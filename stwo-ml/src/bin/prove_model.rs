//! `prove-model` CLI binary.
//!
//! Standalone tool that takes an ONNX model (or HuggingFace model directory)
//! and JSON input, and produces a serialized proof ready for `cairo-prove prove-ml`.
//!
//! **The prover validates the model end-to-end before proving.**
//! If weights are missing, config is broken, or dimensions don't match,
//! the proof is refused. Proofs over an incomplete model are meaningless.
//!
//! ```text
//! # ONNX mode:
//! prove-model \
//!   --model model.onnx \
//!   --input input.json \
//!   --output proof.json \
//!   --format cairo_serde \
//!   --gpu
//!
//! # HuggingFace directory mode:
//! prove-model \
//!   --model-dir /path/to/qwen3-14b \
//!   --layers 1 \
//!   --output proof.json \
//!   --gpu
//!
//! # Validate-only mode (no proving):
//! prove-model \
//!   --model-dir /path/to/qwen3-14b \
//!   --validate
//! ```

#![feature(portable_simd)]

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::Parser;
use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;

use stwo_ml::cairo_serde::{
    MLClaimMetadata, serialize_ml_proof_for_recursive,
    serialize_ml_proof_to_arguments_file,
};
use stwo_ml::compiler::inspect::summarize_model;
use stwo_ml::compiler::onnx::{load_onnx, OnnxModel};
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::gadgets::quantize::{QuantStrategy, quantize_tensor};
use stwo_ml::json_serde;
use stwo_ml::starknet::compute_io_commitment;

/// Output format for the proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    CairoSerde,
    Json,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cairo_serde" | "cairo-serde" | "felt" => Ok(OutputFormat::CairoSerde),
            "json" => Ok(OutputFormat::Json),
            _ => Err(format!("unknown format '{s}', expected 'cairo_serde' or 'json'")),
        }
    }
}

/// stwo-ml prove-model: generic ZKML prover CLI.
///
/// Takes an ONNX model (or HuggingFace model directory) and produces a proof.
/// Validates the model before proving — proofs over broken models are refused.
#[derive(Parser, Debug)]
#[command(name = "prove-model", version, about)]
struct Cli {
    /// Path to ONNX model file.
    /// Use --model for .onnx files, or --model-dir for HuggingFace directories.
    #[arg(long, group = "model_source")]
    model: Option<PathBuf>,

    /// Path to a HuggingFace model directory (with config.json + *.safetensors).
    /// Alternative to --model for loading directly from SafeTensors.
    #[arg(long, group = "model_source")]
    model_dir: Option<PathBuf>,

    /// Number of transformer layers to prove (default: all from config.json).
    /// Only used with --model-dir.
    #[arg(long)]
    layers: Option<usize>,

    /// Path to JSON input file (array of f32 values).
    /// Not required if --inspect is used. If omitted with --model-dir, a random
    /// input of the correct shape is generated.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Path to write output proof.
    #[arg(long, default_value = "proof.json")]
    output: PathBuf,

    /// Output format: 'cairo_serde' (felt252 hex array) or 'json'.
    #[arg(long, default_value = "cairo_serde")]
    format: OutputFormat,

    /// Model ID (hex or decimal) for the on-chain claim.
    #[arg(long, default_value = "0x1")]
    model_id: String,

    /// Use GPU backend if available.
    #[arg(long)]
    gpu: bool,

    /// Print model summary and exit (no proving).
    #[arg(long)]
    inspect: bool,

    /// Validate model directory only (check files, weights, dimensions).
    /// Exits with code 0 if valid, 1 if not.
    #[arg(long)]
    validate: bool,

    /// Channel salt for Fiat-Shamir (optional).
    #[arg(long)]
    salt: Option<u64>,
}

fn parse_model_id(s: &str) -> FieldElement {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        FieldElement::from_hex_be(hex).unwrap_or_else(|_| {
            eprintln!("Error: invalid hex model-id '{s}'");
            process::exit(1);
        })
    } else {
        let val: u64 = s.parse().unwrap_or_else(|_| {
            eprintln!("Error: invalid model-id '{s}', expected hex (0x...) or decimal");
            process::exit(1);
        });
        FieldElement::from(val)
    }
}

fn load_input_json(path: &PathBuf) -> Vec<f32> {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error: cannot read input file '{}': {e}", path.display());
        process::exit(1);
    });
    serde_json::from_str::<Vec<f32>>(&contents).unwrap_or_else(|e| {
        eprintln!("Error: invalid JSON input (expected array of f32): {e}");
        process::exit(1);
    })
}

/// Quantize f32 input values to M31 field elements.
fn quantize_input(values: &[f32], rows: usize, cols: usize) -> M31Matrix {
    let (quantized, _params) = quantize_tensor(values, QuantStrategy::Symmetric8);
    let mut matrix = M31Matrix::new(rows, cols);
    for (i, &v) in quantized.iter().enumerate().take(rows * cols) {
        matrix.data[i] = v;
    }
    matrix
}

/// Generate a deterministic random input for a given shape.
fn generate_random_input(rows: usize, cols: usize) -> M31Matrix {
    let mut matrix = M31Matrix::new(rows, cols);
    for i in 0..(rows * cols) {
        // Simple deterministic pseudo-random values
        matrix.data[i] = M31::from((i as u32 * 7 + 13) % (1 << 20));
    }
    matrix
}

fn load_model(cli: &Cli) -> OnnxModel {
    if let Some(ref model_dir) = cli.model_dir {
        // HuggingFace directory mode — validation is built into load_hf_model
        eprintln!("Loading HuggingFace model from: {}", model_dir.display());
        stwo_ml::compiler::hf_loader::load_hf_model(model_dir, cli.layers)
            .unwrap_or_else(|e| {
                eprintln!("Error loading model directory: {e}");
                process::exit(1);
            })
    } else if let Some(ref model_path) = cli.model {
        // ONNX mode
        eprintln!("Loading ONNX model: {}", model_path.display());
        load_onnx(model_path).unwrap_or_else(|e| {
            eprintln!("Error loading ONNX model: {e}");
            process::exit(1);
        })
    } else {
        eprintln!("Error: specify either --model (ONNX file) or --model-dir (HuggingFace directory)");
        process::exit(1);
    }
}

fn main() {
    let cli = Cli::parse();

    // --validate: run validation only and exit
    if cli.validate {
        if let Some(ref model_dir) = cli.model_dir {
            let report = stwo_ml::compiler::hf_loader::validate_model_directory(
                model_dir,
                cli.layers,
            );
            eprintln!();
            eprintln!("  ── Model Validation ──");
            eprintln!("{}", report.format_report());
            eprintln!();

            if report.passed() {
                eprintln!("Model is valid and ready for proving.");
                process::exit(0);
            } else {
                eprintln!("Model validation FAILED. Fix the issues above.");
                process::exit(1);
            }
        } else {
            eprintln!("--validate requires --model-dir");
            process::exit(1);
        }
    }

    let model = load_model(&cli);

    // --inspect: print summary and exit
    if cli.inspect {
        let summary = summarize_model(&model);
        println!("{summary}");
        return;
    }

    // Build input
    let input = if let Some(ref input_path) = cli.input {
        // Load from JSON
        eprintln!("Loading input: {}", input_path.display());
        let input_f32 = load_input_json(input_path);
        let (in_rows, in_cols) = model.input_shape;
        let expected_len = in_rows * in_cols;
        if input_f32.len() != expected_len {
            eprintln!(
                "Error: input has {} values but model expects {} ({} x {})",
                input_f32.len(),
                expected_len,
                in_rows,
                in_cols,
            );
            process::exit(1);
        }
        quantize_input(&input_f32, in_rows, in_cols)
    } else {
        // Generate random input for --model-dir mode
        let (in_rows, in_cols) = model.input_shape;
        eprintln!("Generating random input: {} x {}", in_rows, in_cols);
        generate_random_input(in_rows, in_cols)
    };

    // Prove
    eprintln!(
        "Proving {} layers (gpu={})",
        model.graph.num_layers(),
        cli.gpu,
    );

    let t0 = Instant::now();

    let proof = if cli.gpu {
        stwo_ml::aggregation::prove_model_aggregated_onchain_auto(
            &model.graph,
            &input,
            &model.weights,
        )
    } else {
        stwo_ml::aggregation::prove_model_aggregated_onchain(
            &model.graph,
            &input,
            &model.weights,
        )
    };

    let prove_elapsed = t0.elapsed();

    let proof = proof.unwrap_or_else(|e| {
        eprintln!("Error: proving failed: {e}");
        process::exit(1);
    });

    eprintln!("Proving completed in {:.2}s", prove_elapsed.as_secs_f64());

    // Build metadata
    let io_commitment = compute_io_commitment(&input, &proof.execution.output);
    let model_id = parse_model_id(&cli.model_id);

    // Determine activation type from the first activation layer
    let activation_type = model
        .graph
        .nodes
        .iter()
        .find_map(|n| match &n.op {
            stwo_ml::compiler::graph::GraphOp::Activation { activation_type, .. } => {
                Some(*activation_type as u8)
            }
            _ => None,
        })
        .unwrap_or(0);

    // Compute weight commitment from all weight matrices
    let weight_commitment = {
        let mut all_weight_data: Vec<M31> = Vec::new();
        for (_, w) in &model.weights.weights {
            all_weight_data.extend_from_slice(&w.data);
        }
        if all_weight_data.is_empty() {
            FieldElement::ZERO
        } else {
            let weight_matrix = M31Matrix {
                rows: 1,
                cols: all_weight_data.len(),
                data: all_weight_data,
            };
            let dummy_output = M31Matrix::new(1, 1);
            compute_io_commitment(&weight_matrix, &dummy_output)
        }
    };

    let metadata = MLClaimMetadata {
        model_id,
        num_layers: model.graph.num_layers() as u32,
        activation_type,
        io_commitment,
        weight_commitment,
    };

    // Serialize
    let output_str = match cli.format {
        OutputFormat::CairoSerde => {
            let felts = serialize_ml_proof_for_recursive(&proof, &metadata, cli.salt);
            serialize_ml_proof_to_arguments_file(&felts)
        }
        OutputFormat::Json => json_serde::proof_to_json(&proof, &metadata),
    };

    std::fs::write(&cli.output, &output_str).unwrap_or_else(|e| {
        eprintln!("Error writing output to '{}': {e}", cli.output.display());
        process::exit(1);
    });

    eprintln!(
        "Proof written to {} ({} bytes, format={:?})",
        cli.output.display(),
        output_str.len(),
        cli.format,
    );
    eprintln!(
        "  matmul_proofs: {}, activation_claims: {}",
        proof.matmul_proofs.len(),
        proof.activation_claims.len(),
    );
    eprintln!(
        "  prove_time: {:.2}s, model: {}, layers: {}",
        prove_elapsed.as_secs_f64(),
        model.metadata.name,
        model.metadata.num_layers,
    );
}
