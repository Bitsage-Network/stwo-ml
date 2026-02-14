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
    DirectProofMetadata, MLClaimMetadata, serialize_ml_proof_for_recursive,
    serialize_ml_proof_to_file,
};
use stwo_ml::compiler::inspect::summarize_model;
use stwo_ml::compiler::onnx::{load_onnx, OnnxModel};
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::gadgets::quantize::{QuantStrategy, quantize_tensor};
use stwo_ml::json_serde;
use stwo_ml::aggregation::compute_io_commitment;
use stwo_ml::starknet::build_starknet_proof_direct;
use stwo_ml::tee::{SecurityLevel, detect_tee_capability};

/// Output format for the proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    CairoSerde,
    Json,
    Direct,
    /// Full ML GKR model proof for on-chain `verify_model_gkr()`.
    /// Uses `prove_model_pure_gkr_auto` → `build_gkr_starknet_proof` pipeline.
    /// Output: JSON with GKR calldata, IO calldata, weight openings.
    MlGkr,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cairo_serde" | "cairo-serde" | "felt" => Ok(OutputFormat::CairoSerde),
            "json" => Ok(OutputFormat::Json),
            "direct" => Ok(OutputFormat::Direct),
            "ml_gkr" | "ml-gkr" | "gkr" => Ok(OutputFormat::MlGkr),
            _ => Err(format!(
                "unknown format '{s}', expected 'cairo_serde', 'json', 'direct', or 'ml_gkr'"
            )),
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

    /// Output format: 'cairo_serde' (felt252 hex for cairo-prove),
    /// 'json' (human-readable), 'direct' (on-chain verify_model_direct),
    /// or 'ml_gkr' (full ML GKR proof for on-chain verify_model_gkr).
    #[arg(long, default_value = "cairo_serde")]
    format: OutputFormat,

    /// Model ID (hex or decimal) for the on-chain claim.
    #[arg(long, default_value = "0x1")]
    model_id: String,

    /// Use GPU backend if available.
    #[arg(long)]
    gpu: bool,

    /// Distribute proving across all available GPUs.
    /// Implies --gpu. Chunks model blocks and assigns them to GPUs
    /// using memory-aware bin-packing.
    #[arg(long)]
    multi_gpu: bool,

    /// Memory budget per chunk in GB (used with --multi-gpu).
    /// Controls how aggressively the model is chunked. Smaller values
    /// create more chunks for better load balancing. Default: 16 GB.
    #[arg(long, default_value = "16")]
    chunk_budget_gb: f64,

    /// Security level: 'auto' (default), 'tee' (require TEE), 'zk-only' (pure ZK).
    ///
    /// - auto: Detect CC hardware at runtime, use TEE if available.
    /// - tee: Require NVIDIA CC-On hardware + nvattest. Fails if unavailable.
    /// - zk-only: Skip TEE even if available. Pure STARK proof only.
    #[arg(long, default_value = "auto")]
    security: SecurityLevel,

    /// Print model summary and exit (no proving).
    #[arg(long)]
    inspect: bool,

    /// Validate model directory only (check files, weights, dimensions).
    /// Exits with code 0 if valid, 1 if not.
    #[arg(long)]
    validate: bool,

    /// Use STWO-native GKR for LogUp component verification (lighter than unified STARK).
    ///
    /// When enabled, activation/quantize/layernorm LogUp gates are proven via STWO's
    /// native GKR batch prover, producing a proof compatible with Cairo's
    /// `partially_verify_batch()`. The unified STARK is still generated for backward
    /// compatibility; GKR provides an additional lighter verification path.
    #[arg(long)]
    gkr: bool,

    /// Channel salt for Fiat-Shamir (optional).
    #[arg(long)]
    salt: Option<u64>,

    /// Skip weight commitment computation (Poseidon hash of all weight matrices).
    /// Useful for benchmarking: the proving pipeline completes without the slow
    /// O(total_weight_elements) Poseidon hash. On-chain submission requires it,
    /// so compute it separately with --commitment-only when needed.
    #[arg(long)]
    skip_commitment: bool,

    /// Submit the GKR proof on-chain via `sncast invoke`.
    /// Requires --format ml_gkr. Calls `verify_model_gkr()` on the contract.
    /// For proofs >5000 felts, uses chunked upload via `upload_proof_chunk`.
    #[arg(long)]
    submit_gkr: bool,

    /// Contract address for --submit-gkr (hex).
    #[arg(long, default_value = "0x00c7845a80d01927826b17032a432ad9cd36ea61be17fe8cc089d9b68c57e710")]
    contract: String,

    /// sncast account name for --submit-gkr.
    #[arg(long, default_value = "deployer")]
    account: String,

    /// Starknet network for --submit-gkr.
    #[arg(long, default_value = "sepolia")]
    network: String,

    /// Max fee in ETH for --submit-gkr transactions.
    #[arg(long, default_value = "0.05")]
    max_fee: String,
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

    // Detect TEE capability and display status
    let tee_cap = detect_tee_capability();
    let resolved = cli.security.resolve();
    eprintln!("Security: {} (resolved: {:?})", cli.security, resolved);
    eprintln!("TEE: {}", tee_cap);

    if matches!(cli.security, SecurityLevel::ZkPlusTee) && !tee_cap.cc_active {
        eprintln!(
            "Error: --security tee requires NVIDIA H100/H200/B200 with CC-On firmware.\n  {}",
            tee_cap.status_message
        );
        process::exit(1);
    }

    // Pre-compute values needed by both proving and commitment paths
    let model_id = parse_model_id(&cli.model_id);
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

    // Check weight commitment cache BEFORE entering the parallel scope.
    // If cached, no background thread needed.
    let cached_commitment = if cli.skip_commitment {
        Some(FieldElement::ZERO)
    } else {
        check_commitment_cache(cli.model_dir.as_deref())
    };

    if cached_commitment.is_some() && !cli.skip_commitment {
        eprintln!("Weight commitment loaded from cache (fingerprint validated)");
    } else if cli.skip_commitment {
        eprintln!("Skipping weight commitment (--skip-commitment)");
    }

    // Pipeline: run proving and weight commitment in parallel via std::thread::scope.
    // Both only need &model.weights (shared ref), so this is safe.
    // Commitment uses CPU-only (Poseidon hash), while proving uses GPU + rayon.
    // During GPU phases of proving, idle CPU cores handle commitment work.
    let use_gpu = cli.gpu || cli.multi_gpu;

    eprintln!(
        "Proving {} layers (gpu={}, multi_gpu={}, security={})",
        model.graph.num_layers(),
        use_gpu,
        cli.multi_gpu,
        cli.security,
    );

    // Multi-GPU: print device discovery info
    #[cfg(feature = "multi-gpu")]
    if cli.multi_gpu {
        let devices = stwo_ml::multi_gpu::discover_devices();
        if devices.is_empty() {
            eprintln!("Error: --multi-gpu specified but no GPUs found");
            process::exit(1);
        }
        eprintln!("Multi-GPU: {} device(s) detected", devices.len());
        for d in &devices {
            eprintln!(
                "  GPU {}: {} ({:.1} GB, SM {}.{}, {} SMs)",
                d.ordinal, d.name,
                d.total_memory as f64 / 1e9,
                d.compute_capability.0, d.compute_capability.1,
                d.sm_count,
            );
        }
    }

    let t0 = Instant::now();

    let (proof_result, weight_commitment, commit_elapsed) = std::thread::scope(|s| {
        // Spawn weight commitment on background thread if not cached
        let commit_handle = if cached_commitment.is_none() {
            Some(s.spawn(|| {
                let t_commit = Instant::now();
                let commitment = compute_weight_commitment(
                    &model.weights,
                    cli.model_dir.as_deref(),
                );
                (commitment, t_commit.elapsed())
            }))
        } else {
            None
        };

        // Prove on main thread
        let proof = if cli.format == OutputFormat::MlGkr {
            // Full ML GKR pipeline: all layers via GKR sumcheck (no individual matmul proofs)
            eprintln!("Using full ML GKR pipeline (--format ml_gkr)");
            stwo_ml::aggregation::prove_model_pure_gkr_auto(
                &model.graph,
                &input,
                &model.weights,
            )
        } else if cli.multi_gpu {
            // Multi-GPU path: chunk model and distribute across GPUs
            #[cfg(feature = "multi-gpu")]
            {
                let memory_budget = (cli.chunk_budget_gb * 1e9) as usize;
                let (chunks, metrics) = stwo_ml::compiler::chunked::prove_model_chunked_multi_gpu_with_metrics(
                    &model.graph, &input, &model.weights, memory_budget,
                ).map_err(|e| stwo_ml::compiler::prove::ModelError::ProvingError {
                    layer: 0,
                    message: format!("Multi-GPU chunked proving: {e}"),
                })?;

                eprintln!(
                    "Multi-GPU proving: {:.2}s total, {} chunks across {} devices",
                    metrics.total_elapsed.as_secs_f64(),
                    chunks.len(),
                    metrics.device_stats.len(),
                );
                for stat in &metrics.device_stats {
                    eprintln!(
                        "  GPU {}: {} chunks, {} matmuls, {:.2}s",
                        stat.device_id,
                        stat.chunks_proven.len(),
                        stat.matmuls_proven,
                        stat.elapsed.as_secs_f64(),
                    );
                }

                stwo_ml::compiler::chunked::compose_chunk_proofs_auto(
                    &chunks, &model.graph, &input, &model.weights,
                ).map_err(|e| stwo_ml::compiler::prove::ModelError::ProvingError {
                    layer: 0,
                    message: format!("Chunk composition: {e}"),
                })
            }
            #[cfg(not(feature = "multi-gpu"))]
            {
                eprintln!("Error: --multi-gpu requires the 'multi-gpu' feature");
                process::exit(1);
            }
        } else if cli.gkr {
            // LogUp GKR pipeline: standard pipeline + STWO-native GKR for LogUp components
            eprintln!("Using STWO-native GKR for LogUp verification");
            if use_gpu {
                stwo_ml::aggregation::prove_model_aggregated_onchain_logup_gkr_auto(
                    &model.graph,
                    &input,
                    &model.weights,
                )
            } else {
                stwo_ml::aggregation::prove_model_aggregated_onchain_logup_gkr(
                    &model.graph,
                    &input,
                    &model.weights,
                )
            }
        } else if use_gpu {
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

        // Collect commitment result
        let (commitment, commit_time) = match (cached_commitment, commit_handle) {
            (Some(c), _) => (c, std::time::Duration::ZERO),
            (None, Some(handle)) => {
                let (c, elapsed) = handle.join().expect("commitment thread panicked");
                (c, elapsed)
            }
            _ => (FieldElement::ZERO, std::time::Duration::ZERO),
        };

        (proof, commitment, commit_time)
    });

    let prove_elapsed = t0.elapsed();

    let proof = proof_result.unwrap_or_else(|e| {
        eprintln!("Error: proving failed: {e}");
        process::exit(1);
    });

    if !commit_elapsed.is_zero() {
        eprintln!(
            "Weight commitment computed in {:.2}s (pipelined with proving — zero added latency)",
            commit_elapsed.as_secs_f64(),
        );
    }

    // Generate TEE attestation if applicable
    let tee_attestation = if matches!(resolved, stwo_ml::tee::ResolvedSecurityLevel::ZkPlusTee) {
        eprintln!("Generating TEE attestation...");
        match stwo_ml::tee::TeeModelProver::with_security(cli.security) {
            Ok(tee_prover) if tee_prover.is_tee() => {
                eprintln!(
                    "  TEE: {} (hw={}, secboot={}, dbg={})",
                    tee_prover.attestation.device_id,
                    tee_prover.attestation.hw_model,
                    tee_prover.attestation.secure_boot,
                    tee_prover.attestation.debug_status,
                );
                Some(tee_prover.attestation.clone())
            }
            Ok(_) => None,
            Err(e) => {
                eprintln!("Warning: TEE attestation failed: {e}");
                None
            }
        }
    } else {
        None
    };

    eprintln!("Proving completed in {:.2}s", prove_elapsed.as_secs_f64());

    // Build metadata
    let io_commitment = compute_io_commitment(&input, &proof.execution.output);

    // Compute TEE attestation hash for on-chain commitment
    let tee_hash = tee_attestation
        .as_ref()
        .filter(|a| a.has_report())
        .map(|a| a.report_hash_felt());

    let metadata = MLClaimMetadata {
        model_id,
        num_layers: model.graph.num_layers() as u32,
        activation_type,
        io_commitment,
        weight_commitment,
        tee_attestation_hash: tee_hash,
    };

    // Serialize
    eprintln!("Serializing proof (format={:?})...", cli.format);
    let t_ser = Instant::now();
    let output_bytes = match cli.format {
        OutputFormat::CairoSerde => {
            let felts = serialize_ml_proof_for_recursive(&proof, &metadata, cli.salt);
            eprintln!("  {} felt252 values, streaming to file...", felts.len());
            serialize_ml_proof_to_file(&felts, &cli.output).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            })
        }
        OutputFormat::Json => {
            let output_str = json_serde::proof_to_json(&proof, &metadata);
            let len = output_str.len();
            std::fs::write(&cli.output, &output_str).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            });
            len
        }
        OutputFormat::MlGkr => {
            use stwo_ml::starknet::build_gkr_starknet_proof;

            let gkr_proof = build_gkr_starknet_proof(&proof, model_id, &input)
                .unwrap_or_else(|e| {
                    eprintln!("Error building GKR starknet proof: {e}");
                    eprintln!("Hint: --format ml_gkr requires the ML GKR pipeline (pure GKR proving).");
                    process::exit(1);
                });

            eprintln!(
                "  gkr_calldata: {} felts, io_calldata: {} felts, weight_openings: {} felts",
                gkr_proof.gkr_calldata.len(),
                gkr_proof.io_calldata.len(),
                gkr_proof.weight_opening_calldata.len(),
            );
            eprintln!(
                "  num_layer_proofs: {}, estimated_gas: {}, total_calldata: {} felts",
                gkr_proof.num_layer_proofs,
                gkr_proof.estimated_gas,
                gkr_proof.total_calldata_size,
            );

            let json_obj = serde_json::json!({
                "format": "ml_gkr",
                "model_id": format!("0x{:064x}", gkr_proof.model_id),
                "io_commitment": format!("0x{:064x}", gkr_proof.io_commitment),
                "num_layer_proofs": gkr_proof.num_layer_proofs,
                "estimated_gas": gkr_proof.estimated_gas,
                "total_calldata_size": gkr_proof.total_calldata_size,
                "gkr_calldata": gkr_proof.gkr_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "io_calldata": gkr_proof.io_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "weight_opening_calldata": gkr_proof.weight_opening_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
            });
            let output_str = serde_json::to_string_pretty(&json_obj).unwrap();
            let len = output_str.len();
            std::fs::write(&cli.output, &output_str).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            });
            len
        }
        OutputFormat::Direct => {
            let direct_metadata = DirectProofMetadata {
                model_id,
                weight_commitment,
                num_layers: model.graph.num_layers() as u32,
                activation_type,
            };
            let direct_proof = build_starknet_proof_direct(&proof, &input, direct_metadata);

            eprintln!(
                "  batched_calldata: {} batches, stark_chunks: {} chunks, has_stark: {}",
                direct_proof.batched_calldata.len(),
                direct_proof.stark_chunks.len(),
                direct_proof.has_activation_stark,
            );
            eprintln!(
                "  estimated_gas: {}, total_calldata: {} felts",
                direct_proof.estimated_gas,
                direct_proof.total_calldata_size,
            );

            // Write as JSON for inspection / pipeline consumption
            let json_obj = serde_json::json!({
                "model_id": format!("0x{:064x}", direct_proof.model_id),
                "io_commitment": format!("0x{:064x}", io_commitment),
                "raw_io_data_len": direct_proof.raw_io_data.len(),
                "weight_commitment": format!("0x{:064x}", direct_proof.weight_commitment),
                "num_layers": direct_proof.num_layers,
                "activation_type": direct_proof.activation_type,
                "has_activation_stark": direct_proof.has_activation_stark,
                "estimated_gas": direct_proof.estimated_gas,
                "total_calldata_size": direct_proof.total_calldata_size,
                "batched_calldata": direct_proof.batched_calldata.iter()
                    .map(|batch| batch.iter().map(|f| format!("0x{:x}", f)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                "stark_chunks": direct_proof.stark_chunks.iter()
                    .map(|chunk| chunk.iter().map(|f| format!("0x{:x}", f)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            });
            let output_str = serde_json::to_string_pretty(&json_obj).unwrap();
            let len = output_str.len();
            std::fs::write(&cli.output, &output_str).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            });
            len
        }
    };

    let ser_elapsed = t_ser.elapsed();

    eprintln!(
        "Proof written to {} ({} bytes, format={:?}, serialize={:.2}s)",
        cli.output.display(),
        output_bytes,
        cli.format,
        ser_elapsed.as_secs_f64(),
    );
    eprintln!(
        "  matmul_proofs: {} individual + {} batched ({} total), activation_claims: {}",
        proof.matmul_proofs.len(),
        proof.batched_matmul_proofs.len(),
        proof.total_matmul_count(),
        proof.activation_claims.len(),
    );
    eprintln!(
        "  prove_time: {:.2}s, model: {}, layers: {}",
        prove_elapsed.as_secs_f64(),
        model.metadata.name,
        model.metadata.num_layers,
    );

    // Print TEE attestation summary
    if let Some(ref att) = tee_attestation {
        let hash_felt = att.report_hash_felt();
        eprintln!(
            "  tee: hw={}, report_hash=0x{:x}, timestamp={}",
            att.hw_model,
            hash_felt,
            att.timestamp,
        );
    } else {
        eprintln!("  tee: none (zk-only)");
    }

    // --submit-gkr: submit GKR proof on-chain via sncast
    if cli.submit_gkr {
        submit_gkr_onchain(&cli, &model, &proof, &input, model_id, io_commitment);
    }
}

/// Submit a GKR proof on-chain via `sncast invoke`.
///
/// Steps:
/// 1. Compile LayeredCircuit from the model graph
/// 2. Build calldata matching `verify_model_gkr()` contract signature
/// 3. Write calldata to temp file (may exceed shell arg limit)
/// 4. Invoke sncast with the calldata
fn submit_gkr_onchain(
    cli: &Cli,
    model: &OnnxModel,
    proof: &stwo_ml::aggregation::AggregatedModelProofOnChain,
    input: &M31Matrix,
    model_id: FieldElement,
    _io_commitment: FieldElement,
) {
    use stwo_ml::starknet::{
        build_verify_model_gkr_calldata, build_register_gkr_calldata,
        build_circuit_descriptor,
    };

    if cli.format != OutputFormat::MlGkr {
        eprintln!("Error: --submit-gkr requires --format ml_gkr");
        process::exit(1);
    }

    let gkr_proof = match &proof.gkr_proof {
        Some(p) => p,
        None => {
            eprintln!("Error: --submit-gkr requires a GKR proof (use --format ml_gkr)");
            process::exit(1);
        }
    };

    // Compile the LayeredCircuit for dimension extraction
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&model.graph)
        .unwrap_or_else(|e| {
            eprintln!("Error compiling GKR circuit: {e}");
            process::exit(1);
        });

    eprintln!();
    eprintln!("=== On-Chain GKR Submission ===");
    eprintln!("  Contract: {}", cli.contract);
    eprintln!("  Account:  {}", cli.account);
    eprintln!("  Network:  {}", cli.network);
    eprintln!("  Max fee:  {} ETH", cli.max_fee);

    // Step 1: Build verify_model_gkr calldata (raw IO data for on-chain recomputation)
    let raw_io_data = stwo_ml::cairo_serde::serialize_raw_io(input, &proof.execution.output);
    let verify_calldata = build_verify_model_gkr_calldata(
        gkr_proof, &circuit, model_id, &raw_io_data,
    );

    eprintln!("  Calldata: {} parts ({} estimated felts)", verify_calldata.total_felts, verify_calldata.total_felts);

    // Write calldata to temp file (may exceed shell arg limit for large proofs)
    let calldata_path = cli.output.with_extension("calldata.txt");
    let calldata_str = verify_calldata.calldata_parts.join(" ");
    std::fs::write(&calldata_path, &calldata_str).unwrap_or_else(|e| {
        eprintln!("Error writing calldata to '{}': {e}", calldata_path.display());
        process::exit(1);
    });
    eprintln!("  Calldata written to: {}", calldata_path.display());

    // Step 2: Check if model needs registration first
    eprintln!("  Checking model registration...");
    let check_result = std::process::Command::new("sncast")
        .args([
            "--account", &cli.account,
            "--network", &cli.network,
            "call",
            "--contract-address", &cli.contract,
            "--function", "get_model_circuit_hash",
            "--calldata", &format!("0x{:x}", model_id),
        ])
        .output();

    let needs_registration = match check_result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // If circuit_hash is 0, model is not registered
            stdout.contains("0x0") || !output.status.success()
        }
        Err(_) => {
            eprintln!("  Warning: could not check registration (sncast not found?)");
            eprintln!("  Attempting to register anyway...");
            true
        }
    };

    if needs_registration {
        eprintln!("  Model not registered — registering for GKR verification...");
        let circuit_desc = build_circuit_descriptor(&circuit);
        let register_calldata = build_register_gkr_calldata(
            model_id,
            &gkr_proof.weight_commitments,
            &circuit_desc,
        );
        let register_str = register_calldata.join(" ");

        let register_result = std::process::Command::new("sncast")
            .args([
                "--account", &cli.account,
                "--network", &cli.network,
                "invoke",
                "--contract-address", &cli.contract,
                "--function", "register_model_gkr",
                "--calldata", &register_str,
                "--max-fee", &cli.max_fee,
            ])
            .output();

        match register_result {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                eprintln!("  Registration submitted: {}", stdout.trim());
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                // May fail if already registered — that's OK
                if stderr.contains("already registered") || stdout.contains("already registered") {
                    eprintln!("  Model already registered (OK)");
                } else {
                    eprintln!("  Warning: registration may have failed:");
                    eprintln!("    stdout: {}", stdout.trim());
                    eprintln!("    stderr: {}", stderr.trim());
                    eprintln!("  Continuing with verification attempt...");
                }
            }
            Err(e) => {
                eprintln!("  Error: could not run sncast: {e}");
                eprintln!("  Make sure sncast is installed and in PATH");
                process::exit(1);
            }
        }
    } else {
        eprintln!("  Model already registered (OK)");
    }

    // Step 3: Submit verification
    eprintln!("  Submitting verify_model_gkr...");

    // For large calldatas, read from file to avoid shell arg limit
    let verify_result = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "sncast --account '{}' --network '{}' invoke \
             --contract-address '{}' \
             --function verify_model_gkr \
             --calldata $(cat '{}') \
             --max-fee {}",
            cli.account,
            cli.network,
            cli.contract,
            calldata_path.display(),
            cli.max_fee,
        ))
        .output();

    match verify_result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("  Verification submitted successfully!");
            eprintln!("  {}", stdout.trim());
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("  Error: verification submission failed");
            eprintln!("    stdout: {}", stdout.trim());
            eprintln!("    stderr: {}", stderr.trim());
            eprintln!("  Calldata saved to: {} (retry manually)", calldata_path.display());
            process::exit(1);
        }
        Err(e) => {
            eprintln!("  Error: could not run sncast: {e}");
            process::exit(1);
        }
    }

    eprintln!("==============================");
}

/// Check if a cached weight commitment exists with valid fingerprint.
fn check_commitment_cache(model_dir: Option<&std::path::Path>) -> Option<FieldElement> {
    let d = model_dir?;
    let fingerprint = compute_fingerprint(d)?;
    let cache_path = d.join(format!(".weight_commitment_{fingerprint}.hex"));
    let hex = std::fs::read_to_string(&cache_path).ok()?;
    FieldElement::from_hex_be(hex.trim().trim_start_matches("0x")).ok()
}

/// Compute a fingerprint from safetensors file metadata (sizes + mtimes).
fn compute_fingerprint(model_dir: &std::path::Path) -> Option<String> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir).ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    for f in &files {
        f.to_str().unwrap_or("").hash(&mut hasher);
        if let Ok(meta) = std::fs::metadata(f) {
            meta.len().hash(&mut hasher);
            if let Ok(mtime) = meta.modified() {
                mtime.hash(&mut hasher);
            }
        }
    }
    Some(format!("{:016x}", hasher.finish()))
}

/// Compute weight commitment: Poseidon hash of all weight matrices.
/// Packed 7:1 (7 M31 values per FieldElement) + parallel Merkle segments.
/// Caches result with fingerprint for instant validated reuse on subsequent runs.
fn compute_weight_commitment(
    weights: &stwo_ml::compiler::graph::GraphWeights,
    model_dir: Option<&std::path::Path>,
) -> FieldElement {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n_weights = weights.weights.len();
    eprintln!("[BG] Computing weight commitment ({} matrices, packed 7:1, parallel)...", n_weights);
    eprintln!("[BG]   First run — will cache with fingerprint for instant validated reuse.");
    let t_commit = Instant::now();

    let weight_list: Vec<_> = weights.weights.iter().collect();
    let done_count = AtomicUsize::new(0);
    let total = weight_list.len();

    // Pack 7 M31 values (7×31=217 bits) into one FieldElement via Horner's method.
    let pack_m31 = |values: &[M31]| -> FieldElement {
        let base = FieldElement::from(1u64 << 31);
        let mut result = FieldElement::ZERO;
        for &v in values.iter().rev() {
            result = result * base + FieldElement::from(v.0 as u64);
        }
        result
    };

    let n_segments = 64usize;

    let hash_segment = |data: &[M31]| -> FieldElement {
        let chunk_size = 4096;
        let packed: Vec<FieldElement> = data.chunks(7)
            .map(|c| pack_m31(c))
            .collect();
        if packed.is_empty() {
            return FieldElement::ZERO;
        }
        let mut hash_inputs: Vec<FieldElement> = Vec::with_capacity(chunk_size + 1);
        let mut running = FieldElement::ZERO;
        for chunk in packed.chunks(chunk_size) {
            hash_inputs.clear();
            hash_inputs.push(running);
            hash_inputs.extend_from_slice(chunk);
            running = starknet_crypto::poseidon_hash_many(&hash_inputs);
        }
        running
    };

    let per_matrix_hashes: Vec<FieldElement> = weight_list.iter().map(|(layer_id, w)| {
        let seg_size = (w.data.len() + n_segments - 1) / n_segments;
        let segment_hashes: Vec<FieldElement> = w.data
            .par_chunks(seg_size.max(1))
            .map(|segment| hash_segment(segment))
            .collect();

        let mut final_inputs: Vec<FieldElement> = Vec::with_capacity(segment_hashes.len() + 3);
        final_inputs.push(FieldElement::from(*layer_id as u64));
        final_inputs.push(FieldElement::from((w.rows as u64) << 32 | w.cols as u64));
        final_inputs.push(FieldElement::from(n_segments as u64));
        final_inputs.extend_from_slice(&segment_hashes);
        let matrix_hash = starknet_crypto::poseidon_hash_many(&final_inputs);

        let finished = done_count.fetch_add(1, Ordering::Relaxed) + 1;
        if finished % 20 == 0 || finished == total {
            let elapsed = t_commit.elapsed().as_secs_f64();
            let eta = if finished > 0 { elapsed * (total as f64 / finished as f64 - 1.0) } else { 0.0 };
            eprintln!(
                "[BG]   Weight commitment: {}/{} ({:.1}s elapsed, ~{:.0}s remaining)",
                finished, total, elapsed, eta,
            );
        }
        matrix_hash
    }).collect();

    let commitment = if per_matrix_hashes.is_empty() {
        FieldElement::ZERO
    } else {
        starknet_crypto::poseidon_hash_many(&per_matrix_hashes)
    };
    eprintln!("[BG] Weight commitment computed in {:.2}s", t_commit.elapsed().as_secs_f64());

    // Cache with fingerprint for validated reuse
    if let Some(d) = model_dir {
        if let Some(fp) = compute_fingerprint(d) {
            let cache_path = d.join(format!(".weight_commitment_{fp}.hex"));
            let hex = format!("0x{:064x}", commitment);
            if let Err(e) = std::fs::write(&cache_path, &hex) {
                eprintln!("[BG]   Warning: could not cache to {}: {e}", cache_path.display());
            } else {
                eprintln!("[BG]   Cached to {} (auto-invalidates if weights change)", cache_path.display());
            }
        }
    }
    commitment
}
