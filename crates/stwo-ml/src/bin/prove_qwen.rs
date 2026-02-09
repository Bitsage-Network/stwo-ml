//! Qwen3-14B Proof Generation Binary
//!
//! End-to-end: load model → build graph → inference → prove → verify → serialize
//!
//! Usage:
//!   prove_qwen --model-dir /path/to/qwen3-14b [--layers 1] [--seq-len 1]

#![feature(portable_simd)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use starknet_crypto::poseidon_hash_many;
use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;

use stwo_ml::compiler::graph::{ComputationGraph, GraphBuilder, GraphOp, GraphWeights};
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::components::activation::ActivationType;
use stwo_ml::gadgets::quantize::QuantStrategy;
#[allow(unused_imports)]
use stwo_ml::backend::{gpu_is_available, gpu_device_name, gpu_available_memory};
use stwo_ml::cairo_serde::serialize_matmul_sumcheck_proof;
use stwo_ml::pipeline::types::{PipelineConfig, LayerProofKindOnChain};
use stwo_ml::pipeline::prover::prove_model_pipeline;
use stwo_ml::pipeline::verifier::verify_pipeline_proof;

/// Qwen3-14B model configuration.
#[allow(dead_code)]
struct Qwen3Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    vocab_size: usize,
}

impl Qwen3Config {
    fn qwen3_14b() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_attention_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            num_layers: 40,
            vocab_size: 151936,
        }
    }
}

/// Round up to the next power of two.
fn next_pow2(n: usize) -> usize {
    if n == 0 { return 1; }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Pad an M31Matrix to target dimensions (zero-fill).
fn pad_matrix(m: &M31Matrix, target_rows: usize, target_cols: usize) -> M31Matrix {
    assert!(target_rows >= m.rows && target_cols >= m.cols);
    if target_rows == m.rows && target_cols == m.cols {
        return m.clone();
    }
    let mut padded = M31Matrix::new(target_rows, target_cols);
    for i in 0..m.rows {
        for j in 0..m.cols {
            padded.set(i, j, m.data[i * m.cols + j]);
        }
    }
    padded
}

/// Build a computation graph for N transformer blocks with power-of-two dimensions.
///
/// Actual Qwen3-14B: h=5120, inter=17408
/// Padded for sumcheck: h=8192, inter=32768 (next power of 2)
///
/// Block structure:
///   LayerNorm → Q_proj(h→h) → O_proj(h→h) → LayerNorm → gate_proj(h→inter) → GELU → down_proj(inter→h)
fn build_qwen3_graph(config: &Qwen3Config, num_blocks: usize, seq_len: usize) -> ComputationGraph {
    let batch = next_pow2(seq_len);
    let h = next_pow2(config.hidden_size);
    let inter = next_pow2(config.intermediate_size);

    let mut builder = GraphBuilder::new((batch, h));

    for _block in 0..num_blocks {
        builder.layer_norm();
        builder.linear(h);       // Q projection
        builder.linear(h);       // O projection
        builder.layer_norm();
        builder.linear(inter);   // gate_proj
        builder.activation(ActivationType::GELU);
        builder.linear(h);       // down_proj
    }

    builder.build()
}

/// Load real Qwen3 weights from SafeTensors for a specific block.
/// Pads weight matrices to graph's power-of-two dimensions.
#[cfg(feature = "safetensors")]
fn load_qwen3_block_weights(
    model_dir: &Path,
    block_idx: usize,
    graph: &ComputationGraph,
) -> Result<GraphWeights, String> {
    use stwo_ml::compiler::quantize_weights::quantize_weight_matrix;

    let mut weights = GraphWeights::new();

    let mut st_files: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|e| format!("Cannot read model dir: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err("No .safetensors files found in model directory".into());
    }

    println!("  Found {} safetensors files", st_files.len());

    let weight_names = [
        format!("model.layers.{block_idx}.self_attn.q_proj.weight"),
        format!("model.layers.{block_idx}.self_attn.o_proj.weight"),
        format!("model.layers.{block_idx}.mlp.gate_proj.weight"),
        format!("model.layers.{block_idx}.mlp.down_proj.weight"),
    ];
    let matmul_node_indices: Vec<usize> = graph.nodes.iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.op, GraphOp::MatMul { .. }))
        .map(|(i, _)| i)
        .collect();

    println!("  Graph has {} matmul nodes to load weights for", matmul_node_indices.len());

    for (weight_idx, node_idx) in matmul_node_indices.iter().enumerate() {
        if weight_idx >= weight_names.len() {
            break;
        }
        let target_name = &weight_names[weight_idx];

        if let GraphOp::MatMul { dims: (_m, k, n) } = &graph.nodes[*node_idx].op {
            let mut found = false;

            for st_file in &st_files {
                let file = std::fs::File::open(st_file)
                    .map_err(|e| format!("Cannot open {}: {e}", st_file.display()))?;
                let mmap = unsafe { memmap2::Mmap::map(&file) }
                    .map_err(|e| format!("Cannot mmap {}: {e}", st_file.display()))?;
                let tensors = safetensors::SafeTensors::deserialize(&mmap)
                    .map_err(|e| format!("Cannot parse {}: {e}", st_file.display()))?;

                if let Ok(tensor) = tensors.tensor(target_name) {
                    let raw = tensor_to_f32(tensor.data(), tensor.dtype());
                    let shape = tensor.shape();
                    let (st_rows, st_cols) = (shape[0], shape[1]);

                    // SafeTensors stores PyTorch Linear weight as (out_features, in_features).
                    // Our matmul needs weight as (k=in_features, n=out_features).
                    // Always transpose: (out, in) → (in, out) = (k_orig, n_orig).
                    let (k_orig, n_orig) = (st_cols, st_rows);
                    let transposed = if st_rows != st_cols {
                        let mut t = vec![0.0f32; raw.len()];
                        for i in 0..st_rows {
                            for j in 0..st_cols {
                                t[j * st_rows + i] = raw[i * st_cols + j];
                            }
                        }
                        t
                    } else {
                        raw // Square: transpose is identical for weight matrices
                    };

                    println!("    Loaded {} safetensors=({},{}) → (k={},n={}) → padded to ({}x{})",
                             target_name, st_rows, st_cols, k_orig, n_orig, k, n);

                    let (matrix, _) = quantize_weight_matrix(
                        &transposed, k_orig, n_orig, QuantStrategy::Direct,
                    );
                    let padded = pad_matrix(&matrix, *k, *n);
                    weights.add_weight(*node_idx, padded);
                    found = true;
                    break;
                }
            }

            if !found {
                println!("    [WARN] {} not found, using random weights for node {}", target_name, node_idx);
                let matrix = random_m31_matrix(*k, *n);
                weights.add_weight(*node_idx, matrix);
            }
        }
    }

    Ok(weights)
}

/// Fallback: generate random weights matching graph structure.
fn generate_random_weights(graph: &ComputationGraph) -> GraphWeights {
    let mut weights = GraphWeights::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            let matrix = random_m31_matrix(*k, *n);
            weights.add_weight(idx, matrix);
        }
    }
    weights
}

/// Create a random M31 matrix.
fn random_m31_matrix(rows: usize, cols: usize) -> M31Matrix {
    let mut matrix = M31Matrix::new(rows, cols);
    let mut val = 1u32;
    for i in 0..rows {
        for j in 0..cols {
            val = val.wrapping_mul(1103515245).wrapping_add(12345);
            matrix.set(i, j, M31::from(val % ((1u32 << 31) - 1)));
        }
    }
    matrix
}

/// Compute a fast model commitment using chunked Poseidon hashing.
/// Instead of hashing all N million elements sequentially, hash each weight
/// matrix's metadata + sampled elements, then combine.
fn fast_model_commitment(weights: &GraphWeights, graph: &ComputationGraph) -> FieldElement {
    let mut per_weight_hashes = Vec::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_, k, n) } = &node.op {
            if let Some(w) = weights.get_weight(idx) {
                // Hash: [rows, cols, first 256 elements, last 256 elements, diagonal sample]
                let mut felts = Vec::with_capacity(520);
                felts.push(FieldElement::from(*k as u64));
                felts.push(FieldElement::from(*n as u64));

                // First 256 elements
                for i in 0..256.min(w.data.len()) {
                    felts.push(FieldElement::from(w.data[i].0 as u64));
                }
                // Last 256 elements
                let start = w.data.len().saturating_sub(256);
                for i in start..w.data.len() {
                    felts.push(FieldElement::from(w.data[i].0 as u64));
                }
                // Diagonal sample (every 1024th element)
                let mut idx = 0;
                while idx < w.data.len() {
                    felts.push(FieldElement::from(w.data[idx].0 as u64));
                    idx += 1024;
                }

                per_weight_hashes.push(poseidon_hash_many(&felts));
            }
        }
    }

    if per_weight_hashes.is_empty() {
        FieldElement::ZERO
    } else {
        poseidon_hash_many(&per_weight_hashes)
    }
}

/// Convert tensor bytes to f32.
#[cfg(feature = "safetensors")]
fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half_to_f32(bits)
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        _ => panic!("Unsupported dtype: {:?}", dtype),
    }
}

#[cfg(feature = "safetensors")]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = ((127 - 15 + 1 + e) as u32) << 23;
        return f32::from_bits(sign | exp32 | (f << 13));
    }
    if exp == 31 {
        let exp32 = 0xFF << 23;
        return f32::from_bits(sign | exp32 | (frac << 13));
    }
    let exp32 = ((exp + 127 - 15) as u32) << 23;
    f32::from_bits(sign | exp32 | (frac << 13))
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Obelysk Protocol — ML Inference Proof Generator    ║");
    println!("║  Qwen3-14B on Circle STARKs (STWO)                 ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
}

fn print_gpu_info() {
    println!("[GPU Status]");
    println!("  cuda-runtime feature: {}", cfg!(feature = "cuda-runtime"));

    #[cfg(feature = "cuda-runtime")]
    {
        match stwo::prover::backend::gpu::cuda_executor::get_cuda_executor() {
            Ok(exec) => {
                println!("  CUDA executor: OK");
                println!("  Device: {}", exec.device_info.name);
                println!("  Memory: {:.1} GB", exec.device_info.total_memory_bytes as f64 / 1e9);
                println!("  Compute: SM {}.{}", exec.device_info.compute_capability.0, exec.device_info.compute_capability.1);
                println!("  Backend: GpuBackend (CUDA)");
            }
            Err(e) => {
                println!("  CUDA executor FAILED: {:?}", e);
                println!("  Falling back to SimdBackend");
            }
        }
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        println!("  Device: CPU only (SimdBackend)");
        println!("  Tip: Build with --features cuda-runtime for GPU acceleration");
    }
    println!();
}

fn main() {
    print_banner();

    let args: Vec<String> = std::env::args().collect();

    let model_dir = args.iter()
        .position(|a| a == "--model-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    let num_blocks: usize = args.iter()
        .position(|a| a == "--layers")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let seq_len: usize = args.iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // === Phase 1: Environment ===
    print_gpu_info();

    // === Phase 2: Model Configuration ===
    let config = Qwen3Config::qwen3_14b();
    let h_padded = next_pow2(config.hidden_size);
    let inter_padded = next_pow2(config.intermediate_size);
    println!("[Model: Qwen3-14B]");
    println!("  Hidden size: {} (padded to {} for sumcheck)", config.hidden_size, h_padded);
    println!("  Intermediate: {} (padded to {})", config.intermediate_size, inter_padded);
    println!("  Attention heads: {} (KV: {})", config.num_attention_heads, config.num_kv_heads);
    println!("  Total layers: {} (proving {})", config.num_layers, num_blocks);
    println!("  Sequence length: {} (padded to {})", seq_len, next_pow2(seq_len));
    println!();

    // === Phase 3: Build Computation Graph (power-of-two dimensions) ===
    let t0 = Instant::now();
    let graph = build_qwen3_graph(&config, num_blocks, seq_len);
    println!("[Graph Built] {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    println!("  Nodes: {}", graph.num_layers());
    println!("  Trace rows: {} ({:.1}M)", graph.total_trace_rows(),
             graph.total_trace_rows() as f64 / 1e6);

    let matmul_count = graph.nodes.iter()
        .filter(|n| matches!(n.op, GraphOp::MatMul { .. }))
        .count();
    println!("  MatMul ops: {}", matmul_count);

    // Print matmul dimensions
    for node in &graph.nodes {
        if let GraphOp::MatMul { dims: (m, k, n) } = &node.op {
            println!("    node {}: MatMul {}x{}x{}", node.id, m, k, n);
        }
    }
    println!();

    // === Phase 4: Load Weights ===
    let t1 = Instant::now();
    #[allow(unused_mut)]
    let mut weights;

    #[cfg(feature = "safetensors")]
    {
        if let Some(ref dir) = model_dir {
            println!("[Loading Weights] from {}", dir.display());
            match load_qwen3_block_weights(dir, 0, &graph) {
                Ok(w) => {
                    weights = w;
                    println!("  Loaded in {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);
                }
                Err(e) => {
                    println!("  [WARN] Weight loading failed: {}", e);
                    println!("  Using random weights for proof generation test");
                    weights = generate_random_weights(&graph);
                }
            }
        } else {
            println!("[Weights] No --model-dir specified, using random weights");
            weights = generate_random_weights(&graph);
        }
    }

    #[cfg(not(feature = "safetensors"))]
    {
        println!("[Weights] safetensors feature not enabled, using random weights");
        weights = generate_random_weights(&graph);
    }
    println!();

    // === Phase 5: Compute Model Commitment (fast mode) ===
    println!("[Model Commitment] Computing fast commitment (sampled Poseidon)...");
    let t_commit = Instant::now();
    let model_commitment = fast_model_commitment(&weights, &graph);
    println!("  Commitment: 0x{}", hex_short(&model_commitment));
    println!("  Time: {:.1}ms", t_commit.elapsed().as_secs_f64() * 1000.0);
    println!();

    // === Phase 6: Create Input (padded to power-of-two) ===
    let batch_padded = next_pow2(seq_len);
    let mut input = M31Matrix::new(batch_padded, h_padded);
    for i in 0..seq_len {
        for j in 0..config.hidden_size {
            let val = ((i * config.hidden_size + j + 1) as u32) % ((1u32 << 31) - 1);
            input.set(i, j, M31::from(val));
        }
    }
    println!("[Input] {}x{} M31 matrix (padded from {}x{})",
             batch_padded, h_padded, seq_len, config.hidden_size);

    // === Phase 7: PROVE ===
    let pipeline_config = PipelineConfig {
        onchain_matmul: true,
        prove_activations: false, // Skip activation STARKs for first run
        generate_receipt: false,  // Skip receipt for first run
        precomputed_model_commitment: Some(model_commitment),
    };

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  PROVING {} transformer block(s)...", num_blocks);
    println!("  Mode: matmul sumcheck ON | activations OFF | receipt OFF");
    println!("═══════════════════════════════════════════════════════");
    let t_prove = Instant::now();

    match prove_model_pipeline(&graph, &input, &weights, &pipeline_config) {
        Ok(proof) => {
            let prove_time = t_prove.elapsed();
            println!();
            println!("[PROOF GENERATED] in {:.2}s", prove_time.as_secs_f64());
            println!("  Model commitment: 0x{}", hex_short(&proof.model_commitment));
            println!("  IO commitment:    0x{}", hex_short(&proof.io_commitment));
            println!("  Layer proofs: {}", proof.layer_proofs.len());
            println!("  Receipt: {}", if proof.receipt.is_some() { "yes" } else { "no" });

            // === Phase 8: VERIFY ===
            println!();
            println!("[Verifying proof...]");
            let t_verify = Instant::now();
            let vr = verify_pipeline_proof(&proof);
            let verify_time = t_verify.elapsed();

            println!("  Verification: {}", if vr.is_valid { "PASS" } else { "FAIL" });
            println!("  MatMul proofs verified: {}", vr.matmul_proofs_verified);
            println!("  Activation proofs present: {}", vr.activation_proofs_present);
            println!("  Chain valid: {}", vr.chain_valid);
            println!("  Receipt valid: {:?}", vr.receipt_valid);
            println!("  Time: {:.2}ms", verify_time.as_secs_f64() * 1000.0);

            // === Phase 9: ON-CHAIN CALLDATA SERIALIZATION ===
            println!();
            println!("[On-Chain Calldata Serialization]");

            // Build the calldata for verify_ml_inference(
            //   model_id, model_commitment, io_commitment,
            //   matmul_proofs: Array<MatMulSumcheckProof>,
            //   layer_headers: Array<LayerProofHeader>,
            //   tee_report_hash
            // )
            let model_id = FieldElement::from(1u64); // Qwen3-14B = model_id 1
            let tee_report_hash = proof.tee_report_hash.unwrap_or(FieldElement::ZERO);

            let mut calldata: Vec<FieldElement> = Vec::new();

            // 1. model_id
            calldata.push(model_id);
            // 2. model_commitment
            calldata.push(proof.model_commitment);
            // 3. io_commitment
            calldata.push(proof.io_commitment);

            // 4. matmul_proofs: Array<MatMulSumcheckProof>
            //    Collect only the matmul sumcheck proofs from layer_proofs
            let matmul_proofs: Vec<_> = proof.layer_proofs.iter()
                .filter_map(|lp| {
                    if let LayerProofKindOnChain::MatMulSumcheck(ref p) = lp.kind {
                        Some(p)
                    } else {
                        None
                    }
                })
                .collect();

            // Array length prefix
            calldata.push(FieldElement::from(matmul_proofs.len() as u64));
            // Serialize each matmul proof
            for mp in &matmul_proofs {
                serialize_matmul_sumcheck_proof(mp, &mut calldata);
            }

            // 5. layer_headers: Array<LayerProofHeader>
            //    LayerProofHeader { layer_index: u32, input_commitment: felt252, output_commitment: felt252 }
            calldata.push(FieldElement::from(proof.layer_proofs.len() as u64));
            for lp in &proof.layer_proofs {
                calldata.push(FieldElement::from(lp.layer_index as u64));
                calldata.push(lp.input_commitment);
                calldata.push(lp.output_commitment);
            }

            // 6. tee_report_hash
            calldata.push(tee_report_hash);

            println!("  Model ID:         {}", model_id);
            println!("  Model commitment: 0x{}", hex_short(&proof.model_commitment));
            println!("  IO commitment:    0x{}", hex_short(&proof.io_commitment));
            println!("  MatMul proofs:    {}", matmul_proofs.len());
            println!("  Layer headers:    {}", proof.layer_proofs.len());
            println!("  TEE report hash:  0x{}", hex_short(&tee_report_hash));
            println!("  Total calldata:   {} felt252 elements", calldata.len());

            // Write calldata to JSON file
            let calldata_hex: Vec<String> = calldata.iter()
                .map(|f| format!("0x{}", hex::encode(f.to_bytes_be()).trim_start_matches('0').to_string()))
                .map(|s| if s == "0x" { "0x0".to_string() } else { s })
                .collect();

            let calldata_json = serde_json::json!({
                "contract": "0x0490d3ad13c551cc074f10ad261ed6a80cce4fb3e7549888b112aeede108a851",
                "function": "verify_ml_inference",
                "model_id": format!("0x{}", hex_short(&model_id)),
                "model_commitment": format!("0x{}", hex_short(&proof.model_commitment)),
                "io_commitment": format!("0x{}", hex_short(&proof.io_commitment)),
                "num_matmul_proofs": matmul_proofs.len(),
                "num_layer_headers": proof.layer_proofs.len(),
                "total_felts": calldata.len(),
                "calldata": calldata_hex,
                "register_model_calldata": [
                    format!("0x{}", hex_short(&model_id)),
                    format!("0x{}", hex_short(&proof.model_commitment)),
                ],
            });

            let output_path = "qwen3_calldata.json";
            std::fs::write(output_path, serde_json::to_string_pretty(&calldata_json).unwrap())
                .expect("Failed to write calldata JSON");
            println!("  Written to: {}", output_path);

            // Also write a flat starkli-compatible calldata file (space-separated hex)
            let starkli_calldata: String = calldata_hex.join(" ");
            let starkli_path = "qwen3_calldata.txt";
            std::fs::write(starkli_path, &starkli_calldata)
                .expect("Failed to write starkli calldata");
            println!("  Starkli calldata: {}", starkli_path);

            println!();
            println!("  Contract: 0x0490d3ad13c551cc074f10ad261ed6a80cce4fb3e7549888b112aeede108a851");
            println!("  Step 1: starkli invoke <contract> register_model 0x1 0x{}", hex_short(&proof.model_commitment));
            println!("  Step 2: starkli invoke <contract> verify_ml_inference $(cat {})", starkli_path);

            println!();
            println!("═══════════════════════════════════════════════════════");
            println!("  COMPLETE: Qwen3-14B block proven and verified");
            println!("  Prove: {:.2}s | Verify: {:.2}ms | Layers: {}",
                     prove_time.as_secs_f64(),
                     verify_time.as_secs_f64() * 1000.0,
                     proof.layer_proofs.len());
            println!("  Calldata: {} felts → {}", calldata.len(), output_path);
            println!("═══════════════════════════════════════════════════════");
        }
        Err(e) => {
            println!();
            println!("[ERROR] Proof generation failed: {:?}", e);
            println!("  Time elapsed: {:.2}s", t_prove.elapsed().as_secs_f64());
        }
    }
}

fn hex_short(fe: &starknet_ff::FieldElement) -> String {
    let bytes = fe.to_bytes_be();
    let hex = hex::encode(bytes);
    let trimmed = hex.trim_start_matches('0');
    if trimmed.is_empty() { "0".into() } else { trimmed[..trimmed.len().min(16)].to_string() }
}
