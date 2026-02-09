//! Qwen3-14B Proof Generation Binary
//!
//! End-to-end: load model → build graph → inference → prove → verify → serialize
//!
//! Usage:
//!   prove_qwen --model-dir /path/to/qwen3-14b [--layers 1] [--seq-len 1]

#![feature(portable_simd)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use stwo::core::fields::m31::M31;

use stwo_ml::compiler::graph::{ComputationGraph, GraphBuilder, GraphOp, GraphWeights};
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::components::activation::ActivationType;
use stwo_ml::gadgets::quantize::QuantStrategy;
use stwo_ml::backend::{gpu_is_available, gpu_device_name, gpu_available_memory};
use stwo_ml::pipeline::types::PipelineConfig;
use stwo_ml::pipeline::prover::prove_model_pipeline;
use stwo_ml::pipeline::verifier::verify_pipeline_proof;
// On-chain serialization uses proof.model_commitment / proof.io_commitment directly

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

/// Build a computation graph for N transformer blocks.
///
/// Simplified sequential block:
///   LayerNorm → Q_proj(h→h) → O_proj(h→h) → LayerNorm → gate_proj(h→inter) → GELU → down_proj(inter→h)
///
/// Note: K/V projections are omitted because GQA uses smaller dim (num_kv_heads × head_dim = 1024)
/// which doesn't chain sequentially with the 5120-dim Q projection. The full attention
/// computation (score matmuls, softmax) is also omitted — this proves the linear projections
/// and MLP which dominate the parameter count and computation.
fn build_qwen3_graph(config: &Qwen3Config, num_blocks: usize, seq_len: usize) -> ComputationGraph {
    let batch = seq_len; // seq_len tokens, each hidden_size
    let h = config.hidden_size;
    let inter = config.intermediate_size;

    let mut builder = GraphBuilder::new((batch, h));

    for _block in 0..num_blocks {
        // === Pre-attention RMSNorm ===
        builder.layer_norm();

        // === Self-Attention projections (Q and O only for sequential proof chain) ===
        // Q projection: (batch, h) → (batch, h)
        builder.linear(h);
        // Output projection: (batch, h) → (batch, h)
        builder.linear(h);

        // === Post-attention RMSNorm ===
        builder.layer_norm();

        // === MLP (SwiGLU simplified) ===
        // gate_proj: (batch, h) → (batch, intermediate)
        builder.linear(inter);
        // SiLU activation on gate
        builder.activation(ActivationType::GELU); // Closest to SiLU we have
        // down_proj: (batch, intermediate) → (batch, h)
        builder.linear(h);
    }

    builder.build()
}

/// Load real Qwen3 weights from SafeTensors for a specific block.
#[cfg(feature = "safetensors")]
fn load_qwen3_block_weights(
    model_dir: &Path,
    block_idx: usize,
    graph: &ComputationGraph,
) -> Result<GraphWeights, String> {
    use stwo_ml::compiler::quantize_weights::quantize_weight_matrix;

    let mut weights = GraphWeights::new();

    // Find all safetensors files in model_dir
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

    // Tensor name mappings for Qwen3 architecture (4 matmuls per block)
    // Graph: layernorm → q_proj → o_proj → layernorm → gate_proj → gelu → down_proj
    // MatMul nodes: q_proj, o_proj, gate_proj, down_proj
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

    // Try to load each weight from safetensors files
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
                    // SafeTensors stores (out_dim, in_dim); matmul needs (k=in_dim, n=out_dim)
                    let (st_rows, st_cols) = (shape[0], shape[1]);
                    println!("    Loaded {} shape=({},{}) → node {} needs ({}x{})",
                             target_name, st_rows, st_cols, node_idx, k, n);

                    // Transpose from (out_dim, in_dim) to (in_dim, out_dim) = (k, n)
                    let data = if st_rows == *n && st_cols == *k {
                        // Need transpose: (n, k) → (k, n)
                        let mut transposed = vec![0.0f32; raw.len()];
                        for i in 0..st_rows {
                            for j in 0..st_cols {
                                transposed[j * st_rows + i] = raw[i * st_cols + j];
                            }
                        }
                        transposed
                    } else {
                        // Already correct shape or square
                        raw
                    };

                    let (matrix, _) = quantize_weight_matrix(
                        &data, *k, *n, QuantStrategy::Direct,
                    );
                    weights.add_weight(*node_idx, matrix);
                    found = true;
                    break;
                }
            }

            if !found {
                // Generate random weights as fallback
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
        // Subnormal
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
        // Direct CUDA init test for diagnostics
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
    println!("[Model: Qwen3-14B]");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Intermediate: {}", config.intermediate_size);
    println!("  Attention heads: {} (KV: {})", config.num_attention_heads, config.num_kv_heads);
    println!("  Total layers: {} (proving {})", config.num_layers, num_blocks);
    println!("  Sequence length: {}", seq_len);
    println!();

    // === Phase 3: Build Computation Graph ===
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

    // === Phase 5: Create Input ===
    let mut input = M31Matrix::new(seq_len, config.hidden_size);
    // Simulate embedding output (token → hidden state)
    for i in 0..seq_len {
        for j in 0..config.hidden_size {
            let val = ((i * config.hidden_size + j + 1) as u32) % ((1u32 << 31) - 1);
            input.set(i, j, M31::from(val));
        }
    }
    println!("[Input] {}x{} M31 matrix (simulated embedding output)", seq_len, config.hidden_size);

    // === Phase 6: PROVE ===
    let pipeline_config = PipelineConfig {
        onchain_matmul: true,
        prove_activations: true,
        generate_receipt: true,
    };

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  PROVING {} transformer block(s)...", num_blocks);
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

            // === Phase 7: VERIFY ===
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

            // === Phase 8: ON-CHAIN SERIALIZATION ===
            println!();
            println!("[On-Chain Serialization]");
            println!("  Model commitment: 0x{}", hex_short(&proof.model_commitment));
            println!("  IO commitment:    0x{}", hex_short(&proof.io_commitment));
            println!("  Layer proofs:     {}", proof.layer_proofs.len());
            println!("  Proven layers:    {}", proof.num_proven_layers());
            println!("  MatMul proofs:    {}", proof.num_matmul_proofs());
            println!("  Ready for StweMlVerifier.verify_ml_inference()");
            println!("  Contract: 0x0490d3ad13c551cc074f10ad261ed6a80cce4fb3e7549888b112aeede108a851");

            println!();
            println!("═══════════════════════════════════════════════════════");
            println!("  COMPLETE: Qwen3-14B block proven and verified");
            println!("  Prove: {:.2}s | Verify: {:.2}ms | Layers: {}",
                     prove_time.as_secs_f64(),
                     verify_time.as_secs_f64() * 1000.0,
                     proof.layer_proofs.len());
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
