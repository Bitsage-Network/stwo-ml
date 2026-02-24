//! GKR prover: layer-by-layer reduction from output to input.
//!
//! Walks a `LayeredCircuit` backwards. At each layer, runs the
//! appropriate reduction protocol:
//! - MatMul → sumcheck over k-dimension
//! - Add → degree-1 split (linearity)
//! - Mul → degree-2 check
//! - Activation → placeholder (LogUp integration in next phase)

use num_traits::{One, Zero};
#[cfg(feature = "cuda-runtime")]
use rayon::prelude::*;
use stwo::core::fields::m31::M31;

use crate::compiler::graph::{GraphExecution, GraphWeights};
use crate::components::attention::{
    attention_forward, split_heads, transpose_m31, AttentionWeights, MultiHeadAttentionConfig,
};
#[cfg(feature = "cuda-runtime")]
use crate::components::matmul::matmul_m31;
use crate::components::matmul::matrix_to_mle_col_major_padded_pub as matrix_to_mle_col_major_padded;
#[cfg(feature = "cuda-runtime")]
use crate::components::matmul::matrix_to_mle_col_major_u32_padded_pub as matrix_to_mle_col_major_u32_padded;
#[cfg(feature = "cuda-runtime")]
use crate::components::matmul::matrix_to_mle_col_major_all_padded;
#[cfg(test)]
use crate::components::matmul::restrict_mle_pub as restrict_mle;
use crate::components::matmul::{
    evaluate_mle_pub as evaluate_mle, matrix_to_mle_pub as matrix_to_mle, pad_matrix_pow2,
    M31Matrix,
};
#[cfg(test)]
use crate::components::matmul::matrix_to_mle_col_major_pub as matrix_to_mle_col_major;
#[cfg(feature = "cuda-runtime")]
use crate::crypto::aggregated_opening::prove_aggregated_binding_gpu;
#[cfg(not(feature = "cuda-runtime"))]
use crate::crypto::aggregated_opening::prove_aggregated_binding;
use crate::crypto::aggregated_opening::AggregatedWeightClaim;
use crate::crypto::poseidon_channel::PoseidonChannel;

use super::circuit::{LayerType, LayeredCircuit};
use super::types::WeightOpeningTranscriptMode;
use super::types::{
    derive_weight_opening_subchannel, GKRClaim, GKRError, GKRProof, LayerProof, SecureField,
};
// ── proof-stream: thread-local sink + zero-overhead macro ─────────────────
#[cfg(feature = "proof-stream")]
use std::cell::RefCell;

#[cfg(feature = "proof-stream")]
thread_local! {
    pub static PROOF_SINK: RefCell<Option<proof_stream::ProofSink>> = RefCell::new(None);
}

/// Install a [`proof_stream::ProofSink`] for this thread.  Returns a RAII guard
/// that flushes and removes the sink on drop.
#[cfg(feature = "proof-stream")]
pub fn set_proof_sink(sink: proof_stream::ProofSink) -> ProofSinkGuard {
    PROOF_SINK.with(|s| *s.borrow_mut() = Some(sink));
    ProofSinkGuard
}

#[cfg(feature = "proof-stream")]
pub struct ProofSinkGuard;
#[cfg(feature = "proof-stream")]
impl Drop for ProofSinkGuard {
    fn drop(&mut self) {
        PROOF_SINK.with(|s| {
            if let Some(sink) = s.borrow().as_ref() {
                sink.flush();
            }
            *s.borrow_mut() = None;
        });
    }
}

/// Emit a proof event if a sink is installed.  The closure is never called
/// (and never compiled) when the `proof-stream` feature is absent.
#[cfg(feature = "proof-stream")]
macro_rules! emit_proof_event {
    ($e:expr) => {
        PROOF_SINK.with(|s| {
            if let Some(sink) = s.borrow().as_ref() {
                sink.emit_if($e);
            }
        })
    };
}
/// No-op: `$($tt:tt)*` avoids compiling the closure body (which may reference
/// `proof_stream::*`) when the feature is disabled.
#[cfg(not(feature = "proof-stream"))]
macro_rules! emit_proof_event {
    ($($tt:tt)*) => {};
}

#[cfg(feature = "proof-stream")]
fn layer_kind_from_type(layer_type: &LayerType) -> proof_stream::LayerKind {
    match layer_type {
        LayerType::MatMul { .. } => proof_stream::LayerKind::MatMul,
        LayerType::Add { .. } => proof_stream::LayerKind::Add,
        LayerType::Mul { .. } => proof_stream::LayerKind::Mul,
        LayerType::Activation { .. } => proof_stream::LayerKind::Activation,
        LayerType::LayerNorm { .. } => proof_stream::LayerKind::LayerNorm,
        LayerType::RMSNorm { .. } => proof_stream::LayerKind::RMSNorm,
        LayerType::Attention { .. } => proof_stream::LayerKind::Attention,
        LayerType::Embedding { .. } => proof_stream::LayerKind::Embedding,
        // Quantize/Dequantize mapped to Add (no dedicated color needed)
        LayerType::Dequantize { .. } | LayerType::Quantize { .. } => proof_stream::LayerKind::Add,
        LayerType::Input | LayerType::Identity => proof_stream::LayerKind::Add,
    }
}

#[cfg(feature = "proof-stream")]
fn estimate_trace_cost(layer_type: &LayerType) -> usize {
    match layer_type {
        LayerType::MatMul { m, k, n, .. } => (m * k * n).next_power_of_two(),
        LayerType::Attention { config } => config.seq_len * config.d_model,
        LayerType::Add { size } | LayerType::Mul { size } => *size,
        LayerType::Activation { size, .. }
        | LayerType::Dequantize { size, .. }
        | LayerType::Quantize { size, .. } => *size,
        LayerType::LayerNorm { dim } | LayerType::RMSNorm { dim } => *dim,
        LayerType::Embedding {
            vocab_size,
            embed_dim,
        } => vocab_size * embed_dim,
        LayerType::Input | LayerType::Identity => 0,
    }
}

#[cfg(feature = "proof-stream")]
#[inline(always)]
fn sf_to_f32(sf: SecureField) -> f32 {
    // QM31 → real M31 component as a float in [0, 1)
    sf.0 .0 .0 as f32 / 0x7fff_ffff_u32 as f32
}

#[cfg(feature = "cuda-runtime")]
#[inline]
fn prepare_weight_opening_mle_u32(weight: &M31Matrix) -> Vec<u32> {
    matrix_to_mle_col_major_u32_padded(weight)
}

#[cfg(feature = "cuda-runtime")]
fn gkr_batch_weight_openings_enabled() -> bool {
    match std::env::var("STWO_GKR_BATCH_WEIGHT_OPENINGS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        // Default ON for GPU builds — parallel weight openings are ~4-8x faster.
        // Set STWO_GKR_BATCH_WEIGHT_OPENINGS=0 to disable.
        #[cfg(feature = "cuda-runtime")]
        Err(_) => true,
        #[cfg(not(feature = "cuda-runtime"))]
        Err(_) => false,
    }
}

#[cfg(feature = "cuda-runtime")]
fn gkr_batch_weight_opening_jobs() -> usize {
    std::env::var("STWO_GKR_BATCH_WEIGHT_OPENING_JOBS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| {
            // Use half of available cores (capped at 8) for weight openings.
            // These are CPU-bound Merkle tree constructions that parallelize well.
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (cpus / 2).clamp(2, 8)
        })
}

fn gkr_aggregate_weight_binding_enabled() -> bool {
    match std::env::var("STWO_GKR_AGGREGATE_WEIGHT_BINDING") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

fn gkr_trustless_mode2_enabled() -> bool {
    match std::env::var("STWO_GKR_TRUSTLESS_MODE2") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

fn gkr_trustless_mode3_enabled() -> bool {
    match std::env::var("STWO_GKR_TRUSTLESS_MODE3") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

fn gkr_aggregated_oracle_sumcheck_enabled() -> bool {
    match std::env::var("STWO_WEIGHT_BINDING") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "aggregated" || v == "oracle_sumcheck" || v == "1"
        }
        Err(_) => false,
    }
}

// =============================================================================
// Shared weight opening helpers (used by both CPU and GPU provers)
// =============================================================================

/// Computed mode flags for weight opening strategy.
struct WeightModeFlags {
    aggregate_weight_binding: bool,
    trustless_mode2: bool,
    trustless_mode3: bool,
}

/// Compute weight opening mode flags from environment variables.
fn compute_weight_mode_flags() -> WeightModeFlags {
    let aggregate_weight_binding_env = gkr_aggregate_weight_binding_enabled();
    let trustless_mode3 = gkr_trustless_mode3_enabled();
    let trustless_mode2 = gkr_trustless_mode2_enabled() && !trustless_mode3;
    if trustless_mode3 && gkr_trustless_mode2_enabled() {
        eprintln!("  [GKR] STWO_GKR_TRUSTLESS_MODE3=1 overrides STWO_GKR_TRUSTLESS_MODE2=1");
    }
    if (trustless_mode2 || trustless_mode3) && aggregate_weight_binding_env {
        let mode = if trustless_mode3 { "MODE3" } else { "MODE2" };
        eprintln!(
            "  [GKR] STWO_GKR_TRUSTLESS_{mode}=1 overrides STWO_GKR_AGGREGATE_WEIGHT_BINDING=on (keeping opening proofs)"
        );
    }
    let aggregate_weight_binding =
        aggregate_weight_binding_env && !(trustless_mode2 || trustless_mode3);
    WeightModeFlags {
        aggregate_weight_binding,
        trustless_mode2,
        trustless_mode3,
    }
}

/// Apply aggregated oracle sumcheck binding for all weight claims.
///
/// Returns (weight_commitments, weight_claims, proof). The `deferred_proofs`
/// are updated in-place with the correct weight commitments.
fn apply_aggregated_oracle_sumcheck(
    weight_data: &[(usize, Vec<SecureField>, SecureField)],
    deferred_weight_claims_data: &[(usize, Vec<SecureField>, SecureField)],
    deferred_proofs: &mut [super::types::DeferredProof],
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
    log_tag: &str,
) -> Result<
    (
        Vec<starknet_ff::FieldElement>,
        Vec<super::types::WeightClaim>,
        Option<crate::crypto::aggregated_opening::AggregatedWeightBindingProof>,
    ),
    GKRError,
> {
    let mut weight_commitments = Vec::with_capacity(weight_data.len());
    let mut weight_claims = Vec::with_capacity(weight_data.len());

    // Build weight claims for main walk
    for (weight_node_id, eval_point, expected_value) in weight_data.iter() {
        weight_claims.push(super::types::WeightClaim {
            weight_node_id: *weight_node_id,
            eval_point: eval_point.clone(),
            expected_value: *expected_value,
        });
    }

    // Collect aggregated claims with MLEs + Merkle root commitments.
    // Each weight matrix needs: MLE conversion + Poseidon Merkle root.
    // For 160 matrices of size 5120x17408, this is the dominant cost.
    //
    // GPU path: sequential per-matrix GPU Poseidon Merkle (~200ms each, ~32s total).
    // CPU fallback: 4-thread parallel pool (~3-5s each, ~5-10min total).
    //
    // Memory optimization: check early if full binding proof is needed.
    // If not (RLC fast path), skip storing per-matrix MLEs — they're only needed
    // for prove_aggregated_binding_gpu(). Without this, 160 matrices × ~4GB each
    // would require ~640 GB RAM, causing OOM on machines with ≤256 GB.
    let need_full_binding = std::env::var("STWO_AGGREGATED_FULL_BINDING")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false);
    let total_claims = weight_data.len() + deferred_weight_claims_data.len();
    eprintln!(
        "  [{log_tag}] aggregated oracle sumcheck: preparing {total_claims} weight commitments..."
    );
    let t_commit = std::time::Instant::now();

    // Combine main + deferred into a single list for ordered processing
    struct WeightPrepInput<'a> {
        weight_node_id: usize,
        eval_point: &'a [SecureField],
        expected_value: SecureField,
        matrix_index: usize,
        is_deferred: bool,
        deferred_idx: usize,
    }
    let mut prep_inputs: Vec<WeightPrepInput<'_>> = Vec::with_capacity(total_claims);
    for (idx, (weight_node_id, eval_point, expected_value)) in weight_data.iter().enumerate() {
        prep_inputs.push(WeightPrepInput {
            weight_node_id: *weight_node_id,
            eval_point,
            expected_value: *expected_value,
            matrix_index: idx,
            is_deferred: false,
            deferred_idx: 0,
        });
    }
    for (deferred_idx, (weight_node_id, eval_point, expected_value)) in
        deferred_weight_claims_data.iter().enumerate()
    {
        prep_inputs.push(WeightPrepInput {
            weight_node_id: *weight_node_id,
            eval_point,
            expected_value: *expected_value,
            matrix_index: weight_data.len() + deferred_idx,
            is_deferred: true,
            deferred_idx,
        });
    }

    struct WeightPrepResult {
        matrix_index: usize,
        n_vars: usize,
        commitment: starknet_ff::FieldElement,
        eval_point: Vec<SecureField>,
        expected_value: SecureField,
        mle: Vec<SecureField>,
        #[cfg(feature = "cuda-runtime")]
        mle_u32: Vec<u32>,
        is_deferred: bool,
        deferred_idx: usize,
    }

    // --- GPU path: sequential per-matrix, GPU Poseidon Merkle root ---
    // Controlled by STWO_GPU_MLE_COMMITMENT (default ON for cuda-runtime builds).
    // Per-matrix GPU failure falls back to CPU for that matrix and continues.
    // STWO_GPU_MLE_COMMITMENT_REQUIRE / STWO_GPU_ONLY → panic on GPU failure.
    #[cfg(feature = "cuda-runtime")]
    let prep_results: Vec<Result<WeightPrepResult, GKRError>> = {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use stwo::prover::backend::gpu::cuda_executor::{
            get_cuda_executor, is_cuda_available, upload_poseidon252_round_constants,
        };
        use crate::crypto::mle_opening::{gpu_mle_commitment_enabled, gpu_mle_commitment_required};

        static GPU_COMMITMENT_LOGGED: AtomicBool = AtomicBool::new(false);
        static GPU_COMMITMENT_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

        let gpu_strict = gpu_mle_commitment_required();
        let gpu_enabled = gpu_mle_commitment_enabled();
        let gpu_available = gpu_enabled && is_cuda_available() && get_cuda_executor().is_ok();

        if gpu_strict && !gpu_available {
            if !gpu_enabled {
                return Err(GKRError::ReductionError {
                    layer_idx: 0,
                    reason: "GPU MLE commitment strict mode enabled, but STWO_GPU_MLE_COMMITMENT is disabled".to_string(),
                });
            }
            return Err(GKRError::ReductionError {
                layer_idx: 0,
                reason: format!(
                    "GPU MLE commitment strict mode enabled, but GPU unavailable (cuda_available={}, executor_ok={})",
                    is_cuda_available(),
                    get_cuda_executor().is_ok(),
                ),
            });
        }

        if gpu_available {
            let executor = get_cuda_executor().map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("cuda init for weight commitments: {e}"),
            })?;
            let d_rc = upload_poseidon252_round_constants(&executor.device)
                .map_err(|e| GKRError::ReductionError {
                    layer_idx: 0,
                    reason: format!("upload poseidon round constants: {e}"),
                })?;

            if !GPU_COMMITMENT_LOGGED.swap(true, AtomicOrdering::Relaxed) {
                eprintln!(
                    "  [{log_tag}] weight commitment backend: GPU Poseidon Merkle (pipelined)",
                );
            }

            // Optimization 1: Reusable limb buffers — allocate once for the max
            // padded matrix size, reuse across all 160+ matrices.
            // Two buffers for double-buffering (optimization 3: pipelining).
            let max_padded_n = prep_inputs.iter().map(|input| {
                let b = weights.get_weight(input.weight_node_id);
                b.map(|m| {
                    m.rows.next_power_of_two() * m.cols.next_power_of_two()
                }).unwrap_or(0)
            }).max().unwrap_or(0);
            let limb_buf_size = max_padded_n * 4;

            // Double-buffer: [0] and [1] swapped each iteration.
            // Using array indexing avoids borrow-checker conflicts with named vars.
            let mut limb_bufs = [vec![0u64; limb_buf_size], vec![0u64; limb_buf_size]];

            // Pipelined processing: while GPU commits matrix N, CPU prepares matrix N+1.
            //
            // Optimization 2: fused matrix_to_mle_col_major_all_padded produces
            // SecureField MLE + u32 MLE + u64 limbs in a single matrix traversal.
            //
            // Optimization 3: std::thread::scope overlaps CPU prep with GPU work.
            struct PreparedMatrix {
                mle: Vec<SecureField>,
                mle_u32: Vec<u32>,
                n_elements: usize,
                n_vars: usize,
            }

            // Prepare first matrix eagerly (no overlap possible for matrix 0)
            let first_input = &prep_inputs[0];
            let first_matrix = weights
                .get_weight(first_input.weight_node_id)
                .ok_or(GKRError::MissingWeight {
                    node_id: first_input.weight_node_id,
                })?;
            let first_n = first_matrix.rows.next_power_of_two()
                * first_matrix.cols.next_power_of_two();
            limb_bufs[0][..first_n * 4].fill(0);
            let (first_mle, first_u32) =
                matrix_to_mle_col_major_all_padded(first_matrix, &mut limb_bufs[0]);
            let mut current_prep = PreparedMatrix {
                n_vars: first_mle.len().trailing_zeros() as usize,
                n_elements: first_n,
                mle: first_mle,
                mle_u32: first_u32,
            };

            let mut results = Vec::with_capacity(total_claims);
            let mut gpu_count = 0usize;
            let mut cpu_fallback_count = 0usize;
            let mut cur_buf_idx: usize = 0; // 0 or 1

            for idx in 0..total_claims {
                let input = &prep_inputs[idx];
                let cur_n_elements = current_prep.n_elements;
                let next_buf_idx = 1 - cur_buf_idx;

                // If there's a next matrix, prepare it in parallel with GPU work.
                let next_prep = if idx + 1 < total_claims {
                    let next_input = &prep_inputs[idx + 1];
                    let next_matrix = weights
                        .get_weight(next_input.weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: next_input.weight_node_id,
                        })?;
                    let next_n = next_matrix.rows.next_power_of_two()
                        * next_matrix.cols.next_power_of_two();

                    // split_at_mut gives us independent &mut to each buffer
                    let (buf_lo, buf_hi) = limb_bufs.split_at_mut(1);
                    let (cur_buf, next_buf) = if cur_buf_idx == 0 {
                        (&buf_lo[0][..], &mut buf_hi[0][..])
                    } else {
                        (&buf_hi[0][..], &mut buf_lo[0][..])
                    };
                    next_buf[..next_n * 4].fill(0);

                    let mut next_result: Option<PreparedMatrix> = None;
                    std::thread::scope(|s| {
                        let cpu_handle = s.spawn(|| {
                            let (mle, u32_mle) =
                                matrix_to_mle_col_major_all_padded(next_matrix, next_buf);
                            PreparedMatrix {
                                n_vars: mle.len().trailing_zeros() as usize,
                                n_elements: next_n,
                                mle,
                                mle_u32: u32_mle,
                            }
                        });

                        // GPU: commit current matrix (runs while CPU thread prepares next)
                        let commitment = match crate::crypto::mle_opening::commit_mle_root_only_gpu_from_limbs(
                            cur_buf, cur_n_elements, executor, &d_rc,
                        ) {
                            Ok(root) => {
                                gpu_count += 1;
                                root
                            }
                            Err(e) => {
                                if gpu_strict {
                                    results.push(Err(GKRError::ReductionError {
                                        layer_idx: 0,
                                        reason: format!(
                                            "GPU MLE commitment strict mode: GPU failed for matrix {}: {e}",
                                            idx
                                        ),
                                    }));
                                    next_result = Some(cpu_handle.join().expect("CPU prep thread panicked"));
                                    return;
                                }
                                if !GPU_COMMITMENT_FALLBACK_LOGGED.swap(true, AtomicOrdering::Relaxed) {
                                    eprintln!(
                                        "  [{log_tag}] weight commitment: GPU failed for matrix {}, falling back to CPU ({e})",
                                        idx
                                    );
                                }
                                cpu_fallback_count += 1;
                                crate::crypto::mle_opening::commit_mle_root_only(&current_prep.mle)
                            }
                        };

                        next_result = Some(cpu_handle.join().expect("CPU prep thread panicked"));

                        if results.last().map(|r| r.is_err()).unwrap_or(false) {
                            return;
                        }

                        let finished = idx + 1;
                        if finished % 10 == 0 || finished == total_claims {
                            eprintln!(
                                "  [{log_tag}] weight commitments: {finished}/{total_claims} done ({:.1}s, gpu={gpu_count} cpu={cpu_fallback_count})",
                                t_commit.elapsed().as_secs_f64(),
                            );
                        }

                        // Drop MLEs immediately when full binding isn't needed,
                        // freeing ~4 GB per matrix before the next one is prepared.
                        let (mle_out, mle_u32_out) = if need_full_binding {
                            (
                                std::mem::take(&mut current_prep.mle),
                                std::mem::take(&mut current_prep.mle_u32),
                            )
                        } else {
                            drop(std::mem::take(&mut current_prep.mle));
                            drop(std::mem::take(&mut current_prep.mle_u32));
                            (Vec::new(), Vec::new())
                        };
                        results.push(Ok(WeightPrepResult {
                            matrix_index: input.matrix_index,
                            n_vars: current_prep.n_vars,
                            commitment,
                            eval_point: input.eval_point.to_vec(),
                            expected_value: input.expected_value,
                            mle: mle_out,
                            mle_u32: mle_u32_out,
                            is_deferred: input.is_deferred,
                            deferred_idx: input.deferred_idx,
                        }));
                    });

                    if results.last().map(|r| r.is_err()).unwrap_or(false) {
                        break;
                    }

                    next_result
                } else {
                    // Last matrix — no next to prefetch, just GPU commit.
                    let commitment = match crate::crypto::mle_opening::commit_mle_root_only_gpu_from_limbs(
                        &limb_bufs[cur_buf_idx], cur_n_elements, executor, &d_rc,
                    ) {
                        Ok(root) => {
                            gpu_count += 1;
                            root
                        }
                        Err(e) => {
                            if gpu_strict {
                                results.push(Err(GKRError::ReductionError {
                                    layer_idx: 0,
                                    reason: format!(
                                        "GPU MLE commitment strict mode: GPU failed for matrix {}: {e}",
                                        idx
                                    ),
                                }));
                                break;
                            }
                            if !GPU_COMMITMENT_FALLBACK_LOGGED.swap(true, AtomicOrdering::Relaxed) {
                                eprintln!(
                                    "  [{log_tag}] weight commitment: GPU failed for matrix {}, falling back to CPU ({e})",
                                    idx
                                );
                            }
                            cpu_fallback_count += 1;
                            crate::crypto::mle_opening::commit_mle_root_only(&current_prep.mle)
                        }
                    };

                    eprintln!(
                        "  [{log_tag}] weight commitments: {}/{total_claims} done ({:.1}s, gpu={gpu_count} cpu={cpu_fallback_count})",
                        idx + 1,
                        t_commit.elapsed().as_secs_f64(),
                    );

                    let (mle_out, mle_u32_out) = if need_full_binding {
                        (
                            std::mem::take(&mut current_prep.mle),
                            std::mem::take(&mut current_prep.mle_u32),
                        )
                    } else {
                        drop(std::mem::take(&mut current_prep.mle));
                        drop(std::mem::take(&mut current_prep.mle_u32));
                        (Vec::new(), Vec::new())
                    };
                    results.push(Ok(WeightPrepResult {
                        matrix_index: input.matrix_index,
                        n_vars: current_prep.n_vars,
                        commitment,
                        eval_point: input.eval_point.to_vec(),
                        expected_value: input.expected_value,
                        mle: mle_out,
                        mle_u32: mle_u32_out,
                        is_deferred: input.is_deferred,
                        deferred_idx: input.deferred_idx,
                    }));
                    None
                };

                if let Some(next) = next_prep {
                    current_prep = next;
                    cur_buf_idx = next_buf_idx;
                }
            }
            results
        } else {
            // GPU unavailable or disabled — use CPU parallel path
            if !GPU_COMMITMENT_FALLBACK_LOGGED.swap(true, AtomicOrdering::Relaxed) {
                eprintln!(
                    "  [{log_tag}] weight commitment backend: CPU parallel (GPU {})",
                    if !gpu_enabled { "disabled" } else { "unavailable" },
                );
            }
            use std::sync::atomic::AtomicUsize;
            let done_count = AtomicUsize::new(0);
            let max_parallel_commitments = std::env::var("STWO_GKR_PARALLEL_COMMITMENTS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(4);

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(max_parallel_commitments)
                .build()
                .map_err(|e| GKRError::ReductionError {
                    layer_idx: 0,
                    reason: format!("failed to build weight-commitment thread pool: {e}"),
                })?;

            eprintln!(
                "  [{log_tag}] using {max_parallel_commitments} parallel CPU threads for commitments",
            );

            pool.install(|| {
                prep_inputs
                .par_iter()
                .map(|input| {
                    let b_matrix = weights
                        .get_weight(input.weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: input.weight_node_id,
                        })?;
                    let b_mle = matrix_to_mle_col_major_padded(b_matrix);
                    let n_vars = b_mle.len().trailing_zeros() as usize;
                    let commitment = crate::crypto::mle_opening::commit_mle_root_only(&b_mle);

                    let mle_u32 = {
                        let u32_mle = matrix_to_mle_col_major_u32_padded(b_matrix);
                        debug_assert_eq!(
                            n_vars,
                            (u32_mle.len() / 4).trailing_zeros() as usize,
                            "n_vars mismatch: SecureField MLE has {n_vars} vars, u32 MLE has {} vars (weight node {})",
                            (u32_mle.len() / 4).trailing_zeros(),
                            input.weight_node_id,
                        );
                        u32_mle
                    };

                    let finished = done_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    if finished % 10 == 0 || finished == total_claims {
                        eprintln!(
                            "  [{log_tag}] weight commitments: {finished}/{total_claims} done ({:.1}s)",
                            t_commit.elapsed().as_secs_f64(),
                        );
                    }

                    let (mle_out, mle_u32_out) = if need_full_binding {
                        (b_mle, mle_u32)
                    } else {
                        drop(b_mle);
                        drop(mle_u32);
                        (Vec::new(), Vec::new())
                    };
                    Ok(WeightPrepResult {
                        matrix_index: input.matrix_index,
                        n_vars,
                        commitment,
                        eval_point: input.eval_point.to_vec(),
                        expected_value: input.expected_value,
                        mle: mle_out,
                        mle_u32: mle_u32_out,
                        is_deferred: input.is_deferred,
                        deferred_idx: input.deferred_idx,
                    })
                })
                .collect()
            })
        }
    };

    // --- CPU-only path (no cuda-runtime feature) ---
    #[cfg(not(feature = "cuda-runtime"))]
    let prep_results: Vec<Result<WeightPrepResult, GKRError>> = {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        let done_count = AtomicUsize::new(0);
        let max_parallel_commitments = std::env::var("STWO_GKR_PARALLEL_COMMITMENTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_parallel_commitments)
            .build()
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("failed to build weight-commitment thread pool: {e}"),
            })?;

        eprintln!(
            "  [{log_tag}] using {max_parallel_commitments} parallel CPU threads for commitments",
        );

        pool.install(|| {
            use rayon::prelude::*;
            prep_inputs
            .par_iter()
            .map(|input| {
                let b_matrix = weights
                    .get_weight(input.weight_node_id)
                    .ok_or(GKRError::MissingWeight {
                        node_id: input.weight_node_id,
                    })?;
                let b_mle = matrix_to_mle_col_major_padded(b_matrix);
                let n_vars = b_mle.len().trailing_zeros() as usize;
                let commitment = crate::crypto::mle_opening::commit_mle_root_only(&b_mle);

                let finished = done_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if finished % 10 == 0 || finished == total_claims {
                    eprintln!(
                        "  [{log_tag}] weight commitments: {finished}/{total_claims} done ({:.1}s)",
                        t_commit.elapsed().as_secs_f64(),
                    );
                }

                let mle_out = if need_full_binding { b_mle } else { drop(b_mle); Vec::new() };
                Ok(WeightPrepResult {
                    matrix_index: input.matrix_index,
                    n_vars,
                    commitment,
                    eval_point: input.eval_point.to_vec(),
                    expected_value: input.expected_value,
                    mle: mle_out,
                    is_deferred: input.is_deferred,
                    deferred_idx: input.deferred_idx,
                })
            })
            .collect()
        })
    };

    // Unpack results in order.
    // Only retain MLEs when full binding proof is needed — otherwise each
    // ~4 GB MLE × 160 matrices would require ~640 GB RAM.
    let mut agg_claims = Vec::with_capacity(total_claims);
    let mut all_mles: Vec<Vec<SecureField>> = if need_full_binding {
        Vec::with_capacity(total_claims)
    } else {
        Vec::new()
    };
    #[cfg(feature = "cuda-runtime")]
    let mut all_mles_u32: Vec<Vec<u32>> = if need_full_binding {
        Vec::with_capacity(total_claims)
    } else {
        Vec::new()
    };

    for result in prep_results {
        let r = result?;
        if !r.is_deferred {
            weight_commitments.push(r.commitment);
        } else {
            // Update deferred proof's weight commitment (only for MatMul kind)
            if let Some(dp) = deferred_proofs.get_mut(r.deferred_idx) {
                if let Some(wc) = dp.weight_commitment_mut() {
                    *wc = r.commitment;
                }
            }
        }
        agg_claims.push(AggregatedWeightClaim {
            matrix_index: r.matrix_index,
            local_n_vars: r.n_vars,
            eval_point: r.eval_point,
            expected_value: r.expected_value,
            commitment: r.commitment,
        });
        if need_full_binding {
            all_mles.push(r.mle);
            #[cfg(feature = "cuda-runtime")]
            all_mles_u32.push(r.mle_u32);
        }
        // When !need_full_binding, r.mle and r.mle_u32 are dropped here,
        // freeing ~4 GB per matrix immediately.
    }
    eprintln!(
        "  [{log_tag}] all {total_claims} weight commitments ready in {:.1}s",
        t_commit.elapsed().as_secs_f64(),
    );

    // Use RLC binding instead of the full aggregated binding proof.
    // The full prove_aggregated_binding_gpu builds a virtual MLE of 2^(selector+max_vars)
    // elements (e.g. 2^29 = 536M for 4 matrices of 134M each) and runs a 29-round
    // MLE opening with GPU Merkle trees at each round — taking 5+ minutes even on H100.
    // RLC binding achieves the same 2^-128 security in ~10ms by combining all expected
    // values into a single scalar via random linear combination.
    // For on-chain verification, the commitments themselves (mixed into Fiat-Shamir)
    // provide the binding; the full opening proof is only needed for trustless mode.
    let binding_proof = if !agg_claims.is_empty() {
        if need_full_binding {
            let mle_refs: Vec<&[SecureField]> = all_mles.iter().map(|m| m.as_slice()).collect();
            #[cfg(feature = "cuda-runtime")]
            let proof = {
                let mle_u32_refs: Vec<&[u32]> = all_mles_u32.iter().map(|m| m.as_slice()).collect();
                prove_aggregated_binding_gpu(&agg_claims, &mle_refs, &mle_u32_refs, channel)
            };
            #[cfg(not(feature = "cuda-runtime"))]
            let proof = prove_aggregated_binding(&agg_claims, &mle_refs, channel);
            eprintln!(
                "  [{log_tag}] aggregated oracle sumcheck: {} claims, ~{} felts calldata (full binding proof)",
                agg_claims.len(),
                proof.estimated_calldata_felts(),
            );
            Some(proof)
        } else {
            // RLC binding: draw random weight, combine all expected values, mix into channel.
            let rho = channel.draw_qm31();
            let mut rho_pow = SecureField::one();
            let mut combined = SecureField::zero();
            for claim in &agg_claims {
                combined = combined + rho_pow * claim.expected_value;
                rho_pow = rho_pow * rho;
            }
            mix_secure_field(channel, combined);
            eprintln!(
                "  [{log_tag}] aggregated oracle sumcheck: {} claims, RLC binding (fast path)",
                agg_claims.len(),
            );
            None
        }
    } else {
        None
    };

    Ok((weight_commitments, weight_claims, binding_proof))
}

/// Apply aggregated RLC weight binding.
///
/// Returns weight claims with the combined expected value mixed into the channel.
///
/// SECURITY: Off-chain verifier only — the Fiat-Shamir ordering (draw rho before
/// committing evaluations) is sound only when the verifier independently recomputes
/// combined_actual via `verify_gkr_with_weights`. Must NOT be used on a trustless
/// verifier path without reordering commit-before-challenge.
fn apply_aggregated_rlc_binding(
    weight_data: &[(usize, Vec<SecureField>, SecureField)],
    deferred_weight_claims_data: &[(usize, Vec<SecureField>, SecureField)],
    channel: &mut PoseidonChannel,
) -> Vec<super::types::WeightClaim> {
    let rho = channel.draw_qm31();
    let mut rho_pow = SecureField::one();
    let mut combined_expected = SecureField::zero();
    for (_, _, expected_value) in deferred_weight_claims_data.iter() {
        combined_expected = combined_expected + rho_pow * *expected_value;
        rho_pow = rho_pow * rho;
    }
    let mut weight_claims = Vec::with_capacity(weight_data.len());
    for (weight_node_id, eval_point, expected_value) in weight_data.iter() {
        weight_claims.push(super::types::WeightClaim {
            weight_node_id: *weight_node_id,
            eval_point: eval_point.clone(),
            expected_value: *expected_value,
        });
        combined_expected = combined_expected + rho_pow * *expected_value;
        rho_pow = rho_pow * rho;
    }
    mix_secure_field(channel, combined_expected);
    eprintln!(
        "  [GKR] aggregated weight binding enabled (RLC): {} claims, {} openings eliminated",
        weight_data.len() + deferred_weight_claims_data.len(),
        weight_data.len() + deferred_weight_claims_data.len()
    );
    weight_claims
}

/// Finalize the weight opening transcript mode after per-opening processing.
fn finalize_weight_transcript_mode(
    flags: &WeightModeFlags,
    aggregate_weight_binding: bool,
) -> Option<WeightOpeningTranscriptMode> {
    if (flags.trustless_mode2 || flags.trustless_mode3) && !aggregate_weight_binding {
        if flags.trustless_mode3 {
            eprintln!(
                "  [GKR] aggregated trustless mode v4 (experimental) enabled: opening proofs retained with mode-3 binding metadata"
            );
            Some(WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental)
        } else {
            eprintln!(
                "  [GKR] aggregated trustless mode v2 enabled: opening proofs retained with mode-2 binding metadata"
            );
            Some(WeightOpeningTranscriptMode::AggregatedTrustlessV2)
        }
    } else {
        None
    }
}

/// Process Add layer reduction result: determine trunk/skip, push deferred claim.
///
/// Returns `(LayerProof, trunk_claim)`.
fn process_add_deferred(
    add_proof: LayerProof,
    current_claim: &GKRClaim,
    input_layers: &[usize],
    deferred_info: &mut Vec<(GKRClaim, usize)>,
) -> (LayerProof, GKRClaim) {
    let LayerProof::Add {
        lhs_eval, rhs_eval, ..
    } = &add_proof
    else {
        unreachable!("process_add_deferred called with non-Add proof")
    };
    // Trunk = higher layer index (encountered next in reverse walk).
    let (trunk_eval, skip_eval, skip_layer_idx, trunk_idx) = if input_layers[1] > input_layers[0] {
        (*rhs_eval, *lhs_eval, input_layers[0], 1u8)
    } else {
        (*lhs_eval, *rhs_eval, input_layers[1], 0u8)
    };
    deferred_info.push((
        GKRClaim {
            point: current_claim.point.clone(),
            value: skip_eval,
        },
        skip_layer_idx,
    ));
    let claim = GKRClaim {
        point: current_claim.point.clone(),
        value: trunk_eval,
    };
    let proof = LayerProof::Add {
        lhs_eval: *lhs_eval,
        rhs_eval: *rhs_eval,
        trunk_idx,
    };
    (proof, claim)
}

/// Compute weight evaluation point from a MatMul reduction and push to weight_data.
///
/// `current_point` is the claim point before reduction. `reduction_claim_point`
/// is the claim point after reduction (contains sumcheck challenges).
fn push_matmul_weight_data(
    weight_node_id: usize,
    m: usize,
    n: usize,
    current_point: &[SecureField],
    reduction_claim_point: &[SecureField],
    final_b_eval: SecureField,
    weight_data: &mut Vec<(usize, Vec<SecureField>, SecureField)>,
) {
    let pm = m.next_power_of_two();
    let log_m = pm.ilog2() as usize;
    let pn = n.next_power_of_two();
    let log_n = pn.ilog2() as usize;
    let r_j = current_point[log_m..log_m + log_n].to_vec();
    let sumcheck_challenges = &reduction_claim_point[log_m..];
    let mut weight_eval_point = r_j;
    weight_eval_point.extend_from_slice(sumcheck_challenges);
    weight_data.push((weight_node_id, weight_eval_point, final_b_eval));
}

/// Prove a full model forward pass using the GKR protocol.
///
/// Walks the circuit from output layer to input layer, running the
/// appropriate reduction at each step. Returns a `GKRProof` that can
/// be verified with `verify_gkr()`.
pub fn prove_gkr(
    circuit: &LayeredCircuit,
    execution: &GraphExecution,
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
) -> Result<GKRProof, GKRError> {
    let d = circuit.layers.len();
    let mut layer_proofs = Vec::with_capacity(d);
    let mut weight_commitments = Vec::with_capacity(d / 2);
    // Capture (weight_node_id, eval_point, final_b_eval) per MatMul for post-walk opening proofs.
    // We intentionally avoid storing full B MLE vectors here to reduce peak memory.
    let mut weight_data: Vec<(usize, Vec<SecureField>, SecureField)> = Vec::with_capacity(d / 2);

    // Seed channel with circuit metadata
    channel.mix_u64(d as u64);
    channel.mix_u64(circuit.input_shape.0 as u64);
    channel.mix_u64(circuit.input_shape.1 as u64);

    // Start with claim on output layer: evaluate the output MLE at a random point
    let output = &execution.output;
    let output_padded = pad_matrix_pow2(output);
    let output_mle = matrix_to_mle(&output_padded);

    let log_out_rows = output_padded.rows.ilog2() as usize;
    let log_out_cols = output_padded.cols.ilog2() as usize;

    let r_out = channel.draw_qm31s(log_out_rows + log_out_cols);
    let output_value = evaluate_mle(&output_mle, &r_out);

    // Mix output claim into channel
    mix_secure_field(channel, output_value);

    let output_claim = GKRClaim {
        point: r_out,
        value: output_value,
    };
    let mut current_claim = output_claim.clone();
    emit_proof_event!(|| proof_stream::ProofEvent::ProofStart {
        model_name: None,
        backend: "cpu-gkr".to_string(),
        num_layers: d,
        input_shape: circuit.input_shape,
        output_shape: circuit.output_shape,
    });

    // Cache GPU device names once before the layer loop — discover_devices() is a
    // syscall that queries NVML/CUDA; calling it per-layer adds measurable overhead.
    #[cfg(all(feature = "proof-stream", feature = "cuda-runtime"))]
    let gkr_gpu_info: Vec<(u32, String, u64)> = crate::multi_gpu::discover_devices()
        .iter()
        .map(|d| (d.ordinal, d.name.clone(), d.total_memory))
        .collect();

    // Deferred claims from DAG Add layers (rhs branches needing separate proofs).
    // Each entry: (rhs_claim, rhs_layer_idx_in_circuit)
    let mut deferred_info: Vec<(GKRClaim, usize)> = Vec::new();

    // Walk layers from output → input
    for layer_idx in (0..d).rev() {
        let layer = &circuit.layers[layer_idx];
        #[cfg(feature = "proof-stream")]
        let _ps_layer_t = std::time::Instant::now();
        emit_proof_event!(|| proof_stream::ProofEvent::LayerStart {
            layer_idx,
            kind: layer_kind_from_type(&layer.layer_type),
            input_shape: layer.input_shape,
            output_shape: layer.output_shape,
            trace_cost: estimate_trace_cost(&layer.layer_type),
            claim_value_approx: sf_to_f32(current_claim.value),
            gpu_device: None,
        });

        let (proof, next_claim) = match &layer.layer_type {
            LayerType::MatMul {
                m,
                k,
                n,
                weight_node_id,
            } => {
                let a_matrix = get_intermediate(execution, layer.node_id)?;
                let b_matrix =
                    weights
                        .get_weight(*weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: *weight_node_id,
                        })?;

                let (reduction, claim) =
                    reduce_matmul_layer(&current_claim, a_matrix, b_matrix, *m, *k, *n, channel)?;

                #[cfg(feature = "proof-stream")]
                {
                    let total_rounds = reduction.round_polys.len();
                    let initial_claim = sf_to_f32(current_claim.value);
                    for (round, rp) in reduction.round_polys.iter().enumerate() {
                        let c0 = proof_stream::SecureFieldMirror {
                            a: rp.c0.0 .0 .0,
                            b: rp.c0.0 .1 .0,
                            c: rp.c0.1 .0 .0,
                            d: rp.c0.1 .1 .0,
                        };
                        let c1 = proof_stream::SecureFieldMirror {
                            a: rp.c1.0 .0 .0,
                            b: rp.c1.0 .1 .0,
                            c: rp.c1.1 .0 .0,
                            d: rp.c1.1 .1 .0,
                        };
                        let c2 = proof_stream::SecureFieldMirror {
                            a: rp.c2.0 .0 .0,
                            b: rp.c2.0 .1 .0,
                            c: rp.c2.1 .0 .0,
                            d: rp.c2.1 .1 .0,
                        };
                        // Geometric decrease: each round halves the claim (one hypercube variable fixed)
                        let claim_approx = initial_claim / (1u32 << round.min(30)) as f32;
                        emit_proof_event!(|| proof_stream::ProofEvent::SumcheckRound {
                            layer_idx,
                            round,
                            total_rounds,
                            poly_deg2: Some(proof_stream::RoundPolyViz { c0, c1, c2 }),
                            poly_deg3: None,
                            claim_value_approx: claim_approx,
                        });
                    }
                }

                push_matmul_weight_data(
                    *weight_node_id,
                    *m,
                    *n,
                    &current_claim.point,
                    &claim.point,
                    reduction.final_b_eval,
                    &mut weight_data,
                );

                (
                    LayerProof::MatMul {
                        round_polys: reduction.round_polys,
                        final_a_eval: reduction.final_a_eval,
                        final_b_eval: reduction.final_b_eval,
                    },
                    claim,
                )
            }

            LayerType::Add { .. } => {
                let (lhs_vals, rhs_vals) = get_binary_op_intermediates(execution, layer, circuit)?;

                let (proof, _claim) =
                    reduce_add_layer(&current_claim, &lhs_vals, &rhs_vals, channel)?;

                process_add_deferred(
                    proof,
                    &current_claim,
                    &layer.input_layers,
                    &mut deferred_info,
                )
            }

            LayerType::Mul { .. } => {
                let (lhs_vals, rhs_vals) = get_binary_op_intermediates(execution, layer, circuit)?;

                reduce_mul_layer(&current_claim, &lhs_vals, &rhs_vals, channel)?
            }

            LayerType::Activation {
                activation_type, ..
            } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;

                reduce_activation_layer(&current_claim, input_matrix, *activation_type, channel)?
            }

            LayerType::LayerNorm { dim, .. } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;

                reduce_layernorm_layer(&current_claim, input_matrix, *dim, channel)?
            }

            LayerType::RMSNorm { dim, .. } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;

                reduce_rmsnorm_layer(&current_claim, input_matrix, *dim, channel)?
            }

            LayerType::Quantize { params, .. } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                reduce_quantize_layer(&current_claim, input_matrix, params, channel)?
            }

            LayerType::Embedding { .. } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                let output_matrix = get_node_output(execution, layer.node_id)?;
                let embed_table =
                    weights
                        .get_weight(layer.node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: layer.node_id,
                        })?;
                reduce_embedding_layer(
                    &current_claim,
                    input_matrix,
                    output_matrix,
                    embed_table,
                    channel,
                )?
            }

            LayerType::Identity => {
                // Claim propagates unchanged
                continue;
            }

            LayerType::Attention { config } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                let attn_weights = get_attention_weights(weights, layer)?;

                reduce_attention_layer(
                    &current_claim,
                    input_matrix,
                    &attn_weights,
                    config,
                    channel,
                )?
            }

            LayerType::Dequantize { params, .. } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;

                reduce_dequantize_layer(&current_claim, input_matrix, params, channel)?
            }

            LayerType::Input => {
                // Should not be reached in normal flow
                break;
            }
        };

        layer_proofs.push(proof);
        current_claim = next_claim;
        #[cfg(feature = "proof-stream")]
        emit_proof_event!(|| proof_stream::ProofEvent::LayerEnd {
            layer_idx,
            kind: proof_stream::LayerProofKind::Sumcheck,
            final_claim_value_approx: sf_to_f32(current_claim.value),
            duration_ms: _ps_layer_t.elapsed().as_millis() as u64,
            rounds_completed: 0,
        });
        #[cfg(feature = "proof-stream")]
        {
            let layers_total = circuit.layers.len();
            emit_proof_event!(|| {
                #[cfg(not(feature = "cuda-runtime"))]
                let devices = vec![proof_stream::GpuSnapshot {
                    device_id: 0,
                    device_name: "CPU (SIMD)".to_string(),
                    utilization: (layer_idx + 1) as f32 / layers_total as f32,
                    free_memory_bytes: None,
                }];
                #[cfg(feature = "cuda-runtime")]
                let devices: Vec<proof_stream::GpuSnapshot> = gkr_gpu_info
                    .iter()
                    .map(|(id, name, mem)| proof_stream::GpuSnapshot {
                        device_id: *id,
                        device_name: name.clone(),
                        utilization: (layer_idx + 1) as f32 / layers_total as f32,
                        free_memory_bytes: Some(*mem),
                    })
                    .collect();
                proof_stream::ProofEvent::GpuStatus {
                    devices,
                    matmul_done: layer_proofs.len(),
                    matmul_total: layers_total,
                    layers_done: layer_idx + 1,
                    layers_total,
                }
            });
        }
    }

    let flags = compute_weight_mode_flags();
    let aggregate_weight_binding = flags.aggregate_weight_binding;
    // In aggregated mode, include deferred MatMul weight claims in the same
    // transcript RLC binding and skip per-deferred Merkle openings.
    let mut deferred_weight_claims_data: Vec<(usize, Vec<SecureField>, SecureField)> =
        Vec::with_capacity(deferred_info.len());

    // Generate deferred proofs for skip branches of DAG Add layers BEFORE
    // weight openings. Fiat-Shamir order: walk → deferred proofs → weight openings.
    let mut deferred_proofs = Vec::with_capacity(deferred_info.len());
    for (deferred_claim, rhs_layer_idx) in &deferred_info {
        let rhs_layer = &circuit.layers[*rhs_layer_idx];
        match &rhs_layer.layer_type {
            LayerType::MatMul {
                m,
                k,
                n,
                weight_node_id,
            } => {
                let a_matrix = get_intermediate(execution, rhs_layer.node_id)?;
                let b_matrix =
                    weights
                        .get_weight(*weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: *weight_node_id,
                        })?;

                mix_secure_field(channel, deferred_claim.value);

                let (reduction, input_claim) =
                    reduce_matmul_layer(deferred_claim, a_matrix, b_matrix, *m, *k, *n, channel)?;

                let pm = m.next_power_of_two();
                let log_m = pm.ilog2() as usize;
                let pn = n.next_power_of_two();
                let log_n = pn.ilog2() as usize;
                let r_j = deferred_claim.point[log_m..log_m + log_n].to_vec();
                let sumcheck_challenges = &input_claim.point[log_m..];
                let mut weight_eval_point = r_j;
                weight_eval_point.extend_from_slice(sumcheck_challenges);

                let deferred_weight_claim = super::types::WeightClaim {
                    weight_node_id: *weight_node_id,
                    eval_point: weight_eval_point.clone(),
                    expected_value: reduction.final_b_eval,
                };
                let (deferred_weight_commitment, deferred_weight_opening) =
                    if aggregate_weight_binding {
                        deferred_weight_claims_data.push((
                            *weight_node_id,
                            weight_eval_point,
                            reduction.final_b_eval,
                        ));
                        (
                            starknet_ff::FieldElement::ZERO,
                            crate::crypto::mle_opening::MleOpeningProof {
                                intermediate_roots: Vec::new(),
                                queries: Vec::new(),
                                final_value: SecureField::zero(),
                            },
                        )
                    } else {
                        #[cfg(feature = "cuda-runtime")]
                        {
                            let b_mle_u32 = matrix_to_mle_col_major_u32_padded(b_matrix);
                            crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                                &b_mle_u32,
                                &deferred_weight_claim.eval_point,
                                channel,
                            )
                        }
                        #[cfg(not(feature = "cuda-runtime"))]
                        {
                            let b_mle = matrix_to_mle_col_major_padded(b_matrix);
                            crate::crypto::mle_opening::prove_mle_opening_with_commitment(
                                &b_mle,
                                &deferred_weight_claim.eval_point,
                                channel,
                            )
                        }
                    };

                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof: LayerProof::MatMul {
                        round_polys: reduction.round_polys,
                        final_a_eval: reduction.final_a_eval,
                        final_b_eval: reduction.final_b_eval,
                    },
                    input_claim,
                    kind: super::types::DeferredProofKind::MatMul {
                        dims: (*m, *k, *n),
                        weight_commitment: deferred_weight_commitment,
                        weight_opening: deferred_weight_opening,
                        weight_claim: deferred_weight_claim,
                    },
                });
            }
            LayerType::Quantize { params, .. } => {
                let input_matrix = get_intermediate(execution, rhs_layer.node_id)?;
                mix_secure_field(channel, deferred_claim.value);
                let (layer_proof, input_claim) =
                    reduce_quantize_layer(deferred_claim, input_matrix, params, channel)?;
                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof,
                    input_claim,
                    kind: super::types::DeferredProofKind::Weightless,
                });
            }
            LayerType::Dequantize { params, .. } => {
                let input_matrix = get_intermediate(execution, rhs_layer.node_id)?;
                mix_secure_field(channel, deferred_claim.value);
                let (layer_proof, input_claim) =
                    reduce_dequantize_layer(deferred_claim, input_matrix, params, channel)?;
                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof,
                    input_claim,
                    kind: super::types::DeferredProofKind::Weightless,
                });
            }
            other => {
                return Err(GKRError::ReductionError {
                    layer_idx: *rhs_layer_idx,
                    reason: format!(
                        "deferred proof not yet supported for layer type {:?}",
                        std::mem::discriminant(other),
                    ),
                });
            }
        }
    }

    // Weight opening strategy dispatch using shared helpers.
    let mut weight_openings = Vec::with_capacity(weight_data.len());
    let mut weight_opening_transcript_mode = WeightOpeningTranscriptMode::Sequential;
    let mut aggregated_binding_proof = None;

    let use_aggregated_oracle_sumcheck = gkr_aggregated_oracle_sumcheck_enabled()
        && (!weight_data.is_empty() || !deferred_weight_claims_data.is_empty());

    let aggregate_weight_binding = aggregate_weight_binding
        && (!weight_data.is_empty() || !deferred_weight_claims_data.is_empty())
        && !use_aggregated_oracle_sumcheck;

    let (weight_claims, weight_commitments_new);

    if use_aggregated_oracle_sumcheck {
        weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        let (wc, claims, proof) = apply_aggregated_oracle_sumcheck(
            &weight_data,
            &deferred_weight_claims_data,
            &mut deferred_proofs,
            weights,
            channel,
            "GKR",
        )?;
        weight_commitments_new = wc;
        weight_claims = claims;
        aggregated_binding_proof = proof;
    } else if aggregate_weight_binding {
        weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;
        weight_claims =
            apply_aggregated_rlc_binding(&weight_data, &deferred_weight_claims_data, channel);
        weight_commitments_new = Vec::new();
    } else {
        weight_commitments_new = Vec::new();
        let opening_seed =
            if (flags.trustless_mode2 || flags.trustless_mode3) && !weight_data.is_empty() {
                Some(channel.draw_felt252())
            } else {
                None
            };

        // Pre-build claims (order matters for subchannel derivation)
        weight_claims = weight_data
            .iter()
            .map(|(wid, ep, ev)| super::types::WeightClaim {
                weight_node_id: *wid,
                eval_point: ep.clone(),
                expected_value: *ev,
            })
            .collect();

        if let Some(seed) = opening_seed {
            // Subchannel mode: each opening is independent → parallelize with rayon
            use rayon::prelude::*;
            let parallel_results: Result<Vec<_>, GKRError> = weight_data
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>()
                .par_iter()
                .map(|(opening_idx, (weight_node_id, _eval_point, _expected_value))| {
                    let b_matrix = weights
                        .get_weight(*weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: *weight_node_id,
                        })?;
                    let claim = &weight_claims[*opening_idx];
                    let mut sub_channel =
                        derive_weight_opening_subchannel(seed, *opening_idx, claim);
                    #[cfg(feature = "cuda-runtime")]
                    {
                        let b_mle_u32 = matrix_to_mle_col_major_u32_padded(b_matrix);
                        Ok(crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                            &b_mle_u32,
                            &claim.eval_point,
                            &mut sub_channel,
                        ))
                    }
                    #[cfg(not(feature = "cuda-runtime"))]
                    {
                        let b_mle = matrix_to_mle_col_major_padded(b_matrix);
                        Ok(crate::crypto::mle_opening::prove_mle_opening_with_commitment(
                            &b_mle,
                            &claim.eval_point,
                            &mut sub_channel,
                        ))
                    }
                })
                .collect();
            let parallel_results = parallel_results?;
            for (commitment, opening) in parallel_results {
                weight_commitments.push(commitment);
                weight_openings.push(opening);
            }
        } else {
            // Sequential mode: openings share the main channel
            for (opening_idx, (weight_node_id, _eval_point, _expected_value)) in
                weight_data.into_iter().enumerate()
            {
                let b_matrix =
                    weights
                        .get_weight(weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: weight_node_id,
                        })?;
                let claim = &weight_claims[opening_idx];
                #[cfg(feature = "cuda-runtime")]
                let (commitment, opening) = {
                    let b_mle_u32 = matrix_to_mle_col_major_u32_padded(b_matrix);
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                        &b_mle_u32,
                        &claim.eval_point,
                        channel,
                    )
                };
                #[cfg(not(feature = "cuda-runtime"))]
                let (commitment, opening) = {
                    let b_mle = matrix_to_mle_col_major_padded(b_matrix);
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment(
                        &b_mle,
                        &claim.eval_point,
                        channel,
                    )
                };
                weight_commitments.push(commitment);
                weight_openings.push(opening);
            }
        }
    }
    weight_commitments.extend(weight_commitments_new);

    if let Some(mode) = finalize_weight_transcript_mode(&flags, aggregate_weight_binding) {
        weight_opening_transcript_mode = mode;
    }

    emit_proof_event!(|| proof_stream::ProofEvent::ProofComplete {
        duration_ms: 0,
        num_layer_proofs: layer_proofs.len(),
        num_weight_openings: weight_openings.len(),
        weight_binding_mode: "sequential".to_string(),
    });
    // Compute IO commitment from model input and output
    let model_input = execution.intermediates.get(&0).unwrap_or(&execution.output);
    let io_commitment = crate::aggregation::compute_io_commitment(model_input, &execution.output);

    Ok(GKRProof {
        layer_proofs,
        output_claim,
        input_claim: current_claim,
        weight_commitments,
        weight_openings,
        weight_claims,
        weight_opening_transcript_mode,
        io_commitment,
        deferred_proofs,
        aggregated_binding: aggregated_binding_proof,
    })
}

/// Prove a full model forward pass using the best available backend.
///
/// Uses `prove_gkr_gpu` when CUDA runtime is enabled and GPU is available
/// (or `OBELYSK_FORCE_GPU=1` is set), otherwise falls back to `prove_gkr`.
pub fn prove_gkr_auto(
    circuit: &LayeredCircuit,
    execution: &GraphExecution,
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
) -> Result<GKRProof, GKRError> {
    #[cfg(feature = "cuda-runtime")]
    {
        if crate::backend::force_gpu() || crate::backend::gpu_is_available() {
            return prove_gkr_gpu(circuit, execution, weights, channel);
        }
    }
    prove_gkr(circuit, execution, weights, channel)
}

// =============================================================================
// GPU-Accelerated GKR Prover
// =============================================================================

/// Prove a full model forward pass using the GKR protocol with GPU acceleration.
///
/// Same protocol as `prove_gkr` but dispatches all hot-path operations to GPU:
/// - MatMul reductions → `GpuSumcheckExecutor::reduce_matmul_layer_gpu()`
/// - MLE evaluations → `GpuSumcheckExecutor::evaluate_mle_gpu()`
///
/// The Fiat-Shamir transcript (Poseidon hashing) runs on GPU during sumcheck
/// rounds, eliminating per-round CPU↔GPU transfers. Channel state is replayed
/// on CPU at the end of each reduction for transcript consistency.
#[cfg(feature = "cuda-runtime")]
pub fn prove_gkr_gpu(
    circuit: &LayeredCircuit,
    execution: &GraphExecution,
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
) -> Result<GKRProof, GKRError> {
    use crate::gpu_sumcheck::GpuSumcheckExecutor;

    let gpu = GpuSumcheckExecutor::cached().map_err(|e| GKRError::ReductionError {
        layer_idx: 0,
        reason: format!("GPU init: {e}"),
    })?;

    let d = circuit.layers.len();
    let mut layer_proofs = Vec::with_capacity(d);
    let mut weight_commitments = Vec::with_capacity(d / 2);
    // Capture (weight_node_id, eval_point, final_b_eval) per MatMul.
    let mut weight_data: Vec<(usize, Vec<SecureField>, SecureField)> = Vec::with_capacity(d / 2);
    let mut deferred_info: Vec<(GKRClaim, usize)> = Vec::new();

    // Seed channel with circuit metadata (same as CPU prover)
    channel.mix_u64(d as u64);
    channel.mix_u64(circuit.input_shape.0 as u64);
    channel.mix_u64(circuit.input_shape.1 as u64);

    // Start with claim on output layer
    let output = &execution.output;
    let output_padded = pad_matrix_pow2(output);
    let output_mle = matrix_to_mle(&output_padded);

    let log_out_rows = output_padded.rows.ilog2() as usize;
    let log_out_cols = output_padded.cols.ilog2() as usize;

    let r_out = channel.draw_qm31s(log_out_rows + log_out_cols);
    let output_value = evaluate_mle(&output_mle, &r_out);

    mix_secure_field(channel, output_value);

    let output_claim = GKRClaim {
        point: r_out,
        value: output_value,
    };
    let mut current_claim = output_claim.clone();

    emit_proof_event!(|| proof_stream::ProofEvent::ProofStart {
        model_name: None,
        backend: "gpu-gkr".to_string(),
        num_layers: circuit.layers.len(),
        input_shape: circuit.input_shape,
        output_shape: circuit.output_shape,
    });

    // Cache GPU device names once — discover_devices() queries NVML per call.
    #[cfg(feature = "proof-stream")]
    let gkr_gpu_info: Vec<(u32, String, u64)> = crate::multi_gpu::discover_devices()
        .iter()
        .map(|d| (d.ordinal, d.name.clone(), d.total_memory))
        .collect();

    let total_work_layers = circuit
        .layers
        .iter()
        .filter(|l| !matches!(l.layer_type, LayerType::Identity | LayerType::Input))
        .count();
    let total_matmul_layers = circuit
        .layers
        .iter()
        .filter(|l| matches!(l.layer_type, LayerType::MatMul { .. }))
        .count();
    let t_gkr_walk = std::time::Instant::now();
    let mut done_work_layers: usize = 0;
    let mut done_matmul_layers: usize = 0;
    let mut last_progress = std::time::Instant::now();

    // Walk layers from output → input
    for layer_idx in (0..d).rev() {
        let layer = &circuit.layers[layer_idx];

        #[cfg(feature = "proof-stream")]
        let _ps_t = std::time::Instant::now();
        emit_proof_event!(|| proof_stream::ProofEvent::LayerStart {
            layer_idx,
            kind: layer_kind_from_type(&layer.layer_type),
            input_shape: layer.input_shape,
            output_shape: layer.output_shape,
            trace_cost: estimate_trace_cost(&layer.layer_type),
            claim_value_approx: sf_to_f32(current_claim.value),
            gpu_device: Some(0),
        });

        let (proof, next_claim) = match &layer.layer_type {
            LayerType::MatMul {
                m,
                k,
                n,
                weight_node_id,
            } => {
                let a_matrix = get_intermediate(execution, layer.node_id)?;
                let b_matrix =
                    weights
                        .get_weight(*weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: *weight_node_id,
                        })?;

                if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
                    eprintln!("[MatMul GPU ENTRY] m={} k={} n={} a={}x{} b={}x{} a_data[0..4]={:?}",
                        m, k, n, a_matrix.rows, a_matrix.cols, b_matrix.rows, b_matrix.cols,
                        &a_matrix.data[..4.min(a_matrix.data.len())]);
                    // Compute C = A*B and show first values
                    let c = crate::components::matmul::matmul_m31(a_matrix, b_matrix);
                    eprintln!("[MatMul GPU ENTRY] c_data[0..4]={:?} c={}x{}",
                        &c.data[..4.min(c.data.len())], c.rows, c.cols);
                }
                let (proof, claim) = reduce_matmul_layer_gpu(
                    &gpu,
                    &current_claim,
                    a_matrix,
                    b_matrix,
                    *m,
                    *k,
                    *n,
                    channel,
                )?;

                let final_b_eval = match &proof {
                    LayerProof::MatMul { final_b_eval, .. } => *final_b_eval,
                    _ => unreachable!("reduce_matmul_layer_gpu returns MatMul"),
                };
                push_matmul_weight_data(
                    *weight_node_id,
                    *m,
                    *n,
                    &current_claim.point,
                    &claim.point,
                    final_b_eval,
                    &mut weight_data,
                );

                #[cfg(feature = "proof-stream")]
                if let LayerProof::MatMul {
                    ref round_polys, ..
                } = proof
                {
                    let n_rounds = round_polys.len();
                    let init = sf_to_f32(current_claim.value);
                    for (round, rp) in round_polys.iter().enumerate() {
                        let c0 = proof_stream::SecureFieldMirror {
                            a: rp.c0.0 .0 .0,
                            b: rp.c0.0 .1 .0,
                            c: rp.c0.1 .0 .0,
                            d: rp.c0.1 .1 .0,
                        };
                        let c1 = proof_stream::SecureFieldMirror {
                            a: rp.c1.0 .0 .0,
                            b: rp.c1.0 .1 .0,
                            c: rp.c1.1 .0 .0,
                            d: rp.c1.1 .1 .0,
                        };
                        let c2 = proof_stream::SecureFieldMirror {
                            a: rp.c2.0 .0 .0,
                            b: rp.c2.0 .1 .0,
                            c: rp.c2.1 .0 .0,
                            d: rp.c2.1 .1 .0,
                        };
                        let claim_approx = init / (1u32 << round.min(30)) as f32;
                        emit_proof_event!(|| proof_stream::ProofEvent::SumcheckRound {
                            layer_idx,
                            round,
                            total_rounds: n_rounds,
                            poly_deg2: Some(proof_stream::RoundPolyViz { c0, c1, c2 }),
                            poly_deg3: None,
                            claim_value_approx: claim_approx,
                        });
                    }
                }

                (proof, claim)
            }

            LayerType::Add { .. } => {
                let (lhs_vals, rhs_vals) = get_binary_op_intermediates(execution, layer, circuit)?;

                let (proof, _claim) =
                    reduce_add_layer_gpu(&gpu, &current_claim, &lhs_vals, &rhs_vals, channel)?;

                process_add_deferred(
                    proof,
                    &current_claim,
                    &layer.input_layers,
                    &mut deferred_info,
                )
            }

            LayerType::Mul { .. } => {
                let (lhs_vals, rhs_vals) = get_binary_op_intermediates(execution, layer, circuit)?;

                reduce_mul_layer_gpu(&gpu, &current_claim, &lhs_vals, &rhs_vals, channel)?
            }

            LayerType::Activation {
                activation_type, ..
            } => {
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                reduce_activation_layer_gpu(
                    &gpu,
                    &current_claim,
                    input_matrix,
                    *activation_type,
                    channel,
                )?
            }

            LayerType::LayerNorm { dim, .. } => {
                // CPU fallback — LayerNorm is memory-bound, not compute-bound
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                reduce_layernorm_layer(&current_claim, input_matrix, *dim, channel)?
            }

            LayerType::RMSNorm { dim, .. } => {
                // CPU fallback — RMSNorm is memory-bound, not compute-bound
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                reduce_rmsnorm_layer(&current_claim, input_matrix, *dim, channel)?
            }

            LayerType::Quantize { params, .. } => {
                // CPU fallback — quantize is table-lookup bound
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                reduce_quantize_layer(&current_claim, input_matrix, params, channel)?
            }

            LayerType::Embedding { .. } => {
                // CPU fallback — embedding lookup is memory-bound and sparse.
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                let output_matrix = get_node_output(execution, layer.node_id)?;
                let embed_table =
                    weights
                        .get_weight(layer.node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: layer.node_id,
                        })?;
                reduce_embedding_layer(
                    &current_claim,
                    input_matrix,
                    output_matrix,
                    embed_table,
                    channel,
                )?
            }

            LayerType::Dequantize { params, .. } => {
                // CPU fallback — dequantize is table-lookup, not compute-bound
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                reduce_dequantize_layer(&current_claim, input_matrix, params, channel)?
            }

            LayerType::Identity => continue,

            LayerType::Attention { config } => {
                // CPU fallback — attention decomposes into matmuls that individually can use GPU
                let input_matrix = get_intermediate(execution, layer.node_id)?;
                let attn_weights = get_attention_weights(weights, layer)?;

                reduce_attention_layer(
                    &current_claim,
                    input_matrix,
                    &attn_weights,
                    config,
                    channel,
                )?
            }

            LayerType::Input => break,
        };

        layer_proofs.push(proof);
        current_claim = next_claim;
        #[cfg(feature = "proof-stream")]
        emit_proof_event!(|| proof_stream::ProofEvent::LayerEnd {
            layer_idx,
            kind: proof_stream::LayerProofKind::Sumcheck,
            final_claim_value_approx: sf_to_f32(current_claim.value),
            duration_ms: _ps_t.elapsed().as_millis() as u64,
            rounds_completed: 0,
        });
        #[cfg(feature = "proof-stream")]
        {
            let layers_total = circuit.layers.len();
            emit_proof_event!(|| {
                let devices: Vec<proof_stream::GpuSnapshot> = gkr_gpu_info
                    .iter()
                    .map(|(id, name, mem)| proof_stream::GpuSnapshot {
                        device_id: *id,
                        device_name: name.clone(),
                        utilization: (layer_idx + 1) as f32 / layers_total as f32,
                        free_memory_bytes: Some(*mem),
                    })
                    .collect();
                proof_stream::ProofEvent::GpuStatus {
                    devices,
                    matmul_done: layer_proofs.len(),
                    matmul_total: layers_total,
                    layers_done: layer_idx + 1,
                    layers_total,
                }
            });
        }

        done_work_layers += 1;
        let is_matmul = matches!(&layer.layer_type, LayerType::MatMul { .. });
        if is_matmul {
            done_matmul_layers += 1;
        }
        let should_log = is_matmul
            || done_work_layers == total_work_layers
            || last_progress.elapsed().as_secs() >= 10;
        if should_log {
            let elapsed = t_gkr_walk.elapsed().as_secs_f64();
            let eta = if done_work_layers > 0 {
                let per_layer = elapsed / done_work_layers as f64;
                per_layer * (total_work_layers.saturating_sub(done_work_layers)) as f64
            } else {
                0.0
            };
            eprintln!(
                "  [GKR] progress: {}/{} layers (matmul {}/{}) elapsed {:.1}s, eta ~{:.0}s",
                done_work_layers,
                total_work_layers,
                done_matmul_layers,
                total_matmul_layers,
                elapsed,
                eta,
            );
            last_progress = std::time::Instant::now();
        }
    }

    eprintln!(
        "  [GKR] layer reductions complete in {:.1}s; entering opening phase (deferred={}, weight_openings={})",
        t_gkr_walk.elapsed().as_secs_f64(),
        deferred_info.len(),
        weight_data.len(),
    );

    // Free GPU memory accumulated during layer reductions before weight opening phase.
    // The GpuSumcheckExecutor::cached() Arc keeps the executor alive, but intermediate
    // buffers from restrict/fold operations are held by CUDA's allocator until sync.
    #[cfg(feature = "cuda-runtime")]
    {
        if let Ok(gpu) = crate::gpu_sumcheck::GpuSumcheckExecutor::cached() {
            if let Err(e) = gpu.device.synchronize() {
                eprintln!("  [GKR] GPU sync warning (non-fatal): {:?}", e);
            }
        }
    }

    let flags = compute_weight_mode_flags();
    let aggregate_weight_binding = flags.aggregate_weight_binding;
    let trustless_mode2 = flags.trustless_mode2;
    let trustless_mode3 = flags.trustless_mode3;
    // In aggregated mode, include deferred MatMul weight claims in the same
    // transcript RLC binding and skip per-deferred Merkle openings.
    let mut deferred_weight_claims_data: Vec<(usize, Vec<SecureField>, SecureField)> =
        Vec::with_capacity(deferred_info.len());

    // Generate deferred proofs for skip branches of DAG Add layers BEFORE weight
    // openings. Fiat-Shamir order: walk → deferred proofs → weight openings.
    let mut deferred_proofs = Vec::with_capacity(deferred_info.len());
    let t_deferred = std::time::Instant::now();
    let deferred_total = deferred_info.len();
    for (i, (deferred_claim, rhs_layer_idx)) in deferred_info.iter().enumerate() {
        let rhs_layer = &circuit.layers[*rhs_layer_idx];
        match &rhs_layer.layer_type {
            LayerType::MatMul {
                m,
                k,
                n,
                weight_node_id,
            } => {
                let a_matrix = get_intermediate(execution, rhs_layer.node_id)?;
                let b_matrix =
                    weights
                        .get_weight(*weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: *weight_node_id,
                        })?;

                mix_secure_field(channel, deferred_claim.value);

                // Use GPU backend for deferred matmul reductions (was CPU-only before).
                let (reduction, input_claim) =
                    reduce_matmul_layer_with_backend::<stwo::prover::backend::gpu::GpuBackend>(
                        deferred_claim,
                        a_matrix,
                        b_matrix,
                        *m,
                        *k,
                        *n,
                        channel,
                    )?;

                let pm = m.next_power_of_two();
                let log_m = pm.ilog2() as usize;
                let pn = n.next_power_of_two();
                let log_n = pn.ilog2() as usize;
                let r_j = deferred_claim.point[log_m..log_m + log_n].to_vec();
                let sumcheck_challenges = &input_claim.point[log_m..];
                let mut weight_eval_point = r_j;
                weight_eval_point.extend_from_slice(sumcheck_challenges);

                let deferred_weight_claim = super::types::WeightClaim {
                    weight_node_id: *weight_node_id,
                    eval_point: weight_eval_point.clone(),
                    expected_value: reduction.final_b_eval,
                };
                let (deferred_weight_commitment, deferred_opening) = if aggregate_weight_binding {
                    deferred_weight_claims_data.push((
                        *weight_node_id,
                        weight_eval_point,
                        reduction.final_b_eval,
                    ));
                    (
                        starknet_ff::FieldElement::ZERO,
                        crate::crypto::mle_opening::MleOpeningProof {
                            intermediate_roots: Vec::new(),
                            queries: Vec::new(),
                            final_value: SecureField::zero(),
                        },
                    )
                } else {
                    let b_mle_u32 = matrix_to_mle_col_major_u32_padded(b_matrix);
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                        &b_mle_u32,
                        &deferred_weight_claim.eval_point,
                        channel,
                    )
                };

                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof: LayerProof::MatMul {
                        round_polys: reduction.round_polys,
                        final_a_eval: reduction.final_a_eval,
                        final_b_eval: reduction.final_b_eval,
                    },
                    input_claim,
                    kind: super::types::DeferredProofKind::MatMul {
                        dims: (*m, *k, *n),
                        weight_commitment: deferred_weight_commitment,
                        weight_opening: deferred_opening,
                        weight_claim: deferred_weight_claim,
                    },
                });
            }
            LayerType::Quantize { params, .. } => {
                let input_matrix = get_intermediate(execution, rhs_layer.node_id)?;
                mix_secure_field(channel, deferred_claim.value);
                let (layer_proof, input_claim) =
                    reduce_quantize_layer(deferred_claim, input_matrix, params, channel)?;
                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof,
                    input_claim,
                    kind: super::types::DeferredProofKind::Weightless,
                });
            }
            LayerType::Dequantize { params, .. } => {
                let input_matrix = get_intermediate(execution, rhs_layer.node_id)?;
                mix_secure_field(channel, deferred_claim.value);
                let (layer_proof, input_claim) =
                    reduce_dequantize_layer(deferred_claim, input_matrix, params, channel)?;
                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof,
                    input_claim,
                    kind: super::types::DeferredProofKind::Weightless,
                });
            }
            other => {
                return Err(GKRError::ReductionError {
                    layer_idx: *rhs_layer_idx,
                    reason: format!(
                        "GPU deferred proof not supported for layer type {:?}",
                        std::mem::discriminant(other),
                    ),
                });
            }
        }

        if deferred_total > 0 && ((i + 1) % 4 == 0 || i + 1 == deferred_total) {
            let elapsed = t_deferred.elapsed().as_secs_f64();
            let eta = if i + 1 > 0 {
                let per = elapsed / (i + 1) as f64;
                per * (deferred_total.saturating_sub(i + 1)) as f64
            } else {
                0.0
            };
            eprintln!(
                "  [GKR] deferred proofs: {}/{} elapsed {:.1}s, eta ~{:.0}s",
                i + 1,
                deferred_total,
                elapsed,
                eta,
            );
        }
    }

    // Generate MLE opening proofs for weight matrices (post-deferred channel state),
    // or use aggregated oracle sumcheck (production on-chain mode).
    let mut weight_openings = Vec::with_capacity(weight_data.len());
    let mut weight_claims = Vec::with_capacity(weight_data.len());
    let mut weight_opening_transcript_mode = WeightOpeningTranscriptMode::Sequential;
    let mut aggregated_binding_proof = None;
    let total_openings = weight_data.len();
    let t_openings = std::time::Instant::now();
    let openings_progress_every = std::env::var("STWO_GKR_OPENINGS_PROGRESS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1);
    let opening_heartbeat_sec = std::env::var("STWO_GKR_OPENING_HEARTBEAT_SEC")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(15);

    let use_aggregated_oracle_sumcheck = gkr_aggregated_oracle_sumcheck_enabled()
        && (!weight_data.is_empty() || !deferred_weight_claims_data.is_empty());

    let aggregate_weight_binding = aggregate_weight_binding
        && (total_openings > 0 || !deferred_weight_claims_data.is_empty())
        && !use_aggregated_oracle_sumcheck;

    let weight_commitments_new;
    if use_aggregated_oracle_sumcheck {
        weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        let (wc, claims, proof) = apply_aggregated_oracle_sumcheck(
            &weight_data,
            &deferred_weight_claims_data,
            &mut deferred_proofs,
            weights,
            channel,
            "GKR-GPU",
        )?;
        weight_commitments_new = wc;
        weight_claims = claims;
        aggregated_binding_proof = proof;
    } else if aggregate_weight_binding {
        weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;
        weight_claims =
            apply_aggregated_rlc_binding(&weight_data, &deferred_weight_claims_data, channel);
        weight_commitments_new = Vec::new();
    } else {
        weight_commitments_new = Vec::new();
        #[cfg(feature = "cuda-runtime")]
        {
            let weight_data = weight_data;
            let batched = gkr_batch_weight_openings_enabled() && total_openings > 0;
            let use_subchannel_openings = trustless_mode2 || trustless_mode3 || batched;
            let opening_seed = if use_subchannel_openings {
                Some(channel.draw_felt252())
            } else {
                None
            };

            if batched {
                use std::sync::atomic::{AtomicUsize, Ordering};
                use std::sync::Arc;

                weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedSubchannelV1;
                let opening_seed = opening_seed.ok_or_else(|| GKRError::ReductionError {
                    layer_idx: 0,
                    reason: "batched weight openings require opening seed but none was drawn"
                        .to_string(),
                })?;
                let jobs = gkr_batch_weight_opening_jobs().min(total_openings.max(1));
                eprintln!(
                    "  [GKR] batched weight openings enabled: {} jobs, {} openings",
                    jobs, total_openings
                );

                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(jobs)
                    .build()
                    .map_err(|e| GKRError::ReductionError {
                        layer_idx: 0,
                        reason: format!("failed to build weight-opening thread pool: {e}"),
                    })?;

                let done = Arc::new(AtomicUsize::new(0));
                let results = pool.install(|| {
                    weight_data
                        .par_iter()
                        .enumerate()
                        .map(|(i, (weight_node_id, eval_point, expected_value))| {
                            let claim = super::types::WeightClaim {
                                weight_node_id: *weight_node_id,
                                eval_point: eval_point.clone(),
                                expected_value: *expected_value,
                            };
                            let b_matrix =
                                weights
                                    .get_weight(*weight_node_id)
                                    .ok_or(GKRError::MissingWeight {
                                        node_id: *weight_node_id,
                                    })?;
                            let b_mle_u32 = prepare_weight_opening_mle_u32(b_matrix);
                            let mut sub_channel =
                                derive_weight_opening_subchannel(opening_seed, i, &claim);
                            let (commitment, opening) =
                                crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                                    &b_mle_u32,
                                    &claim.eval_point,
                                    &mut sub_channel,
                                );

                            let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                            if (finished % openings_progress_every == 0)
                                || finished == total_openings
                            {
                                let elapsed = t_openings.elapsed().as_secs_f64();
                                let eta = if finished > 0 {
                                    let per = elapsed / finished as f64;
                                    per * (total_openings.saturating_sub(finished)) as f64
                                } else {
                                    0.0
                                };
                                eprintln!(
                                    "  [GKR] batched weight openings: {}/{} elapsed {:.1}s, eta ~{:.0}s",
                                    finished, total_openings, elapsed, eta,
                                );
                            }

                            Ok::<_, GKRError>((commitment, opening, claim))
                        })
                        .collect::<Vec<Result<_, GKRError>>>()
                });

                for item in results {
                    let (commitment, opening, claim) = item?;
                    weight_commitments.push(commitment);
                    weight_openings.push(opening);
                    weight_claims.push(claim);
                }
            } else {
                std::thread::scope(|scope| -> Result<(), GKRError> {
                    let mut prefetched_current: Option<Vec<u32>> = None;

                    for (i, (weight_node_id, eval_point, expected_value)) in
                        weight_data.iter().enumerate()
                    {
                        let b_matrix =
                            weights
                                .get_weight(*weight_node_id)
                                .ok_or(GKRError::MissingWeight {
                                    node_id: *weight_node_id,
                                })?;
                        eprintln!(
                            "  [GKR] weight opening {}/{} start: node={}, shape={}x{}",
                            i + 1,
                            total_openings,
                            *weight_node_id,
                            b_matrix.rows,
                            b_matrix.cols,
                        );

                        if prefetched_current.is_none() {
                            prefetched_current = Some(prepare_weight_opening_mle_u32(b_matrix));
                        }

                        let next_prefetch = if i + 1 < total_openings {
                            let next_weight_node_id = weight_data[i + 1].0;
                            Some(scope.spawn(move || -> Result<Vec<u32>, GKRError> {
                                let next_weight = weights.get_weight(next_weight_node_id).ok_or(
                                    GKRError::MissingWeight {
                                        node_id: next_weight_node_id,
                                    },
                                )?;
                                Ok(prepare_weight_opening_mle_u32(next_weight))
                            }))
                        } else {
                            None
                        };

                        let b_mle_u32 =
                            prefetched_current
                                .take()
                                .ok_or_else(|| GKRError::ReductionError {
                                    layer_idx: 0,
                                    reason: format!(
                                        "weight opening {}: prefetched MLE missing for node {}",
                                        i, weight_node_id,
                                    ),
                                })?;
                        let claim = super::types::WeightClaim {
                            weight_node_id: *weight_node_id,
                            eval_point: eval_point.clone(),
                            expected_value: *expected_value,
                        };
                        weight_claims.push(claim.clone());
                        let opening_start = std::time::Instant::now();
                        let opening_hb_stop =
                            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                        let opening_hb = if opening_heartbeat_sec > 0 {
                            let hb_stop = opening_hb_stop.clone();
                            let hb_node = *weight_node_id;
                            let hb_rows = b_matrix.rows;
                            let hb_cols = b_matrix.cols;
                            let hb_idx = i + 1;
                            let hb_total = total_openings;
                            let hb_interval = opening_heartbeat_sec;
                            Some(std::thread::spawn(move || {
                                while !hb_stop.load(std::sync::atomic::Ordering::Relaxed) {
                                    std::thread::sleep(std::time::Duration::from_secs(hb_interval));
                                    if hb_stop.load(std::sync::atomic::Ordering::Relaxed) {
                                        break;
                                    }
                                    eprintln!(
                                    "  [GKR] weight opening {}/{} still running: node={}, shape={}x{}, elapsed {:.1}s",
                                    hb_idx,
                                    hb_total,
                                    hb_node,
                                    hb_rows,
                                    hb_cols,
                                    opening_start.elapsed().as_secs_f64()
                                );
                                }
                            }))
                        } else {
                            None
                        };
                        let (commitment, opening) = if let Some(seed) = opening_seed {
                            let mut sub_channel = derive_weight_opening_subchannel(seed, i, &claim);
                            crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                                &b_mle_u32,
                                &claim.eval_point,
                                &mut sub_channel,
                            )
                        } else {
                            crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                                &b_mle_u32,
                                &claim.eval_point,
                                channel,
                            )
                        };
                        opening_hb_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        if let Some(h) = opening_hb {
                            let _ = h.join();
                        }
                        weight_commitments.push(commitment);
                        weight_openings.push(opening);
                        eprintln!(
                            "  [GKR] weight opening {}/{} done in {:.1}s",
                            i + 1,
                            total_openings,
                            opening_start.elapsed().as_secs_f64()
                        );

                        if let Some(handle) = next_prefetch {
                            prefetched_current =
                                Some(handle.join().map_err(|_| GKRError::ReductionError {
                                    layer_idx: 0,
                                    reason: "weight opening prefetch thread panicked".to_string(),
                                })??);
                        }

                        if total_openings > 0
                            && (((i + 1) % openings_progress_every == 0) || i + 1 == total_openings)
                        {
                            let elapsed = t_openings.elapsed().as_secs_f64();
                            let eta = if i + 1 > 0 {
                                let per = elapsed / (i + 1) as f64;
                                per * (total_openings.saturating_sub(i + 1)) as f64
                            } else {
                                0.0
                            };
                            eprintln!(
                                "  [GKR] weight openings: {}/{} elapsed {:.1}s, eta ~{:.0}s",
                                i + 1,
                                total_openings,
                                elapsed,
                                eta,
                            );
                        }
                    }

                    Ok(())
                })?;
            }
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let opening_seed = if (trustless_mode2 || trustless_mode3) && total_openings > 0 {
                Some(channel.draw_felt252())
            } else {
                None
            };

            for (i, (weight_node_id, eval_point, expected_value)) in
                weight_data.into_iter().enumerate()
            {
                let b_matrix =
                    weights
                        .get_weight(weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: weight_node_id,
                        })?;
                eprintln!(
                    "  [GKR] weight opening {}/{} start: node={}, shape={}x{}",
                    i + 1,
                    total_openings,
                    weight_node_id,
                    b_matrix.rows,
                    b_matrix.cols,
                );

                let claim = super::types::WeightClaim {
                    weight_node_id,
                    eval_point: eval_point.clone(),
                    expected_value,
                };
                weight_claims.push(claim.clone());
                let b_mle = matrix_to_mle_col_major_padded(b_matrix);
                let (commitment, opening) = if let Some(seed) = opening_seed {
                    let mut sub_channel = derive_weight_opening_subchannel(seed, i, &claim);
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment(
                        &b_mle,
                        &claim.eval_point,
                        &mut sub_channel,
                    )
                } else {
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment(
                        &b_mle,
                        &claim.eval_point,
                        channel,
                    )
                };
                weight_commitments.push(commitment);
                weight_openings.push(opening);

                if total_openings > 0
                    && (((i + 1) % openings_progress_every == 0) || i + 1 == total_openings)
                {
                    let elapsed = t_openings.elapsed().as_secs_f64();
                    let eta = if i + 1 > 0 {
                        let per = elapsed / (i + 1) as f64;
                        per * (total_openings.saturating_sub(i + 1)) as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "  [GKR] weight openings: {}/{} elapsed {:.1}s, eta ~{:.0}s",
                        i + 1,
                        total_openings,
                        elapsed,
                        eta,
                    );
                }
            }
        }
    }

    weight_commitments.extend(weight_commitments_new);

    if let Some(mode) = finalize_weight_transcript_mode(&flags, aggregate_weight_binding) {
        weight_opening_transcript_mode = mode;
    }

    let model_input = execution.intermediates.get(&0).unwrap_or(&execution.output);
    let io_commitment = crate::aggregation::compute_io_commitment(model_input, &execution.output);

    emit_proof_event!(|| proof_stream::ProofEvent::ProofComplete {
        duration_ms: t_gkr_walk.elapsed().as_millis() as u64,
        num_layer_proofs: layer_proofs.len(),
        num_weight_openings: weight_openings.len(),
        weight_binding_mode: "gpu-sequential".to_string(),
    });

    Ok(GKRProof {
        layer_proofs,
        output_claim,
        input_claim: current_claim,
        weight_commitments,
        weight_openings,
        weight_claims,
        weight_opening_transcript_mode,
        io_commitment,
        deferred_proofs,
        aggregated_binding: aggregated_binding_proof,
    })
}

/// GPU matmul reduction: dispatches to `GpuSumcheckExecutor::reduce_matmul_layer_gpu`.
#[cfg(feature = "cuda-runtime")]
fn reduce_matmul_layer_gpu(
    _gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    a: &M31Matrix,
    b: &M31Matrix,
    m: usize,
    k: usize,
    n: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    let (reduction, claim) = reduce_matmul_layer_with_backend::<
        stwo::prover::backend::gpu::GpuBackend,
    >(output_claim, a, b, m, k, n, channel)?;

    Ok((
        LayerProof::MatMul {
            round_polys: reduction.round_polys,
            final_a_eval: reduction.final_a_eval,
            final_b_eval: reduction.final_b_eval,
        },
        claim,
    ))
}

/// GPU Add reduction: delegates to `reduce_add_layer_with_backend::<GpuBackend>`.
#[cfg(feature = "cuda-runtime")]
fn reduce_add_layer_gpu(
    _gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    lhs_vals: &[SecureField],
    rhs_vals: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_add_layer_with_backend::<stwo::prover::backend::gpu::GpuBackend>(
        output_claim,
        lhs_vals,
        rhs_vals,
        channel,
    )
}

/// GPU Mul reduction: falls back to CPU eq-sumcheck.
///
/// The eq-sumcheck is memory-bound (not compute-bound), so GPU acceleration
/// has minimal benefit. We use the CPU implementation which is sound.
#[cfg(feature = "cuda-runtime")]
fn reduce_mul_layer_gpu(
    _gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    lhs_vals: &[SecureField],
    rhs_vals: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    // CPU fallback — produces identical proofs to CPU prover
    reduce_mul_layer(output_claim, lhs_vals, rhs_vals, channel)
}

/// GPU-accelerated activation LogUp reduction.
///
/// Runs the full LogUp eq-sumcheck with GPU-accelerated:
/// - MLE evaluation (evaluate_mle_gpu)
/// - Denominator computation (compute_logup_denominators_gpu)
/// - 3-way eq-sumcheck rounds (compute_logup_round_poly_3way)
/// - MLE folding (logup_3way_fold)
///
/// CPU batch_inverse is used for the inverse witnesses (efficient enough on CPU).
#[cfg(feature = "cuda-runtime")]
fn reduce_activation_layer_gpu(
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    activation_type: crate::components::activation::ActivationType,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::{LogUpProof, RoundPolyDeg3};
    use crate::gadgets::lookup_table::PrecomputedTable;
    use crate::gpu_sumcheck::{secure_field_to_u32s, u32s_to_secure_field};
    use stwo::core::fields::FieldExpOps;

    let activation_fn = activation_type.as_fn();

    // Pad input matrix and build MLEs
    let input_padded = pad_matrix_pow2(input_matrix);
    let n = input_padded.rows * input_padded.cols;
    let num_vars = n.ilog2() as usize;

    let input_mle = matrix_to_mle(&input_padded);

    // Compute output by applying activation to each input element
    let output_mle: Vec<SecureField> = input_padded
        .data
        .iter()
        .take(n)
        .map(|&v| SecureField::from(activation_fn(v)))
        .collect();

    // GPU MLE evaluations at claim point
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[ACT GPU] n={} num_vars={} point.len={} input_mle.len={} output_mle.len={} input_rows={} input_cols={} padded_rows={} padded_cols={}",
            n, num_vars, output_claim.point.len(), input_mle.len(), output_mle.len(),
            input_matrix.rows, input_matrix.cols, input_padded.rows, input_padded.cols);
        // Show first few raw input values
        eprintln!("[ACT GPU] input_matrix[0..4] = {:?}", &input_matrix.data[..4.min(input_matrix.data.len())]);
    }
    let input_eval = gpu
        .evaluate_mle_gpu(&input_mle, &output_claim.point)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU eval_mle activation input: {e}"),
        })?;
    let output_eval = gpu
        .evaluate_mle_gpu(&output_mle, &output_claim.point)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU eval_mle activation output: {e}"),
        })?;
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[ACT GPU] input_eval={:?}", input_eval);
        eprintln!("[ACT GPU] output_eval={:?}", output_eval);
        eprintln!("[ACT GPU] output_claim.value={:?}", output_claim.value);
        // CPU cross-check
        let cpu_input_eval = evaluate_mle(&input_mle, &output_claim.point);
        let cpu_output_eval = evaluate_mle(&output_mle, &output_claim.point);
        eprintln!("[ACT GPU] CPU input_eval={:?}", cpu_input_eval);
        eprintln!("[ACT GPU] CPU output_eval={:?}", cpu_output_eval);
        if input_eval != cpu_input_eval {
            eprintln!("[ACT GPU] MISMATCH: GPU input_eval != CPU input_eval");
        }
    }

    // Build activation table
    let table_log_size = activation_type.recommended_table_log_size();
    let table = PrecomputedTable::build_parallel(|x| activation_fn(x), table_log_size);

    // Compute multiplicities (uses hash index for O(1) lookup)
    let trace_inputs_m31: Vec<M31> = input_padded.data[..n].to_vec();
    let multiplicities_m31 =
        crate::components::activation::compute_multiplicities(&trace_inputs_m31, &table);
    let multiplicities: Vec<u32> = multiplicities_m31.iter().map(|m| m.0).collect();

    // Draw LogUp encoding challenges
    channel.mix_u64(0x4C4F47 as u64); // "LOG" tag
    channel.mix_u64(activation_type.type_tag() as u64);
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // Compute denominators on GPU
    let d_vals_device = gpu
        .compute_logup_denominators_gpu(&input_mle, &output_mle, gamma, beta)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU logup denominators: {e}"),
        })?;

    // Download denominators for CPU batch inverse + table sum
    let mut d_vals_flat = vec![0u32; n * 4];
    gpu.device
        .dtoh_sync_copy_into(&d_vals_device, &mut d_vals_flat)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU download denoms: {:?}", e),
        })?;
    let d_vals: Vec<SecureField> = d_vals_flat
        .chunks_exact(4)
        .map(|c| u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
        .collect();

    // Batch inverse on CPU (1 inverse + 3N muls — fast enough)
    let w_vals = SecureField::batch_inverse(&d_vals);

    // LogUp trace-side sum
    let trace_sum: SecureField = w_vals
        .iter()
        .copied()
        .fold(SecureField::zero(), |acc, w| acc + w);

    // Table-side sum with batch inverse
    let nonzero_entries: Vec<(usize, SecureField)> = table
        .inputs
        .iter()
        .zip(&table.outputs)
        .enumerate()
        .filter(|(j, _)| multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            (j, d)
        })
        .collect();
    let table_denoms: Vec<SecureField> = nonzero_entries.iter().map(|(_, d)| *d).collect();
    let table_inv = SecureField::batch_inverse(&table_denoms);
    let table_sum: SecureField = nonzero_entries
        .iter()
        .zip(&table_inv)
        .map(|((j, _), &inv)| SecureField::from(M31::from(multiplicities[*j])) * inv)
        .fold(SecureField::zero(), |acc, v| acc + v);

    if trace_sum != table_sum {
        return Err(GKRError::LogUpError(format!(
            "LogUp sum mismatch: trace={}, table={}",
            trace_sum, table_sum,
        )));
    }

    // Mix claimed sum
    mix_secure_field(channel, trace_sum);

    // Upload eq, w, d to GPU for the eq-sumcheck loop
    let r = &output_claim.point[..num_vars];
    let eq_evals = build_eq_evals(r);

    let eq_flat: Vec<u32> = eq_evals
        .iter()
        .flat_map(|sf| secure_field_to_u32s(*sf))
        .collect();
    let w_flat: Vec<u32> = w_vals
        .iter()
        .flat_map(|sf| secure_field_to_u32s(*sf))
        .collect();

    let mut d_eq = gpu
        .device
        .htod_sync_copy(&eq_flat)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU upload eq: {:?}", e),
        })?;
    let mut d_w = gpu
        .device
        .htod_sync_copy(&w_flat)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU upload w: {:?}", e),
        })?;
    // d_vals_device already on GPU
    let mut d_d = d_vals_device;

    // GPU eq-sumcheck loop
    let mut eq_round_polys = Vec::with_capacity(num_vars);
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let (s0_u32, s1_u32, s2_u32, s3_u32) = gpu
            .compute_logup_round_poly_3way(&d_eq, &d_w, &d_d, mid)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("GPU logup round: {e}"),
            })?;

        let s0 = u32s_to_secure_field(&s0_u32);
        let s1 = u32s_to_secure_field(&s1_u32);
        let s2 = u32s_to_secure_field(&s2_u32);
        let s3 = u32s_to_secure_field(&s3_u32);

        // Newton divided differences for degree-3 interpolation
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;

        eq_round_polys.push(RoundPolyDeg3 { c0, c1, c2, c3 });

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        // GPU 3-way fold
        let challenge_u32 = secure_field_to_u32s(challenge);
        let (new_eq, new_w, new_d) = gpu
            .logup_3way_fold(&d_eq, &d_w, &d_d, cur_n, &challenge_u32)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("GPU logup fold: {e}"),
            })?;
        d_eq = new_eq;
        d_w = new_w;
        d_d = new_d;
        cur_n = mid;
    }

    // Download final w_eval (1 QM31)
    let mut w_eval_u32 = [0u32; 4];
    gpu.device
        .dtoh_sync_copy_into(&d_w, &mut w_eval_u32)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU download w_eval: {:?}", e),
        })?;
    let w_eval = u32s_to_secure_field(&w_eval_u32);

    // Final evaluations at sumcheck challenge point (GPU)
    let in_eval_s = gpu
        .evaluate_mle_gpu(&input_mle, &sumcheck_challenges)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU eval_mle final input: {e}"),
        })?;
    let out_eval_s = gpu
        .evaluate_mle_gpu(&output_mle, &sumcheck_challenges)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("GPU eval_mle final output: {e}"),
        })?;

    // Compute table commitment
    let table_commitment = compute_activation_table_commitment(activation_type, table_log_size);

    // Mix final evaluations
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let logup_proof = LogUpProof {
        eq_round_polys,
        final_evals: (w_eval, in_eval_s, out_eval_s),
        claimed_sum: trace_sum,
        multiplicities,
    };

    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    };

    Ok((
        LayerProof::Activation {
            activation_type,
            logup_proof: Some(logup_proof),
            input_eval,
            output_eval,
            table_commitment,
        },
        claim,
    ))
}

// =============================================================================
// GPU SIMD Block Combination Pipeline
// =============================================================================

/// Prove a GKR proof over N identical transformer blocks in a single pass.
///
/// Instead of running N separate GKR proofs (one per block), draws a random
/// block-selection challenge `r_simd` and combines all block outputs into a
/// single weighted MLE using Lagrange basis. This reduces verifier cost from
/// O(N × depth × log width) to O(depth × log width + log N).
///
/// For Qwen3-14B with 40 decoder blocks: 40x proof compression.
///
/// The prover:
/// 1. Seeds channel with SIMD metadata (num_blocks)
/// 2. Draws r_simd challenges → Lagrange basis weights
/// 3. Combines block outputs: combined[i] = Σ_b weight[b] * block_b[i]
/// 4. Walks layers from output → input, combining per-block intermediates at each step
/// 5. For matmul: restrict each block's A on GPU, combine, then sumcheck with shared B
/// 6. For add/mul: combine operand MLEs on GPU, evaluate combined
/// 7. Produces a standard GKRProof verifiable by `verify_gkr_simd()`
#[cfg(feature = "cuda-runtime")]
pub fn prove_gkr_simd_gpu(
    circuit: &LayeredCircuit,
    block_executions: &[GraphExecution],
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
) -> Result<GKRProof, GKRError> {
    use crate::components::matmul::compute_lagrange_basis_pub;
    use crate::gpu_sumcheck::GpuSumcheckExecutor;

    let simd_config = circuit.simd_config.as_ref().ok_or_else(|| {
        GKRError::SimdError("circuit has no SIMD config — need >= 2 identical blocks".into())
    })?;

    let n_blocks = simd_config.num_blocks;
    if block_executions.len() != n_blocks {
        return Err(GKRError::SimdError(format!(
            "expected {} block executions, got {}",
            n_blocks,
            block_executions.len(),
        )));
    }

    let gpu = GpuSumcheckExecutor::cached().map_err(|e| GKRError::ReductionError {
        layer_idx: 0,
        reason: format!("GPU init: {e}"),
    })?;

    let d = circuit.layers.len();
    let mut layer_proofs = Vec::with_capacity(d);
    let mut weight_commitments = Vec::with_capacity(d / 2);
    // Capture (weight_node_id, eval_point, final_b_eval) per MatMul.
    let mut weight_data: Vec<(usize, Vec<SecureField>, SecureField)> = Vec::with_capacity(d / 2);
    let mut deferred_info: Vec<(GKRClaim, usize)> = Vec::new();

    // Seed channel with circuit + SIMD metadata
    channel.mix_u64(d as u64);
    channel.mix_u64(circuit.input_shape.0 as u64);
    channel.mix_u64(circuit.input_shape.1 as u64);
    channel.mix_u64(n_blocks as u64);

    // Draw SIMD block-selection challenges
    let r_simd = channel.draw_qm31s(simd_config.simd_log_size);
    let block_weights = compute_lagrange_basis_pub(&r_simd);
    // Only use first n_blocks weights (rest are for padding to power-of-2)
    let block_weights = &block_weights[..n_blocks];

    // Combine block outputs into single weighted MLE
    let combined_output =
        combine_block_intermediates_output(block_executions, &block_weights, &gpu)?;

    // Build initial claim on combined output
    let combined_output_padded = pad_matrix_pow2_sf(&combined_output);
    let log_out_rows = combined_output_padded.0.ilog2() as usize;
    let log_out_cols = combined_output_padded.1.ilog2() as usize;

    let r_out = channel.draw_qm31s(log_out_rows + log_out_cols);
    let output_value = evaluate_mle(&combined_output_padded.2, &r_out);
    mix_secure_field(channel, output_value);

    let output_claim = GKRClaim {
        point: r_out,
        value: output_value,
    };
    let mut current_claim = output_claim.clone();

    // Walk layers from output → input (template block structure)
    let template = &simd_config.template_range;
    let template_layers: Vec<usize> = (template.start..template.end).rev().collect();

    for &layer_idx in &template_layers {
        let layer = &circuit.layers[layer_idx];

        let (proof, next_claim) = match &layer.layer_type {
            LayerType::MatMul {
                m,
                k,
                n,
                weight_node_id,
            } => {
                // Shared weight matrix
                let b_matrix =
                    weights
                        .get_weight(*weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: *weight_node_id,
                        })?;

                // Per-block input activations
                let offset_in_template = layer_idx - template.start;
                let block_a_matrices: Vec<&M31Matrix> = (0..n_blocks)
                    .map(|b| {
                        let node_id = simd_block_node_id(circuit, b, offset_in_template)?;
                        get_intermediate(&block_executions[b], node_id)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let (proof, claim) = reduce_matmul_layer_simd_gpu(
                    &gpu,
                    &current_claim,
                    &block_a_matrices,
                    b_matrix,
                    &block_weights,
                    *m,
                    *k,
                    *n,
                    channel,
                )?;

                let final_b_eval = match &proof {
                    LayerProof::MatMul { final_b_eval, .. } => *final_b_eval,
                    _ => unreachable!("reduce_matmul_layer_simd_gpu returns MatMul"),
                };
                push_matmul_weight_data(
                    *weight_node_id,
                    *m,
                    *n,
                    &current_claim.point,
                    &claim.point,
                    final_b_eval,
                    &mut weight_data,
                );

                (proof, claim)
            }

            LayerType::Add { .. } => {
                let (lhs_vals, rhs_vals) = get_combined_binary_op_intermediates(
                    block_executions,
                    layer_idx,
                    template.start,
                    circuit,
                    &block_weights,
                    &gpu,
                )?;
                let (proof, _claim) =
                    reduce_add_layer_gpu(&gpu, &current_claim, &lhs_vals, &rhs_vals, channel)?;

                process_add_deferred(
                    proof,
                    &current_claim,
                    &layer.input_layers,
                    &mut deferred_info,
                )
            }

            LayerType::Mul { .. } => {
                let (lhs_vals, rhs_vals) = get_combined_binary_op_intermediates(
                    block_executions,
                    layer_idx,
                    template.start,
                    circuit,
                    &block_weights,
                    &gpu,
                )?;
                reduce_mul_layer_gpu(&gpu, &current_claim, &lhs_vals, &rhs_vals, channel)?
            }

            LayerType::Activation {
                activation_type, ..
            } => {
                let combined_mle = get_combined_intermediate_mle(
                    block_executions,
                    layer_idx,
                    template.start,
                    circuit,
                    &block_weights,
                    &gpu,
                )?;

                let input_eval = gpu
                    .evaluate_mle_gpu(&combined_mle, &current_claim.point)
                    .map_err(|e| GKRError::ReductionError {
                        layer_idx,
                        reason: format!("GPU eval_mle simd activation: {e}"),
                    })?;

                mix_secure_field(channel, input_eval);
                let claim = GKRClaim {
                    point: current_claim.point.clone(),
                    value: input_eval,
                };
                (
                    LayerProof::Activation {
                        activation_type: *activation_type,
                        logup_proof: None,
                        input_eval,
                        output_eval: current_claim.value,
                        table_commitment: starknet_ff::FieldElement::ZERO,
                    },
                    claim,
                )
            }

            LayerType::LayerNorm { dim } => reduce_layernorm_layer_simd(
                &current_claim,
                block_executions,
                layer_idx,
                template.start,
                circuit,
                &block_weights,
                *dim,
                &gpu,
                channel,
            )?,

            LayerType::RMSNorm { dim } => {
                // CPU fallback for RMSNorm in SIMD path
                let first_exec = &block_executions[0];
                let input_matrix = get_intermediate(first_exec, layer.node_id)?;
                reduce_rmsnorm_layer(&current_claim, input_matrix, *dim, channel)?
            }

            LayerType::Quantize { params, .. } => {
                // CPU fallback — quantize is lookup-heavy
                let first_exec = &block_executions[0];
                let input_matrix = get_intermediate(first_exec, layer.node_id)?;
                reduce_quantize_layer(&current_claim, input_matrix, params, channel)?
            }

            LayerType::Embedding { .. } => {
                // CPU fallback — embedding appears outside SIMD blocks in practice,
                // but keep a correct path for completeness.
                let first_exec = &block_executions[0];
                let input_matrix = get_intermediate(first_exec, layer.node_id)?;
                let output_matrix = get_node_output(first_exec, layer.node_id)?;
                let embed_table =
                    weights
                        .get_weight(layer.node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: layer.node_id,
                        })?;
                reduce_embedding_layer(
                    &current_claim,
                    input_matrix,
                    output_matrix,
                    embed_table,
                    channel,
                )?
            }

            LayerType::Dequantize { params, .. } => {
                // CPU fallback — dequantize is table-lookup, not compute-bound
                let first_exec = &block_executions[0];
                let input_matrix = get_intermediate(first_exec, layer.node_id)?;
                reduce_dequantize_layer(&current_claim, input_matrix, params, channel)?
            }

            LayerType::Identity => continue,
            LayerType::Input => break,

            LayerType::Attention { config } => {
                // SIMD attention: extract per-block input matrices and decompose
                let offset_in_template = layer_idx - template.start;
                let block_input_matrices: Vec<&M31Matrix> = (0..n_blocks)
                    .map(|b| {
                        let node_id = simd_block_node_id(circuit, b, offset_in_template)?;
                        get_intermediate(&block_executions[b], node_id)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let attn_weights = get_attention_weights(weights, layer)?;

                reduce_attention_layer_simd_gpu(
                    &gpu,
                    &current_claim,
                    &block_input_matrices,
                    &attn_weights,
                    config,
                    &block_weights,
                    &r_simd,
                    channel,
                )?
            }
        };

        layer_proofs.push(proof);
        current_claim = next_claim;
    }

    // Generate MLE opening proofs for weight matrices (post-GKR-walk channel state),
    // or use aggregated RLC weight binding (off-chain verification mode),
    // or use aggregated oracle sumcheck (production on-chain mode).
    let mut weight_openings = Vec::with_capacity(weight_data.len());
    let mut weight_claims = Vec::with_capacity(weight_data.len());
    let mut weight_opening_transcript_mode = WeightOpeningTranscriptMode::Sequential;
    let mut aggregated_binding_proof = None;
    let flags = compute_weight_mode_flags();
    let aggregate_weight_binding = flags.aggregate_weight_binding;
    // In aggregated mode, include deferred MatMul weight claims in the same
    // transcript RLC binding and skip per-deferred Merkle openings.
    let mut deferred_weight_claims_data: Vec<(usize, Vec<SecureField>, SecureField)> =
        Vec::with_capacity(deferred_info.len());

    // Generate deferred proofs for skip branches of DAG Add layers BEFORE
    // weight openings. Fiat-Shamir order: walk → deferred proofs → weight openings.
    let mut deferred_proofs = Vec::with_capacity(deferred_info.len());
    for (deferred_claim, rhs_layer_idx) in &deferred_info {
        let rhs_layer = &circuit.layers[*rhs_layer_idx];
        match &rhs_layer.layer_type {
            LayerType::MatMul {
                m,
                k,
                n,
                weight_node_id,
            } => {
                // Shared weight matrix
                let b_matrix =
                    weights
                        .get_weight(*weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: *weight_node_id,
                        })?;

                // Per-block input activations for the skip layer
                let rhs_offset = *rhs_layer_idx - template.start;
                let block_a_matrices: Vec<&M31Matrix> = (0..n_blocks)
                    .map(|b| {
                        let node_id = simd_block_node_id(circuit, b, rhs_offset)?;
                        get_intermediate(&block_executions[b], node_id)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                mix_secure_field(channel, deferred_claim.value);

                let (reduction, input_claim) = reduce_matmul_layer_simd_gpu(
                    &gpu,
                    deferred_claim,
                    &block_a_matrices,
                    b_matrix,
                    &block_weights,
                    *m,
                    *k,
                    *n,
                    channel,
                )?;

                // Build weight eval point (same as push_matmul_weight_data)
                let final_b_eval = match &reduction {
                    LayerProof::MatMul { final_b_eval, .. } => *final_b_eval,
                    _ => unreachable!("reduce_matmul_layer_simd_gpu returns MatMul"),
                };
                let pm = m.next_power_of_two();
                let log_m = pm.ilog2() as usize;
                let pn = n.next_power_of_two();
                let log_n = pn.ilog2() as usize;
                let r_j = deferred_claim.point[log_m..log_m + log_n].to_vec();
                let sumcheck_challenges = &input_claim.point[log_m..];
                let mut weight_eval_point = r_j;
                weight_eval_point.extend_from_slice(sumcheck_challenges);

                let deferred_weight_claim = super::types::WeightClaim {
                    weight_node_id: *weight_node_id,
                    eval_point: weight_eval_point.clone(),
                    expected_value: final_b_eval,
                };

                // In aggregated mode, defer weight opening; otherwise prove inline
                let (deferred_weight_commitment, deferred_weight_opening) =
                    if aggregate_weight_binding {
                        deferred_weight_claims_data.push((
                            *weight_node_id,
                            weight_eval_point,
                            final_b_eval,
                        ));
                        (
                            starknet_ff::FieldElement::ZERO,
                            crate::crypto::mle_opening::MleOpeningProof {
                                intermediate_roots: Vec::new(),
                                queries: Vec::new(),
                                final_value: SecureField::zero(),
                            },
                        )
                    } else {
                        let b_mle_u32 = matrix_to_mle_col_major_u32_padded(b_matrix);
                        crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                            &b_mle_u32,
                            &deferred_weight_claim.eval_point,
                            channel,
                        )
                    };

                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof: reduction,
                    input_claim,
                    kind: super::types::DeferredProofKind::MatMul {
                        dims: (*m, *k, *n),
                        weight_commitment: deferred_weight_commitment,
                        weight_opening: deferred_weight_opening,
                        weight_claim: deferred_weight_claim,
                    },
                });
            }
            LayerType::Quantize { params, .. } => {
                // CPU fallback on block 0 (same as main walk pattern)
                let input_matrix = get_intermediate(&block_executions[0], rhs_layer.node_id)?;
                mix_secure_field(channel, deferred_claim.value);
                let (layer_proof, input_claim) =
                    reduce_quantize_layer(deferred_claim, input_matrix, params, channel)?;
                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof,
                    input_claim,
                    kind: super::types::DeferredProofKind::Weightless,
                });
            }
            LayerType::Dequantize { params, .. } => {
                // CPU fallback on block 0 (same as main walk pattern)
                let input_matrix = get_intermediate(&block_executions[0], rhs_layer.node_id)?;
                mix_secure_field(channel, deferred_claim.value);
                let (layer_proof, input_claim) =
                    reduce_dequantize_layer(deferred_claim, input_matrix, params, channel)?;
                deferred_proofs.push(super::types::DeferredProof {
                    claim: deferred_claim.clone(),
                    layer_proof,
                    input_claim,
                    kind: super::types::DeferredProofKind::Weightless,
                });
            }
            other => {
                return Err(GKRError::ReductionError {
                    layer_idx: *rhs_layer_idx,
                    reason: format!(
                        "SIMD deferred proof not supported for layer type {:?}",
                        std::mem::discriminant(other),
                    ),
                });
            }
        }
    }

    let use_aggregated_oracle_sumcheck = gkr_aggregated_oracle_sumcheck_enabled()
        && (!weight_data.is_empty() || !deferred_weight_claims_data.is_empty());

    let aggregate_weight_binding = aggregate_weight_binding
        && (!weight_data.is_empty() || !deferred_weight_claims_data.is_empty())
        && !use_aggregated_oracle_sumcheck;

    let weight_commitments_new;
    if use_aggregated_oracle_sumcheck {
        weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        let (wc, claims, proof) = apply_aggregated_oracle_sumcheck(
            &weight_data,
            &deferred_weight_claims_data,
            &mut deferred_proofs,
            weights,
            channel,
            "GKR-SIMD-GPU",
        )?;
        weight_commitments_new = wc;
        weight_claims = claims;
        aggregated_binding_proof = proof;
    } else if aggregate_weight_binding {
        weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;
        weight_claims =
            apply_aggregated_rlc_binding(&weight_data, &deferred_weight_claims_data, channel);
        weight_commitments_new = Vec::new();
    } else {
        weight_commitments_new = Vec::new();
        let opening_seed =
            if (flags.trustless_mode2 || flags.trustless_mode3) && !weight_data.is_empty() {
                Some(channel.draw_felt252())
            } else {
                None
            };
        for (opening_idx, (weight_node_id, eval_point, expected_value)) in
            weight_data.into_iter().enumerate()
        {
            let b_matrix = weights
                .get_weight(weight_node_id)
                .ok_or(GKRError::MissingWeight {
                    node_id: weight_node_id,
                })?;

            let claim = super::types::WeightClaim {
                weight_node_id,
                eval_point: eval_point.clone(),
                expected_value,
            };
            weight_claims.push(claim.clone());
            #[cfg(feature = "cuda-runtime")]
            let (commitment, opening) = {
                let b_mle_u32 = matrix_to_mle_col_major_u32_padded(b_matrix);
                if let Some(seed) = opening_seed {
                    let mut sub_channel =
                        derive_weight_opening_subchannel(seed, opening_idx, &claim);
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                        &b_mle_u32,
                        &claim.eval_point,
                        &mut sub_channel,
                    )
                } else {
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32(
                        &b_mle_u32,
                        &claim.eval_point,
                        channel,
                    )
                }
            };
            #[cfg(not(feature = "cuda-runtime"))]
            let (commitment, opening) = {
                let b_mle = matrix_to_mle_col_major_padded(b_matrix);
                if let Some(seed) = opening_seed {
                    let mut sub_channel =
                        derive_weight_opening_subchannel(seed, opening_idx, &claim);
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment(
                        &b_mle,
                        &claim.eval_point,
                        &mut sub_channel,
                    )
                } else {
                    crate::crypto::mle_opening::prove_mle_opening_with_commitment(
                        &b_mle,
                        &claim.eval_point,
                        channel,
                    )
                }
            };
            weight_commitments.push(commitment);
            weight_openings.push(opening);
        }
    }

    weight_commitments.extend(weight_commitments_new);

    if let Some(mode) = finalize_weight_transcript_mode(&flags, aggregate_weight_binding) {
        weight_opening_transcript_mode = mode;
    }

    // IO commitment from first block's input and combined output
    let model_input = block_executions[0]
        .intermediates
        .get(&0)
        .unwrap_or(&block_executions[0].output);
    let io_commitment =
        crate::aggregation::compute_io_commitment(model_input, &block_executions[0].output);

    Ok(GKRProof {
        layer_proofs,
        output_claim,
        input_claim: current_claim,
        weight_commitments,
        weight_openings,
        weight_claims,
        weight_opening_transcript_mode,
        io_commitment,
        deferred_proofs,
        aggregated_binding: aggregated_binding_proof,
    })
}

/// GPU SIMD matmul reduction: restrict each block's A, combine, then sumcheck.
///
/// For N blocks with shared weight B:
///   combined_f_a(x) = Σ_b weight[b] · Ṽ_{A_b}(r_i, x)
///   f_b(x) = Ṽ_B(x, r_j)
///   claim: Σ_x combined_f_a(x) · f_b(x) = combined output value
///
/// The combination is exact because matmul is linear in A:
///   Σ_b w_b · (A_b × B) = (Σ_b w_b · A_b) × B
#[cfg(feature = "cuda-runtime")]
fn reduce_matmul_layer_simd_gpu(
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    block_a_matrices: &[&M31Matrix],
    b: &M31Matrix,
    block_weights: &[SecureField],
    m: usize,
    k: usize,
    n: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    debug_assert!(
        m <= (1 << 30),
        "matmul m={} exceeds 2^30, next_power_of_two would overflow",
        m
    );
    debug_assert!(
        k <= (1 << 30),
        "matmul k={} exceeds 2^30, next_power_of_two would overflow",
        k
    );
    debug_assert!(
        n <= (1 << 30),
        "matmul n={} exceeds 2^30, next_power_of_two would overflow",
        n
    );

    let pm = m.next_power_of_two();
    let pk = k.next_power_of_two();
    let pn = n.next_power_of_two();

    let log_m = pm.ilog2() as usize;
    let log_k = pk.ilog2() as usize;
    let log_n = pn.ilog2() as usize;

    let total_out_vars = log_m + log_n;
    if output_claim.point.len() != total_out_vars {
        return Err(GKRError::ReductionError {
            layer_idx: 0,
            reason: format!(
                "SIMD matmul: claim has {} vars, expected exactly {}",
                output_claim.point.len(),
                total_out_vars,
            ),
        });
    }

    let r_i = &output_claim.point[..log_m];
    let r_j = &output_claim.point[log_m..log_m + log_n];

    // Mix matmul dims + claimed sum
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);
    mix_secure_field(channel, output_claim.value);

    // Step 1: Restrict each block's A on GPU, download to CPU
    let n_blocks = block_a_matrices.len();
    let mut per_block_f_a: Vec<Vec<SecureField>> = Vec::with_capacity(n_blocks);

    for a_matrix in block_a_matrices {
        let d_f_a = gpu
            .restrict_rows(a_matrix, r_i, pk)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("SIMD restrict_rows: {e}"),
            })?;

        // Download restricted vector
        let mut f_a_u32 = vec![0u32; pk * 4];
        gpu.device
            .dtoh_sync_copy_into(&d_f_a, &mut f_a_u32)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("SIMD download f_a: {:?}", e),
            })?;

        let f_a: Vec<SecureField> = f_a_u32
            .chunks_exact(4)
            .map(|c| crate::gpu_sumcheck::u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
            .collect();
        per_block_f_a.push(f_a);
    }

    // Step 2: Combine per-block f_a on GPU
    let combined_f_a = gpu
        .combine_blocks(&per_block_f_a, block_weights)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("SIMD combine_blocks: {e}"),
        })?;

    // Step 3: Restrict shared B (same for all blocks)
    let d_f_b = gpu
        .restrict_cols(b, r_j, pk)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("SIMD restrict_cols: {e}"),
        })?;

    // Step 4: CPU-channel sumcheck on (combined_f_a, f_b).
    //
    // Architecture decision: Fiat-Shamir uses Poseidon252, which has no GPU
    // kernel yet. Rather than attempt GPU-resident Fiat-Shamir (which would
    // require porting Poseidon252 hashing to CUDA), we download the restricted
    // B MLE from GPU and run the sumcheck loop on CPU. The restrict + combine
    // steps (1-3) are the GPU-accelerated hot path; the sumcheck loop is
    // O(k · log k) scalar work — fast on CPU.

    // Download restricted B MLE from GPU
    let mut f_b_u32 = vec![0u32; pk * 4];
    gpu.device
        .dtoh_sync_copy_into(&d_f_b, &mut f_b_u32)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("SIMD download f_b: {:?}", e),
        })?;
    let f_b: Vec<SecureField> = f_b_u32
        .chunks_exact(4)
        .map(|c| crate::gpu_sumcheck::u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
        .collect();

    // CPU sumcheck: degree-2 over (combined_f_a, f_b)
    let two = SecureField::from(M31::from(2u32));
    let inv2 = {
        use stwo::core::fields::FieldExpOps;
        two.inverse()
    };

    let mut f_a_cur = combined_f_a;
    let mut f_b_cur = f_b;
    let mut round_polys = Vec::with_capacity(log_k);
    let mut challenges = Vec::with_capacity(log_k);

    for _round in 0..log_k {
        let mid = f_a_cur.len() / 2;
        let mut s0 = SecureField::zero();
        let mut s1 = SecureField::zero();
        let mut s2 = SecureField::zero();

        for i in 0..mid {
            s0 += f_a_cur[i] * f_b_cur[i];
            s1 += f_a_cur[mid + i] * f_b_cur[mid + i];
            let a2 = f_a_cur[mid + i] + f_a_cur[mid + i] - f_a_cur[i];
            let b2 = f_b_cur[mid + i] + f_b_cur[mid + i] - f_b_cur[i];
            s2 += a2 * b2;
        }

        let c0 = s0;
        let c2 = (s2 - s1 - s1 + s0) * inv2;
        let c1 = s1 - s0 - c2;

        channel.mix_poly_coeffs(c0, c1, c2);
        let alpha = channel.draw_qm31();

        round_polys.push(crate::components::matmul::RoundPoly { c0, c1, c2 });
        challenges.push(alpha);

        // Fold both MLEs at the challenge point
        let mut new_fa = vec![SecureField::zero(); mid];
        let mut new_fb = vec![SecureField::zero(); mid];
        for i in 0..mid {
            new_fa[i] = f_a_cur[i] + alpha * (f_a_cur[mid + i] - f_a_cur[i]);
            new_fb[i] = f_b_cur[i] + alpha * (f_b_cur[mid + i] - f_b_cur[i]);
        }
        f_a_cur = new_fa;
        f_b_cur = new_fb;
    }

    let final_a_eval = if f_a_cur.is_empty() {
        SecureField::zero()
    } else {
        f_a_cur[0]
    };
    let final_b_eval = if f_b_cur.is_empty() {
        SecureField::zero()
    } else {
        f_b_cur[0]
    };

    mix_secure_field(channel, final_a_eval);
    mix_secure_field(channel, final_b_eval);

    let mut input_point = Vec::with_capacity(log_m + log_k);
    input_point.extend_from_slice(r_i);
    input_point.extend_from_slice(&challenges);

    let proof = LayerProof::MatMul {
        round_polys,
        final_a_eval,
        final_b_eval,
    };
    let claim = GKRClaim {
        point: input_point,
        value: final_a_eval,
    };

    Ok((proof, claim))
}

/// GPU SIMD dual-operand matmul reduction: both A and B vary per block.
///
/// For N blocks where BOTH operands vary per block (e.g., attention per-head
/// score = Q_h × K_h^T, context = softmax_h × V_h):
///
/// The linearity trick Σ w_b(A_b × B) = (Σ w_b A_b) × B fails here because
/// B also varies. Instead we use a block-extended 3-factor sumcheck:
///
///   ext_w[b*K + k] = block_weight[b]
///   ext_a[b*K + k] = Ṽ_{A_b}(r_i, k)
///   ext_b[b*K + k] = Ṽ_{B_b}(k, r_j)
///
///   claim = Σ_i ext_w[i] · ext_a[i] · ext_b[i]
///
/// This is the same structure as the eq-sumcheck in reduce_mul_layer,
/// giving us a degree-3 polynomial per round.
#[cfg(feature = "cuda-runtime")]
fn reduce_matmul_layer_dual_simd_gpu(
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    block_a_matrices: &[&M31Matrix],
    block_b_matrices: &[&M31Matrix],
    block_weights: &[SecureField],
    m: usize,
    k: usize,
    n: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, SecureField), GKRError> {
    use crate::gpu_sumcheck::secure_field_to_u32s;

    let n_blocks = block_a_matrices.len();
    assert_eq!(n_blocks, block_b_matrices.len());
    assert_eq!(n_blocks, block_weights.len());
    assert!(n_blocks.is_power_of_two(), "num_blocks must be power of 2");
    assert!(m <= (1 << 30), "matmul m={} exceeds 2^30", m);
    assert!(k <= (1 << 30), "matmul k={} exceeds 2^30", k);
    assert!(n <= (1 << 30), "matmul n={} exceeds 2^30", n);

    let pm = m.next_power_of_two();
    let pk = k.next_power_of_two();
    let pn = n.next_power_of_two();

    let log_m = pm.ilog2() as usize;
    let log_k = pk.ilog2() as usize;
    let log_n = pn.ilog2() as usize;
    let n_block_vars = n_blocks.ilog2() as usize;

    let total_out_vars = log_m + log_n;
    if output_claim.point.len() != total_out_vars {
        return Err(GKRError::ReductionError {
            layer_idx: 0,
            reason: format!(
                "dual SIMD matmul: claim has {} vars, expected exactly {}",
                output_claim.point.len(),
                total_out_vars,
            ),
        });
    }

    let r_i = &output_claim.point[..log_m];
    let r_j = &output_claim.point[log_m..log_m + log_n];

    // Mix dims + claimed sum (matching verifier)
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);
    channel.mix_u64(n_blocks as u64);
    mix_secure_field(channel, output_claim.value);

    // Step 1: Per-block GPU restrict A(r_i, ·) and B(·, r_j)
    // Restrict returns device-resident QM31 vectors of length pk.
    // Download to CPU to build extended MLEs, then re-upload for GPU sumcheck.
    let ext_len = n_blocks * pk;
    let total_vars = n_block_vars + log_k;
    assert!(ext_len.is_power_of_two());

    // Build extended MLEs as flat u32 arrays (4 u32 per QM31 element)
    let mut ext_w_u32 = vec![0u32; ext_len * 4];
    let mut ext_a_u32 = vec![0u32; ext_len * 4];
    let mut ext_b_u32 = vec![0u32; ext_len * 4];

    for b_idx in 0..n_blocks {
        // Restrict A_b rows → device vector of pk QM31 elements
        let d_f_a = gpu
            .restrict_rows(block_a_matrices[b_idx], r_i, pk)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("dual SIMD restrict_rows block {b_idx}: {e}"),
            })?;
        let mut f_a_u32 = vec![0u32; pk * 4];
        gpu.device
            .dtoh_sync_copy_into(&d_f_a, &mut f_a_u32)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("dual SIMD download f_a {b_idx}: {:?}", e),
            })?;

        // Restrict B_b cols → device vector of pk QM31 elements
        let d_f_b = gpu
            .restrict_cols(block_b_matrices[b_idx], r_j, pk)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("dual SIMD restrict_cols block {b_idx}: {e}"),
            })?;
        let mut f_b_u32 = vec![0u32; pk * 4];
        gpu.device
            .dtoh_sync_copy_into(&d_f_b, &mut f_b_u32)
            .map_err(|e| GKRError::ReductionError {
                layer_idx: 0,
                reason: format!("dual SIMD download f_b {b_idx}: {:?}", e),
            })?;

        // Fill extended MLEs: ext[b*pk + k] for each k
        let w_u32 = secure_field_to_u32s(block_weights[b_idx]);
        for ki in 0..pk {
            let base = (b_idx * pk + ki) * 4;
            ext_w_u32[base..base + 4].copy_from_slice(&w_u32);
            ext_a_u32[base..base + 4].copy_from_slice(&f_a_u32[ki * 4..(ki + 1) * 4]);
            ext_b_u32[base..base + 4].copy_from_slice(&f_b_u32[ki * 4..(ki + 1) * 4]);
        }
    }

    // Step 2: Upload extended MLEs to GPU
    let d_ext_w = gpu
        .device
        .htod_sync_copy(&ext_w_u32)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("dual SIMD upload ext_w: {:?}", e),
        })?;
    let d_ext_a = gpu
        .device
        .htod_sync_copy(&ext_a_u32)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("dual SIMD upload ext_a: {:?}", e),
        })?;
    let d_ext_b = gpu
        .device
        .htod_sync_copy(&ext_b_u32)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("dual SIMD upload ext_b: {:?}", e),
        })?;

    // Step 3: Run full 3-factor sumcheck on GPU
    let result = gpu
        .sumcheck_3way(d_ext_w, d_ext_a, d_ext_b, total_vars, channel)
        .map_err(|e| GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("dual SIMD GPU sumcheck: {e}"),
        })?;

    let final_a_eval = result.final_a;
    let final_b_eval = result.final_b;

    // Mix final evals (matching verifier)
    mix_secure_field(channel, final_a_eval);
    mix_secure_field(channel, final_b_eval);

    let proof = LayerProof::MatMulDualSimd {
        round_polys: result.round_polys,
        final_a_eval,
        final_b_eval,
        n_block_vars,
    };

    Ok((proof, output_claim.value))
}

/// Test-accessible wrapper for `reduce_matmul_layer_dual_simd_gpu`.
#[cfg(all(test, feature = "cuda-runtime"))]
pub fn reduce_matmul_layer_dual_simd_gpu_for_test(
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    block_a_matrices: &[&M31Matrix],
    block_b_matrices: &[&M31Matrix],
    block_weights: &[SecureField],
    m: usize,
    k: usize,
    n: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, SecureField), GKRError> {
    reduce_matmul_layer_dual_simd_gpu(
        gpu,
        output_claim,
        block_a_matrices,
        block_b_matrices,
        block_weights,
        m,
        k,
        n,
        channel,
    )
}

// ===== SIMD Helpers =====

/// Combine all block outputs into a single weighted MLE.
#[cfg(feature = "cuda-runtime")]
fn combine_block_intermediates_output(
    block_executions: &[GraphExecution],
    block_weights: &[SecureField],
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
) -> Result<M31Matrix, GKRError> {
    // For now, the combined output is an M31Matrix for compatibility with
    // pad_matrix_pow2. We compute it on CPU since the output is typically small.
    let rows = block_executions[0].output.rows;
    let cols = block_executions[0].output.cols;

    // Build per-block output MLEs
    let block_mles: Vec<Vec<SecureField>> = block_executions
        .iter()
        .map(|exec| {
            let padded = pad_matrix_pow2(&exec.output);
            matrix_to_mle(&padded)
        })
        .collect();

    // Combine on GPU
    let _combined_mle = gpu
        .combine_blocks(&block_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine block outputs: {e}")))?;

    // Convert back to M31Matrix (truncated to real output dims)
    // The MLE is in QM31, but for the initial claim we just need the MLE vector
    // Return a placeholder matrix — the actual evaluation uses the MLE directly
    let result = M31Matrix::new(rows, cols);
    // Store zeros — the actual combined MLE is returned via pad_matrix_pow2_sf
    Ok(result)
}

/// Pad an MLE to power-of-2 dimensions and return (rows, cols, mle_values).
/// Used for combined SIMD outputs that are already in SecureField.
#[cfg(feature = "cuda-runtime")]
fn pad_matrix_pow2_sf(matrix: &M31Matrix) -> (usize, usize, Vec<SecureField>) {
    let padded = pad_matrix_pow2(matrix);
    let mle = matrix_to_mle(&padded);
    (padded.rows, padded.cols, mle)
}

/// Get the combined MLE for a single intermediate at a given layer.
/// Collects per-block intermediates, converts to MLE, combines on GPU.
#[cfg(feature = "cuda-runtime")]
fn get_combined_intermediate_mle(
    block_executions: &[GraphExecution],
    layer_idx: usize,
    template_start: usize,
    circuit: &LayeredCircuit,
    block_weights: &[SecureField],
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
) -> Result<Vec<SecureField>, GKRError> {
    let n_blocks = block_executions.len();
    let offset = layer_idx - template_start;

    let block_mles: Vec<Vec<SecureField>> = (0..n_blocks)
        .map(|b| {
            let node_id = simd_block_node_id(circuit, b, offset)?;
            let matrix = get_intermediate(&block_executions[b], node_id)?;
            let padded = pad_matrix_pow2(matrix);
            Ok(matrix_to_mle(&padded))
        })
        .collect::<Result<Vec<_>, GKRError>>()?;

    gpu.combine_blocks(&block_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine intermediates: {e}")))
}

/// Get combined binary op (Add/Mul) intermediate MLEs for SIMD.
#[cfg(feature = "cuda-runtime")]
fn get_combined_binary_op_intermediates(
    block_executions: &[GraphExecution],
    layer_idx: usize,
    template_start: usize,
    circuit: &LayeredCircuit,
    block_weights: &[SecureField],
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
) -> Result<(Vec<SecureField>, Vec<SecureField>), GKRError> {
    let n_blocks = block_executions.len();
    let layer = &circuit.layers[layer_idx];

    if layer.input_layers.len() < 2 {
        return Err(GKRError::ReductionError {
            layer_idx,
            reason: format!("binary op needs 2 inputs, got {}", layer.input_layers.len()),
        });
    }

    // Get the template's input layer offsets
    let lhs_layer_idx = layer.input_layers[0];
    let rhs_layer_idx = layer.input_layers[1];
    let lhs_offset = lhs_layer_idx - template_start;
    let rhs_offset = rhs_layer_idx - template_start;

    let lhs_mles: Vec<Vec<SecureField>> = (0..n_blocks)
        .map(|b| {
            let block_lhs_idx = circuit.block_ranges[b].start + lhs_offset;
            let node_id = circuit.layers[block_lhs_idx].node_id;
            // Prefer node_outputs (correct: output of input layer)
            let matrix = if let Some(m) = block_executions[b].node_outputs.get(&node_id) {
                m
            } else {
                get_intermediate(&block_executions[b], node_id)?
            };
            let padded = pad_matrix_pow2(matrix);
            Ok(matrix_to_mle(&padded))
        })
        .collect::<Result<Vec<_>, GKRError>>()?;

    let rhs_mles: Vec<Vec<SecureField>> = (0..n_blocks)
        .map(|b| {
            let block_rhs_idx = circuit.block_ranges[b].start + rhs_offset;
            let node_id = circuit.layers[block_rhs_idx].node_id;
            // Prefer node_outputs (correct: output of input layer)
            let matrix = if let Some(m) = block_executions[b].node_outputs.get(&node_id) {
                m
            } else {
                get_intermediate(&block_executions[b], node_id)?
            };
            let padded = pad_matrix_pow2(matrix);
            Ok(matrix_to_mle(&padded))
        })
        .collect::<Result<Vec<_>, GKRError>>()?;

    let combined_lhs = gpu
        .combine_blocks(&lhs_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine lhs: {e}")))?;
    let combined_rhs = gpu
        .combine_blocks(&rhs_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine rhs: {e}")))?;

    Ok((combined_lhs, combined_rhs))
}

// ===== Per-Layer Reduction Functions (CPU) =====

/// Intermediate matmul proof data before packaging into LayerProof.
struct MatMulReduction {
    round_polys: Vec<crate::components::matmul::RoundPoly>,
    final_a_eval: SecureField,
    final_b_eval: SecureField,
}

/// Backend-dispatched MatMul reduction used for phased CPU/GPU unification.
fn reduce_matmul_layer_with_backend<B: crate::backend::ZkmlOps>(
    output_claim: &GKRClaim,
    a: &M31Matrix,
    b: &M31Matrix,
    m: usize,
    k: usize,
    n: usize,
    channel: &mut PoseidonChannel,
) -> Result<(MatMulReduction, GKRClaim), GKRError> {
    // Preserve CPU reducer semantics: derive variable counts from actual
    // matrix shapes (after conceptual pow2 padding), while keeping the
    // original layer dims for transcript mixing.
    assert!(a.rows <= (1 << 30), "matmul rows={} exceeds 2^30", a.rows);
    assert!(a.cols <= (1 << 30), "matmul cols={} exceeds 2^30", a.cols);
    assert!(b.cols <= (1 << 30), "matmul b.cols={} exceeds 2^30", b.cols);
    let pm = a.rows.next_power_of_two();
    let pk = a.cols.next_power_of_two();
    let pn = b.cols.next_power_of_two();

    let log_m = pm.ilog2() as usize;
    let log_n = pn.ilog2() as usize;

    let total_out_vars = log_m + log_n;
    if output_claim.point.len() != total_out_vars {
        return Err(GKRError::ReductionError {
            layer_idx: 0,
            reason: format!(
                "output claim has {} vars, expected exactly {} (log_m={} + log_n={})",
                output_claim.point.len(),
                total_out_vars,
                log_m,
                log_n
            ),
        });
    }

    let r_i = &output_claim.point[..log_m];
    let r_j = &output_claim.point[log_m..log_m + log_n];

    // Mix matmul dims and claimed sum into transcript.
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[MatMul] ch BEFORE seeding: {:?}", channel.digest());
        eprintln!("[MatMul] m={}, k={}, n={}, claim={:?}", m, k, n, output_claim.value);
    }
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);
    mix_secure_field(channel, output_claim.value);
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[MatMul] ch after seeding: {:?}", channel.digest());
    }

    let reduction = B::reduce_matmul_layer(a, b, r_i, r_j, pk, channel).map_err(|e| {
        GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("MatMul reduction backend failure: {e}"),
        }
    })?;

    // Bind final evaluations to transcript.
    mix_secure_field(channel, reduction.final_a_eval);
    mix_secure_field(channel, reduction.final_b_eval);

    let mut input_point = Vec::with_capacity(log_m + reduction.challenges.len());
    input_point.extend_from_slice(r_i);
    input_point.extend_from_slice(&reduction.challenges);

    Ok((
        MatMulReduction {
            round_polys: reduction.round_polys,
            final_a_eval: reduction.final_a_eval,
            final_b_eval: reduction.final_b_eval,
        },
        GKRClaim {
            point: input_point,
            value: reduction.final_a_eval,
        },
    ))
}

/// Reduce a MatMul layer via sumcheck over the inner dimension (CPU path).
///
/// Delegates to `reduce_matmul_layer_with_backend::<SimdBackend>`.
fn reduce_matmul_layer(
    output_claim: &GKRClaim,
    a: &M31Matrix,
    b: &M31Matrix,
    m: usize,
    k: usize,
    n: usize,
    channel: &mut PoseidonChannel,
) -> Result<(MatMulReduction, GKRClaim), GKRError> {
    reduce_matmul_layer_with_backend::<stwo::prover::backend::simd::SimdBackend>(
        output_claim,
        a,
        b,
        m,
        k,
        n,
        channel,
    )
}

/// Reduce an Add layer using a backend-generic MLE evaluator.
///
/// By linearity of MLE: Ṽ_c(r) = Ṽ_a(r) + Ṽ_b(r).
/// Prover sends (a_eval, b_eval), verifier checks a_eval + b_eval == claimed.
///
/// For DAG circuits (residual connections), the main walk follows the lhs
/// branch. The rhs branch gets a deferred proof. Alpha is still drawn for
/// Fiat-Shamir transcript binding but is NOT used to combine claims.
fn reduce_add_layer_with_backend<B: crate::backend::ZkmlOps>(
    output_claim: &GKRClaim,
    lhs_vals: &[SecureField],
    rhs_vals: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    let lhs_eval = B::evaluate_mle_at_point(lhs_vals, &output_claim.point).map_err(|e| {
        GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("Add lhs MLE eval: {e}"),
        }
    })?;
    let rhs_eval = B::evaluate_mle_at_point(rhs_vals, &output_claim.point).map_err(|e| {
        GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("Add rhs MLE eval: {e}"),
        }
    })?;

    // Mix evaluations into channel for Fiat-Shamir binding
    mix_secure_field(channel, lhs_eval);
    mix_secure_field(channel, rhs_eval);

    // Draw alpha for transcript binding (NOT used for claim combination
    // in DAG circuits — the rhs gets a separate deferred proof instead)
    let _alpha = channel.draw_qm31();

    // Main walk follows the lhs branch. The rhs branch claim (r, rhs_eval)
    // will be stored as a deferred claim by the caller.
    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: lhs_eval,
    };

    Ok((
        LayerProof::Add {
            lhs_eval,
            rhs_eval,
            trunk_idx: 0,
        },
        claim,
    ))
}

/// Reduce an Add layer (CPU path via SimdBackend).
fn reduce_add_layer(
    output_claim: &GKRClaim,
    lhs_vals: &[SecureField],
    rhs_vals: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_add_layer_with_backend::<stwo::prover::backend::simd::SimdBackend>(
        output_claim,
        lhs_vals,
        rhs_vals,
        channel,
    )
}

/// Reduce a Mul layer via eq-sumcheck: c[i] = a[i] * b[i].
///
/// Proves: Ṽ_c(r) = Σ_{x ∈ {0,1}^n} eq(r,x) · Ṽ_a(x) · Ṽ_b(x)
///
/// This is a degree-3 sumcheck (eq × a × b). At each round, we evaluate
/// the round polynomial at 4 points (t=0,1,2,3) and Lagrange-interpolate
/// to get degree-3 coefficients (c0, c1, c2, c3).
///
/// After the sumcheck, the final check is:
///   claimed_eval == eq(r, challenges) · lhs_eval · rhs_eval
fn reduce_mul_layer(
    output_claim: &GKRClaim,
    lhs_vals: &[SecureField],
    rhs_vals: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::RoundPolyDeg3;
    use stwo::core::fields::FieldExpOps;

    let n = lhs_vals.len();
    assert!(n.is_power_of_two(), "MLE size must be power of 2");
    assert_eq!(n, rhs_vals.len(), "lhs and rhs must have same length");

    let num_vars = n.ilog2() as usize;
    let r = &output_claim.point;
    assert!(r.len() >= num_vars, "claim point too short for MLE size");

    // Build eq(r, x) for all x ∈ {0,1}^n via tensor product
    let mut eq_evals = build_eq_evals(&r[..num_vars]);

    // Copy the MLE values (we fold these during sumcheck)
    let mut f_a = lhs_vals.to_vec();
    let mut f_b = rhs_vals.to_vec();

    // Mix layer type tag and claimed sum into channel
    channel.mix_u64(0x4D554C as u64); // "MUL" tag
    mix_secure_field(channel, output_claim.value);

    let mut eq_round_polys = Vec::with_capacity(num_vars);
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        // Evaluate round polynomial at t = 0, 1, 2, 3
        let s0 = compute_mul_eq_sum_at_t(&eq_evals, &f_a, &f_b, mid, SecureField::zero());
        let s1 = compute_mul_eq_sum_at_t(&eq_evals, &f_a, &f_b, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &f_a, &f_b, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &f_a, &f_b, mid, three);

        // Lagrange interpolation via Newton divided differences
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let d1 = s1 - s0;
        let d2 = (s2 - s1 - s1 + s0) * inv2;
        let d3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = d1 - d2 + two * d3;
        let c2 = d2 - three * d3;
        let c3 = d3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        eq_round_polys.push(rp);

        // Fiat-Shamir: mix round poly, draw challenge
        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        // Fold all three arrays
        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut f_a, challenge, mid);
        fold_mle(&mut f_b, challenge, mid);
        cur_n = mid;
    }

    // Final evaluations
    assert_eq!(f_a.len(), 1);
    assert_eq!(f_b.len(), 1);
    assert_eq!(eq_evals.len(), 1);
    let lhs_eval = f_a[0];
    let rhs_eval = f_b[0];

    // Mix final evals
    mix_secure_field(channel, lhs_eval);
    mix_secure_field(channel, rhs_eval);

    // Draw combiner for reducing two claims (lhs, rhs) into one
    let alpha = channel.draw_qm31();
    let combined = alpha * lhs_eval + (SecureField::one() - alpha) * rhs_eval;

    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: combined,
    };

    Ok((
        LayerProof::Mul {
            eq_round_polys,
            lhs_eval,
            rhs_eval,
        },
        claim,
    ))
}

/// Test-accessible wrapper for `reduce_mul_layer`.
pub fn reduce_mul_layer_for_test(
    output_claim: &GKRClaim,
    lhs_vals: &[SecureField],
    rhs_vals: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_mul_layer(output_claim, lhs_vals, rhs_vals, channel)
}

/// Reduce an Activation layer via LogUp eq-sumcheck.
///
/// Proves that every (input[i], output[i]) pair exists in the precomputed
/// activation table using the LogUp argument + degree-3 eq-sumcheck.
///
/// Protocol:
/// 1. Build table T for activation function, draw encoding challenges γ, β
/// 2. Compute d_i = γ - in_i - β·out_i, w_i = 1/d_i
/// 3. Verify LogUp sum: Σ w_i = Σ mult_j/(γ - table_encode_j)
/// 4. Eq-sumcheck: Σ eq(r,x)·w(x)·d(x) = 1  (proves w·d = 1 at all boolean points)
fn reduce_activation_layer(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    activation_type: crate::components::activation::ActivationType,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::{LogUpProof, RoundPolyDeg3};
    use crate::gadgets::lookup_table::PrecomputedTable;
    use stwo::core::fields::FieldExpOps;

    let activation_fn = activation_type.as_fn();

    // Pad input matrix and build MLEs
    let input_padded = pad_matrix_pow2(input_matrix);
    let n = input_padded.rows * input_padded.cols;
    let num_vars = n.ilog2() as usize;

    let input_mle = matrix_to_mle(&input_padded);

    // Compute output by applying activation to each input element
    let output_mle: Vec<SecureField> = input_padded
        .data
        .iter()
        .take(n)
        .map(|&v| SecureField::from(activation_fn(v)))
        .collect();

    // Evaluate MLEs at claim point
    let input_eval = evaluate_mle(&input_mle, &output_claim.point);
    let output_eval = evaluate_mle(&output_mle, &output_claim.point);

    // Build activation table (deterministic from activation type + log size)
    let table_log_size = activation_type.recommended_table_log_size();
    let table = PrecomputedTable::build_parallel(|x| activation_fn(x), table_log_size);

    // Compute multiplicities (how many trace entries use each table row)
    let trace_inputs_m31: Vec<M31> = input_padded.data[..n].to_vec();
    let multiplicities_m31 =
        crate::components::activation::compute_multiplicities(&trace_inputs_m31, &table);
    let multiplicities: Vec<u32> = multiplicities_m31.iter().map(|m| m.0).collect();

    // Draw LogUp encoding challenges
    channel.mix_u64(0x4C4F47 as u64); // "LOG" tag
    channel.mix_u64(activation_type.type_tag() as u64);
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // Compute denominators and inverse witnesses
    let d_vals: Vec<SecureField> = input_mle
        .iter()
        .zip(&output_mle)
        .map(|(&inp, &out)| gamma - inp - beta * out)
        .collect();

    // Batch inverse: 1 inverse + 3N muls instead of N individual inverses (~12x speedup)
    let w_vals = SecureField::batch_inverse(&d_vals);

    // Compute LogUp trace-side sum: T = Σ w_i
    let trace_sum: SecureField = w_vals
        .iter()
        .copied()
        .fold(SecureField::zero(), |acc, w| acc + w);

    // Compute table-side sum: S = Σ mult_j / (γ - table_encode_j)
    // Collect non-zero denominators and batch-inverse them
    let nonzero_entries: Vec<(usize, SecureField)> = table
        .inputs
        .iter()
        .zip(&table.outputs)
        .enumerate()
        .filter(|(j, _)| multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            (j, d)
        })
        .collect();
    let table_denoms: Vec<SecureField> = nonzero_entries.iter().map(|(_, d)| *d).collect();
    let table_inv = SecureField::batch_inverse(&table_denoms);
    let table_sum: SecureField = nonzero_entries
        .iter()
        .zip(&table_inv)
        .map(|((j, _), &inv)| SecureField::from(M31::from(multiplicities[*j])) * inv)
        .fold(SecureField::zero(), |acc, v| acc + v);

    if trace_sum != table_sum {
        return Err(GKRError::LogUpError(format!(
            "LogUp sum mismatch: trace={}, table={}",
            trace_sum, table_sum,
        )));
    }

    // Mix claimed sum into channel
    mix_secure_field(channel, trace_sum);

    // Build eq(r, x) for the claim point
    let r = &output_claim.point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut w_folded = w_vals;
    let mut d_folded = d_vals;

    // Run degree-3 eq-sumcheck: Σ eq(r,x) · w(x) · d(x) = 1
    let mut eq_round_polys = Vec::with_capacity(num_vars);
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let s0 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::zero());
        let s1 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, three);

        // Newton divided differences for degree-3 Lagrange interpolation
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        eq_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut w_folded, challenge, mid);
        fold_mle(&mut d_folded, challenge, mid);
        cur_n = mid;
    }

    // Final evaluations at sumcheck challenge point
    assert_eq!(w_folded.len(), 1);
    let w_eval = w_folded[0];
    let in_eval_s = evaluate_mle(&input_mle, &sumcheck_challenges);
    let out_eval_s = evaluate_mle(&output_mle, &sumcheck_challenges);

    // Compute table commitment
    let table_commitment = compute_activation_table_commitment(activation_type, table_log_size);

    // Mix final evaluations into channel
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let logup_proof = LogUpProof {
        eq_round_polys,
        final_evals: (w_eval, in_eval_s, out_eval_s),
        claimed_sum: trace_sum,
        multiplicities,
    };

    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    };

    Ok((
        LayerProof::Activation {
            activation_type,
            logup_proof: Some(logup_proof),
            input_eval,
            output_eval,
            table_commitment,
        },
        claim,
    ))
}

/// Compute a deterministic commitment for an activation table.
/// Since the table is fully determined by (activation_type, table_log_size),
/// the commitment is a Poseidon hash of these parameters.
fn compute_activation_table_commitment(
    activation_type: crate::components::activation::ActivationType,
    table_log_size: u32,
) -> starknet_ff::FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        starknet_ff::FieldElement::from(activation_type.type_tag() as u64),
        starknet_ff::FieldElement::from(table_log_size as u64),
    ])
}

/// Test-accessible wrapper for `reduce_activation_layer`.
pub fn reduce_activation_layer_for_test(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    activation_type: crate::components::activation::ActivationType,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_activation_layer(output_claim, input_matrix, activation_type, channel)
}

/// Test-accessible wrapper for `reduce_activation_layer_gpu`.
#[cfg(feature = "cuda-runtime")]
pub fn reduce_activation_layer_gpu_for_test(
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    activation_type: crate::components::activation::ActivationType,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_activation_layer_gpu(gpu, output_claim, input_matrix, activation_type, channel)
}

// =============================================================================
// Dequantize Reduction
// =============================================================================

/// Reduce a Dequantize layer via LogUp eq-sumcheck.
///
/// Nearly identical to `reduce_activation_layer` but:
/// - 2-element relation: (quantized_input, dequantized_output)
/// - Table built from QuantParams (deterministic: INT4→16 entries, INT8→256 entries)
/// - "DEQLOG" Fiat-Shamir tag instead of "LOG"
///
/// Protocol:
/// 1. Build table T from QuantParams, draw encoding challenges γ, β
/// 2. Compute d_i = γ - in_i - β·out_i, w_i = 1/d_i
/// 3. Verify LogUp sum: Σ w_i = Σ mult_j/(γ - table_encode_j)
/// 4. Eq-sumcheck: Σ eq(r,x)·w(x)·d(x) = 1
fn reduce_dequantize_layer(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    params: &crate::gadgets::quantize::QuantParams,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::{LogUpProof, RoundPolyDeg3};
    use crate::components::dequantize::build_dequantize_table;
    use crate::gadgets::quantize::{dequantize_value, quantize_value, QuantParams, QuantStrategy};
    use stwo::core::fields::FieldExpOps;

    // Pad input matrix and build MLEs
    let input_padded = pad_matrix_pow2(input_matrix);
    let n = input_padded.rows * input_padded.cols;
    let num_vars = n.ilog2() as usize;

    let input_mle = matrix_to_mle(&input_padded);

    // Compute dequantized output: q → dequantize(q) → re-encode as M31 via Direct
    let direct_params = QuantParams {
        strategy: QuantStrategy::Direct,
        scale: 1.0,
        zero_point: 0,
        bits: 31,
    };
    let output_mle: Vec<SecureField> = input_padded
        .data
        .iter()
        .take(n)
        .map(|&v| {
            let f = dequantize_value(v, params);
            SecureField::from(quantize_value(f, &direct_params))
        })
        .collect();

    // Evaluate MLEs at claim point
    let input_eval = evaluate_mle(&input_mle, &output_claim.point);
    let output_eval = evaluate_mle(&output_mle, &output_claim.point);

    // Build dequantize table (deterministic from params)
    let table = build_dequantize_table(params);

    // Compute multiplicities
    let trace_inputs_m31: Vec<M31> = input_padded.data[..n].to_vec();
    let multiplicities_m31 =
        crate::components::activation::compute_multiplicities(&trace_inputs_m31, &table);
    let multiplicities: Vec<u32> = multiplicities_m31.iter().map(|m| m.0).collect();

    // Draw LogUp encoding challenges with dequantize-specific tag
    channel.mix_u64(0x4445514C4F47_u64); // "DEQLOG" tag
    channel.mix_u64(params.bits as u64);
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // Compute denominators and inverse witnesses (batch inverse: ~12x faster)
    let d_vals: Vec<SecureField> = input_mle
        .iter()
        .zip(&output_mle)
        .map(|(&inp, &out)| gamma - inp - beta * out)
        .collect();

    let w_vals = SecureField::batch_inverse(&d_vals);

    // Compute LogUp trace-side sum: T = Σ w_i
    let trace_sum: SecureField = w_vals
        .iter()
        .copied()
        .fold(SecureField::zero(), |acc, w| acc + w);

    // Compute table-side sum with batch inverse
    let nonzero_entries: Vec<(usize, SecureField)> = table
        .inputs
        .iter()
        .zip(&table.outputs)
        .enumerate()
        .filter(|(j, _)| multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            (j, d)
        })
        .collect();
    let table_denoms: Vec<SecureField> = nonzero_entries.iter().map(|(_, d)| *d).collect();
    let table_inv = SecureField::batch_inverse(&table_denoms);
    let table_sum: SecureField = nonzero_entries
        .iter()
        .zip(&table_inv)
        .map(|((j, _), &inv)| SecureField::from(M31::from(multiplicities[*j])) * inv)
        .fold(SecureField::zero(), |acc, v| acc + v);

    if trace_sum != table_sum {
        return Err(GKRError::LogUpError(format!(
            "Dequantize LogUp sum mismatch: trace={}, table={}",
            trace_sum, table_sum,
        )));
    }

    // Mix claimed sum into channel
    mix_secure_field(channel, trace_sum);

    // Build eq(r, x) for the claim point
    let r = &output_claim.point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut w_folded = w_vals;
    let mut d_folded = d_vals;

    // Run degree-3 eq-sumcheck: Σ eq(r,x) · w(x) · d(x) = 1
    let mut eq_round_polys = Vec::with_capacity(num_vars);
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let s0 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::zero());
        let s1 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, three);

        // Newton divided differences for degree-3 Lagrange interpolation
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        eq_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut w_folded, challenge, mid);
        fold_mle(&mut d_folded, challenge, mid);
        cur_n = mid;
    }

    // Final evaluations at sumcheck challenge point
    assert_eq!(w_folded.len(), 1);
    let w_eval = w_folded[0];
    let in_eval_s = evaluate_mle(&input_mle, &sumcheck_challenges);
    let out_eval_s = evaluate_mle(&output_mle, &sumcheck_challenges);

    // Compute table commitment
    let table_commitment = compute_dequantize_table_commitment(params);

    // Mix final evaluations into channel
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let logup_proof = LogUpProof {
        eq_round_polys,
        final_evals: (w_eval, in_eval_s, out_eval_s),
        claimed_sum: trace_sum,
        multiplicities,
    };

    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    };

    Ok((
        LayerProof::Dequantize {
            logup_proof: Some(logup_proof),
            input_eval,
            output_eval,
            table_commitment,
        },
        claim,
    ))
}

/// Test-accessible wrapper for `reduce_dequantize_layer`.
pub fn reduce_dequantize_layer_for_test(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    params: &crate::gadgets::quantize::QuantParams,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_dequantize_layer(output_claim, input_matrix, params, channel)
}

/// Test-accessible wrapper for `reduce_quantize_layer`.
pub fn reduce_quantize_layer_for_test(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    params: &crate::gadgets::quantize::QuantParams,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_quantize_layer(output_claim, input_matrix, params, channel)
}

/// Test-accessible wrapper for `reduce_embedding_layer`.
pub fn reduce_embedding_layer_for_test(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    output_matrix: &M31Matrix,
    embedding_table: &M31Matrix,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_embedding_layer(
        output_claim,
        input_matrix,
        output_matrix,
        embedding_table,
        channel,
    )
}

/// Compute a deterministic commitment for a dequantize table.
/// The table is fully determined by (bits, scale, zero_point), so the
/// commitment is a Poseidon hash of these parameters.
pub fn compute_dequantize_table_commitment(
    params: &crate::gadgets::quantize::QuantParams,
) -> starknet_ff::FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        starknet_ff::FieldElement::from(0x4445515F5441424C45_u128), // "DEQ_TABLE"
        starknet_ff::FieldElement::from(params.bits as u64),
        starknet_ff::FieldElement::from(params.zero_point.unsigned_abs() as u64),
        // Encode scale as integer bits (multiply by 2^32 and truncate)
        starknet_ff::FieldElement::from((params.scale * (1u64 << 32) as f64) as u64),
    ])
}

// =============================================================================
// Quantize Reduction
// =============================================================================

fn quantize_strategy_tag(strategy: crate::gadgets::quantize::QuantStrategy) -> u64 {
    use crate::gadgets::quantize::QuantStrategy;
    match strategy {
        QuantStrategy::Direct => 0,
        QuantStrategy::Symmetric8 => 1,
        QuantStrategy::Asymmetric8 => 2,
        QuantStrategy::Symmetric4 => 3,
        QuantStrategy::Asymmetric4 => 4,
    }
}

fn project_claim_point(point: &[SecureField], n_vars: usize) -> Vec<SecureField> {
    let mut out = point.to_vec();
    if out.len() > n_vars {
        out.truncate(n_vars);
    } else if out.len() < n_vars {
        out.resize(n_vars, SecureField::zero());
    }
    out
}

/// Reduce a Quantize layer via LogUp eq-sumcheck.
///
/// Mirrors Dequantize reduction but uses the quantize table relation
/// `(input, quantized_output)`.
fn reduce_quantize_layer(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    params: &crate::gadgets::quantize::QuantParams,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::{LogUpProof, RoundPolyDeg3};
    use crate::components::quantize::build_quantize_table;
    use crate::gadgets::quantize::{dequantize_value, quantize_value, QuantParams, QuantStrategy};
    use stwo::core::fields::FieldExpOps;

    // Pad input matrix and build MLEs
    let input_padded = pad_matrix_pow2(input_matrix);
    let n = input_padded.rows * input_padded.cols;
    let num_vars = n.ilog2() as usize;
    let out_point = project_claim_point(&output_claim.point, num_vars);
    let input_mle = matrix_to_mle(&input_padded);

    // Compute quantized output in M31 field representation.
    let direct_params = QuantParams {
        strategy: QuantStrategy::Direct,
        scale: 1.0,
        zero_point: 0,
        bits: 31,
    };
    let output_mle: Vec<SecureField> = input_padded
        .data
        .iter()
        .take(n)
        .map(|&v| {
            let f = dequantize_value(v, &direct_params);
            SecureField::from(quantize_value(f, params))
        })
        .collect();

    let input_eval = evaluate_mle(&input_mle, &out_point);
    let output_eval = evaluate_mle(&output_mle, &out_point);

    // Build quantize table from observed inputs and multiplicities over padded trace.
    let table = build_quantize_table(params, &input_padded.data[..n]);
    let mut multiplicities = vec![0u32; table.size()];
    for &inp in &input_padded.data[..n] {
        if let Some(idx) = table.lookup_index(inp) {
            multiplicities[idx] += 1;
        }
    }

    // Draw LogUp encoding challenges with quantize-specific domain separation.
    channel.mix_u64(0x514C4F47_u64); // "QLOG"
    channel.mix_u64(params.bits as u64);
    channel.mix_u64(params.zero_point.unsigned_abs() as u64);
    channel.mix_u64((params.scale * (1u64 << 32) as f64) as u64);
    channel.mix_u64(quantize_strategy_tag(params.strategy));
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // Compute denominators and inverse witnesses.
    let d_vals: Vec<SecureField> = input_mle
        .iter()
        .zip(&output_mle)
        .map(|(&inp, &out)| gamma - inp - beta * out)
        .collect();
    let w_vals = SecureField::batch_inverse(&d_vals);

    let trace_sum: SecureField = w_vals
        .iter()
        .copied()
        .fold(SecureField::zero(), |acc, w| acc + w);

    // Table-side sum.
    let nonzero_entries: Vec<(usize, SecureField)> = table
        .inputs
        .iter()
        .zip(&table.outputs)
        .enumerate()
        .filter(|(j, _)| multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            (j, d)
        })
        .collect();
    let table_denoms: Vec<SecureField> = nonzero_entries.iter().map(|(_, d)| *d).collect();
    let table_inv = SecureField::batch_inverse(&table_denoms);
    let table_sum: SecureField = nonzero_entries
        .iter()
        .zip(&table_inv)
        .map(|((j, _), &inv)| SecureField::from(M31::from(multiplicities[*j])) * inv)
        .fold(SecureField::zero(), |acc, v| acc + v);

    if trace_sum != table_sum {
        return Err(GKRError::LogUpError(format!(
            "Quantize LogUp sum mismatch: trace={}, table={}",
            trace_sum, table_sum
        )));
    }

    mix_secure_field(channel, trace_sum);

    // Degree-3 eq-sumcheck: Σ eq(r,x) · w(x) · d(x) = 1
    let r = &out_point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut w_folded = w_vals;
    let mut d_folded = d_vals;
    let mut eq_round_polys = Vec::with_capacity(num_vars);
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;
        let s0 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::zero());
        let s1 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, three);

        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();
        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;
        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;
        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        eq_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut w_folded, challenge, mid);
        fold_mle(&mut d_folded, challenge, mid);
        cur_n = mid;
    }

    let w_eval = w_folded[0];
    let in_eval_s = evaluate_mle(&input_mle, &sumcheck_challenges);
    let out_eval_s = evaluate_mle(&output_mle, &sumcheck_challenges);

    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let logup_proof = LogUpProof {
        eq_round_polys,
        final_evals: (w_eval, in_eval_s, out_eval_s),
        claimed_sum: trace_sum,
        multiplicities,
    };

    let claim = GKRClaim {
        point: out_point,
        value: input_eval,
    };

    Ok((
        LayerProof::Quantize {
            logup_proof: Some(logup_proof),
            input_eval,
            output_eval,
            table_inputs: table.inputs.clone(),
            table_outputs: table.outputs.clone(),
        },
        claim,
    ))
}

// =============================================================================
// Embedding Reduction
// =============================================================================

/// Reduce an Embedding layer with a LogUp relation on (token_id, col_idx, value).
///
/// Verifier side reconstructs table values from model weights using sparse
/// multiplicities carried in the proof.
fn reduce_embedding_layer(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    output_matrix: &M31Matrix,
    embedding_table: &M31Matrix,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use std::collections::HashMap;

    use super::types::{EmbeddingLogUpProof, RoundPolyDeg3};
    use stwo::core::fields::FieldExpOps;

    if embedding_table.rows == 0 || embedding_table.cols == 0 {
        return Err(GKRError::ReductionError {
            layer_idx: 0,
            reason: "embedding table is empty".to_string(),
        });
    }

    let token_ids: Vec<u32> = input_matrix
        .data
        .iter()
        .map(|m| (m.0 as usize).min(embedding_table.rows - 1) as u32)
        .collect();

    if token_ids.is_empty() {
        return Err(GKRError::ReductionError {
            layer_idx: 0,
            reason: "embedding input is empty".to_string(),
        });
    }

    let out_rows = output_matrix.rows;
    let out_cols = output_matrix.cols;
    if out_rows == 0 || out_cols == 0 {
        return Err(GKRError::ReductionError {
            layer_idx: 0,
            reason: "embedding output is empty".to_string(),
        });
    }

    let padded_rows = out_rows.next_power_of_two();
    let padded_cols = out_cols.next_power_of_two();
    let n = padded_rows * padded_cols;
    let num_vars = n.ilog2() as usize;
    let out_point = project_claim_point(&output_claim.point, num_vars);

    let default_tok = 0u32;
    let default_col = 0u32;
    let default_val = embedding_table.get(0, 0);
    let mut token_trace = vec![SecureField::from(M31::from(default_tok)); n];
    let mut col_trace = vec![SecureField::from(M31::from(default_col)); n];
    let mut value_trace = vec![SecureField::from(default_val); n];

    let mut sparse_mults: HashMap<(u32, u32), u32> = HashMap::new();

    for row in 0..out_rows {
        let tok = token_ids
            .get(row)
            .copied()
            .unwrap_or(default_tok)
            .min((embedding_table.rows - 1) as u32);
        for col in 0..out_cols {
            let idx = row * padded_cols + col;
            let col_u32 = col as u32;
            token_trace[idx] = SecureField::from(M31::from(tok));
            col_trace[idx] = SecureField::from(M31::from(col_u32));
            value_trace[idx] = SecureField::from(output_matrix.get(row, col));
            *sparse_mults.entry((tok, col_u32)).or_insert(0) += 1;
        }
    }

    // Pad with a valid table cell so table-side sums remain well-defined.
    let mut padded_uses = 0u32;
    for row in 0..padded_rows {
        let row_used_cols = if row < out_rows { out_cols } else { 0 };
        padded_uses += (padded_cols - row_used_cols) as u32;
    }
    if padded_uses > 0 {
        *sparse_mults.entry((default_tok, default_col)).or_insert(0) += padded_uses;
    }

    let output_eval = evaluate_mle(&value_trace, &out_point);

    // Input claim is projected to the input MLE variable dimension.
    let input_padded = pad_matrix_pow2(input_matrix);
    let input_mle = matrix_to_mle(&input_padded);
    let input_num_vars = input_mle.len().ilog2() as usize;
    let input_point = project_claim_point(&output_claim.point, input_num_vars);
    let input_eval = evaluate_mle(&input_mle, &input_point);

    channel.mix_u64(0x454D424C4F47_u64); // "EMBLOG"
    channel.mix_u64(embedding_table.rows as u64);
    channel.mix_u64(embedding_table.cols as u64);
    let gamma = channel.draw_qm31();
    let beta_col = channel.draw_qm31();
    let beta_val = channel.draw_qm31();

    let d_vals: Vec<SecureField> = token_trace
        .iter()
        .zip(&col_trace)
        .zip(&value_trace)
        .map(|((&tok, &col), &val)| gamma - tok - beta_col * col - beta_val * val)
        .collect();
    let w_vals = SecureField::batch_inverse(&d_vals);
    let trace_sum: SecureField = w_vals
        .iter()
        .copied()
        .fold(SecureField::zero(), |acc, w| acc + w);

    let mut sparse_entries: Vec<(u32, u32, u32)> = sparse_mults
        .into_iter()
        .map(|((tok, col), mult)| (tok, col, mult))
        .collect();
    sparse_entries.sort_unstable_by_key(|(tok, col, _)| (*tok, *col));

    let table_denoms: Vec<SecureField> = sparse_entries
        .iter()
        .map(|(tok, col, _)| {
            let row = (*tok as usize).min(embedding_table.rows - 1);
            let c = (*col as usize).min(embedding_table.cols - 1);
            let val = SecureField::from(embedding_table.get(row, c));
            gamma
                - SecureField::from(M31::from(*tok))
                - beta_col * SecureField::from(M31::from(*col))
                - beta_val * val
        })
        .collect();
    let table_inv = SecureField::batch_inverse(&table_denoms);
    let table_sum: SecureField = sparse_entries
        .iter()
        .zip(&table_inv)
        .map(|((_, _, mult), inv)| SecureField::from(M31::from(*mult)) * *inv)
        .fold(SecureField::zero(), |acc, v| acc + v);

    if trace_sum != table_sum {
        return Err(GKRError::LogUpError(format!(
            "Embedding LogUp sum mismatch: trace={}, table={}",
            trace_sum, table_sum
        )));
    }

    mix_secure_field(channel, trace_sum);

    // Degree-3 eq-sumcheck: Σ eq(r,x) · w(x) · d(x) = 1
    let r = &out_point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut w_folded = w_vals;
    let mut d_folded = d_vals;
    let mut eq_round_polys = Vec::with_capacity(num_vars);
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;
        let s0 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::zero());
        let s1 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &w_folded, &d_folded, mid, three);

        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();
        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;
        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;
        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        eq_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut w_folded, challenge, mid);
        fold_mle(&mut d_folded, challenge, mid);
        cur_n = mid;
    }

    let w_eval = w_folded[0];
    let tok_eval_s = evaluate_mle(&token_trace, &sumcheck_challenges);
    let col_eval_s = evaluate_mle(&col_trace, &sumcheck_challenges);
    let val_eval_s = evaluate_mle(&value_trace, &sumcheck_challenges);

    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let mut table_tokens = Vec::with_capacity(sparse_entries.len());
    let mut table_cols = Vec::with_capacity(sparse_entries.len());
    let mut multiplicities = Vec::with_capacity(sparse_entries.len());
    for (tok, col, mult) in sparse_entries {
        table_tokens.push(tok);
        table_cols.push(col);
        multiplicities.push(mult);
    }

    let logup_proof = EmbeddingLogUpProof {
        eq_round_polys,
        final_evals: (w_eval, tok_eval_s, col_eval_s, val_eval_s),
        claimed_sum: trace_sum,
        table_tokens,
        table_cols,
        multiplicities,
    };

    let claim = GKRClaim {
        point: input_point,
        value: input_eval,
    };

    Ok((
        LayerProof::Embedding {
            logup_proof: Some(logup_proof),
            input_eval,
            output_eval,
            input_num_vars,
        },
        claim,
    ))
}

// =============================================================================
// LayerNorm Reduction
// =============================================================================

/// Reduce a LayerNorm layer via:
/// 1. Degree-3 eq-sumcheck proving output = (input - mean) × rsqrt
/// 2. LogUp eq-sumcheck proving (variance, rsqrt) ∈ rsqrt_table
///
/// LayerNorm: y[i] = (x[i] - mean_row) × rsqrt(var_row)
/// where mean_row and rsqrt_var are constant per row.
fn reduce_layernorm_layer(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    dim: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::{LogUpProof, RoundPolyDeg3};
    use crate::components::layernorm::{build_rsqrt_table, LayerNormConfig};
    use stwo::core::fields::FieldExpOps;

    let config = LayerNormConfig::new(dim);
    let rsqrt_table = build_rsqrt_table(config.rsqrt_table_log_size);

    // Pad input matrix and build MLE
    let input_padded = pad_matrix_pow2(input_matrix);
    let n = input_padded.rows * input_padded.cols;
    let num_vars = n.ilog2() as usize;
    let cols = input_padded.cols;

    let input_mle = matrix_to_mle(&input_padded);

    // Compute LayerNorm forward pass: mean, variance, rsqrt per row
    let n_active = dim.min(input_padded.cols);
    let inv_n = m31_mod_inverse(n_active as u32);

    let mut mean_mle = vec![SecureField::zero(); n];
    let mut rsqrt_mle = vec![SecureField::zero(); n];
    let mut var_mle = vec![SecureField::zero(); n];
    let mut output_mle = vec![SecureField::zero(); n];
    let mut centered_mle = vec![SecureField::zero(); n];

    for row in 0..input_padded.rows {
        // Mean: sum(x) / n over active columns
        let mut sum = M31::from(0u32);
        for col in 0..n_active {
            sum = sum + input_padded.get(row, col);
        }
        let mean = sum * inv_n;
        let mean_sf = SecureField::from(mean);

        // Variance: sum((x - mean)^2) / n
        let mut var_sum = M31::from(0u32);
        for col in 0..n_active {
            let diff = input_padded.get(row, col) - mean;
            var_sum = var_sum + diff * diff;
        }
        // Reduce variance to rsqrt_table range [0, 2^table_log_size) for LogUp.
        let variance_raw = var_sum * inv_n;
        let variance = M31::from(variance_raw.0 & ((1u32 << config.rsqrt_table_log_size) - 1));
        let variance_sf = SecureField::from(variance);

        let rsqrt = rsqrt_table.lookup(variance).ok_or_else(|| {
            GKRError::LookupTableError(format!(
                "LayerNorm rsqrt lookup failed: variance {} (raw {}) exceeds table range 2^{}",
                variance.0, variance_raw.0, config.rsqrt_table_log_size,
            ))
        })?;
        let rsqrt_sf = SecureField::from(rsqrt);

        for col in 0..cols {
            let idx = row * cols + col;
            mean_mle[idx] = mean_sf;
            var_mle[idx] = variance_sf;
            rsqrt_mle[idx] = rsqrt_sf;

            if col < n_active {
                let x = input_padded.get(row, col);
                let centered = x - mean;
                let out_val = centered * rsqrt;
                centered_mle[idx] = SecureField::from(centered);
                output_mle[idx] = SecureField::from(out_val);
            } else {
                // Padding: identity passthrough
                let x = input_padded.get(row, col);
                centered_mle[idx] = SecureField::from(x);
                output_mle[idx] = SecureField::from(x);
            }
        }
    }

    // Evaluate all MLEs at claim point
    let input_eval = evaluate_mle(&input_mle, &output_claim.point);
    let output_eval = evaluate_mle(&output_mle, &output_claim.point);
    let mean_eval = evaluate_mle(&mean_mle, &output_claim.point);
    let rsqrt_eval = evaluate_mle(&rsqrt_mle, &output_claim.point);

    // ===== Part 1: Linear transform eq-sumcheck =====
    // Proves: output_claim.value = Σ_x eq(r,x) · centered(x) · rsqrt(x)
    channel.mix_u64(0x4C4E as u64); // "LN" tag
    mix_secure_field(channel, mean_eval);
    mix_secure_field(channel, rsqrt_eval);
    mix_secure_field(channel, output_claim.value);

    let r = &output_claim.point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut centered_folded = centered_mle.clone();
    let mut rsqrt_folded = rsqrt_mle.clone();
    let mut linear_round_polys = Vec::with_capacity(num_vars);
    let mut linear_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let s0 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &centered_folded,
            &rsqrt_folded,
            mid,
            SecureField::zero(),
        );
        let s1 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &centered_folded,
            &rsqrt_folded,
            mid,
            SecureField::one(),
        );
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &centered_folded, &rsqrt_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &centered_folded, &rsqrt_folded, mid, three);

        // Newton divided differences for degree-3 Lagrange interpolation
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        linear_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        linear_challenges.push(challenge);

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut centered_folded, challenge, mid);
        fold_mle(&mut rsqrt_folded, challenge, mid);
        cur_n = mid;
    }

    // Final evaluations at sumcheck challenge point
    assert_eq!(centered_folded.len(), 1);
    let centered_final = centered_folded[0];
    let rsqrt_final = rsqrt_folded[0];
    let linear_final_evals = (centered_final, rsqrt_final);

    // Mix final linear evals
    mix_secure_field(channel, centered_final);
    mix_secure_field(channel, rsqrt_final);

    // ===== Part 2: rsqrt LogUp eq-sumcheck =====
    // Proves: all (variance[i], rsqrt[i]) pairs are in the rsqrt table.
    // Reuses the same LogUp structure as activation.

    let table = &rsqrt_table;

    // Compute multiplicities for rsqrt table
    let var_m31: Vec<M31> = var_mle
        .iter()
        .map(|v| {
            // Extract M31 from SecureField (first component)
            M31::from(v.0 .0 .0)
        })
        .collect();
    let multiplicities_m31 = crate::components::activation::compute_multiplicities(&var_m31, table);
    let multiplicities: Vec<u32> = multiplicities_m31.iter().map(|m| m.0).collect();

    // Draw LogUp encoding challenges
    channel.mix_u64(0x4C4F47 as u64); // "LOG" tag
    channel.mix_u64(0x5253 as u64); // "RS" (rsqrt) type tag
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // Compute denominators and inverse witnesses for LogUp (batch inverse: ~12x faster)
    let d_vals: Vec<SecureField> = var_mle
        .iter()
        .zip(&rsqrt_mle)
        .map(|(&v, &rs)| gamma - v - beta * rs)
        .collect();

    let w_vals = SecureField::batch_inverse(&d_vals);

    // Trace-side sum
    let trace_sum: SecureField = w_vals
        .iter()
        .copied()
        .fold(SecureField::zero(), |acc, w| acc + w);

    // Table-side sum with batch inverse
    let nonzero_entries: Vec<(usize, SecureField)> = table
        .inputs
        .iter()
        .zip(&table.outputs)
        .enumerate()
        .filter(|(j, _)| multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            (j, d)
        })
        .collect();
    let table_denoms: Vec<SecureField> = nonzero_entries.iter().map(|(_, d)| *d).collect();
    let table_inv = SecureField::batch_inverse(&table_denoms);
    let table_sum: SecureField = nonzero_entries
        .iter()
        .zip(&table_inv)
        .map(|((j, _), &inv)| SecureField::from(M31::from(multiplicities[*j])) * inv)
        .fold(SecureField::zero(), |acc, v| acc + v);

    if trace_sum != table_sum {
        return Err(GKRError::LogUpError(format!(
            "LayerNorm LogUp sum mismatch: trace={}, table={}",
            trace_sum, table_sum,
        )));
    }

    mix_secure_field(channel, trace_sum);

    // LogUp eq-sumcheck: Σ eq(r_logup, x) · w(x) · d(x) = 1
    let r_logup = &output_claim.point[..num_vars];
    let mut eq_evals_logup = build_eq_evals(r_logup);
    let mut w_folded = w_vals;
    let mut d_folded = d_vals;
    let mut logup_round_polys = Vec::with_capacity(num_vars);
    let mut logup_challenges = Vec::with_capacity(num_vars);
    let mut cur_n_logup = n;

    for _ in 0..num_vars {
        let mid = cur_n_logup / 2;

        let s0 = compute_mul_eq_sum_at_t(
            &eq_evals_logup,
            &w_folded,
            &d_folded,
            mid,
            SecureField::zero(),
        );
        let s1 = compute_mul_eq_sum_at_t(
            &eq_evals_logup,
            &w_folded,
            &d_folded,
            mid,
            SecureField::one(),
        );
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals_logup, &w_folded, &d_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals_logup, &w_folded, &d_folded, mid, three);

        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        logup_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        logup_challenges.push(challenge);

        fold_mle(&mut eq_evals_logup, challenge, mid);
        fold_mle(&mut w_folded, challenge, mid);
        fold_mle(&mut d_folded, challenge, mid);
        cur_n_logup = mid;
    }

    assert_eq!(w_folded.len(), 1);
    let w_eval = w_folded[0];
    // CRITICAL FIX: evaluate var/rsqrt MLEs at the LogUp sumcheck challenge
    // point, NOT the linear sumcheck challenges. The verifier checks
    // d(s) = gamma - var(s_logup) - beta*rsqrt(s_logup) against eq(r, s_logup).
    let var_eval_s = evaluate_mle(&var_mle, &logup_challenges);
    let rsqrt_eval_s = evaluate_mle(&rsqrt_mle, &logup_challenges);

    let logup_proof = LogUpProof {
        eq_round_polys: logup_round_polys,
        final_evals: (w_eval, var_eval_s, rsqrt_eval_s),
        claimed_sum: trace_sum,
        multiplicities,
    };

    // Compute rsqrt table commitment
    let rsqrt_table_commitment = compute_rsqrt_table_commitment(config.rsqrt_table_log_size);

    // Mix final evals
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    };

    Ok((
        LayerProof::LayerNorm {
            logup_proof: Some(logup_proof),
            linear_round_polys,
            linear_final_evals,
            input_eval,
            output_eval,
            mean: mean_eval,
            rsqrt_var: rsqrt_eval,
            rsqrt_table_commitment,
            simd_combined: false,
        },
        claim,
    ))
}

/// Compute a deterministic commitment for a rsqrt lookup table.
pub fn compute_rsqrt_table_commitment(table_log_size: u32) -> starknet_ff::FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        starknet_ff::FieldElement::from(0x5253_u64), // "RS" tag for rsqrt
        starknet_ff::FieldElement::from(table_log_size as u64),
    ])
}

/// Modular inverse of n in M31 via Fermat's little theorem.
fn m31_mod_inverse(n: u32) -> M31 {
    if n == 0 {
        return M31::from(0u32);
    }
    let p: u64 = (1u64 << 31) - 1;
    let mut result: u64 = 1;
    let mut base = n as u64 % p;
    let mut exp = p - 2;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % p;
        }
        base = base * base % p;
        exp >>= 1;
    }
    M31::from(result as u32)
}

/// Test-accessible wrapper for `reduce_layernorm_layer`.
pub fn reduce_layernorm_layer_for_test(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    dim: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_layernorm_layer(output_claim, input_matrix, dim, channel)
}

/// Reduce an RMSNorm layer via:
/// 1. Degree-3 eq-sumcheck proving output = input × rsqrt(rms²)
/// 2. LogUp eq-sumcheck proving (rms², rsqrt) ∈ rsqrt_table
fn reduce_rmsnorm_layer(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    dim: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::{LogUpProof, RoundPolyDeg3};
    use crate::components::rmsnorm::{build_rsqrt_table, RMSNormConfig};
    use stwo::core::fields::FieldExpOps;

    let config = RMSNormConfig::new(dim);
    let rsqrt_table = build_rsqrt_table(config.rsqrt_table_log_size);

    let input_padded = pad_matrix_pow2(input_matrix);
    let n = input_padded.rows * input_padded.cols;
    let num_vars = n.ilog2() as usize;
    let cols = input_padded.cols;
    let input_mle = matrix_to_mle(&input_padded);
    let n_active = dim.min(input_padded.cols);
    let inv_n = m31_mod_inverse(n_active as u32);

    let mut rsqrt_mle = vec![SecureField::zero(); n];
    let mut rms_sq_mle = vec![SecureField::zero(); n];
    let mut output_mle = vec![SecureField::zero(); n];

    for row in 0..input_padded.rows {
        let mut sq_sum = M31::from(0u32);
        for col in 0..n_active {
            let x = input_padded.get(row, col);
            sq_sum = sq_sum + x * x;
        }
        // Reduce rms_sq to rsqrt_table range for LogUp.
        let rms_sq_raw = sq_sum * inv_n;
        let rms_sq = M31::from(rms_sq_raw.0 & ((1u32 << config.rsqrt_table_log_size) - 1));
        let rsqrt = rsqrt_table.lookup(rms_sq).ok_or_else(|| {
            GKRError::LookupTableError(format!(
                "RMSNorm rsqrt lookup failed: rms_sq {} (raw {}) exceeds table range 2^{}",
                rms_sq.0, rms_sq_raw.0, config.rsqrt_table_log_size,
            ))
        })?;

        for col in 0..cols {
            let idx = row * cols + col;
            rms_sq_mle[idx] = SecureField::from(rms_sq);
            rsqrt_mle[idx] = SecureField::from(rsqrt);
            let x = input_padded.get(row, col);
            output_mle[idx] = if col < n_active {
                SecureField::from(x * rsqrt)
            } else {
                SecureField::from(x)
            };
        }
    }

    let input_eval = evaluate_mle(&input_mle, &output_claim.point);
    let output_eval = evaluate_mle(&output_mle, &output_claim.point);
    let rsqrt_eval = evaluate_mle(&rsqrt_mle, &output_claim.point);
    let rms_sq_eval = evaluate_mle(&rms_sq_mle, &output_claim.point);

    // Part 1: eq-sumcheck: output = input × rsqrt
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[RMSNorm] ch BEFORE RN: {:?}", channel.digest());
        eprintln!("[RMSNorm] rms_sq_eval: {:?}", rms_sq_eval);
        eprintln!("[RMSNorm] rsqrt_eval: {:?}", rsqrt_eval);
        eprintln!("[RMSNorm] claim: {:?}", output_claim.value);
        eprintln!("[RMSNorm] input_eval: {:?}", input_eval);
        eprintln!("[RMSNorm] output_eval: {:?}", output_eval);
    }
    channel.mix_u64(0x524E as u64); // "RN" tag
    mix_secure_field(channel, rms_sq_eval);
    mix_secure_field(channel, rsqrt_eval);
    mix_secure_field(channel, output_claim.value);
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[RMSNorm] ch after mix claim: {:?}", channel.digest());
    }

    let r = &output_claim.point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut input_folded = input_mle.clone();
    let mut rsqrt_folded = rsqrt_mle.clone();
    let mut linear_round_polys = Vec::with_capacity(num_vars);
    let mut linear_challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;
        let s0 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &input_folded,
            &rsqrt_folded,
            mid,
            SecureField::zero(),
        );
        let s1 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &input_folded,
            &rsqrt_folded,
            mid,
            SecureField::one(),
        );
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &input_folded, &rsqrt_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &input_folded, &rsqrt_folded, mid, three);

        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();
        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;
        linear_round_polys.push(RoundPolyDeg3 { c0, c1, c2, c3 });

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        linear_challenges.push(challenge);

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut input_folded, challenge, mid);
        fold_mle(&mut rsqrt_folded, challenge, mid);
        cur_n = mid;
    }

    let input_final = input_folded[0];
    let rsqrt_final = rsqrt_folded[0];
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[RMSNorm] ch after {} eq-rounds: {:?}", num_vars, channel.digest());
        eprintln!("[RMSNorm] input_final: {:?}", input_final);
        eprintln!("[RMSNorm] rsqrt_final: {:?}", rsqrt_final);
    }
    mix_secure_field(channel, input_final);
    mix_secure_field(channel, rsqrt_final);
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[RMSNorm] ch after final evals: {:?}", channel.digest());
    }

    // Part 2: rsqrt LogUp eq-sumcheck
    let rms_sq_m31: Vec<M31> = rms_sq_mle.iter().map(|v| M31::from(v.0 .0 .0)).collect();
    let mults_m31 =
        crate::components::activation::compute_multiplicities(&rms_sq_m31, &rsqrt_table);
    let multiplicities: Vec<u32> = mults_m31.iter().map(|m| m.0).collect();

    channel.mix_u64(0x4C4F47 as u64); // "LOG"
    channel.mix_u64(0x524E as u64); // "RN"
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    let d_vals: Vec<SecureField> = rms_sq_mle
        .iter()
        .zip(&rsqrt_mle)
        .map(|(&v, &rs)| gamma - v - beta * rs)
        .collect();
    let w_vals = SecureField::batch_inverse(&d_vals);
    let trace_sum: SecureField = w_vals.iter().copied().sum();

    let nonzero: Vec<(usize, SecureField)> = rsqrt_table
        .inputs
        .iter()
        .zip(&rsqrt_table.outputs)
        .enumerate()
        .filter(|(j, _)| multiplicities[*j] > 0)
        .map(|(j, (&ti, &to))| {
            (
                j,
                gamma - SecureField::from(ti) - beta * SecureField::from(to),
            )
        })
        .collect();
    let tbl_inv = SecureField::batch_inverse(&nonzero.iter().map(|(_, d)| *d).collect::<Vec<_>>());
    let table_sum: SecureField = nonzero
        .iter()
        .zip(&tbl_inv)
        .map(|((j, _), &inv)| SecureField::from(M31::from(multiplicities[*j])) * inv)
        .sum();

    if trace_sum != table_sum {
        return Err(GKRError::LogUpError(format!(
            "RMSNorm LogUp sum mismatch: trace={}, table={}",
            trace_sum, table_sum,
        )));
    }
    mix_secure_field(channel, trace_sum);
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[RMSNorm] ch after mix claimed_sum: {:?}", channel.digest());
        eprintln!("[RMSNorm] claimed_sum (trace_sum): {:?}", trace_sum);
    }

    let r_logup = &output_claim.point[..num_vars];
    let mut eq_logup = build_eq_evals(r_logup);
    let mut w_f = w_vals;
    let mut d_f = d_vals;
    let mut logup_rps = Vec::with_capacity(num_vars);
    let mut logup_chals = Vec::with_capacity(num_vars);
    let mut cur_n2 = n;

    for _ in 0..num_vars {
        let mid = cur_n2 / 2;
        let s0 = compute_mul_eq_sum_at_t(&eq_logup, &w_f, &d_f, mid, SecureField::zero());
        let s1 = compute_mul_eq_sum_at_t(&eq_logup, &w_f, &d_f, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_logup, &w_f, &d_f, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_logup, &w_f, &d_f, mid, three);

        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();
        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;
        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;
        logup_rps.push(RoundPolyDeg3 { c0, c1, c2, c3 });

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let ch = channel.draw_qm31();
        logup_chals.push(ch);

        fold_mle(&mut eq_logup, ch, mid);
        fold_mle(&mut w_f, ch, mid);
        fold_mle(&mut d_f, ch, mid);
        cur_n2 = mid;
    }

    let w_eval = w_f[0];
    let rms_eval_s = evaluate_mle(&rms_sq_mle, &logup_chals);
    let rsqrt_eval_s = evaluate_mle(&rsqrt_mle, &logup_chals);

    let logup_proof = LogUpProof {
        eq_round_polys: logup_rps,
        final_evals: (w_eval, rms_eval_s, rsqrt_eval_s),
        claimed_sum: trace_sum,
        multiplicities,
    };

    let rsqrt_table_commitment = compute_rsqrt_table_commitment(config.rsqrt_table_log_size);

    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[RMSNorm] ch after logup (before input/output mix): {:?}", channel.digest());
        eprintln!("[RMSNorm] logup eq_rounds: {}", logup_proof.eq_round_polys.len());
    }

    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);
    if std::env::var("STWO_CHANNEL_TRACE").is_ok() {
        eprintln!("[RMSNorm] ch FINAL: {:?}", channel.digest());
    }

    Ok((
        LayerProof::RMSNorm {
            logup_proof: Some(logup_proof),
            linear_round_polys,
            linear_final_evals: (input_final, rsqrt_final),
            input_eval,
            output_eval,
            rms_sq_eval,
            rsqrt_eval,
            rsqrt_table_commitment,
            simd_combined: false,
        },
        GKRClaim {
            point: output_claim.point.clone(),
            value: input_eval,
        },
    ))
}

/// Test-accessible wrapper for `reduce_rmsnorm_layer`.
pub fn reduce_rmsnorm_layer_for_test(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    dim: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_rmsnorm_layer(output_claim, input_matrix, dim, channel)
}

/// SIMD reduction for LayerNorm via combined-product approach.
///
/// For batched identical blocks, LayerNorm's non-linearity (mean, rsqrt)
/// prevents the standard factorization used by MatMul/Mul. Instead we define:
///   combined_product[i] = Σ_b w_b · (centered_b[i] * rsqrt_b[i])
/// which equals the combined output on the boolean hypercube.
///
/// The eq-sumcheck proves:
///   output_claim.value = Σ_x eq(r,x) · combined_product(x) · 1
/// where the constant-1 MLE serves as the rsqrt factor.
///
/// LogUp is skipped (logup_proof: None) since combined variance/rsqrt
/// are QM31 sums that don't map to individual table entries.
#[cfg(feature = "cuda-runtime")]
fn reduce_layernorm_layer_simd(
    output_claim: &GKRClaim,
    block_executions: &[GraphExecution],
    layer_idx: usize,
    template_start: usize,
    circuit: &LayeredCircuit,
    block_weights: &[SecureField],
    dim: usize,
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::RoundPolyDeg3;
    use crate::components::layernorm::{build_rsqrt_table, LayerNormConfig};
    use stwo::core::fields::FieldExpOps;

    let config = LayerNormConfig::new(dim);
    let rsqrt_table = build_rsqrt_table(config.rsqrt_table_log_size);

    let n_blocks = block_executions.len();
    let offset = layer_idx - template_start;

    // Phase 1: Per-block forward pass on CPU — build product, mean, rsqrt, input MLEs
    let mut block_product_mles = Vec::with_capacity(n_blocks);
    let mut block_mean_mles = Vec::with_capacity(n_blocks);
    let mut block_rsqrt_mles = Vec::with_capacity(n_blocks);
    let mut block_input_mles = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let node_id = simd_block_node_id(circuit, b, offset)?;
        let input_matrix = get_intermediate(&block_executions[b], node_id)?;
        let padded = pad_matrix_pow2(input_matrix);
        let n = padded.rows * padded.cols;
        let cols = padded.cols;
        let n_active = dim.min(cols);
        let inv_n = m31_mod_inverse(n_active as u32);

        let mut product_mle = vec![SecureField::zero(); n];
        let mut mean_mle = vec![SecureField::zero(); n];
        let mut rsqrt_mle_b = vec![SecureField::zero(); n];
        let input_mle = matrix_to_mle(&padded);

        for row in 0..padded.rows {
            let mut sum = M31::from(0u32);
            for col in 0..n_active {
                sum = sum + padded.get(row, col);
            }
            let mean = sum * inv_n;

            let mut var_sum = M31::from(0u32);
            for col in 0..n_active {
                let diff = padded.get(row, col) - mean;
                var_sum = var_sum + diff * diff;
            }
            // Reduce variance to table range for LogUp consistency.
            let variance_raw = var_sum * inv_n;
            let variance = M31::from(variance_raw.0 & ((1u32 << config.rsqrt_table_log_size) - 1));
            let rsqrt = rsqrt_table.lookup(variance).ok_or_else(|| {
                GKRError::LookupTableError(format!(
                    "SIMD LayerNorm rsqrt lookup failed: variance {} (raw {}) exceeds table range 2^{}",
                    variance.0, variance_raw.0, config.rsqrt_table_log_size,
                ))
            })?;

            let mean_sf = SecureField::from(mean);
            let rsqrt_sf = SecureField::from(rsqrt);

            for col in 0..cols {
                let idx = row * cols + col;
                mean_mle[idx] = mean_sf;
                rsqrt_mle_b[idx] = rsqrt_sf;

                if col < n_active {
                    let x = padded.get(row, col);
                    let centered = x - mean;
                    product_mle[idx] = SecureField::from(centered * rsqrt);
                } else {
                    product_mle[idx] = SecureField::from(padded.get(row, col));
                }
            }
        }

        block_product_mles.push(product_mle);
        block_mean_mles.push(mean_mle);
        block_rsqrt_mles.push(rsqrt_mle_b);
        block_input_mles.push(input_mle);
    }

    // Phase 2: GPU combine
    let combined_product = gpu
        .combine_blocks(&block_product_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine product: {e}")))?;
    let combined_mean = gpu
        .combine_blocks(&block_mean_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine mean: {e}")))?;
    let combined_rsqrt = gpu
        .combine_blocks(&block_rsqrt_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine rsqrt: {e}")))?;
    let combined_input = gpu
        .combine_blocks(&block_input_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine input: {e}")))?;

    let n = combined_product.len();
    let num_vars = n.ilog2() as usize;

    // Phase 3: Evaluate at claim point
    let mean_eval = gpu
        .evaluate_mle_gpu(&combined_mean, &output_claim.point)
        .map_err(|e| GKRError::ReductionError {
            layer_idx,
            reason: format!("GPU eval_mle mean: {e}"),
        })?;
    let rsqrt_eval = gpu
        .evaluate_mle_gpu(&combined_rsqrt, &output_claim.point)
        .map_err(|e| GKRError::ReductionError {
            layer_idx,
            reason: format!("GPU eval_mle rsqrt: {e}"),
        })?;
    let input_eval = gpu
        .evaluate_mle_gpu(&combined_input, &output_claim.point)
        .map_err(|e| GKRError::ReductionError {
            layer_idx,
            reason: format!("GPU eval_mle input: {e}"),
        })?;
    let output_eval = output_claim.value;

    // Phase 4: Channel — match prover/verifier Fiat-Shamir transcript
    channel.mix_u64(0x4C4E as u64); // "LN" tag
    mix_secure_field(channel, mean_eval);
    mix_secure_field(channel, rsqrt_eval);
    mix_secure_field(channel, output_claim.value);

    // Phase 5: Degree-3 eq-sumcheck over combined_product × ones
    let r = &output_claim.point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut centered_folded = combined_product;
    // Constant-1 MLE: (1-r)*1 + r*1 = 1 at every fold, stays all-ones
    let mut rsqrt_folded = vec![SecureField::one(); n];
    let mut linear_round_polys = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let s0 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &centered_folded,
            &rsqrt_folded,
            mid,
            SecureField::zero(),
        );
        let s1 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &centered_folded,
            &rsqrt_folded,
            mid,
            SecureField::one(),
        );
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &centered_folded, &rsqrt_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &centered_folded, &rsqrt_folded, mid, three);

        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        linear_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut centered_folded, challenge, mid);
        fold_mle(&mut rsqrt_folded, challenge, mid);
        cur_n = mid;
    }

    assert_eq!(centered_folded.len(), 1);
    let centered_final = centered_folded[0];
    let rsqrt_final = rsqrt_folded[0];
    let linear_final_evals = (centered_final, rsqrt_final);

    // Mix final linear evals
    mix_secure_field(channel, centered_final);
    mix_secure_field(channel, rsqrt_final);

    // Skip Part 2 (LogUp) — no channel operations

    // Mix input/output evals
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let rsqrt_table_commitment = compute_rsqrt_table_commitment(config.rsqrt_table_log_size);

    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    };

    Ok((
        LayerProof::LayerNorm {
            logup_proof: None,
            linear_round_polys,
            linear_final_evals,
            input_eval,
            output_eval,
            mean: mean_eval,
            rsqrt_var: rsqrt_eval,
            rsqrt_table_commitment,
            simd_combined: true,
        },
        claim,
    ))
}

/// Test-accessible SIMD LayerNorm reduction (CPU only, no GPU needed).
///
/// Takes pre-combined MLEs and runs the degree-3 eq-sumcheck over
/// `combined_product × ones` with `logup_proof: None`.
pub fn reduce_layernorm_simd_for_test(
    output_claim: &GKRClaim,
    combined_product: Vec<SecureField>,
    combined_mean: &[SecureField],
    combined_rsqrt: &[SecureField],
    combined_input: &[SecureField],
    dim: usize,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    use super::types::RoundPolyDeg3;
    use crate::components::layernorm::LayerNormConfig;
    use stwo::core::fields::FieldExpOps;

    let config = LayerNormConfig::new(dim);
    let n = combined_product.len();
    let num_vars = n.ilog2() as usize;

    let mean_eval = evaluate_mle(combined_mean, &output_claim.point);
    let rsqrt_eval = evaluate_mle(combined_rsqrt, &output_claim.point);
    let input_eval = evaluate_mle(combined_input, &output_claim.point);
    let output_eval = output_claim.value;

    channel.mix_u64(0x4C4E as u64); // "LN" tag
    mix_secure_field(channel, mean_eval);
    mix_secure_field(channel, rsqrt_eval);
    mix_secure_field(channel, output_claim.value);

    let r = &output_claim.point[..num_vars];
    let mut eq_evals = build_eq_evals(r);
    let mut centered_folded = combined_product;
    let mut rsqrt_folded = vec![SecureField::one(); n];
    let mut linear_round_polys = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let s0 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &centered_folded,
            &rsqrt_folded,
            mid,
            SecureField::zero(),
        );
        let s1 = compute_mul_eq_sum_at_t(
            &eq_evals,
            &centered_folded,
            &rsqrt_folded,
            mid,
            SecureField::one(),
        );
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_mul_eq_sum_at_t(&eq_evals, &centered_folded, &rsqrt_folded, mid, two);
        let s3 = compute_mul_eq_sum_at_t(&eq_evals, &centered_folded, &rsqrt_folded, mid, three);

        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let dd1 = s1 - s0;
        let dd2 = (s2 - s1 - s1 + s0) * inv2;
        let dd3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = dd1 - dd2 + two * dd3;
        let c2 = dd2 - three * dd3;
        let c3 = dd3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        linear_round_polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();

        fold_mle(&mut eq_evals, challenge, mid);
        fold_mle(&mut centered_folded, challenge, mid);
        fold_mle(&mut rsqrt_folded, challenge, mid);
        cur_n = mid;
    }

    assert_eq!(centered_folded.len(), 1);
    let centered_final = centered_folded[0];
    let rsqrt_final = rsqrt_folded[0];
    let linear_final_evals = (centered_final, rsqrt_final);

    mix_secure_field(channel, centered_final);
    mix_secure_field(channel, rsqrt_final);

    // Skip Part 2 (LogUp)

    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    let rsqrt_table_commitment = compute_rsqrt_table_commitment(config.rsqrt_table_log_size);

    let claim = GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    };

    Ok((
        LayerProof::LayerNorm {
            logup_proof: None,
            linear_round_polys,
            linear_final_evals,
            input_eval,
            output_eval,
            mean: mean_eval,
            rsqrt_var: rsqrt_eval,
            rsqrt_table_commitment,
            simd_combined: true,
        },
        claim,
    ))
}

/// Build eq(r, x) for all x ∈ {0,1}^n via tensor product.
///
/// eq(r, x) = Π_i ((1 - r_i)(1 - x_i) + r_i · x_i)
/// Result has 2^n entries, one for each boolean assignment.
pub(crate) fn build_eq_evals(r: &[SecureField]) -> Vec<SecureField> {
    let n = r.len();
    if n == 0 {
        return vec![SecureField::one()];
    }
    let size = 1 << n;
    let mut evals = Vec::with_capacity(size);

    // Seed with the last coordinate
    let r_last = r[n - 1];
    evals.push(SecureField::one() - r_last);
    evals.push(r_last);

    // Double forward for remaining coordinates (processed in reverse
    // to preserve the same bit ordering as the original implementation).
    // All memory access is forward-sequential: push appends, scale is
    // a forward scan — both are cache-friendly.
    for i in (0..n - 1).rev() {
        let r_i = r[i];
        let one_minus_ri = SecureField::one() - r_i;
        let cur_len = evals.len();
        // Append new entries (bit = 1 for r_i)
        for j in 0..cur_len {
            evals.push(evals[j] * r_i);
        }
        // Scale existing entries in-place (bit = 0 for r_i)
        for j in 0..cur_len {
            evals[j] = evals[j] * one_minus_ri;
        }
    }

    evals
}

/// Compute Σ_{i=0..mid-1} eq_t(i) · a_t(i) · b_t(i) at evaluation point t.
fn compute_mul_eq_sum_at_t(
    eq: &[SecureField],
    a: &[SecureField],
    b: &[SecureField],
    mid: usize,
    t: SecureField,
) -> SecureField {
    debug_assert!(
        eq.len() >= 2 * mid,
        "eq array too small: {} < {}",
        eq.len(),
        2 * mid
    );
    debug_assert!(
        a.len() >= 2 * mid,
        "a array too small: {} < {}",
        a.len(),
        2 * mid
    );
    debug_assert!(
        b.len() >= 2 * mid,
        "b array too small: {} < {}",
        b.len(),
        2 * mid
    );
    let one_minus_t = SecureField::one() - t;
    let mut sum = SecureField::zero();
    for i in 0..mid {
        let eq_t = one_minus_t * eq[i] + t * eq[mid + i];
        let a_t = one_minus_t * a[i] + t * a[mid + i];
        let b_t = one_minus_t * b[i] + t * b[mid + i];
        sum = sum + eq_t * a_t * b_t;
    }
    sum
}

// ===== Helpers =====

/// Compute Σ_{i=0..mid-1} f_a_t(i) · f_b_t(i) where
/// f_a_t(i) = (1-t)*f_a[i] + t*f_a[mid+i] and similarly for f_b.
#[cfg(test)]
fn compute_sum_at_t(
    f_a: &[SecureField],
    f_b: &[SecureField],
    mid: usize,
    t: SecureField,
) -> SecureField {
    let one_minus_t = SecureField::one() - t;
    let mut sum = SecureField::zero();
    for i in 0..mid {
        let a_t = one_minus_t * f_a[i] + t * f_a[mid + i];
        let b_t = one_minus_t * f_b[i] + t * f_b[mid + i];
        sum = sum + a_t * b_t;
    }
    sum
}

/// Fold an MLE in-place at a challenge point: vals[i] = (1-r)*vals[i] + r*vals[mid+i].
/// Truncates the vector to `mid` elements, reusing the allocation.
fn fold_mle(vals: &mut Vec<SecureField>, r: SecureField, mid: usize) {
    let one_minus_r = SecureField::one() - r;
    for i in 0..mid {
        vals[i] = one_minus_r * vals[i] + r * vals[mid + i];
    }
    vals.truncate(mid);
}

/// Mix a SecureField (QM31) into a PoseidonChannel.
/// Mixes each M31 limb as a separate u64 to avoid overflow.
pub(crate) fn mix_secure_field(channel: &mut PoseidonChannel, v: SecureField) {
    channel.mix_u64(v.0 .0 .0 as u64);
    channel.mix_u64(v.0 .1 .0 as u64);
    channel.mix_u64(v.1 .0 .0 as u64);
    channel.mix_u64(v.1 .1 .0 as u64);
}

/// Bounds-checked lookup of a SIMD block's node_id at a given layer offset.
///
/// Translates `(block_index, offset_within_template)` into a circuit layer index
/// and returns the node_id. Errors instead of panicking if the index is out of bounds.
#[cfg(feature = "cuda-runtime")]
fn simd_block_node_id(
    circuit: &LayeredCircuit,
    block_index: usize,
    offset: usize,
) -> Result<usize, GKRError> {
    let block_layer_idx = circuit.block_ranges[block_index].start + offset;
    if block_layer_idx >= circuit.layers.len() {
        return Err(GKRError::SimdError(format!(
            "block {} layer index {} out of bounds (circuit has {} layers)",
            block_index,
            block_layer_idx,
            circuit.layers.len(),
        )));
    }
    Ok(circuit.layers[block_layer_idx].node_id)
}

/// Look up an intermediate result from the execution trace (O(1) HashMap lookup).
fn get_intermediate<'a>(
    execution: &'a GraphExecution,
    node_id: usize,
) -> Result<&'a M31Matrix, GKRError> {
    execution
        .intermediates
        .get(&node_id)
        .ok_or(GKRError::MissingIntermediate { node_id })
}

/// Look up a node output matrix from execution trace.
fn get_node_output<'a>(
    execution: &'a GraphExecution,
    node_id: usize,
) -> Result<&'a M31Matrix, GKRError> {
    execution
        .node_outputs
        .get(&node_id)
        .ok_or(GKRError::MissingIntermediate { node_id })
}

/// Get the two input MLEs for a binary op (Add/Mul) by looking up
/// the OUTPUT of each input layer.
///
/// `intermediates` stores the INPUT to each node, but binary ops need the
/// OUTPUT of their input layers. Use `node_outputs` when available;
/// fall back to `intermediates` for backward compatibility (unit tests).
fn get_binary_op_intermediates(
    execution: &GraphExecution,
    layer: &super::circuit::CircuitLayer,
    circuit: &LayeredCircuit,
) -> Result<(Vec<SecureField>, Vec<SecureField>), GKRError> {
    if layer.input_layers.len() < 2 {
        return Err(GKRError::ReductionError {
            layer_idx: 0,
            reason: format!("binary op needs 2 inputs, got {}", layer.input_layers.len()),
        });
    }

    let lhs_layer = &circuit.layers[layer.input_layers[0]];
    let rhs_layer = &circuit.layers[layer.input_layers[1]];

    // Prefer node_outputs (correct: output of input layer) over intermediates
    // (wrong for binary ops: input to input layer, not its output).
    let lhs_matrix = if let Some(m) = execution.node_outputs.get(&lhs_layer.node_id) {
        m
    } else {
        get_intermediate(execution, lhs_layer.node_id)?
    };
    let rhs_matrix = if let Some(m) = execution.node_outputs.get(&rhs_layer.node_id) {
        m
    } else {
        get_intermediate(execution, rhs_layer.node_id)?
    };

    let lhs_padded = pad_matrix_pow2(lhs_matrix);
    let rhs_padded = pad_matrix_pow2(rhs_matrix);

    Ok((matrix_to_mle(&lhs_padded), matrix_to_mle(&rhs_padded)))
}

/// Extract attention weights from the graph weights for an attention layer.
///
/// The attention layer stores 4 weight matrices (W_Q, W_K, W_V, W_O) at
/// consecutive weight node IDs starting from the layer's first input_layer.
/// Convention: node_id is the input data node; weight_node_ids are stored
/// in the circuit layer's input_layers[1..5] or as node_id+1..node_id+4.
fn get_attention_weights(
    weights: &GraphWeights,
    layer: &super::circuit::CircuitLayer,
) -> Result<AttentionWeights, GKRError> {
    // Try to get 4 weight matrices starting at node_id + 1
    let base = layer.node_id;
    let w_q = weights
        .get_weight(base + 1)
        .ok_or(GKRError::MissingWeight { node_id: base + 1 })?
        .clone();
    let w_k = weights
        .get_weight(base + 2)
        .ok_or(GKRError::MissingWeight { node_id: base + 2 })?
        .clone();
    let w_v = weights
        .get_weight(base + 3)
        .ok_or(GKRError::MissingWeight { node_id: base + 3 })?
        .clone();
    let w_o = weights
        .get_weight(base + 4)
        .ok_or(GKRError::MissingWeight { node_id: base + 4 })?
        .clone();

    Ok(AttentionWeights { w_q, w_k, w_v, w_o })
}

// =============================================================================
// SIMD Attention Decomposition (GPU)
// =============================================================================

/// Reduce an Attention layer across SIMD blocks on GPU.
///
/// Decomposes attention into sub-matmuls, using:
/// - `reduce_matmul_layer_simd_gpu` for shared-weight sub-matmuls (output proj, Q/K/V projections)
/// - `reduce_matmul_layer_dual_simd_gpu` for dual-operand sub-matmuls (per-head score/context)
///
/// The sub-proofs contain a mix of `MatMul` (shared-weight, degree-2) and
/// `MatMulDualSimd` (dual-operand, degree-3). The verifier distinguishes by
/// pattern matching.
///
/// `block_inputs` are the per-block input matrices to the attention layer
/// (the activation feeding into Q/K/V projections).
#[cfg(feature = "cuda-runtime")]
fn reduce_attention_layer_simd_gpu(
    gpu: &std::sync::Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>,
    output_claim: &GKRClaim,
    block_inputs: &[&M31Matrix],
    attn_weights: &AttentionWeights,
    config: &MultiHeadAttentionConfig,
    block_weights: &[SecureField],
    _r_simd: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    let num_heads = config.num_heads;
    let seq_len = config.seq_len;
    let d_model = config.d_model;
    let d_k = config.d_k();
    let n_blocks = block_inputs.len();

    // Run per-block attention forward passes to get all intermediates
    let block_intermediates: Vec<_> = block_inputs
        .iter()
        .map(|input_b| attention_forward(input_b, attn_weights, config, config.causal))
        .collect();

    // Mix attention metadata into channel (same as CPU path)
    channel.mix_u64(0x4154544E_u64); // "ATTN" tag
    channel.mix_u64(num_heads as u64);
    channel.mix_u64(seq_len as u64);
    channel.mix_u64(d_model as u64);
    channel.mix_u64(if config.causal { 1 } else { 0 });

    let expected_count = 4 + 2 * num_heads;
    let mut sub_proofs = Vec::with_capacity(expected_count);
    let mut sub_claim_values = Vec::with_capacity(expected_count);

    // --- Sub-proof 0: Output projection matmul ---
    // final = concat × W_O — shared weight, so use SIMD matmul
    let block_concats: Vec<&M31Matrix> = block_intermediates
        .iter()
        .map(|inter| &inter.concat)
        .collect();

    let (output_proj_proof, _output_proj_claim) = reduce_matmul_layer_simd_gpu(
        gpu,
        output_claim,
        &block_concats,
        &attn_weights.w_o,
        block_weights,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(output_proj_proof);
    sub_claim_values.push(output_claim.value);

    // Split per-block intermediates into per-head
    let block_k_heads: Vec<Vec<M31Matrix>> = block_intermediates
        .iter()
        .map(|inter| split_heads(&inter.k, num_heads))
        .collect();
    let block_v_heads: Vec<Vec<M31Matrix>> = block_intermediates
        .iter()
        .map(|inter| split_heads(&inter.v, num_heads))
        .collect();
    let block_q_heads: Vec<Vec<M31Matrix>> = block_intermediates
        .iter()
        .map(|inter| split_heads(&inter.q, num_heads))
        .collect();

    // --- Per-head sub-proofs (h = H-1..0) ---
    for h in (0..num_heads).rev() {
        // Context matmul: context_h = softmax_h × V_h — dual operand
        // Build combined output MLE for this sub-matmul
        let block_ctx_outputs: Vec<&M31Matrix> = block_intermediates
            .iter()
            .map(|inter| &inter.head_outputs[h])
            .collect();
        let block_ctx_mles: Vec<Vec<SecureField>> = block_ctx_outputs
            .iter()
            .map(|m| {
                let padded = pad_matrix_pow2(m);
                matrix_to_mle(&padded)
            })
            .collect();
        let combined_ctx_mle = gpu
            .combine_blocks(&block_ctx_mles, block_weights)
            .map_err(|e| GKRError::SimdError(format!("combine ctx h={h}: {e}")))?;

        // Draw fresh claim for context
        let ctx_pm = seq_len.next_power_of_two();
        let ctx_pn = d_k.next_power_of_two();
        let ctx_log_rows = ctx_pm.ilog2() as usize;
        let ctx_log_cols = ctx_pn.ilog2() as usize;
        let r_ctx = channel.draw_qm31s(ctx_log_rows + ctx_log_cols);
        let ctx_value = evaluate_mle(&combined_ctx_mle, &r_ctx);
        mix_secure_field(channel, ctx_value);

        let ctx_claim = GKRClaim {
            point: r_ctx,
            value: ctx_value,
        };

        // Dual-operand SIMD matmul: both softmax and V vary per block
        let block_softmax_h: Vec<&M31Matrix> = block_intermediates
            .iter()
            .map(|inter| &inter.softmax_outputs[h])
            .collect();
        let block_v_h: Vec<&M31Matrix> = block_v_heads.iter().map(|heads| &heads[h]).collect();

        let (ctx_proof, _ctx_val) = reduce_matmul_layer_dual_simd_gpu(
            gpu,
            &ctx_claim,
            &block_softmax_h,
            &block_v_h,
            block_weights,
            seq_len,
            seq_len,
            d_k,
            channel,
        )?;
        sub_proofs.push(ctx_proof);
        sub_claim_values.push(ctx_value);

        // Score matmul: raw_scores_h = Q_h × K_h^T (UNSCALED)
        // IMPORTANT: score_matrices[h] includes 1/√d_k scaling, but the sumcheck
        // operates on unscaled Q_h × K_h^T. Must use the raw product.
        let block_q_h: Vec<&M31Matrix> = block_q_heads.iter().map(|heads| &heads[h]).collect();
        let block_k_h_t: Vec<M31Matrix> = block_k_heads
            .iter()
            .map(|heads| transpose_m31(&heads[h]))
            .collect();
        let block_k_h_t_refs: Vec<&M31Matrix> = block_k_h_t.iter().collect();

        let block_raw_scores: Vec<M31Matrix> = (0..n_blocks)
            .map(|b| matmul_m31(block_q_h[b], &block_k_h_t[b]))
            .collect();
        let block_score_mles: Vec<Vec<SecureField>> = block_raw_scores
            .iter()
            .map(|m| {
                let padded = pad_matrix_pow2(m);
                matrix_to_mle(&padded)
            })
            .collect();
        let combined_score_mle = gpu
            .combine_blocks(&block_score_mles, block_weights)
            .map_err(|e| GKRError::SimdError(format!("combine score h={h}: {e}")))?;

        let score_pm = seq_len.next_power_of_two();
        let score_pn = seq_len.next_power_of_two();
        let score_log_rows = score_pm.ilog2() as usize;
        let score_log_cols = score_pn.ilog2() as usize;
        let r_score = channel.draw_qm31s(score_log_rows + score_log_cols);
        let score_value = evaluate_mle(&combined_score_mle, &r_score);
        mix_secure_field(channel, score_value);

        let score_claim = GKRClaim {
            point: r_score,
            value: score_value,
        };

        let (score_proof, _score_val) = reduce_matmul_layer_dual_simd_gpu(
            gpu,
            &score_claim,
            &block_q_h,
            &block_k_h_t_refs,
            block_weights,
            seq_len,
            d_k,
            seq_len,
            channel,
        )?;
        sub_proofs.push(score_proof);
        sub_claim_values.push(score_value);
    }

    // --- Projection matmuls: V, K, Q = input × W_V/K/Q ---
    // Shared weights: use standard SIMD matmul with per-block input matrices

    // V projection (fresh claim)
    let block_v_mles: Vec<Vec<SecureField>> = block_intermediates
        .iter()
        .map(|inter| {
            let padded = pad_matrix_pow2(&inter.v);
            matrix_to_mle(&padded)
        })
        .collect();
    let combined_v_mle = gpu
        .combine_blocks(&block_v_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine v proj: {e}")))?;

    let proj_pm = seq_len.next_power_of_two();
    let proj_pn = d_model.next_power_of_two();
    let proj_log_rows = proj_pm.ilog2() as usize;
    let proj_log_cols = proj_pn.ilog2() as usize;

    let r_v = channel.draw_qm31s(proj_log_rows + proj_log_cols);
    let v_value = evaluate_mle(&combined_v_mle, &r_v);
    mix_secure_field(channel, v_value);

    let v_claim = GKRClaim {
        point: r_v,
        value: v_value,
    };
    let (v_proof, _v_next) = reduce_matmul_layer_simd_gpu(
        gpu,
        &v_claim,
        block_inputs,
        &attn_weights.w_v,
        block_weights,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(v_proof);
    sub_claim_values.push(v_value);

    // K projection (fresh claim)
    let block_k_mles: Vec<Vec<SecureField>> = block_intermediates
        .iter()
        .map(|inter| {
            let padded = pad_matrix_pow2(&inter.k);
            matrix_to_mle(&padded)
        })
        .collect();
    let combined_k_mle = gpu
        .combine_blocks(&block_k_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine k proj: {e}")))?;

    let r_k = channel.draw_qm31s(proj_log_rows + proj_log_cols);
    let k_value = evaluate_mle(&combined_k_mle, &r_k);
    mix_secure_field(channel, k_value);

    let k_claim = GKRClaim {
        point: r_k,
        value: k_value,
    };
    let (k_proof, _k_next) = reduce_matmul_layer_simd_gpu(
        gpu,
        &k_claim,
        block_inputs,
        &attn_weights.w_k,
        block_weights,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(k_proof);
    sub_claim_values.push(k_value);

    // Q projection (fresh claim) — determines the final input claim
    let block_q_mles: Vec<Vec<SecureField>> = block_intermediates
        .iter()
        .map(|inter| {
            let padded = pad_matrix_pow2(&inter.q);
            matrix_to_mle(&padded)
        })
        .collect();
    let combined_q_mle = gpu
        .combine_blocks(&block_q_mles, block_weights)
        .map_err(|e| GKRError::SimdError(format!("combine q proj: {e}")))?;

    let r_q = channel.draw_qm31s(proj_log_rows + proj_log_cols);
    let q_value = evaluate_mle(&combined_q_mle, &r_q);
    mix_secure_field(channel, q_value);

    let q_claim = GKRClaim {
        point: r_q,
        value: q_value,
    };
    let (q_proof, final_input_claim) = reduce_matmul_layer_simd_gpu(
        gpu,
        &q_claim,
        block_inputs,
        &attn_weights.w_q,
        block_weights,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(q_proof);
    sub_claim_values.push(q_value);

    assert_eq!(sub_proofs.len(), expected_count);
    assert_eq!(sub_claim_values.len(), expected_count);

    Ok((
        LayerProof::Attention {
            sub_proofs,
            sub_claim_values,
        },
        final_input_claim,
    ))
}

// =============================================================================
// Attention Decomposition
// =============================================================================

/// Reduce an Attention layer by decomposing it into sub-matmul proofs.
///
/// Standard attention: Output = softmax(Q·K^T / √d_k) · V
///
/// Each sub-matmul in the attention gets a fresh random evaluation point
/// drawn from the Fiat-Shamir channel. The prover evaluates the output MLE
/// at that point and runs the matmul sumcheck. The claimed evaluation values
/// are stored in `sub_claim_values` so the verifier can reconstruct claims.
///
/// Sub-proofs (output → input order):
///   0: Output projection: final = concat × W_O
///   1..2H: Per-head (h = H-1..0): context matmul + score matmul
///   2H+1..2H+3: Projection matmuls (V, K, Q)
///
/// Total sub-proofs: 4 + 2×H
fn reduce_attention_layer(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    attn_weights: &AttentionWeights,
    config: &MultiHeadAttentionConfig,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    let num_heads = config.num_heads;
    let seq_len = config.seq_len;
    let d_model = config.d_model;
    let d_k = config.d_k();

    // Run the full attention forward pass to get all intermediates
    let intermediates = attention_forward(input_matrix, attn_weights, config, config.causal);

    // Mix attention metadata into channel for domain separation
    channel.mix_u64(0x4154544E_u64); // "ATTN" tag
    channel.mix_u64(num_heads as u64);
    channel.mix_u64(seq_len as u64);
    channel.mix_u64(d_model as u64);
    channel.mix_u64(if config.causal { 1 } else { 0 });

    let expected_count = 4 + 2 * num_heads;
    let mut sub_proofs = Vec::with_capacity(expected_count);
    let mut sub_claim_values = Vec::with_capacity(expected_count);

    // Helper: create a fresh claim for a matrix and run matmul reduction
    let prove_sub_matmul = |a: &M31Matrix,
                            b: &M31Matrix,
                            m: usize,
                            k: usize,
                            n: usize,
                            channel: &mut PoseidonChannel|
     -> Result<(LayerProof, GKRClaim, SecureField), GKRError> {
        // Build output MLE
        use crate::components::matmul::matmul_m31;
        let c = matmul_m31(a, b);
        let c_padded = pad_matrix_pow2(&c);
        let c_mle = matrix_to_mle(&c_padded);

        let log_rows = c_padded.rows.ilog2() as usize;
        let log_cols = c_padded.cols.ilog2() as usize;

        // Draw fresh evaluation point from channel
        let r = channel.draw_qm31s(log_rows + log_cols);
        let claimed_value = evaluate_mle(&c_mle, &r);

        // Mix claimed value into channel (binds it to transcript)
        mix_secure_field(channel, claimed_value);

        let fresh_claim = GKRClaim {
            point: r,
            value: claimed_value,
        };

        // Run matmul reduction
        let (proof, input_claim) = reduce_matmul_layer(&fresh_claim, a, b, m, k, n, channel)?;

        let layer_proof = LayerProof::MatMul {
            round_polys: proof.round_polys,
            final_a_eval: proof.final_a_eval,
            final_b_eval: proof.final_b_eval,
        };

        Ok((layer_proof, input_claim, claimed_value))
    };

    // --- Sub-proof 0: Output projection matmul ---
    // final_output = concat × W_O (seq×d_model = seq×d_model × d_model×d_model)
    // Uses the actual output_claim instead of a fresh one (first sub-matmul is linked to parent)
    let (output_proj_proof, _concat_claim) = reduce_matmul_layer(
        output_claim,
        &intermediates.concat,
        &attn_weights.w_o,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(LayerProof::MatMul {
        round_polys: output_proj_proof.round_polys,
        final_a_eval: output_proj_proof.final_a_eval,
        final_b_eval: output_proj_proof.final_b_eval,
    });
    sub_claim_values.push(output_claim.value);

    // Split intermediates per head
    let k_heads = split_heads(&intermediates.k, num_heads);
    let v_heads = split_heads(&intermediates.v, num_heads);
    let q_heads = split_heads(&intermediates.q, num_heads);

    // --- Per-head sub-proofs (h = H-1..0) ---
    for h in (0..num_heads).rev() {
        // Context matmul: context_h = softmax_h × V_h
        // shape: seq×d_k = seq×seq × seq×d_k
        let (ctx_proof, _ctx_input, ctx_value) = prove_sub_matmul(
            &intermediates.softmax_outputs[h],
            &v_heads[h],
            seq_len,
            seq_len,
            d_k,
            channel,
        )?;
        sub_proofs.push(ctx_proof);
        sub_claim_values.push(ctx_value);

        // Score matmul: scores_h = Q_h × K_h^T
        // shape: seq×seq = seq×d_k × d_k×seq
        let k_h_t = transpose_m31(&k_heads[h]);
        let (score_proof, _score_input, score_value) =
            prove_sub_matmul(&q_heads[h], &k_h_t, seq_len, d_k, seq_len, channel)?;
        sub_proofs.push(score_proof);
        sub_claim_values.push(score_value);
    }

    // --- Projection matmuls: V, K, Q = input × W_V/K/Q ---
    // V projection
    let (v_proof, _v_input, v_value) = prove_sub_matmul(
        input_matrix,
        &attn_weights.w_v,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(v_proof);
    sub_claim_values.push(v_value);

    // K projection
    let (k_proof, _k_input, k_value) = prove_sub_matmul(
        input_matrix,
        &attn_weights.w_k,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(k_proof);
    sub_claim_values.push(k_value);

    // Q projection — this one determines the final input claim
    let q = crate::components::matmul::matmul_m31(input_matrix, &attn_weights.w_q);
    let q_padded = pad_matrix_pow2(&q);
    let q_mle = matrix_to_mle(&q_padded);
    let log_rows = q_padded.rows.ilog2() as usize;
    let log_cols = q_padded.cols.ilog2() as usize;
    let r_q = channel.draw_qm31s(log_rows + log_cols);
    let q_value = evaluate_mle(&q_mle, &r_q);
    mix_secure_field(channel, q_value);
    let q_claim = GKRClaim {
        point: r_q,
        value: q_value,
    };
    let (q_reduction, final_input_claim) = reduce_matmul_layer(
        &q_claim,
        input_matrix,
        &attn_weights.w_q,
        seq_len,
        d_model,
        d_model,
        channel,
    )?;
    sub_proofs.push(LayerProof::MatMul {
        round_polys: q_reduction.round_polys,
        final_a_eval: q_reduction.final_a_eval,
        final_b_eval: q_reduction.final_b_eval,
    });
    sub_claim_values.push(q_value);

    assert_eq!(sub_proofs.len(), expected_count);
    assert_eq!(sub_claim_values.len(), expected_count);

    Ok((
        LayerProof::Attention {
            sub_proofs,
            sub_claim_values,
        },
        final_input_claim,
    ))
}

/// Test-accessible wrapper for `reduce_attention_layer`.
pub fn reduce_attention_layer_for_test(
    output_claim: &GKRClaim,
    input_matrix: &M31Matrix,
    attn_weights: &AttentionWeights,
    config: &MultiHeadAttentionConfig,
    channel: &mut PoseidonChannel,
) -> Result<(LayerProof, GKRClaim), GKRError> {
    reduce_attention_layer(output_claim, input_matrix, attn_weights, config, channel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::matmul::M31Matrix;
    use crate::crypto::poseidon_channel::PoseidonChannel;

    /// Simple forward pass for testing: compute matmul C = A × B.
    fn matmul_forward(a: &M31Matrix, b: &M31Matrix) -> M31Matrix {
        let mut c = M31Matrix::new(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum = M31::zero();
                for kk in 0..a.cols {
                    sum = sum + a.get(i, kk) * b.get(kk, j);
                }
                c.set(i, j, sum);
            }
        }
        c
    }

    #[test]
    fn test_reduce_matmul_roundtrip() {
        // 2×4 × 4×2 matmul
        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..2 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                b.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        let c = matmul_forward(&a, &b);

        // Pad and build output MLE
        let c_padded = pad_matrix_pow2(&c);
        let c_mle = matrix_to_mle(&c_padded);

        let log_m = c_padded.rows.ilog2() as usize;
        let log_n = c_padded.cols.ilog2() as usize;

        // Create channel and draw output claim
        let mut channel = PoseidonChannel::new();
        channel.mix_u64(42); // seed
        let r_out = channel.draw_qm31s(log_m + log_n);
        let claimed = evaluate_mle(&c_mle, &r_out);

        let output_claim = GKRClaim {
            point: r_out,
            value: claimed,
        };

        // Run matmul reduction
        let (reduction, _input_claim) =
            reduce_matmul_layer(&output_claim, &a, &b, 2, 4, 2, &mut channel).unwrap();

        // Verify: sum of round polys at 0+1 should match claims
        let log_k = 2; // pad 4 → 4, log2(4)=2
        assert_eq!(reduction.round_polys.len(), log_k);

        // Verify final evals: f_a(r*) * f_b(r*) should equal last round check
        let product = reduction.final_a_eval * reduction.final_b_eval;
        // The product should be consistent with the sumcheck
        // (full verification requires replaying the Fiat-Shamir)
        assert_ne!(
            product,
            SecureField::zero(),
            "product should be non-trivial"
        );
    }

    #[test]
    fn test_reduce_matmul_backend_hook_matches_legacy() {
        // 2x4 x 4x2 matmul
        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..2 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                b.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        let c = matmul_forward(&a, &b);
        let c_padded = pad_matrix_pow2(&c);
        let c_mle = matrix_to_mle(&c_padded);

        let log_m = c_padded.rows.ilog2() as usize;
        let log_n = c_padded.cols.ilog2() as usize;

        // Build identical output claim and transcript pre-state.
        let mut channel_seed = PoseidonChannel::new();
        channel_seed.mix_u64(4242);
        let r_out = channel_seed.draw_qm31s(log_m + log_n);
        let claimed = evaluate_mle(&c_mle, &r_out);
        let output_claim = GKRClaim {
            point: r_out.clone(),
            value: claimed,
        };

        // Legacy reducer path.
        let mut channel_legacy = PoseidonChannel::new();
        channel_legacy.mix_u64(4242);
        let r_out_legacy = channel_legacy.draw_qm31s(log_m + log_n);
        assert_eq!(r_out_legacy, r_out, "legacy and hook transcripts diverged");
        let (legacy_reduction, legacy_claim) =
            reduce_matmul_layer(&output_claim, &a, &b, 2, 4, 2, &mut channel_legacy).unwrap();

        // New backend hook path with SimdBackend implementation.
        let mut channel_hook = PoseidonChannel::new();
        channel_hook.mix_u64(4242);
        let r_out_hook = channel_hook.draw_qm31s(log_m + log_n);
        assert_eq!(r_out_hook, r_out, "legacy and hook transcripts diverged");
        let (hook_reduction, hook_claim) = reduce_matmul_layer_with_backend::<
            stwo::prover::backend::simd::SimdBackend,
        >(
            &output_claim, &a, &b, 2, 4, 2, &mut channel_hook
        )
        .unwrap();

        assert_eq!(
            legacy_reduction.round_polys.len(),
            hook_reduction.round_polys.len()
        );
        for (legacy_rp, hook_rp) in legacy_reduction
            .round_polys
            .iter()
            .zip(hook_reduction.round_polys.iter())
        {
            assert_eq!(legacy_rp.c0, hook_rp.c0);
            assert_eq!(legacy_rp.c1, hook_rp.c1);
            assert_eq!(legacy_rp.c2, hook_rp.c2);
        }
        assert_eq!(legacy_reduction.final_a_eval, hook_reduction.final_a_eval);
        assert_eq!(legacy_reduction.final_b_eval, hook_reduction.final_b_eval);
        assert_eq!(legacy_claim.point, hook_claim.point);
        assert_eq!(legacy_claim.value, hook_claim.value);
    }

    #[test]
    fn test_reduce_add_layer() {
        // Simple add: c = a + b where a=[1,2,3,4], b=[5,6,7,8]
        let a_vals: Vec<SecureField> = (1..=4).map(|x| SecureField::from(M31::from(x))).collect();
        let b_vals: Vec<SecureField> = (5..=8).map(|x| SecureField::from(M31::from(x))).collect();

        let mut channel = PoseidonChannel::new();
        channel.mix_u64(99);
        let r = channel.draw_qm31s(2); // 4 elements → 2 vars

        // Compute claimed output
        let c_vals: Vec<SecureField> = a_vals.iter().zip(&b_vals).map(|(&a, &b)| a + b).collect();
        let claimed = evaluate_mle(&c_vals, &r);

        let output_claim = GKRClaim {
            point: r,
            value: claimed,
        };

        let (proof, _next_claim) =
            reduce_add_layer(&output_claim, &a_vals, &b_vals, &mut channel).unwrap();

        match proof {
            LayerProof::Add {
                lhs_eval, rhs_eval, ..
            } => {
                // a_eval + b_eval should equal claimed sum
                let sum = lhs_eval + rhs_eval;
                assert_eq!(sum, claimed, "add reduction: lhs + rhs != claimed");
            }
            _ => panic!("expected Add proof"),
        }
    }

    #[test]
    fn test_reduce_mul_layer() {
        // Simple mul: c = a * b where a=[2,3,4,5], b=[1,2,3,4]
        let a_vals: Vec<SecureField> = (2..=5).map(|x| SecureField::from(M31::from(x))).collect();
        let b_vals: Vec<SecureField> = (1..=4).map(|x| SecureField::from(M31::from(x))).collect();

        let mut channel = PoseidonChannel::new();
        channel.mix_u64(77);
        let r = channel.draw_qm31s(2);

        // Compute claimed output: element-wise product
        let c_vals: Vec<SecureField> = a_vals.iter().zip(&b_vals).map(|(&a, &b)| a * b).collect();
        let claimed = evaluate_mle(&c_vals, &r);

        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _next_claim) =
            reduce_mul_layer(&output_claim, &a_vals, &b_vals, &mut channel).unwrap();

        match proof {
            LayerProof::Mul {
                eq_round_polys,
                lhs_eval,
                rhs_eval,
            } => {
                // After eq-sumcheck, lhs_eval and rhs_eval are evaluated at
                // the sumcheck challenge point, not at r. Verify the final
                // eq-sumcheck relation: eq(r, challenges) * a(s) * b(s) == claimed
                let num_vars = eq_round_polys.len();
                assert_eq!(num_vars, 2, "4 elements → 2 vars");

                // Replay Fiat-Shamir to get challenges (same channel state)
                let mut replay_channel = PoseidonChannel::new();
                replay_channel.mix_u64(77);
                let _r2 = replay_channel.draw_qm31s(2);
                replay_channel.mix_u64(0x4D554C as u64);
                mix_secure_field(&mut replay_channel, claimed);
                let mut challenges = Vec::new();
                for rp in &eq_round_polys {
                    replay_channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
                    challenges.push(replay_channel.draw_qm31());
                }

                let eq_val = build_eq_evals(&r)[0..1]
                    .iter()
                    .copied()
                    .fold(SecureField::zero(), |_acc, _x| SecureField::zero());
                // Just verify the sumcheck structure: p(0) + p(1) == current sum
                let mut current_sum = claimed;
                for rp in &eq_round_polys {
                    let p0 = rp.c0;
                    let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;
                    assert_eq!(p0 + p1, current_sum, "round sum check");
                    current_sum = rp.eval(
                        challenges[eq_round_polys
                            .iter()
                            .position(|x| std::ptr::eq(x, rp))
                            .unwrap()],
                    );
                }
                // Final: current_sum == eq(r, challenges) * lhs * rhs
                let eq_final = {
                    let mut e = SecureField::one();
                    for (&ri, &ci) in r.iter().zip(challenges.iter()) {
                        e = e * ((SecureField::one() - ri) * (SecureField::one() - ci) + ri * ci);
                    }
                    e
                };
                assert_eq!(
                    current_sum,
                    eq_final * lhs_eval * rhs_eval,
                    "final eq-sumcheck relation"
                );
            }
            _ => panic!("expected Mul proof"),
        }
    }

    #[test]
    fn test_sumcheck_round_poly_consistency() {
        // Verify that c0 + c1 + c2 = sum at t=1 for each round
        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
        }
        for i in 0..8 {
            b.data[i] = M31::from((i + 10) as u32);
        }
        let c = matmul_forward(&a, &b);

        let c_padded = pad_matrix_pow2(&c);
        let c_mle = matrix_to_mle(&c_padded);
        let log_m = c_padded.rows.ilog2() as usize;
        let log_n = c_padded.cols.ilog2() as usize;

        let mut channel = PoseidonChannel::new();
        let r_out = channel.draw_qm31s(log_m + log_n);
        let claimed = evaluate_mle(&c_mle, &r_out);

        let claim = GKRClaim {
            point: r_out,
            value: claimed,
        };

        let (reduction, _) = reduce_matmul_layer(&claim, &a, &b, 2, 4, 2, &mut channel).unwrap();

        // Check first round: p(0) + p(1) should equal the claimed sum
        let rp0 = &reduction.round_polys[0];
        // p(0) = c0, p(1) = c0 + c1 + c2
        let p0 = rp0.c0;
        let p1 = rp0.c0 + rp0.c1 + rp0.c2;
        assert_eq!(p0 + p1, claimed, "first round: p(0)+p(1) != claimed sum");
    }

    // ===== Attention GKR Tests =====

    /// Generate deterministic attention weights for testing.
    fn make_attn_test_weights(
        d_model: usize,
        seed: u64,
    ) -> crate::components::attention::AttentionWeights {
        fn fill_matrix(rows: usize, cols: usize, seed: u64) -> M31Matrix {
            let mut m = M31Matrix::new(rows, cols);
            let mut state = seed;
            for i in 0..rows {
                for j in 0..cols {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let val = ((state >> 33) % 9 + 1) as u32;
                    m.set(i, j, M31::from(val));
                }
            }
            m
        }

        crate::components::attention::AttentionWeights {
            w_q: fill_matrix(d_model, d_model, seed),
            w_k: fill_matrix(d_model, d_model, seed.wrapping_add(1)),
            w_v: fill_matrix(d_model, d_model, seed.wrapping_add(2)),
            w_o: fill_matrix(d_model, d_model, seed.wrapping_add(3)),
        }
    }

    fn make_attn_test_input(seq_len: usize, d_model: usize) -> M31Matrix {
        let mut m = M31Matrix::new(seq_len, d_model);
        for i in 0..seq_len {
            for j in 0..d_model {
                m.set(i, j, M31::from((i * d_model + j + 1) as u32 % 100 + 1));
            }
        }
        m
    }

    #[test]
    fn test_attention_prove_and_verify() {
        // 1 head, d_model=4, seq_len=4
        use crate::components::attention::{attention_forward, MultiHeadAttentionConfig};
        use crate::gkr::verifier::verify_attention_reduction_for_test;

        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let weights = make_attn_test_weights(4, 42);
        let input = make_attn_test_input(4, 4);

        // Run forward pass to get output
        let intermediates = attention_forward(&input, &weights, &config, false);

        // Build output claim
        let output = &intermediates.final_output;
        let output_padded = pad_matrix_pow2(output);
        let output_mle = matrix_to_mle(&output_padded);
        let log_rows = output_padded.rows.ilog2() as usize;
        let log_cols = output_padded.cols.ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xA771);
        let r_out = prover_channel.draw_qm31s(log_rows + log_cols);
        let output_value = evaluate_mle(&output_mle, &r_out);

        let output_claim = GKRClaim {
            point: r_out,
            value: output_value,
        };

        // Prove
        let (proof, input_claim) = reduce_attention_layer(
            &output_claim,
            &input,
            &weights,
            &config,
            &mut prover_channel,
        )
        .unwrap();

        // Check proof structure
        match &proof {
            LayerProof::Attention {
                sub_proofs,
                sub_claim_values,
            } => {
                let expected = 4 + 2 * 1; // 6 sub-proofs for 1 head
                assert_eq!(
                    sub_proofs.len(),
                    expected,
                    "expected {} sub-proofs, got {}",
                    expected,
                    sub_proofs.len()
                );
                assert_eq!(sub_claim_values.len(), expected);

                // Verify
                let mut verifier_channel = PoseidonChannel::new();
                verifier_channel.mix_u64(0xA771);
                let r_out_v = verifier_channel.draw_qm31s(log_rows + log_cols);
                assert_eq!(r_out_v, output_claim.point);

                let verified_claim = verify_attention_reduction_for_test(
                    &output_claim,
                    &config,
                    sub_proofs,
                    sub_claim_values,
                    0,
                    &mut verifier_channel,
                )
                .unwrap();

                assert_eq!(
                    verified_claim.point, input_claim.point,
                    "verified claim point should match prover's input claim point"
                );
                assert_eq!(
                    verified_claim.value, input_claim.value,
                    "verified claim value should match prover's input claim value"
                );
            }
            _ => panic!("expected Attention proof"),
        }
    }

    #[test]
    fn test_attention_multi_head_prove_verify() {
        // 2 heads, d_model=4, seq_len=4
        use crate::components::attention::{attention_forward, MultiHeadAttentionConfig};
        use crate::gkr::verifier::verify_attention_reduction_for_test;

        let config = MultiHeadAttentionConfig::new(2, 4, 4);
        let weights = make_attn_test_weights(4, 77);
        let input = make_attn_test_input(4, 4);

        let intermediates = attention_forward(&input, &weights, &config, false);
        let output = &intermediates.final_output;
        let output_padded = pad_matrix_pow2(output);
        let output_mle = matrix_to_mle(&output_padded);
        let log_rows = output_padded.rows.ilog2() as usize;
        let log_cols = output_padded.cols.ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xA772);
        let r_out = prover_channel.draw_qm31s(log_rows + log_cols);
        let output_value = evaluate_mle(&output_mle, &r_out);
        let output_claim = GKRClaim {
            point: r_out,
            value: output_value,
        };

        let (proof, input_claim) = reduce_attention_layer(
            &output_claim,
            &input,
            &weights,
            &config,
            &mut prover_channel,
        )
        .unwrap();

        match &proof {
            LayerProof::Attention {
                sub_proofs,
                sub_claim_values,
            } => {
                let expected = 4 + 2 * 2; // 8 sub-proofs for 2 heads
                assert_eq!(sub_proofs.len(), expected);
                assert_eq!(sub_claim_values.len(), expected);

                let mut verifier_channel = PoseidonChannel::new();
                verifier_channel.mix_u64(0xA772);
                let _ = verifier_channel.draw_qm31s(log_rows + log_cols);

                let verified_claim = verify_attention_reduction_for_test(
                    &output_claim,
                    &config,
                    sub_proofs,
                    sub_claim_values,
                    0,
                    &mut verifier_channel,
                )
                .unwrap();

                assert_eq!(verified_claim.point, input_claim.point);
                assert_eq!(verified_claim.value, input_claim.value);
            }
            _ => panic!("expected Attention proof"),
        }
    }

    #[test]
    fn test_attention_tampered_sub_proof_fails() {
        // Tamper with a matmul sub-proof and verify the verifier rejects it
        use crate::components::attention::{attention_forward, MultiHeadAttentionConfig};
        use crate::gkr::verifier::verify_attention_reduction_for_test;

        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let weights = make_attn_test_weights(4, 99);
        let input = make_attn_test_input(4, 4);

        let intermediates = attention_forward(&input, &weights, &config, false);
        let output = &intermediates.final_output;
        let output_padded = pad_matrix_pow2(output);
        let output_mle = matrix_to_mle(&output_padded);
        let log_rows = output_padded.rows.ilog2() as usize;
        let log_cols = output_padded.cols.ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xA773);
        let r_out = prover_channel.draw_qm31s(log_rows + log_cols);
        let output_value = evaluate_mle(&output_mle, &r_out);
        let output_claim = GKRClaim {
            point: r_out,
            value: output_value,
        };

        let (proof, _) = reduce_attention_layer(
            &output_claim,
            &input,
            &weights,
            &config,
            &mut prover_channel,
        )
        .unwrap();

        match proof {
            LayerProof::Attention {
                mut sub_proofs,
                sub_claim_values,
            } => {
                // Tamper with the first sub-proof (output projection)
                if let LayerProof::MatMul {
                    ref mut final_a_eval,
                    ..
                } = sub_proofs[0]
                {
                    *final_a_eval = *final_a_eval + SecureField::one();
                }

                let mut verifier_channel = PoseidonChannel::new();
                verifier_channel.mix_u64(0xA773);
                let _ = verifier_channel.draw_qm31s(log_rows + log_cols);

                let result = verify_attention_reduction_for_test(
                    &output_claim,
                    &config,
                    &sub_proofs,
                    &sub_claim_values,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered proof should fail verification");
            }
            _ => panic!("expected Attention proof"),
        }
    }

    #[test]
    fn test_attention_causal_prove_verify() {
        // Causal masking variant
        use crate::components::attention::{attention_forward, MultiHeadAttentionConfig};
        use crate::gkr::verifier::verify_attention_reduction_for_test;

        let config = MultiHeadAttentionConfig::new_causal(1, 4, 4);
        let weights = make_attn_test_weights(4, 55);
        let input = make_attn_test_input(4, 4);

        let intermediates = attention_forward(&input, &weights, &config, true);
        let output = &intermediates.final_output;
        let output_padded = pad_matrix_pow2(output);
        let output_mle = matrix_to_mle(&output_padded);
        let log_rows = output_padded.rows.ilog2() as usize;
        let log_cols = output_padded.cols.ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xA774);
        let r_out = prover_channel.draw_qm31s(log_rows + log_cols);
        let output_value = evaluate_mle(&output_mle, &r_out);
        let output_claim = GKRClaim {
            point: r_out,
            value: output_value,
        };

        let (proof, input_claim) = reduce_attention_layer(
            &output_claim,
            &input,
            &weights,
            &config,
            &mut prover_channel,
        )
        .unwrap();

        match &proof {
            LayerProof::Attention {
                sub_proofs,
                sub_claim_values,
            } => {
                let mut verifier_channel = PoseidonChannel::new();
                verifier_channel.mix_u64(0xA774);
                let _ = verifier_channel.draw_qm31s(log_rows + log_cols);

                let verified_claim = verify_attention_reduction_for_test(
                    &output_claim,
                    &config,
                    sub_proofs,
                    sub_claim_values,
                    0,
                    &mut verifier_channel,
                )
                .unwrap();

                assert_eq!(verified_claim.point, input_claim.point);
                assert_eq!(verified_claim.value, input_claim.value);
            }
            _ => panic!("expected Attention proof"),
        }
    }

    // ===== SIMD Attention Test Helpers =====

    /// CPU degree-2 matmul sumcheck on pre-restricted SecureField vectors.
    /// Matches the prover's `reduce_matmul_layer` channel protocol:
    ///   mix(m, k, n), mix(claimed_value), per-round mix_poly_coeffs + draw,
    ///   mix(final_a), mix(final_b).
    fn cpu_degree2_matmul_sumcheck(
        f_a: &[SecureField],
        f_b: &[SecureField],
        m: usize,
        k: usize,
        n: usize,
        claimed_value: SecureField,
        channel: &mut PoseidonChannel,
    ) -> (
        Vec<crate::components::matmul::RoundPoly>,
        SecureField,
        SecureField,
    ) {
        use crate::components::matmul::RoundPoly;
        use stwo::core::fields::FieldExpOps;

        let pk = f_a.len();
        assert_eq!(pk, f_b.len());
        let log_k = pk.ilog2() as usize;

        channel.mix_u64(m as u64);
        channel.mix_u64(k as u64);
        channel.mix_u64(n as u64);
        mix_secure_field(channel, claimed_value);

        let two = SecureField::from(M31::from(2u32));
        let inv2 = two.inverse();

        let mut round_polys = Vec::with_capacity(log_k);
        let mut fa = f_a.to_vec();
        let mut fb = f_b.to_vec();
        let mut cur_n = pk;

        for _ in 0..log_k {
            let mid = cur_n / 2;
            let s0 = compute_sum_at_t(&fa, &fb, mid, SecureField::zero());
            let s1 = compute_sum_at_t(&fa, &fb, mid, SecureField::one());
            let s2 = compute_sum_at_t(&fa, &fb, mid, two);

            let c0 = s0;
            let c2 = (s2 - s1 - s1 + s0) * inv2;
            let c1 = s1 - s0 - c2;

            round_polys.push(RoundPoly { c0, c1, c2 });
            channel.mix_poly_coeffs(c0, c1, c2);
            let ch = channel.draw_qm31();

            fold_mle(&mut fa, ch, mid);
            fold_mle(&mut fb, ch, mid);
            cur_n = mid;
        }

        let final_a = fa[0];
        let final_b = fb[0];
        mix_secure_field(channel, final_a);
        mix_secure_field(channel, final_b);

        (round_polys, final_a, final_b)
    }

    /// CPU degree-3 dual-operand SIMD sumcheck.
    /// Channel protocol: mix(m,k,n,n_blocks), mix(claim), per-round deg3, mix(finals).
    fn cpu_dual_simd_matmul_sumcheck(
        block_a: &[&[SecureField]],
        block_b: &[&[SecureField]],
        block_weights: &[SecureField],
        pk: usize,
        n_block_vars: usize,
        m: usize,
        k: usize,
        n: usize,
        n_blocks: usize,
        claimed_value: SecureField,
        channel: &mut PoseidonChannel,
    ) -> (
        Vec<crate::gkr::types::RoundPolyDeg3>,
        SecureField,
        SecureField,
    ) {
        use crate::gkr::types::RoundPolyDeg3;
        use stwo::core::fields::FieldExpOps;

        let ext_len = n_blocks * pk;
        let total_vars = n_block_vars + (pk.ilog2() as usize);
        assert_eq!(ext_len, 1 << total_vars);

        let mut ext_w = vec![SecureField::zero(); ext_len];
        let mut ext_a = vec![SecureField::zero(); ext_len];
        let mut ext_b = vec![SecureField::zero(); ext_len];
        for b in 0..n_blocks {
            for ki in 0..pk {
                ext_w[b * pk + ki] = block_weights[b];
                ext_a[b * pk + ki] = block_a[b][ki];
                ext_b[b * pk + ki] = block_b[b][ki];
            }
        }

        channel.mix_u64(m as u64);
        channel.mix_u64(k as u64);
        channel.mix_u64(n as u64);
        channel.mix_u64(n_blocks as u64);
        mix_secure_field(channel, claimed_value);

        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let mut round_polys = Vec::new();
        let mut cur_n = ext_len;

        for _ in 0..total_vars {
            let mid = cur_n / 2;
            let sum_at = |t: SecureField| -> SecureField {
                let ot = SecureField::one() - t;
                (0..mid)
                    .map(|i| {
                        (ot * ext_w[i] + t * ext_w[mid + i])
                            * (ot * ext_a[i] + t * ext_a[mid + i])
                            * (ot * ext_b[i] + t * ext_b[mid + i])
                    })
                    .fold(SecureField::zero(), |a, b| a + b)
            };
            let s0 = sum_at(SecureField::zero());
            let s1 = sum_at(SecureField::one());
            let s2 = sum_at(two);
            let s3 = sum_at(three);

            let d1 = s1 - s0;
            let d2 = (s2 - s1 - s1 + s0) * inv2;
            let d3 = (s3 - s0 - three * (s2 - s1)) * inv6;
            let c0 = s0;
            let c1 = d1 - d2 + two * d3;
            let c2 = d2 - three * d3;
            let c3 = d3;

            round_polys.push(RoundPolyDeg3 { c0, c1, c2, c3 });
            channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
            let ch = channel.draw_qm31();

            let oc = SecureField::one() - ch;
            ext_w = (0..mid)
                .map(|i| oc * ext_w[i] + ch * ext_w[mid + i])
                .collect();
            ext_a = (0..mid)
                .map(|i| oc * ext_a[i] + ch * ext_a[mid + i])
                .collect();
            ext_b = (0..mid)
                .map(|i| oc * ext_b[i] + ch * ext_b[mid + i])
                .collect();
            cur_n = mid;
        }

        let final_a = ext_a[0];
        let final_b = ext_b[0];
        mix_secure_field(channel, final_a);
        mix_secure_field(channel, final_b);

        (round_polys, final_a, final_b)
    }

    // ===== SIMD Attention Tests =====

    /// CPU-only SIMD attention test: 2 blocks, 1 head, d_model=4, seq_len=4.
    /// Constructs a combined proof with MatMul sub-proofs for shared-weight matmuls
    /// and MatMulDualSimd sub-proofs for per-head dual-operand matmuls, then verifies
    /// through `verify_attention_reduction(..., Some(&r_simd), ...)`.
    #[test]
    fn test_simd_attention_dual_operand_prove_verify() {
        use crate::components::attention::{attention_forward, MultiHeadAttentionConfig};
        let d_model = 4;
        let seq_len = 4;
        let num_heads = 1;
        let d_k = d_model / num_heads; // 4
        let n_blocks = 2usize;
        let n_block_vars = n_blocks.ilog2() as usize; // 1

        let config = MultiHeadAttentionConfig::new(num_heads, d_model, seq_len);
        let weights = make_attn_test_weights(d_model, 42);

        // Create 2 different input blocks
        let input_0 = make_attn_test_input(seq_len, d_model);
        let mut input_1 = M31Matrix::new(seq_len, d_model);
        for i in 0..seq_len {
            for j in 0..d_model {
                input_1.set(i, j, M31::from((i * d_model + j + 50) as u32 % 100 + 1));
            }
        }

        // Forward pass both blocks
        let inter_0 = attention_forward(&input_0, &weights, &config, config.causal);
        let inter_1 = attention_forward(&input_1, &weights, &config, config.causal);

        // Build SIMD block weights from r_simd
        let r_simd = vec![SecureField::from(M31::from(5u32))];
        let w_0 = SecureField::one() - r_simd[0];
        let w_1 = r_simd[0];
        let block_weights = vec![w_0, w_1];

        // Helper: combine two MLEs with SIMD weights
        let combine = |mle_0: &[SecureField], mle_1: &[SecureField]| -> Vec<SecureField> {
            mle_0
                .iter()
                .zip(mle_1)
                .map(|(&a, &b)| w_0 * a + w_1 * b)
                .collect()
        };

        // Combine final outputs
        let out_0_padded = pad_matrix_pow2(&inter_0.final_output);
        let out_1_padded = pad_matrix_pow2(&inter_1.final_output);
        let combined_out_mle =
            combine(&matrix_to_mle(&out_0_padded), &matrix_to_mle(&out_1_padded));

        let log_rows = out_0_padded.rows.ilog2() as usize;
        let log_cols = out_0_padded.cols.ilog2() as usize;

        // Start prover channel
        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x51AD_A77E);
        let r_out = prover_channel.draw_qm31s(log_rows + log_cols);
        let output_value = evaluate_mle(&combined_out_mle, &r_out);
        let output_claim = GKRClaim {
            point: r_out,
            value: output_value,
        };

        // Mix attention metadata (matches verifier lines 1721-1725)
        prover_channel.mix_u64(0x4154544E_u64); // "ATTN"
        prover_channel.mix_u64(num_heads as u64);
        prover_channel.mix_u64(seq_len as u64);
        prover_channel.mix_u64(d_model as u64);
        prover_channel.mix_u64(if config.causal { 1 } else { 0 });

        let expected_count = 4 + 2 * num_heads;
        let mut sub_proofs = Vec::with_capacity(expected_count);
        let mut sub_claim_values = Vec::with_capacity(expected_count);

        // --- Sub-proof 0: Output projection (shared W_O, MUST be MatMul) ---
        {
            let combined_concat = combine(
                &matrix_to_mle(&pad_matrix_pow2(&inter_0.concat)),
                &matrix_to_mle(&pad_matrix_pow2(&inter_1.concat)),
            );

            let r_i = &output_claim.point[..log_rows];
            let r_j = &output_claim.point[log_rows..];

            // Restrict combined A (row-major) by rows, B (col-major) by cols
            let f_a = restrict_mle(&combined_concat, r_i);
            let f_b = restrict_mle(
                &matrix_to_mle_col_major(&pad_matrix_pow2(&weights.w_o)),
                r_j,
            );

            let (rps, fa, fb) = cpu_degree2_matmul_sumcheck(
                &f_a,
                &f_b,
                seq_len,
                d_model,
                d_model,
                output_claim.value,
                &mut prover_channel,
            );

            sub_proofs.push(LayerProof::MatMul {
                round_polys: rps,
                final_a_eval: fa,
                final_b_eval: fb,
            });
            sub_claim_values.push(output_claim.value);
        }

        // --- Per-head sub-proofs (dual-operand on CPU) ---
        for h in (0..num_heads).rev() {
            // Context matmul: head_outputs_h = softmax_h × V_h (both vary)
            let combined_ctx = combine(
                &matrix_to_mle(&pad_matrix_pow2(&inter_0.head_outputs[h])),
                &matrix_to_mle(&pad_matrix_pow2(&inter_1.head_outputs[h])),
            );

            let ctx_pm = seq_len.next_power_of_two();
            let ctx_pn = d_k.next_power_of_two();
            let ctx_log_rows = ctx_pm.ilog2() as usize;
            let ctx_log_cols = ctx_pn.ilog2() as usize;

            // Draw fresh claim
            let r_ctx = prover_channel.draw_qm31s(ctx_log_rows + ctx_log_cols);
            let ctx_value = evaluate_mle(&combined_ctx, &r_ctx);
            mix_secure_field(&mut prover_channel, ctx_value);

            // Restrict per-block A (softmax_h) and B (V_h)
            let r_i_ctx = &r_ctx[..ctx_log_rows];
            let r_j_ctx = &r_ctx[ctx_log_rows..];

            let v_heads_0 = split_heads(&inter_0.v, num_heads);
            let v_heads_1 = split_heads(&inter_1.v, num_heads);

            let f_a_ctx_0 = restrict_mle(
                &matrix_to_mle(&pad_matrix_pow2(&inter_0.softmax_outputs[h])),
                r_i_ctx,
            );
            let f_a_ctx_1 = restrict_mle(
                &matrix_to_mle(&pad_matrix_pow2(&inter_1.softmax_outputs[h])),
                r_i_ctx,
            );
            let f_b_ctx_0 = restrict_mle(
                &matrix_to_mle_col_major(&pad_matrix_pow2(&v_heads_0[h])),
                r_j_ctx,
            );
            let f_b_ctx_1 = restrict_mle(
                &matrix_to_mle_col_major(&pad_matrix_pow2(&v_heads_1[h])),
                r_j_ctx,
            );

            let pk_inner = f_a_ctx_0.len();

            let (ctx_rps, ctx_fa, ctx_fb) = cpu_dual_simd_matmul_sumcheck(
                &[&f_a_ctx_0, &f_a_ctx_1],
                &[&f_b_ctx_0, &f_b_ctx_1],
                &block_weights,
                pk_inner,
                n_block_vars,
                seq_len,
                seq_len,
                d_k,
                n_blocks,
                ctx_value,
                &mut prover_channel,
            );

            sub_proofs.push(LayerProof::MatMulDualSimd {
                round_polys: ctx_rps,
                final_a_eval: ctx_fa,
                final_b_eval: ctx_fb,
                n_block_vars,
            });
            sub_claim_values.push(ctx_value);

            // Score matmul: raw Q_h × K_h^T (UNSCALED — score_matrices includes 1/√d_k)
            let q_heads_0 = split_heads(&inter_0.q, num_heads);
            let q_heads_1 = split_heads(&inter_1.q, num_heads);
            let k_heads_0 = split_heads(&inter_0.k, num_heads);
            let k_heads_1 = split_heads(&inter_1.k, num_heads);

            let k_h_t_0 = transpose_m31(&k_heads_0[h]);
            let k_h_t_1 = transpose_m31(&k_heads_1[h]);

            let raw_score_0 = matmul_forward(&q_heads_0[h], &k_h_t_0);
            let raw_score_1 = matmul_forward(&q_heads_1[h], &k_h_t_1);
            let combined_score = combine(
                &matrix_to_mle(&pad_matrix_pow2(&raw_score_0)),
                &matrix_to_mle(&pad_matrix_pow2(&raw_score_1)),
            );

            let score_pm = seq_len.next_power_of_two();
            let score_pn = seq_len.next_power_of_two();
            let score_log_rows = score_pm.ilog2() as usize;
            let score_log_cols = score_pn.ilog2() as usize;

            let r_score = prover_channel.draw_qm31s(score_log_rows + score_log_cols);
            let score_value = evaluate_mle(&combined_score, &r_score);
            mix_secure_field(&mut prover_channel, score_value);

            let r_i_score = &r_score[..score_log_rows];
            let r_j_score = &r_score[score_log_rows..];

            let f_a_s0 = restrict_mle(&matrix_to_mle(&pad_matrix_pow2(&q_heads_0[h])), r_i_score);
            let f_a_s1 = restrict_mle(&matrix_to_mle(&pad_matrix_pow2(&q_heads_1[h])), r_i_score);
            let f_b_s0 = restrict_mle(
                &matrix_to_mle_col_major(&pad_matrix_pow2(&k_h_t_0)),
                r_j_score,
            );
            let f_b_s1 = restrict_mle(
                &matrix_to_mle_col_major(&pad_matrix_pow2(&k_h_t_1)),
                r_j_score,
            );

            let pk_inner_score = f_a_s0.len();

            let (score_rps, score_fa, score_fb) = cpu_dual_simd_matmul_sumcheck(
                &[&f_a_s0, &f_a_s1],
                &[&f_b_s0, &f_b_s1],
                &block_weights,
                pk_inner_score,
                n_block_vars,
                seq_len,
                d_k,
                seq_len,
                n_blocks,
                score_value,
                &mut prover_channel,
            );

            sub_proofs.push(LayerProof::MatMulDualSimd {
                round_polys: score_rps,
                final_a_eval: score_fa,
                final_b_eval: score_fb,
                n_block_vars,
            });
            sub_claim_values.push(score_value);
        }

        // --- V, K, Q projection matmuls (shared weight, MatMul) ---
        let combined_input = combine(
            &matrix_to_mle(&pad_matrix_pow2(&input_0)),
            &matrix_to_mle(&pad_matrix_pow2(&input_1)),
        );

        let proj_pm = seq_len.next_power_of_two();
        let proj_pn = d_model.next_power_of_two();
        let proj_log_rows = proj_pm.ilog2() as usize;
        let proj_log_cols = proj_pn.ilog2() as usize;

        // Helper: prove a fresh-claim shared-weight projection
        let mut prove_projection =
            |combined_out_mle: &[SecureField], weight: &M31Matrix| -> (LayerProof, SecureField) {
                let r = prover_channel.draw_qm31s(proj_log_rows + proj_log_cols);
                let val = evaluate_mle(combined_out_mle, &r);
                mix_secure_field(&mut prover_channel, val);

                let r_i = &r[..proj_log_rows];
                let r_j = &r[proj_log_rows..];

                let f_a = restrict_mle(&combined_input, r_i);
                let f_b = restrict_mle(&matrix_to_mle_col_major(&pad_matrix_pow2(weight)), r_j);

                let (rps, fa, fb) = cpu_degree2_matmul_sumcheck(
                    &f_a,
                    &f_b,
                    seq_len,
                    d_model,
                    d_model,
                    val,
                    &mut prover_channel,
                );

                (
                    LayerProof::MatMul {
                        round_polys: rps,
                        final_a_eval: fa,
                        final_b_eval: fb,
                    },
                    val,
                )
            };

        // V projection
        let combined_v = combine(
            &matrix_to_mle(&pad_matrix_pow2(&inter_0.v)),
            &matrix_to_mle(&pad_matrix_pow2(&inter_1.v)),
        );
        let (v_proof, v_val) = prove_projection(&combined_v, &weights.w_v);
        sub_proofs.push(v_proof);
        sub_claim_values.push(v_val);

        // K projection
        let combined_k = combine(
            &matrix_to_mle(&pad_matrix_pow2(&inter_0.k)),
            &matrix_to_mle(&pad_matrix_pow2(&inter_1.k)),
        );
        let (k_proof, k_val) = prove_projection(&combined_k, &weights.w_k);
        sub_proofs.push(k_proof);
        sub_claim_values.push(k_val);

        // Q projection (final — determines input claim)
        let combined_q = combine(
            &matrix_to_mle(&pad_matrix_pow2(&inter_0.q)),
            &matrix_to_mle(&pad_matrix_pow2(&inter_1.q)),
        );
        let (q_proof, q_val) = prove_projection(&combined_q, &weights.w_q);
        sub_proofs.push(q_proof);
        sub_claim_values.push(q_val);

        assert_eq!(sub_proofs.len(), expected_count);
        assert_eq!(sub_claim_values.len(), expected_count);

        // --- Verify ---
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x51AD_A77E);
        let _r_out_v = verifier_channel.draw_qm31s(log_rows + log_cols);

        let result = crate::gkr::verifier::verify_attention_reduction(
            &output_claim,
            &config,
            &sub_proofs,
            &sub_claim_values,
            Some(&r_simd),
            0,
            &mut verifier_channel,
        );

        assert!(
            result.is_ok(),
            "SIMD attention verification failed: {:?}",
            result.err()
        );
    }

    // ===== GPU Tests =====

    #[cfg(feature = "cuda-runtime")]
    mod gpu_tests {
        use super::*;
        use crate::compiler::graph::{GraphBuilder, GraphExecution, GraphWeights};
        use crate::components::activation::ActivationType;
        use crate::gkr::circuit::LayeredCircuit;
        use crate::gkr::verifier::verify_gkr;

        #[test]
        fn test_prove_gkr_gpu_single_matmul() {
            // 2×4 @ 4×2 matmul — GPU proof verified by CPU verifier
            let mut builder = GraphBuilder::new((2, 4));
            builder.linear(2);
            let graph = builder.build();
            let circuit = LayeredCircuit::from_graph(&graph).unwrap();

            let mut a = M31Matrix::new(2, 4);
            let mut b = M31Matrix::new(4, 2);
            for i in 0..8 {
                a.data[i] = M31::from((i + 1) as u32);
            }
            for i in 0..8 {
                b.data[i] = M31::from((i + 1) as u32);
            }
            let c = matmul_forward(&a, &b);

            let mut weights = GraphWeights::new();
            weights.add_weight(0, b.clone());

            let execution = GraphExecution {
                intermediates: std::collections::HashMap::from([(0, a.clone())]),
                node_outputs: std::collections::HashMap::new(),
                output: c.clone(),
            };

            // Prove with GPU
            let mut prover_channel = PoseidonChannel::new();
            let proof =
                super::super::prove_gkr_gpu(&circuit, &execution, &weights, &mut prover_channel)
                    .unwrap();

            assert_eq!(proof.layer_proofs.len(), 1);
            match &proof.layer_proofs[0] {
                LayerProof::MatMul { round_polys, .. } => {
                    assert_eq!(round_polys.len(), 2); // log2(4) = 2
                }
                _ => panic!("expected MatMul proof"),
            }

            // Verify with CPU verifier (fresh channel)
            let mut verifier_channel = PoseidonChannel::new();
            verify_gkr(&circuit, &proof, &c, &mut verifier_channel).unwrap();
        }

        #[test]
        fn test_prove_gkr_gpu_mlp() {
            // 2-layer MLP: MatMul → ReLU → MatMul
            let mut builder = GraphBuilder::new((2, 4));
            builder.linear(4);
            builder.activation(ActivationType::ReLU);
            builder.linear(2);
            let graph = builder.build();
            let circuit = LayeredCircuit::from_graph(&graph).unwrap();

            let mut w1 = M31Matrix::new(4, 4);
            let mut w2 = M31Matrix::new(4, 2);
            for i in 0..16 {
                w1.data[i] = M31::from((i % 5 + 1) as u32);
            }
            for i in 0..8 {
                w2.data[i] = M31::from((i % 3 + 1) as u32);
            }

            let input = {
                let mut m = M31Matrix::new(2, 4);
                for i in 0..8 {
                    m.data[i] = M31::from((i + 1) as u32);
                }
                m
            };
            let hidden = matmul_forward(&input, &w1);
            let half_p = M31::from((1u32 << 30) as u32);
            let activated = {
                let mut out = M31Matrix::new(hidden.rows, hidden.cols);
                for i in 0..hidden.rows {
                    for j in 0..hidden.cols {
                        let v = hidden.get(i, j);
                        out.set(i, j, if v.0 <= half_p.0 { v } else { M31::zero() });
                    }
                }
                out
            };
            let output = matmul_forward(&activated, &w2);

            let mut weights = GraphWeights::new();
            weights.add_weight(0, w1);
            weights.add_weight(2, w2);

            let execution = GraphExecution {
                intermediates: std::collections::HashMap::from([
                    (0, input.clone()),
                    (1, activated.clone()),
                    (2, activated.clone()),
                ]),
                node_outputs: std::collections::HashMap::new(),
                output: output.clone(),
            };

            // GPU prove
            let mut prover_channel = PoseidonChannel::new();
            let proof =
                super::super::prove_gkr_gpu(&circuit, &execution, &weights, &mut prover_channel)
                    .unwrap();
            assert_eq!(proof.layer_proofs.len(), 3); // MatMul + Activation + MatMul

            // CPU verify
            let mut verifier_channel = PoseidonChannel::new();
            verify_gkr(&circuit, &proof, &output, &mut verifier_channel).unwrap();
        }

        #[test]
        fn test_gpu_cpu_gkr_proof_match() {
            // Verify GPU and CPU provers produce identical proofs
            let mut builder = GraphBuilder::new((2, 4));
            builder.linear(2);
            let graph = builder.build();
            let circuit = LayeredCircuit::from_graph(&graph).unwrap();

            let mut a = M31Matrix::new(2, 4);
            let mut b = M31Matrix::new(4, 2);
            for i in 0..8 {
                a.data[i] = M31::from((i * 3 + 7) as u32 % 31);
            }
            for i in 0..8 {
                b.data[i] = M31::from((i * 5 + 11) as u32 % 31);
            }
            let c = matmul_forward(&a, &b);

            let mut weights = GraphWeights::new();
            weights.add_weight(0, b.clone());

            let execution = GraphExecution {
                intermediates: std::collections::HashMap::from([(0, a.clone())]),
                node_outputs: std::collections::HashMap::new(),
                output: c.clone(),
            };

            // CPU proof
            let mut cpu_channel = PoseidonChannel::new();
            let cpu_proof = prove_gkr(&circuit, &execution, &weights, &mut cpu_channel).unwrap();

            // GPU proof
            let mut gpu_channel = PoseidonChannel::new();
            let gpu_proof =
                super::super::prove_gkr_gpu(&circuit, &execution, &weights, &mut gpu_channel)
                    .unwrap();

            // Compare proof structure
            assert_eq!(cpu_proof.layer_proofs.len(), gpu_proof.layer_proofs.len());

            match (&cpu_proof.layer_proofs[0], &gpu_proof.layer_proofs[0]) {
                (
                    LayerProof::MatMul {
                        round_polys: cpu_rp,
                        final_a_eval: cpu_a,
                        final_b_eval: cpu_b,
                    },
                    LayerProof::MatMul {
                        round_polys: gpu_rp,
                        final_a_eval: gpu_a,
                        final_b_eval: gpu_b,
                    },
                ) => {
                    assert_eq!(cpu_rp.len(), gpu_rp.len());
                    for (i, (cr, gr)) in cpu_rp.iter().zip(gpu_rp).enumerate() {
                        assert_eq!(cr.c0, gr.c0, "round {i} c0 mismatch");
                        assert_eq!(cr.c1, gr.c1, "round {i} c1 mismatch");
                        assert_eq!(cr.c2, gr.c2, "round {i} c2 mismatch");
                    }
                    assert_eq!(cpu_a, gpu_a, "final_a_eval mismatch");
                    assert_eq!(cpu_b, gpu_b, "final_b_eval mismatch");
                }
                _ => panic!("expected both to be MatMul proofs"),
            }

            assert_eq!(cpu_proof.input_claim.point, gpu_proof.input_claim.point);
            assert_eq!(cpu_proof.input_claim.value, gpu_proof.input_claim.value);
        }

        #[test]
        fn test_simd_matmul_reduce_matches_non_simd() {
            // Build two identical blocks: Block0: A0×B, Block1: A1×B (shared weights)
            // Verify that SIMD reduce produces valid sumcheck round polys.
            use crate::gpu_sumcheck::GpuSumcheckExecutor;

            let gpu = std::sync::Arc::new(GpuSumcheckExecutor::new().unwrap());

            // Shared weight matrix B (4×2)
            let mut b = M31Matrix::new(4, 2);
            for i in 0..8 {
                b.data[i] = M31::from((i + 1) as u32);
            }

            // Two different activation matrices A0, A1 (2×4)
            let mut a0 = M31Matrix::new(2, 4);
            for i in 0..8 {
                a0.data[i] = M31::from((i * 3 + 5) as u32 % 31);
            }
            let mut a1 = M31Matrix::new(2, 4);
            for i in 0..8 {
                a1.data[i] = M31::from((i * 7 + 2) as u32 % 31);
            }

            // Compute outputs C0 = A0×B, C1 = A1×B
            let c0 = matmul_forward(&a0, &b);
            let c1 = matmul_forward(&a1, &b);

            // Compute Lagrange block weights for 2 blocks
            let block_weights = crate::components::matmul::compute_lagrange_basis_pub(
                &[SecureField::from(M31::from(17u32))], // 1 challenge for 2 blocks
            );

            // Build combined output MLE
            let c0_padded = pad_matrix_pow2(&c0);
            let c1_padded = pad_matrix_pow2(&c1);
            let c0_mle = matrix_to_mle(&c0_padded);
            let c1_mle = matrix_to_mle(&c1_padded);
            let combined_output: Vec<SecureField> = c0_mle
                .iter()
                .zip(&c1_mle)
                .map(|(&v0, &v1)| block_weights[0] * v0 + block_weights[1] * v1)
                .collect();

            // Create output claim
            let log_m = c0_padded.rows.ilog2() as usize;
            let log_n = c0_padded.cols.ilog2() as usize;
            let mut channel = PoseidonChannel::new();
            channel.mix_u64(0x51BD as u64);
            let r_out = channel.draw_qm31s(log_m + log_n);
            let claimed = evaluate_mle(&combined_output, &r_out);
            let output_claim = GKRClaim {
                point: r_out,
                value: claimed,
            };

            // Run SIMD matmul reduction
            let block_a_matrices: Vec<&M31Matrix> = vec![&a0, &a1];
            let (proof, next_claim) = reduce_matmul_layer_simd_gpu(
                &gpu,
                &output_claim,
                &block_a_matrices,
                &b,
                &block_weights,
                2,
                4,
                2, // m, k, n
                &mut channel,
            )
            .unwrap();

            // Verify proof structure
            match &proof {
                LayerProof::MatMul {
                    round_polys,
                    final_a_eval,
                    final_b_eval,
                } => {
                    let pk = 4usize.next_power_of_two();
                    let log_k = pk.ilog2() as usize;
                    assert_eq!(round_polys.len(), log_k, "should have log2(k) rounds");

                    // Verify round poly consistency: p(0) + p(1) == claim
                    let mut current_sum = claimed;
                    // Replay the channel for dims + claimed_sum mixing
                    let mut replay = PoseidonChannel::new();
                    replay.mix_u64(0x51BD as u64);
                    let _ = replay.draw_qm31s(log_m + log_n);
                    replay.mix_u64(2);
                    replay.mix_u64(4);
                    replay.mix_u64(2);
                    mix_secure_field(&mut replay, claimed);

                    for rp in round_polys {
                        let p0 = rp.c0;
                        let p1 = rp.c0 + rp.c1 + rp.c2;
                        assert_eq!(p0 + p1, current_sum, "round poly sum check");
                        replay.mix_poly_coeffs(rp.c0, rp.c1, rp.c2);
                        let challenge = replay.draw_qm31();
                        // p(challenge) = c0 + c1*t + c2*t²
                        current_sum = rp.c0 + rp.c1 * challenge + rp.c2 * challenge * challenge;
                    }

                    // Final: current_sum == final_a_eval * final_b_eval
                    assert_eq!(
                        current_sum,
                        *final_a_eval * *final_b_eval,
                        "final sumcheck relation"
                    );
                }
                _ => panic!("expected MatMul proof"),
            }

            // Verify next claim has expected dimensionality
            let pm = 2usize.next_power_of_two();
            let pk = 4usize.next_power_of_two();
            assert!(
                next_claim.point.len() >= (pm.ilog2() as usize) + (pk.ilog2() as usize),
                "next claim should have log_m + log_k dimensions"
            );
        }

        #[test]
        fn test_simd_combine_blocks_gpu() {
            // Test that GPU block combination matches CPU reference
            use crate::gpu_sumcheck::GpuSumcheckExecutor;

            let gpu = std::sync::Arc::new(GpuSumcheckExecutor::new().unwrap());

            let n = 8;
            let n_blocks = 3;
            let mut blocks: Vec<Vec<SecureField>> = Vec::new();
            for b in 0..n_blocks {
                let block: Vec<SecureField> = (0..n)
                    .map(|i| SecureField::from(M31::from(((b * n + i) * 7 + 3) as u32 % 31)))
                    .collect();
                blocks.push(block);
            }

            let weights: Vec<SecureField> = vec![
                SecureField::from(M31::from(5u32)),
                SecureField::from(M31::from(11u32)),
                SecureField::from(M31::from(19u32)),
            ];

            // GPU combine
            let gpu_result = gpu.combine_blocks(&blocks, &weights).unwrap();

            // CPU reference
            let cpu_result: Vec<SecureField> = (0..n)
                .map(|i| {
                    let mut sum = SecureField::zero();
                    for b in 0..n_blocks {
                        sum = sum + weights[b] * blocks[b][i];
                    }
                    sum
                })
                .collect();

            assert_eq!(gpu_result.len(), cpu_result.len());
            for (i, (g, c)) in gpu_result.iter().zip(&cpu_result).enumerate() {
                assert_eq!(*g, *c, "combine_blocks mismatch at index {i}");
            }
        }

        #[test]
        fn test_gpu_cpu_activation_proof_match() {
            // Cross-validate: GPU and CPU activation reduction must produce
            // byte-identical LogUp proofs given the same inputs.
            use crate::gpu_sumcheck::GpuSumcheckExecutor;

            let gpu = std::sync::Arc::new(GpuSumcheckExecutor::new().unwrap());

            // Build a 4×4 input matrix with values that stay in ReLU's positive range
            let mut input = M31Matrix::new(4, 4);
            for i in 0..16 {
                input.data[i] = M31::from((i * 7 + 3) as u32 % 100);
            }

            // Pad and build output claim
            let padded = pad_matrix_pow2(&input);
            let n = padded.rows * padded.cols;
            let num_vars = n.ilog2() as usize;

            let input_mle = matrix_to_mle(&padded);
            let activation_fn = ActivationType::ReLU.as_fn();
            let output_mle: Vec<SecureField> = padded
                .data
                .iter()
                .take(n)
                .map(|&v| SecureField::from(activation_fn(v)))
                .collect();

            // Create a claim at a random point
            let mut setup_channel = PoseidonChannel::new();
            setup_channel.mix_u64(0xAC71u64);
            let claim_point = setup_channel.draw_qm31s(num_vars);
            let claimed_value = evaluate_mle(&output_mle, &claim_point);
            let output_claim = GKRClaim {
                point: claim_point,
                value: claimed_value,
            };

            // --- CPU proof ---
            let mut cpu_channel = PoseidonChannel::new();
            let (cpu_proof, cpu_next_claim) = reduce_activation_layer_for_test(
                &output_claim,
                &input,
                ActivationType::ReLU,
                &mut cpu_channel,
            )
            .unwrap();

            // --- GPU proof ---
            let mut gpu_channel = PoseidonChannel::new();
            let (gpu_proof, gpu_next_claim) = super::super::reduce_activation_layer_gpu_for_test(
                &gpu,
                &output_claim,
                &input,
                ActivationType::ReLU,
                &mut gpu_channel,
            )
            .unwrap();

            // Compare next claims
            assert_eq!(
                cpu_next_claim.point, gpu_next_claim.point,
                "next claim point mismatch"
            );
            assert_eq!(
                cpu_next_claim.value, gpu_next_claim.value,
                "next claim value mismatch"
            );

            // Compare proof structure
            match (&cpu_proof, &gpu_proof) {
                (
                    LayerProof::Activation {
                        activation_type: cpu_at,
                        logup_proof: Some(cpu_lp),
                        input_eval: cpu_ie,
                        output_eval: cpu_oe,
                        table_commitment: cpu_tc,
                    },
                    LayerProof::Activation {
                        activation_type: gpu_at,
                        logup_proof: Some(gpu_lp),
                        input_eval: gpu_ie,
                        output_eval: gpu_oe,
                        table_commitment: gpu_tc,
                    },
                ) => {
                    assert_eq!(cpu_at, gpu_at, "activation type mismatch");
                    assert_eq!(cpu_ie, gpu_ie, "input_eval mismatch");
                    assert_eq!(cpu_oe, gpu_oe, "output_eval mismatch");
                    assert_eq!(cpu_tc, gpu_tc, "table_commitment mismatch");

                    // Compare LogUp proof fields
                    assert_eq!(
                        cpu_lp.claimed_sum, gpu_lp.claimed_sum,
                        "claimed_sum mismatch"
                    );
                    assert_eq!(
                        cpu_lp.multiplicities, gpu_lp.multiplicities,
                        "multiplicities mismatch"
                    );
                    assert_eq!(
                        cpu_lp.final_evals, gpu_lp.final_evals,
                        "final_evals mismatch"
                    );

                    // Compare round polynomials
                    assert_eq!(
                        cpu_lp.eq_round_polys.len(),
                        gpu_lp.eq_round_polys.len(),
                        "round poly count mismatch"
                    );
                    for (i, (cr, gr)) in cpu_lp
                        .eq_round_polys
                        .iter()
                        .zip(&gpu_lp.eq_round_polys)
                        .enumerate()
                    {
                        assert_eq!(cr.c0, gr.c0, "round {i} c0 mismatch");
                        assert_eq!(cr.c1, gr.c1, "round {i} c1 mismatch");
                        assert_eq!(cr.c2, gr.c2, "round {i} c2 mismatch");
                        assert_eq!(cr.c3, gr.c3, "round {i} c3 mismatch");
                    }
                }
                _ => panic!("expected both to be Activation proofs with LogUp"),
            }

            // Verify round polynomial sumcheck relation: p(0) + p(1) == current_claim
            // This replays the Fiat-Shamir transcript for the eq-sumcheck portion.
            if let LayerProof::Activation {
                logup_proof: Some(ref lp),
                ..
            } = cpu_proof
            {
                let mut replay = PoseidonChannel::new();
                // Replay pre-sumcheck channel state: LOG tag + type_tag + gamma/beta draws
                replay.mix_u64(0x4C4F47 as u64);
                replay.mix_u64(ActivationType::ReLU.type_tag() as u64);
                let _gamma = replay.draw_qm31();
                let _beta = replay.draw_qm31();
                mix_secure_field(&mut replay, lp.claimed_sum);

                // Verify each round: p(0) + p(1) == running_sum
                let mut running_sum = lp.claimed_sum;
                for (i, rp) in lp.eq_round_polys.iter().enumerate() {
                    let p0 = rp.c0;
                    let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3; // p(1) = c0 + c1 + c2 + c3
                    assert_eq!(
                        p0 + p1,
                        running_sum,
                        "round {i}: p(0) + p(1) != running_sum"
                    );
                    replay.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
                    let challenge = replay.draw_qm31();
                    running_sum = rp.c0
                        + rp.c1 * challenge
                        + rp.c2 * challenge * challenge
                        + rp.c3 * challenge * challenge * challenge;
                }

                // Final sumcheck value should be non-trivially non-zero
                assert_ne!(
                    running_sum,
                    SecureField::zero(),
                    "final sumcheck value should be non-zero for non-trivial activation"
                );
            }
        }
    }
}
