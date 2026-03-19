//! Batch audit prover over time windows.
//!
//! The audit prover replays and proves all inferences in a time window,
//! producing a `BatchAuditResult` with aggregated commitments and
//! proof calldata for on-chain submission.
//!
//! ```text
//! InferenceLog ─[window]─> entries ─[for each]─> replay ─> prove ─> aggregate
//!                                                                        │
//!                                                               BatchAuditResult
//!                                                          (commitments + calldata)
//! ```

use std::time::Instant;

use starknet_ff::FieldElement;
use tracing::info;

use crate::audit::digest::{digest_to_hex, hash_felt_hex_m31, M31Digest, ZERO_DIGEST};
use crate::audit::log::{AuditMerkleTree, InferenceLog};
use crate::audit::replay::verify_replay;
use crate::audit::types::{
    AuditError, AuditRequest, BatchAuditResult, GkrInferenceCalldata, InferenceProofResult,
    ProofMode, VerificationCalldata,
};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::compiler::prove::prove_model;
use crate::components::matmul::M31Matrix;
use crate::crypto::poseidon2_m31::poseidon2_hash;
use crate::starknet::{build_gkr_serializable_proof, prove_for_starknet_onchain};

use stwo::core::fields::m31::M31;

/// Audit prover that batch-proves inferences over a time window.
pub struct AuditProver<'a> {
    graph: &'a ComputationGraph,
    weights: &'a GraphWeights,
    weight_cache: Option<&'a crate::weight_cache::SharedWeightCache>,
}

impl<'a> AuditProver<'a> {
    /// Create a prover for a model.
    pub fn new(graph: &'a ComputationGraph, weights: &'a GraphWeights) -> Self {
        Self { graph, weights, weight_cache: None }
    }

    /// Create a prover with a pre-warmed weight cache for GPU-accelerated proving.
    pub fn with_cache(
        graph: &'a ComputationGraph,
        weights: &'a GraphWeights,
        cache: &'a crate::weight_cache::SharedWeightCache,
    ) -> Self {
        Self { graph, weights, weight_cache: Some(cache) }
    }

    /// Prove all inferences in a time window.
    ///
    /// This is the main entry point for the audit prover:
    /// 1. Query the log for entries in the window
    /// 2. Verify chain integrity
    /// 3. Resolve proving mode from `request.mode`
    /// 4. For each entry: replay, prove with correct pipeline, record result
    /// 5. Aggregate commitments (Merkle roots, combined chain)
    /// 6. Build aggregated proof calldata and verification calldata
    /// 7. Return `BatchAuditResult` with all proof data
    pub fn prove_window(
        &self,
        log: &InferenceLog,
        request: &AuditRequest,
    ) -> Result<BatchAuditResult, AuditError> {
        let total_start = Instant::now();

        // Step 1: Query log for the time window.
        let window = log.query_window(request.start_ns, request.end_ns);
        if window.entries.is_empty() {
            return Err(AuditError::EmptyWindow {
                start: request.start_ns,
                end: request.end_ns,
            });
        }

        // Step 2: Resolve proof mode.
        let mode = ProofMode::from_str(&request.mode);

        info!(
            entries = window.entries.len(),
            start = request.start_ns,
            end = request.end_ns,
            mode = ?mode,
            "Audit prover: proving window"
        );

        // Step 3: Verify chain integrity.
        log.verify_chain()?;

        // Step 4: Apply max_inferences limit.
        let entries = if request.max_inferences > 0 && request.max_inferences < window.entries.len()
        {
            &window.entries[..request.max_inferences]
        } else {
            &window.entries
        };

        // Step 5: Set weight binding mode for GKR prover.
        // AggregatedOracleSumcheck (mode 4) uses RLC binding by default (~0ms).
        // Weight Merkle roots are mixed into the Fiat-Shamir channel before
        // challenges, preventing fabrication. Full MLE opening proof available
        // via STWO_AGGREGATED_FULL_BINDING=1 for trustless on-chain streaming.
        if matches!(mode, ProofMode::Gkr) && !request.weight_binding.is_empty() {
            std::env::set_var("STWO_WEIGHT_BINDING", &request.weight_binding);
        }

        // Step 6: Prove each inference with the resolved mode.
        let mut inference_results = Vec::with_capacity(entries.len());

        #[cfg(feature = "multi-query")]
        {
            use rayon::prelude::*;

            #[cfg(feature = "multi-gpu")]
            let gpu_count = crate::multi_gpu::device_count().max(1);
            #[cfg(not(feature = "multi-gpu"))]
            let gpu_count = 1;

            // On CPU: use all available cores for parallel proving.
            // On GPU: limited by device count.
            let cpu_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            let parallelism = if gpu_count > 1 {
                entries.len().min(gpu_count)
            } else {
                entries.len().min(cpu_threads)
            };
            eprintln!("  [Parallel] Proving {} inferences across {} threads", entries.len(), parallelism);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(parallelism)
                .build()
                .map_err(|e| AuditError::ProvingFailed(format!("Failed to build rayon pool: {e}")))?;

            let results: Vec<(usize, Result<InferenceProofResult, AuditError>)> =
                pool.install(|| {
                    entries
                        .par_iter()
                        .enumerate()
                        .map(|(idx, entry)| {
                            #[cfg(feature = "multi-gpu")]
                            let _guard = crate::multi_gpu::DeviceGuard::new(idx % gpu_count);
                            (idx, self.prove_inference(entry, log, &mode, request.verify_on_chain))
                        })
                        .collect()
                });

            // Sort by original index and collect, propagating first error
            let mut sorted = results;
            sorted.sort_by_key(|(idx, _)| *idx);
            for (_idx, result) in sorted {
                inference_results.push(result?);
            }
        }

        #[cfg(all(not(feature = "multi-query"), feature = "parallel-audit"))]
        {
            use rayon::prelude::*;

            let cpu_threads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            let parallelism = entries.len().min(cpu_threads);
            eprintln!("  [Parallel] Proving {} inferences across {} CPU threads", entries.len(), parallelism);

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(parallelism)
                .build()
                .map_err(|e| AuditError::ProvingFailed(format!("rayon pool: {e}")))?;

            let results: Vec<(usize, Result<InferenceProofResult, AuditError>)> =
                pool.install(|| {
                    entries
                        .par_iter()
                        .enumerate()
                        .map(|(idx, entry)| {
                            (idx, self.prove_inference(entry, log, &mode, request.verify_on_chain))
                        })
                        .collect()
                });

            let mut sorted = results;
            sorted.sort_by_key(|(idx, _)| *idx);
            for (_idx, result) in sorted {
                inference_results.push(result?);
            }
        }

        #[cfg(not(any(feature = "multi-query", feature = "parallel-audit")))]
        for entry in entries {
            let result = self.prove_inference(entry, log, &mode, request.verify_on_chain)?;
            inference_results.push(result);
        }

        // Step 6: Aggregate commitments (M31-native).
        let (io_merkle_root, combined_chain) = compute_batch_commitments(&inference_results);

        // Step 7: Build aggregated proof calldata and verification calldata.
        let proof_calldata = aggregate_proof_calldata(&inference_results);
        let verification_calldata = build_verification_calldata(&inference_results, &mode, log);

        let total_proving_ms = total_start.elapsed().as_millis() as u64;

        // Use actual entry timestamps instead of request bounds (which may be 0).
        let actual_time_start = {
            let ts = entries.iter().map(|e| e.timestamp_ns).min().unwrap_or(request.start_ns);
            if ts == 0 { request.start_ns / 1_000_000_000 } else { ts / 1_000_000_000 }
        };
        let actual_time_end = {
            let ts = entries.iter().map(|e| e.timestamp_ns).max().unwrap_or(request.end_ns);
            if ts == u64::MAX { request.end_ns / 1_000_000_000 } else { ts / 1_000_000_000 }
        };

        let weight_binding_mode = if matches!(mode, ProofMode::Gkr) {
            Some(request.weight_binding.clone())
        } else {
            None
        };

        Ok(BatchAuditResult {
            time_start: actual_time_start,
            time_end: actual_time_end,
            inference_count: inference_results.len() as u32,
            io_merkle_root: digest_to_hex(&io_merkle_root),
            log_merkle_root: digest_to_hex(&window.merkle_root),
            weight_commitment: log.weight_commitment().to_string(),
            combined_chain_commitment: digest_to_hex(&combined_chain),
            inference_results,
            model_id: log.model_id().to_string(),
            proving_time_ms: total_proving_ms,
            proof_calldata,
            verification_calldata,
            weight_binding_mode,
            tee_attestation_hash: None,
        })
    }

    /// Prove a single inference entry using the specified proof mode.
    fn prove_inference(
        &self,
        entry: &crate::audit::types::InferenceLogEntry,
        log: &InferenceLog,
        mode: &ProofMode,
        verify_on_chain: bool,
    ) -> Result<InferenceProofResult, AuditError> {
        let start = Instant::now();

        // Load M31 input from sidecar.
        let (rows, cols, data) = log.read_matrix(entry.matrix_offset, entry.matrix_size)?;
        let input = M31Matrix {
            rows: rows as usize,
            cols: cols as usize,
            data: data.iter().map(|&v| M31::from(v)).collect(),
        };

        match mode {
            // GKR/Direct: skip verify_replay — the proving pipeline runs its own
            // forward pass and computes io_commitment, which we cross-check below.
            // This avoids a redundant 2x forward pass.
            ProofMode::Gkr => self.prove_inference_gkr(entry, &input, start, verify_on_chain),
            ProofMode::Direct => self.prove_inference_direct(entry, &input, start),
            // Legacy: must verify_replay since Blake2s proofs aren't on-chain verifiable.
            ProofMode::Legacy => {
                verify_replay(self.graph, self.weights, entry, log)?;
                self.prove_inference_legacy(entry, &input, start)
            }
        }
    }

    /// GKR pipeline: `prove_for_starknet_ml_gkr()` → `GkrStarknetProof`.
    ///
    /// Produces proof calldata, IO calldata, weight commitments, and
    /// weight opening calldata as hex felt252 strings for
    /// `verify_model_gkr()` on-chain verification.
    ///
    /// Cross-checks the proof's io_commitment against the log entry to detect
    /// any divergence before submitting calldata on-chain.
    fn prove_inference_gkr(
        &self,
        entry: &crate::audit::types::InferenceLogEntry,
        input: &M31Matrix,
        start: Instant,
        verify_on_chain: bool,
    ) -> Result<InferenceProofResult, AuditError> {
        // Parse model_id from the log entry.
        let model_id = FieldElement::from_hex_be(&entry.model_id).map_err(|_| {
            AuditError::ProvingFailed(format!("invalid model_id: {}", entry.model_id))
        })?;

        // When verify_on_chain is requested, we need full aggregated binding for
        // streaming GKR verification. Otherwise use RLC-only (fast path).
        if verify_on_chain {
            std::env::remove_var("STWO_AGGREGATED_RLC_ONLY");
        } else {
            std::env::set_var("STWO_AGGREGATED_RLC_ONLY", "1");
        }

        let agg_proof =
            crate::aggregation::prove_model_pure_gkr_auto_with_cache(
                self.graph, input, self.weights, self.weight_cache,
            )
                .map_err(|e| AuditError::ProvingFailed(format!("GKR proving failed: {}", e)))?;
        let gkr_proof = build_gkr_serializable_proof(&agg_proof, model_id, input)
            .map_err(|e| AuditError::ProvingFailed(format!("GKR proving failed: {}", e)))?;

        let proving_time_ms = start.elapsed().as_millis() as u64;

        // Cross-check: proof's io_commitment vs the log entry's.
        // After fixing the replay forward pass to use unreduced activations
        // (matching the prover), divergence indicates real data corruption.
        let proof_io_hex = format!("{:#066x}", gkr_proof.io_commitment);
        if let Ok(logged_io) = FieldElement::from_hex_be(&entry.io_commitment) {
            if gkr_proof.io_commitment != logged_io {
                return Err(AuditError::ProvingFailed(format!(
                    "io_commitment mismatch for seq {}: logged={}, proof={}",
                    entry.sequence_number, entry.io_commitment, proof_io_hex,
                )));
            }
        }

        // Serialize FieldElement vectors to hex strings.
        let gkr_calldata: Vec<String> = gkr_proof
            .gkr_calldata
            .iter()
            .map(|f| format!("{:#066x}", f))
            .collect();
        let io_calldata: Vec<String> = gkr_proof
            .io_calldata
            .iter()
            .map(|f| format!("{:#066x}", f))
            .collect();
        let weight_commitments_calldata: Vec<String> = gkr_proof
            .weight_commitments
            .iter()
            .map(|f| format!("{:#066x}", f))
            .collect();
        let weight_opening_calldata: Vec<String> = gkr_proof
            .weight_opening_calldata
            .iter()
            .map(|f| format!("{:#066x}", f))
            .collect();

        let proof_size_felts = gkr_proof.total_calldata_size;
        let layer_chain_commitment = format!("{:#066x}", gkr_proof.layer_chain_commitment);

        // Build streaming verification steps when verify_on_chain is requested
        // and the GKR proof is available from the aggregated proof.
        let streaming_steps = if verify_on_chain {
            if let Some(ref gkr) = agg_proof.gkr_proof {
                match self.build_streaming_steps(gkr, model_id, input, &agg_proof.execution.output) {
                    Ok(steps) => {
                        info!(
                            steps = steps.len(),
                            seq = entry.sequence_number,
                            "Built streaming GKR verification steps"
                        );
                        Some(steps)
                    }
                    Err(e) => {
                        tracing::warn!(
                            seq = entry.sequence_number,
                            err = %e,
                            "Failed to build streaming calldata, falling back to single-TX"
                        );
                        None
                    }
                }
            } else {
                tracing::warn!(
                    seq = entry.sequence_number,
                    "verify_on_chain requested but no GKR proof available"
                );
                None
            }
        } else {
            None
        };

        Ok(InferenceProofResult {
            sequence: entry.sequence_number,
            io_commitment: entry.io_commitment.clone(),
            layer_chain_commitment,
            timestamp_ns: entry.timestamp_ns,
            proof_size_felts,
            proving_time_ms,
            proof_calldata: gkr_calldata,
            io_calldata,
            weight_commitments_calldata,
            weight_opening_calldata,
            proof_mode: ProofMode::Gkr,
            streaming_steps,
        })
    }

    /// Build streaming verification steps from a GKR proof.
    ///
    /// Converts the proof into multi-TX streaming calldata using the existing
    /// `build_streaming_gkr_calldata()` infrastructure.
    fn build_streaming_steps(
        &self,
        gkr_proof: &crate::gkr::GKRProof,
        model_id: FieldElement,
        input: &M31Matrix,
        output: &M31Matrix,
    ) -> Result<Vec<crate::audit::types::StreamingVerificationStep>, AuditError> {
        use crate::audit::types::StreamingVerificationStep;

        let circuit = crate::gkr::LayeredCircuit::from_graph(self.graph)
            .map_err(|e| AuditError::ProvingFailed(format!("Circuit compilation failed: {}", e)))?;

        let raw_io = crate::cairo_serde::serialize_raw_io(input, output);

        let streaming = crate::starknet::build_streaming_gkr_calldata(
            gkr_proof, &circuit, model_id, &raw_io, None, None,
        )
        .map_err(|e| AuditError::ProvingFailed(format!("Streaming calldata build failed: {}", e)))?;

        let mut steps = Vec::new();

        // Init step
        steps.push(StreamingVerificationStep {
            filename: "stream_init.txt".to_string(),
            entrypoint: "verify_gkr_stream_init".to_string(),
            calldata: streaming.init_calldata,
        });

        // Output MLE chunks
        for (i, chunk) in streaming.output_mle_chunks.iter().enumerate() {
            steps.push(StreamingVerificationStep {
                filename: format!("stream_output_mle_{}.txt", i),
                entrypoint: "verify_gkr_stream_init_output_mle".to_string(),
                calldata: chunk.calldata.clone(),
            });
        }

        // Layer batches
        for (i, batch) in streaming.stream_batches.iter().enumerate() {
            steps.push(StreamingVerificationStep {
                filename: format!("stream_layers_{}.txt", i),
                entrypoint: "verify_gkr_stream_layers".to_string(),
                calldata: batch.calldata.clone(),
            });
        }

        // Weight binding (packed QM31, separate TX before input MLE)
        steps.push(StreamingVerificationStep {
            filename: "stream_weight_binding.txt".to_string(),
            entrypoint: "verify_gkr_stream_weight_binding".to_string(),
            calldata: streaming.weight_binding_calldata,
        });

        // Input MLE chunks (uniform — no binding data)
        for (i, chunk) in streaming.input_mle_chunks.iter().enumerate() {
            steps.push(StreamingVerificationStep {
                filename: format!("stream_finalize_input_mle_{}.txt", i),
                entrypoint: "verify_gkr_stream_finalize_input_mle".to_string(),
                calldata: chunk.calldata.clone(),
            });
        }

        // Final finalize step
        steps.push(StreamingVerificationStep {
            filename: "stream_finalize.txt".to_string(),
            entrypoint: "verify_gkr_stream_finalize".to_string(),
            calldata: streaming.finalize_calldata,
        });

        Ok(steps)
    }

    /// Direct pipeline: `prove_for_starknet_onchain()` → `StarknetModelProof`.
    ///
    /// Produces combined calldata as hex felt252 strings for on-chain
    /// verification via `verify_model_direct()`.
    ///
    /// Uses the proof's `layer_chain_commitment` (computed from the actual
    /// execution trace) instead of the log entry's potentially stale value.
    /// Cross-checks io_commitment to detect divergence.
    fn prove_inference_direct(
        &self,
        entry: &crate::audit::types::InferenceLogEntry,
        input: &M31Matrix,
        start: Instant,
    ) -> Result<InferenceProofResult, AuditError> {
        let proof = prove_for_starknet_onchain(self.graph, input, self.weights)
            .map_err(|e| AuditError::ProvingFailed(format!("Direct proving failed: {}", e)))?;

        let proving_time_ms = start.elapsed().as_millis() as u64;

        // Cross-check: proof's io_commitment vs the log entry's.
        // After fixing the replay forward pass to use unreduced activations
        // (matching the prover), divergence indicates real data corruption.
        let proof_io_hex = format!("{:#066x}", proof.io_commitment);
        if let Ok(logged_io) = FieldElement::from_hex_be(&entry.io_commitment) {
            if proof.io_commitment != logged_io {
                return Err(AuditError::ProvingFailed(format!(
                    "io_commitment mismatch for seq {}: logged={}, proof={}",
                    entry.sequence_number, entry.io_commitment, proof_io_hex,
                )));
            }
        }

        let proof_calldata: Vec<String> = proof
            .combined_calldata
            .iter()
            .map(|f| format!("{:#066x}", f))
            .collect();

        let proof_size_felts = proof.calldata_size;

        // Use the proof's layer_chain_commitment — it's computed from the actual
        // execution trace via Poseidon(input || intermediates || output).
        // The log entry may have a stale or placeholder value.
        let layer_chain_commitment = format!("{:#066x}", proof.layer_chain_commitment);

        Ok(InferenceProofResult {
            sequence: entry.sequence_number,
            io_commitment: proof_io_hex,
            layer_chain_commitment,
            timestamp_ns: entry.timestamp_ns,
            proof_size_felts,
            proving_time_ms,
            proof_calldata,
            io_calldata: Vec::new(),
            weight_opening_calldata: Vec::new(),
            weight_commitments_calldata: Vec::new(),
            proof_mode: ProofMode::Direct,
            streaming_steps: None,
        })
    }

    /// Legacy pipeline: `prove_model()` → `(Vec<LayerProof>, GraphExecution)`.
    ///
    /// Blake2s proofs — NOT on-chain verifiable. Backward compatible.
    fn prove_inference_legacy(
        &self,
        entry: &crate::audit::types::InferenceLogEntry,
        input: &M31Matrix,
        start: Instant,
    ) -> Result<InferenceProofResult, AuditError> {
        let (proofs, _execution) = prove_model(self.graph, input, self.weights)
            .map_err(|e| AuditError::ProvingFailed(format!("{}", e)))?;

        let proving_time_ms = start.elapsed().as_millis() as u64;
        let proof_size_felts = proofs.len();

        Ok(InferenceProofResult {
            sequence: entry.sequence_number,
            io_commitment: entry.io_commitment.clone(),
            layer_chain_commitment: entry.layer_chain_commitment.clone(),
            timestamp_ns: entry.timestamp_ns,
            proof_size_felts,
            proving_time_ms,
            proof_calldata: Vec::new(),
            io_calldata: Vec::new(),
            weight_opening_calldata: Vec::new(),
            weight_commitments_calldata: Vec::new(),
            proof_mode: ProofMode::Legacy,
            streaming_steps: None,
        })
    }
}

/// Compute aggregated commitments from per-inference results (M31-native).
///
/// - `io_merkle_root`: M31 Merkle root of all io_commitments
/// - `combined_chain`: Poseidon2-M31 hash of all layer_chain_commitments
pub fn compute_batch_commitments(results: &[InferenceProofResult]) -> (M31Digest, M31Digest) {
    // Build M31 Merkle root of io_commitments.
    // io_commitments from ZKML prover are felt252 hex strings — hash into M31 space.
    let mut io_tree = AuditMerkleTree::new();
    for r in results {
        let leaf = hash_felt_hex_m31(&r.io_commitment);
        io_tree.push(leaf);
    }
    let io_merkle_root = io_tree.root();

    // Combined chain commitment: Poseidon2-M31 hash of all chain commitments.
    if results.is_empty() {
        return (io_merkle_root, ZERO_DIGEST);
    }

    let mut chain_input = Vec::new();
    for r in results {
        chain_input.extend_from_slice(&hash_felt_hex_m31(&r.layer_chain_commitment));
    }
    let combined_chain = poseidon2_hash(&chain_input);

    (io_merkle_root, combined_chain)
}

/// Aggregate per-inference proof calldata into a single framed array.
///
/// Format: `[count, len_0, proof_0..., len_1, proof_1..., ...]`
///
/// Returns empty vec if all inferences have empty calldata (Legacy mode).
pub fn aggregate_proof_calldata(results: &[InferenceProofResult]) -> Vec<String> {
    // Skip aggregation if all calldata is empty (Legacy mode).
    let has_calldata = results.iter().any(|r| !r.proof_calldata.is_empty());
    if !has_calldata {
        return Vec::new();
    }

    let mut aggregated = Vec::new();

    // [0] inference count
    aggregated.push(format!("{:#066x}", results.len()));

    // For each inference: [len_i, proof_i...]
    for result in results {
        aggregated.push(format!("{:#066x}", result.proof_calldata.len()));
        aggregated.extend(result.proof_calldata.iter().cloned());
    }

    aggregated
}

/// Build structured verification calldata from per-inference results.
///
/// Returns `None` for Legacy mode (not on-chain verifiable).
fn build_verification_calldata(
    results: &[InferenceProofResult],
    mode: &ProofMode,
    log: &InferenceLog,
) -> Option<VerificationCalldata> {
    match mode {
        ProofMode::Gkr => {
            let per_inference = results
                .iter()
                .map(|r| GkrInferenceCalldata {
                    model_id: log.model_id().to_string(),
                    gkr_calldata: r.proof_calldata.clone(),
                    io_calldata: r.io_calldata.clone(),
                    weight_commitments: r.weight_commitments_calldata.clone(),
                    weight_opening_calldata: r.weight_opening_calldata.clone(),
                })
                .collect();
            Some(VerificationCalldata::Gkr { per_inference })
        }
        ProofMode::Direct => {
            let per_inference = results.iter().map(|r| r.proof_calldata.clone()).collect();
            Some(VerificationCalldata::Direct { per_inference })
        }
        ProofMode::Legacy => None,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregation::compute_io_commitment;
    use crate::audit::digest::ZERO_DIGEST;
    use crate::audit::log::InferenceLog;
    use crate::audit::replay::execute_forward_pass;
    use crate::audit::types::{AuditRequest, InferenceLogEntry, ProofMode};
    use crate::compiler::graph::{GraphBuilder, GraphWeights};
    use crate::components::activation::ActivationType;
    use crate::components::matmul::M31Matrix;
    use std::time::{SystemTime, UNIX_EPOCH};
    use stwo::core::fields::m31::M31;

    fn temp_dir() -> std::path::PathBuf {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("stwo_ml_prover_{}", d))
    }

    /// Helper: build a simple model, create log entries with correct commitments.
    fn setup_test_log(
        dir: &std::path::Path,
        n_entries: usize,
    ) -> (InferenceLog, ComputationGraph, GraphWeights) {
        // Simple model: linear(4->4) -> relu
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU);
        let graph = builder.build();

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i * 4 + j) + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        let mut log = InferenceLog::new(dir, "0x2", "0xabc", "test-model").unwrap();

        let base_ts = 1_000_000_000_000u64;
        for i in 0..n_entries {
            let mut input = M31Matrix::new(1, 4);
            for j in 0..4 {
                input.set(0, j, M31::from((i * 4 + j + 1) as u32));
            }

            // Execute forward pass to get correct output.
            let output = execute_forward_pass(&graph, &input, &weights).unwrap();
            let io_commitment = compute_io_commitment(&input, &output);

            // Write input to sidecar.
            let input_data: Vec<u32> = input.data.iter().map(|m| m.0).collect();
            let (mat_off, mat_sz) = log.write_matrix(1, 4, &input_data).unwrap();

            let entry = InferenceLogEntry {
                inference_id: i as u64,
                sequence_number: 0,
                model_id: "0x2".to_string(),
                weight_commitment: "0xabc".to_string(),
                model_name: "test".to_string(),
                num_layers: 2,
                input_tokens: vec![1, 2, 3, 4],
                output_tokens: vec![],
                matrix_offset: mat_off,
                matrix_size: mat_sz,
                input_rows: 1,
                input_cols: 4,
                output_rows: output.rows as u32,
                output_cols: output.cols as u32,
                io_commitment: format!("{:#066x}", io_commitment),
                layer_chain_commitment: "0x0".to_string(),
                prev_entry_hash: String::new(),
                entry_hash: String::new(),
                timestamp_ns: base_ts + i as u64 * 1_000_000,
                latency_ms: 100,
                gpu_device: "test".to_string(),
                tee_report_hash: "0x0".to_string(),
                task_category: Some("test".to_string()),
                input_preview: None,
                output_preview: None,
            };
            log.append(entry).unwrap();
        }

        (log, graph, weights)
    }

    #[test]
    fn test_prove_single_inference() {
        // Hold env mutex to prevent parallel env var pollution (prove_inference_gkr sets env vars).
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_RLC_ONLY", "1");

        let dir = temp_dir();
        let (log, graph, weights) = setup_test_log(&dir, 1);

        let prover = AuditProver::new(&graph, &weights);
        // Explicit GKR mode — exercises the full on-chain proving pipeline.
        let request = AuditRequest {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            mode: "gkr".to_string(),
            ..AuditRequest::default()
        };

        let result = prover.prove_window(&log, &request).unwrap();
        assert_eq!(result.inference_count, 1);
        assert_eq!(result.inference_results.len(), 1);
        assert!(result.proving_time_ms > 0);
        assert_eq!(result.weight_commitment, "0xabc");
        // GKR mode produces non-empty proof calldata.
        assert!(!result.proof_calldata.is_empty());
        assert!(result.verification_calldata.is_some());
        assert_eq!(result.inference_results[0].proof_mode, ProofMode::Gkr);
        assert!(!result.inference_results[0].proof_calldata.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prove_batch_commitments_deterministic() {
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_RLC_ONLY", "1");

        let dir1 = temp_dir();
        let dir2 = temp_dir();
        let (log1, graph1, weights1) = setup_test_log(&dir1, 3);
        let (log2, graph2, weights2) = setup_test_log(&dir2, 3);

        let prover1 = AuditProver::new(&graph1, &weights1);
        let prover2 = AuditProver::new(&graph2, &weights2);
        // Explicit GKR mode for both provers.
        let request = AuditRequest {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            mode: "gkr".to_string(),
            ..AuditRequest::default()
        };

        let r1 = prover1.prove_window(&log1, &request).unwrap();
        let r2 = prover2.prove_window(&log2, &request).unwrap();

        assert_eq!(r1.io_merkle_root, r2.io_merkle_root);
        assert_eq!(r1.combined_chain_commitment, r2.combined_chain_commitment);
        // GKR proofs should also be deterministic.
        assert_eq!(r1.proof_calldata.len(), r2.proof_calldata.len());

        let _ = std::fs::remove_dir_all(&dir1);
        let _ = std::fs::remove_dir_all(&dir2);
    }

    #[test]
    fn test_empty_window_errors() {
        let dir = temp_dir();
        let (log, graph, weights) = setup_test_log(&dir, 3);

        let prover = AuditProver::new(&graph, &weights);
        let request = AuditRequest {
            start_ns: u64::MAX - 1,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            ..AuditRequest::default()
        };

        let result = prover.prove_window(&log, &request);
        assert!(matches!(result, Err(AuditError::EmptyWindow { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_batch_commitments_computation() {
        // Test the aggregation function directly.
        let results = vec![
            InferenceProofResult {
                sequence: 0,
                io_commitment: "0x1".to_string(),
                layer_chain_commitment: "0xa".to_string(),
                timestamp_ns: 1000,
                proof_size_felts: 100,
                proving_time_ms: 50,
                proof_calldata: Vec::new(),
                io_calldata: Vec::new(),
                weight_opening_calldata: Vec::new(),
                weight_commitments_calldata: Vec::new(),
                proof_mode: ProofMode::Legacy,
                streaming_steps: None,
            },
            InferenceProofResult {
                sequence: 1,
                io_commitment: "0x2".to_string(),
                layer_chain_commitment: "0x14".to_string(),
                timestamp_ns: 2000,
                proof_size_felts: 100,
                proving_time_ms: 50,
                proof_calldata: Vec::new(),
                io_calldata: Vec::new(),
                weight_opening_calldata: Vec::new(),
                weight_commitments_calldata: Vec::new(),
                proof_mode: ProofMode::Legacy,
                streaming_steps: None,
            },
        ];

        let (io_root, combined_chain) = compute_batch_commitments(&results);

        // Both should be non-zero.
        assert_ne!(io_root, ZERO_DIGEST);
        assert_ne!(combined_chain, ZERO_DIGEST);

        // Deterministic: same inputs produce same outputs.
        let (io_root2, combined_chain2) = compute_batch_commitments(&results);
        assert_eq!(io_root, io_root2);
        assert_eq!(combined_chain, combined_chain2);
    }

    #[test]
    fn test_proof_mode_resolution() {
        assert_eq!(ProofMode::from_str("gkr"), ProofMode::Gkr);
        assert_eq!(ProofMode::from_str("direct"), ProofMode::Direct);
        assert_eq!(ProofMode::from_str(""), ProofMode::Legacy);
        assert_eq!(ProofMode::from_str("recursive"), ProofMode::Legacy);
        assert_eq!(ProofMode::from_str("unknown"), ProofMode::Legacy);
    }

    #[test]
    fn test_legacy_proof_calldata_empty() {
        let dir = temp_dir();
        let (log, graph, weights) = setup_test_log(&dir, 1);

        let prover = AuditProver::new(&graph, &weights);
        let request = AuditRequest {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            mode: "".to_string(), // Legacy mode
            ..AuditRequest::default()
        };

        let result = prover.prove_window(&log, &request).unwrap();
        assert_eq!(result.inference_count, 1);
        // Legacy mode: proof_calldata should be empty.
        assert!(result.proof_calldata.is_empty());
        // Legacy mode: verification_calldata should be None.
        assert!(result.verification_calldata.is_none());
        // Per-inference calldata should also be empty.
        assert!(result.inference_results[0].proof_calldata.is_empty());
        assert_eq!(result.inference_results[0].proof_mode, ProofMode::Legacy);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_aggregate_proof_calldata_framing() {
        // Test with non-empty calldata (simulating GKR/Direct mode).
        let results = vec![
            InferenceProofResult {
                sequence: 0,
                io_commitment: "0x1".to_string(),
                layer_chain_commitment: "0x10".to_string(),
                timestamp_ns: 1000,
                proof_size_felts: 3,
                proving_time_ms: 50,
                proof_calldata: vec![
                    format!("{:#066x}", 0xAAu64),
                    format!("{:#066x}", 0xBBu64),
                    format!("{:#066x}", 0xCCu64),
                ],
                io_calldata: Vec::new(),
                weight_opening_calldata: Vec::new(),
                weight_commitments_calldata: Vec::new(),
                proof_mode: ProofMode::Direct,
                streaming_steps: None,
            },
            InferenceProofResult {
                sequence: 1,
                io_commitment: "0x2".to_string(),
                layer_chain_commitment: "0x20".to_string(),
                timestamp_ns: 2000,
                proof_size_felts: 2,
                proving_time_ms: 50,
                proof_calldata: vec![format!("{:#066x}", 0xDDu64), format!("{:#066x}", 0xEEu64)],
                io_calldata: Vec::new(),
                weight_opening_calldata: Vec::new(),
                weight_commitments_calldata: Vec::new(),
                proof_mode: ProofMode::Direct,
                streaming_steps: None,
            },
        ];

        let aggregated = aggregate_proof_calldata(&results);

        // Format: [count=2, len_0=3, AA, BB, CC, len_1=2, DD, EE]
        assert_eq!(aggregated.len(), 8); // 1 + 1+3 + 1+2 = 8
                                         // First element is count = 2.
        assert_eq!(aggregated[0], format!("{:#066x}", 2u64));
        // Second element is len of first proof = 3.
        assert_eq!(aggregated[1], format!("{:#066x}", 3u64));
        // After 3 elements, len of second proof = 2.
        assert_eq!(aggregated[5], format!("{:#066x}", 2u64));
    }

    #[test]
    fn test_aggregate_proof_calldata_empty_for_legacy() {
        let results = vec![InferenceProofResult {
            sequence: 0,
            io_commitment: "0x1".to_string(),
            layer_chain_commitment: "0x10".to_string(),
            timestamp_ns: 1000,
            proof_size_felts: 5,
            proving_time_ms: 50,
            proof_calldata: Vec::new(), // Legacy: empty
            io_calldata: Vec::new(),
            weight_opening_calldata: Vec::new(),
            weight_commitments_calldata: Vec::new(),
            proof_mode: ProofMode::Legacy,
            streaming_steps: None,
        }];

        let aggregated = aggregate_proof_calldata(&results);
        assert!(aggregated.is_empty());
    }

    /// EnvVarGuard: holds ENV_MUTEX and restores env var on drop.
    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let lock = crate::test_utils::ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev, _lock: lock }
        }

        fn remove(key: &'static str) -> Self {
            let lock = crate::test_utils::ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev, _lock: lock }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = self.prev.as_ref() {
                std::env::set_var(self.key, prev);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn test_prove_gkr_with_verify_on_chain_generates_streaming_steps() {
        // Hold the env mutex to prevent RLC_ONLY pollution from parallel tests.
        let _guard = EnvVarGuard::remove("STWO_AGGREGATED_RLC_ONLY");

        let dir = temp_dir();
        let (log, graph, weights) = setup_test_log(&dir, 1);

        let prover = AuditProver::new(&graph, &weights);
        let request = AuditRequest {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            mode: "gkr".to_string(),
            verify_on_chain: true,
            ..AuditRequest::default()
        };

        let result = prover.prove_window(&log, &request).unwrap();
        assert_eq!(result.inference_count, 1);
        assert_eq!(result.inference_results[0].proof_mode, ProofMode::Gkr);

        // verify_on_chain should produce streaming steps
        let steps = result.inference_results[0].streaming_steps.as_ref()
            .expect("verify_on_chain=true should produce streaming_steps");

        // Must have at least: init + finalize
        assert!(steps.len() >= 2, "expected at least 2 streaming steps, got {}", steps.len());

        // First step must be stream_init
        assert_eq!(steps[0].filename, "stream_init.txt");
        assert_eq!(steps[0].entrypoint, "verify_gkr_stream_init");
        assert!(!steps[0].calldata.is_empty(), "init calldata must not be empty");

        // Last step must be stream_finalize
        let last = steps.last().unwrap();
        assert_eq!(last.filename, "stream_finalize.txt");
        assert_eq!(last.entrypoint, "verify_gkr_stream_finalize");

        // All steps must have non-empty calldata
        for step in steps {
            assert!(!step.calldata.is_empty(), "step {} has empty calldata", step.filename);
            assert!(!step.entrypoint.is_empty(), "step {} has empty entrypoint", step.filename);
        }

        // Verify expected entrypoints appear
        let entrypoints: Vec<&str> = steps.iter().map(|s| s.entrypoint.as_str()).collect();
        assert!(entrypoints.contains(&"verify_gkr_stream_init"));
        assert!(entrypoints.contains(&"verify_gkr_stream_finalize"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prove_gkr_without_verify_on_chain_has_no_streaming_steps() {
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_RLC_ONLY", "1");

        let dir = temp_dir();
        let (log, graph, weights) = setup_test_log(&dir, 1);

        let prover = AuditProver::new(&graph, &weights);
        let request = AuditRequest {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            mode: "gkr".to_string(),
            verify_on_chain: false,
            ..AuditRequest::default()
        };

        let result = prover.prove_window(&log, &request).unwrap();
        assert_eq!(result.inference_count, 1);
        assert!(
            result.inference_results[0].streaming_steps.is_none(),
            "verify_on_chain=false should not produce streaming_steps"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
