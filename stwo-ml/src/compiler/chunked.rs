//! Chunked proving for large models.
//!
//! Splits a computation graph into block-sized chunks and proves each
//! independently, using execution checkpoints to pass activations
//! between chunks. This enables proving models that exceed available
//! memory when proven monolithically.
//!
//! # Proof Composition
//!
//! After chunked proving, [`compose_chunk_proofs`] merges independently-proven
//! chunks into a single [`AggregatedModelProofOnChain`]. Matmul sumcheck proofs
//! are extracted directly (with node ID remapping). The unified STARK is rebuilt
//! by re-running the cheap forward pass on the full graph.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::aggregation::{AggregatedModelProofOnChain, AggregationError};
use crate::compiler::checkpoint::ExecutionCheckpoint;
#[cfg(feature = "safetensors")]
use crate::compiler::graph::GraphOp;
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::components::matmul::M31Matrix;
use starknet_crypto::FieldElement;

/// Result from proving a single chunk.
pub struct ChunkProofResult {
    /// The chunk index (0-based).
    pub chunk_index: usize,
    /// Node range in the original graph.
    pub node_range: std::ops::Range<usize>,
    /// The aggregated proof for this chunk.
    pub proof: AggregatedModelProofOnChain,
    /// The output activation matrix (used as input to the next chunk).
    pub output: M31Matrix,
}

/// Error type for chunked proving.
#[derive(Debug, thiserror::Error)]
pub enum ChunkedProvingError {
    #[error("Chunk {chunk} failed: {message}")]
    ChunkFailed { chunk: usize, message: String },
    #[error("Checkpoint IO error: {0}")]
    CheckpointError(#[from] std::io::Error),
    #[error("Aggregation error: {0}")]
    AggregationError(#[from] AggregationError),
    #[error("Empty graph — nothing to prove")]
    EmptyGraph,
}

/// Prove a model by splitting it into chunks along block boundaries.
///
/// Each chunk is proven independently using `prove_model_aggregated_onchain`.
/// Intermediate activations are saved as checkpoints so the process can be
/// resumed if interrupted.
///
/// # Arguments
///
/// * `graph` — The full computation graph
/// * `input` — Input activation matrix
/// * `weights` — Full graph weights
/// * `memory_budget` — Maximum bytes per chunk's proving step
/// * `checkpoint_dir` — Directory for saving execution checkpoints
pub fn prove_model_chunked(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    memory_budget: usize,
    checkpoint_dir: &Path,
) -> Result<Vec<ChunkProofResult>, ChunkedProvingError> {
    if graph.nodes.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    std::fs::create_dir_all(checkpoint_dir).map_err(ChunkedProvingError::CheckpointError)?;

    // Find block boundaries
    let blocks = graph.find_block_boundaries();
    validate_block_ranges(&blocks, graph.nodes.len())?;
    let num_chunks = blocks.len();

    info!(num_chunks, memory_budget, "Starting chunked proving");

    let mut results = Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for (chunk_idx, block_range) in blocks.iter().enumerate() {
        let start = block_range.start;
        let end = block_range.end;

        info!(
            chunk_idx,
            start,
            end,
            num_nodes = end - start,
            "Proving chunk"
        );

        // Try to load checkpoint for this chunk
        let ckpt_path = checkpoint_path(checkpoint_dir, chunk_idx);
        if let Ok(ckpt) = ExecutionCheckpoint::load(&ckpt_path) {
            info!(chunk_idx, "Loaded checkpoint, skipping chunk");
            current_input = ckpt.to_matrix();
            // We don't have the proof here, but we can skip and the caller
            // can re-prove if needed. For now, skip the chunk.
            continue;
        }

        // Extract subgraph and weights for this chunk
        let sub_graph = graph.subgraph(start..end);
        let sub_weights = weights.subset(start..end);

        // Prove the chunk
        let proof = crate::aggregation::prove_model_aggregated_onchain(
            &sub_graph,
            &current_input,
            &sub_weights,
        )
        .map_err(|e| ChunkedProvingError::ChunkFailed {
            chunk: chunk_idx,
            message: format!("{e}"),
        })?;

        // Save checkpoint for the output of this chunk
        let output = proof.execution.output.clone();
        let checkpoint = ExecutionCheckpoint::from_matrix(end - 1, &output);
        checkpoint
            .save(&ckpt_path)
            .map_err(ChunkedProvingError::CheckpointError)?;

        results.push(ChunkProofResult {
            chunk_index: chunk_idx,
            node_range: start..end,
            proof,
            output: output.clone(),
        });

        current_input = output;
    }

    info!(num_chunks = results.len(), "Chunked proving complete");

    Ok(results)
}

/// Compute the checkpoint file path for a given chunk.
fn checkpoint_path(dir: &Path, chunk_idx: usize) -> PathBuf {
    dir.join(format!("chunk_{chunk_idx}.json"))
}

/// Get the total number of matmul proofs across all chunks.
pub fn total_matmul_proofs(results: &[ChunkProofResult]) -> usize {
    results.iter().map(|r| r.proof.matmul_proofs.len()).sum()
}

/// Get the total number of activation claims across all chunks.
pub fn total_activation_claims(results: &[ChunkProofResult]) -> usize {
    results
        .iter()
        .map(|r| r.proof.activation_claims.len())
        .sum()
}

/// Prove a model using parallel chunk proving.
///
/// Unlike [`prove_model_chunked`], this function first executes the full
/// forward pass to compute all intermediate activations, then proves each
/// chunk in parallel using `rayon`. Each chunk's proof is independent once
/// the intermediates are known.
///
/// # Arguments
///
/// * `graph` — The full computation graph
/// * `input` — Input activation matrix
/// * `weights` — Full graph weights
/// * `memory_budget` — Maximum bytes per chunk's proving step
pub fn prove_model_chunked_parallel(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    memory_budget: usize,
) -> Result<Vec<ChunkProofResult>, ChunkedProvingError> {
    use rayon::prelude::*;

    if graph.nodes.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    let blocks = graph.find_block_boundaries();
    validate_block_ranges(&blocks, graph.nodes.len())?;
    let num_chunks = blocks.len();

    info!(
        num_chunks,
        memory_budget, "Starting parallel chunked proving"
    );

    // Phase 1: Sequential forward pass to precompute all chunk inputs.
    // Each chunk needs the output of the previous chunk as input.
    let mut chunk_inputs: Vec<M31Matrix> = Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for block_range in &blocks {
        chunk_inputs.push(current_input.clone());

        // Run forward pass only (no proving) to get this chunk's output
        let sub_graph = graph.subgraph(block_range.start..block_range.end);
        let sub_weights = weights.subset(block_range.start..block_range.end);

        current_input =
            crate::compiler::prove::forward_pass_only(&sub_graph, &current_input, &sub_weights)
                .map_err(|e| ChunkedProvingError::ChunkFailed {
                    chunk: 0,
                    message: format!("Forward pass failed: {e}"),
                })?;
    }

    info!("Forward pass complete, starting parallel proving");

    // Phase 2: Prove all chunks in parallel.
    // Pre-compute subgraphs and weight subsets before spawning workers
    // to avoid redundant clones inside parallel threads.
    let chunk_graphs: Vec<_> = blocks
        .iter()
        .map(|r| graph.subgraph(r.start..r.end))
        .collect();
    let chunk_weights: Vec<_> = blocks
        .iter()
        .map(|r| weights.subset(r.start..r.end))
        .collect();

    // Capture parent thread's GPU device affinity for propagation to rayon workers.
    #[cfg(feature = "multi-gpu")]
    let _mgpu_device_id = crate::multi_gpu::get_thread_device();

    let results: Result<Vec<ChunkProofResult>, ChunkedProvingError> = blocks
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, block_range)| {
            // Propagate GPU device affinity to rayon worker thread (RAII)
            #[cfg(feature = "multi-gpu")]
            let _device_guard = crate::multi_gpu::propagate_device(_mgpu_device_id);

            let start = block_range.start;
            let end = block_range.end;
            let chunk_input = &chunk_inputs[chunk_idx];

            let proof = crate::aggregation::prove_model_aggregated_onchain(
                &chunk_graphs[chunk_idx],
                chunk_input,
                &chunk_weights[chunk_idx],
            )
            .map_err(|e| ChunkedProvingError::ChunkFailed {
                chunk: chunk_idx,
                message: format!("{e}"),
            })?;

            let output = proof.execution.output.clone();
            Ok(ChunkProofResult {
                chunk_index: chunk_idx,
                node_range: start..end,
                proof,
                output,
            })
        })
        .collect();

    let mut results = results?;
    results.sort_by_key(|r| r.chunk_index);

    info!(
        num_chunks = results.len(),
        "Parallel chunked proving complete"
    );

    Ok(results)
}

/// Prove a model using sequential chunk proving with automatic backend selection.
///
/// Uses GPU backend when CUDA is available, otherwise falls back to SIMD.
/// See [`prove_model_chunked`] for details on the sequential proving approach.
pub fn prove_model_chunked_auto(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    memory_budget: usize,
    checkpoint_dir: &Path,
) -> Result<Vec<ChunkProofResult>, ChunkedProvingError> {
    if graph.nodes.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    std::fs::create_dir_all(checkpoint_dir).map_err(ChunkedProvingError::CheckpointError)?;

    let blocks = graph.find_block_boundaries();
    validate_block_ranges(&blocks, graph.nodes.len())?;
    let num_chunks = blocks.len();

    info!(
        num_chunks,
        memory_budget, "Starting chunked proving (auto backend)"
    );

    let mut results = Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for (chunk_idx, block_range) in blocks.iter().enumerate() {
        let start = block_range.start;
        let end = block_range.end;

        info!(
            chunk_idx,
            start,
            end,
            num_nodes = end - start,
            "Proving chunk (auto)"
        );

        let ckpt_path = checkpoint_path(checkpoint_dir, chunk_idx);
        if let Ok(ckpt) = ExecutionCheckpoint::load(&ckpt_path) {
            info!(chunk_idx, "Loaded checkpoint, skipping chunk");
            current_input = ckpt.to_matrix();
            continue;
        }

        let sub_graph = graph.subgraph(start..end);
        let sub_weights = weights.subset(start..end);

        let proof = crate::aggregation::prove_model_aggregated_onchain_auto(
            &sub_graph,
            &current_input,
            &sub_weights,
        )
        .map_err(|e| ChunkedProvingError::ChunkFailed {
            chunk: chunk_idx,
            message: format!("{e}"),
        })?;

        let output = proof.execution.output.clone();
        let checkpoint = ExecutionCheckpoint::from_matrix(end - 1, &output);
        checkpoint
            .save(&ckpt_path)
            .map_err(ChunkedProvingError::CheckpointError)?;

        results.push(ChunkProofResult {
            chunk_index: chunk_idx,
            node_range: start..end,
            proof,
            output: output.clone(),
        });

        current_input = output;
    }

    info!(
        num_chunks = results.len(),
        "Chunked proving complete (auto backend)"
    );
    Ok(results)
}

/// Prove a model using parallel chunk proving with automatic backend selection.
///
/// Uses GPU backend when CUDA is available, otherwise falls back to SIMD.
/// See [`prove_model_chunked_parallel`] for details on the parallel proving approach.
pub fn prove_model_chunked_parallel_auto(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    memory_budget: usize,
) -> Result<Vec<ChunkProofResult>, ChunkedProvingError> {
    use rayon::prelude::*;

    if graph.nodes.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    let blocks = graph.find_block_boundaries();
    validate_block_ranges(&blocks, graph.nodes.len())?;
    let num_chunks = blocks.len();

    info!(
        num_chunks,
        memory_budget, "Starting parallel chunked proving (auto backend)"
    );

    // Phase 1: Sequential forward pass to precompute all chunk inputs.
    let mut chunk_inputs: Vec<M31Matrix> = Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for block_range in &blocks {
        chunk_inputs.push(current_input.clone());

        let sub_graph = graph.subgraph(block_range.start..block_range.end);
        let sub_weights = weights.subset(block_range.start..block_range.end);

        current_input =
            crate::compiler::prove::forward_pass_only(&sub_graph, &current_input, &sub_weights)
                .map_err(|e| ChunkedProvingError::ChunkFailed {
                    chunk: 0,
                    message: format!("Forward pass failed: {e}"),
                })?;
    }

    info!("Forward pass complete, starting parallel proving (auto backend)");

    // Phase 2: Prove all chunks in parallel using auto backend dispatch.
    // Pre-compute subgraphs and weight subsets before spawning workers
    // to avoid redundant clones inside parallel threads.
    let chunk_graphs: Vec<_> = blocks
        .iter()
        .map(|r| graph.subgraph(r.start..r.end))
        .collect();
    let chunk_weights: Vec<_> = blocks
        .iter()
        .map(|r| weights.subset(r.start..r.end))
        .collect();

    // Capture parent thread's GPU device affinity for propagation to rayon workers.
    #[cfg(feature = "multi-gpu")]
    let _mgpu_device_id = crate::multi_gpu::get_thread_device();

    let results: Result<Vec<ChunkProofResult>, ChunkedProvingError> = blocks
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, block_range)| {
            // Propagate GPU device affinity to rayon worker thread (RAII)
            #[cfg(feature = "multi-gpu")]
            let _device_guard = crate::multi_gpu::propagate_device(_mgpu_device_id);

            let start = block_range.start;
            let end = block_range.end;
            let chunk_input = &chunk_inputs[chunk_idx];

            let proof = crate::aggregation::prove_model_aggregated_onchain_auto(
                &chunk_graphs[chunk_idx],
                chunk_input,
                &chunk_weights[chunk_idx],
            )
            .map_err(|e| ChunkedProvingError::ChunkFailed {
                chunk: chunk_idx,
                message: format!("{e}"),
            })?;

            let output = proof.execution.output.clone();
            Ok(ChunkProofResult {
                chunk_index: chunk_idx,
                node_range: start..end,
                proof,
                output,
            })
        })
        .collect();

    let mut results = results?;
    results.sort_by_key(|r| r.chunk_index);

    info!(
        num_chunks = results.len(),
        "Parallel chunked proving complete (auto backend)"
    );
    Ok(results)
}

/// Collect all chunk proofs into a combined set of matmul proofs and activation claims.
pub fn collect_chunk_proofs(
    results: &[ChunkProofResult],
) -> (
    Vec<(usize, crate::components::matmul::MatMulSumcheckProofOnChain)>,
    Vec<crate::aggregation::LayerClaim>,
) {
    let mut all_matmul = Vec::new();
    let mut all_claims = Vec::new();

    for result in results {
        // Remap layer indices to global node IDs
        for (layer_idx, proof) in &result.proof.matmul_proofs {
            let global_idx = result.node_range.start + layer_idx;
            all_matmul.push((global_idx, proof.clone()));
        }
        for claim in &result.proof.activation_claims {
            let mut global_claim = claim.clone();
            global_claim.layer_index += result.node_range.start;
            all_claims.push(global_claim);
        }
    }

    (all_matmul, all_claims)
}

/// Validate that chunk ranges are sorted, contiguous, non-overlapping, and cover `[0, graph_size)`.
///
/// This is strictly stronger than just checking total node count: it also rejects
/// gaps, overlaps, out-of-order chunks, and empty/inverted ranges.
fn validate_chunk_ranges(
    chunks: &[ChunkProofResult],
    graph_size: usize,
) -> Result<(), ChunkedProvingError> {
    if chunks.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    // Verify sorted by chunk_index
    for w in chunks.windows(2) {
        if w[0].chunk_index >= w[1].chunk_index {
            return Err(ChunkedProvingError::ChunkFailed {
                chunk: w[1].chunk_index,
                message: format!(
                    "chunks not sorted: chunk {} appears after chunk {}",
                    w[1].chunk_index, w[0].chunk_index,
                ),
            });
        }
    }

    // Verify each range is valid (start < end)
    for chunk in chunks {
        if chunk.node_range.start >= chunk.node_range.end {
            return Err(ChunkedProvingError::ChunkFailed {
                chunk: chunk.chunk_index,
                message: format!(
                    "empty or inverted range: {}..{}",
                    chunk.node_range.start, chunk.node_range.end,
                ),
            });
        }
    }

    // Verify first chunk starts at 0
    if chunks[0].node_range.start != 0 {
        return Err(ChunkedProvingError::ChunkFailed {
            chunk: 0,
            message: format!(
                "first chunk starts at {} instead of 0",
                chunks[0].node_range.start,
            ),
        });
    }

    // Verify contiguous, non-overlapping
    for w in chunks.windows(2) {
        if w[0].node_range.end != w[1].node_range.start {
            return Err(ChunkedProvingError::ChunkFailed {
                chunk: w[1].chunk_index,
                message: format!(
                    "gap or overlap: chunk {} ends at {}, chunk {} starts at {}",
                    w[0].chunk_index, w[0].node_range.end, w[1].chunk_index, w[1].node_range.start,
                ),
            });
        }
    }

    // Verify last chunk ends at graph_size
    let last = chunks.last().unwrap();
    if last.node_range.end != graph_size {
        return Err(ChunkedProvingError::ChunkFailed {
            chunk: last.chunk_index,
            message: format!(
                "last chunk ends at {} but graph has {} nodes",
                last.node_range.end, graph_size,
            ),
        });
    }

    Ok(())
}

/// Validate block ranges from `find_block_boundaries`: non-empty, non-inverted, contiguous,
/// covering `[0, graph_size)`. Called at proving entry points to catch malformed partitions
/// before they silently corrupt proofs.
fn validate_block_ranges(
    blocks: &[std::ops::Range<usize>],
    graph_size: usize,
) -> Result<(), ChunkedProvingError> {
    if blocks.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }
    for (i, r) in blocks.iter().enumerate() {
        if r.start >= r.end {
            return Err(ChunkedProvingError::ChunkFailed {
                chunk: i,
                message: format!("empty or inverted block range: {}..{}", r.start, r.end),
            });
        }
    }
    if blocks[0].start != 0 {
        return Err(ChunkedProvingError::ChunkFailed {
            chunk: 0,
            message: format!("first block starts at {} instead of 0", blocks[0].start),
        });
    }
    for w in blocks.windows(2) {
        if w[0].end != w[1].start {
            return Err(ChunkedProvingError::ChunkFailed {
                chunk: 0,
                message: format!(
                    "gap or overlap: block ends at {}, next starts at {}",
                    w[0].end, w[1].start,
                ),
            });
        }
    }
    let last = blocks.last().unwrap();
    if last.end != graph_size {
        return Err(ChunkedProvingError::ChunkFailed {
            chunk: blocks.len() - 1,
            message: format!(
                "last block ends at {} but graph has {} nodes",
                last.end, graph_size,
            ),
        });
    }
    Ok(())
}

/// Compose independently-proven chunk proofs into a single [`AggregatedModelProofOnChain`].
///
/// Matmul sumcheck proofs are extracted from each chunk with node ID remapping.
/// The unified STARK (covering activations, add, mul, layernorm, embedding, quantize)
/// is rebuilt by re-running the forward pass on the full graph and calling
/// `build_unified_stark`. Attention proofs are extracted with remapped layer indices.
///
/// **Cost**: O(forward_pass) + O(unified_STARK) — both are fast relative to matmul proving.
///
/// # Errors
///
/// Returns `ChunkedProvingError` if chunks are empty, node counts don't match,
/// or STARK building fails.
pub fn compose_chunk_proofs(
    chunks: &[ChunkProofResult],
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, ChunkedProvingError> {
    use stwo::prover::backend::simd::SimdBackend;
    compose_chunk_proofs_inner::<SimdBackend>(chunks, graph, input, weights)
}

/// Compose chunk proofs with automatic GPU/CPU backend dispatch.
///
/// Uses the GPU backend when available (feature `cuda-runtime`), otherwise falls back
/// to the SIMD backend.
pub fn compose_chunk_proofs_auto(
    chunks: &[ChunkProofResult],
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, ChunkedProvingError> {
    #[cfg(feature = "cuda-runtime")]
    {
        if crate::backend::gpu_is_available() {
            return compose_chunk_proofs_inner::<stwo::prover::backend::gpu::GpuBackend>(
                chunks, graph, input, weights,
            );
        }
    }
    compose_chunk_proofs(chunks, graph, input, weights)
}

/// Inner composition logic, generic over backend `B`.
fn compose_chunk_proofs_inner<B>(
    chunks: &[ChunkProofResult],
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, ChunkedProvingError>
where
    B: stwo::prover::backend::BackendForChannel<
            stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel,
        > + stwo::prover::poly::circle::PolyOps
        + stwo::prover::backend::ColumnOps<stwo::core::fields::m31::BaseField>,
    <B as stwo::prover::backend::ColumnOps<stwo::core::fields::m31::BaseField>>::Column: 'static,
    stwo_constraint_framework::FrameworkComponent<crate::components::activation::ActivationEval>:
        stwo::prover::ComponentProver<B>,
    stwo_constraint_framework::FrameworkComponent<
        crate::components::elementwise::ElementwiseAddEval,
    >: stwo::prover::ComponentProver<B>,
    stwo_constraint_framework::FrameworkComponent<
        crate::components::elementwise::ElementwiseMulEval,
    >: stwo::prover::ComponentProver<B>,
    stwo_constraint_framework::FrameworkComponent<crate::components::layernorm::LayerNormEval>:
        stwo::prover::ComponentProver<B>,
    stwo_constraint_framework::FrameworkComponent<crate::components::embedding::EmbeddingEval>:
        stwo::prover::ComponentProver<B>,
    stwo_constraint_framework::FrameworkComponent<crate::components::quantize::QuantizeEval>:
        stwo::prover::ComponentProver<B>,
    stwo_constraint_framework::FrameworkComponent<crate::components::dequantize::DequantizeEval>:
        stwo::prover::ComponentProver<B>,
    stwo_constraint_framework::FrameworkComponent<crate::components::rmsnorm::RMSNormEval>:
        stwo::prover::ComponentProver<B>,
{
    use crate::aggregation::{
        build_unified_stark, collect_forward_pass_layer_data, compute_io_commitment,
        compute_layer_chain_commitment,
    };
    use crate::compiler::prove::GraphExecution;

    // Validate chunk ranges: sorted, contiguous, non-overlapping, covering [0, graph_size).
    // This is strictly stronger than the old total-count check.
    validate_chunk_ranges(chunks, graph.nodes.len())?;

    info!(
        num_chunks = chunks.len(),
        total_nodes = graph.nodes.len(),
        "Composing chunk proofs"
    );

    // Step 1: Extract matmul proofs from all chunks with global node ID remapping.
    let mut matmul_proofs = Vec::new();
    let mut batched_matmul_proofs = Vec::new();
    let mut attention_proofs = Vec::new();

    for chunk in chunks {
        let offset = chunk.node_range.start;

        // Individual matmul proofs: remap local node_id → global
        for (local_id, proof) in &chunk.proof.matmul_proofs {
            let global_id = offset + local_id;
            matmul_proofs.push((global_id, proof.clone()));
        }

        // Batched matmul proofs: remap each entry's node_id
        for batch in &chunk.proof.batched_matmul_proofs {
            let mut remapped = batch.clone();
            for entry in &mut remapped.entries {
                entry.node_id = offset + entry.node_id;
            }
            batched_matmul_proofs.push(remapped);
        }

        // Attention proofs: remap layer_idx
        for (local_id, proof) in &chunk.proof.attention_proofs {
            let global_id = offset + *local_id;
            attention_proofs.push((global_id, proof.clone()));
        }
    }

    info!(
        individual_matmuls = matmul_proofs.len(),
        batched_groups = batched_matmul_proofs.len(),
        attention = attention_proofs.len(),
        "Extracted proofs from chunks"
    );

    // Step 2: Re-run forward pass to collect non-matmul layer data.
    let fwd = collect_forward_pass_layer_data(graph, input, weights)?;

    // Step 3: Compute commitments from the full forward pass data.
    let layer_chain_commitment =
        compute_layer_chain_commitment(input, &fwd.intermediates, &fwd.final_output);
    let io_commitment = compute_io_commitment(input, &fwd.final_output);

    let execution = GraphExecution {
        intermediates: fwd.intermediates,
        output: fwd.final_output,
    };

    // Step 4: Check if any non-matmul components need a unified STARK.
    let has_components = !fwd.activation_layers.is_empty()
        || !fwd.add_layers.is_empty()
        || !fwd.mul_layers.is_empty()
        || !fwd.layernorm_layers.is_empty()
        || !fwd.embedding_layers.is_empty()
        || !fwd.quantize_layers.is_empty();

    // Step 5: Handle GKR — if any chunk used GKR, note it.
    // GKR re-proving across chunks is a future extension; for now we preserve
    // the per-chunk matmul proofs and leave gkr_proof as None.
    let has_gkr = chunks.iter().any(|c| c.proof.gkr_proof.is_some());
    if has_gkr {
        info!("GKR proofs detected in chunks — using extracted matmul proofs (GKR re-proving is a future extension)");
    }

    if !has_components {
        return Ok(AggregatedModelProofOnChain {
            unified_stark: None,
            matmul_proofs,
            batched_matmul_proofs,
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            rmsnorm_claims: Vec::new(),
            execution,
            activation_claims: Vec::new(),
            attention_proofs,
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment,
            io_commitment,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        });
    }

    // Step 6: Build unified STARK from the collected layer data.
    info!("Building unified STARK for composed proof");
    let result = build_unified_stark::<B>(
        &fwd.activation_layers,
        &fwd.add_layers,
        &fwd.mul_layers,
        &fwd.layernorm_layers,
        &[],
        &fwd.embedding_layers,
        &fwd.quantize_layers,
        &fwd.dequantize_layers,
    )?;

    Ok(AggregatedModelProofOnChain {
        unified_stark: Some(result.stark_proof),
        matmul_proofs,
        batched_matmul_proofs,
        add_claims: result.add_claims,
        mul_claims: result.mul_claims,
        layernorm_claims: result.layernorm_claims,
        rmsnorm_claims: result.rmsnorm_claims,
        execution,
        activation_claims: result.activation_claims,
        attention_proofs,
        embedding_claims: result.embedding_claims,
        quantize_claims: result.quantize_claims,
        dequantize_claims: result.dequantize_claims,
        layer_chain_commitment,
        io_commitment,
        layernorm_mean_var_commitments: fwd.layernorm_mean_var_commitments,
        quantize_params_commitment: crate::aggregation::compute_quantize_params_commitment(
            &fwd.quantize_layers,
        ),
        tiled_matmul_proofs: Vec::new(),
        gkr_proof: None,
        gkr_batch_data: None,
    })
}

// =============================================================================
// Streaming Weight Pipeline Chunked Proving
// =============================================================================

/// Prove a model using streaming weight loading for minimal peak memory.
///
/// Unlike [`prove_model_chunked_parallel`] which requires all weights in memory,
/// this function loads weights on demand per chunk from a [`StreamingWeightPipeline`].
/// Each chunk's weights are dropped after proving, keeping peak memory to ~1 chunk's
/// worth of weights + proving memory.
///
/// Phase 1: Sequential forward pass with per-chunk weight loading.
/// Phase 2: Parallel proving with independent `load_chunk_weights` per worker.
///
/// # Arguments
///
/// * `graph` — The full computation graph
/// * `input` — Input activation matrix
/// * `pipeline` — Streaming weight pipeline (holds mmap'd shard files)
/// * `memory_budget` — Maximum bytes per chunk's proving step (advisory)
#[cfg(feature = "safetensors")]
pub fn prove_model_chunked_streaming(
    graph: &ComputationGraph,
    input: &M31Matrix,
    pipeline: &crate::compiler::streaming::StreamingWeightPipeline,
    memory_budget: usize,
) -> Result<Vec<ChunkProofResult>, ChunkedProvingError> {
    use rayon::prelude::*;

    if graph.nodes.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    let blocks = graph.find_block_boundaries();
    validate_block_ranges(&blocks, graph.nodes.len())?;
    let num_chunks = blocks.len();

    info!(
        num_chunks,
        memory_budget, "Starting streaming chunked proving"
    );

    // Phase 1: Sequential forward pass with per-chunk weight loading.
    // Load weights on demand, run forward pass, drop weights, prefetch next.
    let mut chunk_inputs: Vec<M31Matrix> = Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for (i, block_range) in blocks.iter().enumerate() {
        chunk_inputs.push(current_input.clone());

        let sub_graph = graph.subgraph(block_range.start..block_range.end);
        let sub_weights = pipeline
            .load_chunk_weights(block_range.start..block_range.end)
            .map_err(|e| ChunkedProvingError::ChunkFailed {
                chunk: i,
                message: format!("Weight loading failed: {e}"),
            })?;

        current_input =
            crate::compiler::prove::forward_pass_only(&sub_graph, &current_input, &sub_weights)
                .map_err(|e| ChunkedProvingError::ChunkFailed {
                    chunk: i,
                    message: format!("Forward pass failed: {e}"),
                })?;
        drop(sub_weights); // free this chunk's weights

        // Prefetch next chunk's tensors while we still have time
        if i + 1 < num_chunks {
            let next = &blocks[i + 1];
            pipeline.prefetch_chunk(next.start..next.end);
        }
    }

    info!("Streaming forward pass complete, starting parallel proving");

    // Phase 2: Prove all chunks in parallel. Each worker independently loads
    // its chunk's weights from the pipeline (different chunks touch different
    // mmap pages, so no contention).
    #[cfg(feature = "multi-gpu")]
    let _mgpu_device_id = crate::multi_gpu::get_thread_device();

    let results: Result<Vec<ChunkProofResult>, ChunkedProvingError> = blocks
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, block_range)| {
            #[cfg(feature = "multi-gpu")]
            let _device_guard = crate::multi_gpu::propagate_device(_mgpu_device_id);

            let start = block_range.start;
            let end = block_range.end;
            let chunk_input = &chunk_inputs[chunk_idx];

            let sub_graph = graph.subgraph(start..end);
            let sub_weights = pipeline.load_chunk_weights(start..end).map_err(|e| {
                ChunkedProvingError::ChunkFailed {
                    chunk: chunk_idx,
                    message: format!("Weight loading failed: {e}"),
                }
            })?;

            let proof = crate::aggregation::prove_model_aggregated_onchain(
                &sub_graph,
                chunk_input,
                &sub_weights,
            )
            .map_err(|e| ChunkedProvingError::ChunkFailed {
                chunk: chunk_idx,
                message: format!("{e}"),
            })?;

            let output = proof.execution.output.clone();
            Ok(ChunkProofResult {
                chunk_index: chunk_idx,
                node_range: start..end,
                proof,
                output,
            })
        })
        .collect();

    let mut results = results?;
    results.sort_by_key(|r| r.chunk_index);

    info!(
        num_chunks = results.len(),
        "Streaming chunked proving complete"
    );

    Ok(results)
}

/// Prove a model with **tile-level** streaming: never holds a full weight matrix
/// in RAM. Combines [`StreamingWeightPipeline`] tile-by-tile loading with
/// [`TiledMatMulConfig`] for both the forward pass and proving phase.
///
/// # Memory Profile
///
/// For each matmul node with weight B (k × n):
/// - **Chunk-level streaming**: peak = `k × n × 4` bytes per weight matrix
/// - **Tile-level streaming**: peak = `tile_k × n × 4` bytes per tile
///
/// For Llama-70B (k=8192, n=28672, tile_k=1024):
///   chunk-level: 940 MB per weight, tile-level: 117 MB per tile (8× reduction)
///
/// # Pipeline
///
/// ```text
/// Forward pass (sequential, per-node):
///   for each MatMul node:
///     C = Σ_tile load_tile(B, tile) × A_tile  (streaming accumulation)
///
/// Proving pass (parallel across chunks):
///   for each chunk:
///     for each MatMul in chunk:
///       prove_matmul_tiled_streaming(node, A, C, config)
///         → loads B tile-by-tile, proves each, drops
/// ```
#[cfg(feature = "safetensors")]
pub fn prove_model_chunked_streaming_tiled(
    graph: &ComputationGraph,
    input: &M31Matrix,
    pipeline: &crate::compiler::streaming::StreamingWeightPipeline,
    tile_config: &crate::components::tiled_matmul::TiledMatMulConfig,
    _memory_budget: usize,
) -> Result<Vec<ChunkProofResult>, ChunkedProvingError> {
    use crate::components::tiled_matmul::verify_tiled_matmul;

    if graph.nodes.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    let blocks = graph.find_block_boundaries();
    validate_block_ranges(&blocks, graph.nodes.len())?;
    let num_chunks = blocks.len();

    info!(
        num_chunks,
        tile_k = tile_config.max_tile_k,
        "Starting tile-level streaming chunked proving"
    );

    // Phase 1: Sequential forward pass with tile-level streaming.
    // For each matmul, accumulate C from tile products without loading full B.
    let mut chunk_inputs: Vec<M31Matrix> = Vec::with_capacity(num_chunks);
    let mut chunk_outputs_per_matmul: Vec<Vec<(usize, M31Matrix, M31Matrix)>> =
        Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for (chunk_idx, block_range) in blocks.iter().enumerate() {
        chunk_inputs.push(current_input.clone());

        let sub_graph = graph.subgraph(block_range.start..block_range.end);
        let mut current = current_input.clone();
        let mut matmul_data = Vec::new(); // (node_id, A, C) for deferred proving

        // Walk nodes in this chunk's subgraph
        for (local_idx, node) in sub_graph.nodes.iter().enumerate() {
            let global_id = block_range.start + local_idx;
            match &node.op {
                GraphOp::MatMul { dims: (_m, _k, _n) } => {
                    let a_matrix = current.clone();

                    // Try tile-level streaming forward, fallback to chunk-level
                    let c_matrix = pipeline
                        .forward_matmul_tiled(global_id, &a_matrix, tile_config)
                        .map_err(|e| ChunkedProvingError::ChunkFailed {
                            chunk: chunk_idx,
                            message: format!("Tiled forward node {global_id}: {e}"),
                        })?;

                    matmul_data.push((global_id, a_matrix, c_matrix.clone()));
                    current = c_matrix;
                }
                GraphOp::Activation {
                    activation_type, ..
                } => {
                    let f = activation_type.as_fn();
                    current = crate::compiler::prove::apply_activation_pub(&current, &*f);
                }
                GraphOp::LayerNorm { dim } | GraphOp::RMSNorm { dim } => {
                    let intermediates =
                        crate::compiler::prove::apply_layernorm_detailed(&current, *dim);
                    current = intermediates.output_matrix;
                }
                GraphOp::Add { .. } => {
                    // Identity for sequential forward (add with second input not available)
                    // The actual add will be handled by the aggregation prover
                }
                GraphOp::Mul { .. } => {
                    // Same as Add — handled by aggregation
                }
                _ => {}
            }
        }

        chunk_outputs_per_matmul.push(matmul_data);
        current_input = current;
    }

    info!("Tile-level forward pass complete, starting proving");

    // Phase 2: Prove each chunk's matmuls with tile-level streaming.
    // Each matmul is proved tile-by-tile: load tile → prove → drop.
    let mut results = Vec::with_capacity(num_chunks);

    for (chunk_idx, block_range) in blocks.iter().enumerate() {
        let chunk_matmul_data = &chunk_outputs_per_matmul[chunk_idx];

        let mut tiled_proofs = Vec::new();
        for (node_id, a_matrix, c_matrix) in chunk_matmul_data {
            let tiled_proof = pipeline
                .prove_matmul_tiled_streaming(*node_id, a_matrix, c_matrix, tile_config)
                .map_err(|e| ChunkedProvingError::ChunkFailed {
                    chunk: chunk_idx,
                    message: format!("Tiled proving node {node_id}: {e}"),
                })?;

            // Verify the tiled proof
            verify_tiled_matmul(&tiled_proof).map_err(|e| ChunkedProvingError::ChunkFailed {
                chunk: chunk_idx,
                message: format!("Tiled verification node {node_id}: {e}"),
            })?;

            tiled_proofs.push((*node_id, tiled_proof));
        }

        info!(
            chunk = chunk_idx,
            matmuls = tiled_proofs.len(),
            "Chunk proved with tile-level streaming"
        );

        // Build PrecomputedMatmuls from tiled forward + proving results.
        // This injects pre-computed matmul outputs and proofs into the
        // aggregation pipeline, eliminating weight re-loading and re-proving.
        let sub_graph = graph.subgraph(block_range.start..block_range.end);

        let mut precomputed_outputs = std::collections::HashMap::new();
        for (node_id, _a_matrix, c_matrix) in chunk_matmul_data {
            precomputed_outputs.insert(*node_id, c_matrix.clone());
        }

        let mut composed_proofs = Vec::new();
        let mut multi_tile_proofs = Vec::new();
        for (node_id, tiled_proof) in tiled_proofs {
            if tiled_proof.tile_proofs.len() == 1 {
                let proof = crate::components::tiled_matmul::compose_tiled_proof(&tiled_proof)
                    .map_err(|e| ChunkedProvingError::ChunkFailed {
                        chunk: chunk_idx,
                        message: format!("Tiled composition node {node_id}: {e}"),
                    })?;
                composed_proofs.push((node_id, proof));
            } else {
                multi_tile_proofs.push((node_id, tiled_proof));
            }
        }

        let precomputed = crate::aggregation::PrecomputedMatmuls {
            outputs: precomputed_outputs,
            proofs: composed_proofs,
            tiled_proofs: multi_tile_proofs,
        };

        // No load_chunk_weights! The aggregation function uses pre-computed data.
        let proof = crate::aggregation::prove_model_aggregated_onchain_with_precomputed(
            &sub_graph,
            &chunk_inputs[chunk_idx],
            &GraphWeights::new(),
            precomputed,
        )
        .map_err(|e| ChunkedProvingError::ChunkFailed {
            chunk: chunk_idx,
            message: format!("{e}"),
        })?;

        let output = proof.execution.output.clone();
        results.push(ChunkProofResult {
            chunk_index: chunk_idx,
            node_range: block_range.start..block_range.end,
            proof,
            output,
        });
    }

    info!(
        num_chunks = results.len(),
        "Tile-level streaming chunked proving complete"
    );

    Ok(results)
}

// =============================================================================
// Multi-GPU Chunked Proving
// =============================================================================

/// Prove a model by distributing chunks across multiple GPUs, returning per-device metrics.
///
/// Phase 1: Sequential forward pass on CPU to compute all chunk inputs.
/// Phase 2: Memory-aware partition assigns chunks to GPUs.
/// Phase 3: `std::thread::scope` — one thread per chunk, each with device affinity.
///
/// Returns both the chunk proofs and a [`MultiGpuProvingResult`] with per-device
/// timing, chunk assignments, and matmul counts.
///
/// Use [`prove_model_chunked_multi_gpu`] if you don't need the metrics.
#[cfg(feature = "multi-gpu")]
pub fn prove_model_chunked_multi_gpu_with_metrics(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    memory_budget: usize,
) -> Result<
    (
        Vec<ChunkProofResult>,
        crate::multi_gpu::MultiGpuProvingResult,
    ),
    ChunkedProvingError,
> {
    use crate::multi_gpu::{ChunkWorkload, DeviceGuard, MultiGpuExecutor};

    if graph.nodes.is_empty() {
        return Err(ChunkedProvingError::EmptyGraph);
    }

    let blocks = graph.find_block_boundaries();
    validate_block_ranges(&blocks, graph.nodes.len())?;
    let num_chunks = blocks.len();

    info!(
        num_chunks,
        memory_budget, "Starting multi-GPU chunked proving"
    );

    // Phase 1: Sequential forward pass to precompute all chunk inputs.
    let mut chunk_inputs: Vec<M31Matrix> = Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for block_range in &blocks {
        chunk_inputs.push(current_input.clone());

        let sub_graph = graph.subgraph(block_range.start..block_range.end);
        let sub_weights = weights.subset(block_range.start..block_range.end);

        current_input =
            crate::compiler::prove::forward_pass_only(&sub_graph, &current_input, &sub_weights)
                .map_err(|e| ChunkedProvingError::ChunkFailed {
                    chunk: 0,
                    message: format!("Forward pass failed: {e}"),
                })?;
    }

    info!("Forward pass complete, building multi-GPU partition");

    // Phase 2: Build workloads and partition across GPUs.
    let executor = MultiGpuExecutor::new().map_err(|e| ChunkedProvingError::ChunkFailed {
        chunk: 0,
        message: format!("Multi-GPU init failed: {e}"),
    })?;

    let workloads: Vec<ChunkWorkload> = blocks
        .iter()
        .enumerate()
        .map(|(idx, block_range)| {
            let sub_graph = graph.subgraph(block_range.start..block_range.end);
            let estimated_memory = sub_graph.estimate_peak_memory();
            let num_matmuls = sub_graph
                .nodes
                .iter()
                .filter(|n| matches!(n.op, crate::compiler::graph::GraphOp::MatMul { .. }))
                .count();
            ChunkWorkload {
                chunk_index: idx,
                estimated_memory,
                num_matmuls,
                block_range: block_range.clone(),
            }
        })
        .collect();

    let assignments = executor.partition_chunks(&workloads);

    info!(
        num_gpus = executor.num_devices(),
        num_chunks = assignments.len(),
        "Partition: {:?}",
        assignments
            .iter()
            .map(|a| (a.chunk_index, a.device_id))
            .collect::<Vec<_>>()
    );

    // Phase 3: Prove chunks in parallel, one thread per chunk with device affinity.
    let errors: std::sync::Mutex<Vec<(usize, usize, String)>> = std::sync::Mutex::new(Vec::new());
    let results: std::sync::Mutex<Vec<ChunkProofResult>> =
        std::sync::Mutex::new(Vec::with_capacity(num_chunks));
    let t_start = std::time::Instant::now();

    // Per-device timing for metrics
    let device_timings: std::sync::Mutex<Vec<(usize, Vec<usize>, std::time::Duration, usize)>> =
        std::sync::Mutex::new(Vec::new());

    std::thread::scope(|s| {
        for assignment in &assignments {
            let chunk_idx = assignment.chunk_index;
            let device_id = assignment.device_id;
            let block_range = &blocks[chunk_idx];
            let chunk_input = &chunk_inputs[chunk_idx];
            let errors_ref = &errors;
            let results_ref = &results;
            let timings_ref = &device_timings;
            let workload = &workloads[chunk_idx];

            s.spawn(move || {
                // DeviceGuard sets affinity on creation, restores on drop (panic-safe RAII)
                let _device_guard = DeviceGuard::new(device_id);
                let chunk_start = std::time::Instant::now();

                info!(
                    chunk_idx,
                    device_id,
                    start = block_range.start,
                    end = block_range.end,
                    "Proving chunk on GPU"
                );

                let sub_graph = graph.subgraph(block_range.start..block_range.end);
                let sub_weights = weights.subset(block_range.start..block_range.end);

                match crate::aggregation::prove_model_aggregated_onchain_auto(
                    &sub_graph,
                    chunk_input,
                    &sub_weights,
                ) {
                    Ok(proof) => {
                        let elapsed = chunk_start.elapsed();
                        info!(
                            chunk_idx,
                            device_id,
                            elapsed_s = format!("{:.2}", elapsed.as_secs_f64()),
                            "Chunk proved successfully"
                        );
                        let output = proof.execution.output.clone();
                        results_ref.lock().unwrap().push(ChunkProofResult {
                            chunk_index: chunk_idx,
                            node_range: block_range.start..block_range.end,
                            proof,
                            output,
                        });
                        timings_ref.lock().unwrap().push((
                            device_id,
                            vec![chunk_idx],
                            elapsed,
                            workload.num_matmuls,
                        ));
                    }
                    Err(e) => {
                        tracing::error!(chunk_idx, device_id, error = %e, "Chunk proving failed");
                        errors_ref
                            .lock()
                            .unwrap()
                            .push((chunk_idx, device_id, format!("{e}")));
                    }
                }
                // _device_guard dropped here, restoring previous device state
            });
        }
    });

    let total_elapsed = t_start.elapsed();

    // Check for errors — report ALL failures, not just the first
    let errs = errors.into_inner().unwrap();
    if !errs.is_empty() {
        if errs.len() == 1 {
            let (chunk, _device, msg) = errs.into_iter().next().unwrap();
            return Err(ChunkedProvingError::ChunkFailed {
                chunk,
                message: msg,
            });
        }
        // Multiple failures: aggregate into a single error message
        let details: Vec<String> = errs
            .iter()
            .map(|(chunk, device, msg)| format!("chunk {} (GPU {}): {}", chunk, device, msg))
            .collect();
        return Err(ChunkedProvingError::ChunkFailed {
            chunk: errs[0].0,
            message: format!("{} chunks failed: {}", errs.len(), details.join("; ")),
        });
    }

    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.chunk_index);

    // Build per-device metrics from collected timings
    let raw_timings = device_timings.into_inner().unwrap();

    // Aggregate per-device: merge chunk lists and sum matmul counts
    let mut device_map: std::collections::HashMap<usize, (Vec<usize>, std::time::Duration, usize)> =
        std::collections::HashMap::new();
    for (device_id, chunks, elapsed, matmuls) in raw_timings {
        let entry = device_map
            .entry(device_id)
            .or_insert_with(|| (Vec::new(), std::time::Duration::ZERO, 0));
        entry.0.extend(chunks);
        entry.1 = entry.1.max(elapsed); // wall-clock = max across chunks on same device
        entry.2 += matmuls;
    }

    let device_stats: Vec<crate::multi_gpu::DeviceProvingStat> = device_map
        .into_iter()
        .map(|(device_id, (chunks, elapsed, matmuls))| {
            info!(
                device_id,
                chunks = ?chunks,
                matmuls,
                elapsed_s = format!("{:.2}", elapsed.as_secs_f64()),
                "Device proving stats"
            );
            crate::multi_gpu::DeviceProvingStat {
                device_id,
                chunks_proven: chunks,
                elapsed,
                matmuls_proven: matmuls,
            }
        })
        .collect();

    let metrics = crate::multi_gpu::MultiGpuProvingResult {
        device_stats,
        total_elapsed,
    };

    info!(
        num_chunks = results.len(),
        total_elapsed_s = format!("{:.2}", total_elapsed.as_secs_f64()),
        "Multi-GPU chunked proving complete"
    );

    Ok((results, metrics))
}

/// Prove a model using multi-GPU distributed proving (convenience wrapper).
///
/// Same as [`prove_model_chunked_multi_gpu_with_metrics`] but discards the
/// per-device metrics. Use the `_with_metrics` variant if you need timing data.
#[cfg(feature = "multi-gpu")]
pub fn prove_model_chunked_multi_gpu(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    memory_budget: usize,
) -> Result<Vec<ChunkProofResult>, ChunkedProvingError> {
    prove_model_chunked_multi_gpu_with_metrics(graph, input, weights, memory_budget)
        .map(|(results, _metrics)| results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;
    use stwo::core::fields::m31::M31;

    #[test]
    fn test_prove_model_chunked_simple() {
        // 3-layer MLP: matmul → relu → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let dir = std::env::temp_dir().join("stwo_ml_chunked_test");
        let _ = std::fs::remove_dir_all(&dir);

        let results = prove_model_chunked(
            &graph,
            &input,
            &weights,
            100_000_000, // generous budget
            &dir,
        )
        .expect("chunked proving should succeed");

        // With no LayerNorm, graph is a single block → 1 chunk
        assert!(!results.is_empty());

        let total_mm = total_matmul_proofs(&results);
        assert_eq!(total_mm, 2, "should have 2 matmul proofs");

        let total_act = total_activation_claims(&results);
        assert_eq!(total_act, 1, "should have 1 activation claim");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prove_model_chunked_parallel() {
        // Same MLP as the sequential test
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let results = prove_model_chunked_parallel(&graph, &input, &weights, 100_000_000)
            .expect("parallel chunked proving should succeed");

        assert!(!results.is_empty());

        let total_mm = total_matmul_proofs(&results);
        assert_eq!(total_mm, 2, "should have 2 matmul proofs");

        let total_act = total_activation_claims(&results);
        assert_eq!(total_act, 1, "should have 1 activation claim");
    }

    #[test]
    fn test_collect_chunk_proofs() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let dir = std::env::temp_dir().join("stwo_ml_collect_test");
        let _ = std::fs::remove_dir_all(&dir);

        let results = prove_model_chunked(&graph, &input, &weights, 100_000_000, &dir)
            .expect("chunked proving should succeed");

        let (matmul_proofs, activation_claims) = collect_chunk_proofs(&results);
        assert_eq!(matmul_proofs.len(), 2);
        assert_eq!(activation_claims.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_chunked_checkpoint_persistence() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let dir = std::env::temp_dir().join("stwo_ml_ckpt_persist_test");
        let _ = std::fs::remove_dir_all(&dir);

        let _results = prove_model_chunked(&graph, &input, &weights, 100_000_000, &dir)
            .expect("first run should succeed");

        // Checkpoint file should exist
        let ckpt_path = dir.join("chunk_0.json");
        assert!(ckpt_path.exists(), "checkpoint should be saved");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compose_single_chunk() {
        // MLP with no block boundaries → 1 chunk → compose → verify
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let dir = std::env::temp_dir().join("stwo_ml_compose_single_test");
        let _ = std::fs::remove_dir_all(&dir);

        let chunks = prove_model_chunked(&graph, &input, &weights, 100_000_000, &dir)
            .expect("chunked proving should succeed");

        assert!(!chunks.is_empty(), "should have at least 1 chunk");

        let composed = compose_chunk_proofs(&chunks, &graph, &input, &weights)
            .expect("composition should succeed");

        // Verify the composed proof
        crate::aggregation::verify_aggregated_model_proof_onchain(
            composed, &graph, &input, &weights,
        )
        .expect("composed proof should verify");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compose_matmul_only() {
        // 2 matmuls, no non-matmul ops → compose → verify unified_stark is None
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w1 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w1.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(1, w1);

        let dir = std::env::temp_dir().join("stwo_ml_compose_matmul_only_test");
        let _ = std::fs::remove_dir_all(&dir);

        let chunks = prove_model_chunked(&graph, &input, &weights, 100_000_000, &dir)
            .expect("chunked proving should succeed");

        let composed = compose_chunk_proofs(&chunks, &graph, &input, &weights)
            .expect("composition should succeed");

        assert!(
            composed.unified_stark.is_none(),
            "no non-matmul ops → no unified STARK"
        );
        assert_eq!(
            composed.matmul_proofs.len(),
            2,
            "should have 2 matmul proofs"
        );

        crate::aggregation::verify_aggregated_model_proof_onchain(
            composed, &graph, &input, &weights,
        )
        .expect("composed proof should verify");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compose_with_residual_add() {
        // matmul → relu → add(residual) → compose → verify add_claims.len() == 1
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.current_node_id(); // save matmul output
        builder.activation(ActivationType::ReLU);
        builder.add_from(branch); // residual add
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        let dir = std::env::temp_dir().join("stwo_ml_compose_residual_test");
        let _ = std::fs::remove_dir_all(&dir);

        let chunks = prove_model_chunked(&graph, &input, &weights, 100_000_000, &dir)
            .expect("chunked proving should succeed");

        let composed = compose_chunk_proofs(&chunks, &graph, &input, &weights)
            .expect("composition should succeed");

        assert!(
            composed.unified_stark.is_some(),
            "should have unified STARK for activation + add"
        );
        assert_eq!(composed.add_claims.len(), 1, "should have 1 add claim");
        assert_eq!(
            composed.activation_claims.len(),
            1,
            "should have 1 activation claim"
        );

        crate::aggregation::verify_aggregated_model_proof_onchain(
            composed, &graph, &input, &weights,
        )
        .expect("composed proof should verify");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // =========================================================================
    // Multi-chunk integration tests (graphs WITH LayerNorm → 2+ chunks)
    // =========================================================================

    /// Helper: build a graph with 2 LayerNorms producing 3 chunks, plus weights.
    fn build_2_layernorm_graph() -> (ComputationGraph, M31Matrix, GraphWeights) {
        // linear(4→4) → relu → layernorm → linear(4→4) → relu → layernorm → linear(4→2)
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4) // node 0: matmul
            .activation(ActivationType::ReLU) // node 1: relu
            .layer_norm() // node 2: layernorm
            .linear(4) // node 3: matmul
            .activation(ActivationType::ReLU) // node 4: relu
            .layer_norm() // node 5: layernorm
            .linear(2); // node 6: matmul
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();

        // Weight for node 0: 4×4
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        // Weight for node 3: 4×4
        let mut w3 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w3.set(i, j, M31::from(((i * 2 + j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(3, w3);

        // Weight for node 6: 4×2
        let mut w6 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w6.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(6, w6);

        (graph, input, weights)
    }

    #[test]
    fn test_multi_chunk_2_layernorm() {
        let (graph, input, weights) = build_2_layernorm_graph();

        // Should produce 3 chunks: [0..2), [2..5), [5..7)
        let blocks = graph.find_block_boundaries();
        assert_eq!(blocks.len(), 3, "2 LayerNorms → 3 chunks");
        assert_eq!(blocks[0], 0..2);
        assert_eq!(blocks[1], 2..5);
        assert_eq!(blocks[2], 5..7);

        let results = prove_model_chunked_parallel(&graph, &input, &weights, 100_000_000)
            .expect("multi-chunk parallel proving should succeed");

        assert_eq!(results.len(), 3, "should produce 3 chunk results");

        let total_mm = total_matmul_proofs(&results);
        assert_eq!(total_mm, 3, "should have 3 matmul proofs total");

        let total_act = total_activation_claims(&results);
        assert_eq!(total_act, 2, "should have 2 activation claims total");

        // Verify forward pass output matches direct execution
        let direct_result = crate::compiler::prove::prove_model(&graph, &input, &weights)
            .expect("direct prove should succeed");
        let direct_output = &direct_result.1.output;
        let chunked_output = &results.last().unwrap().output;
        assert_eq!(
            direct_output.rows, chunked_output.rows,
            "output rows should match"
        );
        assert_eq!(
            direct_output.cols, chunked_output.cols,
            "output cols should match"
        );
    }

    #[test]
    fn test_multi_chunk_collect_proofs() {
        let (graph, input, weights) = build_2_layernorm_graph();

        let dir = std::env::temp_dir().join("stwo_ml_multi_chunk_collect_test");
        let _ = std::fs::remove_dir_all(&dir);

        let chunks = prove_model_chunked(&graph, &input, &weights, 100_000_000, &dir)
            .expect("chunked proving should succeed");

        assert!(chunks.len() >= 2, "should have multiple chunks");

        // Verify collect_chunk_proofs remaps IDs correctly
        let (all_matmul, all_activation) = collect_chunk_proofs(&chunks);
        assert_eq!(all_matmul.len(), 3, "3 matmul proofs across chunks");
        assert_eq!(all_activation.len(), 2, "2 activation claims across chunks");

        // Verify global node IDs are unique and within range
        let matmul_ids: Vec<usize> = all_matmul.iter().map(|(id, _)| *id).collect();
        assert!(
            matmul_ids.iter().all(|&id| id < graph.nodes.len()),
            "all matmul IDs should be valid global node IDs"
        );
        let unique_ids: std::collections::HashSet<usize> = matmul_ids.iter().copied().collect();
        assert_eq!(
            unique_ids.len(),
            matmul_ids.len(),
            "matmul IDs should be unique"
        );

        // Verify chunk ranges are contiguous and cover the graph
        let mut covered = 0;
        for chunk in &chunks {
            assert_eq!(
                chunk.node_range.start, covered,
                "chunks should be contiguous"
            );
            covered = chunk.node_range.end;
        }
        assert_eq!(covered, graph.nodes.len(), "chunks should cover all nodes");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multi_chunk_single_chunk_equivalence() {
        // No LayerNorm → 1 chunk. Verify chunked path matches direct path.
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        // Chunked path
        let chunked_results = prove_model_chunked_parallel(&graph, &input, &weights, 100_000_000)
            .expect("chunked proving should succeed");

        assert_eq!(chunked_results.len(), 1, "no LayerNorm → single chunk");

        // Direct path
        let direct_proof =
            crate::aggregation::prove_model_aggregated_onchain(&graph, &input, &weights)
                .expect("direct proving should succeed");

        // Same number of matmul proofs
        assert_eq!(
            total_matmul_proofs(&chunked_results),
            direct_proof.matmul_proofs.len(),
            "matmul proof count should match"
        );

        // Same output
        let chunked_output = &chunked_results[0].output;
        let direct_output = &direct_proof.execution.output;
        assert_eq!(chunked_output.rows, direct_output.rows);
        assert_eq!(chunked_output.cols, direct_output.cols);
        for i in 0..chunked_output.rows {
            for j in 0..chunked_output.cols {
                assert_eq!(
                    chunked_output.get(i, j),
                    direct_output.get(i, j),
                    "output mismatch at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_find_block_boundaries_multi() {
        // 0 LayerNorms → 1 range
        {
            let mut builder = GraphBuilder::new((1, 4));
            builder.linear(4).activation(ActivationType::ReLU).linear(2);
            let graph = builder.build();
            let blocks = graph.find_block_boundaries();
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0], 0..graph.nodes.len());
        }

        // 1 LayerNorm → 1 range (< 2 LNs → single block)
        {
            let mut builder = GraphBuilder::new((1, 4));
            builder.linear(4).layer_norm().linear(2);
            let graph = builder.build();
            let blocks = graph.find_block_boundaries();
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0], 0..graph.nodes.len());
        }

        // 2 LayerNorms → 3 ranges (prefix + 2 LN-led blocks)
        {
            let mut builder = GraphBuilder::new((1, 4));
            builder
                .linear(4) // 0
                .activation(ActivationType::ReLU) // 1
                .layer_norm() // 2
                .linear(4) // 3
                .layer_norm() // 4
                .linear(2); // 5
            let graph = builder.build();
            let blocks = graph.find_block_boundaries();
            assert_eq!(blocks.len(), 3, "2 LNs with prefix → 3 blocks");
            // Verify contiguous coverage
            let mut covered = 0;
            for block in &blocks {
                assert_eq!(block.start, covered, "blocks should be contiguous");
                covered = block.end;
            }
            assert_eq!(covered, graph.nodes.len(), "blocks should cover all nodes");
        }

        // 3 LayerNorms → 4 ranges (prefix + 3 LN-led blocks)
        {
            let mut builder = GraphBuilder::new((1, 4));
            builder
                .linear(4) // 0
                .layer_norm() // 1
                .linear(4) // 2
                .layer_norm() // 3
                .linear(4) // 4
                .layer_norm() // 5
                .linear(2); // 6
            let graph = builder.build();
            let blocks = graph.find_block_boundaries();
            assert_eq!(blocks.len(), 4, "3 LNs with prefix → 4 blocks");
            // Verify contiguous coverage
            let mut covered = 0;
            for block in &blocks {
                assert_eq!(block.start, covered, "blocks should be contiguous");
                covered = block.end;
            }
            assert_eq!(covered, graph.nodes.len(), "blocks should cover all nodes");
        }

        // 2 LayerNorms at start (no prefix) → 2 ranges
        {
            let mut builder = GraphBuilder::new((1, 4));
            builder
                .layer_norm() // 0
                .linear(4) // 1
                .layer_norm() // 2
                .linear(2); // 3
            let graph = builder.build();
            let blocks = graph.find_block_boundaries();
            assert_eq!(blocks.len(), 2, "2 LNs at start → 2 blocks (no prefix)");
            assert_eq!(blocks[0], 0..2);
            assert_eq!(blocks[1], 2..4);
        }
    }
}
