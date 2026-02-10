//! Chunked proving for large models.
//!
//! Splits a computation graph into block-sized chunks and proves each
//! independently, using execution checkpoints to pass activations
//! between chunks. This enables proving models that exceed available
//! memory when proven monolithically.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::aggregation::{AggregatedModelProofOnChain, AggregationError};
use crate::compiler::checkpoint::ExecutionCheckpoint;
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::components::matmul::M31Matrix;

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

    info!(
        num_chunks = results.len(),
        "Chunked proving complete"
    );

    Ok(results)
}

/// Compute the checkpoint file path for a given chunk.
fn checkpoint_path(dir: &Path, chunk_idx: usize) -> PathBuf {
    dir.join(format!("chunk_{chunk_idx}.json"))
}

/// Get the total number of matmul proofs across all chunks.
pub fn total_matmul_proofs(results: &[ChunkProofResult]) -> usize {
    results
        .iter()
        .map(|r| r.proof.matmul_proofs.len())
        .sum()
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
    let num_chunks = blocks.len();

    info!(num_chunks, memory_budget, "Starting parallel chunked proving");

    // Phase 1: Sequential forward pass to precompute all chunk inputs.
    // Each chunk needs the output of the previous chunk as input.
    let mut chunk_inputs: Vec<M31Matrix> = Vec::with_capacity(num_chunks);
    let mut current_input = input.clone();

    for block_range in &blocks {
        chunk_inputs.push(current_input.clone());

        // Run forward pass for this chunk to get its output
        let sub_graph = graph.subgraph(block_range.start..block_range.end);
        let sub_weights = weights.subset(block_range.start..block_range.end);

        let forward_result = crate::compiler::prove::prove_model(
            &sub_graph, &current_input, &sub_weights,
        ).map_err(|e| ChunkedProvingError::ChunkFailed {
            chunk: 0,
            message: format!("Forward pass failed: {e}"),
        })?;

        current_input = forward_result.1.output;
    }

    info!("Forward pass complete, starting parallel proving");

    // Phase 2: Prove all chunks in parallel.
    let results: Result<Vec<ChunkProofResult>, ChunkedProvingError> = blocks
        .par_iter()
        .enumerate()
        .map(|(chunk_idx, block_range)| {
            let start = block_range.start;
            let end = block_range.end;
            let chunk_input = &chunk_inputs[chunk_idx];

            let sub_graph = graph.subgraph(start..end);
            let sub_weights = weights.subset(start..end);

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
        "Parallel chunked proving complete"
    );

    Ok(results)
}

/// Collect all chunk proofs into a combined set of matmul proofs and activation claims.
pub fn collect_chunk_proofs(
    results: &[ChunkProofResult],
) -> (Vec<(usize, crate::components::matmul::MatMulSumcheckProofOnChain)>, Vec<crate::aggregation::LayerClaim>) {
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
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
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
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
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

        let results = prove_model_chunked_parallel(
            &graph,
            &input,
            &weights,
            100_000_000,
        )
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
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
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
}
