//! VM Executor — runs forward pass + proving and captures the result.
//!
//! The executor calls the existing proving pipeline which produces both
//! the model output and the cryptographic proof. Results are wrapped
//! into an `ExecutionTrace` for status tracking and proof management.

use std::time::Instant;

use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::components::attention::ModelKVCache;
use crate::components::matmul::M31Matrix;
use crate::policy::PolicyConfig;
use crate::vm::trace::ExecutionTrace;

/// Execute the forward pass + GKR proof, returning a complete trace.
///
/// Currently runs execution and proving together (the existing pipeline
/// doesn't separate them). The resulting trace includes the proof, output,
/// and all commitment data needed for on-chain submission.
pub fn execute_and_prove(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    kv_cache: Option<&mut ModelKVCache>,
    policy: Option<&PolicyConfig>,
    input_tokens: Vec<u32>,
    model_id: String,
) -> Result<ExecutionTrace, ExecutorError> {
    let t_start = Instant::now();

    let proof_result = if let Some(kv) = kv_cache {
        crate::aggregation::prove_model_pure_gkr_prefill_with_cache(
            graph, input, weights, kv, None, policy,
        )
    } else {
        crate::aggregation::prove_model_pure_gkr_auto_with_cache(graph, input, weights, None, policy)
    };

    let proof = proof_result.map_err(|e| ExecutorError::ForwardError(format!("{e}")))?;
    let inference_time_ms = t_start.elapsed().as_millis() as u64;

    // Extract the output from the proof's execution data
    let output = proof.execution.output.clone();
    let io_commitment = Some(proof.io_commitment);
    let policy_commitment = proof.policy_commitment;

    Ok(ExecutionTrace {
        model_id,
        input_tokens,
        output,
        io_commitment,
        policy_commitment,
        kv_commitment_before: proof.prev_kv_cache_commitment,
        kv_commitment_after: proof.kv_cache_commitment,
        tokenization_commitment: None,
        inference_time_ms,
        num_tokens: input.rows,
        position_offset: 0,
        proof: Some(proof),
    })
}

/// Errors from the VM executor.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("Missing weight for node {0}")]
    MissingWeight(usize),
    #[error("Forward pass error: {0}")]
    ForwardError(String),
}
