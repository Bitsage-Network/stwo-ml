//! VM Prover — generates GKR proofs from captured execution traces.
//!
//! The key function is `prove_from_forward_result()` which takes a
//! `ForwardPassResult` (containing all intermediates from execution)
//! and generates the full cryptographic proof without re-executing.

use crate::aggregation::{AggregatedModelProofOnChain, ForwardPassResult};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::components::matmul::M31Matrix;
use crate::weight_cache::SharedWeightCache;
use crate::policy::PolicyConfig;

/// Prove from a captured forward pass result.
///
/// This is the true trace replay path — the forward pass has already been
/// executed and all intermediates captured in `ForwardPassResult`. This
/// function runs only the cryptographic proof generation (GKR + STARK).
pub fn prove_from_forward_result(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    fwd: ForwardPassResult,
    weight_cache: Option<&SharedWeightCache>,
    policy: Option<&PolicyConfig>,
) -> Result<AggregatedModelProofOnChain, ProverError> {
    crate::aggregation::prove_from_forward_result(
        graph, input, weights, fwd, weight_cache, policy,
    ).map_err(|e| ProverError::ProvingFailed(format!("{e}")))
}

/// Errors from the VM prover.
#[derive(Debug, thiserror::Error)]
pub enum ProverError {
    #[error("Proving failed: {0}")]
    ProvingFailed(String),
    #[error("Empty trace — no intermediates captured")]
    EmptyTrace,
}
