//! VM Prover — placeholder for async proof generation from traces.
//!
//! Currently, the executor runs proving synchronously and stores the proof
//! in the `ExecutionTrace`. The prover module will be expanded to support
//! true async proving from captured traces (Phase 1b of the VM roadmap).

/// Placeholder: in the current architecture, proofs are generated during
/// execution via `execute_and_prove()`. This module will be expanded to
/// support `prove_from_trace()` when execution is decoupled from proving.
pub fn _future_prove_from_trace() {
    // Will be implemented when the executor supports capture-only mode
    // (forward pass without proving) and traces can be replayed.
}
