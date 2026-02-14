//! GKR (Goldwasser-Kalai-Rothblum) protocol engine for ML inference proving.
//!
//! Replaces per-matmul independent proofs with a single layered interactive
//! proof that walks the computation graph from output to input.
//!
//! ## Architecture
//!
//! - **Layer-typed GKR**: Each layer has a specialized reduction protocol
//!   (sumcheck for matmul, identity for add, LogUp for activations).
//! - **GPU-accelerated**: MatMul reductions reuse existing GPU sumcheck kernels.
//! - **SIMD batching**: Identical transformer blocks are batched into a single
//!   GKR pass with a randomized block selection dimension.
//!
//! ## Module Layout
//!
//! - [`types`] — Proof types (`GKRProof`, `LayerProof`, `LogUpProof`), error types, claims.
//! - [`circuit`] — `LayeredCircuit` representation and compiler from `ComputationGraph`.
//! - [`prover`] — Layer-by-layer GKR prover: matmul sumcheck, add/mul reductions,
//!   activation LogUp. GPU variants (`prove_gkr_gpu`, `prove_gkr_simd_gpu`) when `cuda-runtime`.
//! - [`verifier`] — Fiat-Shamir transcript replay verifier. Checks each layer proof
//!   against the circuit structure without needing intermediate values.
//!
//! ## Integration with Aggregation Pipeline
//!
//! GKR is an **optional, additive** layer on top of the standard proving pipeline.
//! When enabled via [`prove_model_aggregated_onchain_gkr()`](crate::aggregation::prove_model_aggregated_onchain_gkr):
//!
//! 1. The standard pipeline runs first (unified STARK + per-matmul sumchecks)
//! 2. The `ComputationGraph` is compiled into a `LayeredCircuit`
//! 3. `prove_gkr()` produces a `GKRProof` from the execution trace
//! 4. The proof is attached as `AggregatedModelProofOnChain.gkr_proof`
//!
//! The verifier checks both the STARK proof and the GKR proof independently.
//!
//! ## Verification
//!
//! `verify_gkr()` is called inside `verify_aggregated_model_proof_onchain()`
//! when `gkr_proof` is present. It replays the Fiat-Shamir transcript and
//! checks each layer proof against the circuit structure.

pub mod types;
pub mod circuit;
pub mod prover;
pub mod verifier;

pub use types::{GKRProof, GKRClaim, GKRError, LayerProof, ReductionOutput, RoundPolyDeg3, LogUpProof, DeferredProof};
pub use circuit::{LayeredCircuit, CircuitLayer, LayerType, LayerCounts, SIMDBatchConfig};
pub use prover::prove_gkr;
#[cfg(feature = "cuda-runtime")]
pub use prover::prove_gkr_gpu;
#[cfg(feature = "cuda-runtime")]
pub use prover::prove_gkr_simd_gpu;
pub use verifier::{verify_gkr, verify_gkr_with_execution, verify_gkr_simd};
