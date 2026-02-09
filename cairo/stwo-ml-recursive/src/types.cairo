/// Types for recursive ML proof aggregation.
///
/// These structs match the Rust-side serialization layout exactly —
/// field order determines Serde deserialization order.

use stwo_ml_verify_core::sumcheck::MatMulSumcheckProof;
use stwo_ml_verify_core::layer_chain::LayerProofHeader;

/// Input to the recursive verifier executable.
///
/// Deserialized from felt252 args via Cairo's `Serde` trait.
/// Field order must exactly match `serialize_recursive_input()` in Rust.
#[derive(Drop, Serde)]
pub struct RecursiveInput {
    /// Registered model identifier.
    pub model_id: felt252,
    /// Poseidon commitment over model weights.
    pub model_commitment: felt252,
    /// Poseidon(input_data || output_data).
    pub io_commitment: felt252,
    /// Number of matmul sumcheck proofs (must equal matmul_proofs.len()).
    pub num_matmul_proofs: u32,
    /// Per-layer matmul sumcheck proofs.
    pub matmul_proofs: Array<MatMulSumcheckProof>,
    /// Layer commitment chain headers.
    pub layer_headers: Array<LayerProofHeader>,
    /// Poseidon commitment of the model's input activations.
    pub model_input_commitment: felt252,
    /// Poseidon commitment of the model's output activations.
    pub model_output_commitment: felt252,
    /// TEE attestation report hash (0 if unavailable).
    pub tee_report_hash: felt252,
}

/// Output of the recursive verifier.
///
/// Returned as Array<felt252> from the executable.
#[derive(Drop)]
pub struct RecursiveOutput {
    /// Aggregate hash binding all verified proofs together.
    pub aggregate_hash: felt252,
    /// Number of matmul proofs successfully verified.
    pub num_verified: u32,
    /// Number of layers in the commitment chain.
    pub num_layers: u32,
    /// Echo of model_commitment (for caller cross-check).
    pub model_commitment: felt252,
    /// Echo of io_commitment (for caller cross-check).
    pub io_commitment: felt252,
}
