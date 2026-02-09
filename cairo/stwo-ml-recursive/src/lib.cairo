/// STWO ML Recursive — Cairo executable for recursive proof aggregation.
///
/// Verifies N matmul sumcheck proofs + layer commitment chain inside the
/// Cairo VM, then the stwo-cairo-prover proves this execution into a
/// single compact Circle STARK proof.
///
/// ```text
/// GPU Prover (Rust)                Cairo Executable                On-Chain
/// ─────────────────                ────────────────                ────────
/// prove_model_pipeline()  →  verify all matmul proofs  →  verify single STARK
///   N matmul proofs              + layer chain              (constant size)
///   N layer headers           → aggregate_hash output
/// ```
pub mod types;
pub mod aggregate;

use types::RecursiveInput;
use aggregate::verify_all_and_aggregate;

/// Entry point for the recursive verifier executable.
///
/// Takes a serialized `RecursiveInput` (deserialized via Serde from felt252 args),
/// verifies all proofs, and returns:
///   [aggregate_hash, num_verified, num_layers, model_commitment, io_commitment]
#[executable]
fn main(input: RecursiveInput) -> Array<felt252> {
    let model_commitment = input.model_commitment;
    let io_commitment = input.io_commitment;

    let (aggregate_hash, num_verified, num_layers) = verify_all_and_aggregate(input);

    array![
        aggregate_hash,
        num_verified.into(),
        num_layers.into(),
        model_commitment,
        io_commitment,
    ]
}
