/// Obelysk ML Verifier — Cairo executable for recursive ML proof verification.
///
/// This is a `#[executable]` that:
///   1. Deserializes an MLProof from program arguments
///   2. Verifies all matmul sumcheck proofs
///   3. Verifies the activation STARK proof (when wired)
///   4. Returns MLVerificationOutput (model_id, commitments, verified status)
///
/// When run through `cairo-prove`, the execution trace of this verifier
/// produces a compact recursive STARK proof (~1KB) that can be verified
/// on-chain in a single transaction.
///
/// Usage:
///   scarb execute --package obelysk_ml_verifier --arguments-file proof.json

use obelysk_ml_air::claim::{MLProof, MLVerificationOutput};
use obelysk_ml_air::verify_ml;

#[executable]
fn main(proof: MLProof) -> MLVerificationOutput {
    verify_ml(proof)
}
