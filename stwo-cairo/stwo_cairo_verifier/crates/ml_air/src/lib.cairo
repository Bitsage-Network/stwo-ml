/// Obelysk ML Air — Cairo verification of ML STARK proofs.
///
/// Verifies ML inference proofs containing:
/// - Matmul sumcheck proofs (verified via round polynomial consistency)
/// - Activation LogUp STARK proofs (verified via `verifier_core::verify`)
///
/// Architecture:
/// ```
/// MLProof {
///     claim: MLClaim,                     // Model metadata + commitments
///     matmul_proofs: Array<MatMulSumcheckProofOnChain>,
///     channel_salt: Option<u64>,
/// }
/// ```
pub mod claim;
pub mod ml_air;
pub mod sumcheck;
pub mod mle;
pub mod components;

use stwo_verifier_core::channel::{Channel, ChannelTrait};
use claim::{MLProof, MLClaimMixTrait, MLVerificationOutput};
use sumcheck::verify_matmul_sumcheck;

/// Verify an ML inference proof.
///
/// 1. Initialize Fiat-Shamir channel
/// 2. Mix claim data into channel
/// 3. Verify each matmul sumcheck proof
/// 4. Return verification output (model_id, io_commitment, etc.)
pub fn verify_ml(proof: @MLProof) -> MLVerificationOutput {
    // Initialize channel
    let mut channel: Channel = Default::default();

    // Apply optional salt for rerandomization
    if let Option::Some(salt) = proof.channel_salt {
        channel.mix_u64(*salt);
    };

    // Mix claim into channel
    let claim = proof.claim;
    claim.mix_into(ref channel);

    // Verify matmul sumcheck proofs
    let mut matmul_idx: u32 = 0;
    let n_matmuls = proof.matmul_proofs.len();
    while matmul_idx < n_matmuls {
        let matmul_proof = proof.matmul_proofs.at(matmul_idx);
        let ok = verify_matmul_sumcheck(ref channel, matmul_proof);
        assert!(ok, "Matmul sumcheck {} verification failed", matmul_idx);
        matmul_idx += 1;
    };

    // Build verification output
    MLVerificationOutput {
        model_id: *claim.model_id,
        io_commitment: *claim.io_commitment,
        weight_commitment: *claim.weight_commitment,
        num_layers: *claim.num_layers,
        num_matmuls: n_matmuls,
        verified: true,
    }
}
