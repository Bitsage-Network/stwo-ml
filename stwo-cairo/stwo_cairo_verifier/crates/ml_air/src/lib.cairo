/// Obelysk ML Air — Cairo verification of ML STARK proofs.
///
/// Verifies ML inference proofs containing:
/// - Matmul sumcheck proofs (verified via round polynomial consistency)
/// - Activation LogUp STARK proofs (verified via `verifier_core::verify`)
///
/// Architecture:
/// ```
/// MLProof {
///     claim: MLClaim,
///     matmul_proofs: Array<MatMulSumcheckProofOnChain>,
///     channel_salt: Option<u64>,
///     activation_stark_proof: Option<ActivationStarkProof>,
/// }
/// ```
pub mod claim;
pub mod ml_air;
pub mod sumcheck;
pub mod mle;
pub mod components;

use core::num::traits::Zero;
use stwo_verifier_core::channel::{Channel, ChannelTrait};
use stwo_verifier_core::verifier::verify;
use stwo_verifier_core::pcs::PcsConfigTrait;
use stwo_verifier_core::pcs::verifier::{
    CommitmentSchemeVerifierImpl, CommitmentSchemeVerifierTrait, get_trace_lde_log_size,
};
use stwo_verifier_core::Hash;
use stwo_constraint_framework::LookupElementsTrait;
use claim::{
    MLProof, MLClaim, MLClaimMixTrait, MLInteractionClaimMixTrait, ActivationStarkProof,
    MLVerificationOutput, MLClaimLogSizesTrait,
};
use sumcheck::verify_matmul_sumcheck;
use components::activation::ActivationLookupElements;
use ml_air::MLAirNewImpl;

/// Minimum security bits for activation STARK verification.
const SECURITY_BITS: u32 = 96;

/// Proof-of-work bits for the interaction phase.
const INTERACTION_POW_BITS: u32 = 20;

/// Verify an ML inference proof.
///
/// Takes ownership of the proof (like `verify_cairo` does) because the generic
/// STARK verifier requires ownership of `StarkProof`.
///
/// 1. Initialize Fiat-Shamir channel
/// 2. Mix claim data into channel
/// 3. Verify each matmul sumcheck proof
/// 4. If activation STARK proof is present, verify it via generic STARK verifier
/// 5. Return verification output (model_id, io_commitment, etc.)
pub fn verify_ml(proof: MLProof) -> MLVerificationOutput {
    let MLProof { claim, matmul_proofs, channel_salt, activation_stark_proof } = proof;

    // Initialize channel
    let mut channel: Channel = Default::default();

    // Apply optional salt for rerandomization
    if let Option::Some(salt) = channel_salt {
        channel.mix_u64(salt);
    };

    // Mix claim into channel
    claim.mix_into(ref channel);

    // Verify matmul sumcheck proofs
    let mut matmul_idx: u32 = 0;
    let n_matmuls = matmul_proofs.len();
    while matmul_idx < n_matmuls {
        let matmul_proof = matmul_proofs.at(matmul_idx);
        let ok = verify_matmul_sumcheck(ref channel, matmul_proof);
        assert!(ok, "Matmul sumcheck {} verification failed", matmul_idx);
        matmul_idx += 1;
    };

    // Verify activation STARK if present
    if let Option::Some(activation_proof) = activation_stark_proof {
        verify_activation_stark(ref channel, @claim, activation_proof);
    }

    // Build verification output
    MLVerificationOutput {
        model_id: claim.model_id,
        io_commitment: claim.io_commitment,
        weight_commitment: claim.weight_commitment,
        num_layers: claim.num_layers,
        num_matmuls: n_matmuls,
        verified: true,
    }
}

/// Full STARK verification for activation LogUp proofs.
///
/// Follows the `verify_cairo()` 13-step pattern:
///   1. Get PCS config from proof
///   2. Mix PCS config into channel
///   3. Create commitment scheme verifier
///   4. Unpack commitments
///   5. Compute log_sizes per tree
///   6. Commit preprocessed trace
///   7. Mix claim into channel, commit trace
///   8. Verify interaction proof-of-work
///   9. Draw interaction lookup elements
///  10. Verify LogUp sum is zero
///  11. Mix interaction claim, commit interaction trace
///  12. Construct MLAir
///  13. Call generic STARK verify
fn verify_activation_stark(
    ref channel: Channel,
    claim: @MLClaim,
    proof: ActivationStarkProof,
) {
    let ActivationStarkProof {
        activation_claims,
        activation_interaction_claims,
        interaction_claim,
        pcs_config,
        interaction_pow,
        stark_proof,
    } = proof;

    // Step 1: Mix PCS config into channel
    let proof_pcs_config = stark_proof.commitment_scheme_proof.config;
    proof_pcs_config.mix_into(ref channel);

    // Step 2: Create commitment scheme verifier
    let mut commitment_scheme = CommitmentSchemeVerifierImpl::new();

    // Step 3: Unpack commitments [preprocessed, trace, interaction_trace, composition]
    let commitments: @Box<[Hash; 4]> = stark_proof
        .commitment_scheme_proof
        .commitments
        .try_into()
        .unwrap();
    let [
        preprocessed_commitment,
        trace_commitment,
        interaction_trace_commitment,
        composition_commitment,
    ] = commitments.unbox();

    // Step 4: Compute log_sizes per tree
    let log_sizes_arr = MLClaimLogSizesTrait::log_sizes(activation_claims.span());
    let preprocessed_log_sizes = log_sizes_arr[0].span();
    let trace_log_sizes = log_sizes_arr[1].span();
    let interaction_trace_log_sizes = log_sizes_arr[2].span();

    let log_blowup_factor = pcs_config.fri_config.log_blowup_factor;

    // Step 5: Commit preprocessed trace
    commitment_scheme.commit(
        preprocessed_commitment,
        preprocessed_log_sizes,
        ref channel,
        log_blowup_factor,
    );
    // Mix claim after preprocessed commit (matches verify_cairo pattern)
    claim.mix_into(ref channel);

    // Step 6: Commit trace
    commitment_scheme.commit(
        trace_commitment,
        trace_log_sizes,
        ref channel,
        log_blowup_factor,
    );

    // Step 7: Verify interaction proof-of-work
    assert!(
        channel.verify_pow_nonce(INTERACTION_POW_BITS, interaction_pow),
        "Activation interaction proof-of-work failed",
    );
    channel.mix_u64(interaction_pow);

    // Step 8: Draw interaction lookup elements
    let activation_lookup_elements: ActivationLookupElements = LookupElementsTrait::draw(
        ref channel,
    );

    // Step 9: Verify LogUp sum — all activation claimed sums should total zero
    assert!(
        interaction_claim.activation_claimed_sum.is_zero(),
        "Invalid activation LogUp sum: must be zero",
    );

    // Step 10: Mix interaction claim into channel
    interaction_claim.mix_into(ref channel);

    // Step 11: Commit interaction trace
    commitment_scheme.commit(
        interaction_trace_commitment,
        interaction_trace_log_sizes,
        ref channel,
        log_blowup_factor,
    );

    // Step 12: Construct MLAir with activation components
    let trace_lde_log_size = get_trace_lde_log_size(@commitment_scheme.trees);
    let trace_log_size = trace_lde_log_size - pcs_config.fri_config.log_blowup_factor;
    let composition_log_degree_bound = trace_log_size + 1;

    let ml_air = MLAirNewImpl::new(
        claim,
        activation_claims.span(),
        activation_interaction_claims.span(),
        @activation_lookup_elements,
        composition_log_degree_bound,
    );

    // Step 13: Call generic STARK verify
    verify(
        ml_air,
        ref channel,
        stark_proof,
        commitment_scheme,
        SECURITY_BITS,
        composition_commitment,
    );
}

#[cfg(test)]
mod tests {
    use core::num::traits::{Zero, One};
    use stwo_verifier_core::channel::{Channel, ChannelTrait};
    use stwo_verifier_core::fields::qm31::{QM31, qm31_const};
    use super::claim::{MLProof, MLClaim, MatMulSumcheckProofOnChain, RoundPoly};
    use super::verify_ml;

    #[test]
    fn test_verify_ml_no_activation_proof() {
        // Backward compatibility: verify_ml with activation_stark_proof = None
        // should only verify matmul sumchecks (existing path).

        let c0 = qm31_const::<5, 0, 0, 0>();
        let c1 = qm31_const::<3, 0, 0, 0>();
        let c2: QM31 = Zero::zero();
        let claimed_sum = qm31_const::<13, 0, 0, 0>();

        // Replay the channel to find the challenge
        let mut replay_channel: Channel = Default::default();
        // mix claim: num_layers=1, activation_type=0
        replay_channel.mix_u64(1);
        replay_channel.mix_u64(0);
        // mix dimensions
        replay_channel.mix_u64(1); // m
        replay_channel.mix_u64(2); // k
        replay_channel.mix_u64(1); // n
        replay_channel.mix_felts([claimed_sum].span());
        replay_channel.mix_felts([c0, c1, c2].span());
        let challenge = replay_channel.draw_secure_felt();

        let p_of_r = c0 + c1 * challenge + c2 * challenge * challenge;

        let final_a_eval = p_of_r;
        let final_b_eval: QM31 = One::one();

        let proof = MLProof {
            claim: MLClaim {
                model_id: 0x1234,
                num_layers: 1,
                activation_type: 0,
                io_commitment: 0xabc,
                weight_commitment: 0xdef,
            },
            matmul_proofs: array![
                MatMulSumcheckProofOnChain {
                    m: 1,
                    k: 2,
                    n: 1,
                    num_rounds: 1,
                    claimed_sum,
                    round_polys: array![RoundPoly { c0, c1, c2 }],
                    final_a_eval,
                    final_b_eval,
                    a_commitment: 0,
                    b_commitment: 0,
                },
            ],
            channel_salt: Option::None,
            activation_stark_proof: Option::None,
        };

        let output = verify_ml(proof);
        assert!(output.verified);
        assert!(output.model_id == 0x1234);
        assert!(output.num_matmuls == 1);
        assert!(output.num_layers == 1);
    }

    #[test]
    fn test_verify_ml_empty_proof() {
        // Verify with zero matmuls and no activation STARK — trivially passes.
        let proof = MLProof {
            claim: MLClaim {
                model_id: 0xbeef,
                num_layers: 0,
                activation_type: 0,
                io_commitment: 0,
                weight_commitment: 0,
            },
            matmul_proofs: array![],
            channel_salt: Option::None,
            activation_stark_proof: Option::None,
        };

        let output = verify_ml(proof);
        assert!(output.verified);
        assert!(output.model_id == 0xbeef);
        assert!(output.num_matmuls == 0);
    }

    #[test]
    fn test_verify_ml_with_salt() {
        // Verify with channel salt and no matmuls — should pass.
        let proof = MLProof {
            claim: MLClaim {
                model_id: 0xcafe,
                num_layers: 1,
                activation_type: 1,
                io_commitment: 0x111,
                weight_commitment: 0x222,
            },
            matmul_proofs: array![],
            channel_salt: Option::Some(42),
            activation_stark_proof: Option::None,
        };

        let output = verify_ml(proof);
        assert!(output.verified);
        assert!(output.io_commitment == 0x111);
        assert!(output.weight_commitment == 0x222);
    }
}
