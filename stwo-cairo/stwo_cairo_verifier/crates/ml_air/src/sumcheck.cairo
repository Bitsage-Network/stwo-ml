/// Sumcheck protocol verifier for matmul proofs.
///
/// Verifies that Σ_{x∈{0,1}^n} g(x) = claimed_sum where g(x) = MLE_A(r_i, x) × MLE_B(x, r_j).
///
/// Protocol:
///   For each round i = 0..num_rounds:
///     1. Prover sends round polynomial p_i(X) = c0 + c1*X + c2*X^2
///     2. Verifier checks: p_i(0) + p_i(1) == current_sum
///     3. Verifier draws challenge r_i from channel
///     4. Update: current_sum = p_i(r_i)
///   Final check: current_sum == final_a_eval × final_b_eval
use core::num::traits::{Zero, One};
use stwo_verifier_core::channel::{Channel, ChannelTrait};
use stwo_verifier_core::fields::qm31::QM31;
use super::claim::{MatMulSumcheckProofOnChain, RoundPoly};

/// Evaluate a round polynomial at a point: c0 + c1*x + c2*x^2.
fn eval_round_poly(poly: @RoundPoly, x: QM31) -> QM31 {
    *poly.c0 + *poly.c1 * x + *poly.c2 * x * x
}

/// Verify a matmul sumcheck proof.
///
/// Returns true if the proof is valid, false otherwise.
/// Mixes the proof data into the channel for Fiat-Shamir binding.
pub fn verify_matmul_sumcheck(
    ref channel: Channel,
    proof: @MatMulSumcheckProofOnChain,
) -> bool {
    let num_rounds = *proof.num_rounds;
    let claimed_sum = *proof.claimed_sum;

    // Mix dimensions into channel for domain separation
    channel.mix_u64((*proof.m).into());
    channel.mix_u64((*proof.k).into());
    channel.mix_u64((*proof.n).into());

    // Mix claimed sum
    channel.mix_felts([claimed_sum].span());

    // Verify each sumcheck round
    let mut current_sum = claimed_sum;
    let mut round_idx: u32 = 0;

    while round_idx < num_rounds {
        let round_poly = proof.round_polys.at(round_idx);

        // Check: p_i(0) + p_i(1) == current_sum
        let eval_at_0 = eval_round_poly(round_poly, Zero::zero());
        let eval_at_1 = eval_round_poly(round_poly, One::one());
        let round_sum = eval_at_0 + eval_at_1;

        if round_sum != current_sum {
            return false;
        }

        // Mix round poly into channel
        channel.mix_felts([*round_poly.c0, *round_poly.c1, *round_poly.c2].span());

        // Draw challenge from channel
        let challenge = channel.draw_secure_felt();

        // Update current_sum = p_i(challenge)
        current_sum = eval_round_poly(round_poly, challenge);

        round_idx += 1;
    };

    // Final check: current_sum == final_a_eval × final_b_eval
    let final_product = *proof.final_a_eval * *proof.final_b_eval;
    if current_sum != final_product {
        return false;
    }

    // Mix final evaluations
    channel.mix_felts([*proof.final_a_eval, *proof.final_b_eval].span());

    true
}

#[cfg(test)]
mod tests {
    use core::num::traits::{Zero, One};
    use stwo_verifier_core::channel::{Channel, ChannelTrait};
    use stwo_verifier_core::fields::qm31::{QM31, QM31Serde, qm31_const};
    use super::{eval_round_poly, verify_matmul_sumcheck};
    use super::super::claim::{MatMulSumcheckProofOnChain, RoundPoly};
    use super::super::components::matmul::num_sumcheck_rounds;

    #[test]
    fn test_eval_round_poly_at_zero() {
        // p(x) = c0 + c1*x + c2*x^2
        // p(0) = c0
        let c0 = qm31_const::<42, 0, 0, 0>();
        let c1 = qm31_const::<7, 0, 0, 0>();
        let c2 = qm31_const::<3, 0, 0, 0>();
        let poly = RoundPoly { c0, c1, c2 };
        let result = eval_round_poly(@poly, Zero::zero());
        assert!(result == c0);
    }

    #[test]
    fn test_eval_round_poly_at_one() {
        // p(1) = c0 + c1 + c2
        let c0 = qm31_const::<10, 0, 0, 0>();
        let c1 = qm31_const::<20, 0, 0, 0>();
        let c2 = qm31_const::<30, 0, 0, 0>();
        let poly = RoundPoly { c0, c1, c2 };
        let result = eval_round_poly(@poly, One::one());
        let expected = qm31_const::<60, 0, 0, 0>();
        assert!(result == expected);
    }

    #[test]
    fn test_verify_valid_single_round_proof() {
        // Construct a valid 1-round sumcheck proof.
        // For k=2 (1 round), we need:
        //   p(0) + p(1) = claimed_sum
        //   After drawing challenge r, p(r) = final_a_eval * final_b_eval
        //
        // Use simple values: c0=5, c1=3, c2=0 (linear poly)
        // p(0) = 5, p(1) = 5+3 = 8
        // claimed_sum = 5 + 8 = 13
        //
        // The challenge is drawn from the channel — we can't predict it,
        // but we can construct the proof to match.
        // For a deterministic test, use the actual channel to pre-compute.

        let mut channel: Channel = Default::default();

        let c0 = qm31_const::<5, 0, 0, 0>();
        let c1 = qm31_const::<3, 0, 0, 0>();
        let c2: QM31 = Zero::zero();
        let claimed_sum = qm31_const::<13, 0, 0, 0>();

        // Replay the channel to find the challenge
        let mut replay_channel: Channel = Default::default();
        replay_channel.mix_u64(1); // m
        replay_channel.mix_u64(2); // k
        replay_channel.mix_u64(1); // n
        replay_channel.mix_felts([claimed_sum].span());
        replay_channel.mix_felts([c0, c1, c2].span());
        let challenge = replay_channel.draw_secure_felt();

        // p(challenge) should equal final_a * final_b
        let poly = RoundPoly { c0, c1, c2 };
        let p_of_r = eval_round_poly(@poly, challenge);

        // Set final_a = p(r), final_b = 1 so product matches
        let final_a_eval = p_of_r;
        let final_b_eval: QM31 = One::one();

        let proof = MatMulSumcheckProofOnChain {
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
        };

        let result = verify_matmul_sumcheck(ref channel, @proof);
        assert!(result, "Valid sumcheck proof should verify");
    }

    #[test]
    fn test_verify_invalid_round_sum() {
        // Construct proof where p(0) + p(1) != claimed_sum
        let mut channel: Channel = Default::default();

        let proof = MatMulSumcheckProofOnChain {
            m: 1,
            k: 2,
            n: 1,
            num_rounds: 1,
            claimed_sum: qm31_const::<100, 0, 0, 0>(), // wrong sum
            round_polys: array![
                RoundPoly {
                    c0: qm31_const::<5, 0, 0, 0>(),
                    c1: qm31_const::<3, 0, 0, 0>(),
                    c2: Zero::zero(),
                },
            ],
            final_a_eval: One::one(),
            final_b_eval: One::one(),
            a_commitment: 0,
            b_commitment: 0,
        };

        let result = verify_matmul_sumcheck(ref channel, @proof);
        assert!(!result, "Invalid round sum should fail");
    }

    #[test]
    fn test_num_sumcheck_rounds() {
        assert!(num_sumcheck_rounds(1) == 0);
        assert!(num_sumcheck_rounds(2) == 1);
        assert!(num_sumcheck_rounds(4) == 2);
        assert!(num_sumcheck_rounds(8) == 3);
        assert!(num_sumcheck_rounds(3) == 2); // ceil(log2(3)) = 2
        assert!(num_sumcheck_rounds(5) == 3); // ceil(log2(5)) = 3
    }
}
