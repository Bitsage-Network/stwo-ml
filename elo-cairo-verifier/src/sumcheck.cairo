// Core Sumcheck Verification
//
// Verifies a sumcheck proof by replaying the Fiat-Shamir transcript.
//
// Matches STWO's partially_verify() + final evaluation check:
//   For each round:
//     1. Check: p_i(0) + p_i(1) = expected_sum
//     2. channel.mix_felts(round_poly coefficients)
//     3. challenge = channel.draw_secure_felt()
//     4. expected_sum ← p_i(challenge)
//   Final: expected_sum = final_a_eval × final_b_eval

use core::poseidon::poseidon_hash_span;
use crate::field::{QM31, qm31_add, qm31_mul, qm31_eq, poly_eval_degree2};
use crate::channel::{PoseidonChannel, channel_mix_poly_coeffs, channel_draw_qm31};
use crate::types::RoundPoly;

/// Verify sumcheck rounds and return (is_valid, proof_hash, assignment).
///
/// The channel state must match the prover's state at sumcheck entry
/// (after mixing dimensions, drawing row/col challenges, mixing claimed_sum
/// and commitments).
///
/// Returns:
/// - `is_valid`: true if all round checks and final check pass
/// - `proof_hash`: Poseidon commitment to the proof transcript
/// - `assignment`: the challenges drawn at each round (needed for MLE opening)
pub fn verify_sumcheck_inner(
    claimed_sum: QM31,
    round_polys: Span<RoundPoly>,
    num_rounds: u32,
    final_a_eval: QM31,
    final_b_eval: QM31,
    ref ch: PoseidonChannel,
) -> (bool, felt252, Array<QM31>) {
    let mut expected_sum = claimed_sum;
    let initial_digest = ch.digest;
    let mut assignment: Array<QM31> = array![];

    let mut round: u32 = 0;
    loop {
        if round >= num_rounds {
            break;
        }

        let poly = *round_polys.at(round);

        // p_i(0) = c0, p_i(1) = c0 + c1 + c2
        let eval_at_0 = poly.c0;
        let eval_at_1 = qm31_add(qm31_add(poly.c0, poly.c1), poly.c2);
        let round_sum = qm31_add(eval_at_0, eval_at_1);

        if !qm31_eq(round_sum, expected_sum) {
            let proof_hash = poseidon_hash_span(
                array![initial_digest, round.into(), 'ROUND_FAIL'].span(),
            );
            return (false, proof_hash, array![]);
        }

        // Mix round polynomial into channel
        channel_mix_poly_coeffs(ref ch, poly.c0, poly.c1, poly.c2);

        // Draw random challenge
        let challenge = channel_draw_qm31(ref ch);
        assignment.append(challenge);

        // Update expected sum: expected_sum ← p_i(challenge)
        expected_sum = poly_eval_degree2(poly.c0, poly.c1, poly.c2, challenge);

        round += 1;
    };

    // Final check: expected_sum = f_A(assignment) × f_B(assignment)
    let product = qm31_mul(final_a_eval, final_b_eval);

    if !qm31_eq(expected_sum, product) {
        let proof_hash = poseidon_hash_span(
            array![initial_digest, num_rounds.into(), 'FINAL_FAIL'].span(),
        );
        return (false, proof_hash, array![]);
    }

    // Compute proof hash for on-chain recording
    let proof_hash = poseidon_hash_span(
        array![
            initial_digest,
            num_rounds.into(),
            claimed_sum.a.a.into(),
            claimed_sum.a.b.into(),
            claimed_sum.b.a.into(),
            claimed_sum.b.b.into(),
            final_a_eval.a.a.into(),
            final_a_eval.a.b.into(),
            final_a_eval.b.a.into(),
            final_a_eval.b.b.into(),
            final_b_eval.a.a.into(),
            final_b_eval.a.b.into(),
            final_b_eval.b.a.into(),
            final_b_eval.b.b.into(),
        ]
            .span(),
    );

    (true, proof_hash, assignment)
}

/// Verify a single sumcheck round check: p(0) + p(1) == expected_sum.
/// Exposed for unit testing.
pub fn check_round_sum(poly: RoundPoly, expected_sum: QM31) -> bool {
    let eval_at_0 = poly.c0;
    let eval_at_1 = qm31_add(qm31_add(poly.c0, poly.c1), poly.c2);
    let round_sum = qm31_add(eval_at_0, eval_at_1);
    qm31_eq(round_sum, expected_sum)
}
