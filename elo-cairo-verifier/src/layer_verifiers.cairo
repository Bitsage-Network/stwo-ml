// ML GKR Per-Layer Verifiers
//
// Ports of stwo-ml/src/gkr/verifier.rs layer reductions to Cairo.
// Each function replays the EXACT Fiat-Shamir transcript from the Rust prover.
//
// Layer tags (matching Rust gkr/types.rs):
//   0=MatMul, 1=Add, 2=Mul, 3=Activation, 4=LayerNorm,
//   5=Attention, 6=Dequantize, 7=MatMulDualSimd, 8=RMSNorm

use crate::field::{
    QM31, qm31_one, qm31_add, qm31_sub, qm31_mul, qm31_eq,
    poly_eval_degree2, poly_eval_degree3, eq_eval, log2_ceil,
};
use crate::channel::{
    PoseidonChannel, channel_mix_u64, channel_mix_secure_field,
    channel_mix_poly_coeffs, channel_mix_poly_coeffs_deg3, channel_draw_qm31,
};
use crate::types::{GKRClaim, CompressedRoundPoly, CompressedGkrRoundPoly};

// ============================================================================
// Helpers
// ============================================================================

/// Clone a GKRClaim's point array (Cairo arrays are move-only).
pub fn clone_point(point: @Array<QM31>) -> Array<QM31> {
    let mut result: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= point.len() {
            break;
        }
        result.append(*point.at(i));
        i += 1;
    };
    result
}

// ============================================================================
// Add Layer Verifier (Tag 1)
// ============================================================================

/// Verify an Add layer reduction in the GKR walk.
///
/// Add layer: output = lhs + rhs (pointwise).
/// No sumcheck needed — linearity gives direct verification.
///
/// In DAG circuits (residual connections), the trunk is the input with
/// the higher layer index. trunk_idx: 0 = lhs is trunk, 1 = rhs is trunk.
/// The claim follows the trunk; the skip branch gets a deferred proof.
///
/// Transcript (matches verifier.rs:908-944):
///   1. Check: lhs_eval + rhs_eval == output_claim.value
///   2. Mix: mix_secure_field(lhs_eval)
///   3. Mix: mix_secure_field(rhs_eval)
///   4. Draw: alpha (for transcript binding, not used)
///   5. Return: claim with value = trunk_eval
pub fn verify_add_layer(
    output_claim: @GKRClaim,
    lhs_eval: QM31,
    rhs_eval: QM31,
    trunk_idx: u32,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    // Step 1: Verify additive decomposition
    let sum = qm31_add(lhs_eval, rhs_eval);
    assert!(qm31_eq(sum, *output_claim.value), "ADD_SUM_MISMATCH");

    // Step 2-3: Mix evaluations into transcript
    channel_mix_secure_field(ref ch, lhs_eval);
    channel_mix_secure_field(ref ch, rhs_eval);

    // Step 4: Draw alpha (for transcript binding, not used for claim)
    let _alpha = channel_draw_qm31(ref ch);

    // Step 5: Return claim with trunk value
    let trunk_eval = if trunk_idx == 1 { rhs_eval } else { lhs_eval };

    GKRClaim {
        point: clone_point(output_claim.point),
        value: trunk_eval,
    }
}

// ============================================================================
// Mul Layer Verifier (Tag 2)
// ============================================================================

/// Verify a Mul layer reduction in the GKR walk.
///
/// Mul layer: output = lhs * rhs (pointwise).
/// Uses degree-3 eq-sumcheck: sum_x eq(r,x) * lhs(x) * rhs(x) = claimed.
///
/// Transcript (matches verifier.rs:817-895):
///   1. Mix: 0x4D554C ("MUL")
///   2. Mix: mix_secure_field(output_claim.value)
///   3. Per round: check p(0)+p(1)==sum, mix poly (deg-3), draw challenge
///   4. Check: sum == eq(r, challenges) * lhs_eval * rhs_eval
///   5. Mix: lhs_eval, rhs_eval
///   6. Draw: alpha
///   7. Return: claim with value = alpha * lhs + (1 - alpha) * rhs
pub fn verify_mul_layer(
    output_claim: @GKRClaim,
    eq_round_polys: Span<CompressedGkrRoundPoly>,
    lhs_eval: QM31,
    rhs_eval: QM31,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let num_vars = eq_round_polys.len();

    // Step 1: Mix "MUL" tag
    channel_mix_u64(ref ch, 0x4D554C);

    // Step 2: Mix output claim value
    channel_mix_secure_field(ref ch, *output_claim.value);

    // Step 3: Degree-3 eq-sumcheck rounds (compressed: reconstruct c1 from current_sum)
    let mut current_sum = *output_claim.value;
    let mut challenges: Array<QM31> = array![];
    let mut round: u32 = 0;
    loop {
        if round >= num_vars {
            break;
        }
        let cpoly = *eq_round_polys.at(round);

        // Reconstruct c1 = current_sum - 2*c0 - c2 - c3
        let c1 = qm31_sub(
            qm31_sub(qm31_sub(current_sum, qm31_add(cpoly.c0, cpoly.c0)), cpoly.c2),
            cpoly.c3,
        );

        // Mix round polynomial (always all 4 coefficients)
        channel_mix_poly_coeffs_deg3(ref ch, cpoly.c0, c1, cpoly.c2, cpoly.c3);

        // Draw challenge
        let challenge = channel_draw_qm31(ref ch);
        challenges.append(challenge);

        // Update: current_sum = p(challenge)
        current_sum = poly_eval_degree3(cpoly.c0, c1, cpoly.c2, cpoly.c3, challenge);

        round += 1;
    };

    // Step 4: Verify final evaluation
    // eq(r[..num_vars], challenges) * lhs * rhs == current_sum
    //
    // Extract the first num_vars elements of claim.point for eq computation
    let mut r_slice: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= num_vars {
            break;
        }
        r_slice.append(*output_claim.point.at(i));
        i += 1;
    };

    let eq_val = eq_eval(r_slice.span(), challenges.span());
    let expected = qm31_mul(eq_val, qm31_mul(lhs_eval, rhs_eval));
    assert!(qm31_eq(current_sum, expected), "MUL_FINAL_MISMATCH");

    // Step 5: Mix final evaluations
    channel_mix_secure_field(ref ch, lhs_eval);
    channel_mix_secure_field(ref ch, rhs_eval);

    // Step 6: Draw combiner
    let alpha = channel_draw_qm31(ref ch);

    // Step 7: Combine claims
    let one = qm31_one();
    let combined = qm31_add(
        qm31_mul(alpha, lhs_eval),
        qm31_mul(qm31_sub(one, alpha), rhs_eval),
    );

    GKRClaim {
        point: clone_point(output_claim.point),
        value: combined,
    }
}

// ============================================================================
// MatMul Layer Verifier (Tag 0)
// ============================================================================

/// Verify a MatMul layer reduction in the GKR walk.
///
/// MatMul layer: C = A x B where A is m x k, B is k x n.
/// Uses degree-2 sumcheck over k: sum_k MLE_A(r_i,k) * MLE_B(k,r_j) = claimed.
///
/// Transcript (matches verifier.rs:546-646):
///   1. Mix: m, k, n (as u64)
///   2. Mix: mix_secure_field(output_claim.value)
///   3. Per round: check p(0)+p(1)==sum, mix poly (degree-2), draw challenge
///   4. Check: sum == final_a * final_b
///   5. Mix: final_a, final_b
///   6. Return: claim with point = [r_i || k_challenges], value = final_a
pub fn verify_matmul_layer(
    output_claim: @GKRClaim,
    round_polys: Span<CompressedRoundPoly>,
    final_a_eval: QM31,
    final_b_eval: QM31,
    m: u32,
    k: u32,
    n: u32,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let log_k = round_polys.len();
    let log_m = log2_ceil(m);

    // Step 1: Mix dimensions
    channel_mix_u64(ref ch, m.into());
    channel_mix_u64(ref ch, k.into());
    channel_mix_u64(ref ch, n.into());

    // Step 2: Mix output claim value
    channel_mix_secure_field(ref ch, *output_claim.value);

    // Step 3: Degree-2 sumcheck rounds (compressed: reconstruct c1 from current_sum)
    let mut current_sum = *output_claim.value;
    let mut k_challenges: Array<QM31> = array![];
    let mut round: u32 = 0;
    loop {
        if round >= log_k {
            break;
        }
        let cpoly = *round_polys.at(round);

        // Reconstruct c1 = current_sum - 2*c0 - c2
        // From: p(0) + p(1) = current_sum, where p(0)=c0, p(1)=c0+c1+c2
        let c1 = qm31_sub(qm31_sub(current_sum, qm31_add(cpoly.c0, cpoly.c0)), cpoly.c2);

        // Mix round polynomial (degree-2) — all 3 coefficients for Fiat-Shamir
        channel_mix_poly_coeffs(ref ch, cpoly.c0, c1, cpoly.c2);

        // Draw challenge
        let challenge = channel_draw_qm31(ref ch);
        k_challenges.append(challenge);

        // Update: current_sum = p(challenge) = c0 + c1*t + c2*t^2
        current_sum = poly_eval_degree2(cpoly.c0, c1, cpoly.c2, challenge);

        round += 1;
    };

    // Step 4: Verify final evaluation: sum == final_a * final_b
    let expected = qm31_mul(final_a_eval, final_b_eval);
    assert!(qm31_eq(current_sum, expected), "MATMUL_FINAL_MISMATCH");

    // Step 5: Mix final evaluations
    channel_mix_secure_field(ref ch, final_a_eval);
    channel_mix_secure_field(ref ch, final_b_eval);

    // Step 6: Build return claim
    // point = [r_i (first log_m coords) || k_challenges]
    let mut new_point: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= log_m {
            break;
        }
        new_point.append(*output_claim.point.at(i));
        i += 1;
    };
    // Append k challenges
    i = 0;
    loop {
        if i >= k_challenges.len() {
            break;
        }
        new_point.append(*k_challenges.at(i));
        i += 1;
    };

    GKRClaim {
        point: new_point,
        value: final_a_eval,
    }
}

// ============================================================================
// LogUp Eq-Sumcheck Helper
// ============================================================================

/// Run a LogUp degree-3 eq-sumcheck verification.
///
/// Common to activation, dequantize, layernorm (Part 2), rmsnorm (Part 2).
/// Proves: sum_{x in {0,1}^n} eq(r,x) * w(x) * d(x) = 1
/// where d(x) = gamma - in(x) - beta*out(x) and w(x) = 1/d(x).
///
/// Transcript steps performed:
///   1. mix_secure_field(claimed_sum)
///   2. Per round: check p(0)+p(1)==sum, mix_poly_coeffs_deg3, draw challenge
///   3. Assert: final_sum == eq(r, challenges) * w(s) * d(s)
fn verify_logup_eq_sumcheck(
    output_claim_point: @Array<QM31>,
    logup_round_polys: Span<CompressedGkrRoundPoly>,
    final_w_eval: QM31,
    final_in_eval: QM31,
    final_out_eval: QM31,
    claimed_sum: QM31,
    gamma: QM31,
    beta: QM31,
    ref ch: PoseidonChannel,
) {
    let num_vars = logup_round_polys.len();

    // Mix claimed sum (same as prover)
    channel_mix_secure_field(ref ch, claimed_sum);

    // Degree-3 eq-sumcheck with initial sum = 1 (compressed: reconstruct c1)
    let mut current_sum = qm31_one();
    let mut challenges: Array<QM31> = array![];
    let mut round: u32 = 0;
    loop {
        if round >= num_vars {
            break;
        }
        let cpoly = *logup_round_polys.at(round);

        // Reconstruct c1 = current_sum - 2*c0 - c2 - c3
        let c1 = qm31_sub(
            qm31_sub(qm31_sub(current_sum, qm31_add(cpoly.c0, cpoly.c0)), cpoly.c2),
            cpoly.c3,
        );

        // Mix round polynomial
        channel_mix_poly_coeffs_deg3(ref ch, cpoly.c0, c1, cpoly.c2, cpoly.c3);

        // Draw challenge
        let challenge = channel_draw_qm31(ref ch);
        challenges.append(challenge);

        // Update: current_sum = p(challenge)
        current_sum = poly_eval_degree3(cpoly.c0, c1, cpoly.c2, cpoly.c3, challenge);

        round += 1;
    };

    // Final check: current_sum == eq(r[..num_vars], challenges) * w(s) * d(s)
    // where d(s) = gamma - in(s) - beta * out(s)
    let d_eval = qm31_sub(qm31_sub(gamma, final_in_eval), qm31_mul(beta, final_out_eval));

    let mut r_slice: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= num_vars {
            break;
        }
        r_slice.append(*output_claim_point.at(i));
        i += 1;
    };

    let eq_val = eq_eval(r_slice.span(), challenges.span());
    let expected = qm31_mul(eq_val, qm31_mul(final_w_eval, d_eval));
    assert!(qm31_eq(current_sum, expected), "LOGUP_FINAL_MISMATCH");
}

// ============================================================================
// Linear Transform Eq-Sumcheck Helper
// ============================================================================

/// Run a linear-transform degree-3 eq-sumcheck verification.
///
/// Common to LayerNorm and RMSNorm Part 1.
/// Proves: claimed = sum_{x} eq(r,x) * lhs(x) * rhs(x)
///
/// Transcript steps performed:
///   Per round: check p(0)+p(1)==sum, mix_poly_coeffs_deg3, draw challenge
///   Assert: final_sum == eq(r, challenges) * lhs_final * rhs_final
fn verify_linear_eq_sumcheck(
    output_claim_point: @Array<QM31>,
    round_polys: Span<CompressedGkrRoundPoly>,
    claimed: QM31,
    lhs_final: QM31,
    rhs_final: QM31,
    ref ch: PoseidonChannel,
) {
    let num_vars = round_polys.len();

    let mut current_sum = claimed;
    let mut challenges: Array<QM31> = array![];
    let mut round: u32 = 0;
    loop {
        if round >= num_vars {
            break;
        }
        let cpoly = *round_polys.at(round);

        // Reconstruct c1 = current_sum - 2*c0 - c2 - c3
        let c1 = qm31_sub(
            qm31_sub(qm31_sub(current_sum, qm31_add(cpoly.c0, cpoly.c0)), cpoly.c2),
            cpoly.c3,
        );

        // Mix round polynomial
        channel_mix_poly_coeffs_deg3(ref ch, cpoly.c0, c1, cpoly.c2, cpoly.c3);

        // Draw challenge
        let challenge = channel_draw_qm31(ref ch);
        challenges.append(challenge);

        // Update: current_sum = p(challenge)
        current_sum = poly_eval_degree3(cpoly.c0, c1, cpoly.c2, cpoly.c3, challenge);

        round += 1;
    };

    // Final check: current_sum == eq(r[..num_vars], challenges) * lhs_final * rhs_final
    let mut r_slice: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= num_vars {
            break;
        }
        r_slice.append(*output_claim_point.at(i));
        i += 1;
    };

    let eq_val = eq_eval(r_slice.span(), challenges.span());
    let expected = qm31_mul(eq_val, qm31_mul(lhs_final, rhs_final));
    assert!(qm31_eq(current_sum, expected), "LINEAR_FINAL_MISMATCH");
}

// ============================================================================
// Activation Layer Verifier (Tag 3)
// ============================================================================

/// Verify an Activation layer reduction in the GKR walk.
///
/// Activation layer uses LogUp argument to prove all (input, output) pairs
/// are in the activation lookup table.
///
/// Transcript (matches verifier.rs:909-1058):
///   1. Mix: 0x4C4F47 ("LOG")
///   2. Mix: activation_type_tag
///   3. Draw: gamma, beta
///   4. [Table-side verification done externally via verify_logup_table_sum]
///   5. Mix: claimed_sum
///   6. LogUp eq-sumcheck (initial sum = 1, degree-3 rounds)
///   7. Assert: final == eq(r, challenges) * w(s) * d(s)
///   8. Mix: input_eval, output_eval
///   9. Return: claim with same point, value = input_eval
pub fn verify_activation_layer(
    output_claim: @GKRClaim,
    activation_type_tag: u64,
    logup_round_polys: Span<CompressedGkrRoundPoly>,
    final_w_eval: QM31,
    final_in_eval: QM31,
    final_out_eval: QM31,
    claimed_sum: QM31,
    input_eval: QM31,
    output_eval: QM31,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    // Step 1-2: Mix tags
    channel_mix_u64(ref ch, 0x4C4F47); // "LOG"
    channel_mix_u64(ref ch, activation_type_tag);

    // Step 3: Draw LogUp encoding challenges
    let gamma = channel_draw_qm31(ref ch);
    let beta = channel_draw_qm31(ref ch);

    // Step 5-7: LogUp eq-sumcheck (table-side check done externally)
    verify_logup_eq_sumcheck(
        output_claim.point,
        logup_round_polys,
        final_w_eval,
        final_in_eval,
        final_out_eval,
        claimed_sum,
        gamma,
        beta,
        ref ch,
    );

    // Step 8: Mix final evals
    channel_mix_secure_field(ref ch, input_eval);
    channel_mix_secure_field(ref ch, output_eval);

    GKRClaim {
        point: clone_point(output_claim.point),
        value: input_eval,
    }
}

// ============================================================================
// Dequantize Layer Verifier (Tag 6)
// ============================================================================

/// Verify a Dequantize layer reduction in the GKR walk.
///
/// Same pattern as Activation but with "DEQLOG" tag + bits parameter.
///
/// Transcript (matches verifier.rs:1066-1207):
///   1. Mix: 0x4445514C4F47 ("DEQLOG")
///   2. Mix: bits (as u64)
///   3. Draw: gamma, beta
///   4. [Table-side verification done externally]
///   5. Mix: claimed_sum
///   6. LogUp eq-sumcheck (initial sum = 1, degree-3 rounds)
///   7. Assert: final == eq(r, challenges) * w(s) * d(s)
///   8. Mix: input_eval, output_eval
///   9. Return: claim with same point, value = input_eval
pub fn verify_dequantize_layer(
    output_claim: @GKRClaim,
    bits: u64,
    logup_round_polys: Span<CompressedGkrRoundPoly>,
    final_w_eval: QM31,
    final_in_eval: QM31,
    final_out_eval: QM31,
    claimed_sum: QM31,
    input_eval: QM31,
    output_eval: QM31,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    // Step 1-2: Mix tags
    channel_mix_u64(ref ch, 0x4445514C4F47); // "DEQLOG"
    channel_mix_u64(ref ch, bits);

    // Step 3: Draw LogUp encoding challenges
    let gamma = channel_draw_qm31(ref ch);
    let beta = channel_draw_qm31(ref ch);

    // Step 5-7: LogUp eq-sumcheck
    verify_logup_eq_sumcheck(
        output_claim.point,
        logup_round_polys,
        final_w_eval,
        final_in_eval,
        final_out_eval,
        claimed_sum,
        gamma,
        beta,
        ref ch,
    );

    // Step 8: Mix final evals
    channel_mix_secure_field(ref ch, input_eval);
    channel_mix_secure_field(ref ch, output_eval);

    GKRClaim {
        point: clone_point(output_claim.point),
        value: input_eval,
    }
}

// ============================================================================
// LayerNorm Layer Verifier (Tag 4)
// ============================================================================

/// Verify a LayerNorm layer reduction in the GKR walk.
///
/// Two-phase verification:
///   Part 1: Linear transform eq-sumcheck — proves output = (input - mean) * rsqrt
///   Part 2: Optional rsqrt LogUp eq-sumcheck — proves rsqrt values are in table
///           (skipped when simd_combined = true)
///
/// Transcript (matches verifier.rs:1216-1441):
///   Part 1:
///     1. Mix: 0x4C4E ("LN")
///     2. Mix: mean_eval, rsqrt_eval, output_claim.value
///     3. Degree-3 eq-sumcheck (initial sum = output_claim.value)
///     4. Assert: final == eq(r, challenges) * centered_final * rsqrt_final
///     5. Mix: centered_final, rsqrt_final
///   Part 2 (if has_logup):
///     6. Mix: 0x4C4F47 ("LOG"), 0x5253 ("RS")
///     7. Draw: gamma, beta
///     8. Mix: logup_claimed_sum
///     9. LogUp eq-sumcheck (initial sum = 1)
///     10. Assert: final == eq(r, challenges) * w(s) * d(s)
///   Final:
///     11. Mix: input_eval, output_eval
///     12. Return: claim with same point, value = input_eval
pub fn verify_layernorm_layer(
    output_claim: @GKRClaim,
    linear_round_polys: Span<CompressedGkrRoundPoly>,
    centered_final: QM31,
    rsqrt_final: QM31,
    mean_eval: QM31,
    rsqrt_eval: QM31,
    has_logup: bool,
    logup_round_polys: Span<CompressedGkrRoundPoly>,
    logup_w_eval: QM31,
    logup_in_eval: QM31,
    logup_out_eval: QM31,
    logup_claimed_sum: QM31,
    input_eval: QM31,
    output_eval: QM31,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    // ===== Part 1: Linear transform eq-sumcheck =====
    channel_mix_u64(ref ch, 0x4C4E); // "LN"
    channel_mix_secure_field(ref ch, mean_eval);
    channel_mix_secure_field(ref ch, rsqrt_eval);
    channel_mix_secure_field(ref ch, *output_claim.value);

    verify_linear_eq_sumcheck(
        output_claim.point,
        linear_round_polys,
        *output_claim.value,
        centered_final,
        rsqrt_final,
        ref ch,
    );

    // Mix final linear evals
    channel_mix_secure_field(ref ch, centered_final);
    channel_mix_secure_field(ref ch, rsqrt_final);

    // ===== Part 2: rsqrt LogUp eq-sumcheck (optional) =====
    if has_logup {
        channel_mix_u64(ref ch, 0x4C4F47); // "LOG"
        channel_mix_u64(ref ch, 0x5253); // "RS"
        let gamma = channel_draw_qm31(ref ch);
        let beta = channel_draw_qm31(ref ch);

        verify_logup_eq_sumcheck(
            output_claim.point,
            logup_round_polys,
            logup_w_eval,
            logup_in_eval,
            logup_out_eval,
            logup_claimed_sum,
            gamma,
            beta,
            ref ch,
        );
    }

    // Mix final evals
    channel_mix_secure_field(ref ch, input_eval);
    channel_mix_secure_field(ref ch, output_eval);

    GKRClaim {
        point: clone_point(output_claim.point),
        value: input_eval,
    }
}

// ============================================================================
// RMSNorm Layer Verifier (Tag 8)
// ============================================================================

/// Verify an RMSNorm layer reduction in the GKR walk.
///
/// Same two-phase structure as LayerNorm but:
///   - Uses "RN" tag (not "LN")
///   - Mixes rms_sq_eval instead of mean_eval
///   - No mean subtraction (output = input * rsqrt directly)
///   - LogUp Part 2 uses "RN" type tag (not "RS")
///
/// Transcript (matches verifier.rs:1449-1670):
///   Part 1:
///     1. Mix: 0x524E ("RN")
///     2. Mix: rms_sq_eval, rsqrt_eval, output_claim.value
///     3. Degree-3 eq-sumcheck (initial sum = output_claim.value)
///     4. Assert: final == eq(r, challenges) * input_final * rsqrt_final
///     5. Mix: input_final, rsqrt_final
///   Part 2 (if has_logup):
///     6. Mix: 0x4C4F47 ("LOG"), 0x524E ("RN")
///     7. Draw: gamma, beta
///     8. Mix: logup_claimed_sum
///     9. LogUp eq-sumcheck (initial sum = 1)
///     10. Assert: final == eq(r, challenges) * w(s) * d(s)
///   Final:
///     11. Mix: input_eval, output_eval
///     12. Return: claim with same point, value = input_eval
pub fn verify_rmsnorm_layer(
    output_claim: @GKRClaim,
    linear_round_polys: Span<CompressedGkrRoundPoly>,
    input_final: QM31,
    rsqrt_final: QM31,
    rms_sq_eval: QM31,
    rsqrt_eval: QM31,
    has_logup: bool,
    logup_round_polys: Span<CompressedGkrRoundPoly>,
    logup_w_eval: QM31,
    logup_in_eval: QM31,
    logup_out_eval: QM31,
    logup_claimed_sum: QM31,
    input_eval: QM31,
    output_eval: QM31,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    // ===== Part 1: Linear transform eq-sumcheck =====
    channel_mix_u64(ref ch, 0x524E); // "RN"
    channel_mix_secure_field(ref ch, rms_sq_eval);
    channel_mix_secure_field(ref ch, rsqrt_eval);
    channel_mix_secure_field(ref ch, *output_claim.value);

    verify_linear_eq_sumcheck(
        output_claim.point,
        linear_round_polys,
        *output_claim.value,
        input_final,
        rsqrt_final,
        ref ch,
    );

    // Mix final linear evals
    channel_mix_secure_field(ref ch, input_final);
    channel_mix_secure_field(ref ch, rsqrt_final);

    // ===== Part 2: rsqrt LogUp eq-sumcheck (optional) =====
    if has_logup {
        channel_mix_u64(ref ch, 0x4C4F47); // "LOG"
        channel_mix_u64(ref ch, 0x524E); // "RN"
        let gamma = channel_draw_qm31(ref ch);
        let beta = channel_draw_qm31(ref ch);

        verify_logup_eq_sumcheck(
            output_claim.point,
            logup_round_polys,
            logup_w_eval,
            logup_in_eval,
            logup_out_eval,
            logup_claimed_sum,
            gamma,
            beta,
            ref ch,
        );
    }

    // Mix final evals
    channel_mix_secure_field(ref ch, input_eval);
    channel_mix_secure_field(ref ch, output_eval);

    GKRClaim {
        point: clone_point(output_claim.point),
        value: input_eval,
    }
}
