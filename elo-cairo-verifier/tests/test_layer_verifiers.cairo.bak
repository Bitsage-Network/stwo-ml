use elo_cairo_verifier::field::{
    QM31, CM31, qm31_new, qm31_zero, qm31_one, qm31_add, qm31_sub, qm31_mul,
    qm31_eq, poly_eval_degree2, poly_eval_degree3, eq_eval, m31_to_qm31,
};
use elo_cairo_verifier::channel::{
    PoseidonChannel, channel_default, channel_mix_u64, channel_mix_secure_field,
    channel_mix_poly_coeffs, channel_mix_poly_coeffs_deg3, channel_draw_qm31,
    channel_mix_felts,
};
use elo_cairo_verifier::types::{RoundPoly, GkrRoundPoly, GKRClaim};
use elo_cairo_verifier::field::qm31_inverse;
use elo_cairo_verifier::layer_verifiers::{
    verify_add_layer, verify_mul_layer, verify_matmul_layer,
    verify_activation_layer, verify_dequantize_layer,
    verify_layernorm_layer, verify_rmsnorm_layer,
};

// ============================================================================
// channel_mix_poly_coeffs_deg3 Tests
// ============================================================================

#[test]
fn test_mix_poly_coeffs_deg3_matches_mix_felts() {
    // Verify: mix_poly_coeffs_deg3(c0,c1,c2,c3) == mix_felts([c0,c1,c2,c3])
    let c0 = qm31_new(100, 200, 300, 400);
    let c1 = qm31_new(500, 600, 700, 800);
    let c2 = qm31_new(900, 1000, 1100, 1200);
    let c3 = qm31_new(1300, 1400, 1500, 1600);

    let mut ch1 = channel_default();
    channel_mix_poly_coeffs_deg3(ref ch1, c0, c1, c2, c3);

    let mut ch2 = channel_default();
    channel_mix_felts(ref ch2, array![c0, c1, c2, c3].span());

    assert!(ch1.digest == ch2.digest, "deg3 should match mix_felts for 4 values");
}

#[test]
fn test_mix_poly_coeffs_deg3_differs_from_deg2() {
    let c0 = qm31_new(1, 2, 3, 4);
    let c1 = qm31_new(5, 6, 7, 8);
    let c2 = qm31_new(9, 10, 11, 12);
    let c3 = qm31_zero();

    let mut ch1 = channel_default();
    channel_mix_poly_coeffs(ref ch1, c0, c1, c2);

    let mut ch2 = channel_default();
    channel_mix_poly_coeffs_deg3(ref ch2, c0, c1, c2, c3);

    assert!(ch1.digest != ch2.digest, "deg3 with c3=0 must differ from deg2");
}

// ============================================================================
// Add Layer Tests
// ============================================================================

#[test]
fn test_verify_add_layer_basic() {
    // Construct a valid Add layer:
    // lhs_eval = 10, rhs_eval = 20, so output claim = 30
    let lhs = qm31_new(10, 0, 0, 0);
    let rhs = qm31_new(20, 0, 0, 0);
    let output_value = qm31_add(lhs, rhs); // 30

    let claim = GKRClaim {
        point: array![qm31_new(42, 0, 0, 0)],
        value: output_value,
    };

    let mut ch = channel_default();
    channel_mix_u64(ref ch, 999); // pre-seed channel

    let result = verify_add_layer(@claim, lhs, rhs, 0, ref ch);

    // Result should have same point as input claim
    assert!(result.point.len() == 1, "point length preserved");
    assert!(qm31_eq(*result.point.at(0), qm31_new(42, 0, 0, 0)), "point value preserved");

    // Value should be alpha * lhs + (1 - alpha) * rhs (not trivially equal to either)
    // We can't predict alpha, but the function should not panic
}

#[test]
fn test_verify_add_layer_general_qm31() {
    // Use general QM31 values to test full field arithmetic
    let lhs = qm31_new(1000, 2000, 3000, 4000);
    let rhs = qm31_new(500, 600, 700, 800);
    let output_value = qm31_add(lhs, rhs);

    let claim = GKRClaim {
        point: array![qm31_new(7, 11, 13, 17), qm31_new(19, 23, 29, 31)],
        value: output_value,
    };

    let mut ch = channel_default();
    let result = verify_add_layer(@claim, lhs, rhs, 0, ref ch);

    assert!(result.point.len() == 2, "point length preserved for multi-dim");
}

#[test]
#[should_panic(expected: "ADD_SUM_MISMATCH")]
fn test_verify_add_layer_mismatch_panics() {
    let lhs = qm31_new(10, 0, 0, 0);
    let rhs = qm31_new(20, 0, 0, 0);
    // Wrong output value (should be 30, not 31)
    let wrong_value = qm31_new(31, 0, 0, 0);

    let claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0)],
        value: wrong_value,
    };

    let mut ch = channel_default();
    verify_add_layer(@claim, lhs, rhs, 0, ref ch);
}

#[test]
fn test_verify_add_layer_deterministic() {
    // Same inputs should produce same output
    let lhs = qm31_new(5, 10, 15, 20);
    let rhs = qm31_new(25, 30, 35, 40);
    let output_value = qm31_add(lhs, rhs);
    let claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0)],
        value: output_value,
    };

    let mut ch1 = channel_default();
    channel_mix_u64(ref ch1, 42);
    let r1 = verify_add_layer(@claim, lhs, rhs, 0, ref ch1);

    let claim2 = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0)],
        value: output_value,
    };
    let mut ch2 = channel_default();
    channel_mix_u64(ref ch2, 42);
    let r2 = verify_add_layer(@claim2, lhs, rhs, 0, ref ch2);

    assert!(qm31_eq(r1.value, r2.value), "deterministic value");
    assert!(ch1.digest == ch2.digest, "deterministic channel state");
}

// ============================================================================
// Mul Layer Tests
// ============================================================================

/// Build a synthetic valid Mul layer proof for testing.
///
/// For a single variable (num_vars=1):
///   claim value V = sum_{x in {0,1}} eq(r,x) * a(x) * b(x)
///                 = eq(r,0) * a(0) * b(0) + eq(r,1) * a(1) * b(1)
///
/// We pick specific values and derive the round polynomial from the sumcheck.
#[test]
fn test_verify_mul_layer_single_var() {
    // 1 variable: r = [r0], sumcheck has 1 round
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();

    // Pick a(0)=2, a(1)=5, b(0)=3, b(1)=7
    let a0 = qm31_new(2, 0, 0, 0);
    let a1 = qm31_new(5, 0, 0, 0);
    let b0 = qm31_new(3, 0, 0, 0);
    let b1 = qm31_new(7, 0, 0, 0);

    // eq(r, 0) = (1 - r0), eq(r, 1) = r0
    let eq_at_0 = qm31_sub(one, r0);
    let eq_at_1 = r0;

    // V = eq(r,0)*a(0)*b(0) + eq(r,1)*a(1)*b(1)
    let v = qm31_add(
        qm31_mul(eq_at_0, qm31_mul(a0, b0)),
        qm31_mul(eq_at_1, qm31_mul(a1, b1)),
    );

    // The sumcheck polynomial p(X) = eq(r,X) * a(X) * b(X) evaluated at X
    // For single var: p(X) = [(1-r0)*(1-X) + r0*X] * [a0*(1-X) + a1*X] * [b0*(1-X) + b1*X]
    // p(0) = (1-r0) * a0 * b0
    // p(1) = r0 * a1 * b1
    // p(0) + p(1) = V (which is the claimed sum)
    let p_at_0 = qm31_mul(eq_at_0, qm31_mul(a0, b0));
    let p_at_1 = qm31_mul(eq_at_1, qm31_mul(a1, b1));

    // For degree-3 polynomial p(X) = c0 + c1*X + c2*X^2 + c3*X^3:
    //   p(0) = c0
    //   p(1) = c0 + c1 + c2 + c3
    //   => c1 + c2 + c3 = p(1) - p(0)
    // We need a valid polynomial. Let's compute p at a few points.
    // p(X) = eq(r,X) * a(X) * b(X)
    //   eq(r,X) = (1-r0)(1-X) + r0*X = (1-r0) + (2r0-1)*X
    //   a(X) = a0 + (a1-a0)*X
    //   b(X) = b0 + (b1-b0)*X
    // So p is degree 3 (product of 3 linear polynomials).

    // To find c0,c1,c2,c3, we can evaluate at X=0,1,2,3 and solve.
    // But a simpler approach: just set c0 = p(0), and compute c1,c2,c3
    // from the polynomial identity.
    //
    // eq(r,X) = (1-r0) + (2*r0-1)*X    [linear in X]
    // a(X) = a0 + (a1-a0)*X            [linear in X]
    // b(X) = b0 + (b1-b0)*X            [linear in X]
    // p(X) = eq * a * b                [degree 3 in X]

    // Let eq_0 = 1-r0, eq_1 = 2*r0-1, da = a1-a0, db = b1-b0
    let eq_coeff_0 = qm31_sub(one, r0); // 1 - r0
    let two_r0 = qm31_add(r0, r0);
    let eq_coeff_1 = qm31_sub(two_r0, one); // 2*r0 - 1
    let da = qm31_sub(a1, a0); // a1 - a0
    let db = qm31_sub(b1, b0); // b1 - b0

    // p(X) = (eq_0 + eq_1*X) * (a0 + da*X) * (b0 + db*X)
    // Expand (eq_0 + eq_1*X)*(a0 + da*X) first:
    //   = eq_0*a0 + (eq_0*da + eq_1*a0)*X + eq_1*da*X^2
    let ea_0 = qm31_mul(eq_coeff_0, a0);
    let ea_1 = qm31_add(qm31_mul(eq_coeff_0, da), qm31_mul(eq_coeff_1, a0));
    let ea_2 = qm31_mul(eq_coeff_1, da);

    // Now multiply by (b0 + db*X):
    //   c0 = ea_0 * b0
    //   c1 = ea_0 * db + ea_1 * b0
    //   c2 = ea_1 * db + ea_2 * b0
    //   c3 = ea_2 * db
    let c0 = qm31_mul(ea_0, b0);
    let c1 = qm31_add(qm31_mul(ea_0, db), qm31_mul(ea_1, b0));
    let c2 = qm31_add(qm31_mul(ea_1, db), qm31_mul(ea_2, b0));
    let c3 = qm31_mul(ea_2, db);

    // Sanity: c0 should equal p(0)
    assert!(qm31_eq(c0, p_at_0), "c0 == p(0)");

    // Sanity: c0+c1+c2+c3 should equal p(1)
    let sum_coeffs = qm31_add(qm31_add(qm31_add(c0, c1), c2), c3);
    assert!(qm31_eq(sum_coeffs, p_at_1), "c0+c1+c2+c3 == p(1)");

    // Now we need to replay the Fiat-Shamir to get the challenge point,
    // and compute lhs_eval = a(challenge), rhs_eval = b(challenge).

    // Set up the verifier channel identically
    let mut prover_ch = channel_default();

    // Mix "MUL" tag + claimed sum (matching what verify_mul_layer expects)
    channel_mix_u64(ref prover_ch, 0x4D554C);
    channel_mix_secure_field(ref prover_ch, v);

    // Mix round poly
    channel_mix_poly_coeffs_deg3(ref prover_ch, c0, c1, c2, c3);

    // Draw challenge
    let s0 = channel_draw_qm31(ref prover_ch);

    // Compute lhs_eval = a(s0) = a0 + da * s0
    let lhs_eval = qm31_add(a0, qm31_mul(da, s0));
    // Compute rhs_eval = b(s0) = b0 + db * s0
    let rhs_eval = qm31_add(b0, qm31_mul(db, s0));

    // Build the round poly struct
    let round_poly = GkrRoundPoly {
        c0: c0, c1: c1, c2: c2, c3: c3, num_coeffs: 4,
    };

    // Build the output claim
    let claim = GKRClaim {
        point: array![r0],
        value: v,
    };

    // Now verify — use a fresh channel
    let mut verify_ch = channel_default();
    let result = verify_mul_layer(
        @claim,
        array![round_poly].span(),
        lhs_eval,
        rhs_eval,
        ref verify_ch,
    );

    // Should not panic — verification passed
    // Check result has same point
    assert!(result.point.len() == 1, "mul result point length");
}

#[test]
#[should_panic(expected: "MUL_ROUND_SUM_MISMATCH")]
fn test_verify_mul_layer_bad_round_poly() {
    let v = qm31_new(100, 0, 0, 0);
    let claim = GKRClaim {
        point: array![qm31_new(3, 0, 0, 0)],
        value: v,
    };

    // Bad round poly: c0 + c1 + c2 + c3 + c0 != v
    let bad_poly = GkrRoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c1: qm31_new(20, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
        c3: qm31_new(40, 0, 0, 0), // sum = 10+20+30+40 = 100, p(0)+p(1) = 10+100 = 110 != 100
        num_coeffs: 4,
    };

    let mut ch = channel_default();
    verify_mul_layer(
        @claim,
        array![bad_poly].span(),
        qm31_zero(),
        qm31_zero(),
        ref ch,
    );
}

// ============================================================================
// MatMul Layer Tests
// ============================================================================

#[test]
fn test_verify_matmul_layer_1x1() {
    // Simplest case: m=1, k=1, n=1 (scalars)
    // C = A * B where all are 1x1 matrices
    // log_k = log2(1) = 0 rounds, so the sumcheck is trivial
    // But log2_ceil(1) = 0 which means 0 rounds.
    // Actually with k=1, padded k = 1, log_k = 0. So no sumcheck rounds.
    // The claim value should directly equal final_a * final_b.

    let a_val = qm31_new(7, 0, 0, 0);
    let b_val = qm31_new(11, 0, 0, 0);
    let c_val = qm31_mul(a_val, b_val); // 77

    let claim = GKRClaim {
        point: array![], // log_m + log_n = 0 + 0 = 0 variables
        value: c_val,
    };

    let mut ch = channel_default();
    let result = verify_matmul_layer(
        @claim,
        array![].span(), // 0 rounds
        a_val,
        b_val,
        1, 1, 1,
        ref ch,
    );

    // Result: point = [] (log_m=0, no k_challenges), value = a_val
    assert!(result.point.len() == 0, "1x1 point empty");
    assert!(qm31_eq(result.value, a_val), "1x1 value = a_val");
}

#[test]
fn test_verify_matmul_layer_2x2() {
    // m=2, k=2, n=2: log_k=1 (one sumcheck round)
    // A = [[a00, a01], [a10, a11]]
    // B = [[b00, b01], [b10, b11]]
    // C = A * B
    //
    // MLE_C(r_i, r_j) = sum_{k in {0,1}} MLE_A(r_i, k) * MLE_B(k, r_j)
    //
    // For the sumcheck over k (1 round):
    //   p(X) = MLE_A(r_i, X) * MLE_B(X, r_j)
    //   p(0) = MLE_A(r_i, 0) * MLE_B(0, r_j)
    //   p(1) = MLE_A(r_i, 1) * MLE_B(1, r_j)
    //   claimed_sum = p(0) + p(1) = MLE_C(r_i, r_j) [by definition]

    // Pick concrete matrix values
    let a00 = qm31_new(1, 0, 0, 0);
    let a01 = qm31_new(2, 0, 0, 0);
    let a10 = qm31_new(3, 0, 0, 0);
    let a11 = qm31_new(4, 0, 0, 0);
    let b00 = qm31_new(5, 0, 0, 0);
    let b01 = qm31_new(6, 0, 0, 0);
    let b10 = qm31_new(7, 0, 0, 0);
    let b11 = qm31_new(8, 0, 0, 0);

    // Choose evaluation point r_i (1 var), r_j (1 var)
    let r_i = qm31_new(3, 0, 0, 0);
    let r_j = qm31_new(5, 0, 0, 0);
    let one = qm31_one();

    // MLE_A(r_i, k):
    //   MLE_A = a00*(1-ri)*(1-k) + a01*(1-ri)*k + a10*ri*(1-k) + a11*ri*k
    //   MLE_A(r_i, 0) = a00*(1-ri) + a10*ri
    //   MLE_A(r_i, 1) = a01*(1-ri) + a11*ri
    let one_minus_ri = qm31_sub(one, r_i);
    let mle_a_ri_0 = qm31_add(qm31_mul(a00, one_minus_ri), qm31_mul(a10, r_i));
    let mle_a_ri_1 = qm31_add(qm31_mul(a01, one_minus_ri), qm31_mul(a11, r_i));

    // MLE_B(k, r_j):
    //   MLE_B(0, r_j) = b00*(1-rj) + b01*rj
    //   MLE_B(1, r_j) = b10*(1-rj) + b11*rj
    let one_minus_rj = qm31_sub(one, r_j);
    let mle_b_0_rj = qm31_add(qm31_mul(b00, one_minus_rj), qm31_mul(b01, r_j));
    let mle_b_1_rj = qm31_add(qm31_mul(b10, one_minus_rj), qm31_mul(b11, r_j));

    // claimed_sum = MLE_A(r_i, 0)*MLE_B(0, r_j) + MLE_A(r_i, 1)*MLE_B(1, r_j)
    let p0 = qm31_mul(mle_a_ri_0, mle_b_0_rj);
    let p1 = qm31_mul(mle_a_ri_1, mle_b_1_rj);
    let claimed_sum = qm31_add(p0, p1);

    // Degree-2 sumcheck polynomial p(X) = MLE_A(r_i, X) * MLE_B(X, r_j)
    //   MLE_A(r_i, X) = mle_a_ri_0 + (mle_a_ri_1 - mle_a_ri_0) * X
    //   MLE_B(X, r_j) = mle_b_0_rj + (mle_b_1_rj - mle_b_0_rj) * X
    //   p(X) = (alpha + beta*X) * (gamma + delta*X) = alpha*gamma + (alpha*delta + beta*gamma)*X + beta*delta*X^2
    let alpha_v = mle_a_ri_0;
    let beta_v = qm31_sub(mle_a_ri_1, mle_a_ri_0);
    let gamma_v = mle_b_0_rj;
    let delta_v = qm31_sub(mle_b_1_rj, mle_b_0_rj);

    let c0 = qm31_mul(alpha_v, gamma_v); // p(0)
    let c1 = qm31_add(qm31_mul(alpha_v, delta_v), qm31_mul(beta_v, gamma_v));
    let c2 = qm31_mul(beta_v, delta_v);

    // Sanity: c0 + (c0+c1+c2) = p(0) + p(1) = claimed_sum
    let check_p1 = qm31_add(qm31_add(c0, c1), c2);
    assert!(qm31_eq(qm31_add(c0, check_p1), claimed_sum), "sanity: p(0)+p(1)=claimed");

    // Replay Fiat-Shamir to find the challenge and final evals
    let mut prover_ch = channel_default();

    // Step 1: Mix dimensions
    channel_mix_u64(ref prover_ch, 2); // m
    channel_mix_u64(ref prover_ch, 2); // k
    channel_mix_u64(ref prover_ch, 2); // n

    // Step 2: Mix claimed sum
    channel_mix_secure_field(ref prover_ch, claimed_sum);

    // Step 3: Mix round poly, draw challenge
    channel_mix_poly_coeffs(ref prover_ch, c0, c1, c2);
    let challenge = channel_draw_qm31(ref prover_ch);

    // final_a = MLE_A(r_i, challenge) = alpha + beta * challenge
    let final_a = qm31_add(alpha_v, qm31_mul(beta_v, challenge));
    // final_b = MLE_B(challenge, r_j) = gamma + delta * challenge
    let final_b = qm31_add(gamma_v, qm31_mul(delta_v, challenge));

    // Sanity: p(challenge) = final_a * final_b
    let p_challenge = poly_eval_degree2(c0, c1, c2, challenge);
    assert!(qm31_eq(p_challenge, qm31_mul(final_a, final_b)), "sanity: p(ch)=a*b");

    // Build proof
    let round_poly = RoundPoly { c0: c0, c1: c1, c2: c2 };
    let claim = GKRClaim {
        point: array![r_i, r_j], // log_m=1, log_n=1
        value: claimed_sum,
    };

    // Verify with a fresh channel
    let mut verify_ch = channel_default();
    let result = verify_matmul_layer(
        @claim,
        array![round_poly].span(),
        final_a,
        final_b,
        2, 2, 2,
        ref verify_ch,
    );

    // Result: point = [r_i, challenge] (log_m=1 from claim + 1 k_challenge)
    assert!(result.point.len() == 2, "2x2 result point has 2 vars");
    assert!(qm31_eq(*result.point.at(0), r_i), "first element is r_i");
    // Second element should be the challenge (derived from Fiat-Shamir)
    assert!(qm31_eq(result.value, final_a), "value = final_a_eval");
}

#[test]
#[should_panic(expected: "MATMUL_ROUND_SUM_MISMATCH")]
fn test_verify_matmul_layer_bad_round_poly() {
    let claimed_sum = qm31_new(100, 0, 0, 0);
    let claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0), qm31_new(2, 0, 0, 0)],
        value: claimed_sum,
    };

    // Bad round poly: c0 + (c0+c1+c2) != claimed_sum
    let bad_poly = RoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c1: qm31_new(20, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
        // p(0)+p(1) = 10 + (10+20+30) = 10+60 = 70 != 100
    };

    let mut ch = channel_default();
    verify_matmul_layer(
        @claim,
        array![bad_poly].span(),
        qm31_zero(),
        qm31_zero(),
        2, 2, 2,
        ref ch,
    );
}

#[test]
#[should_panic(expected: "MATMUL_FINAL_MISMATCH")]
fn test_verify_matmul_layer_bad_final_eval() {
    // Valid round poly that passes the round check, but wrong final evals
    let claimed_sum = qm31_new(100, 0, 0, 0);
    let claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0), qm31_new(2, 0, 0, 0)],
        value: claimed_sum,
    };

    // p(0) = c0 = 30, p(1) = c0+c1+c2 = 30+20+20 = 70
    // p(0) + p(1) = 30 + 70 = 100 = claimed_sum -- passes round check
    let poly = RoundPoly {
        c0: qm31_new(30, 0, 0, 0),
        c1: qm31_new(20, 0, 0, 0),
        c2: qm31_new(20, 0, 0, 0),
    };

    // final_a * final_b won't match p(challenge) for arbitrary wrong values
    let wrong_a = qm31_new(999, 0, 0, 0);
    let wrong_b = qm31_new(1, 0, 0, 0);

    let mut ch = channel_default();
    verify_matmul_layer(
        @claim,
        array![poly].span(),
        wrong_a,
        wrong_b,
        2, 2, 2,
        ref ch,
    );
}

// ============================================================================
// Channel State Consistency Tests
// ============================================================================

#[test]
fn test_add_layer_channel_state_changes() {
    // Verify that the channel state is updated after verification
    let lhs = qm31_new(10, 0, 0, 0);
    let rhs = qm31_new(20, 0, 0, 0);
    let claim = GKRClaim {
        point: array![qm31_one()],
        value: qm31_add(lhs, rhs),
    };

    let mut ch = channel_default();
    let d0 = ch.digest;
    verify_add_layer(@claim, lhs, rhs, 0, ref ch);
    assert!(ch.digest != d0, "channel state should change after add verification");
}

#[test]
fn test_matmul_layer_channel_state_changes() {
    // 1x1x1 case: no rounds, just dimension mixing + final eval mixing
    let a = qm31_new(7, 0, 0, 0);
    let b = qm31_new(11, 0, 0, 0);
    let claim = GKRClaim {
        point: array![],
        value: qm31_mul(a, b),
    };

    let mut ch = channel_default();
    let d0 = ch.digest;
    verify_matmul_layer(@claim, array![].span(), a, b, 1, 1, 1, ref ch);
    assert!(ch.digest != d0, "channel state should change after matmul verification");
}

// ============================================================================
// Activation Layer Tests
// ============================================================================

/// Build a valid 1-variable LogUp eq-sumcheck proof by simulating the prover.
///
/// Strategy: pick r = [r0], use p(X) = eq(r,X) for the degree-3 polynomial
/// (which is valid when w*d = 1 at all boolean points). Then compute
/// final_w_eval at the Fiat-Shamir challenge s such that eq*w*d = p(s) holds.
#[test]
fn test_verify_activation_layer_single_var() {
    let r0 = qm31_new(5, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![r0],
        value: zero, // not used by activation verifier
    };

    // === Simulate prover's Fiat-Shamir to get gamma, beta ===
    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4C4F47); // "LOG"
    let act_tag: u64 = 1; // type tag (e.g. ReLU)
    channel_mix_u64(ref sim_ch, act_tag);
    let gamma = channel_draw_qm31(ref sim_ch);
    let beta = channel_draw_qm31(ref sim_ch);

    // === Build the round polynomial ===
    // For 1 variable, eq(r0, X) = (1-r0)(1-X) + r0*X = (1-r0) + (2r0-1)*X
    // Since w(x)*d(x) = 1 at boolean points, the sumcheck polynomial
    // g(X) = eq(r0, X) * wMLE(X) * dMLE(X) is degree 3.
    //
    // But we need actual coefficients. Use the approach:
    //   p(0) = eq(r0,0) * w(0)*d(0) = (1-r0) * 1 = 1-r0
    //   p(1) = eq(r0,1) * w(1)*d(1) = r0 * 1 = r0
    //   p(0) + p(1) = 1 (initial sum)
    //
    // For simplicity, use p(X) = (1-r0) + (2r0-1)*X as a degree-1 polynomial
    // (c2=c3=0). This is exact when w*d = 1 everywhere (e.g. constant d,w).
    let eq_at_0 = qm31_sub(one, r0); // 1 - r0
    let two_r0 = qm31_add(r0, r0);
    let slope = qm31_sub(two_r0, one); // 2r0 - 1
    let c0 = eq_at_0;
    let c1 = slope;
    let c2 = zero;
    let c3 = zero;

    // claimed_sum = arbitrary (only used for table balance + Fiat-Shamir)
    let claimed_sum = qm31_new(42, 0, 0, 0);

    // Continue simulating prover channel
    channel_mix_secure_field(ref sim_ch, claimed_sum);

    // Mix round poly and draw challenge
    channel_mix_poly_coeffs_deg3(ref sim_ch, c0, c1, c2, c3);
    let s = channel_draw_qm31(ref sim_ch);

    // p(s) = c0 + c1*s = (1-r0) + (2r0-1)*s = eq(r0, s)
    let p_s = poly_eval_degree3(c0, c1, c2, c3, s);

    // To pass the final check: p(s) == eq(r0, s) * w(s) * d(s)
    // Pick final_in = 0, final_out = 0 => d(s) = gamma - 0 - beta*0 = gamma
    // eq(r0, s) = p_s (since our polynomial IS eq)
    // So: p_s == p_s * w(s) * gamma => w(s) = 1/gamma
    let final_in_eval = zero;
    let final_out_eval = zero;
    let final_w_eval = qm31_inverse(gamma);

    // Input/output evals (mixed at the end, arbitrary)
    let input_eval = qm31_new(42, 0, 0, 0);
    let output_eval = qm31_new(99, 0, 0, 0);

    let round_poly = GkrRoundPoly { c0, c1, c2, c3, num_coeffs: 4 };

    // === Verify ===
    let mut ch = channel_default();
    let result = verify_activation_layer(
        @output_claim,
        act_tag,
        array![round_poly].span(),
        final_w_eval,
        final_in_eval,
        final_out_eval,
        claimed_sum,
        input_eval,
        output_eval,
        ref ch,
    );

    // Check result
    assert!(qm31_eq(result.value, input_eval), "activation returns input_eval");
    assert!(result.point.len() == 1, "activation point passthrough");
    assert!(qm31_eq(*result.point.at(0), r0), "activation point value");
}

#[test]
#[should_panic(expected: "LOGUP_ROUND_SUM_MISMATCH")]
fn test_verify_activation_layer_bad_round_poly() {
    let output_claim = GKRClaim {
        point: array![qm31_new(5, 0, 0, 0)],
        value: qm31_zero(),
    };

    // Bad poly: p(0)+p(1) = 10 + (10+20+30+40) = 10+100 = 110 != 1 (initial sum)
    let bad_poly = GkrRoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c1: qm31_new(20, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
        c3: qm31_new(40, 0, 0, 0),
        num_coeffs: 4,
    };

    let mut ch = channel_default();
    verify_activation_layer(
        @output_claim,
        1, // type_tag
        array![bad_poly].span(),
        qm31_zero(),
        qm31_zero(),
        qm31_zero(),
        qm31_one(),
        qm31_zero(),
        qm31_zero(),
        ref ch,
    );
}

#[test]
fn test_activation_channel_state_changes() {
    // Verify channel state is modified during activation verification
    // Use a dummy proof that passes the round check (p(0)+p(1) = 1)
    let one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![zero], // r=0
        value: zero,
    };

    // With r=0: eq(0,0)=1, eq(0,1)=0, so p(0)=1, p(1)=0
    // c0=1, c1+c2+c3 = -1
    let neg_one = qm31_sub(zero, one);
    let round_poly = GkrRoundPoly { c0: one, c1: neg_one, c2: zero, c3: zero, num_coeffs: 4 };

    // Simulate to compute valid w_eval
    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4C4F47);
    channel_mix_u64(ref sim_ch, 1);
    let gamma = channel_draw_qm31(ref sim_ch);
    let _beta = channel_draw_qm31(ref sim_ch);
    let claimed_sum = one;
    channel_mix_secure_field(ref sim_ch, claimed_sum);
    channel_mix_poly_coeffs_deg3(ref sim_ch, one, neg_one, zero, zero);
    let s = channel_draw_qm31(ref sim_ch);

    // p(s) = 1 - s, eq(0,s) = 1-s, d(s) = gamma, so w(s) = 1/gamma
    let final_w_eval = qm31_inverse(gamma);

    let mut ch = channel_default();
    let d0 = ch.digest;
    let _result = verify_activation_layer(
        @output_claim, 1,
        array![round_poly].span(),
        final_w_eval, zero, zero, claimed_sum,
        qm31_new(1, 0, 0, 0), qm31_new(2, 0, 0, 0),
        ref ch,
    );
    assert!(ch.digest != d0, "activation changes channel state");
}

// ============================================================================
// Dequantize Layer Tests
// ============================================================================

#[test]
fn test_verify_dequantize_layer_single_var() {
    // Same proof construction as activation, different tags
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![r0],
        value: zero,
    };

    // Simulate prover channel with DEQLOG tag
    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4445514C4F47); // "DEQLOG"
    let bits: u64 = 8;
    channel_mix_u64(ref sim_ch, bits);
    let gamma = channel_draw_qm31(ref sim_ch);
    let _beta = channel_draw_qm31(ref sim_ch);

    // Round poly: p(X) = eq(r0, X) = (1-r0) + (2r0-1)*X
    let eq_at_0 = qm31_sub(one, r0);
    let slope = qm31_sub(qm31_add(r0, r0), one);
    let c0 = eq_at_0;
    let c1 = slope;

    let claimed_sum = qm31_new(7, 0, 0, 0);
    channel_mix_secure_field(ref sim_ch, claimed_sum);
    channel_mix_poly_coeffs_deg3(ref sim_ch, c0, c1, zero, zero);
    let _s = channel_draw_qm31(ref sim_ch);

    // final_in=0, final_out=0 => d=gamma, w=1/gamma
    let final_w_eval = qm31_inverse(gamma);

    let round_poly = GkrRoundPoly { c0, c1, c2: zero, c3: zero, num_coeffs: 4 };

    let mut ch = channel_default();
    let result = verify_dequantize_layer(
        @output_claim, bits,
        array![round_poly].span(),
        final_w_eval, zero, zero, claimed_sum,
        qm31_new(10, 0, 0, 0), qm31_new(20, 0, 0, 0),
        ref ch,
    );

    assert!(qm31_eq(result.value, qm31_new(10, 0, 0, 0)), "dequantize returns input_eval");
    assert!(result.point.len() == 1, "dequantize point passthrough");
}

#[test]
fn test_dequantize_differs_from_activation_transcript() {
    // Same proof data, but different tags should produce different channel states
    let zero = qm31_zero();
    let one = qm31_one();
    let neg_one = qm31_sub(zero, one);
    let output_claim = GKRClaim { point: array![zero], value: zero };
    let round_poly = GkrRoundPoly { c0: one, c1: neg_one, c2: zero, c3: zero, num_coeffs: 4 };

    // Compute valid w_eval for activation
    let mut sim1 = channel_default();
    channel_mix_u64(ref sim1, 0x4C4F47);
    channel_mix_u64(ref sim1, 1);
    let gamma1 = channel_draw_qm31(ref sim1);
    let _beta1 = channel_draw_qm31(ref sim1);
    channel_mix_secure_field(ref sim1, one);
    channel_mix_poly_coeffs_deg3(ref sim1, one, neg_one, zero, zero);
    let _s1 = channel_draw_qm31(ref sim1);
    let w1 = qm31_inverse(gamma1);

    let mut ch1 = channel_default();
    let _r1 = verify_activation_layer(
        @output_claim, 1,
        array![round_poly].span(),
        w1, zero, zero, one, zero, zero,
        ref ch1,
    );

    // Compute valid w_eval for dequantize
    let output_claim2 = GKRClaim { point: array![zero], value: zero };
    let round_poly2 = GkrRoundPoly { c0: one, c1: neg_one, c2: zero, c3: zero, num_coeffs: 4 };
    let mut sim2 = channel_default();
    channel_mix_u64(ref sim2, 0x4445514C4F47);
    channel_mix_u64(ref sim2, 8);
    let gamma2 = channel_draw_qm31(ref sim2);
    let _beta2 = channel_draw_qm31(ref sim2);
    channel_mix_secure_field(ref sim2, one);
    channel_mix_poly_coeffs_deg3(ref sim2, one, neg_one, zero, zero);
    let _s2 = channel_draw_qm31(ref sim2);
    let w2 = qm31_inverse(gamma2);

    let mut ch2 = channel_default();
    let _r2 = verify_dequantize_layer(
        @output_claim2, 8,
        array![round_poly2].span(),
        w2, zero, zero, one, zero, zero,
        ref ch2,
    );

    // Different tags => different transcripts
    assert!(ch1.digest != ch2.digest, "activation and dequantize must differ");
}

// ============================================================================
// LayerNorm Layer Tests
// ============================================================================

#[test]
fn test_verify_layernorm_layer_part1_only() {
    // Test LayerNorm with has_logup=false (SIMD combined mode).
    // Only Part 1 (linear transform) is verified.
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

    // For the linear eq-sumcheck: output = Σ eq(r,x) * centered(x) * rsqrt(x)
    // With 1 variable: eq(r0, X) = (1-r0) + (2r0-1)*X
    // Pick centered(0)=2, centered(1)=3, rsqrt(0)=5, rsqrt(1)=7
    let c_0 = qm31_new(2, 0, 0, 0);
    let c_1 = qm31_new(3, 0, 0, 0);
    let rs_0 = qm31_new(5, 0, 0, 0);
    let rs_1 = qm31_new(7, 0, 0, 0);

    let eq_at_0 = qm31_sub(one, r0); // 1-r0
    let eq_at_1 = r0;

    // output_claim.value = eq(r0,0)*c(0)*rs(0) + eq(r0,1)*c(1)*rs(1)
    let term0 = qm31_mul(eq_at_0, qm31_mul(c_0, rs_0));
    let term1 = qm31_mul(eq_at_1, qm31_mul(c_1, rs_1));
    let output_value = qm31_add(term0, term1);

    let output_claim = GKRClaim { point: array![r0], value: output_value };

    // Build degree-3 polynomial:
    // g(X) = eq(r0,X) * cMLE(X) * rsMLE(X)
    // eq = A + B*X, cMLE = c_0 + dc*X, rsMLE = rs_0 + drs*X
    let eq_a = eq_at_0;
    let eq_b = qm31_sub(qm31_add(r0, r0), one); // 2r0-1
    let dc = qm31_sub(c_1, c_0);
    let drs = qm31_sub(rs_1, rs_0);

    // (A+BX)*(c_0+dc*X) = A*c_0 + (A*dc + B*c_0)*X + B*dc*X²
    let ea_0 = qm31_mul(eq_a, c_0);
    let ea_1 = qm31_add(qm31_mul(eq_a, dc), qm31_mul(eq_b, c_0));
    let ea_2 = qm31_mul(eq_b, dc);

    // * (rs_0 + drs*X):
    let pc0 = qm31_mul(ea_0, rs_0);
    let pc1 = qm31_add(qm31_mul(ea_0, drs), qm31_mul(ea_1, rs_0));
    let pc2 = qm31_add(qm31_mul(ea_1, drs), qm31_mul(ea_2, rs_0));
    let pc3 = qm31_mul(ea_2, drs);

    // Simulate prover channel to get challenge
    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4C4E); // "LN"
    let mean_eval = qm31_new(10, 0, 0, 0);
    let rsqrt_eval = qm31_new(20, 0, 0, 0);
    channel_mix_secure_field(ref sim_ch, mean_eval);
    channel_mix_secure_field(ref sim_ch, rsqrt_eval);
    channel_mix_secure_field(ref sim_ch, output_value);

    channel_mix_poly_coeffs_deg3(ref sim_ch, pc0, pc1, pc2, pc3);
    let s = channel_draw_qm31(ref sim_ch);

    // Compute final evals at challenge point s
    let centered_final = qm31_add(c_0, qm31_mul(dc, s));
    let rsqrt_final = qm31_add(rs_0, qm31_mul(drs, s));

    let round_poly = GkrRoundPoly { c0: pc0, c1: pc1, c2: pc2, c3: pc3, num_coeffs: 4 };

    let input_eval = qm31_new(100, 0, 0, 0);
    let output_eval = qm31_new(200, 0, 0, 0);

    // Empty logup data (not used when has_logup=false)
    let empty_logup_polys: Array<GkrRoundPoly> = array![];

    let mut ch = channel_default();
    let result = verify_layernorm_layer(
        @output_claim,
        array![round_poly].span(),
        centered_final, rsqrt_final,
        mean_eval, rsqrt_eval,
        false, // has_logup = false (SIMD combined)
        empty_logup_polys.span(),
        zero, zero, zero, zero,
        input_eval, output_eval,
        ref ch,
    );

    assert!(qm31_eq(result.value, input_eval), "layernorm returns input_eval");
    assert!(result.point.len() == 1, "layernorm point passthrough");
    assert!(qm31_eq(*result.point.at(0), r0), "layernorm point value");
}

#[test]
#[should_panic(expected: "LINEAR_ROUND_SUM_MISMATCH")]
fn test_verify_layernorm_layer_bad_linear_poly() {
    let one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0)],
        value: qm31_new(100, 0, 0, 0),
    };

    // Bad poly: p(0)+p(1) = 10 + (10+20+30+40) = 110 != 100
    let bad_poly = GkrRoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c1: qm31_new(20, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
        c3: qm31_new(40, 0, 0, 0),
        num_coeffs: 4,
    };

    let empty: Array<GkrRoundPoly> = array![];

    let mut ch = channel_default();
    verify_layernorm_layer(
        @output_claim,
        array![bad_poly].span(),
        zero, zero, zero, zero,
        false, empty.span(), zero, zero, zero, zero,
        zero, zero,
        ref ch,
    );
}

// ============================================================================
// RMSNorm Layer Tests
// ============================================================================

#[test]
fn test_verify_rmsnorm_layer_part1_only() {
    // Same approach as LayerNorm test but with "RN" tag and rms_sq instead of mean.
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

    // For RMSNorm: output = Σ eq(r,x) * input(x) * rsqrt(x)
    let in_0 = qm31_new(2, 0, 0, 0);
    let in_1 = qm31_new(3, 0, 0, 0);
    let rs_0 = qm31_new(5, 0, 0, 0);
    let rs_1 = qm31_new(7, 0, 0, 0);

    let eq_at_0 = qm31_sub(one, r0);
    let eq_at_1 = r0;

    let output_value = qm31_add(
        qm31_mul(eq_at_0, qm31_mul(in_0, rs_0)),
        qm31_mul(eq_at_1, qm31_mul(in_1, rs_1)),
    );

    let output_claim = GKRClaim { point: array![r0], value: output_value };

    // Build degree-3 polynomial (same structure as LayerNorm)
    let eq_a = eq_at_0;
    let eq_b = qm31_sub(qm31_add(r0, r0), one);
    let din = qm31_sub(in_1, in_0);
    let drs = qm31_sub(rs_1, rs_0);

    let ea_0 = qm31_mul(eq_a, in_0);
    let ea_1 = qm31_add(qm31_mul(eq_a, din), qm31_mul(eq_b, in_0));
    let ea_2 = qm31_mul(eq_b, din);

    let pc0 = qm31_mul(ea_0, rs_0);
    let pc1 = qm31_add(qm31_mul(ea_0, drs), qm31_mul(ea_1, rs_0));
    let pc2 = qm31_add(qm31_mul(ea_1, drs), qm31_mul(ea_2, rs_0));
    let pc3 = qm31_mul(ea_2, drs);

    // Simulate prover channel
    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x524E); // "RN"
    let rms_sq_eval = qm31_new(10, 0, 0, 0);
    let rsqrt_eval = qm31_new(20, 0, 0, 0);
    channel_mix_secure_field(ref sim_ch, rms_sq_eval);
    channel_mix_secure_field(ref sim_ch, rsqrt_eval);
    channel_mix_secure_field(ref sim_ch, output_value);

    channel_mix_poly_coeffs_deg3(ref sim_ch, pc0, pc1, pc2, pc3);
    let s = channel_draw_qm31(ref sim_ch);

    // Final evals at challenge
    let input_final = qm31_add(in_0, qm31_mul(din, s));
    let rsqrt_final = qm31_add(rs_0, qm31_mul(drs, s));

    let round_poly = GkrRoundPoly { c0: pc0, c1: pc1, c2: pc2, c3: pc3, num_coeffs: 4 };
    let empty: Array<GkrRoundPoly> = array![];

    let mut ch = channel_default();
    let result = verify_rmsnorm_layer(
        @output_claim,
        array![round_poly].span(),
        input_final, rsqrt_final,
        rms_sq_eval, rsqrt_eval,
        false, // has_logup = false
        empty.span(), zero, zero, zero, zero,
        qm31_new(100, 0, 0, 0), qm31_new(200, 0, 0, 0),
        ref ch,
    );

    assert!(qm31_eq(result.value, qm31_new(100, 0, 0, 0)), "rmsnorm returns input_eval");
    assert!(result.point.len() == 1, "rmsnorm point passthrough");
}

#[test]
fn test_layernorm_rmsnorm_different_tags() {
    // Same proof data, different tags should produce different channel states
    let zero = qm31_zero();
    let one = qm31_one();
    let r0 = zero;

    // Trivial polynomial: p(X) = 1 (constant, c0=1, c1=c2=c3=0)
    // But p(0)+p(1) = 1+1 = 2, which needs output_value = 2
    // Actually for the linear sumcheck, initial sum = output_claim.value
    // Let's use p(0)=1, p(1)=0, so output_value=1
    let neg_one = qm31_sub(zero, one);
    let pc0 = one;
    let pc1 = neg_one;

    let output_value = one;

    // LayerNorm Part 1: mix "LN" tag first
    let claim1 = GKRClaim { point: array![r0], value: output_value };
    let poly1 = GkrRoundPoly { c0: pc0, c1: pc1, c2: zero, c3: zero, num_coeffs: 4 };

    // Simulate LayerNorm to get valid final evals
    let mut sim1 = channel_default();
    channel_mix_u64(ref sim1, 0x4C4E); // "LN"
    channel_mix_secure_field(ref sim1, zero); // mean
    channel_mix_secure_field(ref sim1, zero); // rsqrt
    channel_mix_secure_field(ref sim1, output_value);
    channel_mix_poly_coeffs_deg3(ref sim1, pc0, pc1, zero, zero);
    let s1 = channel_draw_qm31(ref sim1);

    // With r0=0: eq(0,s1) = 1-s1, p(s1) = 1-s1
    // Need (1-s1) = eq(0,s1) * centered * rsqrt
    // centered=1, rsqrt = (1-s1)/eq(0,s1) = 1 (since eq(0,s1)=1-s1)
    let centered_final1 = one;
    let rsqrt_final1 = one;

    let empty1: Array<GkrRoundPoly> = array![];
    let mut ch1 = channel_default();
    let _r1 = verify_layernorm_layer(
        @claim1,
        array![poly1].span(), centered_final1, rsqrt_final1,
        zero, zero,
        false, empty1.span(), zero, zero, zero, zero,
        zero, zero,
        ref ch1,
    );

    // RMSNorm Part 1: mix "RN" tag first
    let claim2 = GKRClaim { point: array![r0], value: output_value };
    let poly2 = GkrRoundPoly { c0: pc0, c1: pc1, c2: zero, c3: zero, num_coeffs: 4 };
    let empty2: Array<GkrRoundPoly> = array![];
    let mut ch2 = channel_default();
    let _r2 = verify_rmsnorm_layer(
        @claim2,
        array![poly2].span(), one, one,
        zero, zero,
        false, empty2.span(), zero, zero, zero, zero,
        zero, zero,
        ref ch2,
    );

    assert!(ch1.digest != ch2.digest, "layernorm and rmsnorm must have different transcripts");
}

#[test]
#[should_panic(expected: "LINEAR_ROUND_SUM_MISMATCH")]
fn test_verify_rmsnorm_layer_bad_linear_poly() {
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0)],
        value: qm31_new(50, 0, 0, 0),
    };

    // Bad poly: p(0)+p(1) = 5+(5+10+15+20) = 5+50 = 55 != 50
    let bad_poly = GkrRoundPoly {
        c0: qm31_new(5, 0, 0, 0),
        c1: qm31_new(10, 0, 0, 0),
        c2: qm31_new(15, 0, 0, 0),
        c3: qm31_new(20, 0, 0, 0),
        num_coeffs: 4,
    };

    let empty: Array<GkrRoundPoly> = array![];
    let mut ch = channel_default();
    verify_rmsnorm_layer(
        @output_claim,
        array![bad_poly].span(),
        zero, zero, zero, zero,
        false, empty.span(), zero, zero, zero, zero,
        zero, zero,
        ref ch,
    );
}

// ============================================================================
// MUL_FINAL_MISMATCH Test
// ============================================================================

#[test]
#[should_panic(expected: "MUL_FINAL_MISMATCH")]
fn test_verify_mul_layer_bad_final_eval() {
    // Build a claim and round poly where the round check passes
    // but the final eval check fails
    let v = qm31_new(20, 0, 0, 0);
    let r0 = qm31_new(1, 0, 0, 0);
    let claim = GKRClaim {
        point: array![r0],
        value: v,
    };

    // p(0) = 5, p(1) = c0+c1+c2+c3 = 5+5+5+0 = 15
    // p(0)+p(1) = 5 + 15 = 20 = v  -- round check passes
    let poly = GkrRoundPoly {
        c0: qm31_new(5, 0, 0, 0),
        c1: qm31_new(5, 0, 0, 0),
        c2: qm31_new(5, 0, 0, 0),
        c3: qm31_zero(),
        num_coeffs: 4,
    };

    // Wrong final evals: the product eq(r,s)*lhs*rhs won't match p(s)
    let wrong_lhs = qm31_new(999, 0, 0, 0);
    let wrong_rhs = qm31_new(1, 0, 0, 0);

    let mut ch = channel_default();
    verify_mul_layer(
        @claim,
        array![poly].span(),
        wrong_lhs,
        wrong_rhs,
        ref ch,
    );
}
