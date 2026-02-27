use elo_cairo_verifier::field::{
    qm31_new, qm31_zero, qm31_one, qm31_add, qm31_sub, qm31_mul,
    qm31_eq, poly_eval_degree2, poly_eval_degree3,
};
use elo_cairo_verifier::channel::{
    channel_default, channel_mix_u64, channel_mix_secure_field,
    channel_mix_poly_coeffs, channel_mix_poly_coeffs_deg3, channel_draw_qm31,
    channel_mix_felts,
};
use elo_cairo_verifier::types::{CompressedRoundPoly, CompressedGkrRoundPoly, GKRClaim};
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

    assert!(result.point.len() == 1, "point length preserved");
    assert!(qm31_eq(*result.point.at(0), qm31_new(42, 0, 0, 0)), "point value preserved");
}

#[test]
fn test_verify_add_layer_general_qm31() {
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

#[test]
fn test_verify_mul_layer_single_var() {
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();

    let a0 = qm31_new(2, 0, 0, 0);
    let a1 = qm31_new(5, 0, 0, 0);
    let b0 = qm31_new(3, 0, 0, 0);
    let b1 = qm31_new(7, 0, 0, 0);

    let eq_at_0 = qm31_sub(one, r0);
    let eq_at_1 = r0;

    let v = qm31_add(
        qm31_mul(eq_at_0, qm31_mul(a0, b0)),
        qm31_mul(eq_at_1, qm31_mul(a1, b1)),
    );

    let p_at_0 = qm31_mul(eq_at_0, qm31_mul(a0, b0));
    let p_at_1 = qm31_mul(eq_at_1, qm31_mul(a1, b1));

    let eq_coeff_0 = qm31_sub(one, r0);
    let two_r0 = qm31_add(r0, r0);
    let eq_coeff_1 = qm31_sub(two_r0, one);
    let da = qm31_sub(a1, a0);
    let db = qm31_sub(b1, b0);

    let ea_0 = qm31_mul(eq_coeff_0, a0);
    let ea_1 = qm31_add(qm31_mul(eq_coeff_0, da), qm31_mul(eq_coeff_1, a0));
    let ea_2 = qm31_mul(eq_coeff_1, da);

    let c0 = qm31_mul(ea_0, b0);
    let c1 = qm31_add(qm31_mul(ea_0, db), qm31_mul(ea_1, b0));
    let c2 = qm31_add(qm31_mul(ea_1, db), qm31_mul(ea_2, b0));
    let c3 = qm31_mul(ea_2, db);

    assert!(qm31_eq(c0, p_at_0), "c0 == p(0)");
    let sum_coeffs = qm31_add(qm31_add(qm31_add(c0, c1), c2), c3);
    assert!(qm31_eq(sum_coeffs, p_at_1), "c0+c1+c2+c3 == p(1)");

    // Replay prover Fiat-Shamir to get challenge
    let mut prover_ch = channel_default();
    channel_mix_u64(ref prover_ch, 0x4D554C);
    channel_mix_secure_field(ref prover_ch, v);
    channel_mix_poly_coeffs_deg3(ref prover_ch, c0, c1, c2, c3);
    let s0 = channel_draw_qm31(ref prover_ch);

    let lhs_eval = qm31_add(a0, qm31_mul(da, s0));
    let rhs_eval = qm31_add(b0, qm31_mul(db, s0));

    // Build compressed round poly (c1 omitted — verifier reconstructs it)
    let round_poly = CompressedGkrRoundPoly {
        c0: c0, c2: c2, c3: c3,
    };

    let claim = GKRClaim {
        point: array![r0],
        value: v,
    };

    let mut verify_ch = channel_default();
    let result = verify_mul_layer(
        @claim,
        array![round_poly].span(),
        lhs_eval,
        rhs_eval,
        ref verify_ch,
    );

    assert!(result.point.len() == 1, "mul result point length");
}

#[test]
#[should_panic(expected: "MUL_FINAL_MISMATCH")]
fn test_verify_mul_layer_bad_round_poly() {
    let v = qm31_new(100, 0, 0, 0);
    let claim = GKRClaim {
        point: array![qm31_new(3, 0, 0, 0)],
        value: v,
    };

    // Bad poly: c1 is reconstructed from v, so round sum always passes.
    // But final eval check will fail with wrong lhs/rhs.
    let bad_poly = CompressedGkrRoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
        c3: qm31_new(40, 0, 0, 0),
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
    let a_val = qm31_new(7, 0, 0, 0);
    let b_val = qm31_new(11, 0, 0, 0);
    let c_val = qm31_mul(a_val, b_val);

    let claim = GKRClaim {
        point: array![],
        value: c_val,
    };

    let mut ch = channel_default();
    let result = verify_matmul_layer(
        @claim,
        array![].span(),
        a_val,
        b_val,
        1, 1, 1,
        ref ch,
    );

    assert!(result.point.len() == 0, "1x1 point empty");
    assert!(qm31_eq(result.value, a_val), "1x1 value = a_val");
}

#[test]
fn test_verify_matmul_layer_2x2() {
    let a00 = qm31_new(1, 0, 0, 0);
    let a01 = qm31_new(2, 0, 0, 0);
    let a10 = qm31_new(3, 0, 0, 0);
    let a11 = qm31_new(4, 0, 0, 0);
    let b00 = qm31_new(5, 0, 0, 0);
    let b01 = qm31_new(6, 0, 0, 0);
    let b10 = qm31_new(7, 0, 0, 0);
    let b11 = qm31_new(8, 0, 0, 0);

    let r_i = qm31_new(3, 0, 0, 0);
    let r_j = qm31_new(5, 0, 0, 0);
    let one = qm31_one();

    let one_minus_ri = qm31_sub(one, r_i);
    let mle_a_ri_0 = qm31_add(qm31_mul(a00, one_minus_ri), qm31_mul(a10, r_i));
    let mle_a_ri_1 = qm31_add(qm31_mul(a01, one_minus_ri), qm31_mul(a11, r_i));

    let one_minus_rj = qm31_sub(one, r_j);
    let mle_b_0_rj = qm31_add(qm31_mul(b00, one_minus_rj), qm31_mul(b01, r_j));
    let mle_b_1_rj = qm31_add(qm31_mul(b10, one_minus_rj), qm31_mul(b11, r_j));

    let p0 = qm31_mul(mle_a_ri_0, mle_b_0_rj);
    let p1 = qm31_mul(mle_a_ri_1, mle_b_1_rj);
    let claimed_sum = qm31_add(p0, p1);

    let alpha_v = mle_a_ri_0;
    let beta_v = qm31_sub(mle_a_ri_1, mle_a_ri_0);
    let gamma_v = mle_b_0_rj;
    let delta_v = qm31_sub(mle_b_1_rj, mle_b_0_rj);

    let c0 = qm31_mul(alpha_v, gamma_v);
    let c1 = qm31_add(qm31_mul(alpha_v, delta_v), qm31_mul(beta_v, gamma_v));
    let c2 = qm31_mul(beta_v, delta_v);

    let check_p1 = qm31_add(qm31_add(c0, c1), c2);
    assert!(qm31_eq(qm31_add(c0, check_p1), claimed_sum), "sanity: p(0)+p(1)=claimed");

    // Replay Fiat-Shamir
    let mut prover_ch = channel_default();
    channel_mix_u64(ref prover_ch, 2);
    channel_mix_u64(ref prover_ch, 2);
    channel_mix_u64(ref prover_ch, 2);
    channel_mix_secure_field(ref prover_ch, claimed_sum);
    channel_mix_poly_coeffs(ref prover_ch, c0, c1, c2);
    let challenge = channel_draw_qm31(ref prover_ch);

    let final_a = qm31_add(alpha_v, qm31_mul(beta_v, challenge));
    let final_b = qm31_add(gamma_v, qm31_mul(delta_v, challenge));

    let p_challenge = poly_eval_degree2(c0, c1, c2, challenge);
    assert!(qm31_eq(p_challenge, qm31_mul(final_a, final_b)), "sanity: p(ch)=a*b");

    // Build compressed proof (c1 omitted)
    let round_poly = CompressedRoundPoly { c0: c0, c2: c2 };
    let claim = GKRClaim {
        point: array![r_i, r_j],
        value: claimed_sum,
    };

    let mut verify_ch = channel_default();
    let result = verify_matmul_layer(
        @claim,
        array![round_poly].span(),
        final_a,
        final_b,
        2, 2, 2,
        ref verify_ch,
    );

    assert!(result.point.len() == 2, "2x2 result point has 2 vars");
    assert!(qm31_eq(*result.point.at(0), r_i), "first element is r_i");
    assert!(qm31_eq(result.value, final_a), "value = final_a_eval");
}

#[test]
#[should_panic(expected: "MATMUL_FINAL_MISMATCH")]
fn test_verify_matmul_layer_bad_round_poly() {
    let claimed_sum = qm31_new(100, 0, 0, 0);
    let claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0), qm31_new(2, 0, 0, 0)],
        value: claimed_sum,
    };

    // With compressed polys, c1 is reconstructed so round sum always passes.
    // Bad final evals will trigger MATMUL_FINAL_MISMATCH.
    let bad_poly = CompressedRoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
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
    let claimed_sum = qm31_new(100, 0, 0, 0);
    let claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0), qm31_new(2, 0, 0, 0)],
        value: claimed_sum,
    };

    // p(0) = c0 = 30, c1 reconstructed = 100 - 60 - 20 = 20
    // p(1) = 30+20+20 = 70, p(0)+p(1) = 100 = claimed -- passes by construction
    let poly = CompressedRoundPoly {
        c0: qm31_new(30, 0, 0, 0),
        c2: qm31_new(20, 0, 0, 0),
    };

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

#[test]
fn test_verify_activation_layer_single_var() {
    let r0 = qm31_new(5, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![r0],
        value: zero,
    };

    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4C4F47);
    let act_tag: u64 = 1;
    channel_mix_u64(ref sim_ch, act_tag);
    let gamma = channel_draw_qm31(ref sim_ch);
    let _beta = channel_draw_qm31(ref sim_ch);

    let eq_at_0 = qm31_sub(one, r0);
    let two_r0 = qm31_add(r0, r0);
    let slope = qm31_sub(two_r0, one);
    let c0 = eq_at_0;
    let c1 = slope;
    let c2 = zero;
    let c3 = zero;

    let claimed_sum = qm31_new(42, 0, 0, 0);
    channel_mix_secure_field(ref sim_ch, claimed_sum);
    channel_mix_poly_coeffs_deg3(ref sim_ch, c0, c1, c2, c3);
    let s = channel_draw_qm31(ref sim_ch);

    let _p_s = poly_eval_degree3(c0, c1, c2, c3, s);
    let final_in_eval = zero;
    let final_out_eval = zero;
    let final_w_eval = qm31_inverse(gamma);

    let input_eval = qm31_new(42, 0, 0, 0);
    let output_eval = qm31_new(99, 0, 0, 0);

    // Compressed round poly (c1 omitted)
    let round_poly = CompressedGkrRoundPoly { c0, c2, c3 };

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

    assert!(qm31_eq(result.value, input_eval), "activation returns input_eval");
    assert!(result.point.len() == 1, "activation point passthrough");
    assert!(qm31_eq(*result.point.at(0), r0), "activation point value");
}

#[test]
#[should_panic(expected: "LOGUP_FINAL_MISMATCH")]
fn test_verify_activation_layer_bad_round_poly() {
    let output_claim = GKRClaim {
        point: array![qm31_new(5, 0, 0, 0)],
        value: qm31_zero(),
    };

    // With compressed polys, c1 is reconstructed so round sum always passes.
    // Final eval check will fail.
    let bad_poly = CompressedGkrRoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
        c3: qm31_new(40, 0, 0, 0),
    };

    let mut ch = channel_default();
    verify_activation_layer(
        @output_claim,
        1,
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
    let one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![zero],
        value: zero,
    };

    let neg_one = qm31_sub(zero, one);
    // Compressed round poly (c1 omitted)
    let round_poly = CompressedGkrRoundPoly { c0: one, c2: zero, c3: zero };

    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4C4F47);
    channel_mix_u64(ref sim_ch, 1);
    let gamma = channel_draw_qm31(ref sim_ch);
    let _beta = channel_draw_qm31(ref sim_ch);
    let claimed_sum = one;
    channel_mix_secure_field(ref sim_ch, claimed_sum);
    // Verifier reconstructs c1 = 1 - 2*1 - 0 - 0 = -1
    let c1_reconstructed = neg_one;
    channel_mix_poly_coeffs_deg3(ref sim_ch, one, c1_reconstructed, zero, zero);
    let _s = channel_draw_qm31(ref sim_ch);

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
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![r0],
        value: zero,
    };

    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4445514C4F47);
    let bits: u64 = 8;
    channel_mix_u64(ref sim_ch, bits);
    let gamma = channel_draw_qm31(ref sim_ch);
    let _beta = channel_draw_qm31(ref sim_ch);

    let eq_at_0 = qm31_sub(one, r0);
    let slope = qm31_sub(qm31_add(r0, r0), one);
    let c0 = eq_at_0;
    let c1 = slope;

    let claimed_sum = qm31_new(7, 0, 0, 0);
    channel_mix_secure_field(ref sim_ch, claimed_sum);
    channel_mix_poly_coeffs_deg3(ref sim_ch, c0, c1, zero, zero);
    let _s = channel_draw_qm31(ref sim_ch);

    let final_w_eval = qm31_inverse(gamma);

    // Compressed round poly (c1 omitted)
    let round_poly = CompressedGkrRoundPoly { c0, c2: zero, c3: zero };

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
    let zero = qm31_zero();
    let one = qm31_one();
    let neg_one = qm31_sub(zero, one);
    let output_claim = GKRClaim { point: array![zero], value: zero };
    // Compressed round poly (c1 omitted)
    let round_poly = CompressedGkrRoundPoly { c0: one, c2: zero, c3: zero };

    // Compute valid w_eval for activation
    let mut sim1 = channel_default();
    channel_mix_u64(ref sim1, 0x4C4F47);
    channel_mix_u64(ref sim1, 1);
    let gamma1 = channel_draw_qm31(ref sim1);
    let _beta1 = channel_draw_qm31(ref sim1);
    channel_mix_secure_field(ref sim1, one);
    // c1 reconstructed = 1 - 2*1 - 0 - 0 = -1
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
    let round_poly2 = CompressedGkrRoundPoly { c0: one, c2: zero, c3: zero };
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

    assert!(ch1.digest != ch2.digest, "activation and dequantize must differ");
}

// ============================================================================
// LayerNorm Layer Tests
// ============================================================================

#[test]
fn test_verify_layernorm_layer_part1_only() {
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

    let c_0 = qm31_new(2, 0, 0, 0);
    let c_1 = qm31_new(3, 0, 0, 0);
    let rs_0 = qm31_new(5, 0, 0, 0);
    let rs_1 = qm31_new(7, 0, 0, 0);

    let eq_at_0 = qm31_sub(one, r0);
    let eq_at_1 = r0;

    let term0 = qm31_mul(eq_at_0, qm31_mul(c_0, rs_0));
    let term1 = qm31_mul(eq_at_1, qm31_mul(c_1, rs_1));
    let output_value = qm31_add(term0, term1);

    let output_claim = GKRClaim { point: array![r0], value: output_value };

    let eq_a = eq_at_0;
    let eq_b = qm31_sub(qm31_add(r0, r0), one);
    let dc = qm31_sub(c_1, c_0);
    let drs = qm31_sub(rs_1, rs_0);

    let ea_0 = qm31_mul(eq_a, c_0);
    let ea_1 = qm31_add(qm31_mul(eq_a, dc), qm31_mul(eq_b, c_0));
    let ea_2 = qm31_mul(eq_b, dc);

    let pc0 = qm31_mul(ea_0, rs_0);
    let pc1 = qm31_add(qm31_mul(ea_0, drs), qm31_mul(ea_1, rs_0));
    let pc2 = qm31_add(qm31_mul(ea_1, drs), qm31_mul(ea_2, rs_0));
    let pc3 = qm31_mul(ea_2, drs);

    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x4C4E);
    let mean_eval = qm31_new(10, 0, 0, 0);
    let rsqrt_eval = qm31_new(20, 0, 0, 0);
    channel_mix_secure_field(ref sim_ch, mean_eval);
    channel_mix_secure_field(ref sim_ch, rsqrt_eval);
    channel_mix_secure_field(ref sim_ch, output_value);
    channel_mix_poly_coeffs_deg3(ref sim_ch, pc0, pc1, pc2, pc3);
    let s = channel_draw_qm31(ref sim_ch);

    let centered_final = qm31_add(c_0, qm31_mul(dc, s));
    let rsqrt_final = qm31_add(rs_0, qm31_mul(drs, s));

    // Compressed round poly (c1 omitted)
    let round_poly = CompressedGkrRoundPoly { c0: pc0, c2: pc2, c3: pc3 };

    let input_eval = qm31_new(100, 0, 0, 0);
    let output_eval = qm31_new(200, 0, 0, 0);
    let empty_logup_polys: Array<CompressedGkrRoundPoly> = array![];

    let mut ch = channel_default();
    let result = verify_layernorm_layer(
        @output_claim,
        array![round_poly].span(),
        centered_final, rsqrt_final,
        mean_eval, rsqrt_eval,
        false,
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
#[should_panic(expected: "LINEAR_FINAL_MISMATCH")]
fn test_verify_layernorm_layer_bad_linear_poly() {
    let _one = qm31_one();
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0)],
        value: qm31_new(100, 0, 0, 0),
    };

    // With compressed polys, c1 reconstructed so round sum passes.
    // Final eval check will fail with zero lhs/rhs finals.
    let bad_poly = CompressedGkrRoundPoly {
        c0: qm31_new(10, 0, 0, 0),
        c2: qm31_new(30, 0, 0, 0),
        c3: qm31_new(40, 0, 0, 0),
    };

    let empty: Array<CompressedGkrRoundPoly> = array![];

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
    let r0 = qm31_new(3, 0, 0, 0);
    let one = qm31_one();
    let zero = qm31_zero();

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

    let mut sim_ch = channel_default();
    channel_mix_u64(ref sim_ch, 0x524E);
    let rms_sq_eval = qm31_new(10, 0, 0, 0);
    let rsqrt_eval = qm31_new(20, 0, 0, 0);
    channel_mix_secure_field(ref sim_ch, rms_sq_eval);
    channel_mix_secure_field(ref sim_ch, rsqrt_eval);
    channel_mix_secure_field(ref sim_ch, output_value);
    channel_mix_poly_coeffs_deg3(ref sim_ch, pc0, pc1, pc2, pc3);
    let s = channel_draw_qm31(ref sim_ch);

    let input_final = qm31_add(in_0, qm31_mul(din, s));
    let rsqrt_final = qm31_add(rs_0, qm31_mul(drs, s));

    // Compressed round poly (c1 omitted)
    let round_poly = CompressedGkrRoundPoly { c0: pc0, c2: pc2, c3: pc3 };
    let empty: Array<CompressedGkrRoundPoly> = array![];

    let mut ch = channel_default();
    let result = verify_rmsnorm_layer(
        @output_claim,
        array![round_poly].span(),
        input_final, rsqrt_final,
        rms_sq_eval, rsqrt_eval,
        false,
        empty.span(), zero, zero, zero, zero,
        qm31_new(100, 0, 0, 0), qm31_new(200, 0, 0, 0),
        ref ch,
    );

    assert!(qm31_eq(result.value, qm31_new(100, 0, 0, 0)), "rmsnorm returns input_eval");
    assert!(result.point.len() == 1, "rmsnorm point passthrough");
}

#[test]
fn test_layernorm_rmsnorm_different_tags() {
    let zero = qm31_zero();
    let one = qm31_one();
    let r0 = zero;

    let neg_one = qm31_sub(zero, one);
    let pc0 = one;

    let output_value = one;

    let claim1 = GKRClaim { point: array![r0], value: output_value };
    // Compressed round poly (c1 omitted)
    let poly1 = CompressedGkrRoundPoly { c0: pc0, c2: zero, c3: zero };

    let mut sim1 = channel_default();
    channel_mix_u64(ref sim1, 0x4C4E);
    channel_mix_secure_field(ref sim1, zero);
    channel_mix_secure_field(ref sim1, zero);
    channel_mix_secure_field(ref sim1, output_value);
    // c1 reconstructed = 1 - 2*1 - 0 - 0 = -1
    channel_mix_poly_coeffs_deg3(ref sim1, pc0, neg_one, zero, zero);
    let _s1 = channel_draw_qm31(ref sim1);

    let centered_final1 = one;
    let rsqrt_final1 = one;

    let empty1: Array<CompressedGkrRoundPoly> = array![];
    let mut ch1 = channel_default();
    let _r1 = verify_layernorm_layer(
        @claim1,
        array![poly1].span(), centered_final1, rsqrt_final1,
        zero, zero,
        false, empty1.span(), zero, zero, zero, zero,
        zero, zero,
        ref ch1,
    );

    let claim2 = GKRClaim { point: array![r0], value: output_value };
    let poly2 = CompressedGkrRoundPoly { c0: pc0, c2: zero, c3: zero };
    let empty2: Array<CompressedGkrRoundPoly> = array![];
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
#[should_panic(expected: "LINEAR_FINAL_MISMATCH")]
fn test_verify_rmsnorm_layer_bad_linear_poly() {
    let zero = qm31_zero();

    let output_claim = GKRClaim {
        point: array![qm31_new(1, 0, 0, 0)],
        value: qm31_new(50, 0, 0, 0),
    };

    // With compressed polys, round sum passes by construction.
    // Final eval check will fail.
    let bad_poly = CompressedGkrRoundPoly {
        c0: qm31_new(5, 0, 0, 0),
        c2: qm31_new(15, 0, 0, 0),
        c3: qm31_new(20, 0, 0, 0),
    };

    let empty: Array<CompressedGkrRoundPoly> = array![];
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
    let v = qm31_new(20, 0, 0, 0);
    let r0 = qm31_new(1, 0, 0, 0);
    let claim = GKRClaim {
        point: array![r0],
        value: v,
    };

    // Compressed poly (c1 omitted). c1 reconstructed = 20 - 2*5 - 5 - 0 = 5
    let poly = CompressedGkrRoundPoly {
        c0: qm31_new(5, 0, 0, 0),
        c2: qm31_new(5, 0, 0, 0),
        c3: qm31_zero(),
    };

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
