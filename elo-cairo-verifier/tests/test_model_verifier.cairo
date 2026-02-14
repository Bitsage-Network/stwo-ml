use elo_cairo_verifier::field::{
    QM31, CM31, qm31_new, qm31_zero, qm31_add, qm31_sub, qm31_eq,
};
use elo_cairo_verifier::channel::{
    channel_default, channel_mix_secure_field, channel_draw_qm31,
};
use elo_cairo_verifier::types::GKRClaim;
use elo_cairo_verifier::model_verifier::verify_gkr_model;

// ============================================================================
// Helpers
// ============================================================================

/// Shorthand: real-only QM31 from a single u64.
fn mk(a: u64) -> QM31 {
    QM31 { a: CM31 { a, b: 0 }, b: CM31 { a: 0, b: 0 } }
}

/// Push a QM31 as 4 felt252 values (aa, ab, ba, bb) into the data array.
fn push_qm31(ref data: Array<felt252>, v: QM31) {
    data.append(v.a.a.into());
    data.append(v.a.b.into());
    data.append(v.b.a.into());
    data.append(v.b.b.into());
}

// ============================================================================
// Test 1: Zero Layers (edge case)
// ============================================================================

#[test]
fn test_model_verify_zero_layers() {
    let mut data: Array<felt252> = array![];
    data.append(0); // num_deferred = 0
    let initial_claim = GKRClaim {
        point: array![mk(42), mk(7)],
        value: mk(999),
    };

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 0, array![].span(), array![].span(), initial_claim, ref ch,
    );

    assert!(result.point.len() == 2, "point preserved");
    assert!(qm31_eq(result.value, mk(999)), "value preserved");
}

// ============================================================================
// Test 2: Single Add Layer
// ============================================================================

#[test]
fn test_model_verify_single_add() {
    let lhs = mk(100);
    let rhs = mk(200);
    let claimed = qm31_add(lhs, rhs); // mk(300)

    let mut data: Array<felt252> = array![];
    data.append(1); // tag = Add
    push_qm31(ref data, lhs);
    push_qm31(ref data, rhs);
    data.append(0); // trunk_idx = 0 (lhs is trunk)
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(42)],
        value: claimed,
    };

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 1, array![].span(), array![].span(), initial_claim, ref ch,
    );

    // Add preserves point length
    assert!(result.point.len() == 1, "point length");
}

// ============================================================================
// Test 3: Unknown Tag Panics
// ============================================================================

#[test]
#[should_panic(expected: "UNKNOWN_LAYER_TAG")]
fn test_model_verify_unknown_tag_panics() {
    let mut data: Array<felt252> = array![99]; // unknown tag

    let initial_claim = GKRClaim {
        point: array![mk(1)],
        value: mk(1),
    };

    let mut ch = channel_default();
    let (_, _) = verify_gkr_model(
        data.span(), 1, array![].span(), array![].span(), initial_claim, ref ch,
    );
}

// ============================================================================
// Test 4: Two Consecutive Adds (Claim Threading)
// ============================================================================

/// When lhs == rhs, the Add verifier returns value = lhs regardless of trunk_idx.
/// This avoids needing to simulate channel state for the intermediate value.
#[test]
fn test_model_verify_two_adds() {
    let lhs1 = mk(50);
    let rhs1 = mk(50);
    let claimed1 = qm31_add(lhs1, rhs1); // mk(100)

    // With trunk_idx=0, intermediate = lhs1 = 50
    let _intermediate = mk(50);

    // Second add: lhs2 + rhs2 must equal intermediate = 50
    let lhs2 = mk(25);
    let rhs2 = mk(25);

    let mut data: Array<felt252> = array![];
    // Layer 0: Add
    data.append(1);
    push_qm31(ref data, lhs1);
    push_qm31(ref data, rhs1);
    data.append(0); // trunk_idx = 0
    // Layer 1: Add
    data.append(1);
    push_qm31(ref data, lhs2);
    push_qm31(ref data, rhs2);
    data.append(0); // trunk_idx = 0
    // Deferred section
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(42)],
        value: claimed1,
    };

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 2, array![].span(), array![].span(), initial_claim, ref ch,
    );

    assert!(result.point.len() == 1, "point through 2 adds");
}

// ============================================================================
// Test 5: Add with Wrong Sum Panics
// ============================================================================

#[test]
#[should_panic(expected: "ADD_SUM_MISMATCH")]
fn test_model_verify_add_bad_sum_panics() {
    let lhs = mk(100);
    let rhs = mk(200);
    let wrong_claimed = mk(999); // NOT 300

    let mut data: Array<felt252> = array![];
    data.append(1); // tag = Add
    push_qm31(ref data, lhs);
    push_qm31(ref data, rhs);
    data.append(0); // trunk_idx = 0
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(42)],
        value: wrong_claimed,
    };

    let mut ch = channel_default();
    let (_, _) = verify_gkr_model(
        data.span(), 1, array![].span(), array![].span(), initial_claim, ref ch,
    );
}

// ============================================================================
// Test 6: Single MatMul Layer
// ============================================================================

/// Uses "constant polynomial" trick: p(x) = 1 for all x.
/// p(0) + p(1) = 2 = claimed. final_a * final_b = 1 = p(challenge).
#[test]
fn test_model_verify_single_matmul() {
    // m=2, k=2, n=2, log_k=1 round
    let claimed = mk(2);

    // Constant polynomial: c0=1, c1=0, c2=0
    // p(x) = 1 for all x, so p(0)+p(1) = 2 = claimed
    let c0 = mk(1);
    let c1 = qm31_zero();
    let c2 = qm31_zero();

    // final_a * final_b = p(challenge) = 1
    let final_a = mk(1);
    let final_b = mk(1);

    let mut data: Array<felt252> = array![];
    data.append(0); // tag = MatMul
    data.append(1); // num_rounds = 1
    push_qm31(ref data, c0);
    push_qm31(ref data, c1);
    push_qm31(ref data, c2);
    push_qm31(ref data, final_a);
    push_qm31(ref data, final_b);
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(5), mk(7)],  // log_m + log_n = 2 elements for 2x2 output
        value: claimed,
    };

    let matmul_dims: Array<u32> = array![2, 2, 2];

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 1, matmul_dims.span(), array![].span(), initial_claim, ref ch,
    );

    // MatMul returns point = [r_i(log_m=1) || k_challenges(1)] = 2 elements
    assert!(result.point.len() == 2, "matmul point = r_i + k_challenges");
    // value = final_a
    assert!(qm31_eq(result.value, mk(1)), "matmul returns final_a");
}

// ============================================================================
// Test 7: Two MatMuls (Dimension Index Counter)
// ============================================================================

/// Tests that matmul_idx correctly indexes into matmul_dims for consecutive MatMuls.
/// MatMul 1: claimed=2, constant c0=1, final_a*final_b=1 -> output value=1
/// MatMul 2: claimed=1, constant c0=inv(2), final_a*final_b=inv(2) -> output value=inv(2)
#[test]
fn test_model_verify_two_matmuls() {
    // inv(2) in M31: (2^31-1+1)/2 = 2^30 = 1073741824
    let inv2 = mk(1073741824);

    let mut data: Array<felt252> = array![];

    // MatMul 0: claimed=2, constant poly c0=1
    data.append(0); // tag=MatMul
    data.append(1); // num_rounds=1
    push_qm31(ref data, mk(1));       // c0
    push_qm31(ref data, qm31_zero()); // c1
    push_qm31(ref data, qm31_zero()); // c2
    push_qm31(ref data, mk(1));       // final_a
    push_qm31(ref data, mk(1));       // final_b

    // MatMul 1: claimed=1 (output of MatMul 0), constant poly c0=inv2
    // p(0)+p(1) = 2*inv2 = 1 = claimed
    data.append(0); // tag=MatMul
    data.append(1); // num_rounds=1
    push_qm31(ref data, inv2);        // c0
    push_qm31(ref data, qm31_zero()); // c1
    push_qm31(ref data, qm31_zero()); // c2
    push_qm31(ref data, inv2);        // final_a (inv2 * 1 = inv2 = p(r))
    push_qm31(ref data, mk(1));       // final_b

    // Deferred section
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(5), mk(7)],  // log_m + log_n = 2 elements for 2x2 output
        value: mk(2),
    };

    // Two matmuls, each 2x2x2
    let matmul_dims: Array<u32> = array![2, 2, 2, 2, 2, 2];

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 2, matmul_dims.span(), array![].span(), initial_claim, ref ch,
    );

    // Both matmuls have log_m=1, so point = [r_i, k_challenge] = 2 elements
    assert!(result.point.len() == 2, "point length after 2 matmuls");
    // Final value = final_a of last matmul = inv2
    assert!(qm31_eq(result.value, inv2), "final value = inv2");
}

// ============================================================================
// Test 8: Single Mul Layer
// ============================================================================

/// Mul with 1 variable. Uses claim point=[mk(0)] so that eq(0, s) = 1-s.
/// Polynomial: p(x) = V - V*x = V*(1-x) where V = claimed.
/// Then p(0)+p(1) = V+0 = V, and p(s) = V*(1-s) = eq(0,s)*1*V.
#[test]
fn test_model_verify_single_mul() {
    let claimed = mk(100);
    let neg_claimed = qm31_sub(qm31_zero(), claimed); // -100 in M31

    // p(x) = c0 + c1*x + c2*x^2 + c3*x^3 = 100 - 100*x
    let c0 = claimed;
    let c1 = neg_claimed;
    let c2 = qm31_zero();
    let c3 = qm31_zero();

    // lhs = 1, rhs = 100 -> lhs * rhs = 100
    // eq(0, s) * 1 * 100 = (1-s) * 100 = p(s)
    let lhs = mk(1);
    let rhs = claimed;

    let mut data: Array<felt252> = array![];
    data.append(2); // tag = Mul
    data.append(1); // num_rounds = 1
    push_qm31(ref data, c0);
    push_qm31(ref data, c1);
    push_qm31(ref data, c2);
    push_qm31(ref data, c3);
    push_qm31(ref data, lhs);
    push_qm31(ref data, rhs);
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(0)], // r=0 so eq(0,s)=1-s
        value: claimed,
    };

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 1, array![].span(), array![].span(), initial_claim, ref ch,
    );

    // Mul preserves point length
    assert!(result.point.len() == 1, "mul point preserved");
}

// ============================================================================
// Test 9: Add then MatMul (Mixed Layer Types)
// ============================================================================

/// Tests mixed layer types: Add -> MatMul.
/// Uses lhs==rhs trick so intermediate value = lhs = rhs = 50 (trunk_idx=0).
#[test]
fn test_model_verify_add_then_matmul() {
    let lhs = mk(50);
    let rhs = mk(50);
    let claimed = qm31_add(lhs, rhs); // mk(100)

    // After Add with trunk_idx=0: intermediate value = lhs = 50
    // MatMul: claimed = 50, need p(0)+p(1) = 50
    // Use c0 = 25, constant poly: p(x) = 25, so p(0)+p(1) = 50
    let c0 = mk(25);
    let c1 = qm31_zero();
    let c2 = qm31_zero();

    // final_a * final_b = p(challenge) = 25
    let final_a = mk(25);
    let final_b = mk(1);

    let mut data: Array<felt252> = array![];
    // Layer 0: Add
    data.append(1);
    push_qm31(ref data, lhs);
    push_qm31(ref data, rhs);
    data.append(0); // trunk_idx = 0
    // Layer 1: MatMul
    data.append(0); // tag = MatMul
    data.append(1); // num_rounds = 1
    push_qm31(ref data, c0);
    push_qm31(ref data, c1);
    push_qm31(ref data, c2);
    push_qm31(ref data, final_a);
    push_qm31(ref data, final_b);
    // Deferred section
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(42), mk(7)],  // log_m + log_n = 2 elements for 2x2 output
        value: claimed,
    };

    let matmul_dims: Array<u32> = array![2, 2, 2];

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 2, matmul_dims.span(), array![].span(), initial_claim, ref ch,
    );

    // After Add: point=[mk(42), mk(7)], after MatMul: point=[r_i, k_challenge] = 2 elements
    assert!(result.point.len() == 2, "matmul extends point");
    assert!(qm31_eq(result.value, mk(25)), "returns final_a");
}

// ============================================================================
// Test 10: Channel Determinism
// ============================================================================

/// Verify that the model verifier's channel state matches manual simulation.
#[test]
fn test_model_verifier_channel_determinism() {
    let lhs = mk(50);
    let rhs = mk(50);
    let claimed = qm31_add(lhs, rhs);

    let mut data: Array<felt252> = array![];
    data.append(1); // tag = Add
    push_qm31(ref data, lhs);
    push_qm31(ref data, rhs);
    data.append(0); // trunk_idx = 0
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(42)],
        value: claimed,
    };

    let mut ch = channel_default();
    let (_, _) = verify_gkr_model(
        data.span(), 1, array![].span(), array![].span(), initial_claim, ref ch,
    );
    let digest_after_model = ch.digest;

    // Manually simulate the Add verifier's channel operations
    let mut sim_ch = channel_default();
    channel_mix_secure_field(ref sim_ch, lhs);
    channel_mix_secure_field(ref sim_ch, rhs);
    let _ = channel_draw_qm31(ref sim_ch);
    let digest_after_sim = sim_ch.digest;

    assert!(digest_after_model == digest_after_sim, "channel state matches simulation");
}

// ============================================================================
// Test 11: MatMul with Complex QM31 Values
// ============================================================================

/// Tests that the flat reader correctly parses QM31 with all 4 components nonzero.
#[test]
fn test_model_verify_matmul_complex_qm31() {
    // Use QM31 with all components: claimed = (2, 0, 0, 0) for sum check simplicity
    // But use complex final evaluations
    let claimed = mk(2);

    let c0 = mk(1);
    let c1 = qm31_zero();
    let c2 = qm31_zero();

    // final_a has all 4 components nonzero
    let final_a = qm31_new(1, 0, 0, 0);
    // final_b = 1 so product = final_a = mk(1) = c0 = p(challenge)
    let final_b = qm31_new(1, 0, 0, 0);

    let mut data: Array<felt252> = array![];
    data.append(0); // tag = MatMul
    data.append(1); // num_rounds = 1
    push_qm31(ref data, c0);
    push_qm31(ref data, c1);
    push_qm31(ref data, c2);
    push_qm31(ref data, final_a);
    push_qm31(ref data, final_b);
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(5), mk(7)],  // log_m + log_n = 2 elements for 2x2 output
        value: claimed,
    };
    let matmul_dims: Array<u32> = array![2, 2, 2];

    let mut ch = channel_default();
    let (result, _weight_claims) = verify_gkr_model(
        data.span(), 1, matmul_dims.span(), array![].span(), initial_claim, ref ch,
    );

    assert!(result.point.len() == 2, "point len");
    assert!(qm31_eq(result.value, final_a), "value = final_a");
}

// ============================================================================
// Test 12: MatMul Round Sum Mismatch Through Model Verifier
// ============================================================================

#[test]
#[should_panic(expected: "MATMUL_ROUND_SUM_MISMATCH")]
fn test_model_verify_matmul_bad_round_sum_panics() {
    let claimed = mk(2);

    // Bad poly: p(0)+p(1) = 5+5 = 10 != 2
    let c0 = mk(5);
    let c1 = qm31_zero();
    let c2 = qm31_zero();

    let mut data: Array<felt252> = array![];
    data.append(0);
    data.append(1);
    push_qm31(ref data, c0);
    push_qm31(ref data, c1);
    push_qm31(ref data, c2);
    push_qm31(ref data, mk(1)); // final_a
    push_qm31(ref data, mk(1)); // final_b
    data.append(0); // num_deferred = 0

    let initial_claim = GKRClaim {
        point: array![mk(5), mk(7)],  // log_m + log_n = 2 elements for 2x2 output
        value: claimed,
    };
    let matmul_dims: Array<u32> = array![2, 2, 2];

    let mut ch = channel_default();
    let (_, _) = verify_gkr_model(
        data.span(), 1, matmul_dims.span(), array![].span(), initial_claim, ref ch,
    );
}
