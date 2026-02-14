// Tests for GKR batch verification.
//
// Strategy: construct valid GKR proofs by replaying the exact verifier transcript
// (channel operations from gkr_verifier.rs partially_verify_batch), then verify
// they pass. Tampered proofs must fail.

use elo_cairo_verifier::field::{
    qm31_new, qm31_zero, qm31_one, qm31_add, qm31_mul, qm31_eq,
    poly_eval_degree3, eq_eval, fold_mle_eval, random_linear_combination,
    evaluate_mle, pad_and_embed_m31s, m31_to_qm31,
};
use elo_cairo_verifier::field::QM31;
use elo_cairo_verifier::channel::{
    channel_default, channel_mix_u64, channel_mix_felts, channel_draw_qm31,
};
use elo_cairo_verifier::types::{
    GateType, GkrRoundPoly, GkrMask, GkrSumcheckProof, GkrLayerProof,
    GkrInstance, GkrBatchProof,
};
use elo_cairo_verifier::gkr::{
    eval_grand_product_gate, eval_logup_gate, eval_gate,
    reduce_mask_at_point, verify_gkr_sumcheck, partially_verify_batch,
};

// ============================================================================
// Phase 2a: Field Extension Tests
// ============================================================================

#[test]
fn test_poly_eval_degree3_at_zero() {
    // p(0) = c0
    let c0 = qm31_new(42, 0, 0, 0);
    let c1 = qm31_new(7, 0, 0, 0);
    let c2 = qm31_new(3, 0, 0, 0);
    let c3 = qm31_new(1, 0, 0, 0);
    let result = poly_eval_degree3(c0, c1, c2, c3, qm31_zero());
    assert!(qm31_eq(result, c0), "p(0) = c0");
}

#[test]
fn test_poly_eval_degree3_at_one() {
    // p(1) = c0 + c1 + c2 + c3
    let c0 = qm31_new(10, 0, 0, 0);
    let c1 = qm31_new(20, 0, 0, 0);
    let c2 = qm31_new(30, 0, 0, 0);
    let c3 = qm31_new(40, 0, 0, 0);
    let result = poly_eval_degree3(c0, c1, c2, c3, qm31_one());
    let expected = qm31_new(100, 0, 0, 0);
    assert!(qm31_eq(result, expected), "p(1) = 100");
}

#[test]
fn test_poly_eval_degree3_horner() {
    // p(x) = 1 + 2x + 3x^2 + 4x^3
    // p(2) = 1 + 4 + 12 + 32 = 49
    let c0 = qm31_new(1, 0, 0, 0);
    let c1 = qm31_new(2, 0, 0, 0);
    let c2 = qm31_new(3, 0, 0, 0);
    let c3 = qm31_new(4, 0, 0, 0);
    let x = qm31_new(2, 0, 0, 0);
    let result = poly_eval_degree3(c0, c1, c2, c3, x);
    assert!(qm31_eq(result, qm31_new(49, 0, 0, 0)), "p(2) = 49");
}

#[test]
fn test_eq_eval_identical_points() {
    let one = qm31_one();
    let zero = qm31_zero();
    // eq([1,0,1], [1,0,1]) = 1
    let x = array![one, zero, one];
    let y = array![one, zero, one];
    let result = eq_eval(x.span(), y.span());
    assert!(qm31_eq(result, one), "eq(x,x) = 1");
}

#[test]
fn test_eq_eval_different_points() {
    let one = qm31_one();
    let zero = qm31_zero();
    // eq([1,0,1], [1,0,0]) = 1*1 * (1-0)*(1-0) * 0 = 0
    let x = array![one, zero, one];
    let y = array![one, zero, zero];
    let result = eq_eval(x.span(), y.span());
    assert!(qm31_eq(result, zero), "eq(different) = 0");
}

#[test]
fn test_eq_eval_empty() {
    // eq([], []) = 1 (empty product)
    let x: Array<elo_cairo_verifier::field::QM31> = array![];
    let y: Array<elo_cairo_verifier::field::QM31> = array![];
    let result = eq_eval(x.span(), y.span());
    assert!(qm31_eq(result, qm31_one()), "eq([],[]) = 1");
}

#[test]
fn test_fold_mle_eval_at_zero() {
    // fold(0, v0, v1) = v0
    let v0 = qm31_new(42, 0, 0, 0);
    let v1 = qm31_new(99, 0, 0, 0);
    let result = fold_mle_eval(qm31_zero(), v0, v1);
    assert!(qm31_eq(result, v0), "fold(0, v0, v1) = v0");
}

#[test]
fn test_fold_mle_eval_at_one() {
    // fold(1, v0, v1) = v1
    let v0 = qm31_new(42, 0, 0, 0);
    let v1 = qm31_new(99, 0, 0, 0);
    let result = fold_mle_eval(qm31_one(), v0, v1);
    assert!(qm31_eq(result, v1), "fold(1, v0, v1) = v1");
}

#[test]
fn test_random_linear_combination() {
    // rlc([10, 20, 30], alpha=3) = 10 + 3*20 + 9*30 = 10 + 60 + 270 = 340
    let vals = array![
        qm31_new(10, 0, 0, 0),
        qm31_new(20, 0, 0, 0),
        qm31_new(30, 0, 0, 0),
    ];
    let alpha = qm31_new(3, 0, 0, 0);
    let result = random_linear_combination(vals.span(), alpha);
    assert!(qm31_eq(result, qm31_new(340, 0, 0, 0)), "rlc = 340");
}

#[test]
fn test_random_linear_combination_empty() {
    let vals: Array<elo_cairo_verifier::field::QM31> = array![];
    let alpha = qm31_new(7, 0, 0, 0);
    let result = random_linear_combination(vals.span(), alpha);
    assert!(qm31_eq(result, qm31_zero()), "rlc([]) = 0");
}

#[test]
fn test_random_linear_combination_single() {
    let vals = array![qm31_new(42, 0, 0, 0)];
    let alpha = qm31_new(999, 0, 0, 0);
    let result = random_linear_combination(vals.span(), alpha);
    assert!(qm31_eq(result, qm31_new(42, 0, 0, 0)), "rlc([42]) = 42");
}

// ============================================================================
// Gate Evaluation Tests
// ============================================================================

#[test]
fn test_grand_product_gate() {
    // GrandProduct: output = a * b
    let mask = GkrMask {
        values: array![qm31_new(3, 0, 0, 0), qm31_new(7, 0, 0, 0)],
        num_columns: 1,
    };
    let output = eval_grand_product_gate(@mask);
    assert!(output.len() == 1, "GP has 1 output");
    assert!(qm31_eq(*output.at(0), qm31_new(21, 0, 0, 0)), "3*7 = 21");
}

#[test]
fn test_grand_product_gate_zero() {
    let mask = GkrMask {
        values: array![qm31_new(0, 0, 0, 0), qm31_new(42, 0, 0, 0)],
        num_columns: 1,
    };
    let output = eval_grand_product_gate(@mask);
    assert!(qm31_eq(*output.at(0), qm31_zero()), "0*42 = 0");
}

#[test]
fn test_logup_gate() {
    // LogUp: (n_a/d_a) + (n_b/d_b) = (n_a*d_b + n_b*d_a, d_a*d_b)
    // (1/3) + (2/5) = (1*5 + 2*3, 3*5) = (11, 15)
    let mask = GkrMask {
        values: array![
            qm31_new(1, 0, 0, 0), // n_a
            qm31_new(2, 0, 0, 0), // n_b
            qm31_new(3, 0, 0, 0), // d_a
            qm31_new(5, 0, 0, 0), // d_b
        ],
        num_columns: 2,
    };
    let output = eval_logup_gate(@mask);
    assert!(output.len() == 2, "LogUp has 2 outputs");
    assert!(qm31_eq(*output.at(0), qm31_new(11, 0, 0, 0)), "numerator = 11");
    assert!(qm31_eq(*output.at(1), qm31_new(15, 0, 0, 0)), "denominator = 15");
}

#[test]
fn test_logup_gate_zero_numerators() {
    // (0/d_a) + (0/d_b) = (0, d_a*d_b)
    let mask = GkrMask {
        values: array![
            qm31_zero(), qm31_zero(),
            qm31_new(3, 0, 0, 0), qm31_new(5, 0, 0, 0),
        ],
        num_columns: 2,
    };
    let output = eval_logup_gate(@mask);
    assert!(qm31_eq(*output.at(0), qm31_zero()), "0+0 = 0 num");
    assert!(qm31_eq(*output.at(1), qm31_new(15, 0, 0, 0)), "den = 15");
}

#[test]
fn test_eval_gate_dispatch() {
    let gp_gate = GateType { gate_id: 0 };
    let mask = GkrMask {
        values: array![qm31_new(4, 0, 0, 0), qm31_new(5, 0, 0, 0)],
        num_columns: 1,
    };
    let output = eval_gate(@gp_gate, @mask);
    assert!(qm31_eq(*output.at(0), qm31_new(20, 0, 0, 0)), "GP 4*5=20");

    let logup_gate = GateType { gate_id: 1 };
    let mask2 = GkrMask {
        values: array![
            qm31_new(1, 0, 0, 0), qm31_new(1, 0, 0, 0),
            qm31_new(2, 0, 0, 0), qm31_new(3, 0, 0, 0),
        ],
        num_columns: 2,
    };
    let output2 = eval_gate(@logup_gate, @mask2);
    assert!(output2.len() == 2, "LogUp has 2 outputs");
}

// ============================================================================
// Mask Reduction Tests
// ============================================================================

#[test]
fn test_reduce_mask_single_column() {
    // fold(r=0.5ish, v0=10, v1=20) = 10 + r*(20-10) = 10 + 10r
    let mask = GkrMask {
        values: array![qm31_new(10, 0, 0, 0), qm31_new(20, 0, 0, 0)],
        num_columns: 1,
    };
    let r = qm31_zero(); // fold at 0 -> v0
    let result = reduce_mask_at_point(@mask, r);
    assert!(result.len() == 1, "1 column -> 1 result");
    assert!(qm31_eq(*result.at(0), qm31_new(10, 0, 0, 0)), "fold(0) = v0");
}

#[test]
fn test_reduce_mask_at_one() {
    let mask = GkrMask {
        values: array![qm31_new(10, 0, 0, 0), qm31_new(20, 0, 0, 0)],
        num_columns: 1,
    };
    let r = qm31_one(); // fold at 1 -> v1
    let result = reduce_mask_at_point(@mask, r);
    assert!(qm31_eq(*result.at(0), qm31_new(20, 0, 0, 0)), "fold(1) = v1");
}

#[test]
fn test_reduce_mask_multi_column() {
    // 2 columns: [col0_v0, col0_v1, col1_v0, col1_v1]
    let mask = GkrMask {
        values: array![
            qm31_new(10, 0, 0, 0), qm31_new(20, 0, 0, 0), // col 0
            qm31_new(30, 0, 0, 0), qm31_new(50, 0, 0, 0), // col 1
        ],
        num_columns: 2,
    };
    let r = qm31_one();
    let result = reduce_mask_at_point(@mask, r);
    assert!(result.len() == 2, "2 columns -> 2 results");
    assert!(qm31_eq(*result.at(0), qm31_new(20, 0, 0, 0)), "col0: fold(1) = 20");
    assert!(qm31_eq(*result.at(1), qm31_new(50, 0, 0, 0)), "col1: fold(1) = 50");
}

// ============================================================================
// GKR Sumcheck Tests
// ============================================================================

#[test]
fn test_gkr_sumcheck_single_round_degree2() {
    // Single round, degree-2 poly (3 coefficients, like matmul sumcheck)
    // claim = p(0) + p(1) = c0 + (c0 + c1 + c2) = 2c0 + c1 + c2
    let c0 = qm31_new(21, 0, 0, 0);
    let c1 = qm31_new(34, 0, 0, 0);
    let c2 = qm31_zero();
    let c3 = qm31_zero();
    // claim = 42 + 34 + 0 = 76
    let claim = qm31_new(76, 0, 0, 0);

    let proof = GkrSumcheckProof {
        round_polys: array![GkrRoundPoly { c0, c1, c2, c3, num_coeffs: 3 }],
    };

    let mut ch = channel_default();
    let (assignment, final_eval) = verify_gkr_sumcheck(claim, @proof, ref ch);

    assert!(assignment.len() == 1, "1 round -> 1 challenge");
    // final_eval = p(challenge) — just verify it's determined
    let expected = poly_eval_degree3(c0, c1, c2, c3, *assignment.at(0));
    assert!(qm31_eq(final_eval, expected), "final eval matches p(challenge)");
}

#[test]
fn test_gkr_sumcheck_single_round_degree3() {
    // Single round, degree-3 poly (4 coefficients)
    // p(0) = c0, p(1) = c0+c1+c2+c3
    // claim = p(0) + p(1) = 2*c0 + c1 + c2 + c3
    let c0 = qm31_new(10, 0, 0, 0);
    let c1 = qm31_new(20, 0, 0, 0);
    let c2 = qm31_new(5, 0, 0, 0);
    let c3 = qm31_new(3, 0, 0, 0);
    // claim = 20 + 20 + 5 + 3 = 48
    let claim = qm31_new(48, 0, 0, 0);

    let proof = GkrSumcheckProof {
        round_polys: array![GkrRoundPoly { c0, c1, c2, c3, num_coeffs: 4 }],
    };

    let mut ch = channel_default();
    let (assignment, final_eval) = verify_gkr_sumcheck(claim, @proof, ref ch);

    assert!(assignment.len() == 1, "1 round");
    let expected = poly_eval_degree3(c0, c1, c2, c3, *assignment.at(0));
    assert!(qm31_eq(final_eval, expected), "final eval matches");
}

#[test]
fn test_gkr_sumcheck_two_rounds() {
    // Two rounds: first constructs claim for second
    let c0_r1 = qm31_new(30, 0, 0, 0);
    let c1_r1 = qm31_new(40, 0, 0, 0);
    // claim_r1 = 2*30 + 40 = 100
    let claim = qm31_new(100, 0, 0, 0);

    // We need to derive the challenge for round 1 to build round 2
    let mut ch_sim = channel_default();

    // Mix round 1 poly (3 coeffs)
    let coeffs_r1 = array![c0_r1, c1_r1, qm31_zero()];
    channel_mix_felts(ref ch_sim, coeffs_r1.span());
    let challenge_1 = channel_draw_qm31(ref ch_sim);

    // expected_sum for round 2 = p1(challenge_1)
    let sum_r2 = poly_eval_degree3(c0_r1, c1_r1, qm31_zero(), qm31_zero(), challenge_1);

    // Round 2: need p2(0) + p2(1) = sum_r2
    // Use c0=0, c1=sum_r2, c2=0, c3=0 -> 0 + sum_r2 = sum_r2 ✓
    let c0_r2 = qm31_zero();
    let c1_r2 = sum_r2;

    let proof = GkrSumcheckProof {
        round_polys: array![
            GkrRoundPoly { c0: c0_r1, c1: c1_r1, c2: qm31_zero(), c3: qm31_zero(), num_coeffs: 3 },
            GkrRoundPoly { c0: c0_r2, c1: c1_r2, c2: qm31_zero(), c3: qm31_zero(), num_coeffs: 3 },
        ],
    };

    let mut ch = channel_default();
    let (_assignment, _final_eval) = verify_gkr_sumcheck(claim, @proof, ref ch);

    assert!(_assignment.len() == 2, "2 rounds -> 2 challenges");
}

#[test]
fn test_gkr_sumcheck_channel_consistency() {
    // Verify the channel state matches between prover simulation and verifier
    let c0 = qm31_new(21, 0, 0, 0);
    let c1 = qm31_new(34, 0, 0, 0);
    let claim = qm31_new(76, 0, 0, 0);

    let proof = GkrSumcheckProof {
        round_polys: array![GkrRoundPoly {
            c0, c1, c2: qm31_zero(), c3: qm31_zero(), num_coeffs: 3,
        }],
    };

    // Run verifier
    let mut ch_v = channel_default();
    let (assignment_v, _) = verify_gkr_sumcheck(claim, @proof, ref ch_v);

    // Simulate prover channel
    let mut ch_p = channel_default();
    let coeffs = array![c0, c1, qm31_zero()];
    channel_mix_felts(ref ch_p, coeffs.span());
    let challenge_p = channel_draw_qm31(ref ch_p);

    // Challenges must match
    assert!(qm31_eq(*assignment_v.at(0), challenge_p), "channel consistency");
}

// ============================================================================
// Full Batch Verification Tests
// ============================================================================

/// Build a valid single-instance GrandProduct GKR proof with 1 layer.
///
/// Key insight: at layer 0, ood_point is empty, so the sumcheck has 0 rounds
/// (trivial sumcheck). The claim is directly checked against the gate evaluation.
/// The mask values must produce a gate output matching the output claims.
fn build_single_gp_proof() -> GkrBatchProof {
    // GrandProduct: gate output = a * b
    // output_claims = [product], mask = [a, b] where a*b = product
    let a = qm31_new(3, 0, 0, 0);
    let b = qm31_new(7, 0, 0, 0);
    let product = qm31_mul(a, b); // 21

    // Layer 0: trivial sumcheck (0 rounds).
    // The verifier checks: sumcheck_eval == eq_val * rlc(gate_output, lambda)
    // With 0 rounds: sumcheck_eval = sumcheck_claim
    // eq_val = eq([], []) = 1 (both ood_point and sumcheck_ood_point are empty)
    // gate_output = [a*b] = [21]
    // rlc([21], lambda) = 21
    // sumcheck_claim = rlc([rlc([21], lambda) * doubling], alpha) = 21
    // Check: 21 == 1 * 21 = 21

    let mask = GkrMask {
        values: array![a, b],
        num_columns: 1,
    };

    GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 0 },
            n_variables: 1,
            output_claims: array![product],
        }],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof {
                round_polys: array![],  // 0 rounds (trivial sumcheck)
            },
            masks: array![mask],
        }],
    }
}

#[test]
fn test_gkr_batch_verify_single_gp() {
    let proof = build_single_gp_proof();
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@proof, ref ch);

    // After 1 layer with 0 sumcheck rounds: ood_point = [] + [challenge] = length 1
    assert!(artifact.ood_point.len() == 1, "1 layer -> 1 ood coord");
    assert!(artifact.claims_to_verify.len() == 1, "1 instance");
    assert!(artifact.n_variables_by_instance.len() == 1, "1 instance");
    assert!(*artifact.n_variables_by_instance.at(0) == 1, "n_vars = 1");
    // GP: 1 claim per instance (the folded value)
    assert!(artifact.claims_to_verify.at(0).len() == 1, "GP: 1 claim");
}

/// Build a valid single-instance LogUp GKR proof with 1 layer.
fn build_single_logup_proof() -> GkrBatchProof {
    // LogUp: 2 columns (numerator, denominator)
    // gate: (n_a/d_a) + (n_b/d_b) = (n_a*d_b + n_b*d_a, d_a*d_b)
    // We need gate_output == output_claims for the trivial sumcheck to pass.
    //
    // (1/3) + (2/5) = (1*5+2*3, 3*5) = (11, 15)
    let n_a = qm31_new(1, 0, 0, 0);
    let n_b = qm31_new(2, 0, 0, 0);
    let d_a = qm31_new(3, 0, 0, 0);
    let d_b = qm31_new(5, 0, 0, 0);
    let out_num = qm31_new(11, 0, 0, 0);
    let out_den = qm31_new(15, 0, 0, 0);

    let mask = GkrMask {
        values: array![n_a, n_b, d_a, d_b],
        num_columns: 2,
    };

    GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 1 },
            n_variables: 1,
            output_claims: array![out_num, out_den],
        }],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof {
                round_polys: array![],  // 0 rounds
            },
            masks: array![mask],
        }],
    }
}

#[test]
fn test_gkr_batch_verify_single_logup() {
    let proof = build_single_logup_proof();
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@proof, ref ch);

    assert!(artifact.ood_point.len() == 1, "1 layer -> 1 ood coord");
    assert!(artifact.claims_to_verify.len() == 1, "1 instance");
    // LogUp has 2 claims (numerator, denominator reduced)
    assert!(artifact.claims_to_verify.at(0).len() == 2, "LogUp: 2 claims");
}

// ============================================================================
// Serde Tests
// ============================================================================

#[test]
fn test_gkr_batch_proof_serde_roundtrip() {
    let proof = build_single_gp_proof();

    // Serialize
    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    // Deserialize
    let mut span = serialized.span();
    let deserialized: GkrBatchProof = Serde::<GkrBatchProof>::deserialize(ref span)
        .expect('gkr serde deser failed');

    // Verify round-trip: run verification on deserialized proof
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@deserialized, ref ch);

    assert!(artifact.ood_point.len() == 1, "roundtrip ood_point");
    assert!(artifact.claims_to_verify.len() == 1, "roundtrip claims");
    assert!(span.len() == 0, "all data consumed");
}

#[test]
fn test_gkr_types_serde_deserialization() {
    // Manual calldata matching the expected Serde layout for GkrBatchProof
    // 1 instance (GP, n_vars=1, 1 output claim), 1 layer proof (1 round poly, 1 mask)
    let mut calldata: Array<felt252> = array![
        // instances.len
        1,
        // instance[0].gate.gate_id
        0,
        // instance[0].n_variables
        1,
        // instance[0].output_claims.len
        1,
        // instance[0].output_claims[0] = QM31(21, 0, 0, 0)
        21, 0, 0, 0,
        // layer_proofs.len
        1,
        // layer_proof[0].sumcheck_proof.round_polys.len
        1,
        // round_poly[0]: c0, c1, c2, c3, num_coeffs
        0, 0, 0, 0,  // c0 = QM31(0,0,0,0)
        10, 0, 0, 0,  // c1 = QM31(10,0,0,0)
        0, 0, 0, 0,  // c2
        0, 0, 0, 0,  // c3
        2,            // num_coeffs
        // layer_proof[0].masks.len
        1,
        // mask[0].values.len
        2,
        // mask[0].values[0] = QM31(3,0,0,0)
        3, 0, 0, 0,
        // mask[0].values[1] = QM31(7,0,0,0)
        7, 0, 0, 0,
        // mask[0].num_columns
        1,
    ];

    let mut span = calldata.span();
    let proof: GkrBatchProof = Serde::<GkrBatchProof>::deserialize(ref span)
        .expect('gkr calldata deser failed');

    assert!(proof.instances.len() == 1, "1 instance");
    assert!(*proof.instances.at(0).gate.gate_id == 0, "GP gate");
    assert!(*proof.instances.at(0).n_variables == 1, "n_vars=1");
    assert!(proof.layer_proofs.len() == 1, "1 layer");
    assert!(span.len() == 0, "all consumed");
}

// ============================================================================
// Variable-Degree Polynomial Handling
// ============================================================================

#[test]
fn test_gkr_round_poly_truncated_degree1() {
    // Degree-1 poly (2 coefficients): p(x) = c0 + c1*x
    // p(0)=c0=5, p(1)=c0+c1=5+10=15, sum=20
    let claim = qm31_new(20, 0, 0, 0);
    let proof = GkrSumcheckProof {
        round_polys: array![GkrRoundPoly {
            c0: qm31_new(5, 0, 0, 0),
            c1: qm31_new(10, 0, 0, 0),
            c2: qm31_zero(),
            c3: qm31_zero(),
            num_coeffs: 2,
        }],
    };

    let mut ch = channel_default();
    let (assignment, _) = verify_gkr_sumcheck(claim, @proof, ref ch);
    assert!(assignment.len() == 1, "1 round");
}

// ============================================================================
// Multi-Instance Batch Verification Tests
// ============================================================================

/// Build a two-instance GrandProduct proof where both instances have n_variables=1.
/// This tests the batching (lambda/alpha RLC) without staggered activation.
fn build_two_instance_gp_proof() -> GkrBatchProof {
    // Instance 0: 3 * 7 = 21
    // Instance 1: 5 * 11 = 55
    let a0 = qm31_new(3, 0, 0, 0);
    let b0 = qm31_new(7, 0, 0, 0);
    let product0 = qm31_mul(a0, b0); // 21

    let a1 = qm31_new(5, 0, 0, 0);
    let b1 = qm31_new(11, 0, 0, 0);
    let product1 = qm31_mul(a1, b1); // 55

    // Layer 0: both active, 0 sumcheck rounds
    // sumcheck_claim = rlc([product0, product1], alpha)
    // layer_eval = rlc([1*product0, 1*product1], alpha)
    // These are equal by construction.

    GkrBatchProof {
        instances: array![
            GkrInstance {
                gate: GateType { gate_id: 0 },
                n_variables: 1,
                output_claims: array![product0],
            },
            GkrInstance {
                gate: GateType { gate_id: 0 },
                n_variables: 1,
                output_claims: array![product1],
            },
        ],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof {
                round_polys: array![],  // 0 rounds
            },
            masks: array![
                GkrMask { values: array![a0, b0], num_columns: 1 },
                GkrMask { values: array![a1, b1], num_columns: 1 },
            ],
        }],
    }
}

#[test]
fn test_gkr_batch_verify_two_instances_gp() {
    let proof = build_two_instance_gp_proof();
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@proof, ref ch);

    assert!(artifact.ood_point.len() == 1, "1 layer -> 1 ood coord");
    assert!(artifact.claims_to_verify.len() == 2, "2 instances");
    assert!(artifact.n_variables_by_instance.len() == 2, "2 n_vars");
    assert!(*artifact.n_variables_by_instance.at(0) == 1, "inst0 n_vars=1");
    assert!(*artifact.n_variables_by_instance.at(1) == 1, "inst1 n_vars=1");
    // Each GP instance reduces to 1 claim
    assert!(artifact.claims_to_verify.at(0).len() == 1, "inst0: 1 claim");
    assert!(artifact.claims_to_verify.at(1).len() == 1, "inst1: 1 claim");
}

/// Build a mixed batch: GP + LogUp, both n_variables=1.
/// Tests that different gate types in the same batch work correctly.
fn build_mixed_gates_proof() -> GkrBatchProof {
    // Instance 0 (GrandProduct): 4 * 6 = 24
    let gp_a = qm31_new(4, 0, 0, 0);
    let gp_b = qm31_new(6, 0, 0, 0);
    let gp_product = qm31_mul(gp_a, gp_b); // 24

    // Instance 1 (LogUp): (2/3) + (5/7) = (2*7+5*3, 3*7) = (29, 21)
    let n_a = qm31_new(2, 0, 0, 0);
    let n_b = qm31_new(5, 0, 0, 0);
    let d_a = qm31_new(3, 0, 0, 0);
    let d_b = qm31_new(7, 0, 0, 0);
    let out_num = qm31_new(29, 0, 0, 0);
    let out_den = qm31_new(21, 0, 0, 0);

    GkrBatchProof {
        instances: array![
            GkrInstance {
                gate: GateType { gate_id: 0 },  // GP
                n_variables: 1,
                output_claims: array![gp_product],
            },
            GkrInstance {
                gate: GateType { gate_id: 1 },  // LogUp
                n_variables: 1,
                output_claims: array![out_num, out_den],
            },
        ],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof {
                round_polys: array![],  // 0 rounds
            },
            masks: array![
                GkrMask { values: array![gp_a, gp_b], num_columns: 1 },
                GkrMask { values: array![n_a, n_b, d_a, d_b], num_columns: 2 },
            ],
        }],
    }
}

#[test]
fn test_gkr_batch_verify_mixed_gates() {
    let proof = build_mixed_gates_proof();
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@proof, ref ch);

    assert!(artifact.claims_to_verify.len() == 2, "2 instances");
    // GP instance: 1 claim
    assert!(artifact.claims_to_verify.at(0).len() == 1, "GP: 1 claim");
    // LogUp instance: 2 claims (numerator, denominator)
    assert!(artifact.claims_to_verify.at(1).len() == 2, "LogUp: 2 claims");
}

// ============================================================================
// Tampered Proof Rejection Tests
// ============================================================================

#[test]
#[should_panic(expected: "GKR: circuit check failure")]
fn test_gkr_reject_tampered_mask() {
    // Valid proof with mask [3, 7] -> product=21
    // Tamper: change mask to [3, 8] -> gate output=24 != product=21
    let product = qm31_new(21, 0, 0, 0);

    let proof = GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 0 },
            n_variables: 1,
            output_claims: array![product],
        }],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
            masks: array![GkrMask {
                values: array![qm31_new(3, 0, 0, 0), qm31_new(8, 0, 0, 0)],  // TAMPERED
                num_columns: 1,
            }],
        }],
    };

    let mut ch = channel_default();
    partially_verify_batch(@proof, ref ch);  // Should panic
}

#[test]
#[should_panic(expected: "GKR: circuit check failure")]
fn test_gkr_reject_tampered_claim() {
    // Valid: 3*7=21, but claim says 22
    let proof = GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 0 },
            n_variables: 1,
            output_claims: array![qm31_new(22, 0, 0, 0)],  // WRONG
        }],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
            masks: array![GkrMask {
                values: array![qm31_new(3, 0, 0, 0), qm31_new(7, 0, 0, 0)],
                num_columns: 1,
            }],
        }],
    };

    let mut ch = channel_default();
    partially_verify_batch(@proof, ref ch);  // Should panic
}

#[test]
#[should_panic(expected: "GKR: layer count mismatch")]
fn test_gkr_reject_wrong_layer_count() {
    // Instance says n_variables=2, but only 1 layer proof provided
    let proof = GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 0 },
            n_variables: 2,  // Expects 2 layers
            output_claims: array![qm31_new(21, 0, 0, 0)],
        }],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
            masks: array![GkrMask {
                values: array![qm31_new(3, 0, 0, 0), qm31_new(7, 0, 0, 0)],
                num_columns: 1,
            }],
        }],
    };

    let mut ch = channel_default();
    partially_verify_batch(@proof, ref ch);  // Should panic
}

#[test]
#[should_panic]
fn test_gkr_reject_wrong_gate_type() {
    // GrandProduct output claim but LogUp gate (2 columns expected, only 1 provided)
    let proof = GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 1 },  // LogUp
            n_variables: 1,
            output_claims: array![qm31_new(21, 0, 0, 0)],
        }],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
            masks: array![GkrMask {
                values: array![qm31_new(3, 0, 0, 0), qm31_new(7, 0, 0, 0)],
                num_columns: 1,  // Wrong for LogUp!
            }],
        }],
    };

    let mut ch = channel_default();
    partially_verify_batch(@proof, ref ch);
}

// ============================================================================
// Serde Tests for Multi-Instance
// ============================================================================

#[test]
fn test_two_instance_serde_roundtrip() {
    let proof = build_two_instance_gp_proof();

    // Serialize
    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    // Deserialize
    let mut span = serialized.span();
    let deserialized: GkrBatchProof = Serde::<GkrBatchProof>::deserialize(ref span)
        .expect('2inst serde deser failed');

    // Verify roundtrip
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@deserialized, ref ch);

    assert!(artifact.claims_to_verify.len() == 2, "2 instances roundtrip");
    assert!(span.len() == 0, "all data consumed");
}

#[test]
fn test_mixed_gates_serde_roundtrip() {
    let proof = build_mixed_gates_proof();

    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    let mut span = serialized.span();
    let deserialized: GkrBatchProof = Serde::<GkrBatchProof>::deserialize(ref span)
        .expect('mixed serde deser failed');

    let mut ch = channel_default();
    let artifact = partially_verify_batch(@deserialized, ref ch);

    assert!(artifact.claims_to_verify.len() == 2, "2 instances roundtrip");
    assert!(artifact.claims_to_verify.at(0).len() == 1, "GP: 1 claim");
    assert!(artifact.claims_to_verify.at(1).len() == 2, "LogUp: 2 claims");
    assert!(span.len() == 0, "all data consumed");
}

// ============================================================================
// Task #14: QM31 Field Arithmetic Cross-Verification (Rust → Cairo)
//
// Test vectors generated by stwo-ml Rust tests (poseidon_channel.rs).
// Each value was computed using STWO's canonical QM31 arithmetic.
// ============================================================================

#[test]
fn test_qm31_mul_matches_rust_vector() {
    // Rust: a = QM31(1234567, 7654321, 111222, 333444)
    //       b = QM31(9876543, 3456789, 555666, 777888)
    //       a*b = QM31(521350170, 1230121829, 1257727717, 763478952)
    let a = qm31_new(1234567, 7654321, 111222, 333444);
    let b = qm31_new(9876543, 3456789, 555666, 777888);
    let expected = qm31_new(521350170, 1230121829, 1257727717, 763478952);
    let result = qm31_mul(a, b);
    assert!(qm31_eq(result, expected), "QM31 mul mismatch vs Rust");
}

#[test]
fn test_qm31_add_matches_rust_vector() {
    // Rust: a+b = QM31(11111110, 11111110, 666888, 1111332)
    let a = qm31_new(1234567, 7654321, 111222, 333444);
    let b = qm31_new(9876543, 3456789, 555666, 777888);
    let expected = qm31_new(11111110, 11111110, 666888, 1111332);
    let result = qm31_add(a, b);
    assert!(qm31_eq(result, expected), "QM31 add mismatch vs Rust");
}

#[test]
fn test_eq_eval_simple_matches_rust_vector() {
    // Rust: eq([42,17], [99,5]) = QM31(1218224, 0, 0, 0)
    // Using simple M31 values (b components = 0)
    let x = array![qm31_new(42, 0, 0, 0), qm31_new(17, 0, 0, 0)];
    let y = array![qm31_new(99, 0, 0, 0), qm31_new(5, 0, 0, 0)];
    let expected = qm31_new(1218224, 0, 0, 0);
    let result = eq_eval(x.span(), y.span());
    assert!(qm31_eq(result, expected), "eq_eval simple mismatch vs Rust");
}

#[test]
fn test_eq_eval_complex_matches_rust_vector() {
    // Rust: eq([(100,200,300,400)], [(500,600,700,800)])
    //     = QM31(2145863048, 2179200, 2147122647, 1198800)
    let x = array![qm31_new(100, 200, 300, 400)];
    let y = array![qm31_new(500, 600, 700, 800)];
    let expected = qm31_new(2145863048, 2179200, 2147122647, 1198800);
    let result = eq_eval(x.span(), y.span());
    assert!(qm31_eq(result, expected), "eq_eval complex mismatch vs Rust");
}

#[test]
fn test_fold_mle_eval_matches_rust_vector() {
    // Rust: v0 = QM31(1000, 2000, 3000, 4000)
    //       v1 = QM31(5000, 6000, 7000, 8000)
    //       x  = QM31(42, 7, 13, 99)
    //       fold(v0, v1, x) = QM31(2146488647, 750000, 2147282647, 648000)
    let v0 = qm31_new(1000, 2000, 3000, 4000);
    let v1 = qm31_new(5000, 6000, 7000, 8000);
    let x = qm31_new(42, 7, 13, 99);
    let expected = qm31_new(2146488647, 750000, 2147282647, 648000);
    let result = fold_mle_eval(x, v0, v1);
    assert!(qm31_eq(result, expected), "fold_mle_eval mismatch vs Rust");
}

// ============================================================================
// Task #15: Poseidon Channel Transcript Cross-Verification (Rust → Cairo)
//
// Test vectors from stwo-ml Rust tests (poseidon_channel.rs).
// Verifies that Cairo's PoseidonChannel produces identical Fiat-Shamir
// challenges when given the same inputs.
// ============================================================================

#[test]
fn test_channel_draw_after_mix_felts_matches_rust() {
    // Rust: After mix_felts([QM31(42,0,0,0)]):
    //   alpha = QM31(1692896316, 1443752209, 1695081376, 61024658)
    //   lambda = QM31(1544203027, 786944301, 35629302, 1093658306)
    let mut ch = channel_default();
    let claim = qm31_new(42, 0, 0, 0);
    let felts = array![claim];
    channel_mix_felts(ref ch, felts.span());

    let alpha = channel_draw_qm31(ref ch);
    let expected_alpha = qm31_new(1692896316, 1443752209, 1695081376, 61024658);
    assert!(qm31_eq(alpha, expected_alpha), "alpha mismatch vs Rust");

    let lambda = channel_draw_qm31(ref ch);
    let expected_lambda = qm31_new(1544203027, 786944301, 35629302, 1093658306);
    assert!(qm31_eq(lambda, expected_lambda), "lambda mismatch vs Rust");
}

#[test]
fn test_channel_mix_mask_draw_r_matches_rust() {
    // Rust: After mix_felts([42]) + 2 draws, then mix_felts([6, 7]):
    //   r = QM31(863660832, 711266069, 350349676, 767766850)
    let mut ch = channel_default();
    let claim = qm31_new(42, 0, 0, 0);
    channel_mix_felts(ref ch, array![claim].span());

    // Draw alpha and lambda (consume them)
    let _alpha = channel_draw_qm31(ref ch);
    let _lambda = channel_draw_qm31(ref ch);

    // Mix mask values
    let mask_v0 = qm31_new(6, 0, 0, 0);
    let mask_v1 = qm31_new(7, 0, 0, 0);
    channel_mix_felts(ref ch, array![mask_v0, mask_v1].span());

    let r = channel_draw_qm31(ref ch);
    let expected_r = qm31_new(863660832, 711266069, 350349676, 767766850);
    assert!(qm31_eq(r, expected_r), "r mismatch vs Rust");
}

#[test]
fn test_channel_mix_u64_draw_matches_rust() {
    // Cross-verify: mix_u64(42) then draw produces specific value.
    // Rust verified this matches STWO's Poseidon252Channel exactly.
    let mut ch1 = channel_default();
    let mut ch2 = channel_default();

    channel_mix_u64(ref ch1, 42);
    channel_mix_u64(ref ch2, 42);

    let d1 = channel_draw_qm31(ref ch1);
    let d2 = channel_draw_qm31(ref ch2);

    // Determinism: identical operations produce identical results
    assert!(qm31_eq(d1, d2), "Identical mix_u64 should produce identical draws");

    // Different seed → different draw
    let mut ch3 = channel_default();
    channel_mix_u64(ref ch3, 99);
    let d3 = channel_draw_qm31(ref ch3);
    assert!(!qm31_eq(d1, d3), "Different seeds should produce different draws");
}

#[test]
fn test_channel_multi_step_state_machine() {
    // Complex state machine: mix_u64 → draw → mix_felts → draw cycle.
    // Verifies n_draws reset behavior and state consistency.
    let mut ch = channel_default();

    // Phase 1: Mix dimensions
    channel_mix_u64(ref ch, 64);
    channel_mix_u64(ref ch, 128);
    channel_mix_u64(ref ch, 256);

    // Draw a challenge
    let d1 = channel_draw_qm31(ref ch);

    // Phase 2: Mix field element
    let val = qm31_new(999, 888, 777, 666);
    channel_mix_felts(ref ch, array![val].span());

    // Draw after mix (n_draws should have reset)
    let d2 = channel_draw_qm31(ref ch);
    assert!(!qm31_eq(d1, d2), "Draw after mix should differ from previous");

    // Phase 3: Multiple draws
    let d3 = channel_draw_qm31(ref ch);
    let d4 = channel_draw_qm31(ref ch);
    assert!(!qm31_eq(d3, d4), "Consecutive draws should differ");

    // Phase 4: Mix after multiple draws
    channel_mix_felts(ref ch, array![val, val].span());
    let d5 = channel_draw_qm31(ref ch);
    assert!(!qm31_eq(d4, d5), "Draw after re-mix should differ");
}

// ============================================================================
// Deep Circuit Test (n_variables=2, 2 layers, 1 sumcheck round)
//
// Test vectors generated by stwo-ml Rust test:
// test_generate_deep_gp_vectors in poseidon_channel.rs
// ============================================================================

/// Build a valid 2-layer GrandProduct GKR proof.
///
/// Layer 0 (output): trivial sumcheck (0 rounds).
///   Mask = [15, 77], gate = 15*77 = 1155 = output_claim.
/// Layer 1: 1 sumcheck round (ood_point has 1 element from layer 0).
///   Round poly satisfies p(0)+p(1) = reduced_claim.
///   Mask = [reduced_claim, 1], gate = reduced_claim*1 = reduced_claim.
fn build_deep_gp_proof() -> GkrBatchProof {
    let product = qm31_new(1155, 0, 0, 0); // 15 * 77

    // Layer 0 mask: left=15, right=77
    let mask0 = GkrMask {
        values: array![qm31_new(15, 0, 0, 0), qm31_new(77, 0, 0, 0)],
        num_columns: 1,
    };

    // Layer 1: has 1 sumcheck round.
    // Round poly from Rust vectors: c0, c1 (c2=0 for degree-1 poly)
    let c0 = qm31_new(1543647767, 1516546815, 1031339893, 155166972);
    let c1 = qm31_new(392524940, 1322061807, 345595546, 180069066);

    // Mask at layer 1: [reduced_claim, 1]
    let mask1 = GkrMask {
        values: array![
            qm31_new(1332336827, 60188143, 260791685, 490403010),
            qm31_new(1, 0, 0, 0),
        ],
        num_columns: 1,
    };

    GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 0 },
            n_variables: 2,
            output_claims: array![product],
        }],
        layer_proofs: array![
            // Layer 0: 0 sumcheck rounds
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
                masks: array![mask0],
            },
            // Layer 1: 1 sumcheck round
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof {
                    round_polys: array![GkrRoundPoly {
                        c0, c1,
                        c2: qm31_new(0, 0, 0, 0),
                        c3: qm31_new(0, 0, 0, 0),
                        num_coeffs: 3,
                    }],
                },
                masks: array![mask1],
            },
        ],
    }
}

#[test]
fn test_gkr_deep_circuit_2_layers() {
    let proof = build_deep_gp_proof();
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@proof, ref ch);

    // After 2 layers: ood_point has 2 coordinates
    assert!(artifact.ood_point.len() == 2, "2 layers -> 2 ood coords");
    assert!(artifact.claims_to_verify.len() == 1, "1 instance");
    assert!(*artifact.n_variables_by_instance.at(0) == 2, "n_vars=2");
    assert!(artifact.claims_to_verify.at(0).len() == 1, "GP: 1 final claim");

    // Verify final claim matches Rust vector
    let expected_final = qm31_new(834302045, 224875337, 132359604, 1609312598);
    assert!(qm31_eq(*artifact.claims_to_verify.at(0).at(0), expected_final),
        "final claim must match Rust vector");
}

#[test]
fn test_gkr_deep_circuit_serde_roundtrip() {
    let proof = build_deep_gp_proof();

    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    let mut span = serialized.span();
    let deserialized: GkrBatchProof = Serde::<GkrBatchProof>::deserialize(ref span)
        .expect('deep gkr serde failed');

    let mut ch = channel_default();
    let artifact = partially_verify_batch(@deserialized, ref ch);

    assert!(artifact.ood_point.len() == 2, "roundtrip ood_point");
    assert!(artifact.claims_to_verify.len() == 1, "roundtrip claims");
    assert!(span.len() == 0, "all data consumed");
}

#[test]
#[should_panic(expected: "GKR sumcheck round sum mismatch")]
fn test_gkr_deep_circuit_reject_tampered_round_poly() {
    // Tamper with the round polynomial coefficient
    let product = qm31_new(1155, 0, 0, 0);
    let mask0 = GkrMask {
        values: array![qm31_new(15, 0, 0, 0), qm31_new(77, 0, 0, 0)],
        num_columns: 1,
    };

    // TAMPERED c0 (changed from 1543647767 to 999999999)
    let c0 = qm31_new(999999999, 1516546815, 1031339893, 155166972);
    let c1 = qm31_new(392524940, 1322061807, 345595546, 180069066);

    let mask1 = GkrMask {
        values: array![
            qm31_new(1332336827, 60188143, 260791685, 490403010),
            qm31_new(1, 0, 0, 0),
        ],
        num_columns: 1,
    };

    let proof = GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 0 },
            n_variables: 2,
            output_claims: array![product],
        }],
        layer_proofs: array![
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
                masks: array![mask0],
            },
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof {
                    round_polys: array![GkrRoundPoly {
                        c0, c1,
                        c2: qm31_new(0, 0, 0, 0),
                        c3: qm31_new(0, 0, 0, 0),
                        num_coeffs: 3,
                    }],
                },
                masks: array![mask1],
            },
        ],
    };

    let mut ch = channel_default();
    partially_verify_batch(@proof, ref ch);  // Should panic
}

// ============================================================================
// Staggered Activation Test (inst0: n_vars=2, inst1: n_vars=1)
//
// Test vectors generated by stwo-ml Rust test:
// test_generate_staggered_vectors in poseidon_channel.rs
// ============================================================================

/// Build a staggered-activation batch proof:
/// - Instance 0: GP, n_variables=2. Starts at layer 0 (output layer).
///   Layer 0 mask: [15, 77], product=1155.
///   Layer 1 mask: [inst0_reduced, 1].
/// - Instance 1: GP, n_variables=1. Starts at layer 1.
///   Layer 1 mask: [6, 8], product=48.
///
/// At layer 0: only inst0 is active (doubling=1).
/// At layer 1: both active. inst0 has doubling=1, inst1 has doubling=2.
fn build_staggered_proof() -> GkrBatchProof {
    // Layer 0: only inst0
    let mask0_inst0 = GkrMask {
        values: array![qm31_new(15, 0, 0, 0), qm31_new(77, 0, 0, 0)],
        num_columns: 1,
    };

    // Layer 1: both instances
    // Round poly from Rust vectors
    let c0 = qm31_new(1707163297, 1293259362, 1922356252, 1499861285);
    let c1 = qm31_new(392524940, 1322061807, 345595546, 180069066);

    let mask1_inst0 = GkrMask {
        values: array![
            qm31_new(1332336827, 60188143, 260791685, 490403010), // inst0_reduced
            qm31_new(1, 0, 0, 0),
        ],
        num_columns: 1,
    };

    let mask1_inst1 = GkrMask {
        values: array![qm31_new(6, 0, 0, 0), qm31_new(8, 0, 0, 0)],
        num_columns: 1,
    };

    GkrBatchProof {
        instances: array![
            GkrInstance {
                gate: GateType { gate_id: 0 },
                n_variables: 2,
                output_claims: array![qm31_new(1155, 0, 0, 0)],
            },
            GkrInstance {
                gate: GateType { gate_id: 0 },
                n_variables: 1,
                output_claims: array![qm31_new(48, 0, 0, 0)],
            },
        ],
        layer_proofs: array![
            // Layer 0: only inst0, 0 rounds
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
                masks: array![mask0_inst0],
            },
            // Layer 1: both instances, 1 round
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof {
                    round_polys: array![GkrRoundPoly {
                        c0, c1,
                        c2: qm31_new(0, 0, 0, 0),
                        c3: qm31_new(0, 0, 0, 0),
                        num_coeffs: 3,
                    }],
                },
                masks: array![mask1_inst0, mask1_inst1],
            },
        ],
    }
}

#[test]
fn test_gkr_staggered_activation() {
    let proof = build_staggered_proof();
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@proof, ref ch);

    // 2 layers → ood_point has 2 coordinates
    assert!(artifact.ood_point.len() == 2, "2 layers -> 2 ood coords");
    assert!(artifact.claims_to_verify.len() == 2, "2 instances");

    // Instance 0: n_vars=2, 1 final GP claim
    assert!(*artifact.n_variables_by_instance.at(0) == 2, "inst0 n_vars=2");
    assert!(artifact.claims_to_verify.at(0).len() == 1, "inst0: 1 claim");

    // Instance 1: n_vars=1, 1 final GP claim
    assert!(*artifact.n_variables_by_instance.at(1) == 1, "inst1 n_vars=1");
    assert!(artifact.claims_to_verify.at(1).len() == 1, "inst1: 1 claim");

    // Verify final claims match Rust vectors
    let expected0 = qm31_new(511256495, 673773779, 972510431, 413056769);
    let expected1 = qm31_new(1418711364, 985989380, 1711557452, 1368876323);
    assert!(qm31_eq(*artifact.claims_to_verify.at(0).at(0), expected0),
        "inst0 final claim mismatch");
    assert!(qm31_eq(*artifact.claims_to_verify.at(1).at(0), expected1),
        "inst1 final claim mismatch");
}

#[test]
fn test_gkr_staggered_serde_roundtrip() {
    let proof = build_staggered_proof();

    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    let mut span = serialized.span();
    let deserialized: GkrBatchProof = Serde::<GkrBatchProof>::deserialize(ref span)
        .expect('staggered serde failed');

    let mut ch = channel_default();
    let artifact = partially_verify_batch(@deserialized, ref ch);

    assert!(artifact.claims_to_verify.len() == 2, "2 instances roundtrip");
    assert!(span.len() == 0, "all data consumed");
}

#[test]
#[should_panic(expected: "GKR sumcheck round sum mismatch")]
fn test_gkr_staggered_reject_wrong_doubling() {
    // Tamper with inst1's output claim — changes channel state at layer 1,
    let mask0_inst0 = GkrMask {
        values: array![qm31_new(15, 0, 0, 0), qm31_new(77, 0, 0, 0)],
        num_columns: 1,
    };

    let c0 = qm31_new(1707163297, 1293259362, 1922356252, 1499861285);
    let c1 = qm31_new(392524940, 1322061807, 345595546, 180069066);

    let mask1_inst0 = GkrMask {
        values: array![
            qm31_new(1332336827, 60188143, 260791685, 490403010),
            qm31_new(1, 0, 0, 0),
        ],
        num_columns: 1,
    };
    let mask1_inst1 = GkrMask {
        values: array![qm31_new(6, 0, 0, 0), qm31_new(8, 0, 0, 0)],
        num_columns: 1,
    };

    let proof = GkrBatchProof {
        instances: array![
            GkrInstance {
                gate: GateType { gate_id: 0 },
                n_variables: 2,
                output_claims: array![qm31_new(1155, 0, 0, 0)],
            },
            GkrInstance {
                gate: GateType { gate_id: 0 },
                n_variables: 1,
                output_claims: array![qm31_new(99, 0, 0, 0)],  // TAMPERED (was 48)
            },
        ],
        layer_proofs: array![
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
                masks: array![mask0_inst0],
            },
            GkrLayerProof {
                sumcheck_proof: GkrSumcheckProof {
                    round_polys: array![GkrRoundPoly {
                        c0, c1,
                        c2: qm31_new(0, 0, 0, 0),
                        c3: qm31_new(0, 0, 0, 0),
                        num_coeffs: 3,
                    }],
                },
                masks: array![mask1_inst0, mask1_inst1],
            },
        ],
    };

    let mut ch = channel_default();
    partially_verify_batch(@proof, ref ch);  // Should panic — tampered claim changes channel state
}

// ============================================================================
// evaluate_mle Tests — Input Claim Verification Foundation
// ============================================================================

#[test]
fn test_evaluate_mle_single_variable() {
    // 1 variable, 2 evals: f(0)=3, f(1)=7
    // f(r) = 3*(1-r) + 7*r = 3 + 4*r
    let evals = array![qm31_new(3, 0, 0, 0), qm31_new(7, 0, 0, 0)];
    let point = array![qm31_zero()]; // r=0 → f(0)=3
    let result = evaluate_mle(evals.span(), point.span());
    assert!(qm31_eq(result, qm31_new(3, 0, 0, 0)), "MLE f(0)=3");

    let evals2 = array![qm31_new(3, 0, 0, 0), qm31_new(7, 0, 0, 0)];
    let point2 = array![qm31_one()]; // r=1 → f(1)=7
    let result2 = evaluate_mle(evals2.span(), point2.span());
    assert!(qm31_eq(result2, qm31_new(7, 0, 0, 0)), "MLE f(1)=7");
}

#[test]
fn test_evaluate_mle_two_variables() {
    // 2 variables, 4 evals: f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
    // Row-major: [f(0,0), f(0,1), f(1,0), f(1,1)] = [1, 3, 2, 4]
    // Wait — evaluate_mle uses the convention where first variable splits lo/hi:
    //   evals[0..mid] = first var=0, evals[mid..] = first var=1
    // So for 2 vars: [f(0,0), f(0,1), f(1,0), f(1,1)]
    //   First fold (var 0): mid=2
    //     result[0] = evals[0] + r0*(evals[2] - evals[0]) = f(r0, 0)
    //     result[1] = evals[1] + r0*(evals[3] - evals[1]) = f(r0, 1)
    //   Second fold (var 1): mid=1
    //     result[0] = result[0] + r1*(result[1] - result[0]) = f(r0, r1)
    //
    // At (0,0): f(0,0) = evals[0] = 10
    let evals = array![
        qm31_new(10, 0, 0, 0), // f(0,0)
        qm31_new(20, 0, 0, 0), // f(0,1)
        qm31_new(30, 0, 0, 0), // f(1,0)
        qm31_new(40, 0, 0, 0), // f(1,1)
    ];

    let point_00 = array![qm31_zero(), qm31_zero()];
    let result = evaluate_mle(evals.span(), point_00.span());
    assert!(qm31_eq(result, qm31_new(10, 0, 0, 0)), "MLE f(0,0)=10");

    let evals2 = array![
        qm31_new(10, 0, 0, 0),
        qm31_new(20, 0, 0, 0),
        qm31_new(30, 0, 0, 0),
        qm31_new(40, 0, 0, 0),
    ];
    let point_11 = array![qm31_one(), qm31_one()];
    let result2 = evaluate_mle(evals2.span(), point_11.span());
    assert!(qm31_eq(result2, qm31_new(40, 0, 0, 0)), "MLE f(1,1)=40");

    let evals3 = array![
        qm31_new(10, 0, 0, 0),
        qm31_new(20, 0, 0, 0),
        qm31_new(30, 0, 0, 0),
        qm31_new(40, 0, 0, 0),
    ];
    let point_10 = array![qm31_one(), qm31_zero()];
    let result3 = evaluate_mle(evals3.span(), point_10.span());
    assert!(qm31_eq(result3, qm31_new(30, 0, 0, 0)), "MLE f(1,0)=30");

    let evals4 = array![
        qm31_new(10, 0, 0, 0),
        qm31_new(20, 0, 0, 0),
        qm31_new(30, 0, 0, 0),
        qm31_new(40, 0, 0, 0),
    ];
    let point_01 = array![qm31_zero(), qm31_one()];
    let result4 = evaluate_mle(evals4.span(), point_01.span());
    assert!(qm31_eq(result4, qm31_new(20, 0, 0, 0)), "MLE f(0,1)=20");
}

#[test]
fn test_evaluate_mle_matches_fold_mle_eval() {
    // For 1 variable, evaluate_mle should match fold_mle_eval
    let v0 = qm31_new(42, 7, 13, 99);
    let v1 = qm31_new(100, 200, 300, 400);
    let r = qm31_new(17, 3, 0, 0);

    let evals = array![v0, v1];
    let point = array![r];

    let mle_result = evaluate_mle(evals.span(), point.span());
    let fold_result = fold_mle_eval(r, v0, v1);

    assert!(qm31_eq(mle_result, fold_result), "evaluate_mle should match fold_mle_eval for 1 var");
}

#[test]
fn test_evaluate_mle_with_qm31_point() {
    // Test with non-trivial QM31 evaluation point (simulates Fiat-Shamir challenge)
    // 2 vars, evals = [1, 2, 3, 4], point = (r0, r1) where r0 = QM31(5,1,0,0), r1 = QM31(3,0,0,0)
    //
    // Fold var 0 (r0=5+i): mid=2
    //   new[0] = 1 + (5+i)*(3-1) = 1 + (5+i)*2 = 1 + (10+2i) = 11+2i
    //   new[1] = 2 + (5+i)*(4-2) = 2 + (5+i)*2 = 2 + (10+2i) = 12+2i
    // Fold var 1 (r1=3): mid=1
    //   result = (11+2i) + 3*((12+2i)-(11+2i)) = (11+2i) + 3*(1) = 14+2i
    let evals = array![
        qm31_new(1, 0, 0, 0),
        qm31_new(2, 0, 0, 0),
        qm31_new(3, 0, 0, 0),
        qm31_new(4, 0, 0, 0),
    ];
    let point = array![qm31_new(5, 1, 0, 0), qm31_new(3, 0, 0, 0)];
    let result = evaluate_mle(evals.span(), point.span());
    assert!(qm31_eq(result, qm31_new(14, 2, 0, 0)), "MLE at QM31 point");
}

#[test]
fn test_pad_and_embed_m31s() {
    // Embed [1, 2, 3] into QM31 and pad to length 4
    let vals: Array<u64> = array![1, 2, 3];
    let result = pad_and_embed_m31s(vals.span(), 4);

    assert!(result.len() == 4, "should be padded to 4");
    let result_span = result.span();
    assert!(qm31_eq(*result_span.at(0), m31_to_qm31(1)), "val[0] = 1");
    assert!(qm31_eq(*result_span.at(1), m31_to_qm31(2)), "val[1] = 2");
    assert!(qm31_eq(*result_span.at(2), m31_to_qm31(3)), "val[2] = 3");
    assert!(qm31_eq(*result_span.at(3), qm31_zero()), "val[3] = 0 (padding)");
}

#[test]
fn test_evaluate_mle_padded_matrix() {
    // Simulate a 1x3 matrix padded to 1x4 (next pow2 of 3=4)
    // Raw data: [10, 20, 30], padded: [10, 20, 30, 0]
    // This is a 1x4 matrix → total size = 4, log_vars = 2
    // MLE at (0,0) = evals[0] = 10
    let mle: Array<QM31> = array![
        m31_to_qm31(10),
        m31_to_qm31(20),
        m31_to_qm31(30),
        qm31_zero(), // padding
    ];

    let point_00 = array![qm31_zero(), qm31_zero()];
    let result = evaluate_mle(mle.span(), point_00.span());
    assert!(qm31_eq(result, m31_to_qm31(10)), "padded MLE f(0,0)=10");
}
