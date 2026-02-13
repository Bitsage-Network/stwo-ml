// Tests for batched matmul sumcheck verification.
//
// Strategy: construct valid proofs by replaying the exact prover transcript
// (channel operations from gpu_sumcheck.rs prove_matmul_batch_onchain_gpu),
// then verify they pass. Tampered proofs must fail.

use elo_cairo_verifier::field::{
    qm31_new, qm31_zero, qm31_one, qm31_add, qm31_mul, qm31_eq,
    poly_eval_degree2, pack_qm31_to_felt,
};
use elo_cairo_verifier::channel::{
    channel_default, channel_mix_u64, channel_mix_felt,
    channel_draw_qm31, channel_mix_poly_coeffs,
};
use elo_cairo_verifier::types::{RoundPoly, BatchedMatMulEntry, BatchedMatMulProof};
use elo_cairo_verifier::sumcheck::{verify_batched_sumcheck, check_round_sum};

// ============================================================================
// Helpers
// ============================================================================

/// Derive lambda from batch metadata by replaying the prover's Fiat-Shamir.
/// Returns (lambda, channel_state_after_lambda).
fn derive_lambda(
    entries: Span<BatchedMatMulEntry>, k: u32,
) -> elo_cairo_verifier::channel::PoseidonChannel {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, entries.len().into());
    channel_mix_u64(ref ch, k.into());

    let mut i: u32 = 0;
    loop {
        if i >= entries.len() {
            break;
        }
        let e = *entries.at(i);
        channel_mix_felt(ref ch, pack_qm31_to_felt(e.claimed_sum));
        channel_mix_felt(ref ch, e.a_commitment);
        channel_mix_felt(ref ch, e.b_commitment);
        i += 1;
    };
    ch
}

/// Build a valid single-entry batch proof for k=2 (1 round).
///
/// claimed_sum = 76 (arbitrary, represents s(0)+s(1)=21+55).
/// Round poly chosen so p(0)+p(1)=claimed_sum, then final evals satisfy
/// a*b = p(challenge) where challenge is Fiat-Shamir derived.
fn build_valid_single_entry_proof() -> BatchedMatMulProof {
    let claimed_sum = qm31_new(76, 0, 0, 0);
    let a_commitment: felt252 = 0x1234;
    let b_commitment: felt252 = 0x5678;

    let entry = BatchedMatMulEntry {
        node_id: 0,
        m: 2,
        n: 2,
        claimed_sum,
        // Will be set below after computing challenge
        final_a_eval: qm31_zero(), // placeholder
        final_b_eval: qm31_zero(), // placeholder
        a_commitment,
        b_commitment,
    };

    let entries_arr: Array<BatchedMatMulEntry> = array![entry];

    // Step 1: derive lambda via Fiat-Shamir
    let mut ch = derive_lambda(entries_arr.span(), 2);
    let lambda = channel_draw_qm31(ref ch);

    // combined_claimed_sum = λ^0 * claimed_sum = claimed_sum (single entry)
    let combined = claimed_sum;

    // Step 2: construct round poly
    // Need 2*c0 + c1 + c2 = combined_claimed_sum = 76
    // Pick c0 = 21, c1 = 34, c2 = 0 → 42 + 34 + 0 = 76 ✓
    let c0 = qm31_new(21, 0, 0, 0);
    let c1 = qm31_new(34, 0, 0, 0);
    let c2 = qm31_zero();

    // Step 3: derive challenge
    channel_mix_poly_coeffs(ref ch, c0, c1, c2);
    let challenge = channel_draw_qm31(ref ch);

    // Step 4: compute final sum = p(challenge)
    let final_sum = poly_eval_degree2(c0, c1, c2, challenge);

    // Step 5: set final evals so a * b = final_sum
    // Use a_eval = final_sum, b_eval = 1
    let final_a_eval = final_sum;
    let final_b_eval = qm31_one();

    // Rebuild entry with correct final evals
    let entry_final = BatchedMatMulEntry {
        node_id: 0,
        m: 2,
        n: 2,
        claimed_sum,
        final_a_eval,
        final_b_eval,
        a_commitment,
        b_commitment,
    };

    BatchedMatMulProof {
        k: 2,
        num_rounds: 1,
        lambda,
        combined_claimed_sum: combined,
        round_polys: array![RoundPoly { c0, c1, c2 }],
        entries: array![entry_final],
    }
}

/// Build a valid two-entry batch proof for k=2 (1 round).
fn build_valid_two_entry_proof() -> BatchedMatMulProof {
    let claimed_sum_0 = qm31_new(76, 0, 0, 0);
    let claimed_sum_1 = qm31_new(44, 0, 0, 0);
    let a_commit_0: felt252 = 0xAABB;
    let b_commit_0: felt252 = 0xCCDD;
    let a_commit_1: felt252 = 0xEEFF;
    let b_commit_1: felt252 = 0x1122;

    let entry0 = BatchedMatMulEntry {
        node_id: 0,
        m: 2,
        n: 2,
        claimed_sum: claimed_sum_0,
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: a_commit_0,
        b_commitment: b_commit_0,
    };
    let entry1 = BatchedMatMulEntry {
        node_id: 1,
        m: 4,
        n: 4,
        claimed_sum: claimed_sum_1,
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: a_commit_1,
        b_commitment: b_commit_1,
    };

    let entries_arr: Array<BatchedMatMulEntry> = array![entry0, entry1];

    // Derive lambda
    let mut ch = derive_lambda(entries_arr.span(), 2);
    let lambda = channel_draw_qm31(ref ch);

    // combined_claimed_sum = 1*76 + λ*44
    let combined = qm31_add(claimed_sum_0, qm31_mul(lambda, claimed_sum_1));

    // Round poly: need 2*c0 + c1 + c2 = combined
    // Pick c0 = combined/3 approximately... easier: let c2=0, c0 arbitrary
    // p(0) = c0, p(1) = c0 + c1. Need p(0) + p(1) = 2c0 + c1 = combined.
    // Let c0 = qm31_zero, c1 = combined, c2 = 0 → 0 + combined = combined ✓
    // But then p(0)=0 and p(1)=combined. p(0)+p(1)=combined ✓
    let c0 = qm31_zero();
    let c1 = combined;
    let c2 = qm31_zero();

    // Derive challenge
    channel_mix_poly_coeffs(ref ch, c0, c1, c2);
    let challenge = channel_draw_qm31(ref ch);

    // final_sum = p(challenge) = 0 + combined * challenge + 0 = combined * challenge
    let final_sum = poly_eval_degree2(c0, c1, c2, challenge);

    // Need: Σ λ^i * a_i * b_i = final_sum
    // = 1 * a0 * b0 + λ * a1 * b1 = final_sum
    // Set a0*b0 = final_sum, a1*b1 = 0 (simplest)
    // → a0 = final_sum, b0 = 1, a1 = 0, b1 = anything
    let entry0_final = BatchedMatMulEntry {
        node_id: 0,
        m: 2,
        n: 2,
        claimed_sum: claimed_sum_0,
        final_a_eval: final_sum,
        final_b_eval: qm31_one(),
        a_commitment: a_commit_0,
        b_commitment: b_commit_0,
    };
    let entry1_final = BatchedMatMulEntry {
        node_id: 1,
        m: 4,
        n: 4,
        claimed_sum: claimed_sum_1,
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_one(),
        a_commitment: a_commit_1,
        b_commitment: b_commit_1,
    };

    BatchedMatMulProof {
        k: 2,
        num_rounds: 1,
        lambda,
        combined_claimed_sum: combined,
        round_polys: array![RoundPoly { c0, c1, c2 }],
        entries: array![entry0_final, entry1_final],
    }
}

// ============================================================================
// Valid Proof Tests
// ============================================================================

#[test]
fn test_batch_verify_single_entry_valid() {
    let proof = build_valid_single_entry_proof();
    let (is_valid, _proof_hash) = verify_batched_sumcheck(@proof);
    assert!(is_valid, "single-entry batch should verify");
}

#[test]
fn test_batch_verify_two_entry_valid() {
    let proof = build_valid_two_entry_proof();
    let (is_valid, _proof_hash) = verify_batched_sumcheck(@proof);
    assert!(is_valid, "two-entry batch should verify");
}

#[test]
fn test_batch_verify_proof_hash_deterministic() {
    let proof = build_valid_single_entry_proof();
    let (ok1, hash1) = verify_batched_sumcheck(@proof);

    let proof2 = build_valid_single_entry_proof();
    let (ok2, hash2) = verify_batched_sumcheck(@proof2);

    assert!(ok1 && ok2, "both should verify");
    assert!(hash1 == hash2, "same proof should produce same hash");
}

#[test]
fn test_batch_verify_proof_hash_nonzero() {
    let proof = build_valid_single_entry_proof();
    let (is_valid, proof_hash) = verify_batched_sumcheck(@proof);
    assert!(is_valid, "should verify");
    assert!(proof_hash != 0, "proof hash should be non-zero");
}

// ============================================================================
// Lambda Derivation Tests
// ============================================================================

#[test]
fn test_batch_lambda_derivation_deterministic() {
    let entry = BatchedMatMulEntry {
        node_id: 0,
        m: 4,
        n: 4,
        claimed_sum: qm31_new(100, 0, 0, 0),
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0xABCD,
        b_commitment: 0xEF01,
    };
    let entries: Array<BatchedMatMulEntry> = array![entry];

    let mut ch1 = derive_lambda(entries.span(), 8);
    let lambda1 = channel_draw_qm31(ref ch1);

    // Rebuild and re-derive
    let entry2 = BatchedMatMulEntry {
        node_id: 0,
        m: 4,
        n: 4,
        claimed_sum: qm31_new(100, 0, 0, 0),
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0xABCD,
        b_commitment: 0xEF01,
    };
    let entries2: Array<BatchedMatMulEntry> = array![entry2];

    let mut ch2 = derive_lambda(entries2.span(), 8);
    let lambda2 = channel_draw_qm31(ref ch2);

    assert!(qm31_eq(lambda1, lambda2), "same inputs should produce same lambda");
}

#[test]
fn test_batch_lambda_changes_with_different_entries() {
    let entry_a = BatchedMatMulEntry {
        node_id: 0, m: 2, n: 2,
        claimed_sum: qm31_new(10, 0, 0, 0),
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0x1111,
        b_commitment: 0x2222,
    };

    let entry_b = BatchedMatMulEntry {
        node_id: 0, m: 2, n: 2,
        claimed_sum: qm31_new(20, 0, 0, 0), // different claimed_sum
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment: 0x1111,
        b_commitment: 0x2222,
    };

    let mut ch_a = derive_lambda(array![entry_a].span(), 2);
    let lambda_a = channel_draw_qm31(ref ch_a);

    let mut ch_b = derive_lambda(array![entry_b].span(), 2);
    let lambda_b = channel_draw_qm31(ref ch_b);

    assert!(!qm31_eq(lambda_a, lambda_b), "different entries should produce different lambda");
}

// ============================================================================
// Negative Tests — Tampered Proofs
// ============================================================================

#[test]
fn test_batch_verify_empty_batch_fails() {
    let proof = BatchedMatMulProof {
        k: 2,
        num_rounds: 1,
        lambda: qm31_one(),
        combined_claimed_sum: qm31_zero(),
        round_polys: array![RoundPoly { c0: qm31_zero(), c1: qm31_zero(), c2: qm31_zero() }],
        entries: array![],
    };
    let (is_valid, reason) = verify_batched_sumcheck(@proof);
    assert!(!is_valid, "empty batch should fail");
    assert!(reason == 'EMPTY_BATCH', "should return EMPTY_BATCH reason");
}

#[test]
fn test_batch_verify_zero_rounds_fails() {
    let entry = BatchedMatMulEntry {
        node_id: 0, m: 2, n: 2,
        claimed_sum: qm31_one(),
        final_a_eval: qm31_one(),
        final_b_eval: qm31_one(),
        a_commitment: 0x1,
        b_commitment: 0x2,
    };
    let proof = BatchedMatMulProof {
        k: 1,
        num_rounds: 0,
        lambda: qm31_one(),
        combined_claimed_sum: qm31_one(),
        round_polys: array![],
        entries: array![entry],
    };
    let (is_valid, reason) = verify_batched_sumcheck(@proof);
    assert!(!is_valid, "zero rounds should fail");
    assert!(reason == 'ZERO_ROUNDS', "should return ZERO_ROUNDS reason");
}

#[test]
fn test_batch_verify_round_count_mismatch_fails() {
    let entry = BatchedMatMulEntry {
        node_id: 0, m: 2, n: 2,
        claimed_sum: qm31_one(),
        final_a_eval: qm31_one(),
        final_b_eval: qm31_one(),
        a_commitment: 0x1,
        b_commitment: 0x2,
    };
    let proof = BatchedMatMulProof {
        k: 4,
        num_rounds: 2, // says 2 rounds
        lambda: qm31_one(),
        combined_claimed_sum: qm31_one(),
        round_polys: array![
            RoundPoly { c0: qm31_zero(), c1: qm31_zero(), c2: qm31_zero() },
        ], // but only 1 polynomial
        entries: array![entry],
    };
    let (is_valid, reason) = verify_batched_sumcheck(@proof);
    assert!(!is_valid, "round count mismatch should fail");
    assert!(reason == 'ROUND_COUNT', "should return ROUND_COUNT reason");
}

#[test]
fn test_batch_verify_wrong_lambda_fails() {
    let mut proof = build_valid_single_entry_proof();
    // Tamper: use wrong lambda
    let tampered = BatchedMatMulProof {
        k: proof.k,
        num_rounds: proof.num_rounds,
        lambda: qm31_new(999, 888, 777, 666), // wrong lambda
        combined_claimed_sum: proof.combined_claimed_sum,
        round_polys: proof.round_polys,
        entries: proof.entries,
    };
    let (is_valid, reason) = verify_batched_sumcheck(@tampered);
    assert!(!is_valid, "wrong lambda should fail");
    assert!(reason == 'LAMBDA_MISMATCH', "should return LAMBDA_MISMATCH reason");
}

#[test]
fn test_batch_verify_wrong_combined_sum_fails() {
    let proof = build_valid_single_entry_proof();
    // Tamper: wrong combined_claimed_sum
    let tampered = BatchedMatMulProof {
        k: proof.k,
        num_rounds: proof.num_rounds,
        lambda: proof.lambda,
        combined_claimed_sum: qm31_new(999, 0, 0, 0), // wrong
        round_polys: proof.round_polys,
        entries: proof.entries,
    };
    let (is_valid, reason) = verify_batched_sumcheck(@tampered);
    assert!(!is_valid, "wrong combined sum should fail");
    assert!(reason == 'COMBINED_SUM', "should return COMBINED_SUM reason");
}

#[test]
fn test_batch_verify_tampered_round_poly_fails() {
    let proof = build_valid_single_entry_proof();
    // Tamper: modify round polynomial c0
    let tampered = BatchedMatMulProof {
        k: proof.k,
        num_rounds: proof.num_rounds,
        lambda: proof.lambda,
        combined_claimed_sum: proof.combined_claimed_sum,
        round_polys: array![
            RoundPoly {
                c0: qm31_new(999, 0, 0, 0), // tampered
                c1: qm31_new(34, 0, 0, 0),
                c2: qm31_zero(),
            },
        ],
        entries: proof.entries,
    };
    let (is_valid, reason) = verify_batched_sumcheck(@tampered);
    assert!(!is_valid, "tampered round poly should fail");
    assert!(reason == 'ROUND_FAIL', "should return ROUND_FAIL reason");
}

#[test]
fn test_batch_verify_tampered_final_eval_fails() {
    // Build valid proof but change final_a_eval
    let claimed_sum = qm31_new(76, 0, 0, 0);
    let a_commitment: felt252 = 0x1234;
    let b_commitment: felt252 = 0x5678;

    let entry = BatchedMatMulEntry {
        node_id: 0, m: 2, n: 2,
        claimed_sum,
        final_a_eval: qm31_zero(),
        final_b_eval: qm31_zero(),
        a_commitment,
        b_commitment,
    };

    let mut ch = derive_lambda(array![entry].span(), 2);
    let lambda = channel_draw_qm31(ref ch);

    let c0 = qm31_new(21, 0, 0, 0);
    let c1 = qm31_new(34, 0, 0, 0);
    let c2 = qm31_zero();

    channel_mix_poly_coeffs(ref ch, c0, c1, c2);
    let _challenge = channel_draw_qm31(ref ch);

    // Build with WRONG final evals (don't satisfy a*b = p(challenge))
    let entry_bad = BatchedMatMulEntry {
        node_id: 0, m: 2, n: 2,
        claimed_sum,
        final_a_eval: qm31_new(1, 0, 0, 0), // wrong
        final_b_eval: qm31_new(1, 0, 0, 0), // 1*1 = 1 ≠ p(challenge)
        a_commitment,
        b_commitment,
    };

    let proof = BatchedMatMulProof {
        k: 2,
        num_rounds: 1,
        lambda,
        combined_claimed_sum: claimed_sum,
        round_polys: array![RoundPoly { c0, c1, c2 }],
        entries: array![entry_bad],
    };

    let (is_valid, reason) = verify_batched_sumcheck(@proof);
    assert!(!is_valid, "tampered final eval should fail");
    assert!(reason == 'FINAL_EVAL', "should return FINAL_EVAL reason");
}

// ============================================================================
// Algebraic Property Tests
// ============================================================================

#[test]
fn test_batch_lambda_weighting_arithmetic() {
    // Verify: Σ λ^i * x_i with two entries
    let x0 = qm31_new(10, 0, 0, 0);
    let x1 = qm31_new(20, 0, 0, 0);
    let lambda = qm31_new(3, 0, 0, 0);

    // Σ = 1*10 + 3*20 = 10 + 60 = 70
    let mut sum = qm31_zero();
    let mut lp = qm31_one();
    sum = qm31_add(sum, qm31_mul(lp, x0));
    lp = qm31_mul(lp, lambda);
    sum = qm31_add(sum, qm31_mul(lp, x1));

    assert!(qm31_eq(sum, qm31_new(70, 0, 0, 0)), "10 + 3*20 = 70");
}

#[test]
fn test_batch_round_poly_consistency() {
    // For any c0, c1, c2:
    // p(0) + p(1) should equal 2*c0 + c1 + c2
    let c0 = qm31_new(5, 3, 7, 2);
    let c1 = qm31_new(11, 13, 17, 19);
    let c2 = qm31_new(23, 29, 31, 37);

    let p0 = poly_eval_degree2(c0, c1, c2, qm31_zero());
    let p1 = poly_eval_degree2(c0, c1, c2, qm31_one());
    let round_sum = qm31_add(p0, p1);

    // 2*c0 + c1 + c2
    let expected = qm31_add(qm31_add(c0, c0), qm31_add(c1, c2));

    assert!(qm31_eq(round_sum, expected), "p(0)+p(1) = 2c0+c1+c2");
}

#[test]
fn test_batch_round_sum_with_combined_poly() {
    // Simulate the batch sumcheck round check:
    // combined poly from 2 matmuls with lambda
    let lambda = qm31_new(7, 0, 0, 0);

    // Matmul 0: c0=10, c1=20, c2=5
    let c0_0 = qm31_new(10, 0, 0, 0);
    let c1_0 = qm31_new(20, 0, 0, 0);
    let c2_0 = qm31_new(5, 0, 0, 0);

    // Matmul 1: c0=3, c1=4, c2=1
    let c0_1 = qm31_new(3, 0, 0, 0);
    let c1_1 = qm31_new(4, 0, 0, 0);
    let c2_1 = qm31_new(1, 0, 0, 0);

    // Combined: c0 = 1*10 + 7*3 = 31, c1 = 1*20 + 7*4 = 48, c2 = 1*5 + 7*1 = 12
    let c0 = qm31_add(c0_0, qm31_mul(lambda, c0_1));
    let c1 = qm31_add(c1_0, qm31_mul(lambda, c1_1));
    let c2 = qm31_add(c2_0, qm31_mul(lambda, c2_1));

    assert!(qm31_eq(c0, qm31_new(31, 0, 0, 0)), "combined c0 = 31");
    assert!(qm31_eq(c1, qm31_new(48, 0, 0, 0)), "combined c1 = 48");
    assert!(qm31_eq(c2, qm31_new(12, 0, 0, 0)), "combined c2 = 12");

    // p(0) + p(1) = 2*31 + 48 + 12 = 62 + 48 + 12 = 122
    let poly = RoundPoly { c0, c1, c2 };
    let expected_sum = qm31_new(122, 0, 0, 0);
    assert!(check_round_sum(poly, expected_sum), "combined round sum = 122");
}

// ============================================================================
// Cross-Language Serde Deserialization Test
// ============================================================================
//
// Validates that the felt252 layout produced by Rust's
// serialize_batched_matmul_for_recursive() correctly deserializes
// into Cairo's BatchedMatMulProof via #[derive(Serde)].
//
// Layout (matches cairo_serde.rs):
//   k (1 felt) | num_rounds (1 felt) | lambda (4 felts) |
//   combined_claimed_sum (4 felts) |
//   round_polys.len (1 felt) | round_polys[i].c0/c1/c2 (12 felts each) |
//   entries.len (1 felt) | entries[i] fields (17 felts each)
//
// Entry layout: node_id(1) m(1) n(1) claimed_sum(4) final_a(4) final_b(4) a_commit(1) b_commit(1)

#[test]
fn test_serde_deserialization_matches_rust_layout() {
    // Construct a felt252 array matching the exact layout from cairo_serde.rs.
    // Proof: k=2, num_rounds=1, 1 entry, 1 round poly
    let mut calldata: Array<felt252> = array![
        // k
        2,
        // num_rounds
        1,
        // lambda: QM31(100, 200, 300, 400)
        100, 200, 300, 400,
        // combined_claimed_sum: QM31(76, 0, 0, 0)
        76, 0, 0, 0,
        // round_polys.len = 1
        1,
        // round_poly[0].c0 = QM31(21, 0, 0, 0)
        21, 0, 0, 0,
        // round_poly[0].c1 = QM31(34, 0, 0, 0)
        34, 0, 0, 0,
        // round_poly[0].c2 = QM31(0, 0, 0, 0)
        0, 0, 0, 0,
        // entries.len = 1
        1,
        // entry[0]: node_id=5, m=8, n=16
        5, 8, 16,
        // entry[0].claimed_sum = QM31(76, 0, 0, 0)
        76, 0, 0, 0,
        // entry[0].final_a_eval = QM31(99, 0, 0, 0)
        99, 0, 0, 0,
        // entry[0].final_b_eval = QM31(1, 0, 0, 0)
        1, 0, 0, 0,
        // entry[0].a_commitment, b_commitment
        0xDEAD, 0xBEEF,
    ];

    let mut span = calldata.span();
    let proof: BatchedMatMulProof = Serde::<BatchedMatMulProof>::deserialize(ref span)
        .expect('serde deser failed');

    assert!(proof.k == 2, "k should be 2");
    assert!(proof.num_rounds == 1, "num_rounds should be 1");
    assert!(qm31_eq(proof.lambda, qm31_new(100, 200, 300, 400)), "lambda mismatch");
    assert!(qm31_eq(proof.combined_claimed_sum, qm31_new(76, 0, 0, 0)), "combined_sum mismatch");
    assert!(proof.round_polys.len() == 1, "should have 1 round poly");

    let rp = *proof.round_polys.at(0);
    assert!(qm31_eq(rp.c0, qm31_new(21, 0, 0, 0)), "rp.c0 mismatch");
    assert!(qm31_eq(rp.c1, qm31_new(34, 0, 0, 0)), "rp.c1 mismatch");
    assert!(qm31_eq(rp.c2, qm31_zero()), "rp.c2 mismatch");

    assert!(proof.entries.len() == 1, "should have 1 entry");
    let e = *proof.entries.at(0);
    assert!(e.node_id == 5, "node_id mismatch");
    assert!(e.m == 8, "m mismatch");
    assert!(e.n == 16, "n mismatch");
    assert!(qm31_eq(e.claimed_sum, qm31_new(76, 0, 0, 0)), "claimed_sum mismatch");
    assert!(qm31_eq(e.final_a_eval, qm31_new(99, 0, 0, 0)), "final_a mismatch");
    assert!(qm31_eq(e.final_b_eval, qm31_new(1, 0, 0, 0)), "final_b mismatch");
    assert!(e.a_commitment == 0xDEAD, "a_commit mismatch");
    assert!(e.b_commitment == 0xBEEF, "b_commit mismatch");
    assert!(span.len() == 0, "all calldata should be consumed");
}

#[test]
fn test_serde_deserialization_two_entries_two_rounds() {
    // k=4 (2 rounds), 2 entries — tests multi-entry, multi-round layout
    let mut calldata: Array<felt252> = array![
        4, 2,  // k, num_rounds
        7, 11, 13, 17,  // lambda
        500, 100, 0, 0,  // combined_claimed_sum
        2,  // round_polys.len
        10, 0, 0, 0, 20, 0, 0, 0, 5, 0, 0, 0,  // round_poly[0]
        3, 1, 0, 0, 4, 2, 0, 0, 1, 0, 0, 0,    // round_poly[1]
        2,  // entries.len
        // entry[0]
        0, 4, 4,  // node_id, m, n
        300, 50, 0, 0,  // claimed_sum
        42, 0, 0, 0,    // final_a_eval
        7, 0, 0, 0,     // final_b_eval
        0xAAAA, 0xBBBB,  // commitments
        // entry[1]
        3, 8, 2,  // node_id, m, n
        200, 50, 0, 0,  // claimed_sum
        11, 0, 0, 0,    // final_a_eval
        3, 0, 0, 0,     // final_b_eval
        0xCCCC, 0xDDDD,  // commitments
    ];

    let mut span = calldata.span();
    let proof: BatchedMatMulProof = Serde::<BatchedMatMulProof>::deserialize(ref span)
        .expect('serde deser failed');

    assert!(proof.k == 4, "k should be 4");
    assert!(proof.num_rounds == 2, "num_rounds should be 2");
    assert!(qm31_eq(proof.lambda, qm31_new(7, 11, 13, 17)), "lambda mismatch");
    assert!(proof.round_polys.len() == 2, "should have 2 round polys");
    assert!(proof.entries.len() == 2, "should have 2 entries");

    let e0 = *proof.entries.at(0);
    assert!(e0.node_id == 0 && e0.m == 4 && e0.n == 4, "e0 dims");
    assert!(e0.a_commitment == 0xAAAA, "e0 a_commit");

    let e1 = *proof.entries.at(1);
    assert!(e1.node_id == 3 && e1.m == 8 && e1.n == 2, "e1 dims");
    assert!(qm31_eq(e1.claimed_sum, qm31_new(200, 50, 0, 0)), "e1 claimed_sum");
    assert!(e1.a_commitment == 0xCCCC, "e1 a_commit");
    assert!(span.len() == 0, "all calldata consumed");
}

// ============================================================================
// Multi-Round Batch Verification Tests (k=4, 2 rounds)
// ============================================================================

fn build_valid_two_round_proof() -> BatchedMatMulProof {
    let claimed_sum = qm31_new(100, 0, 0, 0);
    let a_commitment: felt252 = 0xAA11;
    let b_commitment: felt252 = 0xBB22;

    let entry = BatchedMatMulEntry {
        node_id: 0, m: 4, n: 4, claimed_sum,
        final_a_eval: qm31_zero(), final_b_eval: qm31_zero(),
        a_commitment, b_commitment,
    };

    let mut ch = derive_lambda(array![entry].span(), 4);
    let lambda = channel_draw_qm31(ref ch);
    let combined = claimed_sum;

    // Round 1: p1(0)+p1(1)=100. c0=30, c1=40, c2=0 → 30+(30+40)=100
    let c0_r1 = qm31_new(30, 0, 0, 0);
    let c1_r1 = qm31_new(40, 0, 0, 0);
    let c2_r1 = qm31_zero();

    channel_mix_poly_coeffs(ref ch, c0_r1, c1_r1, c2_r1);
    let challenge_1 = channel_draw_qm31(ref ch);
    let expected_sum_r2 = poly_eval_degree2(c0_r1, c1_r1, c2_r1, challenge_1);

    // Round 2: p2(0)+p2(1)=expected_sum_r2. c0=0, c1=expected_sum_r2, c2=0
    let c0_r2 = qm31_zero();
    let c1_r2 = expected_sum_r2;
    let c2_r2 = qm31_zero();

    channel_mix_poly_coeffs(ref ch, c0_r2, c1_r2, c2_r2);
    let challenge_2 = channel_draw_qm31(ref ch);
    let final_sum = poly_eval_degree2(c0_r2, c1_r2, c2_r2, challenge_2);

    let entry_final = BatchedMatMulEntry {
        node_id: 0, m: 4, n: 4, claimed_sum,
        final_a_eval: final_sum, final_b_eval: qm31_one(),
        a_commitment, b_commitment,
    };

    BatchedMatMulProof {
        k: 4, num_rounds: 2, lambda,
        combined_claimed_sum: combined,
        round_polys: array![
            RoundPoly { c0: c0_r1, c1: c1_r1, c2: c2_r1 },
            RoundPoly { c0: c0_r2, c1: c1_r2, c2: c2_r2 },
        ],
        entries: array![entry_final],
    }
}

fn build_valid_two_entry_two_round_proof() -> BatchedMatMulProof {
    let cs0 = qm31_new(80, 0, 0, 0);
    let cs1 = qm31_new(60, 0, 0, 0);
    let ac0: felt252 = 0x1111;
    let bc0: felt252 = 0x2222;
    let ac1: felt252 = 0x3333;
    let bc1: felt252 = 0x4444;

    let e0 = BatchedMatMulEntry {
        node_id: 0, m: 4, n: 4, claimed_sum: cs0,
        final_a_eval: qm31_zero(), final_b_eval: qm31_zero(),
        a_commitment: ac0, b_commitment: bc0,
    };
    let e1 = BatchedMatMulEntry {
        node_id: 1, m: 4, n: 8, claimed_sum: cs1,
        final_a_eval: qm31_zero(), final_b_eval: qm31_zero(),
        a_commitment: ac1, b_commitment: bc1,
    };

    let mut ch = derive_lambda(array![e0, e1].span(), 4);
    let lambda = channel_draw_qm31(ref ch);
    let combined = qm31_add(cs0, qm31_mul(lambda, cs1));

    let c0_r1 = qm31_zero();
    let c1_r1 = combined;
    let c2_r1 = qm31_zero();

    channel_mix_poly_coeffs(ref ch, c0_r1, c1_r1, c2_r1);
    let ch1 = channel_draw_qm31(ref ch);
    let sum_r2 = poly_eval_degree2(c0_r1, c1_r1, c2_r1, ch1);

    let c0_r2 = qm31_zero();
    let c1_r2 = sum_r2;
    let c2_r2 = qm31_zero();

    channel_mix_poly_coeffs(ref ch, c0_r2, c1_r2, c2_r2);
    let ch2 = channel_draw_qm31(ref ch);
    let final_sum = poly_eval_degree2(c0_r2, c1_r2, c2_r2, ch2);

    let e0f = BatchedMatMulEntry {
        node_id: 0, m: 4, n: 4, claimed_sum: cs0,
        final_a_eval: final_sum, final_b_eval: qm31_one(),
        a_commitment: ac0, b_commitment: bc0,
    };
    let e1f = BatchedMatMulEntry {
        node_id: 1, m: 4, n: 8, claimed_sum: cs1,
        final_a_eval: qm31_zero(), final_b_eval: qm31_one(),
        a_commitment: ac1, b_commitment: bc1,
    };

    BatchedMatMulProof {
        k: 4, num_rounds: 2, lambda,
        combined_claimed_sum: combined,
        round_polys: array![
            RoundPoly { c0: c0_r1, c1: c1_r1, c2: c2_r1 },
            RoundPoly { c0: c0_r2, c1: c1_r2, c2: c2_r2 },
        ],
        entries: array![e0f, e1f],
    }
}

#[test]
fn test_batch_verify_two_rounds_single_entry() {
    let proof = build_valid_two_round_proof();
    let (is_valid, proof_hash) = verify_batched_sumcheck(@proof);
    assert!(is_valid, "2-round single-entry batch should verify");
    assert!(proof_hash != 0, "proof hash should be non-zero");
}

#[test]
fn test_batch_verify_two_rounds_two_entries() {
    let proof = build_valid_two_entry_two_round_proof();
    let (is_valid, proof_hash) = verify_batched_sumcheck(@proof);
    assert!(is_valid, "2-round two-entry batch should verify");
    assert!(proof_hash != 0, "proof hash should be non-zero");
}

#[test]
fn test_batch_verify_two_rounds_tampered_round2_fails() {
    let cs = qm31_new(100, 0, 0, 0);
    let entry = BatchedMatMulEntry {
        node_id: 0, m: 4, n: 4, claimed_sum: cs,
        final_a_eval: qm31_zero(), final_b_eval: qm31_zero(),
        a_commitment: 0xAA11, b_commitment: 0xBB22,
    };

    let mut ch = derive_lambda(array![entry].span(), 4);
    let lambda = channel_draw_qm31(ref ch);

    let c0_r1 = qm31_new(30, 0, 0, 0);
    let c1_r1 = qm31_new(40, 0, 0, 0);
    let c2_r1 = qm31_zero();

    channel_mix_poly_coeffs(ref ch, c0_r1, c1_r1, c2_r1);
    let _ch1 = channel_draw_qm31(ref ch);

    // Tampered round 2 poly — won't satisfy p2(0)+p2(1)=expected
    let c0_r2_bad = qm31_new(999, 0, 0, 0);
    let c1_r2_bad = qm31_new(111, 0, 0, 0);
    let c2_r2_bad = qm31_zero();

    let entry_f = BatchedMatMulEntry {
        node_id: 0, m: 4, n: 4, claimed_sum: cs,
        final_a_eval: qm31_one(), final_b_eval: qm31_one(),
        a_commitment: 0xAA11, b_commitment: 0xBB22,
    };

    let proof = BatchedMatMulProof {
        k: 4, num_rounds: 2, lambda,
        combined_claimed_sum: cs,
        round_polys: array![
            RoundPoly { c0: c0_r1, c1: c1_r1, c2: c2_r1 },
            RoundPoly { c0: c0_r2_bad, c1: c1_r2_bad, c2: c2_r2_bad },
        ],
        entries: array![entry_f],
    };

    let (is_valid, reason) = verify_batched_sumcheck(@proof);
    assert!(!is_valid, "tampered round 2 should fail");
    assert!(reason == 'ROUND_FAIL', "should fail with ROUND_FAIL");
}

// ============================================================================
// Serde Round-Trip Tests
// ============================================================================

#[test]
fn test_serde_roundtrip_preserves_proof() {
    let original = build_valid_single_entry_proof();

    let mut serialized: Array<felt252> = array![];
    original.serialize(ref serialized);

    let mut span = serialized.span();
    let deserialized: BatchedMatMulProof = Serde::<BatchedMatMulProof>::deserialize(ref span)
        .expect('roundtrip deser failed');

    let (is_valid, _) = verify_batched_sumcheck(@deserialized);
    assert!(is_valid, "roundtrip proof should still verify");
    assert!(span.len() == 0, "all data consumed");
}

#[test]
fn test_serde_roundtrip_two_entry_two_round() {
    let original = build_valid_two_entry_two_round_proof();

    let mut serialized: Array<felt252> = array![];
    original.serialize(ref serialized);

    let mut span = serialized.span();
    let deserialized: BatchedMatMulProof = Serde::<BatchedMatMulProof>::deserialize(ref span)
        .expect('roundtrip deser failed');

    let (is_valid, _) = verify_batched_sumcheck(@deserialized);
    assert!(is_valid, "roundtrip 2-entry 2-round proof should verify");
    assert!(span.len() == 0, "all data consumed");
}
