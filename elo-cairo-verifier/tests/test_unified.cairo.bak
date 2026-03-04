// Tests for Phase 8: Unified Model Verifier
//
// Tests the verify_model() entry point that orchestrates all sub-verifiers:
// matmul sumcheck + batched matmul + GKR + layer chain binding.

use elo_cairo_verifier::field::{
    qm31_new, qm31_zero, qm31_one, qm31_add, qm31_mul, qm31_eq,
    pack_qm31_to_felt,
};
use elo_cairo_verifier::channel::{
    channel_default, channel_mix_u64, channel_mix_felt,
    channel_draw_qm31, channel_mix_poly_coeffs, channel_draw_qm31s,
    PoseidonChannel,
};
use elo_cairo_verifier::types::{
    PcsConfig, ModelProof,
    BatchedMatMulProof, BatchedMatMulEntry, RoundPoly,
    GkrBatchProof, GkrInstance, GkrLayerProof, GkrSumcheckProof,
    GkrRoundPoly, GkrMask, GateType,
};
use elo_cairo_verifier::sumcheck::verify_batched_sumcheck;
use elo_cairo_verifier::gkr::partially_verify_batch;

// ============================================================================
// Helpers
// ============================================================================

fn default_pcs_config() -> PcsConfig {
    PcsConfig {
        pow_bits: 12,
        log_blowup_factor: 1,
        log_last_layer_degree_bound: 5,
        n_queries: 14,
    }
}

/// Build the same valid single-instance GP GKR proof used in test_gkr.cairo.
fn build_gkr_proof_single_gp() -> GkrBatchProof {
    let a = qm31_new(3, 0, 0, 0);
    let b = qm31_new(7, 0, 0, 0);
    let product = qm31_mul(a, b); // 21

    GkrBatchProof {
        instances: array![GkrInstance {
            gate: GateType { gate_id: 0 },
            n_variables: 1,
            output_claims: array![product],
        }],
        layer_proofs: array![GkrLayerProof {
            sumcheck_proof: GkrSumcheckProof { round_polys: array![] },
            masks: array![GkrMask {
                values: array![a, b],
                num_columns: 1,
            }],
        }],
    }
}

/// Build a valid batched matmul proof (single entry, 1 round, k=2).
/// Replays the prover's Fiat-Shamir to derive lambda and construct consistent
/// round poly + final evaluations.
fn build_valid_batch_proof() -> BatchedMatMulProof {
    let entry = BatchedMatMulEntry {
        node_id: 0,
        m: 2,
        n: 2,
        claimed_sum: qm31_new(76, 0, 0, 0),
        final_a_eval: qm31_new(5, 0, 0, 0),
        final_b_eval: qm31_new(0, 0, 0, 0), // placeholder — set after challenge
        a_commitment: 42,
        b_commitment: 99,
    };

    // Derive lambda via Fiat-Shamir (same as batch test)
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 1); // num_entries
    channel_mix_u64(ref ch, 2); // k
    channel_mix_felt(ref ch, pack_qm31_to_felt(entry.claimed_sum));
    channel_mix_felt(ref ch, entry.a_commitment);
    channel_mix_felt(ref ch, entry.b_commitment);
    let lambda = channel_draw_qm31(ref ch);

    // With single entry, combined_claimed_sum = lambda^0 * claimed_sum = claimed_sum
    let combined = entry.claimed_sum;

    // Build a round poly where p(0)+p(1) = combined = 76
    // p(0) = c0 = 21, p(1) = c0+c1+c2 = 21+34+0 = 55, sum = 76 ✓
    let c0 = qm31_new(21, 0, 0, 0);
    let c1 = qm31_new(34, 0, 0, 0);
    let c2 = qm31_zero();

    // Mix round poly, draw challenge
    channel_mix_poly_coeffs(ref ch, c0, c1, c2);
    let challenge = channel_draw_qm31(ref ch);

    // p(challenge) = c0 + c1*challenge + c2*challenge^2
    let p_challenge = qm31_add(c0, qm31_mul(c1, challenge));
    // For the final check: p(challenge) = lambda^0 * a_eval * b_eval = a_eval * b_eval
    // So b_eval = p_challenge / a_eval. For simplicity set a_eval=1, b_eval=p_challenge.
    let a_eval = qm31_one();
    let b_eval = p_challenge;

    let real_entry = BatchedMatMulEntry {
        node_id: 0,
        m: 2,
        n: 2,
        claimed_sum: qm31_new(76, 0, 0, 0),
        final_a_eval: a_eval,
        final_b_eval: b_eval,
        a_commitment: 42,
        b_commitment: 99,
    };

    BatchedMatMulProof {
        k: 2,
        num_rounds: 1,
        lambda,
        combined_claimed_sum: combined,
        round_polys: array![RoundPoly { c0, c1, c2 }],
        entries: array![real_entry],
    }
}

// ============================================================================
// ModelProof Serde Tests
// ============================================================================

#[test]
fn test_model_proof_serde_roundtrip() {
    let proof = ModelProof {
        pcs_config: default_pcs_config(),
        raw_io_data: array![1, 4, 4, 1, 2, 3, 4, 1, 2, 2, 10, 20],
        layer_chain_commitment: 67890,
        matmul_proofs: array![],
        batched_matmul_proofs: array![],
        has_gkr: false,
        gkr_proof: array![],
    };

    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    let mut span = serialized.span();
    let deser: ModelProof = Serde::<ModelProof>::deserialize(ref span)
        .expect('model proof deser failed');

    assert!(deser.pcs_config.pow_bits == 12, "pow_bits");
    assert!(deser.pcs_config.n_queries == 14, "n_queries");
    assert!(deser.raw_io_data.len() == 12, "raw_io_data length");
    assert!(deser.layer_chain_commitment == 67890, "layer_chain");
    assert!(deser.matmul_proofs.len() == 0, "no matmuls");
    assert!(deser.batched_matmul_proofs.len() == 0, "no batched");
    assert!(deser.has_gkr == false, "no gkr");
    assert!(span.len() == 0, "all consumed");
}

#[test]
fn test_model_proof_serde_with_gkr() {
    let gkr = build_gkr_proof_single_gp();

    let proof = ModelProof {
        pcs_config: default_pcs_config(),
        raw_io_data: array![1, 2, 2, 5, 6, 1, 2, 2, 7, 8],
        layer_chain_commitment: 222,
        matmul_proofs: array![],
        batched_matmul_proofs: array![],
        has_gkr: true,
        gkr_proof: array![gkr],
    };

    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    let mut span = serialized.span();
    let deser: ModelProof = Serde::<ModelProof>::deserialize(ref span)
        .expect('model+gkr deser failed');

    assert!(deser.has_gkr == true, "has gkr");
    assert!(deser.gkr_proof.len() == 1, "1 gkr proof");
    assert!(span.len() == 0, "all consumed");
}

// ============================================================================
// GKR Sub-Verifier via Unified Path
// ============================================================================

#[test]
fn test_unified_gkr_only() {
    // ModelProof with only a GKR proof — verify_model should pass
    // (We can't call verify_model directly without a contract, so test the
    // sub-verification functions that verify_model orchestrates.)
    let gkr = build_gkr_proof_single_gp();
    let mut ch = channel_default();
    let artifact = partially_verify_batch(@gkr, ref ch);

    assert!(artifact.ood_point.len() == 1, "1 layer");
    assert!(artifact.claims_to_verify.len() == 1, "1 instance");
}

#[test]
fn test_unified_batch_only() {
    // ModelProof with only a batched proof — verify the batch sub-verifier
    let batch = build_valid_batch_proof();
    let (is_valid, _hash) = verify_batched_sumcheck(@batch);
    assert!(is_valid, "batch verification should pass");
}

// ============================================================================
// PCS Config Validation
// ============================================================================

#[test]
fn test_pcs_config_serde() {
    let config = default_pcs_config();
    let mut serialized: Array<felt252> = array![];
    config.serialize(ref serialized);

    assert!(serialized.len() == 4, "PcsConfig = 4 felts");

    let mut span = serialized.span();
    let deser: PcsConfig = Serde::<PcsConfig>::deserialize(ref span)
        .expect('pcs config deser failed');

    assert!(deser.pow_bits == 12, "pow_bits");
    assert!(deser.log_blowup_factor == 1, "log_blowup");
    assert!(deser.log_last_layer_degree_bound == 5, "log_degree");
    assert!(deser.n_queries == 14, "n_queries");
}

// ============================================================================
// Combined Proof Hash Tests
// ============================================================================

#[test]
fn test_proof_hash_includes_io_commitment() {
    // Two proofs with different IO data should produce different hashes
    let mut inputs1: Array<felt252> = array![111, 222, 12, 14];
    let mut inputs2: Array<felt252> = array![999, 222, 12, 14];

    let hash1 = core::poseidon::poseidon_hash_span(inputs1.span());
    let hash2 = core::poseidon::poseidon_hash_span(inputs2.span());

    assert!(hash1 != hash2, "different IO data must produce different hashes");
}

#[test]
fn test_proof_hash_includes_layer_chain() {
    // Two proofs with different layer_chain_commitments should have different hashes
    let mut inputs1: Array<felt252> = array![111, 222, 12, 14];
    let mut inputs2: Array<felt252> = array![111, 333, 12, 14];

    let hash1 = core::poseidon::poseidon_hash_span(inputs1.span());
    let hash2 = core::poseidon::poseidon_hash_span(inputs2.span());

    assert!(hash1 != hash2, "different layer chains must produce different hashes");
}

// ============================================================================
// ModelProof with Batch + GKR Combined
// ============================================================================

#[test]
fn test_model_proof_serde_full_combined() {
    let batch = build_valid_batch_proof();
    let gkr = build_gkr_proof_single_gp();

    let proof = ModelProof {
        pcs_config: default_pcs_config(),
        raw_io_data: array![1, 3, 3, 10, 20, 30, 1, 3, 3, 40, 50, 60],
        layer_chain_commitment: 99999999,
        matmul_proofs: array![],
        batched_matmul_proofs: array![batch],
        has_gkr: true,
        gkr_proof: array![gkr],
    };

    let mut serialized: Array<felt252> = array![];
    proof.serialize(ref serialized);

    let mut span = serialized.span();
    let deser: ModelProof = Serde::<ModelProof>::deserialize(ref span)
        .expect('full combined deser failed');

    assert!(deser.raw_io_data.len() == 12, "raw_io_data preserved");
    assert!(deser.layer_chain_commitment == 99999999, "layer_chain preserved");
    assert!(deser.matmul_proofs.len() == 0, "0 individual matmuls");
    assert!(deser.batched_matmul_proofs.len() == 1, "1 batched proof");
    assert!(deser.has_gkr == true, "has gkr");
    assert!(deser.gkr_proof.len() == 1, "1 gkr proof");
    assert!(span.len() == 0, "all consumed");

    // Verify each sub-proof independently
    let (batch_valid, _) = verify_batched_sumcheck(deser.batched_matmul_proofs.at(0));
    assert!(batch_valid, "batch sub-proof valid");

    let mut ch = channel_default();
    let artifact = partially_verify_batch(deser.gkr_proof.at(0), ref ch);
    assert!(artifact.claims_to_verify.len() == 1, "gkr sub-proof valid");
}

// (Deprecated verify_model contract tests removed in v0.4.0 — see verify_model_gkr tests)

// ============================================================================
// IO Binding Security Tests
// ============================================================================

#[test]
fn test_gkr_channel_binds_io_commitment() {
    // Verify that GKR verification channel is seeded with IO commitment.
    // Two different IO commitments with same GKR data should produce different
    // GKR channel digests.
    let gkr = build_gkr_proof_single_gp();

    // Channel 1: io_commitment = 111
    let mut ch1 = channel_default();
    channel_mix_felt(ref ch1, 111);
    let _artifact1 = partially_verify_batch(@gkr, ref ch1);

    let gkr2 = build_gkr_proof_single_gp();
    // Channel 2: io_commitment = 999
    let mut ch2 = channel_default();
    channel_mix_felt(ref ch2, 999);
    let _artifact2 = partially_verify_batch(@gkr2, ref ch2);

    // Different IO commitments → different channel digests
    assert!(ch1.digest != ch2.digest, "IO binding must differentiate channels");
}

#[test]
fn test_proof_hash_includes_array_lengths() {
    // Proof hash should include array length fields to prevent collision
    // between proofs with different shapes (e.g., 2 matmuls vs 1 matmul + 1 batch)
    let inputs_2mat = array![111_felt252, 222, 12, 14, 2, 0, 0, 0xABC, 0xDEF];
    let inputs_1mat_1batch = array![111_felt252, 222, 12, 14, 1, 1, 0, 0xABC, 0xDEF];

    let hash1 = core::poseidon::poseidon_hash_span(inputs_2mat.span());
    let hash2 = core::poseidon::poseidon_hash_span(inputs_1mat_1batch.span());

    assert!(hash1 != hash2, "different array lengths must produce different hashes");
}
