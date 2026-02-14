/// Tests for Phase 9: Direct model verification (eliminates Cairo VM Stage 2).
///
/// Tests verify_model_direct() with chunked upload and batch sumcheck proofs.

use snforge_std::{declare, DeclareResultTrait, ContractClassTrait};
use starknet::ContractAddress;
use elo_cairo_verifier::types::{
    RoundPoly, BatchedMatMulProof, BatchedMatMulEntry,
};
use elo_cairo_verifier::verifier::{
    ISumcheckVerifierDispatcher, ISumcheckVerifierDispatcherTrait,
};
use elo_cairo_verifier::field::{QM31, qm31_new, qm31_mul, qm31_add, qm31_one, poly_eval_degree2, pack_qm31_to_felt};
use elo_cairo_verifier::channel::{
    PoseidonChannel, channel_default, channel_mix_u64, channel_draw_qm31,
    channel_mix_felts, channel_mix_felt, channel_mix_poly_coeffs,
};

// ============================================================================
// Helper: deploy contract
// ============================================================================

fn deploy_verifier() -> ISumcheckVerifierDispatcher {
    let contract = declare("SumcheckVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into()]).unwrap();
    ISumcheckVerifierDispatcher { contract_address: address }
}

fn qm31(a0: u64, a1: u64, b0: u64, b1: u64) -> QM31 {
    qm31_new(a0, a1, b0, b1)
}

// ============================================================================
// Tests: upload_proof_chunk
// ============================================================================

#[test]
fn test_upload_single_chunk() {
    let verifier = deploy_verifier();
    let session_id: felt252 = 0x100;

    // Upload one chunk
    verifier.upload_proof_chunk(session_id, 0, array![0x1, 0x2, 0x3, 0x4]);

    // Check chunk count
    let count = verifier.get_session_chunk_count(session_id);
    assert!(count == 1, "Should have 1 chunk");
}

#[test]
fn test_upload_multiple_chunks() {
    let verifier = deploy_verifier();
    let session_id: felt252 = 0x200;

    // Upload 3 chunks sequentially
    verifier.upload_proof_chunk(session_id, 0, array![0x1, 0x2]);
    verifier.upload_proof_chunk(session_id, 1, array![0x3, 0x4, 0x5]);
    verifier.upload_proof_chunk(session_id, 2, array![0x6]);

    let count = verifier.get_session_chunk_count(session_id);
    assert!(count == 3, "Should have 3 chunks");
}

#[test]
#[should_panic(expected: "Chunks must be uploaded sequentially")]
fn test_upload_out_of_order_fails() {
    let verifier = deploy_verifier();
    let session_id: felt252 = 0x300;

    // Try to upload chunk 1 before chunk 0
    verifier.upload_proof_chunk(session_id, 1, array![0x1, 0x2]);
}

#[test]
#[should_panic(expected: "Chunk data cannot be empty")]
fn test_upload_empty_chunk_fails() {
    let verifier = deploy_verifier();
    let session_id: felt252 = 0x400;

    verifier.upload_proof_chunk(session_id, 0, array![]);
}

// ============================================================================
// Tests: verify_model_direct — batch sumcheck only (no activation STARK)
// ============================================================================

#[test]
fn test_verify_model_direct_single_batch() {
    let verifier = deploy_verifier();

    // Register model
    let model_id: felt252 = 0xdead;
    let weight_commitment: felt252 = 0xbeef;
    verifier.register_model(model_id, weight_commitment);

    // Create a valid single-entry batched sumcheck proof.
    //
    // The verifier's Fiat-Shamir protocol (from sumcheck.cairo) is:
    //   1. mix_u64(num_entries), mix_u64(k)
    //   2. per-entry: mix_felt(pack_qm31_to_felt(claimed_sum)),
    //      mix_felt(a_commitment), mix_felt(b_commitment)
    //   3. draw lambda
    //   4. combined_sum = Σ λ^i * claimed_sum_i
    //   5. verify rounds: p(0) + p(1) = current_sum, draw challenge, current_sum = p(challenge)
    //   6. final check: current_sum = Σ λ^i * a_eval_i * b_eval_i

    // Use a simple proof: c0=5, c1=3, c2=0 → p(0)+p(1) = 5 + (5+3+0) = 13
    let c0 = qm31(5, 0, 0, 0);
    let c1 = qm31(3, 0, 0, 0);
    let c2 = qm31(0, 0, 0, 0);
    let claimed_sum = qm31(13, 0, 0, 0);

    let a_commit: felt252 = 0xaaa;
    let b_commit: felt252 = 0xbbb;

    // Replay the verifier's Fiat-Shamir channel to derive lambda
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 1); // num_entries = 1
    channel_mix_u64(ref ch, 2); // k = 2

    // Per-entry: pack claimed_sum, mix commitments
    channel_mix_felt(ref ch, pack_qm31_to_felt(claimed_sum));
    channel_mix_felt(ref ch, a_commit);
    channel_mix_felt(ref ch, b_commit);

    // Draw lambda
    let lambda = channel_draw_qm31(ref ch);

    // combined_sum = lambda^0 * claimed_sum = 1 * claimed_sum = claimed_sum
    let combined_sum = claimed_sum;

    // Mix round polynomial into channel and draw challenge
    channel_mix_poly_coeffs(ref ch, c0, c1, c2);
    let challenge = channel_draw_qm31(ref ch);

    // p(challenge) = c0 + c1*r + c2*r²
    let p_of_r = poly_eval_degree2(c0, c1, c2, challenge);

    // Final check: current_sum == lambda^0 * final_a * final_b
    // So final_a = p_of_r, final_b = 1
    let final_a_eval = p_of_r;
    let final_b_eval = qm31_one();

    let batch_proof = BatchedMatMulProof {
        k: 2,
        num_rounds: 1,
        lambda,
        combined_claimed_sum: combined_sum,
        round_polys: array![RoundPoly { c0, c1, c2 }],
        entries: array![
            BatchedMatMulEntry {
                node_id: 0,
                m: 1,
                n: 1,
                claimed_sum,
                final_a_eval,
                final_b_eval,
                a_commitment: a_commit,
                b_commitment: b_commit,
            },
        ],
    };

    // Build raw_io_data: [in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...]
    let raw_io_data: Array<felt252> = array![
        1, 4, 4, 1, 2, 3, 4,    // input: 1x4, values [1,2,3,4]
        1, 2, 2, 10, 20,        // output: 1x2, values [10,20]
    ];

    // Call verify_model_direct with no activation STARK (empty array)
    let result = verifier.verify_model_direct(
        model_id,
        0x500, // session_id
        raw_io_data, // raw_io_data (IO commitment recomputed on-chain)
        weight_commitment,
        1, // num_layers
        0, // activation_type (ReLU)
        array![batch_proof],
        array![], // no activation STARK data
    );

    assert!(result, "verify_model_direct should pass");

    // Verify count incremented
    let count = verifier.get_verification_count(model_id);
    assert!(count == 1, "Verification count should be 1");
}

#[test]
#[should_panic(expected: "Model not registered")]
fn test_verify_model_direct_unregistered_fails() {
    let verifier = deploy_verifier();

    verifier.verify_model_direct(
        0xdead, // unregistered model
        0x500,
        array![1, 4, 4, 1, 2, 3, 4, 1, 2, 2, 10, 20], // raw_io_data
        0xbeef,
        1,
        0,
        array![],
        array![],
    );
}

#[test]
#[should_panic(expected: "Weight commitment mismatch")]
fn test_verify_model_direct_wrong_commitment_fails() {
    let verifier = deploy_verifier();

    let model_id: felt252 = 0xdead;
    verifier.register_model(model_id, 0xbeef);

    // Call with wrong weight commitment
    verifier.verify_model_direct(
        model_id,
        0x500,
        array![1, 4, 4, 1, 2, 3, 4, 1, 2, 2, 10, 20], // raw_io_data
        0xdead, // wrong commitment (different from 0xbeef)
        1,
        0,
        array![],
        array![],
    );
}

#[test]
#[should_panic(expected: "Raw IO data cannot be empty")]
fn test_verify_model_direct_empty_io_fails() {
    let verifier = deploy_verifier();

    let model_id: felt252 = 0xdead;
    verifier.register_model(model_id, 0xbeef);

    verifier.verify_model_direct(
        model_id,
        0x500,
        array![], // empty raw_io_data
        0xbeef,
        1,
        0,
        array![],
        array![],
    );
}

// ============================================================================
// Tests: verify_model_direct with uploaded STARK chunks
// ============================================================================

#[test]
fn test_verify_model_direct_with_stark_chunks() {
    // Tests that chunk upload + batch-only verification still works.
    // Chunks are stored for data availability; batch proofs verified separately.
    let verifier = deploy_verifier();

    let model_id: felt252 = 0xface;
    let weight_commitment: felt252 = 0xcafe;
    verifier.register_model(model_id, weight_commitment);

    let session_id: felt252 = 0x600;

    // Upload some STARK chunks (stored on-chain for data availability)
    verifier.upload_proof_chunk(session_id, 0, array![0x1, 0x2, 0x3]);
    verifier.upload_proof_chunk(session_id, 1, array![0x4, 0x5, 0x6, 0x7]);
    verifier.upload_proof_chunk(session_id, 2, array![0x8, 0x9]);

    assert!(verifier.get_session_chunk_count(session_id) == 3);

    // Build a valid batch proof — must replay exact Fiat-Shamir protocol
    let c0 = qm31(5, 0, 0, 0);
    let c1 = qm31(3, 0, 0, 0);
    let c2 = qm31(0, 0, 0, 0);
    let claimed_sum = qm31(13, 0, 0, 0); // p(0) + p(1) = 5 + (5+3+0) = 13
    let a_commit: felt252 = 0xaaa;
    let b_commit: felt252 = 0xbbb;

    // Replay verifier's Fiat-Shamir: mix_u64(num_entries), mix_u64(k), per-entry, draw lambda
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 1); // num_entries
    channel_mix_u64(ref ch, 2); // k
    channel_mix_felt(ref ch, pack_qm31_to_felt(claimed_sum));
    channel_mix_felt(ref ch, a_commit);
    channel_mix_felt(ref ch, b_commit);
    let lambda = channel_draw_qm31(ref ch);

    let combined_sum = claimed_sum;
    channel_mix_poly_coeffs(ref ch, c0, c1, c2);
    let challenge = channel_draw_qm31(ref ch);
    let p_of_r = poly_eval_degree2(c0, c1, c2, challenge);
    let final_a_eval = p_of_r;
    let final_b_eval = qm31_one();

    let batch_proof = BatchedMatMulProof {
        k: 2,
        num_rounds: 1,
        lambda,
        combined_claimed_sum: combined_sum,
        round_polys: array![RoundPoly { c0, c1, c2 }],
        entries: array![
            BatchedMatMulEntry {
                node_id: 0,
                m: 1,
                n: 1,
                claimed_sum,
                final_a_eval,
                final_b_eval,
                a_commitment: a_commit,
                b_commitment: b_commit,
            },
        ],
    };

    // Build raw_io_data for IO commitment recomputation
    let raw_io_data: Array<felt252> = array![
        1, 4, 4, 1, 2, 3, 4,    // input: 1x4
        1, 2, 2, 10, 20,        // output: 1x2
    ];

    // Verify with batch proof + no STARK (chunks are stored but not verified as STARK)
    let result = verifier.verify_model_direct(
        model_id,
        session_id,
        raw_io_data,
        weight_commitment,
        2, // num_layers
        0, // activation_type
        array![batch_proof],
        array![], // no STARK data — chunk upload is for data availability
    );

    assert!(result, "verify_model_direct with batch proof should pass");
    assert!(verifier.get_verification_count(model_id) == 1);
}

#[test]
fn test_verify_model_direct_stark_data_bound_by_hash() {
    let verifier = deploy_verifier();

    let model_id: felt252 = 0xface;
    verifier.register_model(model_id, 0xcafe);

    // STARK data is hash-bound (not deserialized on-chain).
    // Contract should succeed — the STARK hash is included in the proof hash.
    let result = verifier.verify_model_direct(
        model_id,
        0x700,
        array![1, 4, 4, 1, 2, 3, 4, 1, 2, 2, 10, 20], // raw_io_data
        0xcafe,
        1,
        0,
        array![],
        array![0xdead, 0xbeef], // data is hashed, not parsed
    );
    assert!(result, "hash-bound STARK data should pass");
    assert!(verifier.get_verification_count(model_id) == 1, "should record verification");
}

// ============================================================================
// Tests: ml_air module (Air<MLAir> construction)
// ============================================================================

use elo_cairo_verifier::ml_air::compute_activation_log_sizes;
use elo_cairo_verifier::ml_air::compute_unified_log_sizes;
use elo_cairo_verifier::ml_air::ActivationClaim;
use elo_cairo_verifier::ml_air::ElementwiseClaim;
use elo_cairo_verifier::ml_air::LayerNormClaim;
use elo_cairo_verifier::ml_air::EmbeddingClaim;

#[test]
fn test_ml_air_log_sizes_empty() {
    let log_sizes = compute_activation_log_sizes(array![].span());
    let log_sizes_span: Span<Array<u32>> = log_sizes.span();
    let preprocessed: Span<u32> = log_sizes_span[0].span();
    let trace: Span<u32> = log_sizes_span[1].span();
    let interaction: Span<u32> = log_sizes_span[2].span();
    assert!(preprocessed.len() == 0, "preprocessed should be empty");
    assert!(trace.len() == 0, "trace should be empty");
    assert!(interaction.len() == 0, "interaction should be empty");
}

#[test]
fn test_ml_air_log_sizes_single_layer() {
    let claims = array![
        ActivationClaim { layer_index: 0, log_size: 8, activation_type: 0 },
    ];
    let log_sizes = compute_activation_log_sizes(claims.span());
    let log_sizes_span: Span<Array<u32>> = log_sizes.span();

    // Preprocessed: 2 columns at log_size 8
    let preprocessed: Span<u32> = log_sizes_span[0].span();
    assert!(preprocessed.len() == 2, "preprocessed should have 2 columns");
    assert!(*preprocessed[0] == 8, "preprocessed[0] = 8");
    assert!(*preprocessed[1] == 8, "preprocessed[1] = 8");

    // Trace: 3 columns at log_size 8
    let trace: Span<u32> = log_sizes_span[1].span();
    assert!(trace.len() == 3, "trace should have 3 columns");

    // Interaction: 4 columns at log_size 8
    let interaction: Span<u32> = log_sizes_span[2].span();
    assert!(interaction.len() == 4, "interaction should have 4 columns");
}

#[test]
fn test_ml_air_log_sizes_multiple_layers() {
    let claims = array![
        ActivationClaim { layer_index: 0, log_size: 8, activation_type: 0 },
        ActivationClaim { layer_index: 1, log_size: 10, activation_type: 1 },
    ];
    let log_sizes = compute_activation_log_sizes(claims.span());
    let log_sizes_span: Span<Array<u32>> = log_sizes.span();

    // 2 layers * 2 preprocessed = 4 columns
    let preprocessed: Span<u32> = log_sizes_span[0].span();
    assert!(preprocessed.len() == 4, "preprocessed should have 4 columns");
    assert!(*preprocessed[0] == 8, "layer 0 preprocessed");
    assert!(*preprocessed[2] == 10, "layer 1 preprocessed");

    // 2 layers * 3 trace = 6 columns
    let trace: Span<u32> = log_sizes_span[1].span();
    assert!(trace.len() == 6, "trace should have 6 columns");

    // 2 layers * 4 interaction = 8 columns
    let interaction: Span<u32> = log_sizes_span[2].span();
    assert!(interaction.len() == 8, "interaction should have 8 columns");
}

// ============================================================================
// Tests: unified log sizes with all component types
// ============================================================================

#[test]
fn test_unified_log_sizes_all_components() {
    // 1 activation (log=8), 1 add (log=6), 1 mul (log=7),
    // 1 layernorm (log=9), 1 embedding (log=10)
    let act_claims = array![ActivationClaim { layer_index: 0, log_size: 8, activation_type: 0 }];
    let add_claims = array![ElementwiseClaim { layer_index: 1, log_size: 6 }];
    let mul_claims = array![ElementwiseClaim { layer_index: 2, log_size: 7 }];
    let ln_claims = array![LayerNormClaim { layer_index: 3, log_size: 9 }];
    let emb_claims = array![EmbeddingClaim { layer_index: 4, log_size: 10 }];

    let log_sizes = compute_unified_log_sizes(
        act_claims.span(), add_claims.span(), mul_claims.span(),
        ln_claims.span(), emb_claims.span(),
    );
    let log_sizes_span: Span<Array<u32>> = log_sizes.span();

    // Tree 0 (preprocessed): act(2) + ln(2) + emb(3) = 7 columns
    let preprocessed: Span<u32> = log_sizes_span[0].span();
    assert!(preprocessed.len() == 7, "preprocessed should have 7 columns");
    assert!(*preprocessed[0] == 8, "act preprocessed[0]");
    assert!(*preprocessed[1] == 8, "act preprocessed[1]");
    assert!(*preprocessed[2] == 9, "ln preprocessed[0]");
    assert!(*preprocessed[3] == 9, "ln preprocessed[1]");
    assert!(*preprocessed[4] == 10, "emb preprocessed[0]");
    assert!(*preprocessed[5] == 10, "emb preprocessed[1]");
    assert!(*preprocessed[6] == 10, "emb preprocessed[2]");

    // Tree 1 (trace): act(3) + add(3) + mul(3) + ln(6) + emb(4) = 19 columns
    let trace: Span<u32> = log_sizes_span[1].span();
    assert!(trace.len() == 19, "trace should have 19 columns");
    // First 3: activation at log=8
    assert!(*trace[0] == 8, "act trace");
    // Next 3: add at log=6
    assert!(*trace[3] == 6, "add trace");
    // Next 3: mul at log=7
    assert!(*trace[6] == 7, "mul trace");
    // Next 6: layernorm at log=9
    assert!(*trace[9] == 9, "ln trace");
    // Next 4: embedding at log=10
    assert!(*trace[15] == 10, "emb trace");

    // Tree 2 (interaction): act(4) + ln(4) + emb(4) = 12 columns
    let interaction: Span<u32> = log_sizes_span[2].span();
    assert!(interaction.len() == 12, "interaction should have 12 columns");
    assert!(*interaction[0] == 8, "act interaction");
    assert!(*interaction[4] == 9, "ln interaction");
    assert!(*interaction[8] == 10, "emb interaction");
}

#[test]
fn test_unified_log_sizes_pure_air_only() {
    // Only Add + Mul — no preprocessed, no interaction
    let add_claims = array![
        ElementwiseClaim { layer_index: 0, log_size: 5 },
        ElementwiseClaim { layer_index: 1, log_size: 6 },
    ];
    let mul_claims = array![ElementwiseClaim { layer_index: 2, log_size: 7 }];

    let log_sizes = compute_unified_log_sizes(
        array![].span(), add_claims.span(), mul_claims.span(),
        array![].span(), array![].span(),
    );
    let log_sizes_span: Span<Array<u32>> = log_sizes.span();

    let preprocessed: Span<u32> = log_sizes_span[0].span();
    assert!(preprocessed.len() == 0, "pure AIR has no preprocessed");

    // 2 add * 3 + 1 mul * 3 = 9 trace columns
    let trace: Span<u32> = log_sizes_span[1].span();
    assert!(trace.len() == 9, "trace should have 9 columns");

    let interaction: Span<u32> = log_sizes_span[2].span();
    assert!(interaction.len() == 0, "pure AIR has no interaction");
}

// ============================================================================
// Tests: Session isolation
// ============================================================================

#[test]
fn test_separate_sessions_independent() {
    let verifier = deploy_verifier();

    let session_a: felt252 = 0xa00;
    let session_b: felt252 = 0xb00;

    verifier.upload_proof_chunk(session_a, 0, array![0x1, 0x2]);
    verifier.upload_proof_chunk(session_a, 1, array![0x3]);

    verifier.upload_proof_chunk(session_b, 0, array![0x4, 0x5, 0x6]);

    assert!(verifier.get_session_chunk_count(session_a) == 2);
    assert!(verifier.get_session_chunk_count(session_b) == 1);
}

// ============================================================================
// Tests: Conditional LookupElements draw order
//
// Uses stwo_verifier_core::channel::Channel (= Blake2sChannel) to match the
// type that LookupElementsTrait::draw expects.  After running the draw
// protocol, we compare channel states by drawing a probe value from each
// channel — equal probes mean identical internal state.
// ============================================================================

use stwo_constraint_framework::LookupElementsTrait;
use stwo_verifier_core::channel::{Channel, ChannelTrait};
use elo_cairo_verifier::ml_air::{ActivationLookupElements, LayerNormLookupElements, EmbeddingLookupElements};

/// Verify that skipping draws for absent component types keeps the channel
/// state identical to one that only drew for present types (activation only).
#[test]
fn test_conditional_draw_order_activation_only() {
    // Simulate the Rust prover: only activation present, no layernorm/embedding.
    let mut prover_channel: Channel = Default::default();
    prover_channel.mix_u64(0); // interaction pow
    let _: ActivationLookupElements = LookupElementsTrait::draw(ref prover_channel);
    // Prover does NOT draw layernorm or embedding.

    // Simulate the FIXED Cairo verifier: conditional draws.
    let mut verifier_channel: Channel = Default::default();
    verifier_channel.mix_u64(0);
    let has_activation = true;
    let has_layernorm = false;
    let has_embedding = false;

    if has_activation {
        let _: ActivationLookupElements = LookupElementsTrait::draw(ref verifier_channel);
    }
    if has_layernorm {
        let _: LayerNormLookupElements = LookupElementsTrait::draw(ref verifier_channel);
    }
    if has_embedding {
        let _: EmbeddingLookupElements = LookupElementsTrait::draw(ref verifier_channel);
    }

    // Draw a probe value from each — identical state → identical output.
    let prover_probe = prover_channel.draw_secure_felt();
    let verifier_probe = verifier_channel.draw_secure_felt();
    assert!(prover_probe == verifier_probe, "conditional draw: channel diverged (activation only)");
}

/// Verify that unconditionally drawing all 3 types causes channel divergence
/// when only activation is present.
#[test]
fn test_unconditional_draws_cause_divergence() {
    // Rust prover: draws activation only.
    let mut prover_channel: Channel = Default::default();
    prover_channel.mix_u64(0);
    let _: ActivationLookupElements = LookupElementsTrait::draw(ref prover_channel);

    // WRONG verifier: draws all 3 unconditionally.
    let mut wrong_channel: Channel = Default::default();
    wrong_channel.mix_u64(0);
    let _: ActivationLookupElements = LookupElementsTrait::draw(ref wrong_channel);
    let _: LayerNormLookupElements = LookupElementsTrait::draw(ref wrong_channel);  // extra!
    let _: EmbeddingLookupElements = LookupElementsTrait::draw(ref wrong_channel);  // extra!

    // Draw probe values — states MUST differ.
    let prover_probe = prover_channel.draw_secure_felt();
    let wrong_probe = wrong_channel.draw_secure_felt();
    assert!(prover_probe != wrong_probe, "unconditional draws should cause divergence");
}

/// Verify correct draw order when all 3 component types are present.
#[test]
fn test_conditional_draw_order_all_present() {
    // Rust prover: all present, draws all 3 in order.
    let mut prover_channel: Channel = Default::default();
    prover_channel.mix_u64(0);
    let _: ActivationLookupElements = LookupElementsTrait::draw(ref prover_channel);
    let _: LayerNormLookupElements = LookupElementsTrait::draw(ref prover_channel);
    let _: EmbeddingLookupElements = LookupElementsTrait::draw(ref prover_channel);

    // Cairo verifier: all conditionals fire.
    let mut verifier_channel: Channel = Default::default();
    verifier_channel.mix_u64(0);
    let has_activation = true;
    let has_layernorm = true;
    let has_embedding = true;

    if has_activation {
        let _: ActivationLookupElements = LookupElementsTrait::draw(ref verifier_channel);
    }
    if has_layernorm {
        let _: LayerNormLookupElements = LookupElementsTrait::draw(ref verifier_channel);
    }
    if has_embedding {
        let _: EmbeddingLookupElements = LookupElementsTrait::draw(ref verifier_channel);
    }

    let prover_probe = prover_channel.draw_secure_felt();
    let verifier_probe = verifier_channel.draw_secure_felt();
    assert!(prover_probe == verifier_probe, "all-present: channel diverged");
}

// ============================================================================
// Upgrade mechanism tests
// ============================================================================

#[test]
fn test_propose_upgrade() {
    let verifier = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(verifier.contract_address, owner);
    snforge_std::start_cheat_block_timestamp(verifier.contract_address, 1000);

    let new_class: starknet::ClassHash = 0xABCD.try_into().unwrap();
    verifier.propose_upgrade(new_class);

    let (pending_hash, proposed_at) = verifier.get_pending_upgrade();
    assert!(pending_hash == new_class, "pending class hash should match");
    assert!(proposed_at == 1000, "proposed_at should match block timestamp");
}

#[test]
#[should_panic(expected: "Only owner")]
fn test_propose_upgrade_non_owner_fails() {
    let verifier = deploy_verifier();
    let attacker: ContractAddress = 0xBAD_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(verifier.contract_address, attacker);

    let new_class: starknet::ClassHash = 0xABCD.try_into().unwrap();
    verifier.propose_upgrade(new_class);
}

#[test]
#[should_panic(expected: "Upgrade delay not elapsed")]
fn test_execute_upgrade_too_early_fails() {
    let verifier = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(verifier.contract_address, owner);

    let new_class: starknet::ClassHash = 0xABCD.try_into().unwrap();
    verifier.propose_upgrade(new_class);
    // Execute immediately (no time has passed)
    verifier.execute_upgrade();
}

#[test]
fn test_cancel_upgrade() {
    let verifier = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(verifier.contract_address, owner);

    let new_class: starknet::ClassHash = 0xABCD.try_into().unwrap();
    verifier.propose_upgrade(new_class);

    // Verify pending
    let (pending, _) = verifier.get_pending_upgrade();
    assert!(pending == new_class, "should be pending");

    // Cancel
    verifier.cancel_upgrade();

    // Verify cleared
    let (pending_after, ts_after) = verifier.get_pending_upgrade();
    let pending_felt: felt252 = pending_after.into();
    assert!(pending_felt == 0, "should be cleared");
    assert!(ts_after == 0, "timestamp should be cleared");
}

#[test]
#[should_panic(expected: "No upgrade pending")]
fn test_cancel_no_pending_fails() {
    let verifier = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(verifier.contract_address, owner);

    verifier.cancel_upgrade();
}

#[test]
#[should_panic(expected: "Upgrade already pending, cancel first")]
fn test_double_propose_fails() {
    let verifier = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(verifier.contract_address, owner);

    let class1: starknet::ClassHash = 0xABCD.try_into().unwrap();
    let class2: starknet::ClassHash = 0xDEAD.try_into().unwrap();
    verifier.propose_upgrade(class1);
    verifier.propose_upgrade(class2);
}

#[test]
fn test_execute_upgrade_after_delay() {
    let verifier = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(verifier.contract_address, owner);

    // Use the contract's own class hash so replace_class_syscall doesn't fail
    let contract = declare("SumcheckVerifierContract").unwrap().contract_class();
    let class_hash = *contract.class_hash;
    verifier.propose_upgrade(class_hash);

    // Warp time forward by 5 minutes (300 seconds)
    let (_, proposed_at) = verifier.get_pending_upgrade();
    snforge_std::start_cheat_block_timestamp(
        verifier.contract_address, proposed_at + 300,
    );

    // Execute — should succeed
    verifier.execute_upgrade();

    // Verify pending state is cleared
    let (pending_after, ts_after) = verifier.get_pending_upgrade();
    let pending_felt: felt252 = pending_after.into();
    assert!(pending_felt == 0, "pending should be cleared after execute");
    assert!(ts_after == 0, "timestamp should be cleared after execute");
}
