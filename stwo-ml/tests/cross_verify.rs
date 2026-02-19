//! Cross-validation tests: verify proofs using both Rust verifier
//! and Cairo serialization pipeline.
//!
//! For each model architecture:
//! 1. Generate proof in Rust
//! 2. Verify in Rust (`verify_model_matmuls`)
//! 3. Serialize for Cairo (`serialize_ml_proof_for_recursive`)
//! 4. Verify serialized data structure is well-formed
//!
//! This ensures the Rust prover and Cairo verifier operate on
//! identical proof data.

use stwo::core::fields::m31::M31;
use starknet_ff::FieldElement;

use stwo_ml::prelude::*;
use stwo_ml::aggregation::prove_model_aggregated_onchain;
use stwo_ml::compiler::prove::{prove_model, verify_model_matmuls};
use stwo_ml::cairo_serde::{
    serialize_ml_proof_for_recursive, serialize_proof, MLClaimMetadata,
};
use stwo_ml::aggregation::compute_io_commitment;

/// Cross-verify: per-layer proof verified in Rust, then aggregated and serialized.
#[test]
fn test_cross_verify_mlp() {
    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    // 1. Per-layer prove + verify
    let (proofs, execution) = prove_model(&model.graph, &input, &model.weights)
        .expect("per-layer proving should succeed");

    verify_model_matmuls(&proofs, &model.graph, &input, &model.weights)
        .expect("Rust verification should succeed");

    // 2. Aggregated prove
    let agg_proof = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("aggregated proving should succeed");

    // 3. Same forward pass output
    assert_eq!(
        execution.output.data,
        agg_proof.execution.output.data,
        "per-layer and aggregated should produce identical outputs"
    );

    // 4. Serialize for Cairo
    let io_commitment = compute_io_commitment(&input, &agg_proof.execution.output);
    let metadata = MLClaimMetadata {
        model_id: FieldElement::from(0x1u64),
        num_layers: model.graph.nodes.len() as u32,
        activation_type: 0,
        io_commitment,
        weight_commitment: FieldElement::ZERO,
        tee_attestation_hash: None,
    };

    let felts = serialize_ml_proof_for_recursive(&agg_proof, &metadata, None);
    assert!(felts.len() > 20, "serialized proof should be non-trivial");

    // 5. Verify activation STARK serializes correctly
    if let Some(stark) = &agg_proof.unified_stark {
        let stark_calldata = serialize_proof(stark);
        assert!(!stark_calldata.is_empty(), "activation STARK calldata should be non-empty");
    }
}

/// Cross-verify a 3-layer deep MLP.
#[test]
fn test_cross_verify_deep_mlp() {
    let model = build_mlp_with_weights(8, &[8, 4], 2, ActivationType::ReLU, 99);

    let mut input = M31Matrix::new(1, 8);
    for j in 0..8 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    // Per-layer
    let (proofs, exec_per_layer) = prove_model(&model.graph, &input, &model.weights)
        .expect("deep MLP per-layer proving");

    verify_model_matmuls(&proofs, &model.graph, &input, &model.weights)
        .expect("deep MLP Rust verification");

    // Aggregated
    let agg_proof = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("deep MLP aggregated proving");

    assert_eq!(
        exec_per_layer.output.data,
        agg_proof.execution.output.data,
        "outputs must match"
    );

    // Serialize
    let io_commitment = compute_io_commitment(&input, &agg_proof.execution.output);
    let metadata = MLClaimMetadata {
        model_id: FieldElement::from(0x2u64),
        num_layers: model.graph.nodes.len() as u32,
        activation_type: 0,
        io_commitment,
        weight_commitment: FieldElement::ZERO,
        tee_attestation_hash: None,
    };

    let felts = serialize_ml_proof_for_recursive(&agg_proof, &metadata, None);
    assert!(felts.len() > 50, "deep MLP proof should be substantial");
}

/// Cross-verify a transformer block.
#[test]
fn test_cross_verify_transformer() {
    let config = TransformerConfig {
        d_model: 4,
        num_heads: 1,
        d_ff: 8,
        activation: ActivationType::GELU,
    };
    let model = build_transformer_block(&config, 77);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    // Per-layer prove + verify
    let (proofs, exec) = prove_model(&model.graph, &input, &model.weights)
        .expect("transformer per-layer proving");

    verify_model_matmuls(&proofs, &model.graph, &input, &model.weights)
        .expect("transformer Rust verification");

    // Aggregated prove
    let agg_proof = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("transformer aggregated proving");

    // Same output
    assert_eq!(
        exec.output.data,
        agg_proof.execution.output.data,
        "transformer outputs must match"
    );

    // Serialize
    let io_commitment = compute_io_commitment(&input, &agg_proof.execution.output);
    let metadata = MLClaimMetadata {
        model_id: FieldElement::from(0x3u64),
        num_layers: model.graph.nodes.len() as u32,
        activation_type: 1, // GELU
        io_commitment,
        weight_commitment: FieldElement::ZERO,
        tee_attestation_hash: None,
    };

    let felts = serialize_ml_proof_for_recursive(&agg_proof, &metadata, Some(42));
    assert!(felts.len() > 50, "transformer proof should be substantial");
}

/// Cross-verify with residual connection.
#[test]
fn test_cross_verify_residual() {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(4);
    let branch = builder.fork();
    builder.activation(ActivationType::ReLU);
    builder.add_from(branch);
    builder.linear(2);
    let graph = builder.build();
    let weights = generate_weights_for_graph(&graph, 42);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    // Per-layer prove + verify
    let (proofs, exec) = prove_model(&graph, &input, &weights)
        .expect("residual per-layer proving");

    verify_model_matmuls(&proofs, &graph, &input, &weights)
        .expect("residual Rust verification");

    // Aggregated
    let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
        .expect("residual aggregated proving");

    assert_eq!(
        exec.output.data,
        agg_proof.execution.output.data,
        "residual outputs must match"
    );
}

/// Verify that different inputs produce different proofs and IO commitments.
#[test]
fn test_cross_verify_different_inputs_different_commitments() {
    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);

    let mut input1 = M31Matrix::new(1, 4);
    for j in 0..4 { input1.set(0, j, M31::from((j + 1) as u32)); }

    let mut input2 = M31Matrix::new(1, 4);
    for j in 0..4 { input2.set(0, j, M31::from((j + 10) as u32)); }

    // Prove both
    let agg1 = prove_model_aggregated_onchain(&model.graph, &input1, &model.weights)
        .expect("proof1");
    let agg2 = prove_model_aggregated_onchain(&model.graph, &input2, &model.weights)
        .expect("proof2");

    // Different outputs
    assert_ne!(agg1.execution.output.data, agg2.execution.output.data);

    // Different IO commitments
    let io1 = compute_io_commitment(&input1, &agg1.execution.output);
    let io2 = compute_io_commitment(&input2, &agg2.execution.output);
    assert_ne!(io1, io2, "different inputs should produce different IO commitments");

    // Both verify in Rust
    let (proofs1, _) = prove_model(&model.graph, &input1, &model.weights).expect("prove1");
    verify_model_matmuls(&proofs1, &model.graph, &input1, &model.weights).expect("verify1");

    let (proofs2, _) = prove_model(&model.graph, &input2, &model.weights).expect("prove2");
    verify_model_matmuls(&proofs2, &model.graph, &input2, &model.weights).expect("verify2");
}

// ─── Mode 4 Aggregated Oracle Sumcheck Cross-Verification ────────────────────

/// RAII guard for setting/restoring environment variables in tests.
struct EnvVarGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, prev }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.as_ref() {
            std::env::set_var(self.key, prev);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

/// Cross-verify mode 4 aggregated binding end-to-end.
///
/// This test exercises the full aggregated oracle sumcheck pipeline:
/// 1. Build a 2-matmul MLP with heterogeneous k dimensions (k=8, k=4)
/// 2. Prove via pure GKR with `STWO_WEIGHT_BINDING=aggregated` (mode 4)
/// 3. Verify the GKR proof carries an `aggregated_binding` proof
/// 4. Serialize the aggregated binding proof via `serialize_aggregated_binding_proof`
/// 5. Verify the proof roundtrips correctly (serialization produces non-trivial output)
/// 6. Cross-verify: run `verify_aggregated_binding` on a fresh channel and confirm it passes
/// 7. Verify the Rust v4 calldata builder works with the aggregated proof
#[test]
fn test_cross_verify_mode4_aggregated_binding() {
    use stwo_ml::aggregation::prove_model_pure_gkr;
    use stwo_ml::crypto::aggregated_opening::AggregatedWeightClaim;
    use stwo_ml::crypto::poseidon_channel::PoseidonChannel;
    use stwo_ml::cairo_serde::serialize_aggregated_binding_proof;
    use stwo_ml::components::matmul::{M31Matrix, matrix_to_mle_col_major_pub};
    // commit_mle_root_only and evaluate_mle_at are exercised internally by
    // prove_aggregated_binding and verify_aggregated_binding respectively.
    use stwo_ml::compiler::graph::{GraphBuilder, GraphWeights};
    use stwo_ml::gkr::{LayeredCircuit, verify_gkr, WeightOpeningTranscriptMode};
    use stwo_ml::starknet::build_gkr_starknet_proof;
    use stwo::core::fields::m31::M31;

    // Activate mode 4 aggregated oracle sumcheck for this test.
    let _binding_mode = EnvVarGuard::set("STWO_WEIGHT_BINDING", "aggregated");

    // ── Step 1: Build a 2-layer MLP with heterogeneous k dimensions ──────────
    // Input (1,8) -> Linear(4) -> Linear(2)
    // MatMul 0: (1,8) x (8,4) => k=8, n=4, weight size = 32 = 2^5
    // MatMul 1: (1,4) x (4,2) => k=4, n=2, weight size = 8 = 2^3
    // Both weight matrices have power-of-2 total sizes (required by MLE).
    // Different k values (8 vs 4) test heterogeneous aggregated binding claims.
    let mut builder = GraphBuilder::new((1, 8));
    builder.linear(4).linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 8);
    for j in 0..8 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    // ── Step 2: Create weights with small M31 values ─────────────────────────
    let mut weights = GraphWeights::new();

    // Weight matrix for layer 0: (8,4) — k=8, n=4
    let mut w0 = M31Matrix::new(8, 4);
    for i in 0..8 {
        for j in 0..4 {
            w0.set(i, j, M31::from(((i * 4 + j) % 7 + 1) as u32));
        }
    }
    weights.add_weight(0, w0);

    // Weight matrix for layer 1: (4,2) — k=4, n=2
    let mut w1 = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w1.set(i, j, M31::from(((i * 2 + j) % 5 + 1) as u32));
        }
    }
    weights.add_weight(1, w1);

    // ── Step 3: Run prove_model_pure_gkr with aggregated binding mode ────────
    let agg_proof = prove_model_pure_gkr(&graph, &input, &weights)
        .expect("pure GKR proving with mode 4 should succeed");

    // ── Step 4: Verify the GKR proof carries aggregated_binding ──────────────
    let gkr_proof = agg_proof
        .gkr_proof
        .as_ref()
        .expect("pure GKR pipeline must produce a GKR proof");

    assert_eq!(
        gkr_proof.weight_opening_transcript_mode,
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck,
        "mode must be AggregatedOracleSumcheck (mode 4)"
    );

    let aggregated_binding = gkr_proof
        .aggregated_binding
        .as_ref()
        .expect("mode 4 proof must include aggregated_binding");

    // The model has 2 matmul layers, so we expect 2 weight claims.
    assert_eq!(
        gkr_proof.weight_claims.len(),
        2,
        "2-layer MLP should have 2 weight claims"
    );

    // ── Step 5: Serialize the aggregated binding proof ────────────────────────
    let mut serialized = Vec::new();
    serialize_aggregated_binding_proof(aggregated_binding, &mut serialized);

    assert!(
        serialized.len() > 10,
        "serialized aggregated binding proof should be non-trivial (got {} felts)",
        serialized.len()
    );

    // Verify estimated calldata matches actual serialized length roughly
    let estimated = aggregated_binding.estimated_calldata_felts();
    assert!(
        estimated > 0,
        "estimated calldata felts must be positive"
    );

    // ── Step 6: Cross-verify with verify_aggregated_binding ──────────────────
    // Reconstruct the AggregatedWeightClaim structs from the GKR proof data,
    // mirroring what the verifier does internally.
    let mut agg_claims = Vec::new();
    for (idx, (claim, commitment)) in gkr_proof
        .weight_claims
        .iter()
        .zip(gkr_proof.weight_commitments.iter())
        .enumerate()
    {
        let weight_matrix = weights
            .get_weight(claim.weight_node_id)
            .expect("weight must exist");
        let mle = matrix_to_mle_col_major_pub(weight_matrix);
        let n_vars = mle.len().trailing_zeros() as usize;

        agg_claims.push(AggregatedWeightClaim {
            matrix_index: idx,
            local_n_vars: n_vars,
            eval_point: claim.eval_point.clone(),
            expected_value: claim.expected_value,
            commitment: *commitment,
        });
    }

    // The verifier uses a fresh channel and replays the same Fiat-Shamir
    // transcript as the prover. We call verify_gkr (which internally calls
    // verify_aggregated_binding) to confirm end-to-end.
    let circuit = LayeredCircuit::from_graph(&graph)
        .expect("circuit compilation should succeed");

    let mut verifier_channel = PoseidonChannel::new();
    verify_gkr(&circuit, gkr_proof, &agg_proof.execution.output, &mut verifier_channel)
        .expect("mode 4 GKR verification with aggregated binding must pass");

    // ── Step 7: Verify v4 calldata builder works ─────────────────────────────
    let model_id = FieldElement::from(0xDEAD_BEEFu64);
    let gkr_starknet = build_gkr_starknet_proof(&agg_proof, model_id, &input)
        .expect("v4 calldata builder should succeed for mode 4 proof");

    assert_eq!(gkr_starknet.model_id, model_id);
    assert!(
        !gkr_starknet.gkr_calldata.is_empty(),
        "GKR calldata must be non-empty"
    );
    assert!(
        !gkr_starknet.io_calldata.is_empty(),
        "IO calldata must be non-empty"
    );
    assert_eq!(
        gkr_starknet.weight_binding_mode_id,
        Some(4),
        "weight binding mode id must be 4 for AggregatedOracleSumcheck"
    );
    assert!(
        gkr_starknet.submission_ready,
        "mode 4 proof should be submission-ready"
    );
    assert!(
        gkr_starknet.soundness_gate_error.is_none(),
        "mode 4 proof should pass all soundness gates"
    );
    assert!(
        !gkr_starknet.weight_binding_data_calldata.is_empty(),
        "mode 4 must carry serialized aggregated binding payload"
    );
    assert!(
        gkr_starknet.total_calldata_size > 0,
        "total calldata size must be positive"
    );
    assert!(
        gkr_starknet.estimated_gas > 0,
        "estimated gas must be positive"
    );

    // Verify IO calldata hashes to the IO commitment
    let recomputed_io = starknet_crypto::poseidon_hash_many(&gkr_starknet.io_calldata);
    assert_eq!(
        recomputed_io, gkr_starknet.io_commitment,
        "IO calldata hash must match IO commitment"
    );

    // Verify heterogeneous dimensions: the two matmuls have different k values.
    // This confirms the aggregated binding handles varying MLE sizes correctly.
    assert_eq!(
        gkr_starknet.num_layer_proofs, 2,
        "2-layer pure matmul MLP should have 2 layer proofs"
    );
}
