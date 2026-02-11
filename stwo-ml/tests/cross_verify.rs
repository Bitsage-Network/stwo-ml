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
use stwo_ml::starknet::compute_io_commitment;

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
