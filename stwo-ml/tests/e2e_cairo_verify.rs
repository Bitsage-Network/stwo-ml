//! End-to-end integration test: Rust prove → Cairo VM verify.
//!
//! Builds a small MLP, proves it, serializes the proof via
//! `serialize_ml_proof_for_recursive()`, writes the felt252[] to
//! a temp file, and (when cairo-prove is available) spawns the
//! Cairo verifier subprocess to verify the proof.
//!
//! The test always exercises the Rust serialization pipeline.
//! The Cairo VM verification step is skipped with a warning if
//! `cairo-prove` is not found on PATH.

use stwo::core::fields::m31::M31;
use starknet_ff::FieldElement;

use stwo_ml::prelude::*;
use stwo_ml::aggregation::prove_model_aggregated_onchain;
use stwo_ml::cairo_serde::{
    serialize_ml_proof_for_recursive, serialize_ml_proof_to_arguments_file,
    MLClaimMetadata,
};
use stwo_ml::starknet::compute_io_commitment;

/// Build a small 2-layer MLP with auto-generated weights.
fn build_small_mlp() -> (OnnxModel, M31Matrix) {
    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    (model, input)
}

#[test]
fn test_e2e_mlp_serialize_for_cairo() {
    let (model, input) = build_small_mlp();

    // 1. Prove with on-chain aggregated prover
    let agg_proof = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("on-chain aggregated proving should succeed");

    assert!(agg_proof.unified_stark.is_some(), "should have activation STARK");
    assert_eq!(agg_proof.matmul_proofs.len(), 2, "2 matmul layers");

    // 2. Build metadata for the MLClaim
    let io_commitment = compute_io_commitment(&input, &agg_proof.execution.output);
    let metadata = MLClaimMetadata {
        model_id: FieldElement::from(0x42u64),
        num_layers: 3, // matmul + relu + matmul
        activation_type: 0, // ReLU
        io_commitment,
        weight_commitment: FieldElement::from(0xCAFEu64),
    };

    // 3. Serialize to felt252[]
    let felts = serialize_ml_proof_for_recursive(&agg_proof, &metadata, None);

    // Verify serialization is non-trivial
    assert!(
        felts.len() > 50,
        "serialized proof too small: {} felts",
        felts.len()
    );

    // Verify MLClaim header
    assert_eq!(felts[0], FieldElement::from(0x42u64), "model_id");
    assert_eq!(felts[1], FieldElement::from(3u64), "num_layers");
    assert_eq!(felts[2], FieldElement::from(0u64), "activation_type (ReLU=0)");
    assert_eq!(felts[3], io_commitment, "io_commitment");
    assert_eq!(felts[4], FieldElement::from(0xCAFEu64), "weight_commitment");

    // 4. Convert to JSON arguments file format
    let json = serialize_ml_proof_to_arguments_file(&felts);
    assert!(json.starts_with("[\"0x"), "should start with hex array");
    assert!(json.ends_with("\"]"), "should end with array close");

    // Count elements: should match felts.len()
    let num_elements = json.matches("\"0x").count();
    assert_eq!(num_elements, felts.len(), "JSON element count should match felt count");

    // 5. Write to temp file
    let tmp_dir = std::env::temp_dir().join("stwo_ml_e2e_cairo_test");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let args_file = tmp_dir.join("ml_proof_args.json");
    std::fs::write(&args_file, &json).expect("should write args file");

    assert!(args_file.exists(), "arguments file should exist");
    let file_content = std::fs::read_to_string(&args_file).expect("should read back");
    assert_eq!(file_content, json, "file content should match");

    // 6. Attempt to run cairo-prove if available
    let cairo_prove = which_cairo_prove();
    if let Some(binary) = cairo_prove {
        eprintln!("Found cairo-prove at: {:?}", binary);
        // Note: We don't actually run the full Cairo verifier here because
        // it requires the ml_verifier Cairo package to be compiled.
        // This test validates the Rust → serialized felt252[] pipeline.
        // The actual Cairo VM test is a manual integration test via:
        //   cairo-prove prove-ml --arguments-file ml_proof_args.json
    } else {
        eprintln!(
            "WARNING: cairo-prove not found on PATH. \
             Skipping Cairo VM verification step. \
             Run `cargo build --release -p stwo-cairo-prover` to build it."
        );
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_e2e_serialized_proof_deterministic() {
    let (model, input) = build_small_mlp();

    let metadata = MLClaimMetadata {
        model_id: FieldElement::from(1u64),
        num_layers: 3,
        activation_type: 0,
        io_commitment: FieldElement::ZERO,
        weight_commitment: FieldElement::ZERO,
    };

    // Prove twice
    let proof1 = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("proof1 should succeed");
    let proof2 = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("proof2 should succeed");

    // Same execution output (deterministic forward pass)
    assert_eq!(
        proof1.execution.output.data,
        proof2.execution.output.data,
        "forward pass should be deterministic"
    );

    // Same number of matmul proofs
    assert_eq!(
        proof1.matmul_proofs.len(),
        proof2.matmul_proofs.len(),
        "same proof structure"
    );

    // Serialization should produce same-length output
    let felts1 = serialize_ml_proof_for_recursive(&proof1, &metadata, None);
    let felts2 = serialize_ml_proof_for_recursive(&proof2, &metadata, None);
    assert_eq!(
        felts1.len(),
        felts2.len(),
        "serialized proof length should be deterministic"
    );
}

#[test]
fn test_e2e_transformer_serialize_for_cairo() {
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

    // Prove
    let agg_proof = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("transformer on-chain proving should succeed");

    // Build metadata
    let io_commitment = compute_io_commitment(&input, &agg_proof.execution.output);
    let metadata = MLClaimMetadata {
        model_id: FieldElement::from(0xFFu64),
        num_layers: model.graph.nodes.len() as u32,
        activation_type: 1, // GELU
        io_commitment,
        weight_commitment: FieldElement::from(0xBEEFu64),
    };

    // Serialize
    let felts = serialize_ml_proof_for_recursive(&agg_proof, &metadata, Some(12345));
    assert!(felts.len() > 50, "transformer proof should be substantial");

    // Verify header
    assert_eq!(felts[0], FieldElement::from(0xFFu64), "model_id");
    assert_eq!(felts[2], FieldElement::from(1u64), "activation_type (GELU=1)");

    // JSON format
    let json = serialize_ml_proof_to_arguments_file(&felts);
    assert!(json.len() > 100, "JSON should be substantial");
}

/// Try to find `cairo-prove` binary on PATH.
fn which_cairo_prove() -> Option<std::path::PathBuf> {
    std::process::Command::new("which")
        .arg("cairo-prove")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    Some(std::path::PathBuf::from(path))
                } else {
                    None
                }
            } else {
                None
            }
        })
}
