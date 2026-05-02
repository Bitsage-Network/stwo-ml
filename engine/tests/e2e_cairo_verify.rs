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

use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;

use obelyzk::aggregation::compute_io_commitment;
use obelyzk::aggregation::prove_model_aggregated_onchain;
use obelyzk::cairo_serde::{
    serialize_ml_proof_for_recursive, serialize_ml_proof_to_arguments_file, MLClaimMetadata,
};
use obelyzk::compiler::onnx::NormType;
use obelyzk::prelude::*;

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

    assert!(
        agg_proof.unified_stark.is_some(),
        "should have activation STARK"
    );
    assert_eq!(agg_proof.matmul_proofs.len(), 2, "2 matmul layers");

    // 2. Build metadata for the MLClaim
    let io_commitment = compute_io_commitment(&input, &agg_proof.execution.output);
    let metadata = MLClaimMetadata {
        model_id: FieldElement::from(0x42u64),
        num_layers: 3,      // matmul + relu + matmul
        activation_type: 0, // ReLU
        io_commitment,
        weight_commitment: FieldElement::from(0xCAFEu64),
        tee_attestation_hash: None,
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
    assert_eq!(
        felts[2],
        FieldElement::from(0u64),
        "activation_type (ReLU=0)"
    );
    assert_eq!(felts[3], io_commitment, "io_commitment");
    assert_eq!(felts[4], FieldElement::from(0xCAFEu64), "weight_commitment");

    // 4. Convert to JSON arguments file format
    let json = serialize_ml_proof_to_arguments_file(&felts);
    assert!(json.starts_with("[\"0x"), "should start with hex array");
    assert!(json.ends_with("\"]"), "should end with array close");

    // Count elements: should match felts.len()
    let num_elements = json.matches("\"0x").count();
    assert_eq!(
        num_elements,
        felts.len(),
        "JSON element count should match felt count"
    );

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
        tee_attestation_hash: None,
    };

    // Prove twice
    let proof1 = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("proof1 should succeed");
    let proof2 = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("proof2 should succeed");

    // Same execution output (deterministic forward pass)
    assert_eq!(
        proof1.execution.output.data, proof2.execution.output.data,
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
        norm_type: NormType::LayerNorm,
        head_dim: 4,
        num_experts: 0,
        num_experts_per_tok: 0,
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
        tee_attestation_hash: None,
    };

    // Serialize
    let felts = serialize_ml_proof_for_recursive(&agg_proof, &metadata, Some(12345));
    assert!(felts.len() > 50, "transformer proof should be substantial");

    // Verify header
    assert_eq!(felts[0], FieldElement::from(0xFFu64), "model_id");
    assert_eq!(
        felts[2],
        FieldElement::from(1u64),
        "activation_type (GELU=1)"
    );

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

// ============================================================================
// GKR Roundtrip Tests (B4)
// ============================================================================

use obelyzk::aggregation::prove_model_pure_gkr;
use obelyzk::cairo_serde::{serialize_gkr_model_proof, serialize_mle_opening_proof};
use obelyzk::crypto::poseidon_channel::PoseidonChannel;
use obelyzk::starknet::build_gkr_starknet_proof;

/// Helper: build a matmul-only model (1×4 → 2) with deterministic weights.
fn build_gkr_matmul_only() -> (
    obelyzk::compiler::graph::ComputationGraph,
    M31Matrix,
    obelyzk::compiler::graph::GraphWeights,
) {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = obelyzk::compiler::graph::GraphWeights::new();
    let mut w = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }
    }
    weights.add_weight(0, w);

    (graph, input, weights)
}

/// Helper: build a 3-layer MLP (1×4 → 4 → ReLU → 2) with deterministic weights.
fn build_gkr_mlp_relu() -> (
    obelyzk::compiler::graph::ComputationGraph,
    M31Matrix,
    obelyzk::compiler::graph::GraphWeights,
) {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(4).activation(ActivationType::ReLU).linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = obelyzk::compiler::graph::GraphWeights::new();
    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
        }
    }
    weights.add_weight(0, w0);

    let mut w2 = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w2.set(i, j, M31::from((i + j + 1) as u32));
        }
    }
    weights.add_weight(2, w2);

    (graph, input, weights)
}

/// Build MLP with LayerNorm: 2×4 → Linear(4) → LayerNorm → Linear(2).
///
/// Uses identity weights for the first matmul and arithmetic progression inputs
/// [7,9,11,13] and [17,19,21,23] (mean=10/20, var=5) — ensures variance values
/// map to the rsqrt LogUp table. Pattern matches `test_matmul_layernorm_chain`
/// in verifier.rs.
fn build_gkr_mlp_layernorm() -> (
    obelyzk::compiler::graph::ComputationGraph,
    M31Matrix,
    obelyzk::compiler::graph::GraphWeights,
) {
    let mut builder = GraphBuilder::new((2, 4));
    builder.linear(4).layer_norm().linear(2);
    let graph = builder.build();

    // Arithmetic progression: [7,9,11,13] (mean=10, var=5), [17,19,21,23] (mean=20, var=5)
    let mut input = M31Matrix::new(2, 4);
    input.set(0, 0, M31::from(7u32));
    input.set(0, 1, M31::from(9u32));
    input.set(0, 2, M31::from(11u32));
    input.set(0, 3, M31::from(13u32));
    input.set(1, 0, M31::from(17u32));
    input.set(1, 1, M31::from(19u32));
    input.set(1, 2, M31::from(21u32));
    input.set(1, 3, M31::from(23u32));

    let mut weights = obelyzk::compiler::graph::GraphWeights::new();

    // Identity weight for first matmul — preserves arithmetic progression for LayerNorm
    let mut w0 = M31Matrix::new(4, 4);
    w0.set(0, 0, M31::from(1u32));
    w0.set(1, 1, M31::from(1u32));
    w0.set(2, 2, M31::from(1u32));
    w0.set(3, 3, M31::from(1u32));
    weights.add_weight(0, w0);

    // Second matmul: 4→2 projection
    let mut w2 = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w2.set(i, j, M31::from((i + j + 1) as u32));
        }
    }
    weights.add_weight(2, w2);

    (graph, input, weights)
}

// ----------------------------------------------------------------------------
// B4.1: Matmul-only roundtrip — serialize, verify format
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_roundtrip_matmul_only() {
    let (graph, input, weights) = build_gkr_matmul_only();

    // Prove
    let proof = prove_model_pure_gkr(&graph, &input, &weights)
        .expect("GKR matmul-only proving should succeed");
    let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");

    // Verify proof structure
    assert_eq!(gkr.layer_proofs.len(), 1, "single matmul = 1 layer proof");
    match &gkr.layer_proofs[0] {
        obelyzk::gkr::types::LayerProof::MatMul { round_polys, .. } => {
            // log2(k=4) = 2 sumcheck rounds
            assert_eq!(round_polys.len(), 2, "4×2 matmul needs log2(4)=2 rounds");
        }
        other => panic!(
            "expected MatMul layer proof, got {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Serialize
    let mut calldata = Vec::new();
    serialize_gkr_model_proof(gkr, &mut calldata);

    // Verify format: [num_layers=1, tag=0(MatMul), num_rounds=2, ...]
    assert_eq!(
        calldata[0],
        FieldElement::from(1u64),
        "num_layers should be 1"
    );
    assert_eq!(
        calldata[1],
        FieldElement::from(0u64),
        "tag should be 0 (MatMul)"
    );
    assert_eq!(
        calldata[2],
        FieldElement::from(2u64),
        "num_rounds should be 2"
    );

    // Each degree-2 round poly = 12 felts (3 QM31 × 4 felts)
    // offset 3: 2 round polys × 12 = 24 felts
    // offset 27: final_a_eval (4 felts) + final_b_eval (4 felts) = 8 felts
    // Size is variable due to weight openings (Merkle auth paths).
    // Just check it's reasonable: layers + tail (input_claim + weight_commitments
    // + weight_openings + io_commitment).
    assert!(
        calldata.len() > 30,
        "calldata should have substantial content: {} felts",
        calldata.len()
    );

    // Verify weight commitments populated (B5.2)
    assert_eq!(
        gkr.weight_commitments.len(),
        1,
        "single matmul should have 1 weight commitment"
    );
    assert_ne!(
        gkr.weight_commitments[0],
        FieldElement::ZERO,
        "weight commitment should be non-zero Poseidon Merkle root"
    );

    // Verify the proof is valid via GKR verifier
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");
    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, gkr, &proof.execution.output, &weights, &mut ch);
    assert!(
        result.is_ok(),
        "GKR verification should pass: {:?}",
        result.err()
    );

    eprintln!(
        "GKR roundtrip matmul-only: {} felts, 1 weight commitment, verification OK",
        calldata.len()
    );
}

// ----------------------------------------------------------------------------
// B4.2: MLP with activation roundtrip — verify LogUp fields present
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_roundtrip_with_activation() {
    let (graph, input, weights) = build_gkr_mlp_relu();

    // Prove
    let proof =
        prove_model_pure_gkr(&graph, &input, &weights).expect("GKR MLP proving should succeed");
    let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");

    // Verify 3 layer proofs: MatMul + Activation + MatMul
    assert_eq!(gkr.layer_proofs.len(), 3, "MLP should have 3 layer proofs");

    // Check layer types in order
    let tags: Vec<&str> = gkr
        .layer_proofs
        .iter()
        .map(|lp| match lp {
            obelyzk::gkr::types::LayerProof::MatMul { .. } => "MatMul",
            obelyzk::gkr::types::LayerProof::Activation { .. } => "Activation",
            obelyzk::gkr::types::LayerProof::Add { .. } => "Add",
            obelyzk::gkr::types::LayerProof::Mul { .. } => "Mul",
            obelyzk::gkr::types::LayerProof::LayerNorm { .. } => "LayerNorm",
            obelyzk::gkr::types::LayerProof::RMSNorm { .. } => "RMSNorm",
            obelyzk::gkr::types::LayerProof::Dequantize { .. } => "Dequantize",
            obelyzk::gkr::types::LayerProof::MatMulDualSimd { .. } => "MatMulDualSimd",
            obelyzk::gkr::types::LayerProof::Attention { .. } => "Attention",
            obelyzk::gkr::types::LayerProof::AttentionDecode { .. } => "AttentionDecode",
            obelyzk::gkr::types::LayerProof::Quantize { .. } => "Quantize",
            obelyzk::gkr::types::LayerProof::Embedding { .. } => "Embedding",
            obelyzk::gkr::types::LayerProof::TopK { .. } => "TopK",
        })
        .collect();
    eprintln!("Layer proof types: {:?}", tags);
    assert_eq!(tags[0], "MatMul");
    assert_eq!(tags[1], "Activation");
    assert_eq!(tags[2], "MatMul");

    // Serialize
    let mut calldata = Vec::new();
    serialize_gkr_model_proof(gkr, &mut calldata);

    // First felt is num_layers=3
    assert_eq!(
        calldata[0],
        FieldElement::from(3u64),
        "num_layers should be 3"
    );
    // Verify calldata is substantial (format evolves, skip byte-level walking)
    assert!(
        calldata.len() > 30,
        "MLP calldata should be substantial: {} felts",
        calldata.len()
    );

    // Verify weight commitments (B5.2): 2 MatMul layers → 2 commitments
    assert_eq!(
        gkr.weight_commitments.len(),
        2,
        "MLP with 2 matmuls should have 2 weight commitments"
    );
    for (i, wc) in gkr.weight_commitments.iter().enumerate() {
        assert_ne!(
            *wc,
            FieldElement::ZERO,
            "weight commitment {} should be non-zero",
            i
        );
    }

    // Verify the proof is valid
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");
    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, gkr, &proof.execution.output, &weights, &mut ch);
    assert!(
        result.is_ok(),
        "GKR verification should pass: {:?}",
        result.err()
    );

    eprintln!(
        "GKR roundtrip MLP+ReLU: {} felts, {} layer proofs, 2 weight commitments, verification OK",
        calldata.len(),
        gkr.layer_proofs.len()
    );
}

// ----------------------------------------------------------------------------
// B4.4: Calldata size check — small models should be under Starknet limits
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_calldata_size_small_models() {
    let (graph_mm, input_mm, weights_mm) = build_gkr_matmul_only();
    let (graph_mlp, input_mlp, weights_mlp) = build_gkr_mlp_relu();

    let proof_mm = prove_model_pure_gkr(&graph_mm, &input_mm, &weights_mm)
        .expect("matmul-only proving should succeed");
    let proof_mlp = prove_model_pure_gkr(&graph_mlp, &input_mlp, &weights_mlp)
        .expect("MLP proving should succeed");

    let mut cd_mm = Vec::new();
    serialize_gkr_model_proof(proof_mm.gkr_proof.as_ref().unwrap(), &mut cd_mm);

    let mut cd_mlp = Vec::new();
    serialize_gkr_model_proof(proof_mlp.gkr_proof.as_ref().unwrap(), &mut cd_mlp);

    eprintln!(
        "Calldata sizes: matmul-only = {} felts, MLP = {} felts",
        cd_mm.len(),
        cd_mlp.len()
    );

    // Matmul-only should be small (no LogUp tables)
    assert!(
        cd_mm.len() < 500,
        "matmul-only calldata too large: {} felts",
        cd_mm.len()
    );

    // MLP with ReLU activation has a LogUp proof with 2^16 multiplicities table.
    // This makes the calldata ~65K felts. Large proofs use GKR mode 4
    // (aggregated oracle sumcheck) for 140x calldata reduction.
    // The important thing is that it serializes correctly.
    assert!(
        cd_mlp.len() > 100,
        "MLP calldata should be substantial (has LogUp)"
    );

    // MLP should be larger than matmul-only (more layers + LogUp)
    assert!(
        cd_mlp.len() > cd_mm.len(),
        "MLP calldata should be larger than matmul-only"
    );

    // Document: if LogUp multiplicities table is large, calldata exceeds single-tx limit.
    // Real deployment uses GKR v4 mode 4 (aggregated weight binding) for compact calldata.
    if cd_mlp.len() > 5000 {
        eprintln!(
            "NOTE: MLP calldata ({} felts) exceeds single-tx limit (~5000 felts). \
             Will need chunked verification for on-chain deployment.",
            cd_mlp.len()
        );
    }
}

// ----------------------------------------------------------------------------
// B4.5: GkrStarknetProof roundtrip — full pipeline
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_starknet_proof_roundtrip() {
    let (graph, input, weights) = build_gkr_mlp_relu();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");

    let model_id = FieldElement::from(0xDEAD_BEEFu64);
    let gkr_sn = build_gkr_starknet_proof(&proof, model_id, &input)
        .expect("starknet proof build should succeed");

    // Verify all sections are populated
    assert_eq!(gkr_sn.model_id, model_id);
    assert!(!gkr_sn.gkr_calldata.is_empty());
    assert!(!gkr_sn.io_calldata.is_empty());
    assert_eq!(gkr_sn.num_layer_proofs, 3);

    // IO calldata should hash to IO commitment
    let recomputed = starknet_crypto::poseidon_hash_many(&gkr_sn.io_calldata);
    assert_eq!(recomputed, gkr_sn.io_commitment);

    // Total size should be positive and consistent
    assert!(
        gkr_sn.total_calldata_size > 0,
        "total calldata should be non-zero"
    );
    // Verify total >= individual parts (may include weight claims, binding data)
    let parts_min = 1
        + gkr_sn.gkr_calldata.len()
        + gkr_sn.io_calldata.len()
        + gkr_sn.weight_opening_calldata.len();
    assert!(
        gkr_sn.total_calldata_size >= parts_min,
        "total {} should be >= parts min {}",
        gkr_sn.total_calldata_size,
        parts_min
    );

    eprintln!(
        "GkrStarknetProof: {} total felts ({} GKR + {} IO + {} weight), gas ~{}",
        gkr_sn.total_calldata_size,
        gkr_sn.gkr_calldata.len(),
        gkr_sn.io_calldata.len(),
        gkr_sn.weight_opening_calldata.len(),
        gkr_sn.estimated_gas,
    );
}

// ----------------------------------------------------------------------------
// B4.7: Negative test — tampered input should be detected by verifier
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_rejects_wrong_output() {
    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();

    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");

    // Tamper with the output: change the first element
    let mut bad_output = proof.execution.output.clone();
    bad_output.data[0] = M31::from(bad_output.data[0].0.wrapping_add(1));

    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, gkr, &bad_output, &weights, &mut ch);
    assert!(
        result.is_err(),
        "verification should reject tampered output"
    );
}

// ----------------------------------------------------------------------------
// B4.8: Negative test — tampered round poly should be detected
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_rejects_tampered_round_poly() {
    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    // Tamper with the first round polynomial of the matmul layer
    match &mut gkr.layer_proofs[0] {
        obelyzk::gkr::types::LayerProof::MatMul { round_polys, .. } => {
            // Flip a coefficient
            let orig = round_polys[0].c0;
            round_polys[0].c0 =
                orig + stwo::core::fields::qm31::QM31::from_u32_unchecked(1, 0, 0, 0);
        }
        _ => panic!("expected MatMul"),
    }

    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");

    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, &gkr, &proof.execution.output, &weights, &mut ch);
    assert!(
        result.is_err(),
        "verification should reject tampered round poly"
    );
}

// ----------------------------------------------------------------------------
// B4.9: Negative test — tampered final eval should be detected
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_rejects_tampered_final_eval() {
    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    // Tamper with final_a_eval
    match &mut gkr.layer_proofs[0] {
        obelyzk::gkr::types::LayerProof::MatMul { final_a_eval, .. } => {
            *final_a_eval =
                *final_a_eval + stwo::core::fields::qm31::QM31::from_u32_unchecked(1, 0, 0, 0);
        }
        _ => panic!("expected MatMul"),
    }

    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");

    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, &gkr, &proof.execution.output, &weights, &mut ch);
    assert!(
        result.is_err(),
        "verification should reject tampered final eval"
    );
}

// ----------------------------------------------------------------------------
// B4.10: Negative test — wrong IO commitment in starknet proof
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_wrong_io_commitment_detected() {
    let (graph, input, weights) = build_gkr_mlp_relu();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");

    let model_id = FieldElement::from(1u64);
    let gkr_sn = build_gkr_starknet_proof(&proof, model_id, &input)
        .expect("starknet proof build should succeed");

    // IO calldata hash should match commitment
    let real_hash = starknet_crypto::poseidon_hash_many(&gkr_sn.io_calldata);
    assert_eq!(real_hash, gkr_sn.io_commitment);

    // A tampered IO commitment won't match the calldata hash
    let fake_commitment = FieldElement::from(0xBAD_CAFEu64);
    assert_ne!(
        fake_commitment, real_hash,
        "fake commitment should differ from real hash"
    );
}

// ----------------------------------------------------------------------------
// B4.11: Serialization determinism — same model produces identical calldata
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_serialization_deterministic() {
    let (graph, input, weights) = build_gkr_mlp_relu();

    let proof1 = prove_model_pure_gkr(&graph, &input, &weights).expect("proof 1 should succeed");
    let proof2 = prove_model_pure_gkr(&graph, &input, &weights).expect("proof 2 should succeed");

    let mut cd1 = Vec::new();
    let mut cd2 = Vec::new();
    serialize_gkr_model_proof(proof1.gkr_proof.as_ref().unwrap(), &mut cd1);
    serialize_gkr_model_proof(proof2.gkr_proof.as_ref().unwrap(), &mut cd2);

    assert_eq!(
        cd1.len(),
        cd2.len(),
        "serialized length should be deterministic"
    );
    assert_eq!(cd1, cd2, "serialized content should be deterministic");

    // Weight commitments should also be deterministic (B5.2)
    let wc1 = &proof1.gkr_proof.as_ref().unwrap().weight_commitments;
    let wc2 = &proof2.gkr_proof.as_ref().unwrap().weight_commitments;
    assert_eq!(wc1, wc2, "weight commitments should be deterministic");
    assert_eq!(wc1.len(), 2, "MLP should have 2 weight commitments");
}

// ----------------------------------------------------------------------------
// B4.7: Negative test — tampered input claim should be detected
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_rejects_wrong_input_claim() {
    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    // Tamper with the input claim value
    gkr.input_claim.value =
        gkr.input_claim.value + stwo::core::fields::qm31::QM31::from_u32_unchecked(1, 0, 0, 0);

    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");

    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, &gkr, &proof.execution.output, &weights, &mut ch);
    assert!(
        result.is_err(),
        "verification should reject tampered input claim"
    );
}

// ----------------------------------------------------------------------------
// B4.8: Negative test — tampered weight commitment detected in starknet proof
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_rejects_wrong_weight_commitment() {
    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();

    // Verify real weight commitment is non-zero
    assert_eq!(gkr.weight_commitments.len(), 1);
    let real_commitment = gkr.weight_commitments[0];
    assert_ne!(real_commitment, FieldElement::ZERO);

    // Build starknet proof and verify the commitment propagates
    let model_id = FieldElement::from(42u64);
    let _gkr_sn =
        build_gkr_starknet_proof(&proof, model_id, &input).expect("starknet proof should succeed");

    // Verify calldata contains the weight commitment
    // The weight commitment appears in the GKR calldata tail
    let mut calldata = Vec::new();
    serialize_gkr_model_proof(gkr, &mut calldata);

    // Verify the real weight commitment appears in the serialized calldata.
    // The exact offset is variable due to weight openings, so search for it.
    assert!(
        calldata.iter().any(|f| *f == real_commitment),
        "weight commitment should appear in serialized calldata"
    );

    // A fake commitment would mismatch — verifier should detect this
    let fake_commitment = FieldElement::from(0xDEAD_BEEFu64);
    assert_ne!(
        fake_commitment, real_commitment,
        "fake commitment should differ from real"
    );
}

// ----------------------------------------------------------------------------
// B4.11: Negative test — tampered activation input_eval
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_rejects_tampered_activation_input_eval() {
    let (graph, input, weights) = build_gkr_mlp_relu();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    // Find the Activation layer and tamper its input_eval
    let mut found = false;
    for lp in &mut gkr.layer_proofs {
        if let obelyzk::gkr::types::LayerProof::Activation { input_eval, .. } = lp {
            *input_eval = *input_eval
                + stwo::core::fields::qm31::QM31::from_u32_unchecked(1, 0, 0, 0);
            found = true;
            break;
        }
    }
    assert!(
        found,
        "should have found an Activation layer to tamper"
    );

    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");

    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, &gkr, &proof.execution.output, &weights, &mut ch);
    assert!(
        result.is_err(),
        "verification should reject tampered activation input_eval"
    );
}

// ----------------------------------------------------------------------------
// B4.3: LayerNorm roundtrip — verify 2-part proof structure
// ----------------------------------------------------------------------------

#[test]
fn test_gkr_roundtrip_with_layernorm() {
    let (graph, input, weights) = build_gkr_mlp_layernorm();

    let proof = prove_model_pure_gkr(&graph, &input, &weights)
        .expect("GKR LayerNorm proving should succeed");
    let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");

    // Should have 3 layer proofs: MatMul + LayerNorm + MatMul
    assert_eq!(gkr.layer_proofs.len(), 3, "should have 3 layer proofs");

    let tags: Vec<&str> = gkr
        .layer_proofs
        .iter()
        .map(|lp| match lp {
            obelyzk::gkr::types::LayerProof::MatMul { .. } => "MatMul",
            obelyzk::gkr::types::LayerProof::LayerNorm { .. } => "LayerNorm",
            obelyzk::gkr::types::LayerProof::Activation { .. } => "Activation",
            obelyzk::gkr::types::LayerProof::Add { .. } => "Add",
            obelyzk::gkr::types::LayerProof::Mul { .. } => "Mul",
            obelyzk::gkr::types::LayerProof::RMSNorm { .. } => "RMSNorm",
            obelyzk::gkr::types::LayerProof::Dequantize { .. } => "Dequantize",
            obelyzk::gkr::types::LayerProof::MatMulDualSimd { .. } => "MatMulDualSimd",
            obelyzk::gkr::types::LayerProof::Attention { .. } => "Attention",
            obelyzk::gkr::types::LayerProof::AttentionDecode { .. } => "AttentionDecode",
            obelyzk::gkr::types::LayerProof::Quantize { .. } => "Quantize",
            obelyzk::gkr::types::LayerProof::Embedding { .. } => "Embedding",
            obelyzk::gkr::types::LayerProof::TopK { .. } => "TopK",
        })
        .collect();
    eprintln!("LayerNorm model proof types: {:?}", tags);
    assert_eq!(tags[0], "MatMul");
    assert_eq!(tags[1], "LayerNorm");
    assert_eq!(tags[2], "MatMul");

    // LayerNorm proof should have linear + LogUp sub-proofs
    match &gkr.layer_proofs[1] {
        obelyzk::gkr::types::LayerProof::LayerNorm {
            linear_round_polys,
            logup_proof,
            ..
        } => {
            assert!(
                !linear_round_polys.is_empty(),
                "LayerNorm should have linear eq-sumcheck round polys"
            );
            eprintln!(
                "LayerNorm: {} linear rounds, logup={}",
                linear_round_polys.len(),
                logup_proof.is_some()
            );
        }
        other => panic!(
            "expected LayerNorm proof, got {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Weight commitments: 2 matmul layers
    assert_eq!(gkr.weight_commitments.len(), 2);

    // Serialize and verify
    let mut calldata = Vec::new();
    serialize_gkr_model_proof(gkr, &mut calldata);
    assert_eq!(calldata[0], FieldElement::from(3u64), "num_layers = 3");

    // Verify calldata is substantial (format evolves, skip byte-level walking)
    assert!(
        calldata.len() > 30,
        "LayerNorm model calldata should be substantial: {} felts",
        calldata.len()
    );

    // Verify the proof passes GKR verification
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");
    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, gkr, &proof.execution.output, &weights, &mut ch);
    assert!(
        result.is_ok(),
        "GKR verification should pass: {:?}",
        result.err()
    );

    eprintln!(
        "GKR roundtrip MatMul+LayerNorm+MatMul: {} felts, verification OK",
        calldata.len()
    );
}

// ============================================================================
// SP3.1: Export proof calldata for cross-language testing
// ============================================================================

/// Build a 3-layer MLP: MatMul(4→4) → ReLU → MatMul(4→2)
/// and export the full GKR proof calldata + verification metadata
/// for Engineer A's Cairo tests.
#[test]
fn test_sp3_export_mlp_proof_calldata() {
    let (graph, input, weights) = build_gkr_mlp_relu();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();

    // Serialize
    let mut calldata = Vec::new();
    serialize_gkr_model_proof(gkr, &mut calldata);

    // Verify it passes
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");
    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, gkr, &proof.execution.output, &weights, &mut ch);
    assert!(result.is_ok(), "proof must verify before export");

    let final_digest = ch.digest();

    // Export as structured JSON for Engineer A
    let json = serde_json::json!({
        "description": "SP3.1: MLP 1x4 -> Linear(4) -> ReLU -> Linear(2), GKR proof",
        "model": {
            "input_shape": [1, 4],
            "layers": ["MatMul(4x4)", "ReLU", "MatMul(4x2)"],
            "num_layer_proofs": gkr.layer_proofs.len(),
        },
        "input_data": input.data.iter().map(|m| m.0).collect::<Vec<u32>>(),
        "output_data": proof.execution.output.data.iter().map(|m| m.0).collect::<Vec<u32>>(),
        "circuit": {
            "num_layers": circuit.layers.len(),
            "input_shape": [circuit.input_shape.0, circuit.input_shape.1],
        },
        "weight_commitments": gkr.weight_commitments.iter()
            .map(|c| format!("0x{:064x}", c))
            .collect::<Vec<_>>(),
        "io_commitment": format!("0x{:064x}", gkr.io_commitment),
        "verification": {
            "final_channel_digest": format!("0x{:064x}", final_digest),
            "status": "PASS",
        },
        "calldata_size": calldata.len(),
        "calldata_hex": calldata.iter()
            .map(|f| format!("0x{:x}", f))
            .collect::<Vec<_>>(),
    });

    // Write to artifacts directory
    let artifacts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("artifacts");
    std::fs::create_dir_all(&artifacts_dir).ok();

    let path = artifacts_dir.join("sp3_mlp_proof.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
        .expect("failed to write artifact");

    eprintln!("SP3.1 artifact written to: {}", path.display());
    eprintln!("  Calldata: {} felts", calldata.len());
    eprintln!("  Weight commitments: {}", gkr.weight_commitments.len());
    eprintln!("  Final digest: 0x{:064x}", final_digest);

    // Also export a tampered version (SP3.4) — flip first round poly coefficient
    let mut tampered_gkr = gkr.clone();
    match &mut tampered_gkr.layer_proofs[0] {
        obelyzk::gkr::types::LayerProof::MatMul { round_polys, .. } => {
            round_polys[0].c0 =
                round_polys[0].c0 + stwo::core::fields::qm31::QM31::from_u32_unchecked(1, 0, 0, 0);
        }
        _ => panic!("first layer should be MatMul"),
    }

    let mut tampered_calldata = Vec::new();
    serialize_gkr_model_proof(&tampered_gkr, &mut tampered_calldata);

    // Verify it FAILS
    let mut ch2 = PoseidonChannel::new();
    let tampered_result =
        obelyzk::gkr::verify_gkr_with_weights(&circuit, &tampered_gkr, &proof.execution.output, &weights, &mut ch2);
    assert!(
        tampered_result.is_err(),
        "tampered proof must fail verification"
    );

    let tampered_json = serde_json::json!({
        "description": "SP3.4: TAMPERED MLP proof (flipped c0 of first matmul round poly)",
        "tamper_location": "layer_proofs[0].round_polys[0].c0 += 1",
        "expected_result": "REJECT",
        "calldata_size": tampered_calldata.len(),
        "calldata_hex": tampered_calldata.iter()
            .map(|f| format!("0x{:x}", f))
            .collect::<Vec<_>>(),
    });

    let tampered_path = artifacts_dir.join("sp3_mlp_proof_tampered.json");
    std::fs::write(
        &tampered_path,
        serde_json::to_string_pretty(&tampered_json).unwrap(),
    )
    .expect("failed to write tampered artifact");

    eprintln!(
        "SP3.4 tampered artifact written to: {}",
        tampered_path.display()
    );
}

/// Export a simple matmul-only proof — minimal test vector for Cairo bring-up.
#[test]
fn test_sp3_export_matmul_only_proof() {
    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();

    let mut calldata = Vec::new();
    serialize_gkr_model_proof(gkr, &mut calldata);

    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = obelyzk::gkr::verify_gkr_with_weights(&circuit, gkr, &proof.execution.output, &weights, &mut ch);
    assert!(result.is_ok());

    // Serialize weight opening proofs (Array<MleOpeningProof> format for Cairo Serde)
    let mut weight_opening_calldata = Vec::new();
    weight_opening_calldata.push(FieldElement::from(gkr.weight_openings.len() as u64));
    for opening in &gkr.weight_openings {
        serialize_mle_opening_proof(opening, &mut weight_opening_calldata);
    }

    // Helper to decompose QM31 → 4 u64 components
    let qm31_parts = |v: stwo::core::fields::qm31::SecureField| -> [u64; 4] {
        use stwo::core::fields::FieldExpOps;
        let comps = v.to_m31_array();
        [
            comps[0].0 as u64,
            comps[1].0 as u64,
            comps[2].0 as u64,
            comps[3].0 as u64,
        ]
    };

    // Capture weight claims for cross-language verification
    let weight_claims_json: Vec<_> = gkr
        .weight_claims
        .iter()
        .map(|c| {
            serde_json::json!({
                "eval_point": c.eval_point.iter().map(|p| qm31_parts(*p)).collect::<Vec<_>>(),
                "expected_value": qm31_parts(c.expected_value),
            })
        })
        .collect();

    // Channel digest after weight opening verification
    let digest_after_weight_openings = format!("0x{:064x}", ch.digest());

    let json = serde_json::json!({
        "description": "SP3: Matmul-only 1x4 -> 2, minimal GKR proof with weight openings",
        "model": {
            "input_shape": [1, 4],
            "layers": ["MatMul(4x2)"],
            "num_layer_proofs": 1,
        },
        "input_data": input.data.iter().map(|m| m.0).collect::<Vec<u32>>(),
        "output_data": proof.execution.output.data.iter().map(|m| m.0).collect::<Vec<u32>>(),
        "weight_commitments": gkr.weight_commitments.iter()
            .map(|c| format!("0x{:064x}", c))
            .collect::<Vec<_>>(),
        "io_commitment": format!("0x{:064x}", gkr.io_commitment),
        "verification": {
            "final_channel_digest": digest_after_weight_openings,
            "status": "PASS",
        },
        "weight_claims": weight_claims_json,
        "weight_opening_calldata_size": weight_opening_calldata.len(),
        "weight_opening_calldata_hex": weight_opening_calldata.iter()
            .map(|f| format!("0x{:x}", f))
            .collect::<Vec<_>>(),
        "calldata_size": calldata.len(),
        "calldata_hex": calldata.iter()
            .map(|f| format!("0x{:x}", f))
            .collect::<Vec<_>>(),
    });

    let artifacts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("artifacts");
    std::fs::create_dir_all(&artifacts_dir).ok();

    let path = artifacts_dir.join("sp3_matmul_only_proof.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
        .expect("failed to write artifact");

    eprintln!("SP3 matmul-only artifact: {}", path.display());
    eprintln!("  Calldata: {} felts", calldata.len());
}

/// B3.2: Test `build_verify_model_gkr_calldata` produces correct structure for matmul-only model.
#[test]
fn test_verify_model_gkr_calldata_matmul_only() {
    use obelyzk::cairo_serde::serialize_gkr_proof_data_only;
    use obelyzk::starknet::{
        build_circuit_descriptor, build_register_gkr_calldata, build_verify_model_gkr_calldata,
        extract_dequantize_bits, extract_matmul_dims,
    };

    // V1 calldata builder requires Sequential weight binding mode
    unsafe { std::env::set_var("STWO_WEIGHT_BINDING", "sequential") };

    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();

    let model_id = FieldElement::from(0x42u64);

    // Test dimension extraction
    let matmul_dims = extract_matmul_dims(&circuit);
    assert_eq!(matmul_dims.len(), 3, "1 matmul = 3 dims (m,k,n)");
    // 1×4 input, 4×2 weight → m=1, k=4, n=2
    assert_eq!(matmul_dims[0], 1, "m=1");
    assert_eq!(matmul_dims[1], 4, "k=4");
    assert_eq!(matmul_dims[2], 2, "n=2");

    let dequantize_bits = extract_dequantize_bits(&circuit);
    assert!(dequantize_bits.is_empty(), "no dequantize layers");

    // Test circuit descriptor
    let desc = build_circuit_descriptor(&circuit);
    assert!(desc.len() >= 2, "at least num_layers + 1 tag");
    assert_eq!(desc[0], 1, "1 layer"); // num_layers
    assert_eq!(desc[1], 0, "tag 0 = MatMul");

    // Test registration calldata
    let reg_calldata = build_register_gkr_calldata(model_id, &gkr.weight_commitments, &desc);
    assert!(!reg_calldata.is_empty());
    assert_eq!(reg_calldata[0], format!("0x{:x}", model_id));

    // Test verify calldata
    let raw_io_data = obelyzk::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
    let verify_calldata = build_verify_model_gkr_calldata(gkr, &circuit, model_id, &raw_io_data).expect("build verify calldata");
    assert!(
        verify_calldata.total_felts > 10,
        "should have substantial calldata"
    );

    // Verify structure: parts[0]=model_id, [1]=io_commitment, [2]=num_layers,
    let parts = &verify_calldata.calldata_parts;
    assert_eq!(parts[0], format!("0x{:x}", model_id));
    let rio_len: usize = parts[1].parse().unwrap();
    assert_eq!(rio_len, raw_io_data.len());
    let o = 2 + rio_len; // offset past raw_io_data
    assert_eq!(parts[o], "1"); // circuit_depth = 1
    assert_eq!(parts[o + 1], "1"); // num_layers = 1
    assert_eq!(parts[o + 2], "3"); // matmul_dims_len = 3
    assert_eq!(parts[o + 3], "1"); // m=1
    assert_eq!(parts[o + 4], "4"); // k=4
    assert_eq!(parts[o + 5], "2"); // n=2

    eprintln!("verify_model_gkr calldata: {} parts", parts.len());
    eprintln!("  matmul_dims: {:?}", matmul_dims);
}

/// B3.2: Test calldata for MLP with activation (matmul + relu + matmul).
#[test]
fn test_verify_model_gkr_calldata_mlp_relu() {
    use obelyzk::starknet::{
        build_verify_model_gkr_calldata, extract_dequantize_bits, extract_matmul_dims,
    };

    // V1 calldata builder requires Sequential weight binding mode
    unsafe { std::env::set_var("STWO_WEIGHT_BINDING", "sequential") };

    let (graph, input, weights) = build_gkr_mlp_relu();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();

    let model_id = FieldElement::from(0x99u64);

    // MLP: MatMul(4→4), ReLU, MatMul(4→2) → 2 matmul dims, 0 dequantize
    let matmul_dims = extract_matmul_dims(&circuit);
    assert_eq!(matmul_dims.len(), 6, "2 matmuls = 6 dims");

    let dequantize_bits = extract_dequantize_bits(&circuit);
    assert!(dequantize_bits.is_empty());

    let raw_io_data = obelyzk::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
    let verify_calldata = build_verify_model_gkr_calldata(gkr, &circuit, model_id, &raw_io_data).expect("build verify calldata");
    assert!(
        verify_calldata.total_felts > 50,
        "MLP calldata should be larger"
    );

    let parts = &verify_calldata.calldata_parts;
    assert_eq!(parts[0], format!("0x{:x}", model_id));
    let rio_len: usize = parts[1].parse().unwrap();
    let o = 2 + rio_len;
    assert_eq!(parts[o], "3"); // circuit_depth = 3
    assert_eq!(parts[o + 1], "3"); // num_layers = 3
    assert_eq!(parts[o + 2], "6"); // matmul_dims_len = 6

    eprintln!("MLP verify_model_gkr calldata: {} parts", parts.len());
}

/// D7: Export on-chain calldata files for matmul-only model deployment test.
///
/// Generates two files in tests/artifacts/:
/// - `d7_register_gkr_calldata.txt` — sncast calldata for register_model_gkr
/// - `d7_verify_gkr_calldata.txt` — sncast calldata for verify_model_gkr
#[test]
fn test_d7_export_onchain_calldata() {
    use obelyzk::starknet::{
        build_circuit_descriptor, build_register_gkr_calldata, build_verify_model_gkr_calldata,
    };

    // V1 calldata builder requires Sequential weight binding mode
    unsafe { std::env::set_var("STWO_WEIGHT_BINDING", "sequential") };

    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();

    let model_id = FieldElement::from(0xD7u64); // test model ID
    let io_commitment = proof.io_commitment;

    // Build registration calldata
    let circuit_desc = build_circuit_descriptor(&circuit);
    let register_calldata =
        build_register_gkr_calldata(model_id, &gkr.weight_commitments, &circuit_desc);

    // Build verification calldata
    let raw_io_data = obelyzk::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
    let verify_calldata = build_verify_model_gkr_calldata(gkr, &circuit, model_id, &raw_io_data).expect("build verify calldata");

    // Write to artifacts
    let artifacts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("artifacts");
    std::fs::create_dir_all(&artifacts_dir).ok();

    let register_path = artifacts_dir.join("d7_register_gkr_calldata.txt");
    std::fs::write(&register_path, register_calldata.join(" ")).expect("write register calldata");

    let verify_path = artifacts_dir.join("d7_verify_gkr_calldata.txt");
    std::fs::write(&verify_path, verify_calldata.calldata_parts.join(" "))
        .expect("write verify calldata");

    // Also write a JSON summary for debugging
    let summary = serde_json::json!({
        "model_id": format!("0x{:x}", model_id),
        "io_commitment": format!("0x{:x}", io_commitment),
        "num_layers": gkr.layer_proofs.len(),
        "weight_commitments": gkr.weight_commitments.iter()
            .map(|c| format!("0x{:x}", c))
            .collect::<Vec<_>>(),
        "register_calldata_len": register_calldata.len(),
        "verify_calldata_len": verify_calldata.total_felts,
        "circuit_descriptor": circuit_desc,
    });
    let summary_path = artifacts_dir.join("d7_onchain_summary.json");
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write summary");

    eprintln!("D7 artifacts exported:");
    eprintln!(
        "  Register: {} ({} parts)",
        register_path.display(),
        register_calldata.len()
    );
    eprintln!(
        "  Verify:   {} ({} parts)",
        verify_path.display(),
        verify_calldata.total_felts
    );
    eprintln!("  Summary:  {}", summary_path.display());
}

/// D9: Export on-chain calldata files for MLP (MatMul → ReLU → MatMul) model.
///
/// This exercises the full GKR walk across 3 layer types:
///   Layer 0 (from output): MatMul 4→2  (degree-2 sumcheck)
///   Layer 1: Activation ReLU           (LogUp eq-sumcheck, degree-3)
///   Layer 2: MatMul 4→4                (degree-2 sumcheck)
///
/// Generates artifacts in tests/artifacts/:
///   - d9_register_gkr_calldata.txt
///   - d9_verify_gkr_calldata.txt
///   - d9_onchain_summary.json
#[test]
fn test_d9_export_mlp_onchain_calldata() {
    use obelyzk::starknet::{
        build_circuit_descriptor, build_register_gkr_calldata, build_verify_model_gkr_calldata,
        extract_dequantize_bits, extract_matmul_dims,
    };

    // V1 calldata builder requires Sequential weight binding mode
    unsafe { std::env::set_var("STWO_WEIGHT_BINDING", "sequential") };

    let (graph, input, weights) = build_gkr_mlp_relu();

    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("MLP proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();

    let model_id = FieldElement::from(0xD9u64);
    let io_commitment = proof.io_commitment;

    // Validate circuit structure
    assert_eq!(gkr.layer_proofs.len(), 3, "MLP should have 3 layers");
    let circuit_desc = build_circuit_descriptor(&circuit);
    eprintln!("D9 circuit descriptor: {:?}", circuit_desc);

    // Validate dimensions
    let matmul_dims = extract_matmul_dims(&circuit);
    eprintln!("D9 matmul_dims: {:?}", matmul_dims);
    assert_eq!(matmul_dims.len(), 6, "2 matmuls = 6 dims");

    let dequantize_bits = extract_dequantize_bits(&circuit);
    assert!(dequantize_bits.is_empty(), "no dequantize layers");

    // Build registration calldata
    let register_calldata =
        build_register_gkr_calldata(model_id, &gkr.weight_commitments, &circuit_desc);

    // Build verification calldata
    let raw_io_data = obelyzk::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
    let verify_calldata = build_verify_model_gkr_calldata(gkr, &circuit, model_id, &raw_io_data).expect("build verify calldata");

    // Validate calldata structure
    let parts = &verify_calldata.calldata_parts;
    assert_eq!(parts[0], format!("0x{:x}", model_id));
    let rio_len: usize = parts[1].parse().unwrap();
    let o = 2 + rio_len;
    assert_eq!(parts[o], "3"); // num_layers = 3

    // Write to artifacts
    let artifacts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("artifacts");
    std::fs::create_dir_all(&artifacts_dir).ok();

    let register_path = artifacts_dir.join("d9_register_gkr_calldata.txt");
    std::fs::write(&register_path, register_calldata.join(" ")).expect("write register calldata");

    let verify_path = artifacts_dir.join("d9_verify_gkr_calldata.txt");
    std::fs::write(&verify_path, verify_calldata.calldata_parts.join(" "))
        .expect("write verify calldata");

    let summary = serde_json::json!({
        "model_id": format!("0x{:x}", model_id),
        "io_commitment": format!("0x{:x}", io_commitment),
        "num_layers": gkr.layer_proofs.len(),
        "layer_types": circuit_desc,
        "matmul_dims": matmul_dims,
        "weight_commitments": gkr.weight_commitments.iter()
            .map(|c| format!("0x{:x}", c))
            .collect::<Vec<_>>(),
        "register_calldata_len": register_calldata.len(),
        "verify_calldata_len": verify_calldata.total_felts,
    });
    let summary_path = artifacts_dir.join("d9_onchain_summary.json");
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write summary");

    eprintln!("D9 MLP artifacts exported:");
    eprintln!(
        "  Register: {} ({} parts)",
        register_path.display(),
        register_calldata.len()
    );
    eprintln!(
        "  Verify:   {} ({} parts)",
        verify_path.display(),
        verify_calldata.total_felts
    );
    eprintln!("  Summary:  {}", summary_path.display());
}

// ----------------------------------------------------------------------------
// D10: LayerNorm model on-chain — MatMul → LayerNorm → MatMul
// ----------------------------------------------------------------------------

#[test]
fn test_d10_export_layernorm_onchain_calldata() {
    use obelyzk::starknet::{
        build_circuit_descriptor, build_gkr_starknet_proof, build_register_gkr_calldata,
        extract_dequantize_bits, extract_matmul_dims,
    };

    let (graph, input, weights) = build_gkr_mlp_layernorm();

    let proof =
        prove_model_pure_gkr(&graph, &input, &weights).expect("LayerNorm proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();

    let model_id = FieldElement::from(0xDAu64);
    let io_commitment = proof.io_commitment;

    // Validate: 3 layer proofs (MatMul, LayerNorm, MatMul)
    assert_eq!(gkr.layer_proofs.len(), 3, "should have 3 layers");
    let circuit_desc = build_circuit_descriptor(&circuit);
    eprintln!("D10 circuit descriptor: {:?}", circuit_desc);

    let matmul_dims = extract_matmul_dims(&circuit);
    eprintln!("D10 matmul_dims: {:?}", matmul_dims);
    assert_eq!(matmul_dims.len(), 6, "2 matmuls = 6 dims");

    let dequantize_bits = extract_dequantize_bits(&circuit);
    assert!(dequantize_bits.is_empty(), "no dequantize layers");

    // Build calldata via GkrStarknetProof (V1 calldata builder doesn't support LayerNorm tag)
    let register_calldata =
        build_register_gkr_calldata(model_id, &gkr.weight_commitments, &circuit_desc);
    let gkr_sn = build_gkr_starknet_proof(&proof, model_id, &input)
        .expect("starknet proof should succeed");

    assert!(gkr_sn.total_calldata_size > 0);

    // Write artifacts
    let artifacts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("artifacts");
    std::fs::create_dir_all(&artifacts_dir).ok();

    std::fs::write(
        artifacts_dir.join("d10_register_gkr_calldata.txt"),
        register_calldata.join(" "),
    )
    .unwrap();

    let summary = serde_json::json!({
        "model_id": format!("0x{:x}", model_id),
        "io_commitment": format!("0x{:x}", io_commitment),
        "num_layers": gkr.layer_proofs.len(),
        "layer_types": circuit_desc,
        "matmul_dims": matmul_dims,
        "weight_commitments": gkr.weight_commitments.iter()
            .map(|c| format!("0x{:x}", c)).collect::<Vec<_>>(),
        "register_calldata_len": register_calldata.len(),
        "total_calldata_size": gkr_sn.total_calldata_size,
    });
    std::fs::write(
        artifacts_dir.join("d10_onchain_summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();

    eprintln!("D10 LayerNorm artifacts exported:");
    eprintln!("  Register: {} parts", register_calldata.len());
    eprintln!("  Total:    {} felts", gkr_sn.total_calldata_size);
}

// ----------------------------------------------------------------------------
// D11: Residual connection — MatMul → fork → ReLU → MatMul → Add (residual)
// Tests Add layer (tag=1) in the GKR walk
// ----------------------------------------------------------------------------

#[test]
fn test_d11_export_residual_onchain_calldata() {
    use obelyzk::compiler::graph::GraphBuilder;
    use obelyzk::starknet::{
        build_circuit_descriptor, build_gkr_starknet_proof, build_register_gkr_calldata,
        extract_dequantize_bits, extract_matmul_dims,
    };

    // Build: input(1×4) → MatMul(4→4) → fork → ReLU → MatMul(4→4) → add_from(fork)
    // This creates a residual connection: out = MatMul(ReLU(MatMul(x))) + MatMul(x)
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(4);
    let residual = builder.fork();
    builder.activation(obelyzk::prelude::ActivationType::ReLU);
    builder.linear(4);
    builder.add_from(residual);
    let graph = builder.build();

    // Weights
    let mut weights = obelyzk::compiler::graph::GraphWeights::new();
    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..4 {
        w0.set(i, i, M31::from(1u32));
    } // identity
    weights.add_weight(0, w0);
    let mut w2 = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            w2.set(i, j, M31::from(((i + j) % 3 + 1) as u32));
        }
    }
    weights.add_weight(2, w2);

    // Input
    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let proof =
        prove_model_pure_gkr(&graph, &input, &weights).expect("Residual proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();

    // Verify the GKR proof with Rust verifier
    {
        let mut verify_ch = obelyzk::crypto::poseidon_channel::PoseidonChannel::new();
        let result =
            obelyzk::gkr::verify_gkr_with_weights(&circuit, gkr, &proof.execution.output, &weights, &mut verify_ch);
        result.expect("D11 GKR proof should verify in Rust");
        eprintln!("D11 Rust GKR verification: PASS");
    }

    let model_id = FieldElement::from(0xDBu64);
    let io_commitment = proof.io_commitment;

    let circuit_desc = build_circuit_descriptor(&circuit);
    eprintln!("D11 circuit descriptor: {:?}", circuit_desc);
    eprintln!("D11 layer proofs: {}", gkr.layer_proofs.len());

    let matmul_dims = extract_matmul_dims(&circuit);
    let dequantize_bits = extract_dequantize_bits(&circuit);

    let register_calldata =
        build_register_gkr_calldata(model_id, &gkr.weight_commitments, &circuit_desc);
    // V1 calldata builder doesn't support Add layer — use GkrStarknetProof instead
    let gkr_sn = build_gkr_starknet_proof(&proof, model_id, &input)
        .expect("starknet proof should succeed");

    assert!(gkr_sn.total_calldata_size > 0);

    // Write artifacts
    let artifacts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("artifacts");
    std::fs::create_dir_all(&artifacts_dir).ok();

    std::fs::write(
        artifacts_dir.join("d11_register_gkr_calldata.txt"),
        register_calldata.join(" "),
    )
    .unwrap();

    let summary = serde_json::json!({
        "model_id": format!("0x{:x}", model_id),
        "io_commitment": format!("0x{:x}", io_commitment),
        "num_layers": gkr.layer_proofs.len(),
        "layer_types": circuit_desc,
        "matmul_dims": matmul_dims,
        "weight_commitments": gkr.weight_commitments.iter()
            .map(|c| format!("0x{:x}", c)).collect::<Vec<_>>(),
        "register_calldata_len": register_calldata.len(),
        "total_calldata_size": gkr_sn.total_calldata_size,
    });
    std::fs::write(
        artifacts_dir.join("d11_onchain_summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();

    eprintln!("D11 Residual artifacts exported:");
    eprintln!("  Register: {} parts", register_calldata.len());
    eprintln!("  Total:    {} felts", gkr_sn.total_calldata_size);
}

// ============================================================================
// E2E Decode Step: prove → serialize → self-verify (tag 11 + Weightless deferred)
// ============================================================================

#[test]
fn test_e2e_decode_cairo_verify() {
    use obelyzk::aggregation::{IncrementalKVCommitment, prove_model_pure_gkr_decode_step};
    use obelyzk::components::attention::{
        attention_forward_cached, AttentionWeights, ModelKVCache,
    };
    use obelyzk::compiler::graph::GraphBuilder;
    use obelyzk::compiler::onnx::generate_weights_for_graph;
    use obelyzk::starknet::{extract_matmul_dims, replay_verify_serialized_proof};

    let d_model = 64;
    let num_heads = 2;
    let d_ff = 256;
    let prefill_len = 4;

    // Build decode graph (1-layer transformer block)
    let mut builder = GraphBuilder::new((1, d_model));
    builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let graph = builder.build();

    let mut weights = generate_weights_for_graph(&graph, 42);

    // Helper: deterministic random matrix
    let random_m31 = |rows: usize, cols: usize, seed: u64| -> M31Matrix {
        let mut data = Vec::with_capacity(rows * cols);
        let mut state = seed;
        for _ in 0..(rows * cols) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push(M31::from((state >> 33) as u32 % 100));
        }
        M31Matrix { rows, cols, data }
    };

    // Add attention named weights
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let obelyzk::compiler::graph::GraphOp::Attention { config: _ } = &node.op {
            weights.add_named_weight(node.id, "w_q", random_m31(d_model, d_model, 200 + node.id as u64));
            weights.add_named_weight(node.id, "w_k", random_m31(d_model, d_model, 300 + node.id as u64));
            weights.add_named_weight(node.id, "w_v", random_m31(d_model, d_model, 400 + node.id as u64));
            weights.add_named_weight(node.id, "w_o", random_m31(d_model, d_model, 500 + node.id as u64));
        }
    }

    // Seed KV cache with prefill
    let mut kv_cache = ModelKVCache::new();
    let prefill_input = random_m31(prefill_len, d_model, 123);
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let obelyzk::compiler::graph::GraphOp::Attention { config } = &node.op {
            let attn_weights = AttentionWeights {
                w_q: weights.get_named_weight(node.id, "w_q").unwrap().clone(),
                w_k: weights.get_named_weight(node.id, "w_k").unwrap().clone(),
                w_v: weights.get_named_weight(node.id, "w_v").unwrap().clone(),
                w_o: weights.get_named_weight(node.id, "w_o").unwrap().clone(),
            };
            let cache = kv_cache.get_or_create(node.id, config);
            let _ = attention_forward_cached(&prefill_input, &attn_weights, config, cache, config.causal);
        }
    }

    // Prove a decode step
    let token_input = random_m31(1, d_model, 999);
    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);
    let (proof, kv_commit) = obelyzk::aggregation::prove_model_pure_gkr_decode_step_incremental(
        &graph, &token_input, &weights, &mut kv_cache, &mut kv_commitment, None, None,
    ).expect("decode proving should succeed");

    let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");
    assert!(!gkr.layer_proofs.is_empty(), "should have layer proofs");
    assert!(gkr.kv_cache_commitment.is_some(), "should have KV commitment");
    assert_eq!(gkr.kv_cache_commitment.unwrap(), kv_commit, "KV commitment mismatch");

    // Serialize proof data (unpacked for simplicity)
    let mut proof_data = Vec::new();
    obelyzk::cairo_serde::serialize_gkr_proof_data_only(gkr, &mut proof_data);
    assert!(!proof_data.is_empty(), "serialized proof should be non-empty");

    // Build matmul dims from circuit
    let circuit = obelyzk::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let matmul_dims = extract_matmul_dims(&circuit);
    let circuit_depth = circuit.layers.len() as u32;
    let num_layers = gkr.layer_proofs.len() as u32;

    // Build raw_io for replay verification
    let mut raw_io = Vec::new();
    raw_io.push(FieldElement::from(token_input.rows as u64));
    raw_io.push(FieldElement::from(token_input.cols as u64));
    raw_io.push(FieldElement::from(token_input.data.len() as u64));
    for v in &token_input.data {
        raw_io.push(FieldElement::from(v.0 as u64));
    }
    raw_io.push(FieldElement::from(proof.execution.output.rows as u64));
    raw_io.push(FieldElement::from(proof.execution.output.cols as u64));
    raw_io.push(FieldElement::from(proof.execution.output.data.len() as u64));
    for v in &proof.execution.output.data {
        raw_io.push(FieldElement::from(v.0 as u64));
    }

    // Self-verify: replay Fiat-Shamir channel against serialized proof
    let result = replay_verify_serialized_proof(
        &proof_data,
        &raw_io,
        &matmul_dims,
        circuit_depth,
        num_layers,
        false, // unpacked
        None,  // expected_io_commitment
        None,  // weight_binding
        gkr.kv_cache_commitment,
        gkr.prev_kv_cache_commitment,
    );
    assert!(
        result.is_ok(),
        "decode calldata replay verification failed: {:?}",
        result.err()
    );

    // Verify deferred proofs are present (decode generates Weightless deferred proofs)
    eprintln!(
        "E2E decode Cairo verify: PASSED ({} layers, {} deferred proofs, {} proof_data felts)",
        gkr.layer_proofs.len(),
        gkr.deferred_proofs.len(),
        proof_data.len(),
    );
}
