//! E2E malicious proof rejection tests.
//!
//! Exercises the full prove → tamper → verify pipeline to confirm that
//! `verify_gkr_with_weights()` catches:
//!
//! 1. Tampered sumcheck round polynomials (single-layer and multi-layer)
//! 2. Forged weight claims (RLC mismatch detection)
//! 3. Wrong activation outputs and swapped activation types
//! 4. Tampered final evaluations in multi-layer chains
//! 5. Tampered deferred proofs (residual DAG branches)

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use stwo_ml::aggregation::prove_model_pure_gkr;
use stwo_ml::compiler::graph::{GraphBuilder, GraphWeights};
use stwo_ml::components::activation::ActivationType;
use stwo_ml::crypto::poseidon_channel::PoseidonChannel;
use stwo_ml::gkr::types::LayerProof;
use stwo_ml::gkr::LayeredCircuit;
use stwo_ml::prelude::*;

// ============================================================================
// Helpers
// ============================================================================

/// Single MatMul: 1×4 @ 4×2
fn build_matmul_only() -> (
    stwo_ml::compiler::graph::ComputationGraph,
    M31Matrix,
    GraphWeights,
) {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = GraphWeights::new();
    let mut w = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }
    }
    weights.add_weight(0, w);

    (graph, input, weights)
}

/// MLP: 1×4 → Linear(4) → ReLU → Linear(2)
fn build_mlp_relu() -> (
    stwo_ml::compiler::graph::ComputationGraph,
    M31Matrix,
    GraphWeights,
) {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(4).activation(ActivationType::ReLU).linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = GraphWeights::new();
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

/// Deep MLP: 1×4 → Linear(4) → ReLU → Linear(4) → GELU → Linear(2)
fn build_deep_mlp() -> (
    stwo_ml::compiler::graph::ComputationGraph,
    M31Matrix,
    GraphWeights,
) {
    let mut builder = GraphBuilder::new((1, 4));
    builder
        .linear(4)
        .activation(ActivationType::ReLU)
        .linear(4)
        .activation(ActivationType::GELU)
        .linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = GraphWeights::new();
    // Node 0: 4×4
    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            w0.set(i, j, M31::from(((i * 3 + j * 5) % 11 + 1) as u32));
        }
    }
    weights.add_weight(0, w0);
    // Node 2: 4×4
    let mut w2 = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            w2.set(i, j, M31::from(((i * 7 + j * 2) % 13 + 1) as u32));
        }
    }
    weights.add_weight(2, w2);
    // Node 4: 4×2
    let mut w4 = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w4.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }
    }
    weights.add_weight(4, w4);

    (graph, input, weights)
}

/// Residual DAG: x → MatMul(0) → MatMul(1) → Add(skip from MatMul(0))
fn build_residual_dag() -> (
    stwo_ml::compiler::graph::ComputationGraph,
    M31Matrix,
    GraphWeights,
) {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(4);
    let residual = builder.fork();
    builder.linear(4);
    builder.add_from(residual);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = GraphWeights::new();
    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..16 {
        w0.data[i] = M31::from(((i % 7) + 1) as u32);
    }
    weights.add_weight(0, w0);
    let mut w1 = M31Matrix::new(4, 4);
    for i in 0..16 {
        w1.data[i] = M31::from((((i * 3) % 11) + 1) as u32);
    }
    weights.add_weight(1, w1);

    (graph, input, weights)
}

// ============================================================================
// 1. Tampered Sumcheck Round Polynomials
// ============================================================================

/// Single-layer matmul: tamper first round polynomial coefficient.
#[test]
fn test_e2e_tampered_round_poly_single_matmul() {
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    match &mut gkr.layer_proofs[0] {
        LayerProof::MatMul { round_polys, .. } => {
            round_polys[0].c0 =
                round_polys[0].c0 + QM31::from_u32_unchecked(1, 0, 0, 0);
        }
        _ => panic!("expected MatMul proof"),
    }

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered round poly must be rejected");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("sumcheck") || msg.contains("mismatch") || msg.contains("round"),
        "error should mention sumcheck failure, got: {msg}"
    );
}

/// Multi-layer MLP: tamper the SECOND matmul's round poly (deeper in the chain).
/// This tests that tampering mid-chain is detected, not just the first layer.
#[test]
fn test_e2e_tampered_round_poly_deep_chain() {
    let (graph, input, weights) = build_deep_mlp();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    // Find the second MatMul layer proof and tamper it
    let mut matmul_count = 0;
    for lp in &mut gkr.layer_proofs {
        if let LayerProof::MatMul { round_polys, .. } = lp {
            matmul_count += 1;
            if matmul_count == 2 {
                round_polys[0].c0 =
                    round_polys[0].c0 + QM31::from_u32_unchecked(0, 1, 0, 0);
                break;
            }
        }
    }
    assert!(matmul_count >= 2, "deep MLP should have at least 2 MatMul layers");

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "mid-chain tampered round poly must be rejected");
}

/// Multi-layer MLP: tamper the LAST round poly of a matmul (not the first).
/// Ensures every round polynomial is checked, not just the first one.
#[test]
fn test_e2e_tampered_last_round_poly() {
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    match &mut gkr.layer_proofs[0] {
        LayerProof::MatMul { round_polys, .. } => {
            let last = round_polys.len() - 1;
            round_polys[last].c1 =
                round_polys[last].c1 + QM31::from_u32_unchecked(0, 0, 1, 0);
        }
        _ => panic!("expected MatMul proof"),
    }

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered last round poly must be rejected");
}

// ============================================================================
// 2. Forged Weight Claims
// ============================================================================

/// Forge weight_claims[0].expected_value — RLC binding detects the mismatch.
#[test]
fn test_e2e_forged_weight_claim_single_matmul() {
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    assert!(!gkr.weight_claims.is_empty(), "matmul must have weight claims");
    gkr.weight_claims[0].expected_value =
        gkr.weight_claims[0].expected_value + QM31::from_u32_unchecked(42, 0, 0, 0);

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "forged weight claim must be rejected");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("weight") || msg.contains("mismatch") || msg.contains("RLC"),
        "error should mention weight binding failure, got: {msg}"
    );
}

/// Forge weight claim in a multi-layer MLP — tamper the second matmul's claim.
#[test]
fn test_e2e_forged_weight_claim_mlp() {
    let (graph, input, weights) = build_mlp_relu();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    assert!(
        gkr.weight_claims.len() >= 2,
        "MLP with 2 matmuls should have >= 2 weight claims"
    );
    // Tamper the second weight claim
    gkr.weight_claims[1].expected_value =
        gkr.weight_claims[1].expected_value + QM31::from_u32_unchecked(1, 2, 3, 4);

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "forged second weight claim must be rejected");
}

/// Forge a weight claim's eval_point — changes which MLE point is evaluated,
/// causing expected vs actual divergence.
#[test]
fn test_e2e_forged_weight_eval_point() {
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    assert!(!gkr.weight_claims.is_empty());
    // Shift the first coordinate of the eval point
    gkr.weight_claims[0].eval_point[0] =
        gkr.weight_claims[0].eval_point[0] + QM31::from_u32_unchecked(1, 0, 0, 0);

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "forged eval point must be rejected");
}

/// Forge weight_node_id to reference a different (wrong) weight matrix.
#[test]
fn test_e2e_forged_weight_node_id() {
    let (graph, input, weights) = build_mlp_relu();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    assert!(gkr.weight_claims.len() >= 2);
    // Swap weight_node_id: point claim 0 at weight matrix 1's node_id
    let other_id = gkr.weight_claims[1].weight_node_id;
    gkr.weight_claims[0].weight_node_id = other_id;

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    // Should fail: either the verifier detects the wrong node_id ordering,
    // or the MLE evaluation against the wrong matrix produces a mismatch.
    assert!(result.is_err(), "swapped weight_node_id must be rejected");
}

// ============================================================================
// 3. Wrong Activations
// ============================================================================

/// Swap activation type in proof (ReLU → GELU) — the verifier checks that
/// the proof's activation type matches the circuit's activation type.
#[test]
fn test_e2e_wrong_activation_type() {
    let (graph, input, weights) = build_mlp_relu();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    let mut found = false;
    for lp in &mut gkr.layer_proofs {
        if let LayerProof::Activation {
            activation_type, ..
        } = lp
        {
            // Swap ReLU for GELU — circuit says ReLU, proof claims GELU
            *activation_type = ActivationType::GELU;
            found = true;
            break;
        }
    }
    assert!(found, "MLP should have an Activation layer proof");

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(
        result.is_err(),
        "wrong activation type must be rejected"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("activation type") || msg.contains("mismatch"),
        "error should mention activation type mismatch, got: {msg}"
    );
}

/// Tamper activation input_eval — breaks claim chaining between layers.
#[test]
fn test_e2e_tampered_activation_input_eval() {
    let (graph, input, weights) = build_mlp_relu();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    let mut found = false;
    for lp in &mut gkr.layer_proofs {
        if let LayerProof::Activation { input_eval, .. } = lp {
            *input_eval = *input_eval + QM31::from_u32_unchecked(0, 0, 0, 1);
            found = true;
            break;
        }
    }
    assert!(found, "MLP should have an Activation layer proof");

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered activation input_eval must be rejected");
}

/// Tamper activation LogUp multiplicities (if present).
/// In GKR mode, logup_proof may be None — skip in that case.
#[test]
fn test_e2e_tampered_activation_logup_multiplicities() {
    let (graph, input, weights) = build_mlp_relu();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    let mut found = false;
    for lp in &mut gkr.layer_proofs {
        if let LayerProof::Activation { logup_proof, .. } = lp {
            if let Some(ref mut logup) = logup_proof {
                if !logup.multiplicities.is_empty() {
                    logup.multiplicities[0] = logup.multiplicities[0].wrapping_add(1);
                    found = true;
                    break;
                }
            }
        }
    }

    if !found {
        // GKR mode skips LogUp — test is not applicable
        eprintln!("SKIP: activation LogUp is None in GKR mode (expected)");
        return;
    }

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered LogUp multiplicities must be rejected");
}

// ============================================================================
// 4. Tampered Final Evaluations
// ============================================================================

/// Tamper final_a_eval in a matmul — breaks the input claim chain.
#[test]
fn test_e2e_tampered_final_a_eval() {
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    match &mut gkr.layer_proofs[0] {
        LayerProof::MatMul { final_a_eval, .. } => {
            *final_a_eval = *final_a_eval + QM31::from_u32_unchecked(1, 0, 0, 0);
        }
        _ => panic!("expected MatMul proof"),
    }

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered final_a_eval must be rejected");
}

/// Tamper final_b_eval in a matmul — breaks weight binding.
#[test]
fn test_e2e_tampered_final_b_eval() {
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    match &mut gkr.layer_proofs[0] {
        LayerProof::MatMul { final_b_eval, .. } => {
            *final_b_eval = *final_b_eval + QM31::from_u32_unchecked(0, 1, 0, 0);
        }
        _ => panic!("expected MatMul proof"),
    }

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered final_b_eval must be rejected");
}

/// Tamper final_a_eval in the LAST matmul of a deep chain.
/// This is the eval that reduces to the model input claim.
#[test]
fn test_e2e_tampered_final_eval_deep_chain() {
    let (graph, input, weights) = build_deep_mlp();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    // Find the last MatMul and tamper its final_a_eval
    let mut last_matmul_idx = None;
    for (i, lp) in gkr.layer_proofs.iter().enumerate() {
        if matches!(lp, LayerProof::MatMul { .. }) {
            last_matmul_idx = Some(i);
        }
    }
    let idx = last_matmul_idx.expect("deep MLP should have MatMul layers");
    match &mut gkr.layer_proofs[idx] {
        LayerProof::MatMul { final_a_eval, .. } => {
            *final_a_eval = *final_a_eval + QM31::from_u32_unchecked(0, 0, 1, 0);
        }
        _ => unreachable!(),
    }

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered final eval in last layer must be rejected");
}

// ============================================================================
// 5. Tampered Deferred Proofs (Residual DAG)
// ============================================================================

/// Tamper a deferred proof's round poly in a residual DAG.
#[test]
fn test_e2e_tampered_deferred_round_poly() {
    let (graph, input, weights) = build_residual_dag();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    assert!(
        !gkr.deferred_proofs.is_empty(),
        "residual DAG should have deferred proofs"
    );

    // Tamper the first deferred proof's layer proof round poly
    let deferred = &mut gkr.deferred_proofs[0];
    if let LayerProof::MatMul { round_polys, .. } = &mut deferred.layer_proof {
        round_polys[0].c0 =
            round_polys[0].c0 + QM31::from_u32_unchecked(1, 0, 0, 0);
    } else {
        panic!("expected MatMul deferred proof");
    }

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered deferred round poly must be rejected");
}

/// Forge a deferred proof's weight claim expected_value.
#[test]
fn test_e2e_forged_deferred_weight_claim() {
    let (graph, input, weights) = build_residual_dag();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    assert!(
        !gkr.deferred_proofs.is_empty(),
        "residual DAG should have deferred proofs"
    );

    // Find a deferred proof with a weight claim and tamper it
    let mut found = false;
    for deferred in &mut gkr.deferred_proofs {
        if let stwo_ml::gkr::types::DeferredProofKind::MatMul {
            ref mut weight_claim,
            ..
        } = deferred.kind
        {
            weight_claim.expected_value =
                weight_claim.expected_value + QM31::from_u32_unchecked(7, 0, 0, 0);
            found = true;
            break;
        }
    }
    assert!(found, "residual DAG deferred proof should have a weight claim");

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "forged deferred weight claim must be rejected");
}

// ============================================================================
// 6. Tampered Output / Input Claims
// ============================================================================

/// Tamper the model output — verifier should detect wrong output MLE.
#[test]
fn test_e2e_tampered_output() {
    let (graph, input, weights) = build_mlp_relu();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();

    let mut bad_output = proof.execution.output.clone();
    bad_output.data[0] = M31::from(bad_output.data[0].0.wrapping_add(1));

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr, &bad_output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered output must be rejected");
}

/// Tamper the input claim value stored in the proof.
#[test]
fn test_e2e_tampered_input_claim() {
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");
    let mut gkr = proof.gkr_proof.as_ref().unwrap().clone();

    gkr.input_claim.value =
        gkr.input_claim.value + QM31::from_u32_unchecked(1, 0, 0, 0);

    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, &gkr, &proof.execution.output, &weights, &mut ch,
    );
    assert!(result.is_err(), "tampered input claim must be rejected");
}

// ============================================================================
// 7. Positive Control — valid proofs pass
// ============================================================================

/// Sanity check: untampered proofs pass verification for all model types.
#[test]
fn test_e2e_valid_proofs_pass() {
    // Single matmul
    let (graph, input, weights) = build_matmul_only();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).unwrap();
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr, &proof.execution.output, &weights, &mut ch,
    )
    .expect("valid single matmul proof should pass");

    // MLP with ReLU
    let (graph, input, weights) = build_mlp_relu();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).unwrap();
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr, &proof.execution.output, &weights, &mut ch,
    )
    .expect("valid MLP proof should pass");

    // Deep MLP
    let (graph, input, weights) = build_deep_mlp();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).unwrap();
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr, &proof.execution.output, &weights, &mut ch,
    )
    .expect("valid deep MLP proof should pass");

    // Residual DAG
    let (graph, input, weights) = build_residual_dag();
    let proof = prove_model_pure_gkr(&graph, &input, &weights).unwrap();
    let gkr = proof.gkr_proof.as_ref().unwrap();
    let circuit = LayeredCircuit::from_graph(&graph).unwrap();
    let mut ch = PoseidonChannel::new();
    stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr, &proof.execution.output, &weights, &mut ch,
    )
    .expect("valid residual DAG proof should pass");
}
