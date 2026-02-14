//! Transcript test vector generation for cross-language Fiat-Shamir verification.
//!
//! These tests generate the exact channel digests at each step of GKR verification,
//! which Engineer A (Cairo verifier) uses to verify their Poseidon channel matches
//! the Rust implementation at every checkpoint.
//!
//! Each test vector includes:
//! - Model description (architecture, weights, input)
//! - GKR proof calldata (felt252 hex array)
//! - Channel digest checkpoints at each verification step
//!
//! If ANY checkpoint disagrees between Rust and Cairo, all GKR verification will fail.
//! This is the SP1 sync point.

use stwo::core::fields::m31::M31;
use stwo_ml::compiler::graph::{GraphBuilder, GraphWeights};
use stwo_ml::components::activation::ActivationType;
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::crypto::poseidon_channel::PoseidonChannel;

/// A checkpoint in the Fiat-Shamir transcript.
#[derive(Debug)]
#[allow(dead_code)]
struct TranscriptCheckpoint {
    label: String,
    digest: starknet_ff::FieldElement,
    n_draws: u32,
}

/// A complete transcript test vector for one model.
#[derive(Debug)]
struct TranscriptTestVector {
    description: String,
    num_layers: usize,
    checkpoints: Vec<TranscriptCheckpoint>,
    gkr_calldata: Vec<starknet_ff::FieldElement>,
    input_data: Vec<u32>,
    output_data: Vec<u32>,
}

/// Build a simple 1×4 → 4 → ReLU → 4 → 2 MLP with deterministic weights.
fn build_mlp_3layer() -> (
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

    // Layer 0: 4×4 weight matrix (MatMul)
    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
        }
    }
    weights.add_weight(0, w0);

    // Layer 2: 4×2 weight matrix (MatMul)
    let mut w2 = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w2.set(i, j, M31::from((i + j + 1) as u32));
        }
    }
    weights.add_weight(2, w2);

    (graph, input, weights)
}

/// Build a matmul-only model: 1×4 → 2.
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

/// Generate transcript checkpoints for GKR verification.
///
/// Captures key channel states:
/// - `initial`: zero state before any mixing
/// - `after_seed`: after mixing circuit dimensions (d, input_rows, input_cols)
/// - `final_after_verify_gkr`: after full GKR verification completes
///
/// The seed checkpoints are replayed manually to give Engineer A granular
/// verification points. The final digest comes from `verify_gkr()` itself.
fn generate_checkpoints(
    circuit: &stwo_ml::gkr::LayeredCircuit,
    proof: &stwo_ml::gkr::GKRProof,
    output: &M31Matrix,
) -> Vec<TranscriptCheckpoint> {
    let mut checkpoints = Vec::new();

    // ===== Checkpoint 0: initial state =====
    checkpoints.push(TranscriptCheckpoint {
        label: "initial".to_string(),
        digest: starknet_ff::FieldElement::ZERO,
        n_draws: 0,
    });

    // ===== Seed channel (replayable by Cairo) =====
    let mut seed_channel = PoseidonChannel::new();
    let d = circuit.layers.len();
    seed_channel.mix_u64(d as u64);

    checkpoints.push(TranscriptCheckpoint {
        label: "after_mix_d".to_string(),
        digest: seed_channel.digest(),
        n_draws: 0,
    });

    seed_channel.mix_u64(circuit.input_shape.0 as u64);

    checkpoints.push(TranscriptCheckpoint {
        label: "after_mix_input_rows".to_string(),
        digest: seed_channel.digest(),
        n_draws: 0,
    });

    seed_channel.mix_u64(circuit.input_shape.1 as u64);

    checkpoints.push(TranscriptCheckpoint {
        label: "after_seed".to_string(),
        digest: seed_channel.digest(),
        n_draws: 0,
    });

    // ===== Final state: run full verify_gkr and capture digest =====
    let mut verify_channel = PoseidonChannel::new();
    let _result = stwo_ml::gkr::verify_gkr(circuit, proof, output, &mut verify_channel);

    checkpoints.push(TranscriptCheckpoint {
        label: "final_after_verify_gkr".to_string(),
        digest: verify_channel.digest(),
        n_draws: 0,
    });

    checkpoints
}

/// Helper: mix a SecureField into the channel (4x individual mix_u64).
#[allow(dead_code)]
fn mix_secure_field(channel: &mut PoseidonChannel, v: stwo::core::fields::qm31::SecureField) {
    channel.mix_u64(v.0 .0 .0 as u64);
    channel.mix_u64(v.0 .1 .0 as u64);
    channel.mix_u64(v.1 .0 .0 as u64);
    channel.mix_u64(v.1 .1 .0 as u64);
}

/// Print a test vector as a Cairo-consumable format.
fn print_test_vector(tv: &TranscriptTestVector) {
    eprintln!("=== Transcript Test Vector: {} ===", tv.description);
    eprintln!("  Layers: {}", tv.num_layers);
    eprintln!("  Calldata size: {} felts", tv.gkr_calldata.len());
    eprintln!("  Input: {:?}", tv.input_data);
    eprintln!("  Output: {:?}", tv.output_data);
    eprintln!();
    for cp in &tv.checkpoints {
        eprintln!(
            "  [{}] digest = 0x{:064x}",
            cp.label,
            cp.digest,
        );
    }
    eprintln!();
    eprintln!("  // Cairo test vector (felt252 hex for calldata):");
    eprintln!("  // First 10 calldata elements:");
    for (i, &felt) in tv.gkr_calldata.iter().take(10).enumerate() {
        eprintln!("  //   calldata[{}] = 0x{:064x}", i, felt);
    }
    eprintln!("===");
}

// ============================================================================
// Test: Matmul-only model transcript
// ============================================================================

#[test]
fn test_transcript_vector_matmul_only() {
    let (graph, input, weights) = build_matmul_only();

    // Prove via GKR
    let proof = stwo_ml::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
        .expect("GKR proving should succeed");
    let gkr_proof = proof.gkr_proof.as_ref().expect("should have GKR proof");

    // Build circuit for verification
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph)
        .expect("circuit compilation should succeed");

    // Generate checkpoints
    let checkpoints = generate_checkpoints(&circuit, gkr_proof, &proof.execution.output);

    // Serialize calldata
    let mut gkr_calldata = Vec::new();
    stwo_ml::cairo_serde::serialize_gkr_model_proof(gkr_proof, &mut gkr_calldata);

    let tv = TranscriptTestVector {
        description: "Matmul-only 1x4 -> 2".to_string(),
        num_layers: gkr_proof.layer_proofs.len(),
        checkpoints,
        gkr_calldata,
        input_data: input.data.iter().map(|m| m.0).collect(),
        output_data: proof.execution.output.data.iter().map(|m| m.0).collect(),
    };

    print_test_vector(&tv);

    // Verify the initial checkpoint is zero
    assert_eq!(
        tv.checkpoints[0].digest,
        starknet_ff::FieldElement::ZERO,
        "initial digest must be zero"
    );

    // Verify seed checkpoint is non-zero (channel was mixed)
    assert_ne!(
        tv.checkpoints[1].digest,
        starknet_ff::FieldElement::ZERO,
        "after_seed digest must be non-zero"
    );

    // Verify final checkpoint exists and is non-zero
    let final_cp = tv.checkpoints.last().unwrap();
    assert_eq!(final_cp.label, "final_after_verify_gkr");
    assert_ne!(final_cp.digest, starknet_ff::FieldElement::ZERO);

    // Verify the GKR proof actually verifies
    let mut verify_channel = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr(&circuit, gkr_proof, &proof.execution.output, &mut verify_channel);
    assert!(result.is_ok(), "GKR verification should pass: {:?}", result.err());

    // Final digest from verification must match our checkpoint
    assert_eq!(
        verify_channel.digest(),
        final_cp.digest,
        "final digest must match verify_gkr output"
    );
}

// ============================================================================
// Test: MLP with activation transcript
// ============================================================================

#[test]
fn test_transcript_vector_mlp_relu() {
    let (graph, input, weights) = build_mlp_3layer();

    // Prove via GKR
    let proof = stwo_ml::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
        .expect("GKR proving should succeed");
    let gkr_proof = proof.gkr_proof.as_ref().expect("should have GKR proof");

    // Build circuit
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph)
        .expect("circuit compilation should succeed");

    // Generate checkpoints
    let checkpoints = generate_checkpoints(&circuit, gkr_proof, &proof.execution.output);

    // Serialize calldata
    let mut gkr_calldata = Vec::new();
    stwo_ml::cairo_serde::serialize_gkr_model_proof(gkr_proof, &mut gkr_calldata);

    let tv = TranscriptTestVector {
        description: "MLP 1x4 -> Linear(4) -> ReLU -> Linear(2)".to_string(),
        num_layers: gkr_proof.layer_proofs.len(),
        checkpoints,
        gkr_calldata: gkr_calldata.clone(),
        input_data: input.data.iter().map(|m| m.0).collect(),
        output_data: proof.execution.output.data.iter().map(|m| m.0).collect(),
    };

    print_test_vector(&tv);

    // Should have 3 layer proofs: MatMul + Activation + MatMul
    assert_eq!(tv.num_layers, 3, "MLP should have 3 layer proofs");

    // Verify all checkpoints are present
    assert!(tv.checkpoints.len() >= 4, "should have at least 4 checkpoints");

    // Verify the GKR proof actually verifies
    let mut verify_channel = PoseidonChannel::new();
    let result = stwo_ml::gkr::verify_gkr(&circuit, gkr_proof, &proof.execution.output, &mut verify_channel);
    assert!(result.is_ok(), "GKR verification should pass: {:?}", result.err());

    // Final digest must match
    let final_cp = tv.checkpoints.last().unwrap();
    assert_eq!(
        verify_channel.digest(),
        final_cp.digest,
        "final digest must match verify_gkr output"
    );

    // Verify calldata format: first felt is num_layers
    assert_eq!(
        gkr_calldata[0],
        starknet_ff::FieldElement::from(3u64),
        "first felt should be num_layers = 3"
    );
}

// ============================================================================
// Test: Deterministic — same model produces same transcript
// ============================================================================

#[test]
fn test_transcript_deterministic() {
    let (graph, input, weights) = build_matmul_only();

    // Run twice
    let proof1 = stwo_ml::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
        .expect("proving should succeed");
    let proof2 = stwo_ml::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
        .expect("proving should succeed");

    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph)
        .expect("circuit should compile");

    let gkr1 = proof1.gkr_proof.as_ref().unwrap();
    let gkr2 = proof2.gkr_proof.as_ref().unwrap();

    // Verify both
    let mut ch1 = PoseidonChannel::new();
    let mut ch2 = PoseidonChannel::new();
    let r1 = stwo_ml::gkr::verify_gkr(&circuit, gkr1, &proof1.execution.output, &mut ch1);
    let r2 = stwo_ml::gkr::verify_gkr(&circuit, gkr2, &proof2.execution.output, &mut ch2);

    assert!(r1.is_ok());
    assert!(r2.is_ok());

    // Transcripts must be identical (same model, same input → deterministic proof → same digest)
    assert_eq!(
        ch1.digest(), ch2.digest(),
        "deterministic proofs must produce identical transcripts"
    );

    // Serialized proofs must be identical
    let mut buf1 = Vec::new();
    let mut buf2 = Vec::new();
    stwo_ml::cairo_serde::serialize_gkr_model_proof(gkr1, &mut buf1);
    stwo_ml::cairo_serde::serialize_gkr_model_proof(gkr2, &mut buf2);
    assert_eq!(buf1, buf2, "deterministic proofs must serialize identically");
}

// ============================================================================
// Test: Export hardcoded digest for Engineer A
// ============================================================================

#[test]
fn test_export_seed_digest_for_cairo() {
    // This test produces a hardcoded digest that Engineer A can verify in Cairo.
    // Model: single matmul 1x4 -> 2
    // Seed: mix_u64(d), mix_u64(input_rows), mix_u64(input_cols)

    let mut channel = PoseidonChannel::new();

    // For a 1-layer circuit with input shape (1, 4):
    // The LayeredCircuit::from_graph adds layers: [Input, MatMul]
    // So d = 2 for a single matmul node (input layer + matmul layer)
    // But let's verify by actually building the circuit:
    let (graph, _input, _weights) = build_matmul_only();
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph)
        .expect("circuit should compile");

    let d = circuit.layers.len();
    let (in_rows, in_cols) = circuit.input_shape;

    eprintln!("=== Seed Digest Vector for Cairo ===");
    eprintln!("  d (num_layers) = {}", d);
    eprintln!("  input_rows = {}", in_rows);
    eprintln!("  input_cols = {}", in_cols);

    channel.mix_u64(d as u64);
    let digest_after_d = channel.digest();
    eprintln!("  digest after mix_u64({}) = 0x{:064x}", d, digest_after_d);

    channel.mix_u64(in_rows as u64);
    let digest_after_rows = channel.digest();
    eprintln!("  digest after mix_u64({}) = 0x{:064x}", in_rows, digest_after_rows);

    channel.mix_u64(in_cols as u64);
    let digest_after_cols = channel.digest();
    eprintln!("  digest after mix_u64({}) = 0x{:064x}", in_cols, digest_after_cols);
    eprintln!("===");

    // The seed digest should be deterministic and non-zero
    assert_ne!(digest_after_cols, starknet_ff::FieldElement::ZERO);

    // Also export a simple mix_u64(42) for basic sanity checking
    let mut ch_basic = PoseidonChannel::new();
    ch_basic.mix_u64(42);
    eprintln!("  Basic: mix_u64(42) digest = 0x{:064x}", ch_basic.digest());
    assert_ne!(ch_basic.digest(), starknet_ff::FieldElement::ZERO);
}
