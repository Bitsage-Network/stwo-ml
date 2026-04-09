//! End-to-end full pipeline integration test.
//!
//! Exercises the complete path from model construction through on-chain
//! proof generation and calldata assembly:
//!
//! ```text
//! GraphBuilder → prove_model_aggregated_onchain() → verify
//!   → prepare_model_registration() → register_model_calldata()
//!   → build_starknet_proof_onchain() → assert consistency
//! ```
//!
//! No actual Starknet submission — validates everything up to calldata.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;

use stwo_ml::aggregation::{
    compute_io_commitment, prove_model_aggregated_onchain, prove_model_aggregated_onchain_gkr,
    verify_aggregated_model_proof_onchain,
};
use stwo_ml::cairo_serde::DirectProofMetadata;
use stwo_ml::prelude::*;
use stwo_ml::starknet::{
    build_starknet_proof_direct, build_starknet_proof_onchain, compute_weight_commitment,
    estimate_gas_from_proof, prepare_model_registration, register_model_calldata,
    register_model_calldata_sumcheck,
};

/// Build a 3-layer MLP: linear(8) → relu → linear(8) → relu → linear(4).
fn build_e2e_mlp() -> (ComputationGraph, M31Matrix, GraphWeights) {
    let mut builder = GraphBuilder::new((1, 8));
    builder
        .linear(8)
        .activation(ActivationType::ReLU)
        .linear(8)
        .activation(ActivationType::ReLU)
        .linear(4);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 8);
    for j in 0..8 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = GraphWeights::new();
    // Node 0: linear(8→8)
    let mut w0 = M31Matrix::new(8, 8);
    for i in 0..8 {
        for j in 0..8 {
            w0.set(i, j, M31::from(((i * 3 + j * 5) % 11 + 1) as u32));
        }
    }
    weights.add_weight(0, w0);
    // Node 2: linear(8→8)
    let mut w2 = M31Matrix::new(8, 8);
    for i in 0..8 {
        for j in 0..8 {
            w2.set(i, j, M31::from(((i * 7 + j * 2) % 13 + 1) as u32));
        }
    }
    weights.add_weight(2, w2);
    // Node 4: linear(8→4)
    let mut w4 = M31Matrix::new(8, 4);
    for i in 0..8 {
        for j in 0..4 {
            w4.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }
    }
    weights.add_weight(4, w4);

    (graph, input, weights)
}

/// Full pipeline integration test: prove → verify → register → serialize → assert.
///
/// Validates 6 phases:
/// 1. On-chain aggregated proving (unified STARK + matmul sumchecks)
/// 2. Proof verification (forward pass replay + commitment checks)
/// 3. Model registration (weight commitment + model ID generation)
/// 4. Registration calldata (Obelysk + Sumcheck verifier formats)
/// 5. Starknet proof building (combined calldata assembly)
/// 6. IO commitment consistency (proof ↔ recomputed)
#[test]
fn test_e2e_full_pipeline() {
    let (graph, input, weights) = build_e2e_mlp();

    // === Phase 1: Prove ===
    let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
        .expect("on-chain aggregated proving should succeed");

    assert!(
        agg_proof.unified_stark.is_some(),
        "MLP with activations needs unified STARK"
    );
    assert_eq!(agg_proof.matmul_proofs.len(), 3, "3 matmul layers");
    assert_eq!(agg_proof.activation_claims.len(), 2, "2 ReLU activations");
    assert!(
        !agg_proof.execution.output.data.is_empty(),
        "forward pass must produce output"
    );

    // === Phase 2: Verify ===
    let proof_for_verify = prove_model_aggregated_onchain(&graph, &input, &weights)
        .expect("second proving for verify should succeed");

    // Verify execution output matches (deterministic forward pass)
    assert_eq!(
        agg_proof.execution.output.data, proof_for_verify.execution.output.data,
        "forward pass must be deterministic"
    );

    verify_aggregated_model_proof_onchain(proof_for_verify, &graph, &input, &weights)
        .expect("on-chain proof verification should succeed");

    // === Phase 3: Model Registration ===
    let registration = prepare_model_registration(&graph, &weights, "e2e-test-mlp");

    assert_ne!(
        registration.model_id,
        FieldElement::ZERO,
        "model_id must be non-zero"
    );
    assert_ne!(
        registration.weight_commitment,
        FieldElement::ZERO,
        "weight_commitment must be non-zero"
    );
    assert!(
        registration.num_layers >= 5,
        "3 matmul + 2 relu = at least 5 layers"
    );

    // Weight commitment should match standalone computation
    let standalone_commitment = compute_weight_commitment(&weights);
    assert_eq!(
        registration.weight_commitment, standalone_commitment,
        "prepare_model_registration weight commitment must match compute_weight_commitment"
    );

    // Model ID must be deterministic
    let registration2 = prepare_model_registration(&graph, &weights, "e2e-test-mlp");
    assert_eq!(
        registration.model_id, registration2.model_id,
        "model_id must be deterministic"
    );

    // === Phase 4: Registration Calldata ===
    let calldata_obelysk = register_model_calldata(&registration);
    assert_eq!(
        calldata_obelysk.len(),
        4,
        "ObelyskVerifier calldata: [model_id, weight_commitment, num_layers, description]"
    );
    assert_eq!(calldata_obelysk[0], registration.model_id);
    assert_eq!(calldata_obelysk[1], registration.weight_commitment);

    let calldata_sumcheck = register_model_calldata_sumcheck(&registration);
    assert_eq!(
        calldata_sumcheck.len(),
        2,
        "SumcheckVerifier calldata: [model_id, weight_commitment]"
    );
    assert_eq!(calldata_sumcheck[0], registration.model_id);
    assert_eq!(calldata_sumcheck[1], registration.weight_commitment);

    // === Phase 5: Build Starknet Proof ===
    let starknet_proof = build_starknet_proof_onchain(&agg_proof, &input);

    // Raw IO data starts at combined_calldata[4] as a length-prefixed array.
    // combined_calldata[4] = raw_io_data.len(), followed by the raw elements.
    // The on-chain verifier recomputes Poseidon(raw_io_data) to derive io_commitment.
    assert!(
        starknet_proof.combined_calldata.len() > 6,
        "combined calldata too small"
    );
    let raw_io_len = starknet_proof.combined_calldata[4];
    let raw_io_len_usize: usize = u64::try_from(raw_io_len).unwrap() as usize;
    assert_eq!(
        raw_io_len_usize,
        starknet_proof.raw_io_data.len(),
        "length prefix must match raw_io_data"
    );
    let recomputed_io = starknet_crypto::poseidon_hash_many(
        &starknet_proof.combined_calldata[5..5 + raw_io_len_usize],
    );
    assert_eq!(
        recomputed_io, starknet_proof.io_commitment,
        "Poseidon(raw_io_data) in combined_calldata must equal io_commitment"
    );
    assert_ne!(starknet_proof.io_commitment, FieldElement::ZERO);

    // Layer chain commitment follows the raw IO data array
    let layer_chain_idx = 5 + raw_io_len_usize;
    assert_eq!(
        starknet_proof.combined_calldata[layer_chain_idx], starknet_proof.layer_chain_commitment,
        "combined_calldata after raw_io_data must equal layer_chain_commitment"
    );
    assert_ne!(starknet_proof.layer_chain_commitment, FieldElement::ZERO);

    // Proof structure
    assert_eq!(starknet_proof.num_matmul_proofs, 3);
    assert_eq!(starknet_proof.layer_claims.len(), 2, "2 activation claims");
    assert_eq!(
        starknet_proof.num_proven_layers, 5,
        "3 matmul + 2 activation"
    );
    assert!(starknet_proof.calldata_size > 0);

    // PCS config must have real security parameters
    assert!(
        starknet_proof.pcs_config.pow_bits > 0,
        "pow_bits must be > 0"
    );
    assert!(
        starknet_proof.pcs_config.fri_config.n_queries > 0,
        "n_queries must be > 0"
    );

    // Gas estimation
    let gas = estimate_gas_from_proof(&starknet_proof);
    assert!(
        gas > starknet_proof.estimated_gas,
        "DA cost should increase gas"
    );
    assert!(gas > 50_000, "gas should exceed base cost");

    // === Phase 6: IO Commitment Consistency ===
    let io_commitment_recomputed = compute_io_commitment(&input, &agg_proof.execution.output);
    assert_eq!(
        starknet_proof.io_commitment, io_commitment_recomputed,
        "IO commitment in proof must match recomputed value"
    );
    assert_eq!(
        agg_proof.io_commitment, io_commitment_recomputed,
        "aggregated proof IO commitment must match recomputed value"
    );

    eprintln!("=== E2E Full Pipeline Test Passed ===");
    eprintln!("  Model: 3-layer MLP (8→8→8→4) with ReLU");
    eprintln!("  Matmul proofs: {}", starknet_proof.num_matmul_proofs);
    eprintln!("  Activation claims: {}", starknet_proof.layer_claims.len());
    eprintln!("  Calldata size: {} felt252s", starknet_proof.calldata_size);
    eprintln!("  Estimated gas: {}", gas);
    eprintln!("  Model ID: {:#066x}", registration.model_id);
    eprintln!(
        "  Weight commitment: {:#066x}",
        registration.weight_commitment
    );
    eprintln!("  IO commitment: {:#066x}", starknet_proof.io_commitment);
}

/// GKR integration pipeline test: standard prove + GKR overlay → serialize → assert.
///
/// Validates that the GKR pipeline:
/// - Produces a valid `GKRProof` with layer proofs
/// - Preserves all standard proof fields (unified STARK, matmul proofs, etc.)
/// - Generates matching IO commitments with the standard pipeline
/// - Serializes GKR calldata into the Starknet proof
#[test]
fn test_e2e_gkr_pipeline() {
    let (graph, input, weights) = build_e2e_mlp();

    // Prove with GKR for matmul layer reductions
    let gkr_proof = prove_model_aggregated_onchain_gkr(&graph, &input, &weights)
        .expect("GKR aggregated proving should succeed");

    // GKR proof must be present
    assert!(gkr_proof.gkr_proof.is_some(), "GKR proof must be populated");
    let gkr_layer_count = gkr_proof.gkr_proof.as_ref().unwrap().layer_proofs.len();
    let gkr_weight_count = gkr_proof
        .gkr_proof
        .as_ref()
        .unwrap()
        .weight_commitments
        .len();
    assert!(gkr_layer_count > 0, "GKR must have layer proofs");

    // Standard fields still populated (GKR is additive — doesn't replace STARK)
    assert!(
        gkr_proof.unified_stark.is_some(),
        "unified STARK still present"
    );
    assert_eq!(gkr_proof.matmul_proofs.len(), 3);
    assert_eq!(gkr_proof.activation_claims.len(), 2);

    // IO commitment must match the non-GKR pipeline
    let standard_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
        .expect("standard proving should succeed");
    assert_eq!(
        gkr_proof.io_commitment, standard_proof.io_commitment,
        "GKR and standard proofs must produce same IO commitment"
    );

    // Starknet proof build works with GKR proof present
    let starknet_proof = build_starknet_proof_onchain(&gkr_proof, &input);
    assert!(
        starknet_proof.gkr_calldata.is_some(),
        "GKR calldata must be present"
    );
    assert!(!starknet_proof.gkr_calldata.as_ref().unwrap().is_empty());

    // Verify GKR proof end-to-end (prover↔verifier channel sync)
    {
        let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
        let mut verify_channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
        stwo_ml::gkr::verifier::verify_gkr(
            &circuit,
            gkr_proof.gkr_proof.as_ref().unwrap(),
            &gkr_proof.execution.output,
            &mut verify_channel,
        )
        .expect("GKR multi-layer verification should succeed");
    }

    eprintln!("=== E2E GKR Pipeline Test Passed ===");
    eprintln!("  GKR layer proofs: {}", gkr_layer_count);
    eprintln!("  Weight commitments: {}", gkr_weight_count);
    eprintln!(
        "  GKR calldata size: {} felts",
        starknet_proof.gkr_calldata.as_ref().unwrap().len()
    );
}

/// Direct verification pipeline test: prove → build_starknet_proof_direct → assert.
///
/// Validates the Phase 2 direct path that eliminates Cairo VM recursion:
/// 1. On-chain aggregated proving (same as recursive path)
/// 2. Direct serialization (batched sumchecks + STARK chunks)
/// 3. Structure validation (chunks, metadata, gas estimate)
/// 4. IO commitment consistency with standard pipeline
#[test]
fn test_e2e_direct_pipeline() {
    let (graph, input, weights) = build_e2e_mlp();

    // === Phase 1: Prove ===
    let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
        .expect("on-chain aggregated proving should succeed");

    // === Phase 2: Build Direct Proof ===
    let io_commitment = compute_io_commitment(&input, &agg_proof.execution.output);
    let weight_commitment = compute_weight_commitment(&weights);
    let model_id = starknet_ff::FieldElement::from(0x42u64);

    let direct_metadata = DirectProofMetadata {
        model_id,
        io_commitment,
        weight_commitment,
        num_layers: graph.num_layers() as u32,
        activation_type: 0, // ReLU
    };

    let direct_proof = build_starknet_proof_direct(&agg_proof, &input, direct_metadata);

    // === Phase 3: Validate Structure ===

    // Metadata preserved
    assert_eq!(direct_proof.model_id, model_id);
    let direct_io_commitment = starknet_crypto::poseidon_hash_many(&direct_proof.raw_io_data);
    assert_eq!(direct_io_commitment, io_commitment);
    assert_eq!(direct_proof.weight_commitment, weight_commitment);
    assert_eq!(direct_proof.num_layers, graph.num_layers() as u32);
    assert_eq!(direct_proof.activation_type, 0);

    // MLP with activations must have activation STARK
    assert!(
        direct_proof.has_activation_stark,
        "MLP with ReLU should have activation STARK"
    );
    assert!(
        !direct_proof.stark_chunks.is_empty(),
        "STARK proof must produce chunks"
    );

    // Each chunk should be at most 4000 felts
    for (i, chunk) in direct_proof.stark_chunks.iter().enumerate() {
        assert!(
            chunk.len() <= 4000,
            "chunk {i} has {} felts, expected <= 4000",
            chunk.len()
        );
        assert!(!chunk.is_empty(), "chunk {i} is empty");
    }

    // Matmul proofs must be present via either batched (CUDA) or individual path.
    // On non-CUDA builds, batched_calldata is empty but matmul_proofs exist in the
    // underlying AggregatedModelProofOnChain.
    let has_matmul_proofs =
        !direct_proof.batched_calldata.is_empty() || !agg_proof.matmul_proofs.is_empty();
    assert!(
        has_matmul_proofs,
        "3-layer MLP should have matmul proofs (batched or individual)"
    );
    for (i, batch) in direct_proof.batched_calldata.iter().enumerate() {
        assert!(!batch.is_empty(), "batch {i} calldata is empty");
    }

    // Gas and calldata size estimates
    assert!(direct_proof.estimated_gas > 0, "estimated gas must be > 0");
    assert!(
        direct_proof.total_calldata_size > 0,
        "total calldata size must be > 0"
    );

    // === Phase 4: IO Commitment Consistency ===
    // Direct proof IO commitment must match standard pipeline
    let standard_proof = build_starknet_proof_onchain(&agg_proof, &input);
    assert_eq!(
        direct_io_commitment, standard_proof.io_commitment,
        "direct and standard proofs must have same IO commitment"
    );

    eprintln!("=== E2E Direct Pipeline Test Passed ===");
    eprintln!("  Model: 3-layer MLP (8→8→8→4) with ReLU");
    eprintln!(
        "  Batched sumcheck proofs: {}",
        direct_proof.batched_calldata.len()
    );
    eprintln!(
        "  STARK chunks: {} ({} felts total)",
        direct_proof.stark_chunks.len(),
        direct_proof
            .stark_chunks
            .iter()
            .map(|c| c.len())
            .sum::<usize>(),
    );
    eprintln!("  Estimated gas: {}", direct_proof.estimated_gas);
    eprintln!(
        "  Total calldata: {} felts",
        direct_proof.total_calldata_size
    );
    eprintln!("  IO commitment: {:#066x}", direct_io_commitment);
}

/// E2E test: prove → health check → dry-run → corruption detection.
///
/// Validates the fast proof validity check and dry-run simulation:
/// 1. Build a small MLP and prove with GKR
/// 2. Serialize to v4 packed IO calldata
/// 3. Run verify_proof_fast() → assert all checks pass
/// 4. Run dry_run_onchain() without RPC → assert healthy
/// 5. Corrupt a single felt → assert health check catches it
/// 6. Truncate calldata → assert tail sentinel failure
#[test]
fn test_e2e_prove_health_check_dry_run() {
    use stwo_ml::aggregation::prove_model_pure_gkr;
    use stwo_ml::gkr::types::WeightOpeningTranscriptMode;
    use stwo_ml::starknet::{
        build_verify_model_gkr_v4_packed_io_calldata, dry_run_onchain, verify_proof_fast,
    };

    // 1. Build a 2-layer MLP: MatMul(4→4) + ReLU + MatMul(4→2)
    let mut builder = GraphBuilder::new((1, 4));
    builder
        .linear(4)
        .activation(ActivationType::ReLU)
        .linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = GraphWeights::new();
    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            w0.set(i, j, M31::from(((i * 3 + j * 5) % 11 + 1) as u32));
        }
    }
    weights.add_weight(0, w0);
    let mut w2 = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w2.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }
    }
    weights.add_weight(2, w2);

    // 2. Prove with GKR
    let mut agg_proof =
        prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
    let gkr = agg_proof
        .gkr_proof
        .as_mut()
        .expect("GKR proof must be present");
    gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
    gkr.aggregated_binding = None;

    // 3. Serialize to v4 packed IO calldata
    let circuit =
        stwo_ml::gkr::LayeredCircuit::from_graph(&graph).expect("circuit should compile");
    let raw_io = stwo_ml::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
    let model_id = FieldElement::from(0x42u64);

    let v4_calldata =
        build_verify_model_gkr_v4_packed_io_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("packed IO calldata should build");

    // Parse calldata from strings to FieldElements
    let calldata_felts: Vec<FieldElement> = v4_calldata
        .calldata_parts
        .iter()
        .filter_map(|s| {
            if s.starts_with("0x") || s.starts_with("0X") {
                FieldElement::from_hex_be(s).ok()
            } else {
                s.parse::<u64>().ok().map(FieldElement::from)
            }
        })
        .collect();

    assert!(
        !calldata_felts.is_empty(),
        "calldata should have been parsed"
    );

    // 4. Run verify_proof_fast() — should pass
    let report = verify_proof_fast(&calldata_felts);
    eprintln!("=== Health Check Report ===");
    for check in &report.checks {
        let status = if check.passed { "PASS" } else { "FAIL" };
        eprintln!("  [{}] {}: {}", status, check.name, check.detail);
    }
    assert!(report.passed, "health check should pass on valid proof");
    assert!(report.total_felts > 0);
    assert!(report.estimated_steps > 0);

    // 5. Run dry_run_onchain() without RPC — should pass
    let dry_run = dry_run_onchain(&calldata_felts, None, None);
    assert!(dry_run.health.passed, "dry-run health should pass");
    assert!(
        dry_run.within_step_limit,
        "small MLP should be within step limit"
    );
    assert!(dry_run.estimated_steps < 10_000_000);
    assert!(dry_run.calldata_size > 0);
    assert!(dry_run.rpc_simulation.is_none(), "no RPC provided");

    // 6. Corrupt a single felt → health check should still detect something
    {
        let mut corrupted = calldata_felts.clone();
        // Zero out the model_id
        corrupted[0] = FieldElement::ZERO;
        let corrupt_report = verify_proof_fast(&corrupted);
        // model_id check should fail
        let model_check = corrupt_report
            .checks
            .iter()
            .find(|c| c.name == "model_id");
        assert!(
            model_check.is_some() && !model_check.unwrap().passed,
            "zeroed model_id should fail model_id check"
        );
        assert!(
            !corrupt_report.passed,
            "corrupted proof should fail overall"
        );
    }

    // 7. Truncate calldata → tail sentinel or length check should fail
    {
        let truncated: Vec<FieldElement> = calldata_felts
            .iter()
            .take(3)
            .cloned()
            .collect();
        let trunc_report = verify_proof_fast(&truncated);
        // With only 3 felts, io_header check should fail (not enough room)
        assert!(
            !trunc_report.passed,
            "truncated proof should fail health check"
        );
    }

    eprintln!("=== E2E Health Check + Dry-Run Test Passed ===");
    eprintln!("  Calldata size: {} felts", calldata_felts.len());
    eprintln!("  Estimated steps: {}", report.estimated_steps);
    eprintln!("  Dry-run fee: {:.4} STRK", dry_run.estimated_fee_strk);
}
