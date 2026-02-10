//! Integration tests for the GPU-aware proving pipeline.
//!
//! Tests the full path: graph build → prove → serialize → verify.
//! All tests run on SimdBackend by default; `cuda-runtime` tests
//! exercise real GPU acceleration.

use stwo::core::fields::m31::M31;
use stwo_ml::prelude::*;

use stwo_ml::aggregation::{
    prove_model_aggregated, prove_model_aggregated_auto,
    prove_model_aggregated_with, AggregatedModelProofFor,
    prove_model_aggregated_onchain, prove_model_aggregated_onchain_auto,
};
use stwo_ml::receipt::{
    ComputeReceipt, prove_receipt, prove_receipt_batch,
    prove_receipt_batch_auto, prove_receipt_batch_with,
};
use stwo_ml::starknet::{prove_for_starknet, build_starknet_proof, build_starknet_proof_onchain, estimate_gas_from_proof, compute_io_commitment};
use stwo_ml::cairo_serde::serialize_proof;
use stwo_ml::compiler::prove::{
    prove_model, prove_model_auto, prove_model_with, verify_model_matmuls,
};
use stwo_ml::gpu::GpuModelProver;
use stwo_ml::backend::BackendInfo;

use starknet_ff::FieldElement;

// === Helper: build a standard test MLP ===

fn build_test_mlp() -> (ComputationGraph, M31Matrix, GraphWeights) {
    let mut builder = GraphBuilder::new((1, 4));
    builder
        .linear(4)
        .activation(ActivationType::ReLU)
        .linear(4)
        .activation(ActivationType::ReLU)
        .linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

    let mut weights = GraphWeights::new();

    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i * 4 + j) % 5 + 1) as u32)); } }
    weights.add_weight(0, w0);

    let mut w2 = M31Matrix::new(4, 4);
    for i in 0..4 { for j in 0..4 { w2.set(i, j, M31::from(((i + j * 3) % 7 + 1) as u32)); } }
    weights.add_weight(2, w2);

    let mut w4 = M31Matrix::new(4, 2);
    for i in 0..4 { for j in 0..2 { w4.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
    weights.add_weight(4, w4);

    (graph, input, weights)
}

fn test_receipt(seq: u32, prev_hash: FieldElement) -> ComputeReceipt {
    let gpu_time_ms = 5000u64;
    let rate_per_sec = 100u64;
    let token_count = 512u32;
    let rate_per_token = 10u64;
    let time_billing = gpu_time_ms * rate_per_sec / 1000;
    let token_billing = token_count as u64 * rate_per_token;

    ComputeReceipt {
        job_id: FieldElement::from(1u64),
        worker_pubkey: FieldElement::from(42u64),
        input_commitment: FieldElement::from(100u64),
        output_commitment: FieldElement::from(200u64),
        model_commitment: FieldElement::from(300u64),
        prev_receipt_hash: prev_hash,
        gpu_time_ms,
        token_count,
        peak_memory_mb: 1024,
        billing_amount_sage: time_billing + token_billing,
        billing_rate_per_sec: rate_per_sec,
        billing_rate_per_token: rate_per_token,
        tee_report_hash: FieldElement::from(500u64),
        tee_timestamp: 1700000000,
        timestamp: 1700000010,
        sequence_number: seq,
    }
}

// === End-to-end pipeline tests ===

#[test]
fn test_full_pipeline_prove_verify_serialize() {
    let (graph, input, weights) = build_test_mlp();

    // 1. Prove
    let (proofs, execution) = prove_model(&graph, &input, &weights)
        .expect("prove_model should succeed");
    assert_eq!(proofs.len(), 5);
    assert_eq!(execution.output.cols, 2);

    // 2. Verify
    verify_model_matmuls(&proofs, &graph, &input, &weights)
        .expect("verification should succeed");
}

#[test]
fn test_full_pipeline_aggregated_to_calldata() {
    let (graph, input, weights) = build_test_mlp();

    // 1. Aggregated prove
    let agg_proof = prove_model_aggregated(&graph, &input, &weights)
        .expect("aggregated proving should succeed");
    assert!(agg_proof.unified_stark.is_some());
    assert_eq!(agg_proof.matmul_proofs.len(), 3);
    assert_eq!(agg_proof.activation_claims.len(), 2);

    // 2. Serialize to felt252 calldata
    let calldata = serialize_proof(agg_proof.unified_stark.as_ref().unwrap());
    assert!(!calldata.is_empty());
    assert!(calldata.len() < 5000, "calldata too large: {} felts", calldata.len());

    // 3. Build starknet proof
    let starknet_proof = build_starknet_proof(&agg_proof);
    assert_eq!(starknet_proof.unified_calldata, calldata);
    assert!(starknet_proof.estimated_gas > 0);

    // 4. DA-aware gas estimate
    let gas_with_da = estimate_gas_from_proof(&starknet_proof);
    assert!(gas_with_da > starknet_proof.estimated_gas);
}

#[test]
fn test_prove_for_starknet_end_to_end() {
    let (graph, input, weights) = build_test_mlp();

    let starknet_proof = prove_for_starknet(&graph, &input, &weights)
        .expect("prove_for_starknet should succeed");

    assert!(!starknet_proof.unified_calldata.is_empty());
    assert_eq!(starknet_proof.num_matmul_proofs, 3);
    assert_eq!(starknet_proof.num_proven_layers, 5);
    assert!(starknet_proof.calldata_size > 0);
}

// === Auto-dispatch tests (use SimdBackend on macOS, GpuBackend on CUDA machines) ===

#[test]
fn test_prove_model_auto_matches_prove_model() {
    let (graph, input, weights) = build_test_mlp();

    let (proofs_auto, exec_auto) = prove_model_auto(&graph, &input, &weights)
        .expect("prove_model_auto should succeed");
    let (proofs_simd, exec_simd) = prove_model(&graph, &input, &weights)
        .expect("prove_model should succeed");

    // Same structure
    assert_eq!(proofs_auto.len(), proofs_simd.len());
    // Same output (deterministic forward pass)
    assert_eq!(exec_auto.output.data, exec_simd.output.data);
}

#[test]
fn test_prove_model_aggregated_auto_matches_simd() {
    let (graph, input, weights) = build_test_mlp();

    let auto_proof = prove_model_aggregated_auto(&graph, &input, &weights)
        .expect("auto should succeed");
    let simd_proof = prove_model_aggregated(&graph, &input, &weights)
        .expect("simd should succeed");

    // Same execution output
    assert_eq!(auto_proof.execution.output.data, simd_proof.execution.output.data);
    // Same structure
    assert_eq!(auto_proof.matmul_proofs.len(), simd_proof.matmul_proofs.len());
    assert_eq!(auto_proof.activation_claims.len(), simd_proof.activation_claims.len());
}

#[test]
fn test_prove_receipt_batch_auto_matches_simd() {
    let r0 = test_receipt(0, FieldElement::ZERO);
    let r1 = test_receipt(1, r0.receipt_hash());

    let auto_proof = prove_receipt_batch_auto(&[r0.clone(), r1.clone()])
        .expect("auto should succeed");
    let simd_proof = prove_receipt_batch(&[r0, r1])
        .expect("simd should succeed");

    assert_eq!(auto_proof.batch_size, simd_proof.batch_size);
    assert_eq!(auto_proof.receipt_hashes, simd_proof.receipt_hashes);
}

// === GpuModelProver unified interface tests ===

#[test]
fn test_gpu_prover_full_pipeline() {
    let (graph, input, weights) = build_test_mlp();
    let prover = GpuModelProver::default();

    // Per-layer proofs
    let (proofs, execution) = prover.prove_model(&graph, &input, &weights)
        .expect("GpuModelProver.prove_model should succeed");
    assert_eq!(proofs.len(), 5);
    assert_eq!(execution.output.cols, 2);

    // Aggregated proof
    let agg_proof = prover.prove_model_aggregated(&graph, &input, &weights)
        .expect("GpuModelProver.prove_model_aggregated should succeed");
    assert!(agg_proof.unified_stark.is_some());
    assert_eq!(agg_proof.execution.output.data, execution.output.data);

    // Receipt
    let r0 = test_receipt(0, FieldElement::ZERO);
    let receipt_proof = prover.prove_receipt_batch(&[r0])
        .expect("GpuModelProver.prove_receipt_batch should succeed");
    assert_eq!(receipt_proof.batch_size, 1);
}

// === Backend info test ===

#[test]
fn test_backend_info_detection() {
    let info = BackendInfo::detect();
    // On dev machines without CUDA, should be SimdBackend
    if !info.gpu_available {
        assert_eq!(info.name, "SimdBackend");
        assert!(info.gpu_device.is_none());
    } else {
        assert_eq!(info.name, "GpuBackend");
        assert!(info.gpu_device.is_some());
    }
}

// === Generic backend explicit tests ===

#[test]
fn test_prove_model_with_explicit_simd() {
    let (graph, input, weights) = build_test_mlp();

    let result = prove_model_with::<SimdBackend, Blake2sMerkleChannel>(
        &graph, &input, &weights,
    );
    assert!(result.is_ok());
}

#[test]
fn test_prove_aggregated_with_explicit_simd() {
    let (graph, input, weights) = build_test_mlp();

    let result = prove_model_aggregated_with::<SimdBackend, Blake2sMerkleChannel>(
        &graph, &input, &weights,
    );
    assert!(result.is_ok());
}

#[test]
fn test_prove_receipt_with_explicit_simd() {
    let r = test_receipt(0, FieldElement::ZERO);

    let result = prove_receipt_batch_with::<SimdBackend, Blake2sMerkleChannel>(
        &[r],
    );
    assert!(result.is_ok());
}

// === Receipt chain + prove integration ===

#[test]
fn test_receipt_chain_prove_and_serialize() {
    let r0 = test_receipt(0, FieldElement::ZERO);
    let r1 = test_receipt(1, r0.receipt_hash());
    let r2 = test_receipt(2, r1.receipt_hash());
    let chain = [r0, r1, r2];

    // Verify chain linking
    stwo_ml::receipt::verify_receipt_chain(&chain).expect("chain should be valid");

    // Prove batch
    let proof = prove_receipt_batch(&chain).expect("batch proving should succeed");
    assert_eq!(proof.batch_size, 3);

    // Serialize to calldata
    let calldata = serialize_proof(&proof.stark_proof);
    assert!(!calldata.is_empty());
    assert!(calldata.len() < 5000, "receipt calldata too large: {} felts", calldata.len());
}

// === Model loading pipeline tests ===

#[test]
fn test_build_mlp_with_weights_full_pipeline() {
    use stwo_ml::compiler::onnx::build_mlp_with_weights;
    use stwo_ml::compiler::inspect::summarize_model;

    // Build auto-weighted MLP
    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

    // 1. Prove
    let (proofs, execution) = prove_model(&model.graph, &input, &model.weights)
        .expect("auto-weighted MLP should prove");
    assert_eq!(proofs.len(), 3);
    assert_eq!(execution.output.cols, 2);

    // 2. Verify
    verify_model_matmuls(&proofs, &model.graph, &input, &model.weights)
        .expect("verification should succeed");

    // 3. Aggregated prove
    let agg_proof = prove_model_aggregated(&model.graph, &input, &model.weights)
        .expect("aggregated proving should succeed");
    assert!(agg_proof.unified_stark.is_some());

    // 4. Summarize
    let summary = summarize_model(&model);
    assert_eq!(summary.num_matmul_layers, 2);
    assert_eq!(summary.total_parameters, 4 * 4 + 4 * 2);
}

#[test]
fn test_transformer_block_full_pipeline() {
    use stwo_ml::compiler::onnx::{build_transformer_block, TransformerConfig};

    let config = TransformerConfig {
        d_model: 4,
        num_heads: 1,
        d_ff: 8,
        activation: ActivationType::GELU,
    };
    let model = build_transformer_block(&config, 77);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

    // Prove
    let (proofs, execution) = prove_model(&model.graph, &input, &model.weights)
        .expect("transformer block should prove");
    assert_eq!(proofs.len(), 7); // LN + Q + O + LN + FFN_up + act + FFN_down
    assert_eq!(execution.output.cols, 4); // d_model preserved

    // Verify matmul proofs
    verify_model_matmuls(&proofs, &model.graph, &input, &model.weights)
        .expect("transformer verification should succeed");
}

#[test]
fn test_prelude_exports_model_loading() {
    // Verify all model-loading types are accessible from prelude
    use stwo_ml::prelude::{
        OnnxModel, ModelMetadata, OnnxError, build_mlp,
        build_mlp_with_weights, generate_weights_for_graph,
        TransformerConfig, build_transformer_block,
        build_transformer, ModelSummary, summarize_model, summarize_graph,
    };

    // Use each type/function to verify they're properly exported
    let model: OnnxModel = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 1);
    let summary: ModelSummary = summarize_model(&model);
    assert_eq!(summary.num_matmul_layers, 2);

    let mlp_model: OnnxModel = build_mlp(4, &[4], 2, ActivationType::ReLU);
    let _meta: &ModelMetadata = &mlp_model.metadata;

    let weights = generate_weights_for_graph(&mlp_model.graph, 42);
    let graph_summary = summarize_graph(&mlp_model.graph, &weights);
    assert!(graph_summary.total_parameters > 0);

    let config = TransformerConfig::new(4, 1);
    let tmodel = build_transformer_block(&config, 2);
    let tsummary = summarize_model(&tmodel);
    assert_eq!(tsummary.num_matmul_layers, 4);

    let transformer = build_transformer(&config, 1, 8, 3);
    assert!(transformer.graph.num_layers() > 0);

    // OnnxError should be constructable
    let _err: OnnxError = OnnxError::IoError("test".into());
}

// === Phase 4: On-Chain Proof Pipeline Integration Tests ===

#[test]
fn test_matmul_onchain_proof_full_pipeline() {
    use stwo_ml::components::matmul::{prove_matmul_sumcheck_onchain, verify_matmul_sumcheck_onchain};
    use stwo_ml::cairo_serde::serialize_matmul_sumcheck_proof;

    // 2x2 matmul → on-chain proof → serialize → verify calldata structure
    let mut a = M31Matrix::new(2, 2);
    a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
    a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

    let mut b = M31Matrix::new(2, 2);
    b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
    b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

    // C = A * B
    let mut c = M31Matrix::new(2, 2);
    c.set(0, 0, M31::from(19)); c.set(0, 1, M31::from(22));
    c.set(1, 0, M31::from(43)); c.set(1, 1, M31::from(50));

    let proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
        .expect("on-chain matmul proof should succeed");

    // Verify locally
    verify_matmul_sumcheck_onchain(&proof)
        .expect("on-chain matmul verification should succeed");

    // Serialize to calldata
    let mut calldata = Vec::new();
    serialize_matmul_sumcheck_proof(&proof, &mut calldata);

    // First 4 felts: m, k, n, num_rounds
    assert_eq!(calldata[0], FieldElement::from(2u64)); // m
    assert_eq!(calldata[1], FieldElement::from(2u64)); // k
    assert_eq!(calldata[2], FieldElement::from(2u64)); // n
    assert!(!calldata.is_empty());
    assert!(calldata.len() > 10, "calldata should have meaningful content");
}

#[test]
fn test_mlp_full_onchain_pipeline() {
    // auto-weighted MLP → aggregated on-chain proof → combined calldata → IO commitment at [4]
    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

    let agg_proof = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("on-chain aggregated proving should succeed");

    assert!(agg_proof.unified_stark.is_some());
    assert_eq!(agg_proof.matmul_proofs.len(), 2); // 2 matmul layers in this MLP

    // Build complete Starknet calldata
    let starknet_proof = build_starknet_proof_onchain(&agg_proof);

    // IO commitment should be at index [4] in combined calldata
    assert!(!starknet_proof.combined_calldata.is_empty());
    assert!(starknet_proof.combined_calldata.len() > 5);
    assert_eq!(starknet_proof.combined_calldata[4], starknet_proof.io_commitment);

    // IO commitment should be non-zero
    assert_ne!(starknet_proof.io_commitment, FieldElement::ZERO);

    // Matmul calldata should be populated
    assert_eq!(starknet_proof.matmul_calldata.len(), 2);
    for mc in &starknet_proof.matmul_calldata {
        assert!(!mc.is_empty());
    }
}

#[test]
fn test_io_commitment_binding() {
    // Prove same model with different inputs → different IO commitments → different calldata[4]
    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);

    let mut input1 = M31Matrix::new(1, 4);
    for j in 0..4 { input1.set(0, j, M31::from((j + 1) as u32)); }

    let mut input2 = M31Matrix::new(1, 4);
    for j in 0..4 { input2.set(0, j, M31::from((j + 10) as u32)); }

    let proof1 = prove_model_aggregated_onchain(&model.graph, &input1, &model.weights)
        .expect("proof1 should succeed");
    let proof2 = prove_model_aggregated_onchain(&model.graph, &input2, &model.weights)
        .expect("proof2 should succeed");

    let sp1 = build_starknet_proof_onchain(&proof1);
    let sp2 = build_starknet_proof_onchain(&proof2);

    // Different inputs → different IO commitments
    assert_ne!(sp1.io_commitment, sp2.io_commitment);

    // IO commitment at calldata[4] should also differ
    assert_ne!(sp1.combined_calldata[4], sp2.combined_calldata[4]);

    // Verify IO commitment matches compute_io_commitment directly
    let io1 = compute_io_commitment(&input1, &proof1.execution.output);
    assert_eq!(sp1.io_commitment, io1);
}

#[test]
fn test_calldata_size_reasonable() {
    // MLP calldata should be < 50,000 felts (gas feasibility check)
    let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 42);

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

    let proof = prove_model_aggregated_onchain(&model.graph, &input, &model.weights)
        .expect("proving should succeed");
    let sp = build_starknet_proof_onchain(&proof);

    assert!(
        sp.combined_calldata.len() < 50_000,
        "calldata too large for on-chain submission: {} felts",
        sp.combined_calldata.len()
    );

    // Each matmul calldata should also be reasonable
    for (i, mc) in sp.matmul_calldata.iter().enumerate() {
        assert!(
            mc.len() < 10_000,
            "matmul {} calldata too large: {} felts",
            i, mc.len()
        );
    }

    // Gas estimate should be populated
    assert!(sp.estimated_gas > 0);
    assert!(sp.calldata_size > 0);
}

// === GPU-specific tests (only run with cuda-runtime feature) ===

#[cfg(feature = "cuda-runtime")]
mod gpu_tests {
    use super::*;
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_prove_model_gpu_backend_explicit() {
        let (graph, input, weights) = build_test_mlp();

        let (proofs, execution) = prove_model_with::<GpuBackend, Blake2sMerkleChannel>(
            &graph, &input, &weights,
        ).expect("GPU proving should succeed");

        assert_eq!(proofs.len(), 5);
        assert_eq!(execution.output.cols, 2);

        // Verify with generic verifier
        verify_model_matmuls(&proofs, &graph, &input, &weights)
            .expect("GPU proof verification should succeed");
    }

    #[test]
    fn test_prove_aggregated_gpu_backend_explicit() {
        let (graph, input, weights) = build_test_mlp();

        let proof = prove_model_aggregated_with::<GpuBackend, Blake2sMerkleChannel>(
            &graph, &input, &weights,
        ).expect("GPU aggregated proving should succeed");

        assert!(proof.unified_stark.is_some());
        assert_eq!(proof.matmul_proofs.len(), 3);
    }

    #[test]
    fn test_prove_receipt_gpu_backend_explicit() {
        let r = test_receipt(0, FieldElement::ZERO);

        let proof = prove_receipt_batch_with::<GpuBackend, Blake2sMerkleChannel>(
            &[r],
        ).expect("GPU receipt proving should succeed");

        assert_eq!(proof.batch_size, 1);
    }

    #[test]
    fn test_gpu_prover_require_gpu() {
        let prover = GpuModelProver::require_gpu()
            .expect("GPU should be available for cuda-runtime tests");
        assert!(prover.is_gpu);
        assert!(!prover.device_name.contains("CPU"));
    }

    #[test]
    fn test_full_gpu_pipeline_to_starknet() {
        let (graph, input, weights) = build_test_mlp();

        // GPU aggregated prove → Starknet calldata
        let agg_proof = prove_model_aggregated_with::<GpuBackend, Blake2sMerkleChannel>(
            &graph, &input, &weights,
        ).expect("GPU aggregated proving should succeed");

        let starknet_proof = build_starknet_proof(&agg_proof);
        assert!(!starknet_proof.unified_calldata.is_empty());
        assert_eq!(starknet_proof.num_proven_layers, 5);
    }
}
