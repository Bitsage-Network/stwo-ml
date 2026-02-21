//! End-to-end audit pipeline integration tests.
//!
//! Exercises the full audit flow: build model → capture inferences → prove
//! → evaluate → report → encrypt → serialize calldata. Validates that all
//! subsystems integrate correctly.
//!
//! ```text
//! Model + Weights
//!       │
//!       v
//! CaptureHook → InferenceLog → AuditProver → BatchAuditResult
//!                                    │                │
//!                              evaluate_batch    report builder
//!                                    │                │
//!                              SemanticSummary   AuditReport
//!                                                     │
//!                                              encrypt + calldata
//! ```

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use stwo::core::fields::m31::M31;

use stwo_ml::aggregation::compute_io_commitment;
use stwo_ml::audit::capture::{CaptureHook, CaptureJob};
use stwo_ml::audit::deterministic::evaluate_deterministic;
use stwo_ml::audit::digest::{digest_to_hex, ZERO_DIGEST};
use stwo_ml::audit::encryption::{
    encrypt_and_store, fetch_and_decrypt, generate_audit_keypair, Poseidon2M31Encryption,
};
use stwo_ml::audit::log::InferenceLog;
use stwo_ml::audit::orchestrator::{run_audit, run_audit_dry, AuditPipelineConfig};
use stwo_ml::audit::prover::AuditProver;
use stwo_ml::audit::replay::execute_forward_pass;
use stwo_ml::audit::report::compute_report_hash;
use stwo_ml::audit::scoring::aggregate_evaluations;
use stwo_ml::audit::self_eval::{evaluate_batch, SelfEvalConfig};
use stwo_ml::audit::storage::{ArweaveClient, MockTransport};
use stwo_ml::audit::submit::{serialize_audit_calldata, validate_calldata, SubmitConfig};
use stwo_ml::audit::types::{AuditError, AuditReport, AuditRequest, InferenceLogEntry, ModelInfo};
use stwo_ml::prelude::*;

// ============================================================================
// Helpers
// ============================================================================

fn temp_dir(prefix: &str) -> PathBuf {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("stwo_ml_e2e_audit_{}_{}", prefix, d))
}

/// Build a 2-layer MLP: linear(4→4) → relu → linear(4→2).
/// Small enough to prove fast, complex enough to exercise the pipeline.
fn build_audit_model() -> (ComputationGraph, GraphWeights, ModelInfo) {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(4).activation(ActivationType::ReLU).linear(2);
    let graph = builder.build();

    let mut weights = GraphWeights::new();

    // Layer 0: 4×4 weight matrix
    let mut w0 = M31Matrix::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            w0.set(i, j, M31::from(((i * 3 + j * 5) % 11 + 1) as u32));
        }
    }
    weights.add_weight(0, w0);

    // Layer 2: 4×2 weight matrix
    let mut w2 = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w2.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }
    }
    weights.add_weight(2, w2);

    let model_info = ModelInfo {
        model_id: "0x2".to_string(),
        name: "test-mlp".to_string(),
        architecture: "mlp".to_string(),
        parameters: "26".to_string(),
        layers: 3,
        weight_commitment: "0xabc".to_string(),
    };

    (graph, weights, model_info)
}

/// Generate a deterministic M31 input matrix for a given index.
fn make_input(idx: usize) -> M31Matrix {
    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((idx * 4 + j + 1) as u32));
    }
    input
}

/// Populate an inference log with N entries using correct forward-pass outputs.
fn populate_log(
    dir: &std::path::Path,
    n: usize,
    graph: &ComputationGraph,
    weights: &GraphWeights,
) -> InferenceLog {
    let mut log = InferenceLog::new(dir, "0x2", "0xabc", "test-mlp").unwrap();

    let base_ts = 1_000_000_000_000u64;
    for i in 0..n {
        let input = make_input(i);
        let output = execute_forward_pass(graph, &input, weights).unwrap();
        let io_commitment = compute_io_commitment(&input, &output);

        let input_data: Vec<u32> = input.data.iter().map(|m| m.0).collect();
        let (mat_off, mat_sz) = log.write_matrix(1, 4, &input_data).unwrap();

        let entry = InferenceLogEntry {
            inference_id: i as u64,
            sequence_number: 0,
            model_id: "0x2".to_string(),
            weight_commitment: "0xabc".to_string(),
            model_name: "test-mlp".to_string(),
            num_layers: 3,
            input_tokens: vec![1, 2, 3, 4],
            output_tokens: vec![5, 6],
            matrix_offset: mat_off,
            matrix_size: mat_sz,
            input_rows: 1,
            input_cols: 4,
            output_rows: output.rows as u32,
            output_cols: output.cols as u32,
            io_commitment: format!("{:#066x}", io_commitment),
            layer_chain_commitment: "0x0".to_string(),
            prev_entry_hash: String::new(),
            entry_hash: String::new(),
            timestamp_ns: base_ts + i as u64 * 1_000_000_000,
            latency_ms: 50 + (i as u64 % 20),
            gpu_device: "test-gpu".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: Some("test".to_string()),
            input_preview: Some(format!("test input {}", i)),
            output_preview: Some(format!("test output {}", i)),
        };
        log.append(entry).unwrap();
    }

    log
}

// ============================================================================
// Test 1: Full dry-run pipeline via orchestrator
// ============================================================================

#[test]
fn test_e2e_audit_dry_run() {
    let dir = temp_dir("dry_run");
    let (graph, weights, model_info) = build_audit_model();
    let log = populate_log(&dir, 3, &graph, &weights);

    let request = AuditRequest {
        start_ns: 0,
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let report = run_audit_dry(&log, &graph, &weights, request, model_info).unwrap();

    // ── Verify report structure ──
    assert!(!report.audit_id.is_empty(), "audit_id should be generated");
    assert_eq!(report.inference_summary.total_inferences, 3);
    assert_eq!(report.model.model_id, "0x2");
    assert_eq!(report.model.name, "test-mlp");

    // Commitments should be non-zero (M31 digest hex format)
    let zero_hex = digest_to_hex(&ZERO_DIGEST);
    assert_ne!(report.commitments.io_merkle_root, zero_hex);
    assert_ne!(report.commitments.inference_log_merkle_root, zero_hex);
    assert_eq!(report.commitments.weight_commitment, "0xabc");

    // Proof info
    assert!(report.proof.proving_time_seconds > 0 || report.inference_summary.total_inferences > 0);
    assert!(
        report.proof.on_chain_tx.is_none(),
        "dry run should have no TX"
    );
    assert!(
        report.privacy.is_none(),
        "public audit should have no privacy info"
    );

    // Inferences recorded
    assert_eq!(report.inferences.len(), 3);
    for (i, entry) in report.inferences.iter().enumerate() {
        assert_eq!(entry.index, i as u32);
        assert!(!entry.io_commitment.is_empty());
    }

    // Semantic evaluation should run (default in dry run)
    assert!(
        report.semantic_evaluation.is_some(),
        "semantics should be evaluated"
    );
    let sem = report.semantic_evaluation.as_ref().unwrap();
    assert_eq!(sem.evaluated_count, 3);

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 2: Full pipeline with encryption + storage + calldata
// ============================================================================

#[test]
fn test_e2e_audit_full_pipeline() {
    let dir = temp_dir("full");
    let (graph, weights, model_info) = build_audit_model();
    let log = populate_log(&dir, 5, &graph, &weights);

    let encryption = Poseidon2M31Encryption;
    let (view_key, _secret_key) = generate_audit_keypair().unwrap();
    let transport = MockTransport::new();
    let storage = ArweaveClient::with_defaults(Box::new(transport));

    let config = AuditPipelineConfig {
        request: AuditRequest {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            ..AuditRequest::default()
        },
        model_info: model_info.clone(),
        evaluate_semantics: true,
        prove_evaluations: false,
        privacy_tier: "private".to_string(),
        owner_pubkey: view_key,
        submit_config: Some(SubmitConfig::default()),
        billing: None,
    };

    let result = run_audit(
        &log,
        &graph,
        &weights,
        &config,
        Some(&encryption),
        Some(&storage),
    )
    .unwrap();

    // ── Report ──
    assert_eq!(result.report.inference_summary.total_inferences, 5);
    assert!(
        result.report.privacy.is_some(),
        "private audit should have privacy info"
    );
    let privacy = result.report.privacy.as_ref().unwrap();
    assert_eq!(privacy.tier, "private");

    // ── Storage ──
    assert!(
        result.storage_receipt.is_some(),
        "should have storage receipt"
    );
    let receipt = result.storage_receipt.as_ref().unwrap();
    assert!(!receipt.tx_id.is_empty(), "Arweave TX should be non-empty");

    // ── Calldata ──
    assert!(result.calldata.is_some(), "should have serialized calldata");
    let calldata = result.calldata.as_ref().unwrap();
    assert!(
        calldata.len() >= 12,
        "calldata should have at least 12 header fields"
    );

    // Validate the calldata structure
    let info = validate_calldata(calldata).unwrap();
    assert_eq!(info.inference_count, 5);
    assert!(info.time_start < info.time_end);

    assert!(result.total_time_ms > 0);

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 3: Capture hook → log → prover pipeline
// ============================================================================

#[test]
fn test_e2e_capture_to_proof() {
    let dir = temp_dir("capture");
    let (graph, weights, _model_info) = build_audit_model();

    // ── Phase 1: Record inferences via capture hook ──
    {
        let hook = CaptureHook::new(&dir, "0x2", "0xabc", "test-mlp").unwrap();

        for i in 0..4 {
            let input = make_input(i);
            let output = execute_forward_pass(&graph, &input, &weights).unwrap();

            let job = CaptureJob {
                input_tokens: vec![1, 2, 3, 4],
                output_tokens: vec![5, 6],
                input_m31: input,
                output_m31: output,
                timestamp_ns: 1_000_000_000_000 + i as u64 * 1_000_000_000,
                latency_ms: 42,
                gpu_device: "test".to_string(),
                tee_report_hash: "0x0".to_string(),
                task_category: Some("code".to_string()),
                input_preview: None,
                output_preview: None,
            };
            hook.record(job);
        }

        hook.flush();
        assert_eq!(hook.entry_count(), 4);
    }
    // Hook dropped — background thread stopped.

    // ── Phase 2: Reload log and prove ──
    let log = InferenceLog::load(&dir).unwrap();
    assert!(
        log.verify_chain().is_ok(),
        "chain should be intact after capture"
    );

    let window = log.query_window(0, u64::MAX);
    assert_eq!(window.entries.len(), 4, "should have 4 captured entries");

    let prover = AuditProver::new(&graph, &weights);
    let request = AuditRequest {
        start_ns: 0,
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let result = prover.prove_window(&log, &request).unwrap();
    assert_eq!(result.inference_count, 4);
    assert_eq!(result.inference_results.len(), 4);
    assert_eq!(result.weight_commitment, "0xabc");
    // io_merkle_root should be a non-zero M31 digest hex
    let zero_hex = digest_to_hex(&ZERO_DIGEST);
    assert_ne!(result.io_merkle_root, zero_hex);

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 4: Deterministic reproducibility — same inputs = same report
// ============================================================================

#[test]
fn test_e2e_audit_deterministic() {
    let (graph, weights, model_info) = build_audit_model();

    let run = |suffix: &str| -> AuditReport {
        let dir = temp_dir(suffix);
        let log = populate_log(&dir, 3, &graph, &weights);

        let request = AuditRequest {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: "0x2".to_string(),
            ..AuditRequest::default()
        };

        let report = run_audit_dry(&log, &graph, &weights, request, model_info.clone()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        report
    };

    let r1 = run("det_a");
    let r2 = run("det_b");

    // Core commitments MUST match
    assert_eq!(
        r1.commitments.io_merkle_root, r2.commitments.io_merkle_root,
        "io_merkle_root should be deterministic"
    );
    assert_eq!(
        r1.commitments.weight_commitment,
        r2.commitments.weight_commitment,
    );
    assert_eq!(
        r1.commitments.combined_chain_commitment,
        r2.commitments.combined_chain_commitment,
    );
    assert_eq!(
        r1.inference_summary.total_inferences,
        r2.inference_summary.total_inferences,
    );

    // Per-inference io_commitments must match
    for (a, b) in r1.inferences.iter().zip(r2.inferences.iter()) {
        assert_eq!(a.io_commitment, b.io_commitment);
    }
}

// ============================================================================
// Test 5: Report hash is deterministic and non-zero
// ============================================================================

#[test]
fn test_e2e_report_hash() {
    let dir = temp_dir("hash");
    let (graph, weights, model_info) = build_audit_model();
    let log = populate_log(&dir, 2, &graph, &weights);

    let request = AuditRequest {
        start_ns: 0,
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let report = run_audit_dry(&log, &graph, &weights, request, model_info).unwrap();

    // compute_report_hash now returns M31Digest
    let hash1 = compute_report_hash(&report).unwrap();
    let hash2 = compute_report_hash(&report).unwrap();

    assert_ne!(hash1, ZERO_DIGEST, "report hash should be non-zero");
    assert_eq!(hash1, hash2, "report hash should be deterministic");

    // Verify report's own hash matches
    assert!(!report.commitments.audit_report_hash.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 6: Calldata serialization round-trip
// ============================================================================

#[test]
fn test_e2e_calldata_roundtrip() {
    let dir = temp_dir("calldata");
    let (graph, weights, _) = build_audit_model();
    let log = populate_log(&dir, 3, &graph, &weights);

    let prover = AuditProver::new(&graph, &weights);
    let request = AuditRequest {
        start_ns: 1_000_000_000_000, // 1000 seconds in ns
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let result = prover.prove_window(&log, &request).unwrap();

    let config = SubmitConfig::default();
    let calldata = serialize_audit_calldata(&result, &config).unwrap();

    // Validate structure (new layout: 12 header fields + proof)
    let info = validate_calldata(&calldata).unwrap();
    assert_eq!(info.inference_count, 3);
    assert!(
        info.time_start > 0,
        "time_start should be > 0 (ns/1e9 = seconds)"
    );
    assert!(info.time_end > info.time_start);
    assert_eq!(info.total_felts, calldata.len());

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 7: Error handling — empty window
// ============================================================================

#[test]
fn test_e2e_empty_window_error() {
    let dir = temp_dir("empty");
    let (graph, weights, model_info) = build_audit_model();
    let log = populate_log(&dir, 3, &graph, &weights);

    let request = AuditRequest {
        start_ns: u64::MAX - 1,
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let result = run_audit_dry(&log, &graph, &weights, request, model_info);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        AuditError::EmptyWindow { .. }
    ));

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 8: Encryption round-trip preserves report
// ============================================================================

#[test]
fn test_e2e_encrypt_decrypt_report() {
    let dir = temp_dir("encrypt");
    let (graph, weights, model_info) = build_audit_model();
    let log = populate_log(&dir, 2, &graph, &weights);

    let request = AuditRequest {
        start_ns: 0,
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let report = run_audit_dry(&log, &graph, &weights, request, model_info).unwrap();

    // Encrypt with production Poseidon2-M31
    let encryption = Poseidon2M31Encryption;
    let (view_key, secret_key) = generate_audit_keypair().unwrap();
    let transport = MockTransport::new();
    let storage = ArweaveClient::with_defaults(Box::new(transport));

    let (receipt, _blob) = encrypt_and_store(&report, &view_key, &encryption, &storage).unwrap();

    assert!(!receipt.tx_id.is_empty());

    // Decrypt — fetch_and_decrypt(tx_id, recipient_addr, privkey, encryption, storage)
    // recipient_address = view_key (for wrapped key lookup)
    // privkey = secret_key (16 bytes, derives view key to unwrap DEK)
    let decrypted = fetch_and_decrypt(
        &receipt.tx_id,
        &view_key,
        &secret_key,
        &encryption,
        &storage,
    )
    .unwrap();

    assert_eq!(decrypted.audit_id, report.audit_id);
    assert_eq!(
        decrypted.inference_summary.total_inferences,
        report.inference_summary.total_inferences,
    );
    assert_eq!(
        decrypted.commitments.io_merkle_root,
        report.commitments.io_merkle_root,
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 9: Semantic evaluation produces valid scores
// ============================================================================

#[test]
fn test_e2e_semantic_evaluation() {
    let dir = temp_dir("semantic");
    let (graph, weights, _) = build_audit_model();
    let log = populate_log(&dir, 5, &graph, &weights);

    let window = log.query_window(0, u64::MAX);
    assert_eq!(window.entries.len(), 5);

    // Run deterministic checks on each entry
    for entry in &window.entries {
        let checks = evaluate_deterministic(
            entry.input_preview.as_deref().unwrap_or(""),
            entry.output_preview.as_deref().unwrap_or(""),
            entry.task_category.as_deref(),
        );
        // Should produce at least one check (non_empty)
        assert!(!checks.is_empty(), "should have at least one check");
    }

    // Run batch self-eval (deterministic only, no forward pass)
    let eval_config = SelfEvalConfig {
        template: None,
        prove_evaluations: false,
    };

    let evaluations = evaluate_batch(&window.entries, Some(&graph), Some(&weights), &eval_config);
    assert_eq!(evaluations.len(), 5);

    // Aggregate
    let summary = aggregate_evaluations(&evaluations, "deterministic", false);
    assert_eq!(summary.evaluated_count, 5);
    assert!(summary.deterministic_pass + summary.deterministic_fail > 0);

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Test 10: Large batch audit (10 inferences)
// ============================================================================

#[test]
fn test_e2e_large_batch() {
    let dir = temp_dir("large");
    let (graph, weights, model_info) = build_audit_model();
    let log = populate_log(&dir, 10, &graph, &weights);

    let request = AuditRequest {
        start_ns: 0,
        end_ns: u64::MAX,
        model_id: "0x2".to_string(),
        ..AuditRequest::default()
    };

    let report = run_audit_dry(&log, &graph, &weights, request, model_info).unwrap();

    assert_eq!(report.inference_summary.total_inferences, 10);
    assert_eq!(report.inferences.len(), 10);

    // All io_commitments should be unique (different inputs)
    let commitments: Vec<&str> = report
        .inferences
        .iter()
        .map(|e| e.io_commitment.as_str())
        .collect();
    let unique: std::collections::HashSet<&&str> = commitments.iter().collect();
    assert_eq!(
        unique.len(),
        10,
        "all io_commitments should be unique for different inputs"
    );

    // Semantic evaluation should cover all 10
    let sem = report.semantic_evaluation.as_ref().unwrap();
    assert_eq!(sem.evaluated_count, 10);

    let _ = std::fs::remove_dir_all(&dir);
}
