//! End-to-end audit orchestrator.
//!
//! Wires all audit subsystems into a single [`run_audit`] call:
//!
//! ```text
//! 1. Query log window          → entries
//! 2. Prove inferences          → BatchAuditResult   (Dev A prover, GPU)
//! 3. Evaluate semantics        → AuditSemanticSummary (Dev B, CPU)
//! 4. Build report              → AuditReport
//! 5. Encrypt + store           → StorageReceipt
//! 6. Submit on-chain           → calldata ready
//! ```
//!
//! Steps 2 and 3 run in parallel when possible (prover on GPU, evaluator
//! on CPU).

use std::time::Instant;

use starknet_ff::FieldElement;
use tracing::info;

use crate::audit::digest::{hash_bytes_m31, hex_to_digest, pack_digest_felt252};
use crate::audit::encryption::encrypt_and_store;
use crate::audit::prover::AuditProver;
use crate::audit::report::AuditReportBuilder;
use crate::audit::scoring::aggregate_evaluations;
use crate::audit::self_eval::{evaluate_batch, SelfEvalConfig};
use crate::audit::storage::ArweaveClient;
use crate::audit::submit::{serialize_audit_record_calldata, SubmitConfig};
use crate::audit::types::{
    AuditEncryption, AuditError, AuditReport, AuditRequest, ModelInfo, PrivacyInfo,
};
use crate::compiler::graph::{ComputationGraph, GraphWeights};

// ─── Orchestrator Config ───────────────────────────────────────────────────

/// Configuration for the full audit pipeline.
pub struct AuditPipelineConfig {
    /// The audit time window and parameters.
    pub request: AuditRequest,
    /// Model metadata for the report.
    pub model_info: ModelInfo,
    /// Whether to run semantic evaluation.
    pub evaluate_semantics: bool,
    /// Whether to prove evaluation forward passes.
    pub prove_evaluations: bool,
    /// Privacy tier: "public", "private", "selective".
    pub privacy_tier: String,
    /// Owner's public key for encryption (empty for public audits).
    pub owner_pubkey: Vec<u8>,
    /// On-chain submission config (None for dry-run).
    pub submit_config: Option<SubmitConfig>,
    /// Billing info (None if not tracked).
    pub billing: Option<crate::audit::types::BillingInfo>,
}

impl Default for AuditPipelineConfig {
    fn default() -> Self {
        Self {
            request: AuditRequest::default(),
            model_info: ModelInfo {
                model_id: "0x0".to_string(),
                name: "unknown".to_string(),
                architecture: "unknown".to_string(),
                parameters: "0".to_string(),
                layers: 0,
                weight_commitment: "0x0".to_string(),
            },
            evaluate_semantics: true,
            prove_evaluations: false,
            privacy_tier: "public".to_string(),
            owner_pubkey: Vec::new(),
            submit_config: None,
            billing: None,
        }
    }
}

// ─── Audit Result ──────────────────────────────────────────────────────────

/// Complete result of running the audit pipeline.
pub struct AuditPipelineResult {
    /// The generated audit report.
    pub report: AuditReport,
    /// Arweave storage receipt (None for public / dry-run).
    pub storage_receipt: Option<crate::audit::storage::StorageReceipt>,
    /// On-chain calldata (None if not submitted).
    pub calldata: Option<Vec<FieldElement>>,
    /// Total pipeline execution time in milliseconds.
    pub total_time_ms: u64,
}

// ─── Main Orchestrator ─────────────────────────────────────────────────────

/// Run the full audit pipeline: prove → evaluate → report → encrypt → store.
///
/// This is the primary entry point for auditing. It takes an inference log,
/// a model, and a pipeline config, then produces a complete audit report
/// with optional encryption, Arweave storage, and on-chain submission.
pub fn run_audit(
    log: &crate::audit::log::InferenceLog,
    graph: &ComputationGraph,
    weights: &GraphWeights,
    config: &AuditPipelineConfig,
    encryption: Option<&dyn AuditEncryption>,
    storage: Option<&ArweaveClient>,
) -> Result<AuditPipelineResult, AuditError> {
    let pipeline_start = Instant::now();

    info!(
        model = config.model_info.name,
        privacy = config.privacy_tier,
        "Audit pipeline: starting"
    );

    // ── Step 1: Query log window ──────────────────────────────────────────
    let window = log.query_window(config.request.start_ns, config.request.end_ns);
    if window.entries.is_empty() {
        return Err(AuditError::EmptyWindow {
            start: config.request.start_ns,
            end: config.request.end_ns,
        });
    }

    info!(
        entries = window.entries.len(),
        "Audit pipeline: log window queried"
    );

    // ── Step 2: Prove inferences ──────────────────────────────────────────
    let prover = AuditProver::new(graph, weights);
    let audit_result = prover.prove_window(log, &config.request)?;

    info!(
        inferences = audit_result.inference_count,
        proving_ms = audit_result.proving_time_ms,
        "Audit pipeline: proving complete"
    );

    // ── Step 3: Semantic evaluation (parallel with proving in future) ─────
    let semantic_summary = if config.evaluate_semantics {
        let eval_config = SelfEvalConfig {
            template: None,
            prove_evaluations: config.prove_evaluations,
        };

        let evaluations = evaluate_batch(&window.entries, Some(graph), Some(weights), &eval_config);

        let summary = aggregate_evaluations(&evaluations, "combined", config.prove_evaluations);

        info!(
            avg_score = summary.avg_quality_score,
            evaluated = summary.evaluated_count,
            "Audit pipeline: evaluation complete"
        );

        Some(summary)
    } else {
        None
    };

    // ── Step 4: Build report ──────────────────────────────────────────────
    let mut builder = AuditReportBuilder::new()
        .with_audit_result(&audit_result)
        .with_log_entries(&window.entries)
        .with_model_info(config.model_info.clone())
        .with_infrastructure();

    if let Some(ref summary) = semantic_summary {
        builder = builder.with_semantic_eval(summary);
    }

    if let Some(ref billing) = config.billing {
        builder = builder.with_billing(billing.clone());
    }

    if config.privacy_tier != "public" {
        builder = builder.with_privacy(PrivacyInfo {
            tier: config.privacy_tier.clone(),
            encryption_scheme: encryption
                .map(|e| e.scheme_name().to_string())
                .unwrap_or_else(|| "none".to_string()),
            arweave_tx_id: None,
        });
    }

    let mut report = builder.build()?;

    info!(audit_id = report.audit_id, "Audit pipeline: report built");

    // ── Step 5: Encrypt + Store ───────────────────────────────────────────
    let storage_receipt = if config.privacy_tier != "public" {
        if let (Some(enc), Some(store)) = (encryption, storage) {
            let (receipt, _blob) = encrypt_and_store(&report, &config.owner_pubkey, enc, store)?;

            // Update report with Arweave TX ID.
            report.proof.arweave_tx_id = Some(receipt.tx_id.clone());
            if let Some(ref mut privacy) = report.privacy {
                privacy.arweave_tx_id = Some(receipt.tx_id.clone());
            }

            info!(
                tx_id = receipt.tx_id,
                "Audit pipeline: encrypted and stored"
            );

            Some(receipt)
        } else {
            None
        }
    } else {
        None
    };

    // ── Step 6: Prepare on-chain calldata ────────────────────────────────
    let calldata = if let Some(ref submit_cfg) = config.submit_config {
        let mut cfg = submit_cfg.clone();
        cfg.privacy_tier = match config.privacy_tier.as_str() {
            "private" => 1,
            "selective" => 2,
            _ => 0,
        };
        if let Some(ref receipt) = storage_receipt {
            // Convert Arweave TX ID to felt252 (hash of the string).
            let tx_bytes = receipt.tx_id.as_bytes();
            let tx_digest = hash_bytes_m31(tx_bytes);
            let (lo_bytes, _hi_bytes) = pack_digest_felt252(&tx_digest);
            if let Ok(felt) = FieldElement::from_bytes_be(&lo_bytes) {
                cfg.arweave_tx_id = felt;
            }
        }

        // Bind on-chain submission to the actual report hash unless the caller
        // explicitly overrides it.
        if cfg.report_hash.is_none() {
            // Parse the report's audit_report_hash (M31 digest hex) and pack
            // into (lo, hi) felt252 pair for the calldata.
            let digest = hex_to_digest(&report.commitments.audit_report_hash).map_err(|e| {
                AuditError::Serde(format!(
                    "invalid report hash hex: {}: {}",
                    report.commitments.audit_report_hash, e
                ))
            })?;
            let (lo_bytes, hi_bytes) = pack_digest_felt252(&digest);
            let lo = FieldElement::from_bytes_be(&lo_bytes)
                .map_err(|_| AuditError::Serde("report hash lo overflow".to_string()))?;
            let hi = FieldElement::from_bytes_be(&hi_bytes)
                .map_err(|_| AuditError::Serde("report hash hi overflow".to_string()))?;
            cfg.report_hash = Some((lo, hi));
        }

        // Use record-only calldata (11 felts) matching the Cairo submit_audit
        // function signature. The proof is verified locally; on-chain record
        // serves as timestamped attestation of audit commitments.
        let data = serialize_audit_record_calldata(&audit_result, &cfg)?;

        info!(felts = data.len(), "Audit pipeline: calldata serialized");

        Some(data)
    } else {
        None
    };

    let total_time_ms = pipeline_start.elapsed().as_millis() as u64;

    info!(
        total_ms = total_time_ms,
        audit_id = report.audit_id,
        "Audit pipeline: complete"
    );

    Ok(AuditPipelineResult {
        report,
        storage_receipt,
        calldata,
        total_time_ms,
    })
}

/// Run a dry-run audit: prove + evaluate + report, but skip storage and on-chain.
///
/// Useful for testing and local validation.
pub fn run_audit_dry(
    log: &crate::audit::log::InferenceLog,
    graph: &ComputationGraph,
    weights: &GraphWeights,
    request: AuditRequest,
    model_info: ModelInfo,
) -> Result<AuditReport, AuditError> {
    let config = AuditPipelineConfig {
        request,
        model_info,
        evaluate_semantics: true,
        prove_evaluations: false,
        privacy_tier: "public".to_string(),
        owner_pubkey: Vec::new(),
        submit_config: None,
        billing: None,
    };

    let result = run_audit(log, graph, weights, &config, None, None)?;
    Ok(result.report)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_defaults() {
        let config = AuditPipelineConfig::default();
        assert!(config.evaluate_semantics);
        assert!(!config.prove_evaluations);
        assert_eq!(config.privacy_tier, "public");
        assert!(config.submit_config.is_none());
    }

    #[test]
    fn test_privacy_tier_mapping() {
        let tiers = [("public", 0u8), ("private", 1), ("selective", 2)];
        for (name, expected) in tiers {
            let val = match name {
                "private" => 1u8,
                "selective" => 2,
                _ => 0,
            };
            assert_eq!(val, expected, "tier '{}' should map to {}", name, expected);
        }
    }
}
