//! Audit report builder and hash computation.
//!
//! Assembles a complete `AuditReport` from audit proving results,
//! semantic evaluation, and infrastructure metadata. The report hash
//! is a Poseidon2-M31 commitment over the canonical JSON representation.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use stwo::core::fields::m31::BaseField as M31;

use crate::audit::digest::{
    digest_to_hex, hash_bytes_m31, hash_felt_hex_m31, u64_to_m31, M31Digest, ZERO_DIGEST,
};
use crate::audit::types::{
    AuditCommitments, AuditError, AuditReport, AuditSemanticSummary, BatchAuditResult,
    BillingInfo, InferenceEntry, InferenceLogEntry, InferenceSummary, InfrastructureInfo,
    ModelInfo, PrivacyInfo, ProofInfo, ReportMetadata, TimeWindow,
};
use crate::crypto::poseidon2_m31::poseidon2_hash;

// ─── Report Hash ────────────────────────────────────────────────────────────

/// Compute the Poseidon2-M31 hash of an audit report.
///
/// Canonical process: serialize to JSON → pack bytes into M31 elements →
/// `poseidon2_hash`. This is the value stored on-chain as `report_hash`.
pub fn compute_report_hash(report: &AuditReport) -> Result<M31Digest, AuditError> {
    let json = serde_json::to_string(report).map_err(|e| AuditError::Serde(e.to_string()))?;

    if json.is_empty() {
        return Ok(ZERO_DIGEST);
    }

    Ok(hash_bytes_m31(json.as_bytes()))
}

// ─── Infrastructure Detection ───────────────────────────────────────────────

/// Auto-detect infrastructure information.
pub fn detect_infrastructure() -> InfrastructureInfo {
    InfrastructureInfo {
        gpu_device: detect_gpu_device(),
        gpu_count: detect_gpu_count(),
        cuda_version: detect_cuda_version(),
        prover_version: env!("CARGO_PKG_VERSION").to_string(),
        tee_active: detect_tee_active(),
        tee_attestation_hash: None,
    }
}

fn detect_gpu_device() -> String {
    #[cfg(feature = "cuda-runtime")]
    {
        crate::backend::gpu_device_name().unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        "cpu-only".to_string()
    }
}

fn detect_gpu_count() -> u32 {
    #[cfg(feature = "cuda-runtime")]
    {
        crate::backend::gpu_device_count().unwrap_or(0)
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        0
    }
}

fn detect_cuda_version() -> String {
    #[cfg(feature = "cuda-runtime")]
    {
        crate::backend::cuda_version().unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        "N/A".to_string()
    }
}

fn detect_tee_active() -> bool {
    #[cfg(feature = "tee")]
    {
        crate::tee::cc_active()
    }
    #[cfg(not(feature = "tee"))]
    {
        false
    }
}

// ─── Timestamp Formatting ───────────────────────────────────────────────────

/// Format nanosecond epoch as ISO 8601 string.
fn format_timestamp_ns(ns: u64) -> String {
    let secs = ns / 1_000_000_000;
    let nanos = (ns % 1_000_000_000) as u32;

    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let (year, month, day) = days_to_ymd(days_since_epoch);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year,
        month,
        day,
        hours,
        minutes,
        seconds,
        nanos / 1_000_000
    )
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m, d)
}

fn now_iso8601() -> String {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    format_timestamp_ns(ns)
}

// ─── Report Builder ─────────────────────────────────────────────────────────

/// Builder for constructing a complete `AuditReport`.
pub struct AuditReportBuilder {
    audit_result: Option<BatchAuditResult>,
    semantic_eval: Option<AuditSemanticSummary>,
    model_info: Option<ModelInfo>,
    infrastructure: Option<InfrastructureInfo>,
    log_entries: Vec<InferenceLogEntry>,
    billing: Option<BillingInfo>,
    privacy: Option<PrivacyInfo>,
}

impl AuditReportBuilder {
    pub fn new() -> Self {
        Self {
            audit_result: None,
            semantic_eval: None,
            model_info: None,
            infrastructure: None,
            log_entries: Vec::new(),
            billing: None,
            privacy: None,
        }
    }

    /// Set the batch audit result (from Dev A's prover).
    pub fn with_audit_result(mut self, result: &BatchAuditResult) -> Self {
        self.audit_result = Some(result.clone());
        self
    }

    /// Set log entries for per-inference detail.
    pub fn with_log_entries(mut self, entries: &[InferenceLogEntry]) -> Self {
        self.log_entries = entries.to_vec();
        self
    }

    /// Set semantic evaluation results.
    pub fn with_semantic_eval(mut self, eval: &AuditSemanticSummary) -> Self {
        self.semantic_eval = Some(eval.clone());
        self
    }

    /// Set model information.
    pub fn with_model_info(mut self, info: ModelInfo) -> Self {
        self.model_info = Some(info);
        self
    }

    /// Auto-detect infrastructure and set it.
    pub fn with_infrastructure(mut self) -> Self {
        self.infrastructure = Some(detect_infrastructure());
        self
    }

    /// Set infrastructure explicitly.
    pub fn with_infrastructure_info(mut self, info: InfrastructureInfo) -> Self {
        self.infrastructure = Some(info);
        self
    }

    /// Set billing information.
    pub fn with_billing(mut self, billing: BillingInfo) -> Self {
        self.billing = Some(billing);
        self
    }

    /// Set privacy information.
    pub fn with_privacy(mut self, privacy: PrivacyInfo) -> Self {
        self.privacy = Some(privacy);
        self
    }

    /// Build the final audit report.
    pub fn build(self) -> Result<AuditReport, AuditError> {
        let result = self
            .audit_result
            .ok_or_else(|| AuditError::Serde("Missing audit result".to_string()))?;

        let infra = self.infrastructure.unwrap_or_else(detect_infrastructure);

        let model = self.model_info.unwrap_or_else(|| ModelInfo {
            model_id: result.model_id.clone(),
            name: String::new(),
            architecture: String::new(),
            parameters: String::new(),
            layers: 0,
            weight_commitment: result.weight_commitment.clone(),
        });

        let time_window = TimeWindow {
            start: format_timestamp_ns(result.time_start * 1_000_000_000),
            end: format_timestamp_ns(result.time_end * 1_000_000_000),
            start_epoch_ns: result.time_start * 1_000_000_000,
            end_epoch_ns: result.time_end * 1_000_000_000,
            duration_seconds: result.time_end.saturating_sub(result.time_start),
        };

        let inference_summary = build_inference_summary(&self.log_entries, &result);
        let inferences = build_inference_entries(&self.log_entries, &self.semantic_eval);

        let commitments = AuditCommitments {
            inference_log_merkle_root: result.log_merkle_root.clone(),
            io_merkle_root: result.io_merkle_root.clone(),
            weight_commitment: result.weight_commitment.clone(),
            combined_chain_commitment: result.combined_chain_commitment.clone(),
            audit_report_hash: String::new(), // Filled after hashing.
        };

        let proof_size: usize = result.proof_calldata.len() * 32;
        let proof_mode = result
            .inference_results
            .first()
            .map(|r| match r.proof_mode {
                crate::audit::types::ProofMode::Gkr => "gkr",
                crate::audit::types::ProofMode::Direct => "direct",
                crate::audit::types::ProofMode::Legacy => "legacy",
            })
            .unwrap_or("legacy")
            .to_string();
        let proof = ProofInfo {
            mode: proof_mode,
            proving_time_seconds: result.proving_time_ms / 1000,
            proof_size_bytes: proof_size,
            on_chain_tx: None,
            on_chain_verified: None,
            arweave_tx_id: None,
            audit_record_id: None,
        };

        // Generate audit ID using Poseidon2-M31.
        let mut audit_id_input = Vec::new();
        audit_id_input.extend_from_slice(&hash_felt_hex_m31(&result.model_id));
        audit_id_input.extend_from_slice(&u64_to_m31(result.time_start));
        audit_id_input.extend_from_slice(&u64_to_m31(result.time_end));
        audit_id_input.push(M31::from(result.inference_count));
        let audit_id_digest = poseidon2_hash(&audit_id_input);

        let mut report = AuditReport {
            version: "1.0.0".to_string(),
            audit_id: digest_to_hex(&audit_id_digest),
            time_window,
            model,
            infrastructure: infra,
            inference_summary,
            semantic_evaluation: self.semantic_eval,
            commitments,
            proof,
            privacy: self.privacy,
            inferences,
            billing: self.billing,
            metadata: ReportMetadata {
                generated_at: now_iso8601(),
                generator: format!("stwo-ml/{}", env!("CARGO_PKG_VERSION")),
            },
        };

        // Compute and set report hash.
        let hash = compute_report_hash(&report)?;
        report.commitments.audit_report_hash = digest_to_hex(&hash);

        Ok(report)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn build_inference_summary(
    entries: &[InferenceLogEntry],
    result: &BatchAuditResult,
) -> InferenceSummary {
    let total = entries.len() as u32;
    let total_input: u64 = entries.iter().map(|e| e.input_tokens.len() as u64).sum();
    let total_output: u64 = entries.iter().map(|e| e.output_tokens.len() as u64).sum();

    let latencies: Vec<u64> = entries.iter().map(|e| e.latency_ms).collect();
    let avg_latency = if latencies.is_empty() {
        0
    } else {
        latencies.iter().sum::<u64>() / latencies.len() as u64
    };

    let p95_latency = if latencies.is_empty() {
        0
    } else {
        let mut sorted = latencies.clone();
        sorted.sort();
        let idx = (sorted.len() as f64 * 0.95) as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    let duration_s = result.time_end.saturating_sub(result.time_start).max(1);
    let throughput = (total_input + total_output) as f32 / duration_s as f32;

    let mut categories: HashMap<String, u32> = HashMap::new();
    for entry in entries {
        let cat = entry
            .task_category
            .as_deref()
            .unwrap_or("general")
            .to_string();
        *categories.entry(cat).or_insert(0) += 1;
    }

    InferenceSummary {
        total_inferences: total,
        total_input_tokens: total_input,
        total_output_tokens: total_output,
        avg_latency_ms: avg_latency,
        p95_latency_ms: p95_latency,
        throughput_tokens_per_sec: throughput,
        categories,
    }
}

fn build_inference_entries(
    entries: &[InferenceLogEntry],
    semantic_eval: &Option<AuditSemanticSummary>,
) -> Vec<InferenceEntry> {
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let score = semantic_eval.as_ref().and_then(|eval| {
                eval.per_inference
                    .iter()
                    .find(|e| e.sequence == entry.sequence_number)
                    .and_then(|e| e.semantic_score)
            });

            InferenceEntry {
                index: i as u32,
                sequence: entry.sequence_number,
                timestamp: format_timestamp_ns(entry.timestamp_ns),
                io_commitment: entry.io_commitment.clone(),
                input_tokens: entry.input_tokens.len() as u32,
                output_tokens: entry.output_tokens.len() as u32,
                latency_ms: entry.latency_ms,
                category: entry.task_category.clone(),
                semantic_score: score,
                input_preview: entry.input_preview.clone(),
                output_preview: entry.output_preview.clone(),
            }
        })
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::types::{InferenceProofResult, ProofMode};

    fn make_batch_result() -> BatchAuditResult {
        BatchAuditResult {
            time_start: 1707900000,
            time_end: 1707903600,
            inference_count: 2,
            io_merkle_root: "0x1234".to_string(),
            log_merkle_root: "0x5678".to_string(),
            weight_commitment: "0xabcd".to_string(),
            combined_chain_commitment: "0xef01".to_string(),
            inference_results: vec![
                InferenceProofResult {
                    sequence: 0,
                    io_commitment: "0xa".to_string(),
                    layer_chain_commitment: "0xb".to_string(),
                    timestamp_ns: 1707900000_000_000_000,
                    proof_size_felts: 100,
                    proving_time_ms: 500,
                    proof_calldata: Vec::new(),
                    io_calldata: Vec::new(),
                    weight_opening_calldata: Vec::new(),
                    weight_commitments_calldata: Vec::new(),
                    proof_mode: ProofMode::Legacy,
                },
                InferenceProofResult {
                    sequence: 1,
                    io_commitment: "0xc".to_string(),
                    layer_chain_commitment: "0xd".to_string(),
                    timestamp_ns: 1707901000_000_000_000,
                    proof_size_felts: 120,
                    proving_time_ms: 600,
                    proof_calldata: Vec::new(),
                    io_calldata: Vec::new(),
                    weight_opening_calldata: Vec::new(),
                    weight_commitments_calldata: Vec::new(),
                    proof_mode: ProofMode::Legacy,
                },
            ],
            model_id: "0x42".to_string(),
            proving_time_ms: 1100,
            proof_calldata: vec!["0x1".to_string(), "0x2".to_string()],
            verification_calldata: None,
            tee_attestation_hash: None,
        }
    }

    #[test]
    fn test_builder_produces_valid_report() {
        let result = make_batch_result();
        let report = AuditReportBuilder::new()
            .with_audit_result(&result)
            .with_infrastructure_info(InfrastructureInfo {
                gpu_device: "test-gpu".to_string(),
                gpu_count: 1,
                cuda_version: "12.4".to_string(),
                prover_version: "0.2.0".to_string(),
                tee_active: false,
                tee_attestation_hash: None,
            })
            .build()
            .unwrap();

        assert_eq!(report.version, "1.0.0");
        assert!(!report.audit_id.is_empty());
        assert_eq!(report.commitments.weight_commitment, "0xabcd");
        assert!(!report.commitments.audit_report_hash.is_empty());
    }

    #[test]
    fn test_report_hash_deterministic() {
        let result = make_batch_result();
        let r1 = AuditReportBuilder::new()
            .with_audit_result(&result)
            .with_infrastructure_info(InfrastructureInfo {
                gpu_device: "test".to_string(),
                gpu_count: 0,
                cuda_version: "N/A".to_string(),
                prover_version: "0.2.0".to_string(),
                tee_active: false,
                tee_attestation_hash: None,
            })
            .build()
            .unwrap();

        // Hash should be non-zero.
        let zero_hex = digest_to_hex(&ZERO_DIGEST);
        assert_ne!(r1.commitments.audit_report_hash, zero_hex);
    }

    #[test]
    fn test_builder_missing_result_errors() {
        let result = AuditReportBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_report_json_roundtrip() {
        let result = make_batch_result();
        let report = AuditReportBuilder::new()
            .with_audit_result(&result)
            .with_infrastructure_info(InfrastructureInfo {
                gpu_device: "test".to_string(),
                gpu_count: 0,
                cuda_version: "N/A".to_string(),
                prover_version: "0.2.0".to_string(),
                tee_active: false,
                tee_attestation_hash: None,
            })
            .build()
            .unwrap();

        let json = serde_json::to_string_pretty(&report).unwrap();
        let roundtrip: AuditReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.audit_id, roundtrip.audit_id);
        assert_eq!(report.version, roundtrip.version);
        assert_eq!(
            report.commitments.audit_report_hash,
            roundtrip.commitments.audit_report_hash
        );
    }

    #[test]
    fn test_timestamp_formatting() {
        let ts = format_timestamp_ns(1771027200_000_000_000);
        assert!(ts.starts_with("2026-02-14"), "Got: {}", ts);
        assert!(ts.ends_with('Z'));
    }
}
