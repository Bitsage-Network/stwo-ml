//! On-chain submission of audit records.
//!
//! Serializes a `BatchAuditResult` into Starknet calldata matching the
//! Cairo `submit_audit` entry point, and provides a dry-run validator.
//!
//! ```text
//! BatchAuditResult ──serialize──> Vec<FieldElement> (calldata)
//!                                        │
//!                   ┌────────────────────┤
//!                   │ (caller provides   │
//!                   │  transport layer)   │
//!                   v                    v
//!             submit_audit()       dry_run_check()
//!              (on-chain)           (local)
//! ```
//!
//! The actual on-chain submission (signing, paymaster, RPC call) is handled
//! by the pipeline scripts (`paymaster_submit.mjs`) or the Rust SDK.
//! This module focuses on calldata construction and local validation.

use starknet_ff::FieldElement;

use crate::audit::digest::{
    hash_felt_hex_m31, hex_to_digest, pack_digest_felt252, u32_to_m31, u64_to_m31, M31Digest,
};
#[cfg(test)]
use crate::audit::types::GkrInferenceCalldata;
use crate::audit::types::{AuditError, BatchAuditResult, VerificationCalldata};
use crate::crypto::poseidon2_m31::poseidon2_hash;

/// Receipt from an on-chain audit submission.
#[derive(Debug, Clone)]
pub struct SubmitReceipt {
    /// On-chain audit ID (hex felt252).
    pub audit_id: String,
    /// Transaction hash (hex).
    pub tx_hash: String,
    /// Block number the TX was included in.
    pub block_number: u64,
    /// Explorer URL for the transaction.
    pub explorer_url: String,
}

/// Configuration for on-chain submission.
#[derive(Debug, Clone)]
pub struct SubmitConfig {
    /// Target contract address (hex felt252).
    pub contract_address: String,
    /// Network: "sepolia" or "mainnet".
    pub network: String,
    /// Report hash as (lo, hi) packed M31 digest.
    /// If None, computed from the BatchAuditResult commitments.
    pub report_hash: Option<(FieldElement, FieldElement)>,
    /// Privacy tier: 0 = public, 1 = encrypted, 2 = selective.
    pub privacy_tier: u8,
    /// Arweave TX ID for the encrypted report blob (felt252, 0 if none).
    pub arweave_tx_id: FieldElement,
}

impl Default for SubmitConfig {
    fn default() -> Self {
        Self {
            contract_address: "0x03f937cb00db86933c94a680ce2cb2df3296e7680df3547c610aa929ffba860c"
                .to_string(),
            network: "sepolia".to_string(),
            report_hash: None,
            privacy_tier: 0,
            arweave_tx_id: FieldElement::ZERO,
        }
    }
}

/// Serialize a `BatchAuditResult` into calldata for `submit_audit`.
///
/// Calldata layout (matches Cairo contract with PackedDigest8):
/// ```text
/// [0]     model_id              : felt252
/// [1-2]   report_hash           : PackedDigest8 (lo, hi)
/// [3-4]   merkle_root           : PackedDigest8 (lo, hi)
/// [5]     weight_commitment     : felt252
/// [6]     time_start            : u64 (as felt252)
/// [7]     time_end              : u64 (as felt252)
/// [8]     inference_count       : u32 (as felt252)
/// [9]     tee_attestation_hash  : felt252
/// [10]    privacy_tier          : u8  (as felt252)
/// [11]    proof_calldata_len    : u32 (as felt252)
/// [12..]  proof_calldata        : felt252[]
/// ```
pub fn serialize_audit_calldata(
    result: &BatchAuditResult,
    config: &SubmitConfig,
) -> Result<Vec<FieldElement>, AuditError> {
    let mut calldata = Vec::new();

    // [0] model_id
    calldata.push(parse_felt(&result.model_id, "model_id")?);

    // [1-2] report_hash — either provided or computed from commitments
    let (rh_lo, rh_hi) = match config.report_hash {
        Some((lo, hi)) => (lo, hi),
        None => compute_report_hash(result)?,
    };
    calldata.push(rh_lo);
    calldata.push(rh_hi);

    // [3-4] inference_log_merkle_root — M31 digest hex → PackedDigest8
    let (mr_lo, mr_hi) = digest_hex_to_felts(&result.log_merkle_root, "log_merkle_root")?;
    calldata.push(mr_lo);
    calldata.push(mr_hi);

    // [5] weight_commitment — stays as felt252
    calldata.push(parse_felt(&result.weight_commitment, "weight_commitment")?);

    // [6] time_start
    calldata.push(FieldElement::from(result.time_start));

    // [7] time_end
    calldata.push(FieldElement::from(result.time_end));

    // [8] inference_count
    calldata.push(FieldElement::from(result.inference_count as u64));

    // [9] tee_attestation_hash
    let tee_hash = result.tee_attestation_hash.as_deref().unwrap_or("0x0");
    calldata.push(parse_felt(tee_hash, "tee_attestation_hash")?);

    // [10] privacy_tier
    calldata.push(FieldElement::from(config.privacy_tier as u64));

    // [11] proof_calldata length prefix (Cairo Span serialization)
    let proof_felts: Vec<FieldElement> = result
        .proof_calldata
        .iter()
        .map(|s| parse_felt(s, "proof_calldata element"))
        .collect::<Result<Vec<_>, _>>()?;
    calldata.push(FieldElement::from(proof_felts.len() as u64));

    // [12..] proof_calldata elements
    calldata.extend_from_slice(&proof_felts);

    Ok(calldata)
}

/// Serialize a `BatchAuditResult` for `submit_audit_record` (no proof).
///
/// Same header as `serialize_audit_calldata` but without proof_calldata fields.
/// For off-chain-verified audits where proof was checked locally.
pub fn serialize_audit_record_calldata(
    result: &BatchAuditResult,
    config: &SubmitConfig,
) -> Result<Vec<FieldElement>, AuditError> {
    let mut calldata = Vec::new();

    calldata.push(parse_felt(&result.model_id, "model_id")?);

    let (rh_lo, rh_hi) = match config.report_hash {
        Some((lo, hi)) => (lo, hi),
        None => compute_report_hash(result)?,
    };
    calldata.push(rh_lo);
    calldata.push(rh_hi);

    let (mr_lo, mr_hi) = digest_hex_to_felts(&result.log_merkle_root, "log_merkle_root")?;
    calldata.push(mr_lo);
    calldata.push(mr_hi);

    calldata.push(parse_felt(&result.weight_commitment, "weight_commitment")?);
    calldata.push(FieldElement::from(result.time_start));
    calldata.push(FieldElement::from(result.time_end));
    calldata.push(FieldElement::from(result.inference_count as u64));

    let tee_hash = result.tee_attestation_hash.as_deref().unwrap_or("0x0");
    calldata.push(parse_felt(tee_hash, "tee_attestation_hash")?);

    calldata.push(FieldElement::from(config.privacy_tier as u64));

    Ok(calldata)
}

/// Compute the report hash from a `BatchAuditResult` (M31-native Poseidon2).
///
/// Hashes the key commitments via Poseidon2-M31, returns packed (lo, hi) felt252 pair:
/// `poseidon2(model_id_m31, io_merkle_root[8], log_merkle_root[8], weight_commitment_m31,
///   combined_chain_commitment[8], time_start[2], time_end[2], inference_count)`
pub fn compute_report_hash(
    result: &BatchAuditResult,
) -> Result<(FieldElement, FieldElement), AuditError> {
    use stwo::core::fields::m31::M31;

    let mut input: Vec<M31> = Vec::new();

    // model_id — hash felt252 hex into M31 space
    input.extend_from_slice(&hash_felt_hex_m31(&result.model_id));

    // io_merkle_root — M31 digest hex → 8 M31 elements
    let io_root = hex_to_digest(&result.io_merkle_root)
        .map_err(|e| AuditError::Serde(format!("io_merkle_root: {}", e)))?;
    input.extend_from_slice(&io_root);

    // log_merkle_root — M31 digest hex → 8 M31 elements
    let log_root = hex_to_digest(&result.log_merkle_root)
        .map_err(|e| AuditError::Serde(format!("log_merkle_root: {}", e)))?;
    input.extend_from_slice(&log_root);

    // weight_commitment — hash felt252 hex into M31 space
    input.extend_from_slice(&hash_felt_hex_m31(&result.weight_commitment));

    // combined_chain_commitment — M31 digest hex → 8 M31 elements
    let chain = hex_to_digest(&result.combined_chain_commitment)
        .map_err(|e| AuditError::Serde(format!("combined_chain_commitment: {}", e)))?;
    input.extend_from_slice(&chain);

    // Scalars
    input.extend_from_slice(&u64_to_m31(result.time_start));
    input.extend_from_slice(&u64_to_m31(result.time_end));
    input.push(u32_to_m31(result.inference_count));

    let digest = poseidon2_hash(&input);
    digest_to_felt_pair(&digest)
}

/// Validate calldata before submission (dry-run check).
///
/// Checks:
/// - Minimum 12-field header present
/// - Inference count > 0
/// - Time window is valid (end > start)
/// - Calldata size matches proof_calldata_len
pub fn validate_calldata(calldata: &[FieldElement]) -> Result<CalldataInfo, AuditError> {
    if calldata.len() < 12 {
        return Err(AuditError::Serde(format!(
            "calldata too short: {} elements (minimum 12)",
            calldata.len()
        )));
    }

    // Field indices in new layout:
    // [0] model_id, [1-2] report_hash, [3-4] merkle_root,
    // [5] weight_commitment, [6] time_start, [7] time_end,
    // [8] inference_count, [9] tee_hash, [10] privacy_tier,
    // [11] proof_calldata_len, [12..] proof_calldata

    let time_start: u64 = calldata[6]
        .try_into()
        .map_err(|_| AuditError::Serde("time_start doesn't fit u64".to_string()))?;
    let time_end: u64 = calldata[7]
        .try_into()
        .map_err(|_| AuditError::Serde("time_end doesn't fit u64".to_string()))?;

    if time_end <= time_start {
        return Err(AuditError::Serde(format!(
            "invalid time window: end ({}) <= start ({})",
            time_end, time_start
        )));
    }

    let inference_count: u64 = calldata[8]
        .try_into()
        .map_err(|_| AuditError::Serde("inference_count doesn't fit u64".to_string()))?;

    if inference_count == 0 {
        return Err(AuditError::Serde("inference_count is 0".to_string()));
    }

    let proof_len: u64 = calldata[11]
        .try_into()
        .map_err(|_| AuditError::Serde("proof_calldata_len doesn't fit u64".to_string()))?;

    let expected_total = 12 + proof_len as usize;
    if calldata.len() != expected_total {
        return Err(AuditError::Serde(format!(
            "calldata length mismatch: got {}, expected {} (12 header + {} proof)",
            calldata.len(),
            expected_total,
            proof_len
        )));
    }

    Ok(CalldataInfo {
        model_id: format!("{:#066x}", calldata[0]),
        report_hash_lo: format!("{:#066x}", calldata[1]),
        report_hash_hi: format!("{:#066x}", calldata[2]),
        time_start,
        time_end,
        inference_count: inference_count as u32,
        proof_calldata_len: proof_len as usize,
        total_felts: calldata.len(),
    })
}

/// Summary of validated calldata.
#[derive(Debug, Clone)]
pub struct CalldataInfo {
    pub model_id: String,
    pub report_hash_lo: String,
    pub report_hash_hi: String,
    pub time_start: u64,
    pub time_end: u64,
    pub inference_count: u32,
    pub proof_calldata_len: usize,
    pub total_felts: usize,
}

/// Build the explorer URL for a transaction.
pub fn explorer_url(network: &str, tx_hash: &str) -> String {
    match network {
        "mainnet" => format!("https://starkscan.co/tx/{}", tx_hash),
        _ => format!("https://sepolia.starkscan.co/tx/{}", tx_hash),
    }
}

/// Configuration for GKR on-chain verification.
///
/// Contains model metadata needed by `verify_model_gkr()` that isn't
/// part of the proof itself.
#[derive(Debug, Clone)]
pub struct GkrVerificationConfig {
    /// Circuit depth (log2 of the largest layer dimension).
    pub circuit_depth: u32,
    /// Number of layers in the model.
    pub num_layers: u32,
    /// Flat matmul dimensions as `[rows0, cols0, k0, rows1, cols1, k1, ...]`.
    /// Matches the Cairo contract's `matmul_dims: Array<u32>`.
    pub matmul_dims: Vec<u32>,
    /// Per-layer dequantize bit widths.
    /// Matches the Cairo contract's `dequantize_bits: Array<u64>`.
    /// Empty if no quantization layers.
    pub dequantize_bits: Vec<u64>,
}

/// Build calldata for `verify_model_gkr()` from verification calldata.
///
/// Assembles per-inference calldata vectors matching the EloVerifier's
/// `verify_model_gkr(model_id, raw_io_data, circuit_depth, num_layers,
///   matmul_dims, dequantize_bits, proof_data, weight_commitments,
///   weight_opening_proofs)` entry point.
///
/// Returns one `Vec<FieldElement>` per inference.
pub fn build_gkr_verification_calldata(
    result: &BatchAuditResult,
    config: &GkrVerificationConfig,
) -> Result<Vec<Vec<FieldElement>>, AuditError> {
    let verification = result.verification_calldata.as_ref().ok_or_else(|| {
        AuditError::Serde("no verification calldata (mode is not GKR)".to_string())
    })?;

    let per_inference = match verification {
        VerificationCalldata::Gkr { per_inference } => per_inference,
        VerificationCalldata::Direct { .. } => {
            return Err(AuditError::Serde(
                "expected GKR verification calldata, got Direct".to_string(),
            ));
        }
    };

    let mut all_calldata = Vec::with_capacity(per_inference.len());

    for inference in per_inference {
        let mut calldata = Vec::new();

        // model_id
        calldata.push(parse_felt(&inference.model_id, "model_id")?);

        // raw_io_data (length-prefixed)
        let io_felts: Vec<FieldElement> = inference
            .io_calldata
            .iter()
            .map(|s| parse_felt(s, "io_calldata"))
            .collect::<Result<Vec<_>, _>>()?;
        calldata.push(FieldElement::from(io_felts.len() as u64));
        calldata.extend_from_slice(&io_felts);

        // circuit_depth
        calldata.push(FieldElement::from(config.circuit_depth as u64));

        // num_layers
        calldata.push(FieldElement::from(config.num_layers as u64));

        // matmul_dims: Array<u32> — flat [len, d0, d1, d2, ...]
        calldata.push(FieldElement::from(config.matmul_dims.len() as u64));
        for &d in &config.matmul_dims {
            calldata.push(FieldElement::from(d as u64));
        }

        // dequantize_bits: Array<u64> — [len, b0, b1, ...]
        calldata.push(FieldElement::from(config.dequantize_bits.len() as u64));
        for &b in &config.dequantize_bits {
            calldata.push(FieldElement::from(b));
        }

        // proof_data: Array<felt252> (length-prefixed)
        let proof_felts: Vec<FieldElement> = inference
            .gkr_calldata
            .iter()
            .map(|s| parse_felt(s, "gkr_calldata"))
            .collect::<Result<Vec<_>, _>>()?;
        calldata.push(FieldElement::from(proof_felts.len() as u64));
        calldata.extend_from_slice(&proof_felts);

        // weight_commitments: Array<felt252> (length-prefixed)
        let wc_felts: Vec<FieldElement> = inference
            .weight_commitments
            .iter()
            .map(|s| parse_felt(s, "weight_commitment"))
            .collect::<Result<Vec<_>, _>>()?;
        calldata.push(FieldElement::from(wc_felts.len() as u64));
        calldata.extend_from_slice(&wc_felts);

        // weight_opening_proofs: Array<MleOpeningProof> (already serialized
        // with the leading array length prefix).
        let weight_felts: Vec<FieldElement> = inference
            .weight_opening_calldata
            .iter()
            .map(|s| parse_felt(s, "weight_opening_calldata"))
            .collect::<Result<Vec<_>, _>>()?;
        if weight_felts.is_empty() {
            return Err(AuditError::Serde(
                "weight_opening_calldata must include Array<MleOpeningProof> length prefix"
                    .to_string(),
            ));
        }
        calldata.extend_from_slice(&weight_felts);

        all_calldata.push(calldata);
    }

    Ok(all_calldata)
}

/// Build calldata for `verify_model_direct()` from verification calldata.
///
/// For Direct mode, the combined calldata from `StarknetModelProof` is already
/// in the correct format. This function just parses the hex strings.
///
/// Returns one `Vec<FieldElement>` per inference.
pub fn build_direct_verification_calldata(
    result: &BatchAuditResult,
) -> Result<Vec<Vec<FieldElement>>, AuditError> {
    let verification = result.verification_calldata.as_ref().ok_or_else(|| {
        AuditError::Serde("no verification calldata (mode is not Direct)".to_string())
    })?;

    let per_inference = match verification {
        VerificationCalldata::Direct { per_inference } => per_inference,
        VerificationCalldata::Gkr { .. } => {
            return Err(AuditError::Serde(
                "expected Direct verification calldata, got GKR".to_string(),
            ));
        }
    };

    let mut all_calldata = Vec::with_capacity(per_inference.len());

    for inference_data in per_inference {
        let calldata: Vec<FieldElement> = inference_data
            .iter()
            .map(|s| parse_felt(s, "direct_calldata"))
            .collect::<Result<Vec<_>, _>>()?;
        all_calldata.push(calldata);
    }

    Ok(all_calldata)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse a hex string into a FieldElement.
pub(crate) fn parse_felt(hex: &str, field_name: &str) -> Result<FieldElement, AuditError> {
    FieldElement::from_hex_be(hex)
        .map_err(|_| AuditError::Serde(format!("invalid hex for {}: {}", field_name, hex)))
}

/// Convert an M31 digest hex string to a (lo, hi) felt252 pair.
pub(crate) fn digest_hex_to_felts(hex: &str, name: &str) -> Result<(FieldElement, FieldElement), AuditError> {
    let digest = hex_to_digest(hex).map_err(|e| AuditError::Serde(format!("{}: {}", name, e)))?;
    digest_to_felt_pair(&digest)
}

/// Pack an M31Digest into a (lo, hi) FieldElement pair.
fn digest_to_felt_pair(digest: &M31Digest) -> Result<(FieldElement, FieldElement), AuditError> {
    let (lo_bytes, hi_bytes) = pack_digest_felt252(digest);
    let lo = FieldElement::from_bytes_be(&lo_bytes)
        .map_err(|_| AuditError::Serde("packed digest lo overflow".to_string()))?;
    let hi = FieldElement::from_bytes_be(&hi_bytes)
        .map_err(|_| AuditError::Serde("packed digest hi overflow".to_string()))?;
    Ok((lo, hi))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::digest::{digest_to_hex, ZERO_DIGEST};
    use crate::audit::types::{BatchAuditResult, InferenceProofResult, ProofMode};
    use stwo::core::fields::m31::M31;

    /// Build a test digest with a single non-zero element for easy identification.
    fn test_digest(val: u32) -> M31Digest {
        let mut d = ZERO_DIGEST;
        d[0] = M31::from(val);
        d
    }

    fn make_batch_result(n: usize) -> BatchAuditResult {
        let results: Vec<InferenceProofResult> = (0..n)
            .map(|i| InferenceProofResult {
                sequence: i as u64,
                io_commitment: format!("{:#066x}", FieldElement::from((i + 1) as u64)),
                layer_chain_commitment: format!("{:#066x}", FieldElement::from((i + 10) as u64)),
                timestamp_ns: 1_000_000_000_000 + i as u64 * 1_000_000,
                proof_size_felts: 100,
                proving_time_ms: 50,
                proof_calldata: Vec::new(),
                io_calldata: Vec::new(),
                weight_opening_calldata: Vec::new(),
                weight_commitments_calldata: Vec::new(),
                proof_mode: ProofMode::Legacy,
            })
            .collect();

        BatchAuditResult {
            time_start: 1000,
            time_end: 2000,
            inference_count: n as u32,
            // M31-native digest hex strings for audit fields
            io_merkle_root: digest_to_hex(&test_digest(0xABC)),
            log_merkle_root: digest_to_hex(&test_digest(0xDEF)),
            combined_chain_commitment: digest_to_hex(&test_digest(0x456)),
            // Felt252 hex for ZKML-boundary fields
            weight_commitment: format!("{:#066x}", FieldElement::from(0x123u64)),
            inference_results: results,
            model_id: "0x2".to_string(),
            proving_time_ms: 1000,
            proof_calldata: vec![
                format!("{:#066x}", FieldElement::from(0x111u64)),
                format!("{:#066x}", FieldElement::from(0x222u64)),
                format!("{:#066x}", FieldElement::from(0x333u64)),
            ],
            verification_calldata: None,
            tee_attestation_hash: None,
        }
    }

    #[test]
    fn test_serialize_roundtrip() {
        let result = make_batch_result(5);
        let config = SubmitConfig::default();

        let calldata = serialize_audit_calldata(&result, &config).unwrap();

        // Header (12) + proof_calldata (3) = 15 total
        assert_eq!(calldata.len(), 15);

        // Validate
        let info = validate_calldata(&calldata).unwrap();
        assert_eq!(info.model_id, format!("{:#066x}", FieldElement::from(2u64)));
        assert_eq!(info.time_start, 1000);
        assert_eq!(info.time_end, 2000);
        assert_eq!(info.inference_count, 5);
        assert_eq!(info.proof_calldata_len, 3);
    }

    #[test]
    fn test_serialize_record_no_proof() {
        let result = make_batch_result(3);
        let config = SubmitConfig::default();

        let calldata = serialize_audit_record_calldata(&result, &config).unwrap();

        // 11 fields (no proof_calldata_len or proof)
        assert_eq!(calldata.len(), 11);

        // model_id
        assert_eq!(calldata[0], FieldElement::from(2u64));
        // time_start at index 6
        assert_eq!(calldata[6], FieldElement::from(1000u64));
        // time_end at index 7
        assert_eq!(calldata[7], FieldElement::from(2000u64));
        // inference_count at index 8
        assert_eq!(calldata[8], FieldElement::from(3u64));
        // privacy_tier at index 10 (default = 0)
        assert_eq!(calldata[10], FieldElement::ZERO);
    }

    #[test]
    fn test_report_hash_deterministic() {
        let result = make_batch_result(5);

        let (lo1, hi1) = compute_report_hash(&result).unwrap();
        let (lo2, hi2) = compute_report_hash(&result).unwrap();
        assert_eq!(lo1, lo2);
        assert_eq!(hi1, hi2);
        // At least one half should be non-zero
        assert!(lo1 != FieldElement::ZERO || hi1 != FieldElement::ZERO);
    }

    #[test]
    fn test_report_hash_changes_with_input() {
        let r1 = make_batch_result(5);
        let mut r2 = make_batch_result(5);
        r2.inference_count = 10;

        let (lo1, hi1) = compute_report_hash(&r1).unwrap();
        let (lo2, hi2) = compute_report_hash(&r2).unwrap();
        assert!(lo1 != lo2 || hi1 != hi2);
    }

    #[test]
    fn test_validate_rejects_short_calldata() {
        let calldata = vec![FieldElement::ZERO; 5];
        assert!(validate_calldata(&calldata).is_err());
    }

    #[test]
    fn test_validate_rejects_invalid_time_window() {
        // Build calldata with end <= start (new layout: 12 fields min)
        let mut calldata = vec![FieldElement::ZERO; 12];
        calldata[6] = FieldElement::from(2000u64); // time_start
        calldata[7] = FieldElement::from(1000u64); // time_end (less than start)
        calldata[8] = FieldElement::from(1u64); // inference_count
        calldata[11] = FieldElement::ZERO; // proof_calldata_len = 0

        assert!(validate_calldata(&calldata).is_err());
    }

    #[test]
    fn test_validate_rejects_zero_inferences() {
        let mut calldata = vec![FieldElement::ZERO; 12];
        calldata[6] = FieldElement::from(1000u64);
        calldata[7] = FieldElement::from(2000u64);
        calldata[8] = FieldElement::ZERO; // inference_count = 0
        calldata[11] = FieldElement::ZERO;

        assert!(validate_calldata(&calldata).is_err());
    }

    #[test]
    fn test_validate_rejects_length_mismatch() {
        let mut calldata = vec![FieldElement::ZERO; 12];
        calldata[6] = FieldElement::from(1000u64);
        calldata[7] = FieldElement::from(2000u64);
        calldata[8] = FieldElement::from(1u64);
        calldata[11] = FieldElement::from(5u64); // says 5 proof elements, but none follow

        assert!(validate_calldata(&calldata).is_err());
    }

    #[test]
    fn test_custom_report_hash() {
        let result = make_batch_result(3);
        let custom_lo = FieldElement::from(0xCAFEu64);
        let custom_hi = FieldElement::from(0xBEEFu64);
        let config = SubmitConfig {
            report_hash: Some((custom_lo, custom_hi)),
            ..SubmitConfig::default()
        };

        let calldata = serialize_audit_calldata(&result, &config).unwrap();
        assert_eq!(calldata[1], custom_lo);
        assert_eq!(calldata[2], custom_hi);
    }

    #[test]
    fn test_calldata_size_reasonable() {
        let mut result = make_batch_result(10);
        result.proof_calldata = (0..1000)
            .map(|i| format!("{:#066x}", FieldElement::from(i as u64)))
            .collect();

        let config = SubmitConfig::default();
        let calldata = serialize_audit_calldata(&result, &config).unwrap();
        assert!(calldata.len() < 100_000);
        assert_eq!(calldata.len(), 1012); // 12 header + 1000 proof
    }

    #[test]
    fn test_explorer_url_sepolia() {
        let url = explorer_url("sepolia", "0xabc");
        assert!(url.contains("sepolia.starkscan.co"));
        assert!(url.contains("0xabc"));
    }

    #[test]
    fn test_explorer_url_mainnet() {
        let url = explorer_url("mainnet", "0xdef");
        assert!(url.contains("starkscan.co"));
        assert!(!url.contains("sepolia"));
    }

    #[test]
    fn test_build_gkr_verification_calldata() {
        let gkr_inference = GkrInferenceCalldata {
            model_id: "0x2".to_string(),
            gkr_calldata: vec![
                format!("{:#066x}", FieldElement::from(0xA1u64)),
                format!("{:#066x}", FieldElement::from(0xA2u64)),
            ],
            io_calldata: vec![format!("{:#066x}", FieldElement::from(0xB1u64))],
            weight_commitments: vec![format!("{:#066x}", FieldElement::from(0xF1u64))],
            weight_opening_calldata: vec![format!("{:#066x}", FieldElement::ZERO)],
        };

        let mut result = make_batch_result(1);
        result.verification_calldata = Some(VerificationCalldata::Gkr {
            per_inference: vec![gkr_inference],
        });

        let config = GkrVerificationConfig {
            circuit_depth: 4,
            num_layers: 2,
            matmul_dims: vec![4, 4, 4],
            dequantize_bits: vec![],
        };

        let calldata = build_gkr_verification_calldata(&result, &config).unwrap();
        assert_eq!(calldata.len(), 1);

        let cd = &calldata[0];
        let mut i = 0;
        assert_eq!(cd[i], FieldElement::from(2u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(1u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(0xB1u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(4u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(2u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(3u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(4u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(4u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(4u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::ZERO);
        i += 1;
        assert_eq!(cd[i], FieldElement::from(2u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(0xA1u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(0xA2u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(1u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::from(0xF1u64));
        i += 1;
        assert_eq!(cd[i], FieldElement::ZERO);
        assert_eq!(cd.len(), 16);
    }

    #[test]
    fn test_build_gkr_rejects_direct_calldata() {
        let mut result = make_batch_result(1);
        result.verification_calldata = Some(VerificationCalldata::Direct {
            per_inference: vec![vec!["0x1".to_string()]],
        });

        let config = GkrVerificationConfig {
            circuit_depth: 4,
            num_layers: 2,
            matmul_dims: vec![],
            dequantize_bits: vec![],
        };

        let err = build_gkr_verification_calldata(&result, &config);
        assert!(err.is_err());
    }

    #[test]
    fn test_build_direct_verification_calldata() {
        let mut result = make_batch_result(2);
        result.verification_calldata = Some(VerificationCalldata::Direct {
            per_inference: vec![
                vec![
                    format!("{:#066x}", FieldElement::from(0xD1u64)),
                    format!("{:#066x}", FieldElement::from(0xD2u64)),
                ],
                vec![format!("{:#066x}", FieldElement::from(0xE1u64))],
            ],
        });

        let calldata = build_direct_verification_calldata(&result).unwrap();
        assert_eq!(calldata.len(), 2);
        assert_eq!(calldata[0].len(), 2);
        assert_eq!(calldata[0][0], FieldElement::from(0xD1u64));
        assert_eq!(calldata[1].len(), 1);
        assert_eq!(calldata[1][0], FieldElement::from(0xE1u64));
    }

    #[test]
    fn test_build_direct_rejects_gkr_calldata() {
        let mut result = make_batch_result(1);
        result.verification_calldata = Some(VerificationCalldata::Gkr {
            per_inference: vec![],
        });

        let err = build_direct_verification_calldata(&result);
        assert!(err.is_err());
    }

    #[test]
    fn test_serialize_audit_calldata_with_proofs() {
        let mut result = make_batch_result(3);
        result.proof_calldata = (0..50)
            .map(|i| format!("{:#066x}", FieldElement::from(i as u64)))
            .collect();

        let config = SubmitConfig::default();
        let calldata = serialize_audit_calldata(&result, &config).unwrap();

        // 12 header + 50 proof = 62 total
        assert_eq!(calldata.len(), 62);

        let info = validate_calldata(&calldata).unwrap();
        assert_eq!(info.proof_calldata_len, 50);
        assert_eq!(info.inference_count, 3);
        assert_eq!(info.total_felts, 62);
    }
}
