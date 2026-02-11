//! TEE (Trusted Execution Environment) integration for confidential proving.
//!
//! Provides a `SecurityLevel` system that prioritizes TEE when available
//! but gracefully degrades to pure-ZK mode on non-CC hardware (3090, 4090, etc.).
//!
//! # SecurityLevel System
//!
//! ```text
//! SecurityLevel::Auto    → detect CC at runtime, use TEE if available
//! SecurityLevel::ZkPlusTee → require TEE or fail
//! SecurityLevel::ZkOnly  → skip TEE even if available
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  TEE Enclave (NVIDIA CC-On)             │
//! │                                          │
//! │  Encrypted Model Weights                 │
//! │       ↓                                  │
//! │  GPU Proving (GpuBackend)               │
//! │       ↓                                  │
//! │  STARK Proof + Attestation Report       │
//! │                                          │
//! │  Memory: AES-XTS encrypted (HBM)       │
//! └─────────────────────────────────────────┘
//!        ↓
//!   Proof (public) — weights never leave TEE
//! ```
//!
//! The ZK proof is always valid regardless of TEE. TEE adds a *privacy and
//! integrity attestation layer* on top — proving the computation ran on verified
//! hardware with encrypted memory. Without TEE, the math is still provably correct;
//! you just lose the hardware privacy guarantee.

use crate::gpu::{GpuModelProver, GpuError};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::compiler::prove::{ModelError, ModelProofResult};
use crate::compiler::onnx::OnnxModel;
use crate::components::matmul::M31Matrix;

use starknet_crypto::poseidon_hash_many;
use starknet_ff::FieldElement;

use std::process::Command;
use std::time::SystemTime;

// ============================================================================
// SecurityLevel System
// ============================================================================

/// Security level controlling TEE usage in the proving pipeline.
///
/// - `Auto` (default): Detect CC at runtime, use TEE if available, fallback to ZkOnly.
/// - `ZkPlusTee`: Require TEE or fail with `TeeError::NotAvailable`.
/// - `ZkOnly`: Skip TEE even if available — pure STARK proof only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Pure ZK — skip TEE even if available.
    ZkOnly,
    /// Require TEE + ZK — fail if TEE is not available.
    ZkPlusTee,
    /// Auto-detect: use TEE if available, fallback to ZkOnly.
    Auto,
}

impl SecurityLevel {
    /// Resolve `Auto` to a concrete level based on detected hardware.
    pub fn resolve(&self) -> ResolvedSecurityLevel {
        match self {
            SecurityLevel::ZkOnly => ResolvedSecurityLevel::ZkOnly,
            SecurityLevel::ZkPlusTee => ResolvedSecurityLevel::ZkPlusTee,
            SecurityLevel::Auto => {
                let cap = detect_tee_capability();
                if cap.cc_active && cap.nvattest_available {
                    ResolvedSecurityLevel::ZkPlusTee
                } else {
                    ResolvedSecurityLevel::ZkOnly
                }
            }
        }
    }
}

impl Default for SecurityLevel {
    fn default() -> Self {
        SecurityLevel::Auto
    }
}

impl std::str::FromStr for SecurityLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(SecurityLevel::Auto),
            "tee" | "zk-plus-tee" | "zk+tee" => Ok(SecurityLevel::ZkPlusTee),
            "zk-only" | "zk" | "none" => Ok(SecurityLevel::ZkOnly),
            _ => Err(format!(
                "unknown security level '{s}', expected 'auto', 'tee', or 'zk-only'"
            )),
        }
    }
}

impl std::fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityLevel::ZkOnly => write!(f, "zk-only"),
            SecurityLevel::ZkPlusTee => write!(f, "zk+tee"),
            SecurityLevel::Auto => write!(f, "auto"),
        }
    }
}

/// Resolved security level (no more `Auto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedSecurityLevel {
    ZkOnly,
    ZkPlusTee,
}

// ============================================================================
// TEE Capability Detection
// ============================================================================

/// Detected TEE capability of the current hardware.
#[derive(Debug, Clone)]
pub struct TeeCapability {
    /// Whether GPU hardware supports CC (Hopper+ architecture, compute >= 9).
    pub cc_supported: bool,
    /// Whether CC mode is currently active on the GPU.
    pub cc_active: bool,
    /// Whether the `nvattest` CLI tool is available on the system.
    pub nvattest_available: bool,
    /// GPU device name (e.g., "NVIDIA H200").
    pub device_name: String,
    /// Human-readable status message.
    pub status_message: String,
}

impl Default for TeeCapability {
    fn default() -> Self {
        Self {
            cc_supported: false,
            cc_active: false,
            nvattest_available: false,
            device_name: String::new(),
            status_message: "No TEE capability detected".to_string(),
        }
    }
}

impl std::fmt::Display for TeeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.cc_active && self.nvattest_available {
            write!(f, "TEE Active ({}, CC-On, attested)", self.device_name)
        } else if self.cc_active {
            write!(f, "TEE Active ({}, CC-On, nvattest missing)", self.device_name)
        } else if self.cc_supported {
            write!(f, "TEE Available ({}, CC-Off)", self.device_name)
        } else if !self.device_name.is_empty() {
            write!(f, "No TEE ({}, pre-Hopper)", self.device_name)
        } else {
            write!(f, "No TEE (no GPU)")
        }
    }
}

/// Detect TEE capability of the current hardware.
///
/// Queries the low-level `tee_status()` from the stwo GPU backend and
/// checks for `nvattest` CLI availability.
pub fn detect_tee_capability() -> TeeCapability {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::tee;
        let info = tee::detect_tee_capability();

        let mut cap = TeeCapability {
            cc_supported: info.cc_supported,
            cc_active: info.cc_active,
            nvattest_available: info.nvattest_available,
            device_name: info.device_name,
            status_message: String::new(),
        };

        cap.status_message = if cap.cc_active && cap.nvattest_available {
            format!("CC-On active on {}, nvattest available", cap.device_name)
        } else if cap.cc_active {
            format!("CC-On active on {}, nvattest NOT found", cap.device_name)
        } else if cap.cc_supported {
            format!("{} supports CC but mode is Off — enable via: nvidia-smi conf-compute -scc on", cap.device_name)
        } else if !cap.device_name.is_empty() {
            format!("{} does not support Confidential Computing (need H100/H200/B200)", cap.device_name)
        } else {
            "No GPU detected".to_string()
        };

        return cap;
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        TeeCapability::default()
    }
}

// ============================================================================
// TEE Attestation
// ============================================================================

/// TEE attestation report.
#[derive(Debug, Clone)]
pub struct TeeAttestation {
    /// GPU attestation report bytes (NVIDIA DCAP / nvattest JWT format).
    pub report: Vec<u8>,
    /// Code measurement hash (SHA-256 of proving binary).
    pub measurement: [u8; 32],
    /// Timestamp of attestation (Unix epoch seconds).
    pub timestamp: u64,
    /// GPU device ID.
    pub device_id: String,
    /// Hardware model from attestation (e.g., "H200").
    pub hw_model: String,
    /// Whether secure boot was verified.
    pub secure_boot: bool,
    /// Debug status from attestation ("disabled-since-boot" for production).
    pub debug_status: String,
}

impl TeeAttestation {
    /// Compute Poseidon hash of the attestation report as a felt252.
    ///
    /// Uses `starknet_crypto::poseidon_hash_many` — the same hash function
    /// used in the ObelyskVerifier contract. This ensures the hash computed
    /// off-chain matches what the contract expects for `tee_attestation_hash`.
    ///
    /// Report bytes are chunked into 31-byte segments (each fits in a felt252),
    /// then Poseidon-hashed. Returns `FieldElement::ZERO` for empty reports
    /// (indicating no TEE attestation).
    pub fn report_hash_felt(&self) -> FieldElement {
        if self.report.is_empty() {
            return FieldElement::ZERO;
        }

        // Chunk report bytes into felt252-sized pieces (31 bytes each, < 2^248).
        let mut felts = Vec::new();
        for chunk in self.report.chunks(31) {
            let mut bytes = [0u8; 32];
            bytes[32 - chunk.len()..].copy_from_slice(chunk);
            felts.push(FieldElement::from_bytes_be(&bytes).unwrap_or(FieldElement::ZERO));
        }
        poseidon_hash_many(&felts)
    }

    /// Compute hash of the attestation report as raw bytes.
    ///
    /// Returns the big-endian bytes of `report_hash_felt()`.
    /// For on-chain use, prefer `report_hash_felt()` directly.
    pub fn report_hash(&self) -> [u8; 32] {
        self.report_hash_felt().to_bytes_be()
    }

    /// Returns true if this attestation has a real TEE report.
    pub fn has_report(&self) -> bool {
        !self.report.is_empty()
    }
}

/// Parsed result from `nvattest` CLI tool.
#[derive(Debug, Clone)]
struct NvatResult {
    /// Raw JWT/JSON attestation report.
    report_bytes: Vec<u8>,
    /// Hardware model claim from EAT token.
    hw_model: String,
    /// Secure boot status.
    secure_boot: bool,
    /// Debug status (production = "disabled-since-boot").
    debug_status: String,
    /// Measurement result digest.
    measurement: [u8; 32],
}

/// TEE-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum TeeError {
    #[error("TEE not available: {0}")]
    NotAvailable(String),
    #[error("Attestation failed: {0}")]
    AttestationFailed(String),
    #[error("nvattest CLI not found — install NVIDIA Attestation SDK")]
    NvattestNotFound,
    #[error("nvattest returned invalid output: {0}")]
    NvattestParseError(String),
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
    #[error("Proving error: {0}")]
    ProvingError(#[from] ModelError),
}

// ============================================================================
// Attestation Generation
// ============================================================================

/// Generate a TEE attestation report.
///
/// Behavior depends on the `tee` feature flag and hardware availability:
///
/// - `tee` feature + CC-On + nvattest: Calls `nvattest attest` subprocess,
///   parses JWT/JSON response, returns real `TeeAttestation`.
/// - `tee` feature + no CC: Returns `TeeError::NotAvailable`.
/// - No `tee` feature: Returns empty attestation (honest about no TEE).
fn generate_attestation(prover: &GpuModelProver) -> Result<TeeAttestation, TeeError> {
    #[cfg(feature = "tee")]
    {
        let cap = detect_tee_capability();

        if !cap.cc_active {
            return Err(TeeError::NotAvailable(format!(
                "TEE requires NVIDIA H100/H200/B200 with CC-On firmware. {}",
                cap.status_message
            )));
        }

        if !cap.nvattest_available {
            return Err(TeeError::NvattestNotFound);
        }

        // Real attestation via nvattest CLI
        let nv_result = call_nvattest()?;

        Ok(TeeAttestation {
            report: nv_result.report_bytes,
            measurement: nv_result.measurement,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            device_id: prover.device_name.clone(),
            hw_model: nv_result.hw_model,
            secure_boot: nv_result.secure_boot,
            debug_status: nv_result.debug_status,
        })
    }

    #[cfg(not(feature = "tee"))]
    {
        Ok(TeeAttestation {
            report: Vec::new(), // Empty = no TEE
            measurement: [0u8; 32],
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            device_id: prover.device_name.clone(),
            hw_model: String::new(),
            secure_boot: false,
            debug_status: "not-applicable".to_string(),
        })
    }
}

/// Generate attestation with SecurityLevel control.
///
/// - `ZkOnly`: Always returns empty attestation (no TEE).
/// - `ZkPlusTee`: Requires real attestation or fails.
/// - `Auto`: Tries real attestation, falls back to empty on non-CC hardware.
fn generate_attestation_with_level(
    prover: &GpuModelProver,
    level: SecurityLevel,
) -> Result<TeeAttestation, TeeError> {
    match level.resolve() {
        ResolvedSecurityLevel::ZkOnly => {
            Ok(TeeAttestation {
                report: Vec::new(),
                measurement: [0u8; 32],
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                device_id: prover.device_name.clone(),
                hw_model: String::new(),
                secure_boot: false,
                debug_status: "zk-only".to_string(),
            })
        }
        ResolvedSecurityLevel::ZkPlusTee => {
            generate_attestation(prover)
        }
    }
}

/// Call the `nvattest` CLI tool and parse the attestation response.
///
/// Runs: `nvattest attest --device gpu --verifier local`
/// Parses the JSON output containing JWT with EAT (Entity Attestation Token) claims.
#[cfg(feature = "tee")]
fn call_nvattest() -> Result<NvatResult, TeeError> {
    let output = Command::new("nvattest")
        .args(["attest", "--device", "gpu", "--verifier", "local"])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                TeeError::NvattestNotFound
            } else {
                TeeError::AttestationFailed(format!("nvattest execution failed: {e}"))
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TeeError::AttestationFailed(format!(
            "nvattest exited with status {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_nvattest_output(&stdout)
}

/// Parse nvattest JSON/JWT output into `NvatResult`.
///
/// The nvattest tool outputs JSON containing:
/// - `attestation_report`: Base64-encoded JWT with EAT claims
/// - `hwmodel`: GPU hardware model
/// - `secboot`: Secure boot status
/// - `dbgstat`: Debug status
/// - `measres`: Measurement result
#[cfg(feature = "tee")]
fn parse_nvattest_output(output: &str) -> Result<NvatResult, TeeError> {
    // nvattest output can be JSON or newline-delimited key-value.
    // Try JSON parse first, then fall back to line parsing.
    let report_bytes = output.trim().as_bytes().to_vec();

    let mut hw_model = String::new();
    let mut secure_boot = false;
    let mut debug_status = String::new();
    let mut measurement = [0u8; 32];

    // Parse key fields from the output
    for line in output.lines() {
        let line = line.trim();

        if line.contains("hwmodel") || line.contains("\"hwmodel\"") {
            if let Some(val) = extract_json_string_value(line, "hwmodel") {
                hw_model = val;
            }
        }
        if line.contains("secboot") || line.contains("\"secboot\"") {
            secure_boot = line.contains("true") || line.contains("enabled");
        }
        if line.contains("dbgstat") || line.contains("\"dbgstat\"") {
            if let Some(val) = extract_json_string_value(line, "dbgstat") {
                debug_status = val;
            }
        }
        if line.contains("measres") || line.contains("\"measres\"") {
            if let Some(val) = extract_json_string_value(line, "measres") {
                // Parse hex measurement into bytes
                let hex = val.trim_start_matches("0x");
                for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
                    if i >= 32 { break; }
                    if let Ok(byte) = u8::from_str_radix(
                        std::str::from_utf8(chunk).unwrap_or("00"),
                        16,
                    ) {
                        measurement[i] = byte;
                    }
                }
            }
        }
    }

    // If we couldn't parse structured fields, still return the raw report
    if hw_model.is_empty() {
        hw_model = "unknown".to_string();
    }
    if debug_status.is_empty() {
        debug_status = "unknown".to_string();
    }

    Ok(NvatResult {
        report_bytes,
        hw_model,
        secure_boot,
        debug_status,
        measurement,
    })
}

/// Extract a JSON string value for a given key from a line.
#[cfg(feature = "tee")]
fn extract_json_string_value(line: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    if let Some(idx) = line.find(&pattern) {
        let after_key = &line[idx + pattern.len()..];
        // Skip ": " or ":"
        let after_colon = after_key.trim_start().strip_prefix(':')?;
        let trimmed = after_colon.trim();
        // Extract quoted string value
        if let Some(start) = trimmed.find('"') {
            let rest = &trimmed[start + 1..];
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
        // Non-quoted value (boolean, number)
        let val = trimmed.trim_end_matches([',', '}', ']']);
        if !val.is_empty() {
            return Some(val.to_string());
        }
    }
    None
}

// ============================================================================
// TEE Model Prover
// ============================================================================

/// TEE-enabled model prover.
///
/// Wraps `GpuModelProver` with attestation. Without the `tee` feature,
/// the prover works but produces empty attestation reports (honest about
/// no TEE). With the `tee` feature enabled, creation fails unless running
/// on real NVIDIA CC-On hardware (H100/H200/B200).
#[derive(Debug)]
pub struct TeeModelProver {
    pub attestation: TeeAttestation,
    pub security_level: SecurityLevel,
    gpu_prover: GpuModelProver,
}

impl TeeModelProver {
    /// Create a TEE prover with auto-detected security level.
    pub fn new() -> Result<Self, TeeError> {
        Self::with_security(SecurityLevel::Auto)
    }

    /// Create a TEE prover with explicit security level.
    pub fn with_security(level: SecurityLevel) -> Result<Self, TeeError> {
        let gpu_prover = GpuModelProver::new()?;
        let attestation = generate_attestation_with_level(&gpu_prover, level)?;

        Ok(Self {
            attestation,
            security_level: level,
            gpu_prover,
        })
    }

    /// Check if running inside a real TEE.
    pub fn is_tee(&self) -> bool {
        !self.attestation.report.is_empty()
    }

    /// Get the resolved security level.
    pub fn resolved_security(&self) -> ResolvedSecurityLevel {
        self.security_level.resolve()
    }

    /// Prove a model with attestation.
    ///
    /// Returns the proof along with a fresh attestation report
    /// binding the proof to this specific TEE instance.
    pub fn prove_with_attestation(
        &self,
        model: &OnnxModel,
        input: &M31Matrix,
    ) -> Result<(ModelProofResult, TeeAttestation), TeeError> {
        let result = self.gpu_prover.prove_model(
            &model.graph,
            input,
            &model.weights,
        )?;

        // Generate fresh attestation for this proof
        let attestation = generate_attestation_with_level(
            &self.gpu_prover,
            self.security_level,
        )?;

        Ok((result, attestation))
    }

    /// Prove a computation graph with attestation.
    pub fn prove_graph(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<(ModelProofResult, TeeAttestation), TeeError> {
        let result = self.gpu_prover.prove_model(graph, input, weights)?;
        let attestation = generate_attestation_with_level(
            &self.gpu_prover,
            self.security_level,
        )?;
        Ok((result, attestation))
    }
}

// ============================================================================
// Verification
// ============================================================================

/// Verify a TEE attestation report.
///
/// Checks: non-empty report, measurement hash, and freshness.
/// Full DCAP signature chain verification requires the nvidia-dcap crate.
pub fn verify_attestation(
    attestation: &TeeAttestation,
    expected_measurement: &[u8; 32],
    max_age_secs: u64,
) -> bool {
    // A valid attestation requires a non-empty TEE report.
    // Empty reports indicate non-TEE mode and cannot pass verification.
    if attestation.report.is_empty() {
        return false;
    }

    // Check measurement matches expected binary hash
    if attestation.measurement != *expected_measurement {
        return false;
    }

    // Check freshness
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if now.saturating_sub(attestation.timestamp) > max_age_secs {
        return false;
    }

    // Structural checks pass (non-empty report, correct measurement, fresh).
    // Full DCAP signature chain verification requires the nvidia-dcap crate.
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_level_parsing() {
        assert_eq!("auto".parse::<SecurityLevel>().unwrap(), SecurityLevel::Auto);
        assert_eq!("tee".parse::<SecurityLevel>().unwrap(), SecurityLevel::ZkPlusTee);
        assert_eq!("zk-only".parse::<SecurityLevel>().unwrap(), SecurityLevel::ZkOnly);
        assert_eq!("zk".parse::<SecurityLevel>().unwrap(), SecurityLevel::ZkOnly);
        assert_eq!("zk+tee".parse::<SecurityLevel>().unwrap(), SecurityLevel::ZkPlusTee);
        assert!("invalid".parse::<SecurityLevel>().is_err());
    }

    #[test]
    fn test_security_level_display() {
        assert_eq!(SecurityLevel::ZkOnly.to_string(), "zk-only");
        assert_eq!(SecurityLevel::ZkPlusTee.to_string(), "zk+tee");
        assert_eq!(SecurityLevel::Auto.to_string(), "auto");
    }

    #[test]
    fn test_security_level_default() {
        assert_eq!(SecurityLevel::default(), SecurityLevel::Auto);
    }

    #[test]
    fn test_tee_capability_detection() {
        let cap = detect_tee_capability();
        // On dev machines: cc_supported=false, cc_active=false
        // On H200 with CC-On: cc_supported=true, cc_active=true
        assert!(!cap.status_message.is_empty());
    }

    #[test]
    fn test_tee_capability_display() {
        let cap = TeeCapability::default();
        let s = format!("{}", cap);
        assert!(s.contains("No TEE"));
    }

    #[test]
    fn test_tee_prover_creation() {
        let result = TeeModelProver::new();
        // Without TEE feature: creates prover with empty attestation.
        // With TEE feature: fails without CC-On hardware.
        #[cfg(not(feature = "tee"))]
        assert!(result.is_ok());
        #[cfg(feature = "tee")]
        let _ = result; // Err on dev machines, Ok on CC-On hardware
    }

    #[test]
    fn test_attestation_non_tee() {
        #[cfg(not(feature = "tee"))]
        {
            let prover = TeeModelProver::new().unwrap();
            assert!(!prover.is_tee());
            assert_eq!(prover.attestation.debug_status, "zk-only");
        }
    }

    #[test]
    fn test_verify_attestation() {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Non-empty report + matching measurement + fresh → pass
        let valid = TeeAttestation {
            report: vec![1, 2, 3, 4],
            measurement: [0u8; 32],
            timestamp: now,
            device_id: "test".to_string(),
            hw_model: "H200".to_string(),
            secure_boot: true,
            debug_status: "disabled-since-boot".to_string(),
        };
        assert!(verify_attestation(&valid, &[0u8; 32], 3600));

        // Wrong measurement → fail
        assert!(!verify_attestation(&valid, &[1u8; 32], 3600));

        // Empty report → fail (no TEE)
        let no_tee = TeeAttestation {
            report: Vec::new(),
            measurement: [0u8; 32],
            timestamp: now,
            device_id: "test".to_string(),
            hw_model: String::new(),
            secure_boot: false,
            debug_status: "not-applicable".to_string(),
        };
        assert!(!verify_attestation(&no_tee, &[0u8; 32], 3600));
    }

    #[test]
    fn test_attestation_expired() {
        let attestation = TeeAttestation {
            report: vec![1, 2, 3, 4], // Non-empty to test expiry path
            measurement: [0u8; 32],
            timestamp: 0, // Epoch = very old
            device_id: "test".to_string(),
            hw_model: "H200".to_string(),
            secure_boot: true,
            debug_status: "disabled-since-boot".to_string(),
        };

        // Max age of 1 hour should fail for timestamp 0
        assert!(!verify_attestation(&attestation, &[0u8; 32], 3600));
    }

    #[test]
    fn test_report_hash() {
        use starknet_ff::FieldElement;

        let att = TeeAttestation {
            report: vec![1, 2, 3, 4],
            measurement: [0u8; 32],
            timestamp: 100,
            device_id: "test".to_string(),
            hw_model: "H200".to_string(),
            secure_boot: true,
            debug_status: "disabled-since-boot".to_string(),
        };

        // Poseidon hash of non-empty report → non-zero
        let hash_felt = att.report_hash_felt();
        assert_ne!(hash_felt, FieldElement::ZERO);

        // Deterministic
        assert_eq!(hash_felt, att.report_hash_felt());

        // report_hash() returns bytes of the felt
        let hash_bytes = att.report_hash();
        assert_eq!(hash_bytes, hash_felt.to_bytes_be());

        // has_report() should be true
        assert!(att.has_report());

        // Empty report → zero hash, no report
        let empty = TeeAttestation {
            report: Vec::new(),
            measurement: [0u8; 32],
            timestamp: 0,
            device_id: "test".to_string(),
            hw_model: String::new(),
            secure_boot: false,
            debug_status: "not-applicable".to_string(),
        };
        assert_eq!(empty.report_hash_felt(), FieldElement::ZERO);
        assert_eq!(empty.report_hash(), [0u8; 32]);
        assert!(!empty.has_report());
    }

    #[test]
    fn test_report_hash_different_reports() {
        let att1 = TeeAttestation {
            report: vec![1, 2, 3, 4],
            measurement: [0u8; 32],
            timestamp: 100,
            device_id: "test".to_string(),
            hw_model: "H200".to_string(),
            secure_boot: true,
            debug_status: "disabled-since-boot".to_string(),
        };
        let att2 = TeeAttestation {
            report: vec![5, 6, 7, 8],
            measurement: [0u8; 32],
            timestamp: 100,
            device_id: "test".to_string(),
            hw_model: "H200".to_string(),
            secure_boot: true,
            debug_status: "disabled-since-boot".to_string(),
        };
        // Different reports → different hashes
        assert_ne!(att1.report_hash_felt(), att2.report_hash_felt());
    }

    #[test]
    fn test_with_security_zk_only() {
        let result = TeeModelProver::with_security(SecurityLevel::ZkOnly);
        assert!(result.is_ok());
        let prover = result.unwrap();
        assert!(!prover.is_tee());
        assert_eq!(prover.security_level, SecurityLevel::ZkOnly);
    }

    #[cfg(feature = "tee")]
    #[test]
    fn test_tee_prove_matches_gpu() {
        let prover = TeeModelProver::new().unwrap();
        let model = crate::compiler::onnx::build_mlp(
            16, &[8], 4, crate::components::activation::ActivationType::ReLU,
        );
        let input = M31Matrix::new(1, 16);

        let result = prover.prove_with_attestation(&model, &input);
        assert!(result.is_ok());
    }
}
