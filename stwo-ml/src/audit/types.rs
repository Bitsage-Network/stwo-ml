//! Shared types for the audit system.
//!
//! Dev A and Dev B code against these types independently.
//! Integration happens when both sides implement their producers/consumers.

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use crate::audit::digest::M31Digest;

// ─── Stage 1: Inference Logging (Dev A produces, Dev B consumes) ────────────

/// A single inference record in the append-only log.
///
/// Contains everything needed to replay the inference during audit proving:
/// the tokenized input/output, the model identifier, and all commitments
/// that will be verified in the aggregated proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceLogEntry {
    // === Identity ===
    /// Unique inference ID (monotonically increasing within session).
    pub inference_id: u64,
    /// Position in the log (0-indexed, equals inference_id for now).
    pub sequence_number: u64,

    // === Model ===
    /// Model identifier (hex-encoded felt252, matches on-chain model_id).
    pub model_id: String,
    /// Poseidon hash of all weight matrices (hex-encoded felt252).
    pub weight_commitment: String,
    /// Human-readable model name.
    pub model_name: String,
    /// Number of layers in the model.
    pub num_layers: u32,

    // === Inference Data ===
    /// Tokenized input (raw token IDs before M31 quantization).
    pub input_tokens: Vec<u32>,
    /// Tokenized output (raw token IDs).
    pub output_tokens: Vec<u32>,
    /// Byte offset into the binary sidecar file where this entry's
    /// M31 input matrix starts. The prover reads from here during replay.
    pub matrix_offset: u64,
    /// Size in bytes of the M31 input matrix in the sidecar.
    pub matrix_size: u64,
    /// Rows and columns of the M31 input matrix.
    pub input_rows: u32,
    pub input_cols: u32,
    /// Rows and columns of the M31 output matrix.
    pub output_rows: u32,
    pub output_cols: u32,

    // === Commitments (hex-encoded felt252) ===
    /// Poseidon(input || output) — binds proof to this specific I/O.
    pub io_commitment: String,
    /// Running Poseidon hash through all layer intermediates.
    pub layer_chain_commitment: String,

    // === Chain Link ===
    /// Poseidon hash of the previous log entry ("0x0" for first).
    pub prev_entry_hash: String,
    /// Poseidon hash of THIS entry (computed on append).
    pub entry_hash: String,

    // === Metadata ===
    /// Inference start timestamp (Unix epoch nanoseconds).
    pub timestamp_ns: u64,
    /// Inference latency in milliseconds.
    pub latency_ms: u64,
    /// GPU device used.
    pub gpu_device: String,
    /// TEE attestation hash ("0x0" if no TEE).
    pub tee_report_hash: String,
    /// Task category — for report aggregation.
    pub task_category: Option<String>,
    /// First 100 chars of detokenized input (for report preview).
    pub input_preview: Option<String>,
    /// First 200 chars of detokenized output (for report preview).
    pub output_preview: Option<String>,
}

/// Query result from the inference log for a time window.
#[derive(Debug)]
pub struct LogWindow {
    /// Entries in the window, ordered by sequence number.
    pub entries: Vec<InferenceLogEntry>,
    /// Merkle root of all entry hashes in the window (M31 Poseidon2 digest).
    pub merkle_root: M31Digest,
    /// Window start (Unix epoch nanoseconds).
    pub start_ns: u64,
    /// Window end (Unix epoch nanoseconds).
    pub end_ns: u64,
    /// Whether chain integrity was verified for these entries.
    pub chain_verified: bool,
}

// ─── Stage 2: Audit Proving (Dev A produces, Dev B consumes) ────────────────

/// Request for an audit over a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequest {
    /// Start of audit window (Unix epoch nanoseconds).
    pub start_ns: u64,
    /// End of audit window (Unix epoch nanoseconds).
    pub end_ns: u64,
    /// Model ID (hex-encoded felt252, must match log entries).
    pub model_id: String,
    /// Proving mode: "gkr" (default, fastest), "direct", "recursive".
    pub mode: String,
    /// Whether to run semantic evaluation (Stage 3).
    pub evaluate_semantics: bool,
    /// Maximum number of inferences to prove (0 = all in window).
    pub max_inferences: usize,
    /// GPU device index for proving (None = default device).
    pub gpu_device: Option<usize>,
}

impl Default for AuditRequest {
    fn default() -> Self {
        Self {
            start_ns: 0,
            end_ns: u64::MAX,
            model_id: String::new(),
            mode: "gkr".to_string(),
            evaluate_semantics: true,
            max_inferences: 0,
            gpu_device: None,
        }
    }
}

/// Which proving pipeline to use for on-chain verification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProofMode {
    /// `verify_model_gkr()` — Poseidon Fiat-Shamir, 100% on-chain via EloVerifier.
    Gkr,
    /// `verify_model_direct()` — unified STARK + Poseidon sumchecks.
    Direct,
    /// `prove_model()` — Blake2s, NOT on-chain verifiable (backward compat).
    Legacy,
}

impl Default for ProofMode {
    fn default() -> Self {
        ProofMode::Legacy
    }
}

impl ProofMode {
    /// Parse a mode string into a `ProofMode`.
    pub fn from_str(mode: &str) -> Self {
        match mode {
            "gkr" => ProofMode::Gkr,
            "direct" => ProofMode::Direct,
            _ => ProofMode::Legacy,
        }
    }
}

/// Result of proving a single inference within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceProofResult {
    /// Sequence number in the log.
    pub sequence: u64,
    /// IO commitment for this inference (hex felt252).
    pub io_commitment: String,
    /// Layer chain commitment for this inference (hex felt252).
    pub layer_chain_commitment: String,
    /// Timestamp of the original inference (nanoseconds).
    pub timestamp_ns: u64,
    /// Size of the proof calldata in felt252 elements.
    pub proof_size_felts: usize,
    /// Time to prove this inference in milliseconds.
    pub proving_time_ms: u64,

    // ── Proof calldata (populated by GKR/Direct pipelines) ──────────

    /// Serialized GKR or combined proof calldata (hex felt252 strings).
    /// Empty for Legacy mode.
    #[serde(default)]
    pub proof_calldata: Vec<String>,
    /// Raw IO calldata for GKR mode (hex felt252 strings).
    /// Empty for Direct/Legacy modes.
    #[serde(default)]
    pub io_calldata: Vec<String>,
    /// Weight opening calldata for GKR mode (hex felt252 strings).
    /// Empty for Direct/Legacy modes.
    #[serde(default)]
    pub weight_opening_calldata: Vec<String>,
    /// Weight commitments for GKR mode (hex felt252 strings).
    /// Empty for Direct/Legacy modes.
    #[serde(default)]
    pub weight_commitments_calldata: Vec<String>,
    /// Which proving pipeline produced this proof.
    #[serde(default)]
    pub proof_mode: ProofMode,
}

/// Per-inference GKR calldata for on-chain verification via `verify_model_gkr()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GkrInferenceCalldata {
    /// Model ID (hex felt252).
    pub model_id: String,
    /// Serialized GKR proof data (hex felt252 strings).
    pub gkr_calldata: Vec<String>,
    /// Raw IO data (hex felt252 strings).
    pub io_calldata: Vec<String>,
    /// Weight commitments (hex felt252 strings).
    #[serde(default)]
    pub weight_commitments: Vec<String>,
    /// Weight opening proofs (hex felt252 strings).
    /// Serialized as Cairo `Array<MleOpeningProof>`:
    /// `[num_openings, opening_0..., opening_1..., ...]`.
    pub weight_opening_calldata: Vec<String>,
}

/// Structured verification calldata for on-chain submission.
///
/// Assembled from per-inference proof results and ready for
/// `build_gkr_verification_calldata()` or `build_direct_verification_calldata()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationCalldata {
    /// GKR mode: per-inference calldata for `verify_model_gkr()`.
    Gkr {
        per_inference: Vec<GkrInferenceCalldata>,
    },
    /// Direct mode: per-inference combined calldata for `verify_model_direct()`.
    Direct {
        per_inference: Vec<Vec<String>>,
    },
}

/// Result of batch audit proving over a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAuditResult {
    // === Window ===
    /// Audit window start (Unix epoch seconds).
    pub time_start: u64,
    /// Audit window end (Unix epoch seconds).
    pub time_end: u64,
    /// Number of inferences proven.
    pub inference_count: u32,

    // === Aggregated Commitments (hex felt252) ===
    /// Merkle root of all io_commitments in the batch.
    pub io_merkle_root: String,
    /// Merkle root of the inference log entries in the window.
    pub log_merkle_root: String,
    /// Weight commitment (constant across all inferences).
    pub weight_commitment: String,
    /// Combined chain commitment (Poseidon of all per-inference chain commits).
    pub combined_chain_commitment: String,

    // === Per-Inference ===
    /// Individual proof results.
    pub inference_results: Vec<InferenceProofResult>,

    // === Proof ===
    /// Model identifier.
    pub model_id: String,
    /// Total proving time in milliseconds.
    pub proving_time_ms: u64,
    /// Aggregated proof calldata for on-chain submission (hex felt252 array).
    /// Format: `[count, len_0, proof_0..., len_1, proof_1..., ...]`
    pub proof_calldata: Vec<String>,

    // === Verification ===
    /// Structured verification calldata for on-chain submission.
    /// None for Legacy mode (proofs are not on-chain verifiable).
    #[serde(default)]
    pub verification_calldata: Option<VerificationCalldata>,

    // === TEE ===
    /// TEE attestation hash (None if no TEE).
    pub tee_attestation_hash: Option<String>,
}

// ─── Stage 3: Semantic Evaluation (Dev B produces) ──────────────────────────

/// Result of a single deterministic check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicCheck {
    /// Check type: "json_valid", "math_correct", "code_syntax", "sql_parse", etc.
    pub check_type: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Error detail if failed, None if passed.
    pub detail: Option<String>,
}

/// Evaluation of a single inference (deterministic + semantic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEvaluation {
    /// Sequence number in the log (joins with InferenceLogEntry).
    pub sequence: u64,
    /// Deterministic check results.
    pub deterministic_checks: Vec<DeterministicCheck>,
    /// Semantic quality score (0.0 - 1.0), None if not evaluated.
    pub semantic_score: Option<f32>,
    /// IO commitment of the self-evaluation forward pass (hex felt252).
    pub eval_io_commitment: Option<String>,
    /// Whether the evaluation forward pass was also proved.
    pub evaluation_proved: bool,
}

/// Aggregated semantic evaluation for an audit window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSemanticSummary {
    /// Evaluation method: "deterministic", "self_eval", "cross_model", "combined".
    pub method: String,
    /// Average quality score across all evaluated inferences.
    pub avg_quality_score: f32,
    /// Score distribution.
    pub excellent_count: u32, // 0.9 - 1.0
    pub good_count: u32,      // 0.7 - 0.9
    pub fair_count: u32,      // 0.5 - 0.7
    pub poor_count: u32,      // 0.0 - 0.5
    /// Deterministic check totals.
    pub deterministic_pass: u32,
    pub deterministic_fail: u32,
    /// How many inferences were evaluated.
    pub evaluated_count: u32,
    /// Whether evaluation forward passes were also proved.
    pub evaluations_proved: bool,
    /// Merkle root of eval_io_commitments (hex felt252).
    pub eval_merkle_root: Option<String>,
    /// Per-inference evaluations.
    pub per_inference: Vec<InferenceEvaluation>,
}

// ─── Stage 4: Report Format (Dev B produces) ────────────────────────────────

/// Complete audit report — the off-chain document bound to on-chain AuditRecord.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Schema version.
    pub version: String,
    /// Unique audit identifier (hex felt252, matches on-chain audit_id).
    pub audit_id: String,
    /// Time window covered by this audit.
    pub time_window: TimeWindow,
    /// Model information.
    pub model: ModelInfo,
    /// Infrastructure (GPU, OS, prover version).
    pub infrastructure: InfrastructureInfo,
    /// Inference statistics.
    pub inference_summary: InferenceSummary,
    /// Semantic evaluation results (None if not requested).
    pub semantic_evaluation: Option<AuditSemanticSummary>,
    /// All cryptographic commitments.
    pub commitments: AuditCommitments,
    /// Proof and on-chain submission info.
    pub proof: ProofInfo,
    /// Privacy and storage info (None for public audits).
    pub privacy: Option<PrivacyInfo>,
    /// Per-inference entries.
    pub inferences: Vec<InferenceEntry>,
    /// Billing information (None if not applicable).
    pub billing: Option<BillingInfo>,
    /// Report generation metadata.
    pub metadata: ReportMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// ISO 8601 start time.
    pub start: String,
    /// ISO 8601 end time.
    pub end: String,
    /// Start as Unix epoch nanoseconds.
    pub start_epoch_ns: u64,
    /// End as Unix epoch nanoseconds.
    pub end_epoch_ns: u64,
    /// Duration in seconds.
    pub duration_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// On-chain model ID (hex felt252).
    pub model_id: String,
    /// Human-readable name.
    pub name: String,
    /// Architecture: "transformer", "cnn", "mlp".
    pub architecture: String,
    /// Parameter count as string ("14.7B").
    pub parameters: String,
    /// Number of layers.
    pub layers: u32,
    /// Poseidon hash of weights (hex felt252).
    pub weight_commitment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureInfo {
    /// GPU device name ("NVIDIA H100 80GB").
    pub gpu_device: String,
    /// Number of GPUs used.
    pub gpu_count: u32,
    /// CUDA version.
    pub cuda_version: String,
    /// Prover software version.
    pub prover_version: String,
    /// Whether TEE was active.
    pub tee_active: bool,
    /// TEE attestation hash (hex felt252).
    pub tee_attestation_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSummary {
    /// Total inferences in the audit window.
    pub total_inferences: u32,
    /// Total input tokens across all inferences.
    pub total_input_tokens: u64,
    /// Total output tokens.
    pub total_output_tokens: u64,
    /// Average inference latency in milliseconds.
    pub avg_latency_ms: u64,
    /// 95th percentile latency.
    pub p95_latency_ms: u64,
    /// Token throughput.
    pub throughput_tokens_per_sec: f32,
    /// Task category distribution.
    pub categories: HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditCommitments {
    /// Merkle root of inference log entries (hex felt252).
    pub inference_log_merkle_root: String,
    /// Merkle root of io_commitments (hex felt252).
    pub io_merkle_root: String,
    /// Model weight commitment (hex felt252).
    pub weight_commitment: String,
    /// Combined layer chain commitment (hex felt252).
    pub combined_chain_commitment: String,
    /// Poseidon hash of this report (hex felt252).
    pub audit_report_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofInfo {
    /// Proving mode: "gkr", "direct", "recursive".
    pub mode: String,
    /// Total proving time in seconds.
    pub proving_time_seconds: u64,
    /// Total proof size in bytes.
    pub proof_size_bytes: usize,
    /// On-chain transaction hash (hex, None if not submitted).
    pub on_chain_tx: Option<String>,
    /// Whether on-chain verification passed.
    pub on_chain_verified: Option<bool>,
    /// Arweave transaction ID for the encrypted report.
    pub arweave_tx_id: Option<String>,
    /// On-chain audit record ID (hex felt252).
    pub audit_record_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyInfo {
    /// Privacy tier: "public", "private", "selective".
    pub tier: String,
    /// Encryption scheme: "poseidon_m31" or "aes_256_gcm".
    pub encryption_scheme: String,
    /// Arweave transaction ID of the encrypted blob.
    pub arweave_tx_id: Option<String>,
}

/// Per-inference entry in the report (summary, not full data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEntry {
    /// Index within this report (0-based).
    pub index: u32,
    /// Sequence number in the log.
    pub sequence: u64,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// IO commitment (hex felt252).
    pub io_commitment: String,
    /// Number of input tokens.
    pub input_tokens: u32,
    /// Number of output tokens.
    pub output_tokens: u32,
    /// Latency in milliseconds.
    pub latency_ms: u64,
    /// Task category.
    pub category: Option<String>,
    /// Semantic quality score (0.0-1.0).
    pub semantic_score: Option<f32>,
    /// Truncated input text (first 100 chars).
    pub input_preview: Option<String>,
    /// Truncated output text (first 200 chars).
    pub output_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingInfo {
    /// Total GPU time in milliseconds.
    pub total_gpu_time_ms: u64,
    /// Total tokens processed.
    pub total_tokens_processed: u64,
    /// Total billed in SAGE tokens.
    pub total_sage_billed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// ISO 8601 timestamp of report generation.
    pub generated_at: String,
    /// Generator software version.
    pub generator: String,
}

// ─── Stage 6: Encryption (VM31 dev implements, Dev B integrates) ────────────

/// Trait for swappable encryption backends.
///
/// VM31 dev implements `PoseidonM31Encryption`.
/// AES-GCM fallback implements `Aes256GcmEncryption`.
pub trait AuditEncryption: Send + Sync {
    /// Encrypt plaintext for an owner.
    fn encrypt(
        &self,
        plaintext: &[u8],
        owner_pubkey: &[u8],
    ) -> Result<EncryptedBlob, EncryptionError>;

    /// Decrypt a blob using the recipient's private key.
    fn decrypt(
        &self,
        blob: &EncryptedBlob,
        recipient_address: &[u8],
        privkey: &[u8],
    ) -> Result<Vec<u8>, EncryptionError>;

    /// Wrap the encryption key for an additional recipient.
    fn wrap_key_for(
        &self,
        blob: &EncryptedBlob,
        owner_privkey: &[u8],
        owner_address: &[u8],
        grantee_pubkey: &[u8],
        grantee_address: &[u8],
    ) -> Result<WrappedKey, EncryptionError>;

    /// Name of this encryption scheme ("poseidon_m31" or "aes_256_gcm").
    fn scheme_name(&self) -> &str;
}

/// Encrypted blob — the ciphertext stored on Arweave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedBlob {
    /// Encrypted report bytes.
    pub ciphertext: Vec<u8>,
    /// Encryption scheme used.
    pub scheme: String,
    /// Nonce / IV.
    pub nonce: Vec<u8>,
    /// Wrapped keys — one per authorized recipient.
    pub wrapped_keys: Vec<WrappedKey>,
    /// Poseidon hash of the plaintext report (hex felt252).
    /// Matches on-chain `audit_report_hash`.
    pub plaintext_hash: String,
}

/// A data key encrypted for a specific Starknet address.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappedKey {
    /// Starknet address of the recipient (hex).
    pub recipient: String,
    /// The encryption key, wrapped with this recipient's public key.
    pub encrypted_key: Vec<u8>,
    /// Role: "owner", "auditor", "regulator".
    pub role: String,
    /// When access was granted (Unix epoch seconds).
    pub granted_at: u64,
}

/// Encryption errors.
#[derive(Debug, thiserror::Error)]
pub enum EncryptionError {
    #[error("Access denied: no wrapped key for this recipient")]
    AccessDenied,
    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),
    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),
    #[error("Key wrapping failed: {0}")]
    WrappingFailed(String),
    #[error("Plaintext hash mismatch: report was tampered")]
    HashMismatch,
}

// ─── Shared Errors ──────────────────────────────────────────────────────────

/// Errors from the audit system.
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("Log error: {0}")]
    LogError(String),
    #[error("Replay mismatch at sequence {sequence}: expected io_commitment {expected}, got {actual}")]
    ReplayMismatch {
        sequence: u64,
        expected: String,
        actual: String,
    },
    #[error("Chain integrity broken at sequence {sequence}: {detail}")]
    ChainBroken { sequence: u64, detail: String },
    #[error("Proving failed: {0}")]
    ProvingFailed(String),
    #[error("No inferences in window [{start}..{end}]")]
    EmptyWindow { start: u64, end: u64 },
    #[error("Weight commitment mismatch: log has {log_value}, model has {model_value}")]
    WeightMismatch { log_value: String, model_value: String },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serde(String),
    #[error("Encryption error: {0}")]
    Encryption(#[from] EncryptionError),
    #[error("Storage error: {0}")]
    Storage(String),
}
