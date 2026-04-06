//! `prove-server` — HTTP API wrapping the stwo-ml proving library.
//!
//! Accepts ONNX models, generates STARK proofs, and returns Starknet calldata.
//!
//! ```text
//! cargo build --release --bin prove-server --features server
//! BIND_ADDR=0.0.0.0:8080 ./target/release/prove-server
//! ```

#![feature(portable_simd)]

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "proof-stream")]
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
#[cfg(feature = "proof-stream")]
use axum::extract::Query;
use axum::{
    extract::{ConnectInfo, Path, State},
    http::{header::{AUTHORIZATION, CONTENT_TYPE}, Method, StatusCode},
    middleware,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use uuid::Uuid;

use stwo::core::fields::m31::M31;
use stwo_ml::circuits::batch::BatchPublicInputs;

use stwo_ml::compiler::hf_loader::load_hf_model;
use stwo_ml::compiler::onnx::load_onnx;
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::gadgets::quantize::{quantize_tensor, QuantStrategy};
use stwo_ml::starknet::{prepare_model_registration, prove_for_starknet_onchain};
use stwo_ml::tee::detect_tee_capability;

use stwo_ml::cairo_serde::serialize_proof;
use stwo_ml::privacy::relayer::{
    build_submit_batch_proof_calldata, compute_withdrawal_binding_digest,
    hash_batch_public_inputs_for_cairo, WithdrawalRecipients,
};

// =============================================================================
// State
// =============================================================================

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Queued,
    Proving,
    Completed,
    Failed,
}

struct ProveJob {
    job_id: String,
    model_id: String,
    status: JobStatus,
    progress_bps: u16,
    started_at: Instant,
    completed_at: Option<Instant>,
    error: Option<String>,
    result: Option<ProveResultPayload>,
}

#[derive(Clone, Serialize)]
struct ProveResultPayload {
    calldata: Vec<String>,
    io_commitment: String,
    weight_commitment: String,
    layer_chain_commitment: String,
    estimated_gas: u64,
    num_matmul_proofs: usize,
    num_layers: usize,
    prove_time_ms: u64,
    tee_attestation_hash: Option<String>,
    /// Policy preset name used for this proof.
    policy: String,
    /// Poseidon commitment of the policy configuration (hex felt252).
    policy_commitment: String,
}

struct LoadedModel {
    model_id: String,
    /// Human-readable name (e.g., "smollm2-135m"), derived from directory name.
    name: String,
    weight_commitment: String,
    num_layers: usize,
    input_shape: (usize, usize),
    /// Arc-wrapped to avoid O(model_size) clones per prove request.
    graph: Arc<stwo_ml::compiler::graph::ComputationGraph>,
    weights: Arc<stwo_ml::compiler::graph::GraphWeights>,
    #[cfg(feature = "server-audit")]
    capture_hook: Option<Arc<stwo_ml::audit::capture::CaptureHook>>,
}

struct PrivacyJob {
    job_id: String,
    status: JobStatus,
    started_at: Instant,
    completed_at: Option<Instant>,
    error: Option<String>,
    result: Option<PrivacyBatchResultPayload>,
    submit_context: Option<PrivacySubmitContext>,
}

#[derive(Clone)]
struct PrivacySubmitContext {
    public_inputs: BatchPublicInputs,
    payout_recipients: Vec<String>,
    credit_recipients: Vec<String>,
}

#[derive(Clone, Serialize)]
struct PrivacyBatchResultPayload {
    num_deposits: usize,
    num_withdrawals: usize,
    num_spends: usize,
    /// Canonical verifier proof hash (`verify_model_direct` event hash).
    /// This is never synthesized locally.
    proof_hash: Option<String>,
    /// `pending_verifier_canonical_hash` until caller supplies canonical hash.
    proof_hash_status: String,
    public_inputs_hash: String,
    prove_time_ms: u64,
    deposits: Vec<PrivacyDepositResult>,
    withdrawals: Vec<PrivacyWithdrawResult>,
    spends: Vec<PrivacySpendResult>,
    /// On-chain calldata for `submit_batch_proof` (full hardened VM31 ABI),
    /// populated only after canonical proof hash is provided.
    calldata_ready: bool,
    calldata: Vec<String>,
    payout_recipients: Vec<String>,
    credit_recipients: Vec<String>,
    /// Serialized STARK proof as felt252 hex values (for on-chain verification).
    stark_proof_calldata: Vec<String>,
}

#[derive(Clone, Serialize)]
struct PrivacyDepositResult {
    commitment: String,
    amount: u64,
    asset_id: u32,
}

#[derive(Clone, Serialize)]
struct PrivacyWithdrawResult {
    nullifier: String,
    amount: u64,
    asset_id: u32,
}

#[derive(Clone, Serialize)]
struct PrivacySpendResult {
    nullifiers: [String; 2],
    output_commitments: [String; 2],
}

/// Stored proof record for the verify endpoint.
#[derive(Clone, Serialize)]
struct StoredProof {
    proof_hash: String,
    model_id: String,
    io_commitment: String,
    weight_commitment: String,
    num_proven_layers: usize,
    prove_time_ms: u64,
    calldata_size: usize,
    created_at_epoch_ms: u64,
}

struct AppState {
    jobs: RwLock<HashMap<String, ProveJob>>,
    privacy_jobs: RwLock<HashMap<String, PrivacyJob>>,
    models: RwLock<HashMap<String, LoadedModel>>,
    /// Proof storage: proof_hash → StoredProof. Populated by /api/v1/infer.
    proofs: RwLock<HashMap<String, StoredProof>>,
    started_at: Instant,
    /// API key for bearer token authentication. None = open access (dev mode).
    api_key: Option<String>,
    /// Per-IP rate limiter state.
    rate_limiter: RateLimiter,
    #[cfg(feature = "proof-stream")]
    ws_sink: proof_stream::WsBroadcastSink,
    validator_url: Option<String>,
    #[cfg(feature = "multi-query")]
    scheduler: stwo_ml::gpu_scheduler::GpuScheduler,
    #[cfg(feature = "multi-query")]
    weight_caches: RwLock<HashMap<String, stwo_ml::weight_cache::SharedWeightCache>>,
    #[cfg(feature = "server-audit")]
    audit_jobs: RwLock<HashMap<String, AuditJob>>,
}

// =============================================================================
// Request / Response types
// =============================================================================

#[derive(Deserialize)]
struct LoadModelRequest {
    /// Path to ONNX model file on the server filesystem.
    model_path: String,
    /// Optional HuggingFace model directory (alternative to ONNX).
    model_dir: Option<String>,
    /// Optional human-readable description for on-chain registration.
    description: Option<String>,
}

#[derive(Deserialize)]
struct LoadHfModelRequest {
    /// Path to HuggingFace model directory on the server filesystem.
    model_dir: String,
    /// Number of transformer layers to load (optional, default: all).
    layers: Option<usize>,
    /// Optional human-readable description for on-chain registration.
    description: Option<String>,
}

#[derive(Serialize)]
struct LoadModelResponse {
    model_id: String,
    weight_commitment: String,
    num_layers: usize,
    input_shape: [usize; 2],
}

#[derive(Serialize)]
struct ModelInfoResponse {
    model_id: String,
    weight_commitment: String,
    num_layers: usize,
    input_shape: [usize; 2],
}

#[derive(Deserialize)]
struct ProveRequest {
    model_id: String,
    /// Flat array of f32 input values.
    input: Option<Vec<f32>>,
    #[serde(default)]
    gpu: bool,
    /// Security level: "auto" | "tee" | "zk-only"
    #[serde(default = "default_security")]
    security: String,
    /// Proof policy: "strict", "standard", or "relaxed".
    /// Defaults to "standard" (on-chain streaming compatible).
    #[serde(default)]
    policy: Option<String>,
}

fn default_security() -> String {
    "auto".to_string()
}

// =============================================================================
// Provable Inference API — POST /api/v1/infer
// =============================================================================

/// Request for the provable inference endpoint.
///
/// Takes a model ID, input data, and returns the model output + proof.
/// This is the primary API for building applications on top of ObelyZK.
#[derive(Deserialize)]
struct InferRequest {
    /// Model ID (must be loaded via POST /api/v1/models first).
    model_id: String,
    /// Input data as flat f32 array. Shape must match model's expected input.
    /// For transformer models: (seq_len, hidden_dim) flattened.
    input: Vec<f32>,
    /// Whether to use GPU proving (default: true if available).
    #[serde(default = "default_true")]
    gpu: bool,
    /// If true, return the full proof calldata (large). Default: false.
    #[serde(default)]
    include_calldata: bool,
    /// If true, return the raw output values. Default: true.
    #[serde(default = "default_true")]
    include_output: bool,
}

fn default_true() -> bool {
    true
}

/// Response from the provable inference endpoint.
#[derive(Serialize)]
struct InferResponse {
    /// Unique proof identifier.
    proof_id: String,
    /// Model output as flat f32 array (if include_output=true).
    output: Option<Vec<f32>>,
    /// Output shape (rows, cols).
    output_shape: (usize, usize),
    /// Poseidon hash of (input || output) — uniquely identifies this inference.
    io_commitment: String,
    /// Weight commitment — identifies the model version.
    weight_commitment: String,
    /// Proof hash for on-chain verification lookup.
    proof_hash: String,
    /// URL to verify this proof.
    verify_url: String,
    /// Number of proven layers (matmul + activation + attention + norm).
    num_proven_layers: usize,
    /// Wall-clock proving time in milliseconds.
    prove_time_ms: u64,
    /// Estimated on-chain verification gas.
    estimated_gas: u64,
    /// Full calldata (if include_calldata=true).
    calldata: Option<Vec<String>>,
    /// Proof size in felts.
    calldata_size: usize,
}

/// Response from the verify endpoint.
#[derive(Serialize)]
struct VerifyResponse {
    /// Whether the proof is valid.
    valid: bool,
    /// Proof hash that was verified.
    proof_hash: String,
    /// Model ID associated with this proof.
    model_id: Option<String>,
    /// IO commitment from the proof.
    io_commitment: Option<String>,
    /// Verification method used.
    method: String,
}

fn normalize_canonical_proof_hash(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("proof_hash must not be empty".to_string());
    }

    let raw_hex = if let Some(stripped) = trimmed.strip_prefix("0x") {
        stripped
    } else if let Some(stripped) = trimmed.strip_prefix("0X") {
        stripped
    } else {
        trimmed
    };

    if raw_hex.is_empty() {
        return Err("proof_hash must contain hex digits".to_string());
    }
    if raw_hex.len() > 64 {
        return Err("proof_hash exceeds felt252 hex length".to_string());
    }
    if !raw_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("proof_hash must be a felt252 hex string".to_string());
    }

    let normalized = raw_hex.trim_start_matches('0').to_ascii_lowercase();
    if normalized.is_empty() {
        return Err("proof_hash cannot be zero".to_string());
    }
    Ok(format!("0x{normalized}"))
}

#[derive(Serialize)]
struct ProveSubmitResponse {
    job_id: String,
    status: JobStatus,
}

#[derive(Serialize)]
struct ProveStatusResponse {
    job_id: String,
    status: JobStatus,
    progress_bps: u16,
    elapsed_secs: f64,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    uptime_secs: f64,
    gpu_available: bool,
    tee_available: bool,
    device_name: String,
    loaded_models: usize,
    active_jobs: usize,
}

// Privacy batch request/response types
#[derive(Deserialize)]
struct PrivacyBatchRequest {
    deposits: Vec<PrivacyDepositRequest>,
    #[serde(default)]
    withdrawals: Vec<PrivacyWithdrawRequest>,
    #[serde(default)]
    spends: Vec<PrivacySpendRequest>,
}

#[derive(Deserialize)]
struct PrivacyDepositRequest {
    amount: u64,
    asset_id: u32,
    recipient_pubkey: [u32; 4],
}

#[derive(Deserialize)]
struct PrivacyWithdrawRequest {
    note: PrivacyNoteRequest,
    spending_key: [u32; 4],
    merkle_siblings: Vec<[u32; 8]>,
    merkle_index: usize,
    merkle_root: [u32; 8],
    payout_recipient: Option<String>,
    #[serde(default)]
    credit_recipient: Option<String>,
}

#[derive(Deserialize)]
struct PrivacySpendRequest {
    inputs: [PrivacySpendInputRequest; 2],
    outputs: [PrivacySpendOutputRequest; 2],
    merkle_root: [u32; 8],
}

#[derive(Deserialize)]
struct PrivacyNoteRequest {
    pub_key: [u32; 4],
    amount_lo: u32,
    amount_hi: u32,
    asset_id: u32,
    blinding: [u32; 4],
}

#[derive(Deserialize)]
struct PrivacySpendInputRequest {
    note: PrivacyNoteRequest,
    spending_key: [u32; 4],
    merkle_siblings: Vec<[u32; 8]>,
    merkle_index: usize,
}

#[derive(Deserialize)]
struct PrivacySpendOutputRequest {
    recipient_pubkey: [u32; 4],
    amount_lo: u32,
    amount_hi: u32,
    asset_id: u32,
    blinding: [u32; 4],
}

#[derive(Serialize)]
struct PrivacyBatchSubmitResponse {
    job_id: String,
    status: JobStatus,
}

#[derive(Serialize)]
struct PrivacyBatchStatusResponse {
    job_id: String,
    status: JobStatus,
    elapsed_secs: f64,
}

#[derive(Deserialize)]
struct PrivacyBatchSubmitCalldataRequest {
    /// Canonical proof hash from verifier `ModelDirectVerified`.
    proof_hash: String,
}

#[derive(Serialize)]
struct PrivacyBatchSubmitCalldataResponse {
    job_id: String,
    proof_hash: String,
    public_inputs_hash: String,
    calldata: Vec<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// =============================================================================
// Security middleware
// =============================================================================

/// In-memory token-bucket rate limiter (per IP).
struct RateLimiter {
    /// requests per minute per IP
    rpm: u32,
    /// burst capacity
    burst: u32,
    buckets: Mutex<HashMap<IpAddr, (Instant, u32)>>,
}

impl RateLimiter {
    fn new(rpm: u32, burst: u32) -> Self {
        Self {
            rpm,
            burst,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    async fn check(&self, ip: IpAddr) -> bool {
        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();
        let entry = buckets.entry(ip).or_insert((now, 0));

        // Reset bucket if more than 60s elapsed
        if now.duration_since(entry.0).as_secs() >= 60 {
            *entry = (now, 0);
        }

        if entry.1 < self.rpm + self.burst {
            entry.1 += 1;
            true
        } else {
            false
        }
    }

    /// Evict stale entries older than 2 minutes to prevent unbounded memory growth.
    async fn evict_stale(&self) {
        let mut buckets = self.buckets.lock().await;
        let now = Instant::now();
        buckets.retain(|_, (t, _)| now.duration_since(*t).as_secs() < 120);
    }
}

/// Constant-time token comparison via SHA-256.
///
/// Hashing both sides ensures equal-length comparison and prevents
/// timing side-channel attacks on the API key.
fn constant_time_eq(a: &str, b: &str) -> bool {
    use sha2::{Sha256, Digest};
    let ha = Sha256::digest(a.as_bytes());
    let hb = Sha256::digest(b.as_bytes());
    // Fixed-length (32 byte) comparison — constant time for equal-length slices
    ha.as_slice()
        .iter()
        .zip(hb.as_slice().iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Bearer token authentication middleware.
///
/// When `PROVE_SERVER_API_KEY` is set, all requests through this layer must
/// include `Authorization: Bearer <key>`. When unset, all requests pass through.
async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    if let Some(ref expected) = state.api_key {
        let auth_ok = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .map(|token| constant_time_eq(token, expected))
            .unwrap_or(false);
        if !auth_ok {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    Ok(next.run(req).await)
}

/// Rate limiting middleware for expensive endpoints.
async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    if !state.rate_limiter.check(addr.ip()).await {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    Ok(next.run(req).await)
}

/// Validate a model path: canonicalize and check against allowlist.
fn validate_model_path(
    p: &std::path::Path,
) -> Result<std::path::PathBuf, (StatusCode, Json<ErrorResponse>)> {
    // First reject obvious traversal in the raw string (before canonicalize, which needs the file to exist)
    let s = p.to_string_lossy();
    if s.contains("..") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Path traversal not allowed".to_string(),
            }),
        ));
    }

    // Canonicalize to resolve symlinks and normalize
    let canonical = std::fs::canonicalize(p).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid path: does not exist or cannot be resolved".to_string(),
            }),
        )
    })?;

    // Check against allowlist if configured
    if let Ok(allowed) = std::env::var("PROVE_SERVER_MODEL_DIR") {
        let allowed_canonical = std::fs::canonicalize(&allowed)
            .unwrap_or_else(|_| std::path::PathBuf::from(&allowed));
        if !canonical.starts_with(&allowed_canonical) {
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: "Path outside allowed model directory".to_string(),
                }),
            ));
        }
    }

    Ok(canonical)
}

// =============================================================================
// Audit types (server-audit)
// =============================================================================

#[cfg(feature = "server-audit")]
struct AuditJob {
    job_id: String,
    model_id: String,
    status: JobStatus,
    started_at: Instant,
    completed_at: Option<Instant>,
    error: Option<String>,
    report: Option<serde_json::Value>,
}

#[cfg(feature = "server-audit")]
#[derive(Deserialize)]
struct AuditTriggerRequest {
    model_id: String,
    /// Proof mode: "gkr" (default) or "direct".
    #[serde(default = "default_audit_mode")]
    mode: String,
    /// Max inferences to audit (0 = all).
    #[serde(default)]
    max_inferences: usize,
    /// Weight binding mode: "aggregated" (default).
    #[serde(default = "default_weight_binding_mode")]
    weight_binding: String,
}

#[cfg(feature = "server-audit")]
fn default_audit_mode() -> String {
    "gkr".to_string()
}
#[cfg(feature = "server-audit")]
fn default_weight_binding_mode() -> String {
    "aggregated".to_string()
}

#[cfg(feature = "server-audit")]
#[derive(Serialize)]
struct AuditSubmitResponse {
    job_id: String,
    status: JobStatus,
}

#[cfg(feature = "server-audit")]
#[derive(Serialize)]
struct AuditStatusResponse {
    job_id: String,
    status: JobStatus,
    elapsed_secs: f64,
    report: Option<serde_json::Value>,
    error: Option<String>,
}

// =============================================================================
// Handlers
// =============================================================================

#[cfg(feature = "server-audit")]
fn create_capture_hook(
    model_id: &str,
    weight_commitment: &str,
    description: &str,
) -> Option<Arc<stwo_ml::audit::capture::CaptureHook>> {
    let audit_dir = match std::env::var("AUDIT_LOG_DIR") {
        Ok(d) => d,
        Err(_) => return None,
    };
    let hook_dir = std::path::Path::new(&audit_dir).join(model_id);
    match stwo_ml::audit::capture::CaptureHook::new(
        &hook_dir,
        model_id,
        weight_commitment,
        description,
    ) {
        Ok(hook) => {
            eprintln!(
                "Audit capture enabled for model {} -> {}",
                model_id,
                hook_dir.display()
            );
            Some(Arc::new(hook))
        }
        Err(e) => {
            eprintln!("Warning: could not enable audit capture: {e}");
            None
        }
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let cap = detect_tee_capability();
    let models = state.models.read().await;
    let jobs = state.jobs.read().await;
    let active = jobs
        .values()
        .filter(|j| j.status == JobStatus::Proving || j.status == JobStatus::Queued)
        .count();

    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_secs: state.started_at.elapsed().as_secs_f64(),
        gpu_available: cap.cc_supported || !cap.device_name.is_empty(),
        tee_available: cap.cc_active && cap.nvattest_available,
        device_name: cap.device_name,
        loaded_models: models.len(),
        active_jobs: active,
    })
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<(StatusCode, Json<LoadModelResponse>), (StatusCode, Json<ErrorResponse>)> {
    // Support either ONNX file or HF directory — with canonicalized path validation
    let (onnx, canonical_path) = if let Some(ref model_dir) = req.model_dir {
        let path = validate_model_path(&std::path::PathBuf::from(model_dir))?;
        let model = load_hf_model(&path, None).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Failed to load HF model: {e}"),
                }),
            )
        })?;
        (model, path)
    } else {
        let path = validate_model_path(&std::path::PathBuf::from(&req.model_path))?;
        let model = load_onnx(&path).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Failed to load model: {e}"),
                }),
            )
        })?;
        // For ONNX files, use the parent directory for cache
        let dir = path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        (model, dir)
    };

    let desc = req.description.as_deref().unwrap_or("api-loaded-model");
    let registration = prepare_model_registration(&onnx.graph, &onnx.weights, desc);
    let model_id = format!("0x{:x}", registration.model_id);
    let weight_commitment = format!("0x{:x}", registration.weight_commitment);

    let resp = LoadModelResponse {
        model_id: model_id.clone(),
        weight_commitment: weight_commitment.clone(),
        num_layers: registration.num_layers,
        input_shape: [onnx.input_shape.0, onnx.input_shape.1],
    };

    let model_name = std::path::Path::new(&req.model_path)
        .file_stem()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "onnx-model".into());

    let loaded = LoadedModel {
        model_id: model_id.clone(),
        name: model_name.clone(),
        weight_commitment: weight_commitment.clone(),
        num_layers: registration.num_layers,
        input_shape: onnx.input_shape,
        graph: Arc::new(onnx.graph),
        weights: Arc::new(onnx.weights),
        #[cfg(feature = "server-audit")]
        capture_hook: create_capture_hook(&model_id, &weight_commitment, desc),
    };

    state.models.write().await.insert(model_id.clone(), loaded);

    // Use the canonicalized path for weight cache (not raw user input)
    #[cfg(feature = "multi-query")]
    {
        let cache =
            stwo_ml::weight_cache::shared_cache_for_model(&canonical_path, &model_id);
        state
            .weight_caches
            .write()
            .await
            .insert(model_id, cache);
    }

    // Suppress unused variable warning when multi-query is disabled
    #[cfg(not(feature = "multi-query"))]
    let _ = canonical_path;

    Ok((StatusCode::CREATED, Json(resp)))
}

async fn load_hf_model_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadHfModelRequest>,
) -> Result<(StatusCode, Json<LoadModelResponse>), (StatusCode, Json<ErrorResponse>)> {
    let path = validate_model_path(&std::path::PathBuf::from(&req.model_dir))?;
    let hf = load_hf_model(&path, req.layers).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Failed to load HF model: {e}"),
            }),
        )
    })?;

    let desc = req.description.as_deref().unwrap_or("hf-loaded-model");
    let registration = prepare_model_registration(&hf.graph, &hf.weights, desc);
    let model_id = format!("0x{:x}", registration.model_id);
    let weight_commitment = format!("0x{:x}", registration.weight_commitment);

    let resp = LoadModelResponse {
        model_id: model_id.clone(),
        weight_commitment: weight_commitment.clone(),
        num_layers: registration.num_layers,
        input_shape: [hf.input_shape.0, hf.input_shape.1],
    };

    let model_name = std::path::Path::new(&req.model_dir)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "hf-model".into());

    let loaded = LoadedModel {
        model_id: model_id.clone(),
        name: model_name,
        weight_commitment: weight_commitment.clone(),
        num_layers: registration.num_layers,
        input_shape: hf.input_shape,
        graph: Arc::new(hf.graph),
        weights: Arc::new(hf.weights),
        #[cfg(feature = "server-audit")]
        capture_hook: create_capture_hook(&model_id, &weight_commitment, desc),
    };

    state.models.write().await.insert(model_id.clone(), loaded);

    // Use the canonicalized path for weight cache (not raw user input)
    #[cfg(feature = "multi-query")]
    {
        let cache = stwo_ml::weight_cache::shared_cache_for_model(&path, &model_id);
        state
            .weight_caches
            .write()
            .await
            .insert(model_id, cache);
    }

    Ok((StatusCode::CREATED, Json(resp)))
}

/// List all loaded models.
async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<ModelListEntry>> {
    let models = state.models.read().await;
    let entries: Vec<ModelListEntry> = models.values().map(|m| ModelListEntry {
        model_id: m.model_id.clone(),
        name: m.name.clone(),
        weight_commitment: m.weight_commitment.clone(),
        num_layers: m.num_layers,
        input_shape: [m.input_shape.0, m.input_shape.1],
    }).collect();
    Json(entries)
}

#[derive(Serialize)]
struct ModelListEntry {
    model_id: String,
    name: String,
    weight_commitment: String,
    num_layers: usize,
    input_shape: [usize; 2],
}

/// Resolve a model by ID or name. Accepts "0x..." model_id or "smollm2-135m" name.
fn resolve_model_id<'a>(
    models: &'a HashMap<String, LoadedModel>,
    id_or_name: &str,
) -> Option<&'a LoadedModel> {
    // Try exact model_id match first
    if let Some(m) = models.get(id_or_name) {
        return Some(m);
    }
    // Try name match (case-insensitive)
    let lower = id_or_name.to_lowercase();
    models.values().find(|m| m.name.to_lowercase() == lower)
}

async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelInfoResponse>, (StatusCode, Json<ErrorResponse>)> {
    let models = state.models.read().await;
    let model = resolve_model_id(&models, &model_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model '{model_id}' not found"),
            }),
        )
    })?;

    Ok(Json(ModelInfoResponse {
        model_id: model.model_id.clone(),
        weight_commitment: model.weight_commitment.clone(),
        num_layers: model.num_layers,
        input_shape: [model.input_shape.0, model.input_shape.1],
    }))
}

async fn submit_prove(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProveRequest>,
) -> Result<(StatusCode, Json<ProveSubmitResponse>), (StatusCode, Json<ErrorResponse>)> {
    // Validate model exists (accepts model_id or name like "smollm2-135m")
    let models = state.models.read().await;
    let model = resolve_model_id(&models, &req.model_id).ok_or_else(|| {
        let available: Vec<String> = models.values().map(|m| m.name.clone()).collect();
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!(
                    "Model '{}' not found. Available: [{}]. Load via POST /api/v1/models/hf",
                    req.model_id,
                    available.join(", ")
                ),
            }),
        )
    })?;

    let (in_rows, in_cols) = model.input_shape;
    let input_matrix = if let Some(ref input_f32) = req.input {
        let expected = in_rows * in_cols;
        if input_f32.len() != expected {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "Input has {} values, model expects {} ({in_rows}x{in_cols})",
                        input_f32.len(),
                        expected
                    ),
                }),
            ));
        }
        // Quantize f32 -> M31 using the same strategy as prove-model CLI
        let (quantized, _params) = quantize_tensor(input_f32, QuantStrategy::Symmetric8);
        let mut matrix = M31Matrix::new(in_rows, in_cols);
        for (i, &v) in quantized.iter().enumerate().take(in_rows * in_cols) {
            matrix.data[i] = v;
        }
        matrix
    } else {
        // Generate deterministic random input
        let mut matrix = M31Matrix::new(in_rows, in_cols);
        for i in 0..(in_rows * in_cols) {
            matrix.data[i] = M31::from((i as u32 * 7 + 13) % (1 << 20));
        }
        matrix
    };

    // Resolve policy from request (defaults to "standard" if not specified).
    let request_policy = match req.policy.as_deref() {
        None | Some("standard") => stwo_ml::policy::PolicyConfig::standard(),
        Some("strict") => stwo_ml::policy::PolicyConfig::strict(),
        Some("relaxed") => stwo_ml::policy::PolicyConfig::relaxed(),
        Some(unknown) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!(
                        "unknown policy '{unknown}'. Valid options: strict, standard, relaxed"
                    ),
                }),
            ));
        }
    };
    let request_policy_name = stwo_ml::policy::preset_name(&request_policy)
        .unwrap_or("custom")
        .to_string();
    let request_policy_commitment_hex = format!("0x{:x}", request_policy.policy_commitment());

    // Arc::clone is O(1) — no deep copy of model weights
    let graph = Arc::clone(&model.graph);
    let weights = Arc::clone(&model.weights);
    let model_weight_commitment = model.weight_commitment.clone();
    let model_id_clone = req.model_id.clone();
    let model_id_for_proofs = req.model_id.clone();
    // Clone for audit capture after proving
    #[cfg(feature = "server-audit")]
    let audit_graph = Arc::clone(&model.graph);
    #[cfg(feature = "server-audit")]
    let audit_weights = Arc::clone(&model.weights);
    #[cfg(feature = "server-audit")]
    let input_matrix_clone = input_matrix.clone();
    #[cfg(feature = "server-audit")]
    let jid_model = req.model_id.clone();
    drop(models);

    let job_id = Uuid::new_v4().to_string();
    let job = ProveJob {
        job_id: job_id.clone(),
        model_id: model_id_clone,
        status: JobStatus::Queued,
        progress_bps: 0,
        started_at: Instant::now(),
        completed_at: None,
        error: None,
        result: None,
    };

    state.jobs.write().await.insert(job_id.clone(), job);

    let resp = ProveSubmitResponse {
        job_id: job_id.clone(),
        status: JobStatus::Queued,
    };

    // -------------------------------------------------------------------------
    // multi-query: Submit to GPU scheduler with bounded concurrency
    // -------------------------------------------------------------------------
    #[cfg(feature = "multi-query")]
    {
        use stwo_ml::gpu_scheduler::{ProveJobResult as GpuJobResult, ScheduledJob};

        // Look up weight cache for this model (if available)
        let weight_cache = {
            let caches = state.weight_caches.read().await;
            caches.get(&req.model_id).cloned()
        };

        #[cfg(feature = "proof-stream")]
        let ws_clone = state.ws_sink.clone();

        let policy_name_for_closure = request_policy_name.clone();
        let policy_commitment_for_closure = request_policy_commitment_hex.clone();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let scheduled = ScheduledJob {
            job_id: job_id.clone(),
            estimated_gpu_memory: 0, // advisory, future use
            prove_fn: Box::new(move |_device_id| {
                // Install proof-stream sink on the blocking thread
                #[cfg(feature = "proof-stream")]
                let _sink_guard = stwo_ml::gkr::prover::set_proof_sink(
                    proof_stream::ProofSink::new(ws_clone),
                );

                let prove_start = Instant::now();
                let proof_result = if let Some(ref cache) = weight_cache {
                    stwo_ml::starknet::prove_for_starknet_onchain_cached(
                        &*graph,
                        &input_matrix,
                        &*weights,
                        cache,
                    )
                } else {
                    prove_for_starknet_onchain(&*graph, &input_matrix, &*weights)
                };

                match proof_result {
                    Ok(proof) => {
                        let elapsed_ms = prove_start.elapsed().as_millis() as u64;
                        // Serialize proof to bytes for transport through the scheduler
                        let payload_json = serde_json::json!({
                            "calldata": proof.combined_calldata.iter()
                                .map(|f| format!("0x{:x}", f)).collect::<Vec<_>>(),
                            "io_commitment": format!("0x{:x}", proof.io_commitment),
                            "layer_chain_commitment": format!("0x{:x}", proof.layer_chain_commitment),
                            "estimated_gas": proof.estimated_gas,
                            "num_matmul_proofs": proof.num_matmul_proofs,
                            "num_proven_layers": proof.num_proven_layers,
                            "tee_attestation_hash": proof.tee_attestation_hash
                                .map(|h| format!("0x{:x}", h)),
                        });
                        Ok(GpuJobResult {
                            data: serde_json::to_vec(&payload_json).unwrap_or_default(),
                            prove_time_ms: elapsed_ms,
                            device_id: _device_id,
                        })
                    }
                    Err(e) => Err(format!("{e}")),
                }
            }),
            result_tx,
            submitted_at: Instant::now(),
        };

        if let Err(e) = state.scheduler.submit(scheduled) {
            // Queue full — return 429 Too Many Requests
            let mut jobs_w = state.jobs.write().await;
            if let Some(j) = jobs_w.get_mut(&job_id) {
                j.status = JobStatus::Failed;
                j.error = Some(format!("{e}"));
                j.completed_at = Some(Instant::now());
            }
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: format!("{e}"),
                }),
            ));
        }

        // Spawn async task to await result and update job status
        let state_clone = state.clone();
        let jid = job_id.clone();
        let validator_url_clone = state_clone.validator_url.clone();
        #[allow(unused_variables)]
        let jid_for_validator = jid.clone();
        tokio::task::spawn(async move {
            // Mark as proving
            {
                let mut jobs = state_clone.jobs.write().await;
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Proving;
                    j.progress_bps = 100;
                }
            }

            match result_rx.await {
                Ok(Ok(gpu_result)) => {
                    // Parse the serialized proof payload
                    let payload_json: serde_json::Value =
                        serde_json::from_slice(&gpu_result.data).unwrap_or_default();

                    let calldata: Vec<String> = payload_json["calldata"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                        .unwrap_or_default();

                    let payload = ProveResultPayload {
                        calldata,
                        io_commitment: payload_json["io_commitment"]
                            .as_str()
                            .unwrap_or("0x0")
                            .to_string(),
                        weight_commitment: model_weight_commitment.clone(),
                        layer_chain_commitment: payload_json["layer_chain_commitment"]
                            .as_str()
                            .unwrap_or("0x0")
                            .to_string(),
                        estimated_gas: payload_json["estimated_gas"].as_u64().unwrap_or(0),
                        num_matmul_proofs: payload_json["num_matmul_proofs"]
                            .as_u64()
                            .unwrap_or(0) as usize,
                        num_layers: payload_json["num_proven_layers"]
                            .as_u64()
                            .unwrap_or(0) as usize,
                        prove_time_ms: gpu_result.prove_time_ms,
                        tee_attestation_hash: payload_json["tee_attestation_hash"]
                            .as_str()
                            .map(String::from),
                        policy: policy_name_for_closure.clone(),
                        policy_commitment: policy_commitment_for_closure.clone(),
                    };

                    let mut jobs = state_clone.jobs.write().await;
                    if let Some(j) = jobs.get_mut(&jid) {
                        j.status = JobStatus::Completed;
                        j.progress_bps = 10000;
                        j.completed_at = Some(Instant::now());
                        j.result = Some(payload);
                    }

                    // Forward proof result to validator if configured
                    #[cfg(any(feature = "audit-http", feature = "server-stream"))]
                    if let Some(ref url) = validator_url_clone {
                        let post_url =
                            format!("{url}/api/v1/workers/job/{jid_for_validator}/result");
                        let elapsed_ms = gpu_result.prove_time_ms;
                        let jid_v = jid_for_validator.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            let body = format!(
                                r#"{{"job_id":"{}","success":true,"generation_time_ms":{}}}"#,
                                jid_v, elapsed_ms
                            );
                            match ureq::post(&post_url)
                                .header("Content-Type", "application/json")
                                .send(body.as_bytes())
                            {
                                Ok(resp) if resp.status() == 200 || resp.status() == 201 => {}
                                Ok(resp) => eprintln!(
                                    "[prove-server] validator returned HTTP {} for job {}",
                                    resp.status(),
                                    jid_v
                                ),
                                Err(e) => eprintln!(
                                    "[prove-server] validator bridge error for job {}: {}",
                                    jid_v, e
                                ),
                            }
                        })
                        .await;
                    }
                }
                Ok(Err(e)) => {
                    let mut jobs = state_clone.jobs.write().await;
                    if let Some(j) = jobs.get_mut(&jid) {
                        j.status = JobStatus::Failed;
                        j.error = Some(e);
                        j.completed_at = Some(Instant::now());
                    }
                }
                Err(_) => {
                    let mut jobs = state_clone.jobs.write().await;
                    if let Some(j) = jobs.get_mut(&jid) {
                        j.status = JobStatus::Failed;
                        j.error = Some("Scheduler channel closed".to_string());
                        j.completed_at = Some(Instant::now());
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------------------
    // Legacy path: unbounded spawn_blocking (no scheduler)
    // -------------------------------------------------------------------------
    #[cfg(not(feature = "multi-query"))]
    {
    let state_clone = state.clone();
    let jid = job_id.clone();
    tokio::task::spawn(async move {
        // Mark as proving
        {
            let mut jobs = state_clone.jobs.write().await;
            if let Some(j) = jobs.get_mut(&jid) {
                j.status = JobStatus::Proving;
                j.progress_bps = 100; // 1%
            }
        }

        let prove_start = Instant::now();

        // Progress ticker: estimate progress based on elapsed time
        // SmolLM2-135M takes ~8s, Qwen2-0.5B ~20s. Use 15s as default estimate.
        let progress_state = state_clone.clone();
        let progress_jid = jid.clone();
        let progress_handle = tokio::spawn(async move {
            let estimated_secs = 15.0_f64;
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let elapsed = prove_start.elapsed().as_secs_f64();
                // Asymptotic progress: approaches 95% but never reaches 100%
                let pct = 1.0 - (-elapsed / estimated_secs).exp();
                let bps = (pct * 9500.0) as u16 + 100; // 100..9600 range
                let mut jobs = progress_state.jobs.write().await;
                match jobs.get_mut(&progress_jid) {
                    Some(j) if j.status == JobStatus::Proving => {
                        j.progress_bps = bps;
                    }
                    _ => break, // job completed or removed
                }
            }
        });

        // Run CPU+GPU heavy proving on a blocking thread
        #[cfg(feature = "proof-stream")]
        let ws_clone = state_clone.ws_sink.clone();
        let validator_url_clone = state_clone.validator_url.clone();
        let jid_for_validator = jid.clone();

        let result = tokio::task::spawn_blocking(move || {
            // Install proof-stream sink on the blocking thread (thread-local must
            // live on the same thread that runs the prover).
            #[cfg(feature = "proof-stream")]
            let _sink_guard =
                stwo_ml::gkr::prover::set_proof_sink(proof_stream::ProofSink::new(ws_clone));
            prove_for_starknet_onchain(&*graph, &input_matrix, &*weights)
        })
        .await;

        // Stop the progress ticker
        progress_handle.abort();

        let prove_elapsed = prove_start.elapsed();
        let mut jobs = state_clone.jobs.write().await;

        match result {
            Ok(Ok(proof)) => {
                let calldata: Vec<String> = proof
                    .combined_calldata
                    .iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect();

                let payload = ProveResultPayload {
                    calldata,
                    io_commitment: format!("0x{:x}", proof.io_commitment),
                    weight_commitment: model_weight_commitment.clone(),
                    layer_chain_commitment: format!("0x{:x}", proof.layer_chain_commitment),
                    estimated_gas: proof.estimated_gas,
                    num_matmul_proofs: proof.num_matmul_proofs,
                    num_layers: proof.num_proven_layers,
                    prove_time_ms: prove_elapsed.as_millis() as u64,
                    tee_attestation_hash: proof.tee_attestation_hash.map(|h| format!("0x{:x}", h)),
                    policy: request_policy_name.clone(),
                    policy_commitment: request_policy_commitment_hex.clone(),
                };

                // Store in proofs map for /api/v1/proofs listing
                let io_str = format!("0x{:x}", proof.io_commitment);
                let proof_hash = format!("0x{:x}", proof.layer_chain_commitment);
                {
                    let stored = StoredProof {
                        proof_hash: proof_hash.clone(),
                        model_id: model_id_for_proofs.clone(),
                        io_commitment: io_str.clone(),
                        weight_commitment: model_weight_commitment.clone(),
                        num_proven_layers: proof.num_proven_layers,
                        prove_time_ms: prove_elapsed.as_millis() as u64,
                        calldata_size: payload.calldata.len(),
                        created_at_epoch_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64,
                    };
                    drop(jobs);
                    state_clone.proofs.write().await.insert(proof_hash, stored);
                    let mut jobs = state_clone.jobs.write().await;
                    if let Some(j) = jobs.get_mut(&jid) {
                        j.status = JobStatus::Completed;
                        j.progress_bps = 10000;
                        j.completed_at = Some(Instant::now());
                        j.result = Some(payload);
                    }
                }

                // Forward proof result to validator if configured
                #[cfg(any(feature = "audit-http", feature = "server-stream"))]
                if let Some(ref url) = validator_url_clone {
                    let post_url = format!("{url}/api/v1/workers/job/{jid_for_validator}/result");
                    let elapsed_ms = prove_elapsed.as_millis() as u64;
                    let jid_v = jid_for_validator.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        let body = format!(
                            r#"{{"job_id":"{}","success":true,"generation_time_ms":{}}}"#,
                            jid_v, elapsed_ms
                        );
                        match ureq::post(&post_url)
                            .header("Content-Type", "application/json")
                            .send(body.as_bytes())
                        {
                            Ok(resp) if resp.status() == 200 || resp.status() == 201 => {}
                            Ok(resp) => eprintln!(
                                "[prove-server] validator returned HTTP {} for job {}",
                                resp.status(),
                                jid_v
                            ),
                            Err(e) => eprintln!(
                                "[prove-server] validator bridge error for job {}: {}",
                                jid_v, e
                            ),
                        }
                    })
                    .await;
                }

                // Record inference for audit log
                #[cfg(feature = "server-audit")]
                {
                    let tee_hash = proof
                        .tee_attestation_hash
                        .map(|h| format!("0x{:x}", h))
                        .unwrap_or_else(|| "0x0".to_string());
                    drop(jobs);
                    let models = state_clone.models.read().await;
                    if let Some(model) = models.get(&jid_model) {
                        if let Some(ref hook) = model.capture_hook {
                            // Replay forward pass to get output for audit record
                            if let Ok(output_m31) = stwo_ml::audit::replay::execute_forward_pass(
                                &audit_graph,
                                &input_matrix_clone,
                                &audit_weights,
                            ) {
                                hook.record(stwo_ml::audit::capture::CaptureJob {
                                    input_tokens: vec![],
                                    output_tokens: vec![],
                                    input_m31: input_matrix_clone,
                                    output_m31,
                                    timestamp_ns: std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_nanos()
                                        as u64,
                                    latency_ms: prove_elapsed.as_millis() as u64,
                                    gpu_device: "server".to_string(),
                                    tee_report_hash: tee_hash,
                                    task_category: Some("prove".to_string()),
                                    input_preview: None,
                                    output_preview: None,
                                });
                            }
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("{e}"));
                    j.completed_at = Some(Instant::now());
                }
            }
            Err(e) => {
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("Task panicked: {e}"));
                    j.completed_at = Some(Instant::now());
                }
            }
        }
    });
    }

    Ok((StatusCode::ACCEPTED, Json(resp)))
}

async fn get_prove_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<ProveStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let jobs = state.jobs.read().await;
    let job = jobs.get(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Job '{job_id}' not found"),
            }),
        )
    })?;

    Ok(Json(ProveStatusResponse {
        job_id: job.job_id.clone(),
        status: job.status,
        progress_bps: job.progress_bps,
        elapsed_secs: job.started_at.elapsed().as_secs_f64(),
    }))
}

async fn get_prove_result(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<ProveResultPayload>, (StatusCode, Json<ErrorResponse>)> {
    let jobs = state.jobs.read().await;
    let job = jobs.get(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Job '{job_id}' not found"),
            }),
        )
    })?;

    match job.status {
        JobStatus::Completed => {
            let result = job.result.as_ref().ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Job completed but result is missing (internal error)".to_string(),
                    }),
                )
            })?;
            Ok(Json(result.clone()))
        }
        JobStatus::Failed => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: job
                    .error
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            }),
        )),
        _ => Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!("Job is still {:?}", job.status),
            }),
        )),
    }
}

// =============================================================================
// Provable Inference Handler — POST /api/v1/infer
// =============================================================================

/// Synchronous provable inference: input → forward pass → GKR proof → response.
///
/// This is the primary endpoint for building applications on ObelyZK.
/// Unlike POST /api/v1/prove (async job), this blocks until the proof is ready.
async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, (StatusCode, Json<ErrorResponse>)> {
    let models = state.models.read().await;
    let model = resolve_model_id(&models, &req.model_id).ok_or_else(|| {
        let available: Vec<String> = models.values().map(|m| m.name.clone()).collect();
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!(
                    "Model '{}' not found. Available: [{}]",
                    req.model_id, available.join(", ")
                ),
            }),
        )
    })?;

    let (in_rows, in_cols) = model.input_shape;
    let expected = in_rows * in_cols;
    if req.input.len() != expected {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Input has {} values, model expects {} ({in_rows}×{in_cols})",
                    req.input.len(),
                    expected
                ),
            }),
        ));
    }

    // Quantize input
    let (quantized, _params) = quantize_tensor(&req.input, QuantStrategy::Symmetric8);
    let mut input_matrix = M31Matrix::new(in_rows, in_cols);
    for (i, &v) in quantized.iter().enumerate().take(in_rows * in_cols) {
        input_matrix.data[i] = v;
    }

    let graph = Arc::clone(&model.graph);
    let weights = Arc::clone(&model.weights);
    let model_id_for_storage = req.model_id.clone();
    let model_id_clone = req.model_id.clone();
    let weight_commitment = model.weight_commitment.clone();
    let include_calldata = req.include_calldata;
    let include_output = req.include_output;

    // Drop the read lock before spawning blocking work
    drop(models);

    // Run inference + proving in a blocking task (CPU/GPU-heavy)
    let result = tokio::task::spawn_blocking(move || {
        let t_start = Instant::now();

        // Full GKR proof (includes forward pass)
        let proof_result = stwo_ml::aggregation::prove_model_aggregated_onchain_gkr_auto(
            &graph,
            &input_matrix,
            &weights,
        );

        match proof_result {
            Ok(proof) => {
                let prove_time_ms = t_start.elapsed().as_millis() as u64;

                // Extract output
                let output_matrix = &proof.execution.output;
                let output_f32 = if include_output {
                    Some(output_matrix.data.iter().map(|v| v.0 as f32).collect::<Vec<f32>>())
                } else {
                    None
                };

                // Compute proof hash (Poseidon of io_commitment + weight_commitment)
                let io_hex = format!("0x{:x}", proof.io_commitment);
                let proof_hash = format!("0x{:x}",
                    starknet_crypto::poseidon_hash_many(&[
                        proof.io_commitment,
                        proof.layer_chain_commitment,
                    ])
                );

                // Serialize calldata
                let gkr_proof = proof.gkr_proof.as_ref();
                let calldata_felts = if let Some(gkr) = gkr_proof {
                    let mut felts = Vec::new();
                    stwo_ml::cairo_serde::serialize_gkr_proof_data_only(gkr, &mut felts);
                    felts
                } else {
                    Vec::new()
                };
                let calldata_size = calldata_felts.len();

                let calldata = if include_calldata {
                    Some(calldata_felts.iter().map(|f| format!("0x{:x}", f)).collect())
                } else {
                    None
                };

                let proof_id = format!("proof-{}", uuid::Uuid::new_v4());
                let num_proven = proof.num_proven_layers();

                Ok(InferResponse {
                    proof_id,
                    output: output_f32,
                    output_shape: (output_matrix.rows, output_matrix.cols),
                    io_commitment: io_hex,
                    weight_commitment,
                    proof_hash: proof_hash.clone(),
                    verify_url: format!("/api/v1/verify/:proof_hash"),
                    num_proven_layers: num_proven,
                    prove_time_ms,
                    estimated_gas: (calldata_size as u64) * 200 + 500_000,
                    calldata,
                    calldata_size,
                })
            }
            Err(e) => Err(format!("Proving failed: {e}")),
        }
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Task join error: {e}"),
            }),
        )
    })?;

    match result {
        Ok(response) => {
            // Store proof for later verification
            let stored = StoredProof {
                proof_hash: response.proof_hash.clone(),
                model_id: model_id_for_storage,
                io_commitment: response.io_commitment.clone(),
                weight_commitment: response.weight_commitment.clone(),
                num_proven_layers: response.num_proven_layers,
                prove_time_ms: response.prove_time_ms,
                calldata_size: response.calldata_size,
                created_at_epoch_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            };
            state.proofs.write().await.insert(response.proof_hash.clone(), stored);

            Ok(Json(response))
        }
        Err(err) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: err }),
        )),
    }
}

/// Verify a proof by hash — GET /api/v1/verify/:proof_hash
///
/// Looks up the proof in local storage. Returns verification status,
/// model ID, and IO commitment for the proof.
///
/// Future: also query Starknet contract for on-chain verification status.
async fn verify_proof(
    State(state): State<Arc<AppState>>,
    Path(proof_hash): Path<String>,
) -> Json<VerifyResponse> {
    let proofs = state.proofs.read().await;
    if let Some(stored) = proofs.get(&proof_hash) {
        Json(VerifyResponse {
            valid: true,
            proof_hash: stored.proof_hash.clone(),
            model_id: Some(stored.model_id.clone()),
            io_commitment: Some(stored.io_commitment.clone()),
            method: "local_storage".to_string(),
        })
    } else {
        Json(VerifyResponse {
            valid: false,
            proof_hash,
            model_id: None,
            io_commitment: None,
            method: "not_found".to_string(),
        })
    }
}

// =============================================================================
// Full Attestation Handler — POST /api/v1/attest
// =============================================================================

/// Full attestation: prove + build streaming calldata + submit on-chain.
///
/// This is the single-call endpoint that does everything:
/// 1. Runs the M31 forward pass through all model layers
/// 2. Generates GKR sumcheck proofs for every MatMul
/// 3. Builds unified STARK proof for activations + norms
/// 4. Chunks calldata into 6 streaming verification steps
/// 5. Submits all 6 transactions to Starknet using deployer key
/// 6. Returns proof hash, tx hashes, and verification status
///
/// The customer never needs a Starknet key or wallet.
#[derive(Deserialize)]
struct AttestRequest {
    /// Model name or hex ID
    model_id: String,
    /// Input tensor (flat f32 array, must match model input_shape)
    input: Vec<f32>,
    /// Use GPU acceleration (default: true)
    #[serde(default = "default_true")]
    gpu: bool,
    /// Submit on-chain after proving (default: true)
    #[serde(default = "default_true")]
    submit_onchain: bool,
}

#[derive(Serialize)]
struct AttestResponse {
    /// Proof identifier
    proof_id: String,
    /// IO commitment: Poseidon(inputs || outputs)
    io_commitment: String,
    /// Weight commitment: Poseidon Merkle root of all weight matrices
    weight_commitment: String,
    /// Number of proven layers (MatMul + activation + norm)
    num_proven_layers: usize,
    /// Total proof generation time in milliseconds
    prove_time_ms: u64,
    /// Calldata statistics
    calldata_felts: usize,
    /// Estimated on-chain gas
    estimated_gas: u64,
    /// On-chain verification status
    onchain: OnChainStatus,
}

#[derive(Serialize)]
struct OnChainStatus {
    /// Whether proof was submitted on-chain
    submitted: bool,
    /// Transaction hashes (one per streaming step, or single for recursive)
    tx_hashes: Vec<String>,
    /// Network (e.g., "starknet-sepolia")
    network: String,
    /// Verifier contract address
    contract: String,
    /// Whether on-chain verification passed
    verified: bool,
    /// Explorer URL for the first/main transaction
    explorer_url: Option<String>,
    /// Error message if submission failed
    error: Option<String>,
}

async fn attest(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AttestRequest>,
) -> Result<Json<AttestResponse>, (StatusCode, Json<ErrorResponse>)> {
    let models = state.models.read().await;
    let model = resolve_model_id(&models, &req.model_id).ok_or_else(|| {
        let available: Vec<String> = models.values().map(|m| m.name.clone()).collect();
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model '{}' not found. Available: [{}]", req.model_id, available.join(", ")),
            }),
        )
    })?;

    let (in_rows, in_cols) = model.input_shape;
    let expected = in_rows * in_cols;
    if req.input.len() != expected {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Input has {} values, model expects {} ({in_rows}x{in_cols})", req.input.len(), expected),
            }),
        ));
    }

    // Quantize input
    let (quantized, _) = quantize_tensor(&req.input, QuantStrategy::Symmetric8);
    let mut input_matrix = M31Matrix::new(in_rows, in_cols);
    for (i, &v) in quantized.iter().enumerate().take(expected) {
        input_matrix.data[i] = v;
    }

    let graph = Arc::clone(&model.graph);
    let weights = Arc::clone(&model.weights);
    let model_weight_commitment = model.weight_commitment.clone();
    let model_name = model.name.clone();
    let model_id_str = model.model_id.clone();
    let model_id_for_artifact = model_id_str.clone();
    drop(models);

    let prove_start = Instant::now();

    // Run full attestation pipeline: GKR prove → streaming calldata
    let model_id_fe = starknet_ff::FieldElement::from_hex_be(&model_id_str)
        .unwrap_or(starknet_ff::FieldElement::from(0x9u64));

    let attestation = tokio::task::spawn_blocking(move || {
        stwo_ml::starknet::prove_full_attestation(&*graph, &input_matrix, &*weights, model_id_fe)
    })
    .await
    .map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: format!("Proof thread panicked: {e}") }),
    ))?
    .map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: format!("Attestation failed: {e}") }),
    ))?;

    let io_commitment = attestation.io_commitment.clone();
    let proof_hash = io_commitment.clone();
    let num_layers = attestation.num_layers;
    let prove_elapsed = attestation.prove_time_ms;
    let calldata_felts = attestation.streaming_calldata.session_metadata.total_felts;
    let num_streaming_batches = attestation.streaming_calldata.stream_batches.len();
    let estimated_gas = (calldata_felts as u64) * 100 + 500_000; // rough estimate

    // Store proof
    let proof_id = Uuid::new_v4().to_string();
    {
        let stored = StoredProof {
            proof_hash: proof_hash.clone(),
            model_id: model_id_str,
            io_commitment: io_commitment.clone(),
            weight_commitment: model_weight_commitment.clone(),
            num_proven_layers: num_layers,
            prove_time_ms: prove_elapsed,
            calldata_size: calldata_felts,
            created_at_epoch_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        state.proofs.write().await.insert(proof_hash.clone(), stored);
    }

    // On-chain submission
    let onchain = if req.submit_onchain {
        // Check if we have Starknet credentials
        let starknet_key = std::env::var("STARKNET_PRIVATE_KEY").ok();
        let starknet_account = std::env::var("STARKNET_ACCOUNT_ADDRESS").ok();
        let contract = std::env::var("CONTRACT_ADDRESS")
            .or_else(|_| std::env::var("OBELYSK_CONTRACT"))
            .unwrap_or_else(|_| "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005".to_string());

        if starknet_key.is_some() && starknet_account.is_some() {
            // Build streaming proof artifact for register_and_submit.mjs
            let streaming = &attestation.streaming_calldata;
            let batch_json: Vec<serde_json::Value> = streaming.stream_batches.iter().map(|b| {
                serde_json::json!({
                    "batch_idx": b.batch_idx,
                    "num_layers": b.num_layers,
                    "calldata": b.calldata,
                })
            }).collect();
            let output_chunks_json: Vec<serde_json::Value> = streaming.output_mle_chunks.iter().map(|c| {
                serde_json::json!({
                    "chunk_offset": c.chunk_offset,
                    "chunk_len": c.chunk_len,
                    "is_last": c.is_last,
                    "calldata": c.calldata,
                })
            }).collect();
            let input_chunks_json: Vec<serde_json::Value> = streaming.input_mle_chunks.iter().map(|c| {
                serde_json::json!({
                    "chunk_offset": c.chunk_offset,
                    "chunk_len": c.chunk_len,
                    "is_last": c.is_last,
                    "calldata": c.calldata,
                })
            }).collect();

            let artifact = serde_json::json!({
                "format": "ml_gkr",
                "model_id": model_id_for_artifact,
                "weight_commitments": attestation.weight_commitments,
                "io_commitment": attestation.io_commitment,
                "verify_calldata": {
                    "schema_version": 3,
                    "entrypoint": "verify_gkr_stream",
                    "mode": "streaming",
                    "model_id": model_id_for_artifact,
                    "total_felts": streaming.session_metadata.total_felts,
                    "circuit_depth": attestation.circuit_depth,
                    "num_layers": attestation.num_layers,
                    "layer_tags": streaming.session_metadata.layer_tags,
                    "init_calldata": streaming.init_calldata,
                    "output_mle_chunks": output_chunks_json,
                    "stream_batches": batch_json,
                    "weight_binding_chunks": streaming.weight_binding_chunks.iter().map(|c| {
                        serde_json::json!({
                            "chunk_idx": c.chunk_idx,
                            "is_last": c.is_last,
                            "entrypoint": c.entrypoint,
                            "calldata": c.calldata,
                        })
                    }).collect::<Vec<serde_json::Value>>(),
                    "input_mle_chunks": input_chunks_json,
                    "finalize_calldata": streaming.finalize_calldata,
                    "chunks": streaming.upload_chunks,
                }
            });

            // Write artifact to temp file
            let tmp_path = format!("/tmp/attest-{}.json", proof_id);
            if let Err(e) = std::fs::write(&tmp_path, serde_json::to_string(&artifact).unwrap_or_default()) {
                return Ok(Json(AttestResponse {
                    proof_id, io_commitment, weight_commitment: model_weight_commitment,
                    num_proven_layers: num_layers, prove_time_ms: prove_elapsed,
                    calldata_felts, estimated_gas,
                    onchain: OnChainStatus {
                        submitted: false, tx_hashes: vec![], network: "starknet-sepolia".into(),
                        contract, verified: false, explorer_url: None,
                        error: Some(format!("Failed to write proof artifact: {e}")),
                    },
                }));
            }

            // Find register_and_submit.mjs relative to the binary
            let exe = std::env::current_exe().unwrap_or_default();
            let scripts_dir = exe.parent().unwrap_or(std::path::Path::new("."))
                .join("../../../scripts/pipeline");
            let submit_script = if scripts_dir.join("register_and_submit.mjs").exists() {
                scripts_dir.join("register_and_submit.mjs")
            } else {
                // Fallback: check working directory
                std::path::PathBuf::from("scripts/pipeline/register_and_submit.mjs")
            };

            let rpc_url = std::env::var("STARKNET_RPC")
                .unwrap_or_else(|_| "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_7/demo".into());

            eprintln!("[attest] Submitting on-chain via {}", submit_script.display());

            // Call register_and_submit.mjs
            // Find node binary — check common locations
            let node_bin = [
                "/home/ubuntu/.nvm/versions/node/v20.20.2/bin/node",
                "/usr/local/bin/node",
                "/usr/bin/node",
                "node",
            ].iter().find(|p| std::path::Path::new(p).exists())
                .unwrap_or(&"node");

            match tokio::process::Command::new(node_bin)
                .arg(submit_script.to_str().unwrap_or("register_and_submit.mjs"))
                .arg(&tmp_path)
                .arg("--skip-register") // model should already be registered
                .env("STARKNET_RPC", &rpc_url)
                .env("STARKNET_ACCOUNT", starknet_account.as_ref().unwrap())
                .env("STARKNET_PRIVATE_KEY", starknet_key.as_ref().unwrap())
                .env("CONTRACT_ADDRESS", &contract)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output()
                .await
            {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);

                    // Parse TX hashes from output
                    let tx_hashes: Vec<String> = stdout.lines()
                        .chain(stderr.lines())
                        .filter(|l| l.contains("TX:") || l.contains("0x"))
                        .filter_map(|l| {
                            l.find("0x").map(|start| {
                                let hash = &l[start..];
                                hash.split_whitespace().next().unwrap_or(hash).to_string()
                            })
                        })
                        .filter(|h| h.len() > 10)
                        .collect();

                    let submitted = output.status.success();
                    let first_tx = tx_hashes.first().cloned();

                    OnChainStatus {
                        submitted,
                        tx_hashes,
                        network: "starknet-sepolia".to_string(),
                        contract: contract.clone(),
                        verified: submitted,
                        explorer_url: first_tx.map(|h| format!("https://sepolia.starkscan.co/tx/{h}")),
                        error: if !submitted { Some(stderr.to_string()) } else { None },
                    }
                }
                Err(e) => OnChainStatus {
                    submitted: false,
                    tx_hashes: vec![],
                    network: "starknet-sepolia".to_string(),
                    contract: contract.clone(),
                    verified: false,
                    explorer_url: None,
                    error: Some(format!("Failed to spawn submission: {e}")),
                },
            }
        } else {
            OnChainStatus {
                submitted: false,
                tx_hashes: vec![],
                network: "starknet-sepolia".to_string(),
                contract,
                verified: false,
                explorer_url: None,
                error: Some("No STARKNET_PRIVATE_KEY configured on prover".to_string()),
            }
        }
    } else {
        OnChainStatus {
            submitted: false,
            tx_hashes: vec![],
            network: "starknet-sepolia".to_string(),
            contract: String::new(),
            verified: false,
            explorer_url: None,
            error: None,
        }
    };

    Ok(Json(AttestResponse {
        proof_id,
        io_commitment,
        weight_commitment: model_weight_commitment,
        num_proven_layers: num_layers,
        prove_time_ms: prove_elapsed,
        calldata_felts,
        estimated_gas,
        onchain,
    }))
}

/// List all proven inferences — GET /api/v1/proofs
///
/// Returns all stored proofs, most recent first.
async fn list_proofs(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<StoredProof>> {
    let proofs = state.proofs.read().await;
    let mut list: Vec<StoredProof> = proofs.values().cloned().collect();
    list.sort_by(|a, b| b.created_at_epoch_ms.cmp(&a.created_at_epoch_ms));
    Json(list)
}

// =============================================================================
// Privacy Handlers
// =============================================================================

async fn submit_privacy_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PrivacyBatchRequest>,
) -> Result<(StatusCode, Json<PrivacyBatchSubmitResponse>), (StatusCode, Json<ErrorResponse>)> {
    if req.deposits.is_empty() && req.withdrawals.is_empty() && req.spends.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Batch must contain at least one transaction".to_string(),
            }),
        ));
    }

    let job_id = Uuid::new_v4().to_string();
    let job = PrivacyJob {
        job_id: job_id.clone(),
        status: JobStatus::Queued,
        started_at: Instant::now(),
        completed_at: None,
        error: None,
        result: None,
        submit_context: None,
    };

    state.privacy_jobs.write().await.insert(job_id.clone(), job);

    let resp = PrivacyBatchSubmitResponse {
        job_id: job_id.clone(),
        status: JobStatus::Queued,
    };

    let state_clone = state.clone();
    let jid = job_id.clone();
    tokio::task::spawn(async move {
        {
            let mut jobs = state_clone.privacy_jobs.write().await;
            if let Some(j) = jobs.get_mut(&jid) {
                j.status = JobStatus::Proving;
            }
        }

        let prove_start = Instant::now();

        let result =
            tokio::task::spawn_blocking(move || {
                use stwo::core::fields::m31::BaseField;
                use stwo_ml::circuits::batch::{prove_privacy_batch, PrivacyBatch};
                use stwo_ml::circuits::deposit::DepositWitness;
                use stwo_ml::circuits::spend::{InputNoteWitness, OutputNoteWitness, SpendWitness};
                use stwo_ml::circuits::withdraw::WithdrawWitness;
                use stwo_ml::crypto::commitment::Note;
                use stwo_ml::crypto::merkle_m31::MerklePath;
                use stwo_ml::crypto::poseidon2_m31::RATE;

                let m31 = |v: u32| BaseField::from_u32_unchecked(v);
                let m31_4 = |arr: [u32; 4]| [m31(arr[0]), m31(arr[1]), m31(arr[2]), m31(arr[3])];
                let m31_8 = |arr: [u32; 8]| -> [BaseField; 8] {
                    [
                        m31(arr[0]),
                        m31(arr[1]),
                        m31(arr[2]),
                        m31(arr[3]),
                        m31(arr[4]),
                        m31(arr[5]),
                        m31(arr[6]),
                        m31(arr[7]),
                    ]
                };

                // Helper: generate random blinding (fallible, rejection-sampled).
                let random_blinding = || -> Result<[BaseField; 4], String> {
                    const P: u32 = (1u32 << 31) - 1; // M31 prime
                    let mut result = [m31(0); 4];
                    for elem in result.iter_mut() {
                        loop {
                            let mut buf = [0u8; 4];
                            getrandom::getrandom(&mut buf)
                                .map_err(|e| format!("entropy source unavailable: {e}"))?;
                            let v = u32::from_le_bytes(buf) >> 1; // 31 bits, uniform in [0, 2^31)
                            if v < P {
                                *elem = m31(v);
                                break;
                            }
                            // v == P (2^31 - 1): reject and retry (~0% probability)
                        }
                    }
                    Ok(result)
                };

                // Max amount that fits in two M31 limbs
                const MAX_AMOUNT: u64 = ((1u64 << 31) - 1) + ((1u64 << 31) - 1) * (1u64 << 31);

                // Build deposit witnesses
                let deposits: Vec<DepositWitness> = {
                    let mut v = Vec::with_capacity(req.deposits.len());
                    for d in &req.deposits {
                        if d.amount > MAX_AMOUNT {
                            return Err(format!(
                                "deposit amount {} exceeds max {}",
                                d.amount, MAX_AMOUNT
                            ));
                        }
                        let pubkey = m31_4(d.recipient_pubkey);
                        let amount_lo = m31((d.amount & 0x7FFF_FFFF) as u32);
                        let amount_hi = m31((d.amount >> 31) as u32);
                        let blinding = random_blinding()?;
                        let note = Note {
                            owner_pubkey: pubkey,
                            amount_lo,
                            amount_hi,
                            asset_id: m31(d.asset_id),
                            blinding,
                        };
                        v.push(DepositWitness {
                            note,
                            amount: d.amount,
                            asset_id: m31(d.asset_id),
                        });
                    }
                    v
                };

                // Build withdraw witnesses and recipient bindings.
                let mut payout_recipients: Vec<String> = Vec::with_capacity(req.withdrawals.len());
                let mut credit_recipients: Vec<String> = Vec::with_capacity(req.withdrawals.len());
                let mut withdrawals: Vec<WithdrawWitness> =
                    Vec::with_capacity(req.withdrawals.len());
                for (idx, w) in req.withdrawals.iter().enumerate() {
                    let payout = w
                        .payout_recipient
                        .clone()
                        .ok_or_else(|| format!("withdrawals[{idx}] missing payout_recipient"))?;
                    let credit = w.credit_recipient.clone().unwrap_or_else(|| payout.clone());
                    let binding = compute_withdrawal_binding_digest(
                        &payout,
                        &credit,
                        w.note.asset_id as u64,
                        w.note.amount_lo as u64,
                        w.note.amount_hi as u64,
                        idx as u32,
                    )
                    .map_err(|e| format!("withdrawals[{idx}] invalid binding inputs: {e}"))?;

                    let note = Note {
                        owner_pubkey: m31_4(w.note.pub_key),
                        amount_lo: m31(w.note.amount_lo),
                        amount_hi: m31(w.note.amount_hi),
                        asset_id: m31(w.note.asset_id),
                        blinding: m31_4(w.note.blinding),
                    };
                    let siblings: Vec<[BaseField; RATE]> =
                        w.merkle_siblings.iter().map(|s| m31_8(*s)).collect();
                    withdrawals.push(WithdrawWitness {
                        note,
                        spending_key: m31_4(w.spending_key),
                        merkle_path: MerklePath {
                            siblings,
                            index: w.merkle_index,
                        },
                        merkle_root: m31_8(w.merkle_root),
                        withdrawal_binding: binding,
                    });
                    payout_recipients.push(payout);
                    credit_recipients.push(credit);
                }

                // Build spend witnesses
                let spends: Vec<SpendWitness> = req
                    .spends
                    .iter()
                    .map(|s| {
                        let inputs: [InputNoteWitness; 2] = [
                            {
                                let inp = &s.inputs[0];
                                let note = Note {
                                    owner_pubkey: m31_4(inp.note.pub_key),
                                    amount_lo: m31(inp.note.amount_lo),
                                    amount_hi: m31(inp.note.amount_hi),
                                    asset_id: m31(inp.note.asset_id),
                                    blinding: m31_4(inp.note.blinding),
                                };
                                let siblings: Vec<[BaseField; RATE]> =
                                    inp.merkle_siblings.iter().map(|sib| m31_8(*sib)).collect();
                                InputNoteWitness {
                                    note,
                                    spending_key: m31_4(inp.spending_key),
                                    merkle_path: MerklePath {
                                        siblings,
                                        index: inp.merkle_index,
                                    },
                                }
                            },
                            {
                                let inp = &s.inputs[1];
                                let note = Note {
                                    owner_pubkey: m31_4(inp.note.pub_key),
                                    amount_lo: m31(inp.note.amount_lo),
                                    amount_hi: m31(inp.note.amount_hi),
                                    asset_id: m31(inp.note.asset_id),
                                    blinding: m31_4(inp.note.blinding),
                                };
                                let siblings: Vec<[BaseField; RATE]> =
                                    inp.merkle_siblings.iter().map(|sib| m31_8(*sib)).collect();
                                InputNoteWitness {
                                    note,
                                    spending_key: m31_4(inp.spending_key),
                                    merkle_path: MerklePath {
                                        siblings,
                                        index: inp.merkle_index,
                                    },
                                }
                            },
                        ];
                        let outputs: [OutputNoteWitness; 2] = [
                            {
                                let out = &s.outputs[0];
                                OutputNoteWitness {
                                    note: Note {
                                        owner_pubkey: m31_4(out.recipient_pubkey),
                                        amount_lo: m31(out.amount_lo),
                                        amount_hi: m31(out.amount_hi),
                                        asset_id: m31(out.asset_id),
                                        blinding: m31_4(out.blinding),
                                    },
                                }
                            },
                            {
                                let out = &s.outputs[1];
                                OutputNoteWitness {
                                    note: Note {
                                        owner_pubkey: m31_4(out.recipient_pubkey),
                                        amount_lo: m31(out.amount_lo),
                                        amount_hi: m31(out.amount_hi),
                                        asset_id: m31(out.asset_id),
                                        blinding: m31_4(out.blinding),
                                    },
                                }
                            },
                        ];
                        SpendWitness {
                            inputs,
                            outputs,
                            merkle_root: m31_8(s.merkle_root),
                        }
                    })
                    .collect();

                let batch = PrivacyBatch {
                    deposits,
                    withdrawals,
                    spends,
                };

                let proof = prove_privacy_batch(&batch).map_err(|e| format!("{e}"))?;
                Ok((proof, payout_recipients, credit_recipients))
            })
            .await;

        let prove_elapsed = prove_start.elapsed();
        let mut jobs = state_clone.privacy_jobs.write().await;

        match result {
            Ok(Ok((batch_proof, payout_recipients, credit_recipients))) => {
                let pi = &batch_proof.public_inputs;

                // Build deposit results
                let deposit_results: Vec<PrivacyDepositResult> = pi
                    .deposits
                    .iter()
                    .map(|d| {
                        let mut hex = String::with_capacity(2 + 8 * 8);
                        hex.push_str("0x");
                        for &e in &d.commitment {
                            hex.push_str(&format!("{:08x}", e.0));
                        }
                        PrivacyDepositResult {
                            commitment: hex,
                            amount: d.amount,
                            asset_id: d.asset_id.0,
                        }
                    })
                    .collect();

                // Build withdraw results
                let withdraw_results: Vec<PrivacyWithdrawResult> = pi
                    .withdrawals
                    .iter()
                    .map(|w| {
                        let mut hex = String::with_capacity(2 + 8 * 8);
                        hex.push_str("0x");
                        for &e in &w.nullifier {
                            hex.push_str(&format!("{:08x}", e.0));
                        }
                        let amount = (w.amount_lo.0 as u64) | ((w.amount_hi.0 as u64) << 31);
                        PrivacyWithdrawResult {
                            nullifier: hex,
                            amount,
                            asset_id: w.asset_id.0,
                        }
                    })
                    .collect();

                // Build spend results
                let spend_results: Vec<PrivacySpendResult> = pi
                    .spends
                    .iter()
                    .map(|s| {
                        let nul_hex = |nul: &[M31]| -> String {
                            let mut h = String::with_capacity(2 + 8 * 8);
                            h.push_str("0x");
                            for &e in nul {
                                h.push_str(&format!("{:08x}", e.0));
                            }
                            h
                        };
                        PrivacySpendResult {
                            nullifiers: [nul_hex(&s.nullifiers[0]), nul_hex(&s.nullifiers[1])],
                            output_commitments: [
                                nul_hex(&s.output_commitments[0]),
                                nul_hex(&s.output_commitments[1]),
                            ],
                        }
                    })
                    .collect();

                // Compute VM31 batch hash using the exact Cairo-compatible encoding.
                let pi_hash = match hash_batch_public_inputs_for_cairo(pi) {
                    Ok(hash) => hash,
                    Err(e) => {
                        if let Some(j) = jobs.get_mut(&jid) {
                            j.status = JobStatus::Failed;
                            j.error = Some(format!("public input hashing failed: {e}"));
                            j.completed_at = Some(Instant::now());
                        }
                        return;
                    }
                };
                let mut pi_hash_hex = String::with_capacity(2 + 8 * 8);
                pi_hash_hex.push_str("0x");
                for &e in &pi_hash {
                    pi_hash_hex.push_str(&format!("{:08x}", e.0));
                }

                // Serialize STARK proof as felt252 calldata for on-chain verification
                let stark_proof_felts = serialize_proof(&batch_proof.stark_proof);
                let stark_proof_calldata: Vec<String> = stark_proof_felts
                    .iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect();

                let submit_context = PrivacySubmitContext {
                    public_inputs: pi.clone(),
                    payout_recipients: payout_recipients.clone(),
                    credit_recipients: credit_recipients.clone(),
                };

                let payload = PrivacyBatchResultPayload {
                    num_deposits: pi.deposits.len(),
                    num_withdrawals: pi.withdrawals.len(),
                    num_spends: pi.spends.len(),
                    proof_hash: None,
                    proof_hash_status: "pending_verifier_canonical_hash".to_string(),
                    public_inputs_hash: pi_hash_hex,
                    prove_time_ms: prove_elapsed.as_millis() as u64,
                    deposits: deposit_results,
                    withdrawals: withdraw_results,
                    spends: spend_results,
                    calldata_ready: false,
                    calldata: Vec::new(),
                    payout_recipients,
                    credit_recipients,
                    stark_proof_calldata,
                };

                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Completed;
                    j.completed_at = Some(Instant::now());
                    j.result = Some(payload);
                    j.submit_context = Some(submit_context);
                }
            }
            Ok(Err(e)) => {
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("{e}"));
                    j.completed_at = Some(Instant::now());
                }
            }
            Err(e) => {
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("Task panicked: {e}"));
                    j.completed_at = Some(Instant::now());
                }
            }
        }
    });

    Ok((StatusCode::ACCEPTED, Json(resp)))
}

async fn get_privacy_batch_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<PrivacyBatchStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let jobs = state.privacy_jobs.read().await;
    let job = jobs.get(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Privacy job '{job_id}' not found"),
            }),
        )
    })?;

    Ok(Json(PrivacyBatchStatusResponse {
        job_id: job.job_id.clone(),
        status: job.status,
        elapsed_secs: job.started_at.elapsed().as_secs_f64(),
    }))
}

async fn get_privacy_batch_result(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<PrivacyBatchResultPayload>, (StatusCode, Json<ErrorResponse>)> {
    let jobs = state.privacy_jobs.read().await;
    let job = jobs.get(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Privacy job '{job_id}' not found"),
            }),
        )
    })?;

    match job.status {
        JobStatus::Completed => {
            let result = job.result.as_ref().ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Privacy job completed but result is missing (internal error)"
                            .to_string(),
                    }),
                )
            })?;
            Ok(Json(result.clone()))
        }
        JobStatus::Failed => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: job
                    .error
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            }),
        )),
        _ => Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!("Privacy job is still {:?}", job.status),
            }),
        )),
    }
}

async fn build_privacy_submit_calldata(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Json(req): Json<PrivacyBatchSubmitCalldataRequest>,
) -> Result<Json<PrivacyBatchSubmitCalldataResponse>, (StatusCode, Json<ErrorResponse>)> {
    let proof_hash = normalize_canonical_proof_hash(&req.proof_hash).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("invalid proof_hash: {e}"),
            }),
        )
    })?;

    let mut jobs = state.privacy_jobs.write().await;
    let job = jobs.get_mut(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Privacy job '{job_id}' not found"),
            }),
        )
    })?;

    match job.status {
        JobStatus::Completed => {
            let submit_ctx = job.submit_context.clone().ok_or_else(|| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Privacy job completed but submit context is missing".to_string(),
                    }),
                )
            })?;
            let public_inputs_hash = job
                .result
                .as_ref()
                .map(|r| r.public_inputs_hash.clone())
                .ok_or_else(|| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "Privacy job completed but result is missing".to_string(),
                        }),
                    )
                })?;

            let recipients = WithdrawalRecipients::new(
                submit_ctx.payout_recipients.clone(),
                submit_ctx.credit_recipients.clone(),
            );
            let calldata = build_submit_batch_proof_calldata(
                &submit_ctx.public_inputs,
                &proof_hash,
                &recipients,
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("failed to build submit calldata: {e}"),
                    }),
                )
            })?;

            if let Some(result) = job.result.as_mut() {
                result.proof_hash = Some(proof_hash.clone());
                result.proof_hash_status = "canonical_ready".to_string();
                result.calldata_ready = true;
                result.calldata = calldata.clone();
            }

            Ok(Json(PrivacyBatchSubmitCalldataResponse {
                job_id: job_id.clone(),
                proof_hash,
                public_inputs_hash,
                calldata,
            }))
        }
        JobStatus::Failed => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: job
                    .error
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            }),
        )),
        _ => Err((
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!("Privacy job is still {:?}", job.status),
            }),
        )),
    }
}

// =============================================================================
// Audit endpoints (server-audit)
// =============================================================================

#[cfg(feature = "server-audit")]
async fn submit_audit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AuditTriggerRequest>,
) -> Result<(StatusCode, Json<AuditSubmitResponse>), (StatusCode, Json<ErrorResponse>)> {
    // Verify model is loaded and has an audit capture hook
    let models = state.models.read().await;
    let model = models.get(&req.model_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Model '{}' not found", req.model_id),
            }),
        )
    })?;

    let capture_hook = model.capture_hook.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Audit capture not enabled for this model. Set AUDIT_LOG_DIR.".to_string(),
            }),
        )
    })?;

    if capture_hook.entry_count() == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No inferences captured yet. Run some proofs first.".to_string(),
            }),
        ));
    }

    let graph = Arc::clone(&model.graph);
    let weights = Arc::clone(&model.weights);
    let model_id = model.model_id.clone();
    let weight_commitment = model.weight_commitment.clone();
    let num_layers = model.num_layers;

    // Flush pending captures before audit
    capture_hook.flush();

    // Resolve the log directory from AUDIT_LOG_DIR
    let audit_dir = std::env::var("AUDIT_LOG_DIR").map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "AUDIT_LOG_DIR not set".to_string(),
            }),
        )
    })?;
    // Sanitize model_id: only allow alphanumeric, underscore, hyphen, dot
    let sanitized_model_id: String = req.model_id.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .collect();
    if sanitized_model_id.is_empty() || sanitized_model_id.contains("..") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "invalid model_id".to_string() }),
        ));
    }
    let log_dir = std::path::PathBuf::from(&audit_dir).join(&sanitized_model_id);

    drop(models);

    let job_id = Uuid::new_v4().to_string();
    let job = AuditJob {
        job_id: job_id.clone(),
        model_id: model_id.clone(),
        status: JobStatus::Queued,
        started_at: Instant::now(),
        completed_at: None,
        error: None,
        report: None,
    };
    state.audit_jobs.write().await.insert(job_id.clone(), job);

    let resp = AuditSubmitResponse {
        job_id: job_id.clone(),
        status: JobStatus::Queued,
    };

    let mode = req.mode.clone();
    let max_inferences = req.max_inferences;
    let weight_binding = req.weight_binding.clone();
    let state_clone = state.clone();
    let jid = job_id.clone();

    tokio::task::spawn(async move {
        {
            let mut jobs = state_clone.audit_jobs.write().await;
            if let Some(j) = jobs.get_mut(&jid) {
                j.status = JobStatus::Proving;
            }
        }

        let result = tokio::task::spawn_blocking(move || {
            // Load the inference log
            let log = stwo_ml::audit::log::InferenceLog::load(&log_dir)
                .map_err(|e| format!("Failed to load audit log: {e}"))?;

            let request = stwo_ml::audit::types::AuditRequest {
                start_ns: 0,
                end_ns: u64::MAX,
                model_id: model_id.clone(),
                mode,
                evaluate_semantics: false,
                max_inferences,
                gpu_device: None,
                weight_binding,
            };

            let model_info = stwo_ml::audit::types::ModelInfo {
                model_id: model_id.clone(),
                name: model_id.clone(),
                architecture: "transformer".to_string(),
                parameters: format!("{} layers", num_layers),
                layers: num_layers as u32,
                weight_commitment: weight_commitment.clone(),
            };

            let config = stwo_ml::audit::orchestrator::AuditPipelineConfig {
                request,
                model_info,
                evaluate_semantics: false,
                prove_evaluations: false,
                privacy_tier: "none".to_string(),
                owner_pubkey: Vec::new(),
                submit_config: None,
                billing: None,
                model_dir: None,
            };

            let pipeline_result =
                stwo_ml::audit::orchestrator::run_audit(&log, &graph, &weights, &config, None, None)
                    .map_err(|e| format!("Audit pipeline failed: {e}"))?;

            serde_json::to_value(&pipeline_result.report)
                .map_err(|e| format!("Failed to serialize report: {e}"))
        })
        .await;

        let mut jobs = state_clone.audit_jobs.write().await;
        match result {
            Ok(Ok(report_json)) => {
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Completed;
                    j.completed_at = Some(Instant::now());
                    j.report = Some(report_json);
                }
            }
            Ok(Err(e)) => {
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Failed;
                    j.error = Some(e);
                    j.completed_at = Some(Instant::now());
                }
            }
            Err(e) => {
                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Failed;
                    j.error = Some(format!("Task panicked: {e}"));
                    j.completed_at = Some(Instant::now());
                }
            }
        }
    });

    Ok((StatusCode::ACCEPTED, Json(resp)))
}

#[cfg(feature = "server-audit")]
async fn get_audit_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<AuditStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let jobs = state.audit_jobs.read().await;
    let job = jobs.get(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Audit job '{job_id}' not found"),
            }),
        )
    })?;

    Ok(Json(AuditStatusResponse {
        job_id: job.job_id.clone(),
        status: job.status,
        elapsed_secs: job.started_at.elapsed().as_secs_f64(),
        report: job.report.clone(),
        error: job.error.clone(),
    }))
}

// =============================================================================
// Dashboard + WebSocket
// =============================================================================

// =============================================================================
// Queue stats endpoint (multi-query)
// =============================================================================

#[cfg(feature = "multi-query")]
async fn get_queue_stats(
    State(state): State<Arc<AppState>>,
) -> Json<stwo_ml::gpu_scheduler::QueueStatsSnapshot> {
    Json(state.scheduler.stats.snapshot())
}

// =============================================================================
// Per-job WebSocket endpoint (multi-query + proof-stream)
// =============================================================================

#[cfg(all(feature = "multi-query", feature = "proof-stream"))]
async fn ws_job_handler(
    ws: WebSocketUpgrade,
    Path(job_id): Path<String>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<axum::response::Response, StatusCode> {
    if let Some(ref expected) = state.api_key {
        match params.get("token") {
            Some(t) if constant_time_eq(t, expected) => {}
            _ => return Err(StatusCode::UNAUTHORIZED),
        }
    }
    let rx = state.ws_sink.subscribe();
    Ok(ws.on_upgrade(move |socket| async move {
        ws_job_client_loop(socket, rx, job_id).await;
    }).into_response())
}

#[cfg(all(feature = "multi-query", feature = "proof-stream"))]
async fn ws_job_client_loop(
    mut socket: WebSocket,
    mut rx: tokio::sync::broadcast::Receiver<String>,
    job_id: String,
) {
    use tokio::sync::broadcast::error::RecvError;

    // Wrap each event with the job_id for filtering
    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                Ok(json) => {
                    // Wrap the raw event with job_id tagging
                    let tagged = format!(r#"{{"job_id":"{}","event":{}}}"#, job_id, json);
                    if socket.send(Message::Text(tagged.into())).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Lagged(n)) => {
                    let warn = format!(
                        r#"{{"job_id":"{}","event":{{"Log":{{"level":"Warn","message":"lagged {} events"}}}}}}"#,
                        job_id, n
                    );
                    if socket.send(Message::Text(warn.into())).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Closed) => break,
            },
            msg = socket.recv() => match msg {
                Some(Ok(Message::Ping(p))) => {
                    let _ = socket.send(Message::Pong(p)).await;
                }
                None | Some(Ok(Message::Close(_))) => break,
                _ => {}
            }
        }
    }
}

async fn dashboard() -> impl IntoResponse {
    Html(include_str!("web_dashboard.html"))
}

#[cfg(feature = "proof-stream")]
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<axum::response::Response, StatusCode> {
    // Authenticate WebSocket via ?token= query param when API key is configured
    if let Some(ref expected) = state.api_key {
        match params.get("token") {
            Some(t) if constant_time_eq(t, expected) => {}
            _ => return Err(StatusCode::UNAUTHORIZED),
        }
    }
    let rx = state.ws_sink.subscribe();
    Ok(ws.on_upgrade(move |socket| async move {
        ws_client_loop(socket, rx).await;
    }).into_response())
}

#[cfg(feature = "proof-stream")]
async fn ws_client_loop(mut socket: WebSocket, mut rx: tokio::sync::broadcast::Receiver<String>) {
    use tokio::sync::broadcast::error::RecvError;
    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                Ok(json) => {
                    if socket.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Lagged(n)) => {
                    let warn = format!(r#"{{"Log":{{"level":"Warn","message":"lagged {} events"}}}}"#, n);
                    if socket.send(Message::Text(warn.into())).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Closed) => break,
            },
            msg = socket.recv() => match msg {
                Some(Ok(Message::Ping(p))) => {
                    let _ = socket.send(Message::Pong(p)).await;
                }
                None | Some(Ok(Message::Close(_))) => break,
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_canonical_proof_hash;

    #[test]
    fn test_normalize_canonical_proof_hash_accepts_hex() {
        assert_eq!(
            normalize_canonical_proof_hash("0x00AbCd").unwrap(),
            "0xabcd"
        );
        assert_eq!(normalize_canonical_proof_hash("1234").unwrap(), "0x1234");
    }

    #[test]
    fn test_normalize_canonical_proof_hash_rejects_invalid_values() {
        assert!(normalize_canonical_proof_hash("").is_err());
        assert!(normalize_canonical_proof_hash("0x").is_err());
        assert!(normalize_canonical_proof_hash("0x0").is_err());
        assert!(normalize_canonical_proof_hash("0xgg").is_err());
        assert!(normalize_canonical_proof_hash(
            "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1"
        )
        .is_err());
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() {
    // ── Proof policy ─────────────────────────────────────────────────
    // Apply the "standard" policy as env vars at startup, BEFORE any proving
    // threads spawn. This ensures all prove paths (aggregated, GKR, audit)
    // use consistent settings.
    //
    // Safety: set_var is called once in main() before tokio spawns threads.
    // Individual STWO_* env vars set by the operator take precedence (we only
    // set vars that aren't already set).
    let server_default_policy = stwo_ml::policy::PolicyConfig::standard();
    for (key, val) in [
        ("STWO_SKIP_RMS_SQ_PROOF", if server_default_policy.skip_rms_sq_proof { "1" } else { "0" }),
        ("STWO_ALLOW_MISSING_NORM_PROOF", if server_default_policy.allow_missing_norm_proof { "1" } else { "0" }),
        ("STWO_PIECEWISE_ACTIVATION", if server_default_policy.piecewise_activation { "1" } else { "0" }),
        ("STWO_ALLOW_LOGUP_ACTIVATION", if server_default_policy.allow_logup_activation { "1" } else { "0" }),
        ("STWO_AGGREGATED_FULL_BINDING", if server_default_policy.aggregated_full_binding { "1" } else { "0" }),
        ("STWO_SKIP_BATCH_TOKENS", if server_default_policy.skip_batch_tokens { "1" } else { "0" }),
        ("STWO_PURE_GKR_SKIP_UNIFIED_STARK", if server_default_policy.skip_unified_stark { "1" } else { "0" }),
    ] {
        if std::env::var(key).is_err() {
            // Safety: called once in main() before any threads are spawned.
            unsafe { std::env::set_var(key, val); }
        }
    }
    eprintln!(
        "  Policy: {}",
        stwo_ml::policy::summary_line(&server_default_policy),
    );

    // ── Worker mode ────────────────────────────────────────────────
    // If COORDINATOR_URL is set, start in worker mode:
    // register with coordinator, send heartbeats, accept jobs via WebSocket.
    let worker_config = stwo_ml::worker::WorkerConfig::from_env();
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let worker_handle = if let Some(ref wc) = worker_config {
        match stwo_ml::worker::start_worker(wc.clone(), shutdown_rx.clone()).await {
            Ok(handle) => {
                eprintln!("  Worker mode: ACTIVE (id: {})", handle.worker_id);
                eprintln!("  Coordinator: {}", wc.coordinator_url);
                eprintln!("  Wallet: {}", if wc.wallet_address.is_empty() { "none" } else { &wc.wallet_address });
                Some(handle)
            }
            Err(e) => {
                eprintln!("  Worker registration failed: {e}");
                eprintln!("  Running in standalone mode");
                None
            }
        }
    } else {
        None
    };

    #[cfg(feature = "proof-stream")]
    let ws_sink = proof_stream::WsBroadcastSink::new(1024);
    let validator_url = std::env::var("VALIDATOR_URL").ok();

    // Initialize GPU scheduler (multi-query mode)
    #[cfg(feature = "multi-query")]
    let scheduler = {
        let max_per_gpu: usize = std::env::var("MAX_CONCURRENT_PER_GPU")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let max_queue: usize = std::env::var("MAX_QUEUE_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let device_ordinals: Vec<usize> = std::env::var("GPU_DEVICES")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|v| v.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default();

        stwo_ml::gpu_scheduler::GpuScheduler::new(stwo_ml::gpu_scheduler::GpuSchedulerConfig {
            max_concurrent_per_gpu: max_per_gpu,
            max_queue_depth: max_queue,
            device_ordinals,
        })
    };

    // Security configuration
    let api_key = std::env::var("PROVE_SERVER_API_KEY").ok();
    let rate_limit_rpm: u32 = std::env::var("PROVE_SERVER_RATE_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let rate_limit_burst: u32 = std::env::var("PROVE_SERVER_BURST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let state = Arc::new(AppState {
        jobs: RwLock::new(HashMap::new()),
        privacy_jobs: RwLock::new(HashMap::new()),
        models: RwLock::new(HashMap::new()),
        proofs: RwLock::new(HashMap::new()),
        started_at: Instant::now(),
        api_key: api_key.clone(),
        rate_limiter: RateLimiter::new(rate_limit_rpm, rate_limit_burst),
        #[cfg(feature = "proof-stream")]
        ws_sink,
        validator_url,
        #[cfg(feature = "multi-query")]
        scheduler,
        #[cfg(feature = "multi-query")]
        weight_caches: RwLock::new(HashMap::new()),
        #[cfg(feature = "server-audit")]
        audit_jobs: RwLock::new(HashMap::new()),
    });

    // Authenticated + rate-limited API routes
    let api_routes = Router::new()
        .route("/api/v1/models", get(list_models).post(load_model))
        .route("/api/v1/models/hf", post(load_hf_model_handler))
        .route("/api/v1/models/:model_id", get(get_model))
        .route("/api/v1/prove", post(submit_prove))
        .route("/api/v1/prove/:job_id", get(get_prove_status))
        .route("/api/v1/prove/:job_id/result", get(get_prove_result))
        .route("/api/v1/infer", post(infer))
        .route("/api/v1/verify/:proof_hash", get(verify_proof))
        .route("/api/v1/proofs", get(list_proofs))
        .route("/api/v1/attest", post(attest))
        .route("/api/v1/privacy/batch", post(submit_privacy_batch))
        .route(
            "/api/v1/privacy/batch/:job_id",
            get(get_privacy_batch_status),
        )
        .route(
            "/api/v1/privacy/batch/{job_id}/result",
            get(get_privacy_batch_result),
        )
        .route(
            "/api/v1/privacy/batch/{job_id}/submit-calldata",
            post(build_privacy_submit_calldata),
        );

    #[allow(unused_mut)]
    let mut api_routes = api_routes;

    #[cfg(feature = "multi-query")]
    {
        api_routes = api_routes.route("/api/v1/queue", get(get_queue_stats));
    }

    #[cfg(feature = "server-audit")]
    {
        api_routes = api_routes
            .route("/api/v1/audit", post(submit_audit))
            .route("/api/v1/audit/:job_id", get(get_audit_status));
    }

    #[cfg(all(feature = "multi-query", feature = "proof-stream"))]
    {
        api_routes = api_routes.route("/api/v1/prove/:job_id/ws", get(ws_job_handler));
    }

    // Apply auth + rate-limit middleware to API routes
    let api_routes = api_routes
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // Public routes (no auth required): dashboard, health, WebSocket
    let public_routes = Router::new()
        .route("/", get(dashboard))
        .route("/health", get(health));

    #[allow(unused_mut)]
    let mut public_routes = public_routes;

    #[cfg(feature = "proof-stream")]
    {
        public_routes = public_routes.route("/ws", get(ws_handler));
    }

    // Merge authenticated API routes into the main router
    let router = public_routes.merge(api_routes);

    // Spawn periodic cleanup: rate limiter + completed jobs (prevent unbounded growth)
    {
        let state_cleanup = state.clone();
        tokio::spawn(async move {
            /// Completed jobs older than this are evicted.
            const JOB_TTL_SECS: u64 = 3600; // 1 hour
            /// Maximum jobs retained per map (hard cap).
            const MAX_JOBS: usize = 10_000;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                state_cleanup.rate_limiter.evict_stale().await;

                // Evict completed jobs older than TTL
                let now = Instant::now();
                let evict = |completed_at: &Option<Instant>| -> bool {
                    match completed_at {
                        Some(t) => now.duration_since(*t).as_secs() < JOB_TTL_SECS,
                        None => true, // keep in-progress jobs
                    }
                };

                {
                    let mut jobs = state_cleanup.jobs.write().await;
                    jobs.retain(|_, j| evict(&j.completed_at));
                    // Hard cap: if still over limit, drop oldest completed first
                    if jobs.len() > MAX_JOBS {
                        let mut to_remove: Vec<_> = jobs.iter()
                            .filter(|(_, j)| j.completed_at.is_some())
                            .map(|(k, j)| (k.clone(), j.started_at))
                            .collect();
                        to_remove.sort_by_key(|(_, t)| *t);
                        for (k, _) in to_remove.iter().take(jobs.len() - MAX_JOBS) {
                            jobs.remove(k);
                        }
                    }
                }
                {
                    let mut pj = state_cleanup.privacy_jobs.write().await;
                    pj.retain(|_, j| evict(&j.completed_at));
                }
                #[cfg(feature = "server-audit")]
                {
                    let mut aj = state_cleanup.audit_jobs.write().await;
                    aj.retain(|_, j| evict(&j.completed_at));
                }
            }
        });
    }

    // ── Worker job processor ─────────────────────────────────────────
    // Reads jobs from the coordinator WebSocket channel, runs them
    // through the local prover, and reports results back.
    if let Some(mut handle) = worker_handle {
        let worker_state = state.clone();
        let worker_id = handle.worker_id.clone();
        let worker_cfg = handle.config.clone();
        tokio::spawn(async move {
            eprintln!("  Worker job processor: listening for coordinator assignments");
            while let Some(job) = handle.job_rx.recv().await {
                eprintln!("  Worker: received job {} (model: {})", job.job_id, job.model_id);

                // Check if model is loaded
                let model_exists = worker_state.models.read().await.contains_key(&job.model_id);
                if !model_exists {
                    eprintln!("  Worker: model {} not loaded, rejecting job", job.model_id);
                    stwo_ml::worker::submit_job_result(
                        &worker_cfg, &worker_id, &job.job_id,
                        Err(format!("Model {} not loaded on this worker", job.model_id)),
                    ).await;
                    continue;
                }

                // Run proof in blocking task
                let state_prove = worker_state.clone();
                let job_id = job.job_id.clone();
                let model_id = job.model_id.clone();
                let input = job.input.clone();
                let gpu = job.gpu;
                let cfg = worker_cfg.clone();
                let wid = worker_id.clone();

                tokio::task::spawn_blocking(move || {
                    let rt = tokio::runtime::Handle::current();
                    let result = rt.block_on(async {
                        // Build input matrix
                        let models = state_prove.models.read().await;
                        let model = match models.get(&model_id) {
                            Some(m) => m,
                            None => {
                                stwo_ml::worker::submit_job_result(
                                    &cfg, &wid, &job_id,
                                    Err("Model disappeared during proving".into()),
                                ).await;
                                return;
                            }
                        };

                        let (rows, cols) = model.input_shape;
                        let q_input: Vec<M31> = input.iter()
                            .map(|&v| M31::from((v.abs() * 1000.0) as u32 % ((1u32 << 31) - 1)))
                            .collect();

                        let input_matrix = M31Matrix {
                            data: q_input,
                            rows,
                            cols,
                        };

                        let graph = model.graph.clone();
                        let weights = model.weights.clone();
                        drop(models);

                        let start = Instant::now();

                        // Run proof
                        match stwo_ml::starknet::prove_for_starknet_onchain(
                            &graph, &input_matrix, &weights,
                        ) {
                            Ok(proof) => {
                                let elapsed = start.elapsed().as_millis() as u64;
                                let io_str = format!("{:#x}", proof.io_commitment);
                                let calldata_size = proof.unified_calldata.len()
                                    + proof.matmul_calldata.iter().map(|c| c.len()).sum::<usize>();

                                stwo_ml::worker::submit_job_result(
                                    &cfg, &wid, &job_id,
                                    Ok(stwo_ml::worker::JobProofResult {
                                        proof_hash: io_str.clone(),
                                        io_commitment: io_str,
                                        weight_commitment: String::new(),
                                        prove_time_ms: elapsed,
                                        calldata_size,
                                    }),
                                ).await;
                            }
                            Err(e) => {
                                stwo_ml::worker::submit_job_result(
                                    &cfg, &wid, &job_id,
                                    Err(format!("Proof failed: {e}")),
                                ).await;
                            }
                        }
                    });
                });
            }
        });
    }

    let bind = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    eprintln!("prove-server listening on {bind}");
    eprintln!("  GPU: {}", detect_tee_capability().device_name);
    #[cfg(feature = "multi-query")]
    eprintln!(
        "  Scheduler: {} GPU(s), queue depth {}",
        state.scheduler.gpu_count(),
        std::env::var("MAX_QUEUE_DEPTH").unwrap_or_else(|_| "64".to_string()),
    );

    // CORS: configurable via PROVE_SERVER_CORS_ORIGINS, permissive in dev mode
    let cors = if let Ok(origins) = std::env::var("PROVE_SERVER_CORS_ORIGINS") {
        let origins: Vec<axum::http::HeaderValue> = origins
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([CONTENT_TYPE, AUTHORIZATION])
    } else {
        eprintln!("  WARNING: CORS permissive (set PROVE_SERVER_CORS_ORIGINS for production)");
        CorsLayer::permissive()
    };

    if api_key.is_some() {
        eprintln!("  Auth: API key required (PROVE_SERVER_API_KEY)");
    } else {
        eprintln!("  WARNING: No API key configured (set PROVE_SERVER_API_KEY for production)");
    }
    eprintln!("  Rate limit: {rate_limit_rpm} req/min, burst {rate_limit_burst}");
    if let Ok(dir) = std::env::var("PROVE_SERVER_MODEL_DIR") {
        eprintln!("  Model dir allowlist: {dir}");
    }

    // Body size limit: 50 MB default, configurable via PROVE_SERVER_MAX_BODY_MB.
    let max_body_mb: usize = std::env::var("PROVE_SERVER_MAX_BODY_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let app = router
        .layer(RequestBodyLimitLayer::new(max_body_mb * 1024 * 1024))
        .layer(cors)
        .with_state(state);
    eprintln!("  Body limit: {max_body_mb} MB");
    eprintln!("  Privacy: POST /api/v1/privacy/batch");
    eprintln!("  Privacy Submit: POST /api/v1/privacy/batch/{{job_id}}/submit-calldata");
    #[cfg(feature = "server-audit")]
    if let Ok(audit_dir) = std::env::var("AUDIT_LOG_DIR") {
        eprintln!("  Audit: enabled (AUDIT_LOG_DIR={audit_dir})");
        eprintln!("  Audit API: POST /api/v1/audit, GET /api/v1/audit/{{job_id}}");
    } else {
        eprintln!("  Audit: disabled (set AUDIT_LOG_DIR to enable)");
    }

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("Failed to bind");

    // Graceful shutdown: drain in-flight jobs + deregister worker
    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => eprintln!("\nReceived SIGINT, shutting down gracefully..."),
            _ = terminate => eprintln!("\nReceived SIGTERM, shutting down gracefully..."),
        }

        // Signal worker tasks to stop (triggers deregistration)
        drop(shutdown_tx);
        // Give heartbeat task time to deregister
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
    .expect("Server error");

    eprintln!("prove-server shut down.");
}
