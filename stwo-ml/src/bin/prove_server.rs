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
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use stwo::core::fields::m31::M31;
use stwo_ml::circuits::batch::BatchPublicInputs;

use stwo_ml::compiler::hf_loader::load_hf_model;
use stwo_ml::compiler::onnx::load_onnx;
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::gadgets::quantize::{QuantStrategy, quantize_tensor};
use stwo_ml::starknet::{prepare_model_registration, prove_for_starknet_onchain};
use stwo_ml::tee::detect_tee_capability;

use stwo_ml::privacy::relayer::{
    WithdrawalRecipients,
    build_submit_batch_proof_calldata,
    compute_withdrawal_binding_digest,
    hash_batch_public_inputs_for_cairo,
};
use stwo_ml::cairo_serde::serialize_proof;

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
}

struct LoadedModel {
    model_id: String,
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

struct AppState {
    jobs: RwLock<HashMap<String, ProveJob>>,
    privacy_jobs: RwLock<HashMap<String, PrivacyJob>>,
    models: RwLock<HashMap<String, LoadedModel>>,
    started_at: Instant,
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
}

fn default_security() -> String {
    "auto".to_string()
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
    match stwo_ml::audit::capture::CaptureHook::new(&hook_dir, model_id, weight_commitment, description) {
        Ok(hook) => {
            eprintln!("Audit capture enabled for model {} -> {}", model_id, hook_dir.display());
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
    let active = jobs.values().filter(|j| j.status == JobStatus::Proving || j.status == JobStatus::Queued).count();

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
    // Support either ONNX file or HF directory
    let onnx = if let Some(ref model_dir) = req.model_dir {
        let path = std::path::PathBuf::from(model_dir);
        load_hf_model(&path, None).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: format!("Failed to load HF model: {e}") }),
            )
        })?
    } else {
        let path = std::path::PathBuf::from(&req.model_path);
        load_onnx(&path).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: format!("Failed to load model: {e}") }),
            )
        })?
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

    let loaded = LoadedModel {
        model_id: model_id.clone(),
        weight_commitment: weight_commitment.clone(),
        num_layers: registration.num_layers,
        input_shape: onnx.input_shape,
        graph: Arc::new(onnx.graph),
        weights: Arc::new(onnx.weights),
        #[cfg(feature = "server-audit")]
        capture_hook: create_capture_hook(&model_id, &weight_commitment, desc),
    };

    state.models.write().await.insert(model_id, loaded);
    Ok((StatusCode::CREATED, Json(resp)))
}

async fn load_hf_model_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadHfModelRequest>,
) -> Result<(StatusCode, Json<LoadModelResponse>), (StatusCode, Json<ErrorResponse>)> {
    let path = std::path::PathBuf::from(&req.model_dir);
    let hf = load_hf_model(&path, req.layers).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: format!("Failed to load HF model: {e}") }),
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

    let loaded = LoadedModel {
        model_id: model_id.clone(),
        weight_commitment: weight_commitment.clone(),
        num_layers: registration.num_layers,
        input_shape: hf.input_shape,
        graph: Arc::new(hf.graph),
        weights: Arc::new(hf.weights),
        #[cfg(feature = "server-audit")]
        capture_hook: create_capture_hook(&model_id, &weight_commitment, desc),
    };

    state.models.write().await.insert(model_id, loaded);
    Ok((StatusCode::CREATED, Json(resp)))
}

async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelInfoResponse>, (StatusCode, Json<ErrorResponse>)> {
    let models = state.models.read().await;
    let model = models.get(&model_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: format!("Model '{model_id}' not found") }),
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
    // Validate model exists and build input
    let models = state.models.read().await;
    let model = models.get(&req.model_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: format!("Model '{}' not found. Load it first via POST /api/v1/models", req.model_id) }),
        )
    })?;

    let (in_rows, in_cols) = model.input_shape;
    let input_matrix = if let Some(ref input_f32) = req.input {
        let expected = in_rows * in_cols;
        if input_f32.len() != expected {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Input has {} values, model expects {} ({in_rows}x{in_cols})", input_f32.len(), expected),
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

    // Arc::clone is O(1) — no deep copy of model weights
    let graph = Arc::clone(&model.graph);
    let weights = Arc::clone(&model.weights);
    let model_weight_commitment = model.weight_commitment.clone();
    let model_id_clone = req.model_id.clone();
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

    // Spawn blocking proving task
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

        // Run CPU+GPU heavy proving on a blocking thread
        let result = tokio::task::spawn_blocking(move || {
            prove_for_starknet_onchain(&*graph, &input_matrix, &*weights)
        })
        .await;

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
                    tee_attestation_hash: proof
                        .tee_attestation_hash
                        .map(|h| format!("0x{:x}", h)),
                };

                if let Some(j) = jobs.get_mut(&jid) {
                    j.status = JobStatus::Completed;
                    j.progress_bps = 10000;
                    j.completed_at = Some(Instant::now());
                    j.result = Some(payload);
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
                            if let Ok(output_m31) =
                                stwo_ml::audit::replay::execute_forward_pass(
                                    &audit_graph,
                                    &input_matrix_clone,
                                    &audit_weights,
                                )
                            {
                                hook.record(stwo_ml::audit::capture::CaptureJob {
                                    input_tokens: vec![],
                                    output_tokens: vec![],
                                    input_m31: input_matrix_clone,
                                    output_m31,
                                    timestamp_ns: std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_nanos() as u64,
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
            Json(ErrorResponse { error: format!("Job '{job_id}' not found") }),
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
            Json(ErrorResponse { error: format!("Job '{job_id}' not found") }),
        )
    })?;

    match job.status {
        JobStatus::Completed => {
            let result = job.result.as_ref().ok_or_else(|| {
                (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                    error: "Job completed but result is missing (internal error)".to_string(),
                }))
            })?;
            Ok(Json(result.clone()))
        }
        JobStatus::Failed => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: job.error.clone().unwrap_or_else(|| "Unknown error".to_string()),
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
// Privacy Handlers
// =============================================================================

async fn submit_privacy_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PrivacyBatchRequest>,
) -> Result<(StatusCode, Json<PrivacyBatchSubmitResponse>), (StatusCode, Json<ErrorResponse>)> {
    if req.deposits.is_empty() && req.withdrawals.is_empty() && req.spends.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "Batch must contain at least one transaction".to_string() }),
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

        let result = tokio::task::spawn_blocking(move || {
            use stwo::core::fields::m31::BaseField;
            use stwo_ml::crypto::commitment::Note;
            use stwo_ml::circuits::deposit::DepositWitness;
            use stwo_ml::circuits::withdraw::WithdrawWitness;
            use stwo_ml::circuits::spend::{SpendWitness, InputNoteWitness, OutputNoteWitness};
            use stwo_ml::circuits::batch::{PrivacyBatch, prove_privacy_batch};
            use stwo_ml::crypto::merkle_m31::MerklePath;
            use stwo_ml::crypto::poseidon2_m31::RATE;

            let m31 = |v: u32| BaseField::from_u32_unchecked(v);
            let m31_4 = |arr: [u32; 4]| [m31(arr[0]), m31(arr[1]), m31(arr[2]), m31(arr[3])];
            let m31_8 = |arr: [u32; 8]| -> [BaseField; 8] {
                [m31(arr[0]), m31(arr[1]), m31(arr[2]), m31(arr[3]),
                 m31(arr[4]), m31(arr[5]), m31(arr[6]), m31(arr[7])]
            };

            // Helper: generate random blinding (fallible)
            let random_blinding = || -> Result<[BaseField; 4], String> {
                let mut buf = [0u8; 16];
                getrandom::getrandom(&mut buf)
                    .map_err(|e| format!("entropy source unavailable: {e}"))?;
                Ok([
                    m31(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) % ((1u32 << 31) - 1)),
                    m31(u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) % ((1u32 << 31) - 1)),
                    m31(u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) % ((1u32 << 31) - 1)),
                    m31(u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) % ((1u32 << 31) - 1)),
                ])
            };

            // Max amount that fits in two M31 limbs
            const MAX_AMOUNT: u64 = ((1u64 << 31) - 1) + ((1u64 << 31) - 1) * (1u64 << 31);

            // Build deposit witnesses
            let deposits: Vec<DepositWitness> = {
                let mut v = Vec::with_capacity(req.deposits.len());
                for d in &req.deposits {
                    if d.amount > MAX_AMOUNT {
                        return Err(format!("deposit amount {} exceeds max {}", d.amount, MAX_AMOUNT));
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
            let mut withdrawals: Vec<WithdrawWitness> = Vec::with_capacity(req.withdrawals.len());
            for (idx, w) in req.withdrawals.iter().enumerate() {
                let payout = w.payout_recipient.clone().ok_or_else(|| {
                    format!("withdrawals[{idx}] missing payout_recipient")
                })?;
                let credit = w
                    .credit_recipient
                    .clone()
                    .unwrap_or_else(|| payout.clone());
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
                let siblings: Vec<[BaseField; RATE]> = w
                    .merkle_siblings
                    .iter()
                    .map(|s| m31_8(*s))
                    .collect();
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
            let spends: Vec<SpendWitness> = req.spends.iter().map(|s| {
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
                        let siblings: Vec<[BaseField; RATE]> = inp.merkle_siblings
                            .iter()
                            .map(|sib| m31_8(*sib))
                            .collect();
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
                        let siblings: Vec<[BaseField; RATE]> = inp.merkle_siblings
                            .iter()
                            .map(|sib| m31_8(*sib))
                            .collect();
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
            }).collect();

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
                let deposit_results: Vec<PrivacyDepositResult> = pi.deposits.iter().map(|d| {
                    let mut hex = String::with_capacity(2 + 8 * 8);
                    hex.push_str("0x");
                    for &e in &d.commitment { hex.push_str(&format!("{:08x}", e.0)); }
                    PrivacyDepositResult {
                        commitment: hex,
                        amount: d.amount,
                        asset_id: d.asset_id.0,
                    }
                }).collect();

                // Build withdraw results
                let withdraw_results: Vec<PrivacyWithdrawResult> = pi.withdrawals.iter().map(|w| {
                    let mut hex = String::with_capacity(2 + 8 * 8);
                    hex.push_str("0x");
                    for &e in &w.nullifier { hex.push_str(&format!("{:08x}", e.0)); }
                    let amount = (w.amount_lo.0 as u64) | ((w.amount_hi.0 as u64) << 31);
                    PrivacyWithdrawResult {
                        nullifier: hex,
                        amount,
                        asset_id: w.asset_id.0,
                    }
                }).collect();

                // Build spend results
                let spend_results: Vec<PrivacySpendResult> = pi.spends.iter().map(|s| {
                    let nul_hex = |nul: &[M31]| -> String {
                        let mut h = String::with_capacity(2 + 8 * 8);
                        h.push_str("0x");
                        for &e in nul { h.push_str(&format!("{:08x}", e.0)); }
                        h
                    };
                    PrivacySpendResult {
                        nullifiers: [nul_hex(&s.nullifiers[0]), nul_hex(&s.nullifiers[1])],
                        output_commitments: [nul_hex(&s.output_commitments[0]), nul_hex(&s.output_commitments[1])],
                    }
                }).collect();

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
                for &e in &pi_hash { pi_hash_hex.push_str(&format!("{:08x}", e.0)); }

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
            Json(ErrorResponse { error: format!("Privacy job '{job_id}' not found") }),
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
            Json(ErrorResponse { error: format!("Privacy job '{job_id}' not found") }),
        )
    })?;

    match job.status {
        JobStatus::Completed => {
            let result = job.result.as_ref().ok_or_else(|| {
                (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
                    error: "Privacy job completed but result is missing (internal error)".to_string(),
                }))
            })?;
            Ok(Json(result.clone()))
        }
        JobStatus::Failed => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: job.error.clone().unwrap_or_else(|| "Unknown error".to_string()),
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
            Json(ErrorResponse { error: format!("invalid proof_hash: {e}") }),
        )
    })?;

    let mut jobs = state.privacy_jobs.write().await;
    let job = jobs.get_mut(&job_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: format!("Privacy job '{job_id}' not found") }),
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
                error: job.error.clone().unwrap_or_else(|| "Unknown error".to_string()),
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
    let state = Arc::new(AppState {
        jobs: RwLock::new(HashMap::new()),
        privacy_jobs: RwLock::new(HashMap::new()),
        models: RwLock::new(HashMap::new()),
        started_at: Instant::now(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/models", post(load_model))
        .route("/api/v1/models/hf", post(load_hf_model_handler))
        .route("/api/v1/models/{model_id}", get(get_model))
        .route("/api/v1/prove", post(submit_prove))
        .route("/api/v1/prove/{job_id}", get(get_prove_status))
        .route("/api/v1/prove/{job_id}/result", get(get_prove_result))
        .route("/api/v1/privacy/batch", post(submit_privacy_batch))
        .route("/api/v1/privacy/batch/{job_id}", get(get_privacy_batch_status))
        .route("/api/v1/privacy/batch/{job_id}/result", get(get_privacy_batch_result))
        .route(
            "/api/v1/privacy/batch/{job_id}/submit-calldata",
            post(build_privacy_submit_calldata),
        )
        .layer(CorsLayer::permissive())
        .with_state(state);

    let bind = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    eprintln!("prove-server listening on {bind}");
    eprintln!("  GPU: {}", detect_tee_capability().device_name);
    eprintln!("  Privacy: POST /api/v1/privacy/batch");
    eprintln!("  Privacy Submit: POST /api/v1/privacy/batch/{{job_id}}/submit-calldata");
    #[cfg(feature = "server-audit")]
    if let Ok(audit_dir) = std::env::var("AUDIT_LOG_DIR") {
        eprintln!("  Audit: enabled (AUDIT_LOG_DIR={audit_dir})");
    } else {
        eprintln!("  Audit: disabled (set AUDIT_LOG_DIR to enable)");
    }

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("Failed to bind");
    axum::serve(listener, app).await.expect("Server error");
}
