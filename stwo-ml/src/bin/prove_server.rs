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

use stwo_ml::compiler::hf_loader::load_hf_model;
use stwo_ml::compiler::onnx::load_onnx;
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::gadgets::quantize::{QuantStrategy, quantize_tensor};
use stwo_ml::starknet::{prepare_model_registration, prove_for_starknet_onchain};
use stwo_ml::tee::detect_tee_capability;

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
}

struct AppState {
    jobs: RwLock<HashMap<String, ProveJob>>,
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

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// =============================================================================
// Handlers
// =============================================================================

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
        weight_commitment,
        num_layers: registration.num_layers,
        input_shape: onnx.input_shape,
        graph: Arc::new(onnx.graph),
        weights: Arc::new(onnx.weights),
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
        weight_commitment,
        num_layers: registration.num_layers,
        input_shape: hf.input_shape,
        graph: Arc::new(hf.graph),
        weights: Arc::new(hf.weights),
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
            let result = job.result.as_ref().unwrap().clone();
            Ok(Json(result))
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
// Main
// =============================================================================

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        jobs: RwLock::new(HashMap::new()),
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
        .layer(CorsLayer::permissive())
        .with_state(state);

    let bind = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    eprintln!("prove-server listening on {bind}");
    eprintln!("  GPU: {}", detect_tee_capability().device_name);

    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("Failed to bind");
    axum::serve(listener, app).await.expect("Server error");
}
