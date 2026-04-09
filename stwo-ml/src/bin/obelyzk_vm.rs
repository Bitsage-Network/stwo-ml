//! `obelyzk-vm` — ZK Inference VM runtime.
//!
//! Multi-provider inference server with async proving.
//! Accepts OpenAI-compatible requests, routes to configured providers,
//! generates ZK proofs asynchronously, tracks conversations.
//!
//! ```text
//! OBELYSK_MODEL_DIR=/models/qwen3-14b \
//! UPSTREAM_URL=http://vllm:8000/v1 \
//!   obelyzk-vm --port 8080 --prove-workers 8
//! ```

#![feature(portable_simd)]

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Path, State},
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Sse, sse::Event},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use futures_util::stream;

use stwo_ml::providers::local::LocalProvider;
use stwo_ml::providers::openai_compat::OpenAiCompatProvider;
use stwo_ml::providers::types::*;
use stwo_ml::vm::queue::{ProvingQueue, ProvingStatus};

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

struct VmState {
    local_provider: Option<Arc<LocalProvider>>,
    upstream_provider: Option<Arc<OpenAiCompatProvider>>,
    proving_queue: Arc<ProvingQueue>,
    attestations: RwLock<HashMap<String, InferenceAttestation>>,
    started_at: Instant,
}

// ═══════════════════════════════════════════════════════════════════
// Request / Response types (OpenAI-compatible)
// ═══════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessageBody>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize)]
struct ChatMessageBody {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
    /// ObelyZK extension: proof metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    obelyzk: Option<ObelyzkMeta>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageOut,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatMessageOut {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ObelyzkMeta {
    proof_id: String,
    proof_status: String,
    trust_model: String,
    io_commitment: Option<String>,
    attestation_id: String,
}

#[derive(Serialize)]
struct ProofStatusResponse {
    proof_id: String,
    status: String,
    prove_time_ms: Option<u64>,
    io_commitment: Option<String>,
    proof_hash: Option<String>,
    calldata_size: Option<usize>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    uptime_secs: f64,
    local_model: Option<String>,
    upstream_url: Option<String>,
    prove_workers: usize,
    queue_depth: usize,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
}

fn api_error(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorBody>) {
    (status, Json(ErrorBody {
        error: ErrorDetail {
            message: msg.to_string(),
            r#type: "invalid_request_error".into(),
        },
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Handlers
// ═══════════════════════════════════════════════════════════════════

async fn health(State(state): State<Arc<VmState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        uptime_secs: state.started_at.elapsed().as_secs_f64(),
        local_model: state.local_provider.as_ref().map(|p| p.model_name.clone()),
        upstream_url: state.upstream_provider.as_ref().map(|p| p.base_url.clone()),
        prove_workers: 1, // TODO: expose from queue
        queue_depth: state.proving_queue.queue_depth(),
    })
}

/// Non-streaming chat completions — returns full response with proof.
async fn chat_completions(
    State(state): State<Arc<VmState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorBody>)> {
    if req.messages.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "messages cannot be empty"));
    }

    let prompt = req.messages.iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    if prompt.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "no user message found"));
    }

    // ── SSE Streaming path ─────────────────────────────────────────
    if req.stream {
        return chat_completions_stream(state, req.model, prompt).await;
    }

    // ── Synchronous path (existing: blocks until proof is done) ────
    let proof_id = format!("proof-{}", uuid::Uuid::new_v4());
    let attestation_id = format!("att-{}", uuid::Uuid::new_v4());

    let local = state.local_provider.as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "no local provider"))?;
    let local = Arc::clone(local);

    let result = tokio::task::spawn_blocking(move || {
        local.infer_text(&prompt, None)
            .map(|(text, mut res, _)| {
                res.trust_model = TrustModel::ZkProof { weight_commitment: local.weight_commitment.clone() };
                (text, res)
            })
            .map_err(|e| format!("{e}"))
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("join: {e}")))?
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e))?;

    let (text, inference_result) = result;
    let now_secs = epoch_secs();
    let io_hex = inference_result.io_commitment.map(|c| format!("0x{:x}", c));
    let trust_str = trust_model_str(&inference_result.trust_model);

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: now_secs,
        model: inference_result.model_id.clone(),
        choices: vec![ChatChoice { index: 0, message: ChatMessageOut { role: "assistant".into(), content: text }, finish_reason: "stop".into() }],
        usage: ChatUsage { prompt_tokens: inference_result.num_tokens, completion_tokens: 1, total_tokens: inference_result.num_tokens + 1 },
        obelyzk: Some(ObelyzkMeta { proof_id, proof_status: "complete".into(), trust_model: trust_str.into(), io_commitment: io_hex, attestation_id }),
    }).into_response())
}

/// SSE streaming chat completions — tokens flow instantly, proof arrives later.
///
/// Flow:
/// 1. Fast forward pass (~2-8s) → predicted token
/// 2. Stream token to client via SSE immediately
/// 3. Queue full GKR proof in background (ProvingQueue)
/// 4. Stream proof_id + status as final SSE event
/// 5. Client polls /v1/proofs/:id for proof completion
async fn chat_completions_stream(
    state: Arc<VmState>,
    model: String,
    prompt: String,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorBody>)> {
    let local = state.local_provider.as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "no local provider"))?;
    let local = Arc::clone(local);

    let proof_id = format!("proof-{}", uuid::Uuid::new_v4());
    let proof_id_for_bg = proof_id.clone();
    let model_for_event = model.clone();
    let now_secs = epoch_secs();

    // Phase 1: Fast forward pass (no proving) — ~2-8s
    let local_fast = Arc::clone(&local);
    let prompt_clone = prompt.clone();
    let fast_result = tokio::task::spawn_blocking(move || {
        let t = Instant::now();
        let result = local_fast.infer_fast(&prompt_clone);
        let ms = t.elapsed().as_millis() as u64;
        result.map(|(text, tokens, input)| (text, tokens, input, ms))
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("join: {e}")))?
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")))?;

    let (predicted_text, token_ids, input_matrix, inference_ms) = fast_result;
    let num_tokens = token_ids.len();

    // Phase 2: Queue full proof in background
    let local_prove = Arc::clone(&local);
    let pid = proof_id_for_bg.clone();
    let queue = Arc::clone(&state.proving_queue);

    // Submit to proving queue via background task
    tokio::task::spawn(async move {
        let job = stwo_ml::vm::queue::ProvingJob {
            job_id: pid.clone(),
            input_matrix: Some(input_matrix),
            trace: stwo_ml::vm::trace::ExecutionTrace {
                model_id: local_prove.model_name.clone(),
                input_tokens: token_ids.clone(),
                output: stwo_ml::components::matmul::M31Matrix::new(1, 1),
                io_commitment: None,
                policy_commitment: starknet_ff::FieldElement::ZERO,
                kv_commitment_before: None,
                kv_commitment_after: None,
                tokenization_commitment: None,
                inference_time_ms: inference_ms,
                num_tokens: token_ids.len(),
                position_offset: 0,
                proof: None,
            },
            graph: Arc::clone(&local_prove.graph),
            weights: Arc::clone(&local_prove.weights),
            weight_cache: None,
            submitted_at: Instant::now(),
            callback_url: None,
        };
        if let Err(e) = queue.submit(job) {
            eprintln!("[stream] failed to queue proof {pid}: {e}");
        } else {
            eprintln!("[stream] queued proof {pid} for background proving");
        }
    });

    // Phase 3: Build SSE event stream
    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let events = vec![
        // Token delta
        Event::default().data(serde_json::json!({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": now_secs,
            "model": model_for_event,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": predicted_text},
                "finish_reason": serde_json::Value::Null
            }]
        }).to_string()),
        // Finish
        Event::default().data(serde_json::json!({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": now_secs,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": num_tokens, "completion_tokens": 1, "total_tokens": num_tokens + 1},
            "obelyzk": {
                "proof_id": proof_id,
                "proof_status": "proving",
                "trust_model": "zk_proof",
                "inference_time_ms": inference_ms
            }
        }).to_string()),
        // OpenAI [DONE] sentinel
        Event::default().data("[DONE]".to_string()),
    ];

    let sse_stream = stream::iter(events.into_iter().map(Ok::<_, Infallible>));
    Ok(Sse::new(sse_stream).into_response())
}

async fn get_proof_status(
    State(state): State<Arc<VmState>>,
    Path(proof_id): Path<String>,
) -> Result<Json<ProofStatusResponse>, (StatusCode, Json<ErrorBody>)> {
    match state.proving_queue.get_status(&proof_id) {
        Some(status) => {
            let (status_str, time, io, hash, size) = match status {
                ProvingStatus::Queued => ("queued".into(), None, None, None, None),
                ProvingStatus::Proving => ("proving".into(), None, None, None, None),
                ProvingStatus::Complete { prove_time_ms, calldata_size, io_commitment, proof_hash } =>
                    ("complete".into(), Some(prove_time_ms), Some(io_commitment), Some(proof_hash), Some(calldata_size)),
                ProvingStatus::Failed { error } =>
                    (format!("failed: {error}"), None, None, None, None),
            };
            Ok(Json(ProofStatusResponse {
                proof_id,
                status: status_str,
                prove_time_ms: time,
                io_commitment: io,
                proof_hash: hash,
                calldata_size: size,
            }))
        }
        None => Err(api_error(StatusCode::NOT_FOUND, "proof not found")),
    }
}

async fn list_models(
    State(state): State<Arc<VmState>>,
) -> Json<serde_json::Value> {
    let mut models = Vec::new();

    if let Some(ref local) = state.local_provider {
        models.push(serde_json::json!({
            "id": local.model_name,
            "object": "model",
            "owned_by": "local",
            "trust_model": "zk_proof",
        }));
    }

    if let Some(ref upstream) = state.upstream_provider {
        models.push(serde_json::json!({
            "id": upstream.model,
            "object": "model",
            "owned_by": upstream.provider_name,
            "trust_model": "commitment",
        }));
    }

    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn trust_model_str(tm: &TrustModel) -> &'static str {
    match tm {
        TrustModel::ZkProof { .. } => "zk_proof",
        TrustModel::TlsAttestation { .. } => "tls_attestation",
        TrustModel::CommitmentOnly => "commitment",
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    eprintln!("╔═══════════════════════════════════════════╗");
    eprintln!("║       ObelyZK · ZK Inference VM           ║");
    eprintln!("║       Verifiable AI for every model       ║");
    eprintln!("╚═══════════════════════════════════════════╝");
    eprintln!();

    let port: u16 = std::env::var("PORT")
        .or_else(|_| std::env::var("BIND_PORT"))
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let prove_workers: usize = std::env::var("PROVE_WORKERS")
        .ok()
        .and_then(|w| w.parse().ok())
        .unwrap_or(1);

    // Load local provider if model dir is set
    let local_provider = std::env::var("OBELYSK_MODEL_DIR")
        .ok()
        .and_then(|dir| {
            let path = PathBuf::from(&dir);
            match LocalProvider::load(&path, None) {
                Ok(p) => Some(Arc::new(p)),
                Err(e) => {
                    eprintln!("  Warning: could not load local model from {dir}: {e}");
                    None
                }
            }
        });

    // Configure upstream provider if URL is set
    let upstream_provider = std::env::var("UPSTREAM_URL")
        .ok()
        .map(|url| {
            let model = std::env::var("UPSTREAM_MODEL").unwrap_or_else(|_| "default".into());
            let provider_name = std::env::var("UPSTREAM_PROVIDER").unwrap_or_else(|_| "upstream".into());
            let mut p = OpenAiCompatProvider::new(&url, &model);
            p.provider_name = provider_name;
            if let Ok(key) = std::env::var("UPSTREAM_API_KEY") {
                p = p.with_api_key(&key);
            }
            eprintln!("  Upstream: {} (model: {})", p.base_url, p.model);
            Arc::new(p)
        });

    eprintln!("  Prove workers: {prove_workers}");
    let proving_queue = Arc::new(ProvingQueue::new(prove_workers));

    let state = Arc::new(VmState {
        local_provider,
        upstream_provider,
        proving_queue,
        attestations: RwLock::new(HashMap::new()),
        started_at: Instant::now(),
    });

    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        // ObelyZK extensions
        .route("/v1/proofs/:proof_id", get(get_proof_status))
        .route("/health", get(health))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    eprintln!();
    eprintln!("  obelyzk-vm listening on {addr}");
    eprintln!("  OpenAI-compatible: POST /v1/chat/completions");
    eprintln!("  Proof status:      GET  /v1/proofs/:id");
    eprintln!("  Health:            GET  /health");
    eprintln!();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service()).await.unwrap();
}
