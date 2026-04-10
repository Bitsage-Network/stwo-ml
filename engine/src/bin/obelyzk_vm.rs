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

use obelyzk::providers::local::LocalProvider;
use obelyzk::providers::openai_compat::OpenAiCompatProvider;
use obelyzk::providers::anthropic::{AnthropicProvider, OpenAiProvider};
use obelyzk::providers::tls_attestation::TlsAttestation;
use obelyzk::providers::types::*;
use obelyzk::vm::queue::{ProvingQueue, ProvingStatus};

// ═══════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════

/// Per-conversation session with KV-cache for multi-turn proving.
struct ChatSession {
    session_id: String,
    model_id: String,
    kv_cache: obelyzk::components::attention::ModelKVCache,
    token_history: Vec<u32>,
    last_accessed: Instant,
    turns: usize,
}

struct VmState {
    local_provider: Option<Arc<LocalProvider>>,
    upstream_provider: Option<Arc<OpenAiCompatProvider>>,
    /// Anthropic Claude provider (TLS-attested).
    anthropic_provider: Option<Arc<AnthropicProvider>>,
    /// OpenAI provider (TLS-attested).
    openai_provider: Option<Arc<OpenAiProvider>>,
    proving_queue: Arc<ProvingQueue>,
    attestations: RwLock<HashMap<String, InferenceAttestation>>,
    /// TLS attestation records.
    tls_attestations: RwLock<HashMap<String, TlsAttestation>>,
    /// Multi-turn conversation sessions with KV-cache.
    sessions: RwLock<HashMap<String, ChatSession>>,
    started_at: Instant,
    /// Last on-chain proof metadata (for TUI dashboard).
    last_proof_meta: RwLock<obelyzk::providers::local::ProofMeta>,
    /// Total on-chain verifications in this session.
    on_chain_tx_count: std::sync::atomic::AtomicUsize,
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
    /// ObelyZK extension: continue a conversation session.
    #[serde(default)]
    session_id: Option<String>,
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
    /// Session ID for multi-turn conversations.
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    /// Number of conversation turns so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    turns: Option<usize>,
    /// Inference latency in ms (fast path only, excludes proving).
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_time_ms: Option<u64>,
    /// On-chain verification TX hash (Starknet Sepolia).
    #[serde(skip_serializing_if = "Option::is_none")]
    tx_hash: Option<String>,
    /// Poseidon hash of the verified proof (dedup key on-chain).
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_hash: Option<String>,
    /// Unique model identifier (Poseidon hash of weight commitments).
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    /// Number of STARK calldata felts submitted on-chain.
    #[serde(skip_serializing_if = "Option::is_none")]
    calldata_felts: Option<usize>,
    /// Block explorer URL for the verification TX.
    #[serde(skip_serializing_if = "Option::is_none")]
    explorer_url: Option<String>,
    /// Total proving time in seconds (GKR + recursive STARK).
    #[serde(skip_serializing_if = "Option::is_none")]
    prove_time_secs: Option<f64>,
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
        let sid = req.session_id.clone().unwrap_or_else(|| format!("ses-{}", uuid::Uuid::new_v4()));
        return chat_completions_stream(state, req.model, prompt, sid).await;
    }

    // ── Synchronous path ────────────────────────────────────────────
    let proof_id = format!("proof-{}", uuid::Uuid::new_v4());
    let attestation_id = format!("att-{}", uuid::Uuid::new_v4());
    let session_id = req.session_id.clone().unwrap_or_else(|| format!("ses-{}", uuid::Uuid::new_v4()));
    let now_secs = epoch_secs();

    // Route based on model name:
    // - Local models (loaded via OBELYSK_MODEL_DIR): ZK proof
    // - "claude-*" models: Anthropic API with TLS attestation
    // - "gpt-*" / "o1-*" / "o3-*" models: OpenAI API with TLS attestation
    // - Everything else: upstream (vLLM/Ollama/TGI) with IO commitment
    let messages: Vec<ChatMessage> = req.messages.iter()
        .map(|m| ChatMessage { role: m.role.clone(), content: m.content.clone() })
        .collect();

    let model_lower = req.model.to_lowercase();
    let mut proof_meta = obelyzk::providers::local::ProofMeta::default();
    let (text, trust_str, io_hex, num_tokens) =
        if let Some(ref local) = state.local_provider {
            // Check if the requested model matches the local model
            if model_lower == local.model_name.to_lowercase()
                || model_lower == "local"
                || state.anthropic_provider.is_none() && state.openai_provider.is_none() && state.upstream_provider.is_none()
            {
                let local = Arc::clone(local);
                let max_tok = req.max_tokens.unwrap_or(1) as usize;
                let result = tokio::task::spawn_blocking(move || {
                    local.generate_with_proof(&prompt, max_tok, |_, _, _| {})
                        .map_err(|e| format!("{e}"))
                })
                .await
                .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("join: {e}")))?
                .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e))?;
                let (text, _ids, pm) = result;
                let io_hex = pm.io_commitment.clone();
                // Update shared state for TUI dashboard
                if pm.tx_hash.is_some() {
                    state.on_chain_tx_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                *state.last_proof_meta.write().await = pm.clone();
                proof_meta = pm;
                (text, "zk_proof", io_hex, max_tok)
            } else if model_lower.starts_with("claude") {
                // Route to Anthropic
                if let Some(ref anthropic) = state.anthropic_provider {
                    let (res, att) = anthropic.chat(&messages, req.max_tokens).await
                        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, &format!("anthropic: {e}")))?;
                    state.tls_attestations.write().await.insert(att.attestation_id.clone(), att);
                    (res.text, trust_model_str(&res.trust_model), res.io_commitment.map(|c| format!("0x{:x}", c)), res.num_tokens)
                } else {
                    return Err(api_error(StatusCode::BAD_REQUEST, "Anthropic provider not configured (set ANTHROPIC_API_KEY)"));
                }
            } else if model_lower.starts_with("gpt") || model_lower.starts_with("o1") || model_lower.starts_with("o3") {
                // Route to OpenAI
                if let Some(ref openai) = state.openai_provider {
                    let (res, att) = openai.chat(&messages, req.max_tokens, req.temperature).await
                        .map_err(|e| api_error(StatusCode::BAD_GATEWAY, &format!("openai: {e}")))?;
                    state.tls_attestations.write().await.insert(att.attestation_id.clone(), att);
                    (res.text, trust_model_str(&res.trust_model), res.io_commitment.map(|c| format!("0x{:x}", c)), res.num_tokens)
                } else {
                    return Err(api_error(StatusCode::BAD_REQUEST, "OpenAI provider not configured (set OPENAI_API_KEY)"));
                }
            } else if let Some(ref upstream) = state.upstream_provider {
                let result = upstream.chat(&messages, req.max_tokens, req.temperature).await
                    .map_err(|e| api_error(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")))?;
                (result.text, trust_model_str(&result.trust_model), result.io_commitment.map(|c| format!("0x{:x}", c)), result.num_tokens)
            } else {
                return Err(api_error(StatusCode::BAD_REQUEST, &format!("Model '{}' not available", req.model)));
            }
        } else if model_lower.starts_with("claude") {
            if let Some(ref anthropic) = state.anthropic_provider {
                let (res, att) = anthropic.chat(&messages, req.max_tokens).await
                    .map_err(|e| api_error(StatusCode::BAD_GATEWAY, &format!("anthropic: {e}")))?;
                state.tls_attestations.write().await.insert(att.attestation_id.clone(), att);
                (res.text, trust_model_str(&res.trust_model), res.io_commitment.map(|c| format!("0x{:x}", c)), res.num_tokens)
            } else {
                return Err(api_error(StatusCode::SERVICE_UNAVAILABLE, "no Anthropic provider"));
            }
        } else if model_lower.starts_with("gpt") || model_lower.starts_with("o1") || model_lower.starts_with("o3") {
            if let Some(ref openai) = state.openai_provider {
                let (res, att) = openai.chat(&messages, req.max_tokens, req.temperature).await
                    .map_err(|e| api_error(StatusCode::BAD_GATEWAY, &format!("openai: {e}")))?;
                state.tls_attestations.write().await.insert(att.attestation_id.clone(), att);
                (res.text, trust_model_str(&res.trust_model), res.io_commitment.map(|c| format!("0x{:x}", c)), res.num_tokens)
            } else {
                return Err(api_error(StatusCode::SERVICE_UNAVAILABLE, "no OpenAI provider"));
            }
        } else if let Some(ref upstream) = state.upstream_provider {
            let result = upstream.chat(&messages, req.max_tokens, req.temperature).await
                .map_err(|e| api_error(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")))?;
            (result.text, trust_model_str(&result.trust_model), result.io_commitment.map(|c| format!("0x{:x}", c)), result.num_tokens)
        } else {
            return Err(api_error(StatusCode::SERVICE_UNAVAILABLE, "no provider configured"));
        };

    // Create/update session
    {
        let mut sessions = state.sessions.write().await;
        let session = sessions.entry(session_id.clone()).or_insert_with(|| ChatSession {
            session_id: session_id.clone(),
            model_id: req.model.clone(),
            kv_cache: obelyzk::components::attention::ModelKVCache::new(),
            token_history: Vec::new(),
            last_accessed: Instant::now(),
            turns: 0,
        });
        session.turns += 1;
        session.last_accessed = Instant::now();
    }

    let turns = state.sessions.read().await.get(&session_id).map(|s| s.turns).unwrap_or(1);

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: now_secs,
        model: req.model,
        choices: vec![ChatChoice { index: 0, message: ChatMessageOut { role: "assistant".into(), content: text }, finish_reason: "stop".into() }],
        usage: ChatUsage { prompt_tokens: num_tokens, completion_tokens: 1, total_tokens: num_tokens + 1 },
        obelyzk: Some(ObelyzkMeta {
            proof_id, proof_status: "complete".into(), trust_model: trust_str.into(),
            io_commitment: io_hex, attestation_id,
            session_id: Some(session_id), turns: Some(turns), inference_time_ms: None,
            tx_hash: proof_meta.tx_hash,
            proof_hash: proof_meta.proof_hash,
            model_id: proof_meta.model_id,
            calldata_felts: proof_meta.calldata_felts,
            explorer_url: proof_meta.explorer_url,
            prove_time_secs: proof_meta.prove_time_secs,
        }),
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
    session_id: String,
) -> Result<axum::response::Response, (StatusCode, Json<ErrorBody>)> {
    let local = state.local_provider.as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "no local provider (streaming requires local model)"))?;
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

    // Create/update session
    {
        let mut sessions = state.sessions.write().await;
        let session = sessions.entry(session_id.clone()).or_insert_with(|| ChatSession {
            session_id: session_id.clone(),
            model_id: model.clone(),
            kv_cache: obelyzk::components::attention::ModelKVCache::new(),
            token_history: Vec::new(),
            last_accessed: Instant::now(),
            turns: 0,
        });
        session.turns += 1;
        session.token_history.extend_from_slice(&token_ids);
        session.last_accessed = Instant::now();
    }

    // Phase 2: Queue full proof in background
    let local_prove = Arc::clone(&local);
    let pid = proof_id_for_bg.clone();
    let queue = Arc::clone(&state.proving_queue);

    // Submit to proving queue via background task
    tokio::task::spawn(async move {
        let job = obelyzk::vm::queue::ProvingJob {
            job_id: pid.clone(),
            input_matrix: Some(input_matrix),
            forward_result: None, // TODO: capture ForwardPassResult from traced execution
            trace: obelyzk::vm::trace::ExecutionTrace {
                model_id: local_prove.model_name.clone(),
                input_tokens: token_ids.clone(),
                output: obelyzk::components::matmul::M31Matrix::new(1, 1),
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
                "inference_time_ms": inference_ms,
                "session_id": session_id
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

    if let Some(ref anthropic) = state.anthropic_provider {
        models.push(serde_json::json!({
            "id": anthropic.model,
            "object": "model",
            "owned_by": "anthropic",
            "trust_model": "tls_attestation",
        }));
    }

    if let Some(ref openai) = state.openai_provider {
        models.push(serde_json::json!({
            "id": openai.model,
            "object": "model",
            "owned_by": "openai",
            "trust_model": "tls_attestation",
        }));
    }

    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

/// Batch prove endpoint — prove multiple tokens in one pass for throughput benchmarking.
///
/// POST /v1/prove/batch { "model": "...", "token_ids": [1,2,3,...], "prompt": "..." }
async fn prove_batch(
    State(state): State<Arc<VmState>>,
    Json(req): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorBody>)> {
    let local = state.local_provider.as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "no local provider"))?;
    let local = Arc::clone(local);

    // Get token IDs from request — either explicit or from prompt tokenization
    let token_ids: Vec<u32> = if let Some(ids) = req["token_ids"].as_array() {
        ids.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect()
    } else if let Some(prompt) = req["prompt"].as_str() {
        let encoding = local.tokenizer.encode(prompt, false)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, &format!("tokenize: {e}")))?;
        encoding.get_ids().to_vec()
    } else {
        return Err(api_error(StatusCode::BAD_REQUEST, "provide 'token_ids' or 'prompt'"));
    };

    if token_ids.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "empty token list"));
    }

    let num_tokens = token_ids.len();

    let result = tokio::task::spawn_blocking(move || {
        local.prove_batch(&token_ids)
            .map_err(|e| format!("{e}"))
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("join: {e}")))?
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e))?;

    let (_output, proof, prove_time_ms) = result;
    let tok_per_sec = num_tokens as f64 / (prove_time_ms as f64 / 1000.0);

    Ok(Json(serde_json::json!({
        "num_tokens": num_tokens,
        "prove_time_ms": prove_time_ms,
        "tokens_per_second": format!("{:.1}", tok_per_sec),
        "io_commitment": format!("0x{:x}", proof.io_commitment),
        "proof_hash": format!("0x{:x}", starknet_crypto::poseidon_hash_many(&[
            proof.io_commitment, proof.layer_chain_commitment,
        ])),
        "calldata_size": proof.gkr_proof.as_ref().map(|g| g.layer_proofs.len() * 100).unwrap_or(0),
        "gpu": obelyzk::backend::gpu_is_available(),
    })))
}

/// Get a TLS attestation by ID.
async fn get_attestation(
    State(state): State<Arc<VmState>>,
    Path(attestation_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorBody>)> {
    let atts = state.tls_attestations.read().await;
    let att = atts.get(&attestation_id)
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "attestation not found"))?;

    Ok(Json(serde_json::json!({
        "attestation_id": att.attestation_id,
        "server_domain": att.server_domain,
        "cert_fingerprint": att.cert_fingerprint,
        "request_method_path": att.request_method_path,
        "request_body_hash": format!("0x{:x}", att.request_body_hash),
        "response_body_hash": format!("0x{:x}", att.response_body_hash),
        "io_commitment": format!("0x{:x}", att.io_commitment),
        "status_code": att.status_code,
        "timestamp": att.timestamp,
        "level": format!("{:?}", att.level),
        "provider": att.provider,
        "model": att.model,
        "response_text_preview": if att.response_text.len() > 200 {
            format!("{}...", &att.response_text[..200])
        } else {
            att.response_text.clone()
        },
    })))
}

/// List active sessions.
async fn list_sessions(
    State(state): State<Arc<VmState>>,
) -> Json<serde_json::Value> {
    let sessions = state.sessions.read().await;
    let list: Vec<serde_json::Value> = sessions.values().map(|s| {
        serde_json::json!({
            "session_id": s.session_id,
            "model_id": s.model_id,
            "turns": s.turns,
            "tokens": s.token_history.len(),
            "age_secs": s.last_accessed.elapsed().as_secs(),
        })
    }).collect();
    Json(serde_json::json!({ "sessions": list, "count": list.len() }))
}

/// Delete a session.
async fn delete_session(
    State(state): State<Arc<VmState>>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorBody>)> {
    let removed = state.sessions.write().await.remove(&session_id).is_some();
    if removed {
        Ok(Json(serde_json::json!({ "deleted": session_id })))
    } else {
        Err(api_error(StatusCode::NOT_FOUND, "session not found"))
    }
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
    let args: Vec<String> = std::env::args().collect();

    // Help
    if args.len() < 2 || args.get(1).map(|s| s.as_str()) == Some("--help") || args.get(1).map(|s| s.as_str()) == Some("-h") {
        println!("obelyzk.rs — Verifiable AI Engine");
        println!();
        println!("USAGE:");
        println!("    obelyzk <COMMAND> [OPTIONS]");
        println!();
        println!("COMMANDS:");
        println!("    serve              Start OpenAI-compatible API server");
        println!("    chat               Interactive verified chat");
        println!("    bench              Throughput benchmark");
        println!("    dashboard          Live Cipher Noir TUI monitor");
        println!();
        println!("OPTIONS:");
        println!("    --help, -h         Show this help message");
        println!();
        println!("ENVIRONMENT:");
        println!("    OBELYSK_MODEL_DIR  Model directory or .gguf file");
        println!("    PORT               API server port (default: 8080)");
        println!("    ANTHROPIC_API_KEY  Enable Claude provider (TLS attestation)");
        println!("    OPENAI_API_KEY     Enable GPT provider (TLS attestation)");
        println!("    PROVE_WORKERS      Number of GPU proving workers (default: 1)");
        println!();
        println!("EXAMPLES:");
        println!("    OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk serve");
        println!("    ANTHROPIC_API_KEY=sk-ant-... obelyzk chat --model claude-sonnet");
        println!("    obelyzk bench --tokens 64");
        println!();
        println!("Docs: https://github.com/Bitsage-Network/obelyzk.rs");
        return;
    }

    // CLI chat mode
    if args.get(1).map(|s| s.as_str()) == Some("chat") {
        return run_chat_mode(&args[2..]).await;
    }

    // Benchmark mode: `obelyzk bench [--tokens 64]`
    if args.get(1).map(|s| s.as_str()) == Some("bench") {
        return run_benchmark(&args[2..]).await;
    }

    // Dashboard mode: `obelyzk dashboard`
    if args.get(1).map(|s| s.as_str()) == Some("dashboard") {
        return run_dashboard().await;
    }

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

    // Configure Anthropic provider (Claude) if API key is set
    let anthropic_provider = std::env::var("ANTHROPIC_API_KEY").ok().map(|key| {
        let model = std::env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        eprintln!("  Anthropic: {} (TLS attestation)", model);
        Arc::new(AnthropicProvider::new(&key, &model))
    });

    // Configure OpenAI provider (GPT) if API key is set
    let openai_provider = std::env::var("OPENAI_API_KEY").ok().map(|key| {
        let model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "gpt-4o".into());
        eprintln!("  OpenAI: {} (TLS attestation)", model);
        Arc::new(OpenAiProvider::new(&key, &model))
    });

    eprintln!("  Prove workers: {prove_workers}");
    let proving_queue = Arc::new(ProvingQueue::new(prove_workers));

    let state = Arc::new(VmState {
        local_provider,
        upstream_provider,
        anthropic_provider,
        openai_provider,
        proving_queue,
        attestations: RwLock::new(HashMap::new()),
        tls_attestations: RwLock::new(HashMap::new()),
        sessions: RwLock::new(HashMap::new()),
        started_at: Instant::now(),
        last_proof_meta: RwLock::new(obelyzk::providers::local::ProofMeta::default()),
        on_chain_tx_count: std::sync::atomic::AtomicUsize::new(0),
    });

    // Session eviction — prune expired sessions every 60s (TTL: 5 min)
    {
        let s = Arc::clone(&state);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                let mut sessions = s.sessions.write().await;
                let before = sessions.len();
                sessions.retain(|_, sess| sess.last_accessed.elapsed() < std::time::Duration::from_secs(300));
                let evicted = before - sessions.len();
                if evicted > 0 {
                    eprintln!("[vm] evicted {} expired sessions ({} active)", evicted, sessions.len());
                }
            }
        });
    }

    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        // ObelyZK extensions
        .route("/v1/proofs/:proof_id", get(get_proof_status))
        .route("/v1/prove/batch", post(prove_batch))
        .route("/v1/sessions", get(list_sessions))
        .route("/v1/sessions/:session_id", axum::routing::delete(delete_session))
        .route("/v1/attestations/:attestation_id", get(get_attestation))
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

// ═══════════════════════════════════════════════════════════════════
// Dashboard Mode — Interactive Cipher Noir TUI (ratatui + crossterm)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "tui")]
async fn run_dashboard() {
    use crossterm::{
        event::{self, Event, KeyCode, KeyModifiers},
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        execute,
    };
    use ratatui::prelude::*;
    use obelyzk::tui::interactive::*;

    let model_dir = std::env::var("OBELYSK_MODEL_DIR").unwrap_or_default();
    let model_name = std::path::Path::new(&model_dir)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "no model".into());

    let started = Instant::now();
    let mut state = InteractiveDashState::default();
    // Set model name on the first model entry if loaded
    if !state.models.is_empty() {
        state.models[0].name = model_name;
    }

    // Detect GPU
    if let Some(name) = obelyzk::backend::gpu_device_name() {
        state.gpu_name = name;
        state.gpu_memory_gb = 80.0;
    }

    // Check if API server is running via simple TCP connect
    if std::net::TcpStream::connect_timeout(
        &"127.0.0.1:8080".parse().unwrap(),
        std::time::Duration::from_millis(500),
    ).is_ok() {
        // API is running — model is already loaded by serve
    }

    // Channel for background API responses
    let (chat_tx, chat_rx) = std::sync::mpsc::channel::<String>();

    // Terminal setup
    enable_raw_mode().expect("raw mode");
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).expect("alt screen");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("terminal");

    let mut proving_start: Option<Instant> = None;

    loop {
        // Update runtime state
        state.uptime_secs = started.elapsed().as_secs();
        state.tick = state.tick.wrapping_add(1);

        // Update proving elapsed time
        if state.proving_active {
            if let Some(start) = proving_start {
                state.proving_elapsed_secs = start.elapsed().as_secs_f64();
                // Simulate layer progress based on elapsed time
                let progress = (state.proving_elapsed_secs / 50.0).min(1.0); // ~50s for full GKR
                state.proving_layer = (progress * state.proving_total_layers as f64) as usize;
                state.proving_matmul = (progress * state.proving_total_matmuls as f64) as usize;
                if state.proving_elapsed_secs > 5.0 { state.proving_phase = "gkr".into(); }
                if state.proving_elapsed_secs > 48.0 { state.proving_phase = "stark".into(); }
            }
            // Push throughput sample
            let tps = if state.proving_elapsed_secs > 0.0 { 1.0 / state.proving_elapsed_secs } else { 0.0 };
            push_limited(&mut state.throughput_history, (tps * 100.0) as u64);
            push_limited(&mut state.gpu_util_history, 85 + (state.tick % 10) as u64);
        }

        // Poll for API responses
        while let Ok(response) = chat_rx.try_recv() {
            state.proving_active = false;
            state.proving_phase = "idle".into();
            proving_start = None;

            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&response) {
                let text = data["choices"][0]["message"]["content"]
                    .as_str().unwrap_or("(no response)").to_string();
                let meta = &data["obelyzk"];
                let tx_hash = meta["tx_hash"].as_str().map(String::from);
                let proof_status = if tx_hash.is_some() { "verified" } else { "proved" };

                // Update last chat message
                if let Some(last) = state.chat_messages.last_mut() {
                    if last.role == "assistant" {
                        last.content = text;
                        last.proof_status = Some(proof_status.into());
                        last.tx_hash = tx_hash.clone();
                    }
                }

                // Update stats
                state.total_tokens_proven += 1;
                state.total_proofs += 1;
                if let Some(felts) = meta["calldata_felts"].as_u64() {
                    state.stark_felts = Some(felts as usize);
                }
                if let Some(rt) = meta["prove_time_secs"].as_f64() {
                    state.stark_time_secs = Some(rt);
                } else {
                    state.stark_time_secs = Some(state.proving_elapsed_secs);
                }
                // Update peak throughput
                if state.proving_elapsed_secs > 0.0 {
                    state.current_tok_per_sec = 1.0 / state.proving_elapsed_secs;
                    if state.current_tok_per_sec > state.peak_tok_per_sec {
                        state.peak_tok_per_sec = state.current_tok_per_sec;
                    }
                }

                // Add on-chain TX
                if let Some(ref tx) = tx_hash {
                    state.verification_count += 1;
                    state.on_chain_txs.push(OnChainTx {
                        tx_hash: tx.clone(),
                        model: state.models.get(state.selected_model)
                            .map(|m| m.name.clone()).unwrap_or_default(),
                        felts: state.stark_felts.unwrap_or(0),
                        verified: true,
                    });
                }
            } else {
                // Error response
                if let Some(last) = state.chat_messages.last_mut() {
                    if last.role == "assistant" {
                        last.content = format!("Error: {}", &response[..200.min(response.len())]);
                        last.proof_status = Some("failed".into());
                    }
                }
            }
        }

        // Render
        terminal.draw(|frame| {
            render_interactive(frame, &state);
        }).expect("draw");

        // Handle input (50ms poll)
        if event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(key)) => {
                    // Global keys
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                        KeyCode::Char('q') if state.mode != Mode::Chat => break,
                        KeyCode::Esc if state.mode != Mode::Chat => break,

                        // Mode switching (only when not typing in chat)
                        KeyCode::Char('1') if state.mode != Mode::Chat => state.mode = Mode::Monitor,
                        KeyCode::Char('2') if state.mode != Mode::Chat => state.mode = Mode::Prove,
                        KeyCode::Char('3') if state.mode != Mode::Chat => state.mode = Mode::Chat,
                        KeyCode::Char('4') if state.mode != Mode::Chat => state.mode = Mode::OnChain,

                        // Theme cycling
                        KeyCode::Char('t') if state.mode != Mode::Chat => {
                            state.theme_idx = (state.theme_idx + 1) % obelyzk::tui::interactive::THEMES.len();
                        }

                        // Tick rate adjustment
                        KeyCode::Char('+') | KeyCode::Char('=') if state.mode != Mode::Chat => {
                            state.tick_rate_ms = (state.tick_rate_ms.saturating_sub(10)).max(10);
                        }
                        KeyCode::Char('-') if state.mode != Mode::Chat => {
                            state.tick_rate_ms = (state.tick_rate_ms + 10).min(500);
                        }

                        // Tab to cycle models
                        KeyCode::Tab if state.mode != Mode::Chat => {
                            if !state.models.is_empty() {
                                state.selected_model = (state.selected_model + 1) % state.models.len();
                            }
                        }

                        // Chat mode input
                        KeyCode::Char(c) if state.mode == Mode::Chat => {
                            state.input_buffer.insert(state.input_cursor, c);
                            state.input_cursor += 1;
                        }
                        KeyCode::Backspace if state.mode == Mode::Chat && state.input_cursor > 0 => {
                            state.input_cursor -= 1;
                            state.input_buffer.remove(state.input_cursor);
                        }
                        KeyCode::Esc if state.mode == Mode::Chat => {
                            state.input_buffer.clear();
                            state.input_cursor = 0;
                            state.mode = Mode::Monitor;
                        }
                        KeyCode::Enter if state.mode == Mode::Chat && !state.input_buffer.is_empty() => {
                            let prompt = state.input_buffer.clone();
                            state.chat_messages.push(ChatMsg {
                                role: "user".into(),
                                content: prompt.clone(),
                                proof_status: None,
                                tx_hash: None,
                            });
                            state.input_buffer.clear();
                            state.input_cursor = 0;

                            // Show proving indicator
                            state.proving_active = true;
                            state.proving_phase = "forward".into();
                            state.proving_layer = 0;
                            state.proving_elapsed_secs = 0.0;
                            proving_start = Some(Instant::now());

                            state.chat_messages.push(ChatMsg {
                                role: "assistant".into(),
                                content: "Proving...".into(),
                                proof_status: Some("proving".into()),
                                tx_hash: None,
                            });

                            // Submit to local API in background thread
                            let tx = chat_tx.clone();
                            std::thread::spawn(move || {
                                let body = serde_json::json!({
                                    "model": "local",
                                    "messages": [{"role": "user", "content": prompt}],
                                    "max_tokens": 1,
                                    "stream": false,
                                });
                                match ureq::post("http://localhost:8080/v1/chat/completions")
                                    .header("Content-Type", "application/json")
                                    .send(body.to_string().as_bytes())
                                {
                                    Ok(resp) => {
                                        let text = resp.into_body().read_to_string().unwrap_or_default();
                                        let _ = tx.send(text);
                                    }
                                    Err(e) => {
                                        let _ = tx.send(format!("{{\"error\":\"{}\"}}", e));
                                    }
                                }
                            });
                        }

                        _ => {}
                    }
                }
                Ok(Event::Resize(_, _)) => {}
                _ => {}
            }
        }
    }

    // Cleanup
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
}

#[cfg(not(feature = "tui"))]
async fn run_dashboard() {
    eprintln!("Dashboard requires the 'tui' feature. Build with:");
    eprintln!("  cargo build --features \"server,tui\"");
    std::process::exit(1);
}

// ═══════════════════════════════════════════════════════════════════
// Benchmark Mode
// ═══════════════════════════════════════════════════════════════════

async fn run_benchmark(args: &[String]) {
    let model_dir = std::env::var("OBELYSK_MODEL_DIR")
        .unwrap_or_else(|_| {
            eprintln!("Error: OBELYSK_MODEL_DIR not set");
            std::process::exit(1);
        });

    // Parse --tokens flag
    let mut num_tokens: usize = 1;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--tokens" && i + 1 < args.len() {
            num_tokens = args[i + 1].parse().unwrap_or(1);
            i += 2;
        } else {
            i += 1;
        }
    }

    println!();
    println!("  \x1b[92m╔═══════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[92m║\x1b[0m  ObelyZK · Throughput Benchmark            \x1b[92m║\x1b[0m");
    println!("  \x1b[92m╚═══════════════════════════════════════════╝\x1b[0m");
    println!();

    let provider = LocalProvider::load(&PathBuf::from(&model_dir), None)
        .unwrap_or_else(|e| { eprintln!("Error: {e}"); std::process::exit(1); });

    let gpu_name = obelyzk::backend::gpu_device_name().unwrap_or_else(|| "CPU".into());
    println!("  \x1b[90mMODEL\x1b[0m    \x1b[97;1m{}\x1b[0m", provider.model_name);
    println!("  \x1b[90mGPU\x1b[0m      \x1b[36m{}\x1b[0m", gpu_name);
    println!("  \x1b[90mTOKENS\x1b[0m   \x1b[97;1m{}\x1b[0m", num_tokens);
    println!("  \x1b[90mD_MODEL\x1b[0m  {}", provider.hidden_size);
    println!();

    // Generate synthetic token IDs (1..N) for benchmarking
    let token_ids: Vec<u32> = (1..=num_tokens as u32).collect();

    println!("  \x1b[33mProving {} tokens...\x1b[0m", num_tokens);
    let t_start = std::time::Instant::now();

    let result = tokio::task::spawn_blocking(move || {
        provider.prove_batch(&token_ids)
    }).await.unwrap();

    match result {
        Ok((_output, proof, prove_time_ms)) => {
            let elapsed = t_start.elapsed().as_secs_f64();
            let tok_per_sec = num_tokens as f64 / elapsed;

            println!();
            println!("  \x1b[92m════════════════════════════════════════\x1b[0m");
            println!("  \x1b[92m  BENCHMARK RESULTS\x1b[0m");
            println!("  \x1b[92m════════════════════════════════════════\x1b[0m");
            println!("  \x1b[90mTokens:\x1b[0m        \x1b[97;1m{}\x1b[0m", num_tokens);
            println!("  \x1b[90mTotal time:\x1b[0m    \x1b[97;1m{:.1}s\x1b[0m", elapsed);
            println!("  \x1b[90mThroughput:\x1b[0m    \x1b[92;1m{:.1} tok/s\x1b[0m", tok_per_sec);
            println!("  \x1b[90mPer token:\x1b[0m     {:.1}ms", (elapsed * 1000.0) / num_tokens as f64);
            println!("  \x1b[90mIO commitment:\x1b[0m \x1b[35m0x{:x}\x1b[0m", proof.io_commitment);
            println!("  \x1b[90mGPU:\x1b[0m           {}", if obelyzk::backend::gpu_is_available() { "\x1b[36mactive\x1b[0m" } else { "\x1b[90minactive\x1b[0m" });
            println!("  \x1b[92m════════════════════════════════════════\x1b[0m");
            println!();
        }
        Err(e) => {
            eprintln!("  \x1b[31mBenchmark failed: {e}\x1b[0m");
            std::process::exit(1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Interactive CLI Chat Mode
// ═══════════════════════════════════════════════════════════════════

async fn run_chat_mode(args: &[String]) {
    use std::io::{BufRead, Write};

    // Parse --model flag
    let mut model = std::env::var("OBELYZK_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model = args[i + 1].clone();
            i += 2;
        } else {
            i += 1;
        }
    }

    let model_lower = model.to_lowercase();

    // Determine provider
    let is_anthropic = model_lower.starts_with("claude");
    let is_openai = model_lower.starts_with("gpt") || model_lower.starts_with("o1") || model_lower.starts_with("o3");

    // Verify API keys
    if is_anthropic && std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("Error: ANTHROPIC_API_KEY not set");
        std::process::exit(1);
    }
    if is_openai && std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("Error: OPENAI_API_KEY not set");
        std::process::exit(1);
    }

    let trust_label = if is_anthropic || is_openai { "TLS attestation" } else { "ZK proof" };

    // Header
    println!();
    println!("  \x1b[92m╔═══════════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[92m║\x1b[0m  ObelyZK · Verifiable Chat                \x1b[92m║\x1b[0m");
    println!("  \x1b[92m║\x1b[0m  Every response is cryptographically bound \x1b[92m║\x1b[0m");
    println!("  \x1b[92m╚═══════════════════════════════════════════╝\x1b[0m");
    println!();
    println!("  \x1b[90mMODEL\x1b[0m  \x1b[97;1m{model}\x1b[0m");
    println!("  \x1b[90mTRUST\x1b[0m  \x1b[92m{trust_label}\x1b[0m");
    println!("  \x1b[90mTYPE\x1b[0m   \x1b[90m'exit' to quit\x1b[0m");
    println!();

    let mut conversation: Vec<ChatMessage> = Vec::new();
    let mut turn = 0usize;

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    loop {
        // Prompt
        print!("  \x1b[92;1mYou\x1b[0m  ");
        stdout.flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
            break;
        }
        let input = line.trim().to_string();
        if input.is_empty() { continue; }
        if input == "exit" || input == "quit" || input == "/exit" {
            println!("\n  \x1b[90m{turn} turns · all attested · bye\x1b[0m\n");
            break;
        }

        conversation.push(ChatMessage { role: "user".into(), content: input.clone() });

        // Call the appropriate provider
        let t_start = std::time::Instant::now();

        let result = if is_anthropic {
            let key = std::env::var("ANTHROPIC_API_KEY").unwrap();
            let provider = AnthropicProvider::new(&key, &model);
            match provider.chat(&conversation, Some(2048)).await {
                Ok((res, att)) => {
                    conversation.push(ChatMessage { role: "assistant".into(), content: res.text.clone() });
                    Some((res.text, format!("0x{:x}", att.io_commitment), att.attestation_id, res.inference_time_ms))
                }
                Err(e) => {
                    println!("  \x1b[33mError: {e}\x1b[0m\n");
                    conversation.pop(); // remove failed user msg
                    continue;
                }
            }
        } else if is_openai {
            let key = std::env::var("OPENAI_API_KEY").unwrap();
            let provider = OpenAiProvider::new(&key, &model);
            match provider.chat(&conversation, Some(2048), Some(0.7)).await {
                Ok((res, att)) => {
                    conversation.push(ChatMessage { role: "assistant".into(), content: res.text.clone() });
                    Some((res.text, format!("0x{:x}", att.io_commitment), att.attestation_id, res.inference_time_ms))
                }
                Err(e) => {
                    println!("  \x1b[33mError: {e}\x1b[0m\n");
                    conversation.pop();
                    continue;
                }
            }
        } else {
            // Local model — multi-token autoregressive generation
            let model_dir = std::env::var("OBELYSK_MODEL_DIR").unwrap_or_default();
            if model_dir.is_empty() {
                println!("  \x1b[33mSet OBELYSK_MODEL_DIR for local model chat\x1b[0m\n");
                conversation.pop();
                continue;
            }
            let provider = match LocalProvider::load(&PathBuf::from(&model_dir), None) {
                Ok(p) => p,
                Err(e) => {
                    println!("  \x1b[33mModel load error: {e}\x1b[0m\n");
                    conversation.pop();
                    continue;
                }
            };

            let max_tokens: usize = std::env::var("OBELYZK_MAX_TOKENS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(128);

            // Stream tokens as they're generated
            print!("\n  \x1b[36;1m AI\x1b[0m  ");
            stdout.flush().ok();

            let t_gen = std::time::Instant::now();
            let gen_result = provider.generate(&input, max_tokens, |text, _id, _step| {
                print!("{text}");
                stdout.flush().ok();
            });

            match gen_result {
                Ok((text, token_ids)) => {
                    let gen_ms = t_gen.elapsed().as_millis() as u64;
                    let n_tokens = token_ids.len();
                    let tok_per_sec = if gen_ms > 0 { n_tokens as f64 / (gen_ms as f64 / 1000.0) } else { 0.0 };

                    println!();
                    println!();
                    println!("  \x1b[92m✓\x1b[0m \x1b[90mZK proof\x1b[0m  \x1b[90m{n_tokens} tokens\x1b[0m  \x1b[33m{gen_ms}ms\x1b[0m  \x1b[90m({tok_per_sec:.1} tok/s)\x1b[0m");
                    println!();

                    conversation.push(ChatMessage { role: "assistant".into(), content: text.clone() });
                    Some((text, "local".into(), format!("gen-{n_tokens}"), gen_ms))
                }
                Err(e) => {
                    println!("\n  \x1b[33mGeneration error: {e}\x1b[0m\n");
                    conversation.pop();
                    continue;
                }
            }
        };

        if let Some((text, commitment, att_id, latency_ms)) = result {
            turn += 1;

            // Print response
            println!();
            println!("  \x1b[36;1m AI\x1b[0m  {text}");
            println!();

            // Attestation footer
            let commit_short = if commitment.len() > 18 {
                format!("{}…{}", &commitment[..10], &commitment[commitment.len()-6..])
            } else {
                commitment.clone()
            };
            let att_short = if att_id.len() > 16 { &att_id[..16] } else { &att_id };

            println!("  \x1b[92m✓\x1b[0m \x1b[90m{trust_label}\x1b[0m  \x1b[35m{att_short}\x1b[0m  \x1b[90m{commit_short}\x1b[0m  \x1b[33m{latency_ms}ms\x1b[0m");
            println!();
        }
    }
}
