//! Worker mode — connects prove-server to the BitSage coordinator.
//!
//! When started with `COORDINATOR_URL=https://api.bitsage.network`,
//! prove-server registers as a GPU worker, sends heartbeats, and
//! accepts proof jobs via WebSocket from the coordinator.
//!
//! ```text
//! prove-server                         coordinator
//!     │                                     │
//!     ├── POST /workers/gpu/register ──────►│  (on startup)
//!     │                                     │
//!     ├── POST /workers/heartbeat ─────────►│  (every 30s)
//!     │                                     │
//!     │◄── WS /ws/worker ──────────────────┤  (job assignments)
//!     │    { job_id, model_id, input }      │
//!     │                                     │
//!     ├── POST /workers/job/:id/result ────►│  (proof complete)
//!     │    { proof_hash, calldata, ... }    │
//!     │                                     │
//!     ├── POST /workers/deregister ────────►│  (on shutdown)
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::watch;
use tracing::{error, info, warn};

// =============================================================================
// Configuration
// =============================================================================

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Coordinator URL (e.g., "https://api.bitsage.network")
    pub coordinator_url: String,
    /// This worker's public address (e.g., "http://prover.bitsage.network:8080")
    pub worker_address: String,
    /// Operator's Starknet wallet address for earnings
    pub wallet_address: String,
    /// GPU model name (auto-detected if empty)
    pub gpu_model: String,
    /// VRAM in GB (auto-detected if 0)
    pub vram_gb: u32,
    /// Worker name (optional, for dashboard display)
    pub worker_name: String,
    /// Region hint (e.g., "us-west-2")
    pub region: String,
    /// Heartbeat interval in seconds
    pub heartbeat_secs: u64,
    /// Prover version string
    pub prover_version: String,
}

impl WorkerConfig {
    /// Build config from environment variables.
    pub fn from_env() -> Option<Self> {
        let coordinator_url = std::env::var("COORDINATOR_URL").ok()?;

        let wallet = std::env::var("WORKER_WALLET")
            .or_else(|_| std::env::var("STARKNET_ACCOUNT_ADDRESS"))
            .unwrap_or_default();

        let worker_address = std::env::var("WORKER_ADDRESS")
            .unwrap_or_else(|_| {
                let port = std::env::var("BIND_ADDR")
                    .unwrap_or_else(|_| "0.0.0.0:8080".into());
                format!("http://{port}")
            });

        // Auto-detect GPU
        let (gpu_model, vram_gb) = detect_gpu();

        Some(Self {
            coordinator_url,
            worker_address,
            wallet_address: wallet,
            gpu_model,
            vram_gb,
            worker_name: std::env::var("WORKER_NAME").unwrap_or_default(),
            region: std::env::var("WORKER_REGION").unwrap_or_default(),
            heartbeat_secs: std::env::var("HEARTBEAT_INTERVAL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30),
            prover_version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }
}

fn detect_gpu() -> (String, u32) {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = s.trim().splitn(2, ',').collect();
            if parts.len() == 2 {
                let name = parts[0].trim().to_string();
                let vram = parts[1].trim().parse::<f32>().unwrap_or(0.0);
                return (name, (vram / 1024.0).ceil() as u32);
            }
        }
    }
    ("unknown".into(), 0)
}

// =============================================================================
// Registration & Heartbeat
// =============================================================================

#[derive(Serialize)]
struct RegisterRequest {
    worker_id: String,
    address: String,
    gpu_model: String,
    vram_gb: u32,
    gpu_backend: String,
    capacity: u32,
    capabilities: serde_json::Value,
    owner_wallet: String,
}

#[derive(Deserialize)]
struct RegisterResponseOuter {
    #[serde(default)]
    success: bool,
    data: Option<RegisterResponseData>,
}

#[derive(Deserialize)]
struct RegisterResponseData {
    worker_id: String,
    #[serde(default)]
    ws_url: String,
    #[serde(default)]
    heartbeat_interval_ms: u64,
}

#[derive(Serialize)]
struct HeartbeatRequest {
    worker_id: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_loaded: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_workload: Option<String>,
    metrics: HeartbeatMetrics,
}

#[derive(Serialize)]
struct HeartbeatMetrics {
    gpu_utilization: f32,
    memory_used: u64,
    memory_total: u64,
    temperature: f32,
    proofs_per_hour: u32,
}

#[derive(Serialize)]
struct JobResultSubmission {
    worker_id: String,
    job_id: String,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    io_commitment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight_commitment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prove_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    calldata_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Incoming job assignment from coordinator via WebSocket.
#[derive(Deserialize, Debug)]
pub struct JobAssignment {
    pub job_id: String,
    pub model_id: String,
    pub input: Vec<f32>,
    #[serde(default)]
    pub gpu: bool,
    #[serde(default)]
    pub layers: Option<usize>,
}

/// Register this worker with the coordinator.
async fn register(config: &WorkerConfig) -> Result<String, String> {
    let url = format!("{}/api/v1/workers/gpu/register", config.coordinator_url);
    let worker_id = format!("{}-{}", config.worker_name, uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("0000"));

    let body = RegisterRequest {
        worker_id: worker_id.clone(),
        address: config.worker_address.clone(),
        gpu_model: config.gpu_model.clone(),
        vram_gb: config.vram_gb,
        gpu_backend: "cuda".to_string(),
        capacity: 1,
        capabilities: serde_json::json!({
            "cuda": true,
            "tee": false,
            "max_concurrent_jobs": 1,
            "supported_circuits": ["AiInference", "GenericCompute"],
        }),
        owner_wallet: config.wallet_address.clone(),
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Registration failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Registration rejected: {status} — {text}"));
    }

    let reg: RegisterResponseOuter = resp.json().await
        .map_err(|e| format!("Bad registration response: {e}"))?;

    match reg.data {
        Some(data) => Ok(data.worker_id),
        None => Err("Registration response missing data".into()),
    }
}

/// Send a heartbeat to the coordinator.
async fn send_heartbeat(
    config: &WorkerConfig,
    worker_id: &str,
    model_loaded: Option<&str>,
    active_jobs: u32,
) {
    let url = format!("{}/api/v1/workers/heartbeat", config.coordinator_url);
    let body = HeartbeatRequest {
        worker_id: worker_id.to_string(),
        status: if active_jobs > 0 { "busy".into() } else { "online".into() },
        model_loaded: model_loaded.map(|s| s.to_string()),
        active_workload: None,
        metrics: HeartbeatMetrics {
            gpu_utilization: 0.0,
            memory_used: 0,
            memory_total: 0,
            temperature: 0.0,
            proofs_per_hour: 0,
        },
    };

    let client = reqwest::Client::new();
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => {},
        Ok(resp) => warn!("Heartbeat rejected: {}", resp.status()),
        Err(e) => warn!("Heartbeat failed: {e}"),
    }
}

/// Deregister this worker from the coordinator.
async fn deregister(config: &WorkerConfig, worker_id: &str) {
    let url = format!("{}/api/v1/workers/deregister", config.coordinator_url);
    let body = serde_json::json!({ "worker_id": worker_id });

    let client = reqwest::Client::new();
    match client.post(&url).json(&body).send().await {
        Ok(_) => info!("Deregistered from coordinator"),
        Err(e) => warn!("Deregister failed: {e}"),
    }
}

/// Submit a completed job result to the coordinator.
pub async fn submit_job_result(
    config: &WorkerConfig,
    worker_id: &str,
    job_id: &str,
    result: Result<JobProofResult, String>,
) {
    let url = format!("{}/api/v1/workers/job/{job_id}/result", config.coordinator_url);

    let body = match result {
        Ok(proof) => JobResultSubmission {
            worker_id: worker_id.to_string(),
            job_id: job_id.to_string(),
            status: "completed".into(),
            proof_hash: Some(proof.proof_hash),
            io_commitment: Some(proof.io_commitment),
            weight_commitment: Some(proof.weight_commitment),
            prove_time_ms: Some(proof.prove_time_ms),
            calldata_size: Some(proof.calldata_size),
            error: None,
        },
        Err(err) => JobResultSubmission {
            worker_id: worker_id.to_string(),
            job_id: job_id.to_string(),
            status: "failed".into(),
            proof_hash: None,
            io_commitment: None,
            weight_commitment: None,
            prove_time_ms: None,
            calldata_size: None,
            error: Some(err),
        },
    };

    let client = reqwest::Client::new();
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => {
            info!("Job {job_id} result submitted to coordinator");
        },
        Ok(resp) => warn!("Job result rejected: {}", resp.status()),
        Err(e) => error!("Failed to submit job result: {e}"),
    }
}

/// Proof result from a completed job.
pub struct JobProofResult {
    pub proof_hash: String,
    pub io_commitment: String,
    pub weight_commitment: String,
    pub prove_time_ms: u64,
    pub calldata_size: usize,
}

// =============================================================================
// Worker lifecycle
// =============================================================================

/// Start the worker mode background tasks.
///
/// This spawns:
/// 1. Registration with the coordinator
/// 2. Periodic heartbeat sender
/// 3. WebSocket listener for job assignments
///
/// Returns a channel sender that the main prove loop can use to report
/// completed jobs back to the coordinator.
pub async fn start_worker(
    config: WorkerConfig,
    shutdown: watch::Receiver<bool>,
) -> Result<WorkerHandle, String> {
    // Register
    info!(
        "Registering with coordinator: {} (GPU: {}, VRAM: {}GB, wallet: {})",
        config.coordinator_url, config.gpu_model, config.vram_gb,
        if config.wallet_address.is_empty() { "none" } else { &config.wallet_address }
    );

    let worker_id = register(&config).await?;
    info!("Registered as worker {worker_id}");

    let config = Arc::new(config);
    let worker_id = Arc::new(worker_id);

    // Job channel: coordinator pushes jobs, prove-server processes them
    let (job_tx, job_rx) = tokio::sync::mpsc::channel::<JobAssignment>(16);

    // Heartbeat task
    let hb_config = config.clone();
    let hb_worker_id = worker_id.clone();
    let mut hb_shutdown = shutdown.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(hb_config.heartbeat_secs)
        );
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    send_heartbeat(&hb_config, &hb_worker_id, None, 0).await;
                }
                _ = hb_shutdown.changed() => {
                    info!("Heartbeat task shutting down");
                    deregister(&hb_config, &hb_worker_id).await;
                    break;
                }
            }
        }
    });

    // WebSocket listener for job assignments
    let ws_config = config.clone();
    let ws_worker_id = worker_id.clone();
    let ws_job_tx = job_tx.clone();
    let mut ws_shutdown = shutdown.clone();
    tokio::spawn(async move {
        loop {
            let ws_url = format!(
                "{}/ws/worker?worker_id={}",
                ws_config.coordinator_url.replace("https://", "wss://").replace("http://", "ws://"),
                ws_worker_id,
            );

            info!("Connecting to coordinator WebSocket: {ws_url}");

            match tokio_tungstenite::connect_async(&ws_url).await {
                Ok((mut ws, _)) => {
                    info!("WebSocket connected to coordinator");
                    use futures_util::StreamExt;

                    loop {
                        tokio::select! {
                            msg = ws.next() => {
                                match msg {
                                    Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                                        match serde_json::from_str::<JobAssignment>(&text) {
                                            Ok(job) => {
                                                info!("Received job {} (model: {})", job.job_id, job.model_id);
                                                if ws_job_tx.send(job).await.is_err() {
                                                    error!("Job channel full, dropping job");
                                                }
                                            }
                                            Err(e) => warn!("Invalid job message: {e}"),
                                        }
                                    }
                                    Some(Ok(_)) => {} // ping/pong/binary
                                    Some(Err(e)) => {
                                        warn!("WebSocket error: {e}");
                                        break;
                                    }
                                    None => {
                                        warn!("WebSocket closed by coordinator");
                                        break;
                                    }
                                }
                            }
                            _ = ws_shutdown.changed() => {
                                info!("WebSocket listener shutting down");
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("WebSocket connection failed: {e}");
                }
            }

            // Reconnect after 5s
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });

    Ok(WorkerHandle {
        worker_id: (*worker_id).clone(),
        config: (*config).clone(),
        job_rx,
    })
}

/// Handle returned by `start_worker` — the prove-server main loop reads jobs from this.
pub struct WorkerHandle {
    pub worker_id: String,
    pub config: WorkerConfig,
    pub job_rx: tokio::sync::mpsc::Receiver<JobAssignment>,
}
