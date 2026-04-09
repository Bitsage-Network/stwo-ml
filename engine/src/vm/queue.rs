//! Async proving queue for the ObelyZK VM.
//!
//! Manages proof jobs across multiple GPU workers. Jobs are submitted
//! by the executor (or API handler) and processed asynchronously.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::vm::trace::ExecutionTrace;
use crate::weight_cache::SharedWeightCache;

/// Status of a proving job.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum ProvingStatus {
    Queued,
    Proving,
    Complete {
        prove_time_ms: u64,
        calldata_size: usize,
        io_commitment: String,
        proof_hash: String,
    },
    Failed {
        error: String,
    },
}

/// A proving job to be processed by a GPU worker.
pub struct ProvingJob {
    pub job_id: String,
    pub trace: ExecutionTrace,
    /// The original input matrix (embedding) for re-proving from scratch.
    pub input_matrix: Option<crate::components::matmul::M31Matrix>,
    /// Pre-computed forward pass result for trace replay proving (no re-execution).
    pub forward_result: Option<crate::aggregation::ForwardPassResult>,
    pub graph: Arc<ComputationGraph>,
    pub weights: Arc<GraphWeights>,
    pub weight_cache: Option<SharedWeightCache>,
    pub submitted_at: Instant,
    pub callback_url: Option<String>,
}

/// Result stored after a job completes.
pub struct ProvingResult {
    pub job_id: String,
    pub status: ProvingStatus,
    pub completed_at: Option<Instant>,
    pub proof: Option<Box<crate::aggregation::AggregatedModelProofOnChain>>,
}

/// Thread-safe proving queue.
///
/// Submit jobs via `submit()`, poll results via `get_status()`.
/// Workers are spawned separately and pull from the internal channel.
pub struct ProvingQueue {
    tx: std::sync::mpsc::Sender<ProvingJob>,
    results: Arc<std::sync::RwLock<HashMap<String, ProvingResult>>>,
}

impl ProvingQueue {
    /// Create a new queue with `num_workers` background threads.
    ///
    /// Each worker runs `prove_from_trace()` on the next available job.
    pub fn new(num_workers: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<ProvingJob>();
        let rx = Arc::new(std::sync::Mutex::new(rx));
        let results: Arc<std::sync::RwLock<HashMap<String, ProvingResult>>> =
            Arc::new(std::sync::RwLock::new(HashMap::new()));

        for worker_id in 0..num_workers {
            let rx = Arc::clone(&rx);
            let results = Arc::clone(&results);

            std::thread::Builder::new()
                .name(format!("prove-worker-{worker_id}"))
                .spawn(move || {
                    eprintln!("[prove-worker-{worker_id}] started");
                    loop {
                        let job = match rx.lock().unwrap().recv() {
                            Ok(j) => j,
                            Err(_) => break, // channel closed
                        };

                        let job_id = job.job_id.clone();
                        eprintln!(
                            "[prove-worker-{worker_id}] proving job {} ({} tokens)",
                            job_id, job.trace.num_tokens
                        );

                        // Mark as proving
                        results.write().unwrap().insert(
                            job_id.clone(),
                            ProvingResult {
                                job_id: job_id.clone(),
                                status: ProvingStatus::Proving,
                                completed_at: None,
                                proof: None,
                            },
                        );

                        let t_start = Instant::now();
                        // Priority order:
                        // 1. Pre-computed proof in trace (synchronous path)
                        // 2. Forward result (trace replay — no re-execution)
                        // 3. Input matrix (full re-execution + proving)
                        let result: Result<crate::aggregation::AggregatedModelProofOnChain, String> =
                            if let Some(proof) = job.trace.proof {
                                Ok(proof)
                            } else if let (Some(fwd), Some(ref input)) = (job.forward_result, &job.input_matrix) {
                                // Trace replay: prove from captured forward pass
                                eprintln!("[prove-worker-{worker_id}] using trace replay path");
                                crate::aggregation::prove_from_forward_result(
                                    &job.graph, input, &job.weights, fwd,
                                    job.weight_cache.as_ref(), None,
                                ).map_err(|e| format!("{e}"))
                            } else if let Some(ref input) = job.input_matrix {
                                // Full re-execution + proving
                                crate::aggregation::prove_model_pure_gkr_auto_with_cache(
                                    &job.graph, input, &job.weights,
                                    job.weight_cache.as_ref(), None,
                                ).map_err(|e| format!("{e}"))
                            } else {
                                Err("No proof, no forward_result, and no input_matrix".into())
                            };

                        let proving_result = match result {
                            Ok(proof) => {
                                let prove_time_ms = t_start.elapsed().as_millis() as u64;
                                let calldata_size = proof
                                    .gkr_proof
                                    .as_ref()
                                    .map(|g| g.layer_proofs.len() * 100) // approximate
                                    .unwrap_or(0);
                                let io_hex = format!("0x{:x}", proof.io_commitment);
                                let proof_hash = format!(
                                    "0x{:x}",
                                    starknet_crypto::poseidon_hash_many(&[
                                        proof.io_commitment,
                                        proof.layer_chain_commitment,
                                    ])
                                );

                                eprintln!(
                                    "[prove-worker-{worker_id}] job {} complete in {prove_time_ms}ms",
                                    job_id
                                );

                                ProvingResult {
                                    job_id: job_id.clone(),
                                    status: ProvingStatus::Complete {
                                        prove_time_ms,
                                        calldata_size,
                                        io_commitment: io_hex,
                                        proof_hash,
                                    },
                                    completed_at: Some(Instant::now()),
                                    proof: Some(Box::new(proof)),
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "[prove-worker-{worker_id}] job {} FAILED: {e}",
                                    job_id
                                );
                                ProvingResult {
                                    job_id: job_id.clone(),
                                    status: ProvingStatus::Failed {
                                        error: format!("{e}"),
                                    },
                                    completed_at: Some(Instant::now()),
                                    proof: None,
                                }
                            }
                        };

                        results.write().unwrap().insert(job_id, proving_result);
                    }
                    eprintln!("[prove-worker-{worker_id}] stopped");
                })
                .expect("failed to spawn prove worker");
        }

        Self { tx, results }
    }

    /// Submit a proving job. Returns immediately.
    pub fn submit(&self, job: ProvingJob) -> Result<(), String> {
        let job_id = job.job_id.clone();
        // Mark as queued
        self.results.write().unwrap().insert(
            job_id.clone(),
            ProvingResult {
                job_id,
                status: ProvingStatus::Queued,
                completed_at: None,
                proof: None,
            },
        );
        self.tx
            .send(job)
            .map_err(|e| format!("Queue send failed: {e}"))
    }

    /// Get the status of a proving job.
    pub fn get_status(&self, job_id: &str) -> Option<ProvingStatus> {
        self.results
            .read()
            .unwrap()
            .get(job_id)
            .map(|r| r.status.clone())
    }

    /// Get the number of pending + in-progress jobs.
    pub fn queue_depth(&self) -> usize {
        self.results
            .read()
            .unwrap()
            .values()
            .filter(|r| matches!(r.status, ProvingStatus::Queued | ProvingStatus::Proving))
            .count()
    }
}
