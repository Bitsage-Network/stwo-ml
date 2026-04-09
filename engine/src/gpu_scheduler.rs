//! GPU-aware job scheduler with bounded concurrency and backpressure.
//!
//! Provides a queue that dispatches proving jobs to the least-loaded GPU,
//! enforces per-GPU concurrency limits via semaphores, and returns 429
//! when the queue is full.
//!
//! ```text
//! POST /prove ──► GpuScheduler.submit() ──► mpsc queue ──► dispatch loop
//!                                                              │
//!                                            ┌─────────────────┤
//!                                            ▼                 ▼
//!                                        GPU 0 (sem)       GPU 1 (sem)
//!                                        spawn_blocking    spawn_blocking
//! ```

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{error, info};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the GPU scheduler.
#[derive(Debug, Clone)]
pub struct GpuSchedulerConfig {
    /// Maximum concurrent proofs per GPU device (default: 1).
    pub max_concurrent_per_gpu: usize,
    /// Maximum queued jobs before rejecting with 429 (default: 64).
    pub max_queue_depth: usize,
    /// Explicit GPU device ordinals. Empty = auto-detect.
    pub device_ordinals: Vec<usize>,
}

impl Default for GpuSchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_per_gpu: 1,
            max_queue_depth: 64,
            device_ordinals: Vec::new(),
        }
    }
}

// =============================================================================
// Types
// =============================================================================

/// Result of a prove job returned through the oneshot channel.
#[derive(Debug)]
pub struct ProveJobResult {
    /// Serialized proof data (format depends on the caller).
    pub data: Vec<u8>,
    /// Wall-clock proving time in milliseconds.
    pub prove_time_ms: u64,
    /// GPU device that executed the job.
    pub device_id: usize,
}

/// Error returned when the job queue is full.
#[derive(Debug)]
pub struct SchedulerFullError {
    pub queue_depth: usize,
    pub max_depth: usize,
}

impl std::fmt::Display for SchedulerFullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Job queue full ({}/{}). Try again later.",
            self.queue_depth, self.max_depth
        )
    }
}

impl std::error::Error for SchedulerFullError {}

/// A job submitted to the scheduler.
pub struct ScheduledJob {
    /// Unique job identifier.
    pub job_id: String,
    /// Estimated GPU memory usage in bytes (advisory, for future memory-aware scheduling).
    pub estimated_gpu_memory: usize,
    /// The proving closure. Receives the assigned GPU device ordinal.
    pub prove_fn: Box<dyn FnOnce(usize) -> Result<ProveJobResult, String> + Send>,
    /// Channel to send the result back to the caller.
    pub result_tx: oneshot::Sender<Result<ProveJobResult, String>>,
    /// When the job was submitted.
    pub submitted_at: Instant,
}

/// Atomic snapshot of queue statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueueStatsSnapshot {
    pub queued: usize,
    pub active: usize,
    pub completed: u64,
    pub failed: u64,
    pub total_prove_time_ms: u64,
    pub avg_prove_time_ms: u64,
    pub per_gpu: Vec<GpuStatsSnapshot>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuStatsSnapshot {
    pub device_id: usize,
    pub active_jobs: usize,
    pub memory_used: usize,
    pub total_memory: usize,
}

/// Shared atomic counters for queue statistics.
pub struct QueueStats {
    pub queued: AtomicUsize,
    pub active: AtomicUsize,
    pub completed: AtomicU64,
    pub failed: AtomicU64,
    pub total_prove_time_ms: AtomicU64,
    pub per_gpu_active: Vec<AtomicUsize>,
    pub per_gpu_memory_used: Vec<AtomicUsize>,
    pub per_gpu_total_memory: Vec<usize>,
}

impl QueueStats {
    fn new(gpu_count: usize, total_memories: Vec<usize>) -> Self {
        let per_gpu_active = (0..gpu_count).map(|_| AtomicUsize::new(0)).collect();
        let per_gpu_memory_used = (0..gpu_count).map(|_| AtomicUsize::new(0)).collect();
        Self {
            queued: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            total_prove_time_ms: AtomicU64::new(0),
            per_gpu_active,
            per_gpu_memory_used,
            per_gpu_total_memory: total_memories,
        }
    }

    /// Take an atomic snapshot of the current stats.
    pub fn snapshot(&self) -> QueueStatsSnapshot {
        let completed = self.completed.load(Ordering::Relaxed);
        let total_ms = self.total_prove_time_ms.load(Ordering::Relaxed);
        let avg = if completed > 0 { total_ms / completed } else { 0 };

        let per_gpu = self
            .per_gpu_active
            .iter()
            .zip(self.per_gpu_memory_used.iter())
            .zip(self.per_gpu_total_memory.iter())
            .enumerate()
            .map(|(i, ((active, mem_used), &total_mem))| GpuStatsSnapshot {
                device_id: i,
                active_jobs: active.load(Ordering::Relaxed),
                memory_used: mem_used.load(Ordering::Relaxed),
                total_memory: total_mem,
            })
            .collect();

        QueueStatsSnapshot {
            queued: self.queued.load(Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            completed,
            failed: self.failed.load(Ordering::Relaxed),
            total_prove_time_ms: total_ms,
            avg_prove_time_ms: avg,
            per_gpu,
        }
    }
}

// =============================================================================
// GPU Slot
// =============================================================================

struct GpuSlot {
    device_id: usize,
    #[allow(dead_code)]
    total_memory: usize,
    permit: Arc<Semaphore>,
}

// =============================================================================
// Scheduler
// =============================================================================

/// GPU-aware job scheduler with bounded concurrency and FIFO dispatch.
pub struct GpuScheduler {
    #[allow(dead_code)]
    config: GpuSchedulerConfig,
    job_tx: mpsc::Sender<ScheduledJob>,
    /// Shared queue statistics (atomic, lock-free reads).
    pub stats: Arc<QueueStats>,
    gpu_count: usize,
}

impl GpuScheduler {
    /// Create a new scheduler and spawn the dispatch loop.
    ///
    /// GPU discovery uses `multi_gpu::discover_devices()` when available,
    /// falling back to a single CPU slot when no GPUs are found.
    pub fn new(config: GpuSchedulerConfig) -> Self {
        let (gpu_slots, total_memories) = Self::build_slots(&config);
        let gpu_count = gpu_slots.len();
        let stats = Arc::new(QueueStats::new(gpu_count, total_memories));

        let (job_tx, job_rx) = mpsc::channel::<ScheduledJob>(config.max_queue_depth);

        // Spawn the dispatch loop
        let dispatch_stats = Arc::clone(&stats);
        tokio::spawn(Self::dispatch_loop(job_rx, gpu_slots, dispatch_stats));

        info!(
            gpus = gpu_count,
            max_per_gpu = config.max_concurrent_per_gpu,
            max_queue = config.max_queue_depth,
            "GPU scheduler started"
        );

        Self {
            config,
            job_tx,
            stats,
            gpu_count,
        }
    }

    /// Build GPU slots from config, falling back to CPU-only.
    fn build_slots(config: &GpuSchedulerConfig) -> (Vec<GpuSlot>, Vec<usize>) {
        // Discover GPUs when multi-gpu feature is available
        let discovered: Vec<(usize, usize)> = {
            #[cfg(feature = "multi-gpu")]
            {
                crate::multi_gpu::discover_devices()
                    .into_iter()
                    .map(|d| (d.ordinal, d.total_memory))
                    .collect()
            }
            #[cfg(not(feature = "multi-gpu"))]
            {
                Vec::new()
            }
        };

        let devices: Vec<(usize, usize)> = if !config.device_ordinals.is_empty() {
            // Use explicitly configured devices
            config
                .device_ordinals
                .iter()
                .map(|&ord| {
                    let mem = discovered
                        .iter()
                        .find(|(o, _)| *o == ord)
                        .map(|(_, m)| *m)
                        .unwrap_or(0);
                    (ord, mem)
                })
                .collect()
        } else if discovered.is_empty() {
            // CPU-only fallback: single slot
            vec![(0, 0)]
        } else {
            discovered
        };

        let mut slots = Vec::with_capacity(devices.len());
        let mut memories = Vec::with_capacity(devices.len());

        for (ord, mem) in devices {
            slots.push(GpuSlot {
                device_id: ord,
                total_memory: mem,
                permit: Arc::new(Semaphore::new(config.max_concurrent_per_gpu)),
            });
            memories.push(mem);
        }

        (slots, memories)
    }

    /// Submit a job to the scheduler. Returns `Err(SchedulerFullError)` if the queue is full.
    pub fn submit(&self, job: ScheduledJob) -> Result<(), SchedulerFullError> {
        self.stats.queued.fetch_add(1, Ordering::Relaxed);
        match self.job_tx.try_send(job) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(job)) => {
                self.stats.queued.fetch_sub(1, Ordering::Relaxed);
                // Drop the job, send error through its channel
                let _ = job.result_tx.send(Err("Queue full".to_string()));
                Err(SchedulerFullError {
                    queue_depth: self.config.max_queue_depth,
                    max_depth: self.config.max_queue_depth,
                })
            }
            Err(mpsc::error::TrySendError::Closed(job)) => {
                self.stats.queued.fetch_sub(1, Ordering::Relaxed);
                let _ = job.result_tx.send(Err("Scheduler shut down".to_string()));
                Err(SchedulerFullError {
                    queue_depth: 0,
                    max_depth: self.config.max_queue_depth,
                })
            }
        }
    }

    /// Number of GPU slots (or 1 for CPU-only).
    pub fn gpu_count(&self) -> usize {
        self.gpu_count
    }

    /// Dispatch loop: receives jobs from the channel, selects the least-loaded
    /// GPU, acquires its semaphore, and spawns the proving task.
    async fn dispatch_loop(
        mut job_rx: mpsc::Receiver<ScheduledJob>,
        gpu_slots: Vec<GpuSlot>,
        stats: Arc<QueueStats>,
    ) {
        while let Some(job) = job_rx.recv().await {
            stats.queued.fetch_sub(1, Ordering::Relaxed);

            // Select GPU with fewest active jobs (uses dispatch-time counters)
            let slot_idx = Self::select_gpu(&stats, gpu_slots.len());
            let slot = &gpu_slots[slot_idx];
            let permit = Arc::clone(&slot.permit);
            let device_id = slot.device_id;
            let job_stats = Arc::clone(&stats);
            let job_id = job.job_id.clone();
            let estimated_mem = job.estimated_gpu_memory;

            // Track active state immediately in the dispatch loop so that
            // subsequent select_gpu calls see the updated count.
            stats.active.fetch_add(1, Ordering::Relaxed);
            stats.per_gpu_active[slot_idx].fetch_add(1, Ordering::Relaxed);
            stats.per_gpu_memory_used[slot_idx].fetch_add(estimated_mem, Ordering::Relaxed);

            // Spawn a task that acquires the GPU permit, then runs the job
            tokio::spawn(async move {
                // Acquire semaphore permit (blocks if GPU is at capacity)
                let _permit = match permit.acquire().await {
                    Ok(p) => p,
                    Err(_) => {
                        // Release counters on failure
                        job_stats.active.fetch_sub(1, Ordering::Relaxed);
                        job_stats.per_gpu_active[slot_idx].fetch_sub(1, Ordering::Relaxed);
                        job_stats.per_gpu_memory_used[slot_idx]
                            .fetch_sub(estimated_mem, Ordering::Relaxed);
                        let _ = job.result_tx.send(Err("Semaphore closed".to_string()));
                        return;
                    }
                };

                let start = Instant::now();
                info!(job_id = %job_id, device = device_id, "Dispatching prove job");

                // Run the proving closure on a blocking thread with GPU affinity
                let result = tokio::task::spawn_blocking(move || {
                    #[cfg(feature = "multi-gpu")]
                    let _guard = crate::multi_gpu::DeviceGuard::new(device_id);
                    (job.prove_fn)(device_id)
                })
                .await;

                let elapsed_ms = start.elapsed().as_millis() as u64;

                // Release active counters
                job_stats.active.fetch_sub(1, Ordering::Relaxed);
                job_stats.per_gpu_active[slot_idx].fetch_sub(1, Ordering::Relaxed);
                job_stats.per_gpu_memory_used[slot_idx]
                    .fetch_sub(estimated_mem, Ordering::Relaxed);

                match result {
                    Ok(Ok(ref _res)) => {
                        job_stats.completed.fetch_add(1, Ordering::Relaxed);
                        job_stats
                            .total_prove_time_ms
                            .fetch_add(elapsed_ms, Ordering::Relaxed);
                        info!(job_id = %job_id, device = device_id, elapsed_ms, "Job completed");
                    }
                    Ok(Err(ref e)) => {
                        job_stats.failed.fetch_add(1, Ordering::Relaxed);
                        error!(job_id = %job_id, device = device_id, error = %e, "Job failed");
                    }
                    Err(ref e) => {
                        job_stats.failed.fetch_add(1, Ordering::Relaxed);
                        error!(job_id = %job_id, device = device_id, error = %e, "Job panicked");
                    }
                }

                // Send result back to caller
                let final_result = match result {
                    Ok(r) => r,
                    Err(e) => Err(format!("Task panicked: {e}")),
                };
                let _ = job.result_tx.send(final_result);

                // _permit dropped here, releasing the semaphore
            });
        }

        info!("GPU scheduler dispatch loop exited");
    }

    /// Select GPU with fewest active jobs.
    fn select_gpu(stats: &QueueStats, gpu_count: usize) -> usize {
        let mut best_idx = 0;
        let mut best_active = usize::MAX;

        for i in 0..gpu_count {
            let active = stats.per_gpu_active[i].load(Ordering::Relaxed);
            if active < best_active {
                best_active = active;
                best_idx = i;
            }
        }

        best_idx
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
    use std::time::Duration;

    fn test_config(max_per_gpu: usize, max_queue: usize) -> GpuSchedulerConfig {
        GpuSchedulerConfig {
            max_concurrent_per_gpu: max_per_gpu,
            max_queue_depth: max_queue,
            device_ordinals: vec![0], // Force single CPU slot for tests
        }
    }

    #[tokio::test]
    async fn test_submit_and_complete() {
        let scheduler = GpuScheduler::new(test_config(2, 8));

        let (result_tx, result_rx) = oneshot::channel();
        let job = ScheduledJob {
            job_id: "test-1".to_string(),
            estimated_gpu_memory: 0,
            prove_fn: Box::new(|device_id| {
                Ok(ProveJobResult {
                    data: vec![1, 2, 3],
                    prove_time_ms: 42,
                    device_id,
                })
            }),
            result_tx,
            submitted_at: Instant::now(),
        };

        scheduler.submit(job).unwrap();
        let result = result_rx.await.unwrap().unwrap();
        assert_eq!(result.data, vec![1, 2, 3]);
        assert_eq!(result.prove_time_ms, 42);

        // Wait a tick for stats to settle
        tokio::time::sleep(Duration::from_millis(10)).await;
        let snap = scheduler.stats.snapshot();
        assert_eq!(snap.completed, 1);
        assert_eq!(snap.active, 0);
    }

    #[tokio::test]
    async fn test_queue_full_rejection() {
        // Queue depth of 1, with a job that blocks
        let scheduler = GpuScheduler::new(test_config(1, 1));

        let (block_tx, block_rx) = std::sync::mpsc::channel::<()>();
        let (result_tx1, _result_rx1) = oneshot::channel();
        let blocking_job = ScheduledJob {
            job_id: "blocker".to_string(),
            estimated_gpu_memory: 0,
            prove_fn: Box::new(move |device_id| {
                block_rx.recv().ok(); // Block until signaled
                Ok(ProveJobResult {
                    data: vec![],
                    prove_time_ms: 0,
                    device_id,
                })
            }),
            result_tx: result_tx1,
            submitted_at: Instant::now(),
        };

        scheduler.submit(blocking_job).unwrap();
        // Give the dispatch loop a moment to pick up the first job
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Now the queue should be empty but the GPU semaphore is held.
        // Fill the queue with one more job
        let (result_tx2, _result_rx2) = oneshot::channel();
        let filler = ScheduledJob {
            job_id: "filler".to_string(),
            estimated_gpu_memory: 0,
            prove_fn: Box::new(|device_id| {
                Ok(ProveJobResult {
                    data: vec![],
                    prove_time_ms: 0,
                    device_id,
                })
            }),
            result_tx: result_tx2,
            submitted_at: Instant::now(),
        };
        scheduler.submit(filler).unwrap();

        // Queue is now full (depth 1). Next submit should fail.
        let (result_tx3, _result_rx3) = oneshot::channel();
        let overflow = ScheduledJob {
            job_id: "overflow".to_string(),
            estimated_gpu_memory: 0,
            prove_fn: Box::new(|device_id| {
                Ok(ProveJobResult {
                    data: vec![],
                    prove_time_ms: 0,
                    device_id,
                })
            }),
            result_tx: result_tx3,
            submitted_at: Instant::now(),
        };
        let err = scheduler.submit(overflow).unwrap_err();
        assert_eq!(err.max_depth, 1);

        // Unblock
        let _ = block_tx.send(());
    }

    #[tokio::test]
    async fn test_fifo_ordering() {
        let scheduler = GpuScheduler::new(test_config(1, 8));
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut receivers = Vec::new();

        for i in 0..3 {
            let (result_tx, result_rx) = oneshot::channel();
            let order_clone = Arc::clone(&order);
            let job = ScheduledJob {
                job_id: format!("job-{i}"),
                estimated_gpu_memory: 0,
                prove_fn: Box::new(move |device_id| {
                    order_clone.lock().unwrap().push(i);
                    Ok(ProveJobResult {
                        data: vec![i as u8],
                        prove_time_ms: 0,
                        device_id,
                    })
                }),
                result_tx,
                submitted_at: Instant::now(),
            };
            scheduler.submit(job).unwrap();
            receivers.push(result_rx);
        }

        // Wait for all to complete
        for rx in receivers {
            rx.await.unwrap().unwrap();
        }

        let completed_order = order.lock().unwrap();
        assert_eq!(*completed_order, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let scheduler = GpuScheduler::new(test_config(4, 16));

        let mut receivers = Vec::new();
        for i in 0..5 {
            let (result_tx, result_rx) = oneshot::channel();
            let job = ScheduledJob {
                job_id: format!("stats-{i}"),
                estimated_gpu_memory: 1024,
                prove_fn: Box::new(|device_id| {
                    std::thread::sleep(Duration::from_millis(5));
                    Ok(ProveJobResult {
                        data: vec![],
                        prove_time_ms: 10,
                        device_id,
                    })
                }),
                result_tx,
                submitted_at: Instant::now(),
            };
            scheduler.submit(job).unwrap();
            receivers.push(result_rx);
        }

        for rx in receivers {
            rx.await.unwrap().unwrap();
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
        let snap = scheduler.stats.snapshot();
        assert_eq!(snap.completed, 5);
        assert_eq!(snap.failed, 0);
        assert_eq!(snap.active, 0);
        assert_eq!(snap.queued, 0);
        assert!(snap.total_prove_time_ms > 0);
    }

    #[tokio::test]
    async fn test_failed_job_tracking() {
        let scheduler = GpuScheduler::new(test_config(2, 8));

        let (result_tx, result_rx) = oneshot::channel();
        let job = ScheduledJob {
            job_id: "fail-1".to_string(),
            estimated_gpu_memory: 0,
            prove_fn: Box::new(|_| Err("intentional failure".to_string())),
            result_tx,
            submitted_at: Instant::now(),
        };

        scheduler.submit(job).unwrap();
        let result = result_rx.await.unwrap();
        assert!(result.is_err());

        tokio::time::sleep(Duration::from_millis(10)).await;
        let snap = scheduler.stats.snapshot();
        assert_eq!(snap.failed, 1);
        assert_eq!(snap.completed, 0);
    }

    #[tokio::test]
    async fn test_gpu_selection_least_loaded() {
        // Use 2 GPU slots with max_concurrent=1.
        // Submit 4 jobs with a small delay, verify both device IDs appear in results.
        let config = GpuSchedulerConfig {
            max_concurrent_per_gpu: 1,
            max_queue_depth: 16,
            device_ordinals: vec![0, 1],
        };
        let scheduler = GpuScheduler::new(config);

        let mut receivers = Vec::new();
        for i in 0..4 {
            let (result_tx, result_rx) = oneshot::channel();
            let job = ScheduledJob {
                job_id: format!("gpu-sel-{i}"),
                estimated_gpu_memory: 0,
                prove_fn: Box::new(move |device_id| {
                    // Simulate some work so the scheduler has time to observe load
                    std::thread::sleep(Duration::from_millis(30));
                    Ok(ProveJobResult {
                        data: vec![device_id as u8],
                        prove_time_ms: 0,
                        device_id,
                    })
                }),
                result_tx,
                submitted_at: Instant::now(),
            };
            scheduler.submit(job).unwrap();
            receivers.push(result_rx);
        }

        let mut device_ids = std::collections::HashSet::new();
        for rx in receivers {
            let result = rx.await.unwrap().unwrap();
            device_ids.insert(result.device_id);
        }

        // With concurrent=1 and blocking jobs, the second slot should be used
        // when the first is busy.
        assert_eq!(device_ids.len(), 2, "Both GPU slots should have been used");
        assert!(device_ids.contains(&0));
        assert!(device_ids.contains(&1));
    }
}
