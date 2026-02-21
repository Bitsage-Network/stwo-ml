//! Multi-GPU distributed proving for large models.
//!
//! Distributes chunk proving across all available GPUs using device-per-chunk
//! parallelism. Thread-local device affinity propagates GPU assignment through
//! the entire proving stack without changing function signatures.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────┐
//! │    MultiGpuExecutor         │
//! │  ┌───────┐  ┌───────┐      │
//! │  │ GPU 0 │  │ GPU 1 │ ...  │
//! │  │ chunk0 │  │ chunk1│      │
//! │  │ chunk3 │  │ chunk2│      │
//! │  └───────┘  └───────┘      │
//! └─────────────────────────────┘
//! ```
//!
//! Each thread sets its device affinity via [`set_thread_device`], and all
//! downstream GPU calls (`GpuSumcheckExecutor::cached()`, etc.) automatically
//! route to the correct device.
//!
//! # Device Affinity Safety
//!
//! Use [`DeviceGuard`] for RAII-based device management. The guard sets the
//! device on creation and restores the previous device on drop — even on panic.
//!
//! ```ignore
//! {
//!     let _guard = DeviceGuard::new(device_id);
//!     // All GPU ops here target device_id
//!     prove_model_aggregated_onchain_auto(&graph, &input, &weights)?;
//! } // device affinity automatically restored
//! ```

use std::cell::Cell;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use tracing::info;

// =============================================================================
// Thread-Local Device Affinity
// =============================================================================

thread_local! {
    static CURRENT_DEVICE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Set the GPU device for the current thread.
///
/// All subsequent GPU operations on this thread will target this device.
/// Prefer [`DeviceGuard`] for scoped, panic-safe device management.
pub fn set_thread_device(device_id: usize) {
    CURRENT_DEVICE.with(|c| c.set(Some(device_id)));
}

/// Get the GPU device for the current thread, if set.
///
/// Returns `None` when no device affinity has been set (default device 0 path).
pub fn get_thread_device() -> Option<usize> {
    CURRENT_DEVICE.with(|c| c.get())
}

/// Clear the GPU device affinity for the current thread.
///
/// After this call, GPU operations will use the default device (0).
pub fn clear_thread_device() {
    CURRENT_DEVICE.with(|c| c.set(None));
}

// =============================================================================
// DeviceGuard — RAII Device Affinity
// =============================================================================

/// RAII guard that sets GPU device affinity on creation and restores
/// the previous affinity on drop.
///
/// This is panic-safe: even if the guarded scope panics, the previous
/// device is restored during stack unwinding.
///
/// # Usage
///
/// ```ignore
/// fn prove_on_gpu(device_id: usize) {
///     let _guard = DeviceGuard::new(device_id);
///     // All GPU ops target device_id
///     do_gpu_work();
/// } // previous device restored here
/// ```
pub struct DeviceGuard {
    previous: Option<usize>,
}

impl DeviceGuard {
    /// Create a new guard that sets the thread's device to `device_id`.
    ///
    /// The previous device affinity is saved and will be restored on drop.
    pub fn new(device_id: usize) -> Self {
        let previous = get_thread_device();
        set_thread_device(device_id);
        Self { previous }
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        match self.previous {
            Some(prev) => set_thread_device(prev),
            None => clear_thread_device(),
        }
    }
}

/// Propagate the current thread's device affinity into a rayon worker.
///
/// Rayon worker threads do NOT inherit `thread_local!` from the parent.
/// Call this at the start of every `par_iter` closure that may invoke GPU ops.
///
/// Returns a `DeviceGuard` so the worker thread's affinity is restored on exit.
///
/// # Usage
///
/// ```ignore
/// let parent_device = get_thread_device();
/// items.par_iter().map(|item| {
///     let _guard = propagate_device(parent_device);
///     do_gpu_work(item) // routes to parent's device
/// }).collect()
/// ```
pub fn propagate_device(parent_device: Option<usize>) -> Option<DeviceGuard> {
    parent_device.map(DeviceGuard::new)
}

// =============================================================================
// Device Discovery
// =============================================================================

/// Information about a single GPU device.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// CUDA device ordinal (0, 1, 2, ...).
    pub ordinal: usize,
    /// Device name (e.g., "NVIDIA H100 80GB HBM3").
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Relative compute power (normalized, 1.0 = A100 baseline).
    pub relative_power: f32,
}

/// Discover all available GPU devices.
///
/// Wraps STWO's `GpuCapabilities::query()` to enumerate devices.
/// Returns an empty vec if no GPUs are found.
pub fn discover_devices() -> Vec<GpuDeviceInfo> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::cuda_executor::get_device_count;
        use stwo::prover::backend::gpu::multi_gpu::GpuCapabilities;

        let count = get_device_count();
        let mut devices = Vec::with_capacity(count);

        for i in 0..count {
            match GpuCapabilities::query(i) {
                Ok(cap) => {
                    devices.push(GpuDeviceInfo {
                        ordinal: i,
                        name: format!(
                            "GPU {} (SM {}.{})",
                            i, cap.compute_capability.0, cap.compute_capability.1
                        ),
                        total_memory: cap.total_memory,
                        compute_capability: cap.compute_capability,
                        sm_count: cap.sm_count,
                        relative_power: cap.relative_power,
                    });
                }
                Err(e) => {
                    tracing::warn!("Failed to query GPU {}: {:?}", i, e);
                }
            }
        }

        devices
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        Vec::new()
    }
}

/// Get the number of available GPU devices.
pub fn device_count() -> usize {
    #[cfg(feature = "cuda-runtime")]
    {
        stwo::prover::backend::gpu::cuda_executor::get_device_count()
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        0
    }
}

// =============================================================================
// Chunk Workload and Assignment
// =============================================================================

/// Describes the work for a single chunk of a model.
#[derive(Debug, Clone)]
pub struct ChunkWorkload {
    /// Index of this chunk in the overall model.
    pub chunk_index: usize,
    /// Estimated peak GPU memory in bytes for proving this chunk.
    pub estimated_memory: usize,
    /// Number of matmul operations in this chunk.
    pub num_matmuls: usize,
    /// Node range in the computation graph.
    pub block_range: Range<usize>,
}

/// Assignment of a chunk to a specific GPU device.
#[derive(Debug, Clone)]
pub struct DeviceAssignment {
    /// GPU device ordinal.
    pub device_id: usize,
    /// Chunk index being assigned.
    pub chunk_index: usize,
}

// =============================================================================
// Proving Result with Device Metrics
// =============================================================================

/// Result from multi-GPU proving with per-device metrics.
#[derive(Debug)]
pub struct MultiGpuProvingResult {
    /// Per-device statistics collected during proving.
    pub device_stats: Vec<DeviceProvingStat>,
    /// Total wall-clock time for proving (all GPUs in parallel).
    pub total_elapsed: std::time::Duration,
}

/// Statistics for a single device's proving work.
#[derive(Debug, Clone)]
pub struct DeviceProvingStat {
    /// Device ordinal.
    pub device_id: usize,
    /// Chunks proven on this device.
    pub chunks_proven: Vec<usize>,
    /// Total proving time on this device.
    pub elapsed: std::time::Duration,
    /// Number of matmuls proven.
    pub matmuls_proven: usize,
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors from multi-GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum MultiGpuError {
    #[error("No GPU devices found")]
    NoDevices,

    #[error("Device {device_id} initialization failed: {message}")]
    DeviceInit { device_id: usize, message: String },

    #[error("Kernel compilation failed on device {device_id}: {message}")]
    KernelCompilation { device_id: usize, message: String },

    #[error("Insufficient memory on device {device_id}: need {needed} bytes, have {available}")]
    InsufficientMemory {
        device_id: usize,
        needed: usize,
        available: usize,
    },

    #[error("Chunk {chunk_index} failed on device {device_id}: {message}")]
    ChunkFailed {
        chunk_index: usize,
        device_id: usize,
        message: String,
    },

    #[error("Multiple chunks failed: {0}")]
    MultipleFailures(String),

    #[error("Warmup validation failed on device {device_id}: {message}")]
    WarmupFailed { device_id: usize, message: String },
}

// =============================================================================
// MultiGpuExecutor
// =============================================================================

/// Coordinator for multi-GPU proving.
///
/// Manages per-device `GpuSumcheckExecutor` instances and assigns chunks
/// to GPUs based on memory-aware bin-packing.
pub struct MultiGpuExecutor {
    /// Per-device executor handles (device_id, executor).
    #[cfg(feature = "cuda-runtime")]
    executors: Vec<(usize, Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>)>,

    /// Device information for all managed GPUs.
    devices: Vec<GpuDeviceInfo>,

    /// Whether warmup validation has been run.
    validated: bool,
}

impl MultiGpuExecutor {
    /// Create a new executor using all available GPUs.
    ///
    /// Compiles CUDA kernels on each device in parallel using `thread::scope`.
    pub fn new() -> Result<Self, MultiGpuError> {
        let devices = discover_devices();
        if devices.is_empty() {
            return Err(MultiGpuError::NoDevices);
        }

        let ordinals: Vec<usize> = devices.iter().map(|d| d.ordinal).collect();
        Self::with_devices_inner(devices, &ordinals)
    }

    /// Create a new executor using specific GPU devices.
    pub fn with_devices(ordinals: &[usize]) -> Result<Self, MultiGpuError> {
        if ordinals.is_empty() {
            return Err(MultiGpuError::NoDevices);
        }

        let all_devices = discover_devices();
        let devices: Vec<GpuDeviceInfo> = ordinals
            .iter()
            .filter_map(|&ord| all_devices.iter().find(|d| d.ordinal == ord).cloned())
            .collect();

        if devices.is_empty() {
            return Err(MultiGpuError::NoDevices);
        }

        Self::with_devices_inner(devices, ordinals)
    }

    fn with_devices_inner(
        devices: Vec<GpuDeviceInfo>,
        ordinals: &[usize],
    ) -> Result<Self, MultiGpuError> {
        #[cfg(feature = "cuda-runtime")]
        {
            use std::sync::Mutex;

            let results: Mutex<
                Vec<Result<(usize, Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>), MultiGpuError>>,
            > = Mutex::new(Vec::with_capacity(ordinals.len()));

            let compile_start = Instant::now();

            // Compile kernels on all devices in parallel
            std::thread::scope(|s| {
                for &device_id in ordinals {
                    let results_ref = &results;
                    s.spawn(move || {
                        info!(device_id, "Compiling sumcheck kernels on GPU");
                        let result =
                            crate::gpu_sumcheck::GpuSumcheckExecutor::cached_for_device(device_id)
                                .map(|exec| (device_id, exec))
                                .map_err(|e| MultiGpuError::KernelCompilation {
                                    device_id,
                                    message: format!("{e}"),
                                });
                        results_ref.lock().unwrap().push(result);
                    });
                }
            });

            let mut executors: Vec<(usize, Arc<crate::gpu_sumcheck::GpuSumcheckExecutor>)> =
                Vec::new();
            for result in results.into_inner().unwrap() {
                executors.push(result?);
            }
            // Sort by device_id for deterministic ordering
            executors.sort_by_key(|(id, _)| *id);

            info!(
                num_gpus = executors.len(),
                compile_ms = compile_start.elapsed().as_millis() as u64,
                "MultiGpuExecutor initialized"
            );

            Ok(Self {
                executors,
                devices,
                validated: false,
            })
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            let _ = ordinals;
            Ok(Self {
                devices,
                validated: false,
            })
        }
    }

    /// Number of GPUs managed by this executor.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get device information for all managed GPUs.
    pub fn devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    /// Whether warmup validation has been run successfully.
    pub fn is_validated(&self) -> bool {
        self.validated
    }

    /// Total GPU memory across all managed devices.
    pub fn total_memory(&self) -> usize {
        self.devices.iter().map(|d| d.total_memory).sum()
    }

    /// Run a lightweight validation on each GPU to confirm kernels work.
    ///
    /// Spawns one thread per GPU, each performing a real 4-element add kernel
    /// (upload → execute → download → verify). This catches driver issues,
    /// context failures, kernel execution errors, or memory transfer bugs
    /// before committing to a long proving run.
    #[cfg(feature = "cuda-runtime")]
    pub fn validate(&mut self) -> Result<(), MultiGpuError> {
        use std::sync::Mutex;
        use stwo::core::fields::m31::M31;

        info!("Running warmup validation on {} GPUs", self.devices.len());
        let errors: Mutex<Vec<MultiGpuError>> = Mutex::new(Vec::new());

        std::thread::scope(|s| {
            for &(device_id, ref _exec) in &self.executors {
                let errors_ref = &errors;
                s.spawn(move || {
                    let _guard = DeviceGuard::new(device_id);

                    // Phase 1: Verify executor cache works for this device.
                    let exec = match crate::gpu_sumcheck::GpuSumcheckExecutor::cached_for_device(
                        device_id,
                    ) {
                        Ok(e) => e,
                        Err(e) => {
                            errors_ref
                                .lock()
                                .unwrap()
                                .push(MultiGpuError::WarmupFailed {
                                    device_id,
                                    message: format!("Executor cache: {e}"),
                                });
                            return;
                        }
                    };

                    // Phase 2: Execute a real 4-element GPU add kernel.
                    // Tests: memory allocation, host→device transfer, kernel launch,
                    // device→host transfer, and numerical correctness.
                    let lhs = [
                        M31::from(3u32),
                        M31::from(7u32),
                        M31::from(11u32),
                        M31::from(0u32),
                    ];
                    let rhs = [
                        M31::from(5u32),
                        M31::from(2u32),
                        M31::from(1u32),
                        M31::from(100u32),
                    ];
                    let expected = [
                        M31::from(8u32),
                        M31::from(9u32),
                        M31::from(12u32),
                        M31::from(100u32),
                    ];

                    match crate::gpu_sumcheck::gpu_elementwise_add(&lhs, &rhs) {
                        Ok(result) => {
                            if result.as_slice() != expected {
                                errors_ref
                                    .lock()
                                    .unwrap()
                                    .push(MultiGpuError::WarmupFailed {
                                        device_id,
                                        message: format!(
                                            "Numerical mismatch: got {:?}, expected {:?}",
                                            result, expected,
                                        ),
                                    });
                            } else {
                                info!(device_id, "Warmup OK (kernel execution verified)");
                            }
                        }
                        Err(e) => {
                            errors_ref
                                .lock()
                                .unwrap()
                                .push(MultiGpuError::WarmupFailed {
                                    device_id,
                                    message: format!("Kernel execution: {e}"),
                                });
                        }
                    }
                });
            }
        });

        let errs = errors.into_inner().unwrap();
        if !errs.is_empty() {
            let summary: Vec<String> = errs.iter().map(|e| format!("{e}")).collect();
            if errs.len() == 1 {
                return Err(errs.into_iter().next().unwrap());
            }
            return Err(MultiGpuError::MultipleFailures(summary.join("; ")));
        }

        self.validated = true;
        info!("Warmup validation passed on all GPUs");
        Ok(())
    }

    /// Partition chunks across GPUs using greedy memory-aware bin-packing.
    ///
    /// Sorts chunks by estimated memory (descending) and assigns each to
    /// the GPU with the most remaining memory capacity. This balances
    /// utilization across heterogeneous GPUs.
    pub fn partition_chunks(&self, chunks: &[ChunkWorkload]) -> Vec<DeviceAssignment> {
        if chunks.is_empty() || self.devices.is_empty() {
            return Vec::new();
        }

        // Sort chunks by estimated memory, descending (largest first for bin-packing)
        let mut sorted_indices: Vec<usize> = (0..chunks.len()).collect();
        sorted_indices
            .sort_by(|&a, &b| chunks[b].estimated_memory.cmp(&chunks[a].estimated_memory));

        // Track remaining capacity per device (80% safety margin)
        let usable: Vec<usize> = self
            .devices
            .iter()
            .map(|d| d.total_memory * 4 / 5)
            .collect();
        let max_single_gpu = usable.iter().copied().max().unwrap_or(0);
        let mut remaining = usable.clone();
        let mut assignments = Vec::with_capacity(chunks.len());
        let mut oversize_warnings: Vec<String> = Vec::new();

        for &chunk_idx in &sorted_indices {
            let chunk = &chunks[chunk_idx];

            // Warn (but still assign) if a single chunk exceeds the largest GPU's usable capacity
            if chunk.estimated_memory > max_single_gpu {
                oversize_warnings.push(format!(
                    "Chunk {} ({} matmuls) needs {:.2} GB but largest GPU has {:.2} GB usable",
                    chunk.chunk_index,
                    chunk.num_matmuls,
                    chunk.estimated_memory as f64 / 1e9,
                    max_single_gpu as f64 / 1e9,
                ));
            }

            // Find GPU with most remaining capacity
            let best_device = remaining
                .iter()
                .enumerate()
                .max_by_key(|(_, &cap)| cap)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            remaining[best_device] = remaining[best_device].saturating_sub(chunk.estimated_memory);

            assignments.push(DeviceAssignment {
                device_id: self.devices[best_device].ordinal,
                chunk_index: chunk.chunk_index,
            });
        }

        // Sort assignments by chunk_index for deterministic output
        assignments.sort_by_key(|a| a.chunk_index);

        // Log warnings for oversized chunks
        for warning in &oversize_warnings {
            tracing::warn!("{}", warning);
        }

        // Log partition summary
        for (dev_idx, device) in self.devices.iter().enumerate() {
            let device_chunks: Vec<usize> = assignments
                .iter()
                .filter(|a| a.device_id == device.ordinal)
                .map(|a| a.chunk_index)
                .collect();
            let total_mem: usize = device_chunks
                .iter()
                .map(|&idx| chunks[idx].estimated_memory)
                .sum();
            let utilization_pct = if usable[dev_idx] > 0 {
                (total_mem as f64 / usable[dev_idx] as f64 * 100.0).min(999.9)
            } else {
                0.0
            };
            info!(
                device = device.ordinal,
                chunks = ?device_chunks,
                estimated_mem_gb = total_mem as f64 / 1e9,
                capacity_gb = device.total_memory as f64 / 1e9,
                utilization_pct = format!("{:.1}%", utilization_pct),
                "Partition assignment"
            );
        }

        assignments
    }

    /// Partition and return per-device statistics (for dry-run / planning).
    pub fn plan_partition(&self, chunks: &[ChunkWorkload]) -> Vec<(usize, Vec<usize>, usize)> {
        let assignments = self.partition_chunks(chunks);
        self.devices
            .iter()
            .map(|device| {
                let device_chunks: Vec<usize> = assignments
                    .iter()
                    .filter(|a| a.device_id == device.ordinal)
                    .map(|a| a.chunk_index)
                    .collect();
                let total_matmuls: usize = device_chunks
                    .iter()
                    .map(|&idx| chunks[idx].num_matmuls)
                    .sum();
                (device.ordinal, device_chunks, total_matmuls)
            })
            .collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_local_affinity() {
        // Default: no device set
        assert_eq!(get_thread_device(), None);

        // Set device
        set_thread_device(2);
        assert_eq!(get_thread_device(), Some(2));

        // Change device
        set_thread_device(0);
        assert_eq!(get_thread_device(), Some(0));

        // Clear
        clear_thread_device();
        assert_eq!(get_thread_device(), None);
    }

    #[test]
    fn test_thread_local_isolation() {
        // Device affinity is per-thread
        set_thread_device(5);

        let child_device = std::thread::spawn(|| {
            // Child thread should NOT inherit parent's device
            let initial = get_thread_device();
            set_thread_device(3);
            let after = get_thread_device();
            (initial, after)
        })
        .join()
        .unwrap();

        assert_eq!(
            child_device.0, None,
            "child should not inherit parent device"
        );
        assert_eq!(child_device.1, Some(3));

        // Parent's device unchanged
        assert_eq!(get_thread_device(), Some(5));

        clear_thread_device();
    }

    #[test]
    fn test_device_guard_basic() {
        clear_thread_device();
        assert_eq!(get_thread_device(), None);

        {
            let _guard = DeviceGuard::new(2);
            assert_eq!(get_thread_device(), Some(2));
        }
        // Guard dropped — restored to None
        assert_eq!(get_thread_device(), None);
    }

    #[test]
    fn test_device_guard_nested() {
        clear_thread_device();

        {
            let _outer = DeviceGuard::new(0);
            assert_eq!(get_thread_device(), Some(0));

            {
                let _inner = DeviceGuard::new(3);
                assert_eq!(get_thread_device(), Some(3));
            }
            // Inner guard dropped — back to 0
            assert_eq!(get_thread_device(), Some(0));
        }
        // Outer guard dropped — back to None
        assert_eq!(get_thread_device(), None);
    }

    #[test]
    fn test_device_guard_panic_safety() {
        clear_thread_device();

        let result = std::panic::catch_unwind(|| {
            let _guard = DeviceGuard::new(7);
            assert_eq!(get_thread_device(), Some(7));
            panic!("intentional test panic");
        });
        assert!(result.is_err());

        // Guard was dropped during unwind — device restored
        assert_eq!(get_thread_device(), None);
    }

    #[test]
    fn test_propagate_device_some() {
        clear_thread_device();
        let parent = Some(4usize);

        let result = std::thread::spawn(move || {
            let _guard = propagate_device(parent);
            get_thread_device()
        })
        .join()
        .unwrap();

        assert_eq!(result, Some(4));
    }

    #[test]
    fn test_propagate_device_none() {
        let parent: Option<usize> = None;

        let result = std::thread::spawn(move || {
            let _guard = propagate_device(parent);
            get_thread_device()
        })
        .join()
        .unwrap();

        // No device was set
        assert_eq!(result, None);
    }

    #[test]
    fn test_discover_devices() {
        // Should not panic even without CUDA
        let devices = discover_devices();
        let _ = devices;
    }

    #[test]
    fn test_device_count() {
        let count = device_count();
        assert_eq!(count, discover_devices().len());
    }

    #[test]
    fn test_partition_empty() {
        let executor = MultiGpuExecutor {
            #[cfg(feature = "cuda-runtime")]
            executors: Vec::new(),
            devices: vec![GpuDeviceInfo {
                ordinal: 0,
                name: "GPU 0".into(),
                total_memory: 80 * 1024 * 1024 * 1024,
                compute_capability: (9, 0),
                sm_count: 132,
                relative_power: 3.0,
            }],
            validated: false,
        };

        let assignments = executor.partition_chunks(&[]);
        assert!(assignments.is_empty());
    }

    #[test]
    fn test_partition_single_gpu() {
        let executor = MultiGpuExecutor {
            #[cfg(feature = "cuda-runtime")]
            executors: Vec::new(),
            devices: vec![GpuDeviceInfo {
                ordinal: 0,
                name: "GPU 0".into(),
                total_memory: 80_000_000_000,
                compute_capability: (9, 0),
                sm_count: 132,
                relative_power: 3.0,
            }],
            validated: false,
        };

        let chunks = vec![
            ChunkWorkload {
                chunk_index: 0,
                estimated_memory: 10_000_000_000,
                num_matmuls: 40,
                block_range: 0..40,
            },
            ChunkWorkload {
                chunk_index: 1,
                estimated_memory: 15_000_000_000,
                num_matmuls: 40,
                block_range: 40..80,
            },
        ];

        let assignments = executor.partition_chunks(&chunks);
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].device_id, 0);
        assert_eq!(assignments[1].device_id, 0);
    }

    #[test]
    fn test_partition_balanced_two_gpus() {
        let executor = MultiGpuExecutor {
            #[cfg(feature = "cuda-runtime")]
            executors: Vec::new(),
            devices: vec![
                GpuDeviceInfo {
                    ordinal: 0,
                    name: "GPU 0".into(),
                    total_memory: 80_000_000_000,
                    compute_capability: (9, 0),
                    sm_count: 132,
                    relative_power: 3.0,
                },
                GpuDeviceInfo {
                    ordinal: 1,
                    name: "GPU 1".into(),
                    total_memory: 80_000_000_000,
                    compute_capability: (9, 0),
                    sm_count: 132,
                    relative_power: 3.0,
                },
            ],
            validated: false,
        };

        let chunks = vec![
            ChunkWorkload {
                chunk_index: 0,
                estimated_memory: 30_000_000_000,
                num_matmuls: 40,
                block_range: 0..40,
            },
            ChunkWorkload {
                chunk_index: 1,
                estimated_memory: 25_000_000_000,
                num_matmuls: 40,
                block_range: 40..80,
            },
            ChunkWorkload {
                chunk_index: 2,
                estimated_memory: 20_000_000_000,
                num_matmuls: 40,
                block_range: 80..120,
            },
            ChunkWorkload {
                chunk_index: 3,
                estimated_memory: 15_000_000_000,
                num_matmuls: 40,
                block_range: 120..160,
            },
        ];

        let assignments = executor.partition_chunks(&chunks);
        assert_eq!(assignments.len(), 4);

        let gpu0_chunks: Vec<_> = assignments.iter().filter(|a| a.device_id == 0).collect();
        let gpu1_chunks: Vec<_> = assignments.iter().filter(|a| a.device_id == 1).collect();

        assert_eq!(gpu0_chunks.len(), 2);
        assert_eq!(gpu1_chunks.len(), 2);
    }

    #[test]
    fn test_partition_heterogeneous_gpus() {
        let executor = MultiGpuExecutor {
            #[cfg(feature = "cuda-runtime")]
            executors: Vec::new(),
            devices: vec![
                GpuDeviceInfo {
                    ordinal: 0,
                    name: "H100".into(),
                    total_memory: 80_000_000_000,
                    compute_capability: (9, 0),
                    sm_count: 132,
                    relative_power: 3.0,
                },
                GpuDeviceInfo {
                    ordinal: 1,
                    name: "A100".into(),
                    total_memory: 40_000_000_000,
                    compute_capability: (8, 0),
                    sm_count: 108,
                    relative_power: 1.0,
                },
            ],
            validated: false,
        };

        let chunks = vec![
            ChunkWorkload {
                chunk_index: 0,
                estimated_memory: 35_000_000_000,
                num_matmuls: 40,
                block_range: 0..40,
            },
            ChunkWorkload {
                chunk_index: 1,
                estimated_memory: 30_000_000_000,
                num_matmuls: 40,
                block_range: 40..80,
            },
            ChunkWorkload {
                chunk_index: 2,
                estimated_memory: 10_000_000_000,
                num_matmuls: 20,
                block_range: 80..100,
            },
        ];

        let assignments = executor.partition_chunks(&chunks);
        assert_eq!(assignments.len(), 3);

        // Greedy with 80% safety: GPU 0 has 64GB usable, GPU 1 has 32GB usable
        // 35GB → GPU 0 (64→29), 30GB → GPU 1 (32→2), 10GB → GPU 0 (29→19)
        assert_eq!(assignments[0].device_id, 0);
        assert_eq!(assignments[1].device_id, 1);
        assert_eq!(assignments[2].device_id, 0);
    }

    #[test]
    fn test_plan_partition() {
        let executor = MultiGpuExecutor {
            #[cfg(feature = "cuda-runtime")]
            executors: Vec::new(),
            devices: vec![
                GpuDeviceInfo {
                    ordinal: 0,
                    name: "GPU 0".into(),
                    total_memory: 80_000_000_000,
                    compute_capability: (9, 0),
                    sm_count: 132,
                    relative_power: 3.0,
                },
                GpuDeviceInfo {
                    ordinal: 1,
                    name: "GPU 1".into(),
                    total_memory: 80_000_000_000,
                    compute_capability: (9, 0),
                    sm_count: 132,
                    relative_power: 3.0,
                },
            ],
            validated: false,
        };

        let chunks = vec![
            ChunkWorkload {
                chunk_index: 0,
                estimated_memory: 10_000_000_000,
                num_matmuls: 20,
                block_range: 0..20,
            },
            ChunkWorkload {
                chunk_index: 1,
                estimated_memory: 10_000_000_000,
                num_matmuls: 30,
                block_range: 20..50,
            },
        ];

        let plan = executor.plan_partition(&chunks);
        assert_eq!(plan.len(), 2);

        let total_matmuls: usize = plan.iter().map(|(_, _, m)| m).sum();
        assert_eq!(total_matmuls, 50);
    }

    #[test]
    fn test_total_memory() {
        let executor = MultiGpuExecutor {
            #[cfg(feature = "cuda-runtime")]
            executors: Vec::new(),
            devices: vec![
                GpuDeviceInfo {
                    ordinal: 0,
                    name: "GPU 0".into(),
                    total_memory: 80_000_000_000,
                    compute_capability: (9, 0),
                    sm_count: 132,
                    relative_power: 3.0,
                },
                GpuDeviceInfo {
                    ordinal: 1,
                    name: "GPU 1".into(),
                    total_memory: 40_000_000_000,
                    compute_capability: (8, 0),
                    sm_count: 108,
                    relative_power: 1.0,
                },
            ],
            validated: false,
        };

        assert_eq!(executor.total_memory(), 120_000_000_000);
    }

    #[test]
    fn test_multi_gpu_error_display() {
        let e = MultiGpuError::NoDevices;
        assert_eq!(format!("{e}"), "No GPU devices found");

        let e = MultiGpuError::InsufficientMemory {
            device_id: 1,
            needed: 100,
            available: 50,
        };
        assert!(format!("{e}").contains("device 1"));

        let e = MultiGpuError::MultipleFailures("chunk 0 on GPU 0, chunk 2 on GPU 1".into());
        assert!(format!("{e}").contains("chunk 0"));

        let e = MultiGpuError::WarmupFailed {
            device_id: 3,
            message: "kernel launch failed".into(),
        };
        assert!(format!("{e}").contains("device 3"));
    }
}
