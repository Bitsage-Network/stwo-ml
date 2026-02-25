//! CUDA FFT Executor - Runtime integration for GPU-accelerated FFT.
//!
//! This module provides the actual CUDA execution layer for the GPU FFT kernels.
//! It handles:
//! - Device initialization and management
//! - Kernel compilation via NVRTC
//! - Memory allocation and transfers
//! - Kernel execution and synchronization
//!
//! # Requirements
//!
//! - CUDA Toolkit 11.0+ installed
//! - NVIDIA GPU with compute capability 7.0+ (Volta or newer recommended)
//! - `gpu` feature enabled in Cargo.toml

#[cfg(feature = "cuda-runtime")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use super::fft::CIRCLE_FFT_CUDA_KERNEL;
#[allow(unused_imports)]
#[cfg(feature = "cuda-runtime")]
use super::fft::{GPU_FFT_THRESHOLD_LOG_SIZE, M31_PRIME};

// =============================================================================
// Global CUDA Context
// =============================================================================

#[cfg(feature = "cuda-runtime")]
static CUDA_FFT_EXECUTOR: OnceLock<Result<CudaFftExecutor, CudaFftError>> = OnceLock::new();

#[cfg(feature = "cuda-runtime")]
use std::sync::Mutex;

#[cfg(feature = "cuda-runtime")]
use std::collections::HashMap;

/// Multi-GPU executor pool for true parallel GPU execution.
///
/// This replaces the global singleton pattern with a pool that supports:
/// - One executor per GPU device
/// - Thread-safe access via Arc<Mutex<>>
/// - Lazy initialization per device
#[cfg(feature = "cuda-runtime")]
static CUDA_EXECUTOR_POOL: OnceLock<Mutex<HashMap<usize, Arc<CudaFftExecutor>>>> = OnceLock::new();

/// Get or create an executor for a specific GPU device.
///
/// This is the preferred method for multi-GPU workloads.
/// Each device gets its own executor with compiled kernels.
///
/// # Arguments
/// * `device_id` - The GPU device ID (0, 1, 2, etc.)
///
/// # Returns
/// An Arc-wrapped executor that can be shared across threads.
#[cfg(feature = "cuda-runtime")]
pub fn get_executor_for_device(device_id: usize) -> Result<Arc<CudaFftExecutor>, CudaFftError> {
    let pool = CUDA_EXECUTOR_POOL.get_or_init(|| Mutex::new(HashMap::new()));

    let mut pool_guard = pool
        .lock()
        .map_err(|_| CudaFftError::DriverInit("Failed to acquire executor pool lock".into()))?;

    // Return cached executor if available
    if let Some(executor) = pool_guard.get(&device_id) {
        return Ok(Arc::clone(executor));
    }

    // Create new executor for this device
    tracing::info!("Creating new CUDA executor for device {}", device_id);
    let executor = CudaFftExecutor::new_on_device(device_id)?;
    let executor_arc = Arc::new(executor);

    pool_guard.insert(device_id, Arc::clone(&executor_arc));

    Ok(executor_arc)
}

/// Get executors for all available GPUs.
///
/// Useful for distributing work across all GPUs.
#[cfg(feature = "cuda-runtime")]
pub fn get_all_executors() -> Result<Vec<(usize, Arc<CudaFftExecutor>)>, CudaFftError> {
    let mut executors = Vec::new();

    // Probe for available GPUs (up to 16)
    for device_id in 0..16 {
        match get_executor_for_device(device_id) {
            Ok(executor) => executors.push((device_id, executor)),
            Err(CudaFftError::NoDevice) => break,
            Err(CudaFftError::DriverInit(_)) => break, // No more GPUs
            Err(e) => return Err(e),
        }
    }

    if executors.is_empty() {
        return Err(CudaFftError::NoDevice);
    }

    Ok(executors)
}

/// Get the number of available CUDA devices.
#[cfg(feature = "cuda-runtime")]
pub fn get_device_count() -> usize {
    let mut count = 0;
    for i in 0..16 {
        if CudaDevice::new(i).is_ok() {
            count = i + 1;
        } else {
            break;
        }
    }
    count
}

/// Get the global CUDA FFT executor instance.
///
/// This lazily initializes the CUDA context on first call.
///
/// **Note:** For multi-GPU workloads, prefer `get_executor_for_device()`.
#[cfg(feature = "cuda-runtime")]
pub fn get_cuda_executor() -> Result<&'static CudaFftExecutor, &'static CudaFftError> {
    CUDA_FFT_EXECUTOR
        .get_or_init(|| CudaFftExecutor::new())
        .as_ref()
}

/// Check if CUDA is available and initialized.
#[cfg(feature = "cuda-runtime")]
pub fn is_cuda_available() -> bool {
    get_cuda_executor().is_ok()
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn is_cuda_available() -> bool {
    false
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn get_device_count() -> usize {
    0
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during CUDA FFT execution.
#[derive(Debug, Clone)]
pub enum CudaFftError {
    /// No CUDA device found
    NoDevice,
    /// CUDA driver initialization failed
    DriverInit(String),
    /// Kernel compilation failed
    KernelCompilation(String),
    /// Memory allocation failed
    MemoryAllocation(String),
    /// Memory transfer failed
    MemoryTransfer(String),
    /// Kernel execution failed
    KernelExecution(String),
    /// Invalid input size
    InvalidSize(String),
    /// Kernel launch failed
    KernelLaunch(String),
}

impl std::fmt::Display for CudaFftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaFftError::NoDevice => write!(f, "No CUDA device found"),
            CudaFftError::DriverInit(s) => write!(f, "CUDA driver init failed: {}", s),
            CudaFftError::KernelCompilation(s) => write!(f, "Kernel compilation failed: {}", s),
            CudaFftError::MemoryAllocation(s) => write!(f, "Memory allocation failed: {}", s),
            CudaFftError::MemoryTransfer(s) => write!(f, "Memory transfer failed: {}", s),
            CudaFftError::KernelExecution(s) => write!(f, "Kernel execution failed: {}", s),
            CudaFftError::InvalidSize(s) => write!(f, "Invalid size: {}", s),
            CudaFftError::KernelLaunch(s) => write!(f, "Kernel launch failed: {}", s),
        }
    }
}

impl std::error::Error for CudaFftError {}

// =============================================================================
// Memory Pressure Management
// =============================================================================

/// Memory usage statistics for a CUDA device.
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    /// Total device memory in bytes
    pub total_bytes: usize,
    /// Free device memory in bytes
    pub free_bytes: usize,
    /// Used device memory in bytes
    pub used_bytes: usize,
    /// Utilization percentage (0-100)
    pub utilization_percent: f32,
}

/// Strategy for handling memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressureStrategy {
    /// Fail immediately with an error
    FailFast,
    /// Fallback to CPU (SIMD) processing
    FallbackToCpu,
    /// Wait and retry (with exponential backoff)
    WaitAndRetry {
        max_retries: u32,
        base_delay_ms: u64,
    },
}

impl Default for MemoryPressureStrategy {
    fn default() -> Self {
        // Default to CPU fallback for robustness
        MemoryPressureStrategy::FallbackToCpu
    }
}

/// Query current GPU memory usage.
#[cfg(feature = "cuda-runtime")]
pub fn get_memory_stats() -> Result<GpuMemoryStats, CudaFftError> {
    use super::compat;

    let (free, total) = compat::mem_get_info().map_err(|e| CudaFftError::DriverInit(e))?;

    let used = total - free;
    let utilization = if total > 0 {
        (used as f32 / total as f32) * 100.0
    } else {
        0.0
    };

    Ok(GpuMemoryStats {
        total_bytes: total,
        free_bytes: free,
        used_bytes: used,
        utilization_percent: utilization,
    })
}

/// Check if there's enough GPU memory for a given allocation.
///
/// Returns `Ok(true)` if sufficient memory is available,
/// `Ok(false)` if not, and `Err` on query failure.
#[cfg(feature = "cuda-runtime")]
pub fn check_memory_available(
    required_bytes: usize,
    safety_margin: f32,
) -> Result<bool, CudaFftError> {
    let stats = get_memory_stats()?;

    // Apply safety margin (e.g., 0.1 = keep 10% free)
    let required_with_margin = (required_bytes as f32 * (1.0 + safety_margin)) as usize;

    Ok(stats.free_bytes >= required_with_margin)
}

/// Calculate memory requirements for proof generation.
///
/// Returns the estimated GPU memory needed for a proof with the given parameters.
pub fn estimate_proof_memory(log_size: u32, num_polynomials: usize) -> usize {
    let n = 1usize << log_size;

    // Base polynomial storage: n * 4 bytes per polynomial
    let poly_storage = n * 4 * num_polynomials;

    // Twiddle factors: approximately 2 * n * 4 bytes
    let twiddle_storage = n * 8;

    // Working buffers: 2-3x polynomial storage for FFT/FRI
    let working_buffers = poly_storage * 3;

    // Merkle tree buffers: roughly 2 * n * 32 bytes for leaves + nodes
    let merkle_storage = n * 64;

    // Add 20% overhead for fragmentation and kernel scratch space
    let total = poly_storage + twiddle_storage + working_buffers + merkle_storage;
    (total as f32 * 1.2) as usize
}

/// Execute with memory pressure handling.
///
/// This wrapper tries GPU execution first, then falls back to CPU if needed.
#[cfg(feature = "cuda-runtime")]
pub fn with_memory_fallback<T, GpuFn, CpuFn>(
    strategy: MemoryPressureStrategy,
    required_bytes: usize,
    mut gpu_fn: GpuFn,
    cpu_fn: CpuFn,
) -> Result<T, CudaFftError>
where
    GpuFn: FnMut() -> Result<T, CudaFftError>,
    CpuFn: FnOnce() -> T,
{
    match strategy {
        MemoryPressureStrategy::FailFast => {
            // Check memory before attempting GPU execution
            if !check_memory_available(required_bytes, 0.1)? {
                let stats = get_memory_stats()?;
                return Err(CudaFftError::MemoryAllocation(format!(
                    "Insufficient GPU memory: need {} MB, only {} MB free",
                    required_bytes / (1024 * 1024),
                    stats.free_bytes / (1024 * 1024)
                )));
            }
            gpu_fn()
        }

        MemoryPressureStrategy::FallbackToCpu => {
            // Check memory first
            if !check_memory_available(required_bytes, 0.1).unwrap_or(false) {
                tracing::warn!(
                    "GPU memory pressure detected ({} MB required), falling back to CPU",
                    required_bytes / (1024 * 1024)
                );
                return Ok(cpu_fn());
            }

            // Try GPU, fallback on error
            match gpu_fn() {
                Ok(result) => Ok(result),
                Err(CudaFftError::MemoryAllocation(_)) => {
                    tracing::warn!("GPU allocation failed, falling back to CPU");
                    Ok(cpu_fn())
                }
                Err(e) => Err(e),
            }
        }

        MemoryPressureStrategy::WaitAndRetry {
            max_retries,
            base_delay_ms,
        } => {
            let mut retries = 0;

            loop {
                if check_memory_available(required_bytes, 0.1)? {
                    match gpu_fn() {
                        Ok(result) => return Ok(result),
                        Err(CudaFftError::MemoryAllocation(msg)) => {
                            if retries >= max_retries {
                                return Err(CudaFftError::MemoryAllocation(format!(
                                    "Out of GPU memory after {} retries: {}",
                                    max_retries, msg
                                )));
                            }
                            retries += 1;
                        }
                        Err(e) => return Err(e),
                    }
                } else if retries >= max_retries {
                    let stats = get_memory_stats()?;
                    return Err(CudaFftError::MemoryAllocation(format!(
                        "Timeout waiting for GPU memory: need {} MB, only {} MB free",
                        required_bytes / (1024 * 1024),
                        stats.free_bytes / (1024 * 1024)
                    )));
                }

                // Exponential backoff
                let delay = base_delay_ms * (1 << retries.min(5));
                tracing::debug!(
                    "Waiting for GPU memory (retry {}/{}), sleeping {} ms",
                    retries + 1,
                    max_retries,
                    delay
                );
                std::thread::sleep(std::time::Duration::from_millis(delay));
                retries += 1;
            }
        }
    }
}

// Non-cuda-runtime stub
#[cfg(not(feature = "cuda-runtime"))]
pub fn get_memory_stats() -> Result<GpuMemoryStats, CudaFftError> {
    Err(CudaFftError::NoDevice)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn check_memory_available(
    _required_bytes: usize,
    _safety_margin: f32,
) -> Result<bool, CudaFftError> {
    Ok(false)
}

// =============================================================================
// CUDA FFT Executor
// =============================================================================

/// CUDA FFT Executor - manages GPU resources for FFT operations.
#[cfg(feature = "cuda-runtime")]
pub struct CudaFftExecutor {
    /// CUDA device handle (public for memory management)
    pub device: Arc<CudaDevice>,
    /// Compiled kernels (public for pipeline access)
    pub kernels: CompiledKernels,
    /// Device info
    pub device_info: DeviceInfo,
    /// Dedicated compute stream for kernel execution (optional)
    /// When set, kernels are launched on this stream instead of the default stream.
    /// This allows overlapping transfers with computation.
    compute_stream: Option<cudarc::driver::CudaStream>,
    /// Transfer stream for H2D/D2H operations
    transfer_stream: Option<cudarc::driver::CudaStream>,
}

// SAFETY: CudaFftExecutor is Send+Sync because:
// - CudaDevice is already Send+Sync (Arc-wrapped)
// - CudaStream contains a raw CUDA stream handle that is safe to send between threads
//   as long as we don't use it concurrently (we don't - access is serialized)
// - CompiledKernels contains CudaFunction handles that are thread-safe
// This is required for OnceLock<Result<CudaFftExecutor, _>> and the executor pool statics.
#[cfg(feature = "cuda-runtime")]
unsafe impl Send for CudaFftExecutor {}
#[cfg(feature = "cuda-runtime")]
unsafe impl Sync for CudaFftExecutor {}

/// Compiled CUDA kernels for proof operations.
#[cfg(feature = "cuda-runtime")]
pub struct CompiledKernels {
    // FFT kernels
    pub bit_reverse: CudaFunction,
    pub ifft_layer: CudaFunction,
    pub fft_layer: CudaFunction,
    // Optimized shared memory FFT kernel
    pub ifft_shared_mem: CudaFunction,
    // Denormalization kernels (fused post-FFT operation)
    pub denormalize: CudaFunction,
    pub denormalize_vec4: CudaFunction,
    // FRI folding kernels
    pub fold_line: CudaFunction,
    pub fold_circle_into_line: CudaFunction,
    pub deinterleave_aos_to_soa: CudaFunction,
    // Quotient accumulation kernels
    pub accumulate_quotients: CudaFunction,
    pub eval_point_accumulate: CudaFunction,
    pub copy_column: CudaFunction, // GPU-resident column copy
    // MLE (GKR) operations kernels
    pub mle_fold_base_to_secure: CudaFunction,
    pub mle_fold_secure: CudaFunction,
    pub gen_eq_evals: CudaFunction,
    // Merkle hashing kernel (Blake2s)
    pub merkle_layer: CudaFunction,
    // Poseidon252 Merkle hashing kernel
    pub poseidon252_merkle_layer: CudaFunction,
    // Poseidon252 chunked hash_many kernel (weight commitments)
    pub poseidon252_hash_many_chunked: CudaFunction,
    // Poseidon252 chunked hash_many kernel over raw M31 inputs (GPU packing)
    pub poseidon252_hash_many_chunked_m31: CudaFunction,
}

/// GPU-resident Poseidon252 Merkle tree layers.
///
/// Stores internal hash layers (not raw leaves):
/// - `layers[0]` has `n_leaf_hashes` hashes (hash of leaf pairs)
/// - `layers[1]` has `n_leaf_hashes/2` hashes
/// - ...
/// - `layers[last]` has 1 hash (root)
#[cfg(feature = "cuda-runtime")]
pub struct Poseidon252MerkleGpuTree {
    device: Arc<CudaDevice>,
    layers: Vec<CudaSlice<u64>>,
    layer_hash_counts: Vec<usize>,
}

#[cfg(feature = "cuda-runtime")]
impl Poseidon252MerkleGpuTree {
    /// Number of internal hash layers (includes root layer).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Number of hashes in a given internal layer.
    pub fn layer_hash_count(&self, layer_idx: usize) -> usize {
        self.layer_hash_counts[layer_idx]
    }

    /// Download one hash node (4 u64 limbs) from an internal layer.
    pub fn node_u64(&self, layer_idx: usize, hash_idx: usize) -> Result<[u64; 4], CudaFftError> {
        if layer_idx >= self.layers.len() {
            return Err(CudaFftError::InvalidSize(format!(
                "layer index out of bounds: {} (layers={})",
                layer_idx,
                self.layers.len()
            )));
        }
        let n = self.layer_hash_counts[layer_idx];
        if hash_idx >= n {
            return Err(CudaFftError::InvalidSize(format!(
                "hash index out of bounds: {} (layer={}, hashes={})",
                hash_idx, layer_idx, n
            )));
        }
        let start = hash_idx * 4;
        let end = start + 4;
        let mut out = [0u64; 4];
        self.device
            .dtoh_sync_copy_into(&self.layers[layer_idx].slice(start..end), &mut out)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        Ok(out)
    }

    /// Download the Merkle root hash (4 u64 limbs).
    pub fn root_u64(&self) -> Result<[u64; 4], CudaFftError> {
        if self.layers.is_empty() {
            return Err(CudaFftError::InvalidSize(
                "cannot read root of empty GPU merkle tree".into(),
            ));
        }
        self.node_u64(self.layers.len() - 1, 0)
    }
}

/// Information about the CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory_bytes: usize,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
}

#[cfg(feature = "cuda-runtime")]
impl CudaFftExecutor {
    /// Create a new CUDA FFT executor on GPU 0.
    ///
    /// This initializes the CUDA context and compiles all FFT kernels.
    pub fn new() -> Result<Self, CudaFftError> {
        Self::new_on_device(0)
    }

    /// Create a new CUDA FFT executor on a specific GPU.
    ///
    /// # Arguments
    /// * `device_id` - The GPU device ID (0, 1, 2, etc.)
    pub fn new_on_device(device_id: usize) -> Result<Self, CudaFftError> {
        // Initialize CUDA device (returns Arc<CudaDevice>)
        let device = CudaDevice::new(device_id)
            .map_err(|e| CudaFftError::DriverInit(format!("GPU {}: {:?}", device_id, e)))?;

        // Get device info
        let device_info = Self::get_device_info(&device)?;

        tracing::info!(
            "CUDA device initialized: {} (SM {}.{}, {} MB)",
            device_info.name,
            device_info.compute_capability.0,
            device_info.compute_capability.1,
            device_info.total_memory_bytes / (1024 * 1024)
        );

        // Compile kernels
        let kernels = Self::compile_kernels(&device)?;

        // Create dedicated streams for overlapped execution
        let compute_stream = device
            .fork_default_stream()
            .map_err(|e| CudaFftError::DriverInit(format!("Compute stream: {:?}", e)))
            .ok();

        let transfer_stream = device
            .fork_default_stream()
            .map_err(|e| CudaFftError::DriverInit(format!("Transfer stream: {:?}", e)))
            .ok();

        if compute_stream.is_some() {
            tracing::info!("CUDA streams enabled for overlapped execution");
        }

        tracing::info!("CUDA FFT kernels compiled successfully");

        Ok(Self {
            device,
            kernels,
            device_info,
            compute_stream,
            transfer_stream,
        })
    }

    // =========================================================================
    // Stream Access Methods
    // =========================================================================

    /// Get the compute stream for kernel execution.
    ///
    /// When a compute stream is available, kernels should be launched on it
    /// to enable overlapped execution with transfers.
    pub fn compute_stream(&self) -> Option<&cudarc::driver::CudaStream> {
        self.compute_stream.as_ref()
    }

    /// Get the transfer stream for H2D/D2H operations.
    pub fn transfer_stream(&self) -> Option<&cudarc::driver::CudaStream> {
        self.transfer_stream.as_ref()
    }

    /// Check if stream-based execution is enabled.
    pub fn has_streams(&self) -> bool {
        self.compute_stream.is_some()
    }

    /// Synchronize the compute stream (wait for all kernels to complete).
    pub fn sync_compute(&self) -> Result<(), CudaFftError> {
        if let Some(stream) = &self.compute_stream {
            self.device.wait_for(stream).map_err(|e| {
                CudaFftError::KernelExecution(format!("Compute stream sync: {:?}", e))
            })?;
        }
        Ok(())
    }

    /// Synchronize the transfer stream (wait for all transfers to complete).
    pub fn sync_transfer(&self) -> Result<(), CudaFftError> {
        if let Some(stream) = &self.transfer_stream {
            self.device.wait_for(stream).map_err(|e| {
                CudaFftError::KernelExecution(format!("Transfer stream sync: {:?}", e))
            })?;
        }
        Ok(())
    }

    /// Synchronize all streams.
    pub fn sync_all(&self) -> Result<(), CudaFftError> {
        self.sync_compute()?;
        self.sync_transfer()?;
        Ok(())
    }

    fn get_device_info(device: &Arc<CudaDevice>) -> Result<DeviceInfo, CudaFftError> {
        use super::compat;
        use cudarc::driver::sys::CUdevice_attribute;

        let cu_device = device.cu_device();

        // Query compute capability
        let major = compat::device_get_attribute(
            *cu_device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .map_err(|e| CudaFftError::DriverInit(e))? as u32;

        let minor = compat::device_get_attribute(
            *cu_device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .map_err(|e| CudaFftError::DriverInit(e))? as u32;

        // Query device name
        let name = compat::device_get_name(*cu_device).unwrap_or_else(|_| "NVIDIA GPU".to_string());

        // Query total memory
        let total_memory_bytes =
            compat::device_total_mem(*cu_device).unwrap_or(8 * 1024 * 1024 * 1024);

        // Query SM count
        let multiprocessor_count = compat::device_get_attribute(
            *cu_device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .map_err(|e| CudaFftError::DriverInit(e))? as u32;

        // Query max threads per block
        let max_threads_per_block = compat::device_get_attribute(
            *cu_device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )
        .map_err(|e| CudaFftError::DriverInit(e))? as u32;

        Ok(DeviceInfo {
            name,
            compute_capability: (major, minor),
            total_memory_bytes,
            multiprocessor_count,
            max_threads_per_block,
        })
    }

    // =========================================================================
    // PTX Caching for Faster Startup
    // =========================================================================

    /// Get the PTX cache directory path.
    ///
    /// Default: `~/.cache/stwo-prover/ptx/`
    fn get_cache_dir() -> Option<std::path::PathBuf> {
        // Try standard cache locations
        if let Some(home) = std::env::var_os("HOME") {
            let cache_dir = std::path::PathBuf::from(home)
                .join(".cache")
                .join("stwo-prover")
                .join("ptx");
            return Some(cache_dir);
        }

        // Fallback to temp directory
        Some(std::env::temp_dir().join("stwo-prover-ptx"))
    }

    /// Compute a hash of the kernel source for cache invalidation.
    ///
    /// Uses blake3 for fast hashing of potentially large kernel sources.
    fn compute_source_hash(source: &str) -> String {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(source.as_bytes());
        // Include library version for invalidation on updates
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());

        let hash = hasher.finalize();
        // Use first 16 bytes (32 hex chars) for reasonable uniqueness
        hex::encode(&hash.as_bytes()[..16])
    }

    /// Try to load pre-compiled PTX, fall back to runtime compilation.
    ///
    /// PTX loading priority:
    /// 1. Build-time compiled PTX (from OUT_DIR via build.rs)
    /// 2. User-provided PTX files (STWO_PTX_DIR environment variable)
    /// 3. Cached PTX markers (for tracking source changes)
    /// 4. Runtime NVRTC compilation (fallback)
    ///
    /// # Note on cudarc Limitations
    ///
    /// cudarc's `Ptx` type is opaque and doesn't expose construction from bytes.
    /// The current implementation uses NVRTC runtime compilation as the primary
    /// path, with cache markers to track compilation state.
    ///
    /// # Arguments
    /// * `kernel_name` - Human-readable kernel name for logging
    /// * `source` - CUDA C++ source code
    ///
    /// # Returns
    /// Compiled PTX (either from cache or freshly compiled)
    fn compile_or_load_cached(
        kernel_name: &str,
        source: &str,
    ) -> Result<cudarc::nvrtc::Ptx, CudaFftError> {
        // Check for build-time PTX directory
        if let Ok(ptx_dir) = std::env::var("STWO_PTX_BUILD_DIR") {
            let marker_path =
                std::path::PathBuf::from(&ptx_dir).join(format!("{}.marker", kernel_name));

            if marker_path.exists() {
                tracing::debug!(
                    "Found build-time PTX marker for {} at {:?}",
                    kernel_name,
                    marker_path
                );
            }
        }

        // Check for user-provided PTX files
        if let Ok(ptx_dir) = std::env::var("STWO_PTX_DIR") {
            let ptx_path = std::path::PathBuf::from(&ptx_dir).join(format!("{}.ptx", kernel_name));

            if ptx_path.exists() {
                tracing::info!(
                    "Loading pre-compiled PTX for {} from {:?}",
                    kernel_name,
                    ptx_path
                );
                // Note: cudarc doesn't support loading PTX from file directly
                // Would need to use CUDA driver API (cuModuleLoad)
            }
        }
        let source_hash = Self::compute_source_hash(source);

        // Check if we have a valid cache marker
        if let Some(cache_dir) = Self::get_cache_dir() {
            let marker_file = cache_dir.join(format!("{}_{}.marker", kernel_name, source_hash));

            if marker_file.exists() {
                // Source hasn't changed since last compilation
                // Still need to recompile due to cudarc limitations, but this is fast
                // as NVRTC uses its own internal caching
                tracing::debug!(
                    "{} source unchanged (hash: {}), recompiling with NVRTC cache",
                    kernel_name,
                    &source_hash[..8]
                );
            } else {
                tracing::info!(
                    "{} source changed or first compile (hash: {})",
                    kernel_name,
                    &source_hash[..8]
                );
            }

            // Compile and update marker
            return Self::compile_and_mark(kernel_name, source, &marker_file, &source_hash);
        }

        // No cache directory available - just compile
        tracing::debug!("PTX caching disabled (no cache directory)");
        Self::compile_with_timing(kernel_name, source)
    }

    /// Compile PTX and save marker file.
    fn compile_and_mark(
        kernel_name: &str,
        source: &str,
        marker_file: &std::path::Path,
        source_hash: &str,
    ) -> Result<cudarc::nvrtc::Ptx, CudaFftError> {
        let ptx = Self::compile_with_timing(kernel_name, source)?;

        // Create marker file for cache validation
        if let Some(parent) = marker_file.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                tracing::debug!("Failed to create cache directory: {}", e);
            } else {
                // Write marker with hash and timestamp
                let marker_content = format!(
                    "{}\n{}\n",
                    source_hash,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                );

                if let Err(e) = std::fs::write(marker_file, marker_content) {
                    tracing::debug!("Failed to write cache marker: {}", e);
                }
            }
        }

        Ok(ptx)
    }

    /// Compile PTX with timing instrumentation.
    fn compile_with_timing(
        kernel_name: &str,
        source: &str,
    ) -> Result<cudarc::nvrtc::Ptx, CudaFftError> {
        let start = std::time::Instant::now();

        let ptx = cudarc::nvrtc::compile_ptx(source).map_err(|e| {
            CudaFftError::KernelCompilation(format!("{} kernel: {:?}", kernel_name, e))
        })?;

        let compile_time = start.elapsed();
        tracing::info!("Compiled {} PTX in {:?}", kernel_name, compile_time);

        Ok(ptx)
    }

    /// Clear the PTX cache.
    ///
    /// Useful when kernel sources have been modified during development.
    pub fn clear_ptx_cache() -> Result<(), std::io::Error> {
        if let Some(cache_dir) = Self::get_cache_dir() {
            if cache_dir.exists() {
                std::fs::remove_dir_all(&cache_dir)?;
                tracing::info!("Cleared PTX cache at {:?}", cache_dir);
            }
        }
        Ok(())
    }

    // =========================================================================
    // Kernel Compilation
    // =========================================================================

    fn compile_kernels(device: &Arc<CudaDevice>) -> Result<CompiledKernels, CudaFftError> {
        // Compile or load cached FFT PTX
        let fft_ptx = Self::compile_or_load_cached("circle_fft", CIRCLE_FFT_CUDA_KERNEL)?;

        // Load FFT PTX into device
        device
            .load_ptx(
                fft_ptx,
                "circle_fft",
                &[
                    "bit_reverse_kernel",
                    "ifft_layer_kernel",
                    "fft_layer_kernel",
                    "ifft_shared_mem_kernel",
                    "denormalize_kernel",
                    "denormalize_vec4_kernel",
                ],
            )
            .map_err(|e| CudaFftError::KernelCompilation(format!("FFT load: {:?}", e)))?;

        // Compile or load cached FRI PTX
        use super::fft::FRI_FOLDING_CUDA_KERNEL;
        let fri_ptx = Self::compile_or_load_cached("fri_folding", FRI_FOLDING_CUDA_KERNEL)?;

        // Load FRI PTX into device
        device
            .load_ptx(
                fri_ptx,
                "fri_folding",
                &[
                    "fold_line_kernel",
                    "fold_circle_into_line_kernel",
                    "deinterleave_aos_to_soa_kernel",
                ],
            )
            .map_err(|e| CudaFftError::KernelCompilation(format!("FRI load: {:?}", e)))?;

        // Compile or load cached Quotient PTX
        use super::fft::QUOTIENT_CUDA_KERNEL;
        let quotient_ptx = Self::compile_or_load_cached("quotient", QUOTIENT_CUDA_KERNEL)?;

        // Load Quotient PTX into device (includes buffer gather and MLE kernels)
        device
            .load_ptx(
                quotient_ptx,
                "quotient",
                &[
                    "accumulate_quotients_kernel",
                    "eval_point_accumulate_kernel",
                    "copy_column_kernel",             // GPU-resident column copy
                    "gather_buffers_kernel",          // GPU-resident buffer gathering
                    "mle_fold_base_to_secure_kernel", // MLE fold: BaseField -> SecureField
                    "mle_fold_secure_kernel",         // MLE fold: SecureField -> SecureField
                    "gen_eq_evals_kernel",            // Generate equality evaluations for GKR
                ],
            )
            .map_err(|e| CudaFftError::KernelCompilation(format!("Quotient load: {:?}", e)))?;

        // Get FFT function handles
        let bit_reverse = device
            .get_func("circle_fft", "bit_reverse_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("bit_reverse_kernel not found".into())
            })?;

        let ifft_layer = device
            .get_func("circle_fft", "ifft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("ifft_layer_kernel not found".into()))?;

        let fft_layer = device
            .get_func("circle_fft", "fft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fft_layer_kernel not found".into()))?;

        let ifft_shared_mem = device
            .get_func("circle_fft", "ifft_shared_mem_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("ifft_shared_mem_kernel not found".into())
            })?;

        // Get denormalization function handles
        let denormalize = device
            .get_func("circle_fft", "denormalize_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("denormalize_kernel not found".into())
            })?;

        let denormalize_vec4 = device
            .get_func("circle_fft", "denormalize_vec4_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("denormalize_vec4_kernel not found".into())
            })?;

        // Get FRI function handles
        let fold_line = device
            .get_func("fri_folding", "fold_line_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fold_line_kernel not found".into()))?;

        let fold_circle_into_line = device
            .get_func("fri_folding", "fold_circle_into_line_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("fold_circle_into_line_kernel not found".into())
            })?;

        let deinterleave_aos_to_soa = device
            .get_func("fri_folding", "deinterleave_aos_to_soa_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("deinterleave_aos_to_soa_kernel not found".into())
            })?;

        // Get Quotient function handle
        let accumulate_quotients = device
            .get_func("quotient", "accumulate_quotients_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("accumulate_quotients_kernel not found".into())
            })?;

        let eval_point_accumulate = device
            .get_func("quotient", "eval_point_accumulate_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("eval_point_accumulate_kernel not found".into())
            })?;

        // Get copy_column kernel for GPU-resident column gathering
        let copy_column = device
            .get_func("quotient", "copy_column_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("copy_column_kernel not found".into())
            })?;

        // Get MLE kernels for GKR operations
        let mle_fold_base_to_secure = device
            .get_func("quotient", "mle_fold_base_to_secure_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("mle_fold_base_to_secure_kernel not found".into())
            })?;

        let mle_fold_secure = device
            .get_func("quotient", "mle_fold_secure_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("mle_fold_secure_kernel not found".into())
            })?;

        let gen_eq_evals = device
            .get_func("quotient", "gen_eq_evals_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("gen_eq_evals_kernel not found".into())
            })?;

        // Compile or load cached Blake2s Merkle PTX
        use super::fft::BLAKE2S_MERKLE_CUDA_KERNEL;
        let merkle_ptx =
            Self::compile_or_load_cached("merkle_blake2s", BLAKE2S_MERKLE_CUDA_KERNEL)?;

        // Load Merkle PTX into device
        device
            .load_ptx(merkle_ptx, "merkle", &["merkle_layer_kernel"])
            .map_err(|e| CudaFftError::KernelCompilation(format!("Merkle load: {:?}", e)))?;

        // Get Merkle function handle
        let merkle_layer = device
            .get_func("merkle", "merkle_layer_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("merkle_layer_kernel not found".into())
            })?;

        // Compile or load cached Poseidon252 Merkle PTX
        use super::fft::POSEIDON252_MERKLE_CUDA_KERNEL;
        let poseidon_ptx =
            Self::compile_or_load_cached("merkle_poseidon252", POSEIDON252_MERKLE_CUDA_KERNEL)?;

        // Load Poseidon252 Merkle PTX into device
        device
            .load_ptx(
                poseidon_ptx,
                "merkle_poseidon252",
                &[
                    "poseidon252_merkle_layer_kernel",
                    "poseidon252_hash_many_chunked_kernel",
                    "poseidon252_hash_many_chunked_m31_kernel",
                ],
            )
            .map_err(|e| {
                CudaFftError::KernelCompilation(format!("Poseidon252 Merkle load: {:?}", e))
            })?;

        // Get Poseidon252 Merkle function handle
        let poseidon252_merkle_layer = device
            .get_func("merkle_poseidon252", "poseidon252_merkle_layer_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation("poseidon252_merkle_layer_kernel not found".into())
            })?;
        let poseidon252_hash_many_chunked = device
            .get_func("merkle_poseidon252", "poseidon252_hash_many_chunked_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation(
                    "poseidon252_hash_many_chunked_kernel not found".into(),
                )
            })?;
        let poseidon252_hash_many_chunked_m31 = device
            .get_func("merkle_poseidon252", "poseidon252_hash_many_chunked_m31_kernel")
            .ok_or_else(|| {
                CudaFftError::KernelCompilation(
                    "poseidon252_hash_many_chunked_m31_kernel not found".into(),
                )
            })?;

        tracing::info!(
            "Compiled FFT, FRI, Quotient, Merkle, and Poseidon252 Merkle kernels successfully"
        );

        Ok(CompiledKernels {
            bit_reverse,
            ifft_layer,
            fft_layer,
            ifft_shared_mem,
            denormalize,
            denormalize_vec4,
            fold_line,
            fold_circle_into_line,
            deinterleave_aos_to_soa,
            accumulate_quotients,
            eval_point_accumulate,
            copy_column,
            mle_fold_base_to_secure,
            mle_fold_secure,
            gen_eq_evals,
            merkle_layer,
            poseidon252_merkle_layer,
            poseidon252_hash_many_chunked,
            poseidon252_hash_many_chunked_m31,
        })
    }

    /// Execute inverse FFT on GPU.
    ///
    /// # Arguments
    /// * `data` - Input/output data (modified in place)
    /// * `twiddles_dbl` - Doubled twiddle factors for each layer
    /// * `log_size` - log2 of the data size
    ///
    /// # Returns
    /// The modified data after IFFT
    pub fn execute_ifft(
        &self,
        data: &mut [u32],
        twiddles_dbl: &[Vec<u32>],
        log_size: u32,
    ) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;

        if data.len() != n {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} elements, got {}",
                n,
                data.len()
            )));
        }

        let _span =
            tracing::span!(tracing::Level::INFO, "CUDA IFFT", log_size = log_size).entered();

        // Allocate device memory
        let mut d_data = self
            .device
            .htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Flatten twiddles and copy to device
        let flat_twiddles: Vec<u32> = twiddles_dbl.iter().flatten().copied().collect();
        let d_twiddles = self
            .device
            .htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Execute IFFT layers
        self.execute_ifft_layers(&mut d_data, &d_twiddles, log_size, twiddles_dbl)?;

        // Copy results back
        self.device
            .dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok(())
    }

    fn execute_ifft_layers(
        &self,
        d_data: &mut CudaSlice<u32>,
        d_twiddles: &CudaSlice<u32>,
        log_size: u32,
        twiddles_dbl: &[Vec<u32>],
    ) -> Result<(), CudaFftError> {
        let block_size = 256u32;
        let num_layers = twiddles_dbl.len();

        // Validate we have the expected number of twiddle layers
        if num_layers != log_size as usize {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} twiddle layers for log_size={}, got {}",
                log_size, log_size, num_layers
            )));
        }

        // Calculate twiddle offsets (as u32 for GPU)
        let mut twiddle_offsets: Vec<u32> = Vec::new();
        let mut offset = 0u32;
        for tw in twiddles_dbl {
            twiddle_offsets.push(offset);
            offset += tw.len() as u32;
        }

        // Shared memory kernel configuration:
        // - SHMEM_ELEMENTS = 1024 elements per block
        // - SHMEM_BLOCK_SIZE = 256 threads per block
        // - Can process up to 10 layers in shared memory (log2(1024) = 10)
        const SHMEM_ELEMENTS: u32 = 1024;
        const SHMEM_LOG_ELEMENTS: u32 = 10;
        const SHMEM_BLOCK_SIZE: u32 = 256;

        // Determine how many layers we can process in shared memory
        // We can process min(log_size, SHMEM_LOG_ELEMENTS) layers
        let shared_mem_layers = std::cmp::min(log_size, SHMEM_LOG_ELEMENTS);
        let n = 1u32 << log_size;

        if shared_mem_layers > 0 && n >= SHMEM_ELEMENTS {
            // Copy twiddle offsets to device
            let d_twiddle_offsets = self
                .device
                .htod_sync_copy(&twiddle_offsets)
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

            // Launch shared memory kernel for first layers
            let num_blocks = n / SHMEM_ELEMENTS;

            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (SHMEM_BLOCK_SIZE, 1, 1),
                shared_mem_bytes: SHMEM_ELEMENTS * 4, // 4 bytes per u32
            };

            unsafe {
                self.kernels
                    .ifft_shared_mem
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut *d_data,
                            d_twiddles,
                            &d_twiddle_offsets,
                            shared_mem_layers,
                            log_size,
                        ),
                    )
                    .map_err(|e| {
                        CudaFftError::KernelExecution(format!("Shared mem kernel: {:?}", e))
                    })?;
            }

            // Sync after shared memory kernel
            self.device
                .synchronize()
                .map_err(|e| CudaFftError::KernelExecution(format!("Shared mem sync: {:?}", e)))?;

            // Process remaining layers with per-layer kernels
            for layer in (shared_mem_layers as usize)..num_layers {
                let n_twiddles = twiddles_dbl[layer].len() as u32;
                let butterflies_per_twiddle = 1u32 << layer;
                let total_butterflies = n_twiddles * butterflies_per_twiddle;
                let grid_size = (total_butterflies + block_size - 1) / block_size;

                let twiddle_offset = twiddle_offsets[layer] as usize;

                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };

                let twiddle_view = d_twiddles.slice(twiddle_offset..);

                unsafe {
                    self.kernels
                        .ifft_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut *d_data,
                                &twiddle_view,
                                layer as u32,
                                log_size,
                                n_twiddles,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        } else {
            // Small FFT: use per-layer kernels only
            for layer in 0..num_layers {
                let n_twiddles = twiddles_dbl[layer].len() as u32;
                let butterflies_per_twiddle = 1u32 << layer;
                let total_butterflies = n_twiddles * butterflies_per_twiddle;
                let grid_size = (total_butterflies + block_size - 1) / block_size;

                let twiddle_offset = twiddle_offsets[layer] as usize;

                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };

                let twiddle_view = d_twiddles.slice(twiddle_offset..);

                unsafe {
                    self.kernels
                        .ifft_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut *d_data,
                                &twiddle_view,
                                layer as u32,
                                log_size,
                                n_twiddles,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        }

        // Final sync
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Final sync failed: {:?}", e)))?;

        Ok(())
    }

    /// Execute IFFT on GPU memory that's already allocated.
    ///
    /// This is the high-performance path for persistent GPU memory.
    /// Data stays on GPU between calls, avoiding transfer overhead.
    ///
    /// # Arguments
    /// * `d_data` - Device memory containing the data (modified in place)
    /// * `d_twiddles` - Device memory containing flattened twiddles
    /// * `d_twiddle_offsets` - Device memory containing twiddle offsets per layer
    /// * `twiddles_dbl` - CPU twiddles (for layer size info)
    /// * `log_size` - log2 of data size
    pub fn execute_ifft_on_device(
        &self,
        d_data: &mut CudaSlice<u32>,
        d_twiddles: &CudaSlice<u32>,
        d_twiddle_offsets: &CudaSlice<u32>,
        twiddles_dbl: &[Vec<u32>],
        log_size: u32,
    ) -> Result<(), CudaFftError> {
        self.execute_ifft_layers_with_offsets(
            d_data,
            d_twiddles,
            d_twiddle_offsets,
            log_size,
            twiddles_dbl,
        )
    }

    fn execute_ifft_layers_with_offsets(
        &self,
        d_data: &mut CudaSlice<u32>,
        d_twiddles: &CudaSlice<u32>,
        d_twiddle_offsets: &CudaSlice<u32>,
        log_size: u32,
        twiddles_dbl: &[Vec<u32>],
    ) -> Result<(), CudaFftError> {
        let block_size = 256u32;
        let num_layers = twiddles_dbl.len();

        // Shared memory kernel configuration
        const SHMEM_ELEMENTS: u32 = 1024;
        const SHMEM_LOG_ELEMENTS: u32 = 10;
        const SHMEM_BLOCK_SIZE: u32 = 256;

        let shared_mem_layers = std::cmp::min(log_size, SHMEM_LOG_ELEMENTS);
        let n = 1u32 << log_size;

        // Calculate twiddle offsets for per-layer kernels
        let mut twiddle_offsets_cpu: Vec<u32> = Vec::new();
        let mut offset = 0u32;
        for tw in twiddles_dbl {
            twiddle_offsets_cpu.push(offset);
            offset += tw.len() as u32;
        }

        if shared_mem_layers > 0 && n >= SHMEM_ELEMENTS {
            // Launch shared memory kernel for first layers
            let num_blocks = n / SHMEM_ELEMENTS;

            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (SHMEM_BLOCK_SIZE, 1, 1),
                shared_mem_bytes: SHMEM_ELEMENTS * 4,
            };

            unsafe {
                self.kernels
                    .ifft_shared_mem
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut *d_data,
                            d_twiddles,
                            d_twiddle_offsets,
                            shared_mem_layers,
                            log_size,
                        ),
                    )
                    .map_err(|e| {
                        CudaFftError::KernelExecution(format!("Shared mem kernel: {:?}", e))
                    })?;
            }

            // Process remaining layers with per-layer kernels
            for layer in (shared_mem_layers as usize)..num_layers {
                let n_twiddles = twiddles_dbl[layer].len() as u32;
                let butterflies_per_twiddle = 1u32 << layer;
                let total_butterflies = n_twiddles * butterflies_per_twiddle;
                let grid_size = (total_butterflies + block_size - 1) / block_size;

                let twiddle_offset = twiddle_offsets_cpu[layer] as usize;

                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };

                let twiddle_view = d_twiddles.slice(twiddle_offset..);

                unsafe {
                    self.kernels
                        .ifft_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut *d_data,
                                &twiddle_view,
                                layer as u32,
                                log_size,
                                n_twiddles,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        } else {
            // Small FFT: use per-layer kernels only
            for layer in 0..num_layers {
                let n_twiddles = twiddles_dbl[layer].len() as u32;
                let butterflies_per_twiddle = 1u32 << layer;
                let total_butterflies = n_twiddles * butterflies_per_twiddle;
                let grid_size = (total_butterflies + block_size - 1) / block_size;

                let twiddle_offset = twiddle_offsets_cpu[layer] as usize;

                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };

                let twiddle_view = d_twiddles.slice(twiddle_offset..);

                unsafe {
                    self.kernels
                        .ifft_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut *d_data,
                                &twiddle_view,
                                layer as u32,
                                log_size,
                                n_twiddles,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        }

        // Sync at the end
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Final sync failed: {:?}", e)))?;

        Ok(())
    }

    /// Execute forward FFT on GPU.
    pub fn execute_fft(
        &self,
        data: &mut [u32],
        twiddles: &[Vec<u32>],
        log_size: u32,
    ) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;

        if data.len() != n {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} elements, got {}",
                n,
                data.len()
            )));
        }

        let _span = tracing::span!(tracing::Level::INFO, "CUDA FFT", log_size = log_size).entered();

        // Allocate device memory
        let mut d_data = self
            .device
            .htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Flatten twiddles and copy to device
        let flat_twiddles: Vec<u32> = twiddles.iter().flatten().copied().collect();
        let d_twiddles = self
            .device
            .htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Execute FFT layers (reverse order of IFFT)
        self.execute_fft_layers(&mut d_data, &d_twiddles, log_size, twiddles)?;

        // Copy results back
        self.device
            .dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok(())
    }

    fn execute_fft_layers(
        &self,
        d_data: &mut CudaSlice<u32>,
        d_twiddles: &CudaSlice<u32>,
        log_size: u32,
        twiddles: &[Vec<u32>],
    ) -> Result<(), CudaFftError> {
        let block_size = 256u32;
        let num_layers = twiddles.len();

        // Validate we have the expected number of twiddle layers
        if num_layers != log_size as usize {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} twiddle layers for log_size={}, got {}",
                log_size, log_size, num_layers
            )));
        }

        // Calculate twiddle offsets
        let mut twiddle_offsets: Vec<usize> = Vec::new();
        let mut offset = 0usize;
        for tw in twiddles {
            twiddle_offsets.push(offset);
            offset += tw.len();
        }

        // Execute layers in reverse order for forward FFT
        // Layer 0 is circle layer, layers 1+ are line layers
        for layer in (0..num_layers).rev() {
            let n_twiddles = twiddles[layer].len() as u32;
            let butterflies_per_twiddle = 1u32 << layer;
            let total_butterflies = n_twiddles * butterflies_per_twiddle;
            let grid_size = (total_butterflies + block_size - 1) / block_size;

            let twiddle_offset = twiddle_offsets[layer];

            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            // Create a twiddle view for this layer
            let twiddle_view = d_twiddles.slice(twiddle_offset..);

            unsafe {
                // Reborrow d_data each iteration to avoid move in loop
                self.kernels
                    .fft_layer
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut *d_data,
                            &twiddle_view,
                            layer as u32,
                            log_size,
                            n_twiddles,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
        }

        // Synchronize
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        Ok(())
    }

    /// Execute bit reversal permutation on GPU.
    pub fn bit_reverse(&self, data: &mut [u32], log_size: u32) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;

        if data.len() != n {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} elements, got {}",
                n,
                data.len()
            )));
        }

        // Allocate and copy
        let mut d_data = self
            .device
            .htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch bit reverse kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .bit_reverse
                .clone()
                .launch(cfg, (&mut d_data, log_size))
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        self.device
            .dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok(())
    }

    /// Get device memory info.
    pub fn memory_info(&self) -> (usize, usize) {
        // (free, total) - cudarc doesn't expose this directly
        (
            self.device_info.total_memory_bytes / 2, // Estimate
            self.device_info.total_memory_bytes,
        )
    }

    // =========================================================================
    // Denormalization Operations
    // =========================================================================

    /// Execute denormalization on GPU memory.
    ///
    /// After IFFT, we need to divide by the domain size to get correct coefficients.
    /// This multiplies each element by the precomputed inverse of the domain size.
    ///
    /// # Arguments
    /// * `d_data` - Device memory containing the data (modified in place)
    /// * `denorm_factor` - Precomputed 1/n mod P
    /// * `n` - Number of elements
    pub fn execute_denormalize_on_device(
        &self,
        d_data: &mut CudaSlice<u32>,
        denorm_factor: u32,
        n: u32,
    ) -> Result<(), CudaFftError> {
        let _span = tracing::span!(tracing::Level::DEBUG, "CUDA denormalize", n = n).entered();

        // Use vectorized kernel for large sizes (must be multiple of 4)
        if n >= 1024 && n % 4 == 0 {
            let block_size = 256u32;
            let grid_size = (n / 4 + block_size - 1) / block_size;

            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.kernels
                    .denormalize_vec4
                    .clone()
                    .launch(cfg, (&mut *d_data, denorm_factor, n))
                    .map_err(|e| {
                        CudaFftError::KernelExecution(format!("Denormalize vec4: {:?}", e))
                    })?;
            }
        } else {
            // Fall back to scalar kernel for small sizes or non-multiple-of-4
            let block_size = 256u32;
            let grid_size = (n + block_size - 1) / block_size;

            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.kernels
                    .denormalize
                    .clone()
                    .launch(cfg, (&mut *d_data, denorm_factor, n))
                    .map_err(|e| CudaFftError::KernelExecution(format!("Denormalize: {:?}", e)))?;
            }
        }

        Ok(())
    }

    /// Execute denormalization on CPU data (transfers to/from GPU).
    pub fn execute_denormalize(
        &self,
        data: &mut [u32],
        denorm_factor: u32,
    ) -> Result<(), CudaFftError> {
        let n = data.len() as u32;

        // Allocate and copy
        let mut d_data = self
            .device
            .htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Execute on device
        self.execute_denormalize_on_device(&mut d_data, denorm_factor, n)?;

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        self.device
            .dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok(())
    }

    // =========================================================================
    // FRI Folding Operations
    // =========================================================================

    /// Execute FRI fold_line on GPU.
    ///
    /// Folds a line evaluation by factor of 2 using the FRI protocol.
    pub fn execute_fold_line(
        &self,
        input: &[u32],
        itwiddles: &[u32],
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        // Validate input
        if input.len() != n * 4 {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} u32 values, got {}",
                n * 4,
                input.len()
            )));
        }
        if itwiddles.len() < n / 2 {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected at least {} twiddles, got {}",
                n / 2,
                itwiddles.len()
            )));
        }

        let _span = tracing::span!(tracing::Level::INFO, "CUDA fold_line", n = n).entered();

        let n_output = n / 2;

        // Allocate device memory
        let d_input = self
            .device
            .htod_sync_copy(input)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_itwiddles = self
            .device
            .htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(n_output * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_output as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_line
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        &d_input,
                        &d_itwiddles,
                        &d_alpha,
                        n as u32,
                        log_n,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; n_output * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!("GPU fold_line completed: {} -> {} elements", n, n_output);

        Ok(output)
    }

    /// Execute FRI fold_circle_into_line on GPU.
    ///
    /// Folds circle evaluation into line evaluation (accumulated).
    pub fn execute_fold_circle_into_line(
        &self,
        dst: &mut [u32],
        src: &[u32],
        itwiddles: &[u32],
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<(), CudaFftError> {
        // Validate input
        if src.len() != n * 4 {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} u32 values in src, got {}",
                n * 4,
                src.len()
            )));
        }
        let n_dst = n / 2;
        if dst.len() != n_dst * 4 {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} u32 values in dst, got {}",
                n_dst * 4,
                dst.len()
            )));
        }
        if itwiddles.len() < n_dst {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected at least {} twiddles, got {}",
                n_dst,
                itwiddles.len()
            )));
        }

        let _span =
            tracing::span!(tracing::Level::INFO, "CUDA fold_circle_into_line", n = n).entered();

        // Allocate device memory
        let d_src = self
            .device
            .htod_sync_copy(src)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_dst = self
            .device
            .htod_sync_copy(dst)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_itwiddles = self
            .device
            .htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_dst as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_circle_into_line
                .clone()
                .launch(
                    cfg,
                    (&mut d_dst, &d_src, &d_itwiddles, &d_alpha, n as u32, log_n),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        self.device
            .dtoh_sync_copy_into(&d_dst, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU fold_circle_into_line completed: {} -> {} elements",
            n,
            n_dst
        );

        Ok(())
    }

    // =========================================================================
    // GPU-Resident FRI Folding Operations
    // =========================================================================

    /// Execute FRI fold_line with GPU-resident input.
    ///
    /// Unlike `execute_fold_line`, this accepts a `CudaSlice<u32>` that is
    /// already on the GPU (from a previous fold round), eliminating the H2D
    /// transfer. Returns both the GPU-resident output (for caching) and a
    /// CPU copy (for decommitment).
    pub fn execute_fold_line_resident(
        &self,
        d_input: &CudaSlice<u32>,
        itwiddles: &[u32],
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<(CudaSlice<u32>, Vec<u32>), CudaFftError> {
        let _span =
            tracing::span!(tracing::Level::INFO, "CUDA fold_line_resident", n = n).entered();

        let n_output = n / 2;

        // Twiddles and alpha still come from CPU (small, ~n/2 u32s)
        let d_itwiddles = self
            .device
            .htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(n_output * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_output as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_line
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        d_input,
                        &d_itwiddles,
                        &d_alpha,
                        n as u32,
                        log_n,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and D2H copy (needed for decommitment)
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut cpu_output = vec![0u32; n_output * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut cpu_output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU fold_line_resident completed: {} -> {} elements (0 H2D)",
            n,
            n_output
        );

        Ok((d_output, cpu_output))
    }

    /// Like `execute_fold_line_resident` but accepts pre-uploaded twiddles on GPU,
    /// eliminating the per-round twiddle H2D transfer.
    pub fn execute_fold_line_resident_preloaded(
        &self,
        d_input: &CudaSlice<u32>,
        d_itwiddles: &CudaSlice<u32>,
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<(CudaSlice<u32>, Vec<u32>), CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA fold_line_resident_preloaded",
            n = n
        )
        .entered();

        let n_output = n / 2;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(n_output * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_output as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_line
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        d_input,
                        d_itwiddles,
                        &d_alpha,
                        n as u32,
                        log_n,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // dtoh_sync_copy_into implicitly waits for kernel completion
        let mut cpu_output = vec![0u32; n_output * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut cpu_output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU fold_line_resident_preloaded completed: {} -> {} elements",
            n,
            n_output
        );

        Ok((d_output, cpu_output))
    }

    /// Execute FRI fold_line keeping output GPU-resident only (no D2H transfer).
    ///
    /// Returns just the `CudaSlice<u32>` output on GPU. The caller is responsible
    /// for deferring the D2H download until the CPU data is actually needed
    /// (e.g., during decommit). This avoids the synchronous D2H that
    /// `execute_fold_line_resident_preloaded` performs every round.
    pub fn execute_fold_line_gpu_only(
        &self,
        d_input: &CudaSlice<u32>,
        d_itwiddles: &CudaSlice<u32>,
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let _span =
            tracing::span!(tracing::Level::INFO, "CUDA fold_line_gpu_only", n = n).entered();

        let n_output = n / 2;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(n_output * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_output as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_line
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        d_input,
                        d_itwiddles,
                        &d_alpha,
                        n as u32,
                        log_n,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize to ensure kernel completion without downloading data
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;

        tracing::info!(
            "GPU fold_line_gpu_only completed: {} -> {} elements (no D2H)",
            n,
            n_output
        );

        Ok(d_output)
    }

    /// Execute FRI fold_circle_into_line with GPU-resident output caching.
    ///
    /// This performs the fold and returns the GPU-resident output `CudaSlice`
    /// alongside the CPU result, so the next `fold_line` can skip H2D.
    pub fn execute_fold_circle_into_line_resident(
        &self,
        dst: &mut [u32],
        src: &[u32],
        itwiddles: &[u32],
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n_dst = n / 2;

        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA fold_circle_into_line_resident",
            n = n
        )
        .entered();

        // All inputs come from CPU for fold_circle_into_line (first fold)
        let d_src = self
            .device
            .htod_sync_copy(src)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Allocate dst on GPU — kernel overwrites all output, no need to upload CPU data
        let mut d_dst: CudaSlice<u32> = unsafe { self.device.alloc::<u32>(n_dst * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_itwiddles = self
            .device
            .htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_dst as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_circle_into_line
                .clone()
                .launch(
                    cfg,
                    (&mut d_dst, &d_src, &d_itwiddles, &d_alpha, n as u32, log_n),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and D2H copy
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        self.device
            .dtoh_sync_copy_into(&d_dst, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU fold_circle_into_line_resident completed: {} -> {} elements",
            n,
            n_dst
        );

        Ok(d_dst)
    }

    /// Like `execute_fold_circle_into_line_resident` but accepts pre-uploaded
    /// twiddles on GPU, eliminating the twiddle H2D transfer.
    pub fn execute_fold_circle_into_line_resident_preloaded(
        &self,
        dst: &mut [u32],
        src: &[u32],
        d_itwiddles: &CudaSlice<u32>,
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n_dst = n / 2;

        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA fold_circle_into_line_resident_preloaded",
            n = n
        )
        .entered();

        let d_src = self
            .device
            .htod_sync_copy(src)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Allocate dst on GPU zeroed — kernel reads dst (dst = dst*alpha_sq + f_prime)
        let mut d_dst: CudaSlice<u32> = self
            .device
            .alloc_zeros::<u32>(n_dst * 4)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_dst as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_circle_into_line
                .clone()
                .launch(
                    cfg,
                    (&mut d_dst, &d_src, d_itwiddles, &d_alpha, n as u32, log_n),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        self.device
            .dtoh_sync_copy_into(&d_dst, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU fold_circle_into_line_resident_preloaded completed: {} -> {} elements",
            n,
            n_dst
        );

        Ok(d_dst)
    }

    /// Like `execute_fold_circle_into_line_resident_preloaded`, but takes
    /// a GPU-resident source (`d_src`) instead of a CPU slice. Avoids the
    /// H2D transfer for `src` when the data is already on the device.
    pub fn execute_fold_circle_into_line_from_gpu(
        &self,
        dst: &mut [u32],
        d_src: &CudaSlice<u32>,
        d_itwiddles: &CudaSlice<u32>,
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n_dst = n / 2;

        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA fold_circle_into_line_from_gpu",
            n = n
        )
        .entered();

        // Allocate dst on GPU zeroed — kernel reads dst (dst = dst*alpha_sq + f_prime)
        let mut d_dst: CudaSlice<u32> = self
            .device
            .alloc_zeros::<u32>(n_dst * 4)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_dst as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_circle_into_line
                .clone()
                .launch(
                    cfg,
                    (&mut d_dst, d_src, d_itwiddles, &d_alpha, n as u32, log_n),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        self.device
            .dtoh_sync_copy_into(&d_dst, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU fold_circle_into_line_from_gpu completed: {} -> {} elements",
            n,
            n_dst
        );

        Ok(d_dst)
    }

    /// GPU-only fold_circle_into_line: src on GPU, dst allocated on GPU (no upload),
    /// no D2H. Returns GPU-resident output. Eliminates the wasted H2D of dst
    /// (kernel overwrites all output) and defers D2H to caller.
    pub fn execute_fold_circle_into_line_gpu_only(
        &self,
        d_src: &CudaSlice<u32>,
        d_itwiddles: &CudaSlice<u32>,
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n_dst = n / 2;

        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA fold_circle_into_line_gpu_only",
            n = n
        )
        .entered();

        // Allocate dst on GPU — no need to upload CPU data since kernel overwrites all output
        let mut d_dst: CudaSlice<u32> = unsafe { self.device.alloc::<u32>(n_dst * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_dst as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_circle_into_line
                .clone()
                .launch(
                    cfg,
                    (&mut d_dst, d_src, d_itwiddles, &d_alpha, n as u32, log_n),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;

        tracing::info!(
            "GPU fold_circle_into_line_gpu_only completed: {} -> {} elements (no D2H)",
            n,
            n_dst
        );

        Ok(d_dst)
    }

    /// Fully GPU-resident fold_circle_into_line: both dst and src on GPU, no D2H.
    /// Used when dst already has GPU-cached data from a prior fold_line (GPU-resident pipeline).
    pub fn execute_fold_circle_into_line_fully_gpu(
        &self,
        d_dst: &CudaSlice<u32>,
        d_src: &CudaSlice<u32>,
        d_itwiddles: &CudaSlice<u32>,
        alpha: &[u32; 4],
        n: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n_dst = n / 2;

        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA fold_circle_into_line_fully_gpu",
            n = n
        )
        .entered();

        // Copy dst to mutable output (kernel accumulates: dst = dst * alpha^2 + fold_result)
        let mut d_output: CudaSlice<u32> = unsafe { self.device.alloc::<u32>(n_dst * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        self.device
            .dtod_copy(d_dst, &mut d_output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod: {:?}", e)))?;

        let d_alpha = self
            .device
            .htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_dst as u32) + block_size - 1) / block_size;
        let log_n = n.ilog2();

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .fold_circle_into_line
                .clone()
                .launch(
                    cfg,
                    (&mut d_output, d_src, d_itwiddles, &d_alpha, n as u32, log_n),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;

        tracing::info!(
            "GPU fold_circle_into_line_fully_gpu completed: {} -> {} elements (no D2H)",
            n,
            n_dst
        );

        Ok(d_output)
    }

    /// De-interleave AoS GPU buffer into 4 SoA column CudaSlices on GPU.
    /// Input: [c0,c1,c2,c3, c0,c1,c2,c3, ...] (n*4 u32s)
    /// Output: 4 CudaSlices each of length n
    pub fn execute_deinterleave_aos_to_soa(
        &self,
        d_aos: &CudaSlice<u32>,
        n: usize,
    ) -> Result<[CudaSlice<u32>; 4], CudaFftError> {
        let mut d_col0 = unsafe { self.device.alloc::<u32>(n) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_col1 = unsafe { self.device.alloc::<u32>(n) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_col2 = unsafe { self.device.alloc::<u32>(n) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_col3 = unsafe { self.device.alloc::<u32>(n) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .deinterleave_aos_to_soa
                .clone()
                .launch(
                    cfg,
                    (
                        d_aos,
                        &mut d_col0,
                        &mut d_col1,
                        &mut d_col2,
                        &mut d_col3,
                        n as u32,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        Ok([d_col0, d_col1, d_col2, d_col3])
    }

    // =========================================================================
    // Quotient Accumulation Operations
    // =========================================================================

    /// Evaluate one OODS point directly from polynomial coefficients on GPU.
    ///
    /// `coeffs` are M31 coefficients in FFT basis order.
    /// `twiddles_aos` are per-coefficient QM31 twiddles packed as AoS:
    /// `[t0.a0,t0.a1,t0.a2,t0.a3, t1.a0,...]`.
    pub fn execute_eval_point_from_coeffs(
        &self,
        coeffs: &[u32],
        twiddles_aos: &[u32],
    ) -> Result<[u32; 4], CudaFftError> {
        let n_coeffs = coeffs.len();
        if twiddles_aos.len() != n_coeffs * 4 {
            return Err(CudaFftError::InvalidSize(format!(
                "twiddles length mismatch: expected {}, got {}",
                n_coeffs * 4,
                twiddles_aos.len()
            )));
        }
        if n_coeffs == 0 {
            return Ok([0; 4]);
        }

        let d_coeffs = self
            .device
            .htod_sync_copy(coeffs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let d_twiddles = self
            .device
            .htod_sync_copy(twiddles_aos)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_accum = self
            .device
            .alloc_zeros::<u64>(4)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_coeffs as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .eval_point_accumulate
                .clone()
                .launch(cfg, (&d_coeffs, &d_twiddles, &mut d_accum, n_coeffs as u32))
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let accum = self
            .device
            .dtoh_sync_copy(&d_accum)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok([
            crate::core::fields::m31::M31::reduce(accum[0]).0,
            crate::core::fields::m31::M31::reduce(accum[1]).0,
            crate::core::fields::m31::M31::reduce(accum[2]).0,
            crate::core::fields::m31::M31::reduce(accum[3]).0,
        ])
    }

    /// Upload quotient columns once and return a reusable device buffer.
    pub fn upload_accumulate_columns(
        &self,
        columns: &[Vec<u32>],
        n_points: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n_columns = columns.len();

        // Flatten columns (interleave by point, not by column)
        // Layout: col0[0], col1[0], col2[0], ..., col0[1], col1[1], ...
        let mut flat_columns: Vec<u32> = Vec::with_capacity(n_columns * n_points);
        for col in columns {
            flat_columns.extend_from_slice(col);
        }

        self.device
            .htod_sync_copy(&flat_columns)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))
    }

    /// Execute quotient accumulation on GPU using pre-uploaded columns.
    pub fn execute_accumulate_quotients_with_device_columns(
        &self,
        d_columns: &CudaSlice<u32>,
        n_columns: usize,
        line_coeffs: &[[u32; 12]],
        denom_inv: &[u32],
        batch_sizes: &[usize],
        col_indices: &[usize],
        n_points: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        let n_batches = batch_sizes.len();

        // Flatten line coefficients
        let flat_line_coeffs: Vec<u32> = line_coeffs
            .iter()
            .flat_map(|coeffs| coeffs.iter().copied())
            .collect();

        // Convert batch_sizes and col_indices to u32
        let batch_sizes_u32: Vec<u32> = batch_sizes.iter().map(|&s| s as u32).collect();
        let col_indices_u32: Vec<u32> = col_indices.iter().map(|&i| i as u32).collect();

        let d_line_coeffs = self
            .device
            .htod_sync_copy(&flat_line_coeffs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_denom_inv = self
            .device
            .htod_sync_copy(denom_inv)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_batch_sizes = self
            .device
            .htod_sync_copy(&batch_sizes_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_col_indices = self
            .device
            .htod_sync_copy(&col_indices_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(n_points * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_points as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .accumulate_quotients
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        d_columns,
                        &d_line_coeffs,
                        &d_denom_inv,
                        &d_batch_sizes,
                        &d_col_indices,
                        n_batches as u32,
                        n_points as u32,
                        n_columns as u32,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; n_points * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok(output)
    }

    /// Execute quotient accumulation on GPU.
    pub fn execute_accumulate_quotients(
        &self,
        columns: &[Vec<u32>],
        line_coeffs: &[[u32; 12]],
        denom_inv: &[u32],
        batch_sizes: &[usize],
        col_indices: &[usize],
        n_points: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA accumulate_quotients",
            n_points = n_points
        )
        .entered();

        let n_columns = columns.len();
        let n_batches = batch_sizes.len();

        let d_columns = self.upload_accumulate_columns(columns, n_points)?;
        let output = self.execute_accumulate_quotients_with_device_columns(
            &d_columns,
            n_columns,
            line_coeffs,
            denom_inv,
            batch_sizes,
            col_indices,
            n_points,
        )?;

        tracing::info!(
            "GPU accumulate_quotients completed: {} points, {} batches",
            n_points,
            n_batches
        );

        Ok(output)
    }

    // =========================================================================
    // Merkle Hashing Operations
    // =========================================================================

    /// Execute Blake2s Merkle hashing on GPU.
    pub fn execute_blake2s_merkle(
        &self,
        columns: &[Vec<u32>],
        prev_layer: Option<&[u8]>,
        n_hashes: usize,
    ) -> Result<Vec<u8>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA blake2s_merkle",
            n_hashes = n_hashes
        )
        .entered();

        let n_columns = columns.len();

        // Flatten columns
        let flat_columns: Vec<u32> = columns.iter().flat_map(|col| col.iter().copied()).collect();

        // Allocate device memory for columns (if any)
        let d_columns = if n_columns > 0 {
            Some(
                self.device
                    .htod_sync_copy(&flat_columns)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        } else {
            None
        };

        // Allocate device memory for previous layer (if any)
        let d_prev_layer = if let Some(prev) = prev_layer {
            Some(
                self.device
                    .htod_sync_copy(prev)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        } else {
            None
        };

        // Allocate output (32 bytes per hash)
        let mut d_output = unsafe { self.device.alloc::<u8>(n_hashes * 32) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_hashes as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let has_prev_layer = if prev_layer.is_some() { 1u32 } else { 0u32 };

        unsafe {
            // We need to handle the optional parameters carefully
            // If columns is None, pass a null-like slice
            // If prev_layer is None, pass a null-like slice

            match (&d_columns, &d_prev_layer) {
                (Some(cols), Some(prev)) => {
                    self.kernels
                        .merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                cols,
                                prev,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev_layer,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (Some(cols), None) => {
                    // Need a dummy buffer for prev_layer
                    let dummy_prev = self
                        .device
                        .alloc::<u8>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                cols,
                                &dummy_prev,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev_layer,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, Some(prev)) => {
                    // Need a dummy buffer for columns
                    let dummy_cols = self
                        .device
                        .alloc::<u32>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                &dummy_cols,
                                prev,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev_layer,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, None) => {
                    return Err(CudaFftError::InvalidSize(
                        "Merkle hashing requires either columns or prev_layer".to_string(),
                    ));
                }
            }
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u8; n_hashes * 32];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!("GPU blake2s_merkle completed: {} hashes", n_hashes);

        Ok(output)
    }

    // =========================================================================
    // Poseidon252 Merkle GPU Kernel
    // =========================================================================

    /// Execute Poseidon252 Merkle layer hashing on GPU.
    ///
    /// # Arguments
    /// * `columns` - Column data (each column is a Vec<u32> of M31 values)
    /// * `prev_layer` - Previous layer hashes (32 bytes / 4×u64 per node), or None for leaves
    /// * `n_hashes` - Number of hash nodes to compute
    /// * `d_round_constants` - Device buffer with 107 compressed Poseidon252 round constants (4×u64 each)
    pub fn execute_poseidon252_merkle(
        &self,
        columns: &[Vec<u32>],
        prev_layer: Option<&[u64]>,
        n_hashes: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<Vec<u64>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA poseidon252_merkle",
            n_hashes = n_hashes
        )
        .entered();

        let n_columns = columns.len();
        let col_stride = n_hashes as u32;

        // Flatten columns (column-major: col0[0..n], col1[0..n], ...)
        let flat_columns: Vec<u32> = columns.iter().flat_map(|col| col.iter().copied()).collect();

        // Upload columns
        let d_columns = if n_columns > 0 {
            Some(
                self.device
                    .htod_sync_copy(&flat_columns)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        } else {
            None
        };

        // Upload prev_layer (4 u64s per node, 2*n_hashes nodes)
        let d_prev_layer = if let Some(prev) = prev_layer {
            Some(
                self.device
                    .htod_sync_copy(prev)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        } else {
            None
        };

        // Output: 4 u64s per hash node
        let mut d_output = unsafe { self.device.alloc::<u64>(n_hashes * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_hashes as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let has_prev: u32 = if prev_layer.is_some() { 1 } else { 0 };

        unsafe {
            match (&d_columns, &d_prev_layer) {
                (Some(cols), Some(prev)) => {
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                cols,
                                prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (Some(cols), None) => {
                    let dummy_prev = self
                        .device
                        .alloc::<u64>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                cols,
                                &dummy_prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, Some(prev)) => {
                    let dummy_cols = self
                        .device
                        .alloc::<u32>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                &dummy_cols,
                                prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, None) => {
                    return Err(CudaFftError::InvalidSize(
                        "Poseidon252 Merkle hashing requires either columns or prev_layer"
                            .to_string(),
                    ));
                }
            }
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u64; n_hashes * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!("GPU poseidon252_merkle completed: {} hashes", n_hashes);
        Ok(output)
    }

    /// Execute chunked Poseidon252 hash_many for many independent segments.
    ///
    /// Each segment `i` is described by `offsets[i]` and `lengths[i]` over `inputs`
    /// (where `inputs` is flattened felt252 data in 4x u64 limbs per element).
    /// The kernel computes:
    ///   running = 0
    ///   for each chunk:
    ///     running = poseidon_hash_many([running] + chunk)
    ///
    /// Output is one felt252 hash per segment (4x u64 limbs each).
    pub fn execute_poseidon252_hash_many_chunked(
        &self,
        inputs: &[u64],
        offsets: &[u32],
        lengths: &[u32],
        chunk_size: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<Vec<u64>, CudaFftError> {
        if offsets.len() != lengths.len() {
            return Err(CudaFftError::InvalidSize(format!(
                "offsets/lengths mismatch: {} vs {}",
                offsets.len(),
                lengths.len()
            )));
        }
        if inputs.len() % 4 != 0 {
            return Err(CudaFftError::InvalidSize(format!(
                "inputs must be packed felt252 limbs (len % 4 == 0), got {}",
                inputs.len()
            )));
        }
        if chunk_size == 0 {
            return Err(CudaFftError::InvalidSize(
                "chunk_size must be > 0".to_string(),
            ));
        }

        let n_segments = offsets.len();
        if n_segments == 0 {
            return Ok(Vec::new());
        }

        let d_inputs = if inputs.is_empty() {
            None
        } else {
            Some(
                self.device
                    .htod_sync_copy(inputs)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        };
        let d_offsets = self
            .device
            .htod_sync_copy(offsets)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let d_lengths = self
            .device
            .htod_sync_copy(lengths)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u64>(n_segments * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_segments as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            if let Some(d_inputs) = &d_inputs {
                self.kernels
                    .poseidon252_hash_many_chunked
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_output,
                            d_inputs,
                            &d_offsets,
                            &d_lengths,
                            d_round_constants,
                            n_segments as u32,
                            chunk_size as u32,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            } else {
                // No felt inputs: segments are empty, so kernel should return 0 per segment.
                let dummy_inputs = self
                    .device
                    .alloc::<u64>(4)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                self.kernels
                    .poseidon252_hash_many_chunked
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_output,
                            &dummy_inputs,
                            &d_offsets,
                            &d_lengths,
                            d_round_constants,
                            n_segments as u32,
                            chunk_size as u32,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u64; n_segments * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        Ok(output)
    }

    /// Execute chunked Poseidon252 hash_many for many independent segments over raw M31 inputs.
    ///
    /// Each segment `i` is described by `offsets[i]` and `lengths[i]` over `inputs_m31`.
    /// M31 values are packed on GPU in base-2^31 groups of 7 (same semantics as CPU packing),
    /// then hashed as:
    ///   running = 0
    ///   for each packed chunk:
    ///     running = poseidon_hash_many([running] + chunk)
    pub fn execute_poseidon252_hash_many_chunked_m31(
        &self,
        inputs_m31: &[u32],
        offsets: &[u32],
        lengths: &[u32],
        chunk_size: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<Vec<u64>, CudaFftError> {
        if offsets.len() != lengths.len() {
            return Err(CudaFftError::InvalidSize(format!(
                "offsets/lengths mismatch: {} vs {}",
                offsets.len(),
                lengths.len()
            )));
        }
        if chunk_size == 0 {
            return Err(CudaFftError::InvalidSize(
                "chunk_size must be > 0".to_string(),
            ));
        }

        let n_segments = offsets.len();
        if n_segments == 0 {
            return Ok(Vec::new());
        }

        let d_inputs = if inputs_m31.is_empty() {
            None
        } else {
            Some(
                self.device
                    .htod_sync_copy(inputs_m31)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        };
        let d_offsets = self
            .device
            .htod_sync_copy(offsets)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let d_lengths = self
            .device
            .htod_sync_copy(lengths)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u64>(n_segments * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_segments as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            if let Some(d_inputs) = &d_inputs {
                self.kernels
                    .poseidon252_hash_many_chunked_m31
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_output,
                            d_inputs,
                            &d_offsets,
                            &d_lengths,
                            d_round_constants,
                            n_segments as u32,
                            chunk_size as u32,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            } else {
                let dummy_inputs = self
                    .device
                    .alloc::<u32>(1)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                self.kernels
                    .poseidon252_hash_many_chunked_m31
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_output,
                            &dummy_inputs,
                            &d_offsets,
                            &d_lengths,
                            d_round_constants,
                            n_segments as u32,
                            chunk_size as u32,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u64; n_segments * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        Ok(output)
    }

    /// Execute Poseidon252 Merkle layer hashing with pre-uploaded GPU column data.
    ///
    /// Like `execute_poseidon252_merkle`, but accepts a `CudaSlice<u32>` that is
    /// already on GPU in column-major SoA layout, eliminating the H2D transfer.
    pub fn execute_poseidon252_merkle_from_gpu(
        &self,
        d_columns: &CudaSlice<u32>,
        n_columns: usize,
        prev_layer: Option<&[u64]>,
        n_hashes: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<Vec<u64>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA poseidon252_merkle_from_gpu",
            n_hashes = n_hashes
        )
        .entered();

        let col_stride = n_hashes as u32;

        let d_prev_layer = if let Some(prev) = prev_layer {
            Some(
                self.device
                    .htod_sync_copy(prev)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        } else {
            None
        };

        let mut d_output = unsafe { self.device.alloc::<u64>(n_hashes * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_hashes as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let has_prev: u32 = if prev_layer.is_some() { 1 } else { 0 };

        unsafe {
            match &d_prev_layer {
                Some(prev) => {
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                d_columns,
                                prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                None => {
                    let dummy_prev = self
                        .device
                        .alloc::<u64>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                d_columns,
                                &dummy_prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u64; n_hashes * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU poseidon252_merkle_from_gpu completed: {} hashes (no column H2D)",
            n_hashes
        );
        Ok(output)
    }

    /// Execute Poseidon252 Merkle layer with GPU-resident prev_layer.
    ///
    /// Takes `d_prev_layer` as an already-uploaded `CudaSlice<u64>` instead of
    /// a CPU slice, eliminating the prev_layer serialize + H2D per layer.
    /// Returns `(Vec<u64>, CudaSlice<u64>)` — both CPU output (for trait) and
    /// GPU output (to cache for next layer).
    pub fn execute_poseidon252_merkle_gpu_prev(
        &self,
        columns: &[Vec<u32>],
        d_prev_layer: Option<&CudaSlice<u64>>,
        n_hashes: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<(Vec<u64>, CudaSlice<u64>), CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA poseidon252_merkle_gpu_prev",
            n_hashes = n_hashes
        )
        .entered();

        let n_columns = columns.len();
        let col_stride = n_hashes as u32;

        // Upload columns (if any)
        let d_columns = if n_columns > 0 {
            let flat_columns: Vec<u32> =
                columns.iter().flat_map(|col| col.iter().copied()).collect();
            Some(
                self.device
                    .htod_sync_copy(&flat_columns)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        } else {
            None
        };

        let mut d_output = unsafe { self.device.alloc::<u64>(n_hashes * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_hashes as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let has_prev: u32 = if d_prev_layer.is_some() { 1 } else { 0 };

        unsafe {
            match (&d_columns, d_prev_layer) {
                (Some(cols), Some(prev)) => {
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                cols,
                                prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (Some(cols), None) => {
                    let dummy_prev = self
                        .device
                        .alloc::<u64>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                cols,
                                &dummy_prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, Some(prev)) => {
                    let dummy_cols = self
                        .device
                        .alloc::<u32>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                &dummy_cols,
                                prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, None) => {
                    return Err(CudaFftError::InvalidSize(
                        "Poseidon252 Merkle requires columns or prev_layer".into(),
                    ));
                }
            }
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync: {:?}", e)))?;

        let mut output = vec![0u64; n_hashes * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::debug!(
            "GPU poseidon252_merkle_gpu_prev: {} hashes (prev on GPU)",
            n_hashes
        );
        Ok((output, d_output))
    }

    /// Execute Poseidon252 Merkle layer with BOTH columns and prev_layer on GPU.
    ///
    /// Fully GPU-resident path: no H2D at all. Returns both CPU and GPU output.
    pub fn execute_poseidon252_merkle_fully_resident(
        &self,
        d_columns: &CudaSlice<u32>,
        n_columns: usize,
        d_prev_layer: Option<&CudaSlice<u64>>,
        n_hashes: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<(Vec<u64>, CudaSlice<u64>), CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA poseidon252_merkle_fully_resident",
            n_hashes = n_hashes
        )
        .entered();

        let col_stride = n_hashes as u32;

        let mut d_output = unsafe { self.device.alloc::<u64>(n_hashes * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((n_hashes as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let has_prev: u32 = if d_prev_layer.is_some() { 1 } else { 0 };

        unsafe {
            match d_prev_layer {
                Some(prev) => {
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                d_columns,
                                prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                None => {
                    let dummy_prev = self
                        .device
                        .alloc::<u64>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .poseidon252_merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                d_columns,
                                &dummy_prev,
                                d_round_constants,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev,
                                col_stride,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync: {:?}", e)))?;

        let mut output = vec![0u64; n_hashes * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::debug!(
            "GPU poseidon252_merkle_fully_resident: {} hashes (fully GPU)",
            n_hashes
        );
        Ok((output, d_output))
    }

    // =========================================================================
    // Full Poseidon252 Merkle Tree (all layers, one GPU pass)
    // =========================================================================

    /// Build an entire Poseidon252 Merkle tree on GPU in one pass.
    ///
    /// Launches all layer kernels sequentially on the same CUDA stream (implicit
    /// ordering) with NO sync between layers. One final sync + bulk D2H.
    ///
    /// Returns `Vec<Vec<u64>>` where index 0 = leaf layer (n_hashes elements × 4 u64),
    /// index 1 = n_hashes/2, etc. down to the root layer (1 element × 4 u64).
    ///
    /// `d_columns` and `n_columns` describe the leaf data (column-major SoA on GPU).
    /// If `d_prev_leaf` is Some, it's incorporated into the leaf hash.
    pub fn execute_poseidon252_merkle_full_tree(
        &self,
        d_columns: &CudaSlice<u32>,
        n_columns: usize,
        d_prev_leaf: Option<&CudaSlice<u64>>,
        n_leaf_hashes: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<Vec<Vec<u64>>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA poseidon252_full_tree",
            n_leaf_hashes = n_leaf_hashes
        )
        .entered();

        if n_leaf_hashes == 0 {
            return Ok(vec![]);
        }

        let block_size = 256u32;

        // Count total layers
        let n_layers = (n_leaf_hashes as f64).log2() as usize + 1;

        // Allocate all output buffers upfront (no reallocation during kernel launches)
        let mut d_layers: Vec<CudaSlice<u64>> = Vec::with_capacity(n_layers);

        // Layer 0: leaf layer
        let mut current_n = n_leaf_hashes;
        for _ in 0..n_layers {
            let d_buf = unsafe { self.device.alloc::<u64>(current_n * 4) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
            d_layers.push(d_buf);
            current_n = (current_n + 1) / 2; // ceiling division for odd sizes
        }

        // Launch leaf layer kernel
        {
            let n = n_leaf_hashes;
            let grid_size = ((n as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let has_prev: u32 = if d_prev_leaf.is_some() { 1 } else { 0 };
            let col_stride = n as u32;

            unsafe {
                match d_prev_leaf {
                    Some(prev) => {
                        self.kernels
                            .poseidon252_merkle_layer
                            .clone()
                            .launch(
                                cfg,
                                (
                                    &mut d_layers[0],
                                    d_columns,
                                    prev,
                                    d_round_constants,
                                    n_columns as u32,
                                    n as u32,
                                    has_prev,
                                    col_stride,
                                ),
                            )
                            .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                    }
                    None => {
                        let dummy_prev = self
                            .device
                            .alloc::<u64>(1)
                            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                        self.kernels
                            .poseidon252_merkle_layer
                            .clone()
                            .launch(
                                cfg,
                                (
                                    &mut d_layers[0],
                                    d_columns,
                                    &dummy_prev,
                                    d_round_constants,
                                    n_columns as u32,
                                    n as u32,
                                    has_prev,
                                    col_stride,
                                ),
                            )
                            .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                    }
                }
            }
        }

        // Launch all subsequent layers (no columns, just hash pairs)
        let dummy_cols = unsafe { self.device.alloc::<u32>(1) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        current_n = n_leaf_hashes;
        for layer_idx in 1..n_layers {
            let next_n = current_n / 2;
            if next_n == 0 {
                break;
            }

            let grid_size = ((next_n as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            // SAFETY: we read from d_layers[layer_idx-1] and write to d_layers[layer_idx].
            // These are distinct allocations. We need to split the borrow.
            let (prev_slice, rest) = d_layers.split_at_mut(layer_idx);
            let d_prev = &prev_slice[layer_idx - 1];
            let d_out = &mut rest[0];

            unsafe {
                self.kernels
                    .poseidon252_merkle_layer
                    .clone()
                    .launch(
                        cfg,
                        (
                            d_out,
                            &dummy_cols,
                            d_prev,
                            d_round_constants,
                            0u32,
                            next_n as u32,
                            1u32,
                            0u32,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }

            current_n = next_n;
        }

        // ONE sync for all layers
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync: {:?}", e)))?;

        // Bulk D2H: download all layers
        let mut results = Vec::with_capacity(n_layers);
        current_n = n_leaf_hashes;
        for layer_idx in 0..n_layers {
            let layer_size = current_n * 4;
            let mut cpu_data = vec![0u64; layer_size];
            self.device
                .dtoh_sync_copy_into(&d_layers[layer_idx].slice(0..layer_size), &mut cpu_data)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            results.push(cpu_data);
            current_n = current_n / 2;
            if current_n == 0 {
                break;
            }
        }

        tracing::info!(
            "GPU poseidon252_full_tree: {} layers, {} leaf hashes (1 sync, 1 bulk D2H)",
            results.len(),
            n_leaf_hashes
        );
        Ok(results)
    }

    /// Build an entire Poseidon252 Merkle tree on GPU and keep internal layers
    /// resident on device memory.
    ///
    /// Unlike `execute_poseidon252_merkle_full_tree`, this method does NOT bulk
    /// download layers. It returns a handle that can fetch only requested nodes
    /// (for Merkle authentication paths), avoiding large D2H transfers.
    pub fn execute_poseidon252_merkle_full_tree_gpu_layers(
        &self,
        d_columns: &CudaSlice<u32>,
        n_columns: usize,
        d_prev_leaf: Option<&CudaSlice<u64>>,
        n_leaf_hashes: usize,
        d_round_constants: &CudaSlice<u64>,
    ) -> Result<Poseidon252MerkleGpuTree, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA poseidon252_full_tree_gpu_layers",
            n_leaf_hashes = n_leaf_hashes
        )
        .entered();

        if n_leaf_hashes == 0 {
            return Err(CudaFftError::InvalidSize(
                "n_leaf_hashes must be > 0".into(),
            ));
        }

        let block_size = 256u32;
        let n_layers = (n_leaf_hashes as f64).log2() as usize + 1;

        let mut d_layers: Vec<CudaSlice<u64>> = Vec::with_capacity(n_layers);
        let mut layer_hash_counts: Vec<usize> = Vec::with_capacity(n_layers);

        let mut current_n = n_leaf_hashes;
        for _ in 0..n_layers {
            let d_buf = unsafe { self.device.alloc::<u64>(current_n * 4) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
            d_layers.push(d_buf);
            layer_hash_counts.push(current_n);
            current_n = (current_n + 1) / 2;
        }

        {
            let n = n_leaf_hashes;
            let grid_size = ((n as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let has_prev: u32 = if d_prev_leaf.is_some() { 1 } else { 0 };
            let col_stride = n as u32;

            unsafe {
                match d_prev_leaf {
                    Some(prev) => {
                        self.kernels
                            .poseidon252_merkle_layer
                            .clone()
                            .launch(
                                cfg,
                                (
                                    &mut d_layers[0],
                                    d_columns,
                                    prev,
                                    d_round_constants,
                                    n_columns as u32,
                                    n as u32,
                                    has_prev,
                                    col_stride,
                                ),
                            )
                            .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                    }
                    None => {
                        let dummy_prev = self
                            .device
                            .alloc::<u64>(1)
                            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                        self.kernels
                            .poseidon252_merkle_layer
                            .clone()
                            .launch(
                                cfg,
                                (
                                    &mut d_layers[0],
                                    d_columns,
                                    &dummy_prev,
                                    d_round_constants,
                                    n_columns as u32,
                                    n as u32,
                                    has_prev,
                                    col_stride,
                                ),
                            )
                            .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                    }
                }
            }
        }

        let dummy_cols = unsafe { self.device.alloc::<u32>(1) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        current_n = n_leaf_hashes;
        for layer_idx in 1..n_layers {
            let next_n = current_n / 2;
            if next_n == 0 {
                break;
            }

            let grid_size = ((next_n as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let (prev_slice, rest) = d_layers.split_at_mut(layer_idx);
            let d_prev = &prev_slice[layer_idx - 1];
            let d_out = &mut rest[0];

            unsafe {
                self.kernels
                    .poseidon252_merkle_layer
                    .clone()
                    .launch(
                        cfg,
                        (
                            d_out,
                            &dummy_cols,
                            d_prev,
                            d_round_constants,
                            0u32,
                            next_n as u32,
                            1u32,
                            0u32,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }

            current_n = next_n;
        }

        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync: {:?}", e)))?;

        tracing::info!(
            "GPU poseidon252_full_tree_gpu_layers: {} layers, {} leaf hashes (GPU-resident)",
            d_layers.len(),
            n_leaf_hashes
        );

        Ok(Poseidon252MerkleGpuTree {
            device: Arc::clone(&self.device),
            layers: d_layers,
            layer_hash_counts,
        })
    }

    /// Build Poseidon252 Merkle tree level-by-level on GPU, keeping only 2 levels
    /// in VRAM at a time. Extracts only the sibling nodes needed for query
    /// authentication paths — avoids allocating the full tree.
    ///
    /// Peak VRAM: ~2× one tree level (vs full tree which stores all levels).
    /// For a 128M-leaf tree: ~12.8 GB peak vs ~20.5 GB for full tree.
    ///
    /// Returns `(root_limbs, per_query_auth_path_siblings)` where each auth path
    /// is a vector of sibling `[u64; 4]` limbs from leaf level to root.
    pub fn execute_poseidon252_merkle_streaming_auth_paths(
        &self,
        d_leaf_limbs: &CudaSlice<u64>,
        n_leaf_hashes: usize,
        d_round_constants: &CudaSlice<u64>,
        query_leaf_indices: &[usize],
    ) -> Result<([u64; 4], Vec<Vec<[u64; 4]>>), CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA poseidon252_streaming_auth_paths",
            n_leaf_hashes = n_leaf_hashes,
            n_queries = query_leaf_indices.len(),
        )
        .entered();

        if n_leaf_hashes == 0 {
            return Err(CudaFftError::InvalidSize(
                "n_leaf_hashes must be > 0".into(),
            ));
        }

        let n_queries = query_leaf_indices.len();
        let block_size = 256u32;
        let n_levels = (n_leaf_hashes as f64).log2().ceil() as usize + 1;

        // Per-query auth path storage: siblings[query][level]
        let mut auth_paths: Vec<Vec<[u64; 4]>> = vec![Vec::with_capacity(n_levels); n_queries];

        // Track node indices per query at the current level
        let mut query_node_indices: Vec<usize> = query_leaf_indices
            .iter()
            .map(|&leaf_idx| leaf_idx / 2)
            .collect();

        // Dummy column buffer for internal levels (no column data, only prev layer)
        let dummy_cols = unsafe { self.device.alloc::<u32>(1) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Level 0: hash leaf pairs from d_leaf_limbs
        let mut d_current = {
            let mut d_out = unsafe { self.device.alloc::<u64>(n_leaf_hashes * 4) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

            let grid_size = ((n_leaf_hashes as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.kernels
                    .poseidon252_merkle_layer
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_out,
                            &dummy_cols,
                            d_leaf_limbs,
                            d_round_constants,
                            0u32,             // n_columns (0 = no column data)
                            n_leaf_hashes as u32,
                            1u32,             // has_prev = 1 (leaf limbs are the "prev" layer)
                            0u32,             // col_stride
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
            self.device
                .synchronize()
                .map_err(|e| CudaFftError::KernelExecution(format!("Sync: {:?}", e)))?;

            d_out
        };
        let mut current_n = n_leaf_hashes;

        // Extract level-0 sibling hashes for each query
        for q in 0..n_queries {
            let sib_idx = query_node_indices[q] ^ 1;
            if sib_idx < current_n {
                let start = sib_idx * 4;
                let end = start + 4;
                let mut out = [0u64; 4];
                self.device
                    .dtoh_sync_copy_into(&d_current.slice(start..end), &mut out)
                    .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
                auth_paths[q].push(out);
            } else {
                // Edge case: sibling out of bounds (odd tree), push zeroes
                auth_paths[q].push([0u64; 4]);
            }
        }

        // Upper levels: build from previous level, extract siblings, free old level
        for _level in 1..n_levels {
            let next_n = (current_n + 1) / 2;
            if next_n == 0 {
                break;
            }

            // Advance query node indices to parent level
            for idx in query_node_indices.iter_mut() {
                *idx >>= 1;
            }

            let mut d_next = unsafe { self.device.alloc::<u64>(next_n * 4) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

            let grid_size = ((next_n as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.kernels
                    .poseidon252_merkle_layer
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_next,
                            &dummy_cols,
                            &d_current,
                            d_round_constants,
                            0u32,
                            next_n as u32,
                            1u32,
                            0u32,
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
            self.device
                .synchronize()
                .map_err(|e| CudaFftError::KernelExecution(format!("Sync: {:?}", e)))?;

            // Free the previous level before extracting (saves VRAM)
            drop(d_current);

            // Extract sibling hashes at this level
            if next_n > 1 {
                for q in 0..n_queries {
                    let sib_idx = query_node_indices[q] ^ 1;
                    if sib_idx < next_n {
                        let start = sib_idx * 4;
                        let end = start + 4;
                        let mut out = [0u64; 4];
                        self.device
                            .dtoh_sync_copy_into(&d_next.slice(start..end), &mut out)
                            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
                        auth_paths[q].push(out);
                    } else {
                        auth_paths[q].push([0u64; 4]);
                    }
                }
            }

            d_current = d_next;
            current_n = next_n;
        }

        // Download root
        let mut root = [0u64; 4];
        self.device
            .dtoh_sync_copy_into(&d_current.slice(0..4), &mut root)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU poseidon252_streaming_auth_paths: {} leaf hashes, {} queries, {} levels",
            n_leaf_hashes,
            n_queries,
            n_levels,
        );

        Ok((root, auth_paths))
    }

    // =========================================================================
    // GPU-Resident Column Operations (eliminate PCIe round-trips)
    // =========================================================================

    /// IFFT that takes CPU data and returns the GPU-resident result.
    ///
    /// Uploads data once, runs IFFT kernels, and returns the `CudaSlice`
    /// without downloading back to CPU. The caller is responsible for
    /// caching the returned slice via `cache_column_gpu`.
    pub fn execute_ifft_to_gpu(
        &self,
        data: &[u32],
        twiddles_dbl: &[Vec<u32>],
        log_size: u32,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n = 1usize << log_size;
        if data.len() != n {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} elements, got {}",
                n,
                data.len()
            )));
        }

        let _span =
            tracing::span!(tracing::Level::INFO, "CUDA IFFT→GPU", log_size = log_size).entered();

        // H2D: data + twiddles
        let mut d_data = self
            .device
            .htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let flat_twiddles: Vec<u32> = twiddles_dbl.iter().flatten().copied().collect();
        let d_twiddles = self
            .device
            .htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Run IFFT layers (all on GPU)
        self.execute_ifft_layers(&mut d_data, &d_twiddles, log_size, twiddles_dbl)?;

        // NO dtoh — return GPU-resident result
        tracing::debug!("IFFT→GPU: {} elements kept on device", n);
        Ok(d_data)
    }

    /// FFT that operates on a GPU-resident input in-place.
    ///
    /// Uploads only twiddle factors (small), runs FFT layers on the
    /// already-resident data, and leaves the result on GPU.
    pub fn execute_fft_on_gpu(
        &self,
        d_data: &mut CudaSlice<u32>,
        twiddles: &[Vec<u32>],
        log_size: u32,
    ) -> Result<(), CudaFftError> {
        let _span =
            tracing::span!(tracing::Level::INFO, "CUDA FFT on GPU", log_size = log_size).entered();

        // H2D: twiddles only
        let flat_twiddles: Vec<u32> = twiddles.iter().flatten().copied().collect();
        let d_twiddles = self
            .device
            .htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Run FFT layers in-place
        self.execute_fft_layers(d_data, &d_twiddles, log_size, twiddles)?;

        // NO dtoh — result stays on GPU
        tracing::debug!(
            "FFT on GPU: {} elements, no PCIe round-trip",
            1u64 << log_size
        );
        Ok(())
    }

    /// Batch IFFT: upload twiddles once, process multiple columns, return GPU-resident results.
    ///
    /// Returns (cpu_results, gpu_slices) — CPU data for polynomial representation,
    /// GPU slices for caching (evaluate can reuse them).
    pub fn execute_batch_ifft_to_gpu(
        &self,
        columns: &[Vec<u32>],
        twiddles_dbl: &[Vec<u32>],
        log_size: u32,
        denorm_val: u32,
    ) -> Result<(Vec<Vec<u32>>, Vec<CudaSlice<u32>>), CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA batch_IFFT→GPU",
            num_cols = columns.len(),
            log_size = log_size
        )
        .entered();

        // Upload twiddles ONCE (shared across all columns)
        let flat_twiddles: Vec<u32> = twiddles_dbl.iter().flatten().copied().collect();
        let d_twiddles = self
            .device
            .htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut cpu_results = Vec::with_capacity(columns.len());
        let mut gpu_slices = Vec::with_capacity(columns.len());

        for col_data in columns {
            // Upload column
            let mut d_data = self
                .device
                .htod_sync_copy(col_data)
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

            // IFFT in-place
            self.execute_ifft_layers(&mut d_data, &d_twiddles, log_size, twiddles_dbl)?;

            // Denormalize on GPU
            self.execute_denormalize_on_device(&mut d_data, denorm_val, 1u32 << log_size)?;
            self.device
                .synchronize()
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;

            // Download to CPU
            let cpu_data = self
                .device
                .dtoh_sync_copy(&d_data)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

            cpu_results.push(cpu_data);
            gpu_slices.push(d_data);
        }

        tracing::info!(
            "Batch IFFT: {} columns × {} elements, twiddles uploaded once",
            columns.len(),
            1u64 << log_size
        );
        Ok((cpu_results, gpu_slices))
    }

    /// Batch FFT: upload twiddles once, process multiple GPU-resident columns.
    ///
    /// Columns must already be on GPU. Returns CPU results + keeps GPU slices for Merkle.
    pub fn execute_batch_fft_on_gpu(
        &self,
        d_columns: &mut [CudaSlice<u32>],
        twiddles: &[Vec<u32>],
        log_size: u32,
    ) -> Result<Vec<Vec<u32>>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA batch_FFT on GPU",
            num_cols = d_columns.len(),
            log_size = log_size
        )
        .entered();

        // Upload twiddles ONCE
        let flat_twiddles: Vec<u32> = twiddles.iter().flatten().copied().collect();
        let d_twiddles = self
            .device
            .htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut cpu_results = Vec::with_capacity(d_columns.len());

        for d_data in d_columns.iter_mut() {
            // FFT in-place
            self.execute_fft_layers(d_data, &d_twiddles, log_size, twiddles)?;

            // Download to CPU
            let cpu_data = self
                .device
                .dtoh_sync_copy(d_data)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            cpu_results.push(cpu_data);
        }

        tracing::info!(
            "Batch FFT: {} columns × {} elements, twiddles uploaded once",
            cpu_results.len(),
            1u64 << log_size
        );
        Ok(cpu_results)
    }

    /// Merkle hashing that takes GPU-resident column data.
    ///
    /// Columns are already on GPU so no H2D is needed for column data.
    /// Only the previous layer hashes (small) are uploaded if present.
    /// Returns hashes to CPU (32 bytes × n_hashes — tiny).
    pub fn execute_blake2s_merkle_from_gpu(
        &self,
        d_columns: &[&CudaSlice<u32>],
        col_lengths: &[usize],
        prev_layer: Option<&[u8]>,
        n_hashes: usize,
    ) -> Result<Vec<u8>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA merkle_from_gpu",
            n_hashes = n_hashes
        )
        .entered();

        let n_columns = d_columns.len();

        // Flatten GPU columns into a single contiguous buffer
        // We need to gather them since the kernel expects contiguous column data
        let total_elements: usize = col_lengths.iter().sum();
        let mut d_flat_columns = unsafe { self.device.alloc::<u32>(total_elements) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Copy each GPU column into the flat buffer using device-to-device copy
        let mut offset = 0usize;
        for (i, d_col) in d_columns.iter().enumerate() {
            let len = col_lengths[i];
            let block_size = 256u32;
            let grid_size = ((len as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels
                    .copy_column
                    .clone()
                    .launch(
                        cfg,
                        (&mut d_flat_columns, *d_col, offset as u32, len as u32),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("copy_column: {:?}", e)))?;
            }
            offset += len;
        }

        // Upload previous layer (small) if present
        let d_prev_layer = if let Some(prev) = prev_layer {
            Some(
                self.device
                    .htod_sync_copy(prev)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?,
            )
        } else {
            None
        };

        // Allocate output
        let mut d_output = unsafe { self.device.alloc::<u8>(n_hashes * 32) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_hashes as u32) + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let has_prev_layer = if prev_layer.is_some() { 1u32 } else { 0u32 };

        unsafe {
            match &d_prev_layer {
                Some(prev) => {
                    self.kernels
                        .merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                &d_flat_columns,
                                prev,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev_layer,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                None => {
                    let dummy_prev = self
                        .device
                        .alloc::<u8>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels
                        .merkle_layer
                        .clone()
                        .launch(
                            cfg,
                            (
                                &mut d_output,
                                &d_flat_columns,
                                &dummy_prev,
                                n_columns as u32,
                                n_hashes as u32,
                                has_prev_layer,
                            ),
                        )
                        .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        }

        // Sync + D2H (tiny: 32 bytes × n_hashes)
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u8; n_hashes * 32];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!("GPU merkle_from_gpu: {} hashes (0 column H2D)", n_hashes);
        Ok(output)
    }

    /// Build an entire Blake2s Merkle tree in one GPU pass (no per-layer sync/D2H).
    ///
    /// Mirrors `execute_poseidon252_merkle_full_tree` but for Blake2s:
    /// 1. Leaf layer: hash columns using existing Blake2s kernel
    /// 2. All subsequent layers: hash pairs from previous layer (no columns)
    /// 3. All kernels on same CUDA stream (implicit ordering, no sync between layers)
    /// 4. ONE sync + bulk D2H at end
    /// 5. Returns `Vec<Vec<u8>>` (each layer is n_hashes × 32 bytes)
    pub fn execute_blake2s_merkle_full_tree(
        &self,
        d_columns: &[&CudaSlice<u32>],
        col_lengths: &[usize],
        n_leaf_hashes: usize,
    ) -> Result<Vec<Vec<u8>>, CudaFftError> {
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA blake2s_full_tree",
            n_leaf_hashes = n_leaf_hashes
        )
        .entered();

        if n_leaf_hashes == 0 {
            return Ok(vec![]);
        }

        let n_columns = d_columns.len();
        let block_size = 256u32;

        // Count total layers
        let n_layers = (n_leaf_hashes as f64).log2() as usize + 1;

        // Flatten GPU columns into a single contiguous buffer
        let total_elements: usize = col_lengths.iter().sum();
        let mut d_flat_columns = unsafe { self.device.alloc::<u32>(total_elements) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut offset = 0usize;
        for (i, d_col) in d_columns.iter().enumerate() {
            let len = col_lengths[i];
            let grid_size = ((len as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.kernels
                    .copy_column
                    .clone()
                    .launch(
                        cfg,
                        (&mut d_flat_columns, *d_col, offset as u32, len as u32),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("copy_column: {:?}", e)))?;
            }
            offset += len;
        }

        // Allocate all output buffers upfront
        let mut d_layers: Vec<CudaSlice<u8>> = Vec::with_capacity(n_layers);
        let mut current_n = n_leaf_hashes;
        for _ in 0..n_layers {
            let d_buf = unsafe { self.device.alloc::<u8>(current_n * 32) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
            d_layers.push(d_buf);
            current_n = (current_n + 1) / 2;
        }

        // Launch leaf layer kernel (columns present, no prev_layer)
        {
            let n = n_leaf_hashes;
            let grid_size = ((n as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let dummy_prev = unsafe { self.device.alloc::<u8>(1) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
            unsafe {
                self.kernels
                    .merkle_layer
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut d_layers[0],
                            &d_flat_columns,
                            &dummy_prev,
                            n_columns as u32,
                            n as u32,
                            0u32, // has_prev_layer = false
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
        }

        // Launch all subsequent layers (no columns, hash pairs from previous layer)
        let dummy_cols = unsafe { self.device.alloc::<u32>(1) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        current_n = n_leaf_hashes;
        for layer_idx in 1..n_layers {
            let next_n = current_n / 2;
            if next_n == 0 {
                break;
            }

            let grid_size = ((next_n as u32) + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            // Split borrow: read from d_layers[layer_idx-1], write to d_layers[layer_idx]
            let (prev_slice, rest) = d_layers.split_at_mut(layer_idx);
            let d_prev = &prev_slice[layer_idx - 1];
            let d_out = &mut rest[0];

            unsafe {
                self.kernels
                    .merkle_layer
                    .clone()
                    .launch(
                        cfg,
                        (
                            d_out,
                            &dummy_cols,
                            d_prev,
                            0u32, // n_columns = 0
                            next_n as u32,
                            1u32, // has_prev_layer = true
                        ),
                    )
                    .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }

            current_n = next_n;
        }

        // ONE sync for all layers
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync: {:?}", e)))?;

        // Bulk D2H: download all layers
        let mut results = Vec::with_capacity(n_layers);
        current_n = n_leaf_hashes;
        for layer_idx in 0..n_layers {
            let layer_size = current_n * 32;
            let mut cpu_data = vec![0u8; layer_size];
            self.device
                .dtoh_sync_copy_into(&d_layers[layer_idx].slice(0..layer_size), &mut cpu_data)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            results.push(cpu_data);
            current_n = current_n / 2;
            if current_n == 0 {
                break;
            }
        }

        tracing::info!(
            "GPU blake2s_full_tree: {} layers, {} leaf hashes (1 sync, 1 bulk D2H)",
            results.len(),
            n_leaf_hashes
        );
        Ok(results)
    }

    // =========================================================================
    // MLE (GKR) Operations
    // =========================================================================

    /// Execute MLE fold operation: BaseField -> SecureField
    ///
    /// Computes: result[i] = lhs[i] * (1 - assignment) + rhs[i] * assignment
    ///
    /// # Arguments
    /// * `lhs` - Left half values (M31, single u32 per element)
    /// * `rhs` - Right half values (M31, single u32 per element)
    /// * `assignment` - QM31 assignment value (4 u32)
    /// * `n` - Number of output elements
    pub fn mle_fold_base_to_secure(
        &self,
        lhs: &[u32],
        rhs: &[u32],
        assignment: &[u32; 4],
        n: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        let _span =
            tracing::span!(tracing::Level::INFO, "CUDA mle_fold_base_to_secure", n = n).entered();

        if lhs.len() != n || rhs.len() != n {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} elements for lhs and rhs, got {} and {}",
                n,
                lhs.len(),
                rhs.len()
            )));
        }

        // Allocate device memory
        let d_lhs = self
            .device
            .htod_sync_copy(lhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_rhs = self
            .device
            .htod_sync_copy(rhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(n * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .mle_fold_base_to_secure
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        &d_lhs,
                        &d_rhs,
                        assignment[0],
                        assignment[1],
                        assignment[2],
                        assignment[3],
                        n as u32,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; n * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!("GPU mle_fold_base_to_secure completed: {} elements", n);

        Ok(output)
    }

    /// Execute MLE fold operation: SecureField -> SecureField
    ///
    /// Computes: result[i] = lhs[i] * (1 - assignment) + rhs[i] * assignment
    ///
    /// # Arguments
    /// * `lhs` - Left half values (QM31, 4 u32 per element)
    /// * `rhs` - Right half values (QM31, 4 u32 per element)
    /// * `assignment` - QM31 assignment value (4 u32)
    /// * `n` - Number of output elements
    pub fn mle_fold_secure(
        &self,
        lhs: &[u32],
        rhs: &[u32],
        assignment: &[u32; 4],
        n: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        let _span = tracing::span!(tracing::Level::INFO, "CUDA mle_fold_secure", n = n).entered();

        if lhs.len() != n * 4 || rhs.len() != n * 4 {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} u32 values for lhs and rhs, got {} and {}",
                n * 4,
                lhs.len(),
                rhs.len()
            )));
        }

        // Allocate device memory
        let d_lhs = self
            .device
            .htod_sync_copy(lhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_rhs = self
            .device
            .htod_sync_copy(rhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(n * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .mle_fold_secure
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        &d_lhs,
                        &d_rhs,
                        assignment[0],
                        assignment[1],
                        assignment[2],
                        assignment[3],
                        n as u32,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; n * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!("GPU mle_fold_secure completed: {} elements", n);

        Ok(output)
    }

    /// Execute gen_eq_evals for GKR.
    ///
    /// Computes: eq_evals[i] = v * prod_j((1 - y[j]) + y[j] * bit_j(i))
    ///
    /// # Arguments
    /// * `y` - QM31 y values (4 u32 per element, n_variables elements)
    /// * `v` - Initial QM31 value (4 u32)
    /// * `n_variables` - Number of variables
    pub fn gen_eq_evals(
        &self,
        y: &[u32],
        v: &[u32; 4],
        n_variables: usize,
    ) -> Result<Vec<u32>, CudaFftError> {
        let output_size = 1usize << n_variables;
        let _span = tracing::span!(
            tracing::Level::INFO,
            "CUDA gen_eq_evals",
            n_variables = n_variables,
            output_size = output_size
        )
        .entered();

        if y.len() != n_variables * 4 {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} u32 values for y, got {}",
                n_variables * 4,
                y.len()
            )));
        }

        // Allocate device memory
        let d_y = self
            .device
            .htod_sync_copy(y)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe { self.device.alloc::<u32>(output_size * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((output_size as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .gen_eq_evals
                .clone()
                .launch(
                    cfg,
                    (
                        &mut d_output,
                        &d_y,
                        v[0],
                        v[1],
                        v[2],
                        v[3],
                        n_variables as u32,
                        output_size as u32,
                    ),
                )
                .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; output_size * 4];
        self.device
            .dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!(
            "GPU gen_eq_evals completed: {} output elements",
            output_size
        );

        Ok(output)
    }
}

// =============================================================================
// High-Level FFT Interface
// =============================================================================

/// Execute IFFT using CUDA if available, otherwise return error.
#[cfg(feature = "cuda-runtime")]
pub fn cuda_ifft(
    data: &mut [u32],
    twiddles_dbl: &[Vec<u32>],
    log_size: u32,
) -> Result<(), CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_ifft(data, twiddles_dbl, log_size)
}

/// Execute FFT using CUDA if available, otherwise return error.
#[cfg(feature = "cuda-runtime")]
pub fn cuda_fft(
    data: &mut [u32],
    twiddles: &[Vec<u32>],
    log_size: u32,
) -> Result<(), CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_fft(data, twiddles, log_size)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_ifft(
    _data: &mut [u32],
    _twiddles_dbl: &[Vec<u32>],
    _log_size: u32,
) -> Result<(), CudaFftError> {
    Err(CudaFftError::NoDevice)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_fft(
    _data: &mut [u32],
    _twiddles: &[Vec<u32>],
    _log_size: u32,
) -> Result<(), CudaFftError> {
    Err(CudaFftError::NoDevice)
}

// =============================================================================
// High-Level FRI Folding Interface
// =============================================================================

/// Execute FRI fold_line using CUDA.
///
/// # Arguments
/// * `input` - Input SecureField values as flat u32 array (4 u32 per element)
/// * `itwiddles` - Inverse twiddle factors
/// * `alpha` - Folding random challenge (4 u32 for QM31)
/// * `n` - Number of input elements
///
/// # Returns
/// Output SecureField values (n/2 elements, 4 u32 each)
#[cfg(feature = "cuda-runtime")]
pub fn cuda_fold_line(
    input: &[u32],
    itwiddles: &[u32],
    alpha: &[u32; 4],
    n: usize,
) -> Result<Vec<u32>, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_fold_line(input, itwiddles, alpha, n)
}

/// Execute FRI fold_circle_into_line using CUDA.
///
/// # Arguments
/// * `dst` - Destination line evaluation (modified in place)
/// * `src` - Source circle evaluation
/// * `itwiddles` - Inverse twiddle factors (y-coordinates)
/// * `alpha` - Folding random challenge (4 u32 for QM31)
/// * `n` - Number of source elements
#[cfg(feature = "cuda-runtime")]
pub fn cuda_fold_circle_into_line(
    dst: &mut [u32],
    src: &[u32],
    itwiddles: &[u32],
    alpha: &[u32; 4],
    n: usize,
) -> Result<(), CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_fold_circle_into_line(dst, src, itwiddles, alpha, n)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_fold_line(
    _input: &[u32],
    _itwiddles: &[u32],
    _alpha: &[u32; 4],
    _n: usize,
) -> Result<Vec<u32>, CudaFftError> {
    Err(CudaFftError::NoDevice)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_fold_circle_into_line(
    _dst: &mut [u32],
    _src: &[u32],
    _itwiddles: &[u32],
    _alpha: &[u32; 4],
    _n: usize,
) -> Result<(), CudaFftError> {
    Err(CudaFftError::NoDevice)
}

// =============================================================================
// High-Level Quotient Accumulation Interface
// =============================================================================

/// Execute quotient accumulation using CUDA.
///
/// # Arguments
/// * `columns` - Column data (each column is a Vec<u32> of M31 values)
/// * `line_coeffs` - Line coefficients (a, b, c as QM31, 12 u32 each)
/// * `denom_inv` - Denominator inverses (CM31, 2 u32 each)
/// * `batch_sizes` - Number of columns per sample batch
/// * `col_indices` - Column indices for each coefficient
/// * `n_points` - Number of domain points
///
/// # Returns
/// Output QM31 values (4 u32 per element)
#[cfg(feature = "cuda-runtime")]
pub fn cuda_accumulate_quotients(
    columns: &[Vec<u32>],
    line_coeffs: &[[u32; 12]],
    denom_inv: &[u32],
    batch_sizes: &[usize],
    col_indices: &[usize],
    n_points: usize,
) -> Result<Vec<u32>, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_accumulate_quotients(
        columns,
        line_coeffs,
        denom_inv,
        batch_sizes,
        col_indices,
        n_points,
    )
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_accumulate_quotients(
    _columns: &[Vec<u32>],
    _line_coeffs: &[[u32; 12]],
    _denom_inv: &[u32],
    _batch_sizes: &[usize],
    _col_indices: &[usize],
    _n_points: usize,
) -> Result<Vec<u32>, CudaFftError> {
    Err(CudaFftError::NoDevice)
}

// =============================================================================
// High-Level Merkle Hashing Interface
// =============================================================================

/// Execute Blake2s Merkle hashing using CUDA.
///
/// # Arguments
/// * `columns` - Column data (each column is a Vec<u32> of M31 values)
/// * `prev_layer` - Previous layer hashes (64 bytes per pair, or None for leaves)
/// * `n_hashes` - Number of hashes to compute
///
/// # Returns
/// Output hashes (32 bytes each)
#[cfg(feature = "cuda-runtime")]
pub fn cuda_blake2s_merkle(
    columns: &[Vec<u32>],
    prev_layer: Option<&[u8]>,
    n_hashes: usize,
) -> Result<Vec<u8>, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    executor.execute_blake2s_merkle(columns, prev_layer, n_hashes)
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn cuda_blake2s_merkle(
    _columns: &[Vec<u32>],
    _prev_layer: Option<&[u8]>,
    _n_hashes: usize,
) -> Result<Vec<u8>, CudaFftError> {
    Err(CudaFftError::NoDevice)
}

// =============================================================================
// Poseidon252 Round Constants
// =============================================================================

/// Compute the 107 compressed Poseidon252 round constants as u64 limbs.
///
/// This replicates the round key compression from starknet-crypto-codegen.
/// The result is 107 × 4 = 428 u64 values (big-endian limb order matching
/// starknet_ff::FieldElement::to_bytes_be layout).
///
/// Layout: [rc0_limb3, rc0_limb2, rc0_limb1, rc0_limb0, rc1_limb3, ...]
/// where limb3 is the most significant (matching how felt252 is stored in the CUDA kernel).
#[cfg(feature = "cuda-runtime")]
pub fn compute_poseidon252_round_constants() -> Vec<u64> {
    use starknet_ff::FieldElement;

    const FULL_ROUNDS: usize = 8;
    const PARTIAL_ROUNDS: usize = 83;

    // Raw round keys (91 rounds × 3 FieldElements)
    let raw_keys = poseidon252_raw_round_keys();

    // Compress round keys (same algorithm as starknet-crypto-codegen)
    let mut comp = Vec::with_capacity(107);

    // First half full rounds (4 rounds × 3 constants = 12)
    for round in &raw_keys[..FULL_ROUNDS / 2] {
        comp.extend_from_slice(round);
    }

    // Compressed partial rounds + first of last full rounds
    {
        let mut state = [FieldElement::ZERO; 3];
        let mut idx = FULL_ROUNDS / 2;

        for _ in 0..PARTIAL_ROUNDS {
            state[0] += raw_keys[idx][0];
            state[1] += raw_keys[idx][1];
            state[2] += raw_keys[idx][2];

            comp.push(state[2]);
            state[2] = FieldElement::ZERO;

            // MixLayer
            let t = state[0] + state[1] + state[2];
            state[0] = t + state[0].double();
            state[1] = t - state[1].double();
            state[2] = t - FieldElement::THREE * state[2];

            idx += 1;
        }

        // First of the last full rounds
        state[0] += raw_keys[idx][0];
        state[1] += raw_keys[idx][1];
        state[2] += raw_keys[idx][2];
        comp.push(state[0]);
        comp.push(state[1]);
        comp.push(state[2]);
    }

    // Last full rounds except the first (3 rounds × 3 = 9)
    for round in &raw_keys[(FULL_ROUNDS / 2 + PARTIAL_ROUNDS + 1)..] {
        comp.extend_from_slice(round);
    }

    assert_eq!(comp.len(), 107, "Expected 107 compressed round constants");

    // Convert to u64 limbs (4 per felt252)
    // starknet_ff stores internally as Montgomery form, but to_bytes_be gives canonical big-endian
    let mut limbs = Vec::with_capacity(107 * 4);
    for fe in &comp {
        let bytes = fe.to_bytes_be();
        // Convert 32 bytes to 4 u64s (big-endian: limb[3] is most significant)
        // CUDA kernel uses: limb[0] = least significant, limb[3] = most significant
        for i in 0..4 {
            let offset = 24 - i * 8; // byte offsets: 24, 16, 8, 0
            let mut val = 0u64;
            for j in 0..8 {
                val = (val << 8) | bytes[offset + j] as u64;
            }
            limbs.push(val);
        }
    }

    limbs
}

/// Upload Poseidon252 round constants to GPU device memory.
/// Returns a CudaSlice that should be cached and reused across kernel calls.
#[cfg(feature = "cuda-runtime")]
pub fn upload_poseidon252_round_constants(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<CudaSlice<u64>, CudaFftError> {
    let limbs = compute_poseidon252_round_constants();
    device
        .htod_sync_copy(&limbs)
        .map_err(|e| CudaFftError::MemoryAllocation(format!("RC upload: {:?}", e)))
}

/// The 91 raw Poseidon252 round keys (decimal strings → FieldElement).
#[cfg(feature = "cuda-runtime")]
fn poseidon252_raw_round_keys() -> Vec<[starknet_ff::FieldElement; 3]> {
    use starknet_ff::FieldElement;

    // Extracted from starknet-crypto-codegen params.rs (poseidon3.txt)
    let raw: [[&str; 3]; 91] = [
        [
            "2950795762459345168613727575620414179244544320470208355568817838579231751791",
            "1587446564224215276866294500450702039420286416111469274423465069420553242820",
            "1645965921169490687904413452218868659025437693527479459426157555728339600137",
        ],
        [
            "2782373324549879794752287702905278018819686065818504085638398966973694145741",
            "3409172630025222641379726933524480516420204828329395644967085131392375707302",
            "2379053116496905638239090788901387719228422033660130943198035907032739387135",
        ],
        [
            "2570819397480941104144008784293466051718826502582588529995520356691856497111",
            "3546220846133880637977653625763703334841539452343273304410918449202580719746",
            "2720682389492889709700489490056111332164748138023159726590726667539759963454",
        ],
        [
            "1899653471897224903834726250400246354200311275092866725547887381599836519005",
            "2369443697923857319844855392163763375394720104106200469525915896159690979559",
            "2354174693689535854311272135513626412848402744119855553970180659094265527996",
        ],
        [
            "2404084503073127963385083467393598147276436640877011103379112521338973185443",
            "950320777137731763811524327595514151340412860090489448295239456547370725376",
            "2121140748740143694053732746913428481442990369183417228688865837805149503386",
        ],
        [
            "2372065044800422557577242066480215868569521938346032514014152523102053709709",
            "2618497439310693947058545060953893433487994458443568169824149550389484489896",
            "3518297267402065742048564133910509847197496119850246255805075095266319996916",
        ],
        [
            "340529752683340505065238931581518232901634742162506851191464448040657139775",
            "1954876811294863748406056845662382214841467408616109501720437541211031966538",
            "813813157354633930267029888722341725864333883175521358739311868164460385261",
        ],
        [
            "71901595776070443337150458310956362034911936706490730914901986556638720031",
            "2789761472166115462625363403490399263810962093264318361008954888847594113421",
            "2628791615374802560074754031104384456692791616314774034906110098358135152410",
        ],
        [
            "3617032588734559635167557152518265808024917503198278888820567553943986939719",
            "2624012360209966117322788103333497793082705816015202046036057821340914061980",
            "149101987103211771991327927827692640556911620408176100290586418839323044234",
        ],
        [
            "1039927963829140138166373450440320262590862908847727961488297105916489431045",
            "2213946951050724449162431068646025833746639391992751674082854766704900195669",
            "2792724903541814965769131737117981991997031078369482697195201969174353468597",
        ],
        [
            "3212031629728871219804596347439383805499808476303618848198208101593976279441",
            "3343514080098703935339621028041191631325798327656683100151836206557453199613",
            "614054702436541219556958850933730254992710988573177298270089989048553060199",
        ],
        [
            "148148081026449726283933484730968827750202042869875329032965774667206931170",
            "1158283532103191908366672518396366136968613180867652172211392033571980848414",
            "1032400527342371389481069504520755916075559110755235773196747439146396688513",
        ],
        [
            "806900704622005851310078578853499250941978435851598088619290797134710613736",
            "462498083559902778091095573017508352472262817904991134671058825705968404510",
            "1003580119810278869589347418043095667699674425582646347949349245557449452503",
        ],
        [
            "619074932220101074089137133998298830285661916867732916607601635248249357793",
            "2635090520059500019661864086615522409798872905401305311748231832709078452746",
            "978252636251682252755279071140187792306115352460774007308726210405257135181",
        ],
        [
            "1766912167973123409669091967764158892111310474906691336473559256218048677083",
            "1663265127259512472182980890707014969235283233442916350121860684522654120381",
            "3532407621206959585000336211742670185380751515636605428496206887841428074250",
        ],
        [
            "2507023127157093845256722098502856938353143387711652912931112668310034975446",
            "3321152907858462102434883844787153373036767230808678981306827073335525034593",
            "3039253036806065280643845548147711477270022154459620569428286684179698125661",
        ],
        [
            "103480338868480851881924519768416587261556021758163719199282794248762465380",
            "2394049781357087698434751577708655768465803975478348134669006211289636928495",
            "2660531560345476340796109810821127229446538730404600368347902087220064379579",
        ],
        [
            "3603166934034556203649050570865466556260359798872408576857928196141785055563",
            "1553799760191949768532188139643704561532896296986025007089826672890485412324",
            "2744284717053657689091306578463476341218866418732695211367062598446038965164",
        ],
        [
            "320745764922149897598257794663594419839885234101078803811049904310835548856",
            "979382242100682161589753881721708883681034024104145498709287731138044566302",
            "1860426855810549882740147175136418997351054138609396651615467358416651354991",
        ],
        [
            "336173081054369235994909356892506146234495707857220254489443629387613956145",
            "1632470326779699229772327605759783482411227247311431865655466227711078175883",
            "921958250077481394074960433988881176409497663777043304881055317463712938502",
        ],
        [
            "3034358982193370602048539901033542101022185309652879937418114324899281842797",
            "25626282149517463867572353922222474817434101087272320606729439087234878607",
            "3002662261401575565838149305485737102400501329139562227180277188790091853682",
        ],
        [
            "2939684373453383817196521641512509179310654199629514917426341354023324109367",
            "1076484609897998179434851570277297233169621096172424141759873688902355505136",
            "2575095284833160494841112025725243274091830284746697961080467506739203605049",
        ],
        [
            "3565075264617591783581665711620369529657840830498005563542124551465195621851",
            "2197016502533303822395077038351174326125210255869204501838837289716363437993",
            "331415322883530754594261416546036195982886300052707474899691116664327869405",
        ],
        [
            "1935011233711290003793244296594669823169522055520303479680359990463281661839",
            "3495901467168087413996941216661589517270845976538454329511167073314577412322",
            "954195417117133246453562983448451025087661597543338750600301835944144520375",
        ],
        [
            "1271840477709992894995746871435810599280944810893784031132923384456797925777",
            "2565310762274337662754531859505158700827688964841878141121196528015826671847",
            "3365022288251637014588279139038152521653896670895105540140002607272936852513",
        ],
        [
            "1660592021628965529963974299647026602622092163312666588591285654477111176051",
            "970104372286014048279296575474974982288801187216974504035759997141059513421",
            "2617024574317953753849168721871770134225690844968986289121504184985993971227",
        ],
        [
            "999899815343607746071464113462778273556695659506865124478430189024755832262",
            "2228536129413411161615629030408828764980855956560026807518714080003644769896",
            "2701953891198001564547196795777701119629537795442025393867364730330476403227",
        ],
        [
            "837078355588159388741598313782044128527494922918203556465116291436461597853",
            "2121749601840466143704862369657561429793951309962582099604848281796392359214",
            "771812260179247428733132708063116523892339056677915387749121983038690154755",
        ],
        [
            "3317336423132806446086732225036532603224267214833263122557471741829060578219",
            "481570067997721834712647566896657604857788523050900222145547508314620762046",
            "242195042559343964206291740270858862066153636168162642380846129622127460192",
        ],
        [
            "2855462178889999218204481481614105202770810647859867354506557827319138379686",
            "3525521107148375040131784770413887305850308357895464453970651672160034885202",
            "1320839531502392535964065058804908871811967681250362364246430459003920305799",
        ],
        [
            "2514191518588387125173345107242226637171897291221681115249521904869763202419",
            "2798335750958827619666318316247381695117827718387653874070218127140615157902",
            "2808467767967035643407948058486565877867906577474361783201337540214875566395",
        ],
        [
            "3551834385992706206273955480294669176699286104229279436819137165202231595747",
            "1219439673853113792340300173186247996249367102884530407862469123523013083971",
            "761519904537984520554247997444508040636526566551719396202550009393012691157",
        ],
        [
            "3355402549169351700500518865338783382387571349497391475317206324155237401353",
            "199541098009731541347317515995192175813554789571447733944970283654592727138",
            "192100490643078165121235261796864975568292640203635147901612231594408079071",
        ],
        [
            "1187019357602953326192019968809486933768550466167033084944727938441427050581",
            "189525349641911362389041124808934468936759383310282010671081989585219065700",
            "2831653363992091308880573627558515686245403755586311978724025292003353336665",
        ],
        [
            "2052859812632218952608271535089179639890275494426396974475479657192657094698",
            "1670756178709659908159049531058853320846231785448204274277900022176591811072",
            "3538757242013734574731807289786598937548399719866320954894004830207085723125",
        ],
        [
            "710549042741321081781917034337800036872214466705318638023070812391485261299",
            "2345013122330545298606028187653996682275206910242635100920038943391319595180",
            "3528369671971445493932880023233332035122954362711876290904323783426765912206",
        ],
        [
            "1167120829038120978297497195837406760848728897181138760506162680655977700764",
            "3073243357129146594530765548901087443775563058893907738967898816092270628884",
            "378514724418106317738164464176041649567501099164061863402473942795977719726",
        ],
        [
            "333391138410406330127594722511180398159664250722328578952158227406762627796",
            "1727570175639917398410201375510924114487348765559913502662122372848626931905",
            "968312190621809249603425066974405725769739606059422769908547372904403793174",
        ],
        [
            "360659316299446405855194688051178331671817370423873014757323462844775818348",
            "1386580151907705298970465943238806620109618995410132218037375811184684929291",
            "3604888328937389309031638299660239238400230206645344173700074923133890528967",
        ],
        [
            "2496185632263372962152518155651824899299616724241852816983268163379540137546",
            "486538168871046887467737983064272608432052269868418721234810979756540672990",
            "1558415498960552213241704009433360128041672577274390114589014204605400783336",
        ],
        [
            "3512058327686147326577190314835092911156317204978509183234511559551181053926",
            "2235429387083113882635494090887463486491842634403047716936833563914243946191",
            "1290896777143878193192832813769470418518651727840187056683408155503813799882",
        ],
        [
            "1143310336918357319571079551779316654556781203013096026972411429993634080835",
            "3235435208525081966062419599803346573407862428113723170955762956243193422118",
            "1293239921425673430660897025143433077974838969258268884994339615096356996604",
        ],
        [
            "236252269127612784685426260840574970698541177557674806964960352572864382971",
            "1733907592497266237374827232200506798207318263912423249709509725341212026275",
            "302004309771755665128395814807589350526779835595021835389022325987048089868",
        ],
        [
            "3018926838139221755384801385583867283206879023218491758435446265703006270945",
            "39701437664873825906031098349904330565195980985885489447836580931425171297",
            "908381723021746969965674308809436059628307487140174335882627549095646509778",
        ],
        [
            "219062858908229855064136253265968615354041842047384625689776811853821594358",
            "1283129863776453589317845316917890202859466483456216900835390291449830275503",
            "418512623547417594896140369190919231877873410935689672661226540908900544012",
        ],
        [
            "1792181590047131972851015200157890246436013346535432437041535789841136268632",
            "370546432987510607338044736824316856592558876687225326692366316978098770516",
            "3323437805230586112013581113386626899534419826098235300155664022709435756946",
        ],
        [
            "910076621742039763058481476739499965761942516177975130656340375573185415877",
            "1762188042455633427137702520675816545396284185254002959309669405982213803405",
            "2186362253913140345102191078329764107619534641234549431429008219905315900520",
        ],
        [
            "2230647725927681765419218738218528849146504088716182944327179019215826045083",
            "1069243907556644434301190076451112491469636357133398376850435321160857761825",
            "2695241469149243992683268025359863087303400907336026926662328156934068747593",
        ],
        [
            "1361519681544413849831669554199151294308350560528931040264950307931824877035",
            "1339116632207878730171031743761550901312154740800549632983325427035029084904",
            "790593524918851401449292693473498591068920069246127392274811084156907468875",
        ],
        [
            "2723400368331924254840192318398326090089058735091724263333980290765736363637",
            "3457180265095920471443772463283225391927927225993685928066766687141729456030",
            "1483675376954327086153452545475557749815683871577400883707749788555424847954",
        ],
        [
            "2926303836265506736227240325795090239680154099205721426928300056982414025239",
            "543969119775473768170832347411484329362572550684421616624136244239799475526",
            "237401230683847084256617415614300816373730178313253487575312839074042461932",
        ],
        [
            "844568412840391587862072008674263874021460074878949862892685736454654414423",
            "151922054871708336050647150237534498235916969120198637893731715254687336644",
            "1299332034710622815055321547569101119597030148120309411086203580212105652312",
        ],
        [
            "487046922649899823989594814663418784068895385009696501386459462815688122993",
            "1104883249092599185744249485896585912845784382683240114120846423960548576851",
            "1458388705536282069567179348797334876446380557083422364875248475157495514484",
        ],
        [
            "850248109622750774031817200193861444623975329881731864752464222442574976566",
            "2885843173858536690032695698009109793537724845140477446409245651176355435722",
            "3027068551635372249579348422266406787688980506275086097330568993357835463816",
        ],
        [
            "3231892723647447539926175383213338123506134054432701323145045438168976970994",
            "1719080830641935421242626784132692936776388194122314954558418655725251172826",
            "1172253756541066126131022537343350498482225068791630219494878195815226839450",
        ],
        [
            "1619232269633026603732619978083169293258272967781186544174521481891163985093",
            "3495680684841853175973173610562400042003100419811771341346135531754869014567",
            "1576161515913099892951745452471618612307857113799539794680346855318958552758",
        ],
        [
            "2618326122974253423403350731396350223238201817594761152626832144510903048529",
            "2696245132758436974032479782852265185094623165224532063951287925001108567649",
            "930116505665110070247395429730201844026054810856263733273443066419816003444",
        ],
        [
            "2786389174502246248523918824488629229455088716707062764363111940462137404076",
            "1555260846425735320214671887347115247546042526197895180675436886484523605116",
            "2306241912153325247392671742757902161446877415586158295423293240351799505917",
        ],
        [
            "411529621724849932999694270803131456243889635467661223241617477462914950626",
            "1542495485262286701469125140275904136434075186064076910329015697714211835205",
            "1853045663799041100600825096887578544265580718909350942241802897995488264551",
        ],
        [
            "2963055259497271220202739837493041799968576111953080503132045092194513937286",
            "2303806870349915764285872605046527036748108533406243381676768310692344456050",
            "2622104986201990620910286730213140904984256464479840856728424375142929278875",
        ],
        [
            "2369987021925266811581727383184031736927816625797282287927222602539037105864",
            "285070227712021899602056480426671736057274017903028992288878116056674401781",
            "3034087076179360957800568733595959058628497428787907887933697691951454610691",
        ],
        [
            "469095854351700119980323115747590868855368701825706298740201488006320881056",
            "360001976264385426746283365024817520563236378289230404095383746911725100012",
            "3438709327109021347267562000879503009590697221730578667498351600602230296178",
        ],
        [
            "63573904800572228121671659287593650438456772568903228287754075619928214969",
            "3470881855042989871434874691030920672110111605547839662680968354703074556970",
            "724559311507950497340993415408274803001166693839947519425501269424891465492",
        ],
        [
            "880409284677518997550768549487344416321062350742831373397603704465823658986",
            "6876255662475867703077362872097208259197756317287339941435193538565586230",
            "2701916445133770775447884812906226786217969545216086200932273680400909154638",
        ],
        [
            "425152119158711585559310064242720816611629181537672850898056934507216982586",
            "1475552998258917706756737045704649573088377604240716286977690565239187213744",
            "2413772448122400684309006716414417978370152271397082147158000439863002593561",
        ],
        [
            "392160855822256520519339260245328807036619920858503984710539815951012864164",
            "1075036996503791536261050742318169965707018400307026402939804424927087093987",
            "2176439430328703902070742432016450246365760303014562857296722712989275658921",
        ],
        [
            "1413865976587623331051814207977382826721471106513581745229680113383908569693",
            "4879283427490523253696177116563427032332223531862961281430108575019551814",
            "3392583297537374046875199552977614390492290683707960975137418536812266544902",
        ],
        [
            "3600854486849487646325182927019642276644093512133907046667282144129939150983",
            "2779924664161372134024229593301361846129279572186444474616319283535189797834",
            "2722699960903170449291146429799738181514821447014433304730310678334403972040",
        ],
        [
            "819109815049226540285781191874507704729062681836086010078910930707209464699",
            "3046121243742768013822760785918001632929744274211027071381357122228091333823",
            "1339019590803056172509793134119156250729668216522001157582155155947567682278",
        ],
        [
            "1933279639657506214789316403763326578443023901555983256955812717638093967201",
            "2138221547112520744699126051903811860205771600821672121643894708182292213541",
            "2694713515543641924097704224170357995809887124438248292930846280951601597065",
        ],
        [
            "2471734202930133750093618989223585244499567111661178960753938272334153710615",
            "504903761112092757611047718215309856203214372330635774577409639907729993533",
            "1943979703748281357156510253941035712048221353507135074336243405478613241290",
        ],
        [
            "684525210957572142559049112233609445802004614280157992196913315652663518936",
            "1705585400798782397786453706717059483604368413512485532079242223503960814508",
            "192429517716023021556170942988476050278432319516032402725586427701913624665",
        ],
        [
            "1586493702243128040549584165333371192888583026298039652930372758731750166765",
            "686072673323546915014972146032384917012218151266600268450347114036285993377",
            "3464340397998075738891129996710075228740496767934137465519455338004332839215",
        ],
        [
            "2805249176617071054530589390406083958753103601524808155663551392362371834663",
            "667746464250968521164727418691487653339733392025160477655836902744186489526",
            "1131527712905109997177270289411406385352032457456054589588342450404257139778",
        ],
        [
            "1908969485750011212309284349900149072003218505891252313183123635318886241171",
            "1025257076985551890132050019084873267454083056307650830147063480409707787695",
            "2153175291918371429502545470578981828372846236838301412119329786849737957977",
        ],
        [
            "3410257749736714576487217882785226905621212230027780855361670645857085424384",
            "3442969106887588154491488961893254739289120695377621434680934888062399029952",
            "3029953900235731770255937704976720759948880815387104275525268727341390470237",
        ],
        [
            "85453456084781138713939104192561924536933417707871501802199311333127894466",
            "2730629666577257820220329078741301754580009106438115341296453318350676425129",
            "178242450661072967256438102630920745430303027840919213764087927763335940415",
        ],
        [
            "2844589222514708695700541363167856718216388819406388706818431442998498677557",
            "3547876269219141094308889387292091231377253967587961309624916269569559952944",
            "2525005406762984211707203144785482908331876505006839217175334833739957826850",
        ],
        [
            "3096397013555211396701910432830904669391580557191845136003938801598654871345",
            "574424067119200181933992948252007230348512600107123873197603373898923821490",
            "1714030696055067278349157346067719307863507310709155690164546226450579547098",
        ],
        [
            "2339895272202694698739231405357972261413383527237194045718815176814132612501",
            "3562501318971895161271663840954705079797767042115717360959659475564651685069",
            "69069358687197963617161747606993436483967992689488259107924379545671193749",
        ],
        [
            "2614502738369008850475068874731531583863538486212691941619835266611116051561",
            "655247349763023251625727726218660142895322325659927266813592114640858573566",
            "2305235672527595714255517865498269719545193172975330668070873705108690670678",
        ],
        [
            "926416070297755413261159098243058134401665060349723804040714357642180531931",
            "866523735635840246543516964237513287099659681479228450791071595433217821460",
            "2284334068466681424919271582037156124891004191915573957556691163266198707693",
        ],
        [
            "1812588309302477291425732810913354633465435706480768615104211305579383928792",
            "2836899808619013605432050476764608707770404125005720004551836441247917488507",
            "2989087789022865112405242078196235025698647423649950459911546051695688370523",
        ],
        [
            "68056284404189102136488263779598243992465747932368669388126367131855404486",
            "505425339250887519581119854377342241317528319745596963584548343662758204398",
            "2118963546856545068961709089296976921067035227488975882615462246481055679215",
        ],
        [
            "2253872596319969096156004495313034590996995209785432485705134570745135149681",
            "1625090409149943603241183848936692198923183279116014478406452426158572703264",
            "179139838844452470348634657368199622305888473747024389514258107503778442495",
        ],
        [
            "1567067018147735642071130442904093290030432522257811793540290101391210410341",
            "2737301854006865242314806979738760349397411136469975337509958305470398783585",
            "3002738216460904473515791428798860225499078134627026021350799206894618186256",
        ],
        [
            "374029488099466837453096950537275565120689146401077127482884887409712315162",
            "973403256517481077805460710540468856199855789930951602150773500862180885363",
            "2691967457038172130555117632010860984519926022632800605713473799739632878867",
        ],
        [
            "3515906794910381201365530594248181418811879320679684239326734893975752012109",
            "148057579455448384062325089530558091463206199724854022070244924642222283388",
            "1541588700238272710315890873051237741033408846596322948443180470429851502842",
        ],
        [
            "147013865879011936545137344076637170977925826031496203944786839068852795297",
            "2630278389304735265620281704608245039972003761509102213752997636382302839857",
            "1359048670759642844930007747955701205155822111403150159614453244477853867621",
        ],
        [
            "2438984569205812336319229336885480537793786558293523767186829418969842616677",
            "2137792255841525507649318539501906353254503076308308692873313199435029594138",
            "2262318076430740712267739371170174514379142884859595360065535117601097652755",
        ],
        [
            "2792703718581084537295613508201818489836796608902614779596544185252826291584",
            "2294173715793292812015960640392421991604150133581218254866878921346561546149",
            "2770011224727997178743274791849308200493823127651418989170761007078565678171",
        ],
    ];

    raw.iter()
        .map(|r| {
            [
                FieldElement::from_dec_str(r[0]).unwrap(),
                FieldElement::from_dec_str(r[1]).unwrap(),
                FieldElement::from_dec_str(r[2]).unwrap(),
            ]
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CudaFftError::NoDevice;
        assert_eq!(format!("{}", err), "No CUDA device found");

        let err = CudaFftError::KernelCompilation("test".to_string());
        assert!(format!("{}", err).contains("test"));
    }

    #[test]
    fn test_cuda_not_available_without_feature() {
        // When cuda-runtime feature is not enabled, CUDA should not be available
        #[cfg(not(feature = "cuda-runtime"))]
        {
            assert!(!is_cuda_available());
        }
    }
}
