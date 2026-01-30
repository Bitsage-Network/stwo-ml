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
    
    let mut pool_guard = pool.lock()
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
    WaitAndRetry { max_retries: u32, base_delay_ms: u64 },
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

    let (free, total) = compat::mem_get_info()
        .map_err(|e| CudaFftError::DriverInit(e))?;

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
pub fn check_memory_available(required_bytes: usize, safety_margin: f32) -> Result<bool, CudaFftError> {
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
                return Err(CudaFftError::MemoryAllocation(
                    format!(
                        "Insufficient GPU memory: need {} MB, only {} MB free",
                        required_bytes / (1024 * 1024),
                        stats.free_bytes / (1024 * 1024)
                    )
                ));
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
        
        MemoryPressureStrategy::WaitAndRetry { max_retries, base_delay_ms } => {
            let mut retries = 0;
            
            loop {
                if check_memory_available(required_bytes, 0.1)? {
                    match gpu_fn() {
                        Ok(result) => return Ok(result),
                        Err(CudaFftError::MemoryAllocation(msg)) => {
                            if retries >= max_retries {
                                return Err(CudaFftError::MemoryAllocation(
                                    format!("Out of GPU memory after {} retries: {}", max_retries, msg)
                                ));
                            }
                            retries += 1;
                        }
                        Err(e) => return Err(e),
                    }
                } else if retries >= max_retries {
                    let stats = get_memory_stats()?;
                    return Err(CudaFftError::MemoryAllocation(
                        format!(
                            "Timeout waiting for GPU memory: need {} MB, only {} MB free",
                            required_bytes / (1024 * 1024),
                            stats.free_bytes / (1024 * 1024)
                        )
                    ));
                }
                
                // Exponential backoff
                let delay = base_delay_ms * (1 << retries.min(5));
                tracing::debug!(
                    "Waiting for GPU memory (retry {}/{}), sleeping {} ms",
                    retries + 1, max_retries, delay
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
pub fn check_memory_available(_required_bytes: usize, _safety_margin: f32) -> Result<bool, CudaFftError> {
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
    // Quotient accumulation kernels
    pub accumulate_quotients: CudaFunction,
    pub copy_column: CudaFunction,  // GPU-resident column copy
    // MLE (GKR) operations kernels
    pub mle_fold_base_to_secure: CudaFunction,
    pub mle_fold_secure: CudaFunction,
    pub gen_eq_evals: CudaFunction,
    // Merkle hashing kernel
    pub merkle_layer: CudaFunction,
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
        let compute_stream = device.fork_default_stream()
            .map_err(|e| CudaFftError::DriverInit(format!("Compute stream: {:?}", e)))
            .ok();
        
        let transfer_stream = device.fork_default_stream()
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
            self.device.wait_for(stream)
                .map_err(|e| CudaFftError::KernelExecution(format!("Compute stream sync: {:?}", e)))?;
        }
        Ok(())
    }
    
    /// Synchronize the transfer stream (wait for all transfers to complete).
    pub fn sync_transfer(&self) -> Result<(), CudaFftError> {
        if let Some(stream) = &self.transfer_stream {
            self.device.wait_for(stream)
                .map_err(|e| CudaFftError::KernelExecution(format!("Transfer stream sync: {:?}", e)))?;
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
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        ).map_err(|e| CudaFftError::DriverInit(e))? as u32;

        let minor = compat::device_get_attribute(
            *cu_device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        ).map_err(|e| CudaFftError::DriverInit(e))? as u32;

        // Query device name
        let name = compat::device_get_name(*cu_device)
            .unwrap_or_else(|_| "NVIDIA GPU".to_string());

        // Query total memory
        let total_memory_bytes = compat::device_total_mem(*cu_device)
            .unwrap_or(8 * 1024 * 1024 * 1024);

        // Query SM count
        let multiprocessor_count = compat::device_get_attribute(
            *cu_device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        ).map_err(|e| CudaFftError::DriverInit(e))? as u32;

        // Query max threads per block
        let max_threads_per_block = compat::device_get_attribute(
            *cu_device,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        ).map_err(|e| CudaFftError::DriverInit(e))? as u32;

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
            let marker_path = std::path::PathBuf::from(&ptx_dir)
                .join(format!("{}.marker", kernel_name));
            
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
            let ptx_path = std::path::PathBuf::from(&ptx_dir)
                .join(format!("{}.ptx", kernel_name));
            
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
        
        let ptx = cudarc::nvrtc::compile_ptx(source)
            .map_err(|e| CudaFftError::KernelCompilation(format!("{} kernel: {:?}", kernel_name, e)))?;
        
        let compile_time = start.elapsed();
        tracing::info!(
            "Compiled {} PTX in {:?}",
            kernel_name,
            compile_time
        );
        
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
        let fft_ptx = Self::compile_or_load_cached(
            "circle_fft",
            CIRCLE_FFT_CUDA_KERNEL,
        )?;
        
        // Load FFT PTX into device
        device.load_ptx(fft_ptx, "circle_fft", &[
            "bit_reverse_kernel",
            "ifft_layer_kernel",
            "fft_layer_kernel",
            "ifft_shared_mem_kernel",
            "denormalize_kernel",
            "denormalize_vec4_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("FFT load: {:?}", e)))?;
        
        // Compile or load cached FRI PTX
        use super::fft::FRI_FOLDING_CUDA_KERNEL;
        let fri_ptx = Self::compile_or_load_cached(
            "fri_folding",
            FRI_FOLDING_CUDA_KERNEL,
        )?;
        
        // Load FRI PTX into device
        device.load_ptx(fri_ptx, "fri_folding", &[
            "fold_line_kernel",
            "fold_circle_into_line_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("FRI load: {:?}", e)))?;
        
        // Compile or load cached Quotient PTX
        use super::fft::QUOTIENT_CUDA_KERNEL;
        let quotient_ptx = Self::compile_or_load_cached(
            "quotient",
            QUOTIENT_CUDA_KERNEL,
        )?;
        
        // Load Quotient PTX into device (includes buffer gather and MLE kernels)
        device.load_ptx(quotient_ptx, "quotient", &[
            "accumulate_quotients_kernel",
            "copy_column_kernel",           // GPU-resident column copy
            "gather_buffers_kernel",        // GPU-resident buffer gathering
            "mle_fold_base_to_secure_kernel", // MLE fold: BaseField -> SecureField
            "mle_fold_secure_kernel",       // MLE fold: SecureField -> SecureField  
            "gen_eq_evals_kernel",          // Generate equality evaluations for GKR
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("Quotient load: {:?}", e)))?;
        
        // Get FFT function handles
        let bit_reverse = device.get_func("circle_fft", "bit_reverse_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("bit_reverse_kernel not found".into()))?;
        
        let ifft_layer = device.get_func("circle_fft", "ifft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("ifft_layer_kernel not found".into()))?;
        
        let fft_layer = device.get_func("circle_fft", "fft_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fft_layer_kernel not found".into()))?;
        
        let ifft_shared_mem = device.get_func("circle_fft", "ifft_shared_mem_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("ifft_shared_mem_kernel not found".into()))?;
        
        // Get denormalization function handles
        let denormalize = device.get_func("circle_fft", "denormalize_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("denormalize_kernel not found".into()))?;
        
        let denormalize_vec4 = device.get_func("circle_fft", "denormalize_vec4_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("denormalize_vec4_kernel not found".into()))?;
        
        // Get FRI function handles
        let fold_line = device.get_func("fri_folding", "fold_line_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fold_line_kernel not found".into()))?;
        
        let fold_circle_into_line = device.get_func("fri_folding", "fold_circle_into_line_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("fold_circle_into_line_kernel not found".into()))?;
        
        // Get Quotient function handle
        let accumulate_quotients = device.get_func("quotient", "accumulate_quotients_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("accumulate_quotients_kernel not found".into()))?;
        
        // Get copy_column kernel for GPU-resident column gathering
        let copy_column = device.get_func("quotient", "copy_column_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("copy_column_kernel not found".into()))?;
        
        // Get MLE kernels for GKR operations
        let mle_fold_base_to_secure = device.get_func("quotient", "mle_fold_base_to_secure_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("mle_fold_base_to_secure_kernel not found".into()))?;
        
        let mle_fold_secure = device.get_func("quotient", "mle_fold_secure_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("mle_fold_secure_kernel not found".into()))?;
        
        let gen_eq_evals = device.get_func("quotient", "gen_eq_evals_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("gen_eq_evals_kernel not found".into()))?;
        
        // Compile or load cached Blake2s Merkle PTX
        use super::fft::BLAKE2S_MERKLE_CUDA_KERNEL;
        let merkle_ptx = Self::compile_or_load_cached(
            "merkle_blake2s",
            BLAKE2S_MERKLE_CUDA_KERNEL,
        )?;
        
        // Load Merkle PTX into device
        device.load_ptx(merkle_ptx, "merkle", &[
            "merkle_layer_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("Merkle load: {:?}", e)))?;
        
        // Get Merkle function handle
        let merkle_layer = device.get_func("merkle", "merkle_layer_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation("merkle_layer_kernel not found".into()))?;
        
        tracing::info!("Compiled FFT, FRI, Quotient, and Merkle kernels successfully");
        
        Ok(CompiledKernels {
            bit_reverse,
            ifft_layer,
            fft_layer,
            ifft_shared_mem,
            denormalize,
            denormalize_vec4,
            fold_line,
            fold_circle_into_line,
            accumulate_quotients,
            copy_column,
            mle_fold_base_to_secure,
            mle_fold_secure,
            gen_eq_evals,
            merkle_layer,
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
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} elements, got {}", n, data.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA IFFT", log_size = log_size).entered();
        
        // Allocate device memory
        let mut d_data = self.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Flatten twiddles and copy to device
        let flat_twiddles: Vec<u32> = twiddles_dbl.iter().flatten().copied().collect();
        let d_twiddles = self.device.htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Execute IFFT layers
        self.execute_ifft_layers(&mut d_data, &d_twiddles, log_size, twiddles_dbl)?;
        
        // Copy results back
        self.device.dtoh_sync_copy_into(&d_data, data)
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
            let d_twiddle_offsets = self.device.htod_sync_copy(&twiddle_offsets)
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
            
            // Launch shared memory kernel for first layers
            let num_blocks = n / SHMEM_ELEMENTS;
            
            let cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (SHMEM_BLOCK_SIZE, 1, 1),
                shared_mem_bytes: SHMEM_ELEMENTS * 4, // 4 bytes per u32
            };
            
            unsafe {
                self.kernels.ifft_shared_mem.clone().launch(
                    cfg,
                    (
                        &mut *d_data,
                        d_twiddles,
                        &d_twiddle_offsets,
                        shared_mem_layers,
                        log_size,
                    ),
                ).map_err(|e| CudaFftError::KernelExecution(format!("Shared mem kernel: {:?}", e)))?;
            }
            
            // Sync after shared memory kernel
            self.device.synchronize()
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
                    self.kernels.ifft_layer.clone().launch(
                        cfg,
                        (&mut *d_data, &twiddle_view, layer as u32, log_size, n_twiddles),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
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
                    self.kernels.ifft_layer.clone().launch(
                        cfg,
                        (&mut *d_data, &twiddle_view, layer as u32, log_size, n_twiddles),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        }
        
        // Final sync
        self.device.synchronize()
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
        self.execute_ifft_layers_with_offsets(d_data, d_twiddles, d_twiddle_offsets, log_size, twiddles_dbl)
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
                self.kernels.ifft_shared_mem.clone().launch(
                    cfg,
                    (
                        &mut *d_data,
                        d_twiddles,
                        d_twiddle_offsets,
                        shared_mem_layers,
                        log_size,
                    ),
                ).map_err(|e| CudaFftError::KernelExecution(format!("Shared mem kernel: {:?}", e)))?;
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
                    self.kernels.ifft_layer.clone().launch(
                        cfg,
                        (&mut *d_data, &twiddle_view, layer as u32, log_size, n_twiddles),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
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
                    self.kernels.ifft_layer.clone().launch(
                        cfg,
                        (&mut *d_data, &twiddle_view, layer as u32, log_size, n_twiddles),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
            }
        }
        
        // Sync at the end
        self.device.synchronize()
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
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} elements, got {}", n, data.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA FFT", log_size = log_size).entered();
        
        // Allocate device memory
        let mut d_data = self.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Flatten twiddles and copy to device
        let flat_twiddles: Vec<u32> = twiddles.iter().flatten().copied().collect();
        let d_twiddles = self.device.htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Execute FFT layers (reverse order of IFFT)
        self.execute_fft_layers(&mut d_data, &d_twiddles, log_size, twiddles)?;
        
        // Copy results back
        self.device.dtoh_sync_copy_into(&d_data, data)
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
                self.kernels.fft_layer.clone().launch(
                    cfg,
                    (&mut *d_data, &twiddle_view, layer as u32, log_size, n_twiddles),
                ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
            }
        }
        
        // Synchronize
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        Ok(())
    }
    
    /// Execute bit reversal permutation on GPU.
    pub fn bit_reverse(&self, data: &mut [u32], log_size: u32) -> Result<(), CudaFftError> {
        let n = 1usize << log_size;
        
        if data.len() != n {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} elements, got {}", n, data.len())
            ));
        }
        
        // Allocate and copy
        let mut d_data = self.device.htod_sync_copy(data)
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
            self.kernels.bit_reverse.clone().launch(
                cfg,
                (&mut d_data, log_size),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        self.device.dtoh_sync_copy_into(&d_data, data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    /// Get device memory info.
    pub fn memory_info(&self) -> (usize, usize) {
        // (free, total) - cudarc doesn't expose this directly
        (
            self.device_info.total_memory_bytes / 2,  // Estimate
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
                self.kernels.denormalize_vec4.clone().launch(
                    cfg,
                    (&mut *d_data, denorm_factor, n),
                ).map_err(|e| CudaFftError::KernelExecution(format!("Denormalize vec4: {:?}", e)))?;
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
                self.kernels.denormalize.clone().launch(
                    cfg,
                    (&mut *d_data, denorm_factor, n),
                ).map_err(|e| CudaFftError::KernelExecution(format!("Denormalize: {:?}", e)))?;
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
        let mut d_data = self.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Execute on device
        self.execute_denormalize_on_device(&mut d_data, denorm_factor, n)?;
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        self.device.dtoh_sync_copy_into(&d_data, data)
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
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} u32 values, got {}", n * 4, input.len())
            ));
        }
        if itwiddles.len() < n / 2 {
            return Err(CudaFftError::InvalidSize(
                format!("Expected at least {} twiddles, got {}", n / 2, itwiddles.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA fold_line", n = n).entered();
        
        let n_output = n / 2;
        
        // Allocate device memory
        let d_input = self.device.htod_sync_copy(input)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_itwiddles = self.device.htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_alpha = self.device.htod_sync_copy(alpha)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut d_output = unsafe {
            self.device.alloc::<u32>(n_output * 4)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
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
            self.kernels.fold_line.clone().launch(
                cfg,
                (&mut d_output, &d_input, &d_itwiddles, &d_alpha, n as u32, log_n),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        let mut output = vec![0u32; n_output * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
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
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} u32 values in src, got {}", n * 4, src.len())
            ));
        }
        let n_dst = n / 2;
        if dst.len() != n_dst * 4 {
            return Err(CudaFftError::InvalidSize(
                format!("Expected {} u32 values in dst, got {}", n_dst * 4, dst.len())
            ));
        }
        if itwiddles.len() < n_dst {
            return Err(CudaFftError::InvalidSize(
                format!("Expected at least {} twiddles, got {}", n_dst, itwiddles.len())
            ));
        }
        
        let _span = tracing::span!(tracing::Level::INFO, "CUDA fold_circle_into_line", n = n).entered();
        
        // Allocate device memory
        let d_src = self.device.htod_sync_copy(src)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut d_dst = self.device.htod_sync_copy(dst)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_itwiddles = self.device.htod_sync_copy(itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_alpha = self.device.htod_sync_copy(alpha)
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
            self.kernels.fold_circle_into_line.clone().launch(
                cfg,
                (&mut d_dst, &d_src, &d_itwiddles, &d_alpha, n as u32, log_n),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        self.device.dtoh_sync_copy_into(&d_dst, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        tracing::info!("GPU fold_circle_into_line completed: {} -> {} elements", n, n_dst);
        
        Ok(())
    }
    
    // =========================================================================
    // Quotient Accumulation Operations
    // =========================================================================
    
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
        let _span = tracing::span!(tracing::Level::INFO, "CUDA accumulate_quotients", n_points = n_points).entered();
        
        let n_columns = columns.len();
        let n_batches = batch_sizes.len();
        
        // Flatten columns (interleave by point, not by column)
        // Layout: col0[0], col1[0], col2[0], ..., col0[1], col1[1], ...
        let mut flat_columns: Vec<u32> = Vec::with_capacity(n_columns * n_points);
        for col in columns {
            flat_columns.extend_from_slice(col);
        }
        
        // Flatten line coefficients
        let flat_line_coeffs: Vec<u32> = line_coeffs.iter()
            .flat_map(|coeffs| coeffs.iter().copied())
            .collect();
        
        // Convert batch_sizes and col_indices to u32
        let batch_sizes_u32: Vec<u32> = batch_sizes.iter().map(|&s| s as u32).collect();
        let col_indices_u32: Vec<u32> = col_indices.iter().map(|&i| i as u32).collect();
        
        // Allocate device memory
        let d_columns = self.device.htod_sync_copy(&flat_columns)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_line_coeffs = self.device.htod_sync_copy(&flat_line_coeffs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_denom_inv = self.device.htod_sync_copy(denom_inv)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_batch_sizes = self.device.htod_sync_copy(&batch_sizes_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let d_col_indices = self.device.htod_sync_copy(&col_indices_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut d_output = unsafe {
            self.device.alloc::<u32>(n_points * 4)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n_points as u32) + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.kernels.accumulate_quotients.clone().launch(
                cfg,
                (
                    &mut d_output,
                    &d_columns,
                    &d_line_coeffs,
                    &d_denom_inv,
                    &d_batch_sizes,
                    &d_col_indices,
                    n_batches as u32,
                    n_points as u32,
                    n_columns as u32,
                ),
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        let mut output = vec![0u32; n_points * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        tracing::info!("GPU accumulate_quotients completed: {} points, {} batches", n_points, n_batches);
        
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
        let _span = tracing::span!(tracing::Level::INFO, "CUDA blake2s_merkle", n_hashes = n_hashes).entered();
        
        let n_columns = columns.len();
        
        // Flatten columns
        let flat_columns: Vec<u32> = columns.iter()
            .flat_map(|col| col.iter().copied())
            .collect();
        
        // Allocate device memory for columns (if any)
        let d_columns = if n_columns > 0 {
            Some(self.device.htod_sync_copy(&flat_columns)
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?)
        } else {
            None
        };
        
        // Allocate device memory for previous layer (if any)
        let d_prev_layer = if let Some(prev) = prev_layer {
            Some(self.device.htod_sync_copy(prev)
                .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?)
        } else {
            None
        };
        
        // Allocate output (32 bytes per hash)
        let mut d_output = unsafe {
            self.device.alloc::<u8>(n_hashes * 32)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
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
                    self.kernels.merkle_layer.clone().launch(
                        cfg,
                        (
                            &mut d_output,
                            cols,
                            prev,
                            n_columns as u32,
                            n_hashes as u32,
                            has_prev_layer,
                        ),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (Some(cols), None) => {
                    // Need a dummy buffer for prev_layer
                    let dummy_prev = self.device.alloc::<u8>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels.merkle_layer.clone().launch(
                        cfg,
                        (
                            &mut d_output,
                            cols,
                            &dummy_prev,
                            n_columns as u32,
                            n_hashes as u32,
                            has_prev_layer,
                        ),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, Some(prev)) => {
                    // Need a dummy buffer for columns
                    let dummy_cols = self.device.alloc::<u32>(1)
                        .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
                    self.kernels.merkle_layer.clone().launch(
                        cfg,
                        (
                            &mut d_output,
                            &dummy_cols,
                            prev,
                            n_columns as u32,
                            n_hashes as u32,
                            has_prev_layer,
                        ),
                    ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
                }
                (None, None) => {
                    return Err(CudaFftError::InvalidSize(
                        "Merkle hashing requires either columns or prev_layer".to_string()
                    ));
                }
            }
        }
        
        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;
        
        let mut output = vec![0u8; n_hashes * 32];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        
        tracing::info!("GPU blake2s_merkle completed: {} hashes", n_hashes);

        Ok(output)
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
        let _span = tracing::span!(tracing::Level::INFO, "CUDA mle_fold_base_to_secure", n = n).entered();

        if lhs.len() != n || rhs.len() != n {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} elements for lhs and rhs, got {} and {}",
                n, lhs.len(), rhs.len()
            )));
        }

        // Allocate device memory
        let d_lhs = self.device.htod_sync_copy(lhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_rhs = self.device.htod_sync_copy(rhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe {
            self.device.alloc::<u32>(n * 4)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels.mle_fold_base_to_secure.clone().launch(
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
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; n * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
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
                n * 4, lhs.len(), rhs.len()
            )));
        }

        // Allocate device memory
        let d_lhs = self.device.htod_sync_copy(lhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let d_rhs = self.device.htod_sync_copy(rhs)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe {
            self.device.alloc::<u32>(n * 4)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels.mle_fold_secure.clone().launch(
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
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; n * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
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
        let _span = tracing::span!(tracing::Level::INFO, "CUDA gen_eq_evals",
            n_variables = n_variables, output_size = output_size).entered();

        if y.len() != n_variables * 4 {
            return Err(CudaFftError::InvalidSize(format!(
                "Expected {} u32 values for y, got {}",
                n_variables * 4, y.len()
            )));
        }

        // Allocate device memory
        let d_y = self.device.htod_sync_copy(y)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let mut d_output = unsafe {
            self.device.alloc::<u32>(output_size * 4)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = ((output_size as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels.gen_eq_evals.clone().launch(
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
            ).map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))?;
        }

        // Synchronize and copy back
        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("Sync failed: {:?}", e)))?;

        let mut output = vec![0u32; output_size * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        tracing::info!("GPU gen_eq_evals completed: {} output elements", output_size);

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
        columns, line_coeffs, denom_inv, batch_sizes, col_indices, n_points
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

