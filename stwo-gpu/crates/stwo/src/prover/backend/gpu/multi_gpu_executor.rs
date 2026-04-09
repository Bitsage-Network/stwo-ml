//! Thread-Safe Multi-GPU Executor
//!
//! This module provides a thread-safe executor that can manage multiple GPUs
//! with proper CUDA context handling for parallel execution.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    MultiGpuExecutorPool                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │   │ GpuExecutor  │  │ GpuExecutor  │  │ GpuExecutor  │  ...     │
//! │   │   (GPU 0)    │  │   (GPU 1)    │  │   (GPU 2)    │          │
//! │   │  Mutex<...>  │  │  Mutex<...>  │  │  Mutex<...>  │          │
//! │   └──────────────┘  └──────────────┘  └──────────────┘          │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each GPU has its own executor wrapped in a Mutex, allowing thread-safe
//! access from multiple threads.

#[cfg(feature = "cuda-runtime")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::{CudaFftError, CudaFftExecutor};
#[cfg(feature = "cuda-runtime")]
use super::fft::{compute_itwiddle_dbls_cpu, compute_twiddle_dbls_cpu};

// =============================================================================
// Global Multi-GPU Executor Pool
// =============================================================================

#[cfg(feature = "cuda-runtime")]
static MULTI_GPU_POOL: OnceLock<Result<MultiGpuExecutorPool, CudaFftError>> = OnceLock::new();

/// Get or initialize the global multi-GPU executor pool.
///
/// This function attempts to initialize GPUs in the following order:
/// 1. All available GPUs
/// 2. Fallback to single GPU (device 0)
/// 3. Returns error if no GPU is available
#[cfg(feature = "cuda-runtime")]
pub fn get_multi_gpu_pool() -> Result<&'static MultiGpuExecutorPool, CudaFftError> {
    let result = MULTI_GPU_POOL.get_or_init(|| {
        // Try all GPUs first
        match MultiGpuExecutorPool::new_all_gpus() {
            Ok(pool) => Ok(pool),
            Err(e) => {
                tracing::warn!("Failed to initialize all GPUs: {:?}, trying single GPU", e);
                // Fallback to single GPU
                match MultiGpuExecutorPool::new_with_devices(&[0]) {
                    Ok(pool) => Ok(pool),
                    Err(e) => {
                        tracing::error!("Failed to initialize any GPU: {:?}", e);
                        Err(e)
                    }
                }
            }
        }
    });

    match result {
        Ok(pool) => Ok(pool),
        Err(e) => Err(e.clone()),
    }
}

// =============================================================================
// Multi-GPU Executor Pool
// =============================================================================

/// Pool of GPU executors for multi-GPU operations.
#[cfg(feature = "cuda-runtime")]
pub struct MultiGpuExecutorPool {
    /// Per-GPU executors (thread-safe)
    executors: Vec<Arc<Mutex<GpuExecutorContext>>>,
    /// Device IDs
    device_ids: Vec<usize>,
}

/// Context for a single GPU with all resources needed for proof generation.
#[cfg(feature = "cuda-runtime")]
pub struct GpuExecutorContext {
    /// The CUDA executor
    pub executor: CudaFftExecutor,
    /// Cached twiddles for common sizes
    pub twiddle_cache: std::collections::HashMap<u32, TwiddleCache>,
}

/// Cached twiddles for a specific log_size.
#[cfg(feature = "cuda-runtime")]
pub struct TwiddleCache {
    pub itwiddles: CudaSlice<u32>,
    pub twiddles: CudaSlice<u32>,
    pub twiddle_offsets: CudaSlice<u32>,
    pub itwiddles_cpu: Vec<Vec<u32>>,
    pub twiddles_cpu: Vec<Vec<u32>>,
}

#[cfg(feature = "cuda-runtime")]
impl MultiGpuExecutorPool {
    /// Create a pool with all available GPUs.
    pub fn new_all_gpus() -> Result<Self, CudaFftError> {
        let device_count = Self::get_device_count()?;
        if device_count == 0 {
            return Err(CudaFftError::NoDevice);
        }
        
        let device_ids: Vec<usize> = (0..device_count).collect();
        Self::new_with_devices(&device_ids)
    }
    
    /// Create a pool with specific GPUs.
    pub fn new_with_devices(device_ids: &[usize]) -> Result<Self, CudaFftError> {
        let mut executors = Vec::new();
        let mut valid_ids = Vec::new();
        
        for &device_id in device_ids {
            match CudaFftExecutor::new_on_device(device_id) {
                Ok(executor) => {
                    let context = GpuExecutorContext {
                        executor,
                        twiddle_cache: std::collections::HashMap::new(),
                    };
                    executors.push(Arc::new(Mutex::new(context)));
                    valid_ids.push(device_id);
                    tracing::info!("Initialized GPU {} for multi-GPU pool", device_id);
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize GPU {}: {:?}", device_id, e);
                }
            }
        }
        
        if executors.is_empty() {
            return Err(CudaFftError::NoDevice);
        }
        
        tracing::info!("Multi-GPU pool initialized with {} GPUs", executors.len());
        
        Ok(Self {
            executors,
            device_ids: valid_ids,
        })
    }
    
    /// Get the number of GPUs in the pool.
    pub fn gpu_count(&self) -> usize {
        self.executors.len()
    }
    
    /// Get device IDs.
    pub fn device_ids(&self) -> &[usize] {
        &self.device_ids
    }
    
    /// Get executor for a specific GPU (by pool index, not device ID).
    pub fn get_executor(&self, pool_index: usize) -> Option<Arc<Mutex<GpuExecutorContext>>> {
        self.executors.get(pool_index).cloned()
    }
    
    /// Execute a function on a specific GPU.
    /// 
    /// This locks the GPU's executor for the duration of the function.
    pub fn with_gpu<F, R>(&self, pool_index: usize, f: F) -> Result<R, CudaFftError>
    where
        F: FnOnce(&mut GpuExecutorContext) -> Result<R, CudaFftError>,
    {
        let executor = self.executors.get(pool_index)
            .ok_or_else(|| CudaFftError::InvalidSize(format!("Invalid GPU index: {}", pool_index)))?;
        
        let mut guard = executor.lock()
            .map_err(|_| CudaFftError::KernelExecution("Failed to lock GPU executor".into()))?;
        
        f(&mut guard)
    }
    
    /// Execute functions on all GPUs in parallel.
    /// 
    /// Returns results from all GPUs.
    pub fn parallel_execute<F, R>(&self, f: F) -> Vec<Result<R, CudaFftError>>
    where
        F: Fn(usize, &mut GpuExecutorContext) -> Result<R, CudaFftError> + Send + Sync + 'static,
        R: Send + 'static,
    {
        use std::thread;
        
        let f = Arc::new(f);
        let mut handles = Vec::new();
        
        for (idx, executor) in self.executors.iter().enumerate() {
            let executor = Arc::clone(executor);
            let f = Arc::clone(&f);
            
            let handle = thread::spawn(move || {
                let mut guard = executor.lock()
                    .map_err(|_| CudaFftError::KernelExecution("Failed to lock GPU executor".into()))?;
                f(idx, &mut guard)
            });
            
            handles.push(handle);
        }
        
        handles.into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(CudaFftError::KernelExecution("Thread panicked".into()))))
            .collect()
    }
    
    fn get_device_count() -> Result<usize, CudaFftError> {
        let mut count = 0;
        for i in 0..16 {
            if CudaDevice::new(i).is_ok() {
                count = i + 1;
            } else {
                break;
            }
        }
        Ok(count)
    }
}

#[cfg(feature = "cuda-runtime")]
impl GpuExecutorContext {
    /// Get or create twiddles for a specific log_size.
    pub fn get_or_create_twiddles(&mut self, log_size: u32) -> Result<&TwiddleCache, CudaFftError> {
        // Check if we need to create the cache first
        if !self.twiddle_cache.contains_key(&log_size) {
            let cache = self.create_twiddle_cache(log_size)?;
            self.twiddle_cache.insert(log_size, cache);
        }

        // Safe: we just ensured the entry exists above
        self.twiddle_cache.get(&log_size)
            .ok_or_else(|| CudaFftError::InvalidSize(
                format!("Twiddle cache entry missing for log_size {} (internal error)", log_size)
            ))
    }
    
    fn create_twiddle_cache(&self, log_size: u32) -> Result<TwiddleCache, CudaFftError> {
        let itwiddles_cpu = compute_itwiddle_dbls_cpu(log_size);
        let twiddles_cpu = compute_twiddle_dbls_cpu(log_size);
        
        let flat_itwiddles: Vec<u32> = itwiddles_cpu.iter().flatten().copied().collect();
        let flat_twiddles: Vec<u32> = twiddles_cpu.iter().flatten().copied().collect();
        
        let itwiddles = self.executor.device.htod_sync_copy(&flat_itwiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let twiddles = self.executor.device.htod_sync_copy(&flat_twiddles)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        let mut offsets: Vec<u32> = Vec::new();
        let mut offset = 0u32;
        for tw in &itwiddles_cpu {
            offsets.push(offset);
            offset += tw.len() as u32;
        }
        let twiddle_offsets = self.executor.device.htod_sync_copy(&offsets)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        
        Ok(TwiddleCache {
            itwiddles,
            twiddles,
            twiddle_offsets,
            itwiddles_cpu,
            twiddles_cpu,
        })
    }
    
    /// Allocate GPU memory for polynomial data.
    pub fn allocate_poly(&self, log_size: u32) -> Result<CudaSlice<u32>, CudaFftError> {
        let n = 1usize << log_size;
        unsafe {
            self.executor.device.alloc::<u32>(n)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))
    }
    
    /// Upload polynomial to GPU.
    pub fn upload_poly(&self, data: &[u32]) -> Result<CudaSlice<u32>, CudaFftError> {
        self.executor.device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))
    }
    
    /// Download polynomial from GPU.
    pub fn download_poly(&self, d_data: &CudaSlice<u32>) -> Result<Vec<u32>, CudaFftError> {
        self.executor.device.dtoh_sync_copy(d_data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))
    }
    
    /// Synchronize the device.
    pub fn sync(&self) -> Result<(), CudaFftError> {
        self.executor.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))
    }
    
    /// Ensure twiddles are cached for a log_size (call before execute_ifft).
    pub fn ensure_twiddles(&mut self, log_size: u32) -> Result<(), CudaFftError> {
        if !self.twiddle_cache.contains_key(&log_size) {
            let cache = self.create_twiddle_cache(log_size)?;
            self.twiddle_cache.insert(log_size, cache);
        }
        Ok(())
    }
    
    /// Execute IFFT on polynomial data.
    /// 
    /// Twiddles must be pre-cached via ensure_twiddles().
    pub fn execute_ifft(&self, d_poly: &mut CudaSlice<u32>, log_size: u32) -> Result<(), CudaFftError> {
        let twiddles = self.twiddle_cache.get(&log_size)
            .ok_or_else(|| CudaFftError::InvalidSize(
                format!("Twiddles not cached for log_size {}. Call ensure_twiddles first.", log_size)
            ))?;
        
        self.executor.execute_ifft_on_device(
            d_poly,
            &twiddles.itwiddles,
            &twiddles.twiddle_offsets,
            &twiddles.itwiddles_cpu,
            log_size,
        )
    }
    
    /// Execute forward FFT on polynomial data using layer-by-layer approach.
    /// 
    /// Twiddles must be pre-cached via ensure_twiddles().
    pub fn execute_fft(&self, d_poly: &mut CudaSlice<u32>, log_size: u32) -> Result<(), CudaFftError> {
        let twiddles = self.twiddle_cache.get(&log_size)
            .ok_or_else(|| CudaFftError::InvalidSize(
                format!("Twiddles not cached for log_size {}. Call ensure_twiddles first.", log_size)
            ))?;
        
        // Execute FFT using the layer kernel
        let block_size = 256u32;
        let num_layers = twiddles.twiddles_cpu.len();
        
        // Calculate twiddle offsets
        let mut twiddle_offsets: Vec<usize> = Vec::new();
        let mut offset = 0usize;
        for tw in &twiddles.twiddles_cpu {
            twiddle_offsets.push(offset);
            offset += tw.len();
        }
        
        // Execute layers in reverse order for forward FFT
        for layer in (0..num_layers).rev() {
            let n_twiddles = twiddles.twiddles_cpu[layer].len();
            let grid_size = ((n_twiddles as u32) + block_size - 1) / block_size;
            
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            
            // Create a view into the twiddles at the correct offset
            let twiddle_offset = twiddle_offsets[layer];
            let twiddle_view = twiddles.twiddles.slice(twiddle_offset..twiddle_offset + n_twiddles);
            
            unsafe {
                self.executor.kernels.fft_layer.clone().launch(
                    cfg,
                    (&mut *d_poly, &twiddle_view, layer as u32, log_size, n_twiddles as u32),
                )
            }.map_err(|e| CudaFftError::KernelExecution(format!("FFT layer {} failed: {:?}", layer, e)))?;
        }
        
        self.executor.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("{:?}", e)))
    }
    
    /// Execute full proof pipeline: IFFT -> FFT -> return result.
    /// 
    /// This is a convenience method that handles twiddle caching internally.
    pub fn execute_proof_pipeline(&mut self, data: &[u32], log_size: u32) -> Result<Vec<u32>, CudaFftError> {
        // Ensure twiddles are cached
        self.ensure_twiddles(log_size)?;
        
        // Upload
        let mut d_poly = self.upload_poly(data)?;
        
        // IFFT
        self.execute_ifft(&mut d_poly, log_size)?;
        
        // FFT
        self.execute_fft(&mut d_poly, log_size)?;
        
        // Sync and download
        self.sync()?;
        self.download_poly(&d_poly)
    }
}

// =============================================================================
// Note: ThreadSafeProofPipeline removed due to CudaSlice not implementing Default.
// Use TrueMultiGpuProver.prove_parallel() instead for multi-GPU workloads.
// =============================================================================

// =============================================================================
// True Multi-GPU Prover
// =============================================================================

/// Multi-GPU prover that truly uses all GPUs in parallel.
#[cfg(feature = "cuda-runtime")]
pub struct TrueMultiGpuProver {
    log_size: u32,
}

#[cfg(feature = "cuda-runtime")]
impl TrueMultiGpuProver {
    /// Create a new multi-GPU prover.
    pub fn new(log_size: u32) -> Result<Self, CudaFftError> {
        // Initialize the pool
        let _ = get_multi_gpu_pool()?;
        Ok(Self { log_size })
    }
    
    /// Get the number of available GPUs.
    pub fn gpu_count(&self) -> Result<usize, CudaFftError> {
        Ok(get_multi_gpu_pool()?.gpu_count())
    }
    
    /// Process proofs in parallel across all GPUs.
    /// 
    /// Each GPU processes a subset of the workloads.
    pub fn prove_parallel<F, R>(&self, workloads: Vec<Vec<u32>>, process_fn: F) -> Vec<Result<R, CudaFftError>>
    where
        F: Fn(usize, &mut GpuExecutorContext, &[u32], u32) -> Result<R, CudaFftError> + Send + Sync + 'static,
        R: Send + 'static,
    {
        use std::thread;
        
        let pool = match get_multi_gpu_pool() {
            Ok(p) => p,
            Err(e) => return vec![Err(e)],
        };
        
        let num_gpus = pool.gpu_count();
        let log_size = self.log_size;
        
        // Distribute workloads across GPUs
        let mut gpu_workloads: Vec<Vec<(usize, Vec<u32>)>> = vec![Vec::new(); num_gpus];
        for (i, workload) in workloads.into_iter().enumerate() {
            let gpu_idx = i % num_gpus;
            gpu_workloads[gpu_idx].push((i, workload));
        }
        
        let process_fn = Arc::new(process_fn);
        let mut handles = Vec::new();
        
        for (gpu_idx, workloads) in gpu_workloads.into_iter().enumerate() {
            let executor = match pool.get_executor(gpu_idx) {
                Some(e) => e,
                None => continue,
            };
            let process_fn = Arc::clone(&process_fn);
            
            let handle = thread::spawn(move || {
                let mut results = Vec::new();
                
                let mut guard = executor.lock()
                    .map_err(|_| CudaFftError::KernelExecution("Lock failed".into()))?;
                
                for (_orig_idx, workload) in workloads {
                    let result = process_fn(gpu_idx, &mut guard, &workload, log_size);
                    results.push(result);
                }
                
                Ok::<Vec<Result<R, CudaFftError>>, CudaFftError>(results)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut all_results = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Ok(results)) => all_results.extend(results),
                Ok(Err(e)) => all_results.push(Err(e)),
                Err(_) => all_results.push(Err(CudaFftError::KernelExecution("Thread panicked".into()))),
            }
        }
        
        all_results
    }
}

#[cfg(test)]
#[cfg(feature = "cuda-runtime")]
mod tests {
    use super::*;
    
    #[test]
    fn test_multi_gpu_pool_creation() {
        // This test will only pass on a system with CUDA GPUs
        if let Ok(pool) = MultiGpuExecutorPool::new_all_gpus() {
            assert!(pool.gpu_count() > 0);
            println!("Found {} GPUs", pool.gpu_count());
        }
    }
}

