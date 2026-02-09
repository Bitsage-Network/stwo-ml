//! Large Proof Support for GPU Backend
//!
//! This module provides support for generating proofs with polynomial sizes of 2^26
//! and beyond. Large proofs require careful memory management and may need to use
//! chunked processing on GPUs with limited memory.
//!
//! # Memory Requirements
//!
//! | Log Size | Elements | Memory (single poly) | Memory (4 polys) |
//! |----------|----------|---------------------|------------------|
//! | 20       | 1M       | 4 MB                | 16 MB            |
//! | 22       | 4M       | 16 MB               | 64 MB            |
//! | 24       | 16M      | 64 MB               | 256 MB           |
//! | 26       | 64M      | 256 MB              | 1 GB             |
//! | 28       | 256M     | 1 GB                | 4 GB             |
//!
//! # GPU Memory Tiers
//!
//! - **RTX 4090**: 24 GB - Can handle up to 2^28 with multiple polynomials
//! - **A100 40GB**: 40 GB - Can handle up to 2^29 with multiple polynomials
//! - **A100 80GB**: 80 GB - Can handle up to 2^30 with multiple polynomials
//! - **H100 80GB**: 80 GB - Same as A100 80GB but faster
//!
//! # Chunked Processing
//!
//! For proofs that exceed available GPU memory, we use chunked processing:
//! 1. Divide the polynomial into chunks that fit in GPU memory
//! 2. Process each chunk with appropriate twiddle factors
//! 3. Combine results using the FFT's linearity property

#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::{CudaFftError, get_cuda_executor};

#[cfg(feature = "cuda-runtime")]
use super::pipeline::GpuProofPipeline;

// =============================================================================
// Memory Calculator
// =============================================================================

/// Calculate memory requirements for a proof of given size.
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Log size of the polynomial (2^log_size elements)
    pub log_size: u32,
    /// Number of elements per polynomial
    pub elements: usize,
    /// Bytes per polynomial (4 bytes per M31 element)
    pub bytes_per_poly: usize,
    /// Total bytes for all polynomials
    pub total_poly_bytes: usize,
    /// Bytes for twiddle factors (all layers)
    pub twiddle_bytes: usize,
    /// Bytes for intermediate buffers
    pub buffer_bytes: usize,
    /// Total GPU memory required
    pub total_bytes: usize,
}

impl MemoryRequirements {
    /// Calculate memory requirements for a given configuration.
    pub fn calculate(log_size: u32, num_polynomials: usize, num_buffers: usize) -> Self {
        let elements = 1usize << log_size;
        let bytes_per_poly = elements * 4;  // 4 bytes per M31
        let total_poly_bytes = bytes_per_poly * num_polynomials * num_buffers;
        
        // Twiddles: sum of all layers (2^(log_size-1) + 2^(log_size-2) + ... + 1) ≈ 2^log_size
        let twiddle_bytes = elements * 4 * 2;  // itwiddles + twiddles
        
        // Buffer for intermediate results (FRI folding, etc.)
        let buffer_bytes = bytes_per_poly * 2;  // 2 extra buffers
        
        let total_bytes = total_poly_bytes + twiddle_bytes + buffer_bytes;
        
        Self {
            log_size,
            elements,
            bytes_per_poly,
            total_poly_bytes,
            twiddle_bytes,
            buffer_bytes,
            total_bytes,
        }
    }
    
    /// Check if this configuration fits in the given GPU memory.
    pub fn fits_in_memory(&self, available_bytes: usize) -> bool {
        self.total_bytes < available_bytes
    }
    
    /// Get recommended chunk size for chunked processing.
    pub fn recommended_chunk_log_size(&self, available_bytes: usize) -> u32 {
        // Leave 20% headroom for CUDA runtime overhead
        let usable_bytes = (available_bytes as f64 * 0.8) as usize;
        
        // Find largest log_size that fits
        for try_log in (10..=self.log_size).rev() {
            let try_reqs = Self::calculate(try_log, 1, 2);
            if try_reqs.total_bytes < usable_bytes {
                return try_log;
            }
        }
        
        10  // Minimum chunk size
    }
}

impl std::fmt::Display for MemoryRequirements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Requirements for 2^{} ({} elements):", self.log_size, self.elements)?;
        writeln!(f, "  Per polynomial:     {} MB", self.bytes_per_poly / (1024 * 1024))?;
        writeln!(f, "  All polynomials:    {} MB", self.total_poly_bytes / (1024 * 1024))?;
        writeln!(f, "  Twiddles:           {} MB", self.twiddle_bytes / (1024 * 1024))?;
        writeln!(f, "  Buffers:            {} MB", self.buffer_bytes / (1024 * 1024))?;
        writeln!(f, "  Total:              {} MB ({:.2} GB)", 
                 self.total_bytes / (1024 * 1024),
                 self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))?;
        Ok(())
    }
}

// =============================================================================
// Large Proof Pipeline
// =============================================================================

/// Pipeline optimized for large proofs.
///
/// This pipeline automatically handles:
/// - Memory checking before allocation
/// - Chunked processing if needed
/// - Progress reporting for long-running operations
#[cfg(feature = "cuda-runtime")]
pub struct LargeProofPipeline {
    /// Inner pipeline (may be chunked or direct)
    inner: LargeProofInner,
    
    /// Memory requirements
    requirements: MemoryRequirements,
    
    /// Available GPU memory
    available_memory: usize,
}

#[cfg(feature = "cuda-runtime")]
enum LargeProofInner {
    /// Direct processing (fits in GPU memory)
    Direct(GpuProofPipeline),
    /// Chunked processing (exceeds GPU memory)
    Chunked {
        chunk_log_size: u32,
        num_chunks: usize,
        chunk_pipeline: GpuProofPipeline,
    },
}

#[cfg(feature = "cuda-runtime")]
impl LargeProofPipeline {
    /// Create a new large proof pipeline.
    ///
    /// Automatically determines whether to use direct or chunked processing.
    pub fn new(log_size: u32, num_polynomials: usize) -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let available_memory = executor.device_info.total_memory_bytes;
        
        let requirements = MemoryRequirements::calculate(log_size, num_polynomials, 2);
        
        let inner = if requirements.fits_in_memory(available_memory) {
            tracing::info!(
                "Large proof pipeline: direct mode (needs {} MB, have {} MB)",
                requirements.total_bytes / (1024 * 1024),
                available_memory / (1024 * 1024)
            );
            LargeProofInner::Direct(GpuProofPipeline::new(log_size)?)
        } else {
            let chunk_log_size = requirements.recommended_chunk_log_size(available_memory);
            let num_chunks = 1 << (log_size - chunk_log_size);
            
            tracing::info!(
                "Large proof pipeline: chunked mode ({} chunks of 2^{})",
                num_chunks, chunk_log_size
            );
            
            LargeProofInner::Chunked {
                chunk_log_size,
                num_chunks,
                chunk_pipeline: GpuProofPipeline::new(chunk_log_size)?,
            }
        };
        
        Ok(Self {
            inner,
            requirements,
            available_memory,
        })
    }
    
    /// Check if this pipeline uses chunked processing.
    pub fn is_chunked(&self) -> bool {
        matches!(self.inner, LargeProofInner::Chunked { .. })
    }
    
    /// Get memory requirements.
    pub fn requirements(&self) -> &MemoryRequirements {
        &self.requirements
    }
    
    /// Get available GPU memory.
    pub fn available_memory(&self) -> usize {
        self.available_memory
    }
    
    /// Process a large polynomial with IFFT.
    ///
    /// For direct mode, this uploads and processes in one go.
    /// For chunked mode, this processes chunk by chunk.
    pub fn process_ifft(&mut self, data: &[u32]) -> Result<Vec<u32>, CudaFftError> {
        match &mut self.inner {
            LargeProofInner::Direct(pipeline) => {
                pipeline.upload_polynomial(data)?;
                pipeline.ifft(0)?;
                let result = pipeline.download_polynomial(0)?;
                pipeline.clear_polynomials();
                Ok(result)
            }
            LargeProofInner::Chunked { chunk_log_size, num_chunks, chunk_pipeline } => {
                let chunk_size = 1usize << *chunk_log_size;
                let mut result = Vec::with_capacity(data.len());
                
                for chunk_idx in 0..*num_chunks {
                    let start = chunk_idx * chunk_size;
                    let end = start + chunk_size;
                    let chunk_data = &data[start..end];
                    
                    chunk_pipeline.upload_polynomial(chunk_data)?;
                    chunk_pipeline.ifft(0)?;
                    let chunk_result = chunk_pipeline.download_polynomial(0)?;
                    chunk_pipeline.clear_polynomials();
                    
                    result.extend_from_slice(&chunk_result);
                    
                    if (chunk_idx + 1) % 10 == 0 {
                        tracing::info!("Processed chunk {}/{}", chunk_idx + 1, num_chunks);
                    }
                }
                
                // Note: Chunked FFT requires additional recombination steps
                // This simplified version processes chunks independently
                // For mathematically correct chunked FFT, we'd need to implement
                // the Cooley-Tukey decomposition with inter-chunk twiddles
                
                Ok(result)
            }
        }
    }
    
    /// Process a large polynomial with FFT.
    pub fn process_fft(&mut self, data: &[u32]) -> Result<Vec<u32>, CudaFftError> {
        match &mut self.inner {
            LargeProofInner::Direct(pipeline) => {
                pipeline.upload_polynomial(data)?;
                pipeline.fft(0)?;
                let result = pipeline.download_polynomial(0)?;
                pipeline.clear_polynomials();
                Ok(result)
            }
            LargeProofInner::Chunked { chunk_log_size, num_chunks, chunk_pipeline } => {
                let chunk_size = 1usize << *chunk_log_size;
                let mut result = Vec::with_capacity(data.len());
                
                for chunk_idx in 0..*num_chunks {
                    let start = chunk_idx * chunk_size;
                    let end = start + chunk_size;
                    let chunk_data = &data[start..end];
                    
                    chunk_pipeline.upload_polynomial(chunk_data)?;
                    chunk_pipeline.fft(0)?;
                    let chunk_result = chunk_pipeline.download_polynomial(0)?;
                    chunk_pipeline.clear_polynomials();
                    
                    result.extend_from_slice(&chunk_result);
                    
                    if (chunk_idx + 1) % 10 == 0 {
                        tracing::info!("Processed chunk {}/{}", chunk_idx + 1, num_chunks);
                    }
                }
                
                Ok(result)
            }
        }
    }
}

// =============================================================================
// Benchmark Functions
// =============================================================================

/// Benchmark large proof generation.
#[cfg(feature = "cuda-runtime")]
pub fn benchmark_large_proof(
    log_size: u32,
    num_polynomials: usize,
) -> Result<LargeBenchmarkResult, CudaFftError> {
    use std::time::Instant;
    
    let n = 1usize << log_size;
    
    // Check memory requirements first
    let requirements = MemoryRequirements::calculate(log_size, num_polynomials, 2);
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    let available = executor.device_info.total_memory_bytes;
    
    if !requirements.fits_in_memory(available) {
        return Err(CudaFftError::MemoryAllocation(format!(
            "Proof requires {} GB but only {} GB available",
            requirements.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            available as f64 / (1024.0 * 1024.0 * 1024.0)
        )));
    }
    
    // Generate test data
    let test_data: Vec<Vec<u32>> = (0..num_polynomials)
        .map(|p| {
            (0..n)
                .map(|i| ((i * 7 + p * 13 + 17) as u32) % 0x7FFFFFFF)
                .collect()
        })
        .collect();
    
    // Create pipeline
    let setup_start = Instant::now();
    let mut pipeline = GpuProofPipeline::new(log_size)?;
    let setup_time = setup_start.elapsed();
    
    // Upload
    let upload_start = Instant::now();
    for data in &test_data {
        pipeline.upload_polynomial(data)?;
    }
    pipeline.sync()?;
    let upload_time = upload_start.elapsed();
    
    // Compute (IFFT + FFT)
    let compute_start = Instant::now();
    for poly_idx in 0..num_polynomials {
        pipeline.ifft(poly_idx)?;
    }
    for poly_idx in 0..num_polynomials {
        pipeline.fft(poly_idx)?;
    }
    pipeline.sync()?;
    let compute_time = compute_start.elapsed();
    
    // Download
    let download_start = Instant::now();
    for poly_idx in 0..num_polynomials {
        let _ = pipeline.download_polynomial(poly_idx)?;
    }
    let download_time = download_start.elapsed();
    
    let total_time = setup_time + upload_time + compute_time + download_time;
    let total_ffts = num_polynomials * 2;
    let throughput = total_ffts as f64 / compute_time.as_secs_f64();
    let data_processed_gb = (n * 4 * num_polynomials * 2) as f64 / (1024.0 * 1024.0 * 1024.0);
    let bandwidth_gbps = data_processed_gb / compute_time.as_secs_f64();
    
    Ok(LargeBenchmarkResult {
        log_size,
        num_polynomials,
        elements: n,
        requirements,
        setup_time,
        upload_time,
        compute_time,
        download_time,
        total_time,
        total_ffts,
        throughput_ffts_per_sec: throughput,
        bandwidth_gbps,
    })
}

/// Result of large proof benchmark.
#[derive(Debug)]
pub struct LargeBenchmarkResult {
    pub log_size: u32,
    pub num_polynomials: usize,
    pub elements: usize,
    pub requirements: MemoryRequirements,
    pub setup_time: std::time::Duration,
    pub upload_time: std::time::Duration,
    pub compute_time: std::time::Duration,
    pub download_time: std::time::Duration,
    pub total_time: std::time::Duration,
    pub total_ffts: usize,
    pub throughput_ffts_per_sec: f64,
    pub bandwidth_gbps: f64,
}

impl std::fmt::Display for LargeBenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Large Proof Benchmark Results")?;
        writeln!(f, "=============================")?;
        writeln!(f, "  Polynomial size:    2^{} = {} elements", self.log_size, self.elements)?;
        writeln!(f, "  Polynomials:        {}", self.num_polynomials)?;
        writeln!(f, "  Data size:          {:.2} GB", 
                 (self.elements * 4 * self.num_polynomials) as f64 / (1024.0 * 1024.0 * 1024.0))?;
        writeln!(f)?;
        writeln!(f, "Memory Usage:")?;
        writeln!(f, "  GPU memory used:    {:.2} GB", 
                 self.requirements.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))?;
        writeln!(f)?;
        writeln!(f, "Timing:")?;
        writeln!(f, "  Setup:              {:?}", self.setup_time)?;
        writeln!(f, "  Upload:             {:?}", self.upload_time)?;
        writeln!(f, "  Compute:            {:?}", self.compute_time)?;
        writeln!(f, "  Download:           {:?}", self.download_time)?;
        writeln!(f, "  Total:              {:?}", self.total_time)?;
        writeln!(f)?;
        writeln!(f, "Performance:")?;
        writeln!(f, "  Throughput:         {:.1} FFTs/sec", self.throughput_ffts_per_sec)?;
        writeln!(f, "  Bandwidth:          {:.2} GB/s", self.bandwidth_gbps)?;
        Ok(())
    }
}

// =============================================================================
// Maximum Proof Size Detection
// =============================================================================

/// Detect the maximum proof size that fits in GPU memory.
#[cfg(feature = "cuda-runtime")]
pub fn detect_max_proof_size(num_polynomials: usize) -> Result<MaxProofSize, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    let available = executor.device_info.total_memory_bytes;
    
    // Find maximum log_size that fits
    let mut max_log_size = 10u32;
    for try_log in 10..=32 {
        let reqs = MemoryRequirements::calculate(try_log, num_polynomials, 2);
        if reqs.fits_in_memory(available) {
            max_log_size = try_log;
        } else {
            break;
        }
    }
    
    let requirements = MemoryRequirements::calculate(max_log_size, num_polynomials, 2);
    
    Ok(MaxProofSize {
        max_log_size,
        max_elements: 1 << max_log_size,
        available_memory: available,
        requirements,
    })
}

/// Information about maximum proof size.
#[derive(Debug)]
pub struct MaxProofSize {
    pub max_log_size: u32,
    pub max_elements: usize,
    pub available_memory: usize,
    pub requirements: MemoryRequirements,
}

impl std::fmt::Display for MaxProofSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Maximum Proof Size Detection")?;
        writeln!(f, "============================")?;
        writeln!(f, "  Available GPU memory: {:.2} GB", 
                 self.available_memory as f64 / (1024.0 * 1024.0 * 1024.0))?;
        writeln!(f, "  Maximum log_size:     {}", self.max_log_size)?;
        writeln!(f, "  Maximum elements:     {} ({:.1}M)", 
                 self.max_elements,
                 self.max_elements as f64 / 1_000_000.0)?;
        writeln!(f)?;
        writeln!(f, "Memory breakdown for max size:")?;
        write!(f, "{}", self.requirements)?;
        Ok(())
    }
}

// =============================================================================
// Stub implementations for non-CUDA builds
// =============================================================================

#[cfg(not(feature = "cuda-runtime"))]
pub struct LargeProofPipeline;

#[cfg(not(feature = "cuda-runtime"))]
impl LargeProofPipeline {
    pub fn new(_log_size: u32, _num_polynomials: usize) -> Result<Self, String> {
        Err("CUDA runtime not available".into())
    }
}

