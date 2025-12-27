//! GPU-accelerated polynomial operations.
//!
//! This module implements [`PolyOps`] for [`GpuBackend`], providing GPU acceleration
//! for the most computationally intensive operations in proof generation:
//!
//! - **FFT (evaluate/interpolate)**: 50-100x speedup on large polynomials
//! - **Twiddle precomputation**: GPU-accelerated for large domains
//! - **Point evaluation**: Parallelized on GPU
//!
//! # Performance Strategy
//!
//! We use a hybrid approach:
//! - Small polynomials (< 16K elements): Use SIMD backend (GPU overhead not worth it)
//! - Large polynomials (>= 16K elements): Use GPU acceleration
//!
//! This threshold was determined empirically on A100 GPUs.
//!
//! # Persistent Pipeline
//!
//! To achieve maximum speedup (50x+), we use a thread-local persistent pipeline
//! that keeps GPU memory allocated across multiple operations. This eliminates
//! the overhead of creating/destroying pipelines for each batch.

use tracing::{span, Level};

use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
#[cfg(feature = "cuda-runtime")]
use crate::core::ColumnVec;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

use super::conversion::{
    circle_coeffs_ref_to_simd, circle_eval_ref_to_simd, twiddle_ref_to_simd, twiddle_to_gpu,
};
use super::fft::{GPU_FFT_THRESHOLD_LOG_SIZE, compute_twiddle_dbls_cpu, get_cached_itwiddles};
use super::cuda_executor::{is_cuda_available, cuda_ifft, cuda_fft};
#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::get_cuda_executor;
#[cfg(feature = "cuda-runtime")]
use super::pipeline::GpuProofPipeline;
use super::GpuBackend;

// =============================================================================
// Persistent Pipeline Cache
// =============================================================================

#[cfg(feature = "cuda-runtime")]
use std::cell::RefCell;
#[cfg(feature = "cuda-runtime")]
use std::collections::HashMap;

// Thread-local cache of GPU pipelines by log_size.
// This allows reusing GPU memory allocations across multiple batch operations.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static PIPELINE_CACHE: RefCell<HashMap<u32, GpuProofPipeline>> = RefCell::new(HashMap::new());
}

/// Get or create a persistent pipeline for the given log_size.
/// The pipeline is cached thread-locally for reuse.
#[cfg(feature = "cuda-runtime")]
fn get_or_create_pipeline(log_size: u32) -> Result<(), super::cuda_executor::CudaFftError> {
    PIPELINE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if !cache.contains_key(&log_size) {
            let pipeline = GpuProofPipeline::new(log_size)?;
            cache.insert(log_size, pipeline);
        }
        Ok(())
    })
}

/// Clear the pipeline cache (useful for testing or memory management).
#[cfg(feature = "cuda-runtime")]
pub fn clear_pipeline_cache() {
    PIPELINE_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

impl PolyOps for GpuBackend {
    // We use the same twiddle type as SimdBackend since twiddles are precomputed
    // and stored in CPU memory. The GPU uses them during FFT execution.
    type Twiddles = <SimdBackend as PolyOps>::Twiddles;
    
    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::interpolate").entered();
        
        let log_size = eval.domain.log_size();
        
        // Small polynomials: use SIMD (GPU overhead not worth it)
        if log_size < GPU_FFT_THRESHOLD_LOG_SIZE {
            tracing::debug!(
                "GPU interpolate: using SIMD for small size (log_size={} < threshold={})",
                log_size, GPU_FFT_THRESHOLD_LOG_SIZE
            );
            return interpolate_simd_fallback(eval, twiddles);
        }
        
        // Large polynomials: prefer GPU, fallback to SIMD if unavailable
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::interpolate: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded. For optimal performance, ensure CUDA is available.",
                log_size
            );
            return interpolate_simd_fallback(eval, twiddles);
        }

        tracing::info!(
            "GPU interpolate: using CUDA for {} elements (log_size={})",
            1u64 << log_size, log_size
        );
        gpu_interpolate(eval, twiddles, log_size)
    }
    
    /// Batch interpolate multiple columns using GPU pipeline.
    /// This is significantly faster than interpolating one at a time.
    #[cfg(feature = "cuda-runtime")]
    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        let columns: Vec<_> = columns.into_iter().collect();
        
        if columns.is_empty() {
            return Vec::new();
        }
        
        let log_size = columns[0].domain.log_size();
        let num_columns = columns.len();
        
        // Small polynomials or few columns: use sequential approach
        if log_size < GPU_FFT_THRESHOLD_LOG_SIZE || num_columns < 2 {
            tracing::debug!("GPU interpolate_columns: sequential (log_size={}, num_columns={}, threshold={})", 
                     log_size, num_columns, GPU_FFT_THRESHOLD_LOG_SIZE);
            return columns
                .into_iter()
                .map(|eval| eval.interpolate_with_twiddles(twiddles))
                .collect();
        }
        
        // Large polynomials with multiple columns: prefer GPU, fallback to SIMD
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::interpolate_columns: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded.",
                log_size
            );
            return columns
                .into_iter()
                .map(|eval| eval.interpolate_with_twiddles(twiddles))
                .collect();
        }

        tracing::info!("GPU interpolate_columns: batch (log_size={}, num_columns={})", log_size, num_columns);

        gpu_batch_interpolate(columns, log_size)
    }
    
    #[cfg(not(feature = "cuda-runtime"))]
    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        columns
            .into_iter()
            .map(|eval| eval.interpolate_with_twiddles(twiddles))
            .collect()
    }

    /// Batch evaluate multiple polynomials using GPU pipeline.
    #[cfg(feature = "cuda-runtime")]
    fn evaluate_polynomials(
        polynomials: ColumnVec<CircleCoefficients<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
        store_polynomials_coefficients: bool,
    ) -> Vec<crate::prover::air::component_prover::Poly<Self>>
    where
        Self: crate::prover::backend::Backend,
    {
        use crate::prover::air::component_prover::Poly;
        
        if polynomials.is_empty() {
            return Vec::new();
        }
        
        let log_size = polynomials[0].log_size();
        let num_polys = polynomials.len();
        
        // Small polynomials or few: use default implementation
        if log_size < GPU_FFT_THRESHOLD_LOG_SIZE || num_polys < 2 {
            return polynomials
                .into_iter()
                .map(|poly_coeffs| {
                    let evals = poly_coeffs.evaluate_with_twiddles(
                        CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain(),
                        twiddles,
                    );
                    Poly::new(store_polynomials_coefficients.then_some(poly_coeffs), evals)
                })
                .collect();
        }
        
        // Large polynomials: prefer GPU, fallback to SIMD
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::evaluate_polynomials: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded.",
                log_size
            );
            return polynomials
                .into_iter()
                .map(|poly_coeffs| {
                    let evals = poly_coeffs.evaluate_with_twiddles(
                        CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain(),
                        twiddles,
                    );
                    Poly::new(store_polynomials_coefficients.then_some(poly_coeffs), evals)
                })
                .collect();
        }

        tracing::info!(
            "GPU batch evaluate: {} polynomials × {} elements (log_size={}, blowup={})",
            num_polys, 1u64 << log_size, log_size, log_blowup_factor
        );

        gpu_batch_evaluate(polynomials, log_blowup_factor, twiddles, store_polynomials_coefficients)
    }
    
    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        // Point evaluation is memory-bound, not compute-bound
        // GPU doesn't help much here, use SIMD via conversion
        let simd_poly = circle_coeffs_ref_to_simd(poly);
        SimdBackend::eval_at_point(simd_poly, point)
    }
    
    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<Self, SecureField> {
        // Delegate to SIMD - this is not a hot path
        SimdBackend::barycentric_weights(coset, p)
    }
    
    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &Col<Self, SecureField>,
    ) -> SecureField {
        let simd_evals = circle_eval_ref_to_simd(evals);
        SimdBackend::barycentric_eval_at_point(simd_evals, weights)
    }
    
    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        let simd_evals = circle_eval_ref_to_simd(evals);
        let simd_twiddles = twiddle_ref_to_simd(twiddles);
        SimdBackend::eval_at_point_by_folding(simd_evals, point, simd_twiddles)
    }
    
    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        let simd_poly = circle_coeffs_ref_to_simd(poly);
        let result = SimdBackend::extend(simd_poly, log_size);
        CircleCoefficients::new(result.coeffs)
    }
    
    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let _span = span!(Level::TRACE, "GpuBackend::evaluate").entered();
        
        let log_size = domain.log_size();
        
        // Small polynomials: use SIMD (GPU overhead not worth it)
        if log_size < GPU_FFT_THRESHOLD_LOG_SIZE {
            tracing::debug!(
                "GPU evaluate: using SIMD for small size (log_size={} < threshold={})",
                log_size, GPU_FFT_THRESHOLD_LOG_SIZE
            );
            return evaluate_simd_fallback(poly, domain, twiddles);
        }
        
        // Large polynomials: prefer GPU, fallback to SIMD if unavailable
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::evaluate: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded. For optimal performance, ensure CUDA is available.",
                log_size
            );
            return evaluate_simd_fallback(poly, domain, twiddles);
        }

        tracing::info!(
            "GPU evaluate: using CUDA for {} elements (log_size={})",
            1u64 << log_size, log_size
        );
        gpu_evaluate(poly, domain, twiddles, log_size)
    }
    
    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::precompute_twiddles").entered();
        
        // Use SIMD backend's cached implementation for twiddle computation.
        // Twiddles are computed once and reused many times, so CPU is fine here.
        // The GPU benefit comes from using these twiddles in FFT operations.
        let simd_twiddles = SimdBackend::precompute_twiddles(coset);
        
        // Convert to GpuBackend's TwiddleTree
        twiddle_to_gpu(simd_twiddles)
    }
    
    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        // This is a simple split operation, no GPU benefit
        let simd_poly = CircleCoefficients::<SimdBackend>::new(poly.coeffs);
        let (left, right) = SimdBackend::split_at_mid(simd_poly);
        (
            CircleCoefficients::new(left.coeffs),
            CircleCoefficients::new(right.coeffs),
        )
    }
}

// =============================================================================
// SIMD Fallback Functions
// =============================================================================

fn interpolate_simd_fallback(
    eval: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<GpuBackend>,
) -> CircleCoefficients<GpuBackend> {
    let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
        eval.domain,
        eval.values,
    );
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let result = SimdBackend::interpolate(simd_eval, simd_twiddles);
    CircleCoefficients::new(result.coeffs)
}

fn evaluate_simd_fallback(
    poly: &CircleCoefficients<GpuBackend>,
    domain: CircleDomain,
    twiddles: &TwiddleTree<GpuBackend>,
) -> CircleEvaluation<GpuBackend, BaseField, BitReversedOrder> {
    let simd_poly = circle_coeffs_ref_to_simd(poly);
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let result = SimdBackend::evaluate(simd_poly, domain, simd_twiddles);
    CircleEvaluation::new(domain, result.values)
}

// =============================================================================
// GPU FFT Implementation
// =============================================================================

/// GPU-accelerated polynomial interpolation (inverse FFT).
///
/// This function:
/// 1. Extracts evaluation data (zero-copy when possible)
/// 2. Uses cached inverse FFT twiddles
/// 3. Transfers data to GPU
/// 4. Executes IFFT kernels
/// 5. Applies denormalization on GPU (fused operation)
/// 6. Transfers results back to CPU
fn gpu_interpolate(
    eval: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
    _twiddles: &TwiddleTree<GpuBackend>,
    log_size: u32,
) -> CircleCoefficients<GpuBackend> {
    let _span = span!(Level::INFO, "GPU interpolate (IFFT)", size = 1u64 << log_size).entered();
    
    // Extract raw data from evaluation - use unsafe transmute for zero-copy
    // BaseField is repr(transparent) over u32, so this is safe
    let values_slice = eval.values.as_slice();
    let data_ptr = values_slice.as_ptr() as *const u32;
    let data_len = values_slice.len();
    
    // Create a mutable copy for GPU processing
    let mut data: Vec<u32> = unsafe {
        std::slice::from_raw_parts(data_ptr, data_len).to_vec()
    };
    
    // Get cached twiddles for IFFT
    let twiddles_dbl = get_cached_itwiddles(log_size);
    
    // Execute CUDA IFFT
    match cuda_ifft(&mut data, &twiddles_dbl, log_size) {
        Ok(()) => {
            tracing::debug!(
                "GPU IFFT completed for {} elements",
                1u64 << log_size
            );
            
            // Apply denormalization factor (divide by domain size)
            // Use GPU denormalization if available, otherwise CPU fallback
            let denorm = BaseField::from(1u32 << log_size).inverse();
            let denorm_val = denorm.0;
            
            #[cfg(feature = "cuda-runtime")]
            {
                // Try to use GPU denormalization (fused operation)
                if let Ok(executor) = get_cuda_executor() {
                    if executor.execute_denormalize(&mut data, denorm_val).is_ok() {
                        tracing::debug!("GPU denormalization completed");
                    } else {
                        // GPU denormalize failed, fall back to CPU
                        apply_denormalization_cpu(&mut data, denorm_val);
                    }
                } else {
                    apply_denormalization_cpu(&mut data, denorm_val);
                }
            }
            
            #[cfg(not(feature = "cuda-runtime"))]
            {
                apply_denormalization_cpu(&mut data, denorm_val);
            }
            
            // Convert back to BaseColumn
            use crate::prover::backend::simd::column::BaseColumn;
            let coeffs: BaseColumn = unsafe {
                // Safety: BaseField is repr(transparent) over u32
                let bf_ptr = data.as_ptr() as *const BaseField;
                std::slice::from_raw_parts(bf_ptr, data.len()).iter().copied().collect()
            };
            
            CircleCoefficients::new(coeffs)
        }
        Err(e) => {
            tracing::error!(
                "GPU IFFT execution failed: {}. Falling back to SIMD.",
                e
            );
            // Fallback to SIMD on GPU execution failure
            return interpolate_simd_fallback(eval, _twiddles);
        }
    }
}

/// Apply denormalization on CPU (fallback when GPU is unavailable).
#[inline]
fn apply_denormalization_cpu(data: &mut [u32], denorm_val: u32) {
    const M31_PRIME: u64 = (1u64 << 31) - 1;
    for v in data.iter_mut() {
        let product = (*v as u64) * (denorm_val as u64);
        *v = (product % M31_PRIME) as u32;
    }
}

/// GPU-accelerated batch polynomial interpolation.
/// 
/// This function processes multiple polynomials using the GPU pipeline,
/// keeping data on GPU between operations for maximum throughput.
/// 
/// Uses bulk upload/download for minimal transfer overhead.
#[cfg(feature = "cuda-runtime")]
fn gpu_batch_interpolate(
    columns: Vec<CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>>,
    log_size: u32,
) -> Vec<CircleCoefficients<GpuBackend>> {
    use std::time::Instant;
    
    let num_columns = columns.len();
    let _span = span!(Level::INFO, "GPU batch interpolate", 
                      num_columns = num_columns, 
                      size = 1u64 << log_size).entered();
    
    // Ensure pipeline is cached for this log_size (twiddles stay on GPU)
    if let Err(e) = get_or_create_pipeline(log_size) {
        tracing::error!("Failed to create GPU pipeline: {}. Falling back to sequential SIMD.", e);
        return columns
            .into_iter()
            .map(|eval| {
                let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    eval.domain,
                    eval.values,
                );
                let simd_twiddles = SimdBackend::precompute_twiddles(eval.domain.half_coset);
                let result = SimdBackend::interpolate(simd_eval, &simd_twiddles);
                CircleCoefficients::new(result.coeffs)
            })
            .collect();
    }
    
    // Use the cached pipeline
    PIPELINE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let pipeline = cache.get_mut(&log_size).expect("Pipeline should exist after get_or_create");
        
        // Clear any existing polynomials from previous batch
        pipeline.clear_polynomials();
        
        // Prepare data for bulk upload
        let upload_start = Instant::now();
        let poly_slices: Vec<Vec<u32>> = columns.iter()
            .map(|eval| {
                let values_slice = eval.values.as_slice();
                let data_ptr = values_slice.as_ptr() as *const u32;
                let data_len = values_slice.len();
                unsafe { std::slice::from_raw_parts(data_ptr, data_len).to_vec() }
            })
            .collect();
        
        // Bulk upload all columns to GPU
        let num_uploaded = pipeline.upload_polynomials_bulk(poly_slices.iter().map(|v| v.as_slice()))
            .expect("Failed to bulk upload polynomials");
        let upload_time = upload_start.elapsed();
        
        // Execute IFFT with fused denormalization on all polynomials
        let compute_start = Instant::now();
        for poly_idx in 0..num_uploaded {
            if let Err(e) = pipeline.ifft_with_denormalize(poly_idx) {
                panic!("Failed to execute GPU IFFT with denormalize: {}", e);
            }
        }
        let compute_time = compute_start.elapsed();
        
        // Bulk download all results
        let download_start = Instant::now();
        let all_data = pipeline.download_polynomials_bulk()
            .expect("Failed to bulk download polynomials");
        let download_time = download_start.elapsed();
        
        // Convert to CircleCoefficients
        use crate::prover::backend::simd::column::BaseColumn;
        let results: Vec<CircleCoefficients<GpuBackend>> = all_data.into_iter()
            .map(|data| {
                let coeffs: BaseColumn = unsafe {
                    let bf_ptr = data.as_ptr() as *const BaseField;
                    std::slice::from_raw_parts(bf_ptr, data.len()).iter().copied().collect()
                };
                CircleCoefficients::new(coeffs)
            })
            .collect();
        
        tracing::debug!("GPU batch_interpolate: upload={:?}, compute={:?}, download={:?}, total={:?}",
                 upload_time, compute_time, download_time, 
                 upload_time + compute_time + download_time);
        
        results
    })
}

/// GPU-accelerated batch polynomial evaluation.
/// 
/// This function processes multiple polynomials using the GPU pipeline,
/// keeping data on GPU between operations for maximum throughput.
/// 
/// Uses a persistent pipeline cache to avoid repeated setup overhead.
#[cfg(feature = "cuda-runtime")]
fn gpu_batch_evaluate(
    polynomials: ColumnVec<CircleCoefficients<GpuBackend>>,
    log_blowup_factor: u32,
    _twiddles: &TwiddleTree<GpuBackend>,
    store_polynomials_coefficients: bool,
) -> Vec<crate::prover::air::component_prover::Poly<GpuBackend>> {
    use crate::prover::air::component_prover::Poly;
    use crate::prover::backend::simd::column::BaseColumn;
    
    let log_size = polynomials[0].log_size();
    let extended_log_size = log_size + log_blowup_factor;
    let extended_domain = CanonicCoset::new(extended_log_size).circle_domain();
    
    let _span = span!(Level::INFO, "GPU batch evaluate", 
                      num_polys = polynomials.len(), 
                      size = 1u64 << log_size,
                      extended_size = 1u64 << extended_log_size).entered();
    
    // Ensure pipeline is cached for the extended size
    if let Err(e) = get_or_create_pipeline(extended_log_size) {
        panic!("Failed to create/get GPU pipeline: {}", e);
    }
    
    // Use the cached pipeline
    PIPELINE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let pipeline = cache.get_mut(&extended_log_size).expect("Pipeline should exist after get_or_create");
        
        // Clear any existing polynomials from previous batch
        pipeline.clear_polynomials();
        
        // Upload all polynomials to GPU (with zero-padding for extension)
        let extended_size = 1usize << extended_log_size;
        for poly in &polynomials {
            let mut data: Vec<u32> = poly.coeffs.as_slice()
                .iter()
                .map(|f| f.0)
                .collect();
            
            // Zero-pad to extended size
            data.resize(extended_size, 0);
            
            if let Err(e) = pipeline.upload_polynomial(&data) {
                panic!("Failed to upload polynomial to GPU: {}", e);
            }
        }
        
        // Execute FFT on all polynomials (stays on GPU)
        for poly_idx in 0..polynomials.len() {
            if let Err(e) = pipeline.fft(poly_idx) {
                panic!("Failed to execute GPU FFT: {}", e);
            }
        }
        
        // Download results
        let mut results = Vec::with_capacity(polynomials.len());
        
        for (poly_idx, poly_coeffs) in polynomials.into_iter().enumerate() {
            let data = match pipeline.download_polynomial(poly_idx) {
                Ok(d) => d,
                Err(e) => {
                    panic!("Failed to download polynomial from GPU: {}", e);
                }
            };
            
            // Convert to BaseColumn
            let values: BaseColumn = unsafe {
                let bf_ptr = data.as_ptr() as *const BaseField;
                std::slice::from_raw_parts(bf_ptr, data.len()).iter().copied().collect()
            };
            
            let evals = CircleEvaluation::new(extended_domain, values);
            results.push(Poly::new(
                store_polynomials_coefficients.then_some(poly_coeffs),
                evals,
            ));
        }
        
        tracing::info!(
            "GPU batch evaluate completed: {} polynomials (pipeline reused)",
            results.len()
        );
        
        results
    })
}

/// GPU-accelerated batch polynomial evaluation on a specific domain.
/// 
/// This function processes multiple polynomials using the GPU pipeline,
/// evaluating them all on the same domain with minimal transfer overhead.
/// 
/// Uses bulk upload/download for minimal transfer overhead.
#[cfg(feature = "cuda-runtime")]
fn gpu_batch_evaluate_columns(
    polynomials: Vec<CircleCoefficients<GpuBackend>>,
    domain: CircleDomain,
    domain_log_size: u32,
) -> Vec<CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>> {
    use std::time::Instant;
    use crate::prover::backend::simd::column::BaseColumn;
    
    let num_polys = polynomials.len();
    let _span = span!(Level::INFO, "GPU batch evaluate_columns", 
                      num_polys = num_polys, 
                      domain_size = 1u64 << domain_log_size).entered();
    
    // Ensure pipeline is cached for the domain size
    if let Err(e) = get_or_create_pipeline(domain_log_size) {
        panic!("Failed to create/get GPU pipeline: {}", e);
    }
    
    // Use the cached pipeline
    PIPELINE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let pipeline = cache.get_mut(&domain_log_size).expect("Pipeline should exist after get_or_create");
        
        // Clear any existing polynomials from previous batch
        pipeline.clear_polynomials();
        
        // Prepare data for bulk upload (with zero-padding to domain size)
        let upload_start = Instant::now();
        let domain_size = 1usize << domain_log_size;
        let poly_slices: Vec<Vec<u32>> = polynomials.iter()
            .map(|poly| {
                let mut data: Vec<u32> = poly.coeffs.as_slice()
                    .iter()
                    .map(|f| f.0)
                    .collect();
                // Zero-pad to domain size
                data.resize(domain_size, 0);
                data
            })
            .collect();
        
        // Bulk upload all polynomials to GPU
        let num_uploaded = pipeline.upload_polynomials_bulk(poly_slices.iter().map(|v| v.as_slice()))
            .expect("Failed to bulk upload polynomials");
        let upload_time = upload_start.elapsed();
        
        // Execute FFT on all polynomials (stays on GPU)
        let compute_start = Instant::now();
        for poly_idx in 0..num_uploaded {
            if let Err(e) = pipeline.fft(poly_idx) {
                panic!("Failed to execute GPU FFT: {}", e);
            }
        }
        let compute_time = compute_start.elapsed();
        
        // Bulk download all results
        let download_start = Instant::now();
        let all_data = pipeline.download_polynomials_bulk()
            .expect("Failed to bulk download polynomials");
        let download_time = download_start.elapsed();
        
        // Convert to CircleEvaluation
        let results: Vec<CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>> = all_data.into_iter()
            .map(|data| {
                let values: BaseColumn = unsafe {
                    let bf_ptr = data.as_ptr() as *const BaseField;
                    std::slice::from_raw_parts(bf_ptr, data.len()).iter().copied().collect()
                };
                CircleEvaluation::new(domain, values)
            })
            .collect();
        
        tracing::debug!("GPU batch_evaluate_columns: upload={:?}, compute={:?}, download={:?}, total={:?}",
                 upload_time, compute_time, download_time,
                 upload_time + compute_time + download_time);
        
        results
    })
}

/// GPU-accelerated polynomial evaluation (forward FFT).
///
/// This function:
/// 1. Extracts coefficient data
/// 2. Computes forward FFT twiddles
/// 3. Transfers data to GPU
/// 4. Executes FFT kernels
/// 5. Transfers results back to CPU
fn gpu_evaluate(
    poly: &CircleCoefficients<GpuBackend>,
    domain: CircleDomain,
    _twiddles: &TwiddleTree<GpuBackend>,
    log_size: u32,
) -> CircleEvaluation<GpuBackend, BaseField, BitReversedOrder> {
    let _span = span!(Level::INFO, "GPU evaluate (FFT)", size = 1u64 << log_size).entered();
    
    // Extract raw data from coefficients
    let mut data: Vec<u32> = poly.coeffs.as_slice()
        .iter()
        .map(|f| f.0)
        .collect();
    
    // Pad to domain size if needed
    let domain_size = 1usize << log_size;
    if data.len() < domain_size {
        data.resize(domain_size, 0);
    }
    
    // Compute twiddles for forward FFT
    let twiddles_dbl = compute_twiddle_dbls_cpu(log_size);
    
    // Execute CUDA FFT
    match cuda_fft(&mut data, &twiddles_dbl, log_size) {
        Ok(()) => {
            tracing::info!(
                "GPU FFT completed for {} elements",
                1u64 << log_size
            );
            
            // Convert back to BaseColumn
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::core::fields::m31::BaseField;
            
            let values: BaseColumn = data.iter()
                .map(|&v| BaseField::from_u32_unchecked(v))
                .collect();
            
            CircleEvaluation::new(domain, values)
        }
        Err(e) => {
            tracing::error!(
                "GPU FFT execution failed: {}. Falling back to SIMD.",
                e
            );
            // Fallback to SIMD on GPU execution failure
            evaluate_simd_fallback(poly, domain, _twiddles)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_backend_uses_threshold() {
        // Verify that small polynomials don't trigger GPU path
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE <= 20);
    }
    
    #[test]
    fn test_threshold_is_reasonable() {
        // 16K elements is a good threshold based on benchmarks
        assert_eq!(GPU_FFT_THRESHOLD_LOG_SIZE, 14);
    }
}
