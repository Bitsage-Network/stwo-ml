//! GPU-accelerated quotient operations.
//!
//! This module implements [`QuotientOps`] for [`GpuBackend`].
//!
//! # Algorithm
//!
//! Quotient accumulation computes:
//! ```text
//! Q(P) = Σ (c·f(P) - a·P.y - b) / denominator(P)
//! ```
//!
//! For each sample batch and each point P in the domain.
//!
//! # GPU Strategy
//!
//! The row accumulation is highly parallel - each row can be computed independently.
//! We implement:
//! 1. **Denominator inverses** - Batch inverse on GPU (or CPU, then transfer)
//! 2. **Row accumulation** - GPU kernel processes all rows in parallel
//! 3. **Extension** - Uses GPU FFT for polynomial evaluation

use tracing::{span, Level};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::core::poly::circle::CircleDomain;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::pcs::quotient_ops::QuotientOps;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;

use super::conversion::{circle_eval_ref_to_simd, secure_eval_to_gpu};
use super::cuda_executor::is_cuda_available;
use super::GpuBackend;

/// Threshold for GPU acceleration (log2 of domain size)
const GPU_QUOTIENT_THRESHOLD_LOG_SIZE: u32 = 14; // 16K elements

impl QuotientOps for GpuBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let _span = span!(Level::TRACE, "GpuBackend::accumulate_quotients").entered();
        
        let log_size = domain.log_size();
        
        // Small domains: use SIMD (GPU overhead not worth it)
        if log_size < GPU_QUOTIENT_THRESHOLD_LOG_SIZE {
            tracing::debug!(
                "GPU quotients: using SIMD for small size (log_size={} < threshold={})",
                log_size, GPU_QUOTIENT_THRESHOLD_LOG_SIZE
            );
            return accumulate_quotients_simd_fallback(
                domain, columns, random_coeff, sample_batches, log_blowup_factor
            );
        }
        
        // Large domains: prefer GPU, fallback to SIMD if unavailable
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::accumulate_quotients: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded. For optimal performance, ensure CUDA is available.",
                log_size
            );
            return accumulate_quotients_simd_fallback(
                domain, columns, random_coeff, sample_batches, log_blowup_factor
            );
        }

        tracing::info!(
            "GPU accumulate_quotients: using CUDA for {} elements (log_size={})",
            domain.size(), log_size
        );

        gpu_accumulate_quotients(domain, columns, random_coeff, sample_batches, log_blowup_factor)
    }
}

// =============================================================================
// SIMD Fallback (for small domains)
// =============================================================================

fn accumulate_quotients_simd_fallback(
    domain: CircleDomain,
    columns: &[&CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>],
    random_coeff: SecureField,
    sample_batches: &[ColumnSampleBatch],
    log_blowup_factor: u32,
) -> SecureEvaluation<GpuBackend, BitReversedOrder> {
    // Convert column references using conversion module
    let simd_columns: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = 
        columns.iter().map(|c| circle_eval_ref_to_simd(*c)).collect();
    
    let result = SimdBackend::accumulate_quotients(
        domain,
        &simd_columns,
        random_coeff,
        sample_batches,
        log_blowup_factor,
    );
    
    secure_eval_to_gpu(result)
}

// =============================================================================
// GPU Implementation
// =============================================================================

/// GPU-accelerated quotient accumulation.
///
/// This implementation:
/// 1. Computes quotient constants (line coefficients, denominator inverses) on CPU
/// 2. Transfers data to GPU
/// 3. Runs parallel row accumulation on GPU
/// 4. Uses GPU FFT for extension
#[cfg(feature = "cuda-runtime")]
fn gpu_accumulate_quotients(
    domain: CircleDomain,
    columns: &[&CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>],
    random_coeff: SecureField,
    sample_batches: &[ColumnSampleBatch],
    log_blowup_factor: u32,
) -> SecureEvaluation<GpuBackend, BitReversedOrder> {
    use super::cuda_executor::cuda_accumulate_quotients;
    use crate::core::pcs::quotients::column_line_coeffs;
    use crate::core::utils::bit_reverse;
    use crate::prover::backend::Column;
    use crate::prover::poly::circle::PolyOps;
    use crate::prover::secure_column::SecureColumnByCoords;
    
    let _span = span!(Level::INFO, "GPU accumulate_quotients", size = domain.size()).entered();
    
    // Split domain
    let (subdomain, mut subdomain_shifts) = domain.split(log_blowup_factor);
    bit_reverse(&mut subdomain_shifts);
    
    // Compute quotient constants on CPU (these are small)
    let line_coeffs = column_line_coeffs(sample_batches, random_coeff);
    
    // Compute denominator inverses
    // For GPU, we compute the denominators, transfer to GPU, batch inverse there
    let denominator_inverses = compute_denominator_inverses_for_gpu(sample_batches, subdomain);
    
    // Prepare column data for GPU
    let column_data: Vec<Vec<u32>> = columns.iter()
        .map(|col| {
            col.values.as_slice().iter().map(|f| f.0).collect()
        })
        .collect();
    
    // Flatten line coefficients for GPU
    let flat_line_coeffs: Vec<[u32; 12]> = line_coeffs.iter()
        .flat_map(|batch_coeffs| {
            batch_coeffs.iter().map(|(a, b, c)| {
                // Each coefficient is QM31 = 4 M31
                [
                    a.0.0.0, a.0.1.0, a.1.0.0, a.1.1.0,  // a
                    b.0.0.0, b.0.1.0, b.1.0.0, b.1.1.0,  // b
                    c.0.0.0, c.0.1.0, c.1.0.0, c.1.1.0,  // c
                ]
            })
        })
        .collect();
    
    // Prepare sample batch metadata for GPU
    let batch_sizes: Vec<usize> = sample_batches.iter()
        .map(|batch| batch.columns_and_values.len())
        .collect();
    
    let column_indices: Vec<usize> = sample_batches.iter()
        .flat_map(|batch| batch.columns_and_values.iter().map(|(idx, _)| *idx))
        .collect();
    
    // Execute GPU kernel
    let subdomain_size = subdomain.size();
    
    match cuda_accumulate_quotients(
        &column_data,
        &flat_line_coeffs,
        &denominator_inverses,
        &batch_sizes,
        &column_indices,
        subdomain_size,
    ) {
        Ok(values_u32) => {
            // Convert result back to SecureColumnByCoords
            let mut values = SecureColumnByCoords::<GpuBackend>::zeros(subdomain_size);
            
            use crate::core::fields::cm31::CM31;
            use crate::core::fields::m31::M31;
            use crate::core::fields::qm31::QM31;
            
            for i in 0..subdomain_size {
                let base = i * 4;
                let val = QM31(
                    CM31(M31(values_u32[base]), M31(values_u32[base + 1])),
                    CM31(M31(values_u32[base + 2]), M31(values_u32[base + 3])),
                );
                values.set(i, val);
            }
            
            // Extend to full domain using GPU FFT
            let twiddles = GpuBackend::precompute_twiddles(subdomain.half_coset);
            
            // Interpolate subdomain values
            let subeval_polys: [_; 4] = std::array::from_fn(|coord| {
                let coord_col = values.columns[coord].clone();
                CircleEvaluation::<GpuBackend, BaseField, BitReversedOrder>::new(subdomain, coord_col)
                    .interpolate_with_twiddles(&twiddles)
            });
            
            // Allocate extended evaluation
            let mut extended_eval = SecureColumnByCoords::<GpuBackend>::zeros(domain.size());
            
            // Extend to each shifted subdomain
            for (ci, &c) in subdomain_shifts.iter().enumerate() {
                let shifted_subdomain = subdomain.shift(c);
                let shifted_twiddles = GpuBackend::precompute_twiddles(shifted_subdomain.half_coset);
                
                for coord in 0..4 {
                    let eval = subeval_polys[coord].evaluate_with_twiddles(shifted_subdomain, &shifted_twiddles);
                    let start = ci * eval.values.len();
                    let end = start + eval.values.len();
                    extended_eval.columns[coord].data[start..end]
                        .copy_from_slice(&eval.values.data);
                }
            }
            
            SecureEvaluation::new(domain, extended_eval)
        }
        Err(e) => {
            tracing::error!(
                "GPU accumulate_quotients CUDA execution failed: {}. Falling back to SIMD.",
                e
            );
            // Fallback to SIMD on CUDA execution failure
            accumulate_quotients_simd_fallback(
                domain, columns, random_coeff, sample_batches, log_blowup_factor
            )
        }
    }
}

#[cfg(not(feature = "cuda-runtime"))]
fn gpu_accumulate_quotients(
    _domain: CircleDomain,
    _columns: &[&CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>],
    _random_coeff: SecureField,
    _sample_batches: &[ColumnSampleBatch],
    _log_blowup_factor: u32,
) -> SecureEvaluation<GpuBackend, BitReversedOrder> {
    panic!("GPU accumulate_quotients requires cuda-runtime feature");
}

/// Compute denominator inverses for GPU.
///
/// Returns flat array of CM31 values (2 u32 per element).
#[cfg(feature = "cuda-runtime")]
fn compute_denominator_inverses_for_gpu(
    sample_batches: &[ColumnSampleBatch],
    domain: CircleDomain,
) -> Vec<u32> {
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::M31;
    use crate::core::fields::FieldExpOps;
    use crate::core::utils::bit_reverse_index;
    
    let domain_size = domain.size();
    let mut result = Vec::with_capacity(sample_batches.len() * domain_size * 2);
    
    for sample_batch in sample_batches {
        // Extract Pr, Pi from the sample point
        let pr_x = sample_batch.point.x.0;
        let pr_y = sample_batch.point.y.0;
        let pi_x = sample_batch.point.x.1;
        let pi_y = sample_batch.point.y.1;
        
        // Compute denominators for all domain points
        let mut denominators: Vec<CM31> = Vec::with_capacity(domain_size);
        
        for i in 0..domain_size {
            let bit_rev_i = bit_reverse_index(i, domain.log_size());
            let p = domain.at(bit_rev_i);
            
            // Line equation: (p - pr) × pi = (p.x - pr.x) * pi.y - (p.y - pr.y) * pi.x
            let dx = CM31(p.x - pr_x.0, M31(0) - pr_x.1);
            let dy = CM31(p.y - pr_y.0, M31(0) - pr_y.1);
            
            let denom = dx * pi_y - dy * pi_x;
            denominators.push(denom);
        }
        
        // Batch inverse
        let inverses = CM31::batch_inverse(&denominators);
        
        // Flatten to u32
        for inv in inverses {
            result.push(inv.0.0);  // real part
            result.push(inv.1.0);  // imag part
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_threshold_reasonable() {
        assert!(GPU_QUOTIENT_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_QUOTIENT_THRESHOLD_LOG_SIZE <= 20);
    }
}
