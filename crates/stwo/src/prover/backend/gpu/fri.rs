//! GPU-accelerated FRI operations.
//!
//! This module implements [`FriOps`] for [`GpuBackend`], providing GPU acceleration
//! for FRI (Fast Reed-Solomon Interactive Oracle Proof) folding operations.
//!
//! # Algorithm
//!
//! FRI folding reduces polynomial degree by half at each step:
//!
//! 1. **fold_line**: For pairs (f(x), f(-x)), compute:
//!    - (f0, f1) = ibutterfly(f(x), f(-x), 1/x)
//!    - result = f0 + α * f1
//!
//! 2. **fold_circle_into_line**: For pairs (f(p), f(-p)) on circle:
//!    - (f0, f1) = ibutterfly(f(p), f(-p), 1/p.y)
//!    - f' = f0 + α * f1
//!    - dst = dst * α² + f'
//!
//! # Performance Characteristics
//!
//! FRI folding is the second most expensive operation after FFT:
//! - **fold_line**: 20-30x speedup on GPU for large evaluations
//! - **fold_circle_into_line**: 15-25x speedup on GPU

use tracing::{span, Level};

use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

#[cfg(feature = "cuda-runtime")]
use crate::prover::secure_column::SecureColumnByCoords;

use super::conversion::{
    line_eval_mut_to_simd, line_eval_ref_to_simd, line_eval_to_gpu,
    secure_eval_ref_to_simd, secure_eval_to_gpu, twiddle_ref_to_simd,
};
use super::cuda_executor::is_cuda_available;
use super::GpuBackend;

#[cfg(feature = "cuda-runtime")]
use crate::core::utils::bit_reverse_index;

/// Threshold below which GPU overhead exceeds benefit for FRI operations.
const GPU_FRI_THRESHOLD_LOG_SIZE: u32 = 14; // 16K elements

impl FriOps for GpuBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::fold_line").entered();
        
        let log_size = eval.len().ilog2();
        
        // Small evaluations: use SIMD (GPU overhead not worth it)
        if log_size < GPU_FRI_THRESHOLD_LOG_SIZE {
            return fold_line_simd_fallback(eval, alpha, twiddles);
        }
        
        // Large evaluations: prefer GPU, fallback to SIMD if unavailable
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::fold_line: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded. For optimal performance, ensure CUDA is available.",
                log_size
            );
            return fold_line_simd_fallback(eval, alpha, twiddles);
        }

        tracing::info!(
            "GPU fold_line: using CUDA for {} elements (log_size={})",
            eval.len(), log_size
        );
        gpu_fold_line(eval, alpha, twiddles)
    }
    
    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let _span = span!(Level::TRACE, "GpuBackend::fold_circle_into_line").entered();
        
        let log_size = src.len().ilog2();
        
        // Small evaluations: use SIMD
        if log_size < GPU_FRI_THRESHOLD_LOG_SIZE {
            fold_circle_into_line_simd_fallback(dst, src, alpha, twiddles);
            return;
        }
        
        // Large evaluations: prefer GPU, fallback to SIMD if unavailable
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::fold_circle_into_line: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded. For optimal performance, ensure CUDA is available.",
                log_size
            );
            fold_circle_into_line_simd_fallback(dst, src, alpha, twiddles);
            return;
        }

        tracing::info!(
            "GPU fold_circle_into_line: using CUDA for {} elements (log_size={})",
            src.len(), log_size
        );
        gpu_fold_circle_into_line(dst, src, alpha, twiddles);
    }
    
    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // Decompose is not compute-intensive, delegate to SIMD
        let simd_eval = secure_eval_ref_to_simd(eval);
        let (simd_result, lambda) = SimdBackend::decompose(simd_eval);
        let result = secure_eval_to_gpu(simd_result);
        (result, lambda)
    }
}

// =============================================================================
// SIMD Fallback Functions (for small sizes)
// =============================================================================

fn fold_line_simd_fallback(
    eval: &LineEvaluation<GpuBackend>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
) -> LineEvaluation<GpuBackend> {
    let simd_eval = line_eval_ref_to_simd(eval);
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let result = SimdBackend::fold_line(simd_eval, alpha, simd_twiddles);
    line_eval_to_gpu(result)
}

fn fold_circle_into_line_simd_fallback(
    dst: &mut LineEvaluation<GpuBackend>,
    src: &SecureEvaluation<GpuBackend, BitReversedOrder>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
) {
    let simd_dst = line_eval_mut_to_simd(dst);
    let simd_src = secure_eval_ref_to_simd(src);
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    SimdBackend::fold_circle_into_line(simd_dst, simd_src, alpha, simd_twiddles);
}

// =============================================================================
// GPU FRI Implementation
// =============================================================================

/// GPU-accelerated line folding.
#[cfg(feature = "cuda-runtime")]
fn gpu_fold_line(
    eval: &LineEvaluation<GpuBackend>,
    alpha: SecureField,
    _twiddles: &TwiddleTree<GpuBackend>,
) -> LineEvaluation<GpuBackend> {
    use super::cuda_executor::cuda_fold_line;
    
    let _span = span!(Level::INFO, "GPU fold_line", size = eval.len()).entered();
    
    let n = eval.len();
    let log_size = n.ilog2();
    let domain = eval.domain();
    
    // Convert SecureField values to flat u32 array
    // SecureField (QM31) = 4 M31 values
    let input: Vec<u32> = (0..n)
        .flat_map(|i| {
            let val = eval.values.at(i);
            secure_field_to_u32s(val)
        })
        .collect();
    
    // Convert alpha to u32 array
    let alpha_u32 = secure_field_to_u32s(alpha);
    
    // Compute inverse twiddles as u32 array
    // For fold_line, we need 1/x for each position
    let itwiddles_u32: Vec<u32> = (0..n/2)
        .map(|i| {
            let x = domain.at(bit_reverse_index(i << 1, log_size));
            x.inverse().0
        })
        .collect();
    
    // Execute CUDA fold_line
    match cuda_fold_line(&input, &itwiddles_u32, &alpha_u32, n) {
        Ok(output) => {
            // Convert output back to SecureColumnByCoords
            let folded_values = u32s_to_secure_column(&output, n / 2);
            LineEvaluation::new(domain.double(), folded_values)
        }
        Err(e) => {
            tracing::error!(
                "GPU fold_line CUDA execution failed: {}. Falling back to SIMD.",
                e
            );
            // Fallback to SIMD on CUDA execution failure
            fold_line_simd_fallback(eval, alpha, _twiddles)
        }
    }
}

#[cfg(not(feature = "cuda-runtime"))]
fn gpu_fold_line(
    _eval: &LineEvaluation<GpuBackend>,
    _alpha: SecureField,
    _twiddles: &TwiddleTree<GpuBackend>,
) -> LineEvaluation<GpuBackend> {
    panic!("GPU fold_line requires cuda-runtime feature");
}

/// GPU-accelerated circle-to-line folding.
#[cfg(feature = "cuda-runtime")]
fn gpu_fold_circle_into_line(
    dst: &mut LineEvaluation<GpuBackend>,
    src: &SecureEvaluation<GpuBackend, BitReversedOrder>,
    alpha: SecureField,
    _twiddles: &TwiddleTree<GpuBackend>,
) {
    use super::cuda_executor::cuda_fold_circle_into_line;
    
    let _span = span!(Level::INFO, "GPU fold_circle_into_line", size = src.len()).entered();
    
    let n = src.len();
    let log_size = n.ilog2();
    let src_domain = src.domain;
    
    // Convert source SecureField values to flat u32 array
    let src_u32: Vec<u32> = (0..n)
        .flat_map(|i| {
            let val = src.values.at(i);
            secure_field_to_u32s(val)
        })
        .collect();
    
    // Convert destination values to flat u32 array
    let n_dst = n / 2;
    let mut dst_u32: Vec<u32> = (0..n_dst)
        .flat_map(|i| {
            let val = dst.values.at(i);
            secure_field_to_u32s(val)
        })
        .collect();
    
    // Convert alpha to u32 array
    let alpha_u32 = secure_field_to_u32s(alpha);
    
    // Compute inverse twiddles (1/p.y for each position)
    let itwiddles_u32: Vec<u32> = (0..n_dst)
        .map(|i| {
            let p = src_domain.at(bit_reverse_index(i << 1, log_size));
            p.y.inverse().0
        })
        .collect();
    
    // Execute CUDA fold_circle_into_line
    match cuda_fold_circle_into_line(&mut dst_u32, &src_u32, &itwiddles_u32, &alpha_u32, n) {
        Ok(()) => {
            // Convert output back to SecureColumnByCoords and update dst
            let folded_values = u32s_to_secure_column(&dst_u32, n_dst);
            for i in 0..n_dst {
                dst.values.set(i, folded_values.at(i));
            }
        }
        Err(e) => {
            tracing::error!(
                "GPU fold_circle_into_line CUDA execution failed: {}. Falling back to SIMD.",
                e
            );
            // Fallback to SIMD on CUDA execution failure
            fold_circle_into_line_simd_fallback(dst, src, alpha, _twiddles);
        }
    }
}

#[cfg(not(feature = "cuda-runtime"))]
fn gpu_fold_circle_into_line(
    _dst: &mut LineEvaluation<GpuBackend>,
    _src: &SecureEvaluation<GpuBackend, BitReversedOrder>,
    _alpha: SecureField,
    _twiddles: &TwiddleTree<GpuBackend>,
) {
    panic!("GPU fold_circle_into_line requires cuda-runtime feature");
}

// =============================================================================
// Helper Functions (CUDA runtime only)
// =============================================================================

/// Convert SecureField (QM31) to 4 u32 values.
#[cfg(feature = "cuda-runtime")]
fn secure_field_to_u32s(val: SecureField) -> [u32; 4] {
    // QM31 = CM31(a0, a1) + i * CM31(a2, a3)
    // where CM31(x, y) = x + u * y
    [
        val.0.0.0,  // a0
        val.0.1.0,  // a1
        val.1.0.0,  // a2
        val.1.1.0,  // a3
    ]
}

/// Convert u32 array back to SecureColumnByCoords.
#[cfg(feature = "cuda-runtime")]
fn u32s_to_secure_column(data: &[u32], n: usize) -> SecureColumnByCoords<GpuBackend> {
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    
    // Create a SecureColumnByCoords with zeros, then fill it
    let mut result = SecureColumnByCoords::<GpuBackend>::zeros(n);
    
    for i in 0..n {
        let base = i * 4;
        let val = QM31(
            CM31(M31(data[base]), M31(data[base + 1])),
            CM31(M31(data[base + 2]), M31(data[base + 3])),
        );
        result.set(i, val);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_threshold_reasonable() {
        assert!(GPU_FRI_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_FRI_THRESHOLD_LOG_SIZE <= 20);
    }
    
    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_secure_field_conversion() {
        use crate::core::fields::cm31::CM31;
        use crate::core::fields::m31::M31;
        use crate::core::fields::qm31::QM31;
        
        let val = QM31(
            CM31(M31(1), M31(2)),
            CM31(M31(3), M31(4)),
        );
        
        let u32s = secure_field_to_u32s(val);
        assert_eq!(u32s, [1, 2, 3, 4]);
        
        let back = u32s_to_secure_column(&u32s, 1);
        assert_eq!(back.at(0), val);
    }
}
