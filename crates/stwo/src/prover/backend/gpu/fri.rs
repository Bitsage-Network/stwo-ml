//! GPU-accelerated FRI operations with CUDA residency.
//!
//! This module implements [`FriOps`] for [`GpuBackend`], providing:
//! - Parallel FRI folding using rayon for multi-core acceleration
//! - CUDA GPU dispatch for large evaluations (≥ 2^12 elements)
//! - GPU memory residency: fold outputs stay on GPU between consecutive
//!   rounds, eliminating N-1 host-to-device transfers in a fold chain.
//!
//! # GPU Residency Architecture
//!
//! ```text
//! fold_circle_into_line: CPU→GPU (H2D) → kernel → cache d_output + D2H
//! fold_line round 1:     cached d_output → kernel → cache d_output + D2H
//! fold_line round 2:     cached d_output → kernel → cache d_output + D2H
//! ...
//! fold_line round N:     cached d_output → kernel → D2H (final)
//! ```

use std::array;
#[cfg(feature = "cuda-runtime")]
use std::cell::RefCell;
#[cfg(feature = "cuda-runtime")]
use std::collections::HashMap;
use std::simd::{u32x16, u32x8};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{span, Level};

// =============================================================================
// Twiddle Cache (thread-local, keyed by domain identity)
// =============================================================================

/// Cached twiddle entry: CPU-side Vec<u32> and optionally a GPU-resident copy.
#[cfg(feature = "cuda-runtime")]
struct TwiddleCacheEntry {
    cpu: Vec<u32>,
    gpu: Option<cudarc::driver::CudaSlice<u32>>,
}

#[cfg(feature = "cuda-runtime")]
thread_local! {
    /// Cache keyed on (coset_initial_index, log_size) → twiddle data.
    /// Each unique FRI domain produces a unique key. Safe to keep per-thread
    /// because FRI proving is single-threaded at the round level.
    static TWIDDLE_CACHE: RefCell<HashMap<(usize, u32), TwiddleCacheEntry>> = RefCell::new(HashMap::new());
}

/// Clear the twiddle cache. Call after a proof is complete to free GPU memory.
#[cfg(feature = "cuda-runtime")]
pub fn clear_twiddle_cache() {
    TWIDDLE_CACHE.with(|cache| cache.borrow_mut().clear());
}

#[cfg(feature = "cuda-runtime")]
use crate::core::utils::bit_reverse_index;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::prover::backend::cpu::{fold_circle_into_line_cpu, fold_line_cpu};
use crate::prover::backend::simd::fft::compute_first_twiddles;
use crate::prover::backend::simd::fft::ifft::simd_ibutterfly;
use crate::prover::backend::simd::m31::LOG_N_LANES;
use crate::prover::backend::simd::qm31::PackedSecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

use super::conversion::{
    line_eval_mut_to_simd, secure_eval_ref_to_simd, secure_eval_to_gpu, twiddle_ref_to_simd,
};
use super::GpuBackend;

/// Minimum log_size for CUDA FRI dispatch. Below this, SIMD is faster.
#[cfg(feature = "cuda-runtime")]
const GPU_FRI_THRESHOLD_LOG_SIZE: u32 = 12;

impl FriOps for GpuBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::fold_line").entered();

        let log_size = eval.len().ilog2();

        // Small evaluations: use CPU fallback
        if log_size <= LOG_N_LANES {
            let simd_eval = unsafe {
                &*(eval as *const LineEvaluation<GpuBackend> as *const LineEvaluation<SimdBackend>)
            };
            let cpu_eval = fold_line_cpu(&simd_eval.to_cpu(), alpha);
            let result: LineEvaluation<SimdBackend> = LineEvaluation::new(
                cpu_eval.domain(),
                cpu_eval.values.into_iter().collect(),
            );
            return unsafe { std::mem::transmute(result) };
        }

        // Try CUDA path for large evaluations
        #[cfg(feature = "cuda-runtime")]
        if log_size >= GPU_FRI_THRESHOLD_LOG_SIZE && super::cuda_executor::is_cuda_available() {
            if let Ok(result) = fold_line_cuda(eval, alpha, twiddles, log_size) {
                return result;
            }
            // Fall through to SIMD on CUDA error
        }

        // SIMD path (parallel with rayon)
        fold_line_simd(eval, alpha, twiddles, log_size)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let _span = span!(Level::TRACE, "GpuBackend::fold_circle_into_line").entered();

        let log_size = src.len().ilog2();

        // Small evaluations: use CPU fallback
        if log_size <= LOG_N_LANES {
            let simd_dst = line_eval_mut_to_simd(dst);
            let simd_src = secure_eval_ref_to_simd(src);
            let mut cpu_dst = simd_dst.to_cpu();
            fold_circle_into_line_cpu(&mut cpu_dst, &simd_src.to_cpu(), alpha);
            *simd_dst = LineEvaluation::new(
                cpu_dst.domain(),
                SecureColumnByCoords::from_cpu(cpu_dst.values),
            );
            return;
        }

        // Try CUDA path for large evaluations
        #[cfg(feature = "cuda-runtime")]
        if log_size >= GPU_FRI_THRESHOLD_LOG_SIZE && super::cuda_executor::is_cuda_available() {
            if fold_circle_into_line_cuda(dst, src, alpha, twiddles, log_size).is_ok() {
                return;
            }
            // Fall through to SIMD on CUDA error
        }

        // SIMD path (parallel with rayon)
        fold_circle_into_line_simd(dst, src, alpha, twiddles, log_size);
    }

    fn resolve_pending_line_evaluation(eval: &mut LineEvaluation<Self>) {
        #[cfg(feature = "cuda-runtime")]
        {
            use super::conversion::aos_to_secure_column;
            use super::memory::pop_next_deferred_fri_fold;

            if let Some(entry) = pop_next_deferred_fri_fold() {
                if entry.n_output == 0 {
                    // Marker entry: already resolved (CPU fallback path)
                    return;
                }
                let executor = match super::cuda_executor::get_cuda_executor() {
                    Ok(e) => e,
                    Err(_) => return,
                };
                let mut cpu_output = vec![0u32; entry.n_output * 4];
                if executor.device.dtoh_sync_copy_into(&entry.d_aos, &mut cpu_output).is_ok() {
                    let resolved = aos_to_secure_column(&cpu_output, entry.n_output);
                    let simd_eval = unsafe {
                        &mut *(eval as *mut LineEvaluation<GpuBackend>
                            as *mut LineEvaluation<SimdBackend>)
                    };
                    simd_eval.values = resolved;
                    tracing::debug!(
                        "GPU-resident FRI: resolved deferred D2H ({} elements)",
                        entry.n_output
                    );
                }
            }
        }
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        let simd_eval = secure_eval_ref_to_simd(eval);
        let (simd_result, lambda) = SimdBackend::decompose(simd_eval);
        let result = secure_eval_to_gpu(simd_result);
        (result, lambda)
    }
}

// =============================================================================
// SIMD Paths (unchanged from original)
// =============================================================================

fn fold_line_simd(
    eval: &LineEvaluation<GpuBackend>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
    log_size: u32,
) -> LineEvaluation<GpuBackend> {
    let domain = eval.domain();
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let itwiddles = domain_line_twiddles_from_tree(domain, &simd_twiddles.itwiddles)[0];

    let simd_eval = unsafe {
        &*(eval as *const LineEvaluation<GpuBackend> as *const LineEvaluation<SimdBackend>)
    };

    let n_vecs = 1usize << (log_size - 1 - LOG_N_LANES);

    let results: Vec<(usize, PackedSecureField)> = {
        #[cfg(feature = "parallel")]
        let iter = (0..n_vecs).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = (0..n_vecs).into_iter();

        iter.map(|vec_index| {
            let twiddle_dbl = u32x16::from_array(array::from_fn(|i| unsafe {
                *itwiddles.get_unchecked(vec_index * 16 + i)
            }));
            let val0 =
                unsafe { simd_eval.values.packed_at(vec_index * 2) }.into_packed_m31s();
            let val1 =
                unsafe { simd_eval.values.packed_at(vec_index * 2 + 1) }.into_packed_m31s();
            let pairs: [_; 4] = array::from_fn(|i| {
                let (a, b) = val0[i].deinterleave(val1[i]);
                simd_ibutterfly(a, b, twiddle_dbl)
            });
            let val0 =
                PackedSecureField::from_packed_m31s(array::from_fn(|i| pairs[i].0));
            let val1 =
                PackedSecureField::from_packed_m31s(array::from_fn(|i| pairs[i].1));
            let value = val0 + PackedSecureField::broadcast(alpha) * val1;
            (vec_index, value)
        })
        .collect()
    };

    let mut folded_values = SecureColumnByCoords::<SimdBackend>::zeros(1 << (log_size - 1));
    for (vec_index, value) in results {
        unsafe { folded_values.set_packed(vec_index, value) };
    }

    let result: LineEvaluation<SimdBackend> = LineEvaluation::new(domain.double(), folded_values);
    unsafe { std::mem::transmute(result) }
}

fn fold_circle_into_line_simd(
    dst: &mut LineEvaluation<GpuBackend>,
    src: &SecureEvaluation<GpuBackend, BitReversedOrder>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
    log_size: u32,
) {
    let domain = src.domain;
    let alpha_sq = alpha * alpha;
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let itwiddles = domain_line_twiddles_from_tree(domain, &simd_twiddles.itwiddles)[0];

    let simd_src = unsafe {
        &*(src as *const SecureEvaluation<GpuBackend, BitReversedOrder>
            as *const SecureEvaluation<SimdBackend, BitReversedOrder>)
    };
    let simd_dst = unsafe {
        &*(dst as *const LineEvaluation<GpuBackend> as *const LineEvaluation<SimdBackend>)
    };

    let n_vecs = 1usize << (log_size - 1 - LOG_N_LANES);

    let results: Vec<(usize, PackedSecureField)> = {
        #[cfg(feature = "parallel")]
        let iter = (0..n_vecs).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = (0..n_vecs).into_iter();

        iter.map(|vec_index| {
            let value = unsafe {
                let twiddle_dbl = u32x8::from_array(array::from_fn(|i| {
                    *itwiddles.get_unchecked(vec_index * 8 + i)
                }));
                let (t0, _) = compute_first_twiddles(twiddle_dbl);
                let val0 = simd_src.values.packed_at(vec_index * 2).into_packed_m31s();
                let val1 =
                    simd_src.values.packed_at(vec_index * 2 + 1).into_packed_m31s();
                let pairs: [_; 4] = array::from_fn(|i| {
                    let (a, b) = val0[i].deinterleave(val1[i]);
                    simd_ibutterfly(a, b, t0)
                });
                let val0 =
                    PackedSecureField::from_packed_m31s(array::from_fn(|i| pairs[i].0));
                let val1 =
                    PackedSecureField::from_packed_m31s(array::from_fn(|i| pairs[i].1));
                val0 + PackedSecureField::broadcast(alpha) * val1
            };
            let dst_val = unsafe { simd_dst.values.packed_at(vec_index) };
            let new_val = dst_val * PackedSecureField::broadcast(alpha_sq) + value;
            (vec_index, new_val)
        })
        .collect()
    };

    let simd_dst_mut = unsafe {
        &mut *(dst as *mut LineEvaluation<GpuBackend> as *mut LineEvaluation<SimdBackend>)
    };
    for (vec_index, value) in results {
        unsafe { simd_dst_mut.values.set_packed(vec_index, value) };
    }
}

// =============================================================================
// CUDA Paths (GPU-resident)
// =============================================================================

#[cfg(feature = "cuda-runtime")]
fn fold_line_cuda(
    eval: &LineEvaluation<GpuBackend>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
    log_size: u32,
) -> Result<LineEvaluation<GpuBackend>, super::cuda_executor::CudaFftError> {
    use super::conversion::{secure_column_to_aos, aos_to_secure_column};
    use super::memory::{
        take_cached_fri_gpu_data, cache_fri_gpu_data, cache_fri_column_gpu,
    };

    let n = eval.len();
    let n_output = n / 2;
    let domain = eval.domain();

    let alpha_u32 = securefield_to_u32(alpha);
    let itwiddles_u32 = compute_fold_line_itwiddles(domain, log_size);

    let simd_eval = unsafe {
        &*(eval as *const LineEvaluation<GpuBackend> as *const LineEvaluation<SimdBackend>)
    };

    let d_input = take_cached_fri_gpu_data(&simd_eval.values);
    let executor = super::cuda_executor::get_cuda_executor().map_err(|e| e.clone())?;

    let coset = domain.coset();
    let cache_key = (coset.initial_index.0, log_size);
    let d_itwiddles = get_or_upload_twiddle_gpu(cache_key, &itwiddles_u32, &executor.device)?;

    // Use GPU-only fold (no D2H inside kernel call) — separates kernel execution
    // from data transfer, enabling cleaner profiling and future async optimization
    let d_output = if let Some(d_cached) = d_input {
        tracing::debug!("FRI fold_line: GPU-only pipeline (log_size={})", log_size);
        executor.execute_fold_line_gpu_only(&d_cached, &d_itwiddles, &alpha_u32, n)?
    } else {
        tracing::debug!("FRI fold_line: H2D + GPU-only (log_size={})", log_size);
        let aos = secure_column_to_aos(&simd_eval.values, n);
        let d_input = executor.device.htod_sync_copy(&aos)
            .map_err(|e| super::cuda_executor::CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        executor.execute_fold_line_gpu_only(&d_input, &d_itwiddles, &alpha_u32, n)?
    };

    // D2H transfer — explicit and separate from kernel call
    let mut cpu_output = vec![0u32; n_output * 4];
    executor.device.dtoh_sync_copy_into(&d_output, &mut cpu_output)
        .map_err(|e| super::cuda_executor::CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
    let folded_values = aos_to_secure_column(&cpu_output, n_output);

    // Deinterleave AoS → 4 SoA columns on GPU for Poseidon252 Merkle.
    // Uses the final folded_values pointers as cache keys for correct lookup.
    if let Ok(soa_cols) = executor.execute_deinterleave_aos_to_soa(&d_output, n_output) {
        for (i, d_col) in soa_cols.into_iter().enumerate() {
            let col_ptr = folded_values.columns[i].as_slice().as_ptr() as usize;
            cache_fri_column_gpu(col_ptr, d_col);
        }
    }

    // Cache AoS for next fold round (consumed by take_cached_fri_gpu_data)
    cache_fri_gpu_data(&folded_values, d_output);

    let result: LineEvaluation<SimdBackend> = LineEvaluation::new(domain.double(), folded_values);
    Ok(unsafe { std::mem::transmute(result) })
}

#[cfg(feature = "cuda-runtime")]
fn fold_circle_into_line_cuda(
    dst: &mut LineEvaluation<GpuBackend>,
    src: &SecureEvaluation<GpuBackend, BitReversedOrder>,
    alpha: SecureField,
    twiddles: &TwiddleTree<GpuBackend>,
    _log_size: u32,
) -> Result<(), super::cuda_executor::CudaFftError> {
    use super::conversion::{secure_column_to_aos, aos_to_secure_column};
    use super::memory::{
        take_cached_fri_gpu_data, cache_fri_gpu_data, cache_fri_column_gpu,
    };

    let n = src.len();
    let n_dst = n / 2;

    let alpha_u32 = securefield_to_u32(alpha);

    let domain = src.domain;
    let itwiddles_u32 = compute_fold_circle_itwiddles(domain);

    let simd_src = unsafe {
        &*(src as *const SecureEvaluation<GpuBackend, BitReversedOrder>
            as *const SecureEvaluation<SimdBackend, BitReversedOrder>)
    };
    let simd_dst = unsafe {
        &*(dst as *const LineEvaluation<GpuBackend> as *const LineEvaluation<SimdBackend>)
    };

    let d_src_cached = take_cached_fri_gpu_data(&simd_src.values);
    // Drop any cached dst GPU data (not used in this path)
    let _ = take_cached_fri_gpu_data(&simd_dst.values);

    let executor = super::cuda_executor::get_cuda_executor().map_err(|e| e.clone())?;

    let log_size = domain.log_size();
    let circle_cache_key = (domain.half_coset.initial_index.0, log_size);
    let d_itwiddles = get_or_upload_twiddle_gpu(circle_cache_key, &itwiddles_u32, &executor.device)?;

    // Upload src and dst to GPU, fold, download result
    let src_aos = secure_column_to_aos(&simd_src.values, n);
    let mut dst_aos = secure_column_to_aos(&simd_dst.values, n_dst);

    // Use GPU-cached src if available (skip H2D for src)
    let d_result = if let Some(d_src_gpu) = d_src_cached {
        tracing::debug!("FRI fold_circle_into_line: cached src (n={})", n);
        executor.execute_fold_circle_into_line_from_gpu(
            &mut dst_aos, &d_src_gpu, &d_itwiddles, &alpha_u32, n,
        )?
    } else {
        tracing::debug!("FRI fold_circle_into_line: full H2D (n={})", n);
        executor.execute_fold_circle_into_line_resident_preloaded(
            &mut dst_aos, &src_aos, &d_itwiddles, &alpha_u32, n,
        )?
    };

    let result_col = aos_to_secure_column(&dst_aos, n_dst);
    let simd_dst_mut = unsafe {
        &mut *(dst as *mut LineEvaluation<GpuBackend> as *mut LineEvaluation<SimdBackend>)
    };
    simd_dst_mut.values = result_col;

    // Deinterleave AoS → SoA on GPU for Poseidon252 Merkle cache
    if let Ok(soa_cols) = executor.execute_deinterleave_aos_to_soa(&d_result, n_dst) {
        for (i, d_col) in soa_cols.into_iter().enumerate() {
            let col_ptr = simd_dst_mut.values.columns[i].as_slice().as_ptr() as usize;
            cache_fri_column_gpu(col_ptr, d_col);
        }
    }

    // Cache AoS for next fold round
    cache_fri_gpu_data(&simd_dst_mut.values, d_result);

    Ok(())
}

/// Convert a `SecureField` (QM31) to its 4 raw u32 components.
#[cfg(feature = "cuda-runtime")]
fn securefield_to_u32(sf: SecureField) -> [u32; 4] {
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::qm31::QM31;
    let QM31(CM31(a, b), CM31(c, d)) = sf;
    [a.0, b.0, c.0, d.0]
}

/// Batch-invert an array of M31 field elements using Montgomery's trick.
/// Computes all inverses with a single field inversion + 3n multiplications.
#[cfg(feature = "cuda-runtime")]
fn batch_inverse_m31(values: &[crate::core::fields::m31::BaseField]) -> Vec<crate::core::fields::m31::BaseField> {
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::FieldExpOps;

    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![values[0].inverse()];
    }

    // Build prefix products
    let mut prefix = Vec::with_capacity(n);
    prefix.push(values[0]);
    for i in 1..n {
        prefix.push(prefix[i - 1] * values[i]);
    }

    // Single inversion of the total product
    let mut inv = prefix[n - 1].inverse();

    // Back-propagate to recover individual inverses
    let mut result = vec![BaseField::from(0u32); n];
    for i in (1..n).rev() {
        result[i] = inv * prefix[i - 1];
        inv = inv * values[i];
    }
    result[0] = inv;
    result
}

/// Compute inverse twiddles for the CUDA fold_line kernel.
///
/// Uses twiddle cache to avoid recomputation across FRI rounds, and
/// Montgomery batch inversion for ~100x speedup over individual inversions.
///
/// Returns a reference-counted cache key so callers can retrieve the
/// GPU-resident copy via `get_cached_twiddle_gpu`.
#[cfg(feature = "cuda-runtime")]
fn compute_fold_line_itwiddles(
    domain: crate::core::poly::line::LineDomain,
    log_size: u32,
) -> Vec<u32> {
    use crate::core::fields::m31::BaseField;

    // Cache key: coset's initial point + log_size uniquely identifies the domain
    let coset = domain.coset();
    let cache_key = (coset.initial_index.0, log_size);

    // Check cache first
    let cached = TWIDDLE_CACHE.with(|cache| {
        cache.borrow().get(&cache_key).map(|entry| entry.cpu.clone())
    });
    if let Some(cpu) = cached {
        return cpu;
    }

    // Compute twiddle values (without inversion)
    let n = 1usize << log_size;
    let values: Vec<BaseField> = (0..n / 2)
        .map(|i| domain.at(bit_reverse_index(i << 1, log_size)))
        .collect();

    // Batch inversion
    let inverses = batch_inverse_m31(&values);
    let itwiddles_u32: Vec<u32> = inverses.iter().map(|x| x.0).collect();

    // Store in cache
    TWIDDLE_CACHE.with(|cache| {
        cache.borrow_mut().insert(cache_key, TwiddleCacheEntry {
            cpu: itwiddles_u32.clone(),
            gpu: None,
        });
    });

    itwiddles_u32
}

/// Compute inverse twiddles for the CUDA fold_circle_into_line kernel.
///
/// Uses twiddle cache and Montgomery batch inversion.
#[cfg(feature = "cuda-runtime")]
fn compute_fold_circle_itwiddles(
    domain: crate::core::poly::circle::CircleDomain,
) -> Vec<u32> {
    use crate::core::fields::m31::BaseField;

    let log_size = domain.log_size();
    let n = 1usize << log_size;

    // Cache key: use the half_coset's initial_index + log_size
    let cache_key = (domain.half_coset.initial_index.0, log_size);

    let cached = TWIDDLE_CACHE.with(|cache| {
        cache.borrow().get(&cache_key).map(|entry| entry.cpu.clone())
    });
    if let Some(cpu) = cached {
        return cpu;
    }

    // Compute y-coordinates (without inversion)
    let values: Vec<BaseField> = (0..n / 2)
        .map(|i| {
            let p = domain.at(bit_reverse_index(i << 1, log_size));
            p.y
        })
        .collect();

    // Batch inversion
    let inverses = batch_inverse_m31(&values);
    let itwiddles_u32: Vec<u32> = inverses.iter().map(|x| x.0).collect();

    TWIDDLE_CACHE.with(|cache| {
        cache.borrow_mut().insert(cache_key, TwiddleCacheEntry {
            cpu: itwiddles_u32.clone(),
            gpu: None,
        });
    });

    itwiddles_u32
}

/// Get or upload twiddles to GPU, caching the CudaSlice for reuse.
/// Returns a clone of the CudaSlice (which is reference-counted in cudarc).
#[cfg(feature = "cuda-runtime")]
fn get_or_upload_twiddle_gpu(
    cache_key: (usize, u32),
    itwiddles: &[u32],
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<cudarc::driver::CudaSlice<u32>, super::cuda_executor::CudaFftError> {
    // Check if GPU copy is already cached
    let cached_gpu = TWIDDLE_CACHE.with(|cache| {
        cache.borrow().get(&cache_key).and_then(|entry| entry.gpu.clone())
    });

    if let Some(d_twiddles) = cached_gpu {
        return Ok(d_twiddles);
    }

    // Upload and cache
    let d_twiddles = device.htod_sync_copy(itwiddles)
        .map_err(|e| super::cuda_executor::CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

    TWIDDLE_CACHE.with(|cache| {
        if let Some(entry) = cache.borrow_mut().get_mut(&cache_key) {
            entry.gpu = Some(d_twiddles.clone());
        }
    });

    Ok(d_twiddles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_n_lanes() {
        assert!(LOG_N_LANES >= 2);
        assert!(LOG_N_LANES <= 6);
    }

    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_gpu_fri_threshold() {
        assert!(GPU_FRI_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_FRI_THRESHOLD_LOG_SIZE <= 20);
    }
}
