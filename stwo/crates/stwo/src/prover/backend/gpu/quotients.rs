//! GPU-accelerated quotient operations.
//!
//! This module implements [`QuotientOps`] for [`GpuBackend`].
//!
//! # Strategy
//!
//! - `accumulate_numerators`: CUDA path for per-batch numerator accumulation.
//! - `compute_quotients_and_combine`: GPU-accelerated for large domains (log_size >= 14), using a
//!   fused CUDA kernel that combines denominator inverse computation and quotient accumulation in a
//!   single pass. Falls back to SIMD for small domains.

use itertools::zip_eq;
#[cfg(feature = "cuda-runtime")]
use num_traits::Zero;

use super::conversion::circle_eval_ref_to_simd;
#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::{get_cuda_executor, is_cuda_available, CudaFftError};
use super::GpuBackend;
use crate::core::fields::m31::BaseField;
#[cfg(feature = "cuda-runtime")]
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{quotient_constants, ColumnSampleBatch};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::pcs::quotient_ops::{AccumulatedNumerators, QuotientOps};
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

impl QuotientOps for GpuBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        #[cfg(feature = "cuda-runtime")]
        {
            if is_cuda_available() {
                match gpu_accumulate_numerators(columns, sample_batches) {
                    Ok(gpu_acc) => {
                        if gpu_quotient_hardening_enabled() {
                            let simd_acc = simd_accumulate_numerators(columns, sample_batches);
                            assert_accumulated_numerators_equal(
                                "accumulate_numerators",
                                &gpu_acc,
                                &simd_acc,
                            );
                        }
                        accumulated_numerators_vec.extend(gpu_acc);
                        return;
                    }
                    Err(err) => {
                        if gpu_quotient_strict_mode() {
                            panic!(
                                "GpuBackend::accumulate_numerators failed in strict mode: {}",
                                err
                            );
                        }
                        tracing::warn!(
                            "[GPU] accumulate_numerators failed ({}), falling back to SIMD",
                            err
                        );
                    }
                }
            } else if gpu_quotient_strict_mode() {
                panic!("GpuBackend::accumulate_numerators strict mode requires CUDA availability");
            }
        }

        accumulated_numerators_vec.extend(simd_accumulate_numerators(columns, sample_batches));
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let max_log_size = accs
            .iter()
            .map(|x| x.partial_numerators_acc.len())
            .max()
            .unwrap()
            .ilog2();

        // GPU dispatch for large domains
        #[cfg(feature = "cuda-runtime")]
        {
            use super::constraints::GPU_QUOTIENT_THRESHOLD_LOG_SIZE;

            if max_log_size >= GPU_QUOTIENT_THRESHOLD_LOG_SIZE {
                match gpu_compute_quotients_and_combine(&accs, max_log_size) {
                    Ok(result) => {
                        tracing::debug!(
                            "[GPU] PCS quotient combination: log_size={}, {} samples",
                            max_log_size,
                            accs.len()
                        );
                        return result;
                    }
                    Err(e) => {
                        tracing::warn!("[GPU] PCS quotient failed ({}), falling back to SIMD", e);
                    }
                }
            }
        }

        // SIMD fallback
        simd_compute_quotients_and_combine(accs)
    }
}

fn simd_accumulate_numerators(
    columns: &[&CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>],
    sample_batches: &[ColumnSampleBatch],
) -> Vec<AccumulatedNumerators<GpuBackend>> {
    let simd_columns: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = columns
        .iter()
        .map(|c| circle_eval_ref_to_simd(*c))
        .collect();

    let mut simd_acc: Vec<AccumulatedNumerators<SimdBackend>> = Vec::new();
    SimdBackend::accumulate_numerators(&simd_columns, sample_batches, &mut simd_acc);

    simd_acc
        .into_iter()
        .map(|acc| AccumulatedNumerators {
            sample_point: acc.sample_point,
            partial_numerators_acc: SecureColumnByCoords {
                columns: acc.partial_numerators_acc.columns,
            },
            first_linear_term_acc: acc.first_linear_term_acc,
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn gpu_accumulate_numerators(
    columns: &[&CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>],
    sample_batches: &[ColumnSampleBatch],
) -> Result<Vec<AccumulatedNumerators<GpuBackend>>, CudaFftError> {
    if sample_batches.is_empty() {
        return Ok(Vec::new());
    }
    if columns.is_empty() {
        return Err(CudaFftError::InvalidSize(
            "accumulate_numerators requires at least one column".into(),
        ));
    }

    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    let n_points = columns[0].values.len();
    let constants = quotient_constants(sample_batches);
    let all_columns: Vec<Vec<u32>> = columns
        .iter()
        .map(|col| base_column_to_u32_vec(&col.values))
        .collect();
    let d_columns = executor.upload_accumulate_columns(&all_columns, n_points)?;
    let denom_inv = unit_cm31_denominators(n_points);

    let mut out = Vec::with_capacity(sample_batches.len());
    for (batch, coeffs) in zip_eq(sample_batches, constants.line_coeffs) {
        if coeffs.is_empty() {
            out.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc: SecureColumnByCoords::zeros(n_points),
                first_linear_term_acc: SecureField::zero(),
            });
            continue;
        }

        let mut col_indices = Vec::with_capacity(batch.cols_vals_randpows.len());
        let mut line_coeffs = Vec::with_capacity(coeffs.len());
        let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, ..)| *a).sum();

        for ((_a, b, c), numerator_data) in coeffs.iter().zip(batch.cols_vals_randpows.iter()) {
            col_indices.push(numerator_data.column_index);

            let b_u32 = securefield_to_u32(*b);
            let c_u32 = securefield_to_u32(*c);
            line_coeffs.push([
                0, 0, 0, 0, // `a` is intentionally omitted in partial numerator accumulation
                b_u32[0], b_u32[1], b_u32[2], b_u32[3], c_u32[0], c_u32[1], c_u32[2], c_u32[3],
            ]);
        }

        let batch_sizes = [line_coeffs.len()];

        let gpu_output = executor.execute_accumulate_quotients_with_device_columns(
            &d_columns,
            all_columns.len(),
            &line_coeffs,
            &denom_inv,
            &batch_sizes,
            &col_indices,
            n_points,
        )?;

        out.push(AccumulatedNumerators {
            sample_point: batch.point,
            partial_numerators_acc: aos_output_to_secure_column_gpu(&gpu_output, n_points),
            first_linear_term_acc,
        });
    }

    Ok(out)
}

#[cfg(feature = "cuda-runtime")]
fn unit_cm31_denominators(n_points: usize) -> Vec<u32> {
    let mut out = vec![0u32; n_points * 2];
    for i in 0..n_points {
        out[i * 2] = 1;
    }
    out
}

#[cfg(feature = "cuda-runtime")]
fn securefield_to_u32(sf: SecureField) -> [u32; 4] {
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::qm31::QM31;
    let QM31(CM31(a, b), CM31(c, d)) = sf;
    [a.0, b.0, c.0, d.0]
}

#[cfg(feature = "cuda-runtime")]
fn aos_output_to_secure_column_gpu(
    output: &[u32],
    n_points: usize,
) -> SecureColumnByCoords<GpuBackend> {
    assert_eq!(output.len(), n_points * 4);
    let mut c0 = Vec::with_capacity(n_points);
    let mut c1 = Vec::with_capacity(n_points);
    let mut c2 = Vec::with_capacity(n_points);
    let mut c3 = Vec::with_capacity(n_points);
    for i in 0..n_points {
        c0.push(output[i * 4]);
        c1.push(output[i * 4 + 1]);
        c2.push(output[i * 4 + 2]);
        c3.push(output[i * 4 + 3]);
    }
    super::conversion::aos_to_secure_column_from_soa(&c0, &c1, &c2, &c3, n_points)
}

#[cfg(feature = "cuda-runtime")]
fn base_column_to_u32_vec(col: &crate::prover::backend::simd::column::BaseColumn) -> Vec<u32> {
    let ptr = col.data.as_ptr() as *const u32;
    let total = col.data.len() * 16;
    let raw = unsafe { std::slice::from_raw_parts(ptr, total) };
    raw[..col.length].to_vec()
}

#[cfg(feature = "cuda-runtime")]
fn gpu_quotient_strict_mode() -> bool {
    gpu_quotient_flag_enabled("STWO_GPU_POLY_STRICT")
}

#[cfg(feature = "cuda-runtime")]
fn gpu_quotient_hardening_enabled() -> bool {
    gpu_quotient_flag_enabled("STWO_GPU_POLY_HARDEN")
}

#[cfg(feature = "cuda-runtime")]
fn gpu_quotient_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

#[cfg(feature = "cuda-runtime")]
fn normalized_m31_eq(a: u32, b: u32) -> bool {
    M31::reduce(a as u64).0 == M31::reduce(b as u64).0
}

#[cfg(feature = "cuda-runtime")]
fn assert_accumulated_numerators_equal(
    op: &str,
    gpu: &[AccumulatedNumerators<GpuBackend>],
    simd: &[AccumulatedNumerators<GpuBackend>],
) {
    assert_eq!(
        gpu.len(),
        simd.len(),
        "{} hardening mismatch: batch count differs (gpu={}, simd={})",
        op,
        gpu.len(),
        simd.len()
    );

    for (batch_idx, (lhs, rhs)) in gpu.iter().zip(simd.iter()).enumerate() {
        assert_eq!(
            lhs.sample_point, rhs.sample_point,
            "{} hardening mismatch at batch {}: sample_point differs",
            op, batch_idx
        );
        assert_eq!(
            lhs.first_linear_term_acc, rhs.first_linear_term_acc,
            "{} hardening mismatch at batch {}: first_linear_term_acc differs",
            op, batch_idx
        );

        let lhs_len = lhs.partial_numerators_acc.len();
        let rhs_len = rhs.partial_numerators_acc.len();
        assert_eq!(
            lhs_len, rhs_len,
            "{} hardening mismatch at batch {}: lengths differ",
            op, batch_idx
        );

        for coord in 0..4 {
            for row in 0..lhs_len {
                let a = lhs.partial_numerators_acc.columns[coord].at(row).0;
                let b = rhs.partial_numerators_acc.columns[coord].at(row).0;
                if !normalized_m31_eq(a, b) {
                    panic!(
                        "{} hardening mismatch at batch {}, coord {}, row {}: gpu={} simd={}",
                        op, batch_idx, coord, row, a, b
                    );
                }
            }
        }
    }
}

/// SIMD fallback for compute_quotients_and_combine.
fn simd_compute_quotients_and_combine(
    accs: Vec<AccumulatedNumerators<GpuBackend>>,
) -> SecureEvaluation<GpuBackend, BitReversedOrder> {
    let simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = accs
        .into_iter()
        .map(|acc| AccumulatedNumerators {
            sample_point: acc.sample_point,
            partial_numerators_acc: SecureColumnByCoords {
                columns: acc.partial_numerators_acc.columns,
            },
            first_linear_term_acc: acc.first_linear_term_acc,
        })
        .collect();

    let result = SimdBackend::compute_quotients_and_combine(simd_accs);

    SecureEvaluation::new(
        result.domain,
        SecureColumnByCoords {
            columns: result.values.columns,
        },
    )
}

/// GPU-accelerated compute_quotients_and_combine using the fused CUDA kernel.
///
/// Transfers domain points, partial numerators, and sample metadata to GPU,
/// runs the fused kernel, and reads back the result.
#[cfg(feature = "cuda-runtime")]
fn gpu_compute_quotients_and_combine(
    accs: &[AccumulatedNumerators<GpuBackend>],
    max_log_size: u32,
) -> Result<SecureEvaluation<GpuBackend, BitReversedOrder>, super::cuda_executor::CudaFftError> {
    use super::constraints::get_gpu_quotient_executor;
    use super::cuda_executor::CudaFftError;
    use crate::core::circle::CirclePoint;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::simd::column::BaseColumn;
    use crate::prover::backend::simd::domain::CircleDomainBitRevIterator;
    use crate::prover::backend::simd::m31::PackedBaseField;

    let executor = get_gpu_quotient_executor().map_err(|e| e.clone())?;
    let device = executor.device();

    let domain_size = 1u32 << max_log_size;
    let domain = CanonicCoset::new(max_log_size).circle_domain();
    let num_samples = accs.len() as u32;

    // Extract domain points as raw M31 values.
    // CircleDomainBitRevIterator yields CirclePoint<PackedBaseField> (16 M31s per pack).
    let mut domain_x_r_host = Vec::with_capacity(domain_size as usize);
    let mut domain_y_r_host = Vec::with_capacity(domain_size as usize);

    for point in CircleDomainBitRevIterator::new(domain) {
        // Each PackedBaseField holds 16 M31 values.
        // Extract raw u32 values via bytemuck-style reinterpretation.
        let x_ptr = &point.x as *const PackedBaseField as *const u32;
        let y_ptr = &point.y as *const PackedBaseField as *const u32;
        unsafe {
            for i in 0..16 {
                domain_x_r_host.push(*x_ptr.add(i));
                domain_y_r_host.push(*y_ptr.add(i));
            }
        }
    }
    // Domain points are base field, so imaginary parts are all zero.
    let domain_xi_host = vec![0u32; domain_size as usize];
    let domain_yi_host = vec![0u32; domain_size as usize];

    // Pack sample data: 13 u32 per sample
    // [sx.0.0, sx.0.1, sx.1.0, sx.1.1, sy.0.0, sy.0.1, sy.1.0, sy.1.1, flt0, flt1, flt2, flt3,
    // log_ratio]
    let mut sample_data_host = Vec::with_capacity(num_samples as usize * 13);
    for acc in accs {
        let sp = acc.sample_point;
        let flt = acc.first_linear_term_acc;
        let log_ratio = max_log_size - acc.partial_numerators_acc.len().ilog2();

        // CirclePoint<SecureField>: x and y are QM31 = (CM31, CM31) = ((M31,M31),(M31,M31))
        // SecureField = QM31 { 0: CM31(M31, M31), 1: CM31(M31, M31) }
        // sp.x.0.0 = real of first CM31, sp.x.0.1 = imag of first CM31
        // sp.x.1.0 = real of second CM31, sp.x.1.1 = imag of second CM31
        sample_data_host.push(sp.x.0 .0 .0); // Pr_x real
        sample_data_host.push(sp.x.0 .1 .0); // Pr_x imag
        sample_data_host.push(sp.x.1 .0 .0); // Pi_x real
        sample_data_host.push(sp.x.1 .1 .0); // Pi_x imag
        sample_data_host.push(sp.y.0 .0 .0); // Pr_y real
        sample_data_host.push(sp.y.0 .1 .0); // Pr_y imag
        sample_data_host.push(sp.y.1 .0 .0); // Pi_y real
        sample_data_host.push(sp.y.1 .1 .0); // Pi_y imag
        sample_data_host.push(flt.0 .0 .0); // first_linear_term c0
        sample_data_host.push(flt.0 .1 .0); // first_linear_term c1
        sample_data_host.push(flt.1 .0 .0); // first_linear_term c2
        sample_data_host.push(flt.1 .1 .0); // first_linear_term c3
        sample_data_host.push(log_ratio);
    }

    // Pack partial numerators: for each sample, flatten 4 BaseColumns
    let mut partial_nums_host = Vec::new();
    for acc in accs {
        let num_size = acc.partial_numerators_acc.len();
        for col_idx in 0..4 {
            let col = &acc.partial_numerators_acc.columns[col_idx];
            let ptr = col.data.as_ptr() as *const u32;
            let raw = unsafe { std::slice::from_raw_parts(ptr, col.data.len() * 16) };
            partial_nums_host.extend_from_slice(&raw[..num_size]);
        }
    }

    // Transfer to GPU
    let d_domain_xr = device
        .htod_copy(domain_x_r_host)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("domain_xr: {}", e)))?;
    let d_domain_xi = device
        .htod_copy(domain_xi_host)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("domain_xi: {}", e)))?;
    let d_domain_yr = device
        .htod_copy(domain_y_r_host)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("domain_yr: {}", e)))?;
    let d_domain_yi = device
        .htod_copy(domain_yi_host)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("domain_yi: {}", e)))?;
    let d_partial_nums = device
        .htod_copy(partial_nums_host)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("partial_nums: {}", e)))?;
    let d_sample_data = device
        .htod_copy(sample_data_host)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("sample_data: {}", e)))?;

    // Allocate output buffers
    let mut d_out_c0 = device
        .alloc_zeros::<u32>(domain_size as usize)
        .map_err(|e| CudaFftError::MemoryAllocation(format!("out_c0: {}", e)))?;
    let mut d_out_c1 = device
        .alloc_zeros::<u32>(domain_size as usize)
        .map_err(|e| CudaFftError::MemoryAllocation(format!("out_c1: {}", e)))?;
    let mut d_out_c2 = device
        .alloc_zeros::<u32>(domain_size as usize)
        .map_err(|e| CudaFftError::MemoryAllocation(format!("out_c2: {}", e)))?;
    let mut d_out_c3 = device
        .alloc_zeros::<u32>(domain_size as usize)
        .map_err(|e| CudaFftError::MemoryAllocation(format!("out_c3: {}", e)))?;

    // Launch kernel
    executor.compute_quotients(
        &d_domain_xr,
        &d_domain_xi,
        &d_domain_yr,
        &d_domain_yi,
        &d_partial_nums,
        &d_sample_data,
        &mut d_out_c0,
        &mut d_out_c1,
        &mut d_out_c2,
        &mut d_out_c3,
        domain_size,
        num_samples,
    )?;

    // Read results back
    let out_c0 = device
        .dtoh_sync_copy(&d_out_c0)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("dtoh c0: {}", e)))?;
    let out_c1 = device
        .dtoh_sync_copy(&d_out_c1)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("dtoh c1: {}", e)))?;
    let out_c2 = device
        .dtoh_sync_copy(&d_out_c2)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("dtoh c2: {}", e)))?;
    let out_c3 = device
        .dtoh_sync_copy(&d_out_c3)
        .map_err(|e| CudaFftError::MemoryTransfer(format!("dtoh c3: {}", e)))?;

    // Convert back to SecureColumnByCoords
    let result_col = super::conversion::aos_to_secure_column_from_soa(
        &out_c0,
        &out_c1,
        &out_c2,
        &out_c3,
        domain_size as usize,
    );

    Ok(SecureEvaluation::new(domain, result_col))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpu_quotient_ops_compiles() {
        // Compile-time check that GpuBackend implements QuotientOps
        fn _assert_impl<T: super::QuotientOps>() {}
        _assert_impl::<super::GpuBackend>();
    }

    #[test]
    fn test_pcs_quotient_kernel_source_compiles() {
        use super::super::constraints::get_pcs_quotient_kernel_source;
        let source = get_pcs_quotient_kernel_source();
        assert!(source.contains("cm31_inv"));
        assert!(source.contains("qm31_mul_cm31"));
        assert!(source.contains("pcs_quotient_combine"));
    }
}
