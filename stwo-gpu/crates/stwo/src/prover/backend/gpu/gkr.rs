//! GPU-accelerated GKR operations.
//!
//! This module implements [`GkrOps`] and [`MleOps`] for [`GpuBackend`].
//!
//! GKR (Goldwasser-Kalai-Rothblum) is used for lookup arguments.
//!
//! # GPU Acceleration Strategy
//!
//! - **MLE fold operations** (`fix_first_variable`): GPU-accelerated for large MLEs (>16K elements)
//! - **Equality evaluations** (`gen_eq_evals`): GPU-accelerated for large outputs
//! - **Next layer** and **sum_as_poly**: Delegate to SIMD (irregular access patterns)
//!
//! # Note on Conversions
//!
//! The GKR types (Mle, Layer, GkrMultivariatePolyOracle) contain backend-specific
//! column types internally. Since GpuBackend and SimdBackend share the same
//! underlying column types, we use the conversion module for type-safe conversions.

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::lookups::gkr_prover::{
    GkrMultivariatePolyOracle, GkrOps, Layer,
};
use crate::prover::lookups::utils::UnivariatePoly;
use crate::prover::lookups::mle::{Mle, MleOps};
#[cfg(feature = "cuda-runtime")]
use crate::prover::backend::Column;

use super::GpuBackend;

/// Minimum MLE size (log2) for GPU acceleration.
/// Below this threshold, SIMD is faster due to GPU overhead.
const GPU_MLE_THRESHOLD_LOG_SIZE: usize = 14; // 16K elements

// =============================================================================
// Type Conversion Utilities
// =============================================================================

/// Convert GpuBackend Mle to SimdBackend.
///
/// # Safety
/// This is safe because both backends use identical column types internally.
#[inline]
fn mle_base_to_simd(mle: Mle<GpuBackend, BaseField>) -> Mle<SimdBackend, BaseField> {
    unsafe { std::mem::transmute(mle) }
}

/// Convert SimdBackend Mle to GpuBackend.
#[inline]
fn mle_secure_to_gpu(mle: Mle<SimdBackend, SecureField>) -> Mle<GpuBackend, SecureField> {
    unsafe { std::mem::transmute(mle) }
}

/// Convert GpuBackend Mle (SecureField) to SimdBackend.
#[inline]
fn mle_secure_to_simd(mle: Mle<GpuBackend, SecureField>) -> Mle<SimdBackend, SecureField> {
    unsafe { std::mem::transmute(mle) }
}

/// Convert GpuBackend Layer reference to SimdBackend.
#[inline]
fn layer_ref_to_simd<'a>(layer: &'a Layer<GpuBackend>) -> &'a Layer<SimdBackend> {
    unsafe { std::mem::transmute(layer) }
}

/// Convert SimdBackend Layer to GpuBackend.
#[inline]
fn layer_to_gpu(layer: Layer<SimdBackend>) -> Layer<GpuBackend> {
    unsafe { std::mem::transmute(layer) }
}

/// Convert GpuBackend GkrMultivariatePolyOracle reference to SimdBackend.
#[inline]
fn gkr_oracle_ref_to_simd<'a>(
    oracle: &'a GkrMultivariatePolyOracle<'a, GpuBackend>
) -> &'a GkrMultivariatePolyOracle<'a, SimdBackend> {
    unsafe { std::mem::transmute(oracle) }
}

// =============================================================================
// MleOps Implementation with GPU Acceleration
// =============================================================================

impl MleOps<BaseField> for GpuBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let n_variables = mle.n_variables();

        // Use GPU for large MLEs
        if n_variables >= GPU_MLE_THRESHOLD_LOG_SIZE {
            #[cfg(feature = "cuda-runtime")]
            {
                tracing::debug!(
                    "GPU MLE fold (BaseField): {} variables, {} elements",
                    n_variables,
                    mle.len()
                );

                match gpu_impl::gpu_fix_first_variable_base(mle, assignment) {
                    Ok(result) => return result,
                    Err(e) => {
                        tracing::warn!("GPU MLE fold failed, falling back to SIMD: {:?}", e);
                        // Cannot recover mle after move, so we need to handle this differently
                        // For now, panic since GPU is expected to work
                        panic!("GPU MLE fold failed and cannot recover: {:?}", e);
                    }
                }
            }
        }

        // SIMD fallback (or small MLEs)
        let simd_mle = mle_base_to_simd(mle);
        let result = SimdBackend::fix_first_variable(simd_mle, assignment);
        mle_secure_to_gpu(result)
    }
}

impl MleOps<SecureField> for GpuBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let n_variables = mle.n_variables();

        // Use GPU for large MLEs
        if n_variables >= GPU_MLE_THRESHOLD_LOG_SIZE {
            #[cfg(feature = "cuda-runtime")]
            {
                tracing::debug!(
                    "GPU MLE fold (SecureField): {} variables, {} elements",
                    n_variables,
                    mle.len()
                );

                match gpu_impl::gpu_fix_first_variable_secure(mle, assignment) {
                    Ok(result) => return result,
                    Err(e) => {
                        tracing::warn!("GPU MLE fold (secure) failed: {:?}", e);
                        panic!("GPU MLE fold failed and cannot recover: {:?}", e);
                    }
                }
            }
        }

        // SIMD fallback (or small MLEs)
        let simd_mle = mle_secure_to_simd(mle);
        let result = SimdBackend::fix_first_variable(simd_mle, assignment);
        mle_secure_to_gpu(result)
    }
}

// =============================================================================
// GkrOps Implementation with GPU Acceleration
// =============================================================================

impl GkrOps for GpuBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let n_variables = y.len();
        #[allow(unused_variables)]
        let output_size = 1usize << n_variables;

        // Use GPU for large outputs
        if n_variables >= GPU_MLE_THRESHOLD_LOG_SIZE {
            #[cfg(feature = "cuda-runtime")]
            {
                tracing::debug!(
                    "GPU gen_eq_evals: {} variables, {} output elements",
                    n_variables,
                    output_size
                );

                match gpu_impl::gpu_gen_eq_evals(y, v) {
                    Ok(result) => return result,
                    Err(e) => {
                        tracing::warn!("GPU gen_eq_evals failed, falling back to SIMD: {:?}", e);
                        // Fall through to SIMD
                    }
                }
            }
        }

        // SIMD fallback
        let result = SimdBackend::gen_eq_evals(y, v);
        mle_secure_to_gpu(result)
    }
    
    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        // next_layer has irregular access patterns and moderate parallelism
        // SIMD is appropriate here
        let simd_layer = layer_ref_to_simd(layer);
        let result = SimdBackend::next_layer(simd_layer);
        layer_to_gpu(result)
    }
    
    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        // This operation has complex, irregular access patterns
        // and produces small output (degree-3 polynomial)
        // SIMD is more appropriate than GPU
        let simd_h = gkr_oracle_ref_to_simd(h);
        SimdBackend::sum_as_poly_in_first_variable(simd_h, claim)
    }
}

// =============================================================================
// GPU Kernel Wrappers (when cuda-runtime is enabled)
// =============================================================================

#[cfg(feature = "cuda-runtime")]
mod gpu_impl {
    use super::*;
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::M31;
    use crate::core::fields::qm31::QM31;
    use crate::prover::backend::Column;
    use crate::prover::backend::simd::column::SecureColumn;
    use super::super::cuda_executor::{get_cuda_executor, CudaFftError};

    /// Convert SecureField (QM31) to 4 u32 values.
    #[inline]
    fn secure_field_to_u32s(val: SecureField) -> [u32; 4] {
        [val.0.0.0, val.0.1.0, val.1.0.0, val.1.1.0]
    }

    /// Convert 4 u32 values to SecureField (QM31).
    #[inline]
    fn u32s_to_secure_field(data: &[u32]) -> SecureField {
        QM31(
            CM31(M31(data[0]), M31(data[1])),
            CM31(M31(data[2]), M31(data[3])),
        )
    }

    /// GPU implementation of fix_first_variable for BaseField MLE.
    ///
    /// Computes: result[i] = lhs[i] * (1 - assignment) + rhs[i] * assignment
    /// where lhs = mle[0..n/2] and rhs = mle[n/2..n]
    pub fn gpu_fix_first_variable_base(
        mle: Mle<GpuBackend, BaseField>,
        assignment: SecureField,
    ) -> Result<Mle<GpuBackend, SecureField>, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let n = mle.len();
        let half_n = n / 2;

        // Extract column data as raw u32
        let evals = mle.into_evals();
        let raw_data: Vec<u32> = evals.to_cpu().iter().map(|m| m.0).collect();

        // Split into lhs and rhs halves
        let lhs = &raw_data[..half_n];
        let rhs = &raw_data[half_n..];

        // Convert assignment to u32 array
        let assignment_u32 = secure_field_to_u32s(assignment);

        // Execute GPU kernel
        let output = executor.mle_fold_base_to_secure(lhs, rhs, &assignment_u32, half_n)?;

        // Convert output back to SecureColumn
        let mut result_col = SecureColumn::zeros(half_n);
        for i in 0..half_n {
            let val = u32s_to_secure_field(&output[i * 4..(i + 1) * 4]);
            result_col.set(i, val);
        }

        Ok(Mle::new(result_col))
    }

    /// GPU implementation of fix_first_variable for SecureField MLE.
    pub fn gpu_fix_first_variable_secure(
        mle: Mle<GpuBackend, SecureField>,
        assignment: SecureField,
    ) -> Result<Mle<GpuBackend, SecureField>, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let n = mle.len();
        let half_n = n / 2;

        // Extract column data as raw u32 (4 per SecureField)
        let evals = mle.into_evals();
        let cpu_data = evals.to_cpu();
        let raw_data: Vec<u32> = cpu_data.iter()
            .flat_map(|sf| secure_field_to_u32s(*sf))
            .collect();

        // Split into lhs and rhs halves (each element is 4 u32s)
        let lhs = &raw_data[..half_n * 4];
        let rhs = &raw_data[half_n * 4..];

        // Convert assignment to u32 array
        let assignment_u32 = secure_field_to_u32s(assignment);

        // Execute GPU kernel
        let output = executor.mle_fold_secure(lhs, rhs, &assignment_u32, half_n)?;

        // Convert output back to SecureColumn
        let mut result_col = SecureColumn::zeros(half_n);
        for i in 0..half_n {
            let val = u32s_to_secure_field(&output[i * 4..(i + 1) * 4]);
            result_col.set(i, val);
        }

        Ok(Mle::new(result_col))
    }

    /// GPU implementation of gen_eq_evals.
    ///
    /// Computes: eq_evals[i] = v * prod_j((1 - y[j]) + y[j] * bit_j(i))
    /// where bit_j(i) is the j-th bit of i
    pub fn gpu_gen_eq_evals(
        y: &[SecureField],
        v: SecureField,
    ) -> Result<Mle<GpuBackend, SecureField>, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let n_variables = y.len();
        let output_size = 1usize << n_variables;

        // Convert y values to flat u32 array
        let y_u32: Vec<u32> = y.iter()
            .flat_map(|sf| secure_field_to_u32s(*sf))
            .collect();

        // Convert v to u32 array
        let v_u32 = secure_field_to_u32s(v);

        // Execute GPU kernel
        let output = executor.gen_eq_evals(&y_u32, &v_u32, n_variables)?;

        // Convert output back to SecureColumn
        let mut result_col = SecureColumn::zeros(output_size);
        for i in 0..output_size {
            let val = u32s_to_secure_field(&output[i * 4..(i + 1) * 4]);
            result_col.set(i, val);
        }

        Ok(Mle::new(result_col))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::qm31::QM31;
    use crate::prover::backend::simd::column::{BaseColumn, SecureColumn};
    use crate::prover::backend::Column;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Generate a random SecureField value.
    fn random_secure_field(rng: &mut SmallRng) -> SecureField {
        QM31::from_m31_array([
            M31::from(rng.gen::<u32>() % P),
            M31::from(rng.gen::<u32>() % P),
            M31::from(rng.gen::<u32>() % P),
            M31::from(rng.gen::<u32>() % P),
        ])
    }

    /// Test that GPU fix_first_variable for BaseField matches SIMD.
    #[test]
    fn test_fix_first_variable_base_matches_simd() {
        let mut rng = SmallRng::seed_from_u64(12345);

        // Test with sizes below and above the GPU threshold
        for log_size in [10, 12, 14, 16] {
            let size = 1 << log_size;

            // Create random BaseField MLE
            let mut base_data = BaseColumn::zeros(size);
            for i in 0..size {
                base_data.set(i, M31::from(rng.gen::<u32>() % P));
            }

            let assignment = random_secure_field(&mut rng);

            // Compute with SIMD
            let simd_mle: Mle<SimdBackend, BaseField> = Mle::new(base_data.clone());
            let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);

            // Compute with GPU backend (will use SIMD for small sizes, GPU for large)
            let gpu_mle: Mle<GpuBackend, BaseField> = Mle::new(base_data);
            let gpu_result = GpuBackend::fix_first_variable(gpu_mle, assignment);

            // Convert both to CPU for comparison
            let simd_cpu = simd_result.into_evals().to_cpu();
            let gpu_cpu = gpu_result.into_evals().to_cpu();

            assert_eq!(
                simd_cpu.len(),
                gpu_cpu.len(),
                "Result lengths differ for log_size={}", log_size
            );

            for i in 0..simd_cpu.len() {
                assert_eq!(
                    simd_cpu[i], gpu_cpu[i],
                    "Mismatch at index {} for log_size={}", i, log_size
                );
            }
        }
    }

    /// Test that GPU fix_first_variable for SecureField matches SIMD.
    #[test]
    fn test_fix_first_variable_secure_matches_simd() {
        let mut rng = SmallRng::seed_from_u64(54321);

        for log_size in [10, 12, 14, 16] {
            let size = 1 << log_size;

            // Create random SecureField MLE
            let mut secure_data = SecureColumn::zeros(size);
            for i in 0..size {
                secure_data.set(i, random_secure_field(&mut rng));
            }

            let assignment = random_secure_field(&mut rng);

            // Compute with SIMD
            let simd_mle: Mle<SimdBackend, SecureField> = Mle::new(secure_data.clone());
            let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);

            // Compute with GPU backend
            let gpu_mle: Mle<GpuBackend, SecureField> = Mle::new(secure_data);
            let gpu_result = GpuBackend::fix_first_variable(gpu_mle, assignment);

            // Compare results
            let simd_cpu = simd_result.into_evals().to_cpu();
            let gpu_cpu = gpu_result.into_evals().to_cpu();

            assert_eq!(
                simd_cpu.len(),
                gpu_cpu.len(),
                "Result lengths differ for log_size={}", log_size
            );

            for i in 0..simd_cpu.len() {
                assert_eq!(
                    simd_cpu[i], gpu_cpu[i],
                    "Mismatch at index {} for log_size={}", i, log_size
                );
            }
        }
    }

    /// Test that GPU gen_eq_evals matches SIMD.
    #[test]
    fn test_gen_eq_evals_matches_simd() {
        let mut rng = SmallRng::seed_from_u64(98765);

        for n_variables in [8, 10, 12, 14, 16] {
            // Generate random y values
            let y: Vec<SecureField> = (0..n_variables)
                .map(|_| random_secure_field(&mut rng))
                .collect();

            let v = random_secure_field(&mut rng);

            // Compute with SIMD
            let simd_result = SimdBackend::gen_eq_evals(&y, v);

            // Compute with GPU backend
            let gpu_result = GpuBackend::gen_eq_evals(&y, v);

            // Compare results
            let simd_cpu = simd_result.into_evals().to_cpu();
            let gpu_cpu = gpu_result.into_evals().to_cpu();

            let expected_size = 1 << n_variables;
            assert_eq!(simd_cpu.len(), expected_size);
            assert_eq!(gpu_cpu.len(), expected_size);

            for i in 0..simd_cpu.len() {
                assert_eq!(
                    simd_cpu[i], gpu_cpu[i],
                    "Mismatch at index {} for n_variables={}", i, n_variables
                );
            }
        }
    }

    /// Test next_layer operation (always uses SIMD, but test the delegation works).
    #[test]
    fn test_next_layer_delegation() {
        let mut rng = SmallRng::seed_from_u64(11111);

        for log_size in [8, 10, 12] {
            let size = 1 << log_size;

            // Create random SecureField column for the layer
            let mut col = SecureColumn::zeros(size);
            for i in 0..size {
                col.set(i, random_secure_field(&mut rng));
            }

            let gpu_mle: Mle<GpuBackend, SecureField> = Mle::new(col.clone());
            let simd_mle: Mle<SimdBackend, SecureField> = Mle::new(col);

            // Create GrandProduct layers
            let gpu_layer = Layer::GrandProduct(gpu_mle);
            let simd_layer = Layer::GrandProduct(simd_mle);

            // Compute next layer
            let gpu_next = GpuBackend::next_layer(&gpu_layer);
            let simd_next = SimdBackend::next_layer(&simd_layer);

            // Extract and compare results
            match (gpu_next, simd_next) {
                (Layer::GrandProduct(gpu_mle), Layer::GrandProduct(simd_mle)) => {
                    let gpu_cpu = gpu_mle.into_evals().to_cpu();
                    let simd_cpu = simd_mle.into_evals().to_cpu();

                    assert_eq!(gpu_cpu.len(), simd_cpu.len());
                    for i in 0..gpu_cpu.len() {
                        assert_eq!(
                            gpu_cpu[i], simd_cpu[i],
                            "Mismatch at index {} for log_size={}", i, log_size
                        );
                    }
                }
                _ => panic!("Unexpected layer variant"),
            }
        }
    }
}
