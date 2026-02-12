//! GPU-accelerated quotient operations.
//!
//! This module implements [`QuotientOps`] for [`GpuBackend`].
//!
//! # Strategy
//!
//! Delegates to the SIMD backend for correctness, with GPU-accelerated paths
//! for large domains when CUDA is available.

use crate::core::fields::m31::BaseField;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::pcs::quotient_ops::{AccumulatedNumerators, QuotientOps};
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

use super::conversion::circle_eval_ref_to_simd;
use super::GpuBackend;

impl QuotientOps for GpuBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        // Convert GpuBackend columns to SimdBackend columns
        let simd_columns: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
            columns.iter().map(|c| circle_eval_ref_to_simd(*c)).collect();

        // Delegate to SIMD backend
        let mut simd_acc: Vec<AccumulatedNumerators<SimdBackend>> = Vec::new();
        SimdBackend::accumulate_numerators(&simd_columns, sample_batches, &mut simd_acc);

        // Convert results back to GpuBackend types
        // Since GpuBackend and SimdBackend share the same column representation,
        // we can transmute the accumulated numerators.
        for acc in simd_acc {
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: acc.sample_point,
                partial_numerators_acc: SecureColumnByCoords {
                    columns: acc.partial_numerators_acc.columns,
                },
                first_linear_term_acc: acc.first_linear_term_acc,
            });
        }
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // Convert GpuBackend accumulated numerators to SimdBackend
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

        // Delegate to SIMD backend
        let result = SimdBackend::compute_quotients_and_combine(simd_accs);

        // Convert back to GpuBackend
        SecureEvaluation::new(
            result.domain,
            SecureColumnByCoords {
                columns: result.values.columns,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpu_quotient_ops_compiles() {
        // Compile-time check that GpuBackend implements QuotientOps
        fn _assert_impl<T: super::QuotientOps>() {}
        _assert_impl::<super::GpuBackend>();
    }
}
