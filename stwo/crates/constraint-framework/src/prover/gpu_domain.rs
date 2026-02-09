//! GPU Domain Evaluator for hardware-accelerated constraint evaluation.
//!
//! This module provides a GPU-accelerated domain evaluator that implements the
//! `EvalAtRow` trait for evaluating constraints on GPU-backed columns.
//!
//! # Architecture
//!
//! The GPU evaluator works with the same data layout as SIMD, but leverages
//! GPU parallelism for large traces. It uses vectorized field operations
//! (VeryPackedBaseField) for efficient constraint evaluation.

use std::ops::Mul;

use num_traits::Zero;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::pcs::TreeVec;
use stwo::core::utils::offset_bit_reversed_circle_domain_index;
use stwo::core::Fraction;
use stwo::prover::backend::gpu::GpuBackend;
use stwo::prover::backend::simd::column::VeryPackedBaseColumn;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::very_packed_m31::{
    VeryPackedBaseField, VeryPackedSecureField, LOG_N_VERY_PACKED_ELEMS,
};
use stwo::prover::backend::Column;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use crate::logup::LogupAtRow;
use crate::{EvalAtRow, INTERACTION_TRACE_IDX};

/// GPU Domain Evaluator for hardware-accelerated constraint evaluation.
///
/// This evaluator works on GPU-backed trace data and produces constraint
/// evaluation results. It uses the same vectorized field representation
/// as SIMD (VeryPackedBaseField) for compatibility, but the underlying
/// data can be GPU-accelerated for bulk operations.
///
/// # Usage
///
/// The evaluator is created for each row (or vector of rows in SIMD mode)
/// and accumulates constraint results with random coefficient weighting.
///
/// ```ignore
/// let eval = GpuDomainEvaluator::new(
///     &trace_eval,
///     vec_row,
///     &random_coeff_powers,
///     domain_log_size,
///     eval_log_size,
///     log_size,
///     claimed_sum,
/// );
/// let result = framework_eval.evaluate(eval);
/// ```
pub struct GpuDomainEvaluator<'a> {
    /// Reference to trace evaluations organized by interaction.
    pub trace_eval:
        &'a TreeVec<Vec<&'a CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>>>,

    /// Current column index for each interaction (for sequential column access).
    pub column_index_per_interaction: Vec<usize>,

    /// The vector row index to evaluate constraints at.
    /// In GPU/SIMD mode, each "row" is actually a vector of elements.
    pub vec_row: usize,

    /// Powers of the random coefficient for constraint accumulation.
    /// Used to combine multiple constraints into a single polynomial.
    pub random_coeff_powers: &'a [SecureField],

    /// Accumulated result for this row's constraint evaluation.
    pub row_res: VeryPackedSecureField,

    /// Current constraint index (for random coefficient selection).
    pub constraint_index: usize,

    /// Log size of the evaluation domain.
    pub domain_log_size: u32,

    /// Log size of the extended evaluation domain.
    pub eval_domain_log_size: u32,

    /// LogUp protocol state for this row.
    pub logup: LogupAtRow<Self>,
}

impl<'a> GpuDomainEvaluator<'a> {
    /// Create a new GPU domain evaluator.
    ///
    /// # Arguments
    ///
    /// * `trace_eval` - Trace evaluations organized by interaction
    /// * `vec_row` - The vector row index to evaluate
    /// * `random_coeff_powers` - Powers of random coefficient for accumulation
    /// * `domain_log_size` - Log2 of the trace domain size
    /// * `eval_log_size` - Log2 of the evaluation domain size
    /// * `log_size` - Log2 of the component size
    /// * `claimed_sum` - Claimed sum for LogUp protocol
    pub fn new(
        trace_eval: &'a TreeVec<Vec<&CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>>>,
        vec_row: usize,
        random_coeff_powers: &'a [SecureField],
        domain_log_size: u32,
        eval_log_size: u32,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace_eval,
            column_index_per_interaction: vec![0; trace_eval.len()],
            vec_row,
            random_coeff_powers,
            row_res: VeryPackedSecureField::zero(),
            constraint_index: 0,
            domain_log_size,
            eval_domain_log_size: eval_log_size,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}

impl EvalAtRow for GpuDomainEvaluator<'_> {
    type F = VeryPackedBaseField;
    type EF = VeryPackedSecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_index = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;

        offsets.map(|off| {
            // If the offset is 0, we can directly access the value at this row.
            if off == 0 {
                unsafe {
                    let col = &self
                        .trace_eval
                        .get_unchecked(interaction)
                        .get_unchecked(col_index)
                        .values;
                    // GpuBackend uses the same column types as SimdBackend
                    let very_packed_col = VeryPackedBaseColumn::transform_under_ref(col);
                    return *very_packed_col.data.get_unchecked(self.vec_row);
                };
            }

            // For non-zero offsets, we need to look up values at offset positions.
            // The domain is bit-reversed circle domain ordered, so we compute
            // the correct index for each element in the vector.
            VeryPackedBaseField::from_array(std::array::from_fn(|i| {
                let row_index = offset_bit_reversed_circle_domain_index(
                    (self.vec_row << (LOG_N_LANES + LOG_N_VERY_PACKED_ELEMS)) + i,
                    self.domain_log_size,
                    self.eval_domain_log_size,
                    off,
                );
                self.trace_eval[interaction][col_index].at(row_index)
            }))
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        // Accumulate constraint with random coefficient weighting
        self.row_res +=
            VeryPackedSecureField::broadcast(self.random_coeff_powers[self.constraint_index])
                * constraint;
        self.constraint_index += 1;
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        VeryPackedSecureField::from_very_packed_m31s(values)
    }

    crate::logup_proxy!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_lanes_constant() {
        // Verify LOG_N_LANES is a reasonable value (typically 4 for 128-bit SIMD)
        assert!(LOG_N_LANES >= 2);
        assert!(LOG_N_LANES <= 8);
    }

    #[test]
    fn test_very_packed_elems_constant() {
        // Verify LOG_N_VERY_PACKED_ELEMS is reasonable (unsigned, always >= 0)
        assert!(LOG_N_VERY_PACKED_ELEMS <= 4);
    }
}
