//! Layer normalization verification.
//!
//! LayerNorm: y = (x - μ) / σ × γ + β
//!
//! Decomposed into provable operations:
//! 1. Mean computation (sumcheck over input vector)
//! 2. Variance computation (sumcheck over squared differences)
//! 3. Reciprocal sqrt via lookup table (LogUp)
//! 4. Scale and shift (element-wise multiply-add)

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry};

use crate::gadgets::lookup_table::PrecomputedTable;

// Relation for reciprocal sqrt lookup.
stwo_constraint_framework::relation!(LayerNormRelation, 2);

impl LayerNormRelation {
    /// Access the inner LookupElements for computing LogUp fractions in the prover.
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<2> {
        &self.0
    }
}

/// LayerNorm configuration.
#[derive(Debug, Clone, Copy)]
pub struct LayerNormConfig {
    pub dim: usize,
    pub rsqrt_table_log_size: u32,
    pub epsilon: u32,
}

impl LayerNormConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            rsqrt_table_log_size: 16,
            epsilon: 1,
        }
    }
}

/// Evaluator for LayerNorm constraints.
#[derive(Debug, Clone)]
pub struct LayerNormEval {
    pub log_n_rows: u32,
    pub dim: usize,
    pub lookup_elements: LayerNormRelation,
    pub claimed_sum: SecureField,
    /// Instance ID — disambiguates preprocessed column names when multiple
    /// LayerNorm layers share the unified STARK. Defaults to 0.
    pub instance_id: usize,
}

impl FrameworkEval for LayerNormEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let table_var = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("layernorm_var_input_{}", self.instance_id).into(),
        });
        let table_rsqrt = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("layernorm_rsqrt_output_{}", self.instance_id).into(),
        });

        let input = eval.next_trace_mask();
        let mean = eval.next_trace_mask();
        let variance = eval.next_trace_mask();
        let rsqrt_val = eval.next_trace_mask();
        let output = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // Constraint: output = (input - mean) * rsqrt_val
        let centered = input - mean;
        eval.add_constraint(output - centered * rsqrt_val.clone());

        // LogUp: table side
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[table_var, table_rsqrt],
        ));

        // LogUp: trace side
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[variance, rsqrt_val],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type LayerNormComponent = FrameworkComponent<LayerNormEval>;

/// Build a reciprocal sqrt lookup table.
pub fn build_rsqrt_table(log_size: u32) -> PrecomputedTable {
    PrecomputedTable::build(
        |x| {
            let val = x.0;
            if val == 0 {
                M31::from((1u32 << 16) - 1)
            } else {
                let scale = 1u32 << 16;
                let sqrt_approx = (val as f64).sqrt();
                let rsqrt = (scale as f64 / sqrt_approx) as u32;
                M31::from(rsqrt.min((1u32 << 31) - 2))
            }
        },
        log_size,
    )
}

/// Generate LayerNorm execution trace columns.
///
/// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
pub fn generate_layernorm_trace<B: ColumnOps<BaseField>>(
    inputs: &[M31],
    means: &[M31],
    variances: &[M31],
    rsqrt_vals: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let cols_data: Vec<&[M31]> = vec![
        inputs,
        means,
        variances,
        rsqrt_vals,
        outputs,
        multiplicities,
    ];
    let mut result = Vec::with_capacity(6);

    for data in cols_data {
        let mut col = Col::<B, BaseField>::zeros(size);
        for (i, &val) in data.iter().enumerate().take(size) {
            col.set(i, val);
        }
        result.push(CircleEvaluation::new(domain, col));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_config() {
        let config = LayerNormConfig::new(768);
        assert_eq!(config.dim, 768);
        assert_eq!(config.rsqrt_table_log_size, 16);
    }

    #[test]
    fn test_rsqrt_table() {
        let table = build_rsqrt_table(4);
        assert_eq!(table.size(), 16);
        let rsqrt_1 = table.lookup(M31::from(1));
        assert!(rsqrt_1.is_some());
    }

    #[test]
    fn test_generate_trace() {
        use stwo::prover::backend::simd::SimdBackend;
        let n = 4;
        let inputs = vec![M31::from(10); n];
        let means = vec![M31::from(5); n];
        let variances = vec![M31::from(4); n];
        let rsqrt_vals = vec![M31::from(32768); n];
        let outputs = vec![M31::from(0); n];
        let mults = vec![M31::from(1); n];

        let trace = generate_layernorm_trace::<SimdBackend>(
            &inputs,
            &means,
            &variances,
            &rsqrt_vals,
            &outputs,
            &mults,
            4,
        );
        assert_eq!(trace.len(), 6);
    }
}
