//! RMSNorm (Root Mean Square Layer Normalization) verification.
//!
//! RMSNorm: y = x / sqrt(mean(x²) + ε) × γ
//!
//! Decomposed into provable operations:
//! 1. RMS computation: rms² = sum(x²) / n
//! 2. Reciprocal sqrt via lookup table (LogUp)
//! 3. Scale: output = input × rsqrt(rms²)
//!
//! Key difference from LayerNorm: no mean subtraction.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry};

use crate::gadgets::lookup_table::PrecomputedTable;

// Relation for reciprocal sqrt lookup: (rms_squared, rsqrt_value).
stwo_constraint_framework::relation!(RMSNormRelation, 2);

impl RMSNormRelation {
    /// Access the inner LookupElements for computing LogUp fractions in the prover.
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<2> {
        &self.0
    }
}

/// RMSNorm configuration.
#[derive(Debug, Clone, Copy)]
pub struct RMSNormConfig {
    pub dim: usize,
    pub rsqrt_table_log_size: u32,
    pub epsilon: u32,
}

impl RMSNormConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            rsqrt_table_log_size: 16,
            epsilon: 1,
        }
    }
}

/// Evaluator for RMSNorm constraints.
///
/// Trace layout (5 columns):
///   Col 0: input       — original value x
///   Col 1: rms_sq      — mean(x²), shared per row
///   Col 2: rsqrt_val   — 1/sqrt(rms_sq), from lookup table
///   Col 3: output       — x × rsqrt_val
///   Col 4: multiplicity — LogUp multiplicity
///
/// Preprocessed (2 columns):
///   Col 0: rms_sq table input
///   Col 1: rsqrt table output
#[derive(Debug, Clone)]
pub struct RMSNormEval {
    pub log_n_rows: u32,
    pub dim: usize,
    pub lookup_elements: RMSNormRelation,
    pub claimed_sum: SecureField,
    /// Instance ID — disambiguates preprocessed column names when multiple
    /// RMSNorm layers share the unified STARK. Defaults to 0.
    pub instance_id: usize,
}

impl FrameworkEval for RMSNormEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let table_rms = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("rmsnorm_rms_input_{}", self.instance_id).into(),
        });
        let table_rsqrt = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("rmsnorm_rsqrt_output_{}", self.instance_id).into(),
        });

        let input = eval.next_trace_mask();
        let rms_sq = eval.next_trace_mask();
        let rsqrt_val = eval.next_trace_mask();
        let output = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // Constraint: output = input × rsqrt_val (no mean subtraction)
        eval.add_constraint(output - input * rsqrt_val.clone());

        // LogUp: table side
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[table_rms, table_rsqrt],
        ));

        // LogUp: trace side — proves (rms_sq, rsqrt_val) ∈ table
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[rms_sq, rsqrt_val],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type RMSNormComponent = FrameworkComponent<RMSNormEval>;

/// Build a reciprocal sqrt lookup table (shared with LayerNorm).
///
/// Table: rms_sq → rsqrt(rms_sq), with fixed-point scale 2^16.
pub fn build_rsqrt_table(log_size: u32) -> PrecomputedTable {
    // Reuse the exact same table as LayerNorm — rsqrt is rsqrt regardless.
    crate::components::layernorm::build_rsqrt_table(log_size)
}

/// Generate RMSNorm execution trace columns (5 columns).
///
/// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
///
/// Padding rows use the first rsqrt table entry for `rms_sq` and `rsqrt_val`
/// so the LogUp evaluator sees valid `(rms_sq, rsqrt_val)` pairs in the table.
pub fn generate_rmsnorm_trace<B: ColumnOps<BaseField>>(
    inputs: &[M31],
    rms_sq_vals: &[M31],
    rsqrt_vals: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    log_size: u32,
    rsqrt_table: &PrecomputedTable,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let pad_rms = rsqrt_table.inputs.first().copied().unwrap_or(M31::from(0));
    let pad_rsqrt = rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));

    let n = inputs.len().min(size);

    let mut input_col = Col::<B, BaseField>::zeros(size);
    let mut rms_col = Col::<B, BaseField>::zeros(size);
    let mut rsqrt_col = Col::<B, BaseField>::zeros(size);
    let mut output_col = Col::<B, BaseField>::zeros(size);
    let mut mult_col = Col::<B, BaseField>::zeros(size);

    for i in 0..n {
        input_col.set(i, inputs[i]);
        rms_col.set(i, rms_sq_vals[i]);
        rsqrt_col.set(i, rsqrt_vals[i]);
        output_col.set(i, outputs[i]);
    }
    // Pad rms_sq and rsqrt with first table entry so LogUp sees valid pairs.
    for i in n..size {
        rms_col.set(i, pad_rms);
        rsqrt_col.set(i, pad_rsqrt);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    vec![
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, rms_col),
        CircleEvaluation::new(domain, rsqrt_col),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, mult_col),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_config() {
        let config = RMSNormConfig::new(768);
        assert_eq!(config.dim, 768);
        assert_eq!(config.rsqrt_table_log_size, 16);
    }

    #[test]
    fn test_rsqrt_table_shared() {
        let table = build_rsqrt_table(4);
        assert_eq!(table.size(), 16);
        let rsqrt_1 = table.lookup(M31::from(1));
        assert!(rsqrt_1.is_some());
        // rsqrt(1) = 2^16 / sqrt(1) = 65536
        assert_eq!(rsqrt_1.unwrap(), M31::from(65536));
    }

    #[test]
    fn test_generate_rmsnorm_trace() {
        use stwo::prover::backend::simd::SimdBackend;
        let n = 4;
        let inputs = vec![M31::from(10); n];
        let rms_sq = vec![M31::from(100); n]; // mean(x²) = 100
        let rsqrt_vals = vec![M31::from(6554); n]; // rsqrt(100) ≈ 6554
        let outputs = vec![M31::from(0); n];
        let mults = vec![M31::from(1); n];

        let table = build_rsqrt_table(4);
        let trace = generate_rmsnorm_trace::<SimdBackend>(
            &inputs,
            &rms_sq,
            &rsqrt_vals,
            &outputs,
            &mults,
            4,
            &table,
        );
        // 5 columns: input, rms_sq, rsqrt, output, multiplicity
        assert_eq!(trace.len(), 5);
    }
}
