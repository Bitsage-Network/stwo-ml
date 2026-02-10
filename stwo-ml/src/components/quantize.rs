//! Quantization verification via range-check STARK proof.
//!
//! Proves that quantized values lie within the valid range [0, 2^bits)
//! using LogUp over a range table. Reuses the range_check gadget infrastructure.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{FrameworkEval, FrameworkComponent, EvalAtRow, RelationEntry};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

// Relation for range-check lookup: (value,).
stwo_constraint_framework::relation!(QuantizeRelation, 1);

/// Evaluator for quantization range-check constraints.
///
/// Verifies each quantized value exists in the range table [0, 2^bits).
#[derive(Debug, Clone)]
pub struct QuantizeEval {
    pub log_n_rows: u32,
    pub lookup_elements: QuantizeRelation,
    pub claimed_sum: SecureField,
}

impl FrameworkEval for QuantizeEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Preprocessed: range table
        let table_value = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "quantize_range_table".into(),
        });

        // Execution trace
        let trace_value = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // LogUp: table side (yield)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[table_value],
        ));

        // LogUp: trace side (use)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[trace_value],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type QuantizeComponent = FrameworkComponent<QuantizeEval>;

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::m31::M31;

    #[test]
    fn test_quantize_eval_log_size() {
        let eval = QuantizeEval {
            log_n_rows: 8,
            lookup_elements: QuantizeRelation::dummy(),
            claimed_sum: SecureField::from(M31::from(0)),
        };
        assert_eq!(eval.log_size(), 8);
    }
}
