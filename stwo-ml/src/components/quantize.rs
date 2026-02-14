//! Quantization verification via 2D LogUp STARK proof.
//!
//! Proves that each (input, output) pair in the quantize trace exists in a
//! precomputed lookup table, ensuring the prover applied the correct
//! `quantize_value(input, params)` formula. Mirrors `DequantizeEval`'s 2D pattern.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{FrameworkEval, FrameworkComponent, EvalAtRow, RelationEntry};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::gadgets::lookup_table::PrecomputedTable;
use crate::gadgets::quantize::{QuantParams, QuantStrategy, dequantize_value, quantize_value};

// Relation for quantize lookup: (input, output).
stwo_constraint_framework::relation!(QuantizeRelation, 2);

impl QuantizeRelation {
    /// Access the inner LookupElements for computing LogUp fractions in the prover.
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<2> {
        &self.0
    }
}

/// Evaluator for quantization 2D lookup constraints.
///
/// Verifies each (input, output) pair in the trace exists in the quantize
/// lookup table, proving the prover applied the correct quantization formula.
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
        // Preprocessed: quantize table (input, output)
        let table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "quantize_table_input".into(),
        });
        let table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "quantize_table_output".into(),
        });

        // Execution trace: (input, output, multiplicity)
        let trace_input = eval.next_trace_mask();
        let trace_output = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // LogUp: table side (yield with -multiplicity)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity.clone()),
            &[table_input, table_output],
        ));

        // LogUp: trace side (use with +1)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[trace_input, trace_output],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

pub type QuantizeComponent = FrameworkComponent<QuantizeEval>;

/// Build a quantize lookup table from quantization parameters and observed input values.
///
/// Creates (input_m31, output_m31) pairs where output = quantize_value(input_as_f32, params).
/// The table contains one entry per distinct input value observed in the data.
/// Inputs are interpreted as Direct-encoded f32 values (M31 → f32 via identity dequant).
pub fn build_quantize_table(params: &QuantParams, input_values: &[M31]) -> PrecomputedTable {
    let direct_params = QuantParams {
        strategy: QuantStrategy::Direct,
        scale: 1.0,
        zero_point: 0,
        bits: 31,
    };

    let mut seen = std::collections::HashSet::new();
    let pairs: Vec<(M31, M31)> = input_values
        .iter()
        .filter(|v| seen.insert(v.0))
        .map(|&inp| {
            let f32_val = dequantize_value(inp, &direct_params);
            let out = quantize_value(f32_val, params);
            (inp, out)
        })
        .collect();

    let log_size = if pairs.is_empty() {
        0
    } else {
        pairs.len().next_power_of_two().ilog2()
    };
    PrecomputedTable::from_pairs(pairs, log_size)
}

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

    #[test]
    fn test_build_quantize_table_int8() {
        let params = QuantParams {
            strategy: QuantStrategy::Symmetric8,
            scale: 1.0 / 127.0,
            zero_point: 127,
            bits: 8,
        };
        // Simulate some Direct-encoded inputs
        let inputs = vec![M31::from(0), M31::from(42), M31::from(100), M31::from(42)];
        let table = build_quantize_table(&params, &inputs);
        // Should have 3 distinct entries (0, 42, 100) — deduplicated
        assert_eq!(table.inputs.len().next_power_of_two(), 1 << table.log_size);
        // Verify each table entry
        let direct_params = QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 31,
        };
        for (&inp, &out) in table.inputs.iter().zip(table.outputs.iter()) {
            if inp == M31::from(0) && out == M31::from(0) {
                continue; // padding
            }
            let f32_val = dequantize_value(inp, &direct_params);
            let expected = quantize_value(f32_val, &params);
            assert_eq!(out, expected, "input={}, expected={}, got={}", inp.0, expected.0, out.0);
        }
    }

    #[test]
    fn test_build_quantize_table_int4() {
        let params = QuantParams {
            strategy: QuantStrategy::Symmetric4,
            scale: 1.0 / 7.0,
            zero_point: 7,
            bits: 4,
        };
        let inputs: Vec<M31> = (0..10).map(|i| M31::from(i)).collect();
        let table = build_quantize_table(&params, &inputs);
        assert!(table.size() >= 10);
    }

    #[test]
    fn test_build_quantize_table_empty() {
        let params = QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 31,
        };
        let table = build_quantize_table(&params, &[]);
        assert_eq!(table.size(), 1); // 2^0 = 1 (minimum)
    }
}
