//! LogUp-based dequantization verification.
//!
//! Proves that each quantized→dequantized mapping exists in a precomputed
//! lookup table. For INT4 the table has 16 entries, for INT8 it has 256.
//! Uses the same LogUp pattern as activation lookups (2D table: input, output).

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry};

use crate::gadgets::lookup_table::PrecomputedTable;
use crate::gadgets::quantize::{dequantize_value, quantize_value, QuantParams, QuantStrategy};

// Relation type for dequantize lookups: (quantized_input, dequantized_output).
stwo_constraint_framework::relation!(DequantizeRelation, 2);

impl DequantizeRelation {
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<2> {
        &self.0
    }
}

/// Evaluator for dequantize lookup constraints.
#[derive(Debug, Clone)]
pub struct DequantizeEval {
    pub log_n_rows: u32,
    pub lookup_elements: DequantizeRelation,
    pub claimed_sum: SecureField,
    /// Instance ID — disambiguates preprocessed column names when multiple
    /// dequantize layers share the unified STARK. Defaults to 0.
    pub instance_id: usize,
}

impl FrameworkEval for DequantizeEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Preprocessed columns (table side)
        let table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("dequantize_table_input_{}", self.instance_id).into(),
        });
        let table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("dequantize_table_output_{}", self.instance_id).into(),
        });

        // Execution trace columns
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

/// Type alias for the dequantize component.
pub type DequantizeComponent = FrameworkComponent<DequantizeEval>;

/// Build a dequantize lookup table from quantization parameters.
///
/// For INT4 (bits=4): 16 entries mapping q ∈ [0..15] → dequantize_to_m31(q, params).
/// For INT8 (bits=8): 256 entries mapping q ∈ [0..255] → dequantize_to_m31(q, params).
///
/// The dequantized value is: f32 = (q - zero_point) * scale, then re-encoded as M31
/// via the Direct strategy.
pub fn build_dequantize_table(params: &QuantParams) -> PrecomputedTable {
    let table_size = 1usize << params.bits;
    let direct_params = QuantParams {
        strategy: QuantStrategy::Direct,
        scale: 1.0,
        zero_point: 0,
        bits: 31,
    };

    let pairs: Vec<(M31, M31)> = (0..table_size)
        .map(|q| {
            let q_m31 = M31::from(q as u32);
            let deq_f32 = dequantize_value(q_m31, params);
            let deq_m31 = quantize_value(deq_f32, &direct_params);
            (q_m31, deq_m31)
        })
        .collect();

    PrecomputedTable::from_pairs(pairs, params.bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_dequantize_table_int4() {
        let params = QuantParams {
            strategy: QuantStrategy::Symmetric4,
            scale: 1.0 / 7.0,
            zero_point: 7,
            bits: 4,
        };
        let table = build_dequantize_table(&params);
        assert_eq!(table.size(), 16);

        // q=7 (zero point) should dequantize to 0
        let zp_idx = table
            .inputs
            .iter()
            .position(|&x| x == M31::from(7))
            .unwrap();
        let deq_val = dequantize_value(M31::from(7), &params);
        assert!(
            deq_val.abs() < 1e-6,
            "zero_point should map to ~0: {deq_val}"
        );

        // Table should have 16 distinct input entries
        let unique: std::collections::HashSet<u32> = table.inputs.iter().map(|m| m.0).collect();
        assert_eq!(unique.len(), 16);
    }

    #[test]
    fn test_build_dequantize_table_int8() {
        let params = QuantParams {
            strategy: QuantStrategy::Symmetric8,
            scale: 1.0 / 127.0,
            zero_point: 127,
            bits: 8,
        };
        let table = build_dequantize_table(&params);
        assert_eq!(table.size(), 256);

        // q=127 (zero point) should map to ~0
        let deq_val = dequantize_value(M31::from(127), &params);
        assert!(
            deq_val.abs() < 1e-6,
            "zero_point should map to ~0: {deq_val}"
        );
    }

    #[test]
    fn test_dequantize_table_consistency() {
        // Build table, then check that every table entry matches the dequantize function
        let params = QuantParams {
            strategy: QuantStrategy::Asymmetric4,
            scale: 0.2,
            zero_point: 0,
            bits: 4,
        };
        let table = build_dequantize_table(&params);

        for (i, (&inp, &out)) in table.inputs.iter().zip(table.outputs.iter()).enumerate() {
            let expected_f32 = dequantize_value(inp, &params);
            let expected_m31 = quantize_value(
                expected_f32,
                &QuantParams {
                    strategy: QuantStrategy::Direct,
                    scale: 1.0,
                    zero_point: 0,
                    bits: 31,
                },
            );
            assert_eq!(
                out, expected_m31,
                "Table entry {i}: input={}, expected output={}, got={}",
                inp.0, expected_m31.0, out.0,
            );
        }
    }
}
