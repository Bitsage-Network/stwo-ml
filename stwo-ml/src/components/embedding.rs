//! Embedding table lookup verification via LogUp.
//!
//! Proves that each token's embedding row was correctly looked up from the
//! embedding table. Uses the same LogUp protocol as activation lookups.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{FrameworkEval, FrameworkComponent, EvalAtRow, RelationEntry};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

// Relation for embedding lookup: (token_id, embed_col_idx, value).
stwo_constraint_framework::relation!(EmbeddingRelation, 3);

/// Evaluator for embedding lookup constraints.
///
/// Verifies that each (token_id, col, value) triple exists in the embedding table.
#[derive(Debug, Clone)]
pub struct EmbeddingEval {
    pub log_n_rows: u32,
    pub lookup_elements: EmbeddingRelation,
    pub claimed_sum: SecureField,
}

impl FrameworkEval for EmbeddingEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Preprocessed: embedding table flattened as (token_id, col_idx, value)
        let table_token = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "embed_table_token".into(),
        });
        let table_col = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "embed_table_col".into(),
        });
        let table_value = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "embed_table_value".into(),
        });

        // Execution trace: looked-up values
        let trace_token = eval.next_trace_mask();
        let trace_col = eval.next_trace_mask();
        let trace_value = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // LogUp: table side (yield)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[table_token, table_col, table_value],
        ));

        // LogUp: trace side (use)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[trace_token, trace_col, trace_value],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type EmbeddingComponent = FrameworkComponent<EmbeddingEval>;

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::m31::M31;

    #[test]
    fn test_embedding_eval_log_size() {
        // Just verify the eval constructs correctly
        let eval = EmbeddingEval {
            log_n_rows: 8,
            lookup_elements: EmbeddingRelation::dummy(),
            claimed_sum: SecureField::from(M31::from(0)),
        };
        assert_eq!(eval.log_size(), 8);
    }
}
