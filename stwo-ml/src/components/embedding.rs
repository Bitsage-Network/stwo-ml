//! Embedding table lookup verification via LogUp.
//!
//! Proves that each token's embedding row was correctly looked up from the
//! embedding table. Uses the same LogUp protocol as activation lookups.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::{FrameworkEval, FrameworkComponent, EvalAtRow, RelationEntry};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::components::matmul::M31Matrix;

/// Perform embedding lookup: for each token_id in `token_ids`,
/// copy the corresponding row from the embedding table.
///
/// Returns `(output_matrix, flat_token_ids, flat_col_indices, flat_values, multiplicities)`.
/// - `output_matrix`: `(num_tokens, embed_dim)` — the embedding result.
/// - The four `Vec<M31>` are the LogUp trace columns for verification.
pub fn embedding_lookup(
    token_ids: &[u32],
    embedding_table: &M31Matrix, // (vocab_size, embed_dim)
) -> (M31Matrix, Vec<M31>, Vec<M31>, Vec<M31>, Vec<M31>) {
    let vocab_size = embedding_table.rows;
    let embed_dim = embedding_table.cols;
    let num_tokens = token_ids.len();

    let mut output = M31Matrix::new(num_tokens, embed_dim);
    let mut flat_token_ids = Vec::with_capacity(num_tokens * embed_dim);
    let mut flat_col_indices = Vec::with_capacity(num_tokens * embed_dim);
    let mut flat_values = Vec::with_capacity(num_tokens * embed_dim);
    let mut multiplicities = vec![M31::from(0); vocab_size * embed_dim];

    for (t, &tid) in token_ids.iter().enumerate() {
        let row_idx = (tid as usize).min(vocab_size - 1);
        for c in 0..embed_dim {
            let val = embedding_table.get(row_idx, c);
            output.set(t, c, val);

            flat_token_ids.push(M31::from(row_idx as u32));
            flat_col_indices.push(M31::from(c as u32));
            flat_values.push(val);

            // Track how many times each table cell is referenced
            multiplicities[row_idx * embed_dim + c] += M31::from(1);
        }
    }

    (output, flat_token_ids, flat_col_indices, flat_values, multiplicities)
}

/// Build the 3 preprocessed table columns for embedding LogUp:
/// `(table_tokens, table_cols, table_values)`.
///
/// Each row corresponds to one cell of the embedding table, flattened row-major.
pub fn build_embedding_table_columns(
    embedding_table: &M31Matrix,
) -> (Vec<M31>, Vec<M31>, Vec<M31>) {
    let vocab_size = embedding_table.rows;
    let embed_dim = embedding_table.cols;
    let total = vocab_size * embed_dim;

    let mut table_tokens = Vec::with_capacity(total);
    let mut table_cols = Vec::with_capacity(total);
    let mut table_values = Vec::with_capacity(total);

    for row in 0..vocab_size {
        for col in 0..embed_dim {
            table_tokens.push(M31::from(row as u32));
            table_cols.push(M31::from(col as u32));
            table_values.push(embedding_table.get(row, col));
        }
    }

    (table_tokens, table_cols, table_values)
}

// Relation for embedding lookup: (token_id, embed_col_idx, value).
stwo_constraint_framework::relation!(EmbeddingRelation, 3);

impl EmbeddingRelation {
    /// Access the inner LookupElements for computing LogUp fractions in the prover.
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<3> {
        &self.0
    }
}

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

    #[test]
    fn test_embedding_lookup_basic() {
        // 4-word vocab, embed_dim=3
        let mut table = M31Matrix::new(4, 3);
        for row in 0..4 {
            for col in 0..3 {
                table.set(row, col, M31::from((row * 10 + col + 1) as u32));
            }
        }

        let token_ids = vec![0, 2, 1, 2];
        let (output, tok_ids, col_ids, values, mults) = embedding_lookup(&token_ids, &table);

        // Output shape: (4 tokens, 3 embed_dim)
        assert_eq!(output.rows, 4);
        assert_eq!(output.cols, 3);

        // Token 0 → row 0: [1, 2, 3]
        assert_eq!(output.get(0, 0), M31::from(1));
        assert_eq!(output.get(0, 1), M31::from(2));
        assert_eq!(output.get(0, 2), M31::from(3));

        // Token 2 → row 2: [21, 22, 23]
        assert_eq!(output.get(1, 0), M31::from(21));
        assert_eq!(output.get(1, 1), M31::from(22));
        assert_eq!(output.get(1, 2), M31::from(23));

        // Trace columns: 4 tokens × 3 dims = 12 entries
        assert_eq!(tok_ids.len(), 12);
        assert_eq!(col_ids.len(), 12);
        assert_eq!(values.len(), 12);

        // Multiplicities: row 2 referenced 2× (tokens 1 and 3)
        // Each cell in row 2 has multiplicity 2
        for col in 0..3 {
            assert_eq!(mults[2 * 3 + col], M31::from(2));
        }
        // Row 0 referenced 1×
        for col in 0..3 {
            assert_eq!(mults[0 * 3 + col], M31::from(1));
        }
        // Row 3 never referenced
        for col in 0..3 {
            assert_eq!(mults[3 * 3 + col], M31::from(0));
        }
    }

    #[test]
    fn test_build_embedding_table_columns() {
        let mut table = M31Matrix::new(3, 2);
        table.set(0, 0, M31::from(10));
        table.set(0, 1, M31::from(11));
        table.set(1, 0, M31::from(20));
        table.set(1, 1, M31::from(21));
        table.set(2, 0, M31::from(30));
        table.set(2, 1, M31::from(31));

        let (tokens, cols, vals) = build_embedding_table_columns(&table);
        assert_eq!(tokens.len(), 6); // 3×2
        assert_eq!(tokens[0], M31::from(0)); // row 0
        assert_eq!(tokens[1], M31::from(0)); // row 0
        assert_eq!(tokens[2], M31::from(1)); // row 1
        assert_eq!(cols[0], M31::from(0));
        assert_eq!(cols[1], M31::from(1));
        assert_eq!(vals[0], M31::from(10));
        assert_eq!(vals[5], M31::from(31));
    }
}
