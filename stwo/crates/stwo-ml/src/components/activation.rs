//! LogUp-based activation function verification.
//!
//! Non-linear operations (ReLU, GELU, sigmoid, softmax) are prohibitively
//! expensive to arithmetize directly. Instead, we precompute lookup tables
//! and use STWO's LogUp protocol to prove each activation value exists in
//! the table.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo_constraint_framework::{
    FrameworkEval, FrameworkComponent,
    EvalAtRow, RelationEntry,
};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::gadgets::lookup_table::PrecomputedTable;

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    Softmax,
    LayerNorm,
}

impl ActivationType {
    /// Recommended lookup table size (log2) for this activation.
    pub fn recommended_table_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 16,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
            ActivationType::LayerNorm => 16,
        }
    }

    /// Production-scale table sizes for real models.
    pub fn production_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 18,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
            ActivationType::LayerNorm => 16,
        }
    }

    /// Whether this activation can be computed exactly (no approximation).
    pub fn is_exact(&self) -> bool {
        matches!(self, ActivationType::ReLU)
    }

    /// Build the activation function as a closure.
    pub fn as_fn(&self) -> Box<dyn Fn(M31) -> M31> {
        use crate::gadgets::lookup_table::activations;
        match self {
            ActivationType::ReLU => Box::new(activations::relu),
            ActivationType::GELU => Box::new(activations::gelu_approx),
            ActivationType::Sigmoid => Box::new(activations::sigmoid_approx),
            ActivationType::Softmax => Box::new(activations::relu),
            ActivationType::LayerNorm => Box::new(|x| x),
        }
    }
}

// Relation type for activation lookups (input, output).
stwo_constraint_framework::relation!(ActivationRelation, 2);

impl ActivationRelation {
    /// Access the inner LookupElements for computing LogUp fractions in the prover.
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<2> {
        &self.0
    }
}

/// Evaluator for activation lookup constraints.
#[derive(Debug, Clone)]
pub struct ActivationEval {
    pub log_n_rows: u32,
    pub lookup_elements: ActivationRelation,
    pub claimed_sum: SecureField,
    pub total_sum: SecureField,
}

impl FrameworkEval for ActivationEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read preprocessed columns (table input, table output)
        let table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "activation_table_input".into(),
        });
        let table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "activation_table_output".into(),
        });

        // Read execution trace columns
        let trace_input = eval.next_trace_mask();
        let trace_output = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // LogUp: table side (yield)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity.clone()),
            &[table_input, table_output],
        ));

        // LogUp: trace side (use)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[trace_input, trace_output],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

/// Type alias for the activation component.
pub type ActivationComponent = FrameworkComponent<ActivationEval>;

/// Generate the preprocessed activation table as trace columns.
pub fn generate_activation_table(
    table: &PrecomputedTable,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    table.generate_trace_columns()
}

/// Generate the main execution trace for activation lookups.
pub fn generate_activation_trace(
    inputs: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    assert_eq!(inputs.len(), outputs.len());
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (i, (&inp, &out)) in inputs.iter().zip(outputs.iter()).enumerate().take(size) {
        input_col.set(i, inp);
        output_col.set(i, out);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    vec![
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, mult_col),
    ]
}

/// Compute multiplicities for the activation table given the trace inputs.
pub fn compute_multiplicities(trace_inputs: &[M31], table: &PrecomputedTable) -> Vec<M31> {
    let mut multiplicities = vec![0u32; table.size()];

    for input in trace_inputs {
        if let Some(idx) = table.inputs.iter().position(|&x| x == *input) {
            multiplicities[idx] += 1;
        }
    }

    multiplicities.into_iter().map(M31::from).collect()
}

/// Error type for activation proving.
#[derive(Debug, thiserror::Error)]
pub enum ActivationError {
    #[error("Input value not found in table: {0:?}")]
    ValueNotInTable(M31),
    #[error("Input/output length mismatch: {0} inputs, {1} outputs")]
    LengthMismatch(usize, usize),
    #[error("Proving error: {0}")]
    ProvingError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_sizes() {
        assert_eq!(ActivationType::ReLU.recommended_table_log_size(), 16);
        assert_eq!(ActivationType::Softmax.recommended_table_log_size(), 20);
        assert!(ActivationType::ReLU.is_exact());
        assert!(!ActivationType::GELU.is_exact());
    }

    #[test]
    fn test_production_sizes() {
        assert_eq!(ActivationType::GELU.production_log_size(), 18);
        assert_eq!(ActivationType::Softmax.production_log_size(), 20);
    }

    #[test]
    fn test_generate_activation_table() {
        let table = PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::relu,
            4,
        );
        let cols = generate_activation_table(&table);
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn test_compute_multiplicities() {
        let table = PrecomputedTable::build(|x| x, 4);
        let inputs = vec![M31::from(0), M31::from(1), M31::from(0), M31::from(3)];
        let mults = compute_multiplicities(&inputs, &table);

        assert_eq!(mults[0], M31::from(2));
        assert_eq!(mults[1], M31::from(1));
        assert_eq!(mults[2], M31::from(0));
        assert_eq!(mults[3], M31::from(1));
    }

    #[test]
    fn test_generate_trace() {
        let inputs = vec![M31::from(1), M31::from(2)];
        let outputs = vec![M31::from(1), M31::from(2)];
        let mults = vec![M31::from(1), M31::from(1)];

        let trace = generate_activation_trace(&inputs, &outputs, &mults, 4);
        assert_eq!(trace.len(), 3);
    }
}
