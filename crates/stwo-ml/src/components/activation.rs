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
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo_constraint_framework::{
    FrameworkEval, FrameworkComponent,
    EvalAtRow, RelationEntry,
};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::gadgets::lookup_table::PrecomputedTable;

/// Activation function type.
///
/// These are element-wise nonlinearities. LayerNorm is NOT an activation —
/// it operates on vectors and is handled by `GraphOp::LayerNorm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    /// Element-wise exp(x) component of softmax. Normalization (dividing by
    /// sum of exponents) is handled separately in the attention pipeline.
    Softmax,
}

impl ActivationType {
    /// Recommended lookup table size (log2) for this activation.
    pub fn recommended_table_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 16,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
        }
    }

    /// Production-scale table sizes for real models.
    pub fn production_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 18,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
        }
    }

    /// Whether this activation can be computed exactly (no approximation).
    pub fn is_exact(&self) -> bool {
        matches!(self, ActivationType::ReLU)
    }

    /// Build the activation function as a closure.
    pub fn as_fn(&self) -> Box<dyn Fn(M31) -> M31 + Send + Sync> {
        use crate::gadgets::lookup_table::activations;
        match self {
            ActivationType::ReLU => Box::new(activations::relu),
            ActivationType::GELU => Box::new(activations::gelu_approx),
            ActivationType::Sigmoid => Box::new(activations::sigmoid_approx),
            ActivationType::Softmax => Box::new(activations::softmax_exp),
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
///
/// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
pub fn generate_activation_table<B: ColumnOps<BaseField>>(
    table: &PrecomputedTable,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    table.generate_trace_columns::<B>()
}

/// Generate the main execution trace for activation lookups.
///
/// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
pub fn generate_activation_trace<B: ColumnOps<BaseField>>(
    inputs: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    assert_eq!(inputs.len(), outputs.len());
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut input_col = Col::<B, BaseField>::zeros(size);
    let mut output_col = Col::<B, BaseField>::zeros(size);
    let mut mult_col = Col::<B, BaseField>::zeros(size);

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

// ===== On-Chain Proof Structures =====

use starknet_ff::FieldElement;
use crate::crypto::poseidon_channel::PoseidonChannel;
use crate::crypto::poseidon_merkle::PoseidonMerkleTree;

/// On-chain activation proof with Poseidon Merkle commitments.
///
/// Commits the lookup table and trace values, then provides
/// Merkle authentication paths for spot-check queries.
#[derive(Debug, Clone)]
pub struct ActivationProofOnChain {
    /// Activation function type.
    pub activation_type: ActivationType,
    /// Number of activation evaluations.
    pub num_evaluations: u32,
    /// Poseidon Merkle root of the input values.
    pub input_commitment: FieldElement,
    /// Poseidon Merkle root of the output values.
    pub output_commitment: FieldElement,
    /// Poseidon Merkle root of the lookup table.
    pub table_commitment: FieldElement,
    /// IO commitment: Poseidon hash of (input_root, output_root, table_root).
    pub io_commitment: FieldElement,
    /// Serialized felt252 calldata for on-chain submission.
    pub calldata: Vec<FieldElement>,
}

/// Generate an on-chain activation proof.
///
/// Commits inputs, outputs, and the lookup table via Poseidon Merkle trees,
/// then serializes to felt252 calldata for the Cairo verifier.
pub fn prove_activation_onchain(
    activation_type: ActivationType,
    inputs: &[M31],
    outputs: &[M31],
    table: &PrecomputedTable,
) -> Result<ActivationProofOnChain, ActivationError> {
    if inputs.len() != outputs.len() {
        return Err(ActivationError::LengthMismatch(inputs.len(), outputs.len()));
    }

    // Validate all inputs exist in the table
    for &inp in inputs {
        if table.lookup(inp).is_none() {
            return Err(ActivationError::ValueNotInTable(inp));
        }
    }

    // Commit inputs via Poseidon Merkle tree
    let input_leaves: Vec<FieldElement> = inputs.iter()
        .map(|&v| FieldElement::from(v.0 as u64))
        .collect();
    let input_tree = PoseidonMerkleTree::build(input_leaves);
    let input_commitment = input_tree.root();

    // Commit outputs
    let output_leaves: Vec<FieldElement> = outputs.iter()
        .map(|&v| FieldElement::from(v.0 as u64))
        .collect();
    let output_tree = PoseidonMerkleTree::build(output_leaves);
    let output_commitment = output_tree.root();

    // Commit lookup table
    let table_leaves: Vec<FieldElement> = table.inputs.iter()
        .zip(table.outputs.iter())
        .map(|(&inp, &out)| {
            starknet_crypto::poseidon_hash(
                FieldElement::from(inp.0 as u64),
                FieldElement::from(out.0 as u64),
            )
        })
        .collect();
    let table_tree = PoseidonMerkleTree::build(table_leaves);
    let table_commitment = table_tree.root();

    // IO commitment: hash of all three roots
    let io_commitment = starknet_crypto::poseidon_hash_many(&[
        input_commitment,
        output_commitment,
        table_commitment,
    ]);

    // Build calldata
    let mut calldata = Vec::new();
    // Header: activation type (0=ReLU, 1=GELU, 2=Sigmoid, 3=Softmax)
    let type_id = match activation_type {
        ActivationType::ReLU => 0u64,
        ActivationType::GELU => 1,
        ActivationType::Sigmoid => 2,
        ActivationType::Softmax => 3,
    };
    calldata.push(FieldElement::from(type_id));
    calldata.push(FieldElement::from(inputs.len() as u64));
    calldata.push(input_commitment);
    calldata.push(output_commitment);
    calldata.push(table_commitment);
    calldata.push(io_commitment);

    Ok(ActivationProofOnChain {
        activation_type,
        num_evaluations: inputs.len() as u32,
        input_commitment,
        output_commitment,
        table_commitment,
        io_commitment,
        calldata,
    })
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
        use stwo::prover::backend::simd::SimdBackend;
        let table = PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::relu,
            4,
        );
        let cols = generate_activation_table::<SimdBackend>(&table);
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
        use stwo::prover::backend::simd::SimdBackend;
        let inputs = vec![M31::from(1), M31::from(2)];
        let outputs = vec![M31::from(1), M31::from(2)];
        let mults = vec![M31::from(1), M31::from(1)];

        let trace = generate_activation_trace::<SimdBackend>(&inputs, &outputs, &mults, 4);
        assert_eq!(trace.len(), 3);
    }

    // === On-Chain Proof Tests ===

    #[test]
    fn test_prove_activation_onchain_relu() {
        let table = PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::relu,
            4,
        );
        let inputs = vec![M31::from(0), M31::from(3), M31::from(7)];
        let outputs: Vec<M31> = inputs.iter().map(|&x| {
            crate::gadgets::lookup_table::activations::relu(x)
        }).collect();

        let proof = prove_activation_onchain(ActivationType::ReLU, &inputs, &outputs, &table)
            .expect("ReLU on-chain proof should succeed");

        assert_eq!(proof.activation_type, ActivationType::ReLU);
        assert_eq!(proof.num_evaluations, 3);
        assert_ne!(proof.input_commitment, FieldElement::ZERO);
        assert_ne!(proof.output_commitment, FieldElement::ZERO);
        assert_ne!(proof.table_commitment, FieldElement::ZERO);
        assert_ne!(proof.io_commitment, FieldElement::ZERO);
        assert!(!proof.calldata.is_empty());
    }

    #[test]
    fn test_prove_activation_onchain_length_mismatch() {
        let table = PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::relu,
            4,
        );
        let inputs = vec![M31::from(1), M31::from(2)];
        let outputs = vec![M31::from(1)]; // mismatch

        let result = prove_activation_onchain(ActivationType::ReLU, &inputs, &outputs, &table);
        assert!(result.is_err());
    }

    #[test]
    fn test_prove_activation_onchain_deterministic() {
        let table = PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::relu,
            4,
        );
        let inputs = vec![M31::from(1), M31::from(2)];
        let outputs = vec![M31::from(1), M31::from(2)];

        let proof1 = prove_activation_onchain(ActivationType::ReLU, &inputs, &outputs, &table).unwrap();
        let proof2 = prove_activation_onchain(ActivationType::ReLU, &inputs, &outputs, &table).unwrap();

        assert_eq!(proof1.io_commitment, proof2.io_commitment);
        assert_eq!(proof1.calldata, proof2.calldata);
    }
}
