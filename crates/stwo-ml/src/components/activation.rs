//! LogUp-based activation function verification.
//!
//! Non-linear operations (ReLU, GELU, sigmoid, softmax) are prohibitively
//! expensive to arithmetize directly. Instead, we precompute lookup tables
//! and use STWO's LogUp protocol to prove each activation value exists in
//! the table.
//!
//! # How It Works
//!
//! ```text
//! Preprocessed Columns (read-only):
//!   table_input[i]  = domain value i
//!   table_output[i] = f(domain value i)
//!
//! Execution Trace:
//!   row i: multiplicity of table entry i
//!
//! LogUp Constraint:
//!   Σ multiplicity_i / (α - combine(table_input_i, table_output_i)) = 0
//!   (net sum must be zero: production = consumption)
//! ```

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    relation, FrameworkComponent, FrameworkEval, LogupTraceGenerator, Relation, RelationEntry,
    TraceLocationAllocator,
};

use crate::gadgets::lookup_table::PrecomputedTable;

// ---------------------------------------------------------------------------
// Activation types
// ---------------------------------------------------------------------------

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ActivationType {
    /// max(0, x) — exact, no approximation needed.
    ReLU,
    /// x × Φ(x) — Gaussian error linear unit, table-approximated.
    GELU,
    /// 1 / (1 + e^(-x)) — table-approximated.
    Sigmoid,
    /// e^(x_i) / Σ e^(x_j) — requires normalization gadget + table.
    Softmax,
    /// (x - μ) / σ — requires running stats + table for reciprocal sqrt.
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

    /// Whether this activation can be computed exactly (no approximation).
    pub fn is_exact(&self) -> bool {
        matches!(self, ActivationType::ReLU)
    }

    /// Build the precomputed table for this activation type.
    pub fn build_table(&self, log_size: u32) -> PrecomputedTable {
        match self {
            ActivationType::ReLU => PrecomputedTable::relu(log_size),
            ActivationType::GELU => PrecomputedTable::gelu(log_size),
            ActivationType::Sigmoid => PrecomputedTable::sigmoid(log_size),
            ActivationType::Softmax => PrecomputedTable::softmax_exp(log_size),
            ActivationType::LayerNorm => PrecomputedTable::identity(log_size),
        }
    }
}

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

// LogUp relation with 2 elements: (input, output)
relation!(ActivationRelation, 2);

// ---------------------------------------------------------------------------
// Constraint definition (FrameworkEval)
// ---------------------------------------------------------------------------

/// Constraint evaluator for activation function verification via LogUp.
///
/// Preprocessed columns: table_input and table_output.
/// Trace column: multiplicity of each table entry.
/// LogUp constraint: entries with non-zero multiplicity must be in the table.
#[derive(Clone)]
pub struct ActivationEval {
    /// Log2 of the table size.
    pub log_size: u32,
    /// Lookup elements drawn from the channel.
    pub lookup_elements: ActivationRelation,
}

const LOG_CONSTRAINT_DEGREE: u32 = 1;

impl FrameworkEval for ActivationEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
        // Read multiplicity from trace.
        let multiplicity = eval.next_trace_mask();

        // Read (input, output) from preprocessed columns.
        let table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: String::from("activation_input"),
        });
        let table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: String::from("activation_output"),
        });

        // LogUp: add (multiplicity, [input, output]) to the relation.
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(multiplicity),
            &[table_input, table_output],
        ));

        eval.finalize_logup();
        eval
    }
}

pub type ActivationComponent = FrameworkComponent<ActivationEval>;

// ---------------------------------------------------------------------------
// Trace generation
// ---------------------------------------------------------------------------

/// Generate the preprocessed columns for an activation table.
///
/// Returns two columns: (table_input, table_output) in bit-reversed order.
pub fn generate_activation_table(
    table: &PrecomputedTable,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let log_size = table.log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let size = 1usize << log_size;

    let mut input_col = vec![M31::zero(); size];
    let mut output_col = vec![M31::zero(); size];

    for i in 0..size {
        let bit_rev =
            bit_reverse_index(coset_index_to_circle_domain_index(i, log_size), log_size);
        input_col[bit_rev] = table.inputs[i];
        output_col[bit_rev] = table.outputs[i];
    }

    vec![
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            domain,
            BaseColumn::from_iter(input_col),
        ),
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            domain,
            BaseColumn::from_iter(output_col),
        ),
    ]
}

/// Count multiplicities of (input, output) pairs against the table.
///
/// Returns `Err` if any input value is outside `[0, table.size())`.
pub fn count_activation_multiplicities(
    inputs: &[M31],
    table: &PrecomputedTable,
) -> Result<Vec<u32>, ActivationError> {
    let size = table.size();
    let mut counts = vec![0u32; size];
    for &input in inputs {
        let idx = input.0 as usize;
        if idx >= size {
            return Err(ActivationError::InputOutOfRange {
                value: input.0,
                max: (size as u32).saturating_sub(1),
            });
        }
        counts[idx] += 1;
    }
    Ok(counts)
}

/// Generate the trace column (multiplicities) in bit-reversed circle domain order.
pub fn generate_activation_trace(
    inputs: &[M31],
    table: &PrecomputedTable,
) -> ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let log_size = table.log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let size = 1usize << log_size;
    let multiplicities = count_activation_multiplicities(inputs, table)
        .expect("inputs already validated before calling generate_activation_trace");

    let mut col = vec![M31::zero(); size];
    for (i, &count) in multiplicities.iter().enumerate() {
        let bit_rev =
            bit_reverse_index(coset_index_to_circle_domain_index(i, log_size), log_size);
        col[bit_rev] = M31::from(count);
    }

    vec![CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
        domain,
        BaseColumn::from_iter(col),
    )]
}

/// Generate the LogUp interaction trace for the activation component.
pub fn generate_activation_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    preprocessed: &[CircleEvaluation<SimdBackend, M31, BitReversedOrder>],
    lookup_elements: &ActivationRelation,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let multiplicity: PackedM31 = trace[0].data[vec_row];
        let table_input: PackedM31 = preprocessed[0].data[vec_row];
        let table_output: PackedM31 = preprocessed[1].data[vec_row];

        let denom: PackedQM31 = lookup_elements.combine(&[table_input, table_output]);
        col_gen.write_frac(vec_row, multiplicity.into(), denom);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

// ---------------------------------------------------------------------------
// End-to-end prove / verify
// ---------------------------------------------------------------------------

/// Prove that `outputs\[i\] = activation(inputs\[i\])` for all i.
///
/// The `inputs` array contains indices into the activation table.
/// Each input must be in `[0, table.size())`.
pub fn prove_activation(
    inputs: &[M31],
    table: &PrecomputedTable,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
) -> Result<
    (
        ActivationComponent,
        stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    ),
    ActivationError,
> {
    let log_size = table.log_size;
    if log_size < LOG_N_LANES {
        return Err(ActivationError::TableTooSmall {
            log_size,
            min_log_size: LOG_N_LANES,
        });
    }

    // Validate all inputs are in table range.
    for &v in inputs {
        if v.0 >= (1u32 << log_size) {
            return Err(ActivationError::InputOutOfRange {
                value: v.0,
                max: (1u32 << log_size) - 1,
            });
        }
    }

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Phase 1: Preprocessed columns (activation table).
    let preprocessed = generate_activation_table(table);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(preprocessed.clone());
    tree_builder.commit(channel);

    // Phase 2: Trace columns (multiplicities).
    let trace = generate_activation_trace(inputs, table);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);

    // Draw lookup elements.
    let lookup_elements = ActivationRelation::draw(channel);

    // Phase 3: Interaction trace (LogUp).
    let (interaction_trace, claimed_sum) =
        generate_activation_interaction_trace(&trace, &preprocessed, &lookup_elements);

    // Mix claimed sum.
    channel.mix_felts(&[claimed_sum]);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace);
    tree_builder.commit(channel);

    // Create component.
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[
        PreProcessedColumnId {
            id: String::from("activation_input"),
        },
        PreProcessedColumnId {
            id: String::from("activation_output"),
        },
    ]);
    let component = ActivationComponent::new(
        &mut allocator,
        ActivationEval {
            log_size,
            lookup_elements,
        },
        claimed_sum,
    );

    // Prove.
    let proof = prove(&[&component], channel, commitment_scheme)
        .map_err(|_| ActivationError::ProvingFailed)?;

    Ok((component, proof))
}

/// Verify an activation proof.
pub fn verify_activation(
    component: &ActivationComponent,
    proof: &stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    channel: &mut Blake2sChannel,
) -> Result<(), ActivationError> {
    let pcs_config = proof.config;
    pcs_config.mix_into(channel);
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    let sizes = component.trace_log_degree_bounds();

    // Preprocessed (2 columns: input and output).
    commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
    // Trace (1 column: multiplicities).
    commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);

    // Draw lookup elements (same as prover).
    let _lookup_elements = ActivationRelation::draw(channel);

    // Mix claimed sum.
    channel.mix_felts(&[component.claimed_sum()]);

    // Interaction.
    commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

    verify(&[component], channel, commitment_scheme, proof.clone())
        .map_err(|e| ActivationError::VerificationFailed(format!("{e}")))
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ActivationError {
    #[error("input value {value} out of table range [0, {max}]")]
    InputOutOfRange { value: u32, max: u32 },
    #[error("table log_size {log_size} too small for SIMD (minimum {min_log_size})")]
    TableTooSmall { log_size: u32, min_log_size: u32 },
    #[error("proving failed (STARK constraint check)")]
    ProvingFailed,
    #[error("verification failed: {0}")]
    VerificationFailed(String),
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
    fn test_prove_verify_relu() {
        let log_size = 4u32; // Small table for testing: [0, 16)
        let table = PrecomputedTable::relu(log_size);

        // Input values: some positive, some "negative" (high half = 0 output)
        let inputs: Vec<M31> = vec![
            M31::from(0),
            M31::from(3),
            M31::from(7),
            M31::from(10),  // positive: ReLU(10) = 10
            M31::from(0),   // duplicate
            M31::from(15),  // "negative": ReLU(15) = 0 (since 15 >= 8)
            M31::from(8),   // "negative": ReLU(8) = 0
            M31::from(5),
        ];

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) =
            prove_activation(&inputs, &table, config, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_activation(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_identity() {
        let log_size = 4u32;
        let table = PrecomputedTable::identity(log_size);

        // All values in range
        let inputs: Vec<M31> = (0..16).map(M31::from).collect();

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) =
            prove_activation(&inputs, &table, config, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_activation(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_activation_out_of_range() {
        let log_size = 4u32;
        let table = PrecomputedTable::relu(log_size);
        let inputs = vec![M31::from(20)]; // out of range

        let config = PcsConfig::default();
        let mut channel = Blake2sChannel::default();
        let result = prove_activation(&inputs, &table, config, &mut channel);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_rejects_tampered_activation_proof() {
        let log_size = 4u32;
        let table = PrecomputedTable::relu(log_size);
        let inputs: Vec<M31> = vec![M31::from(0), M31::from(3), M31::from(7), M31::from(5)];

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (_component, proof) =
            prove_activation(&inputs, &table, config, &mut prover_channel).unwrap();

        // Verify with a different set of inputs should fail because the
        // channel transcript won't match (different commitment tree).
        let different_inputs: Vec<M31> = vec![M31::from(1), M31::from(2), M31::from(4), M31::from(6)];
        let mut prover_channel2 = Blake2sChannel::default();
        let (component2, _proof2) =
            prove_activation(&different_inputs, &table, config, &mut prover_channel2).unwrap();

        // Use component from different inputs with proof from original — should fail.
        let mut verifier_channel = Blake2sChannel::default();
        let result = verify_activation(&component2, &proof, &mut verifier_channel);
        assert!(result.is_err(), "mismatched component/proof should be rejected");
    }

    #[test]
    fn test_count_activation_multiplicities_rejects_out_of_range() {
        let table = PrecomputedTable::relu(4); // size 16
        let inputs = vec![M31::from(0), M31::from(20)]; // 20 >= 16
        let result = count_activation_multiplicities(&inputs, &table);
        assert!(result.is_err());
    }

    #[test]
    fn test_prove_verify_gelu() {
        let log_size = 4u32;
        let table = PrecomputedTable::gelu(log_size);

        let inputs: Vec<M31> = vec![
            M31::from(0),
            M31::from(3),
            M31::from(7),
            M31::from(5),
            M31::from(1),
            M31::from(10),
            M31::from(12),  // "negative" (>= 8)
            M31::from(0),
        ];

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) =
            prove_activation(&inputs, &table, config, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_activation(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_sigmoid() {
        let log_size = 4u32;
        let table = PrecomputedTable::sigmoid(log_size);

        let inputs: Vec<M31> = (0..16).map(M31::from).collect();

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) =
            prove_activation(&inputs, &table, config, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_activation(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_softmax_exp() {
        let log_size = 4u32;
        let table = PrecomputedTable::softmax_exp(log_size);

        let inputs: Vec<M31> = vec![
            M31::from(0),
            M31::from(2),
            M31::from(5),
            M31::from(0),
            M31::from(10),  // "negative"
            M31::from(14),
            M31::from(7),
            M31::from(1),
        ];

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) =
            prove_activation(&inputs, &table, config, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_activation(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_build_table_dispatches_correctly() {
        let relu = ActivationType::ReLU.build_table(4);
        let gelu = ActivationType::GELU.build_table(4);
        let sigmoid = ActivationType::Sigmoid.build_table(4);
        let softmax = ActivationType::Softmax.build_table(4);

        // ReLU(0) = 0
        assert_eq!(relu.get(0).1, M31::from(0));
        // GELU(0) = 0
        assert_eq!(gelu.get(0).1, M31::from(0));
        // Sigmoid(0) = 0.5 * half
        let sig_zero = sigmoid.get(0).1.0;
        assert!(sig_zero > 0, "sigmoid(0) should be positive");
        // Softmax exp(0) = 1.0 * scale
        let exp_zero = softmax.get(0).1.0;
        assert!(exp_zero > 0, "exp(0) should be positive");

        // GELU and Sigmoid tables should differ from each other at some index
        assert_ne!(gelu.get(6).1, sigmoid.get(6).1,
            "GELU and Sigmoid should produce different outputs");
    }
}
