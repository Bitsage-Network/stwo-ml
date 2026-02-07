//! Range check gadget for bounding ML values within valid ranges.
//!
//! Proves that a set of values are all within `[0, 2^bits)` using LogUp lookups
//! against a preprocessed table containing all allowed values.
//!
//! Used for:
//! - INT8 bounds: `[0, 256)` (8-bit range check)
//! - Activation input bounds
//! - Overflow prevention in M31 arithmetic

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

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

// LogUp relation with 1 element: (value,)
relation!(RangeCheckRelation, 1);

// ---------------------------------------------------------------------------
// Constraint definition (FrameworkEval)
// ---------------------------------------------------------------------------

/// Constraint evaluator for range check via LogUp.
///
/// Preprocessed column: all values `[0, 2^log_range_size)`.
/// Trace column 0: multiplicity of each table entry.
/// LogUp constraint: ensures table entries appear with correct multiplicities.
#[derive(Clone)]
pub struct RangeCheckEval {
    /// Log2 of the table size (= range size).
    pub log_size: u32,
    /// Lookup elements drawn from the channel.
    pub lookup_elements: RangeCheckRelation,
}

const LOG_CONSTRAINT_DEGREE: u32 = 1;

impl FrameworkEval for RangeCheckEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
        // Read the multiplicity from the trace.
        let multiplicity = eval.next_trace_mask();
        // Read the table value from the preprocessed column.
        let table_value = eval.get_preprocessed_column(PreProcessedColumnId {
            id: String::from("range_table"),
        });

        // LogUp: add (multiplicity, table_value) to the relation.
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(multiplicity),
            &[table_value],
        ));

        eval.finalize_logup();
        eval
    }
}

pub type RangeCheckComponent = FrameworkComponent<RangeCheckEval>;

// ---------------------------------------------------------------------------
// Trace generation
// ---------------------------------------------------------------------------

/// Generate the preprocessed table column: values [0, 2^log_size).
pub fn generate_range_table(
    log_size: u32,
) -> CircleEvaluation<SimdBackend, M31, BitReversedOrder> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let size = 1usize << log_size;
    let mut col = vec![M31::zero(); size];

    for i in 0..size {
        let bit_rev =
            bit_reverse_index(coset_index_to_circle_domain_index(i, log_size), log_size);
        col[bit_rev] = M31::from(i as u32);
    }

    CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
        domain,
        BaseColumn::from_iter(col),
    )
}

/// Count multiplicities: how many times each value in [0, 2^log_size) appears
/// in the input vector.
///
/// Returns `Err` if any value is outside `[0, 2^log_size)`.
pub fn count_multiplicities(values: &[M31], log_size: u32) -> Result<Vec<u32>, RangeCheckError> {
    let size = 1usize << log_size;
    let mut counts = vec![0u32; size];
    for &v in values {
        let idx = v.0 as usize;
        if idx >= size {
            return Err(RangeCheckError::ValueOutOfRange {
                value: v.0,
                max: (size as u32).saturating_sub(1),
            });
        }
        counts[idx] += 1;
    }
    Ok(counts)
}

/// Generate the trace column (multiplicities) in bit-reversed circle domain order.
pub fn generate_range_trace(
    values: &[M31],
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let size = 1usize << log_size;
    let multiplicities = count_multiplicities(values, log_size)
        .expect("values already validated before calling generate_range_trace");

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

/// Generate the LogUp interaction trace for the range check.
pub fn generate_range_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    preprocessed: &CircleEvaluation<SimdBackend, M31, BitReversedOrder>,
    lookup_elements: &RangeCheckRelation,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let multiplicity: PackedM31 = trace[0].data[vec_row];
        let table_value: PackedM31 = preprocessed.data[vec_row];

        let denom: PackedQM31 = lookup_elements.combine(&[table_value]);
        col_gen.write_frac(vec_row, multiplicity.into(), denom);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

// ---------------------------------------------------------------------------
// End-to-end prove / verify
// ---------------------------------------------------------------------------

/// Prove that all values in `input` are in `[0, 2^log_range_size)`.
pub fn prove_range_check(
    input: &[M31],
    log_range_size: u32,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
) -> Result<
    (
        RangeCheckComponent,
        stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    ),
    RangeCheckError,
> {
    if log_range_size < LOG_N_LANES {
        return Err(RangeCheckError::RangeTooSmall {
            log_size: log_range_size,
            min_log_size: LOG_N_LANES,
        });
    }

    // Validate all values are in range.
    for &v in input {
        if v.0 >= (1u32 << log_range_size) {
            return Err(RangeCheckError::ValueOutOfRange {
                value: v.0,
                max: (1u32 << log_range_size) - 1,
            });
        }
    }

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_range_size + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol.
    config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Phase 1: Preprocessed columns (range table).
    let preprocessed = generate_range_table(log_range_size);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(vec![preprocessed.clone()]);
    tree_builder.commit(channel);

    // Phase 2: Trace columns (multiplicities).
    let trace = generate_range_trace(input, log_range_size);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);

    // Draw lookup elements.
    let lookup_elements = RangeCheckRelation::draw(channel);

    // Phase 3: Interaction trace (LogUp).
    let (interaction_trace, claimed_sum) =
        generate_range_interaction_trace(&trace, &preprocessed, &lookup_elements);

    // Mix claimed sum.
    channel.mix_felts(&[claimed_sum]);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace);
    tree_builder.commit(channel);

    // Create component.
    let mut allocator =
        TraceLocationAllocator::new_with_preprocessed_columns(&[PreProcessedColumnId {
            id: String::from("range_table"),
        }]);
    let component = RangeCheckComponent::new(
        &mut allocator,
        RangeCheckEval {
            log_size: log_range_size,
            lookup_elements,
        },
        claimed_sum,
    );

    // Prove.
    let proof = prove(&[&component], channel, commitment_scheme)
        .map_err(|_| RangeCheckError::ProvingFailed)?;

    Ok((component, proof))
}

/// Verify a range check proof.
pub fn verify_range_check(
    component: &RangeCheckComponent,
    proof: &stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    channel: &mut Blake2sChannel,
) -> Result<(), RangeCheckError> {
    let pcs_config = proof.config;
    pcs_config.mix_into(channel);
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    let sizes = component.trace_log_degree_bounds();

    // Preprocessed.
    commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
    // Trace.
    commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);

    // Draw lookup elements (same as prover).
    let _lookup_elements = RangeCheckRelation::draw(channel);

    // Mix claimed sum.
    channel.mix_felts(&[component.claimed_sum()]);

    // Interaction.
    commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

    verify(&[component], channel, commitment_scheme, proof.clone())
        .map_err(|e| RangeCheckError::VerificationFailed(format!("{e}")))
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RangeCheckError {
    #[error("value {value} out of range [0, {max}]")]
    ValueOutOfRange { value: u32, max: u32 },
    #[error("range log_size {log_size} too small for SIMD (minimum {min_log_size})")]
    RangeTooSmall { log_size: u32, min_log_size: u32 },
    #[error("proving failed (STARK constraint check)")]
    ProvingFailed,
    #[error("verification failed: {0}")]
    VerificationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_multiplicities() {
        let values = vec![M31::from(0), M31::from(1), M31::from(1), M31::from(3)];
        let counts = count_multiplicities(&values, 3).unwrap(); // range [0, 8)
        assert_eq!(counts, vec![1, 2, 0, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_count_multiplicities_rejects_out_of_range() {
        let values = vec![M31::from(0), M31::from(20)]; // 20 >= 2^3 = 8
        let result = count_multiplicities(&values, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_prove_verify_range_check() {
        let log_range = 4u32; // range [0, 16)

        // All values in range.
        let input: Vec<M31> = (0..16).map(|i| M31::from(i % 16)).collect();

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) =
            prove_range_check(&input, log_range, config, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_range_check(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_range_check_out_of_range() {
        let log_range = 4u32; // range [0, 16)
        let input = vec![M31::from(20)]; // out of range

        let config = PcsConfig::default();
        let mut channel = Blake2sChannel::default();
        let result = prove_range_check(&input, log_range, config, &mut channel);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_rejects_tampered_range_proof() {
        let log_range = 4u32;
        // input_a: all zeros — multiplicity [16, 0, 0, ...]
        let input_a: Vec<M31> = vec![M31::from(0); 16];
        // input_b: one of each — multiplicity [1, 1, 1, ...]
        let input_b: Vec<M31> = (0..16).map(M31::from).collect();

        let config = PcsConfig::default();
        let mut ch_a = Blake2sChannel::default();
        let (_comp_a, proof_a) =
            prove_range_check(&input_a, log_range, config, &mut ch_a).unwrap();

        let mut ch_b = Blake2sChannel::default();
        let (comp_b, _proof_b) =
            prove_range_check(&input_b, log_range, config, &mut ch_b).unwrap();

        // Use component from input_b with proof from input_a — should fail
        // because the claimed_sums and commitment trees differ.
        let mut verifier_channel = Blake2sChannel::default();
        let result = verify_range_check(&comp_b, &proof_a, &mut verifier_channel);
        assert!(result.is_err(), "mismatched component/proof should be rejected");
    }
}
