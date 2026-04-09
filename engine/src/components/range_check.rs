//! LogUp-based range check proving component.
//!
//! Proves that a set of values lies within `[min, max]` by constructing a
//! lookup table of valid entries and using STWO's LogUp protocol.
//! This is a 1D lookup (single element per relation entry), simpler than
//! the 2D activation lookup.

use stwo::core::air::Component;
use stwo::core::channel::MerkleChannel;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::core::verifier::verify as stwo_verify;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::prove;
use stwo::prover::CommitmentSchemeProver;

use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, RelationEntry,
    TraceLocationAllocator,
};

use crate::backend::convert_evaluations;
use crate::gadgets::range_check::{compute_range_multiplicities, RangeCheckConfig};

// Relation type for range check lookups: single element (the value).
stwo_constraint_framework::relation!(RangeCheckRelation, 1);

impl RangeCheckRelation {
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<1> {
        &self.0
    }
}

// Evaluator for range check lookup constraints.
//
// Column layout:
//   Tree 0 (preprocessed): 1 col — table values [0, 1, ..., 2^n - 1]
//   Tree 1 (execution):    2 cols — (value, multiplicity)
//   Tree 2 (interaction):  from finalize_logup_in_pairs (2 fracs → 1 batch)
#[derive(Debug, Clone)]
pub struct RangeCheckEval {
    pub log_n_rows: u32,
    pub lookup_elements: RangeCheckRelation,
    pub claimed_sum: SecureField,
}

impl FrameworkEval for RangeCheckEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Preprocessed column (table side): values [0, 1, ..., 2^n - 1]
        let table_value = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "range_check_table".into(),
        });

        // Execution trace columns
        let trace_value = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // LogUp: table side (yield with -multiplicity)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity.clone()),
            &[table_value],
        ));

        // LogUp: trace side (use with +1)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[trace_value],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

pub type RangeCheckComponent = FrameworkComponent<RangeCheckEval>;

#[derive(Debug, thiserror::Error)]
pub enum RangeCheckError {
    #[error("Value out of range at index {index}: {value:?} not in [{min}, {max}]")]
    ValueOutOfRange {
        index: usize,
        value: M31,
        min: u32,
        max: u32,
    },
    #[error("Proving error: {0}")]
    ProvingError(String),
    #[error("Verification error: {0}")]
    VerificationError(String),
}

#[derive(Debug)]
pub struct RangeCheckProof<H: MerkleHasherLifted> {
    pub stark_proof: StarkProof<H>,
    pub claimed_sum: SecureField,
    pub log_size: u32,
}

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// Prove that all `values` lie within `[config.min, config.max]`.
///
/// Returns a standalone STARK proof using Blake2sMerkleChannel.
pub fn prove_range_check(
    values: &[M31],
    config: &RangeCheckConfig,
) -> Result<RangeCheckProof<Blake2sHash>, RangeCheckError> {
    // Validate all values are in range
    for (i, v) in values.iter().enumerate() {
        if v.0 < config.min || v.0 > config.max {
            return Err(RangeCheckError::ValueOutOfRange {
                index: i,
                value: *v,
                min: config.min,
                max: config.max,
            });
        }
    }

    let log_size = config.log_size.max(4); // STWO minimum domain size
    let table_size = 1usize << log_size;

    // Build the preprocessed table column: [min, min+1, ..., max, pad...]
    let range_size = config.range_size() as usize;
    let mut table_values = Vec::with_capacity(table_size);
    for i in 0..range_size {
        table_values.push(M31::from(config.min + i as u32));
    }
    // Pad with first table entry
    let pad_value = table_values[0];
    while table_values.len() < table_size {
        table_values.push(pad_value);
    }

    // Compute multiplicities for the table
    let mut multiplicities = compute_range_multiplicities(values, config);
    // Pad multiplicities to table_size
    while multiplicities.len() < table_size {
        multiplicities.push(M31::from(0));
    }

    // Build SIMD columns
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Preprocessed: table values
    let mut table_col = Col::<SimdBackend, BaseField>::zeros(table_size);
    for (i, &v) in table_values.iter().enumerate() {
        table_col.set(i, v);
    }
    let preprocessed = vec![CircleEvaluation::new(domain, table_col.clone())];

    // Execution: (value, multiplicity)
    // The trace values mirror the table: row i → table entry i.
    // Table side yields -multiplicity[i], trace side uses +1.
    // LogUp sum cancels when multiplicities are correct.
    let mut trace_val_col = Col::<SimdBackend, BaseField>::zeros(table_size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(table_size);
    for (i, &v) in table_values.iter().enumerate() {
        trace_val_col.set(i, v);
    }
    for (i, &m) in multiplicities.iter().enumerate() {
        mult_col.set(i, m);
    }
    let execution = vec![
        CircleEvaluation::new(domain, trace_val_col.clone()),
        CircleEvaluation::new(domain, mult_col),
    ];

    let pcs_config = PcsConfig::default();

    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(pcs_config, &twiddles);

    // Tree 0: Preprocessed
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        preprocessed,
    ));
    tree_builder.commit(channel);

    // Tree 1: Execution
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        execution,
    ));
    tree_builder.commit(channel);

    // Draw lookup elements
    let lookup_elements: RangeCheckRelation = RangeCheckRelation::draw(channel);

    // Compute LogUp interaction trace
    let (interaction_trace, claimed_sum) = compute_range_check_logup_simd(
        &table_col,
        &trace_val_col,
        &multiplicities,
        log_size,
        &lookup_elements,
    );

    // Tree 2: Interaction
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, BaseField>(
        interaction_trace,
    ));
    tree_builder.commit(channel);

    // Build component
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        RangeCheckEval {
            log_n_rows: log_size,
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );

    let stark_proof =
        prove::<SimdBackend, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|e| RangeCheckError::ProvingError(format!("{e:?}")))?;

    Ok(RangeCheckProof {
        stark_proof,
        claimed_sum,
        log_size,
    })
}

/// Verify a standalone range check proof.
pub fn verify_range_check(
    proof: &RangeCheckProof<Blake2sHash>,
    config: &RangeCheckConfig,
    _num_values: usize,
) -> Result<(), RangeCheckError> {
    use stwo::core::pcs::CommitmentSchemeVerifier;

    let log_size = config.log_size.max(4);
    let pcs_config = PcsConfig::default();

    // Build a dummy component to get trace_log_degree_bounds
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(
        &mut allocator,
        RangeCheckEval {
            log_n_rows: log_size,
            lookup_elements: RangeCheckRelation::dummy(),
            claimed_sum: proof.claimed_sum,
        },
        proof.claimed_sum,
    );

    let bounds = Component::trace_log_degree_bounds(&dummy_component);
    let tree0 = bounds[0].clone();
    let tree1 = bounds[1].clone();
    let tree2 = bounds[2].clone();

    // Set up channel and verifier
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    // Replay commitments
    commitment_scheme.commit(proof.stark_proof.commitments[0], &tree0, channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &tree1, channel);

    // Draw lookup elements (same Fiat-Shamir state as prover)
    let lookup_elements: RangeCheckRelation = RangeCheckRelation::draw(channel);

    // Commit interaction trace
    commitment_scheme.commit(proof.stark_proof.commitments[2], &tree2, channel);

    // Build real component with drawn relation elements
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        RangeCheckEval {
            log_n_rows: log_size,
            lookup_elements,
            claimed_sum: proof.claimed_sum,
        },
        proof.claimed_sum,
    );

    // Verify
    stwo_verify::<Blake2sMerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        proof.stark_proof.clone(),
    )
    .map_err(|e| RangeCheckError::VerificationError(format!("{e:?}")))
}

/// Compute LogUp interaction trace for range check on SimdBackend.
///
/// Combines both fractions (table side: -mult/q_table, trace side: +1/q_trace)
/// into a single LogUp column. finalize_logup_in_pairs with 2 fracs → 1 batch
/// → exactly 1 new_col() call.
fn compute_range_check_logup_simd(
    table_col: &Col<SimdBackend, BaseField>,
    trace_val_col: &Col<SimdBackend, BaseField>,
    multiplicities: &[M31],
    log_size: u32,
    lookup_elements: &RangeCheckRelation,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    use stwo::prover::backend::simd::m31::LOG_N_LANES;

    let size = 1usize << log_size;
    let vec_size = size >> LOG_N_LANES;

    let mut logup_gen = LogupTraceGenerator::new(log_size);

    // Single combined fraction: (-mult/q_table) + (1/q_trace)
    // = (q_table - mult * q_trace) / (q_table * q_trace)
    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements
            .lookup_elements()
            .combine(&[table_col.data[vec_row]]);
        let q_trace: PackedSecureField = lookup_elements
            .lookup_elements()
            .combine(&[trace_val_col.data[vec_row]]);

        let mult_packed = mult_packed_at(multiplicities, vec_row);

        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;

        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

/// Helper: build a PackedSecureField from multiplicity slice at a SIMD vector row.
fn mult_packed_at(multiplicities: &[M31], vec_row: usize) -> PackedSecureField {
    use stwo::prover::backend::simd::m31::PackedBaseField;

    let n_lanes = 16usize;
    let base = vec_row * n_lanes;
    let mut vals = [M31::from(0); 16];
    for (i, val) in vals.iter_mut().enumerate() {
        let idx = base + i;
        if idx < multiplicities.len() {
            *val = multiplicities[idx];
        }
    }
    let packed_base = PackedBaseField::from_array(std::array::from_fn(|i| vals[i]));
    packed_base.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    #[test]
    fn test_range_check_prove_verify_uint8() {
        // 256-entry table [0..255], 100 values in range
        let config = RangeCheckConfig::uint8();
        let values: Vec<M31> = (0..100).map(|i| M31::from(i % 256)).collect();

        let proof = prove_range_check(&values, &config).expect("proving should succeed");

        verify_range_check(&proof, &config, values.len()).expect("verification should succeed");
    }

    #[test]
    fn test_range_check_prove_verify_custom() {
        // Small range [0, 15]
        let config = RangeCheckConfig::custom(0, 15);
        let values: Vec<M31> = vec![
            M31::from(0),
            M31::from(5),
            M31::from(10),
            M31::from(15),
            M31::from(7),
            M31::from(3),
            M31::from(12),
            M31::from(1),
        ];

        let proof = prove_range_check(&values, &config).expect("proving should succeed");

        verify_range_check(&proof, &config, values.len()).expect("verification should succeed");
    }

    #[test]
    fn test_range_check_out_of_range_rejected() {
        let config = RangeCheckConfig::custom(0, 15);
        let values = vec![M31::from(16)]; // out of range

        let result = prove_range_check(&values, &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            RangeCheckError::ValueOutOfRange { index: 0, .. } => {}
            e => panic!("Expected ValueOutOfRange, got: {e:?}"),
        }
    }

    #[test]
    fn test_range_check_tampered_fails() {
        let config = RangeCheckConfig::custom(0, 15);
        let values: Vec<M31> = (0..16).map(|i| M31::from(i)).collect();

        let mut proof = prove_range_check(&values, &config).expect("proving should succeed");

        // Tamper with the claimed sum
        proof.claimed_sum = proof.claimed_sum + SecureField::one();

        let result = verify_range_check(&proof, &config, values.len());
        assert!(result.is_err(), "Tampered proof should fail verification");
    }
}
