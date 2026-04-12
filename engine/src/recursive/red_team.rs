//! Feature-gated red-team analysis helpers for the recursive AIR.
//!
//! This module is intentionally limited to synthetic trace construction and
//! AIR-level assertions. It does not build or serialize recursive proofs.

use num_traits::Zero;
use starknet_ff::FieldElement;
use stwo::core::fields::m31::BaseField as M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::TreeVec;
use stwo_constraint_framework::{assert_constraints_on_trace, FrameworkEval};

use super::air::{
    felt252_to_limbs, RecursiveTraceData, RecursiveVerifierEval, COLS_PER_ROW, COLS_PER_STATE,
    LIMBS_PER_FELT,
};

/// Synthetic trace specification used to demonstrate the current AIR statement.
#[derive(Debug, Clone)]
pub(crate) struct SyntheticRecursiveTraceSpec {
    /// Digest chain for the real rows. Must start at zero.
    ///
    /// If `digest_chain = [d0, d1, d2]`, row 0 proves `d0 -> d1` and row 1
    /// proves `d1 -> d2`.
    pub digest_chain: Vec<FieldElement>,
    /// Padded trace height as `2^log_size`.
    pub log_size: u32,
    /// Deterministic filler seed for columns that are currently unconstrained.
    pub unused_column_seed: u64,
}

impl SyntheticRecursiveTraceSpec {
    pub(crate) fn single_row(final_digest: FieldElement, unused_column_seed: u64) -> Self {
        Self {
            digest_chain: vec![FieldElement::ZERO, final_digest],
            log_size: 4,
            unused_column_seed,
        }
    }
}

/// Build a synthetic trace that satisfies the current recursive AIR statement.
///
/// The digest columns are wired consistently, while value/capacity/op_type
/// columns are populated from a deterministic PRNG to demonstrate that the AIR
/// does not constrain them.
pub(crate) fn build_synthetic_recursive_trace(
    spec: &SyntheticRecursiveTraceSpec,
) -> RecursiveTraceData {
    assert!(
        spec.digest_chain.len() >= 2,
        "digest_chain must contain at least one transition"
    );
    assert_eq!(
        spec.digest_chain[0],
        FieldElement::ZERO,
        "synthetic recursive traces must start from zero digest"
    );

    let n_real_rows = spec.digest_chain.len() - 1;
    let n_padded_rows = 1usize << spec.log_size;
    assert!(
        n_real_rows <= n_padded_rows,
        "digest_chain is too long for the requested log_size"
    );

    let mut rng = XorShift64::new(spec.unused_column_seed);
    let mut execution_trace = vec![vec![M31::zero(); n_padded_rows]; COLS_PER_ROW];
    for column in &mut execution_trace {
        for cell in column.iter_mut() {
            *cell = rng.next_m31();
        }
    }

    for row_idx in 0..n_real_rows {
        let digest_before_limbs = felt252_to_limbs(&spec.digest_chain[row_idx]);
        let digest_after_limbs = felt252_to_limbs(&spec.digest_chain[row_idx + 1]);

        for limb in 0..LIMBS_PER_FELT {
            execution_trace[limb][row_idx] = digest_before_limbs[limb];
            execution_trace[COLS_PER_STATE + limb][row_idx] = digest_after_limbs[limb];
        }

        if row_idx + 1 < n_real_rows {
            let next_before_limbs = felt252_to_limbs(&spec.digest_chain[row_idx + 1]);
            for limb in 0..LIMBS_PER_FELT {
                execution_trace[2 * COLS_PER_STATE + limb][row_idx] = next_before_limbs[limb];
            }
        }
    }

    let mut is_first = vec![M31::zero(); n_padded_rows];
    let mut is_last = vec![M31::zero(); n_padded_rows];
    let mut is_chain = vec![M31::zero(); n_padded_rows];
    is_first[0] = M31::from(1);
    is_last[n_real_rows - 1] = M31::from(1);
    for selector in is_chain.iter_mut().take(n_real_rows.saturating_sub(1)) {
        *selector = M31::from(1);
    }

    RecursiveTraceData {
        execution_trace,
        preprocessed_is_first: is_first,
        preprocessed_is_last: is_last,
        preprocessed_is_chain: is_chain,
        log_size: spec.log_size,
        n_real_rows,
        n_channel_ops: n_real_rows,
    }
}

pub(crate) fn current_recursive_air(
    log_size: u32,
    final_digest: FieldElement,
) -> RecursiveVerifierEval {
    RecursiveVerifierEval {
        log_n_rows: log_size,
        n_real_rows: 1 << log_size, // red team: assume full trace
        initial_digest_limbs: felt252_to_limbs(&FieldElement::ZERO),
        final_digest_limbs: felt252_to_limbs(&final_digest),
        hades_lookup: None,
    }
}

pub(crate) fn assert_trace_satisfies_current_recursive_air(trace: &RecursiveTraceData) {
    let air = current_recursive_air(trace.log_size, synthetic_trace_final_digest(trace));
    let preprocessed = vec![
        &trace.preprocessed_is_first,
        &trace.preprocessed_is_last,
        &trace.preprocessed_is_chain,
    ];
    let execution = trace.execution_trace.iter().collect::<Vec<_>>();
    let trees = TreeVec::new(vec![preprocessed, execution, vec![]]);

    assert_constraints_on_trace(
        &trees,
        trace.log_size,
        |row_eval| {
            let _ = air.evaluate(row_eval);
        },
        SecureField::zero(),
    );
}

fn synthetic_trace_final_digest(trace: &RecursiveTraceData) -> FieldElement {
    limbs_to_felt252(
        &trace.execution_trace[COLS_PER_STATE..COLS_PER_STATE + LIMBS_PER_FELT]
            .iter()
            .map(|column| column[trace.n_real_rows - 1])
            .collect::<Vec<_>>(),
    )
}

fn limbs_to_felt252(limbs: &[M31]) -> FieldElement {
    assert_eq!(limbs.len(), LIMBS_PER_FELT);
    let mut bytes = [0u8; 32];

    for (limb_idx, limb) in limbs.iter().enumerate() {
        let value = limb.0;
        let bit_offset = limb_idx * 28;
        for bit in 0..28 {
            if (value >> bit) & 1 == 1 {
                let absolute_bit = bit_offset + bit;
                let byte_index = 31 - absolute_bit / 8;
                let bit_in_byte = absolute_bit % 8;
                bytes[byte_index] |= 1 << bit_in_byte;
            }
        }
    }

    FieldElement::from_byte_slice_be(&bytes).expect("synthetic limbs should encode a valid felt")
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_m31(&mut self) -> M31 {
        M31::from_u32_unchecked((self.next_u64() as u32) & 0x7fff_ffff)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use super::*;
    use stwo_constraint_framework::InfoEvaluator;

    #[test]
    fn test_red_team_recursive_air_reports_current_shape() {
        let info = current_recursive_air(4, FieldElement::from(7u64)).evaluate(InfoEvaluator::new(
            4,
            vec![],
            SecureField::zero(),
        ));

        assert_eq!(info.n_constraints, 27);
        assert_eq!(info.preprocessed_columns.len(), 3);
        assert_eq!(info.mask_offsets[1].len(), COLS_PER_ROW);
        assert!(info.logup_counts.iter().next().is_none());
    }

    #[test]
    fn test_red_team_recursive_air_accepts_arbitrary_unused_columns() {
        let final_digest = FieldElement::from(0x12345u64);

        let trace_a = build_synthetic_recursive_trace(&SyntheticRecursiveTraceSpec::single_row(
            final_digest,
            1,
        ));
        let trace_b = build_synthetic_recursive_trace(&SyntheticRecursiveTraceSpec::single_row(
            final_digest,
            2,
        ));

        assert_trace_satisfies_current_recursive_air(&trace_a);
        assert_trace_satisfies_current_recursive_air(&trace_b);
    }

    #[test]
    fn test_red_team_recursive_air_ignores_value_capacity_and_op_type_columns() {
        let final_digest = FieldElement::from(0xBEEFu64);
        let mut trace = build_synthetic_recursive_trace(&SyntheticRecursiveTraceSpec::single_row(
            final_digest,
            11,
        ));
        assert_trace_satisfies_current_recursive_air(&trace);

        for column in 9..27 {
            trace.execution_trace[column][0] = M31::from(0x1234567);
        }
        for column in 36..54 {
            trace.execution_trace[column][0] = M31::from(0x0765432);
        }
        trace.execution_trace[63][0] = M31::from(1);

        assert_trace_satisfies_current_recursive_air(&trace);
    }

    #[test]
    fn test_red_team_recursive_air_rejects_wrong_final_digest_boundary() {
        let spec = SyntheticRecursiveTraceSpec::single_row(FieldElement::from(0xCAFEu64), 5);
        let trace = build_synthetic_recursive_trace(&spec);

        let wrong_air = current_recursive_air(trace.log_size, FieldElement::from(0xBADu64));
        let preprocessed = vec![
            &trace.preprocessed_is_first,
            &trace.preprocessed_is_last,
            &trace.preprocessed_is_chain,
        ];
        let execution = trace.execution_trace.iter().collect::<Vec<_>>();
        let trees = TreeVec::new(vec![preprocessed, execution, vec![]]);

        let result = catch_unwind(|| {
            assert_constraints_on_trace(
                &trees,
                trace.log_size,
                |row_eval| {
                    let _ = wrong_air.evaluate(row_eval);
                },
                SecureField::zero(),
            );
        });

        assert!(
            result.is_err(),
            "wrong final digest should violate a boundary constraint"
        );
    }

    #[test]
    fn test_red_team_recursive_air_rejects_broken_chain_digest() {
        let spec = SyntheticRecursiveTraceSpec {
            digest_chain: vec![
                FieldElement::ZERO,
                FieldElement::from(0x111u64),
                FieldElement::from(0x222u64),
            ],
            log_size: 4,
            unused_column_seed: 9,
        };
        let mut trace = build_synthetic_recursive_trace(&spec);
        assert_trace_satisfies_current_recursive_air(&trace);

        trace.execution_trace[2 * COLS_PER_STATE][0] += M31::from(1);

        let result = catch_unwind(|| assert_trace_satisfies_current_recursive_air(&trace));
        assert!(
            result.is_err(),
            "broken digest chaining should violate the chain constraint"
        );
    }
}
