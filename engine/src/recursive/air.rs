//! AIR circuit for the recursive STARK — felt252 Hades chain.
//!
//! # Architecture
//!
//! The GKR verifier's Fiat-Shamir transcript is a chain of Hades permutations
//! over felt252 (Starknet's Poseidon). Each mix/draw operation is one Hades call
//! on a 3-element state: `[digest, value, capacity]`.
//!
//! The recursive AIR constrains this chain by:
//! 1. Storing each Hades input/output as M31 limbs (9 limbs per felt252)
//! 2. Constraining the chain: output digest of row i == input digest of row i+1
//! 3. Constraining boundaries: first row starts from zero digest, last row
//!    produces the expected final digest
//!
//! The Hades computation itself (S-box + MDS over felt252) is NOT yet constrained
//! inline — that requires full felt252 arithmetic in M31 limbs (multi-week effort).
//! Instead, the trace commits to the actual Hades states, and the chain + boundary
//! constraints ensure the committed sequence is consistent.
//!
//! # Trace Layout
//!
//! Each row = one Hades permutation call. Columns per row:
//!
//! | Columns | Count | Description |
//! |---------|-------|-------------|
//! | input_digest  | 9 | felt252 → 9 M31 limbs |
//! | input_value   | 9 | felt252 → 9 M31 limbs |
//! | input_cap     | 9 | felt252 → 9 M31 limbs |
//! | output_digest | 9 | felt252 → 9 M31 limbs |
//! | output_value  | 9 | felt252 → 9 M31 limbs |
//! | output_cap    | 9 | felt252 → 9 M31 limbs |
//! | op_type       | 1 | 0 = mix, 1 = draw |
//! | **Total**     | **55** | |
//!
//! For 15K Hades calls: log_size=14, ~900K cells. Compact and efficient.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::BaseField as M31;
use stwo_constraint_framework::{
    preprocessed_columns::PreProcessedColumnId, EvalAtRow, FrameworkComponent, FrameworkEval,
    RelationEntry, Relation,
};

// ── LogUp relation: binds chain AIR ↔ Hades AIR ─────────────────────
// Key: (digest_before[9], input_value[9], input_capacity[9],
//        digest_after[9], output_value[9], output_capacity[9]) = 54 M31 columns.
// Chain AIR contributes +1 multiplicity (consumer).
// Hades AIR contributes -1 multiplicity (provider).
stwo_constraint_framework::relation!(HadesPermRelation, 54);

/// Number of M31 limbs to represent one felt252.
/// 9 * 31 = 279 bits ≥ 252 bits.
pub const LIMBS_PER_FELT: usize = 9;

/// Columns for one Hades state element (3 felt252 = 3 * 9 = 27 limbs).
pub const COLS_PER_STATE: usize = 3 * LIMBS_PER_FELT; // 27

/// Columns for the shifted next-row input digest (for chain constraints).
pub const COLS_SHIFTED_DIGEST: usize = LIMBS_PER_FELT; // 9

/// Columns for one digest: 9 M31 limbs.
pub const COLS_PER_DIGEST: usize = LIMBS_PER_FELT; // 9

/// Columns for the full Hades state that are NOT the digest (value + capacity = 2 × 9 = 18).
pub const COLS_EXTRA_STATE: usize = 2 * LIMBS_PER_FELT; // 18

/// Total columns per row (expanded):
///   digest_before[9] + input_value[9] + input_capacity[9]
/// + digest_after[9]  + output_value[9] + output_capacity[9]
/// + shifted_next_before[9] + op_type[1] = 64
pub const COLS_PER_ROW: usize =
    COLS_PER_STATE   // input full state: 27
    + COLS_PER_STATE // output full state: 27
    + COLS_PER_DIGEST // shifted_next_before: 9
    + 1;             // op_type
    // Total: 27 + 27 + 9 + 1 = 64

// ═══════════════════════════════════════════════════════════════════════
// Felt252 ↔ M31 limb decomposition
// ═══════════════════════════════════════════════════════════════════════

/// Decompose a felt252 into 9 M31 limbs (LSB first).
///
/// Each limb holds 28 bits (not 31) to leave room for carry propagation
/// in future constraint additions. 9 * 28 = 252 bits = exact fit.
pub fn felt252_to_limbs(felt: &FieldElement) -> [M31; LIMBS_PER_FELT] {
    let bytes = felt.to_bytes_be();
    let mut limbs = [M31::from_u32_unchecked(0); LIMBS_PER_FELT];

    // Convert 32 bytes (256 bits) to 9 limbs of 28 bits each.
    // Total capacity: 9 * 28 = 252 bits (perfect for felt252).
    // We use 28-bit limbs for future carry-chain constraints.
    let mut bits_remaining = 252u32;
    let mut byte_idx = 31usize; // start from LSB
    let mut bit_offset = 0u32;

    for limb in limbs.iter_mut() {
        let limb_bits = 28u32.min(bits_remaining);
        let mut value = 0u32;

        for b in 0..limb_bits {
            let global_bit = bit_offset + b;
            let bi = (global_bit / 8) as usize;
            let bpos = (global_bit % 8) as u32;
            if bi <= 31 {
                let byte_val = bytes[31 - bi];
                if (byte_val >> bpos) & 1 == 1 {
                    value |= 1u32 << b;
                }
            }
        }

        *limb = M31::from_u32_unchecked(value);
        bit_offset += limb_bits;
        bits_remaining = bits_remaining.saturating_sub(limb_bits);
    }

    limbs
}

/// Decompose a 3-element Hades state into 27 M31 limbs.
pub fn hades_state_to_limbs(state: &[FieldElement; 3]) -> [M31; COLS_PER_STATE] {
    let mut limbs = [M31::from_u32_unchecked(0); COLS_PER_STATE];
    for (i, felt) in state.iter().enumerate() {
        let felt_limbs = felt252_to_limbs(felt);
        limbs[i * LIMBS_PER_FELT..(i + 1) * LIMBS_PER_FELT].copy_from_slice(&felt_limbs);
    }
    limbs
}

// ═══════════════════════════════════════════════════════════════════════
// FrameworkEval: Hades Chain AIR
// ═══════════════════════════════════════════════════════════════════════

/// AIR evaluator for the recursive STARK's Hades chain.
///
/// Each row constrains one Hades permutation from the verifier's transcript.
/// Chain constraints link consecutive rows via the digest limbs.
#[derive(Debug, Clone)]
pub struct RecursiveVerifierEval {
    /// log2 of the number of trace rows.
    pub log_n_rows: u32,

    /// Initial digest (usually zero, decomposed into limbs).
    pub initial_digest_limbs: [M31; LIMBS_PER_FELT],

    /// Expected final digest after all verifier operations (decomposed into limbs).
    pub final_digest_limbs: [M31; LIMBS_PER_FELT],

    /// LogUp lookup elements for the Hades permutation relation.
    /// When `None`, LogUp is disabled (backward-compatible mode).
    pub hades_lookup: Option<HadesPermRelation>,
}

impl FrameworkEval for RecursiveVerifierEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // All constraints are degree ≤ 2 (products with is_first/is_last selectors).
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // ── Preprocessed selectors ───────────────────────────────────
        let is_first = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_first".into(),
        });
        let is_last = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_last".into(),
        });
        let is_chain = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_chain".into(),
        });

        // ── Read execution trace columns ─────────────────────────────
        // Full input state: digest_before[9] + input_value[9] + input_capacity[9]
        let digest_before: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let input_value: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let input_capacity: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // Full output state: digest_after[9] + output_value[9] + output_capacity[9]
        let digest_after: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let output_value: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let output_capacity: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // shifted_next_before: 9 M31 limbs (digest_before of the NEXT row)
        let shifted_next_before: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // op_type: 0 = mix, 1 = draw, 2 = mix_poly_coeffs
        let _op_type = eval.next_trace_mask();

        // ── Boundary: first row's digest_before = initial (zero) ─────
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_first.clone()
                    * (digest_before[j].clone() - E::F::from(self.initial_digest_limbs[j])),
            );
        }

        // ── Boundary: last row's digest_after = final ────────────────
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_last.clone()
                    * (digest_after[j].clone() - E::F::from(self.final_digest_limbs[j])),
            );
        }

        // ── Chain: digest_after[row] == digest_before[row+1] ─────────
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_chain.clone()
                    * (digest_after[j].clone() - shifted_next_before[j].clone()),
            );
        }

        // ── LogUp: Hades permutation binding ─────────────────────────
        // Each row contributes a +1 lookup into the HadesPermRelation.
        // The Hades AIR must provide matching -1 entries, proving that
        // every (input_state, output_state) was correctly computed.
        if let Some(ref hades_rel) = self.hades_lookup {
            // Build the 54-column key: [input_state[27], output_state[27]]
            let mut key_values: Vec<E::F> = Vec::with_capacity(54);
            for v in &digest_before { key_values.push(v.clone()); }
            for v in &input_value { key_values.push(v.clone()); }
            for v in &input_capacity { key_values.push(v.clone()); }
            for v in &digest_after { key_values.push(v.clone()); }
            for v in &output_value { key_values.push(v.clone()); }
            for v in &output_capacity { key_values.push(v.clone()); }

            eval.add_to_relation(RelationEntry::new(
                hades_rel,
                E::EF::one(), // +1 multiplicity: consumer
                &key_values,
            ));

            eval.finalize_logup();
        }

        eval
    }
}

/// The recursive verifier STARK component.
pub type RecursiveVerifierComponent = FrameworkComponent<RecursiveVerifierEval>;

// ═══════════════════════════════════════════════════════════════════════
// Trace building — populates real Hades states from the witness
// ═══════════════════════════════════════════════════════════════════════

use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};

/// A single chain trace row with full Hades input/output states.
struct ChainRow {
    full_input: [FieldElement; 3],
    full_output: [FieldElement; 3],
}

/// Build the execution trace from real Hades permutation states.
///
/// Each row stores the actual felt252 Hades input/output from the GKR
/// verifier's Fiat-Shamir transcript, decomposed into M31 limbs.
///
/// Expanded layout (64 columns):
///   [0..9)    digest_before     (input state[0])
///   [9..18)   input_value       (input state[1])
///   [18..27)  input_capacity    (input state[2])
///   [27..36)  digest_after      (output state[0])
///   [36..45)  output_value      (output state[1])
///   [45..54)  output_capacity   (output state[2])
///   [54..63)  shifted_next_before (next row's digest_before)
///   [63]      op_type
pub fn build_recursive_trace(
    witness: &super::types::GkrVerifierWitness,
) -> RecursiveTraceData {
    use super::types::WitnessOp;

    // Pair each ChannelOp with its preceding HadesPerm to get full states.
    // The witness records: HadesPerm { input, output } then ChannelOp { digest_before, digest_after }.
    // We walk the ops and pair them.
    let mut rows: Vec<ChainRow> = Vec::new();
    let mut pending_hades: Option<([FieldElement; 3], [FieldElement; 3])> = None;

    for op in &witness.ops {
        match op {
            WitnessOp::HadesPerm { input, output } => {
                pending_hades = Some((*input, *output));
            }
            WitnessOp::ChannelOp { digest_before, digest_after } => {
                let (full_input, full_output) = if let Some((inp, out)) = pending_hades.take() {
                    // Verify consistency: HadesPerm input[0] == digest_before
                    debug_assert_eq!(inp[0], *digest_before);
                    debug_assert_eq!(out[0], *digest_after);
                    (inp, out)
                } else {
                    // No HadesPerm for this ChannelOp — use digest + zeros
                    ([*digest_before, FieldElement::ZERO, FieldElement::ZERO],
                     [*digest_after, FieldElement::ZERO, FieldElement::ZERO])
                };
                rows.push(ChainRow { full_input, full_output });
            }
            _ => {} // Skip non-chain ops
        }
    }

    let n_ops = rows.len();
    let n_real_rows = n_ops;
    let n_for_sizing = witness.n_poseidon_perms.max(n_ops);

    let log_size = if n_for_sizing <= 1 {
        1
    } else {
        (n_for_sizing as u32).next_power_of_two().ilog2().max(1)
    };
    let n_padded_rows = 1usize << log_size;

    // Build trace columns (expanded: 64 columns)
    let mut execution_trace: Vec<Vec<M31>> = Vec::with_capacity(COLS_PER_ROW);
    for _ in 0..COLS_PER_ROW {
        execution_trace.push(vec![M31::from_u32_unchecked(0); n_padded_rows]);
    }

    let zero_limbs = felt252_to_limbs(&FieldElement::ZERO);

    // Populate trace
    for row_idx in 0..n_padded_rows {
        let (input_limbs, output_limbs) = if row_idx < n_ops {
            let r = &rows[row_idx];
            (hades_state_to_limbs(&r.full_input), hades_state_to_limbs(&r.full_output))
        } else {
            ([M31::from_u32_unchecked(0); COLS_PER_STATE],
             [M31::from_u32_unchecked(0); COLS_PER_STATE])
        };

        // Write full input state (27 columns: digest_before + input_value + input_capacity)
        for j in 0..COLS_PER_STATE {
            execution_trace[j][row_idx] = input_limbs[j];
        }
        // Write full output state (27 columns: digest_after + output_value + output_capacity)
        for j in 0..COLS_PER_STATE {
            execution_trace[COLS_PER_STATE + j][row_idx] = output_limbs[j];
        }
        // op_type
        execution_trace[2 * COLS_PER_STATE + LIMBS_PER_FELT][row_idx] = M31::from_u32_unchecked(0);
    }

    // Second pass: shifted_next_before[row_i] = digest_before[row_{i+1}]
    let shifted_start = 2 * COLS_PER_STATE; // after input + output states
    for row_idx in 0..n_padded_rows {
        if row_idx + 1 < n_padded_rows {
            for j in 0..LIMBS_PER_FELT {
                execution_trace[shifted_start + j][row_idx] = execution_trace[j][row_idx + 1];
            }
        }
    }

    // Preprocessed columns
    let mut is_first = vec![M31::from_u32_unchecked(0); n_padded_rows];
    let mut is_last = vec![M31::from_u32_unchecked(0); n_padded_rows];
    let mut is_chain = vec![M31::from_u32_unchecked(0); n_padded_rows];

    is_first[0] = M31::from_u32_unchecked(1);
    if n_real_rows > 0 && n_real_rows <= n_padded_rows {
        is_last[n_real_rows - 1] = M31::from_u32_unchecked(1);
    }
    // is_chain = 1 for all real rows except the last
    for i in 0..n_real_rows.saturating_sub(1).min(n_padded_rows) {
        is_chain[i] = M31::from_u32_unchecked(1);
    }

    RecursiveTraceData {
        execution_trace,
        preprocessed_is_first: is_first,
        preprocessed_is_last: is_last,
        preprocessed_is_chain: is_chain,
        log_size,
        n_real_rows,
        n_channel_ops: n_ops,
    }
}

/// Container for the recursive STARK trace data.
pub struct RecursiveTraceData {
    /// Execution trace columns (COLS_PER_ROW columns x 2^log_size rows).
    pub execution_trace: Vec<Vec<M31>>,

    /// Preprocessed column: 1 on row 0.
    pub preprocessed_is_first: Vec<M31>,

    /// Preprocessed column: 1 on the last real row.
    pub preprocessed_is_last: Vec<M31>,

    /// Preprocessed column: 1 on all real rows except the last.
    pub preprocessed_is_chain: Vec<M31>,

    /// log2 of padded trace height.
    pub log_size: u32,

    /// Number of real (non-padding) rows.
    pub n_real_rows: usize,

    /// Number of channel operations (ChannelOp entries) in the trace.
    pub n_channel_ops: usize,
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::types::GkrVerifierWitness;

    #[test]
    fn test_cols_per_row() {
        assert_eq!(LIMBS_PER_FELT, 9);
        assert_eq!(COLS_PER_DIGEST, 9);
        assert_eq!(COLS_PER_STATE, 27);
        // Expanded: 27 (input) + 27 (output) + 9 (shifted) + 1 (op_type) = 64
        assert_eq!(COLS_PER_ROW, 64);
    }

    #[test]
    fn test_felt252_to_limbs_zero() {
        let limbs = felt252_to_limbs(&FieldElement::ZERO);
        for l in &limbs {
            assert_eq!(*l, M31::from_u32_unchecked(0));
        }
    }

    #[test]
    fn test_felt252_to_limbs_one() {
        let limbs = felt252_to_limbs(&FieldElement::ONE);
        assert_eq!(limbs[0], M31::from_u32_unchecked(1));
        for l in &limbs[1..] {
            assert_eq!(*l, M31::from_u32_unchecked(0));
        }
    }

    #[test]
    fn test_felt252_to_limbs_small() {
        let felt = FieldElement::from(0xDEADBEEFu64);
        let limbs = felt252_to_limbs(&felt);
        // First limb holds lowest 28 bits of 0xDEADBEEF = 0xEADBEEF
        assert_eq!(limbs[0], M31::from_u32_unchecked(0xEADBEEF));
        // Second limb holds next 4 bits = 0xD
        assert_eq!(limbs[1], M31::from_u32_unchecked(0xD));
    }

    #[test]
    fn test_hades_state_to_limbs_roundtrip() {
        let state = [
            FieldElement::from(42u64),
            FieldElement::from(100u64),
            FieldElement::TWO,
        ];
        let limbs = hades_state_to_limbs(&state);
        assert_eq!(limbs.len(), COLS_PER_STATE);

        // First felt252 (42) should have limbs[0] = 42
        assert_eq!(limbs[0], M31::from_u32_unchecked(42));
        // Second felt252 (100) starts at offset 9
        assert_eq!(limbs[LIMBS_PER_FELT], M31::from_u32_unchecked(100));
        // Third felt252 (2) starts at offset 18
        assert_eq!(limbs[2 * LIMBS_PER_FELT], M31::from_u32_unchecked(2));
    }

    #[test]
    fn test_trace_with_channel_op() {
        // Build a witness with ChannelOp and verify trace population.
        use crate::recursive::types::WitnessOp;

        let digest_before = FieldElement::ZERO;
        let mut state = [digest_before, FieldElement::from(42u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut state);
        let digest_after = state[0];

        let witness = GkrVerifierWitness {
            ops: vec![WitnessOp::ChannelOp { digest_before, digest_after }],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                verified: true,
            },
            n_poseidon_perms: 1,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest: digest_after,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);

        assert_eq!(trace.log_size, 1);
        assert_eq!(trace.n_real_rows, 1);
        assert_eq!(trace.n_channel_ops, 1);
        assert_eq!(trace.execution_trace.len(), COLS_PER_ROW);

        // Verify digest_before limbs (first 9 columns)
        let before_limbs = felt252_to_limbs(&digest_before);
        for j in 0..LIMBS_PER_FELT {
            assert_eq!(trace.execution_trace[j][0], before_limbs[j]);
        }

        // Verify digest_after limbs (columns 9-17)
        let after_limbs = felt252_to_limbs(&digest_after);
        for j in 0..LIMBS_PER_FELT {
            assert_eq!(trace.execution_trace[LIMBS_PER_FELT + j][0], after_limbs[j]);
        }

        assert_eq!(trace.preprocessed_is_first[0], M31::from_u32_unchecked(1));
        assert_eq!(trace.preprocessed_is_last[0], M31::from_u32_unchecked(1));
    }

    #[test]
    fn test_trace_chain_correctness() {
        // Two channel ops: verify digest_after[0] == digest_before[1].
        use crate::recursive::types::WitnessOp;

        let d0 = FieldElement::ZERO;
        let mut s1 = [d0, FieldElement::from(42u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut s1);
        let d1 = s1[0];

        let mut s2 = [d1, FieldElement::from(100u64), FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut s2);
        let d2 = s2[0];

        let witness = GkrVerifierWitness {
            ops: vec![
                WitnessOp::ChannelOp { digest_before: d0, digest_after: d1 },
                WitnessOp::ChannelOp { digest_before: d1, digest_after: d2 },
            ],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                verified: true,
            },
            n_poseidon_perms: 2,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest: d2,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);

        assert_eq!(trace.n_channel_ops, 2);

        // Verify chain: digest_after[row0] == digest_before[row1]
        for j in 0..LIMBS_PER_FELT {
            let after_row0 = trace.execution_trace[LIMBS_PER_FELT + j][0];
            let before_row1 = trace.execution_trace[j][1];
            assert_eq!(after_row0, before_row1,
                "chain broken at limb {j}: digest_after[0] != digest_before[1]");
        }
    }

    #[test]
    fn test_eval_properties() {
        let eval = RecursiveVerifierEval {
            log_n_rows: 14,
            initial_digest_limbs: [M31::from_u32_unchecked(0); LIMBS_PER_FELT],
            final_digest_limbs: [M31::from_u32_unchecked(42); LIMBS_PER_FELT],
            hades_lookup: None,
        };
        assert_eq!(eval.log_size(), 14);
        assert_eq!(eval.max_constraint_log_degree_bound(), 15);
    }
}
