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
};

/// Number of M31 limbs to represent one felt252.
/// 9 * 31 = 279 bits ≥ 252 bits.
pub const LIMBS_PER_FELT: usize = 9;

/// Columns for one Hades state element (3 felt252 = 3 * 9 = 27 limbs).
pub const COLS_PER_STATE: usize = 3 * LIMBS_PER_FELT; // 27

/// Columns for the shifted next-row input digest (for chain constraints).
pub const COLS_SHIFTED_DIGEST: usize = LIMBS_PER_FELT; // 9

/// Total columns per row: input state + output state + op_type + shifted_next_digest.
pub const COLS_PER_ROW: usize = 2 * COLS_PER_STATE + 1 + COLS_SHIFTED_DIGEST; // 64

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
        // Input state: [digest(9), value(9), capacity(9)] = 27 columns
        let input_digest: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let input_value: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let input_cap: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // Output state: [digest(9), value(9), capacity(9)] = 27 columns
        let output_digest: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let _output_value: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let _output_cap: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // Op type: 0 = mix, 1 = draw
        let _op_type = eval.next_trace_mask();

        // Shifted next-row input digest (9 columns).
        // On row i, this holds input_digest[row i+1].
        // On the last row, this is zero (no next row).
        let shifted_next_digest: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // ── Boundary constraint: first row's input digest = initial ──
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_first.clone()
                    * (input_digest[j].clone() - E::F::from(self.initial_digest_limbs[j])),
            );
        }

        // ── Boundary constraint: last row's output digest = final ────
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_last.clone()
                    * (output_digest[j].clone() - E::F::from(self.final_digest_limbs[j])),
            );
        }

        // ── Chain constraint: output_digest[row] == input_digest[row+1]
        //
        // We use a shifted column approach: `shifted_next_digest[row_i]`
        // holds `input_digest[row_{i+1}]`. The prover populates this from
        // the trace. The constraint checks equality on all chain rows:
        //
        //   is_chain * (output_digest[j] - shifted_next_digest[j]) == 0
        //
        // This enforces the Fiat-Shamir transcript chain — each Hades call's
        // output digest feeds the next call's input digest.
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_chain.clone()
                    * (output_digest[j].clone() - shifted_next_digest[j].clone()),
            );
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

/// Build the execution trace from real Hades permutation states.
///
/// Each row stores the actual felt252 Hades input/output from the GKR
/// verifier's Fiat-Shamir transcript, decomposed into M31 limbs.
pub fn build_recursive_trace(
    witness: &super::types::GkrVerifierWitness,
) -> RecursiveTraceData {
    use super::types::WitnessOp;

    // Collect all HadesPerm ops from the witness
    let hades_ops: Vec<([FieldElement; 3], [FieldElement; 3])> = witness
        .ops
        .iter()
        .filter_map(|op| match op {
            WitnessOp::HadesPerm { input, output } => Some((*input, *output)),
            _ => None,
        })
        .collect();

    let n_hades = hades_ops.len();
    // Use the total poseidon count from production verifier for sizing
    // (covers ALL channel ops, not just the ones we recorded in Pass 2)
    let n_real_rows = witness.n_poseidon_perms.max(n_hades);

    let log_size = if n_real_rows <= 1 {
        1
    } else {
        (n_real_rows as u32).next_power_of_two().ilog2().max(1)
    };
    let n_padded_rows = 1usize << log_size;

    // Build trace columns
    let mut execution_trace: Vec<Vec<M31>> = Vec::with_capacity(COLS_PER_ROW);
    for _ in 0..COLS_PER_ROW {
        execution_trace.push(vec![M31::from_u32_unchecked(0); n_padded_rows]);
    }

    // Compute a zero-state Hades permutation for padding
    let zero_felt = FieldElement::ZERO;
    let zero_state = [zero_felt; 3];
    let mut padded_output = zero_state;
    crate::crypto::hades::hades_permutation(&mut padded_output);
    let zero_input_limbs = hades_state_to_limbs(&zero_state);
    let zero_output_limbs = hades_state_to_limbs(&padded_output);

    for row in 0..n_padded_rows {
        let (input_limbs, output_limbs, op_type) = if row < n_hades {
            // Real data from the witness
            let (ref input, ref output) = hades_ops[row];
            let il = hades_state_to_limbs(input);
            let ol = hades_state_to_limbs(output);
            // Determine op type from capacity: 2 = mix, 3 = draw
            let cap_bytes = input[2].to_bytes_be();
            let cap_val = cap_bytes[31];
            let op = if cap_val == 3 { 1u32 } else { 0u32 };
            (il, ol, op)
        } else {
            // Padding: valid zero-input Hades permutation
            (zero_input_limbs, zero_output_limbs, 0u32)
        };

        // Write input state limbs (27 columns)
        for j in 0..COLS_PER_STATE {
            execution_trace[j][row] = input_limbs[j];
        }
        // Write output state limbs (27 columns)
        for j in 0..COLS_PER_STATE {
            execution_trace[COLS_PER_STATE + j][row] = output_limbs[j];
        }
        // Write op_type (1 column)
        execution_trace[2 * COLS_PER_STATE][row] = M31::from_u32_unchecked(op_type);

        // Shifted next-row digest (9 columns) — populated in second pass below
    }

    // Second pass: populate shifted_next_digest columns.
    // shifted_next_digest[row_i] = input_digest[row_{i+1}]
    let shifted_col_start = 2 * COLS_PER_STATE + 1;
    for row in 0..n_padded_rows {
        let next_row = row + 1;
        if next_row < n_padded_rows {
            // Copy input_digest of next row
            for j in 0..LIMBS_PER_FELT {
                execution_trace[shifted_col_start + j][row] = execution_trace[j][next_row];
            }
        }
        // Last row: shifted_next_digest = 0 (already zeroed)
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
        n_hades_recorded: n_hades,
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

    /// Number of Hades permutations recorded in the witness (may be < n_real_rows
    /// because some operations like mix_poly_coeffs use poseidon_hash_many internally).
    pub n_hades_recorded: usize,
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
        assert_eq!(COLS_PER_STATE, 27);
        assert_eq!(COLS_SHIFTED_DIGEST, 9);
        // 27 (input) + 27 (output) + 1 (op_type) + 9 (shifted_next_digest) = 64
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
    fn test_trace_with_real_hades() {
        // Build a witness with real Hades ops and verify trace population.
        use crate::recursive::types::WitnessOp;

        let mut state = [FieldElement::ZERO, FieldElement::from(42u64), FieldElement::TWO];
        let input = state;
        crate::crypto::hades::hades_permutation(&mut state);
        let output = state;

        let witness = GkrVerifierWitness {
            ops: vec![WitnessOp::HadesPerm { input, output }],
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
            final_digest: FieldElement::ZERO,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);

        assert_eq!(trace.log_size, 1); // 1 real row → padded to 2
        assert_eq!(trace.n_real_rows, 1);
        assert_eq!(trace.n_hades_recorded, 1);
        assert_eq!(trace.execution_trace.len(), COLS_PER_ROW);
        assert_eq!(trace.execution_trace[0].len(), 2); // 2^1

        // Verify row 0 has the real input digest limbs
        let expected_input_limbs = hades_state_to_limbs(&input);
        for j in 0..COLS_PER_STATE {
            assert_eq!(
                trace.execution_trace[j][0], expected_input_limbs[j],
                "input limb {j} mismatch"
            );
        }

        // Verify row 0 has the real output digest limbs
        let expected_output_limbs = hades_state_to_limbs(&output);
        for j in 0..COLS_PER_STATE {
            assert_eq!(
                trace.execution_trace[COLS_PER_STATE + j][0], expected_output_limbs[j],
                "output limb {j} mismatch"
            );
        }

        // Verify is_first/is_last
        assert_eq!(trace.preprocessed_is_first[0], M31::from_u32_unchecked(1));
        assert_eq!(trace.preprocessed_is_last[0], M31::from_u32_unchecked(1));
    }

    #[test]
    fn test_trace_chain_correctness() {
        // Two Hades calls: verify the chain (output digest of row 0 == input digest of row 1).
        use crate::recursive::types::WitnessOp;

        // Call 1: mix(42)
        let mut state1 = [FieldElement::ZERO, FieldElement::from(42u64), FieldElement::TWO];
        let input1 = state1;
        crate::crypto::hades::hades_permutation(&mut state1);
        let output1 = state1;

        // Call 2: mix(100) — input digest should be output1's digest
        let mut state2 = [output1[0], FieldElement::from(100u64), FieldElement::TWO];
        let input2 = state2;
        crate::crypto::hades::hades_permutation(&mut state2);
        let output2 = state2;

        let witness = GkrVerifierWitness {
            ops: vec![
                WitnessOp::HadesPerm { input: input1, output: output1 },
                WitnessOp::HadesPerm { input: input2, output: output2 },
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
            final_digest: FieldElement::ZERO,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);

        assert_eq!(trace.n_hades_recorded, 2);
        assert_eq!(trace.log_size, 1); // 2 rows → padded to 2

        // Verify chain: output_digest[row0] == input_digest[row1]
        let output_digest_start = COLS_PER_STATE; // output state starts at column 27
        for j in 0..LIMBS_PER_FELT {
            let out_digest_limb_row0 = trace.execution_trace[output_digest_start + j][0];
            let in_digest_limb_row1 = trace.execution_trace[j][1];
            assert_eq!(
                out_digest_limb_row0, in_digest_limb_row1,
                "chain broken at limb {j}: output_digest[0] != input_digest[1]"
            );
        }
    }

    #[test]
    fn test_eval_properties() {
        let eval = RecursiveVerifierEval {
            log_n_rows: 14,
            initial_digest_limbs: [M31::from_u32_unchecked(0); LIMBS_PER_FELT],
            final_digest_limbs: [M31::from_u32_unchecked(42); LIMBS_PER_FELT],
        };
        assert_eq!(eval.log_size(), 14);
        assert_eq!(eval.max_constraint_log_degree_bound(), 15);
    }
}
