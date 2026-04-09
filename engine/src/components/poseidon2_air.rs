//! Poseidon2 AIR constraint helpers for transaction STARKs.
//!
//! Provides shared constraint functions that express Poseidon2-M31 permutation
//! round constraints as degree-2 polynomial constraints over STWO's EvalAtRow.
//! Each permutation occupies 652 execution trace columns:
//!   - 23 × 16 = 368 state columns (before each round + output)
//!   - 8 × 16 = 128 full-round sq auxiliary columns
//!   - 8 × 16 = 128 full-round quad auxiliary columns
//!   - 14 × 1 = 14 partial-round sq auxiliary columns
//!   - 14 × 1 = 14 partial-round quad auxiliary columns

use std::ops::{Add, AddAssign};

use stwo::core::fields::m31::BaseField as M31;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo_constraint_framework::EvalAtRow;

use crate::crypto::poseidon2_m31::{
    apply_external_round_matrix, apply_internal_round_matrix, get_round_constants,
    INTERNAL_DIAG_U32, N_FULL_ROUNDS, N_HALF_FULL_ROUNDS, N_PARTIAL_ROUNDS, RATE, STATE_WIDTH,
};

/// Number of execution trace columns per Poseidon2 permutation.
pub const COLS_PER_PERM: usize = 23 * STATE_WIDTH  // states
    + N_FULL_ROUNDS * STATE_WIDTH                   // full_sq
    + N_FULL_ROUNDS * STATE_WIDTH                   // full_quad
    + N_PARTIAL_ROUNDS                              // partial_sq
    + N_PARTIAL_ROUNDS; // partial_quad
                        // = 368 + 128 + 128 + 14 + 14 = 652

// ──────────────────────── Trace data (prover side) ────────────────────────

/// Intermediate values for one Poseidon2 permutation, used during trace generation.
pub struct PermutationTrace {
    /// state[r][j]: state element j before round r. state[22] = final output.
    pub states: [[M31; STATE_WIDTH]; 23],
    /// S-box sq = (state + rc)² for 8 full rounds, all 16 elements.
    pub full_sq: [[M31; STATE_WIDTH]; N_FULL_ROUNDS],
    /// S-box quad = sq² for 8 full rounds, all 16 elements.
    pub full_quad: [[M31; STATE_WIDTH]; N_FULL_ROUNDS],
    /// S-box sq for 14 partial rounds, element 0 only.
    pub partial_sq: [M31; N_PARTIAL_ROUNDS],
    /// S-box quad for 14 partial rounds, element 0 only.
    pub partial_quad: [M31; N_PARTIAL_ROUNDS],
}

/// Compute the full permutation trace for a given input state.
///
/// Records all 22 round intermediate states plus S-box auxiliary values.
/// The output state matches `poseidon2_permutation(input)`.
pub fn compute_permutation_trace(input: &[M31; STATE_WIDTH]) -> PermutationTrace {
    let rc = get_round_constants();
    let zero = M31::from_u32_unchecked(0);

    let mut states = [[zero; STATE_WIDTH]; 23];
    let mut full_sq = [[zero; STATE_WIDTH]; N_FULL_ROUNDS];
    let mut full_quad = [[zero; STATE_WIDTH]; N_FULL_ROUNDS];
    let mut partial_sq = [zero; N_PARTIAL_ROUNDS];
    let mut partial_quad = [zero; N_PARTIAL_ROUNDS];

    states[0] = *input;
    let mut state = *input;

    // First 4 full rounds
    for r in 0..N_HALF_FULL_ROUNDS {
        for j in 0..STATE_WIDTH {
            let after_rc = state[j] + rc.external[r][j];
            let sq = after_rc * after_rc;
            let quad = sq * sq;
            full_sq[r][j] = sq;
            full_quad[r][j] = quad;
            state[j] = quad * after_rc;
        }
        apply_external_round_matrix(&mut state);
        states[r + 1] = state;
    }

    // 14 partial rounds
    for r in 0..N_PARTIAL_ROUNDS {
        let after_rc = state[0] + rc.internal[r];
        let sq = after_rc * after_rc;
        let quad = sq * sq;
        partial_sq[r] = sq;
        partial_quad[r] = quad;
        state[0] = quad * after_rc;
        apply_internal_round_matrix(&mut state);
        states[r + N_HALF_FULL_ROUNDS + 1] = state;
    }

    // Last 4 full rounds
    for r in 0..N_HALF_FULL_ROUNDS {
        let rc_idx = r + N_HALF_FULL_ROUNDS;
        for j in 0..STATE_WIDTH {
            let after_rc = state[j] + rc.external[rc_idx][j];
            let sq = after_rc * after_rc;
            let quad = sq * sq;
            full_sq[r + N_HALF_FULL_ROUNDS][j] = sq;
            full_quad[r + N_HALF_FULL_ROUNDS][j] = quad;
            state[j] = quad * after_rc;
        }
        apply_external_round_matrix(&mut state);
        states[r + N_HALF_FULL_ROUNDS + N_PARTIAL_ROUNDS + 1] = state;
    }

    PermutationTrace {
        states,
        full_sq,
        full_quad,
        partial_sq,
        partial_quad,
    }
}

/// Compute a dummy permutation trace (zero input) for padding rows.
pub fn dummy_permutation_trace() -> PermutationTrace {
    compute_permutation_trace(&[M31::from_u32_unchecked(0); STATE_WIDTH])
}

/// Compute chained Merkle padding traces starting from a given hash value.
///
/// Each compress perm uses `[prev_output, prev_output]` as input, so that
/// the degree-2 chain constraint `(input[j] - prev[j]) * (input[j+8] - prev[j]) = 0`
/// is trivially satisfied (left half matches). This avoids needing `is_real` gating
/// on Merkle chain constraints, keeping all constraints at degree 2.
pub fn compute_merkle_chain_padding(
    start_hash: &[M31; RATE],
    num_levels: usize,
) -> Vec<PermutationTrace> {
    let zero = M31::from_u32_unchecked(0);
    let mut result = Vec::with_capacity(num_levels);
    let mut current = *start_hash;

    for _ in 0..num_levels {
        let mut input = [zero; STATE_WIDTH];
        input[..RATE].copy_from_slice(&current);
        input[RATE..].copy_from_slice(&current);
        let trace = compute_permutation_trace(&input);
        // Output of this compress: first RATE elements of the output state
        current.copy_from_slice(&trace.states[22][..RATE]);
        result.push(trace);
    }
    result
}

/// Write one permutation's trace data into execution columns at a given row.
///
/// Column layout: states (368) → full_sq (128) → full_quad (128) →
///   partial_sq (14) → partial_quad (14) = 652 total.
pub fn write_permutation_to_trace<B: ColumnOps<M31>>(
    trace: &PermutationTrace,
    cols: &mut [Col<B, M31>],
    col_offset: usize,
    row: usize,
) where
    Col<B, M31>: Column<M31>,
{
    let mut idx = col_offset;

    // States: 23 × 16
    for r in 0..23 {
        for j in 0..STATE_WIDTH {
            cols[idx].set(row, trace.states[r][j]);
            idx += 1;
        }
    }

    // Full sq: 8 × 16
    for r in 0..N_FULL_ROUNDS {
        for j in 0..STATE_WIDTH {
            cols[idx].set(row, trace.full_sq[r][j]);
            idx += 1;
        }
    }

    // Full quad: 8 × 16
    for r in 0..N_FULL_ROUNDS {
        for j in 0..STATE_WIDTH {
            cols[idx].set(row, trace.full_quad[r][j]);
            idx += 1;
        }
    }

    // Partial sq: 14
    for r in 0..N_PARTIAL_ROUNDS {
        cols[idx].set(row, trace.partial_sq[r]);
        idx += 1;
    }

    // Partial quad: 14
    for r in 0..N_PARTIAL_ROUNDS {
        cols[idx].set(row, trace.partial_quad[r]);
        idx += 1;
    }

    debug_assert_eq!(idx - col_offset, COLS_PER_PERM);
}

/// Decompose a u32 value into 16 binary digits (little-endian).
pub fn decompose_to_bits(value: u32) -> [u32; 16] {
    std::array::from_fn(|i| (value >> i) & 1)
}

// ──────────────────────── Constraint columns (verifier side) ──────────────

/// Column references for one Poseidon2 permutation, read from the evaluation trace.
pub struct Poseidon2Columns<F> {
    pub states: [[F; STATE_WIDTH]; 23],
    pub full_sq: [[F; STATE_WIDTH]; N_FULL_ROUNDS],
    pub full_quad: [[F; STATE_WIDTH]; N_FULL_ROUNDS],
    pub partial_sq: [F; N_PARTIAL_ROUNDS],
    pub partial_quad: [F; N_PARTIAL_ROUNDS],
}

impl<F: Clone> Poseidon2Columns<F> {
    pub fn input(&self) -> &[F; STATE_WIDTH] {
        &self.states[0]
    }
    pub fn output(&self) -> &[F; STATE_WIDTH] {
        &self.states[22]
    }
}

/// Read 652 columns for one Poseidon2 permutation and add all round constraints.
///
/// Each call reads the next 652 execution trace columns, adds ~352 degree-2
/// constraints for all 22 rounds, and returns column references for wiring.
pub fn constrain_poseidon2_permutation<E: EvalAtRow>(eval: &mut E) -> Poseidon2Columns<E::F> {
    // Read all columns in layout order
    let states: [[E::F; STATE_WIDTH]; 23] =
        std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));
    let full_sq: [[E::F; STATE_WIDTH]; N_FULL_ROUNDS] =
        std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));
    let full_quad: [[E::F; STATE_WIDTH]; N_FULL_ROUNDS] =
        std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));
    let partial_sq: [E::F; N_PARTIAL_ROUNDS] = std::array::from_fn(|_| eval.next_trace_mask());
    let partial_quad: [E::F; N_PARTIAL_ROUNDS] = std::array::from_fn(|_| eval.next_trace_mask());

    let cols = Poseidon2Columns {
        states,
        full_sq,
        full_quad,
        partial_sq,
        partial_quad,
    };

    // Add round constraints
    let rc = get_round_constants();

    // First 4 full rounds (round_idx 0..3, full_idx 0..3)
    for r in 0..N_HALF_FULL_ROUNDS {
        constrain_full_round(eval, &cols, r, r, &rc.external[r]);
    }

    // 14 partial rounds (round_idx 4..17, partial_idx 0..13)
    for r in 0..N_PARTIAL_ROUNDS {
        constrain_partial_round(eval, &cols, r + N_HALF_FULL_ROUNDS, r, rc.internal[r]);
    }

    // Last 4 full rounds (round_idx 18..21, full_idx 4..7)
    for r in 0..N_HALF_FULL_ROUNDS {
        constrain_full_round(
            eval,
            &cols,
            r + N_HALF_FULL_ROUNDS + N_PARTIAL_ROUNDS,
            r + N_HALF_FULL_ROUNDS,
            &rc.external[r + N_HALF_FULL_ROUNDS],
        );
    }

    cols
}

// ──────────────────────── Round constraint helpers ────────────────────────

/// Add constraints for one full round (S-box on all 16 elements + external matrix).
fn constrain_full_round<E: EvalAtRow>(
    eval: &mut E,
    cols: &Poseidon2Columns<E::F>,
    round_idx: usize,
    full_idx: usize,
    rc: &[M31; STATE_WIDTH],
) {
    let state = &cols.states[round_idx];
    let next_state = &cols.states[round_idx + 1];
    let sq = &cols.full_sq[full_idx];
    let quad = &cols.full_quad[full_idx];

    // Compute S-box outputs as degree-2 expressions
    let sbox_out: [E::F; STATE_WIDTH] = std::array::from_fn(|j| {
        let after_rc = state[j].clone() + E::F::from(rc[j]);

        // sq[j] = after_rc²
        eval.add_constraint(sq[j].clone() - after_rc.clone() * after_rc.clone());
        // quad[j] = sq[j]²
        eval.add_constraint(quad[j].clone() - sq[j].clone() * sq[j].clone());
        // sbox_out[j] = quad[j] * after_rc (degree 2, used inline for matrix)
        quad[j].clone() * after_rc
    });

    // External matrix: next_state = circ(2*M4, M4, M4, M4) × sbox_out
    let matrix_out = apply_external_matrix_exprs(&sbox_out);
    for j in 0..STATE_WIDTH {
        eval.add_constraint(next_state[j].clone() - matrix_out[j].clone());
    }
}

/// Add constraints for one partial round (S-box on element 0 only + internal matrix).
fn constrain_partial_round<E: EvalAtRow>(
    eval: &mut E,
    cols: &Poseidon2Columns<E::F>,
    round_idx: usize,
    partial_idx: usize,
    rc: M31,
) {
    let state = &cols.states[round_idx];
    let next_state = &cols.states[round_idx + 1];
    let sq = &cols.partial_sq[partial_idx];
    let quad = &cols.partial_quad[partial_idx];

    // S-box on element 0
    let after_rc_0 = state[0].clone() + E::F::from(rc);
    eval.add_constraint(sq.clone() - after_rc_0.clone() * after_rc_0.clone());
    eval.add_constraint(quad.clone() - sq.clone() * sq.clone());
    let sbox_out_0 = quad.clone() * after_rc_0;

    // Sum of all sbox outputs: sbox_out_0 + state[1] + ... + state[15]
    let mut sum_sbox = sbox_out_0.clone();
    for j in 1..STATE_WIDTH {
        sum_sbox += state[j].clone();
    }

    // Internal matrix: next_state[j] = DIAG[j] * sbox_out[j] + sum(sbox_out)
    // Element 0: sbox_out = quad * after_rc
    eval.add_constraint(
        next_state[0].clone()
            - (sbox_out_0 * M31::from_u32_unchecked(INTERNAL_DIAG_U32[0]) + sum_sbox.clone()),
    );
    // Elements 1..15: sbox_out[j] = state[j] (identity)
    for j in 1..STATE_WIDTH {
        eval.add_constraint(
            next_state[j].clone()
                - (state[j].clone() * M31::from_u32_unchecked(INTERNAL_DIAG_U32[j])
                    + sum_sbox.clone()),
        );
    }
}

// ──────────────────────── Matrix expression helpers ──────────────────────

/// Apply M4 4×4 matrix as generic expressions (additions only, no multiplications).
///
/// Matches `apply_m4` from poseidon2_m31.rs exactly.
pub fn apply_m4_exprs<F: Clone + Add<Output = F>>(x: &[F; 4]) -> [F; 4] {
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1;
    let t3 = x[3].clone() + x[3].clone() + t0;
    let t4 = t12.clone() + t12 + t3.clone();
    let t5 = t02.clone() + t02 + t2.clone();
    [t3 + t5.clone(), t5, t2 + t4.clone(), t4]
}

/// Apply external round matrix circ(2*M4, M4, M4, M4) as generic expressions.
///
/// Matches `apply_external_round_matrix` from poseidon2_m31.rs exactly.
pub fn apply_external_matrix_exprs<F: Clone + Add<Output = F> + AddAssign>(
    x: &[F; STATE_WIDTH],
) -> [F; STATE_WIDTH] {
    // M4 on each 4-element block
    let b0 = apply_m4_exprs(&[x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone()]);
    let b1 = apply_m4_exprs(&[x[4].clone(), x[5].clone(), x[6].clone(), x[7].clone()]);
    let b2 = apply_m4_exprs(&[x[8].clone(), x[9].clone(), x[10].clone(), x[11].clone()]);
    let b3 = apply_m4_exprs(&[x[12].clone(), x[13].clone(), x[14].clone(), x[15].clone()]);

    let mut result: [F; STATE_WIDTH] = std::array::from_fn(|i| {
        let block = i / 4;
        let pos = i % 4;
        match block {
            0 => b0[pos].clone(),
            1 => b1[pos].clone(),
            2 => b2[pos].clone(),
            3 => b3[pos].clone(),
            _ => unreachable!(),
        }
    });

    // Cross-block column sums
    for j in 0..4 {
        let s = result[j].clone()
            + result[j + 4].clone()
            + result[j + 8].clone()
            + result[j + 12].clone();
        result[j] += s.clone();
        result[j + 4] += s.clone();
        result[j + 8] += s.clone();
        result[j + 12] += s;
    }

    result
}

// ──────────────────────── Tests ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::poseidon2_m31::{
        apply_external_round_matrix, apply_m4, poseidon2_permutation, sbox,
    };

    #[test]
    fn test_compute_permutation_trace_matches() {
        let input =
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map(M31::from_u32_unchecked);
        let trace = compute_permutation_trace(&input);

        // Output should match poseidon2_permutation
        let mut expected = input;
        poseidon2_permutation(&mut expected);
        assert_eq!(trace.states[22], expected);
    }

    #[test]
    fn test_permutation_trace_input_preserved() {
        let input = [42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].map(M31::from_u32_unchecked);
        let trace = compute_permutation_trace(&input);
        assert_eq!(trace.states[0], input);
    }

    #[test]
    fn test_permutation_trace_column_count() {
        assert_eq!(COLS_PER_PERM, 652);
    }

    #[test]
    fn test_sbox_decomposition() {
        // Verify sq/quad values correctly decompose x^5
        let rc = get_round_constants();
        let input = [
            7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
        ]
        .map(M31::from_u32_unchecked);
        let trace = compute_permutation_trace(&input);

        // Check first full round, element 0
        let after_rc = input[0] + rc.external[0][0];
        let expected_sq = after_rc * after_rc;
        let expected_quad = expected_sq * expected_sq;
        let expected_sbox = expected_quad * after_rc; // x^5
        assert_eq!(trace.full_sq[0][0], expected_sq);
        assert_eq!(trace.full_quad[0][0], expected_quad);
        assert_eq!(expected_sbox, sbox(after_rc));

        // Check first partial round, element 0
        let partial_input = trace.states[4][0]; // state before partial round 0
        let after_rc_p = partial_input + rc.internal[0];
        assert_eq!(trace.partial_sq[0], after_rc_p * after_rc_p);
        assert_eq!(
            trace.partial_quad[0],
            trace.partial_sq[0] * trace.partial_sq[0]
        );
    }

    #[test]
    fn test_round_constants_match() {
        let rc = get_round_constants();
        // Verify non-trivial and deterministic
        assert_ne!(rc.external[0][0], M31::from_u32_unchecked(0));
        assert_ne!(rc.internal[0], M31::from_u32_unchecked(0));
        // Verify consistent with a fresh generation
        let rc2 = get_round_constants();
        assert_eq!(rc.external[0], rc2.external[0]);
        assert_eq!(rc.internal[0], rc2.internal[0]);
    }

    #[test]
    fn test_m4_constraint_matches_m4() {
        let input = [10u32, 20, 30, 40].map(M31::from_u32_unchecked);

        // Compute via apply_m4
        let mut expected = input;
        apply_m4(&mut expected);

        // Compute via constraint expression (M31 implements Add + Clone)
        let result = apply_m4_exprs(&input);

        assert_eq!(result, expected, "M4 constraint expression mismatch");
    }

    #[test]
    fn test_external_matrix_matches() {
        let input: [M31; STATE_WIDTH] =
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map(M31::from_u32_unchecked);

        let mut expected = input;
        apply_external_round_matrix(&mut expected);

        let result = apply_external_matrix_exprs(&input);

        assert_eq!(result, expected, "External matrix constraint mismatch");
    }

    #[test]
    fn test_internal_matrix_matches() {
        let input: [M31; STATE_WIDTH] =
            [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
                .map(M31::from_u32_unchecked);

        // Compute expected via apply_internal_round_matrix
        let mut expected = input;
        apply_internal_round_matrix(&mut expected);

        // Compute via the same formula used in constraints:
        // result[j] = DIAG[j] * input[j] + sum(input)
        let sum: M31 = input
            .iter()
            .copied()
            .fold(M31::from_u32_unchecked(0), |a, b| a + b);
        let result: [M31; STATE_WIDTH] =
            std::array::from_fn(|j| input[j] * M31::from_u32_unchecked(INTERNAL_DIAG_U32[j]) + sum);

        assert_eq!(result, expected, "Internal matrix constraint mismatch");
    }
}
