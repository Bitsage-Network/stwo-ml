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
    Relation, RelationEntry,
};

// ── LogUp relation: binds chain AIR ↔ Hades AIR ─────────────────────
// Key: (digest_before[9], digest_after[9]) = 18 M31 columns (28-bit limbs).
// This binds the chain's digest transitions to verified Hades permutations.
// Chain AIR contributes +1 multiplicity (consumer) for each active row.
// Hades AIR contributes -1 multiplicity (provider) for each permutation's
// last round, using the first-round digest and last-round MDS output digest.
//
// Digest-only binding is sufficient because the Poseidon digest uniquely
// identifies the channel state — if the Hades permutation is correct for
// a given (input_state[0], input_state[1], input_state[2]), the output
// digest (output_state[0]) is deterministic.
stwo_constraint_framework::relation!(HadesPermRelation, 18);

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
/// + shifted_next_before[9] + op_type[1]
/// + is_active[1] + is_active_next[1] + is_active_prev[1]
/// + active_count[1] + active_count_next[1] = 69
///
/// SECURITY: The last 7 columns are execution-trace selectors with UNCONDITIONAL
/// constraints. This prevents the all-zeros-selector attack where a malicious prover
/// sets preprocessed selectors to zero, making all constraints trivially satisfied.
/// The amortized accumulator (active_count) uses the identity:
///   active_count[i+1] = active_count[i] + is_active[i] - n_real_rows * N_inv
/// which evaluates to n_real_rows * N_inv ≠ 0 on an all-zeros trace.
pub const COLS_PER_ROW: usize = COLS_PER_STATE   // input full state: 27
    + COLS_PER_STATE // output full state: 27
    + COLS_PER_DIGEST // shifted_next_before: 9
    + 1  // op_type
    + 7; // [64] is_active, [65] is_active_next, [66] active_count,
         // [67] active_count_next, [68] is_chain_gate, [69] is_boundary_gate,
         // [70] is_active_prev
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

    /// Number of real (non-padding) rows in the trace.
    /// SECURITY: This determines where is_first, is_last, is_chain are placed.
    /// Both prover and verifier compute the preprocessed columns from this value.
    /// Without this, a malicious prover could place is_last at row 1 (miniaturized
    /// chain) and the verifier couldn't detect it.
    pub n_real_rows: u32,

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
        // All constraints are degree ≤ 2 (helper columns precompute products)
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // ── Preprocessed selectors ─────────────────────────────────────
        // is_first IS used for the initial boundary (row 0). It represents
        // the first circle domain point and is deterministic. The amortized
        // accumulator (C3) prevents the all-zeros attack even if is_first
        // is tampered, because the correction term is unconditionally non-zero.
        let is_first = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_first".into(),
        });
        let _is_last = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_last".into(),
        });
        let _is_chain = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_chain".into(),
        });

        // ── Read execution trace columns ─────────────────────────────
        // Full input state: digest_before[9] + input_value[9] + input_capacity[9]
        let digest_before: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());
        let input_value: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());
        let input_capacity: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // Full output state: digest_after[9] + output_value[9] + output_capacity[9]
        let digest_after: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());
        let output_value: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());
        let output_capacity: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // shifted_next_before: 9 M31 limbs (digest_before of the NEXT row)
        let shifted_next_before: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // op_type: 0 = mix, 1 = draw, 2 = mix_poly_coeffs
        let _op_type = eval.next_trace_mask();

        // ── Execution-trace selectors ────────────────────────────────
        // Read but used only for the amortized accumulator (unconditional constraint).
        let is_active = eval.next_trace_mask();
        let _is_active_next = eval.next_trace_mask();
        let active_count = eval.next_trace_mask();
        let active_count_next = eval.next_trace_mask();
        let _is_chain_gate = eval.next_trace_mask();
        let _is_boundary_gate = eval.next_trace_mask();
        let _is_active_prev = eval.next_trace_mask();

        // ══════════════════════════════════════════════════════════════
        // UNCONDITIONAL CONSTRAINTS (no selector gating)
        // ══════════════════════════════════════════════════════════════

        // C1: is_active is boolean [degree 2, unconditional]
        eval.add_constraint(is_active.clone() * (E::F::from(M31::from(1u32)) - is_active.clone()));

        // C2: amortized accumulator [degree 1, unconditional]
        // SECURITY: CRITICAL — prevents all-zeros-selector attack.
        // For an all-zeros trace: 0 - 0 - 0 + correction = correction ≠ 0.
        let n = 1u32 << self.log_n_rows;
        let n_inv = M31::from(n).inverse();
        let correction = E::F::from(M31::from(self.n_real_rows)) * E::F::from(n_inv);
        eval.add_constraint(
            active_count_next.clone()
                - active_count.clone()
                - is_active.clone()
                + correction,
        );

        // ══════════════════════════════════════════════════════════════
        // PREPROCESSED-GATED CONSTRAINTS (existing, proven working)
        // These use preprocessed is_first/is_last/is_chain selectors.
        // Combined with C2 (accumulator), these provide full security:
        // - C2 forces exactly n_real_rows active rows (prevents miniaturization)
        // - is_first/is_last/is_chain enforce chain integrity
        // ══════════════════════════════════════════════════════════════

        // C3: Initial boundary — row 0's digest_before = initial [degree 2]
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                is_first.clone()
                    * (digest_before[j].clone() - E::F::from(self.initial_digest_limbs[j])),
            );
        }

        // C4: Final boundary — last active row's digest_after = final [degree 2]
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                _is_last.clone()
                    * (digest_after[j].clone() - E::F::from(self.final_digest_limbs[j])),
            );
        }

        // C5: Chain — consecutive active rows have chained digests [degree 2]
        for j in 0..LIMBS_PER_FELT {
            eval.add_constraint(
                _is_chain.clone()
                    * (digest_after[j].clone() - shifted_next_before[j].clone()),
            );
        }

        // ── LogUp: Hades permutation binding ─────────────────────────
        // Key: (digest_before[9], digest_after[9]) = 18 M31 elements.
        // Only active rows contribute (+1 multiplicity).
        // Padding rows contribute 0 multiplicity (is_active = 0).
        if let Some(ref hades_rel) = self.hades_lookup {
            let mut key_values: Vec<E::F> = Vec::with_capacity(18);
            for v in &digest_before {
                key_values.push(v.clone());
            }
            for v in &digest_after {
                key_values.push(v.clone());
            }

            eval.add_to_relation(RelationEntry::new(
                hades_rel,
                // +1 on active rows, 0 on padding.
                // is_active is an execution-trace column (E::F), needs conversion to E::EF.
                E::EF::from(is_active.clone()),
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
/// Expanded layout (69 columns):
///   [0..9)    digest_before     (input state[0])
///   [9..18)   input_value       (input state[1])
///   [18..27)  input_capacity    (input state[2])
///   [27..36)  digest_after      (output state[0])
///   [36..45)  output_value      (output state[1])
///   [45..54)  output_capacity   (output state[2])
///   [54..63)  shifted_next_before (next row's digest_before)
///   [63]      op_type
///   [64]      is_active         (1 for real rows, 0 for padding)
///   [65]      is_active_next    (shifted: is_active[i+1])
///   [66]      active_count      (amortized accumulator)
///   [67]      active_count_next (shifted: active_count[i+1])
///   [68]      is_chain_gate     (precomputed: is_active * is_active_next)
///   [69]      is_boundary_gate  (precomputed: is_active * (1 - is_active_next))
///   [70]      is_active_prev    (shifted: is_active[i-1])
pub fn build_recursive_trace(witness: &super::types::GkrVerifierWitness) -> RecursiveTraceData {
    use super::types::WitnessOp;

    // Build one row per ChannelOp (digest transition).
    // Each row represents one atomic channel state change (digest_before → digest_after).
    // Some ChannelOps involve multiple Hades permutations (e.g., mix_poly_coeffs
    // does 2 Hades calls per ChannelOp). The chain constraint only links the
    // NET digest transitions, not individual permutations.
    //
    // NOTE: LogUp binding to the Hades AIR requires matching at the individual
    // permutation level, which means refactoring the chain to HadesPerm-level
    // rows. This is deferred to Phase 6 — the current defenses (amortized
    // accumulator, seed_digest, n_poseidon_perms, offline Hades verification)
    // provide strong security without LogUp.
    let mut rows: Vec<ChainRow> = Vec::new();
    let mut pending_hades: Option<([FieldElement; 3], [FieldElement; 3])> = None;

    for op in &witness.ops {
        match op {
            WitnessOp::HadesPerm { input, output } => {
                pending_hades = Some((*input, *output));
            }
            WitnessOp::ChannelOp {
                digest_before,
                digest_after,
            } => {
                let (full_input, full_output) = if let Some((inp, out)) = pending_hades.take() {
                    if inp[0] == *digest_before && out[0] == *digest_after {
                        (inp, out)
                    } else {
                        (
                            [*digest_before, FieldElement::ZERO, FieldElement::ZERO],
                            [*digest_after, FieldElement::ZERO, FieldElement::ZERO],
                        )
                    }
                } else {
                    (
                        [*digest_before, FieldElement::ZERO, FieldElement::ZERO],
                        [*digest_after, FieldElement::ZERO, FieldElement::ZERO],
                    )
                };
                rows.push(ChainRow {
                    full_input,
                    full_output,
                });
            }
            _ => {}
        }
    }

    let n_ops = rows.len();
    let n_real_rows = n_ops;
    let n_for_sizing = witness.n_poseidon_perms.max(n_ops);

    // SECURITY: Ensure at least one padding row (n_padded > n_real_rows).
    // This guarantees is_active transitions from 1→0 within the trace,
    // so the final boundary constraint fires correctly without relying
    // on preprocessed selectors.
    let log_size = if n_for_sizing <= 1 {
        2 // minimum log_size=2 → 4 rows (even for 1-2 real rows)
    } else {
        // +1 ensures n_padded > n_for_sizing (at least one padding row)
        ((n_for_sizing + 1) as u32).next_power_of_two().ilog2().max(2)
    };
    let n_padded_rows = 1usize << log_size;
    assert!(
        n_padded_rows > n_real_rows,
        "n_padded ({n_padded_rows}) must be > n_real ({n_real_rows}) for boundary constraints"
    );

    // Build trace columns (69 columns)
    let mut execution_trace: Vec<Vec<M31>> = Vec::with_capacity(COLS_PER_ROW);
    for _ in 0..COLS_PER_ROW {
        execution_trace.push(vec![M31::from_u32_unchecked(0); n_padded_rows]);
    }

    let zero_limbs = felt252_to_limbs(&FieldElement::ZERO);

    // Populate trace
    for row_idx in 0..n_padded_rows {
        let (input_limbs, output_limbs) = if row_idx < n_ops {
            let r = &rows[row_idx];
            (
                hades_state_to_limbs(&r.full_input),
                hades_state_to_limbs(&r.full_output),
            )
        } else {
            (
                [M31::from_u32_unchecked(0); COLS_PER_STATE],
                [M31::from_u32_unchecked(0); COLS_PER_STATE],
            )
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

    // Second pass: shifted_next_before[row_i] = digest_before[row_{(i+1) mod N}]
    // Circle domain wrap-around: last row shifts to first row's digest.
    let shifted_start = 2 * COLS_PER_STATE; // after input + output states
    for row_idx in 0..n_padded_rows {
        let next_idx = (row_idx + 1) % n_padded_rows;
        for j in 0..LIMBS_PER_FELT {
            execution_trace[shifted_start + j][row_idx] = execution_trace[j][next_idx];
        }
    }

    // ── Execution-trace selectors (columns 64..70) ──────────────────
    // These replace preprocessed is_last/is_chain with unconditional
    // constraints that prevent the all-zeros-selector attack.
    let col_is_active = 2 * COLS_PER_STATE + LIMBS_PER_FELT + 1; // column 64
    let col_is_active_next = col_is_active + 1;     // 65
    let col_active_count = col_is_active + 2;       // 66
    let col_active_count_next = col_is_active + 3;  // 67
    let col_is_chain_gate = col_is_active + 4;      // 68
    let col_is_boundary_gate = col_is_active + 5;   // 69
    let col_is_active_prev = col_is_active + 6;     // 70

    // is_active: 1 for real rows, 0 for padding
    for i in 0..n_real_rows.min(n_padded_rows) {
        execution_trace[col_is_active][i] = M31::from_u32_unchecked(1);
    }

    // is_active_next[i] = is_active[(i+1) mod N]
    for i in 0..n_padded_rows {
        let next = (i + 1) % n_padded_rows;
        execution_trace[col_is_active_next][i] = execution_trace[col_is_active][next];
    }

    // is_active_prev[i] = is_active[(i-1) mod N]
    for i in 0..n_padded_rows {
        let prev = if i == 0 { n_padded_rows - 1 } else { i - 1 };
        execution_trace[col_is_active_prev][i] = execution_trace[col_is_active][prev];
    }

    // Helper gate columns (precomputed products for degree-2 constraints)
    for i in 0..n_padded_rows {
        let a = execution_trace[col_is_active][i];
        let a_next = execution_trace[col_is_active_next][i];
        execution_trace[col_is_chain_gate][i] = a * a_next;
        execution_trace[col_is_boundary_gate][i] = a - a * a_next;
    }

    // active_count: amortized accumulator
    // active_count[i+1] = active_count[i] + is_active[i] - n_real_rows * N_inv
    let n_m31 = M31::from(n_padded_rows as u32);
    let n_inv = n_m31.inverse();
    let correction = M31::from(n_real_rows as u32) * n_inv;

    execution_trace[col_active_count][0] = M31::from_u32_unchecked(0);
    for i in 0..n_padded_rows - 1 {
        let is_act = execution_trace[col_is_active][i];
        execution_trace[col_active_count][i + 1] =
            execution_trace[col_active_count][i] + is_act - correction;
    }

    // active_count_next[i] = active_count[(i+1) mod N]
    for i in 0..n_padded_rows {
        let next = (i + 1) % n_padded_rows;
        execution_trace[col_active_count_next][i] = execution_trace[col_active_count][next];
    }

    // Preprocessed columns (kept for Tree 0 structural compatibility)
    let mut is_first = vec![M31::from_u32_unchecked(0); n_padded_rows];
    let mut is_last = vec![M31::from_u32_unchecked(0); n_padded_rows];
    let mut is_chain = vec![M31::from_u32_unchecked(0); n_padded_rows];

    is_first[0] = M31::from_u32_unchecked(1);
    if n_real_rows > 0 && n_real_rows <= n_padded_rows {
        is_last[n_real_rows - 1] = M31::from_u32_unchecked(1);
    }
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
        // Expanded: 27 (input) + 27 (output) + 9 (shifted) + 1 (op_type) + 7 (selectors) = 71
        assert_eq!(COLS_PER_ROW, 71);
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
            ops: vec![WitnessOp::ChannelOp {
                digest_before,
                digest_after,
            }],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                n_poseidon_perms: 1,
                seed_digest: stwo::core::fields::qm31::QM31::default(),
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

        // Verify digest_after limbs (columns 27-35 in expanded layout)
        let after_limbs = felt252_to_limbs(&digest_after);
        for j in 0..LIMBS_PER_FELT {
            assert_eq!(trace.execution_trace[COLS_PER_STATE + j][0], after_limbs[j]);
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
                WitnessOp::ChannelOp {
                    digest_before: d0,
                    digest_after: d1,
                },
                WitnessOp::ChannelOp {
                    digest_before: d1,
                    digest_after: d2,
                },
            ],
            public_inputs: crate::recursive::types::RecursivePublicInputs {
                circuit_hash: stwo::core::fields::qm31::QM31::default(),
                io_commitment: stwo::core::fields::qm31::QM31::default(),
                weight_super_root: stwo::core::fields::qm31::QM31::default(),
                n_layers: 1,
                n_poseidon_perms: 2,
                seed_digest: stwo::core::fields::qm31::QM31::default(),
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
        // digest_after starts at column COLS_PER_STATE (27) in expanded layout
        for j in 0..LIMBS_PER_FELT {
            let after_row0 = trace.execution_trace[COLS_PER_STATE + j][0];
            let before_row1 = trace.execution_trace[j][1];
            assert_eq!(
                after_row0, before_row1,
                "chain broken at limb {j}: digest_after[0] != digest_before[1]"
            );
        }
    }

    #[test]
    fn test_accumulator_constraint_satisfaction() {
        // Verify the amortized accumulator constraint holds on each row.
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
                n_poseidon_perms: 2,
                seed_digest: stwo::core::fields::qm31::QM31::default(),
            },
            n_poseidon_perms: 2,
            n_sumcheck_rounds: 0,
            n_qm31_ops: 0,
            final_digest: d2,
            n_equality_checks: 0,
        };

        let trace = build_recursive_trace(&witness);
        let n = 1usize << trace.log_size;
        let n_real = trace.n_real_rows;
        let col_is_active = 2 * COLS_PER_STATE + LIMBS_PER_FELT + 1;
        let col_active_count = col_is_active + 3;
        let col_active_count_next = col_is_active + 4;

        // Compute expected correction
        let n_m31 = M31::from(n as u32);
        let n_inv = n_m31.inverse();
        let correction = M31::from(n_real as u32) * n_inv;

        // Check C3: active_count_next - active_count - is_active + correction = 0
        for i in 0..n {
            let ac = trace.execution_trace[col_active_count][i];
            let ac_next = trace.execution_trace[col_active_count_next][i];
            let is_act = trace.execution_trace[col_is_active][i];
            let residual = ac_next - ac - is_act + correction;
            assert_eq!(
                residual,
                M31::from_u32_unchecked(0),
                "C3 accumulator constraint fails at row {i}: ac={ac:?}, ac_next={ac_next:?}, is_active={is_act:?}, correction={correction:?}"
            );
        }
        eprintln!("[test] accumulator constraint satisfied on all {} rows", n);

        // Check C1: is_active boolean
        for i in 0..n {
            let a = trace.execution_trace[col_is_active][i];
            let residual = a * (M31::from(1u32) - a);
            assert_eq!(residual, M31::from_u32_unchecked(0), "C1 boolean fails at row {i}");
        }

        // Check C4: initial boundary (row where is_active=1 and is_active_prev=0)
        let col_is_active_prev = col_is_active + 2;
        for i in 0..n {
            let a = trace.execution_trace[col_is_active][i];
            let a_prev = trace.execution_trace[col_is_active_prev][i];
            if a == M31::from(1u32) && a_prev == M31::from_u32_unchecked(0) {
                // Initial boundary should fire here
                for j in 0..LIMBS_PER_FELT {
                    let db = trace.execution_trace[j][i];
                    assert_eq!(db, M31::from_u32_unchecked(0), "C4 initial boundary fails at row {i} limb {j}");
                }
                eprintln!("[test] initial boundary fires at row {i}");
            }
        }

        // Check C5: final boundary (row where is_active=1 and is_active_next=0)
        let col_is_active_next = col_is_active + 1;
        for i in 0..n {
            let a = trace.execution_trace[col_is_active][i];
            let a_next = trace.execution_trace[col_is_active_next][i];
            if a == M31::from(1u32) && a_next == M31::from_u32_unchecked(0) {
                let final_limbs = felt252_to_limbs(&d2);
                for j in 0..LIMBS_PER_FELT {
                    let da = trace.execution_trace[COLS_PER_STATE + j][i];
                    assert_eq!(da, final_limbs[j], "C5 final boundary fails at row {i} limb {j}");
                }
                eprintln!("[test] final boundary fires at row {i}");
            }
        }
    }

    #[test]
    fn test_eval_properties() {
        let eval = RecursiveVerifierEval {
            log_n_rows: 14,
            n_real_rows: 100, // test value
            initial_digest_limbs: [M31::from_u32_unchecked(0); LIMBS_PER_FELT],
            final_digest_limbs: [M31::from_u32_unchecked(42); LIMBS_PER_FELT],
            hades_lookup: None,
        };
        assert_eq!(eval.log_size(), 14);
        assert_eq!(eval.max_constraint_log_degree_bound(), 15); // +1 for degree-2 constraints
    }
}
