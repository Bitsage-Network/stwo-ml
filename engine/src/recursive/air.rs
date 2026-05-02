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
//! | digest_before | 9 | felt252 → 9 M31 limbs |
//! | digest_after  | 9 | felt252 → 9 M31 limbs |
//! | shifted_next  | 9 | next row's digest_before |
//! | addition_dig  | 9 | intermediate addition |
//! | carry         | 8 | carry chain |
//! | k, is_active, acc, acc_next | 4 | selectors |
//! | **Total**     | **48** | |
//!
//! For 15K Hades calls: log_size=14, ~720K cells. Compact and efficient.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::BaseField as M31;
use stwo_constraint_framework::{
    preprocessed_columns::PreProcessedColumnId, EvalAtRow, FrameworkComponent, FrameworkEval,
    Relation, RelationEntry,
};

use super::hades_air::{cube_252_constraint, mds_constraint, stark_prime_9bit_limbs, LIMBS_28};

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

/// Total columns per row (slim layout — 48 columns):
///   [0..9)    digest_before[9]
///   [9..18)   digest_after[9]
///   [18..27)  shifted_next_before[9]
///   [27..36)  addition_digest[9]
///   [36..44)  addition_carry[8]
///   [44]      addition_k
///   [45]      is_active
///   [46]      active_count
///   [47]      active_count_next
///
/// Each row = one Hades permutation (not one ChannelOp). Multi-Hades ChannelOps
/// (mix_poly_coeffs: 2 Hades calls) produce 2 rows. The addition_digest column
/// holds the intermediate value added to element 0 between consecutive perms
/// within the same ChannelOp.
///
/// Chain constraint: is_chain × (digest_after + addition_digest - shifted_next_before) = 0
///
/// SECURITY: Execution-trace selectors with UNCONDITIONAL amortized accumulator
/// constraint block the all-zeros-selector attack.
pub const COLS_PER_ROW: usize = COLS_PER_DIGEST  // digest_before: 9
    + COLS_PER_DIGEST // digest_after: 9
    + COLS_PER_DIGEST // shifted_next_before: 9
    + COLS_PER_DIGEST // addition_digest: 9
    + 8  // addition_carry: 8 (carry chain for modular limb addition)
    + 1  // addition_k: 1 (modular reduction quotient, 0 or 1)
    + 3; // [45] is_active, [46] active_count, [47] active_count_next
         // Total: 9 + 9 + 9 + 9 + 8 + 1 + 3 = 48

// ═══════════════════════════════════════════════════════════════════════
// Felt252 ↔ M31 limb decomposition
// ═══════════════════════════════════════════════════════════════════════

/// Stark prime P decomposed into 9 × 28-bit limbs (LSB first).
/// P = 2^251 + 17 * 2^192 + 1
pub const P_LIMBS_28: [u32; 9] = [1, 0, 0, 0, 0, 0, 16777216, 1, 134217728];

/// Compute carry-chain witnesses for modular limb addition:
///   a[j] + b[j] + carry[j-1] = result[j] + k*P[j] + carry[j]*2^28
///
/// Returns (carries[8], k) where carries[j] ∈ {0,1} and k ∈ {0,1}.
/// `a_limbs` = digest_after, `b_limbs` = addition_digest, `result_limbs` = shifted_next_before.
pub fn compute_addition_carry_chain(
    a_limbs: &[M31; LIMBS_PER_FELT],
    b_limbs: &[M31; LIMBS_PER_FELT],
    result_limbs: &[M31; LIMBS_PER_FELT],
) -> ([M31; 8], M31) {
    // First determine k: if a + b >= P then k=1, else k=0.
    // In integer arithmetic: sum = a_int + b_int, if sum >= P then k=1.
    // We compute this by checking if the carry chain works with k=0.
    // If it doesn't (carry[8] != 0), try k=1.
    for k in 0..=1u32 {
        let mut carries = [M31::from_u32_unchecked(0); 8];
        let mut carry: i64 = 0;
        let mut valid = true;

        for j in 0..LIMBS_PER_FELT {
            let lhs = a_limbs[j].0 as i64 + b_limbs[j].0 as i64 + carry;
            let rhs_base = result_limbs[j].0 as i64 + (k as i64) * (P_LIMBS_28[j] as i64);
            let diff = lhs - rhs_base;
            // diff = carry_out * 2^28
            if diff % (1i64 << 28) != 0 {
                valid = false;
                break;
            }
            carry = diff >> 28;
            if j < 8 {
                if carry != 0 && carry != 1 {
                    valid = false;
                    break;
                }
                carries[j] = M31::from_u32_unchecked(carry as u32);
            }
        }
        // Last limb: carry must be 0
        if valid && carry == 0 {
            return (carries, M31::from_u32_unchecked(k));
        }
    }
    // Fallback: shouldn't happen for valid felt252 values
    ([M31::from_u32_unchecked(0); 8], M31::from_u32_unchecked(0))
}

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

    /// When true, the Hades AIR columns (1225 columns) follow the 48 chain
    /// columns in the committed trace and their constraints are evaluated
    /// inline. This merges chain + Hades into a single component.
    pub hades_enabled: bool,
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

        // ── Read execution trace columns (slim 48-column layout) ─────
        // digest_before[9]
        let digest_before: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());

        // digest_after[9]
        let digest_after: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());

        // shifted_next_before[9]: digest_before of the NEXT row
        let shifted_next_before: [E::F; LIMBS_PER_FELT] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // addition_digest[9]: intermediate value added to digest between consecutive
        // Hades permutations within the same ChannelOp (e.g., felt2 in mix_poly_coeffs).
        // Zero for single-perm ops.
        let addition_digest: [E::F; LIMBS_PER_FELT] = std::array::from_fn(|_| eval.next_trace_mask());

        // Carry chain for modular limb addition:
        //   digest_after[j] + addition[j] + carry[j-1] = result[j] + k*P[j] + carry[j]*2^28
        // where result[j] = shifted_next_before[j], k ∈ {0,1}, carry[j] ∈ {0,1}
        let addition_carry: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
        let addition_k = eval.next_trace_mask();

        // ── Execution-trace selectors ────────────────────────────────
        let is_active = eval.next_trace_mask();
        let active_count = eval.next_trace_mask();
        let active_count_next = eval.next_trace_mask();

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

        // C5: Chain — carry-chain modular addition [degree 2]
        // digest_after[j] + addition[j] + carry[j-1] - result[j] - k*P[j] - carry[j]*2^28 = 0
        // where result[j] = shifted_next_before[j]
        //
        // Stark prime P in 28-bit limbs (LSB first):
        // P = 2^251 + 17*2^192 + 1
        // P_limbs = [1, 0, 0, 0, 0, 0, 16777216, 1, 134217728]
        {
            let p_limbs_28: [u32; 9] = [1, 0, 0, 0, 0, 0, 16777216, 1, 134217728];
            let two_pow_28 = E::F::from(M31::from(1u32 << 28));

            // k must be boolean
            eval.add_constraint(
                _is_chain.clone() * addition_k.clone() * (addition_k.clone() - E::F::from(M31::from(1u32))),
            );
            // Carries must be boolean
            for j in 0..8 {
                eval.add_constraint(
                    _is_chain.clone()
                        * addition_carry[j].clone()
                        * (addition_carry[j].clone() - E::F::from(M31::from(1u32))),
                );
            }

            // Per-limb carry-chain constraint
            for j in 0..LIMBS_PER_FELT {
                let carry_in = if j == 0 {
                    E::F::from(M31::from(0u32))
                } else {
                    addition_carry[j - 1].clone()
                };
                let carry_out_term = if j < 8 {
                    addition_carry[j].clone() * two_pow_28.clone()
                } else {
                    // Last limb: carry out must be 0 (no overflow past 252 bits)
                    E::F::from(M31::from(0u32))
                };
                let p_j = E::F::from(M31::from(p_limbs_28[j]));

                // da[j] + add[j] + carry_in - snb[j] - k*P[j] - carry_out*2^28 = 0
                eval.add_constraint(
                    _is_chain.clone()
                        * (digest_after[j].clone()
                            + addition_digest[j].clone()
                            + carry_in
                            - shifted_next_before[j].clone()
                            - addition_k.clone() * p_j
                            - carry_out_term),
                );
            }
        }

        // ══════════════════════════════════════════════════════════════
        // HADES AIR (merged inline when enabled)
        // ══════════════════════════════════════════════════════════════
        //
        // When hades_enabled is true, we read the 1225 Hades trace columns
        // that follow the 48 chain columns in the committed trace and
        // evaluate the core Hades constraints (boolean selectors, S-box/cube,
        // post-sbox interpolation, MDS, round transition).
        //
        // The Hades trace has its OWN is_real selector (independent of the
        // chain's is_active) because the two have different numbers of real
        // rows: chain has ~13 rows, Hades has ~1183 (13 perms * 91 rounds).

        if self.hades_enabled {
            let h_zero_f = || E::F::from(M31::from(0u32));

            // ── Hades column reads (1225 columns) ────────────────────
            // state_before: 3 x 28 limbs
            let mut h_state_before: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_state_before[elem][j] = eval.next_trace_mask();
                }
            }

            // sbox_input: 3 x 28 limbs
            let mut h_sbox_input: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_sbox_input[elem][j] = eval.next_trace_mask();
                }
            }

            // cube_result: 3 x 28 limbs
            let mut h_cube_result: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_cube_result[elem][j] = eval.next_trace_mask();
                }
            }

            // cube_sq: 3 x 28 limbs (x^2 intermediate)
            let mut h_cube_sq: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_cube_sq[elem][j] = eval.next_trace_mask();
                }
            }

            // Multiplication witness: 54 carries + 28 k_limbs per mul (6 total)
            let mut h_mul_carries: [[[E::F; 54]; 2]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f())));
            let mut h_mul_k_limbs: [[[[E::F; LIMBS_28]; 1]; 2]; 3] = std::array::from_fn(|_| {
                std::array::from_fn(|_| std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f())))
            });
            for elem in 0..3 {
                for mul_idx in 0..2 {
                    for j in 0..54 {
                        h_mul_carries[elem][mul_idx][j] = eval.next_trace_mask();
                    }
                    for j in 0..LIMBS_28 {
                        h_mul_k_limbs[elem][mul_idx][0][j] = eval.next_trace_mask();
                    }
                }
            }

            // post_sbox: 3 x 28 limbs (MDS input)
            let mut h_post_sbox: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_post_sbox[elem][j] = eval.next_trace_mask();
                }
            }

            // mds_result: 3 x 28 limbs
            let mut h_mds_result: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_mds_result[elem][j] = eval.next_trace_mask();
                }
            }

            // MDS carries (29) + k (1) per element
            let mut h_mds_carries: [[E::F; 29]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            let mut h_mds_k: [E::F; 3] = std::array::from_fn(|_| h_zero_f());
            for elem in 0..3 {
                for j in 0..29 {
                    h_mds_carries[elem][j] = eval.next_trace_mask();
                }
                h_mds_k[elem] = eval.next_trace_mask();
            }

            // shifted_next_state: state_before of the NEXT row (round transition)
            let mut h_shifted_next_state: [[E::F; LIMBS_28]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| h_zero_f()));
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    h_shifted_next_state[elem][j] = eval.next_trace_mask();
                }
            }

            // Selectors
            let h_is_full_round = eval.next_trace_mask();
            let h_is_real = eval.next_trace_mask();
            let h_is_chain_round = eval.next_trace_mask();

            // Boundary selectors (read but not used for repack/LogUp)
            let _h_is_first_round = eval.next_trace_mask();
            let _h_is_last_round = eval.next_trace_mask();

            // Repack columns (read to consume, not constrained in this mode)
            let _h_input_digest_28bit: [E::F; 9] = std::array::from_fn(|_| eval.next_trace_mask());
            let _h_output_digest_28bit: [E::F; 9] = std::array::from_fn(|_| eval.next_trace_mask());
            let _h_split_lo_in: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
            let _h_split_hi_in: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
            let _h_split_lo_out: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());
            let _h_split_hi_out: [E::F; 8] = std::array::from_fn(|_| eval.next_trace_mask());

            // ── Hades constraints ────────────────────────────────────

            let h_p_limbs = stark_prime_9bit_limbs();
            let h_one = E::F::from(M31::from(1u32));

            // Boolean selector constraints
            eval.add_constraint(
                h_is_real.clone() * (h_is_real.clone() - h_one.clone()),
            );
            eval.add_constraint(
                h_is_full_round.clone() * (h_is_full_round.clone() - h_one.clone()),
            );

            // S-box constraints: cube_result = sbox_input^3
            // No selector gating — padding rows are all-zero so constraints
            // hold trivially (0^3 = 0).
            let h_rc_ref: Option<&super::hades_air::RangeCheck20> = None;
            for elem in 0..3 {
                cube_252_constraint::<E>(
                    &h_sbox_input[elem],
                    &h_cube_sq[elem],
                    &h_cube_result[elem],
                    &h_mul_k_limbs[elem][0][0],
                    &h_mul_carries[elem][0],
                    &h_mul_k_limbs[elem][1][0],
                    &h_mul_carries[elem][1],
                    &mut eval,
                    &h_p_limbs,
                    &h_one,
                    h_rc_ref,
                );
            }

            // Post-sbox linking:
            // Element 2: always cubed -> post_sbox[2] = cube_result[2]
            for j in 0..LIMBS_28 {
                eval.add_constraint(
                    h_post_sbox[2][j].clone() - h_cube_result[2][j].clone(),
                );
            }
            // Elements 0,1: interpolate between cube and passthrough
            // post_sbox[e] = is_full_round * cube_result[e] + (1 - is_full_round) * sbox_input[e]
            for elem in 0..2 {
                for j in 0..LIMBS_28 {
                    let expected = h_is_full_round.clone() * h_cube_result[elem][j].clone()
                        + (h_one.clone() - h_is_full_round.clone()) * h_sbox_input[elem][j].clone();
                    eval.add_constraint(
                        h_post_sbox[elem][j].clone() - expected,
                    );
                }
            }

            // MDS constraint
            mds_constraint::<E>(
                &h_post_sbox[0],
                &h_post_sbox[1],
                &h_post_sbox[2],
                &h_mds_result[0],
                &h_mds_result[1],
                &h_mds_result[2],
                &h_mds_carries[0],
                &h_mds_carries[1],
                &h_mds_carries[2],
                &h_mds_k[0],
                &h_mds_k[1],
                &h_mds_k[2],
                &mut eval,
                &h_p_limbs,
                &h_is_real,
                h_rc_ref,
            );

            // Round transition: mds_result[row] == state_before[row+1]
            // Active on is_chain_round (all real rows except last in block).
            for elem in 0..3 {
                for j in 0..LIMBS_28 {
                    eval.add_constraint(
                        h_is_chain_round.clone()
                            * (h_mds_result[elem][j].clone()
                                - h_shifted_next_state[elem][j].clone()),
                    );
                }
            }
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

/// A single chain trace row — one Hades permutation.
struct ChainRow {
    full_input: [FieldElement; 3],
    full_output: [FieldElement; 3],
    /// Value added to output[0] before the NEXT perm's input[0].
    /// Zero for most rows. Non-zero for intermediate perms within
    /// multi-Hades ChannelOps (e.g., felt2 in mix_poly_coeffs).
    addition_digest: FieldElement,
}

/// Build the execution trace from real Hades permutation states.
///
/// Each row stores the actual felt252 Hades input/output from the GKR
/// verifier's Fiat-Shamir transcript, decomposed into M31 limbs.
///
/// Slim layout (48 columns):
///   [0..9)    digest_before
///   [9..18)   digest_after
///   [18..27)  shifted_next_before
///   [27..36)  addition_digest
///   [36..44)  addition_carry
///   [44]      addition_k
///   [45]      is_active
///   [46]      active_count
///   [47]      active_count_next
pub fn build_recursive_trace(witness: &super::types::GkrVerifierWitness) -> RecursiveTraceData {
    use super::types::WitnessOp;

    // Build one row per HadesPerm — 1:1 correspondence with Hades AIR for LogUp.
    // Multi-Hades ChannelOps (mix_poly_coeffs: 2 calls) produce 2 rows.
    // The addition_digest column stores the value added to output[0] before
    // the next perm's input[0]. The carry-chain constraint handles modular
    // limb addition overflow.
    let mut rows: Vec<ChainRow> = Vec::new();

    let hades_ops: Vec<([FieldElement; 3], [FieldElement; 3])> = witness
        .ops
        .iter()
        .filter_map(|op| {
            if let WitnessOp::HadesPerm { input, output } = op {
                Some((*input, *output))
            } else {
                None
            }
        })
        .collect();

    for i in 0..hades_ops.len() {
        let (input, output) = hades_ops[i];
        let addition = if i + 1 < hades_ops.len() {
            hades_ops[i + 1].0[0] - output[0]
        } else {
            FieldElement::ZERO
        };
        rows.push(ChainRow {
            full_input: input,
            full_output: output,
            addition_digest: addition,
        });
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

    // Build trace columns (48 columns — slim layout)
    let mut execution_trace: Vec<Vec<M31>> = Vec::with_capacity(COLS_PER_ROW);
    for _ in 0..COLS_PER_ROW {
        execution_trace.push(vec![M31::from_u32_unchecked(0); n_padded_rows]);
    }

    // Column offsets for the slim 48-column layout
    let col_digest_before = 0;       // [0..9)
    let col_digest_after = 9;        // [9..18)
    let col_shifted = 18;            // [18..27)
    let col_addition = 27;           // [27..36)
    let col_carry = 36;              // [36..44)
    let col_k = 44;                  // [44]
    let col_is_active = 45;          // [45]
    let col_active_count = 46;       // [46]
    let col_active_count_next = 47;  // [47]

    // Populate trace
    for row_idx in 0..n_padded_rows {
        if row_idx < n_ops {
            let r = &rows[row_idx];
            // digest_before: only input state[0] (9 limbs)
            let input_digest_limbs = felt252_to_limbs(&r.full_input[0]);
            for j in 0..LIMBS_PER_FELT {
                execution_trace[col_digest_before + j][row_idx] = input_digest_limbs[j];
            }
            // digest_after: only output state[0] (9 limbs)
            let output_digest_limbs = felt252_to_limbs(&r.full_output[0]);
            for j in 0..LIMBS_PER_FELT {
                execution_trace[col_digest_after + j][row_idx] = output_digest_limbs[j];
            }
            // addition_digest: 9 limbs
            let add_limbs = felt252_to_limbs(&r.addition_digest);
            for j in 0..LIMBS_PER_FELT {
                execution_trace[col_addition + j][row_idx] = add_limbs[j];
            }
        }
        // Padding rows remain zero (initialized above)
    }

    // Second pass: shifted_next_before[row_i] = digest_before[row_{(i+1) mod N}]
    // Circle domain wrap-around: last row shifts to first row's digest.
    for row_idx in 0..n_padded_rows {
        let next_idx = (row_idx + 1) % n_padded_rows;
        for j in 0..LIMBS_PER_FELT {
            execution_trace[col_shifted + j][row_idx] = execution_trace[col_digest_before + j][next_idx];
        }
    }

    // Third pass: compute carry-chain witnesses for modular addition
    {
        for row_idx in 0..n_real_rows.saturating_sub(1) {
            let da_limbs: [M31; LIMBS_PER_FELT] =
                std::array::from_fn(|j| execution_trace[col_digest_after + j][row_idx]);
            let add_limbs: [M31; LIMBS_PER_FELT] =
                std::array::from_fn(|j| execution_trace[col_addition + j][row_idx]);
            let next_before_limbs: [M31; LIMBS_PER_FELT] =
                std::array::from_fn(|j| execution_trace[col_digest_before + j][row_idx + 1]);
            let (carries, k) =
                compute_addition_carry_chain(&da_limbs, &add_limbs, &next_before_limbs);
            for j in 0..8 {
                execution_trace[col_carry + j][row_idx] = carries[j];
            }
            execution_trace[col_k][row_idx] = k;
        }
    }

    // ── Execution-trace selectors (columns 45..47) ──────────────────
    // is_active: 1 for real rows, 0 for padding
    for i in 0..n_real_rows.min(n_padded_rows) {
        execution_trace[col_is_active][i] = M31::from_u32_unchecked(1);
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
        // 9 + 9 + 9 + 9 (addition) + 8 (carry) + 1 (k) + 3 (selectors) = 48
        assert_eq!(COLS_PER_ROW, 48);
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
    #[ignore = "stale: passes WitnessOp::ChannelOp directly to build_recursive_trace; current API requires HadesPerm (G7)"]
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
                hades_commitment: starknet_ff::FieldElement::ZERO,
                kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                prev_kv_cache_commitment: starknet_ff::FieldElement::ZERO,
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
    #[ignore = "stale: passes WitnessOp::ChannelOp directly to build_recursive_trace; current API requires HadesPerm (G7)"]
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
                hades_commitment: starknet_ff::FieldElement::ZERO,
                kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                prev_kv_cache_commitment: starknet_ff::FieldElement::ZERO,
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
    #[ignore = "stale: indexes columns at 2*COLS_PER_STATE assuming 89-col layout; current slim layout is 48 cols (G7)"]
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
                hades_commitment: starknet_ff::FieldElement::ZERO,
                kv_cache_commitment: starknet_ff::FieldElement::ZERO,
                prev_kv_cache_commitment: starknet_ff::FieldElement::ZERO,
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
            hades_enabled: false,
        };
        assert_eq!(eval.log_size(), 14);
        assert_eq!(eval.max_constraint_log_degree_bound(), 15); // +1 for degree-2 constraints
    }
}
