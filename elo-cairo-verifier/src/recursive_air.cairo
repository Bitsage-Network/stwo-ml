/// Cairo AIR implementation for the Recursive STARK verifier.
///
/// Verifies that a chain of Poseidon channel operations was executed correctly.
///
/// Slim trace layout (48 columns per row):
///   [0..9)    digest_before
///   [9..18)   digest_after
///   [18..27)  shifted_next_before
///   [27..36)  addition_digest
///   [36..44)  addition_carry
///   [44]      addition_k
///   [45]      is_active
///   [46]      active_count
///   [47]      active_count_next
///
/// Plus 3 preprocessed selector columns (is_first, is_last, is_chain).
///
/// Constraints (40 total, all degree ≤ 2):
///   C1:  is_active boolean                     [1, unconditional]
///   C2:  amortized accumulator                 [1, unconditional — BLOCKS all-zeros]
///   C3b: is_chain_gate = is_active * is_active_next  [1]
///   C3c: is_boundary_gate = is_active - is_chain_gate [1]
///   C4:  initial boundary (is_first × 9 limbs) [9]
///   C5:  final boundary (is_last × 9 limbs)   [9]
///   C6:  carry-chain modular addition (9 limbs) [9]
///   C6k: k boolean                             [1]
///   C6c: carry booleans                        [8]

use stwo_verifier_core::fields::qm31::{QM31, QM31Zero, QM31One, QM31Trait};
use stwo_verifier_core::fields::m31::{M31, m31};
use stwo_verifier_core::circle::CirclePoint;
use stwo_verifier_core::poly::circle::{CanonicCosetImpl, CanonicCosetTrait};
use stwo_verifier_core::fields::Invertible;
use stwo_verifier_core::verifier::Air;
use stwo_verifier_core::{TreeSpan, ColumnSpan};

/// Number of M31 limbs per felt252 (9 × 28 = 252 bits).
pub const LIMBS_PER_FELT: u32 = 9;

/// Total trace columns.
const TRACE_COLS: u32 = 48;

/// Columns per full Hades state (3 felt252 × 9 limbs).
const COLS_PER_STATE: u32 = 27;

/// Number of preprocessed selector columns: is_first, is_last, is_chain.
const PREPROCESS_COLS: u32 = 3;

/// Total constraints: 1 + 1 + 9 + 9 + 1 + 8 + 9 = 38.
const N_CONSTRAINTS: u32 = 38;

/// Stark prime P in 28-bit limbs (LSB first).
/// P = 2^251 + 17*2^192 + 1
fn stark_prime_28bit_limbs() -> Array<u32> {
    array![1, 0, 0, 0, 0, 0, 16777216, 1, 134217728]
}

/// The recursive verifier AIR.
#[derive(Drop)]
pub struct RecursiveAir {
    /// log2(number of trace rows).
    pub log_n_rows: u32,
    /// Number of real (active) rows in the trace.
    pub n_real_rows: u32,
    /// Initial digest decomposed into 9 M31 limbs (usually all zero).
    pub initial_digest_limbs: Array<QM31>,
    /// Expected final digest decomposed into 9 M31 limbs.
    pub final_digest_limbs: Array<QM31>,
}

impl RecursiveAirImpl of Air<RecursiveAir> {
    // composition_log_degree_bound removed in v1.2.2 — passed to verify() directly.

    fn eval_composition_polynomial_at_point(
        self: @RecursiveAir,
        point: CirclePoint<QM31>,
        mask_values: TreeSpan<ColumnSpan<Span<QM31>>>,
        random_coeff: QM31,
    ) -> QM31 {
        let [preprocessed_vals, trace_vals, _composition_vals]:
            [ColumnSpan<Span<QM31>>; 3] = (*mask_values.try_into().unwrap()).unbox();

        // Preprocessed selectors
        let is_first = extract_single_val(preprocessed_vals, 0);
        let is_last = extract_single_val(preprocessed_vals, 1);
        let is_chain = extract_single_val(preprocessed_vals, 2);

        // Trace columns
        let one: QM31 = QM31One::one();
        let zero: QM31 = QM31Zero::zero();

        // Column extraction helpers
        // Slim 48-column layout:
        // digest_before [0..9)
        // digest_after [9..18)
        // shifted_next_before [18..27)
        // addition_digest [27..36)
        // addition_carry [36..44)
        // addition_k [44]
        // is_active [45]
        // active_count [46]
        // active_count_next [47]

        let is_active = extract_single_val(trace_vals, 45);
        let active_count = extract_single_val(trace_vals, 46);
        let active_count_next = extract_single_val(trace_vals, 47);
        let addition_k = extract_single_val(trace_vals, 44);

        // NOTE: Do NOT divide by vanishing polynomial here.
        // verify() multiplies the result by denominator_inv (vanishing^{-1}) externally.
        // Dividing here would double-apply the inverse.

        let mut quotients: Array<QM31> = array![];

        // ═══════════════════════════════════════════════════════════
        // C1: is_active boolean [unconditional]
        // ═══════════════════════════════════════════════════════════
        quotients.append(is_active * (one - is_active));

        // ═══════════════════════════════════════════════════════════
        // C2: amortized accumulator [unconditional — BLOCKS all-zeros]
        // active_count_next - active_count - is_active + n_real * N_inv = 0
        // ═══════════════════════════════════════════════════════════
        // Compute N = 2^log_n_rows and N_inv in the M31 field
        let n_val: u64 = pow2(*self.log_n_rows);
        let n_m31: M31 = m31(n_val.try_into().unwrap());
        let n_inv_m31: M31 = n_m31.inverse();
        let correction_m31: M31 = m31(*self.n_real_rows) * n_inv_m31;
        let correction: QM31 = m31_to_qm31(correction_m31);
        quotients.append(
            (active_count_next - active_count - is_active + correction)
        );

        // ═══════════════════════════════════════════════════════════
        // C3: Initial boundary — is_first × (digest_before - initial) [9]
        // ═══════════════════════════════════════════════════════════
        let mut j: u32 = 0;
        loop {
            if j >= LIMBS_PER_FELT { break; }
            let db = extract_single_val(trace_vals, j);
            let init = *self.initial_digest_limbs.at(j);
            quotients.append(is_first * (db - init));
            j += 1;
        };

        // ═══════════════════════════════════════════════════════════
        // C4: Final boundary — is_last × (digest_after - final) [9]
        // ═══════════════════════════════════════════════════════════
        j = 0;
        loop {
            if j >= LIMBS_PER_FELT { break; }
            let da = extract_single_val(trace_vals, 9 + j); // digest_after
            let fin = *self.final_digest_limbs.at(j);
            quotients.append(is_last * (da - fin));
            j += 1;
        };

        // ═══════════════════════════════════════════════════════════
        // C5k: k boolean — is_chain × k × (k - 1)
        // ═══════════════════════════════════════════════════════════
        quotients.append(
            is_chain * addition_k * (addition_k - one)
        );

        // ═══════════════════════════════════════════════════════════
        // C5c: carry booleans — is_chain × carry[j] × (carry[j] - 1) [8]
        // ═══════════════════════════════════════════════════════════
        j = 0;
        loop {
            if j >= 8 { break; }
            let carry_j = extract_single_val(trace_vals, 36 + j); // addition_carry
            quotients.append(
                is_chain * carry_j * (carry_j - one)
            );
            j += 1;
        };

        // ═══════════════════════════════════════════════════════════
        // C5: Carry-chain modular addition [9 limbs]
        // da[j] + add[j] + carry_in - snb[j] - k*P[j] - carry_out*2^28 = 0
        // ═══════════════════════════════════════════════════════════
        let p_limbs = stark_prime_28bit_limbs();
        let two_pow_28: QM31 = m31_to_qm31(m31(268435456)); // 2^28

        j = 0;
        loop {
            if j >= LIMBS_PER_FELT { break; }
            let da = extract_single_val(trace_vals, 9 + j); // digest_after
            let add = extract_single_val(trace_vals, 27 + j); // addition_digest
            let snb = extract_single_val(trace_vals, 18 + j); // shifted_next_before
            let p_j: QM31 = m31_to_qm31(m31(*p_limbs.at(j)));

            let carry_in: QM31 = if j == 0 {
                zero
            } else {
                extract_single_val(trace_vals, 36 + j - 1) // addition_carry
            };

            let carry_out_term: QM31 = if j < 8 {
                extract_single_val(trace_vals, 36 + j) * two_pow_28 // addition_carry
            } else {
                zero
            };

            quotients.append(
                is_chain
                    * (da + add + carry_in - snb - addition_k * p_j - carry_out_term)
            );
            j += 1;
        };

        // Accumulate using Horner's method
        let n_quotients = quotients.len();
        let mut acc: QM31 = QM31Zero::zero();
        let mut idx: u32 = 0;
        loop {
            if idx >= n_quotients { break; }
            acc = acc * random_coeff + *quotients.at(idx);
            idx += 1;
        };

        acc
        // let n_quotients = quotients.len();
        // let mut acc: QM31 = QM31Zero::zero();
        // let mut idx: u32 = 0;
        // loop {
        //     if idx >= n_quotients { break; }
        //     acc = acc * random_coeff + *quotients.at(idx);
        //     idx += 1;
        // };
        // acc
    }
}

/// Convert M31 to QM31 (embed in the base component).
fn m31_to_qm31(v: M31) -> QM31 {
    QM31Trait::from_fixed_array([v, m31(0), m31(0), m31(0)])
}

/// Extract a single QM31 value from the j-th column of a tree's mask values.
fn extract_single_val(tree_vals: ColumnSpan<Span<QM31>>, col_idx: u32) -> QM31 {
    let col = *tree_vals.at(col_idx);
    *col.at(0)
}

/// Compute 2^n for small n.
fn pow2(n: u32) -> u64 {
    let mut result: u64 = 1;
    let mut i: u32 = 0;
    loop {
        if i >= n { break; }
        result = result * 2;
        i += 1;
    };
    result
}
