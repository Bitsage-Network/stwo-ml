/// Cairo AIR implementation for the Recursive STARK verifier.
///
/// Verifies that a chain of Poseidon channel operations was executed correctly.
///
/// Expanded trace layout (64 columns per row):
///   [0..9)    digest_before     (input state[0])
///   [9..18)   input_value       (input state[1])
///   [18..27)  input_capacity    (input state[2])
///   [27..36)  digest_after      (output state[0])
///   [36..45)  output_value      (output state[1])
///   [45..54)  output_capacity   (output state[2])
///   [54..63)  shifted_next_before (next row's digest_before)
///   [63]      op_type
///
/// Plus 3 preprocessed selector columns: is_first, is_last, is_chain.
///
/// Constraints (all degree ≤ 2):
///   - Boundary (first row): digest_before[j] == initial_digest_limbs[j]  (9 constraints)
///   - Boundary (last row):  digest_after[j]  == final_digest_limbs[j]    (9 constraints)
///   - Chain (internal rows): digest_after[j]  == shifted_next_before[j]   (9 constraints)

use stwo_verifier_core::fields::qm31::{QM31, QM31Zero, QM31One, QM31Trait};
use stwo_verifier_core::fields::m31::M31;
use stwo_verifier_core::circle::CirclePoint;
use stwo_verifier_core::poly::circle::{CanonicCosetImpl, CanonicCosetTrait};
use stwo_verifier_core::fields::Invertible;
use stwo_verifier_core::verifier::Air;
use stwo_verifier_core::{TreeSpan, ColumnSpan};

/// Number of M31 limbs per felt252 (9 × 31 = 279 bits ≥ 252).
pub const LIMBS_PER_FELT: u32 = 9;

/// Total trace columns (expanded):
///   input_state(27) + output_state(27) + shifted_next(9) + op_type(1) = 64.
const TRACE_COLS: u32 = 64;

/// Columns per full Hades state (3 felt252 × 9 limbs).
const COLS_PER_STATE: u32 = 27;

/// Number of preprocessed selector columns: is_first, is_last, is_chain.
const PREPROCESS_COLS: u32 = 3;

/// Total constraints: 9 boundary_first + 9 boundary_last + 9 chain = 27.
const N_CONSTRAINTS: u32 = 27;

/// The recursive verifier AIR.
#[derive(Drop)]
pub struct RecursiveAir {
    /// log2(number of trace rows).
    pub log_n_rows: u32,
    /// Initial digest decomposed into 9 M31 limbs (usually all zero).
    pub initial_digest_limbs: Array<QM31>,
    /// Expected final digest decomposed into 9 M31 limbs.
    pub final_digest_limbs: Array<QM31>,
}

impl RecursiveAirImpl of Air<RecursiveAir> {
    fn composition_log_degree_bound(self: @RecursiveAir) -> u32 {
        // All constraints are degree ≤ 2 (selector × difference).
        *self.log_n_rows + 1
    }

    /// Evaluate the composition polynomial at an out-of-domain point.
    ///
    /// mask_values layout: [preprocessed(3 cols), trace(28 cols), interaction(0), composition(8)]
    /// Each column has 1 sample (the OOD evaluation).
    fn eval_composition_polynomial_at_point(
        self: @RecursiveAir,
        point: CirclePoint<QM31>,
        mask_values: TreeSpan<ColumnSpan<Span<QM31>>>,
        random_coeff: QM31,
    ) -> QM31 {
        // Destructure the 3 commitment trees (no interaction trace in recursive proof)
        let [preprocessed_vals, trace_vals, _composition_vals]:
            [ColumnSpan<Span<QM31>>; 3] = (*mask_values.try_into().unwrap()).unbox();

        // Extract preprocessed selectors (each is a single QM31 sample)
        let is_first = extract_single_val(preprocessed_vals, 0);
        let is_last = extract_single_val(preprocessed_vals, 1);
        let is_chain = extract_single_val(preprocessed_vals, 2);

        // Extract trace columns (expanded layout):
        // Columns 0-8: digest_before (9 limbs)
        // Columns 9-17: input_value (9 limbs)
        // Columns 18-26: input_capacity (9 limbs)
        // Columns 27-35: digest_after (9 limbs)
        // Columns 36-44: output_value (9 limbs)
        // Columns 45-53: output_capacity (9 limbs)
        // Columns 54-62: shifted_next_before (9 limbs)
        // Column 63: op_type (unused in constraints)

        // Compute the vanishing polynomial inverse at the OOD point.
        // The denominator uses CanonicCoset::new(max_log_degree_bound) where
        // max_log_degree_bound = composition_log_size - COMPOSITION_LOG_SPLIT
        // = (log_size + 1) - 1 = log_size = log_n_rows.
        let constraint_domain = CanonicCosetImpl::new(*self.log_n_rows);
        let domain_vanishing_eval_inv = constraint_domain.eval_vanishing(point).inverse();

        // Collect all 27 constraint quotients in ORDER (same as Rust add_constraint calls)
        let mut quotients: Array<QM31> = array![];

        // ── Boundary: first row's digest_before = initial_digest (9 constraints) ──
        let mut j: u32 = 0;
        loop {
            if j >= LIMBS_PER_FELT { break; }
            let digest_before_j = extract_single_val(trace_vals, j);
            let initial_j = *self.initial_digest_limbs.at(j);
            quotients.append(domain_vanishing_eval_inv * is_first * (digest_before_j - initial_j));
            j += 1;
        };

        // ── Boundary: last row's digest_after = final_digest (9 constraints) ──
        // digest_after is at columns [27..36) in the expanded layout
        j = 0;
        loop {
            if j >= LIMBS_PER_FELT { break; }
            let digest_after_j = extract_single_val(trace_vals, COLS_PER_STATE + j);
            let final_j = *self.final_digest_limbs.at(j);
            quotients.append(domain_vanishing_eval_inv * is_last * (digest_after_j - final_j));
            j += 1;
        };

        // ── Chain: digest_after[row] == shifted_next_before[row] (9 constraints) ──
        // digest_after at [27..36), shifted_next_before at [54..63)
        j = 0;
        loop {
            if j >= LIMBS_PER_FELT { break; }
            let digest_after_j = extract_single_val(trace_vals, COLS_PER_STATE + j);
            let shifted_next_j = extract_single_val(trace_vals, 2 * COLS_PER_STATE + j);
            quotients.append(domain_vanishing_eval_inv * is_chain * (digest_after_j - shifted_next_j));
            j += 1;
        };

        // Accumulate using Horner's method (FORWARD order, matching STWO's
        // PointEvaluationAccumulator::accumulate: acc = acc * r + eval)
        let n = quotients.len();
        let mut acc: QM31 = QM31Zero::zero();
        let mut idx: u32 = 0;
        loop {
            if idx >= n { break; }
            acc = acc * random_coeff + *quotients.at(idx);
            idx += 1;
        };

        acc
    }
}

/// Extract a single QM31 value from the j-th column of a tree's mask values.
fn extract_single_val(tree_vals: ColumnSpan<Span<QM31>>, col_idx: u32) -> QM31 {
    let col = *tree_vals.at(col_idx);
    *col.at(0)
}
