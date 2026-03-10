//! LogUp-based activation function verification.
//!
//! Non-linear operations (ReLU, GELU, sigmoid, softmax) are prohibitively
//! expensive to arithmetize directly. Instead, we precompute lookup tables
//! and use STWO's LogUp protocol to prove each activation value exists in
//! the table.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry};

use crate::gadgets::lookup_table::PrecomputedTable;

/// Activation function type.
///
/// These are element-wise nonlinearities. LayerNorm is NOT an activation —
/// it operates on vectors and is handled by `GraphOp::LayerNorm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    /// Element-wise exp(x) component of softmax. Normalization (dividing by
    /// sum of exponents) is handled separately in the attention pipeline.
    Softmax,
}

impl ActivationType {
    /// Recommended lookup table size (log2) for this activation.
    pub fn recommended_table_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 16,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
        }
    }

    /// Production-scale table sizes for real models.
    pub fn production_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 18,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
        }
    }

    /// Unique type tag for LogUp domain separation (M1 fix).
    /// Ensures ReLU, GELU, Sigmoid, and Softmax use distinct relation entries
    /// even when they share the same ActivationRelation random challenges.
    pub fn type_tag(&self) -> u32 {
        match self {
            ActivationType::ReLU => 1,
            ActivationType::GELU => 2,
            ActivationType::Sigmoid => 3,
            ActivationType::Softmax => 4,
        }
    }

    /// Whether this activation can be computed exactly (no approximation).
    pub fn is_exact(&self) -> bool {
        matches!(self, ActivationType::ReLU)
    }

    /// Build the activation function as a closure.
    pub fn as_fn(&self) -> Box<dyn Fn(M31) -> M31 + Sync> {
        use crate::gadgets::lookup_table::activations;
        match self {
            ActivationType::ReLU => Box::new(activations::relu),
            ActivationType::GELU => Box::new(activations::gelu_approx),
            ActivationType::Sigmoid => Box::new(activations::sigmoid_approx),
            ActivationType::Softmax => Box::new(activations::softmax_exp),
        }
    }
}

// Relation type for activation lookups (type_tag, input, output).
// 3 elements: type_tag distinguishes activation types sharing the same relation.
stwo_constraint_framework::relation!(ActivationRelation, 3);

impl ActivationRelation {
    /// Access the inner LookupElements for computing LogUp fractions in the prover.
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<3> {
        &self.0
    }
}

/// Evaluator for activation lookup constraints.
#[derive(Debug, Clone)]
pub struct ActivationEval {
    pub log_n_rows: u32,
    pub lookup_elements: ActivationRelation,
    pub claimed_sum: SecureField,
    pub total_sum: SecureField,
    /// Type tag for LogUp domain separation (M1 fix).
    pub activation_type_tag: u32,
    /// Instance ID — disambiguates preprocessed column names when multiple
    /// activation layers share the unified STARK. Defaults to 0.
    pub instance_id: usize,
}

impl FrameworkEval for ActivationEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read preprocessed columns (table input, table output)
        // Instance ID disambiguates when multiple activation layers exist in the same STARK.
        let table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("activation_table_input_{}", self.instance_id).into(),
        });
        let table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: format!("activation_table_output_{}", self.instance_id).into(),
        });

        // Read execution trace columns
        let trace_input = eval.next_trace_mask();
        let trace_output = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // Type tag: constant per activation type — domain separates ReLU/GELU/etc.
        let tag = E::F::from(BaseField::from(self.activation_type_tag));

        // LogUp: table side (yield)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity.clone()),
            &[tag.clone(), table_input, table_output],
        ));

        // LogUp: trace side (use)
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[tag, trace_input, trace_output],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

/// Type alias for the activation component.
pub type ActivationComponent = FrameworkComponent<ActivationEval>;

/// Generate the preprocessed activation table as trace columns.
///
/// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
pub fn generate_activation_table<B: ColumnOps<BaseField>>(
    table: &PrecomputedTable,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    table.generate_trace_columns::<B>()
}

/// Generate the main execution trace for activation lookups.
///
/// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
pub fn generate_activation_trace<B: ColumnOps<BaseField>>(
    inputs: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    assert_eq!(inputs.len(), outputs.len());
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut input_col = Col::<B, BaseField>::zeros(size);
    let mut output_col = Col::<B, BaseField>::zeros(size);
    let mut mult_col = Col::<B, BaseField>::zeros(size);

    for (i, (&inp, &out)) in inputs.iter().zip(outputs.iter()).enumerate().take(size) {
        input_col.set(i, inp);
        output_col.set(i, out);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    vec![
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, mult_col),
    ]
}

/// Compute multiplicities for the activation table given the trace inputs.
///
/// Uses the table's hash index (O(1) lookup) when available for tables >= 2^10,
/// falling back to linear scan for smaller tables.
pub fn compute_multiplicities(trace_inputs: &[M31], table: &PrecomputedTable) -> Vec<M31> {
    let mut multiplicities = vec![0u32; table.size()];

    for input in trace_inputs {
        if let Some(idx) = table.lookup_index(*input) {
            multiplicities[idx] += 1;
        }
    }

    multiplicities.into_iter().map(M31::from).collect()
}

/// M31 modulus P = 2^31 - 1.
const M31_P: u32 = (1u32 << 31) - 1;

/// Number of piecewise-linear segments for algebraic activation verification.
pub const PIECEWISE_NUM_SEGMENTS: usize = 16;

/// Bit shift to compute segment index from an M31 value: `val >> 27` gives
/// the top 4 bits, yielding a segment index in `[0, 15]`.
pub const PIECEWISE_SEGMENT_SHIFT: u32 = 27;

/// Piecewise-linear coefficients for approximating an activation function
/// over 16 equal-width segments of the M31 domain `[0, P-1]`.
///
/// Each segment `i` covers `[i*W, (i+1)*W - 1]` where `W = P / 16`.
/// Within segment `i`, the activation is approximated as:
///   `f_i(x) = slopes[i] * x + intercepts[i]`  (all arithmetic in M31)
#[derive(Debug, Clone)]
pub struct PiecewiseLinearCoeffs {
    pub slopes: [M31; PIECEWISE_NUM_SEGMENTS],
    pub intercepts: [M31; PIECEWISE_NUM_SEGMENTS],
    pub segment_width: u32,
}

impl PiecewiseLinearCoeffs {
    /// Compute piecewise-linear coefficients for the given activation type.
    ///
    /// For each segment, evaluates the f64 activation at segment boundaries,
    /// converts to M31 fixed-point, and computes slope/intercept via linear
    /// interpolation.
    pub fn for_activation(act_type: ActivationType) -> Self {
        let w = 1u32 << PIECEWISE_SEGMENT_SHIFT; // segment width = 2^27 (power-of-2 aligned with bit-shift dispatch)
        let mut slopes = [M31::from(0u32); PIECEWISE_NUM_SEGMENTS];
        let mut intercepts = [M31::from(0u32); PIECEWISE_NUM_SEGMENTS];

        for i in 0..PIECEWISE_NUM_SEGMENTS {
            let x_start = (i as u32).wrapping_mul(w);
            let x_end = if i == PIECEWISE_NUM_SEGMENTS - 1 {
                M31_P - 1
            } else {
                ((i + 1) as u32).wrapping_mul(w) - 1
            };

            let y_start = apply_activation_f64(act_type, x_start);
            let y_end = apply_activation_f64(act_type, x_end);

            // slope = (y_end - y_start) / (x_end - x_start)  in M31
            // intercept = y_start - slope * x_start           in M31
            let y_s = M31::from(y_start);
            let y_e = M31::from(y_end);
            let x_s = M31::from(x_start);
            let x_e = M31::from(x_end);

            // Compute slope as (y_e - y_s) * inverse(x_e - x_s) in M31
            let dx = x_e - x_s;
            let dy = y_e - y_s;
            // Use Fermat's little theorem for inverse: a^{P-2} mod P
            let dx_inv = m31_inverse(dx);
            let slope = dy * dx_inv;

            slopes[i] = slope;
            intercepts[i] = y_s - slope * x_s;
        }

        // Verify coefficient non-degeneracy: no two segments share the same (slope, intercept)
        for i in 0..PIECEWISE_NUM_SEGMENTS {
            for j in (i + 1)..PIECEWISE_NUM_SEGMENTS {
                debug_assert!(
                    slopes[i] != slopes[j] || intercepts[i] != intercepts[j],
                    "degenerate piecewise coefficients: segments {i} and {j} are identical"
                );
            }
        }

        Self {
            slopes,
            intercepts,
            segment_width: w,
        }
    }
}

/// Apply activation function to an M31 value, returning the result as u32.
///
/// Interprets the M31 value as a signed integer (values > P/2 are negative),
/// applies the f64 activation, and maps back to M31.
fn apply_activation_f64(act_type: ActivationType, val: u32) -> u32 {
    let half_p = M31_P / 2;
    // Map M31 to signed: values > P/2 are negative
    let signed = if val <= half_p {
        val as f64
    } else {
        val as f64 - M31_P as f64
    };

    // Scale down to [-1, 1] range for activation computation
    let scale = half_p as f64;
    let x = signed / scale;

    let y = match act_type {
        ActivationType::ReLU => x.max(0.0),
        ActivationType::GELU => {
            // GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
        ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        ActivationType::Softmax => x.exp(), // element-wise exp component
    };

    // Scale back and map to M31
    let result = y * scale;
    let result_i64 = result.round() as i64;
    // Map to [0, P-1]
    let result_mod = ((result_i64 % (M31_P as i64)) + M31_P as i64) as u32 % M31_P;
    result_mod
}

/// Compute M31 multiplicative inverse via Fermat's little theorem: a^{P-2} mod P.
fn m31_inverse(a: M31) -> M31 {
    debug_assert!(a != M31::from(0u32), "m31_inverse called with zero");
    // P - 2 = 2^31 - 3
    let mut result = M31::from(1u32);
    let mut base = a;
    let mut exp = M31_P - 2;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base;
        }
        base = base * base;
        exp >>= 1;
    }
    result
}

/// Evaluate the piecewise-linear approximation at a single M31 value.
///
/// Computes segment index from top 4 bits, then returns `slope * val + intercept`.
pub fn piecewise_linear_eval(coeffs: &PiecewiseLinearCoeffs, val: M31) -> M31 {
    let seg_idx = (val.0 >> PIECEWISE_SEGMENT_SHIFT) as usize;
    debug_assert!(seg_idx < PIECEWISE_NUM_SEGMENTS, "segment index {seg_idx} out of range for val {}", val.0);
    // Clamp to valid range (defensive — with 2^27 width, (P-1)>>27 = 15 is always valid)
    let seg_idx = seg_idx.min(PIECEWISE_NUM_SEGMENTS - 1);
    coeffs.slopes[seg_idx] * val + coeffs.intercepts[seg_idx]
}

/// Check whether piecewise activation mode is enabled (default: ON).
///
/// Piecewise-linear algebraic activation proves correctness over the full M31 domain
/// via a 16-segment eq-sumcheck with no lookup tables. Opt-out for legacy LogUp path:
/// `STWO_PIECEWISE_ACTIVATION=0` (or `false`/`no`).
pub fn piecewise_activation_enabled() -> bool {
    std::env::var("STWO_PIECEWISE_ACTIVATION")
        .map(|v| {
            let s = v.trim();
            // Opt-out: 0 / false / no / off → disable piecewise
            !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("no") || s.eq_ignore_ascii_case("off"))
        })
        .unwrap_or(true)
}

/// Error type for activation proving.
#[derive(Debug, thiserror::Error)]
pub enum ActivationError {
    #[error("Input value not found in table: {0:?}")]
    ValueNotInTable(M31),
    #[error("Input/output length mismatch: {0} inputs, {1} outputs")]
    LengthMismatch(usize, usize),
    #[error("Proving error: {0}")]
    ProvingError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_sizes() {
        assert_eq!(ActivationType::ReLU.recommended_table_log_size(), 16);
        assert_eq!(ActivationType::Softmax.recommended_table_log_size(), 20);
        assert!(ActivationType::ReLU.is_exact());
        assert!(!ActivationType::GELU.is_exact());
    }

    #[test]
    fn test_production_sizes() {
        assert_eq!(ActivationType::GELU.production_log_size(), 18);
        assert_eq!(ActivationType::Softmax.production_log_size(), 20);
    }

    #[test]
    fn test_generate_activation_table() {
        use stwo::prover::backend::simd::SimdBackend;
        let table = PrecomputedTable::build(crate::gadgets::lookup_table::activations::relu, 4);
        let cols = generate_activation_table::<SimdBackend>(&table);
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn test_compute_multiplicities() {
        let table = PrecomputedTable::build(|x| x, 4);
        let inputs = vec![M31::from(0), M31::from(1), M31::from(0), M31::from(3)];
        let mults = compute_multiplicities(&inputs, &table);

        assert_eq!(mults[0], M31::from(2));
        assert_eq!(mults[1], M31::from(1));
        assert_eq!(mults[2], M31::from(0));
        assert_eq!(mults[3], M31::from(1));
    }

    #[test]
    fn test_generate_trace() {
        use stwo::prover::backend::simd::SimdBackend;
        let inputs = vec![M31::from(1), M31::from(2)];
        let outputs = vec![M31::from(1), M31::from(2)];
        let mults = vec![M31::from(1), M31::from(1)];

        let trace = generate_activation_trace::<SimdBackend>(&inputs, &outputs, &mults, 4);
        assert_eq!(trace.len(), 3);
    }
}
