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
    /// SiLU (Sigmoid Linear Unit): x * sigmoid(x) = x / (1 + exp(-x)).
    /// Used by Llama, Mistral, and most modern LLMs in the FFN gate.
    SiLU,
}

impl ActivationType {
    /// Recommended lookup table size (log2) for this activation.
    pub fn recommended_table_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 16,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
            ActivationType::SiLU => 16,
        }
    }

    /// Production-scale table sizes for real models.
    pub fn production_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,
            ActivationType::GELU => 18,
            ActivationType::Sigmoid => 16,
            ActivationType::Softmax => 20,
            ActivationType::SiLU => 18,
        }
    }

    /// Unique type tag for LogUp domain separation (M1 fix).
    /// Ensures each activation type uses distinct relation entries
    /// even when they share the same ActivationRelation random challenges.
    pub fn type_tag(&self) -> u32 {
        match self {
            ActivationType::ReLU => 1,
            ActivationType::GELU => 2,
            ActivationType::Sigmoid => 3,
            ActivationType::Softmax => 4,
            ActivationType::SiLU => 5,
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
            ActivationType::SiLU => Box::new(activations::silu_approx),
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

/// Default number of piecewise-linear segments for algebraic activation verification.
///
/// 16 segments (top 4 bits dispatch) gives ~2-5% max deviation.
///
/// Configurable via `PiecewiseLinearCoeffs::with_segments()`:
/// | Segments | Bits | Max error (GELU) | Coefficients |
/// |----------|------|-----------------|-------------|
/// | 16       | 4    | ~2-5%           | 32 M31      |
/// | 64       | 6    | ~0.1%           | 128 M31     |
/// | 256      | 8    | ~0.006%         | 512 M31     |
/// | 1024     | 10   | ~0.001%         | 2048 M31    |
pub const PIECEWISE_NUM_SEGMENTS: usize = 16;

/// Bit shift to compute segment index from an M31 value: `val >> 27` gives
/// the top 4 bits, yielding a segment index in `[0, 15]`.
pub const PIECEWISE_SEGMENT_SHIFT: u32 = 27;

/// Compute the segment shift for a given number of segments.
/// `shift = 31 - log2(num_segments)`.
pub fn segment_shift_for(num_segments: usize) -> u32 {
    assert!(num_segments.is_power_of_two(), "num_segments must be power of 2");
    assert!(num_segments >= 2 && num_segments <= (1 << 16), "num_segments must be in [2, 65536]");
    31 - num_segments.ilog2()
}

/// Piecewise-linear coefficients for approximating an activation function.
///
/// Segments divide the M31 domain `[0, P-1]` into equal-width bins.
/// Within segment `i`, the activation is approximated as:
///   `f_i(x) = slopes[i] * x + intercepts[i]`  (all arithmetic in M31)
///
/// Cost: 1 multiply + 1 add per activation (independent of segment count).
#[derive(Debug, Clone)]
pub struct PiecewiseLinearCoeffs {
    pub slopes: Vec<M31>,
    pub intercepts: Vec<M31>,
    pub segment_width: u32,
    pub num_segments: usize,
    pub segment_shift: u32,
}

impl PiecewiseLinearCoeffs {
    /// Compute piecewise-linear coefficients with the default 16 segments.
    pub fn for_activation(act_type: ActivationType) -> Self {
        Self::with_segments(act_type, PIECEWISE_NUM_SEGMENTS)
    }

    /// Compute piecewise-linear coefficients with a custom segment count.
    ///
    /// `num_segments` must be a power of 2 in [2, 65536].
    /// Uses integer-only activation evaluation for deterministic coefficients.
    pub fn with_segments(act_type: ActivationType, num_segments: usize) -> Self {
        let shift = segment_shift_for(num_segments);
        let w = 1u32 << shift;
        let mut slopes = vec![M31::from(0u32); num_segments];
        let mut intercepts = vec![M31::from(0u32); num_segments];

        for i in 0..num_segments {
            let x_start = (i as u32).wrapping_mul(w);
            let x_end = if i == num_segments - 1 {
                M31_P - 1
            } else {
                ((i + 1) as u32).wrapping_mul(w) - 1
            };

            let y_start = apply_activation_f64(act_type, x_start);
            let y_end = apply_activation_f64(act_type, x_end);

            let y_s = M31::from(y_start);
            let y_e = M31::from(y_end);
            let x_s = M31::from(x_start);
            let x_e = M31::from(x_end);

            let dx = x_e - x_s;
            let dy = y_e - y_s;
            let dx_inv = m31_inverse(dx);
            let slope = dy * dx_inv;

            slopes[i] = slope;
            intercepts[i] = y_s - slope * x_s;
        }

        Self {
            slopes,
            intercepts,
            segment_width: w,
            num_segments,
            segment_shift: shift,
        }
    }
}

/// Apply activation function to an M31 value, returning the result as u32.
///
/// Uses integer-only arithmetic for platform-deterministic results.
/// No IEEE 754 floating-point is used — all computation is in M31.
fn apply_activation_f64(act_type: ActivationType, val: u32) -> u32 {
    let tag = match act_type {
        ActivationType::ReLU => 0u8,
        ActivationType::GELU => 1,
        ActivationType::Sigmoid => 2,
        ActivationType::Softmax => 3,
        ActivationType::SiLU => 4,
    };
    crate::components::integer_math::apply_activation_integer(tag, val)
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
/// Computes segment index from top bits, then returns `slope * val + intercept`.
/// Uses the coefficients' own `segment_shift` and `num_segments` for dispatch.
pub fn piecewise_linear_eval(coeffs: &PiecewiseLinearCoeffs, val: M31) -> M31 {
    let seg_idx = (val.0 >> coeffs.segment_shift) as usize;
    let seg_idx = seg_idx.min(coeffs.num_segments - 1);
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

/// Check piecewise activation from an explicit policy configuration.
///
/// Prefer this over [`piecewise_activation_enabled()`] when a [`PolicyConfig`] is available,
/// to avoid reading environment variables.
pub fn piecewise_activation_from_policy(policy: &crate::policy::PolicyConfig) -> bool {
    policy.piecewise_activation
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

    #[test]
    fn test_configurable_segments() {
        // Verify that different segment counts produce valid coefficients
        for &n in &[16, 64, 256, 1024] {
            let coeffs = PiecewiseLinearCoeffs::with_segments(ActivationType::GELU, n);
            assert_eq!(coeffs.slopes.len(), n);
            assert_eq!(coeffs.intercepts.len(), n);
            assert_eq!(coeffs.num_segments, n);
            assert_eq!(coeffs.segment_shift, segment_shift_for(n));

            // Verify evaluation works for a sample input
            let val = M31::from(1_000_000u32);
            let result = piecewise_linear_eval(&coeffs, val);
            // Result should be a valid M31 value
            assert!(result.0 < M31_P, "result {result:?} out of M31 range");
        }
    }

    #[test]
    fn test_precision_at_segment_boundaries() {
        // At segment boundaries, the piecewise approximation must exactly match
        // the activation function evaluation (since coefficients are derived from
        // boundary values). Test this for multiple segment counts.
        for &n in &[16, 64, 256, 1024] {
            let coeffs = PiecewiseLinearCoeffs::with_segments(ActivationType::GELU, n);
            let mut boundary_errors = 0u32;

            for seg in 0..n {
                let x_start = (seg as u32).wrapping_mul(coeffs.segment_width);
                if x_start >= M31_P { continue; }

                // Piecewise eval at segment start should match the activation value
                // used to derive the coefficients
                let pw_val = piecewise_linear_eval(&coeffs, M31::from(x_start));
                let act_val = apply_activation_f64(ActivationType::GELU, x_start);

                // At exact boundary, slope * x_start + intercept should equal y_start
                // (within M31 arithmetic precision)
                let expected = M31::from(act_val);
                if pw_val != expected {
                    boundary_errors += 1;
                }
            }

            assert_eq!(
                boundary_errors, 0,
                "{n} segments: {boundary_errors} boundary mismatches"
            );
        }
    }

    #[test]
    fn test_segment_width_decreases() {
        // More segments → smaller segment width → finer approximation
        let c16 = PiecewiseLinearCoeffs::with_segments(ActivationType::SiLU, 16);
        let c256 = PiecewiseLinearCoeffs::with_segments(ActivationType::SiLU, 256);
        let c1024 = PiecewiseLinearCoeffs::with_segments(ActivationType::SiLU, 1024);
        assert!(c256.segment_width < c16.segment_width);
        assert!(c1024.segment_width < c256.segment_width);
        eprintln!("Segment widths: 16={}, 256={}, 1024={}", c16.segment_width, c256.segment_width, c1024.segment_width);
    }
}
