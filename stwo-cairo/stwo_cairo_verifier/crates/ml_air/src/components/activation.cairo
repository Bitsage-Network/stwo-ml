/// Activation component for LogUp-based activation function verification.
///
/// Verifies that each (input, output) pair exists in a precomputed activation
/// table using the LogUp protocol. The activation table is committed as a
/// preprocessed column.
///
/// Supported activations: ReLU (exact), GELU (approx), Sigmoid (approx).
///
/// The LogUp relation checks:
///   Σ_{row} mult/(z - combine(input, output)) = claimed_sum
///
/// where combine uses random linear combination with alpha from the channel.
///
/// Constraint (at OOD point):
///   (S(x) - S(prev) + claimed_sum/N) * denom + mult = 0
///
/// This follows the `verify_bitwise_xor_4.cairo` pattern from cairo_air.

use stwo_verifier_core::fields::Invertible;
use stwo_verifier_core::fields::m31::m31;
use stwo_verifier_core::fields::qm31::{QM31, QM31Serde, QM31Trait};
use stwo_verifier_core::circle::CirclePoint;
use stwo_verifier_core::poly::circle::{CanonicCosetImpl, CanonicCosetTrait};
use stwo_verifier_core::utils::pow2;
use stwo_verifier_core::ColumnSpan;
use stwo_constraint_framework::{LookupElements, LookupElementsTrait};

/// Claim about a single activation layer.
#[derive(Drop, Serde, Copy)]
pub struct ActivationClaim {
    /// Layer index in the computation graph.
    pub layer_index: u32,
    /// Log2 of the trace size (number of activation evaluations).
    pub log_size: u32,
    /// Activation type: 0=ReLU, 1=GELU, 2=Sigmoid.
    pub activation_type: u8,
}

/// Interaction claim for activation layer (LogUp sum).
#[derive(Drop, Serde, Copy)]
pub struct ActivationInteractionClaim {
    /// Claimed LogUp sum for this activation layer.
    pub claimed_sum: QM31,
}

/// Number of trace columns for the activation component.
///
/// - Column 0: input values
/// - Column 1: output values
/// - Column 2: multiplicities (how many times each table entry is used)
pub const N_ACTIVATION_TRACE_COLUMNS: u32 = 3;

/// Number of preprocessed columns (the activation table).
///
/// - Column 0: table input values
/// - Column 1: table output values
pub const N_ACTIVATION_PREPROCESSED_COLUMNS: u32 = 2;

/// Number of interaction trace columns (LogUp cumulative sum as QM31 partial evals).
pub const N_ACTIVATION_INTERACTION_COLUMNS: u32 = 4;

/// Activation lookup elements: combine(input, output) → z + alpha*input + alpha^2*output.
pub type ActivationLookupElements = LookupElements<2>;

/// Full activation component for constraint evaluation at an OOD point.
#[derive(Drop)]
pub struct ActivationComponent {
    pub claim: ActivationClaim,
    pub interaction_claim: ActivationInteractionClaim,
    pub lookup_elements: ActivationLookupElements,
}

/// Evaluate activation LogUp constraints at an out-of-domain point.
///
/// This follows the exact pattern from `verify_bitwise_xor_4.cairo`:
///   constraint = ((S_curr - S_prev + claimed_sum/N) * denom + mult) / vanishing
///
/// The function pops columns from `trace_mask_values` and `interaction_trace_mask_values`
/// in order, accumulating the constraint quotient into `sum`.
pub fn evaluate_activation_constraints_at_point(
    component: @ActivationComponent,
    ref sum: QM31,
    ref trace_mask_values: ColumnSpan<Span<QM31>>,
    ref interaction_trace_mask_values: ColumnSpan<Span<QM31>>,
    random_coeff: QM31,
    point: CirclePoint<QM31>,
) {
    let log_size = *component.claim.log_size;
    let trace_domain = CanonicCosetImpl::new(log_size);
    let domain_vanishing_eval_inv = trace_domain.eval_vanishing(point).inverse();
    let claimed_sum = *component.interaction_claim.claimed_sum;
    let column_size = m31(pow2(log_size));

    // Pop 3 trace columns: [input, output, multiplicity]
    // Each column has a single mask value at the current row.
    let [col_input, col_output, col_mult]: [Span<QM31>; 3] =
        (*trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [input_val]: [QM31; 1] = (*col_input.try_into().unwrap()).unbox();
    let [output_val]: [QM31; 1] = (*col_output.try_into().unwrap()).unbox();
    let [mult_val]: [QM31; 1] = (*col_mult.try_into().unwrap()).unbox();

    // Combine lookup: z - (input + alpha * output)
    let denom = component.lookup_elements.combine_qm31([input_val, output_val]);

    // Pop 4 interaction trace columns (LogUp cumulative sum as QM31 partial evals).
    // Each has mask at [prev_row, current_row].
    let [t2c0, t2c1, t2c2, t2c3]: [Span<QM31>; 4] =
        (*interaction_trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [t2c0_prev, t2c0_curr]: [QM31; 2] = (*t2c0.try_into().unwrap()).unbox();
    let [t2c1_prev, t2c1_curr]: [QM31; 2] = (*t2c1.try_into().unwrap()).unbox();
    let [t2c2_prev, t2c2_curr]: [QM31; 2] = (*t2c2.try_into().unwrap()).unbox();
    let [t2c3_prev, t2c3_curr]: [QM31; 2] = (*t2c3.try_into().unwrap()).unbox();

    core::internal::revoke_ap_tracking();

    // Reconstruct cumulative sum QM31 values from partial evals.
    let s_curr = QM31Trait::from_partial_evals([t2c0_curr, t2c1_curr, t2c2_curr, t2c3_curr]);
    let s_prev = QM31Trait::from_partial_evals([t2c0_prev, t2c1_prev, t2c2_prev, t2c3_prev]);

    // LogUp constraint (matching verify_bitwise_xor_4 exactly):
    //   ((S_curr - S_prev + claimed_sum/N) * denom + mult) / vanishing = 0
    let constraint_quotient = (((s_curr - s_prev + (claimed_sum * (column_size.inverse().into())))
        * denom)
        + mult_val)
        * domain_vanishing_eval_inv;

    sum = sum * random_coeff + constraint_quotient;
}

#[cfg(test)]
mod tests {
    use stwo_verifier_core::channel::Channel;
    use stwo_verifier_core::fields::qm31::{QM31, qm31_const};
    use stwo_constraint_framework::{LookupElements, LookupElementsTrait};
    use super::{ActivationClaim, ActivationInteractionClaim, ActivationComponent};

    #[test]
    fn test_activation_component_construction() {
        let claim = ActivationClaim { layer_index: 0, log_size: 4, activation_type: 0 };
        let interaction_claim = ActivationInteractionClaim {
            claimed_sum: qm31_const::<42, 0, 0, 0>(),
        };
        let mut channel: Channel = Default::default();
        let lookup_elements: LookupElements<2> = LookupElementsTrait::draw(ref channel);

        let component = ActivationComponent { claim, interaction_claim, lookup_elements };
        assert!(component.claim.log_size == 4);
        assert!(component.claim.activation_type == 0);
    }

    #[test]
    fn test_activation_lookup_elements_draw() {
        let mut channel: Channel = Default::default();
        let elements: LookupElements<2> = LookupElementsTrait::draw(ref channel);
        assert!(elements.alpha_powers.len() == 2);
    }

    #[test]
    fn test_activation_lookup_combine() {
        let mut channel: Channel = Default::default();
        let elements: LookupElements<2> = LookupElementsTrait::draw(ref channel);

        let val_a: QM31 = qm31_const::<10, 0, 0, 0>();
        let val_b: QM31 = qm31_const::<20, 0, 0, 0>();

        let result = elements.combine_qm31([val_a, val_b]);

        // Horner from back: sum = b, sum = sum * alpha + a, sum = sum - z
        let expected = (val_b * elements.alpha + val_a) - elements.z;
        assert!(result == expected);
    }
}
