/// ML Air trait implementation.
///
/// The MLAir struct holds all ML-specific components and implements the
/// `Air` trait from `verifier_core`, enabling integration with the generic
/// STARK verification pipeline.
///
/// For the initial version, the AIR covers activation LogUp constraints only.
/// Matmul sumcheck proofs are verified separately (not via STARK).

use core::num::traits::Zero;
use stwo_verifier_core::verifier::Air;
use stwo_verifier_core::circle::CirclePoint;
use stwo_verifier_core::fields::qm31::QM31;
use stwo_verifier_core::{ColumnSpan, TreeSpan};
use super::claim::MLClaim;
use super::components::activation::{
    ActivationClaim, ActivationInteractionClaim, ActivationComponent,
    ActivationLookupElements, evaluate_activation_constraints_at_point,
};

/// The ML Air — holds all components for constraint evaluation.
#[derive(Drop)]
pub struct MLAir {
    /// Model claim metadata.
    pub claim: MLClaim,
    /// Log2 of the composition polynomial degree bound.
    pub composition_log_degree_bound: u32,
    /// Activation components for LogUp constraint evaluation (one per layer).
    pub activation_components: Array<ActivationComponent>,
}

/// Create a new MLAir instance.
#[generate_trait]
pub impl MLAirNewImpl of MLAirNewTrait {
    fn new(
        claim: @MLClaim,
        activation_claims: Span<ActivationClaim>,
        activation_interaction_claims: Span<ActivationInteractionClaim>,
        lookup_elements: @ActivationLookupElements,
        composition_log_degree_bound: u32,
    ) -> MLAir {
        let mut activation_components: Array<ActivationComponent> = array![];

        let mut i: u32 = 0;
        while i < activation_claims.len() {
            activation_components
                .append(
                    ActivationComponent {
                        claim: *activation_claims.at(i),
                        interaction_claim: *activation_interaction_claims.at(i),
                        lookup_elements: lookup_elements.clone(),
                    },
                );
            i += 1;
        };

        MLAir {
            claim: *claim,
            composition_log_degree_bound,
            activation_components,
        }
    }
}

/// Air trait implementation for MLAir.
///
/// The composition polynomial evaluation at an OOD point aggregates all
/// activation LogUp constraint evaluations weighted by powers of the random
/// coefficient. This follows the exact pattern from CairoAir.
pub impl MLAirAirImpl of Air<MLAir> {
    fn composition_log_degree_bound(self: @MLAir) -> u32 {
        *self.composition_log_degree_bound
    }

    fn eval_composition_polynomial_at_point(
        self: @MLAir,
        point: CirclePoint<QM31>,
        mask_values: TreeSpan<ColumnSpan<Span<QM31>>>,
        random_coeff: QM31,
    ) -> QM31 {
        let mut sum: QM31 = Zero::zero();

        // Destructure mask_values: [preprocessed, trace, interaction_trace, composition]
        let [
            _preprocessed_mask_values,
            mut trace_mask_values,
            mut interaction_trace_mask_values,
            _composition_trace_mask_values,
        ]: [ColumnSpan<Span<QM31>>; 4] =
            (*mask_values.try_into().unwrap()).unbox();

        // Evaluate activation constraints for each layer.
        // Each call pops its columns from trace_mask_values and
        // interaction_trace_mask_values.
        let mut idx: u32 = 0;
        while idx < self.activation_components.len() {
            evaluate_activation_constraints_at_point(
                self.activation_components.at(idx),
                ref sum,
                ref trace_mask_values,
                ref interaction_trace_mask_values,
                random_coeff,
                point,
            );
            idx += 1;
        };

        sum
    }
}

#[cfg(test)]
mod tests {
    use stwo_verifier_core::channel::Channel;
    use stwo_verifier_core::fields::qm31::qm31_const;
    use stwo_constraint_framework::{LookupElements, LookupElementsTrait};
    use super::{MLAirNewImpl, MLAirAirImpl, Air};
    use super::super::claim::MLClaim;
    use super::super::components::activation::{ActivationClaim, ActivationInteractionClaim};

    #[test]
    fn test_ml_air_construction_empty() {
        // MLAir with no activation components should construct fine.
        let claim = MLClaim {
            model_id: 0x1234,
            num_layers: 2,
            activation_type: 0,
            io_commitment: 0xabc,
            weight_commitment: 0xdef,
        };
        let mut channel: Channel = Default::default();
        let lookup_elements: LookupElements<2> = LookupElementsTrait::draw(ref channel);

        let air = MLAirNewImpl::new(
            @claim,
            array![].span(),
            array![].span(),
            @lookup_elements,
            10,
        );

        assert!(air.activation_components.len() == 0);
        assert!(air.composition_log_degree_bound == 10);
    }

    #[test]
    fn test_ml_air_composition_log_degree_bound() {
        let claim = MLClaim {
            model_id: 0x1234,
            num_layers: 1,
            activation_type: 0,
            io_commitment: 0,
            weight_commitment: 0,
        };
        let mut channel: Channel = Default::default();
        let lookup_elements: LookupElements<2> = LookupElementsTrait::draw(ref channel);

        let air = MLAirNewImpl::new(
            @claim,
            array![].span(),
            array![].span(),
            @lookup_elements,
            15,
        );

        assert!(Air::composition_log_degree_bound(@air) == 15);
    }

    #[test]
    fn test_ml_air_construction_with_components() {
        let claim = MLClaim {
            model_id: 0x1234,
            num_layers: 2,
            activation_type: 0,
            io_commitment: 0,
            weight_commitment: 0,
        };
        let activation_claims = array![
            ActivationClaim { layer_index: 0, log_size: 8, activation_type: 0 },
            ActivationClaim { layer_index: 1, log_size: 8, activation_type: 0 },
        ];
        let activation_interaction_claims = array![
            ActivationInteractionClaim { claimed_sum: qm31_const::<0, 0, 0, 0>() },
            ActivationInteractionClaim { claimed_sum: qm31_const::<0, 0, 0, 0>() },
        ];
        let mut channel: Channel = Default::default();
        let lookup_elements: LookupElements<2> = LookupElementsTrait::draw(ref channel);

        let air = MLAirNewImpl::new(
            @claim,
            activation_claims.span(),
            activation_interaction_claims.span(),
            @lookup_elements,
            12,
        );

        assert!(air.activation_components.len() == 2);
    }
}
