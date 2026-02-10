/// ML Air trait implementation.
///
/// The MLAir struct holds all ML-specific components and implements the
/// `Air` trait from `verifier_core`, enabling integration with the generic
/// STARK verification pipeline.
///
/// For the initial version, the AIR covers activation LogUp constraints only.
/// Matmul sumcheck proofs are verified separately (not via STARK).

use super::claim::{MLClaim, MLInteractionClaim};

/// The ML Air — holds all components for constraint evaluation.
#[derive(Drop)]
pub struct MLAir {
    /// Model claim metadata.
    pub claim: MLClaim,
    /// Log2 of the composition polynomial degree bound.
    pub composition_log_degree_bound: u32,
}

/// Create a new MLAir instance.
#[generate_trait]
pub impl MLAirNewImpl of MLAirNewTrait {
    fn new(
        claim: @MLClaim,
        _interaction_claim: @MLInteractionClaim,
        composition_log_degree_bound: u32,
    ) -> MLAir {
        MLAir {
            claim: *claim,
            composition_log_degree_bound,
        }
    }
}

/// Air trait implementation for MLAir.
///
/// The composition polynomial evaluation at an OOD point aggregates all
/// constraint evaluations weighted by powers of the random coefficient.
///
/// Currently a placeholder — the real constraint evaluation will be added
/// when activation LogUp components are wired in.
#[generate_trait]
pub impl MLAirImpl of MLAirTrait {
    fn composition_log_degree_bound(self: @MLAir) -> u32 {
        *self.composition_log_degree_bound
    }
}
