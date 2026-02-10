/// Claims and proof structures for ML inference verification.
use stwo_verifier_core::channel::{Channel, ChannelTrait};
use stwo_verifier_core::fields::qm31::{QM31, QM31Serde};

/// The complete ML inference proof.
#[derive(Drop, Serde)]
pub struct MLProof {
    /// Model and inference metadata.
    pub claim: MLClaim,
    /// Per-matmul sumcheck proofs (Poseidon Fiat-Shamir).
    pub matmul_proofs: Array<MatMulSumcheckProofOnChain>,
    /// Optional channel salt for rerandomization.
    pub channel_salt: Option<u64>,
}

/// Claim about the ML inference computation.
#[derive(Drop, Serde, Copy)]
pub struct MLClaim {
    /// Unique model identifier (Poseidon hash of architecture + weights).
    pub model_id: felt252,
    /// Number of layers in the model.
    pub num_layers: u32,
    /// Activation function type: 0=ReLU, 1=GELU, 2=Sigmoid.
    pub activation_type: u8,
    /// Poseidon commitment over (input || output) tensors.
    pub io_commitment: felt252,
    /// Poseidon Merkle root of all weight matrices.
    pub weight_commitment: felt252,
}

#[generate_trait]
pub impl MLClaimMixImpl of MLClaimMixTrait {
    fn mix_into(self: @MLClaim, ref channel: Channel) {
        channel.mix_u64((*self.num_layers).into());
        channel.mix_u64((*self.activation_type).into());
    }
}

/// ML interaction claim (logup sums for activation lookups).
#[derive(Drop, Serde, Copy)]
pub struct MLInteractionClaim {
    /// Claimed LogUp sum for all activation layers combined.
    pub activation_claimed_sum: QM31,
}

/// A single round polynomial from the sumcheck protocol.
///
/// Represents p(x) = c0 + c1*x + c2*x^2 where:
///   p(0) + p(1) = current_sum (verified by verifier)
///   p(r) = next_sum (r drawn from channel)
#[derive(Drop, Serde, Copy)]
pub struct RoundPoly {
    pub c0: QM31,
    pub c1: QM31,
    pub c2: QM31,
}

/// On-chain matmul sumcheck proof (Poseidon Fiat-Shamir).
#[derive(Drop, Serde)]
pub struct MatMulSumcheckProofOnChain {
    /// Matrix dimensions.
    pub m: u32,
    pub k: u32,
    pub n: u32,
    /// Number of sumcheck rounds (log2(k_padded)).
    pub num_rounds: u32,
    /// Claimed sum: MLE_C(r_i, r_j).
    pub claimed_sum: QM31,
    /// Round polynomials: one per sumcheck round.
    pub round_polys: Array<RoundPoly>,
    /// Final evaluation of MLE_A at the sumcheck point.
    pub final_a_eval: QM31,
    /// Final evaluation of MLE_B at the sumcheck point.
    pub final_b_eval: QM31,
    /// Poseidon commitment to MLE_A (restricted to r_i).
    pub a_commitment: felt252,
    /// Poseidon commitment to MLE_B (restricted to r_j).
    pub b_commitment: felt252,
}

/// Output of ML verification (returned by the executable).
#[derive(Drop, Serde)]
pub struct MLVerificationOutput {
    pub model_id: felt252,
    pub io_commitment: felt252,
    pub weight_commitment: felt252,
    pub num_layers: u32,
    pub num_matmuls: u32,
    pub verified: bool,
}
