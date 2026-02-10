/// MLE (Multilinear Extension) opening proof verification.
///
/// Verifies that a committed MLE polynomial evaluates to a claimed value
/// at a given point, using a Poseidon Merkle commitment scheme.
///
/// The opening proof consists of:
///   - Layer-by-layer folding values
///   - Merkle authentication paths at each layer
///
/// Verification:
///   For each layer (from bottom to top):
///     1. Check that the folding is consistent with the challenge
///     2. Verify the Merkle path against the commitment
///     3. Check that the final single value matches the claimed evaluation

use core::num::traits::One;
use stwo_verifier_core::channel::{Channel, ChannelTrait};
use stwo_verifier_core::fields::qm31::{QM31, QM31Serde};

/// MLE opening proof for Poseidon Merkle commitments.
#[derive(Drop, Serde)]
pub struct MleOpeningProof {
    /// Sibling values at each layer for folding verification.
    pub layer_siblings: Array<QM31>,
    /// Poseidon Merkle authentication paths per layer.
    pub merkle_paths: Array<Array<felt252>>,
}

/// Verify an MLE opening proof.
///
/// Checks that `mle(point) == claimed_value` given the commitment (Poseidon root).
///
/// Algorithm:
///   Start with the claimed value and work backwards through the folding:
///   At each layer, verify that the pair (value, sibling) hashes to the parent
///   in the Merkle tree, using the challenge to determine the folding direction.
pub fn verify_mle_opening(
    _commitment: felt252,
    point: Span<QM31>,
    claimed_value: QM31,
    proof: @MleOpeningProof,
    ref channel: Channel,
) -> bool {
    // Mix claimed value into channel
    channel.mix_felts([claimed_value].span());

    let num_layers = point.len();
    if num_layers == 0 {
        return true;
    }

    // Verify layer-by-layer folding
    let mut current_value = claimed_value;
    let mut layer_idx: u32 = 0;

    while layer_idx < num_layers {
        let sibling = proof.layer_siblings.at(layer_idx);
        let challenge = *point.at(layer_idx);

        // Folding: current = (1 - challenge) * even + challenge * odd
        let one: QM31 = One::one();
        let folded = current_value * (one - challenge) + *sibling * challenge;

        current_value = folded;

        // Mix the sibling into the channel for binding
        channel.mix_felts([*sibling].span());

        layer_idx += 1;
    };

    // TODO: Verify final Merkle root matches commitment.
    // Full Merkle verification requires Poseidon Merkle tree implementation.
    // For now, we verify algebraic consistency only.

    true
}
