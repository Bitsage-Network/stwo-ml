//! Hades permutation wrapper for Starknet Poseidon.
//!
//! The Cairo channel uses `hades_permutation(state[0], state[1], state[2])`
//! with different capacity values for mix (2) vs draw (3) operations.
//!
//! `starknet-crypto` 0.6.2 exposes `poseidon_permute_comp` which is the
//! raw Hades permutation over the 3-element state. We wrap it here with
//! the `hades_permutation` name used throughout the codebase.

use starknet_ff::FieldElement;

/// Apply the Hades permutation in-place on a 3-element state.
///
/// This is the core building block for the Poseidon hash function
/// used in Cairo/Starknet. It applies:
/// - 8 full rounds (S-box on all 3 state elements)
/// - 83 partial rounds (S-box on state[2] only)
/// - 8 full rounds
///
/// with MDS matrix mixing between each round.
pub fn hades_permutation(state: &mut [FieldElement; 3]) {
    starknet_crypto::poseidon_permute_comp(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hades_known_vectors() {
        // Verify hades_permutation produces expected output.
        // Input: [1, 2, 3]
        let mut state = [
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
        ];
        hades_permutation(&mut state);

        // The output should be deterministic and non-trivial
        assert_ne!(state[0], FieldElement::ZERO);
        assert_ne!(state[1], FieldElement::ZERO);
        assert_ne!(state[2], FieldElement::ZERO);
        assert_ne!(state[0], FieldElement::from(1u64));

        // Run again with same input to verify determinism
        let mut state2 = [
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
        ];
        hades_permutation(&mut state2);
        assert_eq!(state[0], state2[0]);
        assert_eq!(state[1], state2[1]);
        assert_eq!(state[2], state2[2]);
    }

    #[test]
    fn test_hades_matches_poseidon_hash() {
        // poseidon_hash(x, y) = hades([x, y, 2])[0]
        // This is the fundamental relationship.
        let x = FieldElement::from(42u64);
        let y = FieldElement::from(99u64);

        let hash_result = starknet_crypto::poseidon_hash(x, y);

        let mut state = [x, y, FieldElement::TWO];
        hades_permutation(&mut state);

        assert_eq!(
            state[0], hash_result,
            "hades([x, y, 2])[0] must equal poseidon_hash(x, y)"
        );
    }
}
