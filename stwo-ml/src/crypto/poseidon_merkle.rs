//! Binary Merkle tree using Poseidon hash.
//!
//! Provides commitment and authentication path generation for
//! on-chain verification of MLE evaluations.

use starknet_crypto::poseidon_hash;
use starknet_ff::FieldElement;

/// A binary Merkle tree with Poseidon hash.
///
/// Leaves are `FieldElement` values. Internal nodes are computed as
/// `poseidon_hash(left_child, right_child)`.
#[derive(Debug, Clone)]
pub struct PoseidonMerkleTree {
    /// Tree layers from bottom (leaves) to top (root).
    /// `layers[0]` = leaves (padded to power of 2).
    /// `layers[last]` = `[root]`.
    layers: Vec<Vec<FieldElement>>,
}

/// Authentication path for a Merkle proof (bottom-up sibling hashes).
#[derive(Debug, Clone)]
pub struct MerkleAuthPath {
    pub siblings: Vec<FieldElement>,
}

impl PoseidonMerkleTree {
    /// Build a Poseidon Merkle tree from a list of leaves.
    ///
    /// Pads to the next power of 2 with `FieldElement::ZERO`.
    pub fn build(leaves: Vec<FieldElement>) -> Self {
        assert!(!leaves.is_empty(), "cannot build tree from empty leaves");

        // Pad to next power of 2 (minimum 2 leaves)
        let n = leaves.len().next_power_of_two().max(2);
        let mut padded = leaves;
        padded.resize(n, FieldElement::ZERO);

        let mut layers = vec![padded];

        // Build layers bottom-up
        while layers.last().unwrap().len() > 1 {
            let current = layers.last().unwrap();
            let mut next = Vec::with_capacity(current.len() / 2);
            for i in (0..current.len()).step_by(2) {
                next.push(poseidon_hash(current[i], current[i + 1]));
            }
            layers.push(next);
        }

        Self { layers }
    }

    /// Returns the Merkle root.
    pub fn root(&self) -> FieldElement {
        self.layers.last().unwrap()[0]
    }

    /// Returns the number of leaves (including padding).
    pub fn num_leaves(&self) -> usize {
        self.layers[0].len()
    }

    /// Generate an authentication path for the leaf at `index`.
    pub fn prove(&self, index: usize) -> MerkleAuthPath {
        assert!(index < self.layers[0].len(), "index out of bounds");

        let mut siblings = Vec::with_capacity(self.layers.len() - 1);
        let mut idx = index;

        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling_idx = idx ^ 1; // flip last bit
            siblings.push(layer[sibling_idx]);
            idx >>= 1;
        }

        MerkleAuthPath { siblings }
    }

    /// Verify that a leaf value at `index` is consistent with the root.
    pub fn verify(
        root: FieldElement,
        index: usize,
        value: FieldElement,
        path: &MerkleAuthPath,
    ) -> bool {
        let mut current = value;
        let mut idx = index;

        for sibling in &path.siblings {
            current = if idx & 1 == 0 {
                poseidon_hash(current, *sibling)
            } else {
                poseidon_hash(*sibling, current)
            };
            idx >>= 1;
        }

        current == root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_4_leaves() {
        let leaves = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let tree = PoseidonMerkleTree::build(leaves.clone());
        let root = tree.root();

        // Manually compute expected root
        let h01 = poseidon_hash(leaves[0], leaves[1]);
        let h23 = poseidon_hash(leaves[2], leaves[3]);
        let expected_root = poseidon_hash(h01, h23);
        assert_eq!(root, expected_root);

        // Verify all leaf proofs
        for i in 0..4 {
            let path = tree.prove(i);
            assert!(
                PoseidonMerkleTree::verify(root, i, leaves[i], &path),
                "proof for leaf {} should verify",
                i
            );
        }
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        let leaves = vec![FieldElement::from(42u64)];
        let tree = PoseidonMerkleTree::build(leaves);

        // Single leaf padded to 2: [42, 0]
        assert_eq!(tree.num_leaves(), 2);
        let root = tree.root();

        let path = tree.prove(0);
        assert_eq!(path.siblings.len(), 1);
        assert!(PoseidonMerkleTree::verify(root, 0, FieldElement::from(42u64), &path));
    }

    #[test]
    fn test_merkle_tree_tampered_fails() {
        let leaves = vec![
            FieldElement::from(10u64),
            FieldElement::from(20u64),
            FieldElement::from(30u64),
            FieldElement::from(40u64),
        ];

        let tree = PoseidonMerkleTree::build(leaves);
        let root = tree.root();
        let path = tree.prove(0);

        // Wrong value should fail verification
        let wrong_value = FieldElement::from(999u64);
        assert!(
            !PoseidonMerkleTree::verify(root, 0, wrong_value, &path),
            "tampered value should not verify"
        );

        // Wrong index should also fail
        assert!(
            !PoseidonMerkleTree::verify(root, 1, FieldElement::from(10u64), &path),
            "wrong index should not verify"
        );
    }
}
