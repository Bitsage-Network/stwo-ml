//! Poseidon2-M31 Merkle tree for the VM31 privacy pool.
//!
//! Append-only binary Merkle tree using Poseidon2-M31 compression.
//! Leaves: 8 M31 elements (note commitments).
//! Internal nodes: Poseidon2 compress(left, right) → 8 M31 elements.

use std::fmt;

use stwo::core::fields::m31::BaseField as M31;

use super::poseidon2_m31::{poseidon2_compress, RATE};

/// Errors returned by Merkle tree operations.
#[derive(Debug, Clone)]
pub enum MerkleError {
    /// Leaf index is out of range (not yet inserted).
    IndexOutOfRange { index: usize, size: usize },
}

impl fmt::Display for MerkleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MerkleError::IndexOutOfRange { index, size } => {
                write!(
                    f,
                    "Merkle index {index} out of range (tree has {size} leaves)"
                )
            }
        }
    }
}

impl std::error::Error for MerkleError {}

/// Digest type: 8 M31 elements.
pub type Digest = [M31; RATE];

/// Zero digest (empty leaf).
const fn zero_digest() -> Digest {
    [M31::from_u32_unchecked(0); RATE]
}

/// Merkle authentication path: siblings from leaf to root.
#[derive(Clone, Debug)]
pub struct MerklePath {
    /// Sibling digests, from leaf level to root.
    pub siblings: Vec<Digest>,
    /// Leaf index in the tree.
    pub index: usize,
}

/// Append-only Poseidon2-M31 Merkle tree.
///
/// Fixed depth (default 20). Leaves are appended left-to-right.
/// Unused leaves are zero.
pub struct PoseidonMerkleTreeM31 {
    depth: usize,
    /// Number of leaves inserted so far.
    next_index: usize,
    /// Layers: layers[0] = leaves, layers[depth] = [root].
    layers: Vec<Vec<Digest>>,
    /// Precomputed zero hashes for each level (hash of two zero children).
    zero_hashes: Vec<Digest>,
}

impl PoseidonMerkleTreeM31 {
    /// Create an empty tree with the given depth.
    pub fn new(depth: usize) -> Self {
        // Precompute zero hashes for each level
        let mut zero_hashes = vec![zero_digest(); depth + 1];
        for d in 1..=depth {
            zero_hashes[d] = poseidon2_compress(&zero_hashes[d - 1], &zero_hashes[d - 1]);
        }

        // Initialize layers
        let capacity = 1usize << depth;
        let mut layers = Vec::with_capacity(depth + 1);
        layers.push(vec![zero_digest(); capacity]); // leaf layer
        for level in 1..=depth {
            let size = capacity >> level;
            layers.push(vec![zero_hashes[level]; size]);
        }

        Self {
            depth,
            next_index: 0,
            layers,
            zero_hashes,
        }
    }

    /// Append a leaf to the next available position. Returns the leaf index.
    pub fn append(&mut self, leaf: Digest) -> usize {
        let idx = self.next_index;
        assert!(
            idx < (1 << self.depth),
            "Merkle tree full (capacity {})",
            1 << self.depth
        );

        // Set leaf
        self.layers[0][idx] = leaf;

        // Update path from leaf to root
        let mut current_idx = idx;
        for level in 0..self.depth {
            let parent_idx = current_idx / 2;
            let left_idx = parent_idx * 2;
            let right_idx = left_idx + 1;

            let left = self.layers[level][left_idx];
            let right = if right_idx < self.layers[level].len() {
                self.layers[level][right_idx]
            } else {
                self.zero_hashes[level]
            };
            self.layers[level + 1][parent_idx] = poseidon2_compress(&left, &right);

            current_idx = parent_idx;
        }

        self.next_index = idx + 1;
        idx
    }

    /// Get the current root.
    pub fn root(&self) -> Digest {
        self.layers[self.depth][0]
    }

    /// Generate a Merkle proof for the leaf at the given index.
    ///
    /// Returns `Err(MerkleError::IndexOutOfRange)` if the index has not been inserted.
    pub fn prove(&self, index: usize) -> Result<MerklePath, MerkleError> {
        if index >= self.next_index {
            return Err(MerkleError::IndexOutOfRange {
                index,
                size: self.next_index,
            });
        }

        let mut siblings = Vec::with_capacity(self.depth);
        let mut current_idx = index;

        for level in 0..self.depth {
            let sibling_idx = current_idx ^ 1;
            let sibling = if sibling_idx < self.layers[level].len() {
                self.layers[level][sibling_idx]
            } else {
                self.zero_hashes[level]
            };
            siblings.push(sibling);
            current_idx /= 2;
        }

        Ok(MerklePath { siblings, index })
    }

    /// Number of leaves inserted.
    pub fn size(&self) -> usize {
        self.next_index
    }

    /// Tree depth.
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// Verify a Merkle proof against a root.
///
/// `expected_depth` is the tree depth; the proof must have exactly that many siblings.
/// Returns `false` if the path length doesn't match or the root doesn't verify.
pub fn verify_merkle_proof(
    root: &Digest,
    leaf: &Digest,
    path: &MerklePath,
    expected_depth: usize,
) -> bool {
    if path.siblings.len() != expected_depth {
        return false;
    }

    let mut current = *leaf;
    let mut index = path.index;

    for sibling in &path.siblings {
        if index & 1 == 0 {
            // Current node is left child
            current = poseidon2_compress(&current, sibling);
        } else {
            // Current node is right child
            current = poseidon2_compress(sibling, &current);
        }
        index >>= 1;
    }

    current == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leaf(val: u32) -> Digest {
        let mut d = zero_digest();
        d[0] = M31::from_u32_unchecked(val);
        d
    }

    #[test]
    fn test_empty_tree() {
        let tree = PoseidonMerkleTreeM31::new(4);
        assert_eq!(tree.size(), 0);
        // Root should be the zero hash at depth 4
        let expected_root = tree.zero_hashes[4];
        assert_eq!(tree.root(), expected_root);
    }

    #[test]
    fn test_single_insert() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        let leaf = make_leaf(42);
        let idx = tree.append(leaf);
        assert_eq!(idx, 0);
        assert_eq!(tree.size(), 1);

        // Root should have changed
        let empty_root = PoseidonMerkleTreeM31::new(4).root();
        assert_ne!(tree.root(), empty_root);
    }

    #[test]
    fn test_proof_single_leaf() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        let leaf = make_leaf(42);
        tree.append(leaf);

        let proof = tree.prove(0).unwrap();
        assert!(verify_merkle_proof(&tree.root(), &leaf, &proof, 4));
    }

    #[test]
    fn test_proof_multiple_leaves() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        let leaves: Vec<Digest> = (1..=8).map(make_leaf).collect();

        for leaf in &leaves {
            tree.append(*leaf);
        }

        // Verify proof for each leaf
        let root = tree.root();
        for (i, leaf) in leaves.iter().enumerate() {
            let proof = tree.prove(i).unwrap();
            assert!(
                verify_merkle_proof(&root, leaf, &proof, 4),
                "Proof failed for leaf {i}"
            );
        }
    }

    #[test]
    fn test_proof_wrong_leaf_fails() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        tree.append(make_leaf(42));

        let proof = tree.prove(0).unwrap();
        let wrong_leaf = make_leaf(99);
        assert!(
            !verify_merkle_proof(&tree.root(), &wrong_leaf, &proof, 4),
            "Wrong leaf should fail verification"
        );
    }

    #[test]
    fn test_proof_wrong_root_fails() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        tree.append(make_leaf(42));

        let proof = tree.prove(0).unwrap();
        let wrong_root = make_leaf(99);
        assert!(
            !verify_merkle_proof(&wrong_root, &make_leaf(42), &proof, 4),
            "Wrong root should fail verification"
        );
    }

    #[test]
    fn test_root_changes_with_inserts() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        let r0 = tree.root();
        tree.append(make_leaf(1));
        let r1 = tree.root();
        tree.append(make_leaf(2));
        let r2 = tree.root();

        assert_ne!(r0, r1);
        assert_ne!(r1, r2);
        assert_ne!(r0, r2);
    }

    #[test]
    fn test_deterministic_root() {
        let mut t1 = PoseidonMerkleTreeM31::new(4);
        let mut t2 = PoseidonMerkleTreeM31::new(4);

        for i in 1..=5 {
            t1.append(make_leaf(i));
            t2.append(make_leaf(i));
        }

        assert_eq!(t1.root(), t2.root());
    }

    #[test]
    fn test_proof_depth() {
        let depth = 6;
        let mut tree = PoseidonMerkleTreeM31::new(depth);
        tree.append(make_leaf(1));

        let proof = tree.prove(0).unwrap();
        assert_eq!(proof.siblings.len(), depth);
    }

    #[test]
    fn test_larger_tree() {
        let mut tree = PoseidonMerkleTreeM31::new(8); // 256 leaves
        let n = 50;

        for i in 0..n {
            tree.append(make_leaf(i + 1));
        }

        assert_eq!(tree.size(), n as usize);

        // Verify all proofs
        let root = tree.root();
        for i in 0..n {
            let proof = tree.prove(i as usize).unwrap();
            assert!(verify_merkle_proof(&root, &make_leaf(i + 1), &proof, 8));
        }
    }

    #[test]
    fn test_order_sensitive() {
        // Inserting [1, 2] vs [2, 1] should produce different roots
        let mut t1 = PoseidonMerkleTreeM31::new(4);
        let mut t2 = PoseidonMerkleTreeM31::new(4);

        t1.append(make_leaf(1));
        t1.append(make_leaf(2));

        t2.append(make_leaf(2));
        t2.append(make_leaf(1));

        assert_ne!(t1.root(), t2.root());
    }

    #[test]
    fn test_depth_20_creation() {
        // Ensure depth-20 tree can be created (the target for VM31)
        // This allocates ~8 MB for zero_hashes precomputation
        let tree = PoseidonMerkleTreeM31::new(20);
        assert_eq!(tree.depth(), 20);
        assert_eq!(tree.size(), 0);
    }

    #[test]
    fn test_historical_proofs_still_valid() {
        // Proofs generated against older roots remain valid for the same root
        let mut tree = PoseidonMerkleTreeM31::new(4);
        tree.append(make_leaf(1));
        tree.append(make_leaf(2));

        let root_after_2 = tree.root();
        let proof_1 = tree.prove(0).unwrap();

        // Add more leaves
        tree.append(make_leaf(3));
        tree.append(make_leaf(4));

        // Proof against old root should still be valid
        assert!(verify_merkle_proof(
            &root_after_2,
            &make_leaf(1),
            &proof_1,
            4,
        ));

        // But proof against new root should fail (path changed)
        assert!(!verify_merkle_proof(
            &tree.root(),
            &make_leaf(1),
            &proof_1,
            4
        ));

        // Fresh proof against new root should work
        let fresh_proof = tree.prove(0).unwrap();
        assert!(verify_merkle_proof(
            &tree.root(),
            &make_leaf(1),
            &fresh_proof,
            4,
        ));
    }

    #[test]
    fn test_prove_out_of_range_returns_error() {
        let tree = PoseidonMerkleTreeM31::new(4);
        assert!(tree.prove(0).is_err());

        let mut tree2 = PoseidonMerkleTreeM31::new(4);
        tree2.append(make_leaf(1));
        assert!(tree2.prove(1).is_err());
        assert!(tree2.prove(0).is_ok());
    }

    #[test]
    fn test_verify_wrong_depth_fails() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        tree.append(make_leaf(42));
        let proof = tree.prove(0).unwrap();

        // Correct depth passes
        assert!(verify_merkle_proof(&tree.root(), &make_leaf(42), &proof, 4));
        // Wrong depths fail
        assert!(!verify_merkle_proof(
            &tree.root(),
            &make_leaf(42),
            &proof,
            3
        ));
        assert!(!verify_merkle_proof(
            &tree.root(),
            &make_leaf(42),
            &proof,
            5
        ));
    }

    #[test]
    fn test_verify_empty_path_fails() {
        let mut tree = PoseidonMerkleTreeM31::new(4);
        let leaf = make_leaf(42);
        tree.append(leaf);
        let root = tree.root();

        let empty_path = MerklePath {
            siblings: Vec::new(),
            index: 0,
        };
        assert!(!verify_merkle_proof(&root, &leaf, &empty_path, 4));
    }
}
