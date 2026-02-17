//! Binary Merkle tree using Poseidon hash.
//!
//! Provides commitment and authentication path generation for
//! on-chain verification of MLE evaluations.

use rayon::prelude::*;
use starknet_crypto::poseidon_hash;
use starknet_ff::FieldElement;

#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::cuda_executor::{
    get_cuda_executor, is_cuda_available, upload_poseidon252_round_constants,
};

/// Minimum number of pairs per layer to trigger rayon parallelism.
/// Below this, sequential hashing is faster due to rayon overhead.
const PARALLEL_THRESHOLD: usize = 256;
/// Minimum leaf-pair count to attempt GPU Merkle hashing in full-tree mode.
#[cfg(feature = "cuda-runtime")]
const GPU_MERKLE_THRESHOLD_PAIRS: usize = 1 << 14; // 16K pairs

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

    /// Build a Poseidon Merkle tree with rayon-parallel hashing.
    ///
    /// Each layer's hash pairs are independent, so layers with >= PARALLEL_THRESHOLD
    /// pairs use `par_chunks(2)` for parallel Poseidon hashing.
    pub fn build_parallel(leaves: Vec<FieldElement>) -> Self {
        assert!(!leaves.is_empty(), "cannot build tree from empty leaves");

        let n = leaves.len().next_power_of_two().max(2);
        let mut padded = leaves;
        padded.resize(n, FieldElement::ZERO);

        #[cfg(feature = "cuda-runtime")]
        if let Some(gpu_layers) = try_build_parallel_gpu(&padded) {
            return Self { layers: gpu_layers };
        }

        let mut layers = vec![padded];

        while layers.last().unwrap().len() > 1 {
            let current = layers.last().unwrap();
            let n_pairs = current.len() / 2;

            let next = if n_pairs >= PARALLEL_THRESHOLD {
                current.par_chunks(2).map(|pair| poseidon_hash(pair[0], pair[1])).collect()
            } else {
                let mut v = Vec::with_capacity(n_pairs);
                for i in (0..current.len()).step_by(2) {
                    v.push(poseidon_hash(current[i], current[i + 1]));
                }
                v
            };
            layers.push(next);
        }

        Self { layers }
    }

    /// Compute only the Merkle root without storing intermediate layers.
    ///
    /// Uses rayon for large layers. Returns just the root FieldElement.
    /// More memory-efficient than `build()` when only the commitment is needed.
    pub fn root_only_parallel(leaves: Vec<FieldElement>) -> FieldElement {
        assert!(!leaves.is_empty(), "cannot compute root from empty leaves");

        let n = leaves.len().next_power_of_two().max(2);
        let mut current = leaves;
        current.resize(n, FieldElement::ZERO);

        while current.len() > 1 {
            let n_pairs = current.len() / 2;
            current = if n_pairs >= PARALLEL_THRESHOLD {
                current.par_chunks(2).map(|pair| poseidon_hash(pair[0], pair[1])).collect()
            } else {
                let mut v = Vec::with_capacity(n_pairs);
                for i in (0..current.len()).step_by(2) {
                    v.push(poseidon_hash(current[i], current[i + 1]));
                }
                v
            };
        }

        current[0]
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

#[cfg(feature = "cuda-runtime")]
fn gpu_merkle_enabled() -> bool {
    match std::env::var("STWO_GPU_MLE_MERKLE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        // Default ON when CUDA runtime is built in.
        Err(_) => true,
    }
}

#[cfg(feature = "cuda-runtime")]
fn field_element_to_u64_limbs(fe: &FieldElement) -> [u64; 4] {
    let bytes = fe.to_bytes_be();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let offset = 24 - i * 8;
        let mut val = 0u64;
        for j in 0..8 {
            val = (val << 8) | bytes[offset + j] as u64;
        }
        *limb = val;
    }
    limbs
}

#[cfg(feature = "cuda-runtime")]
fn u64_limbs_to_field_element(limbs: &[u64]) -> Option<FieldElement> {
    if limbs.len() != 4 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let limb = limbs[3 - i];
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    // Match STWO GPU Poseidon conversion: keep only low 251 bits.
    bytes[0] &= 0x07;
    FieldElement::from_bytes_be(&bytes).ok()
}

#[cfg(feature = "cuda-runtime")]
fn try_build_parallel_gpu(leaves: &[FieldElement]) -> Option<Vec<Vec<FieldElement>>> {
    if !gpu_merkle_enabled() || !is_cuda_available() {
        return None;
    }
    let n_leaf_hashes = leaves.len() / 2;
    if n_leaf_hashes < GPU_MERKLE_THRESHOLD_PAIRS {
        return None;
    }

    let executor = get_cuda_executor().ok()?;
    let d_round_constants = upload_poseidon252_round_constants(&executor.device).ok()?;

    let mut leaf_limbs = Vec::with_capacity(leaves.len() * 4);
    for fe in leaves {
        leaf_limbs.extend_from_slice(&field_element_to_u64_limbs(fe));
    }
    let d_prev_leaf = executor.device.htod_sync_copy(&leaf_limbs).ok()?;
    let d_dummy_columns = executor.device.htod_sync_copy(&[0u32]).ok()?;

    let raw_layers = executor
        .execute_poseidon252_merkle_full_tree(
            &d_dummy_columns,
            0,
            Some(&d_prev_leaf),
            n_leaf_hashes,
            &d_round_constants,
        )
        .ok()?;

    let mut layers = Vec::with_capacity(raw_layers.len() + 1);
    layers.push(leaves.to_vec());
    for raw_layer in raw_layers {
        if raw_layer.len() % 4 != 0 {
            return None;
        }
        let mut layer = Vec::with_capacity(raw_layer.len() / 4);
        for chunk in raw_layer.chunks_exact(4) {
            layer.push(u64_limbs_to_field_element(chunk)?);
        }
        layers.push(layer);
    }
    Some(layers)
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

    #[test]
    fn test_build_parallel_matches_sequential() {
        let leaves: Vec<FieldElement> = (0..512).map(|i| FieldElement::from(i as u64)).collect();
        let seq = PoseidonMerkleTree::build(leaves.clone());
        let par = PoseidonMerkleTree::build_parallel(leaves);
        assert_eq!(seq.root(), par.root());

        // Verify proofs from parallel tree
        for i in [0, 1, 255, 511] {
            let path = par.prove(i);
            assert!(PoseidonMerkleTree::verify(par.root(), i, FieldElement::from(i as u64), &path));
        }
    }

    #[test]
    fn test_root_only_matches_build() {
        for n in [2, 4, 8, 16, 64, 512] {
            let leaves: Vec<FieldElement> = (0..n).map(|i| FieldElement::from(i as u64)).collect();
            let tree = PoseidonMerkleTree::build(leaves.clone());
            let root = PoseidonMerkleTree::root_only_parallel(leaves);
            assert_eq!(tree.root(), root, "mismatch for n={n}");
        }
    }
}
