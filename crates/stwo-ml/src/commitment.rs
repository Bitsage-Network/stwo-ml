//! Poseidon Merkle commitment and multilinear folding for MLE opening proofs.
//!
//! Provides cryptographic commitments to matrix entries and opening proofs
//! that verify MLE evaluations against committed Merkle roots.
//!
//! # Protocol
//!
//! ```text
//! Commit:  Matrix entries → pad to 2^n → Poseidon Merkle tree → root R₀
//!
//! Open MLE(r) = v:
//!   1. Fold entries with each r_i:  L_{i+1}[j] = (1-r_i)·L_i[2j] + r_i·L_i[2j+1]
//!   2. Commit each intermediate layer → roots R₁, ..., R_{n-1}
//!   3. Mix roots into Fiat-Shamir transcript
//!   4. Draw query indices
//!   5. For each query: provide Merkle proofs at each layer
//!
//! Verify:
//!   For each query, at each layer:
//!     ✓ Merkle proof authenticates values against Rᵢ
//!     ✓ Folding relation: L_{i+1}[j] = (1-rᵢ)·L_i[2j] + rᵢ·L_i[2j+1]
//!   Final value = claimed MLE(r)
//! ```

use rayon::prelude::*;
use starknet_crypto::{poseidon_hash, poseidon_hash_many};
use starknet_ff::FieldElement as FieldElement252;

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;

use crate::backend::{batch_merkle_layer, batch_poseidon_hash_with_domain, PAR_THRESHOLD};
use crate::components::matmul::M31Matrix;

// ============================================================================
// Constants
// ============================================================================

/// Number of spot-check queries for the folding protocol.
/// Each query provides ~1 bit of security per layer.
/// 20 queries × multiple layers gives good overall soundness.
pub const DEFAULT_NUM_QUERIES: usize = 20;

/// Domain separator for leaf hashing (distinguishes leaves from internal nodes).
const LEAF_DOMAIN: FieldElement252 = FieldElement252::ZERO;

// ============================================================================
// Field element conversions
// ============================================================================

/// Convert M31 to felt252.
fn m31_to_felt(v: M31) -> FieldElement252 {
    FieldElement252::from(v.0 as u64)
}

/// Pack QM31 (4 M31 components) into a single felt252.
/// Uses big-endian 2^31 packing, starting from ONE (matching STWO's channel packing).
fn qm31_to_felt(v: SecureField) -> FieldElement252 {
    let shift = FieldElement252::from(1u64 << 31);
    let [m0, m1, m2, m3] = v.to_m31_array();
    let mut result = FieldElement252::ONE;
    result = result * shift + FieldElement252::from(m0.0 as u64);
    result = result * shift + FieldElement252::from(m1.0 as u64);
    result = result * shift + FieldElement252::from(m2.0 as u64);
    result = result * shift + FieldElement252::from(m3.0 as u64);
    result
}

/// Hash an M31 value as a Merkle leaf.
fn hash_m31_leaf(v: M31) -> FieldElement252 {
    poseidon_hash(m31_to_felt(v), LEAF_DOMAIN)
}

/// Hash a QM31 value as a Merkle leaf.
fn hash_qm31_leaf(v: SecureField) -> FieldElement252 {
    poseidon_hash(qm31_to_felt(v), LEAF_DOMAIN)
}

/// Hash two children to form an internal node.
fn hash_node(left: FieldElement252, right: FieldElement252) -> FieldElement252 {
    poseidon_hash(left, right)
}

// ============================================================================
// Poseidon Merkle Tree
// ============================================================================

/// Poseidon hash-based Merkle tree.
///
/// Layers are stored bottom-up: `layers[0]` = leaf hashes, `layers[depth]` = [root].
#[derive(Debug, Clone)]
pub struct PoseidonMerkleTree {
    layers: Vec<Vec<FieldElement252>>,
}

impl PoseidonMerkleTree {
    /// Build a Merkle tree from pre-hashed leaf values.
    pub fn from_leaf_hashes(leaf_hashes: Vec<FieldElement252>) -> Self {
        assert!(
            leaf_hashes.len().is_power_of_two(),
            "leaf count must be power of two, got {}",
            leaf_hashes.len()
        );

        let depth = leaf_hashes.len().ilog2() as usize;
        let mut layers = Vec::with_capacity(depth + 1);
        layers.push(leaf_hashes);

        for i in 0..depth {
            let prev = &layers[i];
            let next = batch_merkle_layer(prev);
            layers.push(next);
        }

        Self { layers }
    }

    /// Build from M31 values (hashes each value as a leaf).
    pub fn from_m31_values(values: &[M31]) -> Self {
        let felt_values: Vec<FieldElement252> = if values.len() >= PAR_THRESHOLD {
            values.par_iter().map(|v| m31_to_felt(*v)).collect()
        } else {
            values.iter().map(|v| m31_to_felt(*v)).collect()
        };
        let leaf_hashes = batch_poseidon_hash_with_domain(&felt_values, LEAF_DOMAIN);
        Self::from_leaf_hashes(leaf_hashes)
    }

    /// Build from QM31 values (hashes each value as a leaf).
    pub fn from_qm31_values(values: &[SecureField]) -> Self {
        let felt_values: Vec<FieldElement252> = if values.len() >= PAR_THRESHOLD {
            values.par_iter().map(|v| qm31_to_felt(*v)).collect()
        } else {
            values.iter().map(|v| qm31_to_felt(*v)).collect()
        };
        let leaf_hashes = batch_poseidon_hash_with_domain(&felt_values, LEAF_DOMAIN);
        Self::from_leaf_hashes(leaf_hashes)
    }

    /// Get the Merkle root.
    pub fn root(&self) -> FieldElement252 {
        self.layers.last().unwrap()[0]
    }

    /// Number of leaves.
    pub fn num_leaves(&self) -> usize {
        self.layers[0].len()
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        self.layers.len() - 1
    }

    /// Generate an authentication path for the leaf at `index`.
    pub fn open(&self, index: usize) -> MerkleProof {
        assert!(index < self.num_leaves());
        let mut siblings = Vec::with_capacity(self.depth());
        let mut idx = index;

        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling_idx = idx ^ 1;
            siblings.push(layer[sibling_idx]);
            idx /= 2;
        }

        MerkleProof { siblings }
    }
}

/// Authentication path in a Poseidon Merkle tree.
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// Sibling hashes from leaf to root (bottom-up).
    pub siblings: Vec<FieldElement252>,
}

impl MerkleProof {
    /// Verify this proof: does `leaf_hash` at `index` produce `root`?
    pub fn verify(
        &self,
        leaf_hash: FieldElement252,
        index: usize,
        root: FieldElement252,
    ) -> bool {
        let mut current = leaf_hash;
        let mut idx = index;

        for sibling in &self.siblings {
            current = if idx.is_multiple_of(2) {
                hash_node(current, *sibling)
            } else {
                hash_node(*sibling, current)
            };
            idx /= 2;
        }

        current == root
    }
}

// ============================================================================
// Matrix Commitment
// ============================================================================

/// Commitment to a matrix: Poseidon Merkle root over padded entries.
#[derive(Debug, Clone)]
pub struct MatrixCommitment {
    /// The Merkle root (the commitment value stored on-chain).
    pub root: FieldElement252,
    /// Number of variables in the MLE (log2 of padded entry count).
    pub num_vars: usize,
    /// Original matrix dimensions.
    pub rows: usize,
    pub cols: usize,
}

/// Commit to a matrix's entries as a Poseidon Merkle tree.
///
/// Entries are stored row-major, padded with zeros to the next power of two.
/// Returns the commitment metadata and the full Merkle tree (needed for opening).
pub fn commit_matrix(matrix: &M31Matrix) -> (MatrixCommitment, PoseidonMerkleTree) {
    let total = matrix.rows * matrix.cols;
    let padded_size = total.next_power_of_two();
    let num_vars = padded_size.ilog2() as usize;

    // Pad entries to power of two
    let mut entries = matrix.data.clone();
    entries.resize(padded_size, M31::from(0));

    let tree = PoseidonMerkleTree::from_m31_values(&entries);
    let commitment = MatrixCommitment {
        root: tree.root(),
        num_vars,
        rows: matrix.rows,
        cols: matrix.cols,
    };

    (commitment, tree)
}

// ============================================================================
// Multilinear Folding Protocol
// ============================================================================

/// Fold a layer of SecureField values with a challenge.
///
/// Pairs consecutive elements: `L_{i+1}[j] = (1 - challenge) * L_i[2j] + challenge * L_i[2j+1]`
///
/// This fixes the **last** MLE variable (the LSB that distinguishes odd/even indices).
/// When opening MLE at `point = [x_0, ..., x_{n-1}]`, fold with variables in **reverse** order:
/// first fold with `x_{n-1}`, then `x_{n-2}`, ..., finally `x_0`.
fn fold_layer(layer: &[SecureField], challenge: SecureField) -> Vec<SecureField> {
    let pairs = layer.len() / 2;
    if pairs >= PAR_THRESHOLD {
        (0..pairs)
            .into_par_iter()
            .map(|j| {
                let lo = layer[2 * j];
                let hi = layer[2 * j + 1];
                challenge * (hi - lo) + lo
            })
            .collect()
    } else {
        (0..pairs)
            .map(|j| {
                let lo = layer[2 * j];
                let hi = layer[2 * j + 1];
                challenge * (hi - lo) + lo
            })
            .collect()
    }
}

/// Data for a single query at a single folding round.
#[derive(Debug, Clone)]
pub struct QueryRoundData {
    /// Value at the even position of the pair (L_i[2j]).
    pub left_value: SecureField,
    /// Value at the odd position of the pair (L_i[2j+1]).
    pub right_value: SecureField,
    /// Merkle proof for the left value against R_i.
    pub left_proof: MerkleProof,
    /// Merkle proof for the right value against R_i.
    pub right_proof: MerkleProof,
}

/// Complete data for a single query across all folding rounds.
#[derive(Debug, Clone)]
pub struct QueryProof {
    /// Initial query pair index in layer 0 (positions 2*index and 2*index+1).
    pub initial_pair_index: usize,
    /// Authentication data at each folding round.
    pub rounds: Vec<QueryRoundData>,
}

/// Opening proof for MLE(point) = claimed_eval using multilinear folding.
#[derive(Debug, Clone)]
pub struct MleOpeningProof {
    /// Merkle roots of intermediate folded layers (R_1, ..., R_{n-1}).
    /// R_0 is the original commitment (not included).
    pub intermediate_roots: Vec<FieldElement252>,
    /// Spot-check query proofs.
    pub queries: Vec<QueryProof>,
    /// The final value after all folds (should equal claimed evaluation).
    pub final_value: SecureField,
}

/// Generate an MLE opening proof via multilinear folding.
///
/// Given matrix entries committed to `tree` (with root `commitment.root`),
/// proves that MLE(point) = claimed_eval.
///
/// The `point` vector has `num_vars` elements — the full opening point
/// (e.g., `[row_challenges..., assignment...]` for matrix A).
pub fn open_mle(
    entries: &[M31],
    point: &[SecureField],
    tree: &PoseidonMerkleTree,
    num_queries: usize,
) -> MleOpeningProof {
    let n = point.len();
    assert_eq!(
        entries.len(),
        1 << n,
        "entries length must be 2^n (got {} for n={})",
        entries.len(),
        n
    );

    // Build all folded layers.
    // fold_layer pairs consecutive elements (fixes the LAST MLE variable),
    // but point[0] is the FIRST MLE variable (MSB, splits array in half).
    // Process variables in reverse so folding matches MLE evaluation convention.
    let reversed_point: Vec<SecureField> = point.iter().rev().copied().collect();

    let mut layers: Vec<Vec<SecureField>> = Vec::with_capacity(n + 1);

    // Layer 0: M31 entries lifted to SecureField
    let layer0: Vec<SecureField> = entries.iter().map(|v| SecureField::from(*v)).collect();
    layers.push(layer0);

    // Fold with each challenge (reversed: last MLE variable first)
    for i in 0..n {
        let folded = fold_layer(&layers[i], reversed_point[i]);
        layers.push(folded);
    }

    // Final value
    assert_eq!(layers[n].len(), 1);
    let final_value = layers[n][0];

    // Build Merkle trees for intermediate layers
    // Layer 0 tree is the input `tree` parameter
    let mut trees: Vec<PoseidonMerkleTree> = Vec::with_capacity(n);
    // trees[0] is a placeholder — we use the input tree for layer 0
    // trees[i] for i >= 1 corresponds to layers[i]
    let mut intermediate_roots: Vec<FieldElement252> = Vec::with_capacity(n - 1);

    for layer in layers.iter().take(n).skip(1) {
        let t = PoseidonMerkleTree::from_qm31_values(layer);
        intermediate_roots.push(t.root());
        trees.push(t);
    }

    // Derive query indices deterministically from the commitment data
    // Use a separate Fiat-Shamir: hash(root, intermediate_roots, point) → seed
    let mut seed_inputs = vec![tree.root()];
    seed_inputs.extend_from_slice(&intermediate_roots);
    for p in point {
        seed_inputs.push(qm31_to_felt(*p));
    }
    let seed = poseidon_hash_many(&seed_inputs);

    // Generate query indices
    let layer0_pairs = entries.len() / 2;
    let query_indices = derive_query_indices(seed, num_queries, layer0_pairs);

    // Generate query proofs (parallel — each query is independent)
    let queries: Vec<QueryProof> = query_indices
        .par_iter()
        .map(|&q| generate_query_proof(q, &layers, tree, &trees, n))
        .collect();

    MleOpeningProof {
        intermediate_roots,
        queries,
        final_value,
    }
}

/// Derive deterministic query indices from a Fiat-Shamir seed.
fn derive_query_indices(seed: FieldElement252, count: usize, range: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    let mut current = seed;

    for i in 0..count {
        // Hash (seed, counter) to get next index
        current = poseidon_hash(current, FieldElement252::from(i as u64));
        // Convert to index: extract lower bits
        let bytes = current.to_bytes_be();
        // Use last 8 bytes as u64, then mod range
        let val = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        indices.push((val as usize) % range);
    }

    indices
}

/// Generate a single query proof across all folding layers.
fn generate_query_proof(
    initial_pair_index: usize,
    layers: &[Vec<SecureField>],
    layer0_tree: &PoseidonMerkleTree,
    intermediate_trees: &[PoseidonMerkleTree],
    num_vars: usize,
) -> QueryProof {
    let mut rounds = Vec::with_capacity(num_vars);
    let mut pair_idx = initial_pair_index;

    for round in 0..num_vars {
        let left_idx = 2 * pair_idx;
        let right_idx = 2 * pair_idx + 1;

        let left_value = layers[round][left_idx];
        let right_value = layers[round][right_idx];

        let (left_proof, right_proof) = if round == 0 {
            // Layer 0: use the original M31 tree
            (
                layer0_tree.open(left_idx),
                layer0_tree.open(right_idx),
            )
        } else {
            // Intermediate layers: use QM31 trees
            // intermediate_trees[0] corresponds to layers[1], etc.
            let tree = &intermediate_trees[round - 1];
            (tree.open(left_idx), tree.open(right_idx))
        };

        rounds.push(QueryRoundData {
            left_value,
            right_value,
            left_proof,
            right_proof,
        });

        // Next round: the pair index halves
        pair_idx /= 2;
    }

    QueryProof {
        initial_pair_index,
        rounds,
    }
}

// ============================================================================
// Verification (for local testing)
// ============================================================================

/// Verify an MLE opening proof locally.
///
/// Checks:
/// 1. All Merkle proofs authenticate values against their layer roots
/// 2. Folding consistency: each layer is correctly derived from the previous
/// 3. Final value matches claimed evaluation
pub fn verify_mle_opening(
    commitment_root: FieldElement252,
    point: &[SecureField],
    claimed_eval: SecureField,
    proof: &MleOpeningProof,
) -> bool {
    let n = point.len();

    // Check intermediate roots count
    if proof.intermediate_roots.len() != n.saturating_sub(1) {
        return false;
    }

    // Check final value
    if proof.final_value != claimed_eval {
        return false;
    }

    // Reverse point to match folding order (fold_layer fixes last variable first)
    let reversed_point: Vec<SecureField> = point.iter().rev().copied().collect();

    // Build roots array: R_0 = commitment_root, R_1..R_{n-1} = intermediate
    let mut roots = Vec::with_capacity(n);
    roots.push(commitment_root);
    roots.extend_from_slice(&proof.intermediate_roots);

    // Re-derive query indices
    let mut seed_inputs = vec![commitment_root];
    seed_inputs.extend_from_slice(&proof.intermediate_roots);
    for p in point {
        seed_inputs.push(qm31_to_felt(*p));
    }
    let seed = poseidon_hash_many(&seed_inputs);
    let layer0_size = 1usize << n;
    let query_indices = derive_query_indices(seed, proof.queries.len(), layer0_size / 2);

    // Verify each query
    for (q_idx, query) in proof.queries.iter().enumerate() {
        if query.initial_pair_index != query_indices[q_idx] {
            return false;
        }

        if query.rounds.len() != n {
            return false;
        }

        let mut pair_idx = query.initial_pair_index;
        let mut prev_fold: Option<(SecureField, usize)> = None;

        for (round, round_data) in query.rounds.iter().enumerate() {
            let left_idx = 2 * pair_idx;
            let right_idx = 2 * pair_idx + 1;

            // Intermediate folding consistency: the fold from round i-1
            // must match the appropriate value authenticated at round i.
            if let Some((prev_val, prev_pair)) = prev_fold {
                let actual = if prev_pair.is_multiple_of(2) {
                    round_data.left_value
                } else {
                    round_data.right_value
                };
                if actual != prev_val {
                    return false;
                }
            }

            // Verify Merkle proofs — layer 0 uses M31 leaf hashing, later layers use QM31
            let (left_hash, right_hash) = if round == 0 {
                // Layer 0 values are M31 (only the .0 component is meaningful)
                let left_m31 = M31::from(round_data.left_value.to_m31_array()[0].0);
                let right_m31 = M31::from(round_data.right_value.to_m31_array()[0].0);
                (hash_m31_leaf(left_m31), hash_m31_leaf(right_m31))
            } else {
                (
                    hash_qm31_leaf(round_data.left_value),
                    hash_qm31_leaf(round_data.right_value),
                )
            };

            if !round_data.left_proof.verify(left_hash, left_idx, roots[round]) {
                return false;
            }

            if !round_data.right_proof.verify(right_hash, right_idx, roots[round]) {
                return false;
            }

            // Verify folding consistency (using reversed point order)
            let challenge = reversed_point[round];
            let expected =
                challenge * (round_data.right_value - round_data.left_value) + round_data.left_value;

            // Store fold result for intermediate consistency check at next round
            prev_fold = Some((expected, pair_idx));

            // For the last round, the folded value must equal final_value
            if round == n - 1 && expected != proof.final_value {
                return false;
            }

            pair_idx /= 2;
        }
    }

    true
}

// ============================================================================
// Serialization helpers
// ============================================================================

/// Serialize a FieldElement252 to calldata (as 4 u64 limbs, big-endian).
pub fn felt252_to_calldata(v: FieldElement252) -> [u64; 4] {
    let bytes = v.to_bytes_be();
    [
        u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    #[test]
    fn test_merkle_tree_basic() {
        let values = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let tree = PoseidonMerkleTree::from_m31_values(&values);

        assert_eq!(tree.num_leaves(), 4);
        assert_eq!(tree.depth(), 2);

        // Verify all authentication paths
        for i in 0..4 {
            let proof = tree.open(i);
            let leaf_hash = hash_m31_leaf(values[i]);
            assert!(
                proof.verify(leaf_hash, i, tree.root()),
                "Merkle proof failed for index {}",
                i
            );
        }
    }

    #[test]
    fn test_merkle_proof_rejects_wrong_leaf() {
        let values = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let tree = PoseidonMerkleTree::from_m31_values(&values);

        let proof = tree.open(0);
        let wrong_leaf = hash_m31_leaf(M31::from(99));
        assert!(!proof.verify(wrong_leaf, 0, tree.root()));
    }

    #[test]
    fn test_merkle_proof_rejects_wrong_index() {
        let values = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let tree = PoseidonMerkleTree::from_m31_values(&values);

        let proof = tree.open(0);
        let leaf_hash = hash_m31_leaf(values[0]);
        // Proof for index 0 should not verify at index 1
        assert!(!proof.verify(leaf_hash, 1, tree.root()));
    }

    #[test]
    fn test_commit_matrix() {
        let matrix =
            M31Matrix::from_data(2, 2, vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)])
                .unwrap();
        let (commitment, tree) = commit_matrix(&matrix);

        assert_eq!(commitment.num_vars, 2); // 4 entries = 2^2
        assert_eq!(commitment.rows, 2);
        assert_eq!(commitment.cols, 2);
        assert_eq!(tree.num_leaves(), 4);
        assert_eq!(commitment.root, tree.root());
    }

    #[test]
    fn test_commit_matrix_non_power_of_two() {
        // 3 entries → padded to 4
        let matrix =
            M31Matrix::from_data(1, 3, vec![M31::from(10), M31::from(20), M31::from(30)])
                .unwrap();
        let (commitment, tree) = commit_matrix(&matrix);

        assert_eq!(commitment.num_vars, 2); // padded to 4 = 2^2
        assert_eq!(tree.num_leaves(), 4);
    }

    #[test]
    fn test_fold_layer_basic() {
        // Layer: [1, 3], fold with challenge 0.5 (use M31 half = 2^30)
        let half = SecureField::from(M31::from(1u32 << 30));
        let layer = vec![
            SecureField::from(M31::from(1)),
            SecureField::from(M31::from(3)),
        ];
        let folded = fold_layer(&layer, half);
        assert_eq!(folded.len(), 1);
        // (1-0.5)*1 + 0.5*3 = 0.5 + 1.5 = 2
        assert_eq!(folded[0], SecureField::from(M31::from(2)));
    }

    #[test]
    fn test_mle_opening_2var() {
        // MLE with 2 variables (4 entries): f(x0, x1) over {0,1}²
        // f(0,0)=1, f(0,1)=2, f(1,0)=3, f(1,1)=4
        // Variable ordering: x0 is the MSB (row), x1 is LSB (col)
        // Index = x0*2 + x1
        let entries = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let tree = PoseidonMerkleTree::from_m31_values(&entries);
        let root = tree.root();

        // Evaluate MLE at a random-ish point
        let r0 = SecureField::from(M31::from(7));
        let r1 = SecureField::from(M31::from(13));
        let point = vec![r0, r1];

        // Manual MLE evaluation:
        // f(r0, r1) = (1-r0)(1-r1)*1 + (1-r0)*r1*2 + r0*(1-r1)*3 + r0*r1*4
        let one = SecureField::one();
        let expected = (one - r0) * (one - r1) * SecureField::from(M31::from(1))
            + (one - r0) * r1 * SecureField::from(M31::from(2))
            + r0 * (one - r1) * SecureField::from(M31::from(3))
            + r0 * r1 * SecureField::from(M31::from(4));

        let proof = open_mle(&entries, &point, &tree, DEFAULT_NUM_QUERIES);
        assert_eq!(proof.final_value, expected);

        // Verify the opening proof
        assert!(verify_mle_opening(root, &point, expected, &proof));
    }

    #[test]
    fn test_mle_opening_3var() {
        // MLE with 3 variables (8 entries)
        let entries: Vec<M31> = (1..=8).map(M31::from).collect();
        let tree = PoseidonMerkleTree::from_m31_values(&entries);
        let root = tree.root();

        let point = vec![
            SecureField::from(M31::from(5)),
            SecureField::from(M31::from(11)),
            SecureField::from(M31::from(17)),
        ];

        // Compute expected via recursive MLE eval
        let entries_sf: Vec<SecureField> =
            entries.iter().map(|v| SecureField::from(*v)).collect();
        let expected = eval_mle_at_point_ref(&entries_sf, &point);

        let proof = open_mle(&entries, &point, &tree, DEFAULT_NUM_QUERIES);
        assert_eq!(proof.final_value, expected);
        assert!(verify_mle_opening(root, &point, expected, &proof));
    }

    #[test]
    fn test_mle_opening_rejects_wrong_eval() {
        let entries = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let tree = PoseidonMerkleTree::from_m31_values(&entries);
        let root = tree.root();

        let point = vec![
            SecureField::from(M31::from(7)),
            SecureField::from(M31::from(13)),
        ];

        let proof = open_mle(&entries, &point, &tree, DEFAULT_NUM_QUERIES);

        // Wrong claimed eval should fail
        let wrong_eval = SecureField::from(M31::from(999));
        assert!(!verify_mle_opening(root, &point, wrong_eval, &proof));
    }

    #[test]
    fn test_mle_opening_rejects_wrong_root() {
        let entries = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let tree = PoseidonMerkleTree::from_m31_values(&entries);

        let point = vec![
            SecureField::from(M31::from(7)),
            SecureField::from(M31::from(13)),
        ];

        let entries_sf: Vec<SecureField> =
            entries.iter().map(|v| SecureField::from(*v)).collect();
        let correct_eval = eval_mle_at_point_ref(&entries_sf, &point);

        let proof = open_mle(&entries, &point, &tree, DEFAULT_NUM_QUERIES);

        // Wrong root should fail
        let wrong_root = poseidon_hash(FieldElement252::from(42u64), FieldElement252::ZERO);
        assert!(!verify_mle_opening(wrong_root, &point, correct_eval, &proof));
    }

    #[test]
    fn test_matrix_commitment_and_opening() {
        // 4×4 matrix: the real use case
        let matrix = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let (commitment, tree) = commit_matrix(&matrix);

        assert_eq!(commitment.num_vars, 4); // 16 entries = 2^4

        // Open at a random point (4 variables)
        let point = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
            SecureField::from(M31::from(11)),
            SecureField::from(M31::from(13)),
        ];

        let mut padded = matrix.data.clone();
        padded.resize(16, M31::from(0));

        let entries_sf: Vec<SecureField> = padded.iter().map(|v| SecureField::from(*v)).collect();
        let expected = eval_mle_at_point_ref(&entries_sf, &point);

        let proof = open_mle(&padded, &point, &tree, DEFAULT_NUM_QUERIES);
        assert_eq!(proof.final_value, expected);
        assert!(verify_mle_opening(commitment.root, &point, expected, &proof));
    }

    // Helper: recursive MLE evaluation for test reference
    fn eval_mle_at_point_ref(evals: &[SecureField], point: &[SecureField]) -> SecureField {
        match point {
            [] => evals[0],
            [p_i, rest @ ..] => {
                let mid = evals.len() / 2;
                let (lhs, rhs) = evals.split_at(mid);
                let l = eval_mle_at_point_ref(lhs, rest);
                let r = eval_mle_at_point_ref(rhs, rest);
                *p_i * (r - l) + l
            }
        }
    }
}
