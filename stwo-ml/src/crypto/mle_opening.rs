//! MLE evaluation opening proof protocol.
//!
//! Implements the multilinear extension opening proof matching Cairo's
//! `MleOpeningProof` struct. Given evaluations on `{0,1}^n` committed
//! via a Poseidon Merkle tree, proves that `evaluate_mle(f, r) = claimed_value`.
//!
//! The protocol iteratively folds the evaluations with sumcheck challenges,
//! building intermediate Merkle commitments at each layer. Queries are
//! drawn from the Poseidon channel for soundness.

use starknet_ff::FieldElement;
use stwo::core::fields::qm31::SecureField;
use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};
use crate::crypto::poseidon_merkle::{PoseidonMerkleTree, MerkleAuthPath};
use rayon::prelude::*;

/// Number of queries for MLE opening (matching STARK FRI query count).
pub const MLE_N_QUERIES: usize = 14;

/// MLE opening proof matching Cairo's `MleOpeningProof`.
#[derive(Debug, Clone)]
pub struct MleOpeningProof {
    /// Merkle roots of intermediate folded layers.
    pub intermediate_roots: Vec<FieldElement>,
    /// Per-query proofs with authentication paths at each folding layer.
    pub queries: Vec<MleQueryProof>,
    /// Final single value after all folding rounds.
    pub final_value: SecureField,
}

/// Proof for a single query through all folding layers.
#[derive(Debug, Clone)]
pub struct MleQueryProof {
    /// Initial pair index in the bottom layer.
    pub initial_pair_index: u32,
    /// Per-round data: left/right values + Merkle auth paths.
    pub rounds: Vec<MleQueryRoundData>,
}

/// Data for one round of a query proof.
#[derive(Debug, Clone)]
pub struct MleQueryRoundData {
    pub left_value: SecureField,
    pub right_value: SecureField,
    pub left_siblings: Vec<FieldElement>,
    pub right_siblings: Vec<FieldElement>,
}

/// Commit to an MLE by building a Poseidon Merkle tree over its evaluations.
///
/// Each SecureField (QM31) is packed into a single FieldElement for hashing.
///
/// Returns (root, tree).
pub fn commit_mle(evals: &[SecureField]) -> (FieldElement, PoseidonMerkleTree) {
    let leaves: Vec<FieldElement> = evals.iter().map(|&sf| securefield_to_felt(sf)).collect();
    let tree = PoseidonMerkleTree::build_parallel(leaves);
    (tree.root(), tree)
}

/// Compute only the MLE commitment root without storing the full tree.
///
/// Uses parallel leaf conversion (rayon) and parallel Merkle hashing.
/// More efficient than `commit_mle` when only the root is needed
/// (e.g., in batch entry preparation where the tree is discarded).
pub fn commit_mle_root_only(evals: &[SecureField]) -> FieldElement {
    use rayon::prelude::*;

    let leaves: Vec<FieldElement> = if evals.len() >= 256 {
        evals.par_iter().map(|&sf| securefield_to_felt(sf)).collect()
    } else {
        evals.iter().map(|&sf| securefield_to_felt(sf)).collect()
    };
    PoseidonMerkleTree::root_only_parallel(leaves)
}

/// Generate an MLE opening proof.
///
/// Given evaluations `evals` on `{0,1}^n`, challenges `challenges` (the sumcheck
/// assignment), and a Poseidon channel for query generation, produces a proof
/// that `evaluate_mle(evals, challenges) = final_value`.
///
/// Protocol:
/// 1. Build initial Merkle tree over `evals`
/// 2. For each challenge `r[i]`:
///    - Fold: `f'[j] = (1-r[i])*f[2j] + r[i]*f[2j+1]`
///    - Commit folded layer → intermediate root
/// 3. Draw query indices from channel
/// 4. Build query proofs with auth paths at each layer
pub fn prove_mle_opening(
    evals: &[SecureField],
    challenges: &[SecureField],
    channel: &mut PoseidonChannel,
) -> MleOpeningProof {
    let (_, proof) = prove_mle_opening_with_commitment(evals, challenges, channel);
    proof
}

/// Generate an MLE opening proof and return the initial commitment root used
/// in the Fiat-Shamir transcript.
///
/// This is equivalent to calling `commit_mle(...)` then `prove_mle_opening(...)`
/// but avoids recomputing the same root in callers that need both values.
pub fn prove_mle_opening_with_commitment(
    evals: &[SecureField],
    challenges: &[SecureField],
    channel: &mut PoseidonChannel,
) -> (FieldElement, MleOpeningProof) {
    assert!(!evals.is_empty());
    assert!(evals.len().is_power_of_two());
    let n_vars = evals.len().ilog2() as usize;
    assert_eq!(challenges.len(), n_vars);

    // Build initial tree and store layers for query generation
    let (initial_root, initial_tree) = commit_mle(evals);
    channel.mix_felt(initial_root);

    // Store all layers (evaluations) and trees for query proof construction
    let mut layer_evals: Vec<Vec<SecureField>> = vec![evals.to_vec()];
    let mut layer_trees: Vec<PoseidonMerkleTree> = vec![initial_tree];
    let mut intermediate_roots: Vec<FieldElement> = Vec::new();

    // Fold through each challenge
    // Variable ordering matches evaluate_mle: first variable splits into lo/hi halves
    for &r in challenges.iter() {
        let current = layer_evals.last().expect("layer_evals is never empty");
        let mid = current.len() / 2;

        let folded: Vec<SecureField> = if mid >= 1 << 16 {
            // Large rounds dominate opening time; parallelize safely (pure map).
            (0..mid)
                .into_par_iter()
                .map(|j| {
                    // f(r, x_rest) = (1-r)*f(0, x_rest) + r*f(1, x_rest)
                    // f(0, x_rest) = current[j], f(1, x_rest) = current[mid + j]
                    current[j] + r * (current[mid + j] - current[j])
                })
                .collect()
        } else {
            let mut v = Vec::with_capacity(mid);
            for j in 0..mid {
                v.push(current[j] + r * (current[mid + j] - current[j]));
            }
            v
        };

        if mid > 1 {
            let (root, tree) = commit_mle(&folded);
            channel.mix_felt(root);
            intermediate_roots.push(root);
            layer_trees.push(tree);
        }
        layer_evals.push(folded);
    }

    let final_value = layer_evals
        .last()
        .expect("layer_evals has final layer")[0];

    // Draw query indices (each query selects an index in [0, n/2))
    let initial_n = evals.len();
    let half_n = initial_n / 2;
    let n_queries = MLE_N_QUERIES.min(half_n);

    let mut query_indices: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let pair_idx = (raw as usize) % half_n;
        query_indices.push(pair_idx);
    }

    // Build query proofs
    let mut queries = Vec::with_capacity(n_queries);
    for &pair_idx in &query_indices {
        let mut rounds = Vec::with_capacity(n_vars);
        let mut current_idx = pair_idx;

        for round in 0..n_vars {
            let layer = &layer_evals[round];
            let mid = layer.len() / 2;
            // With lo/hi folding: left = layer[idx], right = layer[mid + idx]
            let left_idx = current_idx;
            let right_idx = mid + current_idx;

            let left_value = layer[left_idx.min(layer.len() - 1)];
            let right_value = layer[right_idx.min(layer.len() - 1)];

            // Get Merkle auth paths
            let tree = &layer_trees[round.min(layer_trees.len() - 1)];
            let left_path = if left_idx < tree.num_leaves() {
                tree.prove(left_idx)
            } else {
                MerkleAuthPath { siblings: Vec::new() }
            };
            let right_path = if right_idx < tree.num_leaves() {
                tree.prove(right_idx)
            } else {
                MerkleAuthPath { siblings: Vec::new() }
            };

            rounds.push(MleQueryRoundData {
                left_value,
                right_value,
                left_siblings: left_path.siblings,
                right_siblings: right_path.siblings,
            });

            // Next round: the folded layer has `mid` elements.
            // Its lo/hi split is at `mid/2`. Reduce index into [0, mid/2).
            let next_half = (mid / 2).max(1);
            current_idx %= next_half;
        }

        queries.push(MleQueryProof {
            initial_pair_index: pair_idx as u32,
            rounds,
        });
    }

    (
        initial_root,
        MleOpeningProof {
            intermediate_roots,
            queries,
            final_value,
        },
    )
}

/// Verify an MLE opening proof against a committed Poseidon Merkle root.
///
/// Full verification matching Cairo's `verify_mle_opening` in elo-cairo-verifier:
/// 1. Replay Fiat-Shamir transcript (mix commitment + intermediate roots)
/// 2. Draw query indices from channel
/// 3. For each query at each round:
///    a. Verify Merkle auth paths against layer roots
///    b. Check algebraic folding consistency
/// 4. Final folded value must equal `proof.final_value`
pub fn verify_mle_opening(
    commitment: FieldElement,
    proof: &MleOpeningProof,
    challenges: &[SecureField],
    channel: &mut PoseidonChannel,
) -> bool {
    let n_rounds = challenges.len();

    // 1. Replay channel transcript: mix initial commitment + intermediate roots
    channel.mix_felt(commitment);
    for root in &proof.intermediate_roots {
        channel.mix_felt(*root);
    }

    // Build layer roots: layer 0 = commitment, layers 1..n-1 = intermediate_roots
    let layer_roots_len = 1 + proof.intermediate_roots.len();

    if n_rounds == 0 {
        return proof.queries.is_empty();
    }

    // 2. Draw query indices from channel (matching prover's query derivation)
    let half_n = 1usize << (n_rounds - 1);
    let n_queries = MLE_N_QUERIES.min(half_n);

    let mut query_indices = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        query_indices.push((raw as usize) % half_n);
    }

    if proof.queries.len() != n_queries {
        return false;
    }

    // 3. Verify each query chain
    for (q_idx, query) in proof.queries.iter().enumerate() {
        if query.rounds.len() != n_rounds {
            return false;
        }

        // Verify initial pair index matches channel-derived query
        if query.initial_pair_index as usize != query_indices[q_idx] {
            return false;
        }

        let mut current_idx = query.initial_pair_index as usize;
        let mut layer_size = 1usize << n_rounds;

        for (round, round_data) in query.rounds.iter().enumerate() {
            let mid = layer_size / 2;
            let left_idx = current_idx;
            let right_idx = mid + current_idx;

            // 3a. Verify Merkle auth paths for rounds that have trees
            if round < layer_roots_len {
                let layer_root = if round == 0 {
                    commitment
                } else {
                    proof.intermediate_roots[round - 1]
                };

                let left_leaf = securefield_to_felt(round_data.left_value);
                let right_leaf = securefield_to_felt(round_data.right_value);

                let left_path = MerkleAuthPath { siblings: round_data.left_siblings.clone() };
                let right_path = MerkleAuthPath { siblings: round_data.right_siblings.clone() };

                if !PoseidonMerkleTree::verify(layer_root, left_idx, left_leaf, &left_path) {
                    return false;
                }
                if !PoseidonMerkleTree::verify(layer_root, right_idx, right_leaf, &right_path) {
                    return false;
                }
            }

            // 3b. Algebraic fold: f(r) = left + r * (right - left)
            let r = challenges[round];
            let folded = round_data.left_value + r * (round_data.right_value - round_data.left_value);

            // Last round: folded value must equal final_value
            if round == n_rounds - 1 && folded != proof.final_value {
                return false;
            }

            // Advance index for next round
            let next_half = (mid / 2).max(1);
            current_idx %= next_half;
            layer_size = mid;
        }
    }

    true
}

/// Evaluate a multilinear extension at a point (standalone helper).
///
/// Duplicated from matmul.rs for the crypto module's independence.
pub fn evaluate_mle_at(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    assert_eq!(evals.len(), 1 << point.len());
    let mut current = evals.to_vec();
    for &r in point {
        let mid = current.len() / 2;
        let mut next = Vec::with_capacity(mid);
        for i in 0..mid {
            next.push(current[i] + r * (current[mid + i] - current[i]));
        }
        current = next;
    }
    current[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::qm31::QM31;

    fn make_evals(n: usize) -> Vec<SecureField> {
        (0..n)
            .map(|i| SecureField::from(M31::from((i + 1) as u32)))
            .collect()
    }

    #[test]
    fn test_mle_opening_2_vars() {
        // 4 evaluations on {0,1}^2
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(42); // seed

        // Use arbitrary challenge points
        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let proof = prove_mle_opening(&evals, &challenges, &mut ch);

        assert_eq!(proof.final_value, evaluate_mle_at(&evals, &challenges));
        assert!(!proof.queries.is_empty());

        // Verify
        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(42);
        assert!(verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_4_vars() {
        // 16 evaluations on {0,1}^4
        let evals = make_evals(16);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(99);

        let challenges = vec![
            SecureField::from(M31::from(5)),
            SecureField::from(M31::from(11)),
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let proof = prove_mle_opening(&evals, &challenges, &mut ch);
        let expected = evaluate_mle_at(&evals, &challenges);
        assert_eq!(proof.final_value, expected);
    }

    #[test]
    fn test_mle_opening_tampered_fails() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(42);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let mut proof = prove_mle_opening(&evals, &challenges, &mut ch);

        // Tamper with final value
        proof.final_value = SecureField::from(M31::from(999));

        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(42);
        assert!(!verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }

    #[test]
    fn test_mle_opening_wrong_commitment_fails() {
        let evals = make_evals(4);
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(42);

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let proof = prove_mle_opening(&evals, &challenges, &mut ch);

        // Use a WRONG commitment — should fail Merkle verification
        let wrong_commitment = FieldElement::from(0xDEADBEEFu64);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(42);
        assert!(
            !verify_mle_opening(wrong_commitment, &proof, &challenges, &mut ch_v),
            "Wrong commitment should fail Merkle verification"
        );
    }

    #[test]
    fn test_mle_opening_roundtrip() {
        // Prove then verify — must be consistent
        let evals: Vec<SecureField> = vec![
            QM31(CM31(M31::from(10), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            QM31(CM31(M31::from(20), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            QM31(CM31(M31::from(30), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            QM31(CM31(M31::from(40), M31::from(0)), CM31(M31::from(0), M31::from(0))),
        ];

        let challenges = vec![
            SecureField::from(M31::from(2)),
            SecureField::from(M31::from(5)),
        ];

        let mut ch = PoseidonChannel::new();
        ch.mix_u64(7);
        let proof = prove_mle_opening(&evals, &challenges, &mut ch);

        // Final value must match direct MLE evaluation
        let direct = evaluate_mle_at(&evals, &challenges);
        assert_eq!(proof.final_value, direct);

        // Verify
        let (commitment, _) = commit_mle(&evals);
        let mut ch_v = PoseidonChannel::new();
        ch_v.mix_u64(7);
        assert!(verify_mle_opening(commitment, &proof, &challenges, &mut ch_v));
    }
}
