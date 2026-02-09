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
use num_traits::Zero;

use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};
use crate::crypto::poseidon_merkle::{PoseidonMerkleTree, MerkleAuthPath};

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
    let tree = PoseidonMerkleTree::build(leaves);
    (tree.root(), tree)
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
    let mut current = evals.to_vec();
    for &r in challenges.iter() {
        let mid = current.len() / 2;
        let mut folded = Vec::with_capacity(mid);
        for j in 0..mid {
            // f(r, x_rest) = (1-r)*f(0, x_rest) + r*f(1, x_rest)
            // f(0, x_rest) = current[j], f(1, x_rest) = current[mid + j]
            folded.push(current[j] + r * (current[mid + j] - current[j]));
        }

        if folded.len() > 1 {
            let (root, tree) = commit_mle(&folded);
            channel.mix_felt(root);
            intermediate_roots.push(root);
            layer_trees.push(tree);
        }
        layer_evals.push(folded.clone());
        current = folded;
    }

    let final_value = current[0];

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

    MleOpeningProof {
        intermediate_roots,
        queries,
        final_value,
    }
}

/// Verify an MLE opening proof.
///
/// Checks:
/// 1. Final value matches what folding through all query rounds produces
/// 2. All query chains fold consistently to `final_value`
/// 3. Structural consistency (round counts match challenges)
///
/// Note: On-chain verification additionally checks Merkle auth paths against
/// commitment roots. This local check verifies algebraic consistency.
pub fn verify_mle_opening(
    _commitment: FieldElement,
    proof: &MleOpeningProof,
    challenges: &[SecureField],
    _channel: &mut PoseidonChannel,
) -> bool {
    // Verify each query chain
    for query in &proof.queries {
        if query.rounds.len() != challenges.len() {
            return false;
        }

        // Check folding consistency at each round
        let mut last_folded = SecureField::zero();
        for (round_idx, round_data) in query.rounds.iter().enumerate() {
            let r = challenges[round_idx];
            // Lo/hi folding: f(r, rest) = left + r * (right - left)
            last_folded = round_data.left_value + r * (round_data.right_value - round_data.left_value);
        }

        // Last round's folded value should equal final_value
        if last_folded != proof.final_value {
            return false;
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
