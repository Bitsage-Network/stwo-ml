// MLE Commitment Opening Verification
//
// Verifies that a multilinear extension committed via Poseidon Merkle tree
// evaluates to a claimed value at a given point.
//
// Protocol (matching Rust's mle_opening.rs):
//   1. Prover commits evaluations in a Poseidon Merkle tree (root R₀)
//   2. Prover folds with lo/hi split:
//      L_{i+1}[j] = L_i[j] + r[i] * (L_i[mid+j] - L_i[j])
//   3. Prover commits intermediate layers → roots R₁, ..., R_{n-1}
//   4. Channel absorbs R₀, R₁, ..., R_{n-1} via mix_felt
//   5. Verifier draws query indices from channel
//   6. For each query, verifier checks Merkle proofs + folding consistency
//
// Leaf values: raw packed QM31 (securefield_to_felt), no per-leaf hash
// Internal nodes: poseidon_hash(left, right)
// Folding order: forward (first variable first)

use core::poseidon::hades_permutation;
use crate::field::{QM31, qm31_add, qm31_sub, qm31_mul, qm31_eq, pack_qm31_to_felt, pow2};
use crate::channel::{PoseidonChannel, channel_mix_felt, channel_draw_query_indices};
use crate::types::MleOpeningProof;

/// Number of spot-check queries for the MLE folding protocol.
/// Matches Rust's MLE_N_QUERIES = 14.
pub const MLE_NUM_QUERIES: u32 = 14;

/// Verify a Poseidon Merkle authentication path.
/// Sibling order is bottom-up (leaf → root).
fn verify_merkle_path(
    leaf_hash: felt252, index: u32, siblings: Span<felt252>, root: felt252,
) -> bool {
    let mut current = leaf_hash;
    let mut idx = index;
    let mut i: u32 = 0;
    loop {
        if i >= siblings.len() {
            break;
        }
        let sibling = *siblings.at(i);
        if idx % 2 == 0 {
            let (s0, _, _) = hades_permutation(current, sibling, 2);
            current = s0;
        } else {
            let (s0, _, _) = hades_permutation(sibling, current, 2);
            current = s0;
        }
        idx = idx / 2;
        i += 1;
    };
    current == root
}

/// Compute the next query pair index after folding.
fn next_query_pair_index(current_idx: u32, layer_mid: u32) -> u32 {
    let next_half = layer_mid / 2;
    if next_half == 0 {
        0
    } else {
        current_idx % next_half
    }
}

/// Verify an MLE opening proof against a committed Poseidon Merkle root.
///
/// Matches Rust's verify_mle_opening() from mle_opening.rs exactly:
/// 1. Channel-based Fiat-Shamir transcript (mix commitment + intermediate roots)
/// 2. Channel-based query derivation (draw_query_indices)
/// 3. Lo/hi split folding (not consecutive pairs)
/// 4. Raw packed QM31 leaves (no poseidon_hash(leaf, 0))
/// 5. Forward folding order (first variable first)
/// 6. MLE_NUM_QUERIES = 14
pub fn verify_mle_opening(
    commitment_root: felt252,
    proof: @MleOpeningProof,
    challenges: Span<QM31>,
    ref ch: PoseidonChannel,
) -> bool {
    let n_rounds: u32 = challenges.len();

    // Replay channel transcript: mix initial commitment and intermediate roots
    channel_mix_felt(ref ch, commitment_root);
    let intermediate_roots_span = proof.intermediate_roots.span();
    let mut ir_i: u32 = 0;
    loop {
        if ir_i >= intermediate_roots_span.len() {
            break;
        }
        channel_mix_felt(ref ch, *intermediate_roots_span.at(ir_i));
        ir_i += 1;
    };

    // Build layer roots: layer 0 = commitment, layers 1..n-1 = intermediate_roots
    let layer_roots_len: u32 = 1 + intermediate_roots_span.len();

    if n_rounds == 0 {
        return proof.queries.len() == 0;
    }

    // Draw query indices
    let half_n: u32 = pow2(n_rounds - 1);
    let n_queries: u32 = if MLE_NUM_QUERIES < half_n {
        MLE_NUM_QUERIES
    } else {
        half_n
    };
    let query_indices = channel_draw_query_indices(ref ch, half_n, n_queries);

    let queries_span = proof.queries.span();
    if queries_span.len() != n_queries {
        return false;
    }

    // Verify each query chain
    let mut q_idx: u32 = 0;
    loop {
        if q_idx >= n_queries {
            break;
        }

        let query = queries_span.at(q_idx);
        let rounds_span = query.rounds.span();

        if rounds_span.len() != n_rounds {
            return false;
        }

        // Verify initial pair index matches channel-derived query
        if *query.initial_pair_index != *query_indices.at(q_idx) {
            return false;
        }

        let mut current_idx: u32 = *query.initial_pair_index;
        let mut layer_size: u32 = pow2(n_rounds);

        let mut round: u32 = 0;
        loop {
            if round >= n_rounds {
                break;
            }

            let rd = rounds_span.at(round);
            let left_value: QM31 = *rd.left_value;
            let right_value: QM31 = *rd.right_value;
            let left_siblings = rd.left_siblings.span();
            let right_siblings = rd.right_siblings.span();

            // Lo/hi split: left = layer[idx], right = layer[mid + idx]
            let mid: u32 = layer_size / 2;
            let left_idx: u32 = current_idx;
            let right_idx: u32 = mid + current_idx;

            // Verify Merkle paths for rounds that have trees
            if round < layer_roots_len {
                let layer_root = if round == 0 {
                    commitment_root
                } else {
                    *intermediate_roots_span.at(round - 1)
                };

                let left_leaf = pack_qm31_to_felt(left_value);
                let right_leaf = pack_qm31_to_felt(right_value);

                if !verify_merkle_path(left_leaf, left_idx, left_siblings, layer_root) {
                    return false;
                }
                if !verify_merkle_path(right_leaf, right_idx, right_siblings, layer_root) {
                    return false;
                }
            }

            // Algebraic fold: f(r) = left + r * (right - left)
            let challenge: QM31 = *challenges.at(round);
            let diff = qm31_sub(right_value, left_value);
            let fold_val = qm31_add(left_value, qm31_mul(challenge, diff));

            // Last round: folded value must equal final_value
            if round == n_rounds - 1 {
                if !qm31_eq(fold_val, *proof.final_value) {
                    return false;
                }
            }

            current_idx = next_query_pair_index(current_idx, mid);
            layer_size = mid;

            round += 1;
        };

        q_idx += 1;
    };

    true
}
