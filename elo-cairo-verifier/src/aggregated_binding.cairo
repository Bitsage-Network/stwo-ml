/// Aggregated weight binding verification via unified oracle mismatch sumcheck.
///
/// Verifies M weight claims in one shot: mismatch sumcheck + single MLE opening.
/// Produces ~17K felts calldata instead of ~2.4M (160 separate openings).

use core::poseidon::poseidon_hash_span;
use crate::field::{
    QM31, qm31_add, qm31_sub, qm31_mul, qm31_eq, qm31_zero, qm31_one,
    poly_eval_degree2, eq_eval, pack_qm31_to_felt,
};
use crate::channel::{
    PoseidonChannel, channel_mix_felt, channel_mix_felts, channel_mix_poly_coeffs,
    channel_draw_qm31,
};
use crate::mle::verify_mle_opening;
use crate::types::MleOpeningProof;
use crate::model_verifier::WeightClaimData;

/// Configuration for the aggregated binding protocol.
#[derive(Drop, Copy, Serde)]
pub struct AggregatedBindingConfig {
    pub selector_bits: u32,
    pub n_max: u32,
    pub m_padded: u32,
    pub n_global: u32,
    pub n_claims: u32,
}

/// Aggregated weight binding proof.
#[derive(Drop, Serde)]
pub struct AggregatedWeightBindingProof {
    pub config: AggregatedBindingConfig,
    /// Degree-2 sumcheck round polynomials (c0, c1, c2) per round.
    pub round_polys: Array<(QM31, QM31, QM31)>,
    /// Oracle evaluation W_global(s) at the sumcheck challenge point.
    pub oracle_eval_at_s: QM31,
    /// Super-root hash.
    pub super_root: felt252,
    /// Per-matrix subtree roots (padded to m_padded).
    pub subtree_roots: Array<felt252>,
    /// Single MLE opening proof against super-root.
    pub opening_proof: MleOpeningProof,
}

/// Compute the Poseidon hash of (a, b) using 2-to-1 hashing.
fn poseidon_hash_2(a: felt252, b: felt252) -> felt252 {
    poseidon_hash_span(array![a, b].span())
}

/// Build Merkle root from a power-of-2 array of leaves.
fn merkle_root_from_leaves(leaves: Span<felt252>) -> felt252 {
    let len = leaves.len();
    assert!(len > 0, "empty leaves");
    if len == 1 {
        return *leaves.at(0);
    }

    // Build first layer of hashes
    let mut current: Array<felt252> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= len {
            break;
        }
        current.append(poseidon_hash_2(*leaves.at(i), *leaves.at(i + 1)));
        i += 2;
    };

    // Reduce until single root
    loop {
        if current.len() <= 1 {
            break;
        }
        let mut next: Array<felt252> = array![];
        let mut j: u32 = 0;
        let clen = current.len();
        loop {
            if j >= clen {
                break;
            }
            next.append(poseidon_hash_2(*current.at(j), *current.at(j + 1)));
            j += 2;
        };
        current = next;
    };

    *current.at(0)
}

/// Verify the super-root matches the subtree roots.
///
/// The super-root is the Merkle root of the subtree_roots array.
/// The verifier reconstructs it from claimed subtree_roots and checks
/// it matches the claimed super_root.
fn verify_super_root(
    claimed_root: felt252,
    subtree_roots: Span<felt252>,
) -> bool {
    let computed = merkle_root_from_leaves(subtree_roots);
    computed == claimed_root
}

/// Compute the Poseidon Merkle root of a tree with 2^n_vars zero QM31 leaves.
///
/// Matches Rust's `compute_zero_tree_root()` in aggregated_opening.rs.
/// Each leaf is `pack_qm31_to_felt(QM31::zero())`.
/// The tree is built bottom-up: h_{level} = poseidon_hash_2(h_{level-1}, h_{level-1}).
fn compute_zero_tree_root(n_vars: u32) -> felt252 {
    let zero_leaf = pack_qm31_to_felt(qm31_zero());
    if n_vars == 0 {
        return zero_leaf;
    }
    // For n_vars >= 1: bottom hash = hash(zero_leaf, zero_leaf), then self-hash
    let mut h = poseidon_hash_2(zero_leaf, zero_leaf);
    let mut level: u32 = 1;
    loop {
        if level >= n_vars {
            break;
        }
        h = poseidon_hash_2(h, h);
        level += 1;
    };
    h
}

/// Extend a commitment root from depth `local_n_vars` to depth `n_max`
/// by pairing with zero-tree roots at each level.
///
/// Matches Rust's extension logic in `build_super_root()`.
fn extend_commitment_to_depth(
    commitment: felt252,
    local_n_vars: u32,
    n_max: u32,
) -> felt252 {
    if local_n_vars == n_max {
        return commitment;
    }
    let levels_to_extend = n_max - local_n_vars;
    let mut extended = commitment;
    let mut zero_h = compute_zero_tree_root(local_n_vars);
    let mut lvl: u32 = 0;
    loop {
        if lvl >= levels_to_extend {
            break;
        }
        extended = poseidon_hash_2(extended, zero_h);
        zero_h = poseidon_hash_2(zero_h, zero_h);
        lvl += 1;
    };
    extended
}

/// Pre-compute zero-tree roots for levels 0..n_max (inclusive).
///
/// cache[0] = pack(QM31::zero)  (the zero leaf)
/// cache[i] = poseidon_hash_2(cache[i-1], cache[i-1])  for i >= 1
///
/// This avoids redundant recomputation inside `verify_subtree_commitments`.
fn build_zero_tree_cache(n_max: u32) -> Array<felt252> {
    let mut cache: Array<felt252> = array![];
    let zero_leaf = pack_qm31_to_felt(qm31_zero());
    cache.append(zero_leaf); // level 0
    let mut h = zero_leaf;
    let mut level: u32 = 0;
    loop {
        if level >= n_max {
            break;
        }
        h = poseidon_hash_2(h, h);
        cache.append(h); // level+1
        level += 1;
    };
    cache
}

/// Extend a commitment root from `local_n_vars` to `n_max` using a pre-built cache.
///
/// cache[i] is the zero-tree root at depth i. At each extension level,
/// we pair with cache[lvl] (the zero-tree sibling at that level).
fn extend_commitment_to_depth_cached(
    commitment: felt252,
    local_n_vars: u32,
    n_max: u32,
    cache: @Array<felt252>,
) -> felt252 {
    if local_n_vars == n_max {
        return commitment;
    }
    let mut extended = commitment;
    let mut lvl = local_n_vars;
    loop {
        if lvl >= n_max {
            break;
        }
        extended = poseidon_hash_2(extended, *cache.at(lvl));
        lvl += 1;
    };
    extended
}

/// Verify that subtree roots match registered weight commitments.
///
/// For each claim i < n_claims:
///   - Compute expected subtree root by extending weight_commitments[i]
///     from local_n_vars to n_max depth using zero-tree hashing.
///   - Check subtree_roots[i] == expected.
/// For padding slots i >= n_claims:
///   - Check subtree_roots[i] == zero_tree_root(n_max).
///
/// Uses a pre-built zero-tree cache to avoid redundant Poseidon hashing.
fn verify_subtree_commitments(
    subtree_roots: Span<felt252>,
    weight_commitments: Span<felt252>,
    weight_claims: Span<WeightClaimData>,
    n_claims: u32,
    n_max: u32,
    m_padded: u32,
) -> bool {
    // Check that we have the right number of subtree roots
    if subtree_roots.len() != m_padded {
        return false;
    }
    if weight_commitments.len() < n_claims {
        return false;
    }

    // Pre-compute zero-tree roots for all levels 0..n_max once
    let cache = build_zero_tree_cache(n_max);

    // For actual claims: extend commitment to n_max depth and compare
    let mut i: u32 = 0;
    loop {
        if i >= n_claims {
            break;
        }
        let commitment = *weight_commitments.at(i);
        // local_n_vars = length of the eval_point for this claim
        let local_n_vars = weight_claims.at(i).eval_point.len();
        let expected = extend_commitment_to_depth_cached(commitment, local_n_vars, n_max, @cache);
        if *subtree_roots.at(i) != expected {
            return false;
        }
        i += 1;
    };

    // Padding slots must be zero-tree root at n_max depth (read from cache)
    let zero_root = *cache.at(n_max);
    loop {
        if i >= m_padded {
            break;
        }
        if *subtree_roots.at(i) != zero_root {
            return false;
        }
        i += 1;
    };

    true
}

/// Main aggregated binding verification.
///
/// Verifies:
/// 1. Super-root is correctly built from subtree roots
/// 2. Mismatch sumcheck: R(t) = Σ β_i * eq(g_i, t) * (W(t) - v_i) ≡ 0
/// 3. Oracle evaluation at challenge point via single MLE opening
pub fn verify_aggregated_binding(
    proof: @AggregatedWeightBindingProof,
    weight_claims: Span<WeightClaimData>,
    weight_commitments: Span<felt252>,
    ref ch: PoseidonChannel,
) -> bool {
    let config = proof.config;

    // 1. Verify super-root from subtree roots
    if !verify_super_root(*proof.super_root, proof.subtree_roots.span()) {
        return false;
    }

    // 1b. Verify subtree roots match registered weight commitments
    //     (with zero-tree extension for smaller matrices).
    //     This is the critical binding: ensures the prover's claimed subtree
    //     roots correspond to the actual registered weight MLE commitments.
    if !verify_subtree_commitments(
        proof.subtree_roots.span(),
        weight_commitments,
        weight_claims,
        *config.n_claims,
        *config.n_max,
        *config.m_padded,
    ) {
        return false;
    }

    // 2. Mix super-root into channel
    channel_mix_felt(ref ch, *proof.super_root);

    // 3. Draw β = ρ^i weights
    let rho = channel_draw_qm31(ref ch);
    let n_claims = *config.n_claims;
    let n_global = *config.n_global;

    // 4. Verify sumcheck rounds
    let round_polys = proof.round_polys.span();
    if round_polys.len() != n_global {
        return false;
    }

    let mut current_sum = qm31_zero(); // Expected sum = 0 (mismatch is zero)
    let mut challenge_point: Array<QM31> = array![];

    let mut round: u32 = 0;
    loop {
        if round >= n_global {
            break;
        }

        let (c0, c1, c2) = *round_polys.at(round);

        // p(0) = c0
        let p0 = c0;
        // p(1) = c0 + c1 + c2
        let p1 = qm31_add(c0, qm31_add(c1, c2));

        // Check: p(0) + p(1) == current_sum
        let round_sum = qm31_add(p0, p1);
        if !qm31_eq(round_sum, current_sum) {
            return false;
        }

        // Mix round polynomial into channel
        channel_mix_poly_coeffs(ref ch, c0, c1, c2);

        // Draw challenge for this round
        let r = channel_draw_qm31(ref ch);
        challenge_point.append(r);

        // Update: current_sum = p(r) = c0 + c1*r + c2*r^2
        current_sum = poly_eval_degree2(c0, c1, c2, r);

        round += 1;
    };

    // 5. Final check: sumcheck output = Σ β_i * eq(g_i, s) * (oracle_eval - v_i)
    let oracle_eval = *proof.oracle_eval_at_s;
    let selector_bits = *config.selector_bits;
    let n_max = *config.n_max;

    let mut verifier_sum = qm31_zero();
    let mut rho_pow = qm31_one();
    let challenge_span = challenge_point.span();

    let mut claim_i: u32 = 0;
    loop {
        if claim_i >= n_claims {
            break;
        }

        let claim = weight_claims.at(claim_i);

        // Build global point g_i = [selector_bits || padded_local_point]
        let mut g_i: Array<QM31> = array![];

        // Selector bits (MSB first): binary encoding of claim_i
        let mut bit_idx: u32 = selector_bits;
        loop {
            if bit_idx == 0 {
                break;
            }
            bit_idx -= 1;
            let bit_val: u32 = (claim_i / pow2(bit_idx)) % 2;
            if bit_val == 1 {
                g_i.append(qm31_one());
            } else {
                g_i.append(qm31_zero());
            }
        };

        // Local bits: zero-pad to n_max, then eval_point
        let eval_len = claim.eval_point.len();
        let pad = n_max - eval_len;
        let mut p: u32 = 0;
        loop {
            if p >= pad {
                break;
            }
            g_i.append(qm31_zero());
            p += 1;
        };

        let mut ep: u32 = 0;
        loop {
            if ep >= eval_len {
                break;
            }
            g_i.append(*claim.eval_point.at(ep));
            ep += 1;
        };

        // eq(g_i, challenge_point)
        let eq_val = eq_eval(g_i.span(), challenge_span);

        // β_i * eq(g_i, s) * (oracle_eval - v_i)
        let mismatch = qm31_sub(oracle_eval, *claim.expected_value);
        let contribution = qm31_mul(rho_pow, qm31_mul(eq_val, mismatch));
        verifier_sum = qm31_add(verifier_sum, contribution);

        rho_pow = qm31_mul(rho_pow, rho);
        claim_i += 1;
    };

    if !qm31_eq(current_sum, verifier_sum) {
        return false;
    }

    // 6. Mix oracle eval into channel
    let oracle_arr: Array<QM31> = array![oracle_eval];
    channel_mix_felts(ref ch, oracle_arr.span());

    // 7. Verify single MLE opening against super-root
    verify_mle_opening(
        *proof.super_root,
        proof.opening_proof,
        challenge_span,
        ref ch,
    )
}

/// Power of 2 helper: 2^n for small n (up to 31).
fn pow2(n: u32) -> u32 {
    if n == 0 {
        1_u32
    } else if n == 1 {
        2_u32
    } else if n == 2 {
        4_u32
    } else if n == 3 {
        8_u32
    } else if n == 4 {
        16_u32
    } else if n == 5 {
        32_u32
    } else if n == 6 {
        64_u32
    } else if n == 7 {
        128_u32
    } else if n == 8 {
        256_u32
    } else {
        // For n > 8, compute iteratively
        let mut result: u32 = 256;
        let mut i: u32 = 8;
        loop {
            if i >= n {
                break;
            }
            result = result * 2;
            i += 1;
        };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{
        verify_aggregated_binding,
        AggregatedBindingConfig, AggregatedWeightBindingProof,
        merkle_root_from_leaves, pow2, poseidon_hash_2, verify_super_root,
        compute_zero_tree_root, extend_commitment_to_depth,
        build_zero_tree_cache, extend_commitment_to_depth_cached,
        verify_subtree_commitments,
    };
    use crate::field::{QM31, qm31_new, qm31_zero, qm31_eq, pack_qm31_to_felt};
    use crate::channel::channel_default;
    use crate::types::MleOpeningProof;
    use crate::model_verifier::WeightClaimData;

    #[test]
    fn test_pow2_values() {
        assert!(pow2(0) == 1, "2^0 = 1");
        assert!(pow2(1) == 2, "2^1 = 2");
        assert!(pow2(8) == 256, "2^8 = 256");
        assert!(pow2(10) == 1024, "2^10 = 1024");
    }

    #[test]
    fn test_merkle_root_single_leaf() {
        let leaves: Array<felt252> = array![0x42];
        let root = merkle_root_from_leaves(leaves.span());
        assert!(root == 0x42, "single leaf root should be the leaf itself");
    }

    #[test]
    fn test_merkle_root_two_leaves() {
        let a: felt252 = 0x1;
        let b: felt252 = 0x2;
        let leaves: Array<felt252> = array![a, b];
        let expected = poseidon_hash_2(a, b);
        let root = merkle_root_from_leaves(leaves.span());
        assert!(root == expected, "two leaf root should be hash(a, b)");
    }

    #[test]
    fn test_verify_super_root() {
        let a: felt252 = 0x1;
        let b: felt252 = 0x2;
        let subtree_roots: Array<felt252> = array![a, b];
        let expected_root = poseidon_hash_2(a, b);
        assert!(verify_super_root(expected_root, subtree_roots.span()), "super root should verify");
        assert!(
            !verify_super_root(0x9999, subtree_roots.span()),
            "wrong super root should fail",
        );
    }

    #[test]
    fn test_zero_tree_root_depth0() {
        // Depth 0 = single leaf = pack(QM31::zero)
        let zero_leaf = pack_qm31_to_felt(qm31_zero());
        assert!(compute_zero_tree_root(0) == zero_leaf, "depth 0 should be zero leaf");
    }

    #[test]
    fn test_zero_tree_root_depth1() {
        // Depth 1 = hash(zero_leaf, zero_leaf)
        let zero_leaf = pack_qm31_to_felt(qm31_zero());
        let expected = poseidon_hash_2(zero_leaf, zero_leaf);
        assert!(compute_zero_tree_root(1) == expected, "depth 1 mismatch");
    }

    #[test]
    fn test_zero_tree_root_depth2() {
        // Depth 2 = hash(h1, h1) where h1 = hash(z, z)
        let zero_leaf = pack_qm31_to_felt(qm31_zero());
        let h1 = poseidon_hash_2(zero_leaf, zero_leaf);
        let expected = poseidon_hash_2(h1, h1);
        assert!(compute_zero_tree_root(2) == expected, "depth 2 mismatch");
    }

    #[test]
    fn test_extend_commitment_same_depth() {
        // No extension needed when local_n_vars == n_max
        let commitment: felt252 = 0xABCD;
        assert!(
            extend_commitment_to_depth(commitment, 4, 4) == commitment,
            "same depth should return commitment unchanged",
        );
    }

    #[test]
    fn test_extend_commitment_one_level() {
        // Extend from depth 3 to depth 4:
        // extended = hash(commitment, zero_tree_root(3))
        let commitment: felt252 = 0xABCD;
        let zero_h_3 = compute_zero_tree_root(3);
        let expected = poseidon_hash_2(commitment, zero_h_3);
        assert!(
            extend_commitment_to_depth(commitment, 3, 4) == expected,
            "one-level extension mismatch",
        );
    }

    #[test]
    fn test_verify_subtree_commitments_matching() {
        // Two claims, both at n_max=4, m_padded=2
        let commitment_0: felt252 = 0x111;
        let commitment_1: felt252 = 0x222;
        let subtree_roots: Array<felt252> = array![commitment_0, commitment_1];
        let weight_commitments: Array<felt252> = array![commitment_0, commitment_1];
        // Claims with eval_point length = n_max = 4
        let claims: Array<WeightClaimData> = array![
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
        ];
        assert!(
            verify_subtree_commitments(
                subtree_roots.span(), weight_commitments.span(), claims.span(), 2, 4, 2,
            ),
            "matching commitments should pass",
        );
    }

    #[test]
    fn test_verify_subtree_commitments_wrong_root_fails() {
        // Prover sends fake subtree root
        let real_commitment: felt252 = 0x111;
        let fake_root: felt252 = 0x999;
        let subtree_roots: Array<felt252> = array![fake_root, fake_root];
        let weight_commitments: Array<felt252> = array![real_commitment, real_commitment];
        let claims: Array<WeightClaimData> = array![
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
        ];
        assert!(
            !verify_subtree_commitments(
                subtree_roots.span(), weight_commitments.span(), claims.span(), 2, 4, 2,
            ),
            "fake subtree root should fail",
        );
    }

    #[test]
    fn test_verify_subtree_commitments_padding() {
        // 1 claim, m_padded=2 — padding slot must be zero tree root
        let commitment: felt252 = 0x111;
        let zero_root = compute_zero_tree_root(4);
        let subtree_roots: Array<felt252> = array![commitment, zero_root];
        let weight_commitments: Array<felt252> = array![commitment];
        let claims: Array<WeightClaimData> = array![
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
        ];
        assert!(
            verify_subtree_commitments(
                subtree_roots.span(), weight_commitments.span(), claims.span(), 1, 4, 2,
            ),
            "valid padding should pass",
        );
    }

    #[test]
    fn test_verify_subtree_commitments_bad_padding_fails() {
        // Padding slot is NOT zero tree root
        let commitment: felt252 = 0x111;
        let subtree_roots: Array<felt252> = array![commitment, 0xBAD];
        let weight_commitments: Array<felt252> = array![commitment];
        let claims: Array<WeightClaimData> = array![
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
        ];
        assert!(
            !verify_subtree_commitments(
                subtree_roots.span(), weight_commitments.span(), claims.span(), 1, 4, 2,
            ),
            "bad padding should fail",
        );
    }

    #[test]
    fn test_build_zero_tree_cache_depth0() {
        let cache = build_zero_tree_cache(0);
        // Only one entry: the zero leaf
        assert!(cache.len() == 1, "cache depth 0 should have 1 entry");
        let zero_leaf = pack_qm31_to_felt(qm31_zero());
        assert!(*cache.at(0) == zero_leaf, "cache[0] should be zero leaf");
    }

    #[test]
    fn test_build_zero_tree_cache_matches_compute() {
        // Cache entries should match compute_zero_tree_root for each depth
        let cache = build_zero_tree_cache(4);
        assert!(cache.len() == 5, "cache depth 4 should have 5 entries");
        let mut d: u32 = 0;
        loop {
            if d > 4 {
                break;
            }
            assert!(
                *cache.at(d) == compute_zero_tree_root(d),
                "cache mismatch at depth",
            );
            d += 1;
        };
    }

    #[test]
    fn test_extend_cached_matches_uncached() {
        // Both extension functions should produce identical results
        let commitment: felt252 = 0xDEAD;
        let cache = build_zero_tree_cache(6);
        let mut local: u32 = 0;
        loop {
            if local > 6 {
                break;
            }
            let uncached = extend_commitment_to_depth(commitment, local, 6);
            let cached = extend_commitment_to_depth_cached(commitment, local, 6, @cache);
            assert!(uncached == cached, "cached/uncached mismatch");
            local += 1;
        };
    }

    // ---- Mode 4 integration tests ----

    #[test]
    fn test_config_serde_roundtrip() {
        let config = AggregatedBindingConfig {
            selector_bits: 2,
            n_max: 8,
            m_padded: 4,
            n_global: 10,
            n_claims: 3,
        };

        let mut output: Array<felt252> = array![];
        config.serialize(ref output);

        // AggregatedBindingConfig has 5 u32 fields → 5 felts
        assert!(output.len() == 5, "config should serialize to 5 felts");

        let mut span = output.span();
        let deserialized: AggregatedBindingConfig = Serde::deserialize(ref span).unwrap();
        assert!(deserialized.selector_bits == 2, "selector_bits mismatch");
        assert!(deserialized.n_max == 8, "n_max mismatch");
        assert!(deserialized.m_padded == 4, "m_padded mismatch");
        assert!(deserialized.n_global == 10, "n_global mismatch");
        assert!(deserialized.n_claims == 3, "n_claims mismatch");
    }

    #[test]
    fn test_proof_serde_roundtrip() {
        // Construct a minimal AggregatedWeightBindingProof and verify Serde roundtrip
        let config = AggregatedBindingConfig {
            selector_bits: 1,
            n_max: 2,
            m_padded: 2,
            n_global: 3,
            n_claims: 1,
        };

        let q1 = qm31_new(1, 2, 3, 4);
        let q2 = qm31_new(5, 6, 7, 8);
        let q3 = qm31_new(9, 10, 11, 12);

        let round_polys: Array<(QM31, QM31, QM31)> = array![
            (q1, q2, q3),
            (q2, q3, q1),
            (q3, q1, q2),
        ];

        let oracle_eval = qm31_new(100, 200, 300, 400);

        let proof = AggregatedWeightBindingProof {
            config,
            round_polys,
            oracle_eval_at_s: oracle_eval,
            super_root: 0xCAFE,
            subtree_roots: array![0xAA, 0xBB],
            opening_proof: MleOpeningProof {
                intermediate_roots: array![0x11, 0x22],
                queries: array![],
                final_value: qm31_new(42, 43, 44, 45),
            },
        };

        let mut output: Array<felt252> = array![];
        proof.serialize(ref output);
        assert!(output.len() > 0, "proof should serialize to non-empty array");

        let mut span = output.span();
        let d: AggregatedWeightBindingProof = Serde::deserialize(ref span).unwrap();

        // Verify config fields
        assert!(d.config.selector_bits == 1, "config.selector_bits");
        assert!(d.config.n_max == 2, "config.n_max");
        assert!(d.config.m_padded == 2, "config.m_padded");
        assert!(d.config.n_global == 3, "config.n_global");
        assert!(d.config.n_claims == 1, "config.n_claims");

        // Verify round_polys count
        assert!(d.round_polys.len() == 3, "round_polys count");

        // Verify oracle eval
        assert!(qm31_eq(d.oracle_eval_at_s, oracle_eval), "oracle_eval");

        // Verify super_root
        assert!(d.super_root == 0xCAFE, "super_root");

        // Verify subtree_roots
        assert!(d.subtree_roots.len() == 2, "subtree_roots count");
        assert!(*d.subtree_roots.at(0) == 0xAA, "subtree_roots[0]");
        assert!(*d.subtree_roots.at(1) == 0xBB, "subtree_roots[1]");

        // Verify opening proof
        assert!(d.opening_proof.intermediate_roots.len() == 2, "intermediate_roots count");
        assert!(qm31_eq(d.opening_proof.final_value, qm31_new(42, 43, 44, 45)), "final_value");
    }

    #[test]
    fn test_selector_bit_encoding_2bits() {
        // For selector_bits=2, claim indices 0..3 should encode as:
        //   claim 0 → [0, 0]
        //   claim 1 → [0, 1]
        //   claim 2 → [1, 0]
        //   claim 3 → [1, 1]
        // The encoding is MSB-first via (claim_i / pow2(bit_idx)) % 2
        let selector_bits: u32 = 2;

        // claim 0: bit1 = (0/2)%2 = 0, bit0 = (0/1)%2 = 0 → [0, 0]
        let mut bits_0: Array<u32> = array![];
        let mut bi: u32 = selector_bits;
        loop {
            if bi == 0 { break; }
            bi -= 1;
            bits_0.append((0_u32 / pow2(bi)) % 2);
        };
        assert!(*bits_0.at(0) == 0 && *bits_0.at(1) == 0, "claim 0 = [0,0]");

        // claim 1: bit1 = (1/2)%2 = 0, bit0 = (1/1)%2 = 1 → [0, 1]
        let mut bits_1: Array<u32> = array![];
        bi = selector_bits;
        loop {
            if bi == 0 { break; }
            bi -= 1;
            bits_1.append((1_u32 / pow2(bi)) % 2);
        };
        assert!(*bits_1.at(0) == 0 && *bits_1.at(1) == 1, "claim 1 = [0,1]");

        // claim 2: bit1 = (2/2)%2 = 1, bit0 = (2/1)%2 = 0 → [1, 0]
        let mut bits_2: Array<u32> = array![];
        bi = selector_bits;
        loop {
            if bi == 0 { break; }
            bi -= 1;
            bits_2.append((2_u32 / pow2(bi)) % 2);
        };
        assert!(*bits_2.at(0) == 1 && *bits_2.at(1) == 0, "claim 2 = [1,0]");

        // claim 3: bit1 = (3/2)%2 = 1, bit0 = (3/1)%2 = 1 → [1, 1]
        let mut bits_3: Array<u32> = array![];
        bi = selector_bits;
        loop {
            if bi == 0 { break; }
            bi -= 1;
            bits_3.append((3_u32 / pow2(bi)) % 2);
        };
        assert!(*bits_3.at(0) == 1 && *bits_3.at(1) == 1, "claim 3 = [1,1]");
    }

    #[test]
    fn test_verify_aggregated_binding_rejects_tampered_super_root() {
        // Construct a proof with mismatched super_root.
        // verify_aggregated_binding should return false at step 1 (verify_super_root).
        let a: felt252 = 0x1111;
        let b: felt252 = 0x2222;
        let _correct_root = poseidon_hash_2(a, b);
        let wrong_root: felt252 = 0xDEAD;

        let config = AggregatedBindingConfig {
            selector_bits: 1,
            n_max: 2,
            m_padded: 2,
            n_global: 3,
            n_claims: 1,
        };

        let proof = AggregatedWeightBindingProof {
            config,
            round_polys: array![
                (qm31_zero(), qm31_zero(), qm31_zero()),
                (qm31_zero(), qm31_zero(), qm31_zero()),
                (qm31_zero(), qm31_zero(), qm31_zero()),
            ],
            oracle_eval_at_s: qm31_zero(),
            super_root: wrong_root,          // <-- does NOT match subtree_roots
            subtree_roots: array![a, b],
            opening_proof: MleOpeningProof {
                intermediate_roots: array![],
                queries: array![],
                final_value: qm31_zero(),
            },
        };

        let claims: Array<WeightClaimData> = array![
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
        ];
        let commitments: Array<felt252> = array![a];

        let mut ch = channel_default();
        let result = verify_aggregated_binding(
            @proof, claims.span(), commitments.span(), ref ch,
        );
        assert!(!result, "tampered super_root should fail verification");
    }

    #[test]
    fn test_verify_aggregated_binding_rejects_tampered_subtree_root() {
        // Proof has a correct super_root but subtree_roots don't match weight commitments.
        // Should fail at step 1b (verify_subtree_commitments).
        let real_commitment: felt252 = 0x1111;
        let fake_subtree: felt252 = 0x9999;
        let zero_root = compute_zero_tree_root(2);

        // Build super_root from the fake subtree roots
        let super_root = poseidon_hash_2(fake_subtree, zero_root);

        let config = AggregatedBindingConfig {
            selector_bits: 1,
            n_max: 2,
            m_padded: 2,
            n_global: 3,
            n_claims: 1,
        };

        let proof = AggregatedWeightBindingProof {
            config,
            round_polys: array![
                (qm31_zero(), qm31_zero(), qm31_zero()),
                (qm31_zero(), qm31_zero(), qm31_zero()),
                (qm31_zero(), qm31_zero(), qm31_zero()),
            ],
            oracle_eval_at_s: qm31_zero(),
            super_root,                        // matches subtree_roots...
            subtree_roots: array![fake_subtree, zero_root],  // ...but doesn't match weight commitment
            opening_proof: MleOpeningProof {
                intermediate_roots: array![],
                queries: array![],
                final_value: qm31_zero(),
            },
        };

        let claims: Array<WeightClaimData> = array![
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
        ];
        let commitments: Array<felt252> = array![real_commitment]; // doesn't match fake_subtree

        let mut ch = channel_default();
        let result = verify_aggregated_binding(
            @proof, claims.span(), commitments.span(), ref ch,
        );
        assert!(!result, "tampered subtree root should fail at commitment check");
    }

    #[test]
    fn test_verify_aggregated_binding_rejects_wrong_round_count() {
        // Proof has correct super-root and subtree bindings, but wrong number of sumcheck rounds.
        // Should fail at step 4 (round_polys.len != n_global).
        let commitment: felt252 = 0x1111;
        let zero_root = compute_zero_tree_root(2);
        // commitment has eval_point len=2 == n_max=2 → no extension needed
        let super_root = poseidon_hash_2(commitment, zero_root);

        let config = AggregatedBindingConfig {
            selector_bits: 1,
            n_max: 2,
            m_padded: 2,
            n_global: 3,   // expects 3 rounds
            n_claims: 1,
        };

        let proof = AggregatedWeightBindingProof {
            config,
            round_polys: array![
                (qm31_zero(), qm31_zero(), qm31_zero()),
                // Only 1 round instead of 3
            ],
            oracle_eval_at_s: qm31_zero(),
            super_root,
            subtree_roots: array![commitment, zero_root],
            opening_proof: MleOpeningProof {
                intermediate_roots: array![],
                queries: array![],
                final_value: qm31_zero(),
            },
        };

        let claims: Array<WeightClaimData> = array![
            WeightClaimData {
                eval_point: array![qm31_zero(), qm31_zero()],
                expected_value: qm31_zero(),
            },
        ];
        let commitments: Array<felt252> = array![commitment];

        let mut ch = channel_default();
        let result = verify_aggregated_binding(
            @proof, claims.span(), commitments.span(), ref ch,
        );
        assert!(!result, "wrong round count should fail");
    }
}
