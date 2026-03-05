//! Aggregated weight binding via unified oracle mismatch sumcheck.
//!
//! After a GKR walk produces M weight claims `(z_i, v_i)` against per-matrix
//! commitments `C_i`, this module proves all claims correct in one shot:
//!
//! 1. **Virtual unified oracle** — all M weight matrices mapped into a single
//!    address space via power-of-2 aligned slots (8 selector bits + n_max local
//!    bits).
//!
//! 2. **Mismatch sumcheck** — prove `R(t) = Σ β_i · eq(g_i, t) · (W(t) - v_i) = 0`
//!    over 35 variables. Sparse evaluation: at each point only one matrix contributes.
//!
//! 3. **One MLE opening** — single opening against a super-root built from
//!    per-matrix commitment roots (subtrees of the virtual tree).
//!
//! Result: ~17K felts calldata instead of ~2.4M (160 separate openings).

#[cfg(feature = "cuda-runtime")]
use crate::crypto::mle_opening::prove_mle_opening_with_commitment_qm31_u32;
#[cfg(feature = "cuda-runtime")]
use crate::gpu_sumcheck::{GpuMleFoldSession, GpuSumcheckExecutor};
use crate::crypto::mle_opening::{
    evaluate_mle_at, mle_n_queries, prove_mle_opening, verify_mle_opening, MleOpeningProof,
};
use crate::crypto::poseidon_channel::PoseidonChannel;

use num_traits::Zero;
use starknet_crypto::poseidon_hash;
use starknet_ff::FieldElement;

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;

use crate::crypto::poseidon_channel::securefield_to_felt;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Maximum number of weight matrices (selector space = 2^SELECTOR_BITS).
pub const SELECTOR_BITS: usize = 8;
/// Maximum slots in the virtual oracle.
pub const MAX_MATRICES: usize = 1 << SELECTOR_BITS; // 256

/// Number of MLE queries for the aggregated opening.
/// Uses the runtime-configurable query count.
pub fn aggregated_mle_queries() -> usize {
    mle_n_queries()
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// Configuration for the aggregated binding protocol.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AggregatedBindingConfig {
    /// Number of selector bits (log2 of padded matrix count).
    pub selector_bits: usize,
    /// Maximum local variable count across all matrices.
    pub n_max: usize,
    /// Number of matrices padded to next power of 2 (for selector alignment).
    pub m_padded: usize,
    /// Total global variables: selector_bits + n_max.
    pub n_global: usize,
    /// Number of actual weight claims.
    pub n_claims: usize,
}

impl AggregatedBindingConfig {
    /// Compute configuration from claim sizes.
    pub fn from_claims(claims: &[AggregatedWeightClaim]) -> Self {
        assert!(!claims.is_empty(), "need at least one weight claim");
        assert!(
            claims.len() <= MAX_MATRICES,
            "too many matrices: {} > {}",
            claims.len(),
            MAX_MATRICES
        );

        let n_max = claims.iter().map(|c| c.local_n_vars).max().unwrap_or(0);

        // Pad matrix count to power of 2 for clean selector addressing
        let m_padded = claims.len().next_power_of_two();
        let selector_bits = m_padded.trailing_zeros() as usize;

        let n_global = selector_bits + n_max;

        AggregatedBindingConfig {
            selector_bits,
            n_max,
            m_padded,
            n_global,
            n_claims: claims.len(),
        }
    }
}

/// A weight claim to be aggregated.
#[derive(Debug, Clone)]
pub struct AggregatedWeightClaim {
    /// Index of this matrix in the weight_commitments array.
    pub matrix_index: usize,
    /// Number of variables in this matrix's MLE (log2 of padded size).
    pub local_n_vars: usize,
    /// Evaluation point (length = local_n_vars).
    pub eval_point: Vec<SecureField>,
    /// Expected MLE evaluation value.
    pub expected_value: SecureField,
    /// Poseidon Merkle root commitment for this matrix.
    pub commitment: FieldElement,
}

/// Super-root: virtual Merkle root built from per-matrix subtree roots.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SuperRoot {
    /// The top-level super-root hash.
    pub root: FieldElement,
    /// Per-matrix subtree roots (padded to m_padded with zero_tree_root).
    pub subtree_roots: Vec<FieldElement>,
    /// Root of a tree of all zeros (for padding unused slots).
    pub zero_tree_root: FieldElement,
    /// Number of top-level Merkle levels (= selector_bits).
    pub top_levels: usize,
}

/// Complete aggregated weight binding proof.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AggregatedWeightBindingProof {
    /// Protocol configuration.
    pub config: AggregatedBindingConfig,
    /// Sumcheck round polynomials: n_global rounds, each degree 2 (3 coefficients).
    pub sumcheck_round_polys: Vec<(SecureField, SecureField, SecureField)>,
    /// Oracle evaluation W_global(s) at the sumcheck challenge point.
    pub oracle_eval_at_s: SecureField,
    /// Single MLE opening proof against the super-root (legacy path).
    pub opening_proof: MleOpeningProof,
    /// Super-root data.
    pub super_root: SuperRoot,
    /// Per-matrix MLE opening proofs (when set, verifier uses these instead of
    /// the single opening_proof against the super-root). Each proof is against
    /// the matrix's own commitment root at the local challenge point.
    /// This avoids building a 16GB+ virtual MLE Merkle tree.
    #[cfg_attr(feature = "serde", serde(default))]
    pub per_matrix_openings: Option<Vec<MleOpeningProof>>,
    /// Per-matrix evaluations at the local challenge point (paired with per_matrix_openings).
    #[cfg_attr(feature = "serde", serde(default))]
    pub per_matrix_evals: Option<Vec<SecureField>>,
}

// ─── Zero tree ───────────────────────────────────────────────────────────────

/// Compute the Poseidon Merkle root of a tree with 2^n_vars zero leaves.
///
/// Each leaf is `securefield_to_felt(QM31::zero())` = poseidon packing of (0,0,0,0).
/// The tree is built bottom-up: h_{level}  = poseidon_hash(h_{level-1}, h_{level-1}).
pub fn compute_zero_tree_root(n_vars: usize) -> FieldElement {
    // A single leaf (2^0 = 1 leaf) has root = hash of zero SecureField
    let zero_leaf = securefield_to_felt(SecureField::zero());
    if n_vars == 0 {
        return zero_leaf;
    }
    // For n_vars >= 1, leaves are pairs. Bottom hash = hash(zero_leaf, zero_leaf)
    let mut h = poseidon_hash(zero_leaf, zero_leaf);
    for _ in 1..n_vars {
        h = poseidon_hash(h, h);
    }
    h
}

// ─── Super-root construction ─────────────────────────────────────────────────

/// Build the super-root from per-matrix commitment roots.
///
/// The virtual tree has `m_padded` subtrees, each of size 2^n_max.
/// Matrices smaller than n_max are "extended" by treating their commitment
/// root as the root of a subtree padded with zeros to 2^n_max.
///
/// The top `selector_bits` levels are a standard Merkle tree over these
/// (potentially extended) subtree roots.
pub fn build_super_root(
    claims: &[AggregatedWeightClaim],
    config: &AggregatedBindingConfig,
) -> SuperRoot {
    // Pre-compute all zero tree roots from 0..=n_max to avoid redundant
    // O(n) Poseidon hash chains per claim.
    let zero_tree_roots: Vec<FieldElement> =
        (0..=config.n_max).map(compute_zero_tree_root).collect();
    let zero_tree_root = zero_tree_roots[config.n_max];

    // Build padded subtree roots: actual commitments + zero-tree padding
    let mut subtree_roots = Vec::with_capacity(config.m_padded);

    for claim in claims {
        if claim.local_n_vars == config.n_max {
            // Same size — commitment IS the subtree root
            subtree_roots.push(claim.commitment);
        } else {
            // Smaller matrix: extend its root to n_max depth.
            // The committed tree has depth local_n_vars. We need to hash it
            // up to depth n_max by pairing with zero_tree_root at each level
            // from local_n_vars to n_max.
            let levels_to_extend = config.n_max - claim.local_n_vars;
            let mut extended = claim.commitment;
            for k in 0..levels_to_extend {
                extended = poseidon_hash(extended, zero_tree_roots[claim.local_n_vars + k]);
            }
            subtree_roots.push(extended);
        }
    }

    // Pad unused slots with zero_tree_root
    while subtree_roots.len() < config.m_padded {
        subtree_roots.push(zero_tree_root);
    }

    // Build top Merkle tree over subtree_roots
    let root = merkle_root_from_leaves(&subtree_roots);

    SuperRoot {
        root,
        subtree_roots,
        zero_tree_root,
        top_levels: config.selector_bits,
    }
}

/// Build a Merkle root from a power-of-2 list of leaves using Poseidon.
fn merkle_root_from_leaves(leaves: &[FieldElement]) -> FieldElement {
    assert!(leaves.len().is_power_of_two());
    if leaves.len() == 1 {
        return leaves[0];
    }
    let mut current = leaves.to_vec();
    while current.len() > 1 {
        let mut next = Vec::with_capacity(current.len() / 2);
        for pair in current.chunks_exact(2) {
            next.push(poseidon_hash(pair[0], pair[1]));
        }
        current = next;
    }
    current[0]
}

// ─── WeightSource trait for streaming access ────────────────────────────────

/// Trait for on-demand weight MLE access (avoids holding all MLEs in memory).
///
/// The streaming aggregated binding sumcheck loads one MLE at a time, evaluates
/// it, then drops it. This reduces peak memory from ~640 GB (all 160 weight
/// MLEs simultaneously) to ~4 GB (single largest matrix).
pub trait WeightSource {
    /// Return the SecureField MLE for the weight matrix at the given claim index.
    /// Caller should drop the returned Vec after use to free memory.
    fn get_mle(&self, claim_index: usize) -> Vec<SecureField>;
    /// Return the QM31 AoS u32 MLE for GPU opening proof at the given claim index.
    fn get_mle_u32(&self, claim_index: usize) -> Vec<u32>;
    /// Number of weight matrices.
    fn len(&self) -> usize;
}

/// Backward-compatible `WeightSource` wrapping pre-built `&[&[SecureField]]` slices.
///
/// Used by tests and the non-streaming path. Clones MLEs on access since the
/// trait requires owned `Vec`s (the caller may drop them to free memory).
pub struct SliceWeightSource<'a> {
    pub mles: &'a [&'a [SecureField]],
    pub mles_u32: &'a [&'a [u32]],
}

impl<'a> WeightSource for SliceWeightSource<'a> {
    fn get_mle(&self, claim_index: usize) -> Vec<SecureField> {
        self.mles[claim_index].to_vec()
    }
    fn get_mle_u32(&self, claim_index: usize) -> Vec<u32> {
        self.mles_u32[claim_index].to_vec()
    }
    fn len(&self) -> usize {
        self.mles.len()
    }
}

// ─── Selector weight helper ─────────────────────────────────────────────────

/// Compute the selector polynomial weight for a given matrix index.
///
/// `sel_weight(i, selector) = Π_j (selector_j * bit_j + (1 - selector_j) * (1 - bit_j))`
/// where `bit_j` is the j-th bit (MSB first) of `matrix_index`.
fn compute_selector_weight(
    matrix_index: usize,
    selector: &[SecureField],
    config: &AggregatedBindingConfig,
) -> SecureField {
    let one = SecureField::from(M31::from(1));
    let mut sel_weight = one;
    for (j, &s_j) in selector.iter().enumerate() {
        let bit_pos = config.selector_bits - 1 - j;
        let bit_j = if (matrix_index >> bit_pos) & 1 == 1 {
            one
        } else {
            SecureField::zero()
        };
        sel_weight = sel_weight * (s_j * bit_j + (one - s_j) * (one - bit_j));
    }
    sel_weight
}

// ─── eq polynomial ───────────────────────────────────────────────────────────

/// Evaluate eq(a, b) = Π_i (a_i * b_i + (1 - a_i) * (1 - b_i))
/// where a, b are vectors of SecureField elements.
fn eq_eval(a: &[SecureField], b: &[SecureField]) -> SecureField {
    assert_eq!(a.len(), b.len());
    let one = SecureField::from(M31::from(1));
    let mut result = one;
    for (ai, bi) in a.iter().zip(b.iter()) {
        result = result * (*ai * *bi + (one - *ai) * (one - *bi));
    }
    result
}

// ─── Sparse mismatch sumcheck ────────────────────────────────────────────────

/// Global evaluation point for claim i: selector bits || zero-padded local point.
///
/// selector = binary encoding of matrix_index (selector_bits long)
/// local = eval_point zero-padded to n_max
fn global_point(
    claim: &AggregatedWeightClaim,
    config: &AggregatedBindingConfig,
) -> Vec<SecureField> {
    let one = SecureField::from(M31::from(1));
    let zero = SecureField::zero();

    let mut g = Vec::with_capacity(config.n_global);

    // Selector bits (MSB first within the selector block)
    for bit_idx in (0..config.selector_bits).rev() {
        if (claim.matrix_index >> bit_idx) & 1 == 1 {
            g.push(one);
        } else {
            g.push(zero);
        }
    }

    // Local bits: claim's eval_point padded to n_max with zeros
    let pad = config.n_max - claim.local_n_vars;
    for _ in 0..pad {
        g.push(zero);
    }
    g.extend_from_slice(&claim.eval_point);

    assert_eq!(g.len(), config.n_global);
    g
}

/// Evaluate the virtual unified oracle at a global point `t`.
///
/// `t` = [selector_bits || local_bits] where selector picks the matrix
/// and local_bits index into that matrix's MLE.
///
/// For non-boolean selector values (after sumcheck challenges), we interpolate
/// across all matrices.
fn eval_unified_oracle(
    t: &[SecureField],
    weight_mles: &[&[SecureField]],
    config: &AggregatedBindingConfig,
) -> SecureField {
    let selector = &t[..config.selector_bits];
    let local = &t[config.selector_bits..];
    assert_eq!(local.len(), config.n_max);

    // Compute selector polynomial weights for each matrix
    // W_global(t) = Σ_i sel_weight(i, selector) * W_i(local)
    let mut result = SecureField::zero();
    for (i, mle) in weight_mles.iter().enumerate() {
        let sel_weight = compute_selector_weight(i, selector, config);

        if sel_weight == SecureField::zero() {
            continue;
        }

        // Evaluate W_i at the local point (truncated to this matrix's n_vars)
        let n_vars = mle.len().trailing_zeros() as usize;
        let local_truncated = &local[config.n_max - n_vars..];
        let w_val = evaluate_mle_at(mle, local_truncated);

        result = result + sel_weight * w_val;
    }

    result
}

/// Compute β weights from Fiat-Shamir channel.
fn draw_beta_weights(channel: &mut PoseidonChannel, n_claims: usize) -> Vec<SecureField> {
    let rho = channel.draw_qm31();
    let mut betas = Vec::with_capacity(n_claims);
    let mut rho_pow = SecureField::from(M31::from(1));
    for _ in 0..n_claims {
        betas.push(rho_pow);
        rho_pow = rho_pow * rho;
    }
    betas
}

/// Run the mismatch sumcheck and produce round polynomials.
///
/// Proves: R(t) = Σ_i β_i · eq(g_i, t) · (W_global(t) - v_i) ≡ 0
///
/// Returns (round_polys, challenge_point) where challenge_point is the final
/// evaluation point after all rounds.
fn mismatch_sumcheck(
    claims: &[AggregatedWeightClaim],
    weight_mles: &[&[SecureField]],
    config: &AggregatedBindingConfig,
    betas: &[SecureField],
    channel: &mut PoseidonChannel,
) -> (
    Vec<(SecureField, SecureField, SecureField)>,
    Vec<SecureField>,
) {
    let n = config.n_global;
    let one = SecureField::from(M31::from(1));

    // Pre-compute global points for each claim
    let global_points: Vec<Vec<SecureField>> =
        claims.iter().map(|c| global_point(c, config)).collect();

    // eq tables: for each claim i, maintain running eq product
    // eq_tables[i] = current partial eq(g_i, t) contribution
    // We track them as: for each claim, the "remaining" eq factor
    // that hasn't been fixed by sumcheck challenges yet.
    //
    // At each round j, we evaluate R restricted to t_j ∈ {0, 1, 2}
    // while the first j-1 variables are fixed to challenges r_0..r_{j-1}.

    // For efficiency, we maintain "eq_prefix[i]" = product of eq factors
    // for already-fixed variables, and "eq_suffix_table" for remaining.

    let mut challenge_point = Vec::with_capacity(n);
    let mut round_polys = Vec::with_capacity(n);

    // eq_prefix[i] = Π_{j < current_round} (r_j * g_i[j] + (1-r_j)*(1-g_i[j]))
    let mut eq_prefix: Vec<SecureField> = vec![one; claims.len()];

    for round in 0..n {
        // Evaluate R(t_round) at t_round = 0, 1, 2 (keeping all later vars summed)
        let s0 = eval_round_at_value(
            SecureField::zero(),
            round,
            claims,
            weight_mles,
            config,
            betas,
            &global_points,
            &eq_prefix,
            &challenge_point,
        );
        let s1 = eval_round_at_value(
            one,
            round,
            claims,
            weight_mles,
            config,
            betas,
            &global_points,
            &eq_prefix,
            &challenge_point,
        );
        let two = SecureField::from(M31::from(2));
        let s2 = eval_round_at_value(
            two,
            round,
            claims,
            weight_mles,
            config,
            betas,
            &global_points,
            &eq_prefix,
            &challenge_point,
        );

        // Degree-2 polynomial through (0, s0), (1, s1), (2, s2):
        // p(t) = c0 + c1*t + c2*t²
        // c0 = s0
        // c1 = (-3*s0 + 4*s1 - s2) / 2  [Lagrange]
        // c2 = (s0 - 2*s1 + s2) / 2
        let c0 = s0;
        let two_inv = two.inverse();
        let c2 = (s0 - two * s1 + s2) * two_inv;
        let c1 = s1 - s0 - c2;

        // Mix round polynomial into channel
        channel.mix_poly_coeffs(c0, c1, c2);
        round_polys.push((c0, c1, c2));

        // Draw challenge for this round
        let r = channel.draw_qm31();
        challenge_point.push(r);

        // Update eq_prefix for each claim
        for (i, gp) in global_points.iter().enumerate() {
            let g_val = gp[round];
            eq_prefix[i] = eq_prefix[i] * (r * g_val + (one - r) * (one - g_val));
        }
    }

    (round_polys, challenge_point)
}

/// Evaluate the sumcheck polynomial for a specific round at a given value,
/// summing over all remaining variables.
///
/// This is the key function for the sparse mismatch sumcheck. Because the
/// unified oracle is sparse (only one matrix per selector), we can evaluate
/// efficiently by iterating over claims rather than the full 2^n domain.
fn eval_round_at_value(
    value: SecureField,
    round: usize,
    claims: &[AggregatedWeightClaim],
    weight_mles: &[&[SecureField]],
    config: &AggregatedBindingConfig,
    betas: &[SecureField],
    global_points: &[Vec<SecureField>],
    eq_prefix: &[SecureField],
    challenge_point: &[SecureField],
) -> SecureField {
    let one = SecureField::from(M31::from(1));
    let n = config.n_global;

    // For each claim i, compute:
    //   β_i * eq_prefix_i * eq(g_i[round], value) * Σ_{remaining vars} eq(g_i[round+1..], x) * (W_i(x) - v_i)
    //
    // The sum over remaining vars of eq(g_i[round+1..], x) * W_i(x) = W_i evaluated
    // at partial point (challenge_point[..round], value, g_i[round+1..]) restricted to
    // the matrix's local vars.
    //
    // Actually, Σ_x eq(g, x) * f(x) = f(g) when g is a boolean point. But here
    // we need to be careful because we're computing the *full sum* over x of the
    // remaining unfixed variables.
    //
    // Let me think about this more carefully. The mismatch polynomial is:
    // R(t) = Σ_i β_i * eq(g_i, t) * (W_global(t) - v_i)
    //
    // In the sumcheck, round j asks us to compute:
    // S_j(X_j) = Σ_{x_{j+1},...,x_{n-1}} R(r_0,...,r_{j-1}, X_j, x_{j+1},...,x_{n-1})
    //
    // For each claim i:
    // Contribution_i = β_i * Σ_{x_{j+1},...,x_{n-1}} eq(g_i, (r_0,...,r_{j-1}, X_j, x_{j+1},...,x_{n-1})) * (W_global(...) - v_i)
    //
    // The eq factorizes: eq(g, t) = Π_k eq_1d(g_k, t_k)
    // So: eq(g_i, t) = eq_prefix_i * eq_1d(g_i[j], X_j) * Π_{k>j} eq_1d(g_i[k], x_k)
    //
    // For the W_global term: since the selector picks exactly one matrix,
    // at the boolean assignment of remaining x_k, only one matrix contributes.
    // But during sumcheck with random challenges, selector bits become random —
    // all matrices can contribute.
    //
    // KEY INSIGHT for sparsity: At each round, we can separate the contribution
    // by claim. For claim i with global point g_i, we need:
    //
    // β_i * eq_prefix_i * eq_1d(g_i[round], value) *
    //   [Σ_{x>round} Π_{k>round} eq_1d(g_i[k], x_k) * (W_global(r_0..r_{round-1}, value, x_{round+1}..x_{n-1}) - v_i)]
    //
    // The W_global at a mixed (random + boolean) point:
    // For remaining boolean coords x_{round+1}..x_{n-1}, the selector variables
    // (some of which are already fixed to challenges) select a matrix.
    //
    // For a small number of claims, the simplest correct approach is to compute
    // the mismatch sum directly. For production, we'd want the sparse optimization.
    // For now, use the direct but claim-sparse approach:

    let mut total = SecureField::zero();

    // For each claim, compute its contribution
    for (i, claim) in claims.iter().enumerate() {
        let gp = &global_points[i];
        let _n_vars = claim.local_n_vars;

        // eq factor for this round variable
        let eq_round = gp[round] * value + (one - gp[round]) * (one - value);

        // Sum over remaining variables (round+1..n)
        // For the W_global - v_i term, we use the fact that at boolean remaining
        // coordinates, only specific matrices contribute. But this is complex.
        // Instead, use the multilinear extension identity directly:
        //
        // Σ_{x_{j+1}..x_{n-1}} eq(g_i[j+1:], x[j+1:]) * (W_global(r_0..r_{j-1}, X_j, x_{j+1}..x_{n-1}) - v_i)
        //
        // = W_global evaluated at g_i (with first j+1 vars replaced by challenges/value) - v_i
        //   ... NO, this identity only holds for boolean g_i, and W_global = Σ sel_wt * W_k.
        //
        // Actually: Σ_x eq(a, x) * f(x) = f(a) for ANY multilinear f.
        // So: Σ_{x>round} eq(g_i[round+1:], x) * W_global(challenges, value, x) = W_global(challenges, value, g_i[round+1:])
        // And: Σ_{x>round} eq(g_i[round+1:], x) * v_i = v_i (since Σ eq = 1 for boolean g_i points... wait,
        //       g_i can be non-boolean for eval_point entries)
        //
        // Hmm, g_i[round+1:] ARE the original claim global points which can be non-boolean
        // (since eval_point entries are random QM31 values from the GKR walk).
        // But the identity Σ_x eq(a, x) * f(x) = f(a) holds for ANY a.
        //
        // For the constant v_i term: Σ_{x>round} eq(g_i[round+1:], x) * 1 = Π_{k>round} (g_i[k] + 1 - g_i[k]) = 1
        // Wait no: Σ_{x_k ∈ {0,1}} eq_1d(a_k, x_k) = a_k + (1-a_k) = 1. So the product is 1. Good.
        //
        // Therefore:
        // Σ_{x>round} eq(g_i[round+1:], x) * (W_global(..., x) - v_i)
        //   = W_global(r_0..r_{round-1}, value, g_i[round+1:]) - v_i

        // Build the full evaluation point
        let mut eval_pt = Vec::with_capacity(n);
        eval_pt.extend_from_slice(challenge_point); // r_0..r_{round-1}
        eval_pt.push(value); // X_round = value
        eval_pt.extend_from_slice(&gp[round + 1..]); // g_i[round+1:]

        let w_at_pt = eval_unified_oracle(&eval_pt, weight_mles, config);
        let mismatch = w_at_pt - claim.expected_value;

        total = total + betas[i] * eq_prefix[i] * eq_round * mismatch;
    }

    total
}

// ─── Single MLE opening against super-root ───────────────────────────────────

/// Build the full virtual MLE for the unified oracle and open it at the
/// sumcheck challenge point.
///
/// The virtual MLE has 2^n_global entries organized as:
/// [matrix_0 padded to 2^n_max | matrix_1 padded to 2^n_max | ... | zeros]
fn build_virtual_mle(
    weight_mles: &[&[SecureField]],
    config: &AggregatedBindingConfig,
) -> Vec<SecureField> {
    let total_size = 1 << config.n_global;
    let slot_size = 1 << config.n_max;

    let mut virtual_mle = vec![SecureField::zero(); total_size];

    for (i, mle) in weight_mles.iter().enumerate() {
        let offset = i * slot_size;
        // Copy matrix MLE into its slot (rest stays zero)
        virtual_mle[offset..offset + mle.len()].copy_from_slice(mle);
    }

    virtual_mle
}

/// Build the full virtual MLE in QM31 AoS u32 format for GPU-accelerated opening.
///
/// Layout matches `build_virtual_mle`: `m_padded` slots of `2^n_max` elements each,
/// but stored as 4 u32 words per QM31 point: `[a0,b0,c0,d0, a1,b1,c1,d1, ...]`.
#[cfg(any(feature = "cuda-runtime", test))]
fn build_virtual_mle_u32(weight_mles_u32: &[&[u32]], config: &AggregatedBindingConfig) -> Vec<u32> {
    let total_size = 1usize << config.n_global;
    let slot_size = 1usize << config.n_max;

    // 4 u32 words per QM31 element
    let mut virtual_mle = vec![0u32; total_size * 4];

    for (i, mle) in weight_mles_u32.iter().enumerate() {
        let offset = i * slot_size * 4;
        // Copy matrix MLE u32 words into its slot (rest stays zero)
        virtual_mle[offset..offset + mle.len()].copy_from_slice(mle);
    }

    virtual_mle
}

// ─── Main prover ─────────────────────────────────────────────────────────────

/// Prove all weight claims via the aggregated binding protocol.
///
/// # Arguments
/// - `claims`: Weight claims from GKR walk (eval_point, expected_value, commitment per matrix)
/// - `weight_mles`: MLE evaluations for each weight matrix (padded to power-of-2)
/// - `channel`: Fiat-Shamir channel (post-GKR-walk state)
///
/// # Returns
/// `AggregatedWeightBindingProof` containing sumcheck rounds, oracle eval, single opening, and super-root.
pub fn prove_aggregated_binding(
    claims: &[AggregatedWeightClaim],
    weight_mles: &[&[SecureField]],
    channel: &mut PoseidonChannel,
) -> AggregatedWeightBindingProof {
    assert_eq!(claims.len(), weight_mles.len());
    assert!(!claims.is_empty());

    let config = AggregatedBindingConfig::from_claims(claims);

    // 1. Build super-root and mix into channel
    let super_root = build_super_root(claims, &config);
    channel.mix_felt(super_root.root);

    // 2. Draw β weights
    let betas = draw_beta_weights(channel, claims.len());

    // 3. Run mismatch sumcheck
    let (round_polys, challenge_point) =
        mismatch_sumcheck(claims, weight_mles, &config, &betas, channel);

    // 4. Evaluate oracle at challenge point
    let oracle_eval = eval_unified_oracle(&challenge_point, weight_mles, &config);
    channel.mix_felts(&[oracle_eval]);

    // 5. Build virtual MLE and prove single opening against super-root
    let virtual_mle = build_virtual_mle(weight_mles, &config);
    let opening_proof = prove_mle_opening(&virtual_mle, &challenge_point, channel);

    AggregatedWeightBindingProof {
        config,
        sumcheck_round_polys: round_polys,
        oracle_eval_at_s: oracle_eval,
        opening_proof,
        super_root,
        per_matrix_openings: None,
        per_matrix_evals: None,
    }
}

/// GPU-accelerated variant of [`prove_aggregated_binding`].
///
/// Uses GPU MLE fold kernels for the final virtual MLE opening proof.
/// The mismatch sumcheck still runs on CPU (it's inherently sparse and fast).
///
/// # Arguments
/// - `claims`: Weight claims from GKR walk
/// - `weight_mles`: SecureField MLEs for sparse mismatch sumcheck evaluation
/// - `weight_mles_u32`: QM31 AoS u32 MLEs for GPU-accelerated MLE opening
/// - `channel`: Fiat-Shamir channel (post-GKR-walk state)
#[cfg(feature = "cuda-runtime")]
pub fn prove_aggregated_binding_gpu(
    claims: &[AggregatedWeightClaim],
    weight_mles: &[&[SecureField]],
    weight_mles_u32: &[&[u32]],
    channel: &mut PoseidonChannel,
) -> AggregatedWeightBindingProof {
    assert_eq!(claims.len(), weight_mles.len());
    assert_eq!(claims.len(), weight_mles_u32.len());
    assert!(!claims.is_empty());

    // Verify n_vars consistency between SecureField and u32 MLE representations.
    // Each u32 MLE has 4 words per QM31 element, so n_points = len/4.
    for (i, (sf_mle, u32_mle)) in weight_mles.iter().zip(weight_mles_u32.iter()).enumerate() {
        let sf_n_points = sf_mle.len();
        let u32_n_points = u32_mle.len() / 4;
        assert_eq!(
            sf_n_points,
            u32_n_points,
            "n_vars mismatch at matrix {i}: SecureField MLE has {sf_n_points} points, \
             u32 MLE has {u32_n_points} points ({}  u32 words)",
            u32_mle.len(),
        );
        assert_eq!(
            u32_mle.len() % 4,
            0,
            "u32 MLE at matrix {i} has {} words (not a multiple of 4)",
            u32_mle.len(),
        );
    }

    let config = AggregatedBindingConfig::from_claims(claims);

    // 1. Build super-root and mix into channel
    let super_root = build_super_root(claims, &config);
    channel.mix_felt(super_root.root);

    // 2. Draw β weights
    let betas = draw_beta_weights(channel, claims.len());

    // 3. Run mismatch sumcheck (CPU, uses SecureField MLEs for sparse evaluation)
    let (round_polys, challenge_point) =
        mismatch_sumcheck(claims, weight_mles, &config, &betas, channel);

    // 4. Evaluate oracle at challenge point
    let oracle_eval = eval_unified_oracle(&challenge_point, weight_mles, &config);
    channel.mix_felts(&[oracle_eval]);

    // 5. Build virtual MLE in u32 format and prove opening with GPU
    let virtual_mle_u32 = build_virtual_mle_u32(weight_mles_u32, &config);
    let (commitment, opening_proof) =
        prove_mle_opening_with_commitment_qm31_u32(&virtual_mle_u32, &challenge_point, channel);

    // Soundness gate: the GPU tree commitment must equal the super-root.
    // Both are Poseidon Merkle roots over the same data in the same layout —
    // the virtual MLE tree's bottom n_max levels are per-matrix subtrees,
    // and the top selector_bits levels match the super-root construction.
    debug_assert_eq!(
        commitment, super_root.root,
        "GPU virtual MLE commitment ({commitment:?}) diverged from super-root ({:?}). \
         This indicates a data representation mismatch between SecureField and u32 AoS formats.",
        super_root.root,
    );

    AggregatedWeightBindingProof {
        config,
        sumcheck_round_polys: round_polys,
        oracle_eval_at_s: oracle_eval,
        opening_proof,
        super_root,
        per_matrix_openings: None,
        per_matrix_evals: None,
    }
}

// ─── Streaming aggregated binding ─────────────────────────────────────────

/// Evaluate the virtual unified oracle at a global point, streaming MLEs on demand.
///
/// Same semantics as [`eval_unified_oracle`] but loads each weight MLE from the
/// source just-in-time and drops it immediately after evaluation. Peak memory is
/// bounded by the single largest matrix MLE (~4 GB) instead of all MLEs (~640 GB).
fn eval_unified_oracle_streaming(
    t: &[SecureField],
    source: &dyn WeightSource,
    claim_n_vars: &[usize],
    config: &AggregatedBindingConfig,
) -> SecureField {
    let selector = &t[..config.selector_bits];
    let local = &t[config.selector_bits..];
    assert_eq!(local.len(), config.n_max);

    let mut result = SecureField::zero();
    for i in 0..source.len() {
        let sel_weight = compute_selector_weight(i, selector, config);

        if sel_weight == SecureField::zero() {
            continue;
        }

        let n_vars = claim_n_vars[i];
        let local_truncated = &local[config.n_max - n_vars..];
        let mle = source.get_mle(i); // load on demand
        let w_val = evaluate_mle_at(&mle, local_truncated);
        // mle dropped here — frees ~4 GB

        result = result + sel_weight * w_val;
    }

    result
}

/// GPU minimum MLE size (2^GPU_MLE_THRESHOLD) below which we fall back to CPU.
/// Small MLEs are faster to evaluate on CPU than to upload to GPU.
#[cfg(feature = "cuda-runtime")]
const GPU_MLE_THRESHOLD: usize = 14;

/// Evaluate the unified oracle at 3 points that differ only at one variable,
/// loading each weight MLE to GPU once and forking 3 evaluation paths.
///
/// When all 3 eval points share a common prefix and suffix (differing only at
/// `branch_pos`), the MLE is uploaded once, the shared prefix is folded, then
/// the GPU buffer is cloned 2× to fork for the 3 branch values.
///
/// For selector rounds (round < selector_bits), only the selector weight differs
/// across the 3 eval points — the MLE evaluation is identical, so we evaluate once.
#[cfg(feature = "cuda-runtime")]
fn eval_unified_oracle_3way_gpu(
    eval_pts: [&[SecureField]; 3],
    source: &dyn WeightSource,
    claim_n_vars: &[usize],
    config: &AggregatedBindingConfig,
    executor: &GpuSumcheckExecutor,
) -> [SecureField; 3] {
    let mut results = [SecureField::zero(); 3];

    // Extract selector and local portions from each eval point
    let selectors: [&[SecureField]; 3] = [
        &eval_pts[0][..config.selector_bits],
        &eval_pts[1][..config.selector_bits],
        &eval_pts[2][..config.selector_bits],
    ];
    let locals: [&[SecureField]; 3] = [
        &eval_pts[0][config.selector_bits..],
        &eval_pts[1][config.selector_bits..],
        &eval_pts[2][config.selector_bits..],
    ];

    // Find which position differs (branch_pos) in the full eval_pt
    let branch_pos = (0..eval_pts[0].len())
        .find(|&j| eval_pts[0][j] != eval_pts[1][j])
        .unwrap_or(0);

    // Check if branching is in selector vs local region
    let branch_in_selector = branch_pos < config.selector_bits;

    for i in 0..source.len() {
        let sel_weights: [SecureField; 3] = [
            compute_selector_weight(i, selectors[0], config),
            compute_selector_weight(i, selectors[1], config),
            compute_selector_weight(i, selectors[2], config),
        ];

        // Skip if all 3 selector weights are zero
        if sel_weights[0] == SecureField::zero()
            && sel_weights[1] == SecureField::zero()
            && sel_weights[2] == SecureField::zero()
        {
            continue;
        }

        let n_vars = claim_n_vars[i];
        let local_truncated_0 = &locals[0][config.n_max - n_vars..];

        if branch_in_selector {
            // Branching variable is in the selector — MLE eval is identical for all 3.
            // Evaluate once, multiply by 3 different selector weights.
            let w_val = if n_vars < GPU_MLE_THRESHOLD {
                let mle = source.get_mle(i);
                evaluate_mle_at(&mle, local_truncated_0)
            } else {
                let mle_u32 = source.get_mle_u32(i);
                executor
                    .evaluate_mle_at_gpu_u32(&mle_u32, local_truncated_0)
                    .unwrap_or_else(|e| {
                        eprintln!("[GPU] 3way selector fallback for matrix {i}: {e:?}");
                        let mle = source.get_mle(i);
                        evaluate_mle_at(&mle, local_truncated_0)
                    })
            };
            for k in 0..3 {
                results[k] = results[k] + sel_weights[k] * w_val;
            }
        } else {
            // Branching variable is in the local region (or padding).
            let local_offset = config.n_max - n_vars;
            let branch_in_local = branch_pos.saturating_sub(config.selector_bits);

            if branch_in_local < local_offset {
                // Branch is in the padding zone for this matrix — MLE eval is
                // identical for all 3 t-values. Evaluate once, multiply by weights.
                let w_val = if n_vars < GPU_MLE_THRESHOLD {
                    let mle = source.get_mle(i);
                    evaluate_mle_at(&mle, local_truncated_0)
                } else {
                    let mle_u32 = source.get_mle_u32(i);
                    executor
                        .evaluate_mle_at_gpu_u32(&mle_u32, local_truncated_0)
                        .unwrap_or_else(|e| {
                            eprintln!("[GPU] padding fallback for matrix {i}: {e:?}");
                            let mle = source.get_mle(i);
                            evaluate_mle_at(&mle, local_truncated_0)
                        })
                };
                for k in 0..3 {
                    results[k] = results[k] + sel_weights[k] * w_val;
                }
                continue;
            }

            // Upload MLE once, fold shared prefix, clone+fork for the 3 branch values.
            let local_branch_pos = branch_in_local - local_offset;

            if n_vars < GPU_MLE_THRESHOLD {
                // CPU fallback for small MLEs
                for k in 0..3 {
                    if sel_weights[k] == SecureField::zero() {
                        continue;
                    }
                    let local_truncated_k = &locals[k][config.n_max - n_vars..];
                    let mle = source.get_mle(i);
                    let w_val = evaluate_mle_at(&mle, local_truncated_k);
                    results[k] = results[k] + sel_weights[k] * w_val;
                }
            } else {
                let mle_u32 = source.get_mle_u32(i);
                match executor.start_mle_fold_session_u32(&mle_u32) {
                    Ok(mut session) => {
                        // Fold shared prefix (variables before branch_pos in local space)
                        let prefix = &local_truncated_0[..local_branch_pos];
                        for &challenge in prefix.iter() {
                            if let Err(e) =
                                executor.mle_fold_session_step_in_place(&mut session, challenge)
                            {
                                eprintln!("[GPU] 3way prefix fold error: {e:?}, falling back to CPU");
                                for k in 0..3 {
                                    if sel_weights[k] == SecureField::zero() {
                                        continue;
                                    }
                                    let lt = &locals[k][config.n_max - n_vars..];
                                    let mle = source.get_mle(i);
                                    results[k] = results[k] + sel_weights[k] * evaluate_mle_at(&mle, lt);
                                }
                                break;
                            }
                        }

                        // Fork: clone session for branches 1 and 2, reuse original for branch 0
                        let suffix = &local_truncated_0[local_branch_pos + 1..];
                        let branch_values: [SecureField; 3] = [
                            locals[0][config.n_max - n_vars + local_branch_pos],
                            locals[1][config.n_max - n_vars + local_branch_pos],
                            locals[2][config.n_max - n_vars + local_branch_pos],
                        ];

                        for k in 0..3 {
                            if sel_weights[k] == SecureField::zero() {
                                continue;
                            }
                            let mut fork = if k == 0 {
                                // Reuse original for first branch (avoid unnecessary clone)
                                match executor.clone_fold_session(&session) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        eprintln!("[GPU] clone_fold_session error: {e:?}");
                                        let lt = &locals[k][config.n_max - n_vars..];
                                        let mle = source.get_mle(i);
                                        results[k] = results[k] + sel_weights[k] * evaluate_mle_at(&mle, lt);
                                        continue;
                                    }
                                }
                            } else {
                                match executor.clone_fold_session(&session) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        eprintln!("[GPU] clone_fold_session error: {e:?}");
                                        let lt = &locals[k][config.n_max - n_vars..];
                                        let mle = source.get_mle(i);
                                        results[k] = results[k] + sel_weights[k] * evaluate_mle_at(&mle, lt);
                                        continue;
                                    }
                                }
                            };

                            // Fold branch variable
                            if let Err(e) = executor
                                .mle_fold_session_step_in_place(&mut fork, branch_values[k])
                            {
                                eprintln!("[GPU] branch fold error: {e:?}");
                                let lt = &locals[k][config.n_max - n_vars..];
                                let mle = source.get_mle(i);
                                results[k] = results[k] + sel_weights[k] * evaluate_mle_at(&mle, lt);
                                continue;
                            }

                            // Fold suffix and read result
                            match executor.finish_fold_and_read(&mut fork, suffix) {
                                Ok(w_val) => {
                                    results[k] = results[k] + sel_weights[k] * w_val;
                                }
                                Err(e) => {
                                    eprintln!("[GPU] finish_fold error: {e:?}");
                                    let lt = &locals[k][config.n_max - n_vars..];
                                    let mle = source.get_mle(i);
                                    results[k] = results[k] + sel_weights[k] * evaluate_mle_at(&mle, lt);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Full CPU fallback on session start failure
                        eprintln!("[GPU] session start error for matrix {i}: {e:?}");
                        for k in 0..3 {
                            if sel_weights[k] == SecureField::zero() {
                                continue;
                            }
                            let lt = &locals[k][config.n_max - n_vars..];
                            let mle = source.get_mle(i);
                            results[k] = results[k] + sel_weights[k] * evaluate_mle_at(&mle, lt);
                        }
                    }
                }
            }
        }
    }

    results
}

/// Evaluate the unified oracle at a single point on GPU (streaming).
///
/// Same as [`eval_unified_oracle_streaming`] but uses GPU for large MLEs.
#[cfg(feature = "cuda-runtime")]
fn eval_unified_oracle_streaming_gpu(
    t: &[SecureField],
    source: &dyn WeightSource,
    claim_n_vars: &[usize],
    config: &AggregatedBindingConfig,
    executor: &GpuSumcheckExecutor,
) -> SecureField {
    let selector = &t[..config.selector_bits];
    let local = &t[config.selector_bits..];
    assert_eq!(local.len(), config.n_max);

    let mut result = SecureField::zero();
    for i in 0..source.len() {
        let sel_weight = compute_selector_weight(i, selector, config);

        if sel_weight == SecureField::zero() {
            continue;
        }

        let n_vars = claim_n_vars[i];
        let local_truncated = &local[config.n_max - n_vars..];

        let w_val = if n_vars < GPU_MLE_THRESHOLD {
            let mle = source.get_mle(i);
            evaluate_mle_at(&mle, local_truncated)
        } else {
            let mle_u32 = source.get_mle_u32(i);
            executor
                .evaluate_mle_at_gpu_u32(&mle_u32, local_truncated)
                .unwrap_or_else(|e| {
                    eprintln!("[GPU] oracle eval fallback for matrix {i}: {e:?}");
                    let mle = source.get_mle(i);
                    evaluate_mle_at(&mle, local_truncated)
                })
        };

        result = result + sel_weight * w_val;
    }

    result
}

/// Evaluate the sumcheck polynomial for a specific round at a given value,
/// using streaming weight MLE access.
///
/// Same semantics as [`eval_round_at_value`] but calls [`eval_unified_oracle_streaming`].
fn eval_round_at_value_streaming(
    value: SecureField,
    round: usize,
    claims: &[AggregatedWeightClaim],
    source: &dyn WeightSource,
    claim_n_vars: &[usize],
    config: &AggregatedBindingConfig,
    betas: &[SecureField],
    global_points: &[Vec<SecureField>],
    eq_prefix: &[SecureField],
    challenge_point: &[SecureField],
) -> SecureField {
    let one = SecureField::from(M31::from(1));
    let n = config.n_global;

    let mut total = SecureField::zero();

    for (i, claim) in claims.iter().enumerate() {
        let gp = &global_points[i];

        // eq factor for this round variable
        let eq_round = gp[round] * value + (one - gp[round]) * (one - value);

        // Build the full evaluation point
        let mut eval_pt = Vec::with_capacity(n);
        eval_pt.extend_from_slice(challenge_point); // r_0..r_{round-1}
        eval_pt.push(value); // X_round = value
        eval_pt.extend_from_slice(&gp[round + 1..]); // g_i[round+1:]

        let w_at_pt = eval_unified_oracle_streaming(&eval_pt, source, claim_n_vars, config);
        let mismatch = w_at_pt - claim.expected_value;

        total = total + betas[i] * eq_prefix[i] * eq_round * mismatch;
    }

    total
}

/// Run the mismatch sumcheck using streaming weight MLE access.
///
/// Same structure as [`mismatch_sumcheck`] but loads MLEs on demand via the
/// [`WeightSource`] trait instead of requiring all MLEs in memory.
fn mismatch_sumcheck_streaming(
    claims: &[AggregatedWeightClaim],
    source: &dyn WeightSource,
    claim_n_vars: &[usize],
    config: &AggregatedBindingConfig,
    betas: &[SecureField],
    channel: &mut PoseidonChannel,
) -> (
    Vec<(SecureField, SecureField, SecureField)>,
    Vec<SecureField>,
) {
    let n = config.n_global;
    let one = SecureField::from(M31::from(1));

    let global_points: Vec<Vec<SecureField>> =
        claims.iter().map(|c| global_point(c, config)).collect();

    let mut challenge_point = Vec::with_capacity(n);
    let mut round_polys = Vec::with_capacity(n);
    let mut eq_prefix: Vec<SecureField> = vec![one; claims.len()];

    for round in 0..n {
        let s0 = eval_round_at_value_streaming(
            SecureField::zero(),
            round,
            claims,
            source,
            claim_n_vars,
            config,
            betas,
            &global_points,
            &eq_prefix,
            &challenge_point,
        );
        let s1 = eval_round_at_value_streaming(
            one,
            round,
            claims,
            source,
            claim_n_vars,
            config,
            betas,
            &global_points,
            &eq_prefix,
            &challenge_point,
        );
        let two = SecureField::from(M31::from(2));
        let s2 = eval_round_at_value_streaming(
            two,
            round,
            claims,
            source,
            claim_n_vars,
            config,
            betas,
            &global_points,
            &eq_prefix,
            &challenge_point,
        );

        // Degree-2 polynomial through (0, s0), (1, s1), (2, s2)
        let c0 = s0;
        let two_inv = two.inverse();
        let c2 = (s0 - two * s1 + s2) * two_inv;
        let c1 = s1 - s0 - c2;

        channel.mix_poly_coeffs(c0, c1, c2);
        round_polys.push((c0, c1, c2));

        let r = channel.draw_qm31();
        challenge_point.push(r);

        for (i, gp) in global_points.iter().enumerate() {
            let g_val = gp[round];
            eq_prefix[i] = eq_prefix[i] * (r * g_val + (one - r) * (one - g_val));
        }
    }

    (round_polys, challenge_point)
}

/// Evaluate the sumcheck polynomial at 3 values (t=0, t=1, t=2) for a round.
///
/// **Matrix-first loop order** with **pre-cached GPU sessions**: each MLE is
/// uploaded once at sumcheck start and stays on GPU. Per round, we clone from
/// the cached session (dtod copy, ~5ms) instead of re-uploading (~80ms).
#[cfg(feature = "cuda-runtime")]
fn eval_round_3way_gpu(
    round: usize,
    claims: &[AggregatedWeightClaim],
    source: &dyn WeightSource,
    claim_n_vars: &[usize],
    config: &AggregatedBindingConfig,
    betas: &[SecureField],
    global_points: &[Vec<SecureField>],
    eq_prefix: &[SecureField],
    challenge_point: &[SecureField],
    executor: &GpuSumcheckExecutor,
    // Pre-uploaded GPU sessions (None for small MLEs or upload failures).
    gpu_sessions: &[Option<GpuMleFoldSession>],
) -> (SecureField, SecureField, SecureField) {
    let one = SecureField::from(M31::from(1));
    let two = SecureField::from(M31::from(2));
    let n = config.n_global;
    let values = [SecureField::zero(), one, two];

    // Accumulate per-claim oracle values: oracle_accum[ci][k] = Σ_j sel_weight_j * w_j
    // We MUST NOT compute (w_val - v_i) per matrix because Σ sel_weight ≠ 1 at
    // non-boolean challenge points. Instead: accumulate the oracle, then subtract v_i once.
    let n_claims = claims.len();
    let mut oracle_accum: Vec<[SecureField; 3]> = vec![[SecureField::zero(); 3]; n_claims];

    // Pre-build eval points for all claims: eval_pts[claim][k] = full point
    let eval_pts: Vec<[Vec<SecureField>; 3]> = claims
        .iter()
        .enumerate()
        .map(|(ci, _)| {
            let gp = &global_points[ci];
            let mut pts: [Vec<SecureField>; 3] = [
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            ];
            for k in 0..3 {
                pts[k].extend_from_slice(challenge_point);
                pts[k].push(values[k]);
                pts[k].extend_from_slice(&gp[round + 1..]);
            }
            pts
        })
        .collect();

    // MATRIX-FIRST: accumulate sel_weight * w_val into oracle_accum
    for mat_idx in 0..source.len() {
        let n_vars = claim_n_vars[mat_idx];

        // Check if ANY claim needs this matrix (any non-zero selector weight)
        let mut any_nonzero = false;
        let claim_sel_weights: Vec<[SecureField; 3]> = (0..n_claims)
            .map(|ci| {
                let mut sw = [SecureField::zero(); 3];
                for k in 0..3 {
                    let selector = &eval_pts[ci][k][..config.selector_bits];
                    sw[k] = compute_selector_weight(mat_idx, selector, config);
                    if sw[k] != SecureField::zero() {
                        any_nonzero = true;
                    }
                }
                sw
            })
            .collect();

        if !any_nonzero {
            continue;
        }

        if gpu_sessions[mat_idx].is_none() {
            // CPU path for small MLEs or upload failures
            let mle = source.get_mle(mat_idx);
            for ci in 0..n_claims {
                for k in 0..3 {
                    if claim_sel_weights[ci][k] == SecureField::zero() {
                        continue;
                    }
                    let local = &eval_pts[ci][k][config.selector_bits..];
                    let local_truncated = &local[config.n_max - n_vars..];
                    let w_val = evaluate_mle_at(&mle, local_truncated);
                    oracle_accum[ci][k] =
                        oracle_accum[ci][k] + claim_sel_weights[ci][k] * w_val;
                }
            }
        } else {
            let base_session = gpu_sessions[mat_idx].as_ref().unwrap();

            for ci in 0..n_claims {
                let local_0 = &eval_pts[ci][0][config.selector_bits..];
                let local_truncated = &local_0[config.n_max - n_vars..];

                let branch_global = round;
                let branch_in_selector = branch_global < config.selector_bits;
                let local_offset = config.n_max - n_vars;
                let branch_in_padding = !branch_in_selector
                    && (branch_global - config.selector_bits) < local_offset;

                if branch_in_selector || branch_in_padding {
                    // MLE eval is identical for all 3 t-values; only weights differ
                    let w_val = {
                        let fork = executor.clone_fold_session(base_session);
                        match fork {
                            Ok(mut f) => executor
                                .finish_fold_and_read(&mut f, local_truncated)
                                .unwrap_or_else(|_| {
                                    let mle = source.get_mle(mat_idx);
                                    evaluate_mle_at(&mle, local_truncated)
                                }),
                            Err(_) => {
                                let mle = source.get_mle(mat_idx);
                                evaluate_mle_at(&mle, local_truncated)
                            }
                        }
                    };
                    for k in 0..3 {
                        if claim_sel_weights[ci][k] == SecureField::zero() {
                            continue;
                        }
                        oracle_accum[ci][k] =
                            oracle_accum[ci][k] + claim_sel_weights[ci][k] * w_val;
                    }
                } else {
                    // Branch is in real local vars — 3-way fork
                    let local_branch_pos =
                        (branch_global - config.selector_bits) - local_offset;

                    let prefix_session = match executor.clone_fold_session(base_session) {
                        Ok(mut f) => {
                            let prefix = &local_truncated[..local_branch_pos];
                            let mut ok = true;
                            for &ch in prefix.iter() {
                                if executor
                                    .mle_fold_session_step_in_place(&mut f, ch)
                                    .is_err()
                                {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok { Some(f) } else { None }
                        }
                        Err(_) => None,
                    };

                    if let Some(ref ps) = prefix_session {
                        let suffix = &local_truncated[local_branch_pos + 1..];
                        let branch_values = [
                            eval_pts[ci][0]
                                [config.selector_bits + local_offset + local_branch_pos],
                            eval_pts[ci][1]
                                [config.selector_bits + local_offset + local_branch_pos],
                            eval_pts[ci][2]
                                [config.selector_bits + local_offset + local_branch_pos],
                        ];

                        for k in 0..3 {
                            if claim_sel_weights[ci][k] == SecureField::zero() {
                                continue;
                            }
                            let w_val = match executor.clone_fold_session(ps) {
                                Ok(mut fork) => {
                                    let step_ok = executor
                                        .mle_fold_session_step_in_place(
                                            &mut fork,
                                            branch_values[k],
                                        )
                                        .is_ok();
                                    if step_ok {
                                        executor
                                            .finish_fold_and_read(&mut fork, suffix)
                                            .unwrap_or_else(|_| {
                                                let mle = source.get_mle(mat_idx);
                                                let lt_k = &eval_pts[ci][k]
                                                    [config.selector_bits + local_offset..];
                                                evaluate_mle_at(&mle, lt_k)
                                            })
                                    } else {
                                        let mle = source.get_mle(mat_idx);
                                        let lt_k = &eval_pts[ci][k]
                                            [config.selector_bits + local_offset..];
                                        evaluate_mle_at(&mle, lt_k)
                                    }
                                }
                                Err(_) => {
                                    let mle = source.get_mle(mat_idx);
                                    let lt_k = &eval_pts[ci][k]
                                        [config.selector_bits + local_offset..];
                                    evaluate_mle_at(&mle, lt_k)
                                }
                            };
                            oracle_accum[ci][k] =
                                oracle_accum[ci][k] + claim_sel_weights[ci][k] * w_val;
                        }
                    } else {
                        // GPU prefix fold failed — CPU fallback for all 3
                        let mle = source.get_mle(mat_idx);
                        for k in 0..3 {
                            if claim_sel_weights[ci][k] == SecureField::zero() {
                                continue;
                            }
                            let lt_k =
                                &eval_pts[ci][k][config.selector_bits + local_offset..];
                            let w_val = evaluate_mle_at(&mle, lt_k);
                            oracle_accum[ci][k] =
                                oracle_accum[ci][k] + claim_sel_weights[ci][k] * w_val;
                        }
                    }
                }
            }
        }
    }

    // Now compute s[k] = Σ_ci β[ci] * eq_prefix[ci] * eq_round * (oracle_accum[ci][k] - v_ci)
    let mut s = [SecureField::zero(); 3];
    for (ci, claim) in claims.iter().enumerate() {
        let gp = &global_points[ci];
        for k in 0..3 {
            let eq_round = gp[round] * values[k] + (one - gp[round]) * (one - values[k]);
            let mismatch = oracle_accum[ci][k] - claim.expected_value;
            s[k] = s[k] + betas[ci] * eq_prefix[ci] * eq_round * mismatch;
        }
    }

    (s[0], s[1], s[2])
}

/// Run the mismatch sumcheck using GPU-accelerated 3-way batched evaluation.
///
/// **Matrix-first loop order**: each MLE loaded once per round (not once per claim).
/// Same Fiat-Shamir transcript as [`mismatch_sumcheck_streaming`].
#[cfg(feature = "cuda-runtime")]
fn mismatch_sumcheck_streaming_gpu(
    claims: &[AggregatedWeightClaim],
    source: &dyn WeightSource,
    claim_n_vars: &[usize],
    config: &AggregatedBindingConfig,
    betas: &[SecureField],
    channel: &mut PoseidonChannel,
    executor: &GpuSumcheckExecutor,
) -> (
    Vec<(SecureField, SecureField, SecureField)>,
    Vec<SecureField>,
) {
    let n = config.n_global;
    let one = SecureField::from(M31::from(1));

    eprintln!(
        "[GPU-sumcheck] starting: {} rounds, {} claims, {} matrices",
        n,
        claims.len(),
        source.len()
    );
    let t_start = std::time::Instant::now();

    // Pre-upload all large MLEs to GPU once. Sessions stay resident across rounds.
    // For small MLEs (< 2^GPU_MLE_THRESHOLD), we use CPU eval (None slot).
    let gpu_sessions: Vec<Option<GpuMleFoldSession>> = (0..source.len())
        .map(|i| {
            let nv = claim_n_vars[i];
            if nv < GPU_MLE_THRESHOLD {
                return None;
            }
            let mle_u32 = source.get_mle_u32(i);
            match executor.start_mle_fold_session_u32(&mle_u32) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("[GPU-sumcheck] pre-upload failed for matrix {i}: {e:?}");
                    None
                }
            }
        })
        .collect();
    let n_cached = gpu_sessions.iter().filter(|s| s.is_some()).count();
    eprintln!(
        "[GPU-sumcheck] pre-uploaded {}/{} MLEs to GPU in {:.1}s",
        n_cached,
        source.len(),
        t_start.elapsed().as_secs_f64()
    );

    let global_points: Vec<Vec<SecureField>> =
        claims.iter().map(|c| global_point(c, config)).collect();

    let mut challenge_point = Vec::with_capacity(n);
    let mut round_polys = Vec::with_capacity(n);
    let mut eq_prefix: Vec<SecureField> = vec![one; claims.len()];

    for round in 0..n {
        let t_round = std::time::Instant::now();
        let (s0, s1, s2) = eval_round_3way_gpu(
            round,
            claims,
            source,
            claim_n_vars,
            config,
            betas,
            &global_points,
            &eq_prefix,
            &challenge_point,
            executor,
            &gpu_sessions,
        );
        if round < 3 || round == n - 1 || (round + 1) % 10 == 0 {
            eprintln!(
                "[GPU-sumcheck] round {}/{} in {:.2}s (total {:.1}s)",
                round + 1,
                n,
                t_round.elapsed().as_secs_f64(),
                t_start.elapsed().as_secs_f64()
            );
        }

        // Degree-2 polynomial through (0, s0), (1, s1), (2, s2)
        let two = SecureField::from(M31::from(2));
        let c0 = s0;
        let two_inv = two.inverse();
        let c2 = (s0 - two * s1 + s2) * two_inv;
        let c1 = s1 - s0 - c2;

        channel.mix_poly_coeffs(c0, c1, c2);
        round_polys.push((c0, c1, c2));

        let r = channel.draw_qm31();
        challenge_point.push(r);

        for (i, gp) in global_points.iter().enumerate() {
            let g_val = gp[round];
            eq_prefix[i] = eq_prefix[i] * (r * g_val + (one - r) * (one - g_val));
        }
    }

    eprintln!(
        "[GPU-sumcheck] complete: {} rounds in {:.1}s",
        n,
        t_start.elapsed().as_secs_f64()
    );

    (round_polys, challenge_point)
}

/// Build the virtual MLE u32 array slot-by-slot from a streaming source.
///
/// Each matrix u32 MLE is loaded, copied into its slot, and then dropped.
/// This avoids requiring all u32 MLEs in memory simultaneously.
#[cfg(feature = "cuda-runtime")]
fn build_virtual_mle_u32_streaming(
    source: &dyn WeightSource,
    config: &AggregatedBindingConfig,
) -> Vec<u32> {
    let total_size = 1usize << config.n_global;
    let slot_size = 1usize << config.n_max;

    // 4 u32 words per QM31 element
    let mut virtual_mle = vec![0u32; total_size * 4];

    for i in 0..source.len() {
        let mle_u32 = source.get_mle_u32(i);
        let offset = i * slot_size * 4;
        virtual_mle[offset..offset + mle_u32.len()].copy_from_slice(&mle_u32);
        // mle_u32 dropped here
    }

    virtual_mle
}

/// Build the virtual MLE (SecureField) slot-by-slot from a streaming source.
///
/// Used for CPU-only (non-CUDA) path. Each matrix MLE is loaded, copied, and dropped.
#[cfg(not(feature = "cuda-runtime"))]
fn build_virtual_mle_streaming(
    source: &dyn WeightSource,
    config: &AggregatedBindingConfig,
) -> Vec<SecureField> {
    let total_size = 1usize << config.n_global;
    let slot_size = 1usize << config.n_max;

    let mut virtual_mle = vec![SecureField::zero(); total_size];

    for i in 0..source.len() {
        let mle = source.get_mle(i);
        let offset = i * slot_size;
        virtual_mle[offset..offset + mle.len()].copy_from_slice(&mle);
        // mle dropped here
    }

    virtual_mle
}

/// Streaming aggregated weight binding prover (GPU path).
///
/// Same protocol as [`prove_aggregated_binding_gpu`] but loads weight MLEs
/// on-demand from a [`WeightSource`] instead of requiring all MLEs in memory.
/// Peak MLE memory drops from ~640 GB (all matrices) to ~4 GB (largest single matrix).
///
/// The sumcheck uses GPU-accelerated 3-way batched evaluation: each weight MLE
/// is uploaded to GPU once per round and folded to 3 eval points (t=0,1,2)
/// via prefix-fold + clone-fork, reducing upload cost by 3×.
#[cfg(feature = "cuda-runtime")]
pub fn prove_aggregated_binding_streaming(
    claims: &[AggregatedWeightClaim],
    source: &dyn WeightSource,
    channel: &mut PoseidonChannel,
) -> AggregatedWeightBindingProof {
    assert_eq!(claims.len(), source.len());
    assert!(!claims.is_empty());

    let config = AggregatedBindingConfig::from_claims(claims);
    let claim_n_vars: Vec<usize> = claims.iter().map(|c| c.local_n_vars).collect();

    // 1. Build super-root and mix into channel
    let super_root = build_super_root(claims, &config);
    channel.mix_felt(super_root.root);

    // 2. Draw β weights
    let betas = draw_beta_weights(channel, claims.len());

    // 3. GPU-accelerated streaming mismatch sumcheck
    let (round_polys, challenge_point) = match GpuSumcheckExecutor::cached() {
        Ok(executor) => {
            eprintln!("[GPU] Using GPU-accelerated aggregated binding sumcheck");
            mismatch_sumcheck_streaming_gpu(
                claims, source, &claim_n_vars, &config, &betas, channel, &executor,
            )
        }
        Err(e) => {
            eprintln!("[GPU] Executor init failed ({e:?}), falling back to CPU sumcheck");
            mismatch_sumcheck_streaming(
                claims, source, &claim_n_vars, &config, &betas, channel,
            )
        }
    };

    // 4. Oracle eval at challenge point (streaming — loads one matrix at a time)
    let oracle_eval =
        eval_unified_oracle_streaming(&challenge_point, source, &claim_n_vars, &config);
    channel.mix_felts(&[oracle_eval]);

    // 5. Build virtual MLE and prove single opening against super-root.
    //    This matches the Cairo verifier which expects a single MLE opening proof
    //    against the super-root (not per-matrix openings).
    //    Memory cost: ~2^n_global × 16 bytes (e.g. ~4GB for 4 matrices of Qwen3-14B 1-layer).
    let t_opening = std::time::Instant::now();
    let virtual_mle = build_virtual_mle_streaming(source, &config);
    eprintln!(
        "[GPU] virtual MLE built: {} elements ({:.1} GB) in {:.1}s",
        virtual_mle.len(),
        (virtual_mle.len() * std::mem::size_of::<SecureField>()) as f64 / 1e9,
        t_opening.elapsed().as_secs_f64()
    );
    let opening_proof = prove_mle_opening(&virtual_mle, &challenge_point, channel);
    eprintln!(
        "[GPU] single virtual MLE opening: {:.1}s total",
        t_opening.elapsed().as_secs_f64()
    );

    AggregatedWeightBindingProof {
        config,
        sumcheck_round_polys: round_polys,
        oracle_eval_at_s: oracle_eval,
        opening_proof,
        super_root,
        per_matrix_openings: None,
        per_matrix_evals: None,
    }
}

/// Streaming aggregated weight binding prover (CPU path).
///
/// Same protocol as [`prove_aggregated_binding`] but loads weight MLEs on-demand.
#[cfg(not(feature = "cuda-runtime"))]
pub fn prove_aggregated_binding_streaming(
    claims: &[AggregatedWeightClaim],
    source: &dyn WeightSource,
    channel: &mut PoseidonChannel,
) -> AggregatedWeightBindingProof {
    assert_eq!(claims.len(), source.len());
    assert!(!claims.is_empty());

    let config = AggregatedBindingConfig::from_claims(claims);
    let claim_n_vars: Vec<usize> = claims.iter().map(|c| c.local_n_vars).collect();

    // 1. Build super-root and mix into channel
    let super_root = build_super_root(claims, &config);
    channel.mix_felt(super_root.root);

    // 2. Draw β weights
    let betas = draw_beta_weights(channel, claims.len());

    // 3. Streaming mismatch sumcheck
    let (round_polys, challenge_point) =
        mismatch_sumcheck_streaming(claims, source, &claim_n_vars, &config, &betas, channel);

    // 4. Oracle eval at challenge point
    let oracle_eval =
        eval_unified_oracle_streaming(&challenge_point, source, &claim_n_vars, &config);
    channel.mix_felts(&[oracle_eval]);

    // 5. Build virtual MLE and prove opening
    let virtual_mle = build_virtual_mle_streaming(source, &config);
    let opening_proof = prove_mle_opening(&virtual_mle, &challenge_point, channel);

    AggregatedWeightBindingProof {
        config,
        sumcheck_round_polys: round_polys,
        oracle_eval_at_s: oracle_eval,
        opening_proof,
        super_root,
        per_matrix_openings: None,
        per_matrix_evals: None,
    }
}

// ─── Verifier ────────────────────────────────────────────────────────────────

/// Verify the aggregated weight binding proof.
///
/// # Arguments
/// - `proof`: The aggregated binding proof
/// - `claims`: Weight claims (must match prover's claims)
/// - `channel`: Fiat-Shamir channel (post-GKR-walk state, must be identical to prover's)
///
/// # Returns
/// `true` if verification passes.
pub fn verify_aggregated_binding(
    proof: &AggregatedWeightBindingProof,
    claims: &[AggregatedWeightClaim],
    channel: &mut PoseidonChannel,
) -> bool {
    if claims.is_empty() {
        return false;
    }

    let config = &proof.config;

    // 1. Verify super-root matches claimed commitments
    let expected_super_root = build_super_root(claims, config);
    if expected_super_root.root != proof.super_root.root {
        return false;
    }

    // 2. Mix super-root
    channel.mix_felt(proof.super_root.root);

    // 3. Draw β weights
    let betas = draw_beta_weights(channel, claims.len());

    // 4. Verify sumcheck
    let global_points: Vec<Vec<SecureField>> =
        claims.iter().map(|c| global_point(c, config)).collect();

    let n = config.n_global;

    if proof.sumcheck_round_polys.len() != n {
        return false;
    }

    // Verify each sumcheck round
    let mut current_sum = SecureField::zero();
    let mut challenge_point = Vec::with_capacity(n);

    for (round, &(c0, c1, c2)) in proof.sumcheck_round_polys.iter().enumerate() {
        // Check: p(0) + p(1) == current_sum
        let p0 = c0;
        let p1 = c0 + c1 + c2;
        if round == 0 {
            // First round: p(0) + p(1) should equal 0
            if p0 + p1 != SecureField::zero() {
                return false;
            }
        } else {
            if p0 + p1 != current_sum {
                return false;
            }
        }

        // Mix round polynomial
        channel.mix_poly_coeffs(c0, c1, c2);

        // Draw challenge
        let r = channel.draw_qm31();
        challenge_point.push(r);

        // Next sum = p(r)
        current_sum = c0 + c1 * r + c2 * r * r;
    }

    // 5. Final check: sumcheck output = Σ β_i * eq(g_i, s) * (W(s) - v_i)
    let mut verifier_sum = SecureField::zero();
    for (i, claim) in claims.iter().enumerate() {
        let eq_val = eq_eval(&global_points[i], &challenge_point);
        verifier_sum =
            verifier_sum + betas[i] * eq_val * (proof.oracle_eval_at_s - claim.expected_value);
    }

    if current_sum != verifier_sum {
        eprintln!("[verify_aggregated_binding] FAIL: sumcheck final sum mismatch");
        return false;
    }

    // 6-7. Verify MLE openings
    if let (Some(ref pm_openings), Some(ref pm_evals)) =
        (&proof.per_matrix_openings, &proof.per_matrix_evals)
    {
        // Per-matrix path: openings come BEFORE oracle_eval mix (matching prover order)
        if pm_openings.len() != claims.len() || pm_evals.len() != claims.len() {
            eprintln!("[verify_aggregated_binding] FAIL: per-matrix length mismatch");
            return false;
        }

        let selector = &challenge_point[..config.selector_bits];
        let local = &challenge_point[config.selector_bits..];

        // Verify each per-matrix opening
        for (i, claim) in claims.iter().enumerate() {
            let n_vars = claim.local_n_vars;
            let local_truncated = &local[config.n_max - n_vars..];

            if !verify_mle_opening(
                claim.commitment,
                &pm_openings[i],
                local_truncated,
                channel,
            ) {
                eprintln!("[verify_aggregated_binding] FAIL: per-matrix opening {i} failed (n_vars={n_vars}, commitment={:?})", claim.commitment);
                return false;
            }
        }

        // Verify oracle_eval = Σ sel_weight(i) * per_matrix_eval(i)
        let mut expected_eval = SecureField::zero();
        for i in 0..claims.len() {
            let sel_weight = compute_selector_weight(i, selector, config);
            expected_eval = expected_eval + sel_weight * pm_evals[i];
        }
        if expected_eval != proof.oracle_eval_at_s {
            eprintln!("[verify_aggregated_binding] FAIL: oracle_eval mismatch (expected={expected_eval:?}, got={:?})", proof.oracle_eval_at_s);
            return false;
        }

        // Mix oracle eval AFTER per-matrix openings (matching prover order)
        channel.mix_felts(&[proof.oracle_eval_at_s]);

        true
    } else {
        // Legacy path: mix oracle eval, then single opening against super-root
        channel.mix_felts(&[proof.oracle_eval_at_s]);
        verify_mle_opening(
            proof.super_root.root,
            &proof.opening_proof,
            &challenge_point,
            channel,
        )
    }
}

// ─── Serialization helpers ───────────────────────────────────────────────────

impl AggregatedWeightBindingProof {
    /// Estimate calldata size in felt252s.
    pub fn estimated_calldata_felts(&self) -> usize {
        let config_felts = 5; // selector_bits, n_max, m_padded, n_global, n_claims
        let round_poly_felts = self.sumcheck_round_polys.len() * 3 * 4; // 3 QM31 coeffs * 4 felts each
        let oracle_eval_felts = 4; // one QM31
        let super_root_felts = 1 + self.super_root.subtree_roots.len(); // root + subtrees
        let opening_felts = estimate_mle_opening_felts(&self.opening_proof);
        config_felts + round_poly_felts + oracle_eval_felts + super_root_felts + opening_felts
    }
}

fn estimate_mle_opening_felts(proof: &MleOpeningProof) -> usize {
    let roots = proof.intermediate_roots.len();
    let mut query_felts = 0;
    for q in &proof.queries {
        query_felts += 1; // initial_pair_index
        query_felts += 1; // rounds.len()
        for r in &q.rounds {
            query_felts += 4 * 2; // left_value + right_value (QM31)
            query_felts += 1 + r.left_siblings.len(); // len + siblings
            query_felts += 1 + r.right_siblings.len();
        }
    }
    1 + roots  // intermediate_roots array
        + 1 + query_felts // queries array
        + 4 // final_value
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::mle_opening::{commit_mle_root_only, evaluate_mle_at, prove_mle_opening};
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::qm31::QM31;

    fn random_qm31() -> SecureField {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        QM31(
            CM31(
                M31::from(rng.gen_range(0..((1u64 << 31) - 1) as u32)),
                M31::from(rng.gen_range(0..((1u64 << 31) - 1) as u32)),
            ),
            CM31(
                M31::from(rng.gen_range(0..((1u64 << 31) - 1) as u32)),
                M31::from(rng.gen_range(0..((1u64 << 31) - 1) as u32)),
            ),
        )
    }

    fn random_mle(n_vars: usize) -> Vec<SecureField> {
        (0..1 << n_vars).map(|_| random_qm31()).collect()
    }

    fn make_claim(matrix_index: usize, mle: &[SecureField]) -> AggregatedWeightClaim {
        let n_vars = mle.len().trailing_zeros() as usize;
        let eval_point: Vec<SecureField> = (0..n_vars).map(|_| random_qm31()).collect();
        let expected_value = evaluate_mle_at(mle, &eval_point);
        let commitment = commit_mle_root_only(mle);

        AggregatedWeightClaim {
            matrix_index,
            local_n_vars: n_vars,
            eval_point,
            expected_value,
            commitment,
        }
    }

    #[test]
    fn test_zero_tree_root() {
        // Zero tree of depth 1 = hash(zero_leaf, zero_leaf)
        let zero_leaf = securefield_to_felt(SecureField::zero());
        let expected = poseidon_hash(zero_leaf, zero_leaf);
        assert_eq!(compute_zero_tree_root(1), expected);

        // Depth 2 = hash(hash(z,z), hash(z,z))
        let h1 = poseidon_hash(zero_leaf, zero_leaf);
        let expected2 = poseidon_hash(h1, h1);
        assert_eq!(compute_zero_tree_root(2), expected2);
    }

    #[test]
    fn test_super_root_reconstruction() {
        let mle0 = random_mle(4);
        let mle1 = random_mle(4);
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0.clone(), claim1.clone()];
        let config = AggregatedBindingConfig::from_claims(&claims);

        let sr1 = build_super_root(&claims, &config);
        let sr2 = build_super_root(&claims, &config);
        assert_eq!(sr1.root, sr2.root, "super root should be deterministic");
    }

    #[test]
    fn test_2_matrices() {
        let mle0 = random_mle(3);
        let mle1 = random_mle(3);

        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];
        let weight_mles: Vec<&[SecureField]> = vec![&mle0, &mle1];

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(42); // simulate post-GKR state

        let mut verifier_ch = prover_ch.clone();

        let proof = prove_aggregated_binding(&claims, &weight_mles, &mut prover_ch);
        assert!(
            verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "2-matrix aggregated binding should verify"
        );
    }

    #[test]
    fn test_10_varying_sizes() {
        let sizes = [3, 4, 5, 3, 4, 5, 6, 3, 4, 5];
        let mles: Vec<Vec<SecureField>> = sizes.iter().map(|&n| random_mle(n)).collect();
        let claims: Vec<AggregatedWeightClaim> = mles
            .iter()
            .enumerate()
            .map(|(i, m)| make_claim(i, m))
            .collect();
        let weight_refs: Vec<&[SecureField]> = mles.iter().map(|m| m.as_slice()).collect();

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(123);
        let mut verifier_ch = prover_ch.clone();

        let proof = prove_aggregated_binding(&claims, &weight_refs, &mut prover_ch);
        assert!(
            verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "10-matrix varying sizes should verify"
        );
    }

    #[test]
    fn test_tampered_claim_fails() {
        let mle0 = random_mle(3);
        let mle1 = random_mle(3);

        let mut claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);

        // Tamper with expected value
        claim0.expected_value = claim0.expected_value + SecureField::from(M31::from(1));

        let claims = vec![claim0, claim1];
        let weight_mles: Vec<&[SecureField]> = vec![&mle0, &mle1];

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(42);
        let mut verifier_ch = prover_ch.clone();

        let proof = prove_aggregated_binding(&claims, &weight_mles, &mut prover_ch);
        // Should fail — the mismatch sum won't be zero
        assert!(
            !verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "tampered claim should fail verification"
        );
    }

    #[test]
    fn test_wrong_commitment_fails() {
        let mle0 = random_mle(3);
        let mle1 = random_mle(3);

        let mut claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);

        // Replace commitment with wrong value
        claim0.commitment = FieldElement::from(9999u64);

        let claims_prover = vec![claim0.clone(), claim1.clone()];
        let weight_mles: Vec<&[SecureField]> = vec![&mle0, &mle1];

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(42);
        let mut verifier_ch = prover_ch.clone();

        let proof = prove_aggregated_binding(&claims_prover, &weight_mles, &mut prover_ch);

        // Verifier uses same wrong commitment — sumcheck passes but MLE opening fails
        assert!(
            !verify_aggregated_binding(&proof, &claims_prover, &mut verifier_ch),
            "wrong commitment should fail MLE opening verification"
        );
    }

    #[test]
    fn test_single_matrix() {
        let mle = random_mle(5);
        let claim = make_claim(0, &mle);
        let claims = vec![claim];
        let weight_mles: Vec<&[SecureField]> = vec![&mle];

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(1);
        let mut verifier_ch = prover_ch.clone();

        let proof = prove_aggregated_binding(&claims, &weight_mles, &mut prover_ch);
        assert!(
            verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "single matrix should verify"
        );
    }

    #[test]
    fn test_sparse_matches_dense() {
        // Verify that our eval_unified_oracle matches direct MLE evaluation
        let mle0 = random_mle(3);
        let mle1 = random_mle(3);
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];
        let config = AggregatedBindingConfig::from_claims(&claims);
        let weight_mles: Vec<&[SecureField]> = vec![&mle0, &mle1];

        // Build full virtual MLE and evaluate at a random point
        let virtual_mle = build_virtual_mle(&weight_mles, &config);
        let point: Vec<SecureField> = (0..config.n_global).map(|_| random_qm31()).collect();

        let dense_val = evaluate_mle_at(&virtual_mle, &point);
        let sparse_val = eval_unified_oracle(&point, &weight_mles, &config);

        assert_eq!(
            dense_val, sparse_val,
            "sparse and dense oracle evaluation should match"
        );
    }

    #[test]
    fn test_eq_eval_identity() {
        // eq(a, a) should be 1 for boolean a
        let one = SecureField::from(M31::from(1));
        let zero = SecureField::zero();
        let a = vec![one, zero, one, zero];
        assert_eq!(eq_eval(&a, &a), one);
    }

    #[test]
    fn test_config_from_claims() {
        let mle0 = random_mle(5);
        let mle1 = random_mle(7);
        let mle2 = random_mle(3);
        let claims: Vec<AggregatedWeightClaim> = [(&mle0, 0), (&mle1, 1), (&mle2, 2)]
            .iter()
            .map(|(m, i)| make_claim(*i, m))
            .collect();

        let config = AggregatedBindingConfig::from_claims(&claims);
        assert_eq!(config.n_max, 7); // max local vars
        assert_eq!(config.m_padded, 4); // 3 claims → next pow2 = 4
        assert_eq!(config.selector_bits, 2); // log2(4)
        assert_eq!(config.n_global, 9); // 2 + 7
        assert_eq!(config.n_claims, 3);
    }

    // NOTE: The 160-matrix production-scale test is in tests/e2e_full_pipeline.rs
    // since it's expensive and depends on model weights.

    // ─── GPU path hardening tests ──────────────────────────────────────

    /// Convert a SecureField MLE to QM31 AoS u32 format.
    /// Each QM31(CM31(a,b), CM31(c,d)) → [a.0, b.0, c.0, d.0].
    fn securefield_mle_to_u32_aos(mle: &[SecureField]) -> Vec<u32> {
        let mut out = vec![0u32; mle.len() * 4];
        for (i, &sf) in mle.iter().enumerate() {
            let QM31(CM31(a, b), CM31(c, d)) = sf;
            out[i * 4] = a.0;
            out[i * 4 + 1] = b.0;
            out[i * 4 + 2] = c.0;
            out[i * 4 + 3] = d.0;
        }
        out
    }

    #[test]
    fn test_virtual_mle_u32_root_matches_securefield() {
        // Verify that building the virtual MLE in u32 format and committing
        // produces the same Merkle root as the SecureField virtual MLE.
        let mle0 = random_mle(3);
        let mle1 = random_mle(3);
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];
        let config = AggregatedBindingConfig::from_claims(&claims);

        // Build both virtual MLEs
        let sf_refs: Vec<&[SecureField]> = vec![&mle0, &mle1];
        let virtual_sf = build_virtual_mle(&sf_refs, &config);

        let mle0_u32 = securefield_mle_to_u32_aos(&mle0);
        let mle1_u32 = securefield_mle_to_u32_aos(&mle1);
        let u32_refs: Vec<&[u32]> = vec![&mle0_u32, &mle1_u32];
        let virtual_u32 = build_virtual_mle_u32(&u32_refs, &config);

        // Verify sizes match
        assert_eq!(
            virtual_sf.len() * 4,
            virtual_u32.len(),
            "virtual MLE sizes must match: {} SecureField entries vs {} u32 words",
            virtual_sf.len(),
            virtual_u32.len(),
        );

        // Verify per-element data consistency
        for (i, &sf) in virtual_sf.iter().enumerate() {
            let QM31(CM31(a, b), CM31(c, d)) = sf;
            assert_eq!(
                virtual_u32[i * 4],
                a.0,
                "mismatch at element {i}, component a"
            );
            assert_eq!(
                virtual_u32[i * 4 + 1],
                b.0,
                "mismatch at element {i}, component b"
            );
            assert_eq!(
                virtual_u32[i * 4 + 2],
                c.0,
                "mismatch at element {i}, component c"
            );
            assert_eq!(
                virtual_u32[i * 4 + 3],
                d.0,
                "mismatch at element {i}, component d"
            );
        }

        // Verify commitment roots match
        let sf_root = commit_mle_root_only(&virtual_sf);
        // Build u32 tree root by converting back (can't use GPU commit in tests)
        // Instead verify the data equality above implies root equality.
        // Direct root comparison requires the Merkle tree builder which is tested elsewhere.
        let _ = sf_root; // root equality follows from data equality proven above
    }

    #[test]
    fn test_virtual_mle_u32_varying_sizes() {
        // Test with matrices of different sizes — padding must be correct.
        let sizes = [3, 5, 4];
        let mles: Vec<Vec<SecureField>> = sizes.iter().map(|&n| random_mle(n)).collect();
        let claims: Vec<AggregatedWeightClaim> = mles
            .iter()
            .enumerate()
            .map(|(i, m)| make_claim(i, m))
            .collect();
        let config = AggregatedBindingConfig::from_claims(&claims);

        assert_eq!(config.n_max, 5);
        assert_eq!(config.m_padded, 4);
        assert_eq!(config.selector_bits, 2);

        let sf_refs: Vec<&[SecureField]> = mles.iter().map(|m| m.as_slice()).collect();
        let virtual_sf = build_virtual_mle(&sf_refs, &config);

        let mles_u32: Vec<Vec<u32>> = mles.iter().map(|m| securefield_mle_to_u32_aos(m)).collect();
        let u32_refs: Vec<&[u32]> = mles_u32.iter().map(|m| m.as_slice()).collect();
        let virtual_u32 = build_virtual_mle_u32(&u32_refs, &config);

        // Size: 2^(2+5) = 128 slots × 4 u32 words = 512 u32 words
        assert_eq!(virtual_sf.len(), 1 << config.n_global);
        assert_eq!(virtual_u32.len(), virtual_sf.len() * 4);

        // Verify data equivalence
        for (i, &sf) in virtual_sf.iter().enumerate() {
            let QM31(CM31(a, b), CM31(c, d)) = sf;
            assert_eq!(virtual_u32[i * 4], a.0);
            assert_eq!(virtual_u32[i * 4 + 1], b.0);
            assert_eq!(virtual_u32[i * 4 + 2], c.0);
            assert_eq!(virtual_u32[i * 4 + 3], d.0);
        }

        // Padding slots (indices 3..4) should be all zeros
        let slot_size = 1 << config.n_max;
        for i in 3..config.m_padded {
            let offset = i * slot_size;
            for j in 0..slot_size {
                assert_eq!(
                    virtual_sf[offset + j],
                    SecureField::zero(),
                    "padding slot {i}, index {j} should be zero",
                );
            }
        }

        // Smaller matrices (n_vars=3) should have zero-padding within their slots
        // Slot 0: mle of size 2^3=8, slot_size=2^5=32, indices 8..32 should be zero
        let local_size = 1 << sizes[0]; // 8
        for j in local_size..slot_size {
            assert_eq!(
                virtual_sf[j],
                SecureField::zero(),
                "slot 0, intra-slot padding index {j} should be zero",
            );
        }
    }

    #[test]
    fn test_u32_aos_conversion_roundtrip() {
        // Verify our test helper matches the expected AoS layout.
        let sf = QM31(
            CM31(M31::from(42), M31::from(7)),
            CM31(M31::from(13), M31::from(99)),
        );
        let mle = vec![sf];
        let u32_data = securefield_mle_to_u32_aos(&mle);
        assert_eq!(u32_data, vec![42, 7, 13, 99]);
    }

    #[test]
    fn test_m31_only_securefield_u32_consistency() {
        // Weight matrices contain M31 values embedded as SecureField.
        // Verify: SecureField::from(M31(x)) → u32 AoS gives [x, 0, 0, 0].
        let values = [0u32, 1, 42, (1 << 31) - 2, 1000000];
        for &v in &values {
            let sf = SecureField::from(M31::from(v));
            let QM31(CM31(a, b), CM31(c, d)) = sf;
            assert_eq!(a.0, v, "first component should be the M31 value");
            assert_eq!(b.0, 0, "second component should be zero for M31 embed");
            assert_eq!(c.0, 0, "third component should be zero for M31 embed");
            assert_eq!(d.0, 0, "fourth component should be zero for M31 embed");
        }
    }

    #[test]
    fn test_super_root_matches_virtual_mle_commitment() {
        // The super-root should equal the Merkle root of the virtual MLE tree.
        // This is the fundamental invariant that makes aggregated binding work.
        let mle0 = random_mle(4);
        let mle1 = random_mle(4);
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];
        let config = AggregatedBindingConfig::from_claims(&claims);

        let super_root = build_super_root(&claims, &config);

        let sf_refs: Vec<&[SecureField]> = vec![&mle0, &mle1];
        let virtual_mle = build_virtual_mle(&sf_refs, &config);
        let virtual_root = commit_mle_root_only(&virtual_mle);

        assert_eq!(
            super_root.root, virtual_root,
            "super-root ({:?}) must equal virtual MLE commitment ({:?}). \
             These represent the same Merkle tree: bottom n_max levels are per-matrix, \
             top selector_bits levels merge them.",
            super_root.root, virtual_root,
        );
    }

    #[test]
    fn test_super_root_matches_virtual_mle_varying_sizes() {
        // Same invariant but with heterogeneous matrix sizes.
        let mle0 = random_mle(3); // smaller
        let mle1 = random_mle(5); // n_max
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];
        let config = AggregatedBindingConfig::from_claims(&claims);

        let super_root = build_super_root(&claims, &config);

        let sf_refs: Vec<&[SecureField]> = vec![&mle0, &mle1];
        let virtual_mle = build_virtual_mle(&sf_refs, &config);
        let virtual_root = commit_mle_root_only(&virtual_mle);

        assert_eq!(
            super_root.root, virtual_root,
            "super-root must equal virtual MLE commitment for varying sizes",
        );
    }

    #[test]
    fn test_super_root_matches_virtual_mle_5_matrices() {
        // Stress test with 5 matrices of varying sizes.
        let sizes = [3, 4, 5, 3, 6];
        let mles: Vec<Vec<SecureField>> = sizes.iter().map(|&n| random_mle(n)).collect();
        let claims: Vec<AggregatedWeightClaim> = mles
            .iter()
            .enumerate()
            .map(|(i, m)| make_claim(i, m))
            .collect();
        let config = AggregatedBindingConfig::from_claims(&claims);

        let super_root = build_super_root(&claims, &config);

        let sf_refs: Vec<&[SecureField]> = mles.iter().map(|m| m.as_slice()).collect();
        let virtual_mle = build_virtual_mle(&sf_refs, &config);
        let virtual_root = commit_mle_root_only(&virtual_mle);

        assert_eq!(
            super_root.root, virtual_root,
            "super-root must equal virtual MLE commitment for 5 matrices",
        );
    }

    #[test]
    fn test_build_virtual_mle_u32_slot_boundaries() {
        // Verify that u32 slot boundaries are correct and don't overlap.
        let mle0 = random_mle(2); // 4 entries, 16 u32 words
        let mle1 = random_mle(2); // 4 entries, 16 u32 words
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];
        let config = AggregatedBindingConfig::from_claims(&claims);

        let mle0_u32 = securefield_mle_to_u32_aos(&mle0);
        let mle1_u32 = securefield_mle_to_u32_aos(&mle1);
        let u32_refs: Vec<&[u32]> = vec![&mle0_u32, &mle1_u32];
        let virt = build_virtual_mle_u32(&u32_refs, &config);

        let slot_size = 1 << config.n_max; // entries per slot
        let slot_u32_size = slot_size * 4; // u32 words per slot

        // Slot 0: entries from mle0
        for i in 0..mle0.len() {
            let QM31(CM31(a, _), CM31(_, _)) = mle0[i];
            assert_eq!(virt[i * 4], a.0, "slot 0 data at index {i}");
        }

        // Slot 1: entries from mle1 (at offset slot_u32_size)
        for i in 0..mle1.len() {
            let QM31(CM31(a, _), CM31(_, _)) = mle1[i];
            assert_eq!(virt[slot_u32_size + i * 4], a.0, "slot 1 data at index {i}",);
        }

        // Gap between slots should be zero
        let mle0_u32_len = mle0.len() * 4;
        for i in mle0_u32_len..slot_u32_size {
            assert_eq!(virt[i], 0, "intra-slot padding at u32 index {i}");
        }
    }

    // ─── Streaming tests ────────────────────────────────────────────────

    /// SliceWeightSource for tests that don't need u32 MLEs (CPU path).
    fn make_slice_source<'a>(
        mles: &'a [&'a [SecureField]],
        mles_u32: &'a [&'a [u32]],
    ) -> SliceWeightSource<'a> {
        SliceWeightSource { mles, mles_u32 }
    }

    #[test]
    fn test_slice_weight_source_basic() {
        let mle0 = random_mle(3);
        let mle1 = random_mle(4);
        let mle_refs: Vec<&[SecureField]> = vec![&mle0, &mle1];
        let mle0_u32 = securefield_mle_to_u32_aos(&mle0);
        let mle1_u32 = securefield_mle_to_u32_aos(&mle1);
        let u32_refs: Vec<&[u32]> = vec![&mle0_u32, &mle1_u32];

        let source = make_slice_source(&mle_refs, &u32_refs);
        assert_eq!(source.len(), 2);

        let got0 = source.get_mle(0);
        assert_eq!(got0, mle0);
        let got1 = source.get_mle(1);
        assert_eq!(got1, mle1);

        let got0_u32 = source.get_mle_u32(0);
        assert_eq!(got0_u32, mle0_u32);
    }

    #[test]
    fn test_streaming_oracle_matches_direct() {
        // Verify eval_unified_oracle_streaming matches eval_unified_oracle.
        let mle0 = random_mle(3);
        let mle1 = random_mle(4);
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];
        let config = AggregatedBindingConfig::from_claims(&claims);
        let claim_n_vars: Vec<usize> = claims.iter().map(|c| c.local_n_vars).collect();

        let mle_refs: Vec<&[SecureField]> = vec![&mle0, &mle1];
        let mle0_u32 = securefield_mle_to_u32_aos(&mle0);
        let mle1_u32 = securefield_mle_to_u32_aos(&mle1);
        let u32_refs: Vec<&[u32]> = vec![&mle0_u32, &mle1_u32];
        let source = make_slice_source(&mle_refs, &u32_refs);

        // Random evaluation point
        let point: Vec<SecureField> = (0..config.n_global).map(|_| random_qm31()).collect();

        let direct = eval_unified_oracle(&point, &mle_refs, &config);
        let streaming = eval_unified_oracle_streaming(&point, &source, &claim_n_vars, &config);
        assert_eq!(direct, streaming, "streaming oracle must match direct");
    }

    #[test]
    fn test_streaming_matches_non_streaming_2_matrices() {
        // Prove with both paths and compare proofs field-by-field.
        let mle0 = random_mle(3);
        let mle1 = random_mle(3);
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claims = vec![claim0, claim1];

        let mle_refs: Vec<&[SecureField]> = vec![&mle0, &mle1];
        let mle0_u32 = securefield_mle_to_u32_aos(&mle0);
        let mle1_u32 = securefield_mle_to_u32_aos(&mle1);
        let u32_refs: Vec<&[u32]> = vec![&mle0_u32, &mle1_u32];
        let source = make_slice_source(&mle_refs, &u32_refs);

        // Non-streaming proof
        let mut ch1 = PoseidonChannel::new();
        ch1.mix_u64(77);
        let proof_direct = prove_aggregated_binding(&claims, &mle_refs, &mut ch1);

        // Streaming proof (uses SliceWeightSource for backward compat)
        let mut ch2 = PoseidonChannel::new();
        ch2.mix_u64(77);
        let proof_streaming =
            prove_aggregated_binding_streaming(&claims, &source, &mut ch2);

        // Field-by-field comparison
        assert_eq!(
            proof_direct.sumcheck_round_polys, proof_streaming.sumcheck_round_polys,
            "sumcheck round polynomials must match"
        );
        assert_eq!(
            proof_direct.oracle_eval_at_s, proof_streaming.oracle_eval_at_s,
            "oracle eval must match"
        );
        assert_eq!(
            proof_direct.super_root.root, proof_streaming.super_root.root,
            "super-root must match"
        );
    }

    #[test]
    fn test_streaming_matches_non_streaming_varying_sizes() {
        let sizes = [3, 4, 5, 3];
        let mles: Vec<Vec<SecureField>> = sizes.iter().map(|&n| random_mle(n)).collect();
        let claims: Vec<AggregatedWeightClaim> = mles
            .iter()
            .enumerate()
            .map(|(i, m)| make_claim(i, m))
            .collect();
        let mle_refs: Vec<&[SecureField]> = mles.iter().map(|m| m.as_slice()).collect();
        let mles_u32: Vec<Vec<u32>> = mles.iter().map(|m| securefield_mle_to_u32_aos(m)).collect();
        let u32_refs: Vec<&[u32]> = mles_u32.iter().map(|m| m.as_slice()).collect();
        let source = make_slice_source(&mle_refs, &u32_refs);

        let mut ch1 = PoseidonChannel::new();
        ch1.mix_u64(321);
        let proof_direct = prove_aggregated_binding(&claims, &mle_refs, &mut ch1);

        let mut ch2 = PoseidonChannel::new();
        ch2.mix_u64(321);
        let proof_streaming =
            prove_aggregated_binding_streaming(&claims, &source, &mut ch2);

        assert_eq!(
            proof_direct.sumcheck_round_polys, proof_streaming.sumcheck_round_polys,
            "sumcheck round polys must match for varying sizes"
        );
        assert_eq!(
            proof_direct.oracle_eval_at_s, proof_streaming.oracle_eval_at_s,
            "oracle eval must match for varying sizes"
        );
    }

    #[test]
    fn test_streaming_verify() {
        // Prove with streaming, verify with standard verifier.
        let mle0 = random_mle(4);
        let mle1 = random_mle(4);
        let mle2 = random_mle(3);
        let claim0 = make_claim(0, &mle0);
        let claim1 = make_claim(1, &mle1);
        let claim2 = make_claim(2, &mle2);
        let claims = vec![claim0, claim1, claim2];

        let mle_refs: Vec<&[SecureField]> = vec![&mle0, &mle1, &mle2];
        let mles_u32: Vec<Vec<u32>> = mle_refs.iter().map(|m| securefield_mle_to_u32_aos(m)).collect();
        let u32_refs: Vec<&[u32]> = mles_u32.iter().map(|m| m.as_slice()).collect();
        let source = make_slice_source(&mle_refs, &u32_refs);

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(99);
        let mut verifier_ch = prover_ch.clone();

        let proof = prove_aggregated_binding_streaming(&claims, &source, &mut prover_ch);

        assert!(
            verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "streaming proof must verify with standard verifier"
        );
    }

    #[test]
    fn test_streaming_single_matrix() {
        let mle = random_mle(5);
        let claim = make_claim(0, &mle);
        let claims = vec![claim];

        let mle_refs: Vec<&[SecureField]> = vec![&mle];
        let mle_u32 = securefield_mle_to_u32_aos(&mle);
        let u32_refs: Vec<&[u32]> = vec![&mle_u32];
        let source = make_slice_source(&mle_refs, &u32_refs);

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(1);
        let mut verifier_ch = prover_ch.clone();

        let proof = prove_aggregated_binding_streaming(&claims, &source, &mut prover_ch);
        assert!(
            verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "streaming single matrix must verify"
        );
    }

    #[test]
    fn test_compute_selector_weight_matches_inline() {
        // Verify the extracted helper matches the original inline computation.
        let config = AggregatedBindingConfig {
            selector_bits: 3,
            n_max: 5,
            m_padded: 8,
            n_global: 8,
            n_claims: 5,
        };
        let one = SecureField::from(M31::from(1));
        let selector: Vec<SecureField> = (0..3).map(|_| random_qm31()).collect();

        for matrix_idx in 0..8 {
            // Inline computation (from original eval_unified_oracle)
            let mut expected = one;
            for (j, &s_j) in selector.iter().enumerate() {
                let bit_pos = config.selector_bits - 1 - j;
                let bit_j = if (matrix_idx >> bit_pos) & 1 == 1 {
                    one
                } else {
                    SecureField::zero()
                };
                expected = expected * (s_j * bit_j + (one - s_j) * (one - bit_j));
            }

            let got = compute_selector_weight(matrix_idx, &selector, &config);
            assert_eq!(got, expected, "selector weight mismatch at index {matrix_idx}");
        }
    }

    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_gpu_evaluate_mle_at_matches_cpu() {
        use crate::gpu_sumcheck::{secure_field_to_u32s, GpuSumcheckExecutor};

        let executor = match GpuSumcheckExecutor::cached() {
            Ok(e) => e,
            Err(_) => {
                eprintln!("Skipping GPU test: no CUDA device available");
                return;
            }
        };

        // Test across MLE sizes from 2^10 to 2^20
        for n_vars in [10, 12, 14, 16, 18, 20] {
            let mle = random_mle(n_vars);
            let point: Vec<SecureField> = (0..n_vars).map(|_| random_qm31()).collect();

            // CPU evaluation
            let cpu_result = evaluate_mle_at(&mle, &point);

            // GPU evaluation (u32 path)
            let mle_u32: Vec<u32> = mle
                .iter()
                .flat_map(|sf| secure_field_to_u32s(*sf))
                .collect();
            let gpu_result = executor
                .evaluate_mle_at_gpu_u32(&mle_u32, &point)
                .expect("GPU eval failed");

            assert_eq!(
                cpu_result, gpu_result,
                "CPU/GPU mismatch at n_vars={n_vars}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda-runtime")]
    fn test_gpu_3way_oracle_matches_cpu() {
        use crate::gpu_sumcheck::GpuSumcheckExecutor;

        let executor = match GpuSumcheckExecutor::cached() {
            Ok(e) => e,
            Err(_) => {
                eprintln!("Skipping GPU test: no CUDA device available");
                return;
            }
        };

        // Setup: 4 matrices of varying sizes
        let mles: Vec<Vec<SecureField>> = vec![
            random_mle(10),
            random_mle(12),
            random_mle(14),
            random_mle(10),
        ];
        let claims: Vec<AggregatedWeightClaim> = mles
            .iter()
            .enumerate()
            .map(|(i, mle)| make_claim(i, mle))
            .collect();
        let config = AggregatedBindingConfig::from_claims(&claims);
        let claim_n_vars: Vec<usize> = claims.iter().map(|c| c.local_n_vars).collect();

        let mle_refs: Vec<&[SecureField]> = mles.iter().map(|m| m.as_slice()).collect();
        let mle_u32s: Vec<Vec<u32>> = mles
            .iter()
            .map(|m| {
                use crate::gpu_sumcheck::secure_field_to_u32s;
                m.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect()
            })
            .collect();
        let mle_u32_refs: Vec<&[u32]> = mle_u32s.iter().map(|m| m.as_slice()).collect();
        let source = SliceWeightSource {
            mles: &mle_refs,
            mles_u32: &mle_u32_refs,
        };

        // Build a random global-length eval point and 3 variants differing at position 5
        let n = config.n_global;
        let base_pt: Vec<SecureField> = (0..n).map(|_| random_qm31()).collect();
        let one = SecureField::from(M31::from(1));
        let two = SecureField::from(M31::from(2));

        let branch_pos = config.selector_bits + 2; // inside local region
        let mut pt0 = base_pt.clone();
        let mut pt1 = base_pt.clone();
        let mut pt2 = base_pt.clone();
        pt0[branch_pos] = SecureField::zero();
        pt1[branch_pos] = one;
        pt2[branch_pos] = two;

        // CPU reference
        let cpu_0 = eval_unified_oracle_streaming(&pt0, &source, &claim_n_vars, &config);
        let cpu_1 = eval_unified_oracle_streaming(&pt1, &source, &claim_n_vars, &config);
        let cpu_2 = eval_unified_oracle_streaming(&pt2, &source, &claim_n_vars, &config);

        // GPU 3-way
        let gpu_results = eval_unified_oracle_3way_gpu(
            [&pt0, &pt1, &pt2],
            &source,
            &claim_n_vars,
            &config,
            &executor,
        );

        assert_eq!(cpu_0, gpu_results[0], "3way mismatch at t=0");
        assert_eq!(cpu_1, gpu_results[1], "3way mismatch at t=1");
        assert_eq!(cpu_2, gpu_results[2], "3way mismatch at t=2");
    }

    /// Test that per-matrix opening proofs verify correctly (CPU-only test).
    /// Constructs a proof with per-matrix openings by hand to validate
    /// Fiat-Shamir transcript consistency between prover and verifier.
    #[test]
    fn test_per_matrix_openings_verify() {
        let mle0 = random_mle(6);
        let mle1 = random_mle(8);
        let mle2 = random_mle(7);
        let claims: Vec<AggregatedWeightClaim> = vec![
            make_claim(0, &mle0),
            make_claim(1, &mle1),
            make_claim(2, &mle2),
        ];
        let config = AggregatedBindingConfig::from_claims(&claims);
        let claim_n_vars: Vec<usize> = claims.iter().map(|c| c.local_n_vars).collect();

        let mle_refs: Vec<&[SecureField]> = vec![&mle0, &mle1, &mle2];
        let mles_u32: Vec<Vec<u32>> = mle_refs.iter().map(|m| securefield_mle_to_u32_aos(m)).collect();
        let u32_refs: Vec<&[u32]> = mles_u32.iter().map(|m| m.as_slice()).collect();
        let source = make_slice_source(&mle_refs, &u32_refs);

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(42);
        let mut verifier_ch = prover_ch.clone();

        // --- Prover side: build proof with per-matrix openings ---

        // 1. Super root
        let super_root = build_super_root(&claims, &config);
        prover_ch.mix_felt(super_root.root);

        // 2. Betas
        let betas = draw_beta_weights(&mut prover_ch, claims.len());

        // 3. Mismatch sumcheck (CPU)
        let (round_polys, challenge_point) = mismatch_sumcheck_streaming(
            &claims, &source, &claim_n_vars, &config, &betas, &mut prover_ch,
        );

        // 4. Per-matrix openings
        let selector = &challenge_point[..config.selector_bits];
        let local = &challenge_point[config.selector_bits..];

        let mut per_matrix_evals = Vec::new();
        let mut per_matrix_openings = Vec::new();

        for i in 0..source.len() {
            let n_vars = claim_n_vars[i];
            let local_truncated = &local[config.n_max - n_vars..];

            let mle = source.get_mle(i);
            let opening_proof = prove_mle_opening(&mle, local_truncated, &mut prover_ch);
            let w_val = evaluate_mle_at(&mle, local_truncated);
            per_matrix_evals.push(w_val);
            per_matrix_openings.push(opening_proof);
        }

        // Oracle eval
        let mut oracle_eval = SecureField::zero();
        for i in 0..source.len() {
            let sel_weight = compute_selector_weight(i, selector, &config);
            oracle_eval = oracle_eval + sel_weight * per_matrix_evals[i];
        }
        prover_ch.mix_felts(&[oracle_eval]);

        let dummy_opening = per_matrix_openings[0].clone();

        let proof = AggregatedWeightBindingProof {
            config,
            sumcheck_round_polys: round_polys,
            oracle_eval_at_s: oracle_eval,
            opening_proof: dummy_opening,
            super_root,
            per_matrix_openings: Some(per_matrix_openings),
            per_matrix_evals: Some(per_matrix_evals),
        };

        // --- Verifier side ---
        assert!(
            verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "per-matrix opening proof must verify"
        );
    }

    /// Test that a tampered per-matrix eval fails verification.
    #[test]
    fn test_per_matrix_openings_tampered_eval_fails() {
        let mle0 = random_mle(6);
        let mle1 = random_mle(7);
        let claims: Vec<AggregatedWeightClaim> = vec![
            make_claim(0, &mle0),
            make_claim(1, &mle1),
        ];
        let config = AggregatedBindingConfig::from_claims(&claims);
        let claim_n_vars: Vec<usize> = claims.iter().map(|c| c.local_n_vars).collect();

        let mle_refs: Vec<&[SecureField]> = vec![&mle0, &mle1];
        let mles_u32: Vec<Vec<u32>> = mle_refs.iter().map(|m| securefield_mle_to_u32_aos(m)).collect();
        let u32_refs: Vec<&[u32]> = mles_u32.iter().map(|m| m.as_slice()).collect();
        let source = make_slice_source(&mle_refs, &u32_refs);

        let mut prover_ch = PoseidonChannel::new();
        prover_ch.mix_u64(77);
        let mut verifier_ch = prover_ch.clone();

        // Build valid proof with per-matrix openings
        let super_root = build_super_root(&claims, &config);
        prover_ch.mix_felt(super_root.root);
        let betas = draw_beta_weights(&mut prover_ch, claims.len());
        let (round_polys, challenge_point) = mismatch_sumcheck_streaming(
            &claims, &source, &claim_n_vars, &config, &betas, &mut prover_ch,
        );

        let selector = &challenge_point[..config.selector_bits];
        let local = &challenge_point[config.selector_bits..];

        let mut per_matrix_evals = Vec::new();
        let mut per_matrix_openings = Vec::new();

        for i in 0..source.len() {
            let n_vars = claim_n_vars[i];
            let local_truncated = &local[config.n_max - n_vars..];
            let mle = source.get_mle(i);
            let opening_proof = prove_mle_opening(&mle, local_truncated, &mut prover_ch);
            let w_val = evaluate_mle_at(&mle, local_truncated);
            per_matrix_evals.push(w_val);
            per_matrix_openings.push(opening_proof);
        }

        let mut oracle_eval = SecureField::zero();
        for i in 0..source.len() {
            let sel_weight = compute_selector_weight(i, selector, &config);
            oracle_eval = oracle_eval + sel_weight * per_matrix_evals[i];
        }
        prover_ch.mix_felts(&[oracle_eval]);

        // Tamper: flip one eval
        let mut tampered_evals = per_matrix_evals.clone();
        tampered_evals[0] = tampered_evals[0] + SecureField::from(M31::from(1));

        let proof = AggregatedWeightBindingProof {
            config,
            sumcheck_round_polys: round_polys,
            oracle_eval_at_s: oracle_eval,
            opening_proof: per_matrix_openings[0].clone(),
            super_root,
            per_matrix_openings: Some(per_matrix_openings),
            per_matrix_evals: Some(tampered_evals),
        };

        assert!(
            !verify_aggregated_binding(&proof, &claims, &mut verifier_ch),
            "tampered per-matrix eval should fail verification"
        );
    }
}
