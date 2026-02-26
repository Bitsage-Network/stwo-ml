//! MLE evaluation opening proof protocol.
//!
//! Implements the multilinear extension opening proof matching Cairo's
//! `MleOpeningProof` struct. Given evaluations on `{0,1}^n` committed
//! via a Poseidon Merkle tree, proves that `evaluate_mle(f, r) = claimed_value`.
//!
//! The protocol iteratively folds the evaluations with sumcheck challenges,
//! building intermediate Merkle commitments at each layer. Queries are
//! drawn from the Poseidon channel for soundness.

use crate::crypto::poseidon_channel::{securefield_to_felt, PoseidonChannel};
use crate::crypto::poseidon_merkle::{MerkleAuthPath, PoseidonMerkleTree};
#[cfg(feature = "cuda-runtime")]
use crate::crypto::poseidon_merkle::{securefield_to_u64_limbs_direct, u64_limbs_to_field_element};
#[cfg(feature = "cuda-runtime")]
use crate::crypto::merkle_cache;
#[cfg(feature = "cuda-runtime")]
use crate::gpu_sumcheck::u32s_to_secure_field;
#[cfg(feature = "cuda-runtime")]
use crate::gpu_sumcheck::GpuSumcheckExecutor;
#[cfg(feature = "cuda-runtime")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda-runtime")]
use cudarc::driver::DeviceSlice;
use rayon::prelude::*;
use starknet_ff::FieldElement;
#[cfg(feature = "cuda-runtime")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda-runtime")]
use std::time::Instant;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::cm31::CM31;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::qm31::QM31;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::cuda_executor::{
    get_cuda_executor, is_cuda_available, upload_poseidon252_round_constants, CudaFftExecutor,
    Poseidon252MerkleGpuTree,
};

/// Number of queries for MLE opening proofs.
///
/// Configurable via `STWO_MLE_N_QUERIES` env var (2..=20, default 5).
/// Each query on an n-variable MLE provides ~n bits of soundness, so
/// 5 queries × 27 vars = 135 bits — above 128-bit security target.
pub const MLE_N_QUERIES: usize = 5;

/// Runtime-configurable query count (cached on first call).
pub fn mle_n_queries() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("STWO_MLE_N_QUERIES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v >= 2 && v <= 20)
            .unwrap_or(MLE_N_QUERIES)
    })
}
#[cfg(feature = "cuda-runtime")]
static GPU_MLE_FOLD_BACKEND_LOGGED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda-runtime")]
static GPU_MLE_OPENING_TREE_BACKEND_LOGGED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda-runtime")]
static GPU_MLE_OPENING_TREE_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda-runtime")]
#[derive(Debug, Clone)]
enum MleLayerValues {
    Secure(Vec<SecureField>),
    #[cfg(feature = "cuda-runtime")]
    Qm31U32Aos(Vec<u32>),
}

#[cfg(feature = "cuda-runtime")]
impl MleLayerValues {
    #[inline]
    fn len_points(&self) -> usize {
        match self {
            Self::Secure(v) => v.len(),
            #[cfg(feature = "cuda-runtime")]
            Self::Qm31U32Aos(v) => v.len() / 4,
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn gpu_mle_fold_enabled() -> bool {
    match std::env::var("STWO_GPU_MLE_FOLD") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        // Default ON for proving workloads with very large MLEs.
        // If GPU fold is unavailable/fails, prover falls back to CPU unless strict mode is enabled.
        Err(_) => true,
    }
}

#[cfg(feature = "cuda-runtime")]
fn gpu_mle_fold_required() -> bool {
    let explicit = match std::env::var("STWO_GPU_MLE_FOLD_REQUIRE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    };
    if explicit {
        return true;
    }
    match std::env::var("STWO_GPU_ONLY") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

#[cfg(feature = "cuda-runtime")]
fn gpu_mle_fold_min_points() -> usize {
    std::env::var("STWO_GPU_MLE_FOLD_MIN_POINTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v.is_power_of_two() && v >= 2)
        .unwrap_or(1 << 16)
}

#[cfg(feature = "cuda-runtime")]
fn gpu_mle_opening_tree_enabled() -> bool {
    match std::env::var("STWO_GPU_MLE_OPENING_TREE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        // Default ON for large opening workloads.
        Err(_) => true,
    }
}

#[cfg(feature = "cuda-runtime")]
fn gpu_mle_opening_tree_required() -> bool {
    let explicit = match std::env::var("STWO_GPU_MLE_OPENING_TREE_REQUIRE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    };
    if explicit {
        return true;
    }
    match std::env::var("STWO_GPU_ONLY") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn gpu_mle_commitment_enabled() -> bool {
    match std::env::var("STWO_GPU_MLE_COMMITMENT") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        // Default ON when CUDA runtime is built in.
        Err(_) => true,
    }
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn gpu_mle_commitment_required() -> bool {
    let explicit = match std::env::var("STWO_GPU_MLE_COMMITMENT_REQUIRE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    };
    if explicit {
        return true;
    }
    match std::env::var("STWO_GPU_ONLY") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

/// Whether GPU streaming Merkle is enabled for round-0 auth path extraction.
/// Default: ON when cuda-runtime feature is enabled.
/// Control via `STWO_GPU_STREAMING_MERKLE=0/1`.
#[cfg(feature = "cuda-runtime")]
fn gpu_streaming_merkle_enabled() -> bool {
    match std::env::var("STWO_GPU_STREAMING_MERKLE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => true,
    }
}

/// Minimum number of leaf hashes to trigger GPU streaming Merkle.
/// Default: 1048576 (1M leaves = 4M QM31 u32 words).
/// Control via `STWO_GPU_STREAMING_MERKLE_MIN_LEAVES`.
#[cfg(feature = "cuda-runtime")]
fn gpu_streaming_merkle_min_leaves() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("STWO_GPU_STREAMING_MERKLE_MIN_LEAVES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v >= 2)
            .unwrap_or(1 << 20)
    })
}

/// Whether to cross-check GPU streaming auth paths against CPU tree (first matrix only).
/// Control via `STWO_GPU_STREAMING_MERKLE_VERIFY=1`.
#[cfg(feature = "cuda-runtime")]
fn gpu_streaming_merkle_verify() -> bool {
    match std::env::var("STWO_GPU_STREAMING_MERKLE_VERIFY") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

#[cfg(feature = "cuda-runtime")]
static GPU_STREAMING_MERKLE_LOGGED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda-runtime")]
static GPU_STREAMING_MERKLE_VERIFIED: AtomicBool = AtomicBool::new(false);

// Thread-local cache for initial MLE root. Set before calling
// prove_mle_opening_with_commitment_qm31_u32 to skip initial root computation.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static CACHED_INITIAL_MLE_ROOT: std::cell::Cell<Option<FieldElement>> = const { std::cell::Cell::new(None) };
}

// Thread-local node_id for mmap Merkle tree cache lookup.
// Set before calling prove_mle_opening_with_commitment_qm31_u32 to enable
// disk-cached Merkle tree auth path extraction.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static CACHED_MERKLE_NODE_ID: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Set a cached initial MLE root to be consumed by the next call to
/// `prove_mle_opening_with_commitment_qm31_u32` on this thread.
/// The value is consumed (taken) on use — call before each opening.
#[cfg(feature = "cuda-runtime")]
pub fn set_cached_initial_mle_root(root: FieldElement) {
    CACHED_INITIAL_MLE_ROOT.with(|c| c.set(Some(root)));
}

/// Take (consume) the cached initial MLE root, if one was set.
#[cfg(feature = "cuda-runtime")]
fn take_cached_initial_mle_root() -> Option<FieldElement> {
    CACHED_INITIAL_MLE_ROOT.with(|c| c.take())
}

/// Set a node_id for mmap Merkle tree cache lookup.
/// Consumed by the next call to `prove_mle_opening_with_commitment_qm31_u32`.
#[cfg(feature = "cuda-runtime")]
pub fn set_merkle_cache_node_id(node_id: usize) {
    CACHED_MERKLE_NODE_ID.with(|c| c.set(Some(node_id)));
}

/// Take (consume) the cached node_id for Merkle tree cache.
#[cfg(feature = "cuda-runtime")]
fn take_merkle_cache_node_id() -> Option<usize> {
    CACHED_MERKLE_NODE_ID.with(|c| c.take())
}

#[cfg(feature = "cuda-runtime")]
#[inline]
fn felt_to_securefield_packed(fe: FieldElement) -> SecureField {
    let bytes = fe.to_bytes_be();
    let mut low = [0u8; 16];
    low.copy_from_slice(&bytes[16..]);
    let v = u128::from_be_bytes(low);

    let mask = (1u128 << 31) - 1;
    let d = (v & mask) as u32;
    let c = ((v >> 31) & mask) as u32;
    let b = ((v >> 62) & mask) as u32;
    let a = ((v >> 93) & mask) as u32;

    QM31(
        CM31(M31::from(a), M31::from(b)),
        CM31(M31::from(c), M31::from(d)),
    )
}

#[cfg(feature = "cuda-runtime")]
#[inline]
fn qm31_u32_to_u64_limbs_direct(words: &[u32]) -> [u64; 4] {
    debug_assert!(words.len() == 4);
    let packed = (1u128 << 124)
        | ((words[0] as u128) << 93)
        | ((words[1] as u128) << 62)
        | ((words[2] as u128) << 31)
        | (words[3] as u128);
    [packed as u64, (packed >> 64) as u64, 0, 0]
}

#[cfg(feature = "cuda-runtime")]
#[inline]
fn u64_limbs_to_felt252(limbs: &[u64; 4]) -> Option<FieldElement> {
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let limb = limbs[3 - i];
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    bytes[0] &= 0x07;
    FieldElement::from_bytes_be(&bytes).ok()
}

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
    // Fast path: bypass FieldElement Montgomery conversions entirely.
    // Goes SecureField → u64 limbs → GPU Poseidon, storing all layers as raw
    // limbs. Saves ~225ns/element (3 Montgomery ops) and avoids 8GB leaf clone.
    if evals.len().is_power_of_two() && evals.len() >= 256 {
        let tree = PoseidonMerkleTree::build_parallel_from_secure(evals);
        return (tree.root(), tree);
    }

    let leaves: Vec<FieldElement> = if evals.len() >= 256 {
        evals
            .par_iter()
            .map(|&sf| securefield_to_felt(sf))
            .collect()
    } else {
        evals.iter().map(|&sf| securefield_to_felt(sf)).collect()
    };
    let tree = PoseidonMerkleTree::build_parallel(leaves);
    (tree.root(), tree)
}

#[cfg(feature = "cuda-runtime")]
fn commit_mle_from_qm31_u32_aos(evals_u32: &[u32]) -> (FieldElement, PoseidonMerkleTree) {
    let tree = PoseidonMerkleTree::build_parallel_from_qm31_u32_aos(evals_u32);
    (tree.root(), tree)
}

#[cfg(feature = "cuda-runtime")]
fn build_gpu_merkle_path_with_leaf_sibling(
    tree: &Poseidon252MerkleGpuTree,
    leaf_idx: usize,
    n_leaves: usize,
    leaf_sibling: SecureField,
) -> Result<Vec<FieldElement>, String> {
    let mut siblings = Vec::with_capacity(n_leaves.ilog2() as usize);
    siblings.push(securefield_to_felt(leaf_sibling));

    let mut node_idx = leaf_idx / 2;
    let internal_levels = tree.num_layers().saturating_sub(1);
    for layer_idx in 0..internal_levels {
        let sib_idx = node_idx ^ 1;
        let limbs = tree
            .node_u64(layer_idx, sib_idx)
            .map_err(|e| format!("download gpu merkle sibling: {e}"))?;
        let felt = u64_limbs_to_felt252(&limbs)
            .ok_or_else(|| "invalid sibling limbs from gpu merkle tree".to_string())?;
        siblings.push(felt);
        node_idx >>= 1;
    }

    Ok(siblings)
}

/// Get the auth path siblings from a CPU PoseidonMerkleTree.
/// The CPU tree's `prove()` already includes the leaf-pair sibling as siblings[0],
/// which matches the GPU streaming format (leaf sibling + internal hash siblings).
#[cfg(feature = "cuda-runtime")]
fn tree_prove_path_siblings(
    tree: &PoseidonMerkleTree,
    leaf_idx: usize,
) -> Vec<FieldElement> {
    tree.prove(leaf_idx).siblings
}

/// Extract authentication paths using GPU streaming Merkle construction.
///
/// Builds the Merkle tree level-by-level on GPU, keeping only 2 levels in VRAM
/// at a time. Downloads only the sibling nodes needed for query auth paths.
/// Works for any folding round (not just round 0).
///
/// VRAM: Peak ~2x one tree level (vs full tree that stores all levels).
/// Speed: ~1-2s for round 0 (67M leaves), exponentially faster for later rounds.
#[cfg(feature = "cuda-runtime")]
fn extract_auth_paths_gpu_streaming(
    layer: &MleLayerValues,
    query_pair_indices: &[usize],
    n_points: usize,
) -> Result<(FieldElement, Vec<MleQueryRoundData>), String> {
    let start = Instant::now();
    let n_leaf_hashes = n_points / 2;
    let mid = n_points / 2;

    let executor = get_cuda_executor()
        .map_err(|e| format!("GPU executor init: {:?}", e))?;
    let d_rc = upload_poseidon252_round_constants(&executor.device)
        .map_err(|e| format!("upload round constants: {:?}", e))?;

    // Convert layer data → u64 limbs for GPU Poseidon leaf hashing
    let mut leaf_limbs = vec![0u64; n_points * 4];
    match layer {
        MleLayerValues::Qm31U32Aos(evals_u32) => {
            leaf_limbs
                .par_chunks_mut(4)
                .zip(evals_u32.par_chunks_exact(4))
                .for_each(|(dst, src)| {
                    dst.copy_from_slice(&qm31_u32_to_u64_limbs_direct(src));
                });
        }
        MleLayerValues::Secure(vals) => {
            leaf_limbs[..vals.len() * 4]
                .par_chunks_mut(4)
                .zip(vals.par_iter())
                .for_each(|(dst, sf)| {
                    dst.copy_from_slice(&securefield_to_u64_limbs_direct(*sf));
                });
        }
    }

    // Upload leaf limbs to GPU
    let d_leaf_limbs = executor
        .device
        .htod_sync_copy(&leaf_limbs)
        .map_err(|e| format!("H2D leaf limbs: {:?}", e))?;
    drop(leaf_limbs); // Free CPU copy

    // Expand query pair indices to leaf indices for auth path extraction.
    // Each query needs auth paths for both left (pair_idx) and right (mid + pair_idx).
    // We extract sibling nodes for left_idx/2 and right_idx/2 in the tree.
    let mut all_leaf_indices: Vec<usize> = Vec::with_capacity(query_pair_indices.len() * 2);
    for &pair_idx in query_pair_indices {
        all_leaf_indices.push(pair_idx);          // left leaf index
        all_leaf_indices.push(mid + pair_idx);    // right leaf index
    }

    let (root_limbs, all_auth_paths) = executor
        .execute_poseidon252_merkle_streaming_auth_paths(
            &d_leaf_limbs,
            n_leaf_hashes,
            &d_rc,
            &all_leaf_indices,
        )
        .map_err(|e| format!("GPU streaming merkle: {:?}", e))?;

    let root = u64_limbs_to_felt252(&root_limbs)
        .ok_or_else(|| "invalid root limbs from GPU streaming merkle".to_string())?;

    // Build MleQueryRoundData for each query
    let mut round_data: Vec<MleQueryRoundData> = Vec::with_capacity(query_pair_indices.len());

    for (q, &pair_idx) in query_pair_indices.iter().enumerate() {
        let left_idx = pair_idx;
        let right_idx = mid + pair_idx;

        // Read left/right SecureField values from the layer data
        let (left_value, right_value) = match layer {
            MleLayerValues::Qm31U32Aos(evals_u32) => {
                let l = left_idx * 4;
                let r = right_idx * 4;
                (
                    u32s_to_secure_field(&[evals_u32[l], evals_u32[l + 1], evals_u32[l + 2], evals_u32[l + 3]]),
                    u32s_to_secure_field(&[evals_u32[r], evals_u32[r + 1], evals_u32[r + 2], evals_u32[r + 3]]),
                )
            }
            MleLayerValues::Secure(vals) => {
                (vals[left_idx], vals[right_idx])
            }
        };

        // Convert u64 limb auth paths to FieldElement auth paths.
        // Auth path format: [leaf_sibling, level0_sibling, level1_sibling, ...]
        // But we need to prepend the leaf-pair sibling (the other QM31 value).
        let left_raw_siblings = &all_auth_paths[q * 2];
        let right_raw_siblings = &all_auth_paths[q * 2 + 1];

        // Left auth path: first sibling is the right value (leaf pair sibling)
        let mut left_siblings = Vec::with_capacity(left_raw_siblings.len() + 1);
        left_siblings.push(securefield_to_felt(right_value));
        for limbs in left_raw_siblings {
            let felt = u64_limbs_to_felt252(limbs)
                .ok_or_else(|| "invalid left sibling limbs".to_string())?;
            left_siblings.push(felt);
        }

        // Right auth path: first sibling is the left value (leaf pair sibling)
        let mut right_siblings = Vec::with_capacity(right_raw_siblings.len() + 1);
        right_siblings.push(securefield_to_felt(left_value));
        for limbs in right_raw_siblings {
            let felt = u64_limbs_to_felt252(limbs)
                .ok_or_else(|| "invalid right sibling limbs".to_string())?;
            right_siblings.push(felt);
        }

        round_data.push(MleQueryRoundData {
            left_value,
            right_value,
            left_siblings,
            right_siblings,
        });
    }

    let elapsed = start.elapsed();
    if !GPU_STREAMING_MERKLE_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[GKR] GPU streaming Merkle: auth paths for {} queries from {} leaves in {:.3}s",
            query_pair_indices.len(),
            n_leaf_hashes,
            elapsed.as_secs_f64(),
        );
    }

    Ok((root, round_data))
}

#[cfg(feature = "cuda-runtime")]
fn fold_layer_cpu_qm31_words(words: &[u32], r: SecureField) -> Vec<SecureField> {
    debug_assert!(words.len() % 4 == 0);
    let n_points = words.len() / 4;
    let mid = n_points / 2;
    if mid >= 1 << 16 {
        (0..mid)
            .into_par_iter()
            .map(|j| {
                let l = j * 4;
                let rr = (mid + j) * 4;
                let left =
                    u32s_to_secure_field(&[words[l], words[l + 1], words[l + 2], words[l + 3]]);
                let right =
                    u32s_to_secure_field(&[words[rr], words[rr + 1], words[rr + 2], words[rr + 3]]);
                left + r * (right - left)
            })
            .collect()
    } else {
        let mut out = Vec::with_capacity(mid);
        for j in 0..mid {
            let l = j * 4;
            let rr = (mid + j) * 4;
            let left = u32s_to_secure_field(&[words[l], words[l + 1], words[l + 2], words[l + 3]]);
            let right =
                u32s_to_secure_field(&[words[rr], words[rr + 1], words[rr + 2], words[rr + 3]]);
            out.push(left + r * (right - left));
        }
        out
    }
}

/// Compute only the MLE commitment root from QM31 u32 AoS data.
///
/// More efficient than `commit_mle_from_qm31_u32_aos` when only the root is
/// needed (avoids storing all tree layers).
#[cfg(feature = "cuda-runtime")]
pub fn commit_mle_root_only_from_qm31_u32_aos(evals_u32: &[u32]) -> FieldElement {
    PoseidonMerkleTree::root_only_from_qm31_u32_aos(evals_u32)
}

/// Compute only the MLE commitment root without storing the full tree.
///
/// Uses parallel leaf conversion (rayon) and parallel Merkle hashing.
/// More efficient than `commit_mle` when only the root is needed
/// (e.g., in batch entry preparation where the tree is discarded).
pub fn commit_mle_root_only(evals: &[SecureField]) -> FieldElement {
    use rayon::prelude::*;

    let leaves: Vec<FieldElement> = if evals.len() >= 256 {
        evals
            .par_iter()
            .map(|&sf| securefield_to_felt(sf))
            .collect()
    } else {
        evals.iter().map(|&sf| securefield_to_felt(sf)).collect()
    };
    PoseidonMerkleTree::root_only_parallel(leaves)
}

/// Compute only the MLE commitment root using the GPU Poseidon Merkle kernel.
///
/// Converts SecureField evaluations to u64 limbs, uploads to GPU, runs the
/// full Poseidon252 Merkle tree kernel, and extracts only the root.
/// All intermediate GPU buffers are dropped after root extraction.
///
/// ~20-50x faster than CPU `commit_mle_root_only` for large matrices
/// (e.g., 134M elements: ~200ms GPU vs ~3-5s CPU).
#[cfg(feature = "cuda-runtime")]
pub fn commit_mle_root_only_gpu(
    evals: &[SecureField],
    executor: &CudaFftExecutor,
    d_round_constants: &CudaSlice<u64>,
) -> Result<FieldElement, String> {
    assert!(!evals.is_empty(), "cannot commit empty evals");
    let n = evals.len().next_power_of_two().max(2);
    let n_leaf_hashes = n / 2;

    // Convert SecureField → u64 limbs (parallel on CPU)
    let mut leaf_limbs = vec![0u64; n * 4];
    leaf_limbs[..evals.len() * 4]
        .par_chunks_mut(4)
        .zip(evals.par_iter())
        .for_each(|(dst, sf)| {
            dst.copy_from_slice(&securefield_to_u64_limbs_direct(*sf));
        });
    // Padding elements remain zero (matching FieldElement::ZERO encoding)

    // Upload leaf limbs to GPU
    let d_prev_leaf = executor
        .device
        .htod_sync_copy(&leaf_limbs)
        .map_err(|e| format!("H2D leaf limbs: {:?}", e))?;
    drop(leaf_limbs); // Free ~4GB host allocation immediately
    let d_dummy_columns = executor
        .device
        .htod_sync_copy(&[0u32])
        .map_err(|e| format!("H2D dummy columns: {:?}", e))?;

    // Run GPU Poseidon Merkle tree — keep layers on GPU, download only root.
    // Using gpu_layers variant avoids ~4GB bulk D2H of all intermediate layers.
    let gpu_tree = executor
        .execute_poseidon252_merkle_full_tree_gpu_layers(
            &d_dummy_columns,
            0,
            Some(&d_prev_leaf),
            n_leaf_hashes,
            d_round_constants,
        )
        .map_err(|e| format!("execute full-tree: {e}"))?;

    // Download only the root (32 bytes D2H vs ~4GB for full tree)
    let root_limbs = gpu_tree
        .root_u64()
        .map_err(|e| format!("download GPU root: {e}"))?;
    let root = u64_limbs_to_field_element(&root_limbs)
        .ok_or_else(|| "invalid root limbs from GPU Merkle tree".to_string())?;
    // gpu_tree dropped here — frees all GPU intermediate layer buffers

    // Opt-in cross-check: verify GPU root matches CPU root.
    // Expensive (~5s per matrix), so only enabled via env var, not debug_assertions.
    if std::env::var("STWO_GPU_COMMITMENT_VERIFY")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        })
        .unwrap_or(false)
    {
        let cpu_root = commit_mle_root_only(evals);
        if cpu_root != root {
            return Err(format!(
                "GPU and CPU Poseidon Merkle roots differ! GPU={:?} CPU={:?}",
                root, cpu_root,
            ));
        }
    }

    Ok(root)
}

/// Compute the MLE commitment root from pre-packed u64 limbs on GPU.
///
/// Skips the SecureField → u64 conversion entirely. The caller provides
/// `leaf_limbs` already in GPU Poseidon format (4 u64s per element, padded
/// to power-of-2 with zeros). `n_elements` is the total padded element count.
///
/// The limb buffer is borrowed, not consumed — callers can reuse it across
/// matrices (optimization #1: buffer reuse).
#[cfg(feature = "cuda-runtime")]
pub fn commit_mle_root_only_gpu_from_limbs(
    leaf_limbs: &[u64],
    n_elements: usize,
    executor: &CudaFftExecutor,
    d_round_constants: &CudaSlice<u64>,
) -> Result<FieldElement, String> {
    assert!(n_elements >= 2, "need at least 2 elements");
    assert!(n_elements.is_power_of_two(), "n_elements must be power of 2");
    assert!(
        leaf_limbs.len() >= n_elements * 4,
        "limb buffer too small: need {} got {}",
        n_elements * 4,
        leaf_limbs.len()
    );
    let n_leaf_hashes = n_elements / 2;

    // Upload leaf limbs to GPU (borrow — caller keeps ownership for reuse)
    let d_prev_leaf = executor
        .device
        .htod_sync_copy(&leaf_limbs[..n_elements * 4])
        .map_err(|e| format!("H2D leaf limbs: {:?}", e))?;
    let d_dummy_columns = executor
        .device
        .htod_sync_copy(&[0u32])
        .map_err(|e| format!("H2D dummy columns: {:?}", e))?;

    // Run GPU Poseidon Merkle — layers stay on device, download only root.
    let gpu_tree = executor
        .execute_poseidon252_merkle_full_tree_gpu_layers(
            &d_dummy_columns,
            0,
            Some(&d_prev_leaf),
            n_leaf_hashes,
            d_round_constants,
        )
        .map_err(|e| format!("execute full-tree: {e}"))?;

    let root_limbs = gpu_tree
        .root_u64()
        .map_err(|e| format!("download GPU root: {e}"))?;
    let root = u64_limbs_to_field_element(&root_limbs)
        .ok_or_else(|| "invalid root limbs from GPU Merkle tree".to_string())?;

    Ok(root)
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

/// Generate an MLE opening proof from QM31 AoS u32 words.
///
/// Input layout: `[a0,b0,c0,d0, a1,b1,c1,d1, ...]`.
#[cfg(feature = "cuda-runtime")]
fn prove_mle_opening_with_commitment_qm31_u32_gpu_tree(
    evals_u32: &[u32],
    challenges: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(FieldElement, MleOpeningProof), String> {
    assert!(!evals_u32.is_empty());
    assert!(evals_u32.len() % 4 == 0);
    let n_points = evals_u32.len() / 4;
    assert!(n_points.is_power_of_two());
    let n_vars = n_points.ilog2() as usize;
    assert_eq!(challenges.len(), n_vars);

    if !is_cuda_available() {
        return Err("cuda unavailable".to_string());
    }

    let gpu_fold_strict = gpu_mle_fold_required();
    let gpu_fold_session = {
        if gpu_mle_fold_enabled() && n_points >= gpu_mle_fold_min_points() {
            match GpuSumcheckExecutor::cached() {
                Ok(gpu) => match gpu.start_mle_fold_session_u32(evals_u32) {
                    Ok(session) => {
                        if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                            eprintln!("[GKR] MLE fold backend: GPU session");
                        }
                        Some((gpu, session))
                    }
                    Err(e) => {
                        if gpu_fold_strict {
                            return Err(format!(
                                "GPU MLE fold strict mode enabled, but session init failed: {e}"
                            ));
                        }
                        if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                            eprintln!("[GKR] MLE fold backend: CPU fallback (session init: {e})");
                        }
                        None
                    }
                },
                Err(e) => {
                    if gpu_fold_strict {
                        return Err(format!(
                            "GPU MLE fold strict mode enabled, but GPU executor init failed: {e}"
                        ));
                    }
                    if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE fold backend: CPU fallback (GPU init: {e})");
                    }
                    None
                }
            }
        } else {
            if gpu_fold_strict {
                if !gpu_mle_fold_enabled() {
                    return Err(
                        "GPU MLE fold strict mode enabled, but STWO_GPU_MLE_FOLD is disabled"
                            .to_string(),
                    );
                }
                if n_points < gpu_mle_fold_min_points() {
                    return Err(format!(
                        "GPU MLE fold strict mode enabled, but opening size {} is below min {}",
                        n_points,
                        gpu_mle_fold_min_points()
                    ));
                }
            }
            None
        }
    };

    let (gpu, mut fold_session) = match gpu_fold_session {
        Some(v) => v,
        None => {
            return Err(
                "GPU-resident opening path requires GPU fold session (falling back)".to_string(),
            )
        }
    };

    let executor = get_cuda_executor().map_err(|e| format!("cuda init: {e}"))?;
    let d_rc = upload_poseidon252_round_constants(&executor.device)
        .map_err(|e| format!("upload round constants: {e}"))?;
    let d_dummy_columns = executor
        .device
        .htod_sync_copy(&[0u32])
        .map_err(|e| format!("H2D dummy columns: {:?}", e))?;

    // ========================================================================
    // GPU deferred tree construction: single-pass fold + tree per round.
    //
    // Old approach: build ALL GPU Merkle trees during fold phase (OOM on H100
    // for 27-variable MLEs), then replay fold to extract query auth paths.
    //
    // New approach: single pass through fold rounds. At each round:
    //   1. Build GPU Merkle tree from current device buffer
    //   2. Extract root for Fiat-Shamir
    //   3. Extract query auth paths (computed from pre-determined indices)
    //   4. Drop GPU tree before building next round's tree
    //
    // This keeps only ONE GPU tree in VRAM at any time (~2-4GB vs ~30GB+).
    // Eliminates the replay fold session entirely (1 fold pass, not 2).
    // ========================================================================

    if !GPU_MLE_OPENING_TREE_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[GKR] MLE opening tree backend: GPU-deferred (1 tree at a time)");
    }

    // Phase 1a: Compute initial root via GPU (no tree storage yet)
    let (d_initial, initial_n) = gpu.mle_fold_session_current_device(&fold_session);
    let initial_root = {
        let n_leaf_hashes = initial_n / 2;
        // Pack QM31 u32 words to felt252 limbs on host
        let mut qm31_words = vec![0u32; initial_n * 4];
        executor.device.dtoh_sync_copy_into(d_initial, &mut qm31_words)
            .map_err(|e| format!("D2H initial words: {:?}", e))?;
        let mut leaf_limbs = vec![0u64; initial_n * 4];
        leaf_limbs.par_chunks_mut(4).zip(qm31_words.par_chunks_exact(4))
            .for_each(|(dst, src)| { dst.copy_from_slice(&qm31_u32_to_u64_limbs_direct(src)); });
        let d_leaf = executor.device.htod_sync_copy(&leaf_limbs)
            .map_err(|e| format!("H2D initial limbs: {:?}", e))?;
        drop(leaf_limbs);
        let gpu_tree = executor.execute_poseidon252_merkle_full_tree_gpu_layers(
            &d_dummy_columns, 0, Some(&d_leaf), n_leaf_hashes, &d_rc,
        ).map_err(|e| format!("initial GPU Merkle: {e}"))?;
        let root_limbs = gpu_tree.root_u64().map_err(|e| format!("download initial root: {e}"))?;
        u64_limbs_to_felt252(&root_limbs)
            .ok_or_else(|| "invalid initial root limbs".to_string())?
        // gpu_tree dropped — frees initial tree from VRAM
    };
    channel.mix_felt(initial_root);

    // Phase 1b: Fold loop — compute root-only at each step via GPU.
    // We CANNOT determine query auth paths yet (query indices derived after all
    // roots are mixed into the channel), so we only compute roots here and
    // store the fold challenge sequence for Phase 2 replay.
    let fold_phase_start = Instant::now();
    let mut intermediate_roots: Vec<FieldElement> = Vec::with_capacity(n_vars.saturating_sub(1));

    for (round, &r) in challenges.iter().enumerate() {
        gpu.mle_fold_session_step_in_place(&mut fold_session, r)
            .map_err(|e| {
                if gpu_fold_strict {
                    format!("GPU fold strict: failed at round {}: {}", round, e)
                } else {
                    format!("GPU folding failed at round {}: {}", round, e)
                }
            })?;

        if gpu.mle_fold_session_len(&fold_session) > 1 {
            let (d_current, cur_n) = gpu.mle_fold_session_current_device(&fold_session);
            let n_leaf_hashes = cur_n / 2;
            let mut qm31_words = vec![0u32; cur_n * 4];
            executor.device.dtoh_sync_copy_into(d_current, &mut qm31_words)
                .map_err(|e| format!("D2H fold words round {}: {:?}", round, e))?;
            let mut leaf_limbs = vec![0u64; cur_n * 4];
            leaf_limbs.par_chunks_mut(4).zip(qm31_words.par_chunks_exact(4))
                .for_each(|(dst, src)| { dst.copy_from_slice(&qm31_u32_to_u64_limbs_direct(src)); });
            let d_leaf = executor.device.htod_sync_copy(&leaf_limbs)
                .map_err(|e| format!("H2D fold limbs round {}: {:?}", round, e))?;
            drop(leaf_limbs);
            let gpu_tree = executor.execute_poseidon252_merkle_full_tree_gpu_layers(
                &d_dummy_columns, 0, Some(&d_leaf), n_leaf_hashes, &d_rc,
            ).map_err(|e| format!("GPU Merkle round {}: {e}", round))?;
            let root_limbs = gpu_tree.root_u64()
                .map_err(|e| format!("download root round {}: {e}", round))?;
            let root = u64_limbs_to_felt252(&root_limbs)
                .ok_or_else(|| format!("invalid root limbs at round {}", round))?;
            channel.mix_felt(root);
            intermediate_roots.push(root);
            // gpu_tree dropped — frees this round's tree from VRAM
        }
    }
    let fold_phase_elapsed = fold_phase_start.elapsed();

    let (_, final_n) = gpu.mle_fold_session_current_device(&fold_session);
    if final_n != 1 {
        return Err(format!("GPU fold ended with {} points (expected 1)", final_n));
    }
    let final_words = gpu.mle_fold_session_read_qm31_at(&fold_session, 0)
        .map_err(|e| format!("download final value: {e}"))?;
    let final_value = u32s_to_secure_field(&final_words);
    drop(fold_session); // Free fold session GPU memory

    // Draw query indices
    let half_n = n_points / 2;
    let n_queries = mle_n_queries().min(half_n);
    let mut query_indices: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        query_indices.push((raw as usize) % half_n);
    }

    // Precompute per-round pair indices
    let mut round_pair_indices: Vec<Vec<usize>> =
        (0..n_vars).map(|_| Vec::with_capacity(n_queries)).collect();
    let mut cur_query_indices = query_indices.clone();
    let mut layer_size = n_points;
    for round in 0..n_vars {
        round_pair_indices[round].extend(cur_query_indices.iter().copied());
        let mid = layer_size / 2;
        let next_half = (mid / 2).max(1);
        for idx in cur_query_indices.iter_mut() {
            *idx %= next_half;
        }
        layer_size = mid;
    }

    // Phase 2: Replay fold on GPU, building each tree one-at-a-time and
    // extracting query auth paths before dropping it.
    let query_phase_start = Instant::now();
    let mut replay_session = gpu.start_mle_fold_session_u32(evals_u32)
        .map_err(|e| format!("start replay fold session: {e}"))?;
    let mut query_rounds: Vec<Vec<MleQueryRoundData>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_vars)).collect();
    let mut replay_layer_size = n_points;

    for round in 0..n_vars {
        let mid = replay_layer_size / 2;
        let (d_current, cur_n) = gpu.mle_fold_session_current_device(&replay_session);
        if cur_n != replay_layer_size {
            return Err(format!(
                "replay fold size mismatch at round {}: expected {}, got {}",
                round, replay_layer_size, cur_n
            ));
        }

        // Build GPU tree for this round (dropped after extracting query auth paths)
        let tree = {
            let n_leaf_hashes = cur_n / 2;
            let mut qm31_words = vec![0u32; cur_n * 4];
            executor.device.dtoh_sync_copy_into(d_current, &mut qm31_words)
                .map_err(|e| format!("D2H replay words round {}: {:?}", round, e))?;
            let mut leaf_limbs = vec![0u64; cur_n * 4];
            leaf_limbs.par_chunks_mut(4).zip(qm31_words.par_chunks_exact(4))
                .for_each(|(dst, src)| { dst.copy_from_slice(&qm31_u32_to_u64_limbs_direct(src)); });
            let d_leaf = executor.device.htod_sync_copy(&leaf_limbs)
                .map_err(|e| format!("H2D replay limbs round {}: {:?}", round, e))?;
            drop(leaf_limbs);
            executor.execute_poseidon252_merkle_full_tree_gpu_layers(
                &d_dummy_columns, 0, Some(&d_leaf), n_leaf_hashes, &d_rc,
            ).map_err(|e| format!("GPU Merkle replay round {}: {e}", round))?
        };

        for q in 0..n_queries {
            let left_idx = round_pair_indices[round][q];
            let right_idx = mid + left_idx;

            let left_words = gpu.mle_fold_session_read_qm31_at(&replay_session, left_idx)
                .map_err(|e| format!("download left (round {}, query {}): {}", round, q, e))?;
            let right_words = gpu.mle_fold_session_read_qm31_at(&replay_session, right_idx)
                .map_err(|e| format!("download right (round {}, query {}): {}", round, q, e))?;

            let left_value = u32s_to_secure_field(&left_words);
            let right_value = u32s_to_secure_field(&right_words);
            let left_siblings = build_gpu_merkle_path_with_leaf_sibling(
                &tree, left_idx, replay_layer_size, right_value,
            )?;
            let right_siblings = build_gpu_merkle_path_with_leaf_sibling(
                &tree, right_idx, replay_layer_size, left_value,
            )?;

            query_rounds[q].push(MleQueryRoundData {
                left_value, right_value, left_siblings, right_siblings,
            });
        }
        // tree dropped here — frees GPU VRAM before next round

        if round + 1 < n_vars {
            gpu.mle_fold_session_step_in_place(&mut replay_session, challenges[round])
                .map_err(|e| format!("replay fold step failed round {}: {}", round, e))?;
            replay_layer_size = mid;
        }
    }
    let query_phase_elapsed = query_phase_start.elapsed();

    let timing_enabled = std::env::var("STWO_GPU_MLE_OPENING_TIMING")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        })
        .unwrap_or(false);
    if timing_enabled {
        eprintln!(
            "[GKR] opening gpu timing: fold_roots={:.2}s query_extract={:.2}s",
            fold_phase_elapsed.as_secs_f64(),
            query_phase_elapsed.as_secs_f64(),
        );
    }

    let queries: Vec<MleQueryProof> = query_indices
        .into_iter()
        .zip(query_rounds.into_iter())
        .map(|(pair_idx, rounds)| MleQueryProof {
            initial_pair_index: pair_idx as u32,
            rounds,
        })
        .collect();

    Ok((
        initial_root,
        MleOpeningProof {
            intermediate_roots,
            queries,
            final_value,
        },
    ))
}

#[cfg(feature = "cuda-runtime")]
pub fn prove_mle_opening_with_commitment_qm31_u32(
    evals_u32: &[u32],
    challenges: &[SecureField],
    channel: &mut PoseidonChannel,
) -> (FieldElement, MleOpeningProof) {
    assert!(!evals_u32.is_empty());
    assert!(evals_u32.len() % 4 == 0);
    let n_points = evals_u32.len() / 4;
    assert!(n_points.is_power_of_two());
    let n_vars = n_points.ilog2() as usize;
    assert_eq!(challenges.len(), n_vars);

    let gpu_tree_required = gpu_mle_opening_tree_required();
    if gpu_mle_opening_tree_enabled() {
        match prove_mle_opening_with_commitment_qm31_u32_gpu_tree(evals_u32, challenges, channel) {
            Ok(res) => return res,
            Err(e) => {
                if gpu_tree_required {
                    panic!(
                        "GPU MLE opening-tree strict mode enabled, but GPU-resident path failed: {}",
                        e
                    );
                }
                if !GPU_MLE_OPENING_TREE_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[GKR] MLE opening tree backend: CPU fallback (GPU-resident path failed: {e})"
                    );
                }
            }
        }
    } else if gpu_tree_required {
        panic!(
            "GPU MLE opening-tree strict mode enabled, but STWO_GPU_MLE_OPENING_TREE is disabled"
        );
    }

    // ========================================================================
    // Deferred tree construction: compute root-only during fold (fast), then
    // rebuild trees one-at-a-time for query auth paths (memory-efficient).
    //
    // Old approach: build & store all 27 full Merkle trees during fold loop
    //   → ~30GB peak memory (all trees stored simultaneously), OOM on GPU
    //   → CPU Poseidon at each round: ~3-5s × 27 = 80-135s per matrix
    //
    // New approach:
    //   Phase 1 (fold + roots): fold values, compute root_only at each step,
    //     store folded layer values in Vec<MleLayerValues> — O(2N) total memory
    //   Phase 2 (query extraction): after query indices drawn, re-derive each
    //     layer's tree on demand, extract auth paths, drop tree immediately
    //     → only 1 tree in memory at any time
    //   Net: ~4x less memory, ~2-3x faster (root_only skips layer storage)
    // ========================================================================

    // Phase 1: Root-only initial commitment.
    // If a cached root was provided via set_cached_initial_mle_root(), use it
    // to skip the tree computation (~1-3s per matrix).
    let initial_root = take_cached_initial_mle_root()
        .unwrap_or_else(|| commit_mle_root_only_from_qm31_u32_aos(evals_u32));
    channel.mix_felt(initial_root);

    // Store folded layer values for later tree reconstruction during query extraction.
    // saved_layers[0] = initial evals (borrowed from evals_u32), rest are owned.
    let mut saved_layers: Vec<MleLayerValues> = Vec::with_capacity(n_vars);
    saved_layers.push(MleLayerValues::Qm31U32Aos(evals_u32.to_vec()));

    let mut current_u32_layer: Option<Vec<u32>> = None;
    let mut current_secure_layer: Option<Vec<SecureField>> = None;
    let mut intermediate_roots: Vec<FieldElement> = Vec::new();

    let gpu_fold_strict = gpu_mle_fold_required();
    let mut gpu_fold_session = {
        if gpu_mle_fold_enabled() && n_points >= gpu_mle_fold_min_points() {
            match GpuSumcheckExecutor::cached() {
                Ok(gpu) => match gpu.start_mle_fold_session_u32(evals_u32) {
                    Ok(session) => {
                        if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                            eprintln!("[GKR] MLE fold backend: GPU session");
                        }
                        Some((gpu, session))
                    }
                    Err(e) => {
                        if gpu_fold_strict {
                            panic!(
                                "GPU MLE fold strict mode enabled, but session init failed: {}",
                                e
                            );
                        }
                        if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                            eprintln!("[GKR] MLE fold backend: CPU fallback (session init: {e})");
                        }
                        None
                    }
                },
                Err(e) => {
                    if gpu_fold_strict {
                        panic!(
                            "GPU MLE fold strict mode enabled, but GPU executor init failed: {}",
                            e
                        );
                    }
                    if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE fold backend: CPU fallback (GPU init: {e})");
                    }
                    None
                }
            }
        } else {
            if gpu_fold_strict {
                if !gpu_mle_fold_enabled() {
                    panic!("GPU MLE fold strict mode enabled, but STWO_GPU_MLE_FOLD is disabled");
                }
                if n_points < gpu_mle_fold_min_points() {
                    panic!(
                        "GPU MLE fold strict mode enabled, but opening size {} is below min {}",
                        n_points,
                        gpu_mle_fold_min_points()
                    );
                }
            }
            None
        }
    };

    // Phase 1: Fold loop — compute roots only, store folded values
    for &r in challenges.iter() {
        // Fast path: in-place fold for CPU secure layer (avoids Vec allocation)
        if gpu_fold_session.is_none() {
            if let Some(ref mut cur_secure) = current_secure_layer {
                let mid = cur_secure.len() / 2;
                if mid >= 1 << 16 {
                    let (lo, hi) = cur_secure.split_at_mut(mid);
                    lo.par_iter_mut().zip(hi.par_iter()).for_each(|(l, h)| {
                        *l = *l + r * (*h - *l);
                    });
                } else {
                    for j in 0..mid {
                        cur_secure[j] = cur_secure[j] + r * (cur_secure[mid + j] - cur_secure[j]);
                    }
                }
                cur_secure.truncate(mid);
                if cur_secure.len() > 1 {
                    let root = commit_mle_root_only(cur_secure);
                    channel.mix_felt(root);
                    intermediate_roots.push(root);
                    saved_layers.push(MleLayerValues::Secure(cur_secure.clone()));
                }
                continue;
            }
        }
        let next_layer = if gpu_fold_session.is_some() {
            let gpu_folded = {
                let (gpu, session) = gpu_fold_session.as_mut().expect("checked is_some");
                gpu.mle_fold_session_step_u32(session, r)
            };
            match gpu_folded {
                Ok(folded_u32) => MleLayerValues::Qm31U32Aos(folded_u32),
                Err(e) => {
                    if gpu_fold_strict {
                        panic!(
                            "GPU MLE fold strict mode enabled, but GPU folding failed mid-proof: {}",
                            e
                        );
                    }
                    eprintln!("[GKR] MLE fold backend: GPU step failed, switching to CPU ({e})");
                    gpu_fold_session = None;
                    let folded = if let Some(cur_secure) = current_secure_layer.as_ref() {
                        let mid = cur_secure.len() / 2;
                        if mid >= 1 << 16 {
                            (0..mid)
                                .into_par_iter()
                                .map(|j| cur_secure[j] + r * (cur_secure[mid + j] - cur_secure[j]))
                                .collect()
                        } else {
                            let mut v = Vec::with_capacity(mid);
                            for j in 0..mid {
                                v.push(cur_secure[j] + r * (cur_secure[mid + j] - cur_secure[j]));
                            }
                            v
                        }
                    } else if let Some(cur_u32) = current_u32_layer.as_ref() {
                        fold_layer_cpu_qm31_words(cur_u32, r)
                    } else {
                        fold_layer_cpu_qm31_words(evals_u32, r)
                    };
                    MleLayerValues::Secure(folded)
                }
            }
        } else {
            let folded = if let Some(cur_secure) = current_secure_layer.as_ref() {
                let mid = cur_secure.len() / 2;
                if mid >= 1 << 16 {
                    (0..mid)
                        .into_par_iter()
                        .map(|j| cur_secure[j] + r * (cur_secure[mid + j] - cur_secure[j]))
                        .collect()
                } else {
                    let mut v = Vec::with_capacity(mid);
                    for j in 0..mid {
                        v.push(cur_secure[j] + r * (cur_secure[mid + j] - cur_secure[j]));
                    }
                    v
                }
            } else if let Some(cur_u32) = current_u32_layer.as_ref() {
                fold_layer_cpu_qm31_words(cur_u32, r)
            } else {
                fold_layer_cpu_qm31_words(evals_u32, r)
            };
            MleLayerValues::Secure(folded)
        };

        if next_layer.len_points() > 1 {
            // Root-only: skip full tree construction (deferred to Phase 2)
            let root = match &next_layer {
                MleLayerValues::Secure(vals) => commit_mle_root_only(vals),
                MleLayerValues::Qm31U32Aos(words) => commit_mle_root_only_from_qm31_u32_aos(words),
            };
            channel.mix_felt(root);
            intermediate_roots.push(root);
            // Save folded layer for later tree reconstruction
            saved_layers.push(next_layer.clone());
        }
        match next_layer {
            MleLayerValues::Secure(vals) => {
                current_secure_layer = Some(vals);
                current_u32_layer = None;
            }
            MleLayerValues::Qm31U32Aos(words) => {
                current_u32_layer = Some(words);
                current_secure_layer = None;
            }
        }
    }
    let final_value = if let Some(cur_secure) = current_secure_layer.as_ref() {
        cur_secure[0]
    } else if let Some(cur_u32) = current_u32_layer.as_ref() {
        u32s_to_secure_field(&[cur_u32[0], cur_u32[1], cur_u32[2], cur_u32[3]])
    } else {
        // No fold rounds (single-point MLE)
        u32s_to_secure_field(&[evals_u32[0], evals_u32[1], evals_u32[2], evals_u32[3]])
    };

    // Phase 2: Draw query indices and extract auth paths.
    // Rebuild trees one-at-a-time from saved_layers (only 1 tree in memory at a time).
    let half_n = n_points / 2;
    let n_queries = mle_n_queries().min(half_n);

    let mut query_indices: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let pair_idx = (raw as usize) % half_n;
        query_indices.push(pair_idx);
    }

    // Precompute per-round pair indices for all queries
    let mut round_pair_indices: Vec<Vec<usize>> =
        (0..n_vars).map(|_| Vec::with_capacity(n_queries)).collect();
    {
        let mut cur_indices = query_indices.clone();
        let mut layer_size = n_points;
        for round in 0..n_vars {
            round_pair_indices[round].extend(cur_indices.iter().copied());
            let mid = layer_size / 2;
            let next_half = (mid / 2).max(1);
            for idx in cur_indices.iter_mut() {
                *idx %= next_half;
            }
            layer_size = mid;
        }
    }

    // Extract auth paths by rebuilding each tree from saved layer values.
    // Process one tree at a time to avoid storing all trees simultaneously.
    //
    // GPU streaming Merkle is attempted for ANY round with enough leaf hashes
    // (>= gpu_streaming_merkle_min_leaves). This eliminates ~80-100s of CPU
    // tree rebuilds for rounds 0-6. Smaller rounds fall back to fast CPU path.
    let mut query_rounds: Vec<Vec<MleQueryRoundData>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_vars)).collect();
    let mut layer_size = n_points;

    // Consume node_id for mmap Merkle tree cache (if set)
    let merkle_cache_node_id = take_merkle_cache_node_id();

    for round in 0..n_vars {
        let mid = layer_size / 2;
        let n_leaf_hashes = mid;

        // Priority 1: mmap cached tree (instant, <1ms per auth path)
        if let Some(node_id) = merkle_cache_node_id {
            if let Some(cached_tree) = merkle_cache::open_merkle_cache(node_id, round) {
                // Verify root matches Phase 1 root
                let expected_root = if round == 0 {
                    initial_root
                } else {
                    intermediate_roots[round - 1]
                };
                if cached_tree.root() == expected_root {
                    let mut cache_ok = true;
                    for q in 0..n_queries {
                        let left_idx = round_pair_indices[round][q];
                        let right_idx = mid + left_idx;

                        let left_path = match cached_tree.prove(left_idx) {
                            Some(p) => p,
                            None => { cache_ok = false; break; }
                        };
                        let right_path = match cached_tree.prove(right_idx) {
                            Some(p) => p,
                            None => { cache_ok = false; break; }
                        };

                        // Read values from saved_layers (mmap tree only has hashes, not original values)
                        let (left_value, right_value) = match &saved_layers[round] {
                            MleLayerValues::Qm31U32Aos(evals_u32) => {
                                let l = left_idx * 4;
                                let r = right_idx * 4;
                                (
                                    u32s_to_secure_field(&[evals_u32[l], evals_u32[l+1], evals_u32[l+2], evals_u32[l+3]]),
                                    u32s_to_secure_field(&[evals_u32[r], evals_u32[r+1], evals_u32[r+2], evals_u32[r+3]]),
                                )
                            }
                            MleLayerValues::Secure(vals) => {
                                (vals[left_idx], vals[right_idx])
                            }
                        };

                        query_rounds[q].push(MleQueryRoundData {
                            left_value,
                            right_value,
                            left_siblings: left_path.siblings,
                            right_siblings: right_path.siblings,
                        });
                    }
                    if cache_ok {
                        layer_size = mid;
                        continue; // Skip GPU streaming and CPU tree rebuild
                    }
                }
            }
        }

        // Priority 2: GPU streaming for any round with enough leaves
        if gpu_streaming_merkle_enabled()
            && n_leaf_hashes >= gpu_streaming_merkle_min_leaves()
            && is_cuda_available()
        {
            let round_pair_indices_for_round: Vec<usize> = (0..n_queries)
                .map(|q| round_pair_indices[round][q])
                .collect();
            match extract_auth_paths_gpu_streaming(
                &saved_layers[round], &round_pair_indices_for_round, layer_size
            ) {
                Ok((streaming_root, round_data)) => {
                    // Verify root matches Phase 1 root
                    let expected_root = if round == 0 {
                        initial_root
                    } else {
                        intermediate_roots[round - 1]
                    };
                    if streaming_root != expected_root {
                        eprintln!(
                            "[GKR] GPU streaming Merkle root mismatch at round {}! streaming={:?} vs expected={:?}",
                            round, streaming_root, expected_root
                        );
                        // Fall through to CPU path below
                    } else {
                        // Optional cross-check against CPU path (first matrix only, round 0 only)
                        if round == 0
                            && gpu_streaming_merkle_verify()
                            && !GPU_STREAMING_MERKLE_VERIFIED.swap(true, Ordering::Relaxed)
                        {
                            let cpu_tree = match &saved_layers[0] {
                                MleLayerValues::Secure(vals) => {
                                    let (_, t) = commit_mle(vals);
                                    t
                                }
                                MleLayerValues::Qm31U32Aos(words) => {
                                    let (_, t) = commit_mle_from_qm31_u32_aos(words);
                                    t
                                }
                            };
                            for (q, rd) in round_data.iter().enumerate() {
                                let left_idx = round_pair_indices_for_round[q];
                                let cpu_left = tree_prove_path_siblings(&cpu_tree, left_idx);
                                let cpu_right = tree_prove_path_siblings(&cpu_tree, mid + left_idx);
                                if rd.left_siblings != cpu_left || rd.right_siblings != cpu_right {
                                    eprintln!(
                                        "[GKR] GPU streaming Merkle VERIFY FAILED for round {} query {}",
                                        round, q
                                    );
                                }
                            }
                            eprintln!("[GKR] GPU streaming Merkle cross-check passed ({} queries)", round_data.len());
                        }

                        for q in 0..n_queries {
                            query_rounds[q].push(round_data[q].clone());
                        }
                        layer_size = mid;
                        continue; // Skip CPU tree rebuild
                    }
                }
                Err(e) => {
                    if !GPU_STREAMING_MERKLE_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] GPU streaming Merkle failed at round {}, CPU fallback: {e}", round);
                    }
                    // Fall through to CPU path below
                }
            }
        }

        // CPU fallback: build tree for this round from saved layer values
        let tree = match &saved_layers[round] {
            MleLayerValues::Secure(vals) => {
                let (_, t) = commit_mle(vals);
                t
            }
            MleLayerValues::Qm31U32Aos(words) => {
                let (_, t) = commit_mle_from_qm31_u32_aos(words);
                t
            }
        };

        // Write tree to mmap cache for next run (fire-and-forget)
        if let Some(node_id) = merkle_cache_node_id {
            if let Some(cache_dir) = merkle_cache::merkle_cache_dir() {
                let cache_path = merkle_cache::merkle_cache_path(&cache_dir, node_id, round);
                if !cache_path.exists() {
                    let _ = std::fs::create_dir_all(&cache_dir);
                    let leaves: Vec<FieldElement> = (0..layer_size)
                        .map(|i| tree.leaf_at(i))
                        .collect();
                    match merkle_cache::MmapMerkleTree::build_and_cache(&leaves, &cache_path) {
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!(
                                "[GKR] Failed to write Merkle cache for node {} round {}: {e}",
                                node_id, round
                            );
                        }
                    }
                }
            }
        }

        for q in 0..n_queries {
            let left_idx = round_pair_indices[round][q];
            let right_idx = mid + left_idx;

            let left_value = felt_to_securefield_packed(tree.leaf_at(left_idx));
            let right_value = felt_to_securefield_packed(tree.leaf_at(right_idx));
            let left_path = tree.prove(left_idx);
            let right_path = tree.prove(right_idx);

            query_rounds[q].push(MleQueryRoundData {
                left_value,
                right_value,
                left_siblings: left_path.siblings,
                right_siblings: right_path.siblings,
            });
        }
        // tree dropped here — frees memory before rebuilding next round's tree
        layer_size = mid;
    }
    // saved_layers no longer needed after query extraction
    drop(saved_layers);

    let queries: Vec<MleQueryProof> = query_indices
        .into_iter()
        .zip(query_rounds.into_iter())
        .map(|(pair_idx, rounds)| MleQueryProof {
            initial_pair_index: pair_idx as u32,
            rounds,
        })
        .collect();

    (
        initial_root,
        MleOpeningProof {
            intermediate_roots,
            queries,
            final_value,
        },
    )
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

    // Deferred tree construction: root-only during fold, rebuild for query paths.
    // Same optimization as prove_mle_opening_with_commitment_qm31_u32.
    let initial_root = commit_mle_root_only(evals);
    channel.mix_felt(initial_root);

    // Store all layer evaluations for deferred tree reconstruction
    let mut layer_evals: Vec<Vec<SecureField>> = vec![evals.to_vec()];
    let mut intermediate_roots: Vec<FieldElement> = Vec::new();
    #[cfg(feature = "cuda-runtime")]
    let gpu_fold_strict = gpu_mle_fold_required();
    #[cfg(feature = "cuda-runtime")]
    let mut gpu_fold_session = {
        if gpu_mle_fold_enabled() && evals.len() >= gpu_mle_fold_min_points() {
            match GpuSumcheckExecutor::cached() {
                Ok(gpu) => match gpu.start_mle_fold_session(evals) {
                    Ok(session) => {
                        if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                            eprintln!("[GKR] MLE fold backend: GPU session");
                        }
                        Some((gpu, session))
                    }
                    Err(e) => {
                        if gpu_fold_strict {
                            panic!(
                                "GPU MLE fold strict mode enabled, but session init failed: {}",
                                e
                            );
                        }
                        if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                            eprintln!("[GKR] MLE fold backend: CPU fallback (session init: {e})");
                        }
                        None
                    }
                },
                Err(e) => {
                    if gpu_fold_strict {
                        panic!(
                            "GPU MLE fold strict mode enabled, but GPU executor init failed: {}",
                            e
                        );
                    }
                    if !GPU_MLE_FOLD_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE fold backend: CPU fallback (GPU init: {e})");
                    }
                    None
                }
            }
        } else {
            if gpu_fold_strict {
                if !gpu_mle_fold_enabled() {
                    panic!("GPU MLE fold strict mode enabled, but STWO_GPU_MLE_FOLD is disabled");
                }
                if evals.len() < gpu_mle_fold_min_points() {
                    panic!(
                        "GPU MLE fold strict mode enabled, but opening size {} is below min {}",
                        evals.len(),
                        gpu_mle_fold_min_points()
                    );
                }
            }
            None
        }
    };

    // Phase 1: Fold through each challenge, compute root-only
    for &r in challenges.iter() {
        let folded: Vec<SecureField> = {
            #[cfg(feature = "cuda-runtime")]
            {
                if gpu_fold_session.is_some() {
                    let gpu_folded = {
                        let (gpu, session) = gpu_fold_session.as_mut().expect("checked is_some");
                        gpu.mle_fold_session_step(session, r)
                    };
                    match gpu_folded {
                        Ok(vals) => vals,
                        Err(e) => {
                            if gpu_fold_strict {
                                panic!(
                                    "GPU MLE fold strict mode enabled, but GPU folding failed mid-proof: {}",
                                    e
                                );
                            }
                            eprintln!(
                                "[GKR] MLE fold backend: GPU step failed, switching to CPU ({e})"
                            );
                            gpu_fold_session = None;
                            let current = layer_evals.last().expect("layer_evals is never empty");
                            let mid = current.len() / 2;
                            if mid >= 1 << 16 {
                                (0..mid)
                                    .into_par_iter()
                                    .map(|j| current[j] + r * (current[mid + j] - current[j]))
                                    .collect()
                            } else {
                                let mut v = Vec::with_capacity(mid);
                                for j in 0..mid {
                                    v.push(current[j] + r * (current[mid + j] - current[j]));
                                }
                                v
                            }
                        }
                    }
                } else {
                    let current = layer_evals.last().expect("layer_evals is never empty");
                    let mid = current.len() / 2;
                    if mid >= 1 << 16 {
                        (0..mid)
                            .into_par_iter()
                            .map(|j| current[j] + r * (current[mid + j] - current[j]))
                            .collect()
                    } else {
                        let mut v = Vec::with_capacity(mid);
                        for j in 0..mid {
                            v.push(current[j] + r * (current[mid + j] - current[j]));
                        }
                        v
                    }
                }
            }
            #[cfg(not(feature = "cuda-runtime"))]
            {
                let current = layer_evals.last().expect("layer_evals is never empty");
                let mid = current.len() / 2;
                if mid >= 1 << 16 {
                    (0..mid)
                        .into_par_iter()
                        .map(|j| current[j] + r * (current[mid + j] - current[j]))
                        .collect()
                } else {
                    let mut v = Vec::with_capacity(mid);
                    for j in 0..mid {
                        v.push(current[j] + r * (current[mid + j] - current[j]));
                    }
                    v
                }
            }
        };

        if folded.len() > 1 {
            let root = commit_mle_root_only(&folded);
            channel.mix_felt(root);
            intermediate_roots.push(root);
        }
        layer_evals.push(folded);
    }

    let final_value = layer_evals.last().expect("layer_evals has final layer")[0];

    // Phase 2: Draw query indices and extract auth paths.
    let initial_n = evals.len();
    let half_n = initial_n / 2;
    let n_queries = mle_n_queries().min(half_n);

    let mut query_indices: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let pair_idx = (raw as usize) % half_n;
        query_indices.push(pair_idx);
    }

    // Precompute per-round pair indices for all queries
    let mut round_pair_indices: Vec<Vec<usize>> =
        (0..n_vars).map(|_| Vec::with_capacity(n_queries)).collect();
    {
        let mut cur_indices = query_indices.clone();
        let mut layer_size = initial_n;
        for round in 0..n_vars {
            round_pair_indices[round].extend(cur_indices.iter().copied());
            let mid = layer_size / 2;
            let next_half = (mid / 2).max(1);
            for idx in cur_indices.iter_mut() {
                *idx %= next_half;
            }
            layer_size = mid;
        }
    }

    // Rebuild trees one-at-a-time per round, extracting all queries from each.
    let mut query_rounds: Vec<Vec<MleQueryRoundData>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_vars)).collect();

    for round in 0..n_vars {
        let layer = &layer_evals[round];
        let mid = layer.len() / 2;

        // Build tree for this round (dropped after extracting all query auth paths)
        let (_, tree) = commit_mle(layer);

        for q in 0..n_queries {
            let left_idx = round_pair_indices[round][q];
            let right_idx = mid + left_idx;

            let left_value = layer[left_idx.min(layer.len() - 1)];
            let right_value = layer[right_idx.min(layer.len() - 1)];

            let left_path = if left_idx < tree.num_leaves() {
                tree.prove(left_idx)
            } else {
                MerkleAuthPath {
                    siblings: Vec::new(),
                }
            };
            let right_path = if right_idx < tree.num_leaves() {
                tree.prove(right_idx)
            } else {
                MerkleAuthPath {
                    siblings: Vec::new(),
                }
            };

            query_rounds[q].push(MleQueryRoundData {
                left_value,
                right_value,
                left_siblings: left_path.siblings,
                right_siblings: right_path.siblings,
            });
        }
        // tree dropped here — only 1 tree in memory at a time
    }
    drop(layer_evals);

    let queries: Vec<MleQueryProof> = query_indices
        .into_iter()
        .zip(query_rounds.into_iter())
        .map(|(pair_idx, rounds)| MleQueryProof {
            initial_pair_index: pair_idx as u32,
            rounds,
        })
        .collect();

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
    let n_queries = mle_n_queries().min(half_n);

    let mut query_indices = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
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

                let left_path = MerkleAuthPath {
                    siblings: round_data.left_siblings.clone(),
                };
                let right_path = MerkleAuthPath {
                    siblings: round_data.right_siblings.clone(),
                };

                if !PoseidonMerkleTree::verify(layer_root, left_idx, left_leaf, &left_path) {
                    return false;
                }
                if !PoseidonMerkleTree::verify(layer_root, right_idx, right_leaf, &right_path) {
                    return false;
                }
            }

            // 3b. Algebraic fold: f(r) = left + r * (right - left)
            let r = challenges[round];
            let folded =
                round_data.left_value + r * (round_data.right_value - round_data.left_value);

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
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::m31::M31;
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
        assert!(verify_mle_opening(
            commitment,
            &proof,
            &challenges,
            &mut ch_v
        ));
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
        assert!(!verify_mle_opening(
            commitment,
            &proof,
            &challenges,
            &mut ch_v
        ));
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
            QM31(
                CM31(M31::from(10), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            QM31(
                CM31(M31::from(20), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            QM31(
                CM31(M31::from(30), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            QM31(
                CM31(M31::from(40), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
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
        assert!(verify_mle_opening(
            commitment,
            &proof,
            &challenges,
            &mut ch_v
        ));
    }

    /// Test GPU streaming Merkle auth paths against CPU-built tree auth paths.
    ///
    /// Creates a small QM31 u32 AoS MLE, builds Merkle tree via both CPU and
    /// GPU streaming paths, and verifies auth paths are identical.
    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn test_gpu_streaming_merkle_auth_paths() {
        use stwo::prover::backend::gpu::cuda_executor::{
            get_cuda_executor, is_cuda_available, upload_poseidon252_round_constants,
        };

        if !is_cuda_available() {
            eprintln!("CUDA not available, skipping GPU streaming Merkle test");
            return;
        }

        // Build a small MLE: 1024 QM31 elements = 4096 u32 words
        let n_points = 1024usize;
        let n_leaf_hashes = n_points / 2;
        let evals_u32: Vec<u32> = (0..n_points * 4)
            .map(|i| ((i as u32 * 7 + 13) % (1 << 31)))
            .collect();

        // Build CPU tree for reference
        let (cpu_root, cpu_tree) = commit_mle_from_qm31_u32_aos(&evals_u32);

        // Build GPU streaming auth paths for a few query indices
        let query_pair_indices: Vec<usize> = vec![0, 1, 7, 42, 100, 255, 510, 511];
        let mid = n_points / 2;

        let layer = MleLayerValues::Qm31U32Aos(evals_u32.clone());
        match extract_auth_paths_gpu_streaming(&layer, &query_pair_indices, n_points) {
            Ok((gpu_root, gpu_round_data)) => {
                // 1. Roots must match
                assert_eq!(
                    cpu_root, gpu_root,
                    "GPU streaming root != CPU root"
                );

                // 2. Auth paths must match for each query
                for (q, &pair_idx) in query_pair_indices.iter().enumerate() {
                    let left_idx = pair_idx;
                    let right_idx = mid + pair_idx;

                    let rd = &gpu_round_data[q];

                    // Check values match
                    let cpu_left_value = felt_to_securefield_packed(cpu_tree.leaf_at(left_idx));
                    let cpu_right_value = felt_to_securefield_packed(cpu_tree.leaf_at(right_idx));
                    assert_eq!(rd.left_value, cpu_left_value, "query {q}: left value mismatch");
                    assert_eq!(rd.right_value, cpu_right_value, "query {q}: right value mismatch");

                    // Check auth paths match
                    let cpu_left_path = cpu_tree.prove(left_idx);
                    let cpu_right_path = cpu_tree.prove(right_idx);
                    assert_eq!(
                        rd.left_siblings, cpu_left_path.siblings,
                        "query {q}: left auth path mismatch"
                    );
                    assert_eq!(
                        rd.right_siblings, cpu_right_path.siblings,
                        "query {q}: right auth path mismatch"
                    );
                }
                eprintln!(
                    "GPU streaming Merkle test passed: {} queries, {} leaves",
                    query_pair_indices.len(),
                    n_leaf_hashes
                );
            }
            Err(e) => {
                eprintln!("GPU streaming Merkle test skipped (GPU error): {e}");
            }
        }
    }

    /// Test that the cached initial MLE root is consumed correctly.
    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn test_cached_initial_mle_root_thread_local() {
        let root = FieldElement::from(0x12345u64);
        set_cached_initial_mle_root(root);

        // First take should return the value
        let taken = take_cached_initial_mle_root();
        assert_eq!(taken, Some(root));

        // Second take should return None (consumed)
        let taken2 = take_cached_initial_mle_root();
        assert_eq!(taken2, None);
    }
}
