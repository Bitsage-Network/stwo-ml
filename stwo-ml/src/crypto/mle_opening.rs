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

/// Number of queries for MLE opening (matching STARK FRI query count).
pub const MLE_N_QUERIES: usize = 14;
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
        .unwrap_or(1 << 20)
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
fn build_gpu_opening_tree_from_qm31_u32_device_with_ctx(
    executor: &CudaFftExecutor,
    d_rc: &CudaSlice<u64>,
    d_dummy_columns: &CudaSlice<u32>,
    d_qm31_aos: &CudaSlice<u32>,
    n_points: usize,
) -> Result<(FieldElement, Poseidon252MerkleGpuTree), String> {
    assert!(n_points > 0);
    assert!(n_points.is_power_of_two());
    if d_qm31_aos.len() != n_points * 4 {
        return Err(format!(
            "device QM31 AoS size mismatch: expected {} words, got {}",
            n_points * 4,
            d_qm31_aos.len()
        ));
    }

    let n_leaf_hashes = n_points / 2;
    if n_leaf_hashes == 0 {
        return Err("cannot build gpu opening tree for single-point input".to_string());
    }

    // Pack QM31 AoS words into felt252 limbs.
    // NOTE: this uses a host-side conversion step when a dedicated CUDA pack
    // kernel is unavailable in the linked stwo backend.
    let mut qm31_words = vec![0u32; n_points * 4];
    executor
        .device
        .dtoh_sync_copy_into(d_qm31_aos, &mut qm31_words)
        .map_err(|e| format!("D2H qm31 words: {:?}", e))?;
    let mut leaf_limbs = vec![0u64; n_points * 4];
    leaf_limbs
        .par_chunks_mut(4)
        .zip(qm31_words.par_chunks_exact(4))
        .for_each(|(dst, src)| {
            dst.copy_from_slice(&qm31_u32_to_u64_limbs_direct(src));
        });
    let d_prev_leaf = executor
        .device
        .htod_sync_copy(&leaf_limbs)
        .map_err(|e| format!("H2D packed felt252 limbs: {:?}", e))?;

    let tree = executor
        .execute_poseidon252_merkle_full_tree_gpu_layers(
            d_dummy_columns,
            0,
            Some(&d_prev_leaf),
            n_leaf_hashes,
            d_rc,
        )
        .map_err(|e| format!("execute full-tree gpu layers: {e}"))?;
    let root_limbs = tree
        .root_u64()
        .map_err(|e| format!("download gpu root: {e}"))?;
    let root = u64_limbs_to_felt252(&root_limbs)
        .ok_or_else(|| "invalid root limbs from gpu merkle tree".to_string())?;
    Ok((root, tree))
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

    let (d_initial, initial_n) = gpu.mle_fold_session_current_device(&fold_session);
    let (initial_root, initial_tree) = build_gpu_opening_tree_from_qm31_u32_device_with_ctx(
        executor,
        &d_rc,
        &d_dummy_columns,
        d_initial,
        initial_n,
    )?;
    channel.mix_felt(initial_root);

    if !GPU_MLE_OPENING_TREE_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!("[GKR] MLE opening tree backend: GPU-resident (no bulk D2H)");
    }

    let tree_phase_start = Instant::now();
    let mut layer_trees: Vec<Poseidon252MerkleGpuTree> = Vec::with_capacity(n_vars);
    layer_trees.push(initial_tree);
    let mut intermediate_roots: Vec<FieldElement> = Vec::with_capacity(n_vars.saturating_sub(1));

    for (round, &r) in challenges.iter().enumerate() {
        gpu.mle_fold_session_step_in_place(&mut fold_session, r)
            .map_err(|e| {
                if gpu_fold_strict {
                    format!(
                        "GPU MLE fold strict mode enabled, but GPU folding failed at round {}: {}",
                        round, e
                    )
                } else {
                    format!("GPU MLE folding failed at round {}: {}", round, e)
                }
            })?;

        if gpu.mle_fold_session_len(&fold_session) > 1 {
            let (d_current, cur_n) = gpu.mle_fold_session_current_device(&fold_session);
            let (root, tree) = build_gpu_opening_tree_from_qm31_u32_device_with_ctx(
                executor,
                &d_rc,
                &d_dummy_columns,
                d_current,
                cur_n,
            )?;
            channel.mix_felt(root);
            intermediate_roots.push(root);
            layer_trees.push(tree);
        }
    }
    let tree_phase_elapsed = tree_phase_start.elapsed();

    if layer_trees.len() != n_vars {
        return Err(format!(
            "GPU opening tree construction produced {} layer trees, expected {}",
            layer_trees.len(),
            n_vars
        ));
    }

    let (_, final_n) = gpu.mle_fold_session_current_device(&fold_session);
    if final_n != 1 {
        return Err(format!(
            "GPU fold session ended with {} points (expected 1)",
            final_n
        ));
    }
    let final_words = gpu
        .mle_fold_session_read_qm31_at(&fold_session, 0)
        .map_err(|e| format!("download final folded value: {e}"))?;
    let final_value = u32s_to_secure_field(&final_words);

    let half_n = n_points / 2;
    let n_queries = MLE_N_QUERIES.min(half_n);

    let mut query_indices: Vec<usize> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let felt = channel.draw_felt252();
        let bytes = felt.to_bytes_be();
        let raw = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        query_indices.push((raw as usize) % half_n);
    }

    // Precompute per-round pair indices for each query once, then replay only
    // lightweight folds to extract the queried leaf pairs.
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

    let query_phase_start = Instant::now();
    let mut replay_session = gpu
        .start_mle_fold_session_u32(evals_u32)
        .map_err(|e| format!("start replay MLE fold session: {e}"))?;
    let mut query_rounds: Vec<Vec<MleQueryRoundData>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_vars)).collect();
    let mut replay_layer_size = n_points;

    for round in 0..n_vars {
        let mid = replay_layer_size / 2;
        let (_, cur_n) = gpu.mle_fold_session_current_device(&replay_session);
        if cur_n != replay_layer_size {
            return Err(format!(
                "replay fold size mismatch at round {}: expected {}, got {}",
                round, replay_layer_size, cur_n
            ));
        }

        let tree = &layer_trees[round];
        for q in 0..n_queries {
            let left_idx = round_pair_indices[round][q];
            let right_idx = mid + left_idx;

            let left_words = gpu
                .mle_fold_session_read_qm31_at(&replay_session, left_idx)
                .map_err(|e| {
                    format!(
                        "download left query value (round {}, query {}): {}",
                        round, q, e
                    )
                })?;
            let right_words = gpu
                .mle_fold_session_read_qm31_at(&replay_session, right_idx)
                .map_err(|e| {
                    format!(
                        "download right query value (round {}, query {}): {}",
                        round, q, e
                    )
                })?;

            let left_value = u32s_to_secure_field(&left_words);
            let right_value = u32s_to_secure_field(&right_words);
            let left_siblings = build_gpu_merkle_path_with_leaf_sibling(
                tree,
                left_idx,
                replay_layer_size,
                right_value,
            )?;
            let right_siblings = build_gpu_merkle_path_with_leaf_sibling(
                tree,
                right_idx,
                replay_layer_size,
                left_value,
            )?;

            query_rounds[q].push(MleQueryRoundData {
                left_value,
                right_value,
                left_siblings,
                right_siblings,
            });
        }

        if round + 1 < n_vars {
            gpu.mle_fold_session_step_in_place(&mut replay_session, challenges[round])
                .map_err(|e| format!("replay GPU fold step failed at round {}: {}", round, e))?;
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
            "[GKR] opening gpu timing: tree_build={:.2}s query_extract={:.2}s",
            tree_phase_elapsed.as_secs_f64(),
            query_phase_elapsed.as_secs_f64()
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

    let (initial_root, initial_tree) = commit_mle_from_qm31_u32_aos(evals_u32);
    channel.mix_felt(initial_root);

    // Keep the initial layer borrowed to avoid a full-size clone of `evals_u32`.
    // On large matrices this saves a multi-GB copy before opening generation starts.
    let mut current_u32_layer: Option<Vec<u32>> = None;
    let mut current_secure_layer: Option<Vec<SecureField>> = None;
    let mut layer_trees: Vec<PoseidonMerkleTree> = vec![initial_tree];
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

    for &r in challenges.iter() {
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
                        // Already in secure-field mode.
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
            let (root, tree) = match &next_layer {
                MleLayerValues::Secure(vals) => commit_mle(vals),
                MleLayerValues::Qm31U32Aos(words) => commit_mle_from_qm31_u32_aos(words),
            };
            channel.mix_felt(root);
            intermediate_roots.push(root);
            layer_trees.push(tree);
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

    // Draw query indices (each query selects an index in [0, n/2))
    let half_n = n_points / 2;
    let n_queries = MLE_N_QUERIES.min(half_n);

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

    let mut queries = Vec::with_capacity(n_queries);
    for &pair_idx in &query_indices {
        let mut rounds = Vec::with_capacity(n_vars);
        let mut current_idx = pair_idx;
        let mut layer_size = n_points;

        for round in 0..n_vars {
            let mid = layer_size / 2;
            let left_idx = current_idx;
            let right_idx = mid + current_idx;

            let tree = &layer_trees[round];
            let left_value = felt_to_securefield_packed(tree.leaf_at(left_idx));
            let right_value = felt_to_securefield_packed(tree.leaf_at(right_idx));

            let left_path = tree.prove(left_idx);
            let right_path = tree.prove(right_idx);

            rounds.push(MleQueryRoundData {
                left_value,
                right_value,
                left_siblings: left_path.siblings,
                right_siblings: right_path.siblings,
            });

            let next_half = (mid / 2).max(1);
            current_idx %= next_half;
            layer_size = mid;
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

    // Fold through each challenge
    // Variable ordering matches evaluate_mle: first variable splits into lo/hi halves
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
        let mid = folded.len();

        if mid > 1 {
            let (root, tree) = commit_mle(&folded);
            channel.mix_felt(root);
            intermediate_roots.push(root);
            layer_trees.push(tree);
        }
        layer_evals.push(folded);
    }

    let final_value = layer_evals.last().expect("layer_evals has final layer")[0];

    // Draw query indices (each query selects an index in [0, n/2))
    let initial_n = evals.len();
    let half_n = initial_n / 2;
    let n_queries = MLE_N_QUERIES.min(half_n);

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
}
