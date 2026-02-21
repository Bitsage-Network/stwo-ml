//! Binary Merkle tree using Poseidon hash.
//!
//! Provides commitment and authentication path generation for
//! on-chain verification of MLE evaluations.

use rayon::prelude::*;
use starknet_crypto::poseidon_hash;
use starknet_ff::FieldElement;
#[cfg(feature = "cuda-runtime")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::qm31::SecureField;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::qm31::QM31;

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
#[cfg(feature = "cuda-runtime")]
static GPU_MLE_MERKLE_BACKEND_LOGGED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cuda-runtime")]
static GPU_MLE_MERKLE_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

/// A binary Merkle tree with Poseidon hash.
///
/// Leaves are `FieldElement` values. Internal nodes are computed as
/// `poseidon_hash(left_child, right_child)`.
#[derive(Debug, Clone)]
pub struct PoseidonMerkleTree {
    /// Tree layers from bottom (leaves) to top (root).
    /// `layers[0]` = leaves (padded to power of 2).
    /// `layers[last]` = `[root]`.
    layers: Vec<MerkleLayer>,
}

/// Authentication path for a Merkle proof (bottom-up sibling hashes).
#[derive(Debug, Clone)]
pub struct MerkleAuthPath {
    pub siblings: Vec<FieldElement>,
}

#[derive(Debug, Clone)]
enum MerkleLayer {
    Felt(Vec<FieldElement>),
    #[cfg(feature = "cuda-runtime")]
    Limbs(Vec<u64>),
}

impl MerkleLayer {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Felt(v) => v.len(),
            #[cfg(feature = "cuda-runtime")]
            Self::Limbs(v) => v.len() / 4,
        }
    }

    #[inline]
    fn at(&self, idx: usize) -> FieldElement {
        match self {
            Self::Felt(v) => v[idx],
            #[cfg(feature = "cuda-runtime")]
            Self::Limbs(v) => {
                let s = idx * 4;
                u64_limbs_to_field_element(&v[s..s + 4]).expect("invalid GPU Merkle limbs")
            }
        }
    }
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

        let mut layers = vec![MerkleLayer::Felt(padded)];

        // Build layers bottom-up
        while layers.last().unwrap().len() > 1 {
            let current = match layers.last().unwrap() {
                MerkleLayer::Felt(v) => v,
                #[cfg(feature = "cuda-runtime")]
                MerkleLayer::Limbs(_) => unreachable!("CPU tree layers are always felt"),
            };
            let mut next = Vec::with_capacity(current.len() / 2);
            for i in (0..current.len()).step_by(2) {
                next.push(poseidon_hash(current[i], current[i + 1]));
            }
            layers.push(MerkleLayer::Felt(next));
        }

        Self { layers }
    }

    /// Build a Poseidon Merkle tree with rayon-parallel hashing.
    ///
    /// Each layer's hash pairs are independent, so layers with >= PARALLEL_THRESHOLD
    /// pairs use `par_chunks(2)` for parallel Poseidon hashing.
    pub fn build_parallel(leaves: Vec<FieldElement>) -> Self {
        Self::build_parallel_prepacked(leaves, None)
    }

    /// Build a Poseidon Merkle tree with rayon-parallel hashing and optional
    /// prepacked leaf limbs for the GPU backend.
    #[allow(unused_variables)]
    pub fn build_parallel_prepacked(
        leaves: Vec<FieldElement>,
        prepacked_leaf_limbs: Option<Vec<u64>>,
    ) -> Self {
        assert!(!leaves.is_empty(), "cannot build tree from empty leaves");

        let n = leaves.len().next_power_of_two().max(2);
        let mut padded = leaves;
        padded.resize(n, FieldElement::ZERO);

        #[cfg(feature = "cuda-runtime")]
        {
            let strict_gpu = gpu_merkle_required();
            match try_build_parallel_gpu(&padded, prepacked_leaf_limbs.as_deref()) {
                Ok(Some(gpu_layers)) => {
                    if !GPU_MLE_MERKLE_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE Merkle backend: GPU Poseidon full-tree");
                    }
                    return Self { layers: gpu_layers };
                }
                Ok(None) => {
                    if strict_gpu {
                        panic!(
                            "GPU MLE Merkle strict mode enabled, but GPU path unavailable \
                             (pairs={}, threshold={}, cuda_available={})",
                            padded.len() / 2,
                            GPU_MERKLE_THRESHOLD_PAIRS,
                            is_cuda_available()
                        );
                    }
                    if !GPU_MLE_MERKLE_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[GKR] MLE Merkle backend: CPU fallback (GPU unavailable; pairs={}, threshold={}, cuda_available={})",
                            padded.len() / 2,
                            GPU_MERKLE_THRESHOLD_PAIRS,
                            is_cuda_available()
                        );
                    }
                }
                Err(err) => {
                    if strict_gpu {
                        panic!(
                            "GPU MLE Merkle strict mode enabled, but GPU full-tree hashing failed: {}",
                            err
                        );
                    }
                    if !GPU_MLE_MERKLE_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE Merkle backend: CPU fallback ({err})");
                    }
                }
            }
        }

        let mut layers = vec![MerkleLayer::Felt(padded)];

        while layers.last().unwrap().len() > 1 {
            let current = match layers.last().unwrap() {
                MerkleLayer::Felt(v) => v,
                #[cfg(feature = "cuda-runtime")]
                MerkleLayer::Limbs(_) => unreachable!("CPU fallback tree layers are felt"),
            };
            let n_pairs = current.len() / 2;

            let next = if n_pairs >= PARALLEL_THRESHOLD {
                current
                    .par_chunks(2)
                    .map(|pair| poseidon_hash(pair[0], pair[1]))
                    .collect()
            } else {
                let mut v = Vec::with_capacity(n_pairs);
                for i in (0..current.len()).step_by(2) {
                    v.push(poseidon_hash(current[i], current[i + 1]));
                }
                v
            };
            layers.push(MerkleLayer::Felt(next));
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
                current
                    .par_chunks(2)
                    .map(|pair| poseidon_hash(pair[0], pair[1]))
                    .collect()
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
        self.layers.last().unwrap().at(0)
    }

    /// Returns the number of leaves (including padding).
    pub fn num_leaves(&self) -> usize {
        self.layers[0].len()
    }

    /// Return the leaf value at `index` as a felt.
    ///
    /// Works for both felt-backed and limb-backed layers.
    pub fn leaf_at(&self, index: usize) -> FieldElement {
        assert!(index < self.layers[0].len(), "leaf index out of bounds");
        self.layers[0].at(index)
    }

    /// Generate an authentication path for the leaf at `index`.
    pub fn prove(&self, index: usize) -> MerkleAuthPath {
        assert!(index < self.layers[0].len(), "index out of bounds");

        let mut siblings = Vec::with_capacity(self.layers.len() - 1);
        let mut idx = index;

        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling_idx = idx ^ 1; // flip last bit
            siblings.push(layer.at(sibling_idx));
            idx >>= 1;
        }

        MerkleAuthPath { siblings }
    }

    /// Build a Poseidon Merkle tree directly from SecureField evaluations.
    ///
    /// On cuda-runtime builds with GPU available, this bypasses all FieldElement
    /// Montgomery conversions (~225ns per element saved). Both leaf and internal
    /// layers are stored as raw u64 limbs and converted lazily on auth path access.
    ///
    /// Falls back to the standard FieldElement path when GPU is unavailable.
    pub fn build_parallel_from_secure(evals: &[SecureField]) -> Self {
        assert!(!evals.is_empty(), "cannot build tree from empty evals");
        assert!(
            evals.len().is_power_of_two(),
            "evals must be power-of-2 length"
        );

        #[cfg(feature = "cuda-runtime")]
        {
            let strict_gpu = gpu_merkle_required();
            match try_build_gpu_from_secure_fields(evals) {
                Ok(Some(gpu_layers)) => {
                    if !GPU_MLE_MERKLE_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE Merkle backend: GPU Poseidon direct-secure");
                    }
                    return Self { layers: gpu_layers };
                }
                Ok(None) => {
                    if strict_gpu {
                        panic!(
                            "GPU MLE Merkle strict mode enabled, but direct-secure GPU path unavailable \
                             (pairs={}, threshold={}, cuda_available={})",
                            evals.len() / 2,
                            GPU_MERKLE_THRESHOLD_PAIRS,
                            is_cuda_available()
                        );
                    }
                    if !GPU_MLE_MERKLE_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[GKR] MLE Merkle backend: CPU fallback (direct-secure unavailable; pairs={}, threshold={}, cuda_available={})",
                            evals.len() / 2,
                            GPU_MERKLE_THRESHOLD_PAIRS,
                            is_cuda_available()
                        );
                    }
                }
                Err(err) => {
                    if strict_gpu {
                        panic!(
                            "GPU MLE Merkle strict mode enabled, but direct-secure GPU hashing failed: {}",
                            err
                        );
                    }
                    if !GPU_MLE_MERKLE_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE Merkle backend: CPU fallback ({err})");
                    }
                }
            }
        }

        // CPU fallback: convert through FieldElement
        let leaves: Vec<FieldElement> = if evals.len() >= 256 {
            evals
                .par_iter()
                .map(|sf| crate::crypto::poseidon_channel::securefield_to_felt(*sf))
                .collect()
        } else {
            evals
                .iter()
                .map(|sf| crate::crypto::poseidon_channel::securefield_to_felt(*sf))
                .collect()
        };
        Self::build_parallel(leaves)
    }

    /// Build a Poseidon Merkle tree directly from QM31 words in AoS u32 format.
    ///
    /// Input layout: `[a0,b0,c0,d0, a1,b1,c1,d1, ...]` where each 4-word tuple
    /// is a QM31 value.
    ///
    /// On cuda-runtime builds this keeps the path in limb form and avoids
    /// per-element Montgomery conversions.
    pub fn build_parallel_from_qm31_u32_aos(words: &[u32]) -> Self {
        assert!(!words.is_empty(), "cannot build tree from empty words");
        assert!(words.len() % 4 == 0, "QM31 AoS words must be multiple of 4");
        let n_points = words.len() / 4;
        assert!(n_points.is_power_of_two(), "eval count must be power-of-2");

        #[cfg(feature = "cuda-runtime")]
        {
            let strict_gpu = gpu_merkle_required();
            match try_build_gpu_from_qm31_u32(words) {
                Ok(Some(gpu_layers)) => {
                    if !GPU_MLE_MERKLE_BACKEND_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE Merkle backend: GPU Poseidon direct-u32");
                    }
                    return Self { layers: gpu_layers };
                }
                Ok(None) => {
                    if strict_gpu {
                        panic!(
                            "GPU MLE Merkle strict mode enabled, but direct-u32 GPU path unavailable \
                             (pairs={}, threshold={}, cuda_available={})",
                            n_points / 2,
                            GPU_MERKLE_THRESHOLD_PAIRS,
                            is_cuda_available()
                        );
                    }
                    if !GPU_MLE_MERKLE_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[GKR] MLE Merkle backend: CPU fallback (direct-u32 unavailable; pairs={}, threshold={}, cuda_available={})",
                            n_points / 2,
                            GPU_MERKLE_THRESHOLD_PAIRS,
                            is_cuda_available()
                        );
                    }
                }
                Err(err) => {
                    if strict_gpu {
                        panic!(
                            "GPU MLE Merkle strict mode enabled, but direct-u32 GPU hashing failed: {}",
                            err
                        );
                    }
                    if !GPU_MLE_MERKLE_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!("[GKR] MLE Merkle backend: CPU fallback ({err})");
                    }
                }
            }
        }

        let leaves: Vec<FieldElement> = if n_points >= 256 {
            words
                .par_chunks_exact(4)
                .map(|c| {
                    let packed = (1u128 << 124)
                        | ((c[0] as u128) << 93)
                        | ((c[1] as u128) << 62)
                        | ((c[2] as u128) << 31)
                        | (c[3] as u128);
                    FieldElement::from(packed)
                })
                .collect()
        } else {
            words
                .chunks_exact(4)
                .map(|c| {
                    let packed = (1u128 << 124)
                        | ((c[0] as u128) << 93)
                        | ((c[1] as u128) << 62)
                        | ((c[2] as u128) << 31)
                        | (c[3] as u128);
                    FieldElement::from(packed)
                })
                .collect()
        };
        Self::build_parallel(leaves)
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
fn gpu_merkle_required() -> bool {
    let explicit = match std::env::var("STWO_GPU_MLE_MERKLE_REQUIRE") {
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

/// Convert a SecureField (QM31) directly to the u64 limb representation
/// used by the GPU Poseidon kernel, bypassing FieldElement Montgomery form.
///
/// This is equivalent to `field_element_to_u64_limbs(&securefield_to_felt(sf))`
/// but ~30x faster because it skips two Montgomery multiplications.
///
/// The packed value `2^124 + (a<<93) + (b<<62) + (c<<31) + d` fits in 125 bits,
/// so limbs[2] and limbs[3] are always 0.
#[cfg(feature = "cuda-runtime")]
#[inline]
pub(crate) fn securefield_to_u64_limbs_direct(sf: SecureField) -> [u64; 4] {
    let QM31(CM31(a, b), CM31(c, d)) = sf;
    let packed = (1u128 << 124)
        | ((a.0 as u128) << 93)
        | ((b.0 as u128) << 62)
        | ((c.0 as u128) << 31)
        | (d.0 as u128);
    [packed as u64, (packed >> 64) as u64, 0, 0]
}

/// Convert QM31 AoS words `[a,b,c,d]` directly to the u64 limb representation
/// consumed by the GPU Poseidon kernel.
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
pub(crate) fn u64_limbs_to_field_element(limbs: &[u64]) -> Option<FieldElement> {
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
fn try_build_parallel_gpu(
    leaves: &[FieldElement],
    prepacked_leaf_limbs: Option<&[u64]>,
) -> Result<Option<Vec<MerkleLayer>>, String> {
    if !gpu_merkle_enabled() || !is_cuda_available() {
        return Ok(None);
    }
    let n_leaf_hashes = leaves.len() / 2;
    if n_leaf_hashes < GPU_MERKLE_THRESHOLD_PAIRS {
        return Ok(None);
    }

    let executor = get_cuda_executor().map_err(|e| format!("cuda init: {e}"))?;
    let d_round_constants = upload_poseidon252_round_constants(&executor.device)
        .map_err(|e| format!("upload round constants: {e}"))?;

    let leaf_limbs: Vec<u64> = if let Some(prepacked) = prepacked_leaf_limbs {
        if prepacked.len() != leaves.len() * 4 {
            return Err(format!(
                "invalid prepacked limb length: got {}, expected {}",
                prepacked.len(),
                leaves.len() * 4,
            ));
        }
        prepacked.to_vec()
    } else {
        let mut v = vec![0u64; leaves.len() * 4];
        v.par_chunks_mut(4)
            .zip(leaves.par_iter())
            .for_each(|(dst, fe)| {
                let limbs = field_element_to_u64_limbs(fe);
                dst.copy_from_slice(&limbs);
            });
        v
    };
    let d_prev_leaf = executor
        .device
        .htod_sync_copy(&leaf_limbs)
        .map_err(|e| format!("H2D leaf limbs: {:?}", e))?;
    let d_dummy_columns = executor
        .device
        .htod_sync_copy(&[0u32])
        .map_err(|e| format!("H2D dummy columns: {:?}", e))?;

    let raw_layers = executor
        .execute_poseidon252_merkle_full_tree(
            &d_dummy_columns,
            0,
            Some(&d_prev_leaf),
            n_leaf_hashes,
            &d_round_constants,
        )
        .map_err(|e| format!("execute full-tree: {e}"))?;

    let mut layers = Vec::with_capacity(raw_layers.len() + 1);
    layers.push(MerkleLayer::Felt(leaves.to_vec()));
    for raw_layer in raw_layers {
        if raw_layer.len() % 4 != 0 {
            return Err(format!(
                "invalid raw layer limb length: {}",
                raw_layer.len()
            ));
        }
        layers.push(MerkleLayer::Limbs(raw_layer));
    }
    Ok(Some(layers))
}

/// Build GPU Merkle tree directly from SecureField evaluations,
/// bypassing all FieldElement Montgomery conversions.
///
/// Saves ~225ns per element (3 Montgomery ops) compared to the
/// FieldElement path. For 268M elements that's ~60s saved.
#[cfg(feature = "cuda-runtime")]
fn try_build_gpu_from_secure_fields(
    evals: &[SecureField],
) -> Result<Option<Vec<MerkleLayer>>, String> {
    if !gpu_merkle_enabled() || !is_cuda_available() {
        return Ok(None);
    }
    let n_leaf_hashes = evals.len() / 2;
    if n_leaf_hashes < GPU_MERKLE_THRESHOLD_PAIRS {
        return Ok(None);
    }

    let executor = get_cuda_executor().map_err(|e| format!("cuda init: {e}"))?;
    let d_round_constants = upload_poseidon252_round_constants(&executor.device)
        .map_err(|e| format!("upload round constants: {e}"))?;

    // Direct SecureField → u64 limbs, skipping FieldElement Montgomery form entirely.
    let mut leaf_limbs = vec![0u64; evals.len() * 4];
    leaf_limbs
        .par_chunks_mut(4)
        .zip(evals.par_iter())
        .for_each(|(dst, sf)| {
            dst.copy_from_slice(&securefield_to_u64_limbs_direct(*sf));
        });

    let d_prev_leaf = executor
        .device
        .htod_sync_copy(&leaf_limbs)
        .map_err(|e| format!("H2D leaf limbs: {:?}", e))?;
    let d_dummy_columns = executor
        .device
        .htod_sync_copy(&[0u32])
        .map_err(|e| format!("H2D dummy columns: {:?}", e))?;

    let raw_layers = executor
        .execute_poseidon252_merkle_full_tree(
            &d_dummy_columns,
            0,
            Some(&d_prev_leaf),
            n_leaf_hashes,
            &d_round_constants,
        )
        .map_err(|e| format!("execute full-tree: {e}"))?;

    let mut layers = Vec::with_capacity(raw_layers.len() + 1);
    // Store leaf layer as Limbs — no FieldElement allocation or 8GB clone.
    layers.push(MerkleLayer::Limbs(leaf_limbs));
    // Internal layers stay as raw limbs from GPU — no Montgomery conversion.
    for raw_layer in raw_layers {
        if raw_layer.len() % 4 != 0 {
            return Err(format!(
                "invalid raw layer limb length: {}",
                raw_layer.len()
            ));
        }
        layers.push(MerkleLayer::Limbs(raw_layer));
    }
    Ok(Some(layers))
}

/// Build GPU Merkle tree directly from QM31 words (AoS u32), bypassing
/// SecureField/FieldElement conversions.
#[cfg(feature = "cuda-runtime")]
fn try_build_gpu_from_qm31_u32(words: &[u32]) -> Result<Option<Vec<MerkleLayer>>, String> {
    if !gpu_merkle_enabled() || !is_cuda_available() {
        return Ok(None);
    }
    if words.len() % 4 != 0 {
        return Err(format!("invalid QM31 word length: {}", words.len()));
    }
    let n_points = words.len() / 4;
    let n_leaf_hashes = n_points / 2;
    if n_leaf_hashes < GPU_MERKLE_THRESHOLD_PAIRS {
        return Ok(None);
    }

    let executor = get_cuda_executor().map_err(|e| format!("cuda init: {e}"))?;
    let d_round_constants = upload_poseidon252_round_constants(&executor.device)
        .map_err(|e| format!("upload round constants: {e}"))?;

    let mut leaf_limbs = vec![0u64; n_points * 4];
    leaf_limbs
        .par_chunks_mut(4)
        .zip(words.par_chunks_exact(4))
        .for_each(|(dst, src)| {
            dst.copy_from_slice(&qm31_u32_to_u64_limbs_direct(src));
        });

    let d_prev_leaf = executor
        .device
        .htod_sync_copy(&leaf_limbs)
        .map_err(|e| format!("H2D leaf limbs: {:?}", e))?;
    let d_dummy_columns = executor
        .device
        .htod_sync_copy(&[0u32])
        .map_err(|e| format!("H2D dummy columns: {:?}", e))?;

    let raw_layers = executor
        .execute_poseidon252_merkle_full_tree(
            &d_dummy_columns,
            0,
            Some(&d_prev_leaf),
            n_leaf_hashes,
            &d_round_constants,
        )
        .map_err(|e| format!("execute full-tree: {e}"))?;

    let mut layers = Vec::with_capacity(raw_layers.len() + 1);
    layers.push(MerkleLayer::Limbs(leaf_limbs));
    for raw_layer in raw_layers {
        if raw_layer.len() % 4 != 0 {
            return Err(format!(
                "invalid raw layer limb length: {}",
                raw_layer.len()
            ));
        }
        layers.push(MerkleLayer::Limbs(raw_layer));
    }
    Ok(Some(layers))
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
        assert!(PoseidonMerkleTree::verify(
            root,
            0,
            FieldElement::from(42u64),
            &path
        ));
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
            assert!(PoseidonMerkleTree::verify(
                par.root(),
                i,
                FieldElement::from(i as u64),
                &path
            ));
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

    #[test]
    fn test_build_from_secure_matches_felt_path() {
        use crate::crypto::poseidon_channel::securefield_to_felt;
        use stwo::core::fields::m31::M31;

        // Create power-of-2 SecureField evaluations
        for n in [4, 16, 64, 256, 512] {
            let evals: Vec<SecureField> = (0..n)
                .map(|i| SecureField::from(M31::from((i + 1) as u32)))
                .collect();

            // Build via the standard FieldElement path
            let leaves: Vec<FieldElement> =
                evals.iter().map(|sf| securefield_to_felt(*sf)).collect();
            let felt_tree = PoseidonMerkleTree::build_parallel(leaves);

            // Build via the new direct SecureField path
            let secure_tree = PoseidonMerkleTree::build_parallel_from_secure(&evals);

            assert_eq!(
                felt_tree.root(),
                secure_tree.root(),
                "root mismatch for n={n}: felt vs direct-secure path"
            );

            // Verify auth paths from the secure tree agree with the felt tree
            for idx in [0, 1, n / 2, n - 1] {
                let felt_path = felt_tree.prove(idx);
                let secure_path = secure_tree.prove(idx);
                assert_eq!(
                    felt_path.siblings, secure_path.siblings,
                    "auth path mismatch for n={n}, idx={idx}"
                );
            }
        }
    }
}
