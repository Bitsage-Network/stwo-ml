//! GPU-accelerated Merkle tree operations.
//!
//! This module implements [`MerkleOps`] for [`GpuBackend`].
//!
//! # Algorithm
//!
//! Merkle tree commitment involves:
//! 1. Hashing leaf data (columns) using Blake2s
//! 2. Building the tree by hashing pairs of child hashes
//!
//! # GPU Strategy
//!
//! Blake2s hashing is moderately parallelizable on GPU:
//! - Each hash computation is independent
//! - GPU processes all leaves/nodes in parallel
//! - Expected speedup: 2-4x for large trees (>64K leaves)
//!
//! For smaller trees, CPU SIMD is faster due to transfer overhead.

use starknet_ff::FieldElement as FieldElement252;
use tracing::{span, Level};

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleHasher;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::vcs::ops::MerkleOps;

use super::conversion::{base_col_ref_to_simd, hash_col_ref_to_simd, hash_col_to_gpu};
use super::cuda_executor::is_cuda_available;
use super::GpuBackend;

/// Threshold for GPU acceleration (log2 of tree size).
#[cfg(test)]
const GPU_MERKLE_THRESHOLD_LOG_SIZE: u32 = 16; // GPU Merkle only when columns are pre-cached (>64K leaves)

/// Threshold for GPU Blake2s acceleration (log2 of tree size).
/// Blake2s is 32-bit bitwise ops — GPU-friendly for large trees >= 2^14 = 16K nodes.
const GPU_BLAKE2S_THRESHOLD_LOG_SIZE: u32 = 14;

impl MerkleOps<Blake2sMerkleHasher> for GpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        let _span = span!(Level::TRACE, "GpuBackend::commit_on_layer (Blake2s)").entered();

        #[cfg(feature = "cuda-runtime")]
        if log_size >= GPU_BLAKE2S_THRESHOLD_LOG_SIZE && is_cuda_available() {
            // Check precomputed cache first (from full-tree build)
            use super::memory::take_precomputed_blake2s_layer;
            if let Some(result) = take_precomputed_blake2s_layer(1usize << log_size) {
                return result;
            }
            return gpu_commit_on_layer_blake2s(log_size, prev_layer, columns);
        }

        commit_on_layer_simd_blake2s(log_size, prev_layer, columns)
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for GpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        let _span = span!(Level::TRACE, "GpuBackend::commit_on_layer (Blake2sM31)").entered();
        commit_on_layer_simd_blake2s_m31(log_size, prev_layer, columns)
    }
}

impl MerkleOps<Poseidon252MerkleHasher> for GpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, FieldElement252>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, FieldElement252> {
        let _span = span!(Level::TRACE, "GpuBackend::commit_on_layer (Poseidon252)").entered();

        // Try GPU path for large trees when CUDA is available
        #[cfg(feature = "cuda-runtime")]
        if log_size >= GPU_POSEIDON252_THRESHOLD_LOG_SIZE && is_cuda_available() {
            match gpu_commit_on_layer_poseidon252(log_size, prev_layer, columns) {
                Ok(result) => return result,
                Err(e) => {
                    tracing::warn!("GPU Poseidon252 Merkle failed: {}, falling back to SIMD", e);
                }
            }
        }

        // SIMD fallback
        let simd_columns: Vec<&Col<SimdBackend, BaseField>> = columns.iter()
            .map(|c| base_col_ref_to_simd(*c))
            .collect();

        <SimdBackend as MerkleOps<Poseidon252MerkleHasher>>::commit_on_layer(
            log_size,
            prev_layer,
            &simd_columns,
        )
    }
}

/// Threshold for GPU Poseidon252 acceleration (log2 of tree size).
/// Poseidon252 is algebraic and GPU-friendly — use GPU for trees >= 2^14 = 16K nodes.
const GPU_POSEIDON252_THRESHOLD_LOG_SIZE: u32 = 14;

// =============================================================================
// SIMD Fallback (for small trees)
// =============================================================================

fn commit_on_layer_simd_blake2s(
    log_size: u32,
    prev_layer: Option<&Col<GpuBackend, Blake2sHash>>,
    columns: &[&Col<GpuBackend, BaseField>],
) -> Col<GpuBackend, Blake2sHash> {
    // Convert using conversion module - hash columns are Vec<Blake2sHash> which is identical
    let simd_prev = prev_layer.map(|p| hash_col_ref_to_simd(p));
    
    // Base columns are BaseColumn which is identical between backends
    let simd_columns: Vec<&Col<SimdBackend, BaseField>> = columns.iter()
        .map(|c| base_col_ref_to_simd(*c))
        .collect();
    
    let result = <SimdBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
        log_size, 
        simd_prev, 
        &simd_columns
    );
    
    hash_col_to_gpu(result)
}

fn commit_on_layer_simd_blake2s_m31(
    log_size: u32,
    prev_layer: Option<&Col<GpuBackend, Blake2sHash>>,
    columns: &[&Col<GpuBackend, BaseField>],
) -> Col<GpuBackend, Blake2sHash> {
    // Convert using conversion module
    let simd_prev = prev_layer.map(|p| hash_col_ref_to_simd(p));
    
    let simd_columns: Vec<&Col<SimdBackend, BaseField>> = columns.iter()
        .map(|c| base_col_ref_to_simd(*c))
        .collect();
    
    let result = <SimdBackend as MerkleOps<Blake2sM31MerkleHasher>>::commit_on_layer(
        log_size, 
        simd_prev, 
        &simd_columns
    );
    
    hash_col_to_gpu(result)
}

// =============================================================================
// GPU Implementation
// =============================================================================

/// GPU-accelerated Blake2s Merkle tree layer commitment.
#[cfg(feature = "cuda-runtime")]
fn gpu_commit_on_layer_blake2s(
    log_size: u32,
    prev_layer: Option<&Col<GpuBackend, Blake2sHash>>,
    columns: &[&Col<GpuBackend, BaseField>],
) -> Col<GpuBackend, Blake2sHash> {
    use super::cuda_executor::{cuda_blake2s_merkle, get_cuda_executor};
    use super::memory::{take_cached_column_gpu, take_fri_column_gpu, cache_precomputed_blake2s_layer};

    let _span = span!(Level::INFO, "GPU commit_on_layer (Blake2s)", log_size = log_size).entered();

    let n_hashes = 1usize << log_size;

    // Try GPU-resident path: check if all columns are cached on GPU
    // Check both the general column cache and the FRI column cache
    let cached_slices: Vec<_> = columns.iter()
        .map(|col| {
            let ptr = col.data.as_ptr() as usize;
            let len = col.data.len();
            take_cached_column_gpu(ptr, len)
                .or_else(|| take_fri_column_gpu(ptr))
        })
        .collect();

    let all_cached = cached_slices.iter().all(|s| s.is_some());

    if all_cached && !cached_slices.is_empty() {
        tracing::info!("GPU Merkle Blake2s: ALL {} columns cached on GPU — zero H2D", columns.len());

        let gpu_slices: Vec<_> = cached_slices.into_iter().map(|s| s.unwrap()).collect();
        let gpu_slice_refs: Vec<&cudarc::driver::CudaSlice<u32>> = gpu_slices.iter().collect();
        let col_lengths: Vec<usize> = columns.iter().map(|c| c.as_slice().len()).collect();

        let is_leaf_layer = prev_layer.is_none() && !columns.is_empty();

        if let Ok(executor) = get_cuda_executor() {
            // FULL-TREE PATH: build entire Merkle tree in one GPU pass for leaf layers
            if is_leaf_layer {
                match executor.execute_blake2s_merkle_full_tree(
                    &gpu_slice_refs, &col_lengths, n_hashes,
                ) {
                    Ok(all_layers) => {
                        // Cache all layers except leaf (index 0) for subsequent commit_on_layer calls
                        for (idx, layer_data) in all_layers.iter().enumerate().skip(1) {
                            let layer_n = n_hashes >> idx;
                            if layer_n > 0 {
                                let hashes: Vec<Blake2sHash> = layer_data
                                    .chunks_exact(32)
                                    .map(|chunk| {
                                        let mut hash = [0u8; 32];
                                        hash.copy_from_slice(chunk);
                                        Blake2sHash(hash)
                                    })
                                    .collect();
                                cache_precomputed_blake2s_layer(layer_n, hashes);
                            }
                        }
                        tracing::info!("GPU Blake2s Merkle FULL TREE: {} leaf hashes, {} layers precomputed",
                            n_hashes, all_layers.len() - 1);
                        // Return leaf layer result
                        return all_layers[0]
                            .chunks_exact(32)
                            .map(|chunk| {
                                let mut hash = [0u8; 32];
                                hash.copy_from_slice(chunk);
                                Blake2sHash(hash)
                            })
                            .collect();
                    }
                    Err(e) => {
                        tracing::warn!("GPU Blake2s full-tree failed: {}, trying single-layer", e);
                    }
                }
            }

            // Single-layer fallback for non-leaf layers
            let prev_hashes: Option<Vec<u8>> = prev_layer.map(|prev| {
                prev.iter()
                    .flat_map(|hash| hash.as_ref().iter().copied())
                    .collect()
            });

            match executor.execute_blake2s_merkle_from_gpu(
                &gpu_slice_refs, &col_lengths, prev_hashes.as_deref(), n_hashes
            ) {
                Ok(result_bytes) => {
                    return result_bytes
                        .chunks_exact(32)
                        .map(|chunk| {
                            let mut hash = [0u8; 32];
                            hash.copy_from_slice(chunk);
                            Blake2sHash(hash)
                        })
                        .collect();
                }
                Err(e) => {
                    tracing::warn!("GPU Merkle from GPU failed: {}, falling back to standard path", e);
                }
            }
        }
        // gpu_slices dropped here, freeing GPU memory
    } else {
        // Return unclaimed cached slices (they'll be dropped/freed)
        // This is fine — partial cache hits aren't worth the complexity
        if cached_slices.iter().any(|s| s.is_some()) {
            tracing::debug!(
                "GPU Merkle: partial cache hit ({}/{}), using standard path",
                cached_slices.iter().filter(|s| s.is_some()).count(),
                columns.len()
            );
        }
    }

    // Standard path: H2D all columns
    let column_data: Vec<Vec<u32>> = columns.iter()
        .map(|col| {
            col.as_slice().iter().map(|f| f.0).collect()
        })
        .collect();

    let prev_hashes: Option<Vec<u8>> = prev_layer.map(|prev| {
        prev.iter()
            .flat_map(|hash| hash.as_ref().iter().copied())
            .collect()
    });

    match cuda_blake2s_merkle(&column_data, prev_hashes.as_deref(), n_hashes) {
        Ok(result_bytes) => {
            result_bytes
                .chunks_exact(32)
                .map(|chunk| {
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(chunk);
                    Blake2sHash(hash)
                })
                .collect()
        }
        Err(e) => {
            tracing::error!(
                "GPU Merkle hashing CUDA execution failed: {}. Falling back to SIMD.",
                e
            );
            commit_on_layer_simd_blake2s(log_size, prev_layer, columns)
        }
    }
}

/// GPU-accelerated Blake2sM31 Merkle tree layer commitment.
#[cfg(feature = "cuda-runtime")]
#[allow(dead_code)]
fn gpu_commit_on_layer_blake2s_m31(
    log_size: u32,
    prev_layer: Option<&Col<GpuBackend, Blake2sHash>>,
    columns: &[&Col<GpuBackend, BaseField>],
) -> Col<GpuBackend, Blake2sHash> {
    use crate::core::vcs::blake2_hash::reduce_to_m31;
    
    // Use the regular Blake2s implementation
    let mut result = gpu_commit_on_layer_blake2s(log_size, prev_layer, columns);
    
    // Reduce each hash to M31
    for hash in result.iter_mut() {
        hash.0 = reduce_to_m31(hash.0);
    }
    
    result
}

/// GPU-accelerated Poseidon252 Merkle tree layer commitment.
///
/// Uses a full-tree precomputation strategy: when called for the leaf layer
/// (columns present, no prev_layer), builds ALL subsequent layers in one GPU
/// pass (no per-layer sync/D2H). Subsequent calls return precomputed results.
#[cfg(feature = "cuda-runtime")]
fn gpu_commit_on_layer_poseidon252(
    log_size: u32,
    prev_layer: Option<&Col<GpuBackend, FieldElement252>>,
    columns: &[&Col<GpuBackend, BaseField>],
) -> Result<Col<GpuBackend, FieldElement252>, String> {
    use super::cuda_executor::{get_cuda_executor, upload_poseidon252_round_constants};
    use super::memory::{take_fri_column_gpu, take_precomputed_merkle_layer, cache_precomputed_merkle_layer};
    use std::sync::OnceLock;
    use cudarc::driver::CudaSlice;

    static RC_CACHE: OnceLock<Result<CudaSlice<u64>, String>> = OnceLock::new();

    let _span = span!(Level::INFO, "GPU commit_on_layer (Poseidon252)", log_size = log_size).entered();

    let n_hashes = 1usize << log_size;

    // Fast path: check if this layer was precomputed by a previous leaf-layer call
    if let Some(result_u64) = take_precomputed_merkle_layer(n_hashes) {
        use rayon::prelude::*;
        let result_fe: Vec<FieldElement252> = result_u64.par_chunks_exact(4)
            .map(|limbs| u64_limbs_to_felt252(limbs))
            .collect();
        tracing::debug!("Poseidon252 Merkle: precomputed layer (log_size={}, {} hashes)", log_size, n_hashes);
        return Ok(result_fe);
    }

    let executor = get_cuda_executor().map_err(|e| format!("{}", e))?;

    let d_rc = RC_CACHE.get_or_init(|| {
        upload_poseidon252_round_constants(&executor.device)
            .map_err(|e| format!("RC upload: {}", e))
    });
    let d_rc = d_rc.as_ref().map_err(|e| e.clone())?;

    // Check if all columns are cached on GPU (from FRI fold_line deinterleave).
    let gpu_cols: Vec<Option<CudaSlice<u32>>> = columns.iter()
        .map(|col| {
            let col_ptr = col.as_slice().as_ptr() as usize;
            take_fri_column_gpu(col_ptr)
        })
        .collect();
    let all_cols_cached = !gpu_cols.is_empty() && gpu_cols.iter().all(|c| c.is_some());

    // FULL-TREE PATH: if we have columns and no prev_layer (leaf layer for FRI commits),
    // build the entire tree in one GPU pass.
    let is_leaf_layer = prev_layer.is_none() && !columns.is_empty();

    if is_leaf_layer && all_cols_cached {
        let cached_slices: Vec<CudaSlice<u32>> = gpu_cols.into_iter()
            .map(|c| c.unwrap()).collect();
        let n_cols = cached_slices.len();
        let mut d_flat: CudaSlice<u32> = unsafe {
            executor.device.alloc::<u32>(n_cols * n_hashes)
        }.map_err(|e| format!("alloc: {:?}", e))?;
        for (i, d_col) in cached_slices.iter().enumerate() {
            let offset = i * n_hashes;
            executor.device.dtod_copy(d_col, &mut d_flat.slice_mut(offset..offset + n_hashes))
                .map_err(|e| format!("dtod: {:?}", e))?;
        }

        tracing::debug!("Poseidon252 Merkle: FULL TREE from GPU columns (log_size={}, {} layers)", log_size, log_size + 1);
        let all_layers = executor.execute_poseidon252_merkle_full_tree(
            &d_flat, n_cols, None, n_hashes, d_rc,
        ).map_err(|e| format!("{}", e))?;

        // Cache all layers except the leaf (index 0) which we return now
        for (idx, layer_data) in all_layers.iter().enumerate().skip(1) {
            let layer_n = n_hashes >> idx;
            if layer_n > 0 {
                cache_precomputed_merkle_layer(layer_n, layer_data.clone());
            }
        }

        use rayon::prelude::*;
        let result_fe: Vec<FieldElement252> = all_layers[0].par_chunks_exact(4)
            .map(|limbs| u64_limbs_to_felt252(limbs))
            .collect();
        tracing::info!("GPU Poseidon252 Merkle FULL TREE: {} leaf hashes, {} layers precomputed", n_hashes, all_layers.len() - 1);
        return Ok(result_fe);
    }

    if is_leaf_layer {
        // Columns on CPU, no prev_layer — upload columns and build full tree
        drop(gpu_cols);
        let column_data: Vec<Vec<u32>> = columns.iter()
            .map(|col| col.as_slice().iter().map(|f| f.0).collect())
            .collect();
        let n_cols = column_data.len();
        let flat_columns: Vec<u32> = column_data.iter()
            .flat_map(|col| col.iter().copied())
            .collect();
        let d_flat = executor.device.htod_sync_copy(&flat_columns)
            .map_err(|e| format!("H2D: {:?}", e))?;

        tracing::debug!("Poseidon252 Merkle: FULL TREE from CPU columns (log_size={})", log_size);
        let all_layers = executor.execute_poseidon252_merkle_full_tree(
            &d_flat, n_cols, None, n_hashes, d_rc,
        ).map_err(|e| format!("{}", e))?;

        for (idx, layer_data) in all_layers.iter().enumerate().skip(1) {
            let layer_n = n_hashes >> idx;
            if layer_n > 0 {
                cache_precomputed_merkle_layer(layer_n, layer_data.clone());
            }
        }

        use rayon::prelude::*;
        let result_fe: Vec<FieldElement252> = all_layers[0].par_chunks_exact(4)
            .map(|limbs| u64_limbs_to_felt252(limbs))
            .collect();
        tracing::info!("GPU Poseidon252 Merkle FULL TREE: {} leaf hashes, {} layers precomputed", n_hashes, all_layers.len() - 1);
        return Ok(result_fe);
    }

    // Non-leaf layer that wasn't precomputed (columns present at this layer too).
    // This happens for trace commits where columns span multiple sizes.
    // Fall back to per-layer approach.
    drop(gpu_cols);
    let column_data: Vec<Vec<u32>> = columns.iter()
        .map(|col| col.as_slice().iter().map(|f| f.0).collect())
        .collect();
    let prev_data: Option<Vec<u64>> = prev_layer.map(|prev| {
        prev.iter()
            .flat_map(|fe| {
                let bytes = fe.to_bytes_be();
                (0..4).map(move |i| {
                    let offset = 24 - i * 8;
                    let mut val = 0u64;
                    for j in 0..8 {
                        val = (val << 8) | bytes[offset + j] as u64;
                    }
                    val
                })
            })
            .collect()
    });
    let result = executor.execute_poseidon252_merkle(
        &column_data, prev_data.as_deref(), n_hashes, d_rc,
    ).map_err(|e| format!("{}", e))?;

    use rayon::prelude::*;
    let result_fe: Vec<FieldElement252> = result.par_chunks_exact(4)
        .map(|limbs| u64_limbs_to_felt252(limbs))
        .collect();
    tracing::info!("GPU Poseidon252 Merkle: {} hashes at log_size={} (per-layer fallback)", n_hashes, log_size);
    Ok(result_fe)
}

/// Convert 4 u64 limbs to a FieldElement252.
#[cfg(feature = "cuda-runtime")]
fn u64_limbs_to_felt252(limbs: &[u64]) -> FieldElement252 {
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let offset = 24 - i * 8;
        let val = limbs[i];
        for j in 0..8 {
            bytes[offset + j] = ((val >> (56 - j * 8)) & 0xFF) as u8;
        }
    }
    bytes[0] &= 0x07;
    FieldElement252::from_bytes_be(&bytes).expect("valid felt252 from GPU")
}

#[cfg(not(feature = "cuda-runtime"))]
fn gpu_commit_on_layer_blake2s(
    _log_size: u32,
    _prev_layer: Option<&Col<GpuBackend, Blake2sHash>>,
    _columns: &[&Col<GpuBackend, BaseField>],
) -> Col<GpuBackend, Blake2sHash> {
    panic!("GPU Merkle hashing requires cuda-runtime feature");
}

#[cfg(not(feature = "cuda-runtime"))]
fn gpu_commit_on_layer_blake2s_m31(
    _log_size: u32,
    _prev_layer: Option<&Col<GpuBackend, Blake2sHash>>,
    _columns: &[&Col<GpuBackend, BaseField>],
) -> Col<GpuBackend, Blake2sHash> {
    panic!("GPU Merkle hashing requires cuda-runtime feature");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_threshold_reasonable() {
        assert!(GPU_MERKLE_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_MERKLE_THRESHOLD_LOG_SIZE <= 22);
    }
}
