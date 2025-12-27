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

use tracing::{span, Level};

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::vcs::ops::MerkleOps;

use super::conversion::{base_col_ref_to_simd, hash_col_ref_to_simd, hash_col_to_gpu};
use super::cuda_executor::is_cuda_available;
use super::GpuBackend;

/// Threshold for GPU acceleration (log2 of tree size)
const GPU_MERKLE_THRESHOLD_LOG_SIZE: u32 = 16; // 64K leaves

impl MerkleOps<Blake2sMerkleHasher> for GpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        let _span = span!(Level::TRACE, "GpuBackend::commit_on_layer (Blake2s)").entered();
        
        // Small trees: use SIMD (GPU overhead not worth it)
        if log_size < GPU_MERKLE_THRESHOLD_LOG_SIZE {
            tracing::debug!(
                "GPU Merkle: using SIMD for small size (log_size={} < threshold={})",
                log_size, GPU_MERKLE_THRESHOLD_LOG_SIZE
            );
            return commit_on_layer_simd_blake2s(log_size, prev_layer, columns);
        }
        
        // Large trees: prefer GPU, fallback to SIMD if unavailable
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::commit_on_layer: CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded. For optimal performance, ensure CUDA is available.",
                log_size
            );
            return commit_on_layer_simd_blake2s(log_size, prev_layer, columns);
        }

        tracing::info!(
            "GPU Merkle: using CUDA for {} leaves (log_size={})",
            1u64 << log_size, log_size
        );

        gpu_commit_on_layer_blake2s(log_size, prev_layer, columns)
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for GpuBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Col<Self, Blake2sHash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        let _span = span!(Level::TRACE, "GpuBackend::commit_on_layer (Blake2sM31)").entered();
        
        // Small trees: use SIMD
        if log_size < GPU_MERKLE_THRESHOLD_LOG_SIZE {
            return commit_on_layer_simd_blake2s_m31(log_size, prev_layer, columns);
        }
        
        // Large trees: prefer GPU, fallback to SIMD if unavailable
        if !is_cuda_available() {
            tracing::warn!(
                "GpuBackend::commit_on_layer (M31): CUDA unavailable for log_size={}, falling back to SIMD. \
                 Performance will be degraded. For optimal performance, ensure CUDA is available.",
                log_size
            );
            return commit_on_layer_simd_blake2s_m31(log_size, prev_layer, columns);
        }

        tracing::info!(
            "GPU Merkle (M31): using CUDA for {} leaves (log_size={})",
            1u64 << log_size, log_size
        );

        gpu_commit_on_layer_blake2s_m31(log_size, prev_layer, columns)
    }
}

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
    use super::cuda_executor::cuda_blake2s_merkle;
    
    let _span = span!(Level::INFO, "GPU commit_on_layer (Blake2s)", log_size = log_size).entered();
    
    let n_hashes = 1usize << log_size;
    
    // Flatten column data
    let column_data: Vec<Vec<u32>> = columns.iter()
        .map(|col| {
            col.as_slice().iter().map(|f| f.0).collect()
        })
        .collect();
    
    // Flatten previous layer hashes (if any)
    let prev_hashes: Option<Vec<u8>> = prev_layer.map(|prev| {
        prev.iter()
            .flat_map(|hash| hash.as_ref().iter().copied())
            .collect()
    });
    
    // Execute GPU Merkle hashing
    match cuda_blake2s_merkle(&column_data, prev_hashes.as_deref(), n_hashes) {
        Ok(result_bytes) => {
            // Convert bytes back to Blake2sHash
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
            // Fallback to SIMD on CUDA execution failure
            commit_on_layer_simd_blake2s(log_size, prev_layer, columns)
        }
    }
}

/// GPU-accelerated Blake2sM31 Merkle tree layer commitment.
#[cfg(feature = "cuda-runtime")]
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
        assert!(GPU_MERKLE_THRESHOLD_LOG_SIZE <= 20);
    }
}
