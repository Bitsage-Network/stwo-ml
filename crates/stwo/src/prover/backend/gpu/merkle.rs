//! GPU-accelerated Merkle tree operations.
//!
//! This module implements [`MerkleOpsLifted`] for [`GpuBackend`].
//!
//! # Algorithm
//!
//! Merkle tree commitment involves:
//! 1. Hashing leaf data (columns) using Blake2s or Poseidon252
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
//!
//! Currently, the GPU backend delegates to `SimdBackend`'s `MerkleOpsLifted`
//! implementation. CUDA-specific acceleration can be restored later.

use starknet_ff::FieldElement as FieldElement252;
use tracing::{span, Level};

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
use crate::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

use super::GpuBackend;

/// Threshold for GPU acceleration (log2 of tree size)
#[allow(dead_code)]
const GPU_MERKLE_THRESHOLD_LOG_SIZE: u32 = 16;

impl<const IS_M31_OUTPUT: bool> MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>
    for GpuBackend
{
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, Blake2sHash> {
        let _span = span!(
            Level::TRACE,
            "GpuBackend::build_leaves (Blake2s)",
            is_m31 = IS_M31_OUTPUT
        )
        .entered();

        // Delegate to SimdBackend. The column types are identical between
        // GpuBackend and SimdBackend (both use BaseColumn / Vec<Blake2sHash>),
        // so we can safely reinterpret the references.
        let simd_columns: Vec<&Col<SimdBackend, BaseField>> = columns
            .iter()
            .map(|c| *c as &Col<SimdBackend, BaseField>)
            .collect();

        <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_leaves(
            &simd_columns,
        )
    }

    fn build_next_layer(
        prev_layer: &Col<Self, Blake2sHash>,
    ) -> Col<Self, Blake2sHash> {
        let _span = span!(
            Level::TRACE,
            "GpuBackend::build_next_layer (Blake2s)",
            is_m31 = IS_M31_OUTPUT
        )
        .entered();

        <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_next_layer(
            prev_layer,
        )
    }
}

impl MerkleOpsLifted<Poseidon252MerkleHasher> for GpuBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, FieldElement252> {
        let _span =
            span!(Level::TRACE, "GpuBackend::build_leaves (Poseidon252)").entered();

        let simd_columns: Vec<&Col<SimdBackend, BaseField>> = columns
            .iter()
            .map(|c| *c as &Col<SimdBackend, BaseField>)
            .collect();

        <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_leaves(
            &simd_columns,
        )
    }

    fn build_next_layer(
        prev_layer: &Col<Self, FieldElement252>,
    ) -> Col<Self, FieldElement252> {
        let _span =
            span!(Level::TRACE, "GpuBackend::build_next_layer (Poseidon252)").entered();

        <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_next_layer(
            prev_layer,
        )
    }
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
