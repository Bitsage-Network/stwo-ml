//! GPU-accelerated lifted Merkle tree operations.
//!
//! Implements [`MerkleOpsLifted`] for [`GpuBackend`] by delegating to [`SimdBackend`].
//!
//! Since GpuBackend and SimdBackend share the same column types (BaseColumn,
//! SecureColumn, Vec<Blake2sHash>, Vec<FieldElement252>), the delegation is
//! zero-copy: we reinterpret the reference types and call the SIMD implementation
//! directly.

use starknet_ff::FieldElement as FieldElement252;

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
use crate::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

use super::conversion::base_col_ref_to_simd;
use super::GpuBackend;

// =============================================================================
// Blake2s Lifted Merkle (generic over IS_M31_OUTPUT)
// =============================================================================

impl<const IS_M31_OUTPUT: bool> MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>
    for GpuBackend
{
    fn build_leaves(columns: &[&Col<Self, BaseField>]) -> Col<Self, Blake2sHash> {
        // GpuBackend::Column<BaseField> = BaseColumn = SimdBackend::Column<BaseField>
        // GpuBackend::Column<Blake2sHash> = Vec<Blake2sHash> = SimdBackend::Column<Blake2sHash>
        // So we can pass through directly.
        let simd_columns: Vec<&Col<SimdBackend, BaseField>> =
            columns.iter().map(|c| base_col_ref_to_simd(*c)).collect();
        <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_leaves(
            &simd_columns,
        )
    }

    fn build_next_layer(prev_layer: &Col<Self, Blake2sHash>) -> Col<Self, Blake2sHash> {
        // Vec<Blake2sHash> is identical between backends
        <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>>::build_next_layer(
            prev_layer,
        )
    }
}

// =============================================================================
// Poseidon252 Lifted Merkle
// =============================================================================

impl MerkleOpsLifted<Poseidon252MerkleHasher> for GpuBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
    ) -> Col<Self, FieldElement252> {
        let simd_columns: Vec<&Col<SimdBackend, BaseField>> =
            columns.iter().map(|c| base_col_ref_to_simd(*c)).collect();
        <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_leaves(&simd_columns)
    }

    fn build_next_layer(
        prev_layer: &Col<Self, FieldElement252>,
    ) -> Col<Self, FieldElement252> {
        // Vec<FieldElement252> is identical between backends
        <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_next_layer(prev_layer)
    }
}
