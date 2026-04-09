//! GPU-accelerated accumulation operations.
//!
//! This module implements [`AccumulationOps`] for [`GpuBackend`].
//!
//! # Note
//!
//! Accumulation is a simple element-wise addition. GPU could help for very
//! large columns, but the overhead usually isn't worth it. We delegate to SIMD.

use crate::core::fields::qm31::SecureField;
use crate::prover::AccumulationOps;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::secure_column::SecureColumnByCoords;

use super::conversion::secure_col_coords_mut_to_simd;
use super::conversion::secure_col_coords_ref_to_simd;
use super::GpuBackend;

impl AccumulationOps for GpuBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        // Accumulation is a simple element-wise addition
        // GPU could help for very large columns, but overhead usually not worth it
        let simd_column = secure_col_coords_mut_to_simd(column);
        let simd_other = secure_col_coords_ref_to_simd(other);
        SimdBackend::accumulate(simd_column, simd_other);
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        // Power generation is sequential, no GPU benefit
        SimdBackend::generate_secure_powers(felt, n_powers)
    }

    fn lift_and_accumulate(
        cols: Vec<SecureColumnByCoords<Self>>,
    ) -> Option<SecureColumnByCoords<Self>> {
        // GpuBackend and SimdBackend share the same column layout (BaseColumn = Vec<PackedM31>),
        // so we can safely reinterpret the columns between backends.
        let simd_cols: Vec<SecureColumnByCoords<SimdBackend>> = cols
            .into_iter()
            .map(|col| SecureColumnByCoords { columns: col.columns })
            .collect();
        let result = SimdBackend::lift_and_accumulate(simd_cols)?;
        Some(SecureColumnByCoords { columns: result.columns })
    }
}
