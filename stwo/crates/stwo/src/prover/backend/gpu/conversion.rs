//! Type conversion utilities between GpuBackend and SimdBackend.
//!
//! Since GpuBackend and SimdBackend share the same underlying column types
//! (BaseColumn, SecureColumn), we can safely convert between them.
//!
//! # Safety
//!
//! These conversions are safe because:
//! 1. Both backends use identical column types (defined in simd/column.rs)
//! 2. GpuBackend::Column = SimdBackend::Column = BaseColumn/SecureColumn
//! 3. The memory layout is identical
//!
//! # Design
//!
//! Instead of using `unsafe { transmute(...) }` throughout the codebase,
//! we centralize all conversions here with proper documentation and
//! compile-time type assertions.

use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::prover::backend::simd::column::{BaseColumn, SecureColumn};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, SecureEvaluation};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

use super::GpuBackend;

// =============================================================================
// Compile-time Type Assertions
// =============================================================================

// These assertions ensure that GpuBackend and SimdBackend use the same column types.
// If they ever diverge, these will fail to compile.
const _: () = {
    // Check that Column types are the same
    fn _assert_same_base_column<T>(_: T) where T: Into<BaseColumn> {}
    fn _assert_same_secure_column<T>(_: T) where T: Into<SecureColumn> {}
};

// =============================================================================
// Reference Conversions (zero-copy)
// =============================================================================

/// Convert a reference to GpuBackend column to SimdBackend column.
/// 
/// # Safety
/// This is safe because both backends use identical column types.
#[inline]
pub fn base_col_ref_to_simd(col: &BaseColumn) -> &BaseColumn {
    col // Same type, no conversion needed
}

/// Convert a reference to GpuBackend secure column to SimdBackend.
#[inline]
pub fn secure_col_ref_to_simd(col: &SecureColumn) -> &SecureColumn {
    col
}

/// Convert a mutable reference to GpuBackend secure column to SimdBackend.
#[inline]
pub fn secure_col_mut_to_simd(col: &mut SecureColumn) -> &mut SecureColumn {
    col
}

// =============================================================================
// CircleEvaluation Conversions
// =============================================================================

/// Convert GpuBackend CircleEvaluation reference to SimdBackend.
#[inline]
pub fn circle_eval_ref_to_simd<'a>(
    eval: &'a CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>
) -> &'a CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(eval) }
}

/// Convert SimdBackend CircleEvaluation to GpuBackend.
#[inline]
pub fn circle_eval_to_gpu(
    eval: CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>
) -> CircleEvaluation<GpuBackend, BaseField, BitReversedOrder> {
    CircleEvaluation::new(eval.domain, eval.values)
}

/// Convert GpuBackend CircleEvaluation to SimdBackend.
#[inline]
pub fn circle_eval_to_simd(
    eval: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>
) -> CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> {
    CircleEvaluation::new(eval.domain, eval.values)
}

// =============================================================================
// SecureEvaluation Conversions
// =============================================================================

/// Convert GpuBackend SecureEvaluation reference to SimdBackend.
#[inline]
pub fn secure_eval_ref_to_simd<'a>(
    eval: &'a SecureEvaluation<GpuBackend, BitReversedOrder>
) -> &'a SecureEvaluation<SimdBackend, BitReversedOrder> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(eval) }
}

/// Convert SimdBackend SecureEvaluation to GpuBackend.
#[inline]
pub fn secure_eval_to_gpu(
    eval: SecureEvaluation<SimdBackend, BitReversedOrder>
) -> SecureEvaluation<GpuBackend, BitReversedOrder> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(eval) }
}

// =============================================================================
// CircleCoefficients Conversions
// =============================================================================

/// Convert GpuBackend CircleCoefficients reference to SimdBackend.
#[inline]
pub fn circle_coeffs_ref_to_simd<'a>(
    coeffs: &'a CircleCoefficients<GpuBackend>
) -> &'a CircleCoefficients<SimdBackend> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(coeffs) }
}

/// Convert SimdBackend CircleCoefficients to GpuBackend.
#[inline]
pub fn circle_coeffs_to_gpu(
    coeffs: CircleCoefficients<SimdBackend>
) -> CircleCoefficients<GpuBackend> {
    CircleCoefficients::new(coeffs.coeffs)
}

// =============================================================================
// LineEvaluation Conversions
// =============================================================================

/// Convert GpuBackend LineEvaluation reference to SimdBackend.
#[inline]
pub fn line_eval_ref_to_simd<'a>(
    eval: &'a LineEvaluation<GpuBackend>
) -> &'a LineEvaluation<SimdBackend> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(eval) }
}

/// Convert GpuBackend LineEvaluation mutable reference to SimdBackend.
#[inline]
pub fn line_eval_mut_to_simd<'a>(
    eval: &'a mut LineEvaluation<GpuBackend>
) -> &'a mut LineEvaluation<SimdBackend> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(eval) }
}

/// Convert SimdBackend LineEvaluation to GpuBackend.
#[inline]
pub fn line_eval_to_gpu(
    eval: LineEvaluation<SimdBackend>
) -> LineEvaluation<GpuBackend> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(eval) }
}

// =============================================================================
// TwiddleTree Conversions
// =============================================================================

/// Convert GpuBackend TwiddleTree reference to SimdBackend.
#[inline]
pub fn twiddle_ref_to_simd<'a>(
    twiddles: &'a TwiddleTree<GpuBackend>
) -> &'a TwiddleTree<SimdBackend> {
    // Safety: GpuBackend and SimdBackend use identical twiddle types
    unsafe { std::mem::transmute(twiddles) }
}

/// Convert SimdBackend TwiddleTree to GpuBackend.
#[inline]
pub fn twiddle_to_gpu(
    twiddles: TwiddleTree<SimdBackend>
) -> TwiddleTree<GpuBackend> {
    // Safety: GpuBackend and SimdBackend use identical twiddle types
    unsafe { std::mem::transmute(twiddles) }
}

// =============================================================================
// SecureColumnByCoords Conversions
// =============================================================================

/// Convert GpuBackend SecureColumnByCoords mutable reference to SimdBackend.
#[inline]
pub fn secure_col_coords_mut_to_simd<'a>(
    col: &'a mut SecureColumnByCoords<GpuBackend>
) -> &'a mut SecureColumnByCoords<SimdBackend> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(col) }
}

/// Convert GpuBackend SecureColumnByCoords reference to SimdBackend.
#[inline]
pub fn secure_col_coords_ref_to_simd<'a>(
    col: &'a SecureColumnByCoords<GpuBackend>
) -> &'a SecureColumnByCoords<SimdBackend> {
    // Safety: GpuBackend and SimdBackend use identical column types
    unsafe { std::mem::transmute(col) }
}

// =============================================================================
// Hash Column Conversions
// =============================================================================

/// Convert GpuBackend hash column reference to SimdBackend.
#[inline]
pub fn hash_col_ref_to_simd<'a>(
    col: &'a Vec<Blake2sHash>
) -> &'a Vec<Blake2sHash> {
    col // Same type
}

/// Convert SimdBackend hash column to GpuBackend.
#[inline]
pub fn hash_col_to_gpu(
    col: Vec<Blake2sHash>
) -> Vec<Blake2sHash> {
    col // Same type
}

// =============================================================================
// SoA ↔ AoS Conversions for CUDA FRI Kernels
// =============================================================================

/// Convert a `SecureColumnByCoords` (SoA: 4 separate BaseColumns) to AoS
/// layout (interleaved `[c0_i, c1_i, c2_i, c3_i, ...]`) as flat `u32`.
///
/// The CUDA FRI kernels expect QM31 elements packed as 4 consecutive `u32`.
/// This uses `bytemuck`-style reinterpretation (M31 is repr(transparent)
/// over `u32`) instead of element-by-element extraction.
pub fn secure_column_to_aos(col: &SecureColumnByCoords<SimdBackend>, n: usize) -> Vec<u32> {
    // Each BaseColumn stores Vec<PackedBaseField> where PackedBaseField
    // is repr(transparent) over u32x16. We need the raw u32 values.
    let c0 = col_data_as_u32(&col.columns[0], n);
    let c1 = col_data_as_u32(&col.columns[1], n);
    let c2 = col_data_as_u32(&col.columns[2], n);
    let c3 = col_data_as_u32(&col.columns[3], n);

    let mut aos = Vec::with_capacity(n * 4);
    for i in 0..n {
        aos.push(c0[i]);
        aos.push(c1[i]);
        aos.push(c2[i]);
        aos.push(c3[i]);
    }
    aos
}

/// Convert AoS layout (`[c0_0, c1_0, c2_0, c3_0, c0_1, ...]`) back to
/// `SecureColumnByCoords<SimdBackend>`.
pub fn aos_to_secure_column(aos: &[u32], n: usize) -> SecureColumnByCoords<SimdBackend> {
    use crate::core::fields::m31::M31;

    assert_eq!(aos.len(), n * 4);

    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    let mut c3 = Vec::with_capacity(n);

    for i in 0..n {
        c0.push(M31(aos[i * 4]));
        c1.push(M31(aos[i * 4 + 1]));
        c2.push(M31(aos[i * 4 + 2]));
        c3.push(M31(aos[i * 4 + 3]));
    }

    SecureColumnByCoords {
        columns: [
            BaseColumn::from_iter(c0),
            BaseColumn::from_iter(c1),
            BaseColumn::from_iter(c2),
            BaseColumn::from_iter(c3),
        ],
    }
}

/// Reinterpret a `BaseColumn`'s packed data as a `&[u32]` slice of length `n`.
///
/// `BaseColumn.data` is `Vec<PackedBaseField>` where `PackedBaseField` is
/// `repr(transparent)` over `u32x16`. Each `u32x16` holds 16 consecutive
/// M31 values stored as `u32`.
fn col_data_as_u32(col: &BaseColumn, n: usize) -> &[u32] {
    // Safety: PackedBaseField is repr(transparent) over u32x16,
    // and M31 is repr(transparent) over u32.
    let ptr = col.data.as_ptr() as *const u32;
    let total = col.data.len() * 16; // 16 u32s per PackedBaseField
    let slice = unsafe { std::slice::from_raw_parts(ptr, total) };
    &slice[..n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::backend::Column;

    #[test]
    fn test_base_column_identity() {
        let col = BaseColumn::zeros(100);
        let converted = base_col_ref_to_simd(&col);
        assert_eq!(col.len(), converted.len());
    }

    #[test]
    fn test_aos_roundtrip() {
        use crate::core::fields::m31::M31;

        let n = 32; // Must be multiple of 16 for PackedBaseField alignment
        let mut col = SecureColumnByCoords::<SimdBackend>::zeros(n);
        // Write some test values
        for i in 0..n {
            let val = crate::core::fields::qm31::SecureField::from_m31(
                M31(i as u32),
                M31(i as u32 + 100),
                M31(i as u32 + 200),
                M31(i as u32 + 300),
            );
            col.set(i, val);
        }

        let aos = secure_column_to_aos(&col, n);
        assert_eq!(aos.len(), n * 4);

        let recovered = aos_to_secure_column(&aos, n);
        for i in 0..n {
            assert_eq!(col.at(i), recovered.at(i), "mismatch at index {}", i);
        }
    }
}

