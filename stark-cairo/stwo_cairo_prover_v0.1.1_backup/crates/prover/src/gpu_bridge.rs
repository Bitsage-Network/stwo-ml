//! SimdToGpu bridge for zero-copy column transmutation.
//!
//! This module provides [`SimdToGpuTreeBuilder`], an adapter that implements
//! [`TreeBuilder<SimdBackend>`] but delegates to a GPU-backed tree builder internally.
//!
//! # Architecture
//!
//! Cairo witness generation is hardcoded to `SimdBackend` (96 files produce
//! `CircleEvaluation<SimdBackend, ...>`). Rather than genericizing all witness code,
//! we exploit the fact that `GpuBackend` and `SimdBackend` use **identical column types**
//! (`Vec<PackedM31>`). A zero-copy transmute converts the type parameter, then all
//! subsequent STARK proving runs on `GpuBackend` (GPU FFT, FRI, Merkle, constraints).
//!
//! This pattern is already used throughout `stwo/crates/stwo/src/prover/backend/gpu/conversion.rs`.

#[cfg(feature = "cuda-runtime")]
use stwo::core::channel::MerkleChannel;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::m31::M31;
#[cfg(feature = "cuda-runtime")]
use stwo::core::pcs::TreeSubspan;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::GpuBackend;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::simd::SimdBackend;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::BackendForChannel;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::poly::circle::CircleEvaluation;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::poly::BitReversedOrder;

#[cfg(feature = "cuda-runtime")]
use crate::witness::utils::TreeBuilder;

/// Adapter that accepts `SimdBackend` evaluations and commits them via `GpuBackend`.
///
/// Witness generators call `extend_evals` with `CircleEvaluation<SimdBackend, ...>`.
/// This adapter transmutes them to `CircleEvaluation<GpuBackend, ...>` (zero-copy,
/// identical memory layout) and delegates to the inner `GpuBackend` tree builder
/// for GPU-accelerated interpolation and Merkle commitment.
#[cfg(feature = "cuda-runtime")]
pub struct SimdToGpuTreeBuilder<'a, 'b, MC: MerkleChannel>
where
    GpuBackend: BackendForChannel<MC>,
{
    inner: stwo::prover::TreeBuilder<'a, 'b, GpuBackend, MC>,
}

#[cfg(feature = "cuda-runtime")]
impl<'a, 'b, MC: MerkleChannel> SimdToGpuTreeBuilder<'a, 'b, MC>
where
    GpuBackend: BackendForChannel<MC>,
{
    /// Wrap a `GpuBackend` tree builder so it can accept `SimdBackend` evaluations.
    pub fn new(inner: stwo::prover::TreeBuilder<'a, 'b, GpuBackend, MC>) -> Self {
        Self { inner }
    }

    /// Commit the accumulated polynomials.
    pub fn commit(self, channel: &mut MC::C) {
        self.inner.commit(channel);
    }
}

#[cfg(feature = "cuda-runtime")]
impl<MC: MerkleChannel> TreeBuilder<SimdBackend> for SimdToGpuTreeBuilder<'_, '_, MC>
where
    GpuBackend: BackendForChannel<MC>,
{
    fn extend_evals(
        &mut self,
        columns: impl IntoIterator<Item = CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ) -> TreeSubspan {
        // Zero-copy transmute: SimdBackend and GpuBackend have identical column layouts.
        // This is the same pattern used in stwo/crates/stwo/src/prover/backend/gpu/conversion.rs.
        let gpu_columns: Vec<CircleEvaluation<GpuBackend, M31, BitReversedOrder>> = columns
            .into_iter()
            .map(|eval| {
                // Safety: GpuBackend::Column == SimdBackend::Column == BaseColumn (Vec<PackedM31>).
                // CircleEvaluation<B, F, O> is { domain: CircleDomain, values: Col<B, F> }.
                // The domain is backend-independent, and the column type is identical.
                unsafe { std::mem::transmute(eval) }
            })
            .collect();
        self.inner.extend_evals(gpu_columns)
    }
}
