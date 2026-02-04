//! GPU-accelerated polynomial operations.
//!
//! This module implements [`PolyOps`] for [`GpuBackend`].
//!
//! # Strategy
//!
//! FFT/IFFT operations delegate to the SIMD backend which uses rayon for parallelism.
//! The GPU accelerates Merkle commitments and FRI folding (see merkle.rs, fri.rs).
//!
//! The SIMD backend's FFT uses a transpose-based algorithm (above CACHED_FFT_LOG_SIZE=16)
//! that produces coefficients in a representation specific to its butterfly network.
//! The GPU kernel's scalar FFT uses a different butterfly ordering that is self-consistent
//! but cross-incompatible with SIMD at sizes > 2^16. Both produce correct roundtrips
//! independently, but SIMD coefficients cannot be evaluated by GPU FFT and vice versa.
//!
//! The [`GpuProofPipeline`] in pipeline.rs provides 50-100x GPU acceleration by keeping
//! all data on GPU and using the GPU's own FFT throughout the full proof pipeline.

use tracing::{span, Level};

use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
#[cfg(feature = "cuda-runtime")]
use crate::core::ColumnVec;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;

use super::conversion::{
    circle_coeffs_ref_to_simd, circle_eval_ref_to_simd, twiddle_ref_to_simd, twiddle_to_gpu,
};
use super::GpuBackend;

impl PolyOps for GpuBackend {
    type Twiddles = <SimdBackend as PolyOps>::Twiddles;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        interpolate_simd_fallback(eval, twiddles)
    }

    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        let simd_twiddles = twiddle_ref_to_simd(twiddles);
        let simd_columns = columns.into_iter().map(|eval| {
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                eval.domain, eval.values,
            )
        });
        let results = SimdBackend::interpolate_columns(simd_columns, simd_twiddles);
        results.into_iter().map(|r| CircleCoefficients::new(r.coeffs)).collect()
    }

    #[cfg(feature = "cuda-runtime")]
    fn evaluate_polynomials(
        polynomials: ColumnVec<CircleCoefficients<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
        store_polynomials_coefficients: bool,
    ) -> Vec<crate::prover::air::component_prover::Poly<Self>>
    where
        Self: crate::prover::backend::Backend,
    {
        use crate::prover::air::component_prover::Poly;

        let simd_twiddles = twiddle_ref_to_simd(twiddles);
        let simd_polys: ColumnVec<CircleCoefficients<SimdBackend>> = polynomials
            .iter()
            .map(|p| CircleCoefficients::<SimdBackend>::new(p.coeffs.clone()))
            .collect();

        let simd_results = SimdBackend::evaluate_polynomials(
            simd_polys, log_blowup_factor, simd_twiddles, false,
        );

        simd_results
            .into_iter()
            .zip(polynomials)
            .map(|(simd_poly, original_coeffs)| {
                let evals: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder> =
                    CircleEvaluation::new(simd_poly.evals.domain, simd_poly.evals.values);
                Poly::new(
                    store_polynomials_coefficients.then_some(original_coeffs),
                    evals,
                )
            })
            .collect()
    }

    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        let simd_poly = circle_coeffs_ref_to_simd(poly);
        SimdBackend::eval_at_point(simd_poly, point)
    }

    fn barycentric_weights(
        coset: CanonicCoset,
        p: CirclePoint<SecureField>,
    ) -> Col<Self, SecureField> {
        SimdBackend::barycentric_weights(coset, p)
    }

    fn barycentric_eval_at_point(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        weights: &Col<Self, SecureField>,
    ) -> SecureField {
        let simd_evals = circle_eval_ref_to_simd(evals);
        SimdBackend::barycentric_eval_at_point(simd_evals, weights)
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        let simd_evals = circle_eval_ref_to_simd(evals);
        let simd_twiddles = twiddle_ref_to_simd(twiddles);
        SimdBackend::eval_at_point_by_folding(simd_evals, point, simd_twiddles)
    }

    fn extend(poly: &CircleCoefficients<Self>, log_size: u32) -> CircleCoefficients<Self> {
        let simd_poly = circle_coeffs_ref_to_simd(poly);
        let result = SimdBackend::extend(simd_poly, log_size);
        CircleCoefficients::new(result.coeffs)
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        evaluate_simd_fallback(poly, domain, twiddles)
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        let _span = span!(Level::TRACE, "GpuBackend::precompute_twiddles").entered();
        let simd_twiddles = SimdBackend::precompute_twiddles(coset);
        twiddle_to_gpu(simd_twiddles)
    }

    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let simd_poly = CircleCoefficients::<SimdBackend>::new(poly.coeffs);
        let (left, right) = SimdBackend::split_at_mid(simd_poly);
        (
            CircleCoefficients::new(left.coeffs),
            CircleCoefficients::new(right.coeffs),
        )
    }
}

fn interpolate_simd_fallback(
    eval: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<GpuBackend>,
) -> CircleCoefficients<GpuBackend> {
    let simd_eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
        eval.domain,
        eval.values,
    );
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let result = SimdBackend::interpolate(simd_eval, simd_twiddles);
    CircleCoefficients::new(result.coeffs)
}

fn evaluate_simd_fallback(
    poly: &CircleCoefficients<GpuBackend>,
    domain: CircleDomain,
    twiddles: &TwiddleTree<GpuBackend>,
) -> CircleEvaluation<GpuBackend, BaseField, BitReversedOrder> {
    let simd_poly = circle_coeffs_ref_to_simd(poly);
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let result = SimdBackend::evaluate(simd_poly, domain, simd_twiddles);
    CircleEvaluation::new(domain, result.values)
}
