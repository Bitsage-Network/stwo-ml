//! GPU-accelerated polynomial operations.
//!
//! This module implements [`PolyOps`] for [`GpuBackend`].
//!
//! # Strategy
//!
//! For CUDA builds we use a GPU-first path for interpolation/extension and keep
//! intermediate columns resident on GPU using the column cache in `memory.rs`.
//! This removes repeated CPU<->GPU transfers in the hottest PCS operations.
//!
//! Soundness hardening controls:
//! - `STWO_GPU_POLY_STRICT=1`: fail-closed (no silent SIMD fallback on GPU errors)
//! - `STWO_GPU_POLY_HARDEN=1`: cross-check GPU outputs against SIMD and panic on mismatch

use tracing::{span, Level};

use super::conversion::{
    circle_coeffs_ref_to_simd, circle_eval_ref_to_simd, twiddle_ref_to_simd, twiddle_to_gpu,
};
#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::{get_cuda_executor, is_cuda_available, CudaFftError};
#[cfg(feature = "cuda-runtime")]
use super::fft::{extract_itwiddles_for_gpu, extract_twiddles_for_gpu};
#[cfg(feature = "cuda-runtime")]
use super::memory::{cache_column_gpu, take_cached_column_gpu};
use super::GpuBackend;
use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
#[cfg(feature = "cuda-runtime")]
use crate::core::fields::m31::M31;
#[cfg(feature = "cuda-runtime")]
use crate::core::fields::m31::P;
use crate::core::fields::qm31::SecureField;
#[cfg(feature = "cuda-runtime")]
use crate::core::fields::FieldExpOps;
use crate::core::poly::circle::{CanonicCoset, CircleDomain};
#[cfg(feature = "cuda-runtime")]
use crate::core::poly::line::LineDomain;
#[cfg(feature = "cuda-runtime")]
use crate::core::poly::utils::domain_line_twiddles_from_tree;
#[cfg(feature = "cuda-runtime")]
use crate::core::poly::utils::get_folding_alphas;
#[cfg(feature = "cuda-runtime")]
use crate::core::ColumnVec;
#[cfg(feature = "cuda-runtime")]
use crate::prover::air::component_prover::Poly;
#[cfg(feature = "cuda-runtime")]
use crate::prover::backend::simd::fft::CACHED_FFT_LOG_SIZE;
#[cfg(feature = "cuda-runtime")]
use crate::prover::backend::simd::m31::LOG_N_LANES;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Col;
#[cfg(feature = "cuda-runtime")]
use crate::prover::fri::FriOps;
#[cfg(feature = "cuda-runtime")]
use crate::prover::line::LineEvaluation;
#[cfg(feature = "cuda-runtime")]
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
#[cfg(feature = "cuda-runtime")]
use crate::prover::secure_column::SecureColumnByCoords;

impl PolyOps for GpuBackend {
    type Twiddles = <SimdBackend as PolyOps>::Twiddles;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        #[cfg(feature = "cuda-runtime")]
        {
            // Reuse the batched path for a single column.
            return Self::interpolate_columns(vec![eval], twiddles)
                .into_iter()
                .next()
                .expect("single interpolation should return one polynomial");
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            interpolate_simd_fallback(eval, twiddles)
        }
    }

    fn interpolate_columns(
        columns: Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        #[cfg(feature = "cuda-runtime")]
        {
            if columns.is_empty() {
                return Vec::new();
            }

            let harden_inputs = gpu_poly_hardening_enabled().then(|| columns.clone());

            if is_cuda_available() {
                match interpolate_columns_gpu(&columns, twiddles) {
                    Ok(gpu_result) => {
                        if let Some(harden_columns) = harden_inputs {
                            let simd_result =
                                interpolate_columns_simd_fallback(harden_columns, twiddles);
                            assert_coeff_vectors_equal(
                                "interpolate_columns",
                                &gpu_result,
                                &simd_result,
                            );
                        }
                        return gpu_result;
                    }
                    Err(err) => {
                        if gpu_poly_strict_mode() {
                            panic!(
                                "GpuBackend::interpolate_columns failed in strict mode: {}",
                                err
                            );
                        }
                        tracing::warn!(
                            "GpuBackend::interpolate_columns GPU path failed ({}), falling back to SIMD",
                            err
                        );
                    }
                }
            } else if gpu_poly_strict_mode() {
                panic!("GpuBackend::interpolate_columns strict mode requires CUDA availability");
            }
        }

        interpolate_columns_simd_fallback(columns, twiddles)
    }

    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        #[cfg(feature = "cuda-runtime")]
        {
            if is_cuda_available() {
                match eval_at_point_gpu(poly, point) {
                    Ok(value) => {
                        if gpu_poly_hardening_enabled() {
                            let simd_poly = circle_coeffs_ref_to_simd(poly);
                            let simd_value = SimdBackend::eval_at_point(simd_poly, point);
                            assert_securefield_equal("eval_at_point", value, simd_value);
                        }
                        return value;
                    }
                    Err(err) => {
                        if gpu_poly_strict_mode() {
                            panic!("GpuBackend::eval_at_point failed in strict mode: {}", err);
                        }
                        tracing::warn!(
                            "GpuBackend::eval_at_point GPU path failed ({}), falling back to SIMD",
                            err
                        );
                    }
                }
            } else if gpu_poly_strict_mode() {
                panic!("GpuBackend::eval_at_point strict mode requires CUDA availability");
            }
        }

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
        #[cfg(feature = "cuda-runtime")]
        {
            if is_cuda_available() {
                return eval_at_point_by_folding_gpu(evals, point, twiddles);
            }
            if gpu_poly_strict_mode() {
                panic!(
                    "GpuBackend::eval_at_point_by_folding strict mode requires CUDA availability"
                );
            }
        }

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
        #[cfg(feature = "cuda-runtime")]
        {
            // Keep non-canonic domains on the proven SIMD path.
            if domain.is_canonic() && domain.log_size() >= poly.log_size() {
                let mut polys = Self::evaluate_polynomials(
                    vec![poly.clone()],
                    domain.log_size() - poly.log_size(),
                    twiddles,
                    false,
                );
                if let Some(single) = polys.pop() {
                    return single.evals;
                }
            }
        }

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
        if polynomials.is_empty() {
            return Vec::new();
        }

        let harden_snapshot = gpu_poly_hardening_enabled().then(|| polynomials.clone());

        if is_cuda_available() {
            match evaluate_polynomials_gpu(
                &polynomials,
                log_blowup_factor,
                twiddles,
                store_polynomials_coefficients,
            ) {
                Ok(gpu_result) => {
                    if let Some(snapshot) = harden_snapshot {
                        let simd_result = evaluate_polynomials_simd_fallback(
                            snapshot,
                            log_blowup_factor,
                            twiddles,
                            store_polynomials_coefficients,
                        );
                        assert_poly_eval_vectors_equal(
                            "evaluate_polynomials",
                            &gpu_result,
                            &simd_result,
                        );
                    }
                    return gpu_result;
                }
                Err(err) => {
                    if gpu_poly_strict_mode() {
                        panic!(
                            "GpuBackend::evaluate_polynomials failed in strict mode: {}",
                            err
                        );
                    }
                    tracing::warn!(
                        "GpuBackend::evaluate_polynomials GPU path failed ({}), falling back to SIMD",
                        err
                    );
                }
            }
        } else if gpu_poly_strict_mode() {
            panic!("GpuBackend::evaluate_polynomials strict mode requires CUDA availability");
        }

        evaluate_polynomials_simd_fallback(
            polynomials,
            log_blowup_factor,
            twiddles,
            store_polynomials_coefficients,
        )
    }
}

fn interpolate_columns_simd_fallback(
    columns: Vec<CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>>,
    twiddles: &TwiddleTree<GpuBackend>,
) -> Vec<CircleCoefficients<GpuBackend>> {
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let simd_columns: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = columns
        .into_iter()
        .map(|eval| {
            CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                eval.domain,
                eval.values,
            )
        })
        .collect();
    let results = SimdBackend::interpolate_columns(simd_columns, simd_twiddles);
    results
        .into_iter()
        .map(|r| CircleCoefficients::new(r.coeffs))
        .collect()
}

#[cfg(feature = "cuda-runtime")]
fn evaluate_polynomials_simd_fallback(
    polynomials: ColumnVec<CircleCoefficients<GpuBackend>>,
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<GpuBackend>,
    store_polynomials_coefficients: bool,
) -> Vec<Poly<GpuBackend>> {
    let simd_twiddles = twiddle_ref_to_simd(twiddles);
    let simd_polys: ColumnVec<CircleCoefficients<SimdBackend>> = polynomials
        .iter()
        .map(|p| CircleCoefficients::<SimdBackend>::new(p.coeffs.clone()))
        .collect();

    let simd_results =
        SimdBackend::evaluate_polynomials(simd_polys, log_blowup_factor, simd_twiddles, false);

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

fn interpolate_simd_fallback(
    eval: CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<GpuBackend>,
) -> CircleCoefficients<GpuBackend> {
    let simd_eval =
        CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(eval.domain, eval.values);
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

#[cfg(feature = "cuda-runtime")]
fn gpu_poly_strict_mode() -> bool {
    gpu_poly_flag_enabled("STWO_GPU_POLY_STRICT")
}

#[cfg(feature = "cuda-runtime")]
fn gpu_poly_hardening_enabled() -> bool {
    gpu_poly_flag_enabled("STWO_GPU_POLY_HARDEN")
}

#[cfg(feature = "cuda-runtime")]
fn gpu_poly_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

#[cfg(feature = "cuda-runtime")]
fn base_column_to_u32_vec(col: &crate::prover::backend::simd::column::BaseColumn) -> Vec<u32> {
    // Safety: BaseColumn stores PackedBaseField which is repr(transparent) over u32x16.
    let ptr = col.data.as_ptr() as *const u32;
    let total = col.data.len() * 16;
    let raw = unsafe { std::slice::from_raw_parts(ptr, total) };
    raw[..col.length].to_vec()
}

#[cfg(feature = "cuda-runtime")]
fn u32_vec_to_base_column(values: Vec<u32>) -> crate::prover::backend::simd::column::BaseColumn {
    values.into_iter().map(M31::from_u32_unchecked).collect()
}

#[cfg(feature = "cuda-runtime")]
fn column_cache_key(col: &crate::prover::backend::simd::column::BaseColumn) -> (usize, usize) {
    (col.data.as_ptr() as usize, col.data.len())
}

#[cfg(feature = "cuda-runtime")]
fn normalized_m31_eq(a: u32, b: u32) -> bool {
    M31::reduce(a as u64).0 == M31::reduce(b as u64).0
}

#[cfg(feature = "cuda-runtime")]
fn assert_coeff_vectors_equal(
    op: &str,
    gpu: &[CircleCoefficients<GpuBackend>],
    simd: &[CircleCoefficients<GpuBackend>],
) {
    assert_eq!(
        gpu.len(),
        simd.len(),
        "{} hardening mismatch: result length differs (gpu={}, simd={})",
        op,
        gpu.len(),
        simd.len()
    );
    for (poly_idx, (lhs, rhs)) in gpu.iter().zip(simd.iter()).enumerate() {
        assert_eq!(
            lhs.log_size(),
            rhs.log_size(),
            "{} hardening mismatch at polynomial {}: log_size differs",
            op,
            poly_idx
        );
        let lhs_vals = lhs.coeffs.as_slice();
        let rhs_vals = rhs.coeffs.as_slice();
        assert_eq!(
            lhs_vals.len(),
            rhs_vals.len(),
            "{} hardening mismatch at polynomial {}: length differs",
            op,
            poly_idx
        );
        for (i, (a, b)) in lhs_vals.iter().zip(rhs_vals.iter()).enumerate() {
            if !normalized_m31_eq(a.0, b.0) {
                panic!(
                    "{} hardening mismatch at polynomial {}, coeff {}: gpu={} simd={} (mod P {} vs {})",
                    op,
                    poly_idx,
                    i,
                    a.0,
                    b.0,
                    a.0 % P,
                    b.0 % P
                );
            }
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn assert_poly_eval_vectors_equal(op: &str, gpu: &[Poly<GpuBackend>], simd: &[Poly<GpuBackend>]) {
    assert_eq!(
        gpu.len(),
        simd.len(),
        "{} hardening mismatch: result length differs (gpu={}, simd={})",
        op,
        gpu.len(),
        simd.len()
    );
    for (poly_idx, (lhs, rhs)) in gpu.iter().zip(simd.iter()).enumerate() {
        assert_eq!(
            lhs.evals.domain, rhs.evals.domain,
            "{} hardening mismatch at polynomial {}: domains differ",
            op, poly_idx
        );
        let lhs_vals = lhs.evals.values.as_slice();
        let rhs_vals = rhs.evals.values.as_slice();
        assert_eq!(
            lhs_vals.len(),
            rhs_vals.len(),
            "{} hardening mismatch at polynomial {}: eval lengths differ",
            op,
            poly_idx
        );
        for (i, (a, b)) in lhs_vals.iter().zip(rhs_vals.iter()).enumerate() {
            if !normalized_m31_eq(a.0, b.0) {
                panic!(
                    "{} hardening mismatch at polynomial {}, eval {}: gpu={} simd={} (mod P {} vs {})",
                    op,
                    poly_idx,
                    i,
                    a.0,
                    b.0,
                    a.0 % P,
                    b.0 % P
                );
            }
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn assert_securefield_equal(op: &str, gpu: SecureField, simd: SecureField) {
    let gpu_coords = gpu.to_m31_array();
    let simd_coords = simd.to_m31_array();
    for (idx, (a, b)) in gpu_coords.iter().zip(simd_coords.iter()).enumerate() {
        if !normalized_m31_eq(a.0, b.0) {
            panic!(
                "{} hardening mismatch at coordinate {}: gpu={} simd={} (mod P {} vs {})",
                op,
                idx,
                a.0,
                b.0,
                a.0 % P,
                b.0 % P
            );
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn eval_at_point_gpu(
    poly: &CircleCoefficients<GpuBackend>,
    point: CirclePoint<SecureField>,
) -> Result<SecureField, CudaFftError> {
    if poly.log_size() <= 8 {
        let simd_poly = circle_coeffs_ref_to_simd(poly);
        return Ok(SimdBackend::eval_at_point(simd_poly, point));
    }

    let coeffs = base_column_to_u32_vec(&poly.coeffs);
    let twiddles_aos = generate_eval_twiddles_aos(point, poly.log_size());
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    let out = executor.execute_eval_point_from_coeffs(&coeffs, &twiddles_aos)?;

    Ok(SecureField::from_m31(
        M31::from_u32_unchecked(out[0]),
        M31::from_u32_unchecked(out[1]),
        M31::from_u32_unchecked(out[2]),
        M31::from_u32_unchecked(out[3]),
    ))
}

#[cfg(feature = "cuda-runtime")]
fn eval_at_point_by_folding_gpu(
    evals: &CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>,
    point: CirclePoint<SecureField>,
    twiddles: &TwiddleTree<GpuBackend>,
) -> SecureField {
    let log_size = evals.domain.log_size();
    let mut folding_alphas = get_folding_alphas(point, log_size as usize);
    let mut layer_evaluation =
        LineEvaluation::<GpuBackend>::new_zero(LineDomain::new(Coset::half_odds(log_size - 1)));

    let secure_evals = SecureEvaluation::<GpuBackend, BitReversedOrder>::new(
        evals.domain,
        SecureColumnByCoords::from_base_field_col(&evals.values),
    );

    GpuBackend::fold_circle_into_line(
        &mut layer_evaluation,
        &secure_evals,
        folding_alphas.pop().expect("missing first folding alpha"),
        twiddles,
    );

    while layer_evaluation.len() > 1 {
        layer_evaluation = GpuBackend::fold_line(
            &layer_evaluation,
            folding_alphas.pop().expect("missing folding alpha"),
            twiddles,
        );
    }
    GpuBackend::resolve_pending_line_evaluation(&mut layer_evaluation);

    layer_evaluation.values.at(0) / SecureField::from(2_u32.pow(log_size))
}

#[cfg(feature = "cuda-runtime")]
fn generate_eval_twiddles_aos(point: CirclePoint<SecureField>, log_size: u32) -> Vec<u32> {
    let mappings = generate_evaluation_mappings(point, log_size);
    let steps = twiddle_steps(&mappings);
    let n = 1usize << log_size;

    let mut twiddles = Vec::with_capacity(n * 4);
    let mut twiddle = SecureField::one();
    for i in 0..n {
        let coords = twiddle.to_m31_array();
        twiddles.push(coords[0].0);
        twiddles.push(coords[1].0);
        twiddles.push(coords[2].0);
        twiddles.push(coords[3].0);
        twiddle *= steps[i.trailing_ones() as usize];
    }
    twiddles
}

#[cfg(feature = "cuda-runtime")]
fn generate_evaluation_mappings(
    point: CirclePoint<SecureField>,
    log_size: u32,
) -> Vec<SecureField> {
    let mut mappings = vec![point.y, point.x];
    let mut x = point.x;
    for _ in 2..log_size {
        x = CirclePoint::double_x(x);
        mappings.push(x);
    }

    // Match SIMD transposed FFT-basis ordering for large polynomials.
    if log_size > CACHED_FFT_LOG_SIZE {
        mappings.reverse();
        let n = mappings.len();
        let n0 = (n - LOG_N_LANES as usize) / 2;
        let n1 = (n - LOG_N_LANES as usize).div_ceil(2);
        let (ab, c) = mappings.split_at_mut(n1);
        let (a, _b) = ab.split_at_mut(n0);
        a.swap_with_slice(&mut c[0..n0]);
        mappings.reverse();
    }

    mappings
}

#[cfg(feature = "cuda-runtime")]
fn twiddle_steps(mappings: &[SecureField]) -> Vec<SecureField> {
    let mut denominators = Vec::with_capacity(mappings.len());
    denominators.push(mappings[0]);
    for i in 1..mappings.len() {
        denominators.push(denominators[i - 1] * mappings[i]);
    }

    let denom_inverses = SecureField::batch_inverse(&denominators);

    let mut steps = Vec::with_capacity(mappings.len() + 1);
    steps.push(mappings[0]);
    for (m, d) in mappings.iter().skip(1).zip(denom_inverses.iter()) {
        steps.push(*m * *d);
    }
    steps.push(SecureField::one());
    steps
}

#[cfg(feature = "cuda-runtime")]
fn build_gpu_twiddles_from_line_layers(line_layers: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let circle_twiddles: Vec<u32> = if !line_layers.is_empty() && !line_layers[0].is_empty() {
        let first_line = &line_layers[0];
        first_line
            .chunks_exact(2)
            .flat_map(|chunk| {
                let x = BaseField::from_u32_unchecked(chunk[0] / 2);
                let y = BaseField::from_u32_unchecked(chunk[1] / 2);
                [y.0 * 2, (-y).0 * 2, (-x).0 * 2, x.0 * 2]
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut result = Vec::with_capacity(line_layers.len() + 1);
    result.push(circle_twiddles);
    result.extend(line_layers.iter().cloned());
    result
}

#[cfg(feature = "cuda-runtime")]
fn interpolate_columns_gpu(
    columns: &[CircleEvaluation<GpuBackend, BaseField, BitReversedOrder>],
    twiddles: &TwiddleTree<GpuBackend>,
) -> Result<Vec<CircleCoefficients<GpuBackend>>, CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;

    if columns.is_empty() {
        return Ok(Vec::new());
    }

    let first_domain = columns[0].domain;
    let same_domain = columns.iter().all(|c| c.domain == first_domain);

    if same_domain {
        let log_size = first_domain.log_size();
        let tws = extract_itwiddles_for_gpu(twiddles, first_domain);
        let denorm_factor = BaseField::from(first_domain.size()).inverse().0;

        let column_data: Vec<Vec<u32>> = columns
            .iter()
            .map(|eval| base_column_to_u32_vec(&eval.values))
            .collect();

        let (cpu_results, gpu_slices) =
            executor.execute_batch_ifft_to_gpu(&column_data, &tws, log_size, denorm_factor)?;

        let mut out = Vec::with_capacity(cpu_results.len());
        for (cpu_data, d_col) in cpu_results.into_iter().zip(gpu_slices.into_iter()) {
            let coeff_col = u32_vec_to_base_column(cpu_data);
            let key = column_cache_key(&coeff_col);
            cache_column_gpu(key.0, key.1, d_col);
            out.push(CircleCoefficients::new(coeff_col));
        }
        return Ok(out);
    }

    // Mixed domains: still run on GPU, one column at a time.
    let mut out = Vec::with_capacity(columns.len());
    for eval in columns {
        let log_size = eval.domain.log_size();
        let tws = extract_itwiddles_for_gpu(twiddles, eval.domain);
        let denorm_factor = BaseField::from(eval.domain.size()).inverse().0;
        let raw = base_column_to_u32_vec(&eval.values);

        let mut d_col = executor.execute_ifft_to_gpu(&raw, &tws, log_size)?;
        executor.execute_denormalize_on_device(&mut d_col, denorm_factor, 1u32 << log_size)?;
        executor
            .device
            .synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;

        let cpu_data = executor
            .device
            .dtoh_sync_copy(&d_col)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtoh: {:?}", e)))?;
        let coeff_col = u32_vec_to_base_column(cpu_data);
        let key = column_cache_key(&coeff_col);
        cache_column_gpu(key.0, key.1, d_col);
        out.push(CircleCoefficients::new(coeff_col));
    }
    Ok(out)
}

#[cfg(feature = "cuda-runtime")]
fn evaluate_polynomials_gpu(
    polynomials: &[CircleCoefficients<GpuBackend>],
    log_blowup_factor: u32,
    twiddles: &TwiddleTree<GpuBackend>,
    store_polynomials_coefficients: bool,
) -> Result<Vec<Poly<GpuBackend>>, CudaFftError> {
    use std::collections::BTreeMap;

    let executor = get_cuda_executor().map_err(|e| e.clone())?;

    // Group by coefficient log_size so we can batch FFT per group.
    let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (idx, poly) in polynomials.iter().enumerate() {
        groups.entry(poly.log_size()).or_default().push(idx);
    }

    let mut out: Vec<Option<Poly<GpuBackend>>> = (0..polynomials.len()).map(|_| None).collect();

    for (fft_log_size, indices) in groups {
        if fft_log_size < 2 {
            return Err(CudaFftError::InvalidSize(format!(
                "GPU FFT requires log_size >= 2, got {}",
                fft_log_size
            )));
        }

        let eval_log_size = fft_log_size + log_blowup_factor;
        let domain = CanonicCoset::new(eval_log_size).circle_domain();
        let coeff_size = 1usize << fft_log_size;
        let eval_size = 1usize << eval_log_size;
        let n_subdomains = 1usize << (eval_log_size - fft_log_size);

        let mut d_coeffs = Vec::with_capacity(indices.len());
        for &poly_idx in &indices {
            let coeff_col = &polynomials[poly_idx].coeffs;
            let key = column_cache_key(coeff_col);
            if let Some(d_cached) = take_cached_column_gpu(key.0, key.1) {
                d_coeffs.push(d_cached);
            } else {
                let raw = base_column_to_u32_vec(coeff_col);
                let d_uploaded = executor
                    .device
                    .htod_sync_copy(&raw)
                    .map_err(|e| CudaFftError::MemoryAllocation(format!("htod: {:?}", e)))?;
                d_coeffs.push(d_uploaded);
            }
        }

        let mut d_work_cols = Vec::with_capacity(indices.len());
        let mut d_eval_cols = Vec::with_capacity(indices.len());
        for _ in 0..indices.len() {
            let work = unsafe { executor.device.alloc::<u32>(coeff_size) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("alloc work: {:?}", e)))?;
            let eval = unsafe { executor.device.alloc::<u32>(eval_size) }
                .map_err(|e| CudaFftError::MemoryAllocation(format!("alloc eval: {:?}", e)))?;
            d_work_cols.push(work);
            d_eval_cols.push(eval);
        }

        let domain_line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);
        let full_twiddles = (n_subdomains == 1).then(|| extract_twiddles_for_gpu(twiddles, domain));

        let mut eval_buffers: Vec<Vec<u32>> =
            (0..indices.len()).map(|_| vec![0u32; eval_size]).collect();

        for sub_idx in 0..n_subdomains {
            for (d_coeff, d_work) in d_coeffs.iter().zip(d_work_cols.iter_mut()) {
                executor.device.dtod_copy(d_coeff, d_work).map_err(|e| {
                    CudaFftError::MemoryTransfer(format!("dtod coeff->work: {:?}", e))
                })?;
            }

            let sub_twiddles = if let Some(tw) = &full_twiddles {
                tw.clone()
            } else {
                let mut line_layers = Vec::with_capacity((fft_log_size - 1) as usize);
                for layer_i in 0..(fft_log_size - 1) {
                    let shift = (fft_log_size - 2 - layer_i) as usize;
                    let start = sub_idx << shift;
                    let end = (sub_idx + 1) << shift;
                    line_layers.push(domain_line_twiddles[layer_i as usize][start..end].to_vec());
                }
                build_gpu_twiddles_from_line_layers(&line_layers)
            };

            let chunk_results =
                executor.execute_batch_fft_on_gpu(&mut d_work_cols, &sub_twiddles, fft_log_size)?;

            let offset = sub_idx * coeff_size;
            for (local_idx, chunk) in chunk_results.into_iter().enumerate() {
                eval_buffers[local_idx][offset..offset + coeff_size].copy_from_slice(&chunk);

                let mut dst = d_eval_cols[local_idx].slice_mut(offset..offset + coeff_size);
                executor
                    .device
                    .dtod_copy(&d_work_cols[local_idx], &mut dst)
                    .map_err(|e| {
                        CudaFftError::MemoryTransfer(format!(
                            "dtod work->eval chunk {}: {:?}",
                            local_idx, e
                        ))
                    })?;
            }
        }

        for ((poly_idx, eval_buf), d_eval_col) in indices
            .iter()
            .copied()
            .zip(eval_buffers.into_iter())
            .zip(d_eval_cols.into_iter())
        {
            let eval_col = u32_vec_to_base_column(eval_buf);
            let evals = CircleEvaluation::new(domain, eval_col);
            let key = column_cache_key(&evals.values);
            cache_column_gpu(key.0, key.1, d_eval_col);

            let coeffs = store_polynomials_coefficients.then(|| polynomials[poly_idx].clone());
            out[poly_idx] = Some(Poly::new(coeffs, evals));
        }
    }

    Ok(out
        .into_iter()
        .map(|p| p.expect("all grouped polynomials must be filled"))
        .collect())
}
