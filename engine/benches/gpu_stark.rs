//! GPU STARK Acceleration Benchmark
//!
//! Compares STWO's SimdBackend vs GpuBackend on core STARK operations:
//!   1. FFT/IFFT (polynomial evaluate/interpolate)
//!   2. FRI fold (line and circle-to-line)
//!   3. Merkle tree commitment
//!   4. Full STARK prove on a synthetic AIR
//!
//! Run:
//!   cargo bench --bench gpu_stark --features cuda-runtime
//!
//! Without GPU:
//!   cargo bench --bench gpu_stark
//!   (only SIMD baseline numbers)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;

/// Generate a random-looking M31 column of given size.
fn make_column<B: ColumnOps<BaseField>>(log_size: u32) -> Col<B, BaseField> {
    let size = 1usize << log_size;
    let mut col = B::Column::zeros(size);
    for i in 0..size {
        col.set(i, M31::from(((i as u64 * 7 + 13) % ((1u64 << 31) - 1)) as u32));
    }
    col
}

/// Benchmark FFT (evaluate): coefficients → evaluations
fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_evaluate");
    group.measurement_time(Duration::from_secs(10));

    for log_size in [16, 18, 20, 22, 24] {
        // SIMD baseline
        let coset = CanonicCoset::new(log_size);
        let twiddles = SimdBackend::precompute_twiddles(coset.coset());

        group.bench_with_input(
            BenchmarkId::new("simd", format!("2^{log_size}")),
            &log_size,
            |bench, &log_size| {
                let col = make_column::<SimdBackend>(log_size);
                let poly = stwo::prover::poly::circle::CircleCoefficients::<SimdBackend>::new(col);
                let domain = CanonicCoset::new(log_size).circle_domain();
                bench.iter(|| {
                    let _eval = SimdBackend::evaluate(&poly, domain, &twiddles);
                });
            },
        );

        // GPU benchmark (only when cuda-runtime is available)
        #[cfg(feature = "cuda-runtime")]
        {
            use stwo::prover::backend::gpu::GpuBackend;

            if GpuBackend::is_available() {
                let gpu_twiddles = GpuBackend::precompute_twiddles(coset.coset());

                group.bench_with_input(
                    BenchmarkId::new("gpu", format!("2^{log_size}")),
                    &log_size,
                    |bench, &log_size| {
                        let col = make_column::<GpuBackend>(log_size);
                        let poly =
                            stwo::prover::poly::circle::CircleCoefficients::<GpuBackend>::new(col);
                        let domain = CanonicCoset::new(log_size).circle_domain();
                        bench.iter(|| {
                            let _eval = GpuBackend::evaluate(&poly, &domain, &gpu_twiddles);
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

/// Benchmark IFFT (interpolate): evaluations → coefficients
fn bench_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_interpolate");
    group.measurement_time(Duration::from_secs(10));

    for log_size in [16, 18, 20, 22, 24] {
        let coset = CanonicCoset::new(log_size);
        let twiddles = SimdBackend::precompute_twiddles(coset.coset());

        group.bench_with_input(
            BenchmarkId::new("simd", format!("2^{log_size}")),
            &log_size,
            |bench, &log_size| {
                let col = make_column::<SimdBackend>(log_size);
                let domain = CanonicCoset::new(log_size).circle_domain();
                let eval = CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    domain, col,
                );
                bench.iter(|| {
                    let _poly = SimdBackend::interpolate(eval.clone(), &twiddles);
                });
            },
        );

        #[cfg(feature = "cuda-runtime")]
        {
            use stwo::prover::backend::gpu::GpuBackend;

            if GpuBackend::is_available() {
                let gpu_twiddles = GpuBackend::precompute_twiddles(coset.coset());

                group.bench_with_input(
                    BenchmarkId::new("gpu", format!("2^{log_size}")),
                    &log_size,
                    |bench, &log_size| {
                        let col = make_column::<GpuBackend>(log_size);
                        let domain = CanonicCoset::new(log_size).circle_domain();
                        let eval =
                            CircleEvaluation::<GpuBackend, BaseField, BitReversedOrder>::new(
                                domain, col,
                            );
                        bench.iter(|| {
                            let _poly = GpuBackend::interpolate(eval.clone(), &gpu_twiddles);
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

/// Benchmark our M31 matmul: SIMD vs GPU
fn bench_matmul_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("m31_matmul");
    group.measurement_time(Duration::from_secs(10));

    use stwo_ml::components::matmul::{matmul_m31, M31Matrix};

    for size in [64, 256, 1024, 4096] {
        let a = {
            let mut m = M31Matrix::new(size, size);
            for i in 0..size * size {
                m.data[i] = M31::from((i as u32 * 7 + 3) % 251);
            }
            m
        };
        let b = a.clone();

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{size}x{size}")),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _c = matmul_m31(&a, &b);
                });
            },
        );

        #[cfg(feature = "cuda-runtime")]
        {
            use stwo_ml::gpu_sumcheck::gpu_matmul_m31_full;

            group.bench_with_input(
                BenchmarkId::new("gpu", format!("{size}x{size}")),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        let _c = gpu_matmul_m31_full(&a, &b);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_fft, bench_ifft, bench_matmul_gpu);
criterion_main!(benches);
