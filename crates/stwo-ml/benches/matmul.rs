use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo_ml::components::matmul::{prove_matmul, verify_matmul, M31Matrix};
use stwo_ml::starknet::{prove_matmul_for_starknet, prove_matmul_for_starknet_with_config, ProverConfig};

extern crate rayon;

fn bench_matmul_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_sumcheck");

    for size in [2, 4, 8, 16, 32] {
        let a = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from(i % 2147483647)).collect(),
        )
        .unwrap();
        let b = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from((i + 1) % 2147483647)).collect(),
        )
        .unwrap();
        let c_mat = M31Matrix::multiply(&a, &b).unwrap();

        group.bench_function(BenchmarkId::new("prove", format!("{size}x{size}")), |bench| {
            bench.iter(|| {
                let mut channel = Blake2sChannel::default();
                prove_matmul(&a, &b, &c_mat, &mut channel).unwrap()
            })
        });

        // Pre-generate proof for verify benchmark
        let mut prover_channel = Blake2sChannel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c_mat, &mut prover_channel).unwrap();

        group.bench_function(BenchmarkId::new("verify", format!("{size}x{size}")), |bench| {
            bench.iter(|| {
                let mut channel = Blake2sChannel::default();
                verify_matmul(&a, &b, &c_mat, &proof, &aux, &mut channel).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_starknet_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("starknet_proof");
    group.sample_size(10); // Fewer samples for larger sizes

    for size in [8, 16, 32, 64, 128, 256, 512] {
        let a = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from((i * 7 + 3) % 2147483647)).collect(),
        )
        .unwrap();
        let b = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from((i * 13 + 11) % 2147483647)).collect(),
        )
        .unwrap();
        let c_mat = M31Matrix::multiply(&a, &b).unwrap();

        group.bench_function(
            BenchmarkId::new("full_prove", format!("{size}x{size}")),
            |bench| bench.iter(|| prove_matmul_for_starknet(&a, &b, &c_mat).unwrap()),
        );
    }
    group.finish();
}

fn bench_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("seq_vs_par");
    group.sample_size(10);

    for size in [64, 128, 256, 512] {
        let a = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from((i * 7 + 3) % 2147483647)).collect(),
        )
        .unwrap();
        let b = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from((i * 13 + 11) % 2147483647)).collect(),
        )
        .unwrap();
        let c_mat = M31Matrix::multiply(&a, &b).unwrap();

        // Sequential (1 thread)
        group.bench_function(
            BenchmarkId::new("1_thread", format!("{size}x{size}")),
            |bench| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                bench.iter(|| {
                    pool.install(|| prove_matmul_for_starknet(&a, &b, &c_mat).unwrap())
                })
            },
        );

        // Parallel (all cores)
        group.bench_function(
            BenchmarkId::new("all_cores", format!("{size}x{size}")),
            |bench| bench.iter(|| prove_matmul_for_starknet(&a, &b, &c_mat).unwrap()),
        );
    }
    group.finish();
}

fn bench_production_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("production_mode");
    group.sample_size(10);

    let prod_config = ProverConfig::production();

    for size in [64, 128, 256] {
        let a = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from((i * 7 + 3) % 2147483647)).collect(),
        )
        .unwrap();
        let b = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32).map(|i| M31::from((i * 13 + 11) % 2147483647)).collect(),
        )
        .unwrap();
        let c_mat = M31Matrix::multiply(&a, &b).unwrap();

        group.bench_function(
            BenchmarkId::new("default", format!("{size}x{size}")),
            |bench| bench.iter(|| prove_matmul_for_starknet(&a, &b, &c_mat).unwrap()),
        );

        group.bench_function(
            BenchmarkId::new("production", format!("{size}x{size}")),
            |bench| {
                bench.iter(|| {
                    prove_matmul_for_starknet_with_config(&a, &b, &c_mat, &prod_config).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_sumcheck,
    bench_starknet_proof,
    bench_sequential_vs_parallel,
    bench_production_mode
);
criterion_main!(benches);
