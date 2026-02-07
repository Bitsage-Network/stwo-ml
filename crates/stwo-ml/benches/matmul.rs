use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo_ml::components::matmul::{prove_matmul, verify_matmul, M31Matrix};
use stwo_ml::starknet::prove_matmul_for_starknet;

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

    for size in [8, 16, 32, 64, 128, 256] {
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

criterion_group!(benches, bench_matmul_sumcheck, bench_starknet_proof);
criterion_main!(benches);
