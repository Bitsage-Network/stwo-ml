use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo_ml::components::matmul::{prove_matmul, verify_matmul, M31Matrix};

fn bench_matmul_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_sumcheck");

    for size in [2, 4, 8] {
        let a = M31Matrix::from_data(
            size,
            size,
            (1..=(size * size) as u32).map(M31::from).collect(),
        )
        .unwrap();
        let b = M31Matrix::from_data(
            size,
            size,
            ((size * size + 1) as u32..=(2 * size * size) as u32)
                .map(M31::from)
                .collect(),
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

criterion_group!(benches, bench_matmul_sumcheck);
criterion_main!(benches);
