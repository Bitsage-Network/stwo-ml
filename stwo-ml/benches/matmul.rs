use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::fields::m31::M31;
use stwo_ml::components::matmul::{
    matmul_m31, prove_matmul_sumcheck, prove_matmul_sumcheck_onchain, verify_matmul_sumcheck,
    verify_matmul_sumcheck_onchain, M31Matrix,
};
use stwo_ml::crypto::mle_opening::commit_mle;

fn make_matrix(rows: usize, cols: usize) -> M31Matrix {
    let mut m = M31Matrix::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m.set(i, j, M31::from(((i * cols + j) % 251 + 1) as u32));
        }
    }
    m
}

fn bench_matmul_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_sumcheck_prove");
    for size in [2, 4, 8, 16] {
        let a = make_matrix(size, size);
        let b = make_matrix(size, size);
        let result = matmul_m31(&a, &b);

        group.bench_with_input(
            BenchmarkId::new("prove", format!("{size}x{size}")),
            &size,
            |bench, _| {
                bench.iter(|| {
                    prove_matmul_sumcheck(&a, &b, &result).unwrap();
                });
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("matmul_sumcheck_verify");
    for size in [2, 4, 8, 16] {
        let a = make_matrix(size, size);
        let b = make_matrix(size, size);
        let result = matmul_m31(&a, &b);
        let proof = prove_matmul_sumcheck(&a, &b, &result).unwrap();

        group.bench_with_input(
            BenchmarkId::new("verify", format!("{size}x{size}")),
            &size,
            |bench, _| {
                bench.iter(|| {
                    verify_matmul_sumcheck(&proof, &a, &b, &result).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_matmul_onchain(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_onchain");
    for size in [2, 4, 8] {
        let a = make_matrix(size, size);
        let b = make_matrix(size, size);
        let result = matmul_m31(&a, &b);

        group.bench_with_input(
            BenchmarkId::new("prove", format!("{size}x{size}")),
            &size,
            |bench, _| {
                bench.iter(|| {
                    prove_matmul_sumcheck_onchain(&a, &b, &result).unwrap();
                });
            },
        );

        let proof = prove_matmul_sumcheck_onchain(&a, &b, &result).unwrap();
        group.bench_with_input(
            BenchmarkId::new("verify", format!("{size}x{size}")),
            &size,
            |bench, _| {
                bench.iter(|| {
                    verify_matmul_sumcheck_onchain(&proof).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_commit_mle(c: &mut Criterion) {
    use stwo::core::fields::qm31::SecureField;

    let mut group = c.benchmark_group("commit_mle");
    for log_size in [2, 4, 6, 8] {
        let size = 1usize << log_size;
        let evals: Vec<SecureField> = (0..size)
            .map(|i| SecureField::from(M31::from((i + 1) as u32)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("commit", format!("2^{log_size}")),
            &log_size,
            |bench, _| {
                bench.iter(|| {
                    commit_mle(&evals);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_sumcheck,
    bench_matmul_onchain,
    bench_commit_mle
);
criterion_main!(benches);
