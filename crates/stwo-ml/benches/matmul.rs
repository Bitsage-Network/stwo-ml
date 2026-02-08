use criterion::{criterion_group, criterion_main, Criterion};

fn bench_matmul_sumcheck(_c: &mut Criterion) {
    // TODO: Benchmark sumcheck-based matmul vs naive trace
}

criterion_group!(benches, bench_matmul_sumcheck);
criterion_main!(benches);
