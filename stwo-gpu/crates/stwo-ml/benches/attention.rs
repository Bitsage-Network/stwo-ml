use criterion::{criterion_group, criterion_main, Criterion};

fn bench_attention_sumcheck(_c: &mut Criterion) {
    // TODO: Benchmark composed attention (QK^T + softmax + attn×V)
}

criterion_group!(benches, bench_attention_sumcheck);
criterion_main!(benches);
