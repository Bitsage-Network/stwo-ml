use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo_ml::components::attention::MultiHeadAttentionConfig;

fn bench_attention_cost_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_cost_model");

    let configs = [
        ("tiny_2h_64d_32s", 2, 64, 32),
        ("small_4h_256d_128s", 4, 256, 128),
        ("bert_12h_768d_512s", 12, 768, 512),
        ("gpt2_12h_768d_1024s", 12, 768, 1024),
    ];

    for (name, heads, d_model, seq_len) in configs {
        group.bench_with_input(
            BenchmarkId::new("sumcheck_rows", name),
            &name,
            |bench, _| {
                bench.iter(|| {
                    let config = MultiHeadAttentionConfig::new(heads, d_model, seq_len);
                    config.sumcheck_trace_rows()
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("speedup", name), &name, |bench, _| {
            bench.iter(|| {
                let config = MultiHeadAttentionConfig::new(heads, d_model, seq_len);
                config.speedup()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_attention_cost_model);
criterion_main!(benches);
