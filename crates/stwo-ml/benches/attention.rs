use criterion::{criterion_group, criterion_main, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo_ml::components::attention::{prove_attention_head, verify_attention_head};
use stwo_ml::components::matmul::M31Matrix;

fn bench_attention_sumcheck(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_sumcheck");

    // 4x4 attention head (seq_len=4, d_k=4, d_v=4)
    let q = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
    let k = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
    let v = M31Matrix::from_data(4, 4, (33..=48).map(M31::from).collect()).unwrap();
    let kt = k.transpose();
    let scores = M31Matrix::multiply(&q, &kt).unwrap();
    let weights = scores.clone();
    let output = M31Matrix::multiply(&weights, &v).unwrap();

    group.bench_function("prove_4x4_head", |bench| {
        bench.iter(|| {
            let mut channel = Blake2sChannel::default();
            prove_attention_head(&q, &k, &v, &scores, &weights, &output, &mut channel).unwrap()
        })
    });

    let mut prover_channel = Blake2sChannel::default();
    let proof =
        prove_attention_head(&q, &k, &v, &scores, &weights, &output, &mut prover_channel)
            .unwrap();

    group.bench_function("verify_4x4_head", |bench| {
        bench.iter(|| {
            let mut channel = Blake2sChannel::default();
            verify_attention_head(
                &q, &k, &v, &scores, &weights, &output, &proof, &mut channel,
            )
            .unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_attention_sumcheck);
criterion_main!(benches);
