use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo_ml::components::layernorm::{
    compute_mean, prove_layernorm_stark, verify_layernorm_stark, LayerNormParams,
};

fn bench_layernorm_prove_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm_prove_verify");
    group.sample_size(10);

    for log_size in [4, 8] {
        let size = 1usize << log_size;
        let inputs: Vec<M31> = (0..size as u32).map(|i| M31::from(i % (size as u32))).collect();
        let mean = compute_mean(&inputs).unwrap();
        let inv_std = M31::from(1);
        let params = LayerNormParams::identity(size);

        group.bench_function(BenchmarkId::new("prove", log_size), |bench| {
            bench.iter(|| {
                let config = PcsConfig::default();
                let mut channel = Blake2sChannel::default();
                prove_layernorm_stark(
                    &inputs,
                    log_size as u32,
                    mean,
                    inv_std,
                    &params,
                    config,
                    &mut channel,
                )
                .unwrap()
            })
        });

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) = prove_layernorm_stark(
            &inputs,
            log_size as u32,
            mean,
            inv_std,
            &params,
            config,
            &mut prover_channel,
        )
        .unwrap();

        group.bench_function(BenchmarkId::new("verify", log_size), |bench| {
            bench.iter(|| {
                let mut channel = Blake2sChannel::default();
                verify_layernorm_stark(&component, &proof, &mut channel).unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_layernorm_prove_verify);
criterion_main!(benches);
