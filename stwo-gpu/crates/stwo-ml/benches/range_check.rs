use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo_ml::gadgets::range_check::{prove_range_check, verify_range_check};

fn bench_range_check_prove_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_check_prove_verify");

    for log_range in [4, 8] {
        let max_val = 1u32 << log_range;
        let inputs: Vec<M31> = (0..64).map(|i| M31::from(i % max_val)).collect();

        group.bench_function(BenchmarkId::new("prove", log_range), |bench| {
            bench.iter(|| {
                let config = PcsConfig::default();
                let mut channel = Blake2sChannel::default();
                prove_range_check(&inputs, log_range, config, &mut channel).unwrap()
            })
        });

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) =
            prove_range_check(&inputs, log_range, config, &mut prover_channel).unwrap();

        group.bench_function(BenchmarkId::new("verify", log_range), |bench| {
            bench.iter(|| {
                let mut channel = Blake2sChannel::default();
                verify_range_check(&component, &proof, &mut channel).unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_range_check_prove_verify);
criterion_main!(benches);
