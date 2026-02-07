use criterion::{criterion_group, criterion_main, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo_ml::components::activation::{prove_activation, verify_activation};
use stwo_ml::gadgets::lookup_table::PrecomputedTable;

fn bench_relu_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_relu");

    let log_size = 4u32; // 16-entry table (small for benchmark speed)
    let table = PrecomputedTable::relu(log_size);
    let inputs: Vec<M31> = (0..16u32).map(|i| M31::from(i % 16)).collect();

    group.bench_function("prove_relu_16", |bench| {
        bench.iter(|| {
            let config = PcsConfig::default();
            let mut channel = Blake2sChannel::default();
            prove_activation(&inputs, &table, config, &mut channel).unwrap()
        })
    });

    let config = PcsConfig::default();
    let mut prover_channel = Blake2sChannel::default();
    let (component, proof) =
        prove_activation(&inputs, &table, config, &mut prover_channel).unwrap();

    group.bench_function("verify_relu_16", |bench| {
        bench.iter(|| {
            let mut channel = Blake2sChannel::default();
            verify_activation(&component, &proof, &mut channel).unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_relu_activation);
criterion_main!(benches);
