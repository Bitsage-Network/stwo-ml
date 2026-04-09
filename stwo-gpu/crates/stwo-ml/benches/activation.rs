use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo_ml::components::activation::{prove_activation, verify_activation, ActivationType};
use stwo_ml::gadgets::lookup_table::PrecomputedTable;

fn bench_activation_prove_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_prove_verify");

    let activations = [
        ("relu", ActivationType::ReLU),
        ("gelu", ActivationType::GELU),
        ("sigmoid", ActivationType::Sigmoid),
        ("softmax_exp", ActivationType::Softmax),
    ];

    let log_size = 4u32; // 16-entry table
    let inputs: Vec<M31> = (0..16u32).map(|i| M31::from(i % 16)).collect();

    for (name, act_type) in &activations {
        let table = act_type.build_table(log_size);

        group.bench_function(BenchmarkId::new("prove", *name), |bench| {
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

        group.bench_function(BenchmarkId::new("verify", *name), |bench| {
            bench.iter(|| {
                let mut channel = Blake2sChannel::default();
                verify_activation(&component, &proof, &mut channel).unwrap()
            })
        });
    }

    group.finish();
}

fn bench_activation_table_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_table_build");

    for log_size in [4, 8, 12, 16] {
        group.bench_function(BenchmarkId::new("relu", log_size), |bench| {
            bench.iter(|| PrecomputedTable::relu(log_size))
        });

        group.bench_function(BenchmarkId::new("gelu", log_size), |bench| {
            bench.iter(|| PrecomputedTable::gelu(log_size))
        });

        group.bench_function(BenchmarkId::new("sigmoid", log_size), |bench| {
            bench.iter(|| PrecomputedTable::sigmoid(log_size))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_activation_prove_verify, bench_activation_table_build);
criterion_main!(benches);
