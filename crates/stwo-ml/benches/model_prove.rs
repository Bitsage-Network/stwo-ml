use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::fields::m31::M31;
use stwo_ml::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use stwo_ml::compiler::prove::{prove_model, verify_model};
use stwo_ml::components::activation::ActivationType;
use stwo_ml::components::matmul::M31Matrix;

fn bench_model_matmul_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_matmul_relu");
    group.sample_size(10);

    for size in [4, 8] {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input {
                rows: size,
                cols: size,
            },
            GraphOp::MatMul {
                weight_rows: size,
                weight_cols: size,
            },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
        ])
        .unwrap();

        let input = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32)
                .map(|i| M31::from(i % 4))
                .collect(),
        )
        .unwrap();

        let weight = M31Matrix::from_data(
            size,
            size,
            (0..(size * size) as u32)
                .map(|i| M31::from((i + 1) % 3))
                .collect(),
        )
        .unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, weight);

        group.bench_function(BenchmarkId::new("prove", format!("{size}x{size}")), |bench| {
            bench.iter(|| prove_model(&graph, &input, &weights).unwrap())
        });

        let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();

        group.bench_function(
            BenchmarkId::new("verify", format!("{size}x{size}")),
            |bench| {
                bench.iter(|| verify_model(&proof, &execution, &graph, &weights).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_model_two_layer_mlp(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_two_layer_mlp");
    group.sample_size(10);

    let size = 4;
    let graph = ComputationGraph::sequential(vec![
        GraphOp::Input {
            rows: size,
            cols: size,
        },
        GraphOp::MatMul {
            weight_rows: size,
            weight_cols: size,
        },
        GraphOp::Activation {
            activation: ActivationType::ReLU,
            log_table_size: 8,
        },
        GraphOp::MatMul {
            weight_rows: size,
            weight_cols: size,
        },
    ])
    .unwrap();

    let input = M31Matrix::from_data(
        size,
        size,
        (0..(size * size) as u32)
            .map(|i| M31::from(i % 3))
            .collect(),
    )
    .unwrap();

    let w1 = M31Matrix::from_data(
        size,
        size,
        (0..(size * size) as u32)
            .map(|i| M31::from((i + 1) % 2))
            .collect(),
    )
    .unwrap();

    let w2 = M31Matrix::from_data(
        size,
        size,
        (0..(size * size) as u32)
            .map(|i| M31::from(i % 2))
            .collect(),
    )
    .unwrap();

    let mut weights = GraphWeights::new();
    weights.matmul_weights.insert(1, w1);
    weights.matmul_weights.insert(3, w2);

    group.bench_function("prove_4x4", |bench| {
        bench.iter(|| prove_model(&graph, &input, &weights).unwrap())
    });

    let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();

    group.bench_function("verify_4x4", |bench| {
        bench.iter(|| verify_model(&proof, &execution, &graph, &weights).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_model_matmul_relu, bench_model_two_layer_mlp);
criterion_main!(benches);
