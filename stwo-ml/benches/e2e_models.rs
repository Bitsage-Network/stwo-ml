//! End-to-end model proving benchmarks.
//!
//! Benchmarks different model architectures through the full proving pipeline:
//! forward pass → per-layer proof generation → verification.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::fields::m31::M31;

use stwo_ml::compiler::graph::GraphBuilder;
use stwo_ml::compiler::onnx::generate_weights_for_graph;
use stwo_ml::compiler::prove::{prove_model, verify_model_matmuls};
use stwo_ml::components::activation::ActivationType;
use stwo_ml::components::matmul::M31Matrix;

/// Build a test input matrix.
fn make_input(rows: usize, cols: usize) -> M31Matrix {
    let mut m = M31Matrix::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m.set(i, j, M31::from(((i * cols + j) % 9 + 1) as u32));
        }
    }
    m
}

/// Benchmark: MLP (MNIST-like)
fn bench_mlp_mnist(c: &mut Criterion) {
    let mut builder = GraphBuilder::new((1, 16));
    builder
        .linear(16)
        .activation(ActivationType::ReLU)
        .linear(8)
        .activation(ActivationType::ReLU)
        .linear(4);
    let graph = builder.build();
    let weights = generate_weights_for_graph(&graph, 42);
    let input = make_input(1, 16);

    c.bench_function("mlp_16_16_8_4", |b| {
        b.iter(|| {
            let (proofs, _) = prove_model(&graph, &input, &weights).unwrap();
            verify_model_matmuls(&proofs, &graph, &input, &weights).unwrap();
        })
    });
}

/// Benchmark: Transformer block (tiny)
fn bench_transformer_tiny(c: &mut Criterion) {
    use stwo_ml::compiler::onnx::{build_transformer_block, TransformerConfig};

    let config = TransformerConfig {
        d_model: 4,
        num_heads: 1,
        d_ff: 8,
        activation: ActivationType::GELU,
    };
    let model = build_transformer_block(&config, 42);
    let input = make_input(1, 4);

    c.bench_function("transformer_block_d4_h1", |b| {
        b.iter(|| {
            prove_model(&model.graph, &input, &model.weights).unwrap();
        })
    });
}

/// Benchmark: MatMul sizes (sumcheck)
fn bench_matmul_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_sumcheck");

    for size in [4, 8, 16, 32] {
        let mut builder = GraphBuilder::new((1, size));
        builder.linear(size);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 42);
        let input = make_input(1, size);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter(|| {
                    prove_model(&graph, &input, &weights).unwrap();
                })
            },
        );
    }
    group.finish();
}

/// Benchmark: Residual connection (Add node)
fn bench_residual_block(c: &mut Criterion) {
    let mut builder = GraphBuilder::new((1, 8));
    builder.linear(8);
    let branch = builder.fork();
    builder.activation(ActivationType::ReLU);
    builder.linear(8);
    builder.add_from(branch);

    let graph = builder.build();
    let weights = generate_weights_for_graph(&graph, 42);
    let input = make_input(1, 8);

    c.bench_function("residual_block_d8", |b| {
        b.iter(|| {
            prove_model(&graph, &input, &weights).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_mlp_mnist,
    bench_transformer_tiny,
    bench_matmul_sizes,
    bench_residual_block,
);
criterion_main!(benches);
