use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stwo::core::fields::m31::M31;
use stwo_ml::aggregation::{
    prove_model_pure_gkr_decode_step, prove_model_pure_gkr_prefill,
};
use stwo_ml::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use stwo_ml::components::attention::{ModelKVCache, MultiHeadAttentionConfig};
use stwo_ml::components::matmul::M31Matrix;

const D_MODEL: usize = 8;
const NUM_HEADS: usize = 2;
const PREFILL_LEN: usize = 4;

fn make_matrix(rows: usize, cols: usize) -> M31Matrix {
    let mut m = M31Matrix::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            m.set(i, j, M31::from((i * cols + j + 1) as u32 % 100 + 1));
        }
    }
    m
}

fn build_test_graph(
    seq_len: usize,
) -> (ComputationGraph, GraphWeights) {
    let config = MultiHeadAttentionConfig {
        d_model: D_MODEL,
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_HEADS,
        seq_len,
        causal: true,
    };
    let mut graph = ComputationGraph::new((seq_len, D_MODEL));
    let mm_id = graph.add_node(
        GraphOp::MatMul {
            dims: (seq_len, D_MODEL, D_MODEL),
        },
        vec![],
        (seq_len, D_MODEL),
    );
    let attn_id = graph.add_node(
        GraphOp::Attention { config },
        vec![mm_id],
        (seq_len, D_MODEL),
    );

    let mut weights = GraphWeights::new();
    weights.add_weight(mm_id, make_matrix(D_MODEL, D_MODEL));
    let wq = make_matrix(D_MODEL, D_MODEL);
    let wk = make_matrix(D_MODEL, D_MODEL);
    let wv = make_matrix(D_MODEL, D_MODEL);
    let wo = make_matrix(D_MODEL, D_MODEL);
    weights.add_named_weight(attn_id, "w_q", wq.clone());
    weights.add_named_weight(attn_id, "w_k", wk.clone());
    weights.add_named_weight(attn_id, "w_v", wv.clone());
    weights.add_named_weight(attn_id, "w_o", wo.clone());
    weights.add_weight(attn_id + 1, wq);
    weights.add_weight(attn_id + 2, wk);
    weights.add_weight(attn_id + 3, wv);
    weights.add_weight(attn_id + 4, wo);

    (graph, weights)
}

fn prefill(graph: &ComputationGraph, weights: &GraphWeights) -> ModelKVCache {
    let input = make_matrix(PREFILL_LEN, D_MODEL);
    let mut kvc = ModelKVCache::new();
    prove_model_pure_gkr_prefill(graph, &input, weights, &mut kvc)
        .expect("prefill should succeed");
    kvc
}

fn bench_decode_step_base(c: &mut Criterion) {
    let (graph, weights) = build_test_graph(PREFILL_LEN);
    let token = make_matrix(1, D_MODEL);

    c.bench_function("decode_step_base", |b| {
        b.iter_batched(
            || prefill(&graph, &weights),
            |mut kvc| {
                prove_model_pure_gkr_decode_step(
                    &graph, &token, &weights, &mut kvc, None,
                )
                .unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_decode_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_scaling");

    for cache_len in [4, 8, 16] {
        let (graph, weights) = build_test_graph(cache_len);
        let token = make_matrix(1, D_MODEL);

        group.bench_with_input(
            BenchmarkId::new("step", cache_len),
            &cache_len,
            |b, _| {
                b.iter_batched(
                    || {
                        // Prefill to the target cache length
                        let input = make_matrix(cache_len, D_MODEL);
                        let mut kvc = ModelKVCache::new();
                        prove_model_pure_gkr_prefill(&graph, &input, &weights, &mut kvc)
                            .expect("prefill should succeed");
                        kvc
                    },
                    |mut kvc| {
                        prove_model_pure_gkr_decode_step(
                            &graph, &token, &weights, &mut kvc, None,
                        )
                        .unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decode_step_base, bench_decode_scaling);
criterion_main!(benches);
