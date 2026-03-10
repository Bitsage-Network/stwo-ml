//! Decode-step GKR proving benchmark.
//!
//! Measures per-step decode latency to inform the proving protocol
//! (prove every token vs batch every N).
//!
//! Run: `cargo test --features std decode_benchmark -- --ignored --nocapture`
//!
//! Env vars for scale:
//!   BENCH_D_MODEL    (default 128)
//!   BENCH_NUM_HEADS  (default 4)
//!   BENCH_D_FF       (default 512)
//!   BENCH_PREFILL_LEN (default 8)
//!   BENCH_DECODE_STEPS (default 10)

use std::time::Instant;

use stwo::core::fields::m31::M31;
use stwo_ml::aggregation::{IncrementalKVCommitment, prove_model_pure_gkr_decode_step};
use stwo_ml::components::attention::{
    attention_forward_cached, AttentionWeights, ModelKVCache,
};
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::compiler::graph::GraphBuilder;
use stwo_ml::compiler::onnx::generate_weights_for_graph;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn random_m31_matrix(rows: usize, cols: usize, seed: u64) -> M31Matrix {
    let mut data = Vec::with_capacity(rows * cols);
    let mut state = seed;
    for _ in 0..(rows * cols) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        data.push(M31::from((state >> 33) as u32 % 100));
    }
    M31Matrix { rows, cols, data }
}

#[test]
#[ignore]
fn decode_benchmark() {
    let d_model = env_usize("BENCH_D_MODEL", 128);
    let num_heads = env_usize("BENCH_NUM_HEADS", 4);
    let d_ff = env_usize("BENCH_D_FF", 512);
    let prefill_len = env_usize("BENCH_PREFILL_LEN", 8);
    let decode_steps = env_usize("BENCH_DECODE_STEPS", 10);

    eprintln!("=== Decode Benchmark ===");
    eprintln!(
        "  d_model={}, heads={}, d_ff={}, prefill={}, steps={}",
        d_model, num_heads, d_ff, prefill_len, decode_steps
    );

    // Build decode graph (seq_len=1 for single-token decode)
    let mut decode_builder = GraphBuilder::new((1, d_model));
    decode_builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let decode_graph = decode_builder.build();

    // Generate deterministic weights for matmul nodes
    let mut weights = generate_weights_for_graph(&decode_graph, 42);

    // Add attention weights for Attention nodes.
    // GKR prover expects positional weights at node_id+1..+4 AND named weights.
    let topo = decode_graph.topological_order();
    for &node_id in &topo {
        let node = &decode_graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config: _ } = &node.op {
            let w_q = random_m31_matrix(d_model, d_model, 200 + node.id as u64);
            let w_k = random_m31_matrix(d_model, d_model, 300 + node.id as u64);
            let w_v = random_m31_matrix(d_model, d_model, 400 + node.id as u64);
            let w_o = random_m31_matrix(d_model, d_model, 500 + node.id as u64);
            weights.add_named_weight(node.id, "w_q", w_q);
            weights.add_named_weight(node.id, "w_k", w_k);
            weights.add_named_weight(node.id, "w_v", w_v);
            weights.add_named_weight(node.id, "w_o", w_o);
        }
    }

    // Seed KV cache with prefill tokens
    let mut kv_cache = ModelKVCache::new();
    let prefill_input = random_m31_matrix(prefill_len, d_model, 123);

    // Find the attention node to seed the cache
    for &node_id in &topo {
        let node = &decode_graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config } = &node.op {
            let w_q = weights.get_named_weight(node.id, "w_q").unwrap();
            let w_k = weights.get_named_weight(node.id, "w_k").unwrap();
            let w_v = weights.get_named_weight(node.id, "w_v").unwrap();
            let w_o = weights.get_named_weight(node.id, "w_o").unwrap();
            let attn_weights = AttentionWeights {
                w_q: w_q.clone(),
                w_k: w_k.clone(),
                w_v: w_v.clone(),
                w_o: w_o.clone(),
            };
            let cache = kv_cache.get_or_create(node.id, config);
            let _prefill_inter = attention_forward_cached(
                &prefill_input,
                &attn_weights,
                config,
                cache,
                config.causal,
            );
            eprintln!(
                "  Prefill seeded: cache_len={} for node {}",
                cache.len(),
                node.id
            );
        }
    }

    let max_seq_len = prefill_len + decode_steps;
    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, max_seq_len);
    eprintln!("  Initial KV commitment: {:?}", kv_commitment.commitment());

    // Weight cache: first step populates, subsequent steps hit all-cached fast path
    let weight_cache = stwo_ml::weight_cache::shared_cache("decode-bench");

    // Decode loop
    let mut decode_times = Vec::with_capacity(decode_steps);
    for step in 0..decode_steps {
        let token_input = random_m31_matrix(1, d_model, 1000 + step as u64);
        let t = Instant::now();
        let result = prove_model_pure_gkr_decode_step(
            &decode_graph,
            &token_input,
            &weights,
            &mut kv_cache,
            &mut kv_commitment,
            Some(&weight_cache),
        );
        let elapsed = t.elapsed();
        decode_times.push(elapsed);

        match result {
            Ok((proof, kv_commit)) => {
                eprintln!(
                    "  Step {}: {:.1}ms, kv_commit={:?}, gkr_layers={}",
                    step,
                    elapsed.as_secs_f64() * 1000.0,
                    kv_commit,
                    proof
                        .gkr_proof
                        .as_ref()
                        .map(|p| p.layer_proofs.len())
                        .unwrap_or(0),
                );
            }
            Err(e) => {
                eprintln!("  Step {} FAILED: {:?}", step, e);
            }
        }
    }

    // Report
    let times_ms: Vec<f64> = decode_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let avg = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tokens_per_sec = 1000.0 / avg;

    eprintln!("\n=== Decode Benchmark Results ===");
    eprintln!("  d_model={}, heads={}, d_ff={}", d_model, num_heads, d_ff);
    eprintln!("  Prefill len: {}", prefill_len);
    eprintln!("  Decode steps: {}", decode_steps);
    eprintln!("  Per-step: avg={:.1}ms, min={:.1}ms, max={:.1}ms", avg, min, max);
    eprintln!("  Throughput: {:.2} tokens/sec (proving)", tokens_per_sec);
    eprintln!("================================");
}

/// Prove a single decode step and verify the GKR proof end-to-end.
#[test]
fn decode_prove_verify_roundtrip() {
    let d_model = 64;
    let num_heads = 2;
    let d_ff = 256;
    let prefill_len = 4;

    // Build decode graph
    let mut builder = GraphBuilder::new((1, d_model));
    builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let graph = builder.build();

    let mut weights = generate_weights_for_graph(&graph, 42);

    // Add attention named weights
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config: _ } = &node.op {
            weights.add_named_weight(node.id, "w_q", random_m31_matrix(d_model, d_model, 200 + node.id as u64));
            weights.add_named_weight(node.id, "w_k", random_m31_matrix(d_model, d_model, 300 + node.id as u64));
            weights.add_named_weight(node.id, "w_v", random_m31_matrix(d_model, d_model, 400 + node.id as u64));
            weights.add_named_weight(node.id, "w_o", random_m31_matrix(d_model, d_model, 500 + node.id as u64));
        }
    }

    // Seed KV cache with prefill
    let mut kv_cache = ModelKVCache::new();
    let prefill_input = random_m31_matrix(prefill_len, d_model, 123);
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config } = &node.op {
            let attn_weights = AttentionWeights {
                w_q: weights.get_named_weight(node.id, "w_q").unwrap().clone(),
                w_k: weights.get_named_weight(node.id, "w_k").unwrap().clone(),
                w_v: weights.get_named_weight(node.id, "w_v").unwrap().clone(),
                w_o: weights.get_named_weight(node.id, "w_o").unwrap().clone(),
            };
            let cache = kv_cache.get_or_create(node.id, config);
            let _ = attention_forward_cached(&prefill_input, &attn_weights, config, cache, config.causal);
        }
    }

    // Prove a decode step
    let token_input = random_m31_matrix(1, d_model, 999);
    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);
    let result = prove_model_pure_gkr_decode_step(
        &graph, &token_input, &weights, &mut kv_cache, &mut kv_commitment, None,
    );
    let (proof, kv_commit) = result.expect("decode proving should succeed");

    // Verify: replay channel and verify GKR proof
    let gkr_proof = proof.gkr_proof.as_ref().expect("should have GKR proof");
    assert!(
        !gkr_proof.layer_proofs.is_empty(),
        "should have layer proofs"
    );
    assert!(
        gkr_proof.kv_cache_commitment.is_some(),
        "should have KV commitment"
    );
    assert_eq!(
        gkr_proof.kv_cache_commitment.unwrap(),
        kv_commit,
        "KV commitment should match"
    );

    // Replay the verification channel with same KV commitment mixing
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut verify_channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    verify_channel.mix_felt(gkr_proof.kv_cache_commitment.unwrap());
    verify_channel.mix_felt(gkr_proof.prev_kv_cache_commitment.unwrap());

    let verify_result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit,
        gkr_proof,
        &proof.execution.output,
        &weights,
        &mut verify_channel,
    );
    assert!(
        verify_result.is_ok(),
        "GKR verification should succeed: {:?}",
        verify_result.err()
    );
    eprintln!("Decode prove+verify roundtrip: PASSED");
}

/// Verify that KV commitment chain is consistent across multiple decode steps.
#[test]
fn decode_kv_commitment_chain() {
    let d_model = 64;
    let num_heads = 2;
    let d_ff = 256;
    let prefill_len = 4;
    let decode_steps = 5;

    let mut builder = GraphBuilder::new((1, d_model));
    builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let graph = builder.build();

    let mut weights = generate_weights_for_graph(&graph, 42);
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config: _ } = &node.op {
            weights.add_named_weight(node.id, "w_q", random_m31_matrix(d_model, d_model, 200 + node.id as u64));
            weights.add_named_weight(node.id, "w_k", random_m31_matrix(d_model, d_model, 300 + node.id as u64));
            weights.add_named_weight(node.id, "w_v", random_m31_matrix(d_model, d_model, 400 + node.id as u64));
            weights.add_named_weight(node.id, "w_o", random_m31_matrix(d_model, d_model, 500 + node.id as u64));
        }
    }

    // Seed KV cache
    let mut kv_cache = ModelKVCache::new();
    let prefill_input = random_m31_matrix(prefill_len, d_model, 123);
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config } = &node.op {
            let attn_weights = AttentionWeights {
                w_q: weights.get_named_weight(node.id, "w_q").unwrap().clone(),
                w_k: weights.get_named_weight(node.id, "w_k").unwrap().clone(),
                w_v: weights.get_named_weight(node.id, "w_v").unwrap().clone(),
                w_o: weights.get_named_weight(node.id, "w_o").unwrap().clone(),
            };
            let cache = kv_cache.get_or_create(node.id, config);
            let _ = attention_forward_cached(&prefill_input, &attn_weights, config, cache, config.causal);
        }
    }

    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, prefill_len + decode_steps);
    let initial_commitment = kv_commitment.commitment();
    let weight_cache = stwo_ml::weight_cache::shared_cache("chain-test");

    // Run decode steps and verify commitment chain
    let mut prev_commitment = initial_commitment;
    for step in 0..decode_steps {
        let token_input = random_m31_matrix(1, d_model, 1000 + step as u64);
        let (proof, new_commitment) =
            prove_model_pure_gkr_decode_step(
                &graph, &token_input, &weights, &mut kv_cache, &mut kv_commitment,
                Some(&weight_cache),
            )
                .unwrap_or_else(|e| panic!("step {} failed: {:?}", step, e));

        let gkr_proof = proof.gkr_proof.as_ref().unwrap();

        // Chain integrity: prev_kv_cache_commitment of step N must match
        // kv_cache_commitment from step N-1 (or initial commitment for step 0)
        assert_eq!(
            gkr_proof.prev_kv_cache_commitment.unwrap(),
            prev_commitment,
            "Step {}: prev_kv_commitment mismatch",
            step
        );
        assert_eq!(
            gkr_proof.kv_cache_commitment.unwrap(),
            new_commitment,
            "Step {}: new_kv_commitment mismatch",
            step
        );

        // Commitments should change each step (cache grows)
        assert_ne!(
            prev_commitment, new_commitment,
            "Step {}: commitment should change as cache grows",
            step
        );

        eprintln!(
            "  Step {}: chain OK (prev={:?} -> new={:?})",
            step, prev_commitment, new_commitment
        );
        prev_commitment = new_commitment;
    }
    eprintln!(
        "KV commitment chain ({} steps): PASSED",
        decode_steps
    );
}

/// Basic test: IncrementalPoseidonMerkle root matches full PoseidonMerkleTree::build.
#[test]
fn incremental_merkle_basic() {
    use starknet_crypto::poseidon_hash;
    use starknet_ff::FieldElement;
    use stwo_ml::crypto::poseidon_merkle::{IncrementalPoseidonMerkle, PoseidonMerkleTree};

    for n in [1usize, 2, 3, 4, 7, 8, 15, 16, 32] {
        let leaves: Vec<FieldElement> = (0..n).map(|i| FieldElement::from(i as u64 + 1)).collect();

        // Build incrementally
        let cap = n.next_power_of_two().max(2);
        let mut inc = IncrementalPoseidonMerkle::new(cap);
        for &leaf in &leaves {
            inc.push(leaf);
        }

        // Build from full tree
        let full = PoseidonMerkleTree::build(leaves.clone());

        assert_eq!(
            inc.root(),
            full.root(),
            "root mismatch for n={n}: incremental vs full"
        );

        // Also verify incrementally at each step — pad partial leaves to same capacity
        let mut inc2 = IncrementalPoseidonMerkle::new(cap);
        for (i, &leaf) in leaves.iter().enumerate() {
            inc2.push(leaf);
            // Pad partial leaves to same capacity so the Merkle tree shape matches
            let mut partial: Vec<FieldElement> = leaves[..=i].to_vec();
            partial.resize(cap, FieldElement::ZERO);
            let partial_full = PoseidonMerkleTree::build(partial);
            assert_eq!(
                inc2.root(),
                partial_full.root(),
                "partial root mismatch after {}/{n} leaves",
                i + 1
            );
        }
    }

    // Verify basic structure: 2-leaf tree
    let a = FieldElement::from(10u64);
    let b = FieldElement::from(20u64);
    let mut t = IncrementalPoseidonMerkle::new(2);
    t.push(a);
    t.push(b);
    assert_eq!(t.root(), poseidon_hash(a, b));

    eprintln!("incremental_merkle_basic: PASSED");
}

/// Test that automatic capacity growth preserves root consistency.
#[test]
fn incremental_merkle_grow() {
    use starknet_ff::FieldElement;
    use stwo_ml::crypto::poseidon_merkle::{IncrementalPoseidonMerkle, PoseidonMerkleTree};

    // Start with capacity 4, push 8 leaves (forces one grow)
    let mut inc = IncrementalPoseidonMerkle::new(4);
    assert_eq!(inc.capacity(), 4);

    let leaves: Vec<FieldElement> = (0..8).map(|i| FieldElement::from(i as u64 + 1)).collect();
    for &leaf in &leaves {
        inc.push(leaf);
    }
    assert_eq!(inc.capacity(), 8); // should have grown
    assert_eq!(inc.len(), 8);

    // Compare against a tree built with capacity 8 from the start
    let mut inc_8 = IncrementalPoseidonMerkle::new(8);
    for &leaf in &leaves {
        inc_8.push(leaf);
    }
    assert_eq!(inc.root(), inc_8.root(), "grown tree root should match pre-sized tree");

    // Also verify against PoseidonMerkleTree::build
    let full = PoseidonMerkleTree::build(leaves.clone());
    assert_eq!(inc.root(), full.root(), "grown tree should match full build");

    // Push more — force a second grow (to capacity 16)
    let extra: Vec<FieldElement> = (8..12).map(|i| FieldElement::from(i as u64 + 1)).collect();
    for &leaf in &extra {
        inc.push(leaf);
    }
    assert_eq!(inc.capacity(), 16);

    let all: Vec<FieldElement> = (0..12).map(|i| FieldElement::from(i as u64 + 1)).collect();
    let mut all_padded = all.clone();
    all_padded.resize(16, FieldElement::ZERO);
    let full_16 = PoseidonMerkleTree::build(all_padded);
    assert_eq!(inc.root(), full_16.root(), "double-grown tree should match");

    eprintln!("incremental_merkle_grow: PASSED");
}

/// Verify IncrementalKVCommitment chain consistency: each step produces a
/// deterministic commitment that changes as the cache grows.
#[test]
fn incremental_kv_commitment_matches() {
    let d_model = 64;
    let num_heads = 2;
    let prefill_len = 4;

    let mut builder = GraphBuilder::new((1, d_model));
    builder.transformer_block(num_heads, num_heads, 1, 256);
    let graph = builder.build();

    let mut weights = generate_weights_for_graph(&graph, 42);
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config: _ } = &node.op {
            weights.add_named_weight(node.id, "w_q", random_m31_matrix(d_model, d_model, 200 + node.id as u64));
            weights.add_named_weight(node.id, "w_k", random_m31_matrix(d_model, d_model, 300 + node.id as u64));
            weights.add_named_weight(node.id, "w_v", random_m31_matrix(d_model, d_model, 400 + node.id as u64));
            weights.add_named_weight(node.id, "w_o", random_m31_matrix(d_model, d_model, 500 + node.id as u64));
        }
    }

    // Seed KV cache
    let mut kv_cache = ModelKVCache::new();
    let prefill_input = random_m31_matrix(prefill_len, d_model, 123);
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config } = &node.op {
            let attn_weights = AttentionWeights {
                w_q: weights.get_named_weight(node.id, "w_q").unwrap().clone(),
                w_k: weights.get_named_weight(node.id, "w_k").unwrap().clone(),
                w_v: weights.get_named_weight(node.id, "w_v").unwrap().clone(),
                w_o: weights.get_named_weight(node.id, "w_o").unwrap().clone(),
            };
            let cache = kv_cache.get_or_create(node.id, config);
            let _ = attention_forward_cached(&prefill_input, &attn_weights, config, cache, config.causal);
        }
    }

    // Build two incremental commitments from the same state — should match
    let kv1 = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);
    let kv2 = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);
    assert_eq!(
        kv1.commitment(),
        kv2.commitment(),
        "same cache state should produce same commitment"
    );

    // Build with different capacity — root should differ (different tree shape)
    // but both are valid commitments
    let kv3 = IncrementalKVCommitment::from_kv_cache(&kv_cache, 32);
    // These may differ because the Merkle tree capacity/shape differs,
    // but commitment is still deterministic for the same capacity
    let kv3b = IncrementalKVCommitment::from_kv_cache(&kv_cache, 32);
    assert_eq!(kv3.commitment(), kv3b.commitment());

    // Verify commitment changes after appending
    let mut kv_mut = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);
    let before = kv_mut.commitment();

    // Simulate one decode step's KV append
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config } = &node.op {
            let attn_weights = AttentionWeights {
                w_q: weights.get_named_weight(node.id, "w_q").unwrap().clone(),
                w_k: weights.get_named_weight(node.id, "w_k").unwrap().clone(),
                w_v: weights.get_named_weight(node.id, "w_v").unwrap().clone(),
                w_o: weights.get_named_weight(node.id, "w_o").unwrap().clone(),
            };
            let cache = kv_cache.get_or_create(node.id, config);
            let step_input = random_m31_matrix(1, d_model, 777);
            let _ = attention_forward_cached(&step_input, &attn_weights, config, cache, config.causal);
        }
    }
    kv_mut.append_step(&kv_cache, 1);
    let after = kv_mut.commitment();
    assert_ne!(before, after, "commitment should change after append");

    eprintln!("incremental_kv_commitment_matches: PASSED");
}
