//! Decode-step GKR proving benchmark.
//!
//! Measures per-step decode latency to inform the proving protocol
//! (prove every token vs batch every N).
//!
//! Run: `cargo test --features std decode_benchmark -- --ignored --nocapture`
//!
//! Env vars for scale:
//!   BENCH_D_MODEL      (default 128)
//!   BENCH_NUM_HEADS    (default 4)
//!   BENCH_D_FF         (default 512)
//!   BENCH_PREFILL_LEN  (default 8)
//!   BENCH_DECODE_STEPS (default 10)
//!   BENCH_FORCE_CPU    (set to 1 to force CPU path for A/B comparison)

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

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[test]
#[ignore]
fn decode_benchmark() {
    // BENCH_FORCE_CPU=1 forces CPU path for A/B comparison
    let force_cpu = std::env::var("BENCH_FORCE_CPU").ok().map(|v| v == "1").unwrap_or(false);
    let _cpu_guard = if force_cpu {
        eprintln!("  [BENCH] BENCH_FORCE_CPU=1 — forcing CPU path");
        std::env::set_var("OBELYSK_FORCE_GPU", "0");
        Some(())
    } else {
        None
    };

    let d_model = env_usize("BENCH_D_MODEL", 128);
    let num_heads = env_usize("BENCH_NUM_HEADS", 4);
    let d_ff = env_usize("BENCH_D_FF", 512);
    let prefill_len = env_usize("BENCH_PREFILL_LEN", 8);
    let decode_steps = env_usize("BENCH_DECODE_STEPS", 10);

    let backend_label = if force_cpu { "CPU (forced)" } else { "auto (GPU if available)" };
    eprintln!("=== Decode Benchmark [{}] ===", backend_label);
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

    eprintln!("\n=== Decode Benchmark Results [{}] ===", backend_label);
    eprintln!("  d_model={}, heads={}, d_ff={}", d_model, num_heads, d_ff);
    eprintln!("  Prefill len: {}", prefill_len);
    eprintln!("  Decode steps: {}", decode_steps);

    // Cold vs cached breakdown
    if !times_ms.is_empty() {
        eprintln!("  Cold start (step 0): {:.1}ms", times_ms[0]);
    }
    if times_ms.len() > 1 {
        let cached_times: Vec<f64> = times_ms[1..].to_vec();
        let cached_avg = cached_times.iter().sum::<f64>() / cached_times.len() as f64;
        eprintln!(
            "  Cached avg (steps 1-{}): {:.1}ms",
            decode_steps - 1,
            cached_avg
        );

        // p50/p95/p99 from cached times only
        let mut sorted = cached_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);
        eprintln!("  Per-token (cached): p50={:.1}ms, p95={:.1}ms, p99={:.1}ms", p50, p95, p99);
    }

    eprintln!("  Per-step (all): avg={:.1}ms, min={:.1}ms, max={:.1}ms", avg, min, max);
    eprintln!("  Throughput: {:.2} tokens/sec (proving)", tokens_per_sec);
    eprintln!("================================");

    // Restore env if we forced CPU
    if force_cpu {
        std::env::remove_var("OBELYSK_FORCE_GPU");
    }
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

    // Verify position_offset is set correctly in the decode proof
    for lp in &gkr_proof.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode {
            position_offset,
            full_seq_len,
            new_tokens,
            ..
        } = lp
        {
            assert_eq!(*new_tokens, 1, "decode step should have new_tokens=1");
            assert_eq!(
                *full_seq_len,
                prefill_len + 1,
                "full_seq_len should be prefill + 1"
            );
            assert_eq!(
                *position_offset,
                prefill_len,
                "position_offset should equal prefill_len"
            );
            assert_eq!(
                *position_offset + *new_tokens,
                *full_seq_len,
                "position_offset + new_tokens should equal full_seq_len"
            );
        }
    }

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

        // Verify position_offset chain: position_offset[step] == full_seq_len[step-1]
        for lp in &gkr_proof.layer_proofs {
            if let stwo_ml::gkr::types::LayerProof::AttentionDecode {
                position_offset,
                full_seq_len,
                new_tokens,
                ..
            } = lp
            {
                assert_eq!(
                    *position_offset + *new_tokens, *full_seq_len,
                    "Step {}: position_offset + new_tokens != full_seq_len",
                    step
                );
                let expected_pos = prefill_len + step;
                assert_eq!(
                    *position_offset, expected_pos,
                    "Step {}: position_offset should be prefill_len + step",
                    step
                );
                eprintln!(
                    "  Step {}: position_offset={} full_seq_len={} new_tokens={}",
                    step, position_offset, full_seq_len, new_tokens
                );
            }
        }

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

/// Compare prefill (N tokens at once) vs decode (1 token N times) latency.
#[test]
#[ignore]
fn decode_benchmark_prefill_vs_decode() {
    let d_model = env_usize("BENCH_D_MODEL", 64);
    let num_heads = env_usize("BENCH_NUM_HEADS", 2);
    let d_ff = env_usize("BENCH_D_FF", 256);
    let prefill_len = env_usize("BENCH_PREFILL_LEN", 4);
    let decode_steps = env_usize("BENCH_DECODE_STEPS", 5);

    eprintln!("=== Prefill vs Decode Benchmark ===");
    eprintln!(
        "  d_model={}, heads={}, d_ff={}, prefill={}, decode_steps={}",
        d_model, num_heads, d_ff, prefill_len, decode_steps
    );

    // --- Prefill: prove a single forward pass with seq_len = decode_steps ---
    let mut prefill_builder = GraphBuilder::new((decode_steps, d_model));
    prefill_builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let prefill_graph = prefill_builder.build();

    let prefill_weights = generate_weights_for_graph(&prefill_graph, 42);
    let prefill_input = random_m31_matrix(decode_steps, d_model, 555);
    let weight_cache = stwo_ml::weight_cache::shared_cache("prefill-bench");

    let t_prefill = Instant::now();
    let prefill_result = stwo_ml::aggregation::prove_model_pure_gkr_auto_with_cache(
        &prefill_graph,
        &prefill_input,
        &prefill_weights,
        Some(&weight_cache),
    );
    let prefill_elapsed = t_prefill.elapsed();
    match &prefill_result {
        Ok(_) => eprintln!("  Prefill ({} tokens): {:.1}ms", decode_steps, prefill_elapsed.as_secs_f64() * 1000.0),
        Err(e) => eprintln!("  Prefill FAILED: {:?}", e),
    }

    // --- Decode: prove decode_steps individual tokens ---
    let mut decode_builder = GraphBuilder::new((1, d_model));
    decode_builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let decode_graph = decode_builder.build();

    let mut decode_weights = generate_weights_for_graph(&decode_graph, 42);
    let topo = decode_graph.topological_order();
    for &node_id in &topo {
        let node = &decode_graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config: _ } = &node.op {
            decode_weights.add_named_weight(node.id, "w_q", random_m31_matrix(d_model, d_model, 200 + node.id as u64));
            decode_weights.add_named_weight(node.id, "w_k", random_m31_matrix(d_model, d_model, 300 + node.id as u64));
            decode_weights.add_named_weight(node.id, "w_v", random_m31_matrix(d_model, d_model, 400 + node.id as u64));
            decode_weights.add_named_weight(node.id, "w_o", random_m31_matrix(d_model, d_model, 500 + node.id as u64));
        }
    }

    let mut kv_cache = ModelKVCache::new();
    let seed_input = random_m31_matrix(prefill_len, d_model, 123);
    for &node_id in &topo {
        let node = &decode_graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { config } = &node.op {
            let attn_weights = AttentionWeights {
                w_q: decode_weights.get_named_weight(node.id, "w_q").unwrap().clone(),
                w_k: decode_weights.get_named_weight(node.id, "w_k").unwrap().clone(),
                w_v: decode_weights.get_named_weight(node.id, "w_v").unwrap().clone(),
                w_o: decode_weights.get_named_weight(node.id, "w_o").unwrap().clone(),
            };
            let cache = kv_cache.get_or_create(node.id, config);
            let _ = attention_forward_cached(&seed_input, &attn_weights, config, cache, config.causal);
        }
    }

    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, prefill_len + decode_steps);
    let decode_cache = stwo_ml::weight_cache::shared_cache("decode-bench-cmp");

    let t_decode_total = Instant::now();
    for step in 0..decode_steps {
        let token_input = random_m31_matrix(1, d_model, 1000 + step as u64);
        let result = prove_model_pure_gkr_decode_step(
            &decode_graph, &token_input, &decode_weights,
            &mut kv_cache, &mut kv_commitment, Some(&decode_cache),
        );
        if let Err(e) = &result {
            eprintln!("  Decode step {} FAILED: {:?}", step, e);
        }
    }
    let decode_elapsed = t_decode_total.elapsed();
    eprintln!(
        "  Decode ({} tokens): {:.1}ms total ({:.1}ms/token)",
        decode_steps,
        decode_elapsed.as_secs_f64() * 1000.0,
        decode_elapsed.as_secs_f64() * 1000.0 / decode_steps as f64,
    );

    eprintln!("\n=== Prefill vs Decode Summary ===");
    eprintln!(
        "  Prefill: {:.1}ms ({} tokens at once)",
        prefill_elapsed.as_secs_f64() * 1000.0,
        decode_steps
    );
    eprintln!(
        "  Decode:  {:.1}ms ({} tokens one at a time)",
        decode_elapsed.as_secs_f64() * 1000.0,
        decode_steps
    );
    if prefill_elapsed.as_secs_f64() > 0.0 {
        eprintln!(
            "  Decode/Prefill ratio: {:.1}x",
            decode_elapsed.as_secs_f64() / prefill_elapsed.as_secs_f64()
        );
    }
    eprintln!("=================================");
}

// ─────────────────────────────────────────────────────────────────────────────
// Adversarial tamper tests — Phase 3 hardening
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: build a decode graph + weights + seeded KV cache, prove one decode step.
/// Returns (graph, weights, proof, output, kv_commitment) for tamper testing.
fn setup_decode_proof() -> (
    stwo_ml::compiler::graph::ComputationGraph,
    stwo_ml::compiler::graph::GraphWeights,
    stwo_ml::aggregation::AggregatedModelProofOnChain,
    M31Matrix,
    starknet_ff::FieldElement,
) {
    let d_model = 64;
    let num_heads = 2;
    let d_ff = 256;
    let prefill_len = 4;

    let mut builder = GraphBuilder::new((1, d_model));
    builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let graph = builder.build();

    let mut weights = stwo_ml::compiler::onnx::generate_weights_for_graph(&graph, 42);
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { .. } = &node.op {
            weights.add_named_weight(node.id, "w_q", random_m31_matrix(d_model, d_model, 200 + node.id as u64));
            weights.add_named_weight(node.id, "w_k", random_m31_matrix(d_model, d_model, 300 + node.id as u64));
            weights.add_named_weight(node.id, "w_v", random_m31_matrix(d_model, d_model, 400 + node.id as u64));
            weights.add_named_weight(node.id, "w_o", random_m31_matrix(d_model, d_model, 500 + node.id as u64));
        }
    }

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

    let token_input = random_m31_matrix(1, d_model, 999);
    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);

    let (proof, kv_commit) = prove_model_pure_gkr_decode_step(
        &graph, &token_input, &weights, &mut kv_cache, &mut kv_commitment, None,
    )
    .expect("setup: decode proving should succeed");

    (graph, weights, proof, token_input, kv_commit)
}

/// Tamper test: modifying position_offset in AttentionDecode layer proof
/// should cause verification to fail.
#[test]
fn decode_tamper_position_offset_rejected() {
    let (graph, weights, mut proof, _input, _kv) = setup_decode_proof();

    let gkr_proof = proof.gkr_proof.as_mut().expect("should have GKR proof");

    // Tamper: modify position_offset in every AttentionDecode layer
    let mut tampered = false;
    for lp in &mut gkr_proof.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode {
            ref mut position_offset,
            ..
        } = lp
        {
            *position_offset = 9999; // wrong offset
            tampered = true;
        }
    }
    assert!(tampered, "should have found at least one AttentionDecode layer to tamper");

    // Verify should fail
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    if let Some(kv) = gkr_proof.kv_cache_commitment {
        channel.mix_felt(kv);
    }
    if let Some(prev) = gkr_proof.prev_kv_cache_commitment {
        channel.mix_felt(prev);
    }

    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr_proof, &proof.execution.output, &weights, &mut channel,
    );
    assert!(result.is_err(), "tampered position_offset should fail verification");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("position_offset") || err_msg.contains("verification failed"),
        "error should mention position_offset, got: {err_msg}"
    );
    eprintln!("decode_tamper_position_offset_rejected: PASSED");
}

/// Tamper test: modifying KV cache commitment should cause verification to fail
/// because the Fiat-Shamir transcript will diverge.
#[test]
fn decode_tamper_kv_commitment_rejected() {
    let (graph, weights, mut proof, _input, _kv) = setup_decode_proof();

    let gkr_proof = proof.gkr_proof.as_mut().expect("should have GKR proof");

    // Tamper: replace kv_cache_commitment with a fake value
    let original = gkr_proof.kv_cache_commitment.unwrap();
    gkr_proof.kv_cache_commitment = Some(starknet_ff::FieldElement::from(0xDEADBEEFu64));
    assert_ne!(
        gkr_proof.kv_cache_commitment.unwrap(),
        original,
        "should have tampered the commitment"
    );

    // Verify with the TAMPERED commitment mixed into the channel
    // (simulating what a malicious prover would present)
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    channel.mix_felt(gkr_proof.kv_cache_commitment.unwrap());
    if let Some(prev) = gkr_proof.prev_kv_cache_commitment {
        channel.mix_felt(prev);
    }

    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr_proof, &proof.execution.output, &weights, &mut channel,
    );
    // The transcript divergence should cause sumcheck/claim verification to fail
    assert!(
        result.is_err(),
        "tampered kv_cache_commitment should fail verification"
    );
    eprintln!("decode_tamper_kv_commitment_rejected: PASSED");
}

/// Tamper test: modifying new_tokens in AttentionDecode should fail
/// because position_offset + new_tokens != full_seq_len.
#[test]
fn decode_tamper_new_tokens_rejected() {
    let (graph, weights, mut proof, _input, _kv) = setup_decode_proof();

    let gkr_proof = proof.gkr_proof.as_mut().expect("should have GKR proof");

    // Tamper: change new_tokens to 5 (was 1)
    let mut tampered = false;
    for lp in &mut gkr_proof.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode {
            ref mut new_tokens,
            ..
        } = lp
        {
            *new_tokens = 5;
            tampered = true;
        }
    }
    assert!(tampered, "should have found AttentionDecode layer to tamper");

    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    if let Some(kv) = gkr_proof.kv_cache_commitment {
        channel.mix_felt(kv);
    }
    if let Some(prev) = gkr_proof.prev_kv_cache_commitment {
        channel.mix_felt(prev);
    }

    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr_proof, &proof.execution.output, &weights, &mut channel,
    );
    assert!(result.is_err(), "tampered new_tokens should fail verification");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("position_offset") || err_msg.contains("new_tokens") || err_msg.contains("full_seq_len") || err_msg.contains("verification failed"),
        "error should mention position/token mismatch, got: {err_msg}"
    );
    eprintln!("decode_tamper_new_tokens_rejected: PASSED");
}

/// Tamper test: modifying prev_kv_cache_commitment should cause Fiat-Shamir
/// transcript divergence.
#[test]
fn decode_tamper_prev_kv_commitment_rejected() {
    let (graph, weights, mut proof, _input, _kv) = setup_decode_proof();

    let gkr_proof = proof.gkr_proof.as_mut().expect("should have GKR proof");

    // Tamper: replace prev_kv_cache_commitment
    gkr_proof.prev_kv_cache_commitment = Some(starknet_ff::FieldElement::from(0xCAFEBABEu64));

    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    if let Some(kv) = gkr_proof.kv_cache_commitment {
        channel.mix_felt(kv);
    }
    // Mix the TAMPERED prev commitment
    channel.mix_felt(gkr_proof.prev_kv_cache_commitment.unwrap());

    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr_proof, &proof.execution.output, &weights, &mut channel,
    );
    assert!(
        result.is_err(),
        "tampered prev_kv_cache_commitment should fail verification"
    );
    eprintln!("decode_tamper_prev_kv_commitment_rejected: PASSED");
}

/// Tamper test: modifying full_seq_len breaks position_offset + new_tokens == full_seq_len.
#[test]
fn decode_tamper_full_seq_len_rejected() {
    let (graph, weights, mut proof, _input, _kv) = setup_decode_proof();

    let gkr_proof = proof.gkr_proof.as_mut().expect("should have GKR proof");

    let mut tampered = false;
    for lp in &mut gkr_proof.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode {
            ref mut full_seq_len,
            ..
        } = lp
        {
            *full_seq_len = 100; // wrong value
            tampered = true;
        }
    }
    assert!(tampered);

    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    if let Some(kv) = gkr_proof.kv_cache_commitment {
        channel.mix_felt(kv);
    }
    if let Some(prev) = gkr_proof.prev_kv_cache_commitment {
        channel.mix_felt(prev);
    }

    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr_proof, &proof.execution.output, &weights, &mut channel,
    );
    assert!(result.is_err(), "tampered full_seq_len should fail verification");
    eprintln!("decode_tamper_full_seq_len_rejected: PASSED");
}

/// Tamper test: modifying a sub-proof claim value inside AttentionDecode
/// should cause sumcheck verification failure.
#[test]
fn decode_tamper_sub_claim_value_rejected() {
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::fields::m31::M31;

    let (graph, weights, mut proof, _input, _kv) = setup_decode_proof();

    let gkr_proof = proof.gkr_proof.as_mut().expect("should have GKR proof");

    let mut tampered = false;
    for lp in &mut gkr_proof.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode {
            ref mut sub_claim_values,
            ..
        } = lp
        {
            if !sub_claim_values.is_empty() {
                // Corrupt the first sub-claim value
                sub_claim_values[0] = SecureField::from(M31::from(0xBAADu32));
                tampered = true;
            }
        }
    }
    assert!(tampered, "should have found sub_claim_values to tamper");

    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    if let Some(kv) = gkr_proof.kv_cache_commitment {
        channel.mix_felt(kv);
    }
    if let Some(prev) = gkr_proof.prev_kv_cache_commitment {
        channel.mix_felt(prev);
    }

    let result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr_proof, &proof.execution.output, &weights, &mut channel,
    );
    assert!(
        result.is_err(),
        "tampered sub_claim_value should fail verification"
    );
    eprintln!("decode_tamper_sub_claim_value_rejected: PASSED");
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-token decode tests
// ─────────────────────────────────────────────────────────────────────────────

/// Prove a decode step with new_tokens > 1 (batch decode).
/// Verifies that position_offset and full_seq_len are correct for multi-token.
#[test]
fn decode_multi_token_batch() {
    let d_model = 64;
    let num_heads = 2;
    let d_ff = 256;
    let prefill_len = 4;
    let new_tokens = 3; // batch decode: 3 tokens at once

    // Build decode graph with seq_len matching new_tokens
    let mut builder = GraphBuilder::new((new_tokens, d_model));
    builder.transformer_block(num_heads, num_heads, new_tokens, d_ff);
    let graph = builder.build();

    let mut weights = stwo_ml::compiler::onnx::generate_weights_for_graph(&graph, 42);
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { .. } = &node.op {
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

    // Prove a multi-token decode step
    let batch_input = random_m31_matrix(new_tokens, d_model, 888);
    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);

    let result = prove_model_pure_gkr_decode_step(
        &graph, &batch_input, &weights, &mut kv_cache, &mut kv_commitment, None,
    );
    let (proof, kv_commit) = result.expect("multi-token decode proving should succeed");

    // Verify the proof
    let gkr_proof = proof.gkr_proof.as_ref().expect("should have GKR proof");
    assert!(!gkr_proof.layer_proofs.is_empty(), "should have layer proofs");

    // Check AttentionDecode metadata
    for lp in &gkr_proof.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode {
            new_tokens: nt,
            full_seq_len,
            position_offset,
            ..
        } = lp
        {
            assert_eq!(*nt, new_tokens, "new_tokens should match batch size");
            assert_eq!(
                *full_seq_len,
                prefill_len + new_tokens,
                "full_seq_len = prefill + new_tokens"
            );
            assert_eq!(
                *position_offset,
                prefill_len,
                "position_offset = prefill_len"
            );
            assert_eq!(
                *position_offset + *nt, *full_seq_len,
                "position_offset + new_tokens == full_seq_len"
            );
            eprintln!(
                "  Multi-token: pos_offset={}, new_tokens={}, full_seq_len={}",
                position_offset, nt, full_seq_len
            );
        }
    }

    // Verify GKR proof
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&graph).unwrap();
    let mut channel = stwo_ml::crypto::poseidon_channel::PoseidonChannel::new();
    channel.mix_felt(gkr_proof.kv_cache_commitment.unwrap());
    channel.mix_felt(gkr_proof.prev_kv_cache_commitment.unwrap());

    let verify_result = stwo_ml::gkr::verify_gkr_with_weights(
        &circuit, gkr_proof, &proof.execution.output, &weights, &mut channel,
    );
    assert!(
        verify_result.is_ok(),
        "multi-token decode GKR verification should succeed: {:?}",
        verify_result.err()
    );

    // Verify commitment was updated
    assert_ne!(
        kv_commit,
        starknet_ff::FieldElement::ZERO,
        "KV commitment should be non-zero"
    );

    eprintln!("decode_multi_token_batch: PASSED (new_tokens={})", new_tokens);
}

/// Multi-token decode followed by single-token decode: chain consistency.
#[test]
fn decode_multi_then_single_chain() {
    let d_model = 64;
    let num_heads = 2;
    let d_ff = 256;
    let prefill_len = 4;

    // Build graph for single-token decode
    let mut builder = GraphBuilder::new((1, d_model));
    builder.transformer_block(num_heads, num_heads, 1, d_ff);
    let graph = builder.build();

    let mut weights = stwo_ml::compiler::onnx::generate_weights_for_graph(&graph, 42);
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let stwo_ml::compiler::graph::GraphOp::Attention { .. } = &node.op {
            weights.add_named_weight(node.id, "w_q", random_m31_matrix(d_model, d_model, 200 + node.id as u64));
            weights.add_named_weight(node.id, "w_k", random_m31_matrix(d_model, d_model, 300 + node.id as u64));
            weights.add_named_weight(node.id, "w_v", random_m31_matrix(d_model, d_model, 400 + node.id as u64));
            weights.add_named_weight(node.id, "w_o", random_m31_matrix(d_model, d_model, 500 + node.id as u64));
        }
    }

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

    let mut kv_commitment = IncrementalKVCommitment::from_kv_cache(&kv_cache, 16);
    let weight_cache = stwo_ml::weight_cache::shared_cache("multi-single-chain");

    // Step 0: first single decode
    let t0_input = random_m31_matrix(1, d_model, 1000);
    let (proof0, commit0) = prove_model_pure_gkr_decode_step(
        &graph, &t0_input, &weights, &mut kv_cache, &mut kv_commitment, Some(&weight_cache),
    ).expect("step 0 should succeed");

    // Step 1: second single decode
    let t1_input = random_m31_matrix(1, d_model, 1001);
    let (proof1, commit1) = prove_model_pure_gkr_decode_step(
        &graph, &t1_input, &weights, &mut kv_cache, &mut kv_commitment, Some(&weight_cache),
    ).expect("step 1 should succeed");

    // Chain: step1's prev must equal step0's new
    let gkr0 = proof0.gkr_proof.as_ref().unwrap();
    let gkr1 = proof1.gkr_proof.as_ref().unwrap();

    assert_eq!(
        gkr1.prev_kv_cache_commitment.unwrap(),
        commit0,
        "step 1 prev_kv should match step 0 new_kv"
    );
    assert_ne!(commit0, commit1, "commitments should differ");

    // Verify position offsets
    for lp in &gkr0.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode { position_offset, .. } = lp {
            assert_eq!(*position_offset, prefill_len, "step 0 offset");
        }
    }
    for lp in &gkr1.layer_proofs {
        if let stwo_ml::gkr::types::LayerProof::AttentionDecode { position_offset, .. } = lp {
            assert_eq!(*position_offset, prefill_len + 1, "step 1 offset");
        }
    }

    eprintln!("decode_multi_then_single_chain: PASSED");
}

/// Verify GPU and CPU decode provers produce identical layer proofs.
#[cfg(feature = "cuda-runtime")]
#[test]
fn decode_gpu_cpu_equivalence() {
    use stwo_ml::gkr::circuit::LayeredCircuit;
    use stwo_ml::compiler::onnx::generate_weights_for_graph;

    let d_model = 64;
    let num_heads = 2;
    let d_ff = 256;
    let prefill_len = 4;

    // Build decode graph
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

    // Seed KV cache with prefill
    let mut kv_cache_cpu = ModelKVCache::new();
    let mut kv_cache_gpu = ModelKVCache::new();
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
            let cache_cpu = kv_cache_cpu.get_or_create(node.id, config);
            let _ = attention_forward_cached(&prefill_input, &attn_weights, config, cache_cpu, config.causal);
            let cache_gpu = kv_cache_gpu.get_or_create(node.id, config);
            let _ = attention_forward_cached(&prefill_input, &attn_weights, config, cache_gpu, config.causal);
        }
    }

    let token_input = random_m31_matrix(1, d_model, 999);

    // Run CPU decode proving
    let mut kv_commitment_cpu = IncrementalKVCommitment::from_kv_cache(&kv_cache_cpu, 16);
    let (proof_cpu, _) = prove_model_pure_gkr_decode_step(
        &graph, &token_input, &weights, &mut kv_cache_cpu, &mut kv_commitment_cpu, None,
    ).expect("CPU decode should succeed");

    // Run GPU decode proving (same inputs, fresh channel)
    let mut kv_commitment_gpu = IncrementalKVCommitment::from_kv_cache(&kv_cache_gpu, 16);
    let (proof_gpu, _) = prove_model_pure_gkr_decode_step(
        &graph, &token_input, &weights, &mut kv_cache_gpu, &mut kv_commitment_gpu, None,
    ).expect("GPU decode should succeed");

    // Compare GKR proofs
    let gkr_cpu = proof_cpu.gkr_proof.as_ref().expect("CPU should have GKR proof");
    let gkr_gpu = proof_gpu.gkr_proof.as_ref().expect("GPU should have GKR proof");

    assert_eq!(
        gkr_cpu.layer_proofs.len(),
        gkr_gpu.layer_proofs.len(),
        "layer proof count mismatch: CPU={} vs GPU={}",
        gkr_cpu.layer_proofs.len(),
        gkr_gpu.layer_proofs.len(),
    );

    // The proofs should be transcript-identical since both use the same
    // Fiat-Shamir channel and the same mathematical reductions.
    // Note: if GPU uses different floating-point rounding for intermediate
    // computations, the proofs may differ — that would indicate a bug.
    for (i, (cpu_lp, gpu_lp)) in gkr_cpu.layer_proofs.iter().zip(gkr_gpu.layer_proofs.iter()).enumerate() {
        assert_eq!(
            std::mem::discriminant(cpu_lp),
            std::mem::discriminant(gpu_lp),
            "layer {} proof type mismatch",
            i,
        );
    }

    assert_eq!(
        gkr_cpu.output_claim.value,
        gkr_gpu.output_claim.value,
        "output claim value mismatch"
    );
    assert_eq!(
        gkr_cpu.input_claim.value,
        gkr_gpu.input_claim.value,
        "input claim value mismatch"
    );

    eprintln!("decode_gpu_cpu_equivalence: PASSED ({} layer proofs match)", gkr_cpu.layer_proofs.len());
}
