# Transformer Architecture — Full Llama-Style Proving Pipeline

## Overview

`stwo-ml` supports proving full transformer decoder blocks as used in Llama, Qwen, Mistral, and similar architectures. Each block follows the pre-norm residual pattern:

```
                    ┌───────────────┐
          Input ────┤   Identity    ├──────────────────────┐
                    └───────┬───────┘                      │
                            ▼                              │ (residual)
                    ┌───────────────┐                      │
                    │   RMSNorm     │                      │
                    └───────┬───────┘                      │
                            ▼                              │
                    ┌───────────────┐                      │
                    │  Attention    │  (GQA/MQA/MHA)       │
                    │  Q×K^T→soft→V│                       │
                    └───────┬───────┘                      │
                            ▼                              │
                    ┌───────────────┐                      │
                    │     Add       │◄─────────────────────┘
                    └───────┬───────┘
                            │──────────────────────────────┐
                            ▼                              │ (residual)
                    ┌───────────────┐                      │
                    │   RMSNorm     │                      │
                    └───────┬───────┘                      │
                            ▼                              │
                    ┌───────────────┐                      │
                    │  FFN: Linear  │  (d_model → ffn_dim) │
                    └───────┬───────┘                      │
                            ▼                              │
                    ┌───────────────┐                      │
                    │     GELU      │                      │
                    └───────┬───────┘                      │
                            ▼                              │
                    ┌───────────────┐                      │
                    │  FFN: Linear  │  (ffn_dim → d_model) │
                    └───────┬───────┘                      │
                            ▼                              │
                    ┌───────────────┐                      │
                    │     Add       │◄─────────────────────┘
                    └───────┬───────┘
                         Output
```

## Builder API

The `GraphBuilder::transformer_block()` method constructs this entire pattern in one call:

```rust
use stwo_ml::compiler::graph::GraphBuilder;

let mut builder = GraphBuilder::new((seq_len, d_model));

// Stack N transformer blocks
for _ in 0..num_layers {
    builder.transformer_block(
        num_heads,      // Q heads (e.g., 32)
        num_kv_heads,   // KV heads (e.g., 8 for GQA)
        seq_len,        // sequence length
        ffn_dim,        // feed-forward intermediate dim (e.g., 4 × d_model)
    );
}

let graph = builder.build();
```

Each block produces 9 graph nodes: Identity, RMSNorm, Attention, Add, RMSNorm, Linear, GELU, Linear, Add.

## Components

### RMSNorm (`components/rmsnorm.rs`)

Root Mean Square Layer Normalization — used in Llama/Qwen instead of LayerNorm.

**Formula**: `y = x / sqrt(mean(x^2) + epsilon) * gamma`

**Key difference from LayerNorm**: No mean subtraction. Cheaper to compute and prove.

**Proving approach**: Decomposed into three provable operations:
1. Compute `rms^2 = sum(x^2) / n` via M31 arithmetic
2. Reciprocal sqrt `rsqrt(rms^2)` via LogUp lookup table (precomputed table of 2^16 entries)
3. Scale: `output = input * rsqrt_val`

**Trace layout** (5 columns):

| Column | Name | Description |
|--------|------|-------------|
| 0 | input | Original value x |
| 1 | rms_sq | mean(x^2), shared per row |
| 2 | rsqrt_val | 1/sqrt(rms_sq), from lookup |
| 3 | output | x * rsqrt_val |
| 4 | multiplicity | LogUp multiplicity |

**Constraint**: Uses `RMSNormRelation` (2-element LogUp) with `finalize_logup_in_pairs()`.

### RoPE — Rotary Positional Embedding (`components/rope.rs`)

Position-dependent rotations applied to Q and K vectors before attention scoring.

**Formula**:
```
x' = x * cos(theta * m) - y * sin(theta * m)
y' = x * sin(theta * m) + y * cos(theta * m)
```

where `theta_j = base^(-2j/d)`, `m` = position index.

**Proving approach**:
1. **Precompute rotation table**: All (position, dim_pair) -> (cos_val, sin_val) in M31
2. **Element-wise rotation**: Apply rotation using table values
3. **LogUp proof**: Every (cos, sin) pair comes from the precomputed table

The rotation factors are deterministic from `(seq_len, head_dim, base)`, so the verifier reconstructs the table independently.

**Configuration**:
```rust
pub struct RoPEConfig {
    pub seq_len: usize,      // number of positions
    pub head_dim: usize,     // per-head dimension (must be even)
    pub base: f64,           // frequency base (default: 10000)
    pub max_seq_len: usize,  // max positions for table precomputation
}
```

### Grouped Query Attention — GQA/MQA (`components/attention.rs`)

The attention component supports three modes:

| Mode | KV Heads | Description |
|------|----------|-------------|
| **MHA** | `num_kv_heads == num_heads` | Standard multi-head attention |
| **GQA** | `1 < num_kv_heads < num_heads` | Groups of Q heads share K/V |
| **MQA** | `num_kv_heads == 1` | All Q heads share one K/V head |

GQA is used by Llama 3, Qwen 2/3, Mistral, and most modern LLMs because it reduces KV-cache memory by `num_heads / num_kv_heads` with minimal quality loss.

**How it works**: K and V are projected to `num_kv_heads` heads instead of `num_heads`. Each Q head `h` uses KV head `h / group_size`:

```rust
let group_size = num_heads / num_kv_heads;
for h in 0..num_heads {
    let kv_idx = h / group_size;
    // Q_h uses K[kv_idx] and V[kv_idx]
    let scores = matmul(&q_heads[h], &transpose(&kv_heads_k[kv_idx]));
    let context = matmul(&softmax(&scores), &kv_heads_v[kv_idx]);
}
```

**Proof decomposition**: Same as standard attention (Q/K/V projections + per-head score + softmax + context + output projection), but K/V projections produce fewer heads. The sumcheck proofs for shared K/V heads are reused across Q head groups.

**Constructors**:
```rust
// Standard MHA
MultiHeadAttentionConfig::new(32, 4096, 2048)

// GQA: 32 Q heads, 8 KV heads
MultiHeadAttentionConfig::new_gqa(32, 8, 4096, 2048, true)

// MQA: all Q heads share 1 KV head
MultiHeadAttentionConfig::new_mqa(32, 4096, 2048, true)
```

### KV-Cache — Incremental Decoding

For autoregressive generation, the KV-Cache stores previously computed K/V projections so each new token only requires O(1) new computation instead of reprocessing the full sequence.

```rust
pub struct KVCache {
    pub k_cache: Vec<M31Matrix>,  // per KV-head: (cached_len, d_k)
    pub v_cache: Vec<M31Matrix>,  // per KV-head: (cached_len, d_k)
    pub cached_len: usize,
    pub num_kv_heads: usize,
    pub d_k: usize,
}
```

**Usage**:
```rust
let config = MultiHeadAttentionConfig::new_gqa(32, 8, 4096, 1, true);
let mut cache = KVCache::new(&config);

// Step 1: process first token
let out1 = attention_forward_cached(&input_tok1, &weights, &config, &mut cache);
// cache.cached_len == 1

// Step 2: process next token (uses cached K/V from step 1)
let out2 = attention_forward_cached(&input_tok2, &weights, &config, &mut cache);
// cache.cached_len == 2
```

**Causal masking**: The cached attention correctly handles position offsets — new queries attend to all cached positions plus the current token, with causal masking applied at the correct offset.

**Multi-layer cache**: `ModelKVCache` wraps per-layer caches:
```rust
let mut model_cache = ModelKVCache::new(num_layers, &attn_config);
for layer in 0..num_layers {
    output = attention_forward_cached(&output, &weights[layer], &config, &mut model_cache.layers[layer]);
}
```

## Proof Structure

A single transformer block generates:

| Component | Count | Protocol |
|-----------|-------|----------|
| RMSNorm | 2 | LogUp STARK (rsqrt table) |
| Attention | 1 | 4+2H composed sumcheck + LogUp (softmax) |
| FFN MatMul | 2 | Sumcheck over MLE |
| GELU | 1 | LogUp STARK (activation table) |
| Add (residual) | 2 | Linear split (no proof needed in GKR) |

For a 32-head GQA model with 8 KV heads, one block produces 69 matmul sumcheck proofs + 2 LogUp STARKs + 1 softmax STARK.

## Integration with GKR

When using the GKR protocol, the transformer block is compiled into a `LayeredCircuit` where each graph node becomes one or more layers:

```
GKR Output claim
    → RMSNorm (eq-sumcheck + LogUp)
    → Attention (composed sub-matmuls)
    → Add (linear split)
    → RMSNorm (eq-sumcheck + LogUp)
    → Linear (matmul sumcheck)
    → GELU (LogUp eq-sumcheck)
    → Linear (matmul sumcheck)
    → Add (linear split)
    → Input claim
```

The GKR proof for the entire block is a single interactive proof, replacing what would otherwise be ~75 independent STARK proofs.

## Example: Full Pipeline

```rust
use stwo_ml::compiler::graph::GraphBuilder;
use stwo_ml::aggregation::{prove_model_aggregated, verify_aggregated_model_proof};

// Build a 2-layer transformer
let mut builder = GraphBuilder::new((2, 64)); // seq_len=2, d_model=64
builder
    .transformer_block(4, 2, 2, 128)  // 4 heads, 2 KV heads, seq=2, ffn=128
    .transformer_block(4, 2, 2, 128);

let graph = builder.build();

// Create weights for all MatMul and Attention nodes
let weights = create_weights_for_graph(&graph);

// Prove
let proof = prove_model_aggregated(&graph, &input, &weights)
    .expect("proving failed");

// Verify
verify_aggregated_model_proof(proof, &graph, &input, &weights)
    .expect("verification failed");
```
