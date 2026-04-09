# Changelog — stwo-ml v0.2.0

**Branch**: `feat/batch-sumcheck-verifier`
**Date**: February 2026
**Base**: Merged from PR #37 (`feat: batch sumcheck verification`)

---

## Summary

This release transforms stwo-ml from a matmul+activation proving library into a full transformer inference proving system. The major additions are:

1. **GKR Protocol Engine** — Layer-by-layer interactive proof replacing per-layer independent STARKs
2. **Full Transformer Block Support** — RMSNorm, RoPE, GQA/MQA attention, KV-cache, causal masking
3. **Quantized Inference (INT4/INT8)** — LogUp dequantize/quantize with Poseidon-committed parameters
4. **Multi-GPU Distributed Proving** — Device affinity, chunk partitioning, streamed weight loading
5. **Cairo On-Chain Verifier** — Decomposed layer-by-layer GKR verification on Starknet
6. **Security Audit (24 findings)** — All addressed across Critical/High/Medium/Quantize tiers
7. **442 passing tests** (up from ~380)

---

## New Components

### RMSNorm (`components/rmsnorm.rs`)

Root Mean Square Layer Normalization for Llama/Qwen pre-norm architecture. Decomposes into:
- `rms^2 = sum(x^2) / n` via M31 arithmetic
- Reciprocal sqrt via LogUp lookup table (2^16 entries)
- `output = input * rsqrt_val`

5-column trace layout with `finalize_logup_in_pairs()`. Unique PreProcessedColumnId strings per instance to avoid allocator collision.

### RoPE (`components/rope.rs`)

Rotary Positional Embedding for position-dependent Q/K rotations. Precomputes `(cos, sin)` rotation table deterministically from `(seq_len, head_dim, base)`. LogUp proves every rotation pair used exists in the table. Verifier reconstructs the table independently.

### Dequantize (`components/dequantize.rs`)

LogUp 2D lookup table for INT4 (16 entries) and INT8 (256 entries) dequantization. Proves `(quantized_input, dequantized_output)` pair membership. Same `FrameworkEval` pattern as `ActivationEval`.

### GQA/MQA Attention (`components/attention.rs`)

Extended `MultiHeadAttentionConfig` with:
- `num_kv_heads` field for Grouped Query / Multi-Query Attention
- `causal: bool` flag for autoregressive masking
- `new_gqa()`, `new_mqa()`, `new_causal()` constructors
- `KVCache` for incremental decoding with position offset tracking

### Transformer Block Builder (`compiler/graph.rs`)

`GraphBuilder::transformer_block(num_heads, num_kv_heads, seq_len, ffn_dim)` produces a 9-node subgraph:
```
Identity → RMSNorm → Attention → Add → RMSNorm → Linear → GELU → Linear → Add
```

---

## GKR Protocol Engine (`gkr/`)

Complete layer-typed GKR implementation:

| File | Purpose |
|------|---------|
| `types.rs` | `GKRProof`, `LayerProof`, `GKRClaim`, `RoundPolyDeg3`, `LogUpProof` |
| `circuit.rs` | `LayeredCircuit` compiler from `ComputationGraph` |
| `prover.rs` | Layer reductions: matmul sumcheck, add split, mul eq-sumcheck, activation LogUp, layernorm combined-product, attention composed sub-matmuls |
| `verifier.rs` | Fiat-Shamir transcript replay verification |

GPU variants: `prove_gkr_gpu()` (single block), `prove_gkr_simd_gpu()` (SIMD batching across blocks).

SIMD block batching leverages linearity: `Σ w_b (A_b × B) = (Σ w_b A_b) × B` — only combine activations, shared weights need no block extension.

---

## Aggregation Pipeline (`aggregation.rs`)

Major expansion (+3714 lines) adding:

- **RMSNorm claims**: `rmsnorm_claims: Vec<LayerClaim>` in proof structs
- **Dequantize claims**: `dequantize_claims: Vec<LayerClaim>` in proof structs
- **Quantization params commitment**: `compute_quantize_params_commitment()` — Poseidon hash of (strategy, scale, zero_point, bits) per layer
- **Node-identity completeness**: `verify_model_completeness()` now matches by `HashSet<usize>` of node IDs, not just counts (H4 fix)
- **IO domain separation**: Length/dimension prefixes in `compute_io_commitment()` (H5 fix)
- **Activation type tags**: 3-element LogUp relation `(type_tag, input, output)` (M1 fix)
- **Interaction PoW sync**: `channel.mix_u64(0)` after Tree 1 commit (H3 fix)
- **PrecomputedMatmuls injection**: Pre-computed matmul outputs + proofs bypass redundant weight loading
- **GKR integration**: `prove_model_aggregated_onchain_gkr()`, `prove_model_pure_gkr_auto()`

---

## Cairo On-Chain Verifier (`elo-cairo-verifier/`)

New files for decomposed GKR verification:

| File | Purpose |
|------|---------|
| `src/gkr.cairo` | GKR batch verifier matching STWO's `partially_verify_batch()` |
| `src/model_verifier.cairo` | Full GKR model walk: `verify_gkr_model()` |
| `src/layer_verifiers.cairo` | Per-layer verifiers for all op types |
| `src/logup.cairo` | LogUp table-side sum verification with Montgomery batch inversion |
| `src/channel.cairo` | PoseidonChannel with QM31 field arithmetic |
| `src/field.cairo` | M31/QM31 field operations |
| `src/types.cairo` | GKR proof types, RoundPoly, GKRClaim structs |

Test files: `test_batch.cairo`, `test_direct.cairo`, `test_gkr.cairo`, `test_layer_verifiers.cairo`, `test_logup.cairo`, `test_model_verifier.cairo`, `test_unified.cairo`.

---

## GPU Acceleration (`gpu_sumcheck.rs`)

+898 lines of CUDA kernel additions:

- **Fused MLE restrict**: `m31_restrict_rows_kernel`, `m31_restrict_cols_kernel` — takes original M31 matrix + QM31 Lagrange basis → restricted vector on GPU
- **GEMM**: `m31_gemm_kernel` for multi-row matmul (2D grid, 16×16 blocks)
- **Element-wise ops**: `m31_add_kernel`, `m31_mul_kernel`, `m31_relu_kernel`
- **GKR kernels**: `combine_blocks_kernel` for SIMD block batching
- **Kernel management**: `ForwardKernels` + `RestrictKernels` lazy-compiled groups

Multi-GPU support (`multi_gpu.rs`): device affinity via thread-local storage, chunk-to-GPU assignment, `DeviceGuard` RAII.

---

## Tile-Level Streaming (`compiler/streaming.rs`, `compiler/chunked.rs`)

For models exceeding GPU memory:

- **Chunked proving** (+1533 lines): Chunk-level memory-efficient proving with execution checkpoints
- **Tile streaming**: Per-chunk weight loading from mmap'd SafeTensors, double-buffered pipeline (load tile N+1 while proving tile N)
- **Peak RAM**: 1 tile instead of full weight matrix
- **Tiled matmul**: `TiledMatMulProof` for tile-level composition via `compose_tiled_proof()`

---

## Infrastructure

### Poseidon Channel (`crypto/poseidon_channel.rs`)

+1028 lines: Full Poseidon252-based Fiat-Shamir channel for Cairo verifier integration. Matches the Cairo verifier's channel state exactly.

### Cairo Serde (`cairo_serde.rs`)

+648 lines: Rust `StarkProof` → `Vec<felt252>` serialization with GKR proof support, `WeightMleOpening` for direct on-chain verification.

### Starknet Integration (`starknet.rs`)

+662 lines: `DirectStarknetProof`, `ModelRegistration`, `compute_weight_commitment()`, GKR calldata serialization.

### Prove Model CLI (`bin/prove_model.rs`)

+231 lines: New flags `--multi-gpu`, `--chunk-budget-gb`, `--gkr`. Output formats: `Direct`, `MlGkr`.

### Prove Server (`bin/prove_server.rs`)

New HTTP API server (axum) wrapping the proving library. Endpoints: `/health`, `/api/v1/models`, `/api/v1/prove`, `/api/v1/prove/{id}/result`.

---

## Input Claim Verification (C8)

Critical soundness fix: the on-chain GKR verifier now evaluates MLEs against raw I/O data at both endpoints of the GKR walk, preventing proof/data substitution attacks. See [`docs/input-claim-verification.md`](input-claim-verification.md).

- `evaluate_mle()` and `pad_and_embed_m31s()` added to Cairo field module
- `verify_model_gkr` contract interface simplified: raw I/O replaces 4 prover-supplied parameters
- `build_verify_model_gkr_calldata()` updated to pass raw I/O data
- 6 new Cairo MLE tests, all 24 Rust e2e tests updated

---

## IO Commitment Recomputation (H4 Phase 2)

Eliminates the last trusted caller input: `io_commitment: felt252` is no longer a function parameter. All three on-chain verification paths now accept `raw_io_data: Array<felt252>` and recompute `Poseidon(raw_io_data)` on-chain.

**Motivation**: Even after Phase 1 (H4) bound the IO commitment to the Fiat-Shamir transcript, a malicious caller could still supply a fabricated `io_commitment` value that passes the sumcheck/GKR checks but represents different I/O than what was actually computed.

**Changes**:

| Component | Before | After |
|-----------|--------|-------|
| `types.cairo` | `ModelProof.io_commitment: felt252` | `ModelProof.raw_io_data: Array<felt252>` |
| `types.cairo` | `DirectModelProof.io_commitment: felt252` | `DirectModelProof.raw_io_data: Array<felt252>` |
| `verifier.cairo` | Trust caller `io_commitment` | `poseidon_hash_span(raw_io_data.span())` on-chain |
| `starknet.rs` | `build_starknet_proof_onchain(proof)` | `build_starknet_proof_onchain(proof, input)` |
| Calldata `[4]` | `io_commitment` (1 felt) | `raw_io_data.len()` (length-prefixed array) |

**Raw IO Data Layout**: `[in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...]`

- Domain-separated dimension prefixes prevent input/output confusion
- Length prefixes enable deterministic on-chain parsing
- GKR path evaluates `MLE(output, r_out)` and `MLE(input, final_point)` on-chain

**Tests updated**: 20/20 Rust starknet, 24/24 e2e_cairo_verify, 246/246 Cairo snforge.

---

## Security Fixes

25 findings across 4 severity tiers — all fixed. See [`docs/security-audit.md`](security-audit.md) for details.

### Committed (3 commits)

| Commit | Findings | Description |
|--------|----------|-------------|
| `245f06f` | H3 | Sync `channel.mix_u64(0)` after Tree 1 commit (interaction PoW) |
| `154f8cd` | H4, H5, M1 | Node-identity completeness, IO domain separation, activation type tags |
| `e5a8b04` | — | Merge PR #37 (batch sumcheck verifier baseline) |

### Uncommitted (this branch)

| Finding | Description |
|---------|-------------|
| C8 | Input/output claim verification: on-chain MLE evaluation against raw data |
| Q1 | Quantize forward pass: real formula instead of simple clamp |
| Q2 | QuantizeEval: 1D range-check → 2D LogUp (input, output) binding |
| Q3 | Quantization parameters Poseidon commitment |
| Q4 | QuantizeLayerData: added input_values and full QuantParams |
| Q5 | GQA verifier: split K/V by `num_kv_heads` not `num_heads` |
| Q6 | Quantize trace: 1D → 2D columns across Trees 0/1/2 |

---

## Bug Fixes

### GQA Verifier Head Splitting (Q5)

`verify_attention_proof_blake2s()` split K and V heads by `num_heads` (Q head count) instead of `num_kv_heads`. This worked coincidentally for MHA where the counts are equal, but crashed with `zip_eq() reached end of one iterator` for GQA/MQA models.

**Root cause**: `split_heads(&inter.k, config.num_heads)` should have been `split_heads(&inter.k, config.num_kv_heads)`.

**Impact**: The `test_transformer_block_prove_verify` test (first to combine Activation + Add + RMSNorm + GQA Attention in a single unified STARK) exposed this bug.

**Fix**: Split K/V by `num_kv_heads`, index via `kv_idx = h / group_size`.

### Interaction PoW Channel Sync (H3)

Rust prover/verifier skipped `channel.mix_u64(0)` after Tree 1 commit, but the Cairo verifier always calls it (even with `INTERACTION_POW_BITS=0`). This caused Fiat-Shamir transcript divergence — on-chain verification would fail because lookup elements are drawn from different channel states.

---

## Test Results

```
test result: ok. 442 passed; 0 failed; 0 ignored; 0 measured (lib)
test result: ok. 5 passed (cross_verify)
test result: ok. 24 passed (e2e_cairo_verify)
test result: ok. 3 passed (e2e_full_pipeline)
test result: ok. 4 passed (transcript_vectors)
```

Key new tests:
- `test_transformer_block_prove_verify` — full GQA transformer block end-to-end
- `test_gqa_prove_and_verify` — GQA-specific attention verification
- `test_mqa_prove_onchain` — MQA on-chain path
- `test_attention_proof_tampered_fails_verification` — tamper detection
- `test_attention_proof_verified_in_aggregation` — attention in unified pipeline

---

## Documentation

| Document | Status |
|----------|--------|
| [`docs/transformer-architecture.md`](transformer-architecture.md) | New |
| [`docs/gkr-protocol.md`](gkr-protocol.md) | New |
| [`docs/gpu-acceleration.md`](gpu-acceleration.md) | New |
| [`docs/simd-block-batching.md`](simd-block-batching.md) | New |
| [`docs/input-claim-verification.md`](input-claim-verification.md) | New |
| [`docs/security-audit.md`](security-audit.md) | Updated (C8 + counts) |
| [`docs/tile-streaming-architecture.md`](tile-streaming-architecture.md) | New |
| [`docs/changelog-v0.2.md`](changelog-v0.2.md) | This file |
| [`README.md`](../README.md) | Updated |
