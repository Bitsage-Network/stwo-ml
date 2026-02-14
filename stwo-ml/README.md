# stwo-ml

ML inference proving circuits built on [STWO](https://github.com/starkware-libs/stwo) — StarkWare's Circle STARK prover over M31.

## What This Is

`stwo-ml` adds ML-specific proving circuits on top of STWO's Circle STARK backend:

- **Sumcheck-based MatMul** — Verify matrix multiplication via multilinear extensions (42–1700x trace reduction vs naive encoding)
- **GKR Protocol** — Layer-by-layer interactive proof replaces per-layer independent STARKs with a single pass from output to input
- **SIMD Block Batching** — Prove N identical transformer blocks in one GKR pass with log(N) extra sumcheck rounds per layer
- **GPU Acceleration** — CUDA kernels for sumcheck rounds, fused MLE restrict, GEMM, element-wise ops, and multi-GPU distributed proving
- **LogUp Activation Tables** — ReLU, GELU, Sigmoid, Softmax via lookup proofs with GPU eq-sumcheck
- **Attention (GQA/MQA)** — Grouped Query Attention with composed Q/K/V sumchecks, KV-cache for incremental decoding, and dual-operand 3-factor sumcheck
- **RMSNorm** — Root Mean Square normalization with LogUp rsqrt lookup table (Llama/Qwen pre-norm)
- **RoPE** — Rotary Positional Embedding with precomputed rotation tables and LogUp membership proof
- **LayerNorm** — Combined-product MLE for sound SIMD reduction of non-linear mean/rsqrt operations
- **Transformer Block Builder** — One-call `GraphBuilder::transformer_block()` composing RMSNorm → GQA Attention → Residual → RMSNorm → FFN → Residual
- **Quantized Inference (INT4/INT8)** — Sound 2D LogUp lookup tables for both quantize and dequantize, Poseidon-committed parameters, native packed-INT4 SafeTensors loading
- **ONNX Compiler** — Import models directly from PyTorch/TensorFlow via tract-onnx
- **Dual-Track Execution** — Simultaneous f32 inference and M31 proving for meaningful float output alongside verifiable proofs

## Measured Performance

| Metric | Value | Scope |
|--------|-------|-------|
| GPU prove time | 37.64s | 1 transformer block of Qwen3-14B (H200) |
| Verification | 206ms | CPU |
| Recursive STARK | 46.76s | cairo-prove over ML verifier trace (eliminated by direct verify) |
| Proof size | 17 MB | Constant regardless of model size |
| MatMul trace reduction | 42–255x | Sumcheck vs naive row-by-row |
| GPU FFT speedup | 50–112x | NTT/INTT vs CPU SIMD backend |
| Security | 96-bit | pow_bits=26, n_queries=70, log_blowup=1 |
| On-chain verify | 1 tx | Starknet Sepolia, < 0.31 STRK |

> The 37.64s benchmark covers a single transformer block (1 of 40 in Qwen3-14B), not a full forward pass. Full-model benchmarks are in progress.

## Why It's Fast

| Advantage | Detail |
|-----------|--------|
| M31 field | `p = 2^31 - 1`. Single-cycle reduction. 2–4x faster than 256-bit primes. |
| Circle group | `p + 1 = 2^31` — maximal power-of-two FFT structure via circle group. |
| GPU backend | CUDA kernels for sumcheck rounds, fused MLE restrict, GEMM, element-wise ops. Multi-GPU with device-affine chunk partitioning. |
| GKR protocol | Single interactive proof for entire computation graph vs per-layer STARKs. |
| SIMD batching | N identical transformer blocks proved in 1 pass with log(N) overhead. |
| Sumcheck | Matrix multiply proof in O(log k) rounds instead of O(m·k·n) trace rows. |
| Fused restrict | GPU kernel maps original M31 matrix + Lagrange basis → restricted vector directly. Saves ~1 GB/matrix. |
| Transparent | FRI commitment — no trusted setup ceremony. |
| Native verification | Proofs verify in Cairo on Starknet. |

## Security

The prover/verifier pipeline has undergone a comprehensive security audit (February 2026) covering 24 findings across all severity tiers — all addressed:

| Category | Findings | Key Areas |
|----------|----------|-----------|
| **Critical (C1–C7)** | 7 fixed | Activation STARK soundness, sumcheck Fiat-Shamir binding, domain separation, MLE opening proofs, LayerNorm commitment, softmax normalization, QKV weight binding |
| **High (H1–H5)** | 5 fixed | LayerNorm mean/variance commitment, on-chain softmax STARK, batch lambda commitment, IO commitment binding, weight commitment scope |
| **Medium (M1–M6)** | 6 fixed | Activation type tags in LogUp, LayerClaim type binding, Blake2s proof serialization, recursive MLE openings, PCS config dedup, causal mask propagation |
| **Quantize (Q1–Q6)** | 6 fixed | 1D→2D LogUp relation, forward pass formula, QuantParams commitment, QuantizeLayerData completeness, GQA verifier K/V head splitting, 2D trace building |

See [`docs/security-audit.md`](docs/security-audit.md) for the full audit report with root causes, fixes, and verification details.

## GKR Protocol

The GKR interactive proof engine walks the computation graph from output to input in a single pass, replacing per-layer independent STARK proofs:

```
Output claim → MatMul sumcheck → Activation LogUp → LayerNorm → ... → Input claim
```

Each layer type has a specialized reduction protocol:

| Layer Type | Protocol | Degree | Rounds |
|-----------|----------|--------|--------|
| MatMul | Sumcheck over inner dim | 2 | log(k) |
| MatMul (dual SIMD) | Block-extended 3-factor sumcheck | 3 | log(blocks) + log(k) |
| Add | Linear split | 1 | 0 |
| Mul | Eq-sumcheck | 3 | log(n) |
| Activation | LogUp eq-sumcheck | 3 | log(n) |
| RMSNorm | LogUp rsqrt lookup + eq-sumcheck | 3 | log(n) |
| LayerNorm | Combined-product eq-sumcheck | 3 | log(n) |
| RoPE | LogUp rotation table | 3 | log(n) |
| Dequantize | LogUp 2D lookup (INT4: 16, INT8: 256) | 3 | log(n) |
| Quantize | LogUp 2D lookup (data-dependent table) | 3 | log(n) |
| Attention | 4+2H composed sub-matmuls | 2/3 | varies |

### SIMD Block Batching

For models with repeated identical blocks (transformers), SIMD batching proves N blocks simultaneously:

- **Shared-weight matmuls** (Q/K/V/output projections): combine inputs, standard degree-2 sumcheck
- **Dual-operand matmuls** (per-head Q×K^T, softmax×V): block-extended 3-factor sumcheck with log(N) extra rounds
- **LayerNorm**: combined-product MLE handles non-linear mean/rsqrt

See [`docs/gkr-protocol.md`](docs/gkr-protocol.md) and [`docs/simd-block-batching.md`](docs/simd-block-batching.md) for details.

## Transformer Architecture

Full Llama-style transformer blocks are supported end-to-end:

```
Input → RMSNorm → GQA Attention → +Residual → RMSNorm → FFN (Linear→GELU→Linear) → +Residual → Output
```

Build with a single call:

```rust
let mut builder = GraphBuilder::new((seq_len, d_model));
builder.transformer_block(32, 8, seq_len, 4 * d_model); // 32 Q heads, 8 KV heads (GQA)
let graph = builder.build();
```

Key components:

| Component | Module | Proving Protocol |
|-----------|--------|-----------------|
| RMSNorm | `components/rmsnorm.rs` | LogUp rsqrt lookup table |
| GQA/MQA Attention | `components/attention.rs` | Composed sumcheck + softmax LogUp |
| RoPE | `components/rope.rs` | LogUp rotation table (verifier-reconstructable) |
| KV-Cache | `components/attention.rs` | Incremental K/V storage for autoregressive decoding |
| Dequantize (INT4/INT8) | `components/dequantize.rs` | LogUp 2D table (16 or 256 entries) |
| Quantize (INT4/INT8) | `components/quantize.rs` | LogUp 2D table (data-dependent, Poseidon-committed params) |

See [`docs/transformer-architecture.md`](docs/transformer-architecture.md) for the full block diagram, builder API, component details, and proof structure.

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/transformer-architecture.md`](docs/transformer-architecture.md) | Full transformer block: RMSNorm, GQA/MQA attention, RoPE, KV-cache, builder API |
| [`docs/gkr-protocol.md`](docs/gkr-protocol.md) | GKR protocol: layer types, reduction protocols, proof types, Fiat-Shamir transcript |
| [`docs/gpu-acceleration.md`](docs/gpu-acceleration.md) | GPU pipeline: CUDA kernels, fused MLE restrict, multi-GPU distributed proving |
| [`docs/simd-block-batching.md`](docs/simd-block-batching.md) | SIMD batching: shared-weight vs dual-operand matmuls, attention decomposition, LayerNorm |
| [`docs/security-audit.md`](docs/security-audit.md) | Security audit report (24 findings, all fixed) |
| [`docs/tile-streaming-architecture.md`](docs/tile-streaming-architecture.md) | Tile-level streaming pipeline for memory-bounded proving |
| [`docs/changelog-v0.2.md`](docs/changelog-v0.2.md) | v0.2.0 changelog: GKR, transformer blocks, quantization, multi-GPU |

## Structure

```
stwo-ml/
├── src/
│   ├── components/           # ML AIR components
│   │   ├── matmul.rs         # Sumcheck-based matrix multiplication
│   │   ├── activation.rs     # LogUp-based non-linear operations (ReLU, GELU, Sigmoid, Softmax)
│   │   ├── attention.rs      # GQA/MQA/MHA attention with KV-cache
│   │   ├── rmsnorm.rs        # RMSNorm with LogUp rsqrt lookup
│   │   ├── rope.rs           # Rotary Positional Embedding with LogUp rotation table
│   │   ├── dequantize.rs     # Dequantization LogUp table (INT4/INT8)
│   │   ├── layernorm.rs      # LayerNorm with rsqrt lookup
│   │   ├── embedding.rs      # Token embedding lookup
│   │   ├── elementwise.rs    # Add/Mul pure AIR constraints
│   │   ├── conv2d.rs         # Convolution via im2col + matmul
│   │   ├── tiled_matmul.rs   # Tiled sumcheck for large k
│   │   └── quantize.rs       # 2D LogUp quantization verification
│   │
│   ├── gkr/                  # GKR interactive proof engine
│   │   ├── mod.rs            # Module entry point
│   │   ├── types.rs          # GKRProof, LayerProof, GKRClaim, RoundPolyDeg3
│   │   ├── circuit.rs        # LayeredCircuit compiler from ComputationGraph
│   │   ├── prover.rs         # Layer reductions: matmul, activation, attention, GPU/SIMD
│   │   └── verifier.rs       # Fiat-Shamir transcript replay verifier
│   │
│   ├── compiler/             # Model → Circuit
│   │   ├── graph.rs          # Computation DAG + transformer_block() builder
│   │   ├── onnx.rs           # ONNX model import via tract-onnx
│   │   ├── safetensors.rs    # SafeTensors weight loading (f16/bf16/f32/INT4/INT8)
│   │   ├── quantize_weights.rs # Quantization strategies (Direct, Symmetric8, INT4)
│   │   ├── prove.rs          # Per-layer proof generation
│   │   ├── dual.rs           # f32/M31 dual-track execution
│   │   ├── chunked.rs        # Memory-bounded chunk proving + multi-GPU
│   │   ├── streaming.rs      # mmap-based streaming weight pipeline
│   │   └── checkpoint.rs     # Checkpoint persistence
│   │
│   ├── aggregation.rs        # Unified STARK for non-matmul layers
│   ├── cairo_serde.rs        # Rust → felt252 serialization
│   ├── starknet.rs           # On-chain calldata + direct proof generation
│   ├── gpu.rs                # GPU-accelerated prover dispatch
│   ├── gpu_sumcheck.rs       # CUDA kernels: sumcheck, fused restrict, LogUp, GEMM
│   ├── multi_gpu.rs          # Multi-GPU: device affinity, chunk partitioning, DeviceGuard
│   ├── backend.rs            # GPU/CPU auto-dispatch
│   └── receipt.rs            # Verifiable compute receipts (SVCR)
│
├── docs/                     # Technical documentation
├── benches/                  # Performance benchmarks
└── tests/                    # Integration tests (442 lib + 5 cross-verify + 12 e2e)
```

## Tile-Level Streaming

For large models where even a single weight matrix exceeds available memory, the tile-level streaming pipeline splits the inner dimension `k` into tiles and processes each tile directly from mmap'd SafeTensors shards:

```text
Chunk-level:  load full B (k x n) -> prove -> drop
Tile-level:   load B[0..tile_k, :] -> prove tile 0 -> drop
              load B[tile_k..2*tile_k, :] -> prove tile 1 -> drop
              ...
Peak RAM: 1 tile (tile_k x n) instead of full matrix (k x n)
```

**Double-buffered pipeline**: Both the forward pass and proving path use `std::thread::scope` to load tile N+1 on a background thread while the main thread computes/proves tile N. For the proving path, the ~1-3ms tile load is completely hidden behind the 50-500ms sumcheck — effectively free I/O.

**PrecomputedMatmuls injection**: The aggregation pipeline accepts pre-computed matmul outputs and proofs, eliminating redundant weight re-loading and matmul re-proving. The forward pass uses pre-computed C matrices, Phase 2 (proving) is skipped entirely, and only the STARK for non-matmul components (activations, add, mul, layernorm) is built fresh.

For a 160-matmul transformer block (Qwen3-14B), this eliminates ~44 GB of redundant weight I/O per chunk.

See [`docs/tile-streaming-architecture.md`](docs/tile-streaming-architecture.md) for the full architecture, double-buffered pipeline details, data flow diagrams, and memory analysis.

## Binaries

### prove-model (CLI)

One-command ONNX-to-proof pipeline:

```bash
cargo build --release --bin prove-model --features cli

# Prove a model
prove-model --model model.onnx --output proof.json --gpu

# Inspect model structure
prove-model --model model.onnx --inspect

# HuggingFace SafeTensors
prove-model --model-dir /path/to/hf/model --layers 1 --format cairo_serde --gpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | ONNX model file |
| `--model-dir` | — | HuggingFace model directory |
| `--layers` | all | Limit to first N transformer layers |
| `--input` | random | JSON array of f32 input values |
| `--output` | `proof.json` | Output file |
| `--format` | `cairo_serde` | `cairo_serde` (felt252 hex), `json`, or `direct` |
| `--gpu` | off | GPU acceleration |
| `--security` | `auto` | `auto`, `tee`, or `zk-only` |
| `--inspect` | — | Print model summary and exit |

#### Output Formats

| Format | Description |
|--------|-------------|
| `cairo_serde` | felt252 hex array for `cairo-prove prove-ml` recursive path |
| `json` | Human-readable JSON with proof components |
| `direct` | JSON with `batched_calldata`, `stark_chunks`, `metadata` for `EloVerifier.verify_model_direct()` — eliminates 46.8s Cairo VM recursion (Stage 2) |

### prove-server (REST API)

HTTP server wrapping the proving library. Accepts ONNX models, generates STARK proofs, returns Starknet calldata.

```bash
cargo build --release --bin prove-server --features server

# Start server
BIND_ADDR=0.0.0.0:8080 ./target/release/prove-server

# H200 with GPU + TEE
BIND_ADDR=0.0.0.0:8080 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 \
  ./target/release/prove-server
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status, GPU/TEE detection, loaded models |
| `POST` | `/api/v1/models` | Load ONNX model, compute weight commitment |
| `GET` | `/api/v1/models/{id}` | Get model info |
| `POST` | `/api/v1/prove` | Submit prove job (returns 202 + job_id) |
| `GET` | `/api/v1/prove/{id}` | Poll job status + progress |
| `GET` | `/api/v1/prove/{id}/result` | Get completed proof (calldata, commitments, gas) |

#### Example

```bash
# Load a model
curl -X POST http://localhost:8080/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "/path/to/model.onnx"}'
# -> {"model_id": "0x...", "weight_commitment": "0x...", "num_layers": 40, "input_shape": [1, 5120]}

# Submit proving job
curl -X POST http://localhost:8080/api/v1/prove \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "0x...", "gpu": true}'
# -> 202 {"job_id": "uuid", "status": "queued"}

# Poll until complete
curl http://localhost:8080/api/v1/prove/{job_id}
# -> {"status": "completed", "progress_bps": 10000, "elapsed_secs": 40.5}

# Get result
curl http://localhost:8080/api/v1/prove/{job_id}/result
# -> {"calldata": ["0x..."], "raw_io_data": ["0x...", ...], "estimated_gas": 350000, ...}
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIND_ADDR` | `127.0.0.1:8080` | Server bind address |
| `LD_LIBRARY_PATH` | — | Must include CUDA libs for GPU mode |

## On-Chain Verification

Proofs generated by `stwo-ml` are verified on Starknet via the [EloVerifier contract](../elo-cairo-verifier/):

| Pipeline | Stage 1 | Stage 2 | On-Chain |
|----------|---------|---------|----------|
| **Direct (recommended)** | GPU prove (37.6s) | **Eliminated** | `verify_model_direct()` |
| Recursive | GPU prove (37.6s) | Cairo VM STARK (46.8s) | `verify<CairoAir>` |

```bash
# Direct pipeline (2-stage, no Cairo VM)
prove-model --model model.onnx --format direct --gpu --output proof.json

# Recursive pipeline (3-stage, via Cairo VM)
prove-model --model model.onnx --format cairo_serde --gpu --output args.json
cairo-prove prove-ml stwo_ml_recursive.executable.json args.json
```

**IO Commitment**: All verification paths accept raw I/O data and recompute `Poseidon(raw_io_data)` on-chain — no caller-supplied commitments are trusted. The GKR path additionally evaluates MLEs on-chain to bind the proof to exact inputs and outputs.

**Current contract** (Sepolia): [`0x0068c7...86eb7`](https://sepolia.starkscan.co/contract/0x0068c7023d6edcb1c086bed57e0ce2b3b5dd007f50f0d6beaec3e57427c86eb7)

## Feature Flags

| Flag | Enables | Requires |
|------|---------|----------|
| `std` (default) | Standard library + STWO prover | — |
| `gpu` | GPU kernel source | — |
| `cuda-runtime` | Full CUDA + GPU sumcheck kernels | CUDA 12.4+ |
| `multi-gpu` | Multi-GPU proving | `cuda-runtime` |
| `tee` | NVIDIA Confidential Computing | `cuda-runtime` + H100+ |
| `onnx` | ONNX model import | — |
| `safetensors` | SafeTensors weight loading | — |
| `model-loading` | ONNX + SafeTensors | — |
| `cli` | `prove-model` binary | `model-loading` |
| `server` | `prove-server` HTTP API | `model-loading` |

## License

Apache 2.0
