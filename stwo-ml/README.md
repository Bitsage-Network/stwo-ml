# stwo-ml

ML inference proving circuits built on [STWO](https://github.com/starkware-libs/stwo) — StarkWare's Circle STARK prover over M31.

## What This Is

`stwo-ml` adds ML-specific proving circuits on top of STWO's Circle STARK backend:

- **Sumcheck-based MatMul** — Verify matrix multiplication via multilinear extensions (42–1700x trace reduction vs naive encoding)
- **LogUp Activation Tables** — ReLU, GELU, Sigmoid, Softmax via lookup proofs
- **Attention Component** — Composed Q/K/V sumchecks + softmax LogUp for multi-head attention
- **LayerNorm** — Mean/variance/rsqrt constraints with lookup verification
- **ONNX Compiler** — Import models directly from PyTorch/TensorFlow via tract-onnx
- **Dual-Track Execution** — Simultaneous f32 inference and M31 proving for meaningful float output alongside verifiable proofs

## Measured Performance

| Metric | Value | Scope |
|--------|-------|-------|
| GPU prove time | 37.64s | 1 transformer block of Qwen3-14B (H200) |
| Verification | 206ms | CPU |
| Recursive STARK | 46.76s | cairo-prove over ML verifier trace |
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
| GPU backend | 20,000+ lines CUDA. GPU residency, CUDA Graphs, multi-GPU. |
| Sumcheck | Matrix multiply proof in O(log k) rounds instead of O(m·k·n) trace rows. |
| Transparent | FRI commitment — no trusted setup ceremony. |
| Native verification | Proofs verify in Cairo on Starknet. |

## Structure

```
stwo-ml/
├── src/
│   ├── components/           # ML AIR components
│   │   ├── matmul.rs         # Sumcheck-based matrix multiplication
│   │   ├── activation.rs     # LogUp-based non-linear operations
│   │   ├── attention.rs      # Composed attention mechanism
│   │   ├── layernorm.rs      # Normalization with rsqrt lookup
│   │   ├── embedding.rs      # Token embedding lookup
│   │   ├── elementwise.rs    # Add/Mul pure AIR constraints
│   │   ├── conv2d.rs         # Convolution via im2col + matmul
│   │   ├── tiled_matmul.rs   # Tiled sumcheck for large k
│   │   └── quantize.rs       # Range-check quantization
│   │
│   ├── compiler/             # Model → Circuit
│   │   ├── onnx.rs           # ONNX model import via tract-onnx
│   │   ├── graph.rs          # Computation DAG (topological sort)
│   │   ├── prove.rs          # Per-layer proof generation
│   │   ├── dual.rs           # f32/M31 dual-track execution
│   │   ├── chunked.rs        # Memory-bounded chunk proving
│   │   └── checkpoint.rs     # Checkpoint persistence
│   │
│   ├── aggregation.rs        # Unified STARK for non-matmul layers
│   ├── cairo_serde.rs        # Rust → felt252 serialization
│   ├── starknet.rs           # On-chain calldata generation
│   ├── gpu.rs                # GPU-accelerated prover dispatch
│   ├── gpu_sumcheck.rs       # CUDA sumcheck round kernels
│   ├── backend.rs            # GPU/CPU auto-dispatch
│   └── receipt.rs            # Verifiable compute receipts (SVCR)
│
├── benches/                  # Performance benchmarks
└── tests/                    # Integration tests
```

## License

Apache 2.0
