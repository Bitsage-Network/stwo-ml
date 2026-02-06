# stwo-ml

ML inference proving circuits built on [STWO](https://github.com/starkware-libs/stwo) — the fastest STARK prover in the world.

## What This Is

`stwo-ml` adds ML-specific proving circuits on top of STWO's Circle STARK backend:

- **Sumcheck-based MatMul** — Verify matrix multiplication in O(n) instead of O(n³)
- **LogUp Activation Tables** — ReLU, GELU, sigmoid, softmax via lookup proofs
- **Attention Gadget** — Composed sumcheck + lookup for transformer attention
- **ONNX Compiler** — Import models directly from PyTorch/TensorFlow

## Why It's Fast

STWO already has every primitive needed for ML proving (sumcheck, LogUp, MLEs, GKR) with production-grade GPU acceleration. `stwo-ml` wires them together for neural network inference:

| Advantage | Detail |
|-----------|--------|
| M31 field | Single-cycle reduction (2^31-1). 2-4x faster than 256-bit primes. |
| GPU backend | 20,000+ lines of CUDA. 174x FFT speedup. CUDA Graphs. Multi-GPU. |
| Memory residency | Entire proof stays on GPU. One transfer in, one transfer out. |
| Transparent | FRI commitment — no trusted setup ceremony. |
| Native verification | Proofs verify in Cairo on Starknet for 0.31 STRK. |

## Structure

```
stwo-ml/
├── src/
│   ├── components/        # ML AIR components
│   │   ├── matmul.rs      # Sumcheck-based matrix multiplication
│   │   ├── activation.rs  # LogUp-based non-linear operations
│   │   ├── attention.rs   # Composed attention mechanism
│   │   └── layernorm.rs   # Normalization gadget
│   ├── gadgets/           # Reusable constraint gadgets
│   │   ├── range_check.rs # Value bounding for quantized arithmetic
│   │   ├── lookup_table.rs# Precomputed function tables
│   │   └── quantize.rs    # FP32 → INT8 → M31 mapping
│   └── compiler/          # Model → Circuit
│       ├── onnx.rs        # ONNX model import
│       └── graph.rs       # Computation graph builder
└── benches/               # Performance benchmarks
```

## Status

Phase 1 (in progress): MatMul sumcheck, ReLU/GELU lookup tables, basic compiler.

## License

Apache 2.0
