<div align="center">

# STWO-ML

**ML Inference Proving on Circle STARKs**

[![Starknet](https://img.shields.io/badge/Starknet-Sepolia-blue?style=for-the-badge)](https://sepolia.starkscan.co)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-H100-green?style=for-the-badge)](https://www.nvidia.com/en-us/data-center/h100/)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange?style=for-the-badge)](LICENSE)

GPU-accelerated Circle STARK prover with ML inference circuits.
Fork of [StarkWare's STWO](https://github.com/starkware-libs/stwo) extended for verifiable AI.

[Paper](https://eprint.iacr.org/2024/278) | [Benchmarks](https://starkware-libs.github.io/stwo/dev/bench/index.html) | [BitSage Network](https://bitsage.network)

</div>

---

## What This Is

STWO-ML combines two things:

1. **STWO** — The fastest STARK prover in the world (Circle STARKs over M31, GPU-accelerated)
2. **stwo-ml** — ML-specific proving circuits built on top (sumcheck matmul, LogUp activations, attention)

Together they enable **proof of inference**: cryptographic proof that a specific AI model produced a specific output, verified on Starknet L2.

## Why STWO for ML

| Property | STWO | Other zkML Systems |
|----------|------|--------------------|
| **Field** | M31 (2^31-1), single-cycle reduction | 256-bit primes, multi-cycle |
| **MatMul** | Sumcheck: O(n) verifier | Naive trace: O(n³) rows |
| **GPU** | 20,000+ lines CUDA, memory-resident | CPU-only or partial GPU |
| **Setup** | Transparent (FRI) — no ceremony | Trusted setup required |
| **Verification** | On-chain on Starknet L2 | Off-chain or expensive L1 |
| **Projected speed** | 10-50x faster than zkLLM | 15 min for 13B params |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  STWO-ML Repository                                              │
│                                                                   │
│  crates/stwo-ml/          ML inference circuits (NEW)            │
│  ├── components/          Sumcheck matmul, LogUp activations     │
│  ├── gadgets/             Range checks, quantization, lookups    │
│  └── compiler/            ONNX import, computation graph         │
│                                                                   │
│  crates/stwo/             Core STARK prover (upstream fork)      │
│  ├── backend/cpu/         SIMD-optimized CPU proving             │
│  ├── backend/gpu/         CUDA GPU proving (20K+ lines)          │
│  ├── fri.rs               FRI commitment (GPU-resident)          │
│  └── prover.rs            Circle STARK prove/verify              │
│                                                                   │
│  crates/constraint-framework/   LogUp, Sumcheck, GKR             │
│  crates/air-utils/              AIR helper macros                 │
│  crates/examples/               Demo provers                     │
├──────────────────────────────────────────────────────────────────┤
│  Starknet L2              On-chain proof verification            │
│  └── StwoVerifier         0.31 STRK per verification             │
└──────────────────────────────────────────────────────────────────┘
```

## Proof Pipeline

```
Model (ONNX) ──► stwo-ml compiler ──► Circuit (AIR) ──► STWO Prover (GPU) ──► Proof ──► Starknet L2
                                                              │
                                                     CUDA kernels:
                                                     Circle FFT, FRI,
                                                     Merkle, Sumcheck
```

Each proof submission on Starknet generates 12 events through a 4-call multicall:

```
__execute__
├── ProofGatedPayment.register_job_payment
├── StwoVerifier.submit_and_verify_with_io_binding
├── PaymentRouter.register_job
└── PaymentRouter.pay_with_sage
    ├── Worker   80%
    ├── Treasury 18%
    └── Stakers   2%
```

## Benchmarks

**Hardware:** NVIDIA H100 PCIe 80GB | **Network:** Starknet Sepolia

### Proof Generation

| Workload | Trace Steps | GPU Time | On-Chain Cost |
|----------|-------------|----------|---------------|
| ML Inference (small) | 132 | 21ms | 0.31 STRK |
| 64K Steps | 65,536 | 159ms | 0.36 STRK |
| 256K Steps | 262,144 | 335ms | 0.39 STRK |
| 1M Steps | 1,048,576 | 1,107ms | 0.43 STRK |

### ML Trace Reduction (stwo-ml)

| Operation | Naive Rows | Sumcheck Rows | Reduction |
|-----------|-----------|---------------|-----------|
| 128x128 MatMul | 2,097,152 | 49,152 | 42x |
| 768x768 MatMul (BERT) | 452,984,832 | 1,769,472 | 255x |
| BERT-base Attention | 200M+ | 11.7M | 17x |

### Verified Transactions (Sepolia)

| Workload | Transaction |
|----------|-------------|
| ML GPU | [`0x06854...cf53`](https://sepolia.voyager.online/tx/0x068545dbe5b18a52328b0c0b74a661c6f0f7f689d4847247b055bd217a46cf53) |
| ML CPU | [`0x051ee...1eac`](https://sepolia.voyager.online/tx/0x051ee2466af84d94b439fae15bcb1662317a4a7116ee3e7ccb3a3f07ae731eac) |
| GPU 64K | [`0x03cc2...bfb1`](https://sepolia.voyager.online/tx/0x03cc26baf34abbed4c753ce60e53854d8728723a73acc3f7fa9f687fc6f9bfb1) |
| GPU 256K | [`0x0384d...4607`](https://sepolia.voyager.online/tx/0x0384d3daa5f08e083115c228b91d19a2a79d3d73117eb57f666f9ec8b3574607) |
| GPU 1M | [`0x05d0a...8ec9`](https://sepolia.voyager.online/tx/0x05d0ae5280523e1ec31802a8aa7ffec28eea943c498d7b1694a495087557eec9) |

## Deployed Contracts (Starknet Sepolia)

| Contract | Address |
|----------|---------|
| StwoVerifier | `0x575968af96f814da648442daf1b8a09d43b650c06986e17b2bab7719418ddfb` |
| ProofGatedPayment | `0x7e74d191b1cca7cac00adc03bc64eaa6236b81001f50c61d1d70ec4bfde8af0` |
| PaymentRouter | `0x001a7c5974eaa8a4d8c145765e507f73d56ee1d05419cbcffcae79ed3cd50f4d` |
| SAGE Token | `0x072349097c8a802e7f66dc96b95aca84e4d78ddad22014904076c76293a99850` |

## Building

### Prerequisites

- Rust nightly (see `rust-toolchain.toml`)
- CUDA Toolkit 12.x (for GPU features)
- NVIDIA Driver 535+

### Build

```bash
# CPU only
cargo build --release

# With GPU acceleration
cargo build --release --features cuda

# ML circuits only
cargo build --release -p stwo-ml
```

### Test

```bash
# All tests
cargo test

# ML circuits
cargo test -p stwo-ml

# Benchmarks
cargo bench -p stwo-ml
```

## Upstream Sync

We track StarkWare's upstream STWO and merge regularly:

```bash
git fetch upstream
git merge upstream/dev
```

Current base: **STWO v2.0.0** (synced Feb 2026)

Our additions live in `crates/stwo-ml/` and GPU integration patches in `crates/stwo/src/core/backend/gpu/`.

## Related Repositories

| Repository | Description |
|------------|-------------|
| [Bitsage-Network/rust-node](https://github.com/Bitsage-Network/rust-node) | GPU Worker Node (ObelyskVM + proof pipeline) |
| [Bitsage-Network/BitSage-Cairo-Smart-Contracts](https://github.com/Bitsage-Network/BitSage-Cairo-Smart-Contracts) | Starknet contracts (StwoVerifier, PaymentRouter) |
| [Bitsage-Network/obelysk](https://github.com/Bitsage-Network/obelysk) | Privacy protocol (shielded swaps, confidential transfers) |
| [starkware-libs/stwo](https://github.com/starkware-libs/stwo) | Upstream STWO prover |

## License

Apache 2.0 — Same as upstream STWO.
