<p align="center">
  <img src="https://raw.githubusercontent.com/Bitsage-Network/Obelysk-Protocol/main/apps/web/public/obelysk-logo.png" alt="Obelysk" width="180"/>
</p>

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ██╗                    ║
║    ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝██╔════╝██║ ██╔╝                    ║
║    ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝ ███████╗█████╔╝                     ║
║    ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝  ╚════██║██╔═██╗                     ║
║    ╚██████╔╝██████╔╝███████╗███████╗██║   ███████║██║  ██╗                    ║
║     ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝                    ║
║                                                                               ║
║        ███████╗████████╗██╗    ██╗ ██████╗     ███╗   ███╗██╗                 ║
║        ██╔════╝╚══██╔══╝██║    ██║██╔═══██╗    ████╗ ████║██║                 ║
║        ███████╗   ██║   ██║ █╗ ██║██║   ██║    ██╔████╔██║██║                 ║
║        ╚════██║   ██║   ██║███╗██║██║   ██║    ██║╚██╔╝██║██║                 ║
║        ███████║   ██║   ╚███╔███╔╝╚██████╔╝    ██║ ╚═╝ ██║███████╗           ║
║        ╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝     ╚═╝     ╚═╝╚══════╝           ║
║                                                                               ║
║          GPU-Accelerated ZK Proofs for Verifiable AI on Starknet              ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

<p align="center">
  <a href="https://github.com/Bitsage-Network/stwo-ml/stargazers"><img src="https://img.shields.io/github/stars/Bitsage-Network/stwo-ml?style=for-the-badge&color=yellow" alt="Stars"></a>
  <a href="https://github.com/Bitsage-Network/stwo-ml/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/starknet-sepolia-purple?style=for-the-badge" alt="Starknet">
  <img src="https://img.shields.io/badge/CUDA-12.4+-green?style=for-the-badge&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/rust-nightly-orange?style=for-the-badge&logo=rust" alt="Rust">
</p>

<p align="center">
  <strong>Prove Qwen3-14B inference in 40 seconds. Verify on-chain for < $0.01.</strong>
</p>

---

## What is STWO ML?

STWO ML is the proving engine behind [Obelysk Protocol](https://github.com/Bitsage-Network/Obelysk-Protocol) — a GPU-accelerated system that generates zero-knowledge proofs of ML inference and settles them on Starknet in a single transaction.

It extends [StarkWare's STWO](https://github.com/starkware-libs/stwo) (the fastest STARK prover) with **20,000+ lines of CUDA**, ML-specific circuits, and an end-to-end pipeline from model weights to on-chain verification.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   "Did this AI actually run Qwen3-14B on my input?"                         │
│                                                                             │
│    stwo-ml proves it did — with math, not trust.                            │
│                                                                             │
│    Anyone can verify. Nobody can fake it. Settled on Starknet.              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Numbers

```
╔══════════════════════════════╦══════════════════════════════════════════════╗
║  METRIC                      ║  VALUE                                      ║
╠══════════════════════════════╬══════════════════════════════════════════════╣
║  Qwen3-14B Prove Time        ║  40.52 seconds (NVIDIA H200)                ║
║  On-Chain Verify Cost         ║  < 0.31 STRK (< $0.01)                     ║
║  Proof Compression            ║  17 MB → ~1 KB (recursive STARK)           ║
║  MatMul Trace Reduction       ║  42-255x (sumcheck protocol)               ║
║  GPU FFT Speedup              ║  174x vs CPU SIMD                          ║
║  Security Level               ║  96-bit (configurable)                     ║
║  Trusted Setup                ║  NONE (FRI-based, fully transparent)       ║
║  GPU Backend                  ║  20,000+ lines CUDA (H100/H200/B200)      ║
╚══════════════════════════════╩══════════════════════════════════════════════╝
```

---

## Quick Start

### One-Command Install

```bash
# Clone the repo
git clone https://github.com/Bitsage-Network/stwo-ml.git && cd stwo-ml

# Build everything (CPU mode — works on any machine)
cargo build --release -p stwo-ml

# Run tests to verify
cargo test --lib -p stwo-ml
```

### GPU Mode (NVIDIA H100/H200/B200)

```bash
# Requires: CUDA 12.4+, NVIDIA driver 550+
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Build with GPU acceleration
cargo build --release -p stwo-ml --features cuda-runtime

# Prove ML inference
cargo run --release -p prove-qwen --features cuda-runtime -- \
  --model qwen3-14b \
  --input "What is zero-knowledge?" \
  --output proof.json \
  --gpu
```

### Full Pipeline (Prove + Verify On-Chain)

```bash
# Step 1: Prove ML inference on GPU (40s)
prove-qwen --model qwen3-14b --input "Hello" --output ml_proof.json --gpu

# Step 2: Compress to recursive proof (~1 KB)
cairo-prove prove-ml \
  --verifier-executable ml_verifier.executable.json \
  --ml-proof ml_proof.json \
  --output recursive.json

# Step 3: Submit to Starknet (1 transaction, 7 events)
starkli invoke $OBELYSK_VERIFIER verify_and_pay \
  $MODEL_ID $PROOF_HASH $IO_COMMITMENT $WEIGHT_COMMITMENT \
  $NUM_LAYERS $JOB_ID $WORKER $SAGE_AMOUNT

# Step 4: Anyone can verify
starkli call $OBELYSK_VERIFIER is_verified $PROOF_ID
# → 0x1 (TRUE)
```

---

## How It Works

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         STWO ML — PROVING PIPELINE                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║   STEP 1: GPU PROVING                                                       ║
║   ══════════════════                                                        ║
║                                                                             ║
║   Model Weights ─────→ stwo-ml (CUDA) ─────→ STARK Proofs                  ║
║   (SafeTensors)            │                                                ║
║                            ├─ MatMul ──→ Sumcheck Protocol (O(m+k+n))      ║
║                            │             42-255x trace reduction             ║
║                            │                                                ║
║                            ├─ Activation → LogUp Lookup Tables              ║
║                            │               ReLU / GELU / Sigmoid            ║
║                            │                                                ║
║                            ├─ Attention ─→ Composed Q/K/V Sumchecks        ║
║                            │               Multi-head with softmax          ║
║                            │                                                ║
║                            └─ LayerNorm ─→ Mean/Variance Constraints       ║
║                                                                             ║
║   Output: AggregatedModelProof (~17 MB, 4 sumcheck + 1 activation STARK)   ║
║                                                                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                    │                                        ║
║                                    ▼                                        ║
║   STEP 2: RECURSIVE COMPRESSION                                            ║
║   ══════════════════════════════                                            ║
║                                                                             ║
║   ML Proof ──→ Cairo ML Verifier ──→ cairo-prove ──→ Recursive STARK       ║
║   (17 MB)      (obelysk_ml_air)      (proves the       (~1 KB)             ║
║                 Sumcheck + MLE         verifier's                           ║
║                 verification           execution)                           ║
║                                                                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                    │                                        ║
║                                    ▼                                        ║
║   STEP 3: ON-CHAIN SETTLEMENT                                              ║
║   ════════════════════════════                                              ║
║                                                                             ║
║   Recursive Proof ──→ ObelyskVerifier (Starknet) ──→ VERIFIED              ║
║   (~1 KB)              │                                                    ║
║                        ├─ Verify proof fact                                 ║
║                        ├─ Transfer SAGE payment (client → GPU worker)       ║
║                        ├─ Record on-chain (proof_id → verified)             ║
║                        └─ Emit 7 events for indexers & frontends            ║
║                                                                             ║
║   Single TX. Atomic. Trustless.                                             ║
║                                                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Sumcheck MatMul — The Core Innovation

Proving matrix multiplication `C = A @ B` the naive way requires materializing the entire O(m x k x n) computation as constraint rows. STWO ML uses the **sumcheck protocol** to reduce this to O(m + k + n):

```
                        SUMCHECK PROTOCOL FOR MATMUL

  Traditional:    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + ... + A[i,k]*B[k,j]
                  → Must prove ALL m×k×n multiplications in trace
                  → 128×128: 2,097,152 constraint rows

  Sumcheck:       Encode A, B, C as multilinear extensions (MLEs)
                  → log(k) rounds of interaction
                  → Each round: prover sends p(X) = c₀ + c₁X + c₂X²
                  → Verifier checks: p(0) + p(1) = claimed_sum
                  → Verifier draws random challenge r
                  → Final: single MLE opening check
                  → 128×128: 49,152 rows (42x reduction)
                  → 256×256: ~768 rows (255x reduction)
```

### LogUp for Activations

Non-linear functions can't be arithmetized efficiently. Instead, STWO ML precomputes lookup tables and uses the **LogUp protocol** to prove every activation output exists in the table:

```
  ┌──────────────────┐     ┌──────────────────────────────┐
  │ Activation Input  │────→│ Lookup Table (precomputed)    │
  │ x = 1.5          │     │ ReLU(0.0) = 0.0              │
  │                   │     │ ReLU(0.1) = 0.1              │
  │ Output: 1.5       │     │ ...                          │
  │ (ReLU(1.5) = 1.5) │←───│ ReLU(1.5) = 1.5  ← MATCH    │
  └──────────────────┘     │ ...                          │
                           └──────────────────────────────┘

  LogUp proves the lookup without revealing WHICH entry was used.
  Supports: ReLU, GELU, Sigmoid, Softmax
```

### Recursive Compression

The ML verifier is itself a Cairo program. STWO proves its execution, producing a proof-of-the-verifier — a compact recursive STARK:

```
  17 MB ML Proof ──→ Cairo VM (verifies proof) ──→ Execution Trace ──→ 1 KB STARK
                     └─ obelysk_ml_air:                                    │
                        verify_matmul_sumcheck()                           │
                        verify_mle_opening()                               │
                        mix_fiat_shamir_channel()                          ▼
                                                                    Fits in 1 TX
```

---

## Repository Structure

```
stwo-ml/
│
├── stwo/                              STWO CORE — StarkWare's Prover + Our GPU Backend
│   └── crates/stwo/
│       └── src/prover/backend/
│           ├── simd/                  StarkWare's CPU backend
│           ├── cpu/                   StarkWare's baseline
│           └── gpu/                   ████ OURS ████  20,000+ lines CUDA
│               ├── fft.rs             Circle FFT kernels (M31 field)
│               ├── fri.rs             FRI commitment with GPU residency
│               ├── merkle.rs          Blake2s GPU hashing
│               ├── quotients.rs       Quotient accumulation
│               ├── gkr.rs             MLE fold kernels
│               ├── cuda_executor.rs   NVRTC compilation, multi-GPU
│               ├── pipeline.rs        H2D → compute → D2H pipeline
│               └── tee/               NVIDIA Confidential Computing
│
├── stwo-ml/                           ML PROVING LIBRARY — 11,000+ lines Rust
│   └── src/
│       ├── components/
│       │   ├── matmul.rs              Sumcheck-based matrix multiplication (1,224 loc)
│       │   ├── activation.rs          LogUp activation tables (255 loc)
│       │   ├── attention.rs           Multi-head attention proofs (1,120 loc)
│       │   ├── layernorm.rs           Layer normalization (182 loc)
│       │   └── f32_ops.rs             Dual-track float verification (338 loc)
│       │
│       ├── compiler/
│       │   ├── onnx.rs                ONNX model import (711 loc)
│       │   ├── graph.rs               Computation graph builder (411 loc)
│       │   ├── prove.rs               Proving pipeline (913 loc)
│       │   ├── safetensors.rs         Weight loading (471 loc)
│       │   ├── dual.rs                F32 dual-track verification (466 loc)
│       │   └── inspect.rs             Model introspection (224 loc)
│       │
│       ├── aggregation.rs             Proof composition into single STARK (916 loc)
│       ├── cairo_serde.rs             Rust → felt252 serialization bridge (776 loc)
│       ├── starknet.rs                On-chain calldata generation (417 loc)
│       ├── backend.rs                 GPU/CPU auto-dispatch (310 loc)
│       ├── gpu.rs                     GPU-accelerated pipeline (403 loc)
│       ├── tee.rs                     TEE attestation integration (274 loc)
│       ├── receipt.rs                 Verifiable Compute Receipts (632 loc)
│       └── crypto/
│           ├── poseidon_channel.rs    Fiat-Shamir channel (249 loc)
│           ├── mle_opening.rs         MLE commitments & proofs (352 loc)
│           └── poseidon_merkle.rs     On-chain compatible Merkle (176 loc)
│
├── stwo-cairo/                        CAIRO PROVING & RECURSIVE VERIFICATION
│   ├── cairo-prove/                   CLI: prove | verify | prove-ml
│   ├── stwo_cairo_prover/             Rust prover for Cairo VM traces
│   └── stwo_cairo_verifier/           Cairo verifier workspace
│       └── crates/
│           ├── ml_air/                ML-specific AIR (sumcheck, MLE, LogUp)
│           ├── ml_verifier/           Cairo #[executable] for recursive proofs
│           ├── verifier_core/         Generic STARK verifier (FRI, Merkle)
│           └── constraint_framework/  Constraint evaluation primitives
│
├── stwo-ml-verifier/                  ON-CHAIN CONTRACT (Cairo/Starknet)
│   └── src/
│       ├── contract.cairo             ObelyskVerifier: verify_and_pay, 7 events
│       └── interfaces.cairo           IObelyskVerifier, IERC20 dispatcher
│
└── docs/                              Documentation & technical specs
```

---

## Building & Testing

### Prerequisites

| Tool | Version | Required For |
|------|---------|-------------|
| Rust | nightly 1.88+ | Core library |
| CUDA Toolkit | 12.4+ | GPU proving |
| Scarb | 2.12+ | Cairo contracts |
| starkli | latest | Starknet deployment |

### Build Commands

```bash
# ─── CPU Mode (works everywhere) ──────────────────────────────────
cargo build --release -p stwo-ml

# ─── GPU Mode (NVIDIA H100/H200/B200) ────────────────────────────
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
cargo build --release -p stwo-ml --features cuda-runtime

# ─── Cairo ML Verifier ────────────────────────────────────────────
cd stwo-cairo/stwo_cairo_verifier && scarb build

# ─── ObelyskVerifier Contract ─────────────────────────────────────
cd stwo-ml-verifier && scarb build
```

### Test Suite

```bash
# ─── All library tests (177 tests, ~2 min) ────────────────────────
cargo test --lib -p stwo-ml

# ─── Cairo sumcheck verifier (5 tests) ────────────────────────────
cd stwo-cairo/stwo_cairo_verifier && scarb test -p obelysk_ml_air

# ─── GPU integration tests (requires NVIDIA GPU) ──────────────────
cargo test --test gpu_pipeline -p stwo-ml --features cuda-runtime

# ─── Scale tests (512x512 → 2048x2048 matmul) ────────────────────
cargo test --test scale_matmul -p stwo-ml --release
```

```
╔════════════════════════╦════════╦══════════════════════════════════════════╗
║  Test Suite             ║ Count  ║ Coverage                                ║
╠════════════════════════╬════════╬══════════════════════════════════════════╣
║  stwo-ml (lib)          ║  177   ║ MatMul, activations, attention, ONNX,   ║
║                         ║        ║ graph, aggregation, crypto, serialization║
╠════════════════════════╬════════╬══════════════════════════════════════════╣
║  stwo-ml (integration)  ║   32   ║ GPU pipeline, scale testing (2048x2048) ║
╠════════════════════════╬════════╬══════════════════════════════════════════╣
║  obelysk_ml_air (Cairo) ║    5   ║ Sumcheck verifier, round polynomials    ║
╠════════════════════════╬════════╬══════════════════════════════════════════╣
║  cairo_serde            ║   14   ║ Rust→felt252 roundtrips                 ║
╠════════════════════════╬════════╬══════════════════════════════════════════╣
║  ObelyskVerifier        ║   --   ║ Compiles, produces contract artifacts   ║
╚════════════════════════╩════════╩══════════════════════════════════════════╝
```

---

## Feature Flags

```
╔══════════════════╦══════════════════════════════════════╦═══════════════════════╗
║  Flag             ║  What It Does                        ║  Requires             ║
╠══════════════════╬══════════════════════════════════════╬═══════════════════════╣
║  std (default)    ║  Standard library + STWO prover      ║  —                    ║
║  gpu              ║  GPU kernel sources (no runtime)     ║  —                    ║
║  cuda-runtime     ║  Full CUDA GPU execution             ║  CUDA 12.4+           ║
║  multi-gpu        ║  Distributed multi-GPU proving       ║  cuda-runtime         ║
║  tee              ║  NVIDIA Confidential Computing       ║  cuda-runtime + H100+ ║
║  onnx             ║  ONNX model import via tract-onnx    ║  —                    ║
║  safetensors      ║  SafeTensors weight loading          ║  —                    ║
║  model-loading    ║  Both ONNX + SafeTensors             ║  —                    ║
╚══════════════════╩══════════════════════════════════════╩═══════════════════════╝
```

---

## Performance

```
╔══════════════════════════════╦════════════╦══════════════════════════════════╗
║  Operation                    ║  Time      ║  Hardware                        ║
╠══════════════════════════════╬════════════╬══════════════════════════════════╣
║  128×128 MatMul prove         ║  11 ms     ║  Apple M3 (release)              ║
║  512×512 MatMul prove         ║  11 ms     ║  Apple M3 (release)              ║
║  2048×2048 MatMul prove       ║  189 ms    ║  Apple M3 (496 MB memory)        ║
║  Qwen3-14B full prove         ║  40.52 s   ║  NVIDIA H200 (GPU)               ║
║  Qwen3-14B verify             ║  209 ms    ║  CPU                             ║
║  Recursive compression        ║  ~30 s     ║  NVIDIA H200 (GPU)               ║
║  On-chain verification        ║  1 TX      ║  Starknet Sepolia                ║
╠══════════════════════════════╬════════════╬══════════════════════════════════╣
║  GPU FFT speedup              ║  174x      ║  vs CPU SIMD backend             ║
║  MatMul trace reduction       ║  42-255x   ║  vs naive constraint approach    ║
║  Proof compression            ║  17000x    ║  17 MB → ~1 KB (recursive)       ║
╚══════════════════════════════╩════════════╩══════════════════════════════════╝
```

---

## Deployed Contracts (Starknet Sepolia)

| Contract | Address | Explorer |
|----------|---------|---------|
| **ObelyskVerifier v3** | `0x04f8c537...a965a15` | [View on Voyager](https://sepolia.voyager.online/contract/0x04f8c5377d94baa15291832dc3821c2fc235a95f0823f86add32f828ea965a15) |
| **SAGE Token** | `0x0723490...a99850` | [View on Voyager](https://sepolia.voyager.online/contract/0x072349097c8a802e7f66dc96b95aca84e4d78ddad22014904076c76293a99850) |
| **StweMlStarkVerifier** | `0x005928a...74fba` | [View on Voyager](https://sepolia.voyager.online/contract/0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba) |

---

## Security

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  96-bit security (configurable)                                 │
  │  No trusted setup — FRI-based, fully transparent                │
  │  Fiat-Shamir via Poseidon (matches Cairo verifier exactly)      │
  │  TEE option — model weights never leave encrypted GPU memory    │
  │  NVIDIA Confidential Computing (CC-On) on H100/H200/B200       │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork → Branch → Code → Test → PR
git checkout -b feat/your-feature
cargo test --lib -p stwo-ml
# Open PR against main
```

---

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

Built on [STWO](https://github.com/starkware-libs/stwo) by [StarkWare](https://starkware.co).

---

<p align="center">
  <img src="https://raw.githubusercontent.com/Bitsage-Network/Obelysk-Protocol/main/apps/web/public/obelysk-logo.png" alt="Obelysk" width="60"/>
</p>

<p align="center">
  <strong><a href="https://github.com/Bitsage-Network">Bitsage Network</a></strong> — Verifiable AI on Starknet
</p>

<p align="center">
  <a href="https://github.com/Bitsage-Network/Obelysk-Protocol">Obelysk Protocol</a> · <a href="https://github.com/Bitsage-Network/stwo-ml">STWO ML</a> · <a href="https://github.com/Bitsage-Network/stwo-ml/tree/main/docs">Docs</a>
</p>
