<p align="center">
  <img src="https://raw.githubusercontent.com/Bitsage-Network/stwo-ml/main/resources/img/stwo-logo.png" alt="STWO ML" width="200"/>
</p>

<h1 align="center">STWO ML</h1>

<p align="center">
  <strong>GPU-accelerated STARK prover for verifiable ML inference on Starknet</strong>
</p>

<p align="center">
  <a href="https://github.com/Bitsage-Network/stwo-ml/actions"><img src="https://img.shields.io/github/actions/workflow/status/Bitsage-Network/stwo-ml/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://github.com/Bitsage-Network/stwo-ml/stargazers"><img src="https://img.shields.io/github/stars/Bitsage-Network/stwo-ml" alt="Stars"></a>
  <a href="https://github.com/Bitsage-Network/stwo-ml/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

---

STWO ML is a production-grade system for proving ML inference with zero-knowledge proofs and verifying results on Starknet. It extends [StarkWare's STWO prover](https://github.com/starkware-libs/stwo) with ML-specific circuits, 20,000+ lines of CUDA acceleration, and an end-to-end pipeline from model weights to on-chain verification.

**Prove a Qwen3-14B inference in 40 seconds. Verify on-chain for < $0.01.**

## Key Results

| Metric | Value |
|--------|-------|
| Qwen3-14B prove time (H200 GPU) | 40.52s |
| On-chain verification cost | < 0.31 STRK |
| Recursive proof size | ~1 KB (from 17 MB raw) |
| MatMul trace reduction | 42-255x (via sumcheck) |
| GPU FFT speedup | 174x vs CPU |
| Security | 96-bit (configurable) |

## Architecture

```
                          STWO ML — End-to-End Pipeline

 ┌─────────────────────────────────────────────────────────────────────┐
 │                         1. PROVE (GPU)                              │
 │                                                                     │
 │   Model Weights ──→ stwo-ml ──→ Per-layer STARK proofs             │
 │   (SafeTensors)      │                                              │
 │                      ├── MatMul: Sumcheck protocol (O(m+k+n))       │
 │                      ├── Activations: LogUp lookup tables           │
 │                      ├── Attention: Composed Q/K/V sumchecks        │
 │                      └── LayerNorm: Mean/variance constraints       │
 │                                                                     │
 │   Output: AggregatedModelProof (~17 MB)                             │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────────────┐
 │                      2. RECURSIVE COMPRESS                          │
 │                                                                     │
 │   ML proof ──→ Cairo ML Verifier ──→ cairo-prove ──→ Recursive proof│
 │                (obelysk_ml_air)       (STARK of          (~1 KB)    │
 │                                        the verifier)                │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────────────┐
 │                    3. ON-CHAIN VERIFY                                │
 │                                                                     │
 │   Recursive proof ──→ ObelyskVerifier ──→ 7 events + SAGE payment  │
 │                       (Starknet contract)                           │
 │                                                                     │
 │   Single TX: verify proof + transfer SAGE + emit rich events        │
 └─────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
stwo-ml/
├── stwo/                    # Fork of StarkWare's STWO prover
│   └── crates/stwo/         #   Circle STARK core + 20K lines GPU/CUDA
│
├── stwo-ml/                 # ML inference proving library (Rust)
│   └── src/
│       ├── components/      #   MatMul, Activation, Attention, LayerNorm, F32
│       ├── compiler/        #   ONNX import, graph builder, proving pipeline
│       ├── aggregation.rs   #   Compose layer proofs into single STARK
│       ├── cairo_serde.rs   #   Rust → felt252 serialization for Cairo
│       ├── starknet.rs      #   On-chain calldata generation
│       ├── backend.rs       #   GPU/CPU backend selection
│       ├── gpu.rs           #   GPU-accelerated proof pipeline
│       ├── tee.rs           #   NVIDIA TEE (confidential GPU) integration
│       ├── receipt.rs       #   Streaming Verifiable Compute Receipts
│       ├── crypto/          #   Poseidon channel, MLE commitments, Merkle
│       └── gadgets/         #   Lookup tables, quantization, range checks
│
├── stwo-cairo/              # Cairo proving & verification
│   ├── cairo-prove/         #   CLI: prove, verify, prove-ml subcommands
│   ├── stwo_cairo_prover/   #   Rust prover for Cairo VM execution traces
│   └── stwo_cairo_verifier/ #   Cairo verifier workspace
│       └── crates/
│           ├── ml_air/      #     ML-specific AIR (sumcheck, MLE, LogUp)
│           ├── ml_verifier/ #     Cairo executable for recursive proving
│           ├── verifier_core/     STARK verifier primitives
│           └── ...          #     constraint_framework, utils, etc.
│
├── stwo-ml-verifier/        # On-chain ObelyskVerifier contract (Cairo)
│   └── src/
│       ├── contract.cairo   #   verify_and_pay, register_model, 7 events
│       └── interfaces.cairo #   IObelyskVerifier, IERC20 dispatcher
│
└── docs/                    # Documentation
```

## Components

### STWO (Fork) — Circle STARK Prover

GPU-accelerated fork of [StarkWare's STWO](https://github.com/starkware-libs/stwo) — the fastest STARK prover, using Circle STARKs over the Mersenne-31 field (2^31 - 1).

**What we added:**
- **20,000+ lines of CUDA code** — FFT, FRI, Merkle, quotient evaluation, GKR on GPU
- **Multi-GPU support** — distributed proving across multiple H100/H200 GPUs
- **TEE integration** — NVIDIA Confidential Computing for encrypted proof generation
- **Configurable GPU thresholds** — automatic CPU/GPU dispatch based on problem size

### stwo-ml — ML Proving Library

The core Rust library that turns neural network inference into STARK proofs.

#### ML Components

| Component | Technique | Trace Reduction | What It Proves |
|-----------|-----------|-----------------|----------------|
| **MatMul** | Sumcheck protocol | 42-255x | Matrix multiplication A @ B = C |
| **Activation** | LogUp lookup tables | N/A (table-based) | ReLU, GELU, Sigmoid, Softmax |
| **Attention** | Composed sumchecks | Per-head | Q/K/V projections + softmax + context |
| **LayerNorm** | Mean/variance constraints | ~4x dim | Normalization over last dimension |
| **F32 Ops** | Dual-track proving | N/A | Float32 with fixed-point verification |

#### Sumcheck MatMul — The Core Innovation

Traditional approach to proving `C = A @ B` requires O(m x k x n) constraint rows. Our sumcheck-based approach reduces this to O(m + k + n):

```
Given: A (m x k), B (k x n), C (m x n)

1. Encode A, B, C as multilinear extensions (MLEs) over boolean hypercube
2. Reduce C[i,j] = sum_l A[i,l]*B[l,j] to sumcheck protocol
3. Prover sends round polynomials p_r(X) = c0 + c1*X + c2*X^2
4. Verifier checks p_r(0) + p_r(1) = claimed_sum per round
5. Final: verify MLE openings at challenge point

128x128 MatMul: 2.1M rows → 49K rows (42x reduction)
256x256 MatMul: 16M rows → 768 rows (255x reduction)
```

#### Model Compilation Pipeline

```
ONNX/SafeTensors → ComputationGraph → Per-layer Proofs → Aggregated STARK
     │                    │                    │                  │
  load_onnx()      GraphBuilder         prove_model()    prove_model_aggregated()
                   .matmul()              (parallel)       (single STARK for
                   .activation()                            all activations)
                   .attention()
                   .layernorm()
```

#### Proof Aggregation

Multiple activation proofs are composed into a single aggregated STARK, while matmul sumcheck proofs remain separate (they're already compact):

```
Layer 0: MatMul₀ (sumcheck) + ReLU₀ ─┐
Layer 1: MatMul₁ (sumcheck) + ReLU₁ ─┤──→ Single STARK (all activations)
Layer 2: MatMul₂ (sumcheck) + GELU₂ ─┘    + N sumcheck proofs (matmuls)
```

### stwo-cairo — Recursive Proof Compression

#### cairo-prove CLI

```bash
# Prove a Cairo program execution
cairo-prove prove <executable.json> <proof.json> --arguments 42,100

# Verify a proof
cairo-prove verify <proof.json>

# Generate recursive ML proof (compresses 17MB → ~1KB)
cairo-prove prove-ml \
  --verifier-executable ml_verifier.executable.json \
  --ml-proof ml_proof.json \
  --output recursive_proof.json \
  --gpu
```

#### ML Cairo Verifier (obelysk_ml_air)

Cairo implementation of the ML proof verifier, designed to run inside the Cairo VM so its execution can itself be proven:

- **Sumcheck verifier** — validates round polynomials and challenge derivation
- **MLE opening verifier** — Poseidon Merkle path verification
- **Fiat-Shamir channel** — deterministic challenge generation matching Rust prover
- **5 Cairo tests** covering round evaluation, proof validation, and edge cases

### ObelyskVerifier — On-Chain Contract

Starknet contract for single-transaction verification with SAGE token payment:

```
verify_and_pay(model_id, proof_hash, io_commitment, weight_commitment,
               num_layers, job_id, worker, sage_amount) → bool
```

**7 events per verification:**
1. `JobCreated` — job registered
2. `ProofSubmitted` — proof fact recorded
3. `InferenceVerified` — verification passed with full metadata
4. `PaymentProcessed` — SAGE transferred from client to worker
5. `WorkerRewarded` — worker payment confirmed
6. `VerificationComplete` — final summary with proof ID

**Additional functions:** `register_model`, `is_verified`, `get_model_verification_count`

## Getting Started

### Prerequisites

- **Rust** nightly (1.88+, 1.89+ for cairo-prove)
- **CUDA Toolkit** 12.4+ (for GPU proving)
- **Scarb** 2.12+ (for Cairo contracts)
- **starkli** (for Starknet deployment)

### Build

```bash
# CPU-only build
cd stwo-ml && cargo build --release

# GPU build (requires CUDA)
cd stwo-ml && cargo build --release --features cuda-runtime

# Build Cairo ML verifier
cd stwo-cairo/stwo_cairo_verifier && scarb build

# Build ObelyskVerifier contract
cd stwo-ml-verifier && scarb build
```

### Run Tests

```bash
# stwo-ml library tests (177 tests)
cd stwo-ml && cargo test --lib

# Cairo ML AIR tests (5 tests)
cd stwo-cairo/stwo_cairo_verifier && scarb test -p obelysk_ml_air

# GPU pipeline integration tests (requires H200)
cd stwo-ml && cargo test --test gpu_pipeline --features cuda-runtime

# Scale tests (512x512 → 2048x2048 matmul)
cd stwo-ml && cargo test --test scale_matmul --release
```

### Prove ML Inference (Full Pipeline)

```bash
# Step 1: Prove ML inference on GPU
prove-qwen --model qwen3-14b --input "Hello world" --output ml_proof.json --gpu

# Step 2: Generate recursive proof
cairo-prove prove-ml \
  --verifier-executable ml_verifier.executable.json \
  --ml-proof ml_proof.json \
  --output recursive_proof.json

# Step 3: Submit to Starknet
starkli invoke $OBELYSK_VERIFIER verify_and_pay \
  $MODEL_ID $PROOF_HASH $IO_COMMITMENT $WEIGHT_COMMITMENT \
  $NUM_LAYERS $JOB_ID $WORKER $SAGE_AMOUNT

# Step 4: Check verification
starkli call $OBELYSK_VERIFIER is_verified $PROOF_ID
```

## Feature Flags

| Flag | Description | Requires |
|------|-------------|----------|
| `std` (default) | Standard library + STWO prover | — |
| `gpu` | GPU kernel sources (no runtime) | — |
| `cuda-runtime` | Full CUDA GPU execution | CUDA 12.4+ |
| `multi-gpu` | Distributed multi-GPU proving | `cuda-runtime` |
| `tee` | NVIDIA Confidential Computing | `cuda-runtime` + H100/H200 |
| `onnx` | ONNX model import | — |
| `safetensors` | SafeTensors weight loading | — |
| `model-loading` | Both ONNX + SafeTensors | — |

## Test Coverage

| Crate | Tests | What's Covered |
|-------|-------|----------------|
| stwo-ml (lib) | 177 | MatMul sumcheck, activations, attention, ONNX, graph, aggregation, serialization, crypto |
| stwo-ml (integration) | 32 | GPU pipeline, scale testing (up to 2048x2048) |
| obelysk_ml_air (Cairo) | 5 | Sumcheck verifier, round polynomial evaluation |
| cairo_serde | 14 | Rust→felt252 serialization roundtrips |
| ObelyskVerifier | Compiles | Contract artifact generation |

## Deployed Contracts (Starknet Sepolia)

| Contract | Address |
|----------|---------|
| ObelyskVerifier (v3) | `0x04f8c5377d94baa15291832dc3821c2fc235a95f0823f86add32f828ea965a15` |
| SAGE Token | `0x072349097c8a802e7f66dc96b95aca84e4d78ddad22014904076c76293a99850` |

## How It Works

### 1. Sumcheck Protocol for MatMul

The key insight: matrix multiplication `C = A @ B` can be expressed as a sum over a multilinear extension (MLE), then verified via the sumcheck protocol in logarithmic rounds instead of materializing the full trace.

Each round, the prover sends a degree-2 polynomial. The verifier:
1. Checks `p(0) + p(1) = claimed_sum`
2. Draws a random challenge `r`
3. Updates `claimed_sum = p(r)`

After `log(k)` rounds, the verifier makes a single MLE opening query to confirm the final evaluation.

### 2. LogUp for Activations

Non-linear functions (ReLU, GELU, sigmoid) are proven via precomputed lookup tables using the LogUp protocol. The prover demonstrates that every activation output exists in the table without revealing which entry was used.

### 3. Recursive Compression

The ML verifier itself is a Cairo program. When it verifies an ML proof, its execution trace can be proven by STWO, producing a compact recursive STARK. This compresses a 17 MB ML proof down to ~1 KB that fits in a single Starknet transaction.

### 4. On-Chain Settlement

The ObelyskVerifier contract accepts recursive proofs and atomically:
- Verifies the proof fact (trusted submitter pattern, upgradeable to Integrity fact registry)
- Transfers SAGE tokens from client to GPU worker
- Emits 7 indexable events for frontends and analytics

## Performance Benchmarks

| Operation | Time | Hardware |
|-----------|------|----------|
| 128x128 MatMul prove | 11ms | Apple M3 (release) |
| 512x512 MatMul prove | 11ms | Apple M3 (release) |
| 2048x2048 MatMul prove | 189ms | Apple M3 (496 MB) |
| Qwen3-14B full prove | 40.52s | NVIDIA H200 |
| Qwen3-14B verify | 209ms | CPU |
| On-chain verification | 1 TX | Starknet Sepolia |

## Security

- **96-bit security** (configurable via PCS params)
- **No trusted setup** — FRI-based, fully transparent
- **Fiat-Shamir** via Poseidon hash (matches Cairo verifier exactly)
- **TEE option** — model weights never leave encrypted GPU memory (NVIDIA CC-On)

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

Built on [STWO](https://github.com/starkware-libs/stwo) by StarkWare.

---

<p align="center">
  <strong>Bitsage Network</strong> — Verifiable AI on Starknet
</p>
