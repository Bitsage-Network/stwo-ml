# stwo-ml

ML inference proving and privacy protocol built on [STWO](https://github.com/starkware-libs/stwo) — StarkWare's Circle STARK prover over M31. GKR sumcheck over M31 multilinear extensions. Zero f64 in the proving path — integer-only M31 arithmetic end to end.

**935+ tests, 0 failures** | **7 model families proven** | **9 adversarial attacks detected** | **Recursive STARK: 0.42s, 4KB proof** | **On-chain verified on Starknet Sepolia (6/6 TX succeeded)**

## Proven Models

Cryptographic self-verification on Apple Silicon (CPU). Every model runs through the full GKR sumcheck pipeline — no mock proofs, no skipped layers.

| Model | Family | Parameters | Prove Time | Notes |
|-------|--------|-----------|------------|-------|
| **Qwen2-0.5B** | Qwen | 0.5B | **0.57s** | Fastest full-model proof |
| **Qwen2-1.5B** | Qwen | 1.5B | **1.14s** | |
| **SmolLM2-135M** | Llama | 135M | **3.41s** | Llama-family architecture |
| **Phi-3 Mini 3.8B** | Phi | 3.8B | **48.86s** | Fused QKV + gate_up weight splitting |
| **Yi-1.5-6B** | Yi | 6B | **86.58s** | |
| **Mistral-7B-v0.3** | Mistral | 7B | **88.19s** | |
| **Llama-3.2-3B** | Meta Llama | 3B | **48.48s** | |

Architecture highlights:
- **Recursive STARK**: 0.42s recursive proof, 4KB size, 22/22 tests passing, single-TX on-chain ready
- **MoE TopK**: Complete pipeline — prover + verifier + serializer + multi-expert K branches
- **Conv2D**: im2col + MatMul lowering in GKR pipeline (unlocks YOLO, ViT, ResNet)
- **Gated FFN (SwiGLU)**: gate * up multiplication correctly modeled in the proving circuit
- **Fused weight support**: QKV splitting (Phi-3), gate_up splitting (Phi-3) handled natively
- **Zero f64 in proving path**: integer-only M31 arithmetic (cos/sin table, integer sigmoid/gelu/silu)
- **Configurable piecewise activation precision**: 16 to 4096 segments
- **RMSNorm gamma affine scale**: committed and proven
- **Binary serialization**: OZKP format via bincode
- **Streaming proof pipeline**: StreamingProofPipeline for chunk decode
- **GPU kernels**: 7K LOC CUDA + Metal shaders + STWO's 27K LOC GPU backend

### Adversarial Testing

9 adversarial attacks tested, ALL detected:

| Attack | Description | Result |
|--------|-------------|--------|
| Weight substitution | Replace model weights with different values | DETECTED |
| Output fabrication | Return fabricated outputs without running model | DETECTED |
| Proof reuse | Replay a valid proof for a different input | DETECTED |
| Model swap | Prove with model A, claim it was model B | DETECTED |
| Commitment chain tampering | Alter historical commitment chain entries | DETECTED |
| TopK fraud | Falsify MoE expert routing selection | DETECTED |
| Gamma substitution | Replace RMSNorm gamma scale factors | DETECTED |
| Platform divergence | Submit proof from mismatched platform | DETECTED |
| Activation swap | Replace activation function (e.g., GELU for SiLU) | DETECTED |

## Quick Start

```bash
# Build (macOS with Metal)
cargo build --release --bin prove-model --features cli,metal

# Prove Qwen2-0.5B (fastest — 0.57s)
./target/release/prove-model --model-dir ~/.obelysk/models/qwen2-0.5b --gkr --format ml_gkr --output proof.json

# Prove Phi-3 Mini 3.8B (fused QKV + gate_up splitting)
./target/release/prove-model --model-dir ~/.obelysk/models/phi-3-mini --gkr --format ml_gkr --output proof.json

# Prove Llama-3.2-3B
./target/release/prove-model --model-dir ~/.obelysk/models/llama-3.2-3b --gkr --format ml_gkr --output proof.json

# Prove a single transformer layer (fast iteration)
./target/release/prove-model --model-dir ./model --layers 1 --gkr --format ml_gkr

# Recursive STARK (constant-size proof, single TX)
./target/release/prove-model --model-dir ~/.obelysk/models/qwen2-0.5b --gkr --recursive --output recursive_proof.bin
```

## Python SDK and REST API

```bash
# Start the prove server
cargo build --release --bin prove-server --features server
BIND_ADDR=0.0.0.0:8080 ./target/release/prove-server
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/infer` | Provable inference — run model + generate proof |
| `GET` | `/api/v1/verify/:hash` | Verify a proof by hash |
| `GET` | `/api/v1/proofs` | List all generated proofs |

Docker deployment available via `Dockerfile` in the repository root.

## What This Is

`stwo-ml` is a dual-purpose library:

1. **Verifiable ML Inference** — Prove that a neural network forward pass was executed correctly, from ONNX model to on-chain verified proof.
2. **VM31 Privacy Protocol** — Shielded transactions (deposit/withdraw/spend) over the M31 field with STARK-based zero-knowledge proofs, Poseidon2-M31 commitments, and append-only Merkle pool.

### ML Inference Proving

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

### VM31 Privacy Protocol

- **Poseidon2-M31** — Native Poseidon2 hash over the Mersenne-31 field (t=16, rate=8, x^5 S-box, 22 rounds)
- **Note Commitments** — Poseidon2(pubkey || asset || amount || blinding) with 124-bit hiding
- **Nullifiers** — Poseidon2(spending_key || commitment) for double-spend prevention
- **Merkle Pool** — Append-only depth-20 Poseidon2-M31 Merkle tree (1M notes)
- **Symmetric Encryption** — Poseidon2-M31 counter-mode encryption for note memos
- **Transaction Circuits** — Deposit, withdraw, and 2-in/2-out spend with computational integrity proofs
- **Transaction STARKs** — Full STWO STARK proofs wrapping transaction circuits for zero-knowledge (verifier sees only public inputs)
- **Verifiable Audit** — Append-only inference logging with batch proving and on-chain submission

## Verified On-Chain — Starknet Sepolia (March 2026)

**First-ever full GKR streaming proof verification of ML inference on Starknet.** 6/6 streaming transactions confirmed SUCCEEDED on Starknet Sepolia.

| Step | Description | TX Hash | Status |
|------|-------------|---------|--------|
| 1. stream_init | IO commitment + Fiat-Shamir channel seed | [`0x5493...1a1`](https://sepolia.starkscan.co/tx/0x5493310a8e2deb5d2f25b07e2402e84692aaf5926141b5acc203a1892a181a1) | SUCCEEDED |
| 2. output_mle | Output MLE evaluation at random challenge | [`0x7cab...7e`](https://sepolia.starkscan.co/tx/0x7cabd35f5382c11334c6509e40b7a758ccd7e03e83e75b66a3c569f5d7b7a7e) | SUCCEEDED |
| 3. layers | 8 GKR layer proofs: MatMul + RMSNorm + SiLU | [`0x5346...918`](https://sepolia.starkscan.co/tx/0x53465edc957c5f8a6054739a0633beecf814ee37e3e22c23a570448a5be5918) | SUCCEEDED |
| 4. weight_binding | Aggregated weight commitment via Poseidon Merkle | [`0x5f54...3fc`](https://sepolia.starkscan.co/tx/0x5f549a1e6cc1ebefea3615c2458cdd0fd8f45fd505bf72e5b9dd8417c9be3fc) | SUCCEEDED |
| 5. input_mle | Input MLE consistency proof | [`0x2395...bab`](https://sepolia.starkscan.co/tx/0x239545b66f94387a3d1b5dbc55dedba6b7de1d5384f1930e77e152a986d5bab) | SUCCEEDED |
| 6. finalize | Final assertion + proof recording | [`0x4b08...a41`](https://sepolia.starkscan.co/tx/0x4b081156d4be88ea159533223d2597d76cd3f99911501d8326e156f12051a41) | SUCCEEDED |

**Contract**: [`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) (v39 class `0x0473c81da9df0522f5c239f022889f7730ef866fb97e4f092ad1e8793fb22feb`)

**Model**: Qwen2-0.5B, 1 transformer layer (8 GKR layers: 3 RMSNorm + 4 MatMul + 1 SiLU) | **Proof**: 5,526 felts streaming calldata, 5 MLE opening queries, schema v3 streaming v25

### How to Reproduce

```bash
# Generate on-chain compatible proof
./scripts/prove_onchain.sh --model-dir ~/.obelysk/models/qwen2-0.5b --layers 1 --format ml_gkr --gkr --output /tmp/proof.json

# Submit to Starknet
export STARKNET_PRIVATE_KEY="0x..."
export STARKNET_ACCOUNT_ADDRESS="0x..."
./scripts/prove_and_submit.sh

# Interactive TUI (chat + prove + on-chain)
./target/release/obelysk
```

**Required env vars for on-chain compatibility:**
```bash
STWO_SKIP_RMS_SQ_PROOF=1
STWO_ALLOW_MISSING_NORM_PROOF=1
STWO_PIECEWISE_ACTIVATION=0
STWO_ALLOW_LOGUP_ACTIVATION=1
STWO_AGGREGATED_FULL_BINDING=1
STWO_SKIP_BATCH_TOKENS=1
STWO_MLE_N_QUERIES=5
```

## Measured Performance

| Metric | Value | Scope |
|--------|-------|-------|
| Full model prove | **103s** | Qwen3-14B, all 40 layers (H100 NVL, cached) |
| Per-layer amortized | 2.58s | 160 MatMul sumchecks via GKR |
| Recursive STARK | **0.42s** | Constant-size proof (4KB), single TX on-chain |
| Verification | 206ms | CPU |
| Proof size (GKR) | 7.1 MB | Single inference, 281 layers |
| Proof size (recursive) | **4 KB** | Recursive STARK wrapping GKR proof |
| Audit (3 inferences) | 5m 11s | 40 layers × 3 inferences + audit report |
| MatMul trace reduction | 42–255x | Sumcheck vs naive row-by-row |
| GPU FFT speedup | 50–112x | NTT/INTT vs CPU SIMD backend |
| Security | 96-bit | pow_bits=26, n_queries=70, log_blowup=1 |
| On-chain verify | 6 TXs | Streaming GKR v25 on Starknet Sepolia (all SUCCEEDED) |
| Tests | **935+** | 0 failures expected |

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

## STWO GPU Backend

The STWO prover includes a full CUDA GPU backend at `stwo/crates/stwo/src/prover/backend/gpu/` that accelerates every stage of proof generation. All GPU paths use hybrid dispatch — SIMD fallback for small operations (< 2^12–2^14 elements), CUDA kernels for large.

### Core Runtime

| Module | Description |
|--------|-------------|
| `mod.rs` | `GpuBackend` marker struct, global context (`is_available()`, `device_name()`, `available_memory()`, `compute_capability()`) |
| `cuda_executor.rs` | NVRTC kernel compilation, device init, memory allocation. Global singleton via `get_cuda_executor()`, per-device executors via `get_executor_for_device(device_id)` |
| `compat.rs` | cudarc 0.11+ FFI bindings for peer access, CUDA graphs, async transfers, `mem_get_info()` |

### Memory & Data Transfer

| Module | Description |
|--------|-------------|
| `memory.rs` | Type-safe `GpuBuffer<T>`, FRI GPU residency cache (`FriGpuState`), pinned memory pool for fast H2D/D2H |
| `column.rs` | GPU bit-reversal for columns > 2^14 elements, column transfer utilities (`base_column_to_gpu()`, `gpu_to_base_column()`) |
| `conversion.rs` | Zero-copy transmutes between `GpuBackend` and `SimdBackend` (identical column types, compile-time size assertions) |

### Computation Kernels

| Module | Description |
|--------|-------------|
| `fft.rs` | Circle FFT over M31 with shared-memory butterfly, cached twiddles. **50–112x speedup** vs SIMD for 1M–4M elements |
| `constraints.rs` | Direct GPU evaluation of AIR constraints with embedded M31 modular arithmetic |
| `quotients.rs` | Fused GPU quotient kernel for large domains (log_size >= 14), SIMD fallback for small |
| `poly_ops.rs` | Polynomial interpolate/evaluate/extend (mostly SIMD-delegated; real GPU acceleration via `GpuProofPipeline`) |
| `fri.rs` | FRI folding with GPU residency — large folds (> 2^12) on GPU, thread-local twiddle cache |
| `gkr.rs` | GKR lookup ops: `fix_first_variable()` GPU-accelerated for > 16K elements, `gen_eq_evals()` for large outputs |

### Merkle & Accumulation

| Module | Description |
|--------|-------------|
| `merkle.rs` | GPU Blake2s hashing for trees > 2^14 leaves (2–4x speedup for > 64K leaves). Supports Blake2s, Blake2sM31, Poseidon252 hashers |
| `merkle_lifted.rs` | Lifted (compressed) Merkle ops delegating to SIMD via zero-copy conversion |
| `accumulation.rs` | Element-wise accumulation (SIMD-delegated — no GPU benefit at this granularity) |
| `grind.rs` | Proof-of-work grinding (SIMD-delegated) |

### Pipeline & Optimization

| Module | Description |
|--------|-------------|
| `pipeline.rs` | `GpuProofPipeline` — persistent GPU memory across trace→FFT→quotient→FRI→Merkle. Multi-GPU via per-device executors. Optional CUDA Graphs (20–40% speedup) |
| `cuda_streams.rs` | `CudaStreamManager` — overlapped H2D/compute/D2H with 3 primary streams (~10–15% latency hiding) |
| `optimizations.rs` | CUDA Graphs (kernel capture/replay), pinned memory pool (`get_pinned_pool_u32()`), global memory pool with RAII cleanup |
| `large_proofs.rs` | Support for 2^26+ polynomials. `MemoryRequirements` calculator, chunked processing for proofs exceeding GPU memory |
| `multi_gpu.rs` | `GpuCapabilities` for device introspection (compute capability, SM count, Tensor Cores). Throughput mode (independent proofs) and distributed mode (single proof across GPUs) |
| `multi_gpu_executor.rs` | Thread-safe `MultiGpuExecutorPool` — per-GPU executors in `Mutex`, global singleton, fallback logic |

### TEE Confidential Computing (`tee/`)

Hardware-based attestation and encrypted memory for H100/H200/B200 GPUs:

| Module | Description |
|--------|-------------|
| `tee/mod.rs` | `ConfidentialGpu` enum (H100 80GB, H100 NVL 94GB, H200 141GB, B200 192GB), `SessionState` lifecycle |
| `tee/attestation.rs` | GPU attestation (SPDM protocol) and CPU attestation (TDX/SEV-SNP) via nvTrust SDK |
| `tee/crypto.rs` | AES-GCM-256 encryption matching GPU DMA, HKDF key derivation, SHA-256 hashing |
| `tee/nvtrust.rs` | nvTrust SDK client — `GpuInfo`, attestation evidence, CC mode queries via Python SDK or `nvidia-smi conf-compute` |
| `tee/cc_mode.rs` | Confidential Computing mode configuration |

### Key Design Patterns

- **Hybrid dispatch**: Every GPU path falls back to SIMD below a threshold (2^12–2^14 elements)
- **GPU residency**: FRI folds and proof pipeline keep data on-device between stages — single H2D in, single D2H out
- **Zero-copy conversion**: `GpuBackend` ↔ `SimdBackend` via `unsafe` transmute with compile-time size assertions
- **Thread-local caching**: Twiddle factors cached per-thread in FFT and FRI to avoid recomputation
- **CUDA Graphs**: Optional kernel sequence capture eliminates launch overhead (20–40% for repeated patterns)
- **Multi-GPU pool**: `get_executor_for_device(id)` → true per-device parallelism with `Mutex`-wrapped executor pool

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

## VM31 Privacy Protocol

The VM31 privacy protocol implements shielded transactions over the M31 field. Users deposit public funds into a shielded pool (Merkle tree of note commitments), transfer privately within the pool, and withdraw back to public.

### Cryptographic Primitives (`crypto/`)

All cryptography is native M31 — no field conversions, no felt252 wrappers.

| Primitive | Module | Description |
|-----------|--------|-------------|
| Poseidon2-M31 | `crypto/poseidon2_m31.rs` | t=16, rate=8, x^5 S-box, 4+14+4 rounds. Hash output: 8 M31 elements (~124-bit collision resistance) |
| Note Commitment | `crypto/commitment.rs` | Poseidon2 sponge hash of pk, asset, amount, blinding (11 M31 input) |
| Nullifier | `crypto/commitment.rs` | Poseidon2 sponge hash of spending\_key and commitment (12 M31 input) |
| Key Derivation | `crypto/commitment.rs` | Domain-separated Poseidon2 hash: `"spend"` → pubkey, `"view"` → viewing key |
| Merkle Tree | `crypto/merkle_m31.rs` | Append-only depth-20 tree, Poseidon2 compress for internal nodes |
| Encryption | `crypto/encryption.rs` | Poseidon2-M31 counter-mode, key derivation, checksum validation |

```rust
use stwo_ml::crypto::commitment::{Note, derive_pubkey, compute_commitment, compute_nullifier};
use stwo_ml::crypto::merkle_m31::PoseidonMerkleTreeM31;

// Create a note
let sk = [M31::from(42), M31::from(43), M31::from(44), M31::from(45)];
let pk = derive_pubkey(&sk);
let note = Note::new(pk, asset_id, amount_lo, amount_hi, blinding);
let commitment = compute_commitment(&note);
let nullifier = compute_nullifier(&sk, &commitment);

// Insert into Merkle pool
let mut tree = PoseidonMerkleTreeM31::new(20);
let index = tree.insert(&commitment);
let proof = tree.prove(index);
assert!(tree.verify(&commitment, &proof));
```

### Transaction Circuits (`circuits/`)

Three transaction types, each with computational integrity proofs (Phase 3) and STARK zero-knowledge proofs (Phase 4).

| Transaction | Circuit | STARK | Perms | Trace Width |
|------------|---------|-------|-------|-------------|
| **Deposit** | `circuits/deposit.rs` | `circuits/stark_deposit.rs` | 2 | ~1,372 cols |
| **Withdraw** | `circuits/withdraw.rs` | `circuits/stark_withdraw.rs` | 25 (real) + 7 (pad) = 32 | ~20,932 cols |
| **Spend** | `circuits/spend.rs` | `circuits/stark_spend.rs` | 54 (real) + 10 (pad) = 64 | ~41,864 cols |

**Deposit** — Convert public funds to a shielded note:
```
Public: (amount, asset_id) → commitment
Proof: commitment = Poseidon2(pk || asset || amount || blinding)
```

**Withdraw** — Reveal a shielded note to public:
```
Public: (merkle_root, nullifier, amount, asset_id)
Proof: ownership + commitment + nullifier + Merkle membership (20 levels)
```

**Spend** — Private 2-in/2-out transfer:
```
Public: (merkle_root, nullifier_0, nullifier_1, output_commitment_0, output_commitment_1)
Proof: per-input (ownership + commitment + nullifier + Merkle) ×2
     + per-output (commitment hash) ×2
     + balance conservation + asset consistency
```

### Transaction STARKs (Zero-Knowledge)

Phase 4 wraps each transaction circuit into a single-row STWO STARK proof. The verifier sees **only public inputs** — all private data (keys, blinding factors, Merkle paths, note contents) is hidden by the FRI protocol.

**Architecture**: Each transaction type is a single wide-row `FrameworkEval` where one row = one complete transaction. All Poseidon2 permutations are unrolled within the row (no cross-row constraints). Each permutation occupies 652 trace columns (23×16 state columns + S-box auxiliaries).

**Key constraint techniques**:
- **S-box decomposition**: x^5 → `sq = (s+rc)^2`, `quad = sq^2`, `out = quad * (s+rc)` — all degree 2
- **Merkle chain**: `(input[j] - prev_out[j]) * (input[j+8] - prev_out[j]) = 0` — degree-2 without `is_real` selector
- **Range check**: Bit decomposition (16 bits per sub-limb), no LogUp table needed
- **No interaction trace**: Only preprocessed + execution trees (no `finalize_logup`)

```rust
use stwo_ml::circuits::stark_spend::{prove_spend_stark, verify_spend_stark};

// Prove a private transfer
let proof = prove_spend_stark(&witness)?;

// Verify with only public inputs
verify_spend_stark(&proof, &public_inputs)?;
// public_inputs: merkle_root, nullifiers, output_commitments — no private data
```

See [`docs/vm31-privacy-protocol.md`](docs/vm31-privacy-protocol.md) for the full protocol design, constraint details, and security analysis.

### Privacy SDK (`privacy/`)

High-level wallet, transaction builder, and pool interaction layer for the VM31 shielded pool.

| Module | Description |
|--------|-------------|
| `privacy/wallet.rs` | Key management — create, load, encrypt/decrypt wallet files (`~/.vm31/wallet.json`) |
| `privacy/tx_builder.rs` | Transaction builder — compose deposit/withdraw/spend batches for proving |
| `privacy/pool_client.rs` | On-chain pool queries via JSON-RPC — Merkle root, tree size, nullifier checks, event scanning |
| `privacy/tree_sync.rs` | Global Merkle tree sync from on-chain `NoteInserted` events with disk cache (`~/.vm31/tree_cache.json`) |
| `privacy/note_store.rs` | Local note tracking — persistent JSON store, balance queries, note selection, memo scanning |
| `privacy/relayer.rs` | Relayer interaction for batch submission |
| `privacy/serde_utils.rs` | Batch proof serialization/deserialization |

#### Pool Client

Reads on-chain state from the VM31 pool contract via `starknet_call` and `starknet_getEvents` JSON-RPC:

```rust
use stwo_ml::privacy::pool_client::{PoolClient, PoolClientConfig};

let config = PoolClientConfig::from_env("sepolia"); // uses default pool address
let client = PoolClient::new(config);

let root = client.get_merkle_root()?;     // [M31; 8]
let size = client.get_tree_size()?;        // u64
let spent = client.is_nullifier_spent(&nullifier)?;

// Scan NoteInserted events (paginated)
let events = client.get_note_inserted_events(from_block)?;
```

Function selectors use proper `sn_keccak` (Keccak-256 truncated to 250 bits).

#### Tree Sync

Rebuilds the full depth-20 Merkle tree locally by scanning on-chain events, then generates Merkle proofs against the real pool root:

```rust
use stwo_ml::privacy::tree_sync::TreeSync;

let mut tree = TreeSync::load_or_create(&TreeSync::default_cache_path())?;
let result = tree.sync(&pool_client)?;  // fetch new events, append, verify root
println!("Synced: {} leaves ({} new)", result.total_leaves, result.events_added);

let path = tree.prove(leaf_index)?;     // Merkle proof for a note
let root = tree.root();                 // current synced root
let idx = tree.find_commitment(&commitment); // lookup by commitment
```

The sync follows the standard Tornado Cash pattern: scan `NoteInserted` events, rebuild sequentially, verify root matches on-chain. Cache persists to `~/.vm31/tree_cache.json` for incremental sync (~20ms to reload 1000 leaves).

#### Deployed Pool Contract (Starknet Sepolia)

| Field | Value |
|-------|-------|
| **Address** | `0x07cf94e27a60b94658ec908a00a9bb6dfff03358e952d9d48a8ed0be080ce1f9` |
| **Class Hash** | `0x046d316ca9ffe36adfdd3760003e9f8aa433cb34105619edcdc275315a2c8405` |
| **Owner/Relayer** | `0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344` |
| **Verifier** | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` (EloVerifier v11) |
| **Upgradability** | 5-minute timelocked `propose_upgrade` / `execute_upgrade` |

### Verifiable Audit System (`audit/`)

Captures real-time model inference into an append-only log, then batch-proves correctness over a time window with optional semantic evaluation.

```text
HOT PATH (serving)              AUDIT PATH (on-demand)
User → Server → Response        "Prove last hour" →
          │                              │
          v                    ┌─────────┼─────────┐
   Inference Log               v         v         v
   (append-only)          Summary   Prover    Evaluator
                          (instant) (GPU)     (parallel)
                              │         │         │
                              └─────────┼─────────┘
                                        v
                                  Audit Report
                                  → Encrypt → Arweave → On-chain
```

| Module | Description |
|--------|-------------|
| `audit/log.rs` | Append-only inference log with Merkle commitment |
| `audit/capture.rs` | Non-blocking capture hook for model servers |
| `audit/replay.rs` | Replay verification before proving |
| `audit/prover.rs` | Batch audit prover over time windows |
| `audit/submit.rs` | On-chain submission of audit records |
| `audit/deterministic.rs` | Deterministic output checks |
| `audit/self_eval.rs` | Self-evaluation via model forward pass |
| `audit/scoring.rs` | Aggregate semantic scoring |
| `audit/report.rs` | Report builder and hash computation |
| `audit/storage.rs` | Arweave upload/download client |
| `audit/encryption.rs` | Encryption integration with VM31 |
| `audit/orchestrator.rs` | End-to-end audit orchestration |

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/vm31-privacy-protocol.md`](docs/vm31-privacy-protocol.md) | VM31 privacy protocol: Poseidon2-M31, commitments, Merkle pool, transaction circuits, STARKs |
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
│   │   ├── quantize.rs       # 2D LogUp quantization verification
│   │   ├── poseidon2_air.rs  # Poseidon2-M31 AIR constraints (shared by transaction STARKs)
│   │   └── range_check.rs    # Range check gadgets for bounding values
│   │
│   ├── crypto/               # VM31 cryptographic primitives
│   │   ├── poseidon2_m31.rs  # Poseidon2 hash over M31 (t=16, rate=8, x^5, 22 rounds)
│   │   ├── commitment.rs     # Note commitment, nullifier, key derivation
│   │   ├── merkle_m31.rs     # Append-only Poseidon2-M31 Merkle tree (depth 20)
│   │   ├── encryption.rs     # Poseidon2-M31 counter-mode encryption
│   │   ├── hades.rs          # Hades permutation (felt252, for Cairo compatibility)
│   │   ├── poseidon_merkle.rs # Poseidon-based Merkle tree (felt252)
│   │   ├── mle_opening.rs    # MLE commitment and opening proofs
│   │   └── poseidon_channel.rs # Fiat-Shamir channel matching Cairo verifier
│   │
│   ├── circuits/             # VM31 transaction circuits
│   │   ├── poseidon_circuit.rs # Poseidon2-M31 batch permutation proving
│   │   ├── helpers.rs        # Permutation-recording helpers for composed proofs
│   │   ├── deposit.rs        # Deposit: public amount → shielded note (Phase 3)
│   │   ├── withdraw.rs       # Withdraw: shielded note → public amount (Phase 3)
│   │   ├── spend.rs          # 2-in/2-out private transfer (Phase 3)
│   │   ├── stark_deposit.rs  # Deposit STARK: ZK proof, 2 perms (Phase 4)
│   │   ├── stark_withdraw.rs # Withdraw STARK: ZK proof, 32 perms (Phase 4)
│   │   └── stark_spend.rs    # Spend STARK: ZK proof, 64 perms (Phase 4)
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
│   ├── privacy/              # VM31 privacy SDK
│   │   ├── wallet.rs         # Key management, create/load/encrypt wallet files
│   │   ├── tx_builder.rs     # Transaction builder — compose batches for proving
│   │   ├── pool_client.rs    # On-chain pool queries via JSON-RPC + event scanning
│   │   ├── tree_sync.rs      # Global Merkle tree sync from NoteInserted events
│   │   ├── note_store.rs     # Local note tracking (persistent JSON, balance, selection)
│   │   ├── relayer.rs        # Relayer interaction for batch submission
│   │   └── serde_utils.rs    # Batch proof serialization/deserialization
│   │
│   ├── audit/                # Verifiable inference audit system
│   │   ├── types.rs          # InferenceLogEntry, AuditReport, shared types
│   │   ├── log.rs            # Append-only inference log with Merkle commitment
│   │   ├── capture.rs        # Non-blocking capture hook for model servers
│   │   ├── replay.rs         # Replay verification before proving
│   │   ├── prover.rs         # Batch audit prover over time windows
│   │   ├── submit.rs         # On-chain submission of audit records
│   │   ├── deterministic.rs  # Deterministic output checks
│   │   ├── self_eval.rs      # Self-evaluation via model forward pass
│   │   ├── scoring.rs        # Aggregate semantic scoring
│   │   ├── report.rs         # Report builder and hash computation
│   │   ├── storage.rs        # Arweave upload/download client
│   │   ├── encryption.rs     # Encryption integration with VM31
│   │   └── orchestrator.rs   # End-to-end audit orchestration
│   │
│   ├── gadgets/              # Reusable constraint gadgets
│   │   └── range_check.rs    # Range check tables (uint8, uint16, int8)
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
└── tests/                    # Integration tests (935+ total: lib + e2e_audit + cli_audit)
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

#### Privacy Subcommands

```bash
# Wallet management
prove-model wallet --create                     # Generate new spending key + viewing key
prove-model wallet --info                       # Show address + balances

# Deposit (public -> shielded)
prove-model deposit --amount 1000 --asset 0     # Deposit asset 0 (STRK)

# Withdraw (shielded -> public, syncs global Merkle tree)
prove-model withdraw --amount 500 --asset 0 --pool-contract 0x07cf94...

# Transfer (2-in/2-out private spend)
prove-model transfer --amount 300 --asset 0 \
  --to 0x<recipient_pubkey> --to-viewing-key 0x<recipient_vk>

# Scan for incoming notes + sync tree
prove-model scan --pool-contract 0x07cf94...

# Query pool state
prove-model pool-status --pool-contract 0x07cf94...
prove-model pool-status --check-nullifier 0x<hex>
prove-model pool-status --check-root 0x<hex>

# Batch proving (deposits only via file)
prove-model batch --tx-file transactions.json
```

The `withdraw` and `transfer` commands sync the global Merkle tree from on-chain `NoteInserted` events before generating proofs, ensuring the Merkle root matches the pool contract. The `scan` command updates pending note indices and confirms notes on-chain.

#### Output Formats

| Format | Description |
|--------|-------------|
| `cairo_serde` | felt252 hex array for `cairo-prove prove-ml` recursive path |
| `json` | Human-readable JSON with proof components |
| `direct` | **(Deprecated)** JSON with `batched_calldata` for legacy `verify_model_direct()` — removed from EloVerifier v11, use `ml_gkr` instead |

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
| **GKR (recommended)** | GPU prove (~103s full model) | — | `verify_model_gkr_v4()` (streaming) |
| Recursive | GPU prove (~103s full model) | Cairo VM STARK (46.8s) | `verify<CairoAir>` |

```bash
# GKR pipeline (recommended, single tx)
prove-model --model model.onnx --format ml_gkr --gpu --output proof.json

# Recursive pipeline (3-stage, via Cairo VM)
prove-model --model model.onnx --format cairo_serde --gpu --output args.json
cairo-prove prove-ml stwo_ml_recursive.executable.json args.json
```

**IO Commitment**: All verification paths accept raw I/O data and recompute `Poseidon(raw_io_data)` on-chain — no caller-supplied commitments are trusted. The GKR path additionally evaluates MLEs on-chain to bind the proof to exact inputs and outputs.

**Current contract** (Sepolia): [`0x0121d1...c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) (EloVerifier v11)

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
| `audit` | Verifiable inference audit system | — |
| `audit-http` | Audit with HTTP submission | `audit` |
| `aes-fallback` | AES-GCM encryption fallback for audit | `audit` |
| `server-audit` | prove-server with audit integration | `server` + `audit` |

## License

Apache 2.0
