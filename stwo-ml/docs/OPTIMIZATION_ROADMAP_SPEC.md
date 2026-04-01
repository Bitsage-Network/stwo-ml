# ObelyZK Optimization Roadmap — Deep Specification

**Version**: 1.0 | **Date**: March 24, 2026 | **Author**: Bitsage Network

> Engineering specification for achieving 500 proven tokens/second on a single H100,
> 0.001% activation precision, full platform determinism, and mathematical guardrail
> enforcement. Every section includes architecture, files to modify, success criteria,
> and estimated effort.

---

## Current Baseline (March 24, 2026)

### Performance

| Metric | Value | Hardware |
|--------|-------|----------|
| Full model prove (Qwen3-14B, 40 layers) | **103s** | H100 NVL (cached) |
| Single layer prove | **3s** | H100 |
| Current throughput (seq_len=1) | ~0.01 tok/s | H100 |
| Estimated throughput (seq_len=10K) | **~100 tok/s** | H100 (not yet benchmarked) |
| Security tests | 41/41 pass | — |
| Test suite | 818 + 10 e2e + 95 Cairo | — |

### 103-Second Breakdown (Warm Model)

| Phase | Time | Backend | Optimization Target |
|-------|------|---------|-------------------|
| Forward pass (281 nodes) | 38s | **CPU** | → GPU (Phase A) |
| GKR walk (160 matmuls) | 47s | **Our GPU kernels** | → Parallelism (Phase B) |
| Unified STARK (121 components) | 5s | **CPU** (STWO bug) | → GPU (Phase C) |
| Serialization | 2s | CPU | → Binary (Phase D) |
| **Total (warm)** | **92s** | | **→ Target: ~22s** |

### Accuracy

| Component | Current Precision | Status |
|-----------|------------------|--------|
| MatMul (M31 arithmetic) | Exact | No change needed |
| Layer-to-layer flow | Exact (no re-quantization) | No change needed |
| ReLU | Exact lookup | No change needed |
| INT8 Symmetric quantization | ±0.02 per value | No change needed |
| GELU/SiLU (piecewise, 16 segments) | ~2-5% max deviation | → 0.001% (Phase F) |
| Sigmoid (piecewise, 16 segments) | ~2-5% max deviation | → 0.001% (Phase F) |
| Softmax (LogUp, 2^20 table) | ~0.01% max deviation | → 0.001% (Phase F) |
| Platform determinism | **NOT guaranteed** (f64 in RoPE, tables) | → Guaranteed (Phase E) |
| Guardrail enforcement | **Not implemented** | → Enforced (Phase G) |

### Custom GPU Stack (19 CUDA Kernels + 4 Metal Shaders)

| Kernel Suite | Kernels | Purpose |
|-------------|---------|---------|
| Sumcheck protocol | 3 (round, reduce, fold) | On-device sumcheck — never leaves GPU |
| Forward pass M31 | 7 (GEMM, GEMV, add, mul, ReLU, LayerNorm, RMSNorm) | M31 field arithmetic on GPU |
| Fused MLE restrict | 2 (restrict_rows, restrict_cols) | Eliminates 2GB allocation + 24s CPU folding |
| LogUp activation | 5 (denominator, 3way_round, 4way_reduce, 3way_fold, combine) | Full GPU LogUp eq-sumcheck |
| Metal (Apple Silicon) | 4 (matmul, sumcheck, dispatch, device) | M1/M2/M3/M4 support |

---

## Target Performance (After All Phases)

### Single H100

| Mode | Current | Target | Speedup |
|------|---------|--------|---------|
| Full prefill (seq_len=1) | ~92s | ~22s | **4.2x** |
| Batched prefill (seq_len=10K) | ~125s (est.) | ~31s | **4x** |
| Batched prefill (seq_len=20K) | ~150s (est.) | ~45s | **3.3x** |
| Incremental delta (3K new tokens) | ~45s (est.) | ~18s | **2.5x** |
| **Peak throughput** | ~100 tok/s | **~500 tok/s** | **5x** |
| **Sustained incremental** | ~40 tok/s (est.) | **~200 tok/s** | **5x** |

### Multi-GPU and Next-Gen Hardware

| Configuration | Peak tok/s | Sustained tok/s | Est. cost/hr |
|--------------|-----------|----------------|-------------|
| 1× H100 | 500 | 200 | ~$3.50 |
| 8× H100 | 4,000 | 1,600 | ~$10-28 |
| 1× H200 (4.8 TB/s BW) | 700 | 280 | ~$4-5 |
| 8× H200 | 5,600 | 2,240 | ~$32-40 |
| 1× B200 (8.0 TB/s BW) | 1,200 | 480 | ~$5-6 |
| 8× B200 | 9,600 | 3,840 | ~$40-50 |
| 1× B300 (12.0 TB/s BW) | 1,800 | 720 | TBD |
| 8× B300 | 14,400 | 5,760 | TBD |

### Cost Economics

| Cluster | Sustained tok/hr | Cost/hr | $/M proven tokens |
|---------|-----------------|---------|-------------------|
| 8× H100 | 5.76M | ~$10 | **$1.74** |
| 8× B200 | 13.82M | ~$45 | **$3.26** |
| 8× B300 | 20.74M | ~$50 est. | **$2.41** |

### Target Accuracy (After All Phases)

| Component | Current | Target | Phase |
|-----------|---------|--------|-------|
| GELU/SiLU/Sigmoid | ~2-5% | **~0.001%** | Phase F |
| Platform determinism | Not guaranteed | **Bit-identical everywhere** | Phase E |
| Unverified operations | RoPE*, softmax sum*, mask* | **None** | Phases 1B-1E (closed) |
| Guardrail enforcement | None | **Mathematical — policy commitment** | Phase G |

*Note: Phases 1B, 1C, 1D are already CLOSED as of March 17, 2026.

---

## Phase A: GPU Forward Pass

**Goal**: Move the 38-second CPU forward pass to GPU using existing kernels.

**Status**: OPEN — kernels exist, wiring needed.

**Impact**: 38s → ~5-8s (forward pass), ~92s → ~57-60s total.

### Architecture

The GPU forward pass kernels already exist in `gpu_sumcheck.rs`:
- `m31_gemm_kernel` — dense matrix multiply (16×16 workgroups)
- `m31_gemv_kernel` — single-row matrix-vector (decode path)
- `m31_add_kernel` — element-wise modular addition
- `m31_mul_kernel` — element-wise modular multiplication
- `m31_relu_kernel` — ReLU activation (threshold at P/2)
- `m31_layernorm_kernel` — fused mean + variance + rsqrt + normalize
- `m31_rmsnorm_kernel` — fused RMS² + rsqrt + normalize

Currently these are called individually. The forward pass in `aggregation.rs` executes
on CPU via the `ComputationGraph` traversal. The change is to replace CPU node execution
with GPU kernel dispatch.

### Files to Modify

| File | Change |
|------|--------|
| `src/aggregation.rs` | `collect_forward_pass_layer_data()` — dispatch to GPU when `cuda-runtime` feature enabled |
| `src/gpu_sumcheck.rs` | Add `gpu_forward_pass()` orchestrator — sequential kernel launches per node |
| `src/compiler/prove.rs` | Add GPU path in `execute_graph_forward()` |
| `src/components/attention.rs` | `attention_forward()` — GPU score/context matmuls |

### Key Constraints

- Intermediate outputs must remain on GPU memory between layers (avoid CPU↔GPU roundtrips)
- Attention Q×K^T requires `(seq_len, d_k) × (d_k, seq_len)` — GPU GEMM handles this
- LayerNorm/RMSNorm statistics (mean, variance) needed both for forward pass AND LogUp proof
- GPU memory budget: ~80GB H100. Qwen3-14B intermediates at seq_len=1: ~40MB (trivial).
  At seq_len=40K: ~1.6GB per layer intermediate (manageable).

### Implementation Steps

1. Add `GpuForwardExecutor` struct that holds device handle + kernel references
2. Implement `execute_node_gpu()` for each `GraphOp` variant (MatMul, Add, Mul, Activation, LayerNorm, RMSNorm, Attention)
3. Keep intermediates in GPU device memory (`CudaSlice<u32>`) between nodes
4. Download final output + all intermediates only after full forward pass completes
5. Wire into `prove_model_pure_gkr_auto_with_cache()` when GPU available

### Success Criteria

- [ ] Forward pass runs entirely on GPU (no CPU matmul fallback)
- [ ] Intermediates remain on GPU memory between layers
- [ ] `prove_model_pure_gkr_auto_with_cache()` uses GPU forward when available
- [ ] Benchmark: forward pass < 8s on H100 for Qwen3-14B (seq_len=1)
- [ ] All 818 tests pass
- [ ] E2e audit tests pass with GPU forward

### Estimated Effort: 2 weeks

---

## Phase B: GKR Walk Parallelism

**Goal**: Parallelize independent matmul sumchecks within each transformer layer.

**Status**: OPEN.

**Impact**: 47s → ~15-20s (GKR walk).

### Architecture

Within each transformer layer, some matmul sumchecks are independent:
- Q, K, V projections are independent of each other
- FFN up and FFN gate projections are independent
- These can run concurrently on the same GPU (multiple CUDA streams)

The GKR walk in `prove_gkr_gpu_with_cache()` currently processes all 160 matmuls
sequentially. Within each layer, 2-4 matmuls can run in parallel.

### Approach

1. **Identify independent matmuls per layer**: Group by data dependency (Q/K/V parallel, output sequential after attention, FFN up/gate parallel)
2. **Multi-stream execution**: Launch independent sumcheck kernels on separate CUDA streams
3. **Synchronize at layer boundary**: `cudaStreamSynchronize()` before next layer's GKR claim chain

### Files to Modify

| File | Change |
|------|--------|
| `src/gkr/prover.rs` | `prove_gkr_gpu_with_cache()` — identify parallel groups, dispatch to streams |
| `src/gpu_sumcheck.rs` | `GpuSumcheckExecutor` — accept stream parameter for kernel launches |
| `src/gpu_sumcheck.rs` | `GpuMatMulOracle` — stream-aware `sum_as_poly_in_first_variable()` |

### Key Constraints

- GKR claim chaining is sequential between layers (layer L's output claim feeds layer L+1)
- Within-layer parallelism limited to 2-4 concurrent matmuls (GPU memory for MLE buffers)
- CUDA stream creation overhead: negligible (reuse stream pool)

### Expected Speedup

| Layer type | MatMuls | Parallelizable | Sequential depth |
|-----------|---------|---------------|-----------------|
| Attention (Q,K,V) | 3 | 3-way parallel | 1 |
| Attention (output) | 1 | Sequential (depends on attention) | 1 |
| FFN (up, gate) | 2 | 2-way parallel | 1 |
| FFN (down) | 1 | Sequential (depends on activation) | 1 |
| **Per layer** | **4-7** | | **~3 sequential steps** |

Theoretical: 47s × (3/7) ≈ 20s. Conservative estimate: **~18-22s**.

### Success Criteria

- [ ] Within-layer parallel sumcheck dispatch via CUDA streams
- [ ] GKR claim chain integrity maintained (cross-layer sequential)
- [ ] Benchmark: GKR walk < 22s on H100 for Qwen3-14B
- [ ] All 818 tests pass
- [ ] Tamper tests still reject invalid proofs

### Estimated Effort: 2-3 weeks

---

## Phase C: GPU Unified STARK

**Goal**: Move the 5-second unified STARK from CPU to GPU.

**Status**: BLOCKED — requires STWO library fix for preprocessed column deduplication bug.

**Impact**: 5s → ~1-2s.

### Blocker

STWO's `GpuBackend` preprocessed column allocator deduplicates columns by name.
Multi-instance components (e.g., 40 RMSNorm layers) read wrong column data, causing
`ConstraintsNotSatisfied`. Instance ID workaround fixes `SimdBackend` but not `GpuBackend`.

### Resolution Path

1. **Option A**: Fix in STWO upstream — add instance ID to preprocessed column key
2. **Option B**: Workaround — unique column names per instance (`rmsnorm_0_col`, `rmsnorm_1_col`, ...)
3. **Option C**: Skip — 5s → 2s is low priority relative to 38s + 47s targets

### Files to Modify (when unblocked)

| File | Change |
|------|--------|
| `src/aggregation.rs` | Switch `SimdBackend` → `GpuBackend` for unified STARK |
| STWO library | Fix preprocessed column allocator (instance-aware keying) |

### Success Criteria

- [ ] Unified STARK proves on `GpuBackend` without `ConstraintsNotSatisfied`
- [ ] Benchmark: unified STARK < 2s on H100
- [ ] All 121 component constraints pass

### Estimated Effort: 1 week (after STWO fix available)

---

## Phase D: Binary Serialization

**Goal**: Replace JSON hex encoding with binary proof format.

**Status**: OPEN.

**Impact**: 2s → 0.3s serialization, 7MB → 3.6MB proof size.

### Architecture

Current: `serde_json` serialization with hex-encoded field elements.
Target: Compact binary format with direct `FieldElement` byte encoding.

### Files to Modify

| File | Change |
|------|--------|
| `src/gkr/types.rs` | Add `to_bytes()` / `from_bytes()` for `GKRProof` |
| `src/aggregation.rs` | Add `to_bytes()` / `from_bytes()` for `AggregatedModelProofOnChain` |
| `src/bin/prove_model.rs` | `--format binary` flag |
| `src/starknet.rs` | Binary calldata builder (parallel to JSON path) |

### Success Criteria

- [ ] Binary roundtrip: `proof == deserialize(serialize(proof))`
- [ ] Benchmark: serialization < 0.5s
- [ ] Proof size < 4MB for Qwen3-14B 40-layer
- [ ] JSON format still supported (backward compatibility)

### Estimated Effort: 1 week

---

## Phase E: Platform Determinism (Phase 1E from ROADMAP)

**Goal**: Eliminate all f64-dependent computation. Ensure bit-identical M31 values on every platform.

**Status**: OPEN — CRITICAL for distributed proving and accuracy enforcement.

**Impact**: Enables cross-platform proof reproducibility. Required for accuracy rails to be meaningful.

### What Uses f64 Today

| Operation | Where | f64 Usage |
|-----------|-------|-----------|
| RoPE angles | `src/components/rope.rs` | `base.powf(-2.0 * j / d)`, `angle.cos()`, `angle.sin()` |
| Activation tables | `src/components/activation.rs:308-338` | `apply_activation_f64()` uses `tanh()`, `exp()` |
| Piecewise coefficients | `src/components/activation.rs:265-266` | `apply_activation_f64()` at segment boundaries |
| Quantization rounding | `src/gadgets/quantize.rs:106` | `(value as f64 / params.scale).round()` |

### Fix Strategy

#### E1. Integer-Only RoPE Angles

Replace `powf` / `cos` / `sin` with fixed-point Chebyshev polynomial approximation over M31:

```
cos(θ) ≈ 1 - θ²/2 + θ⁴/24 - θ⁶/720 + ...  (all in M31 fixed-point)
sin(θ) ≈ θ - θ³/6 + θ⁵/120 - θ⁷/5040 + ... (all in M31 fixed-point)
```

- Precompute all angle values using only M31 arithmetic
- Store as committed table (part of model registration)
- Degree-10 Chebyshev gives ~10^-7 precision (sufficient for M31 range)

**Files**: `src/components/rope.rs`

#### E2. Integer-Only Activation Tables

Replace `apply_activation_f64()` with exhaustive M31 enumeration or integer polynomial:

- For GELU (16-bit table = 65536 entries): precompute ALL values using Chebyshev in M31
- For piecewise coefficients: compute segment boundaries using M31 arithmetic only
- Store canonical tables as committed artifacts

**Files**: `src/components/activation.rs`, `src/gadgets/lookup_table.rs`

#### E3. Deterministic Quantization

Replace `f64` rounding with M31-native rounding:

```rust
// Current (f64-dependent):
let q = (value as f64 / params.scale).round() as i64 + params.zero_point as i64;

// Target (integer-only):
let q = m31_fixed_point_div(value_m31, scale_m31) + zero_point;
```

**Files**: `src/gadgets/quantize.rs`

### Test Plan

- [ ] Cross-platform determinism test: compute all tables on x86, ARM, GPU — assert identical M31 outputs
- [ ] Roundtrip: integer-only RoPE matches f64 RoPE within ε for all positions in [0, max_seq_len]
- [ ] Activation table: integer-only GELU table matches f64 table within ε for all 2^16 inputs
- [ ] No `f64` or `f32` used anywhere in the proving path (enforced by `#[deny(clippy::float_arithmetic)]` on proving modules)

### Success Criteria

- [ ] `grep -r "as f64\|as f32\|powf\|\.cos()\|\.sin()\|\.exp()\|\.tanh()" src/components/ src/gadgets/` returns zero matches in proving-path code
- [ ] Cross-platform test passes (x86 + ARM produce identical M31 values)
- [ ] All 818 tests pass
- [ ] Benchmark: no measurable performance regression

### Estimated Effort: 3-4 weeks

---

## Phase F: High-Precision Activation Rails

**Goal**: Increase piecewise activation segments from 16 to 1024. Reduce max deviation from ~2-5% to ~0.001%.

**Status**: OPEN.

**Impact**: Activation precision improvement with zero runtime cost increase.

### Architecture

Current (`src/components/activation.rs:227`):
```rust
pub const PIECEWISE_NUM_SEGMENTS: usize = 16;
pub const PIECEWISE_SEGMENT_SHIFT: u32 = 27;  // top 4 bits → segment index
```

Each activation evaluation is 1 multiply + 1 add regardless of segment count. Only the
number of committed coefficients changes:

| Segments | Coefficients | Bits for dispatch | Max error (GELU) | Runtime cost |
|----------|-------------|-------------------|-----------------|-------------|
| 16 | 32 M31 values | 4 | ~2-5% | 1 mul + 1 add |
| 64 | 128 M31 values | 6 | ~0.1% | 1 mul + 1 add |
| 256 | 512 M31 values | 8 | ~0.006% | 1 mul + 1 add |
| 1024 | 2048 M31 values | 10 | ~0.001% | 1 mul + 1 add |

### Implementation

1. Make `PIECEWISE_NUM_SEGMENTS` configurable per model registration (not a global constant)
2. Compute `PIECEWISE_SEGMENT_SHIFT` dynamically: `31 - log2(segments)`
3. Store segment count in `ModelRegistration` → part of `model_id` hash
4. Generate coefficients at registration time, commit on-chain

### Files to Modify

| File | Change |
|------|--------|
| `src/components/activation.rs` | `PiecewiseLinearCoeffs` — accept `num_segments` parameter |
| `src/components/activation.rs` | `piecewise_linear_eval()` — dynamic shift instead of constant |
| `src/starknet.rs` | `ModelRegistration` — add `activation_commitment: FieldElement` |
| `src/starknet.rs` | `ModelRegistration` — add `num_piecewise_segments: u32` |
| `src/aggregation.rs` | Proof generation — include activation commitment in io_commitment |
| `src/bin/prove_model.rs` | `--precision {low,medium,high,exact}` CLI flag |

### Precision Presets

| Preset | Segments | Use Case |
|--------|----------|----------|
| `low` | 16 | Development, testing |
| `medium` | 256 | Production (default) |
| `high` | 1024 | Regulated industries |
| `exact` | 4096 | Maximum precision (exceeds f32) |

### Success Criteria

- [ ] `PiecewiseLinearCoeffs::for_activation(act_type, num_segments)` accepts configurable segments
- [ ] `piecewise_linear_eval()` uses dynamic dispatch (no hardcoded shift)
- [ ] `ModelRegistration` includes activation_commitment
- [ ] CLI `--precision` flag works
- [ ] Benchmark: zero performance regression (same 1 mul + 1 add per activation)
- [ ] Accuracy test: 1024 segments → max GELU deviation < 0.002% from f64 reference
- [ ] All 818 tests pass

### Estimated Effort: 1-2 weeks

### Dependency: Phase E (integer-only coefficients required for determinism)

---

## Phase G: Policy Commitment Binding

**Goal**: Mathematically enforce operational guardrails as part of the proof.

**Status**: OPEN.

**Impact**: Proves "model X, under guardrail policy Y, produced output Z for input W."

### Architecture

Extend `ModelRegistration` to include a policy commitment:

```rust
pub struct ModelRegistration {
    pub model_id: FieldElement,
    pub weight_commitment: FieldElement,
    pub activation_commitment: FieldElement,    // NEW (Phase F)
    pub policy_commitment: FieldElement,         // NEW (Phase G)
    pub quantization_commitment: FieldElement,   // NEW (Phase G)
    pub num_layers: usize,
    pub num_piecewise_segments: u32,             // NEW (Phase F)
}
```

The `policy_commitment` is `Poseidon(system_prompt_tokens)` — the hash of the guardrail
system prompt that the enterprise registers on-chain.

The `io_commitment` is extended to include the policy:

```rust
// Current:
io_commitment = Poseidon(packed_inputs || packed_outputs)

// New:
io_commitment = Poseidon(policy_commitment || packed_inputs || packed_outputs)
```

The on-chain verifier checks:
```
proof.policy_commitment == registered.policy_commitment
```

If anyone strips the system prompt, changes the guardrails, or uses a different policy,
the io_commitment won't match and verification fails.

### The `model_id` Becomes the Full Mathematical Definition

```rust
model_id = Poseidon(
    weight_commitment,          // What the model knows
    activation_commitment,      // How it computes (GELU/SiLU coefficients)
    quantization_commitment,    // How it represents numbers (scale/zero_point)
    architecture_hash,          // How it's wired (graph structure)
    policy_commitment,          // What guardrails it operates under
)
```

This is the complete cryptographic identity of the model + its operational constraints.
Any deviation from any component = proof failure.

### Files to Modify

| File | Change |
|------|--------|
| `src/starknet.rs` | `ModelRegistration` — add fields |
| `src/starknet.rs` | `prepare_model_registration()` — compute activation, policy, quantization commitments |
| `src/starknet.rs` | `register_model_calldata()` — include new fields |
| `src/aggregation.rs` | `compute_io_commitment()` — include policy_commitment in hash |
| `src/gadgets/quantize.rs` | `compute_quantization_commitment()` — Poseidon(all QuantParams) |
| `src/bin/prove_model.rs` | `--policy-file <path>` — load system prompt for commitment |
| `elo-cairo-verifier/src/model_verifier.cairo` | Verify policy_commitment matches registration |

### On-Chain Contract Changes

```cairo
// Register model with full identity
fn register_model(
    model_id: felt252,
    weight_commitment: felt252,
    activation_commitment: felt252,
    policy_commitment: felt252,
    quantization_commitment: felt252,
    num_layers: u32,
) { ... }

// Verify proof matches registered model
fn verify_model_gkr(...) {
    // Existing checks
    assert(proof.weight_commitment == registered.weight_commitment);

    // NEW checks
    assert(proof.activation_commitment == registered.activation_commitment);
    assert(proof.policy_commitment == registered.policy_commitment);
}
```

### Success Criteria

- [ ] `ModelRegistration` includes all 5 commitments
- [ ] `model_id` hash includes all commitments
- [ ] `io_commitment` includes `policy_commitment`
- [ ] `--policy-file` CLI flag works
- [ ] Tamper test: altered system prompt → proof fails
- [ ] Tamper test: missing policy → io_commitment mismatch
- [ ] Cairo verifier checks policy_commitment against registration
- [ ] All 818 tests pass

### Estimated Effort: 2 weeks

---

## Phase H: Incremental Delta Proving

**Goal**: Prove only new tokens per conversation turn, not the full context.

**Status**: OPEN — foundation exists (decode-step architecture), needs chunk extension.

**Impact**: Sustained ~200 proven tok/s (vs ~40 tok/s without incremental).

### Architecture

The decode-step infrastructure already exists:
- `prove_model_pure_gkr_decode_step_incremental()` proves 1 new token with KV-cache chain
- `IncrementalPoseidonMerkle` tracks KV state with O(log N) appends
- `IncrementalKVCommitment` maintains per-layer Merkle trees
- `prove_model_pure_gkr_decode_sequence()` chains multiple decode steps

The extension: allow decode steps to process **chunks of new tokens** (not just 1).

### Current Decode Path (seq_len=1)

```
prove_model_pure_gkr_decode_step_incremental():
  1. Assert input.rows == 1
  2. Forward pass using attention_forward_cached()
  3. GKR proof for single token
  4. KV commit: append_step(cache, 1)
  5. Return (proof, new_kv_commitment)
```

### Target: Chunk Decode Path (seq_len=chunk_size)

```
prove_model_pure_gkr_decode_chunk_incremental():
  1. Accept input.rows == chunk_size (e.g., 3000)
  2. Forward pass: process chunk through all layers with cached KV
  3. Attention: new chunk scores against full cache + self
     - Score: (chunk, d_k) × (d_k, cache_len + chunk) → O(chunk × total)
     - Context: (chunk, cache_len + chunk) × (cache_len + chunk, d_k)
  4. GKR proof for chunk (larger matmuls, more sumcheck rounds)
  5. KV commit: append_step(cache, chunk_size) — batch append
  6. Return (proof, new_kv_commitment)
```

### Attention Cost: Chunk vs Full Prefill

| Scenario | Score matmul | Cost | vs Full Prefill |
|----------|-------------|------|----------------|
| Full prefill (20K tokens) | (20K, 128) × (128, 20K) | O(20K²) = 400M/head | baseline |
| Chunk decode (3K new at pos 20K) | (3K, 128) × (128, 23K) | O(3K × 23K) = 69M/head | **5.8x cheaper** |
| Chunk decode (1K new at pos 20K) | (1K, 128) × (128, 21K) | O(1K × 21K) = 21M/head | **19x cheaper** |

### Files to Modify

| File | Change |
|------|--------|
| `src/aggregation.rs` | `prove_model_pure_gkr_decode_chunk_incremental()` — new function |
| `src/aggregation.rs` | Remove `assert!(input.rows == 1)` constraint in decode path |
| `src/components/attention.rs` | `attention_forward_cached_chunk()` — chunk×cache attention |
| `src/gkr/prover.rs` | `reduce_attention_layer_decode_chunk()` — chunk-aware attention reduction |
| `src/bin/prove_model.rs` | `--decode-chunk-size N` CLI flag |

### KV-Cache Batch Append

`IncrementalKVCommitment::append_step()` already accepts a `new_tokens` count.
For chunk_size=3000, it appends 3000 leaves to each layer's Merkle tree.
Cost: 3000 × log₂(capacity) × num_layers Poseidon hashes. At capacity 40K, 40 layers:
3000 × 16 × 40 = 1.92M hashes. At ~1μs/hash on CPU: ~1.9 seconds. Acceptable.

### Proof Chain Integrity

Each chunk proof includes:
- `prev_kv_cache_commitment` — must match previous proof's `kv_cache_commitment`
- `kv_cache_commitment` — new state after appending chunk
- `position_offset` — starting position of chunk in full sequence

The chain is validated by `prove_model_pure_gkr_decode_sequence()`:
```rust
proof[i].kv_cache_commitment == proof[i+1].prev_kv_cache_commitment
```

### Conversation Flow Example

```
Turn 1: prefill 2K tokens          → full proof, ~22s (optimized)
Turn 2: chunk 1.5K new tokens      → delta proof, ~14s
Turn 3: chunk 2K new tokens        → delta proof, ~16s
Turn 4: chunk 3K new tokens        → delta proof, ~18s
...
Turn 20: chunk 4K new tokens       → delta proof, ~22s

Total: 22s + 19 × ~18s avg = ~364s ≈ 6 minutes
Context: ~35K tokens, every turn proven
Throughput: 35K / 364s ≈ 96 tok/s (sustained)
```

With GPU forward pass (Phase A): delta proofs drop to ~12-15s avg → ~200 tok/s sustained.

### Success Criteria

- [ ] `prove_model_pure_gkr_decode_chunk_incremental()` works for chunk_size > 1
- [ ] KV commitment chain validates across chunks
- [ ] Attention reduction handles (chunk, cache_len) asymmetric matmuls
- [ ] Benchmark: 3K-token chunk at position 20K < 20s on H100 (with Phase A)
- [ ] Benchmark: sustained throughput > 150 tok/s across 20-turn conversation
- [ ] All decode_benchmark tests pass
- [ ] Cairo on-chain verification accepts chunk proofs

### Estimated Effort: 3-4 weeks

### Dependencies: Phase A (GPU forward pass) for target throughput numbers

---

## Phase I: Private I/O Verification

**Goal**: Decouple raw input/output data from on-chain calldata. Verify via commitment only.

**Status**: OPEN.

**Impact**: Users can prove inference without revealing conversation content on-chain.

### Current State

The on-chain verifier receives `raw_io_data` (packed inputs and outputs) in calldata.
It reconstructs the MLE from this data and checks against the GKR claim. This means
anyone reading the blockchain sees the exact inputs and outputs.

### Target Architecture

```
Current:
  Prover sends: io_commitment (hash) + raw_io_data (PUBLIC)
  Verifier:     Rebuilds MLE from raw_io_data, checks against commitment

Private:
  Prover sends: io_commitment (hash) + mle_evaluation_at_challenge (single field element)
  Verifier:     Checks MLE evaluation is consistent with commitment + GKR claims
```

The verifier doesn't need raw I/O. It only needs the MLE evaluation at the random
challenge point produced by the Fiat-Shamir transcript. The `io_commitment` binds the
proof to specific I/O without revealing it.

### Files to Modify

| File | Change |
|------|--------|
| `src/starknet.rs` | `build_verify_model_gkr_calldata()` — option to omit raw_io_data |
| `src/aggregation.rs` | `compute_io_commitment()` — standalone commitment proof |
| `elo-cairo-verifier/src/verifier.cairo` | Accept commitment + evaluation instead of raw data |
| `src/bin/prove_model.rs` | `--private-io` flag |

### Success Criteria

- [ ] `--private-io` flag omits raw I/O from calldata
- [ ] On-chain verifier accepts commitment + MLE evaluation
- [ ] Proof size reduction: ~1281 packed felts → ~10 felts (io section)
- [ ] Tamper test: wrong commitment → rejected
- [ ] Full GKR verification still passes

### Estimated Effort: 2 weeks

---

## Phase J: Baseline Throughput Benchmark (Sprint 0)

**Goal**: Establish the verified "100 tokens/second today" claim by benchmarking at seq_len=10000.

**Status**: PREREQUISITE — must run before any optimization work begins.

**Impact**: Establishes the published baseline that all future improvements reference.

### What Needs to Happen

The current 103s benchmark runs at `seq_len=1` (single token). We claim ~100 tok/s is
achievable today at `seq_len=10000` based on logarithmic sumcheck scaling. This must be
measured, not estimated.

### Benchmark Matrix

| seq_len | Expected proof time | Expected tok/s | Must measure |
|---------|-------------------|---------------|-------------|
| 1 | 103s (known) | 0.01 | Already benchmarked |
| 100 | ~104-106s | ~1 | Yes |
| 1,000 | ~108-112s | ~9 | Yes |
| 5,000 | ~115-120s | ~42-43 | Yes |
| 10,000 | ~120-130s | ~77-83 | **Yes — critical** |
| 20,000 | ~130-150s | ~133-154 | Yes |
| 40,000 (max) | ~150-200s | ~200-267 | Yes (if memory allows) |

### Implementation

```bash
# Run on H100 NVL with cached weights
for seq_len in 1 100 1000 5000 10000 20000 40000; do
  ./target/release/prove-model \
    --model-dir ~/.obelysk/models/qwen3-14b \
    --layers 40 \
    --gkr \
    --format ml_gkr \
    --seq-len $seq_len \
    --bench \
    --output /tmp/bench_seq${seq_len}.json
done
```

### Files to Modify

| File | Change |
|------|--------|
| `src/bin/prove_model.rs` | Add `--seq-len N` flag to override default seq_len=1 |
| `src/compiler/hf_loader.rs` | `load_hf_model_full()` — accept seq_len parameter from CLI |
| `src/bin/prove_model.rs` | Add `--bench` flag — report wall clock + tok/s |

### Memory Constraints

| seq_len | Input matrix size | Attention per head | Estimated peak VRAM |
|---------|------------------|-------------------|-------------------|
| 1,000 | 5MB | 1M elements | ~20 GB |
| 10,000 | 50MB | 100M elements | ~40 GB |
| 20,000 | 100MB | 400M elements | ~60 GB |
| 40,000 | 200MB | 1.6B elements | ~80 GB (H100 limit) |

### Success Criteria

- [ ] `--seq-len` and `--bench` flags implemented
- [ ] Benchmark results for all 7 seq_len values published
- [ ] Confirmed: seq_len=10000 achieves ≥80 tok/s on H100
- [ ] Results documented in GETTING_STARTED.md performance table

### Estimated Effort: 3-5 days

---

## Phase K: Production Deployment Model (Commit / Sample / Prove)

**Goal**: Define the three-tier production architecture for enterprise model integrity at scale.

**Status**: OPEN.

**Impact**: Makes model integrity practical for enterprises with 50K+ tasks/day without requiring
full proof of every turn.

### Architecture: Three Tiers

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: COMMIT (every turn, always, ~$0/day)                │
│                                                             │
│ Every inference turn produces:                              │
│   io_hash = Poseidon(input_embedding || output_logits)      │
│   chain = Poseidon(prev_chain || io_hash || timestamp)      │
│                                                             │
│ Chain head posted on-chain periodically (hourly/daily)      │
│ Cost: 1 TX/period (~$0.01)                                  │
│ Guarantee: tamper evidence — any historical alteration       │
│            breaks the chain                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 2: SAMPLE (background, asynchronous)                   │
│                                                             │
│ Random selection of turns for full GKR proof:               │
│   - Critical actions (deploy, transact): 100%               │
│   - Customer-facing decisions: 10-25%                       │
│   - Internal operations: 1-5%                               │
│   - Freshness: at least 1 proof/hour                        │
│                                                             │
│ Selection is unpredictable (Fiat-Shamir from chain hash)    │
│ Cost: ~$2-10/day depending on sample rate and hardware      │
│ Guarantee: statistical — cheating on >1% of turns detected  │
│            with >63% probability per audit window            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: PROVE (on-demand, synchronous when needed)          │
│                                                             │
│ Full GKR proof before critical actions:                     │
│   proof = prove_turn(input, model, weights)                 │
│   if proof.verifies(): execute_action()                     │
│   else: HALT — integrity violation                          │
│                                                             │
│ Also triggered by: disputes, audits, regulatory requests    │
│ Cost: ~$0.10-0.50 per proof (single turn on H100)           │
│ Guarantee: mathematical — STARK proof, unchallengeable      │
└─────────────────────────────────────────────────────────────┘
```

### Commitment Chain Implementation

#### Data Structures

```rust
/// Per-turn commitment record
pub struct TurnCommitment {
    pub turn_id: u64,
    pub timestamp: u64,
    pub model_id: FieldElement,          // registered model identity
    pub policy_commitment: FieldElement,  // guardrails hash
    pub io_hash: FieldElement,           // Poseidon(input || output)
    pub chain_hash: FieldElement,        // Poseidon(prev_chain || io_hash || timestamp)
    pub proof_status: ProofStatus,       // Committed | Sampled | Proven
}

pub enum ProofStatus {
    Committed,                           // Hash chain only — no proof generated
    Sampled { proof_hash: FieldElement }, // Selected for sampling, proof generated
    Proven { proof: GKRProof },          // Full proof available
}

/// Commitment chain state
pub struct CommitmentChain {
    pub chain_head: FieldElement,
    pub turn_count: u64,
    pub last_anchor_tx: Option<String>,  // on-chain TX hash
    pub last_anchor_time: u64,
}
```

#### On-Chain Anchor Contract

```cairo
/// Anchor the commitment chain head on-chain
fn anchor_chain(
    model_id: felt252,
    chain_head: felt252,
    turn_count: u64,
    timestamp: u64,
) {
    // Verify model is registered
    assert(is_registered(model_id));
    // Store chain head (overwrite previous)
    chain_anchors::write(model_id, ChainAnchor { chain_head, turn_count, timestamp });
    // Emit event for indexers
    emit ChainAnchored { model_id, chain_head, turn_count, timestamp };
}

/// Verify a specific turn's membership in the anchored chain
fn verify_chain_membership(
    model_id: felt252,
    turn_commitment: TurnCommitment,
    merkle_proof: Span<felt252>,     // path from turn to chain_head
) -> bool { ... }
```

### Sampling Strategy

The sampling selector must be unpredictable to prevent gaming:

```rust
/// Determine if a turn should be sampled for proving
pub fn should_sample(turn: &TurnCommitment, config: &SamplingConfig) -> bool {
    // Critical actions always proven
    if turn.is_critical_action() { return true; }

    // Deterministic but unpredictable: hash(chain_state || turn_id)
    let selector = poseidon_hash(&[turn.chain_hash, FieldElement::from(turn.turn_id)]);
    let threshold = config.sample_rate_bps; // basis points (100 = 1%)

    // selector mod 10000 < threshold → selected
    (selector.0 % 10000) < threshold as u32
}
```

### Files to Create/Modify

| File | Change |
|------|--------|
| `src/commitment_chain.rs` | **NEW** — `CommitmentChain`, `TurnCommitment`, chain operations |
| `src/sampling.rs` | **NEW** — `SamplingConfig`, `should_sample()`, sampling strategy |
| `src/audit/orchestrator.rs` | Integrate commitment chain into audit pipeline |
| `src/bin/prove_model.rs` | `--commit-chain` mode, `--sample-rate N` flag |
| `elo-cairo-verifier/src/lib.cairo` | `anchor_chain()`, `verify_chain_membership()` |

### Enterprise Configuration

```toml
# obelysk.toml — enterprise proving configuration
[commitment]
chain_enabled = true
anchor_interval = "1h"           # post chain head on-chain every hour
anchor_contract = "0x0121d1..."  # Starknet contract address

[sampling]
critical_actions = 1.0           # 100% — always prove
customer_facing = 0.10           # 10%
internal = 0.01                  # 1%
freshness_interval = "1h"        # at least 1 proof per hour

[proving]
gpu_budget = 4                   # max GPUs for proving
off_peak_only = true             # only prove during off-peak hours
priority = "critical_first"      # prove critical actions immediately
```

### Cost Analysis

| Enterprise Size | Turns/day | Commitments | Sampled Proofs | On-Demand Proofs | Daily Cost |
|----------------|-----------|-------------|---------------|-----------------|-----------|
| Small (10 agents) | 1,500 | 1,500 (free) | ~30 (2%) | ~5 | ~$3 |
| Medium (100 agents) | 15,000 | 15,000 (free) | ~300 (2%) | ~50 | ~$25 |
| Large (1000 agents) | 150,000 | 150,000 (free) | ~3,000 (2%) | ~500 | ~$250 |
| Enterprise (5000 agents) | 750,000 | 750,000 (free) | ~15,000 (2%) | ~2,500 | ~$1,250 |

### Success Criteria

- [ ] `CommitmentChain` correctly chains all turns with Poseidon
- [ ] On-chain anchor contract deployed and tested
- [ ] `should_sample()` produces unpredictable but deterministic selection
- [ ] `obelysk.toml` configuration works for all tiers
- [ ] E2e test: 100 turns committed → 5 sampled → 2 proven on-demand → chain verifies
- [ ] Tamper test: alter historical turn → chain breaks → detected

### Estimated Effort: 3-4 weeks

---

## Phase L: Recursive Proof Integration with Incremental Proving

**Goal**: Compose incremental delta proofs into constant-size recursive proofs for on-chain submission.

**Status**: OPEN — depends on Phase 4A (recursive STARK, in progress on `feat/recursive-stark`).

**Impact**: 1 on-chain TX per epoch covering hundreds of proven turns, regardless of how many.

### Architecture

Phase 4A delivers recursive STARK: compress one GKR proof (~87K felts) into ~500 felts.
This phase extends recursion to batch incremental delta proofs:

```
Turn 1: delta proof (GKR)  ──┐
Turn 2: delta proof (GKR)  ──┤
Turn 3: delta proof (GKR)  ──┼──→ Recursive fold ──→ 1 STARK proof (~500 felts)
...                           │                       ──→ 1 TX on Starknet
Turn N: delta proof (GKR)  ──┘
```

### Folding Strategy

**Option A: Sequential folding (simple)**
```
proof_1 → recursive_1
(recursive_1, proof_2) → recursive_2
(recursive_2, proof_3) → recursive_3
...
```
Cost: N × ~45s recursive proving. For N=100: ~75 minutes.

**Option B: Tree folding (parallel)**
```
(proof_1, proof_2)   → recursive_12
(proof_3, proof_4)   → recursive_34
(recursive_12, recursive_34) → recursive_1234
...
```
Cost: log₂(N) levels × ~45s. For N=100: ~7 levels × ~45s = ~5 minutes.
Parallelizable across GPUs at each level.

**Option C: Accumulation (IVC-style, best)**
```
acc_0 = proof_1
acc_1 = fold(acc_0, proof_2)    // ~5-10s per fold
acc_2 = fold(acc_1, proof_3)
...
acc_N = fold(acc_{N-1}, proof_N)
finalize(acc_N) → recursive STARK proof
```
Cost: N × ~5-10s fold + 1 × ~45s finalize. For N=100: ~500s fold + 45s = ~9 minutes.
Runs incrementally — each turn folds into the accumulator immediately.

### Recommended: Option C (Accumulation)

The accumulator approach is best for streaming:
- Each delta proof folds into the accumulator in ~5-10s
- The accumulator is always up-to-date
- Finalize on-demand when ready to anchor on-chain
- 1 TX per epoch regardless of turn count

### Files to Create/Modify

| File | Change |
|------|--------|
| `src/recursive/accumulator.rs` | **NEW** — `ProofAccumulator`, `fold()`, `finalize()` |
| `src/recursive/mod.rs` | Export accumulator API |
| `src/commitment_chain.rs` | Integrate accumulator with commitment chain |
| `src/bin/prove_model.rs` | `--accumulate` mode — fold each proof into running accumulator |

### Integration with Commitment Chain (Phase K)

```
Commitment chain runs continuously (all turns)
Sampling selects turns for full proof (Tier 2)
Each proven turn's GKR proof folds into accumulator
Periodically: finalize accumulator → recursive STARK → 1 TX on-chain

On-chain TX proves: "N turns were proven correct, accumulator valid"
Chain anchor proves: "all M turns are committed, N of them proven"
```

### Success Criteria

- [ ] Phase 4A (recursive STARK) complete and working
- [ ] `ProofAccumulator::fold()` correctly accumulates delta proofs
- [ ] `ProofAccumulator::finalize()` produces valid recursive STARK
- [ ] Tree folding works for parallel proving
- [ ] Benchmark: fold < 10s per proof, finalize < 60s
- [ ] E2e: 20 delta proofs → accumulate → finalize → 1 TX → verified on-chain
- [ ] Integration with commitment chain and sampling

### Estimated Effort: 4-5 weeks

### Dependencies: Phase 4A (recursive STARK), Phase H (incremental delta proving)

---

## Phase M: Consumer Hardware Optimization

**Goal**: Optimize proving for individual developers on RTX 4090 and Apple Silicon.

**Status**: OPEN.

**Impact**: Individual developers can prove their own models locally without H100 access.

### RTX 4090 Path (CUDA)

The RTX 4090 has 24GB VRAM and ~82 TFLOPS FP32. Our CUDA kernels already run on it.
The bottleneck is memory, not compute.

| Model | Full proof (current) | Target | VRAM needed |
|-------|---------------------|--------|-------------|
| GPT-2 (124M, 12L) | ~20-30s | ~8-12s | ~4 GB |
| Qwen2-0.5B (12L) | ~20-30s | ~8-12s | ~8 GB |
| Phi-3 Mini (3.8B, 32L) | ~80-100s | ~30-40s | ~20 GB |
| Qwen3-14B (1-2L) | ~3-6s | ~2-3s | ~12 GB |

#### Optimizations

1. **Memory-efficient forward pass**: Stream intermediates layer by layer (don't store all 40 layers)
2. **Lower GPU Merkle threshold**: Tune `STWO_GPU_MERKLE_THRESHOLD` for 24GB cards
3. **Mixed-precision MLE**: Use u16 for MLE storage where possible (2x memory savings)
4. **Checkpoint/resume**: Save proving state to disk, resume across multiple GPU sessions

### Apple Silicon Path (Metal)

Metal compute shaders exist in `src/metal/`:
- `matmul.rs` — M31 GEMM
- `sumcheck.rs` — sumcheck round polynomial
- `dispatch.rs` — auto-dispatch with size thresholds

#### Optimizations

1. **Unified memory advantage**: M-series has shared CPU/GPU memory — zero-copy MLE access
2. **Metal Performance Shaders**: Leverage MPS for matrix operations where applicable
3. **Tile-based proving**: Break large proofs into GPU-sized tiles for M1/M2 (16GB)
4. **M4 Pro/Max targeting**: M4 Max has 128GB unified memory — can prove Phi-3 Mini easily

### Target Performance

| Hardware | Model | Target proof time | Target tok/s (seq_len=1K) |
|----------|-------|------------------|--------------------------|
| RTX 4090 | GPT-2 (124M) | ~10s | ~100 |
| RTX 4090 | Phi-3 Mini (3.8B) | ~35s | ~29 |
| M4 Max (128GB) | Phi-3 Mini (3.8B) | ~50s | ~20 |
| M4 Pro (24GB) | GPT-2 (124M) | ~15s | ~67 |
| M2 (16GB) | GPT-2 (124M) | ~25s | ~40 |

### Files to Modify

| File | Change |
|------|--------|
| `src/metal/matmul.rs` | Optimize for unified memory (zero-copy) |
| `src/metal/sumcheck.rs` | Lower dispatch threshold for M-series |
| `src/gpu_sumcheck.rs` | Memory-efficient mode for 24GB cards |
| `src/aggregation.rs` | Layer-streaming forward pass (don't store all intermediates) |
| `src/bin/prove_model.rs` | `--memory-budget N` flag to limit GPU memory usage |

### Success Criteria

- [ ] RTX 4090: Phi-3 Mini full proof < 40s
- [ ] RTX 4090: GPT-2 full proof < 12s
- [ ] M4 Max: Phi-3 Mini full proof < 55s
- [ ] `--memory-budget` flag works (limits VRAM usage)
- [ ] Layer-streaming forward pass reduces peak memory by 50%+
- [ ] Metal path benchmarked on M1, M2, M4

### Estimated Effort: 3-4 weeks

---

## Phase Sequencing and Dependencies

```
SPRINT 0 (prerequisite):
┌──────────┐
│ Phase J  │  Baseline Benchmark — establish "100 tok/s today"
│ (3-5 d)  │  No dependencies
└──────────┘

PERFORMANCE TRACK:
┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐
│ Phase A  │───────→│ Phase B  │───────→│ Phase H  │───────→│ Phase L  │
│ GPU Fwd  │        │ GKR Para │        │ Incremnl │        │ Recursive│
│ (2 wk)   │        │ (2-3 wk) │        │ (3-4 wk) │        │ (4-5 wk) │
└──────────┘        └──────────┘        └──────────┘        └──────────┘
                                              ↑
ACCURACY TRACK:                               │ (incremental needs determinism)
┌──────────┐        ┌──────────┐              │
│ Phase E  │───────→│ Phase F  │──────────────┘
│ Platform │        │ HiPrec   │
│ (3-4 wk) │        │ (1-2 wk) │
└──────────┘        └──────────┘
                         │
INTEGRITY TRACK:         │
                         ▼
                    ┌──────────┐        ┌──────────┐
                    │ Phase G  │───────→│ Phase K  │
                    │ Policy   │        │ Commit/  │
                    │ (2 wk)   │        │ Sample   │
                    └──────────┘        │ (3-4 wk) │
                                        └──────────┘

INDEPENDENT (can run in parallel with any track):
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Phase D  │    │ Phase I  │    │ Phase M  │
│ Binary   │    │ Private  │    │ Consumer │
│ (1 wk)   │    │ I/O      │    │ Hardware │
│          │    │ (2 wk)   │    │ (3-4 wk) │
└──────────┘    └──────────┘    └──────────┘

BLOCKED:
┌──────────┐
│ Phase C  │  GPU STARK — blocked on STWO fix
│ (1 wk)   │
└──────────┘
```

### Recommended Execution Order

| Sprint | Phases | Duration | Milestone |
|--------|--------|----------|-----------|
| Sprint 0 | **J** (baseline benchmark) | 3-5 days | Published: 100 tok/s at seq_len=10K |
| Sprint 1 | **A** (GPU forward) + **D** (binary serialization) | 2 weeks | Forward pass < 8s, binary proofs |
| Sprint 2 | **E** (platform determinism) + **I** (private I/O) | 3-4 weeks | Bit-identical cross-platform, I/O privacy |
| Sprint 3 | **B** (GKR parallelism) + **F** (high-precision rails) | 3 weeks | GKR < 22s, 0.001% activations |
| Sprint 4 | **G** (policy commitment) + **M** (consumer hardware) | 3-4 weeks | Guardrail enforcement, RTX 4090/Metal |
| Sprint 5 | **H** (incremental delta) | 3-4 weeks | Sustained ~200 tok/s |
| Sprint 6 | **K** (commit/sample/prove) + **C** (GPU STARK if ready) | 3-4 weeks | Production deployment model |
| Sprint 7 | **L** (recursive integration) + final benchmarks | 4-5 weeks | 1 TX/epoch, ~500 tok/s peak confirmed |

**Total: ~22-28 weeks (5-7 months)**

### Track Parallelism

Sprints 2-4 can overlap across tracks if team capacity allows:
- **Performance track** (A → B → H → L): throughput-focused engineer(s)
- **Accuracy track** (E → F): precision/determinism engineer(s)
- **Integrity track** (G → K): on-chain/contract engineer(s)
- **Independent** (D, I, M): can be assigned to any available engineer

With 3 parallel engineers, total wall clock reduces to **~16-20 weeks (4-5 months)**.

---

## Benchmark Suite

### Performance Benchmarks (run after each phase)

| Benchmark | Command | Target |
|-----------|---------|--------|
| Full prefill (seq_len=1) | `prove-model --layers 40 --gkr --bench` | < 22s |
| Batched prefill (seq_len=10K) | `prove-model --layers 40 --gkr --seq-len 10000 --bench` | < 31s |
| Batched prefill (seq_len=20K) | `prove-model --layers 40 --gkr --seq-len 20000 --bench` | < 45s |
| Incremental delta (3K at pos 20K) | `prove-model --decode-chunk 3000 --cache-pos 20000 --bench` | < 18s |
| Single layer | `prove-model --layers 1 --gkr --bench` | < 1s |
| Peak tok/s | `prove-model --layers 40 --gkr --seq-len 20000 --bench --report-throughput` | > 450 tok/s |
| Sustained tok/s | `prove-model --decode-session 20 --bench --report-throughput` | > 150 tok/s |

### Accuracy Benchmarks

| Benchmark | Command | Target |
|-----------|---------|--------|
| GELU piecewise error (16 seg) | `prove-model --test-activation-precision --segments 16` | < 5% |
| GELU piecewise error (256 seg) | `prove-model --test-activation-precision --segments 256` | < 0.01% |
| GELU piecewise error (1024 seg) | `prove-model --test-activation-precision --segments 1024` | < 0.002% |
| Cross-platform determinism | `prove-model --test-determinism --platforms x86,arm` | 0 diff |
| Quantization roundtrip | `prove-model --test-quantization-roundtrip` | < 0.02 |

### Integrity Benchmarks

| Benchmark | Target |
|-----------|--------|
| Weight tamper → rejection | PASS |
| Activation tamper → rejection | PASS |
| Policy mismatch → rejection | PASS |
| IO alteration → rejection | PASS |
| KV chain break → rejection | PASS |

---

## All Phases Summary

| Phase | Name | Effort | Key Deliverable |
|-------|------|--------|----------------|
| **J** | Baseline Benchmark | 3-5 days | Published 100 tok/s at seq_len=10K |
| **A** | GPU Forward Pass | 2 weeks | 38s → 5-8s (wire existing kernels) |
| **B** | GKR Walk Parallelism | 2-3 weeks | 47s → 15-20s (multi-stream sumcheck) |
| **C** | GPU Unified STARK | 1 week | 5s → 1-2s (blocked on STWO) |
| **D** | Binary Serialization | 1 week | 2s → 0.3s, 7MB → 3.6MB |
| **E** | Platform Determinism | 3-4 weeks | Bit-identical M31 on every platform |
| **F** | High-Precision Rails | 1-2 weeks | ~2-5% → ~0.001% activation precision |
| **G** | Policy Commitment | 2 weeks | Guardrail enforcement on-chain |
| **H** | Incremental Delta Proving | 3-4 weeks | Sustained ~200 tok/s |
| **I** | Private I/O Verification | 2 weeks | Hide I/O from chain |
| **J** | Baseline Benchmark | 3-5 days | Establish published baseline |
| **K** | Commit/Sample/Prove | 3-4 weeks | Production deployment model |
| **L** | Recursive + Incremental | 4-5 weeks | 1 TX/epoch, proof accumulation |
| **M** | Consumer Hardware | 3-4 weeks | RTX 4090 + Apple Silicon targets |

---

## Summary: Before and After

### Performance

| Metric | Today | After All Phases |
|--------|-------|-----------------|
| Full prefill proof time | 92s (warm) | **~22s** |
| Peak throughput (1 H100) | ~100 tok/s | **~500 tok/s** |
| Sustained throughput (1 H100) | ~40 tok/s (est.) | **~200 tok/s** |
| 8× H100 sustained | ~320 tok/s (est.) | **~1,600 tok/s** |
| 8× B300 sustained | N/A | **~5,760 tok/s** |
| Cost per M proven tokens (8× H100) | ~$6.94 | **~$1.74** |
| On-chain TXs per epoch | 6-18 per proof | **1 (recursive accumulation)** |
| Consumer hardware (RTX 4090, Phi-3) | ~80-100s | **~35s** |
| Apple Silicon (M4 Max, Phi-3) | Untested | **~50s** |

### Accuracy

| Metric | Today | After All Phases |
|--------|-------|-----------------|
| Activation precision (GELU/SiLU) | ~2-5% | **~0.001%** |
| Platform determinism | Not guaranteed | **Bit-identical everywhere** |
| Unverified operations | Phase 1E gap | **None** |
| Guardrail enforcement | None | **Policy commitment on-chain** |
| Accuracy enforcement model | Verify after the fact | **Impossible to deviate — rails define computation** |

### Model Identity

| Component | Today | After All Phases |
|-----------|-------|-----------------|
| Weight commitment | Yes | Yes |
| Activation commitment | No | **Yes (Phase F)** |
| Policy commitment | No | **Yes (Phase G)** |
| Quantization commitment | No | **Yes (Phase G)** |
| Full model_id hash | Partial | **Complete cryptographic identity** |

### Production Deployment

| Capability | Today | After All Phases |
|-----------|-------|-----------------|
| Commitment chain | Not implemented | **All turns committed (free)** |
| Sampling strategy | Manual audit pipeline | **Configurable per-enterprise tiers** |
| On-demand proving | CLI only | **Integrated with commitment chain** |
| Recursive batching | Not implemented | **Accumulator → 1 TX/epoch** |
| Enterprise config | Env vars | **obelysk.toml declarative config** |

### Privacy

| Aspect | Today | After All Phases |
|--------|-------|-----------------|
| Weights | Hidden (Poseidon commitment) | Hidden |
| Intermediates | Hidden (MLE commitment) | Hidden |
| I/O data | **PUBLIC in calldata** | **Private (Phase I)** |
| Hardware attestation | Optional (TEE) | Optional (TEE) |

### Consumer Hardware

| Hardware | Today | After All Phases |
|----------|-------|-----------------|
| RTX 4090 (Phi-3 Mini) | ~80-100s | **~35s** |
| RTX 4090 (GPT-2) | ~20-30s | **~10s** |
| M4 Max (Phi-3 Mini) | Untested | **~50s** |
| M2 (GPT-2) | Untested | **~25s** |
| CPU-only (any model, 1L) | ~30-150s | **~20-80s** |
