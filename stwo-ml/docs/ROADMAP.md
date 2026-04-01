# ObelyZK Protocol Roadmap

**Version**: 2.1 | **Date**: March 21, 2026 | **Author**: Bitsage Network

> Strategy: Deep research per phase, implement, test end-to-end on H100 GPU, verify all security properties, then advance to next phase.

---

## Current Baseline (March 16, 2026)

| Metric | Value |
|--------|-------|
| Model | Qwen3-14B (9.2B params, 40 layers, 160 MatMuls) |
| Prove time (cached) | **103s** on H100 NVL |
| Audit (3 inferences) | **5m 11s** |
| Security tests | 41/41 pass (18 tamper + 23 integration gates) |
| On-chain | Starknet Sepolia, **6-TX streaming v25, v39 Cairo verifier (6/6 SUCCEEDED)** |
| Proof system | GKR sumcheck + Poseidon2-M31 commitments (no FRI) |
| Field | M31 -> CM31 -> QM31 (124-bit algebraic security) |

---

## Phase 1: Close Soundness Gaps

**Goal**: Every arithmetic operation in a transformer forward pass is provably verified. No unverified operations remain.

**Success criteria**: A malicious prover cannot fabricate any intermediate value without detection.

### 1A. LayerNorm/RMSNorm Mean & Variance Verification

**Status**: CLOSED (verified March 2026, confirmed March 16 2026).

The prover implements a 3-part protocol:
- Part 0: Plain sumcheck binding mean/variance to input (Σx/n for mean, Σ(x-μ)²/n for variance)
- Part 1: eq-sumcheck for output = (input - mean) * rsqrt
- Part 2: LogUp eq-sumcheck for (variance, rsqrt) table lookup

Tamper tests confirm: `test_layernorm_multi_row_tampered_mean_rejected`, `test_rmsnorm_multi_row_tampered_rms_sq_rejected`, `test_layernorm_logup_none_rejected`.

**No further work needed.** Previously listed as a gap in the paper — paper corrected.

**Current behavior**: LayerNorm reduction accepts mean and variance as trace inputs without constraining them to equal the actual statistics of the input vector.

**Attack**: Malicious prover computes correct MatMul outputs but substitutes arbitrary normalization parameters, producing different outputs while passing verification.

**Protocol fix**:

1. **Mean verification** via inner-product sumcheck:
   - Claim: `mean = (1/n) * sum(x_i)` for input vector x of length n
   - Reduce to: `n * mean = sum(x_i)`
   - Express as inner product: `<x, ones> = n * mean`
   - Standard inner-product sumcheck (log(n) rounds)
   - Cost: ~log(5120) = 13 sumcheck rounds per norm layer

2. **Variance verification** via degree-3 eq-sumcheck:
   - Claim: `variance = (1/n) * sum((x_i - mean)^2)`
   - Expand: `n * variance = sum(x_i^2) - 2*mean*sum(x_i) + n*mean^2`
   - Since `sum(x_i) = n*mean` (from step 1): `n * variance = sum(x_i^2) - n*mean^2`
   - Reduce to: prove `sum(x_i^2)` via sumcheck over MLE of x^2
   - Cost: ~13 sumcheck rounds + 1 MLE evaluation

3. **Integration**:
   - Add `MeanVarianceProof` to `LayerProof::LayerNorm` and `LayerProof::RMSNorm`
   - Verifier replays sumcheck and checks consistency
   - Mix proof into Fiat-Shamir channel between norm reduction and next layer

**Files to modify**:
- `src/gkr/prover.rs`: Add mean/variance sumcheck after norm reduction
- `src/gkr/verifier.rs`: Verify mean/variance proof
- `src/gkr/types.rs`: Add `MeanVarianceProof` type
- `src/components/layernorm.rs`: Compute and export witness
- `src/components/rmsnorm.rs`: Same (variance only, no mean for RMSNorm)

**Estimated calldata overhead**: ~200 felts per norm layer (13 rounds * 3 coefficients + evaluations). For 81 RMSNorm + 0 LayerNorm in Qwen3-14B: ~16,200 extra felts (~19% increase).

**Estimated proving overhead**: ~1-2s total (81 norm layers * ~15ms each).

**Test plan**:
- Unit test: prove mean/variance for known vector, verify
- Tamper test: corrupt mean, verify rejection
- Tamper test: corrupt variance, verify rejection
- Integration: full 40-layer proof with mean/variance constraints
- H100 benchmark: measure overhead vs baseline

---

### 1B. Softmax Sum Constraint

**Status**: CRITICAL GAP — verifier accepts any `sum_exp` value without checking it equals the actual sum of exponentials.

**Current behavior**: `SoftmaxNormEval` constraint checks `weights[i] * sum_exp == exp_values[i]` but `sum_exp` is provided by the prover as an input, not derived from `exp_values`.

**Attack**: Malicious prover claims `sum_exp = 1`, making `weights[i] = exp_values[i]` (unnormalized softmax). Verifier accepts.

**Additional bug**: If `sum_exp == 0 (mod P)`, the forward pass returns unnormalized exp values (attention.rs line 856-863). This is a correctness bug independent of the soundness gap.

**Protocol fix**:

1. **Sum accumulator column**: Add a running-sum interaction column to the SoftmaxNorm STARK component that proves `sum_exp = sum(exp_values[0..seq_len])`.

2. **Implementation**: In `SoftmaxNormEval`, add constraint:
   ```
   accumulator[0] = exp_values[0]
   accumulator[i] = accumulator[i-1] + exp_values[i]
   accumulator[seq_len-1] = sum_exp
   ```

3. **Guard clause**: Fix the sum=0 edge case — reject or handle gracefully.

**Files to modify**:
- `src/components/attention.rs`: Add accumulator column, fix sum=0 bug
- `src/gkr/verifier.rs`: Verify accumulator constraint
- `src/aggregation.rs`: Wire accumulator into trace building

**Estimated overhead**: ~50 felts per attention layer, ~1s total proving.

**Test plan**:
- Unit test: verify correct softmax sum
- Tamper test: wrong sum_exp, verify rejection
- Edge test: all-zero input (sum_exp = seq_len), verify behavior
- Edge test: sum_exp = 0 mod P, verify graceful handling
- H100 benchmark: full model with softmax sum constraint

---

### 1C. RoPE Position Encoding Arithmetization

**Status**: CRITICAL GAP — RoPE is compiled as `LayerType::Identity`, meaning the rotation is completely unverified.

**Current behavior**: `GraphOp::RoPE => LayerType::Identity` (circuit.rs:271). The GKR claim propagates unchanged — no rotation constraint is checked.

**Attack**: Malicious prover can skip RoPE entirely or apply wrong rotations, producing attention patterns that don't correspond to the model's actual position encoding.

**Why this is hard**: RoPE uses trigonometric functions (`cos`, `sin`) which are computed via f64 in the current implementation. Proving cos/sin in a finite field requires either:
- (a) Precomputed lookup tables (LogUp, like activations)
- (b) Taylor series approximation (high-degree polynomial constraint)
- (c) Cordic-style iterative computation (complex circuit)

**Protocol fix (recommended: LogUp table approach)**:

1. **Precompute angle table**: For each position `pos` in [0, max_seq_len] and each dimension pair `j` in [0, d/2]:
   - `theta_j = base^(-2j/d)` (computed in integer fixed-point)
   - `cos_table[pos][j] = fixed_point_cos(pos * theta_j)`
   - `sin_table[pos][j] = fixed_point_sin(pos * theta_j)`

2. **Commit table**: Poseidon Merkle root of the angle table (registered on-chain like weight commitments).

3. **Prove rotation**: For each (pos, dim_pair), prove via LogUp:
   - `(pos, j, cos_val, sin_val)` is in the committed table
   - Output constraint: `out[2j] = in[2j]*cos - in[2j+1]*sin`, `out[2j+1] = in[2j]*sin + in[2j+1]*cos`

4. **Integer-only computation**: Replace f64 `powf/cos/sin` with fixed-point integer arithmetic using Chebyshev polynomials or precomputed tables. This eliminates platform divergence.

**Files to modify**:
- `src/gkr/circuit.rs`: Change `GraphOp::RoPE => LayerType::RoPE { .. }` (new layer type)
- `src/gkr/prover.rs`: Add `reduce_rope_layer()` with LogUp proof
- `src/gkr/verifier.rs`: Add RoPE verification
- `src/components/rope.rs`: Integer-only angle computation + table generation
- `src/aggregation.rs`: Wire RoPE into unified STARK for LogUp

**Estimated overhead**: ~500 felts per attention layer (table lookup proof). For 40 layers: ~20,000 extra felts.

**Estimated proving overhead**: ~3-5s total (table commitment + LogUp for 40 layers).

**Test plan**:
- Unit test: prove rotation for known angles
- Tamper test: wrong rotation angle, verify rejection
- Tamper test: wrong position offset, verify rejection
- Consistency test: integer vs f64 computation match
- H100 benchmark: full model with RoPE constraints

---

### 1D. Causal Attention Mask Verification

**Status**: CRITICAL GAP — prover applies mask in forward pass but verifier doesn't check which positions are masked.

**Current behavior**: Causal mask sets `scores[i][j] = P-2` for `j > i + cache_offset`. This happens in the prover's forward pass but the softmax STARK component doesn't constrain it.

**Attack**: Malicious prover masks wrong positions (e.g., allows future tokens), breaking autoregressive property.

**Protocol fix**:

1. **Mask constraint in SoftmaxExp**: For each score entry, the constraint checks:
   ```
   if col_idx > row_idx + position_offset:
       score must equal MASK_VALUE (P-2)
   ```

2. **Implementation**: Add a preprocessed column encoding the mask pattern. The STARK constraint verifies that masked positions have the sentinel value.

3. **Position offset binding**: Link `position_offset` to the KV-cache commitment chain, ensuring consistency across decode steps.

**Files to modify**:
- `src/components/attention.rs`: Add mask constraint column
- `src/aggregation.rs`: Wire mask into trace

**Estimated overhead**: Minimal — mask is a preprocessed column (no prover work).

**Test plan**:
- Unit test: correct mask accepted
- Tamper test: future token unmasked, verify rejection
- Decode test: mask consistent with KV-cache position

---

### 1E. Eliminate f64 Platform Divergence

**Status**: CRITICAL — RoPE angles, activation tables, quantization all use f64. Different platforms may produce different M31 values.

**Current behavior**: `base.powf(-2.0 * j / d)`, `angle.cos()`, `angle.sin()`, GELU tanh approximation all use hardware f64.

**Protocol fix**:

1. **Integer-only RoPE**: Chebyshev polynomial cos/sin over fixed-point M31 values
2. **Integer-only activation tables**: Precompute GELU/Sigmoid/Softmax tables using only M31 arithmetic (polynomial approximation or exhaustive enumeration for 16-bit range)
3. **Deterministic quantization**: Replace f32 intermediate with M31-native rounding

**Test plan**:
- Cross-platform test: compute tables on x86, ARM, GPU — verify identical M31 outputs
- Roundtrip test: encode -> decode matches for all valid inputs

---

## Phase 2: Performance Optimization

**Goal**: Reduce proving overhead where possible. 103s baseline is already strong.

**Analysis (March 17, 2026)**: Deep research revealed the GPU dispatch threshold (2A)
was a red herring — all matmuls already use GPU via `gpu_matmul_m31_full` and
`reduce_matmul_layer_gpu`. The 103s breakdown is:

| Phase | Time | Backend | Optimization Potential |
|-------|------|---------|----------------------|
| Weight loading | 35s | CPU I/O (mmap + transpose) | Parallel shard extraction |
| Forward pass | 38s | GPU (`gpu_matmul_m31_full`) | Fused activation kernels |
| GKR walk | 47s | GPU (`reduce_matmul_layer_gpu`) | Already optimized |
| Unified STARK | 5s | CPU (SimdBackend) | Blocked by STWO GpuBackend bug |
| Serialization | 2s | CPU | Binary format |

### 2A. GPU Dispatch Threshold — RESOLVED (Not a Bottleneck)

**Status**: CLOSED. Investigation confirmed all matmuls already dispatch to GPU.
The 16384 comment in gpu_sumcheck.rs is outdated documentation, not an active threshold.

### 2B. GPU Unified STARK — BLOCKED (STWO Library Bug)

**Status**: DEFERRED. The GpuBackend hits `ConstraintsNotSatisfied` due to a bug in
STWO's preprocessed column allocator that deduplicates columns by name. Multi-instance
components (e.g., 40 RMSNorm layers) read wrong column data. Instance ID workaround
fixes SimdBackend but not GpuBackend. Fix requires STWO library changes.
**Impact**: 5s → 2s (low priority, already fast).

### 2C. Fused GPU Kernels

**Status**: OPEN. Forward pass does CPU↔GPU transfers between matmul and activation.
Fusing activation into the GPU kernel would eliminate round-trips.
**Impact**: ~5s savings on forward pass.

### 2D. Binary Serialization

**Status**: OPEN. JSON hex encoding is 2x larger than necessary.
**Impact**: 2s → 0.5s, 7MB → 3.6MB.

---

## Phase 3: Multi-Model Support & Benchmarking

**Goal**: Prove the protocol works across diverse architectures — dense, MoE, and vision.

### 3A. Model Support Matrix (Updated March 17, 2026)

**Tier 1: Ready NOW (zero code changes after SiLU fix)**

| Model | Params | Norm | Activation | Attention | Status |
|-------|--------|------|-----------|-----------|--------|
| **Qwen3-14B** | 14B | RMSNorm | SiLU | GQA (40h/8kv) | **PROVEN** (103s, 40L) |
| **Llama-3-8B** | 8B | RMSNorm | SiLU | GQA (32h/8kv) | **READY** (download + run) |
| **Llama-3-70B** | 70B | RMSNorm | SiLU | GQA (64h/8kv) | **READY** (multi-GPU memory) |
| **Mistral-7B** | 7B | RMSNorm | SiLU | GQA (32h/8kv) | **READY** |
| **Phi-3 Mini** | 3.8B | RMSNorm | GELU | GQA | **READY** |
| **GPT-2** | 124M-1.5B | LayerNorm | GELU | MHA (12h) | **READY** |
| **GLM-4-9B** | 9B | RMSNorm | SiLU | GQA | **READY** (standard transformer) |

**Tier 2: Needs MoE routing (single feature unlocks all)**

| Model | Total Params | Active | Experts | Router | Source |
|-------|-------------|--------|---------|--------|--------|
| **Mixtral-8x7B** | 47B | 13B | 8, top-2 | MatMul | Open-weight (Mistral AI) |
| **Kimi K2** | 1T | 32B | MoE | MatMul | Open-weight (Moonshot AI) |
| **GLM-5** | 744B | 40B | MoE | MatMul | Open-weight (Zhipu AI, MIT) |
| **DeepSeek-V3** | 671B | ~37B | 256, top-8 | MatMul | Open-weight |
| **Kimi K2.5** | 1T+ | 32B | MoE + Vision | MatMul | Open-weight (Moonshot AI) |

**Tier 3: Needs specialized components**

| Model | Params | Blocker | New Component | Effort |
|-------|--------|---------|---------------|--------|
| **MiniMax-01** | 456B | Lightning (linear) attention | `GraphOp::LinearAttention` | HIGH (2-4 weeks) |
| **MiniMax-M1** | 456B | Same + reasoning | Same | HIGH |
| **YOLOv8** | 3-68M | Vision pipeline | Image preprocessing, Conv2D wiring | MEDIUM (2 weeks) |
| **ViT** | 86-632M | Patch embedding | Conv2D + reshape | MEDIUM |

### 3B. MoE Routing Protocol (KEY UNLOCK — Tier 2)

**Impact**: Single implementation unlocks Mixtral, Kimi K2/K2.5, GLM-5, DeepSeek-V3.

**How MoE works**:
```
For each token:
  1. router_logits = input × W_router          (MatMul: hidden_dim → num_experts)
  2. gate_weights = softmax(top-k(router_logits))  (top-k selection + normalize)
  3. For each selected expert i:
       expert_out_i = FFN_i(input)             (MatMul: hidden → ff → hidden)
  4. output = Σ gate_weight_i × expert_out_i   (weighted sum)
```

**What needs proving**:

| Step | Operation | Already Supported? | New Component Needed |
|------|-----------|-------------------|---------------------|
| Router MatMul | `input × W_router` | **Yes** (standard MatMul) | None |
| Top-k selection | Find k largest logits | **No** | `TopKProof` |
| Gate softmax | softmax over selected logits | **Yes** (Phase 1B softmax sum) | Wire to MoE |
| Expert FFN | Per-expert MatMul chain | **Yes** (standard MatMul) | None |
| Weighted sum | `Σ gate_i × expert_i` | **Yes** (Mul + Add) | None |

**The only new primitive**: `TopKProof` — prove that the k selected indices are the k largest values.

**TopK verification protocol** (comparison-based):
1. Prover provides: selected indices `[i_1, ..., i_k]` and values `[v_1, ..., v_k]`
2. Prover provides: rejected indices `[j_1, ..., j_{n-k}]` and values `[u_1, ..., u_{n-k}]`
3. Constraint 1: all values at selected indices match router_logits (LogUp lookup)
4. Constraint 2: all values at rejected indices match router_logits (LogUp lookup)
5. Constraint 3: `min(selected) >= max(rejected)` (comparison proof)
6. Constraint 4: union of selected + rejected = all indices (permutation argument)

**Estimated complexity**: ~1 week for TopK constraint + 1 week for MoE graph wiring.

**Files to modify**:
- `src/compiler/graph.rs`: Add `GraphOp::MoE { num_experts, top_k, expert_dims }`
- `src/gkr/circuit.rs`: Add `LayerType::MoE { ... }`
- `src/gkr/prover.rs`: Add `reduce_moe_layer()` that decomposes into router + TopK + experts
- `src/components/topk.rs`: New file — TopK STARK constraint + trace generation
- `src/compiler/hf_loader.rs`: Detect MoE architecture from config.json, build expert graph

### 3C. Lightning Attention Protocol (MiniMax-specific)

**Blocker for**: MiniMax-01, MiniMax-M1 only. Not needed for other MoE models.

Lightning Attention replaces `softmax(QK^T/√d)V` with linear attention:
`output = φ(Q) × (φ(K)^T × V)` where φ is a kernel function.

This changes the attention from O(n²) to O(n) but requires a completely different
arithmetization — no softmax, no score matrix, different matmul decomposition.

**Protocol**: Would need `GraphOp::LinearAttention` with:
1. Kernel function application: `φ(Q)`, `φ(K)` — element-wise (like activation, LogUp provable)
2. Accumulated KV: `S = Σ φ(K_i)^T × V_i` — running sum (MatMul + Add)
3. Output: `out_i = φ(Q_i) × S` — MatMul

**Effort**: 2-4 weeks. Deferred until MoE routing is complete.

### 3D. CNN Support (YOLOv8, Vision Transformers)

`GraphOp::Conv2D` already exists in the graph IR. Implementation via im2col:
- `Conv2D(input, kernel) = MatMul(im2col(input), reshape(kernel))`
- im2col is deterministic index mapping — provable via permutation argument
- BatchNorm: same as LayerNorm (mean + variance + affine, already proven)
- Detection head (NMS, anchor boxes): post-processing, not part of arithmetic trace

**Effort**: 2 weeks. Conv2D IR exists, need inference pipeline + im2col proof.

### 3E. Competitive Benchmarking

| Competitor | Max Params | Prove Time | On-Chain | Proof System |
|-----------|-----------|-----------|----------|-------------|
| **EZKL** | ~1M | seconds | Solidity (Ethereum) | Halo2/KZG |
| **zkLLM** | 13B | 1-15 min | None | Custom |
| **Giza** | ~10M | minutes | Cairo (Stone) | STARK |
| **Expander** | ? | ? | None | GKR |
| **DeepProve** | GPT-2 | ? | None | GKR |
| **ObelyZK** | **14B (dense), 1T+ (MoE target)** | **103s** | **Cairo (Starknet)** | **GKR + STARK** |

**Benchmark protocol**:
1. Standardize: GPT-2-124M, Llama-3-8B, Mixtral-8x7B (when MoE ready)
2. Measure: prove time, verify time, proof size, calldata, security level
3. Hardware: H100 NVL (single GPU), document exact specs
4. Reproducible: publish scripts + model weights + expected outputs
5. Open-source benchmark suite for community verification

---

## Phase 4: Protocol Innovation

### 4A. Recursive Proof Composition

**Goal**: Compress 112K felt calldata into ~500 felts via recursive STARK.

**Approach**: Prove the GKR verifier execution in a STARK circuit (the verifier becomes the witness). The recursive proof attests "I verified the GKR proof and it passed." On-chain, verify only the recursive proof.

**Impact**: Constant-size on-chain verification regardless of model size. A 14B model and a 400B model have the same verification cost.

### 4B. Streaming Proof Aggregation

**Goal**: Aggregate multiple inference proofs into a single on-chain proof.

**Approach**: Batch N inference proofs into one recursive proof. On-chain verifier checks one proof for N inferences.

**Use case**: Audit pipeline proves 100 inferences, submits 1 aggregated proof. Cost: ~0.25 STRK total instead of 100 * 0.25 = 25 STRK.

### 4C. Verifiable Fine-Tuning

**Goal**: Prove that fine-tuning was performed correctly (DP-SGD, LoRA, etc.)

**Approach**: Extend GKR to backward pass gradients. Prove:
- Loss function evaluation
- Gradient computation (chain rule through verified forward pass)
- Weight update rule (SGD/Adam step)
- Differential privacy noise injection (if required)

### 4D. Cross-Model Pipeline Verification

**Goal**: Prove that a pipeline (model A -> post-process -> model B) executed correctly.

**Approach**: Chain proof commitments across models. Proof A's output commitment becomes Proof B's input commitment. The on-chain verifier checks both proofs and the linking commitment.

---

## Phase 5: Production Infrastructure

### 5A. Inference Engine Integration

**Goal**: Drop-in plugin for vLLM, TGI, Ollama that captures M31 intermediates.

**Approach**:
- Hook into the inference engine's forward pass at the matmul/norm/activation boundaries
- Capture intermediates as M31 matrices (quantize from f16/bf16)
- Write to append-only inference log
- Proving runs asynchronously from serving

### 5B. Prover Fleet Orchestration

**Goal**: Horizontal scaling across GPU fleet.

**Approach**:
- Queue-based: inference log entries → prover job queue → GPU workers
- Worker pools: dedicated H100s for proving (separate from serving)
- Priority scheduling: high-value inferences first

### 5C. Verification API

**Goal**: Public REST API for proof verification.

**Approach**:
- `GET /verify/{proof_hash}` → verified/not-verified + model details
- `GET /model/{model_id}/proofs` → list of verified proofs
- `POST /challenge/{inference_id}` → trigger on-demand proof generation

---

## On-Chain Streaming Verification — COMPLETE (March 21, 2026)

**First-ever full GKR streaming proof verification of ML inference on Starknet.** 6/6 TX SUCCEEDED on Starknet Sepolia.

| Step | TX Hash | Status |
|------|---------|--------|
| stream_init | [`0x5493...1a1`](https://sepolia.starkscan.co/tx/0x5493310a8e2deb5d2f25b07e2402e84692aaf5926141b5acc203a1892a181a1) | SUCCEEDED |
| output_mle | [`0x7cab...7e`](https://sepolia.starkscan.co/tx/0x7cabd35f5382c11334c6509e40b7a758ccd7e03e83e75b66a3c569f5d7b7a7e) | SUCCEEDED |
| layers | [`0x5346...918`](https://sepolia.starkscan.co/tx/0x53465edc957c5f8a6054739a0633beecf814ee37e3e22c23a570448a5be5918) | SUCCEEDED |
| weight_binding | [`0x5f54...3fc`](https://sepolia.starkscan.co/tx/0x5f549a1e6cc1ebefea3615c2458cdd0fd8f45fd505bf72e5b9dd8417c9be3fc) | SUCCEEDED |
| input_mle | [`0x2395...bab`](https://sepolia.starkscan.co/tx/0x239545b66f94387a3d1b5dbc55dedba6b7de1d5384f1930e77e152a986d5bab) | SUCCEEDED |
| finalize | [`0x4b08...a41`](https://sepolia.starkscan.co/tx/0x4b081156d4be88ea159533223d2597d76cd3f99911501d8326e156f12051a41) | SUCCEEDED |

- **Model**: Qwen2-0.5B, 1 transformer layer (8 GKR layers: 3 RMSNorm + 4 MatMul + 1 SiLU)
- **Contract (v39)**: `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`
- **Class hash**: `0x0473c81da9df0522f5c239f022889f7730ef866fb97e4f092ad1e8793fb22feb`
- **Proof**: 5,526 felts streaming calldata, 5 MLE opening queries, streaming v25 protocol

---

## Execution Timeline (Updated March 21, 2026)

| Phase | Status | Deliverable | Test Gate |
|-------|--------|-------------|-----------|
| **Streaming Verification** | **COMPLETE** (March 21) | 6-TX on-chain GKR streaming, v39 contract | 6/6 TX SUCCEEDED on Sepolia |
| **1A** LayerNorm mean/variance | **CLOSED** (was already done) | Plain sumcheck + LogUp | Tamper tests pass |
| **1B** Softmax sum | **CLOSED** (March 17) | Plain sumcheck + row-sum binding | 5 tamper tests pass |
| **1C** RoPE arithmetization | **CLOSED** (March 17) | Full STARK (rotation + LogUp) | 8 RoPE tests + circuit test |
| **1D** Causal mask | **CLOSED** (March 17) | Fiat-Shamir binding | Causal mismatch test |
| **1E** f64 elimination | **CLOSED** (April 1) | Integer-only cos/sin/exp/sigmoid/gelu/isqrt | 7 integer_math + 7 rope + 48 attention tests |
| **1F** Attention scale 1/√d | **CLOSED** (April 1) | Integer Newton-Raphson isqrt | 48 attention tests pass |
| **1G** Softmax sum=0 | **CLOSED** (April 1) | Graceful uniform fallback | No panic on degenerate input |
| **1H** LayerNorm γ/β affine | **CLOSED** (April 1) | γ commitment + scale_mle prover | 2 RMSNorm gamma tests |
| **2A** GPU threshold | **CLOSED** (not a bottleneck) | Already on GPU | Confirmed via profiling |
| **2B** GPU unified STARK | DEFERRED (STWO bug) | Needs STWO library fix | — |
| **2C** Fused kernels | OPEN | CUDA matmul+activation | ~5s savings |
| **2D** Binary serialization | **CLOSED** (April 1) | bincode OZKP format | 7 binary_serde tests |
| **2E** Configurable precision | **CLOSED** (April 1) | 16/64/256/1024 segments | 8 activation tests |
| **3A** SiLU activation | **CLOSED** (March 17) | Native SiLU LogUp | 4 unit tests |
| **3B** MoE routing (TopK) | **IN PROGRESS** | TopK proof + MoE graph | Mixtral/Kimi/GLM-5 proven |
| **3C** Lightning Attention | FUTURE | Linear attention protocol | MiniMax-01 proven |
| **3D** CNN (YOLOv8) | FUTURE | im2col proof | YOLOv8 proven |
| **3E** Benchmarks | NEXT (after 3B) | 5+ models, comparison table | Published |
| **4A** Recursive composition | FUTURE | Constant-size on-chain proof | Single TX on Starknet |

---

## Competitive Position After Roadmap

| Capability | EZKL | zkLLM | Giza | Expander | **ObelyZK (Target)** |
|-----------|------|-------|------|----------|---------------------|
| Max params | 1M | 13B | 10M | ? | **400B+ (MoE)** |
| Prove time | seconds | 1-15 min | minutes | ? | **<30s (14B dense)** |
| Full semantics | Yes | Yes | Yes | ? | **Yes (all ops verified)** |
| On-chain verifier | Solidity | None | Cairo | None | **Cairo (Starknet)** |
| Proof system | Halo2/KZG | Custom | STARK | GKR | **GKR + STARK (no FRI)** |
| MoE support | No | No | No | ? | **Yes (router + expert)** |
| CNN support | Yes | No | No | ? | **Yes (im2col)** |
| Recursive proofs | No | No | No | ? | **Yes (constant-size)** |
| Multi-inference audit | No | No | No | No | **Yes (batch proving)** |
| KV-cache chain | No | No | No | No | **Yes (incremental Merkle)** |
| Deployed | Ethereum | No | Starknet | No | **Starknet** |

---

## Development Methodology

For each phase item:

1. **Research**: Deep dive into the protocol extension, write formal specification
2. **Implement**: Code the prover and verifier changes
3. **Unit test**: Local tests with small matrices (fast iteration)
4. **Security test**: Tamper detection tests (corrupt each witness element)
5. **Integration test**: Full pipeline with existing test suite (41+ tests must pass)
6. **H100 GPU test**: End-to-end on real hardware with real model weights
7. **Benchmark**: Measure overhead vs baseline, report in this document
8. **Paper update**: Update obelyzk-paper.tex with new results
9. **Push**: Commit, push, verify CI passes

No phase item is considered complete until it passes the H100 GPU end-to-end test with the full 40-layer Qwen3-14B model.
