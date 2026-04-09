# GPU-Accelerated Provable Computation Platform

**Version**: 1.0 | **Date**: April 3, 2026

## What We Need to Build

### Phase 1: GPU STARK Benchmark (1 week)

**Goal**: Prove the 80x claim with real numbers.

Write a standalone benchmark that runs the same STWO operations on both SIMD and GPU backends, measuring wall-clock time for each:

```
benchmarks/
  gpu_vs_simd/
    ntt_bench.rs          — NTT/IFFT on M31 polynomials (2^20 to 2^24 elements)
    merkle_bench.rs       — Poseidon Merkle tree commitment (2^20 to 2^24 leaves)
    constraint_bench.rs   — AIR constraint evaluation (1M to 100M rows)
    fri_bench.rs          — FRI fold + query generation
    e2e_stark_bench.rs    — Full STARK prove on synthetic AIR
```

For each benchmark:
- STWO SimdBackend (baseline)
- Our CUDA kernels
- Our Metal shaders
- Report: ops/sec, latency, throughput, GPU utilization

**Files to create/modify**:
- `benchmarks/gpu_vs_simd/` — new benchmark suite
- `src/gpu_stark.rs` — new module: GPU wrappers for STWO prove operations
- Need to extract NTT, Merkle, constraint eval from STWO internals

**Key insight**: We already have `gpu_matmul_m31_full` (matmul), `poseidon_merkle` (Merkle on GPU), `gpu_sumcheck` (MLE evaluation). We need NTT and constraint eval GPU kernels.

---

### Phase 2: Fix STWO GpuBackend (2 weeks)

**Goal**: Make STWO's `prove()` work on GpuBackend for multi-component AIRs.

The bug: STWO's `GpuBackend` deduplicates preprocessed columns by name. When multiple instances of the same component type exist (e.g., 40 RMSNorm layers), they share column names but need different column data. The GPU allocator returns the wrong column.

**Fix approach**:
1. Add instance ID to preprocessed column names (e.g., `rmsnorm_0_rsqrt`, `rmsnorm_1_rsqrt`)
2. Or: bypass name-based dedup, use positional column allocation
3. Submit PR to STWO upstream (or maintain a fork patch)

**Files to modify**:
- `stwo/crates/stwo/src/prover/backend/gpu/` — preprocessed column allocator
- Our `src/aggregation.rs` — instance ID generation per component

**Impact**: Unified STARK proves on GPU (5s → 1-2s). More importantly, proves we can fix STWO internals.

---

### Phase 3: GPU NTT Kernel (2 weeks)

**Goal**: GPU-accelerated Number Theoretic Transform over M31.

NTT is ~30% of STARK proving time. STWO uses Cooley-Tukey butterfly on CPU (SIMD). We need a CUDA kernel.

M31 NTT specifics:
- Field: p = 2^31 - 1 (Mersenne prime)
- Circle group: order p+1 = 2^31 (perfect for radix-2 NTT)
- Twiddle factors: precomputed roots of unity on the circle
- STWO uses DCCT (Discrete Cosine Circle Transform), not standard NTT

**Implementation**:
```rust
// src/gpu_ntt.rs
pub fn ntt_gpu(values: &mut [M31], log_size: u32) { ... }
pub fn intt_gpu(values: &mut [M31], log_size: u32) { ... }

// CUDA kernel: gpu_ntt.cu
__global__ void m31_butterfly_kernel(
    uint32_t* values,
    const uint32_t* twiddles,
    int stage, int half_size
) {
    // Cooley-Tukey butterfly in M31 arithmetic
    uint32_t a = values[i];
    uint32_t b = m31_mul(values[i + half_size], twiddles[j]);
    values[i] = m31_add(a, b);
    values[i + half_size] = m31_sub(a, b);
}
```

**Benchmark target**: 20-50x over SIMD for polynomials of size 2^22+.

---

### Phase 4: GPU Constraint Evaluation (2 weeks)

**Goal**: Evaluate AIR constraints on GPU instead of CPU.

STWO evaluates constraints row-by-row using the `FrameworkEval` trait. Each row computes the constraint polynomial at one point. This is embarrassingly parallel — each row is independent.

**Implementation**:
- Compile constraint expressions into a GPU kernel at prove time
- OR: write a generic constraint evaluator that takes column data + constraint coefficients
- Each GPU thread evaluates one row's constraints

**Files**:
- `src/gpu_constraint_eval.rs` — GPU constraint evaluation engine
- Integrate with STWO's `prove()` pipeline as a custom evaluator backend

---

### Phase 5: DeFi Pricing Proof-of-Concept (1 week)

**Goal**: Prove a DeFi computation using our MatMul sumcheck.

Example: Multi-pool AMM pricing
```
prices = price_matrix × liquidity_vector
```

Where:
- `price_matrix[pool][token]` = exchange rate (N pools × M tokens)
- `liquidity_vector[token]` = amount per token
- `prices[pool]` = portfolio value in each pool

This is a MatMul. Our prover handles it TODAY. We just need to:
1. Define the input format (price feeds as M31)
2. Run the proof
3. Verify on-chain

**Files**:
- `examples/defi_pricing.rs` — standalone example
- `src/applications/defi.rs` — DeFi-specific helpers (price quantization, etc.)

---

### Phase 6: Recursive STARK Completion (3 weeks)

**Goal**: Single-TX on-chain verification via recursive proof.

Foundation exists (2.2K LOC on `feat/recursive-stark`):
- `src/recursive/air.rs` — 4 sub-components (Poseidon chain, sumcheck, QM31 arith, public inputs)
- `src/recursive/prover.rs` — prove_recursive()
- `src/recursive/verifier.rs` — verify_recursive()
- `src/recursive/witness.rs` — instrumented channel

What's needed:
- Fix the AIR constraint satisfaction (was passing but needs validation with new code)
- Generate valid STARK proof for the 40-layer Qwen3-14B GKR verifier
- Cairo recursive verifier: `elo-cairo-verifier/src/recursive_verifier.cairo`
- End-to-end: GKR proof → recursive STARK → single TX on Starknet

---

### Phase 7: Tensor VM Specification (2 weeks)

**Goal**: Formalize the "specialized tensor VM" as a spec.

Instructions:
```
MATMUL(A_id, B_id, out_id, m, k, n)     — GKR sumcheck
ACTIVATE(in_id, out_id, type, segments)  — piecewise algebraic
NORMALIZE(in_id, out_id, dim, gamma_id)  — eq-sumcheck + LogUp
ATTEND(Q_id, K_id, V_id, out_id, config) — composed sub-matmuls
EMBED(token_ids, table_id, out_id)       — LogUp
TOPK(logits_id, out_id, N, K)           — threshold + completeness
```

Each instruction maps to a single GKR reduction. The VM state is a set of named M31 matrices. Programs are directed acyclic graphs of instructions.

**Deliverable**: Technical specification document + reference interpreter.

---

## Implementation Priority

```
Week 1:    Phase 1 (GPU benchmark) — proves the 80x claim
Week 2-3:  Phase 2 (fix STWO GpuBackend) — unblocks GPU STARK
Week 3-4:  Phase 3 (GPU NTT kernel) — 30% of proving time
Week 4-5:  Phase 5 (DeFi PoC) — broadens beyond ML
Week 5-6:  Phase 4 (GPU constraint eval) — 25% of proving time
Week 6-9:  Phase 6 (recursive STARK) — single TX on-chain
Week 9-10: Phase 7 (Tensor VM spec) — the platform vision
```

## What Exists vs What's New

| Component | Exists Today | New Development |
|-----------|-------------|-----------------|
| M31 GPU arithmetic | 4,523 LOC (gpu_sumcheck.rs) | Extend to NTT |
| Poseidon GPU Merkle | Working (poseidon_merkle.rs) | Benchmark vs SIMD |
| Multi-GPU dispatch | 1,184 LOC (multi_gpu.rs) | Apply to STARK proving |
| Metal shaders | 634 LOC (metal/) | NTT + constraint eval |
| GKR sumcheck | Battle-tested (920+ tests) | DeFi/SQL applications |
| MatMul prover | Proven on real models | DeFi pricing example |
| Recursive STARK | 2.2K LOC foundation | Completion + testing |
| On-chain verifier | Deployed on Sepolia | Recursive verifier |
| Tensor VM | Implicit in graph compiler | Formalize as spec |
