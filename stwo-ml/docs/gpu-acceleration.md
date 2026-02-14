# GPU Acceleration

## Overview

stwo-ml uses CUDA for all compute-intensive operations in the proving pipeline. GPU acceleration spans three layers:

1. **Sumcheck kernels** — Round polynomial evaluation + MLE folding (inner loop)
2. **Fused MLE restrict** — Direct matrix-to-restricted-vector without intermediate allocation
3. **Forward pass ops** — MatMul, element-wise add/mul/relu for model execution
4. **Multi-GPU** — Distributed chunk proving across multiple GPUs

All GPU code is gated behind `feature = "cuda-runtime"` (requires CUDA 12.4+).

## CUDA Kernel Architecture

### Kernel Groups

| Group | Constant | Kernels | Used By |
|-------|----------|---------|---------|
| `M31_FORWARD_KERNEL` | Forward pass | `m31_gemv_kernel`, `m31_gemm_kernel`, `m31_add_kernel`, `m31_mul_kernel`, `m31_relu_kernel` | Model execution |
| `MLE_RESTRICT_KERNEL` | Fused restrict | `m31_restrict_rows_kernel`, `m31_restrict_cols_kernel` | MatMul sumcheck setup |
| `GKR_CUDA_KERNEL` | GKR operations | `combine_blocks_kernel`, `evaluate_mle_kernel`, sumcheck kernels | SIMD block batching |
| LogUp kernels | 3-factor sumcheck | `logup_3way_round_kernel`, `logup_3way_fold_kernel`, `logup_4way_reduce_kernel` | Activation/dual-operand sumcheck |

Kernels are lazily compiled via `ForwardKernels`, `RestrictKernels` singletons on `GpuSumcheckExecutor`.

### Field Arithmetic in CUDA

All kernels operate on M31 (`p = 2^31 - 1`) and QM31 (degree-4 extension) field elements. CUDA kernels use `_r` suffix variants of QM31 arithmetic to avoid name collisions between kernel groups:

```cuda
// QM31 multiply: (a0 + a1·i) + (a2 + a3·i)·u
__device__ void qm31_mul_r(uint32_t* out, const uint32_t* a, const uint32_t* b) { ... }
```

## Fused MLE Restrict

### Problem

The standard matmul sumcheck setup requires:
1. `pad_matrix_pow2(A)` — copy + zero-pad to power-of-2 dimensions
2. `matrix_to_mle(A_padded)` — convert to MLE evaluation table (M31 → SecureField)
3. `restrict_mle(A_mle, r_i)` — fold variables to get restricted vector

For a 5120×5120 weight matrix padded to 8192×8192, this allocates 67M SecureField elements (~1 GB per matrix).

### Solution

Fused GPU kernels take the **original M31 matrix** and the **QM31 Lagrange basis** and produce the restricted vector directly on GPU:

```
gpu.restrict_rows(A: &M31Matrix, r_i: &[SecureField], pk: usize) → Vec<SecureField>
gpu.restrict_cols(B: &M31Matrix, r_j: &[SecureField], pk: usize) → Vec<SecureField>
```

| Metric | Before | After |
|--------|--------|-------|
| Memory per matrix | ~1 GB | O(k + n) |
| Compute ops (5120→8192) | 67M | 26M |
| Wall time (160 matmuls) | ~18s | eliminated |

### Lagrange Basis

`compute_lagrange_basis(challenges)` builds the multilinear Lagrange basis weights via tensor product in `O(n log n)`:

```
For challenges (r₀, r₁, ..., r_{d-1}):
  L[b₀b₁...b_{d-1}] = Π_i ((1-r_i)(1-b_i) + r_i·b_i)
```

The fused kernel computes `Σ_i L[i] · M[i, k]` in a single GPU pass.

## GPU Sumcheck

### Round Polynomial Evaluation

For degree-2 (matmul):
```
p(t) = Σ_{i=0}^{mid-1} [(1-t)·f_a[i] + t·f_a[mid+i]] · [(1-t)·f_b[i] + t·f_b[mid+i]]
```

Evaluated at `t = 0, 1, 2` to get `(s₀, s₁, s₂)`, then Newton interpolation → `(c₀, c₁, c₂)`.

For degree-3 (mul/dual-operand/activation):
```
p(t) = Σ_{i=0}^{mid-1} w(i,t) · a(i,t) · b(i,t)
```

Evaluated at `t = 0, 1, 2, 3` → Newton divided differences → `(c₀, c₁, c₂, c₃)`.

### MLE Fold

After each round, the sumcheck challenge `r` folds the MLE in-place:
```
f[i] = (1 - r) · f[i] + r · f[mid + i]    for i = 0..mid
```

GPU kernel dispatches one thread per element.

### Dispatch Threshold

GPU kernels are only used when `k >= 2^MLE_THRESHOLD` (default 16384 elements). Below this, CPU is faster due to kernel launch overhead (~5-10us per launch).

## GPU GEMM

`gpu_matmul_m31_full()` provides M31 matrix multiplication on GPU:

- **GEMV** (m=1): Single-row dot product, 1D grid
- **GEMM** (m>1): Full matrix multiply, 2D grid with 16×16 thread blocks

Used in model forward pass execution and weight commitment computation.

## GPU Element-wise Operations

| Kernel | Function | Threshold |
|--------|----------|-----------|
| `m31_add_kernel` | `gpu_elementwise_add()` | len >= 4096 |
| `m31_mul_kernel` | `gpu_elementwise_mul()` | len >= 4096 |
| `m31_relu_kernel` | `gpu_relu()` | len >= 4096 |

All operations fall back to CPU below threshold. Error return (not panic) on dimension mismatch enables graceful fallback.

## GKR GPU Operations

| Method | Description |
|--------|-------------|
| `evaluate_mle_gpu(mle, point)` | MLE evaluation at a point on GPU |
| `combine_blocks(block_mles, weights)` | `Σ_b w_b · MLE_b` — SIMD block combination |
| `reduce_matmul_layer_gpu(claim, A, B, m, k, n, channel)` | Full matmul reduction on GPU |
| `restrict_rows(A, r_i, pk)` | Fused row restriction |
| `restrict_cols(B, r_j, pk)` | Fused column restriction |
| `sumcheck_3way(ext_w, ext_a, ext_b, n, channel)` | 3-factor GPU sumcheck (reuses LogUp kernels) |

## Multi-GPU Distributed Proving

### Architecture

```
┌─────────────────────────────────┐
│       MultiGpuExecutor          │
│  ┌─────────┐  ┌─────────┐      │
│  │  GPU 0   │  │  GPU 1   │ ... │
│  │  chunk 0 │  │  chunk 1 │     │
│  │  chunk 3 │  │  chunk 2 │     │
│  └─────────┘  └─────────┘      │
└─────────────────────────────────┘
```

### Thread-Local Device Affinity

```rust
thread_local! { CURRENT_DEVICE: Cell<Option<usize>> }
```

- `set_thread_device(id)` / `get_thread_device()` — set/query affinity
- `DeviceGuard::new(id)` — RAII guard, restores previous device on drop (panic-safe)
- `propagate_device(parent)` — for rayon worker threads (they don't inherit `thread_local!`)

### GpuSumcheckExecutor Per-Device Pool

```rust
GpuSumcheckExecutor::cached()           // uses thread-local device
GpuSumcheckExecutor::cached_for_device(id)  // explicit device
```

Per-device singleton pool via `OnceLock<Mutex<HashMap<usize, Arc<Self>>>>`. One `cached()` call routes all downstream GPU operations to the correct device.

### Chunk Partitioning

`MultiGpuExecutor::partition_chunks(chunks)` uses greedy bin-packing with 80% memory safety margin:

1. Sort chunks by estimated memory (descending)
2. Assign each chunk to the GPU with the most free memory
3. Warn on oversized chunks exceeding single GPU capacity

### Proving Flow

```rust
prove_model_chunked_multi_gpu(graph, input, weights, executor)
```

1. CPU forward pass → chunk inputs
2. `partition_chunks()` → device assignments
3. `std::thread::scope` with `DeviceGuard` per chunk thread
4. Collect all results (ALL errors, not just first)
5. Return `MultiGpuProvingResult` with per-device stats

### CLI

```bash
prove-model --model model.onnx --gpu --multi-gpu --chunk-budget-gb 8
```

## Pipeline Optimizations

### Parallel Merkle Tree

`PoseidonMerkleTree::build_parallel()` + `root_only_parallel()`:
- `par_chunks(2)` for layers with >= 256 pairs
- `root_only_parallel`: no intermediate layer storage (just the root)
- Used in `commit_mle_root_only()` for batch entry prep

### Pipelined Weight Commitment

```rust
std::thread::scope(|s| {
    s.spawn(|| commit_weights_background(...));  // [BG] prefix on output
    prove_layers(...);  // proving runs concurrently
});
```

Zero added latency — weight commitment runs on a background thread during proving.

### Weight Loading

Two-phase bulk extract + parallel process:
1. Phase 1: Extract raw f32 from ALL SafeTensor shards
2. Phase 2: Single rayon parallel pass (160 tasks for Qwen3-14B)
3. `madvise(MADV_SEQUENTIAL | MADV_WILLNEED)` on shard mmaps for OS prefetch

### Streaming Serialization

`serialize_ml_proof_to_file()` writes via `BufWriter` (1 MB buffer) instead of `Vec<String>` + `.join()`, eliminating large intermediate allocations for proofs.
