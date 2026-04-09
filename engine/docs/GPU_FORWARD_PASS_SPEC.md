# GPU Forward Pass — Phase A Specification

**Goal**: Move the 38-second CPU forward pass to GPU using existing CUDA kernels.  
**Impact**: 96s total → ~58-66s (forward: 38s → 8s). 5x faster interactive chat.  
**Status**: OPEN — kernels exist, orchestrator needed.

---

## Current Bottleneck

The forward pass in `aggregation.rs:4691-5265` runs ~600 lines of CPU code iterating the computation graph. Each node executes sequentially on CPU, even though GPU kernels exist for every operation.

**Current timing breakdown (SmolLM2-135M, A10G):**

| Phase | Time | Backend |
|-------|------|---------|
| Forward pass (211 nodes) | ~8s | CPU (with GPU matmul) |
| GKR walk (120 matmuls) | ~80s | GPU sumcheck |
| Unified STARK | ~5s | CPU |
| **Total** | **~93s** | |

For Qwen3-14B on H100, the forward pass is ~38s due to larger matrices.

## Existing GPU Kernels (gpu_sumcheck.rs)

All forward-pass CUDA kernels already exist in `ForwardKernels`:

| Kernel | Function | Status |
|--------|----------|--------|
| `m31_gemv_kernel` | Matrix-vector multiply (m=1) | Exists |
| `m31_gemm_kernel` | Matrix-matrix multiply (m>1) | Exists |
| `m31_add_kernel` | Element-wise addition | Exists |
| `m31_mul_kernel` | Element-wise multiplication | Exists |
| `m31_relu_kernel` | ReLU activation | Exists |
| `m31_layernorm_kernel` | LayerNorm + stats output | Exists |
| `m31_rmsnorm_kernel` | RMSNorm + stats output | Exists |

**Already wired for matmul**: `gpu_matmul_m31_full()` is called in the forward pass. But each call uploads, computes, and downloads — the intermediate stays on CPU between ops.

## Architecture: GpuForwardExecutor

### New File: `engine/src/gpu_forward.rs`

```rust
/// GPU-resident tensor — stays on device between operations.
pub struct GpuTensor {
    data: CudaSlice<u32>,  // M31 values as u32 on GPU
    rows: usize,
    cols: usize,
}

/// Orchestrates the forward pass entirely on GPU.
pub struct GpuForwardExecutor {
    device: Arc<CudaDevice>,
    kernels: ForwardKernels,
    /// Pre-uploaded weight tensors (cached across inferences).
    weight_cache: HashMap<usize, GpuTensor>,
}

impl GpuForwardExecutor {
    /// Upload M31Matrix to GPU.
    pub fn upload(&self, m: &M31Matrix) -> GpuTensor;
    
    /// Download GPU tensor to M31Matrix.
    pub fn download(&self, t: &GpuTensor) -> M31Matrix;
    
    /// GPU matmul: C = A × B (both on device).
    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor;
    
    /// GPU element-wise add.
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor;
    
    /// GPU element-wise mul.
    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor;
    
    /// GPU ReLU activation.
    pub fn relu(&self, a: &GpuTensor) -> GpuTensor;
    
    /// GPU RMSNorm (returns normalized + stats for STARK).
    pub fn rmsnorm(&self, a: &GpuTensor, dim: usize) -> (GpuTensor, Vec<u32>, Vec<u32>);
    
    /// GPU LayerNorm.
    pub fn layernorm(&self, a: &GpuTensor, dim: usize) -> (GpuTensor, Vec<u32>, Vec<u32>, Vec<u32>);
    
    /// Pre-upload all weight matrices to GPU.
    pub fn cache_weights(&mut self, weights: &GraphWeights);
    
    /// Run the full forward pass on GPU.
    /// Returns: (intermediates for GKR, all layer data for STARK, output).
    pub fn forward_pass(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<ForwardPassResult, GpuForwardError>;
}
```

### Key Design Decisions

1. **Keep intermediates on GPU between ops**: The main speedup. Currently each `gpu_matmul_m31_full()` uploads input, computes, downloads output. Instead: upload once, chain operations, download intermediates only when needed for GKR/STARK.

2. **Download intermediates at collection points**: The GKR prover needs CPU `M31Matrix` for sumcheck. Download after each matmul (needed for claim chaining) but keep the running `current` tensor on GPU.

3. **Pre-upload weights at startup**: Weight matrices don't change between inferences. Upload once, keep on GPU. For SmolLM2 (120 weights, 576 dim): ~150MB GPU memory. For Qwen3-14B (160 weights, 5120 dim): ~6GB.

4. **Ops that stay on CPU**:
   - **RoPE**: Complex sincos computation, no GPU kernel. ~1ms per layer (fast enough).
   - **Embedding lookup**: Row index into table. CPU is fine.
   - **Attention softmax**: Inside attention_forward. CPU for now.
   - **SiLU/GELU (piecewise)**: No GPU kernel for piecewise activation. Use CPU.

   For these: download current tensor to CPU, run op, upload result back. The roundtrip is ~0.1ms per op — negligible vs the matmul savings.

5. **Collect layer data for STARK**: The `activation_layers`, `layernorm_layers`, etc. need CPU data for the unified STARK. Collect these during the forward pass (same as the CPU path) by downloading after each non-matmul op.

## Integration Point

In `aggregation.rs` at line 4665 (start of forward pass):

```rust
// GPU forward pass when available
#[cfg(feature = "cuda-runtime")]
if crate::backend::gpu_is_available() {
    let gpu_exec = GpuForwardExecutor::cached()?;
    let fwd = gpu_exec.forward_pass(graph, input, weights)?;
    // fwd contains: intermediates, node_outputs, output, layer data
    // Skip the 600-line CPU forward pass loop
    // Continue to Phase 2 (GKR) and Phase 3 (STARK) as normal
}
```

## Expected Speedup

| Op | CPU Time | GPU Time | Speedup |
|----|----------|----------|---------|
| MatMul (576×576) | ~1ms | ~0.1ms | 10x |
| MatMul (576×1536) | ~3ms | ~0.3ms | 10x |
| RMSNorm (576 dim) | ~0.1ms | ~0.05ms | 2x |
| Upload/download per op | 0ms (CPU) | ~0.05ms | overhead |
| **Total forward (SmolLM2)** | **~8s** | **~2-3s** | **3-4x** |
| **Total forward (Qwen3-14B)** | **~38s** | **~5-8s** | **5-7x** |

The SmolLM2 speedup is modest (data too small for GPU). The Qwen3-14B speedup is significant (GPU well-utilized at d_model=5120).

## Files to Create/Modify

| File | Change |
|------|--------|
| `engine/src/gpu_forward.rs` | **New** — GpuForwardExecutor, GpuTensor, forward_pass() |
| `engine/src/lib.rs` | Add `pub mod gpu_forward;` |
| `engine/src/aggregation.rs:4665` | GPU branch before CPU forward pass loop |
| `engine/src/gpu_sumcheck.rs` | Make ForwardKernels pub(crate) |

## Weight Memory Budget

| Model | Weights | Dim | GPU Memory |
|-------|---------|-----|------------|
| SmolLM2-135M | 120 | 576 | ~150 MB |
| Qwen2-0.5B | 120 | 896 | ~400 MB |
| Llama-3.2-3B | 160 | 3072 | ~6 GB |
| Qwen3-14B | 160 | 5120 | ~16 GB |

H100 (80GB): comfortably fits Qwen3-14B weights + intermediates.
A10G (24GB): fits SmolLM2 and Qwen2-0.5B. Tight for 3B+ models.

## Verification

```bash
# GPU forward pass produces same output as CPU
OBELYZK_GPU_FORWARD=1 obelyzk bench --tokens 1
# Compare io_commitment with CPU path — must match

# Benchmark speedup
OBELYZK_GPU_FORWARD=0 obelyzk bench --tokens 1  # CPU baseline
OBELYZK_GPU_FORWARD=1 obelyzk bench --tokens 1  # GPU path
# GPU should be 3-7x faster

# All tests still pass
cargo test --lib --features std
```
