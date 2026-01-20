# STWO GPU Backend Completion Plan

## Current Status: 87% Complete → Target: 100%

**Total Effort**: 100-130 hours (2.5-3.5 weeks)

---

## Task Summary

| # | Task | Hours | Priority | Impact |
|---|------|-------|----------|--------|
| 1 | GPU CI Testing | 15-20 | **HIGH** | Enables all testing |
| 2 | CUDA Graphs | 20-25 | **HIGH** | +20-40% speedup |
| 3 | Multi-GPU Distributed | 25-30 | MEDIUM | 4-8x on multi-GPU |
| 4 | Direct GPU Constraints | 30-35 | MEDIUM-HIGH | 2-5x constraint eval |
| 5 | Pinned Memory Pool | 10-15 | MEDIUM | 2-3x transfer speed |

---

## Phase 1: GPU CI Testing (15-20 hours)

### Problem
No automated GPU testing in CI pipeline.

### Files to Create
```
.github/workflows/gpu-ci.yaml        # GPU testing workflow
.github/actions/setup_cuda.yml       # CUDA setup action
crates/stwo/tests/gpu_integration_tests.rs
crates/stwo/tests/gpu_unit_tests.rs
```

### Implementation

```yaml
# .github/workflows/gpu-ci.yaml
name: GPU CI

on:
  push:
    branches: [main, dev]
  pull_request:

jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu]  # Or use cloud GPU runners
    steps:
      - uses: actions/checkout@v4
      - name: Setup CUDA
        uses: ./.github/actions/setup_cuda
      - name: Build with CUDA
        run: cargo build --features cuda-runtime
      - name: Run GPU tests
        run: cargo test --features cuda-runtime --lib prover::backend::gpu
      - name: Run integration tests
        run: cargo test --features cuda-runtime --test '*gpu*'
```

### Success Criteria
- CI workflow passes on GPU runner
- All GPU unit tests pass
- Feature parity tests (GPU vs SIMD) pass

---

## Phase 2: CUDA Graphs Utilization (20-25 hours)

### Problem
Framework exists in `optimizations.rs` but not actually used.

### Files to Modify
```
src/prover/backend/gpu/optimizations.rs  # Implement actual CUDA graph calls
src/prover/backend/gpu/cuda_executor.rs  # Add raw CUDA bindings
src/prover/backend/gpu/pipeline.rs       # Integrate graph capture
src/prover/backend/gpu/fft.rs            # Graph capture for FFT
```

### Key Implementation

```rust
// src/prover/backend/gpu/optimizations.rs
#[cfg(feature = "cuda-runtime")]
impl CudaGraph {
    pub fn begin_capture(&mut self) -> Result<(), CudaFftError> {
        unsafe {
            use cudarc::driver::sys;
            let result = sys::cuStreamBeginCapture(
                self.capture_stream.stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            );
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(CudaFftError::DriverInit("Begin capture failed".into()));
            }
        }
        self.is_capturing = true;
        Ok(())
    }

    pub fn end_capture(&mut self) -> Result<(), CudaFftError> {
        unsafe {
            use cudarc::driver::sys;
            let mut graph: sys::CUgraph = std::ptr::null_mut();
            sys::cuStreamEndCapture(self.capture_stream.stream, &mut graph);

            let mut graph_exec: sys::CUgraphExec = std::ptr::null_mut();
            sys::cuGraphInstantiate_v2(&mut graph_exec, graph, std::ptr::null_mut());

            self.graph_exec = Some(GraphExec {
                raw_graph: graph as usize,
                raw_exec: graph_exec as usize,
            });
        }
        Ok(())
    }

    pub fn launch(&self) -> Result<(), CudaFftError> {
        if let Some(ref exec) = self.graph_exec {
            unsafe {
                use cudarc::driver::sys;
                sys::cuGraphLaunch(
                    exec.raw_exec as sys::CUgraphExec,
                    self.capture_stream.stream,
                );
            }
        }
        Ok(())
    }
}
```

### Success Criteria
- Graph capture/launch reduces kernel overhead by 60%+
- No correctness regressions vs non-graph path

---

## Phase 3: Multi-GPU Distributed Mode (25-30 hours)

### Problem
Only throughput mode (independent proofs) works. Distributed mode incomplete.

### Files to Modify
```
src/prover/backend/gpu/multi_gpu.rs          # Major enhancements
src/prover/backend/gpu/multi_gpu_executor.rs # New traits
src/prover/backend/gpu/cuda_streams.rs       # P2P streams
```

### Key Implementation

```rust
// src/prover/backend/gpu/multi_gpu.rs

/// NVLink topology detection
pub struct NvlinkTopology {
    bandwidth_matrix: Vec<Vec<f32>>,  // GB/s between GPU pairs
    is_nvlink: Vec<Vec<bool>>,
}

impl NvlinkTopology {
    pub fn detect() -> Result<Self, CudaFftError> {
        // Use cudaDeviceGetP2PAttribute to query NVLink capabilities
        let device_count = get_device_count();
        let mut bandwidth = vec![vec![0.0f32; device_count]; device_count];
        let mut nvlink = vec![vec![false; device_count]; device_count];

        for i in 0..device_count {
            for j in 0..device_count {
                if i != j {
                    // Query P2P bandwidth
                    unsafe {
                        let mut access = 0i32;
                        cudarc::driver::sys::cuDeviceCanAccessPeer(&mut access, i as i32, j as i32);
                        nvlink[i][j] = access != 0;
                        bandwidth[i][j] = if nvlink[i][j] { 300.0 } else { 32.0 }; // NVLink vs PCIe
                    }
                }
            }
        }

        Ok(Self { bandwidth_matrix: bandwidth, is_nvlink: nvlink })
    }
}

/// Distributed FFT with cross-GPU butterfly operations
pub struct DistributedFft {
    log_size: u32,
    num_gpus: usize,
    topology: NvlinkTopology,
}

impl DistributedFft {
    pub fn execute(&self, pipelines: &mut [GpuProofPipeline]) -> Result<(), CudaFftError> {
        let local_layers = self.log_size as usize - (self.num_gpus as f32).log2().ceil() as usize;

        // Phase 1: Local FFT on each GPU (first log2(n/num_gpus) layers)
        for pipeline in pipelines.iter_mut() {
            pipeline.fft_local_layers(local_layers)?;
        }

        // Phase 2: Cross-GPU butterfly operations
        for layer in local_layers..self.log_size as usize {
            self.inter_gpu_butterfly(pipelines, layer)?;
        }

        Ok(())
    }

    fn inter_gpu_butterfly(&self, pipelines: &mut [GpuProofPipeline], layer: usize) -> Result<(), CudaFftError> {
        // Determine GPU pairs that exchange data
        let stride = 1 << (layer - (self.num_gpus as f32).log2().ceil() as usize);

        for i in (0..self.num_gpus).step_by(stride * 2) {
            let src = i;
            let dst = i + stride;

            // P2P transfer
            p2p_exchange(&pipelines[src], &pipelines[dst])?;
        }

        Ok(())
    }
}
```

### Success Criteria
- 4-GPU FFT achieves 3.5x+ speedup vs single GPU
- NVLink utilized when available
- Graceful fallback to PCIe

---

## Phase 4: Direct GPU Constraint Kernels (30-35 hours)

### Problem
Constraint evaluation uses SIMD vectorization, not direct GPU kernels.

### Files to Create/Modify
```
src/prover/backend/gpu/constraints.rs              # NEW - kernel implementations
src/prover/backend/gpu/mod.rs                      # Export constraints
crates/constraint-framework/src/prover/gpu_component_prover.rs
```

### Key Implementation

```rust
// src/prover/backend/gpu/constraints.rs (NEW)

pub const CONSTRAINT_EVAL_KERNEL: &str = r#"
__global__ void eval_constraints(
    const uint32_t* trace_data,
    uint32_t* constraint_out,
    const uint32_t* random_coeff,
    uint32_t domain_size,
    uint32_t num_constraints
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    uint32_t accumulator = 0;
    for (uint32_t j = 0; j < num_constraints; ++j) {
        uint32_t val = eval_constraint_j(trace_data, idx, j);
        uint32_t coeff = m31_pow(random_coeff[0], j);
        accumulator = m31_add(accumulator, m31_mul(val, coeff));
    }

    constraint_out[idx] = accumulator;
}

__device__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    return r >= 0x7FFFFFFF ? r - 0x7FFFFFFF : r;
}

__device__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t w = (uint64_t)a * b;
    uint32_t lo = (uint32_t)w;
    uint32_t hi = (uint32_t)(w >> 31);
    return m31_add(lo & 0x7FFFFFFF, hi);
}
"#;

pub struct ConstraintKernel {
    function: CudaFunction,
}

impl ConstraintKernel {
    pub fn compile(device: &Arc<CudaDevice>) -> Result<Self, CudaFftError> {
        let function = device.get_or_load_func("eval_constraints", CONSTRAINT_EVAL_KERNEL)?;
        Ok(Self { function })
    }

    pub fn launch(
        &self,
        trace: &CudaSlice<u32>,
        output: &mut CudaSlice<u32>,
        coeffs: &[u32],
        domain_size: u32,
        num_constraints: u32,
    ) -> Result<(), CudaFftError> {
        let config = LaunchConfig {
            grid_dim: ((domain_size + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        self.function.launch(config, (trace, output, coeffs, domain_size, num_constraints))?;
        Ok(())
    }
}
```

### Success Criteria
- 2x+ speedup vs SIMD for constraint-heavy AIRs
- Correctness verified against SIMD results

---

## Phase 5: Pinned Memory Pool (10-15 hours)

### Problem
Pinned buffer exists but no reuse pool, causing allocation overhead.

### Files to Modify
```
src/prover/backend/gpu/optimizations.rs  # Complete pool implementation
src/prover/backend/gpu/pipeline.rs       # Use pinned pool
```

### Key Implementation

```rust
// src/prover/backend/gpu/optimizations.rs

pub struct PinnedMemoryPool<T: Copy + Default> {
    available: HashMap<usize, Vec<PinnedBuffer<T>>>,
    stats: PinnedPoolStats,
}

impl<T: Copy + Default> PinnedMemoryPool<T> {
    pub fn acquire(&mut self, size: usize) -> Result<PinnedBuffer<T>, CudaFftError> {
        let rounded = size.next_power_of_two();

        // Try cache first
        if let Some(buffers) = self.available.get_mut(&rounded) {
            if let Some(buf) = buffers.pop() {
                self.stats.hits += 1;
                return Ok(buf);
            }
        }

        // Allocate new
        self.stats.misses += 1;
        PinnedBuffer::new(rounded)
    }

    pub fn release(&mut self, buffer: PinnedBuffer<T>) {
        let size = buffer.len().next_power_of_two();
        self.available.entry(size).or_default().push(buffer);
    }
}
```

### Success Criteria
- 80%+ cache hit rate after warmup
- 2x+ transfer speedup vs non-pinned

---

## Implementation Schedule

```
Week 1-2:  Phase 1 (GPU CI)           [BLOCKING - enables testing]
Week 2-3:  Phase 2 (CUDA Graphs)      [HIGH - 20-40% speedup]
Week 3-4:  Phase 5 (Pinned Memory)    [Can parallelize with Phase 2]
Week 4-5:  Phase 3 (Multi-GPU)        [MEDIUM - multi-GPU scaling]
Week 5-6:  Phase 4 (Constraints)      [MEDIUM-HIGH - AIR speedup]
```

## Dependency Graph

```
Phase 1 (CI) ──┬──> Phase 2 (Graphs)
               ├──> Phase 3 (Multi-GPU)
               ├──> Phase 4 (Constraints)
               └──> Phase 5 (Pinned Memory)

Phase 2 ──> Phase 3 (optional, can use graphs for local ops)
```

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| CI GPU Tests | 0 | 50+ |
| Kernel Launch Overhead | 100% | 40% (with graphs) |
| Multi-GPU Scaling (4x) | 1x | 3.5x |
| Constraint Eval Speed | 1x (SIMD) | 2-5x (GPU) |
| Transfer Efficiency | 1x | 2x (pinned pool) |

---

## Quick Start Commands

```bash
# After Phase 1: Run GPU tests locally
cargo test --features cuda-runtime --lib prover::backend::gpu

# After Phase 2: Benchmark graph optimization
cargo run --example gpu_vs_simd_real_benchmark --features cuda-runtime

# After Phase 3: Test multi-GPU
cargo run --example multi_gpu_benchmark --features multi-gpu

# After Phase 4: Test constraint kernel
cargo test --features cuda-runtime constraint_kernel
```
