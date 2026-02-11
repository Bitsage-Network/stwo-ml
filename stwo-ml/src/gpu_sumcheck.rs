//! GPU-accelerated sumcheck prover for matrix multiplication.
//!
//! Provides CUDA kernels for the two hot-path operations in the sumcheck protocol:
//! 1. **Round polynomial reduction** — parallel sum over `k/2` element pairs
//! 2. **MLE fold** — fixing a variable in the multilinear extension (reuses stwo's kernel)
//!
//! The `GpuMatMulOracle` implements `MultivariatePolyOracle`, allowing the existing
//! `sumcheck::prove_batch()` protocol to run transparently on GPU without any
//! protocol-level changes.
//!
//! # Dispatch Threshold
//!
//! GPU is only used when `k >= 2^MLE_THRESHOLD` (default 16384 elements).
//! Below this, CPU is faster due to kernel launch overhead (~5-10us per launch).

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use num_traits::{One, Zero};

#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::m31::M31;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::qm31::{QM31, SecureField};
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::cm31::CM31;
#[cfg(feature = "cuda-runtime")]
use stwo::core::channel::{Blake2sChannel, Channel};
#[cfg(feature = "cuda-runtime")]
use stwo::prover::lookups::sumcheck::{self, MultivariatePolyOracle, SumcheckProof};
#[cfg(feature = "cuda-runtime")]
use stwo::prover::lookups::utils::UnivariatePoly;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::cuda_executor::{get_cuda_executor, CudaFftExecutor, CudaFftError};

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use crate::components::matmul::{
    M31Matrix, MatMulSumcheckProof, MatMulError,
    evaluate_mle_pub, restrict_mle_pub,
    matrix_to_mle_pub, matrix_to_mle_col_major_pub,
};

// =============================================================================
// CUDA Kernel Source
// =============================================================================

/// CUDA kernel for the sumcheck round polynomial computation.
///
/// Computes three QM31 sums in parallel over `mid = n_points/2` pairs:
/// - `s0 = Σ f_a[i] * f_b[i]`          (evaluation at t=0)
/// - `s1 = Σ f_a[mid+i] * f_b[mid+i]`  (evaluation at t=1)
/// - `s2 = Σ (2*f_a[mid+i] - f_a[i]) * (2*f_b[mid+i] - f_b[i])`  (evaluation at t=2)
///
/// Uses shared memory block-level tree reduction (256 threads/block).
/// A second `sumcheck_reduce_kernel` finishes cross-block reduction when grid_dim > 1.
#[cfg(feature = "cuda-runtime")]
const SUMCHECK_CUDA_KERNEL: &str = r#"
#define M31_PRIME 0x7FFFFFFFu

__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return (sum >= M31_PRIME) ? (sum - M31_PRIME) : sum;
}

__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (a + M31_PRIME - b);
}

__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t result = lo + hi;
    return (result >= M31_PRIME) ? (result - M31_PRIME) : result;
}

// CM31 = (real, imag) with u^2 = 2
struct CM31 {
    uint32_t real;
    uint32_t imag;
};

__device__ __forceinline__ CM31 cm31_add(CM31 a, CM31 b) {
    CM31 r;
    r.real = m31_add(a.real, b.real);
    r.imag = m31_add(a.imag, b.imag);
    return r;
}

__device__ __forceinline__ CM31 cm31_sub(CM31 a, CM31 b) {
    CM31 r;
    r.real = m31_sub(a.real, b.real);
    r.imag = m31_sub(a.imag, b.imag);
    return r;
}

// CM31 multiplication: (a + ub)(c + ud) = (ac + 2bd) + u(ad + bc)
__device__ __forceinline__ CM31 cm31_mul(CM31 a, CM31 b) {
    uint32_t ac = m31_mul(a.real, b.real);
    uint32_t bd = m31_mul(a.imag, b.imag);
    uint32_t ad = m31_mul(a.real, b.imag);
    uint32_t bc = m31_mul(a.imag, b.real);
    CM31 r;
    r.real = m31_add(ac, m31_add(bd, bd));
    r.imag = m31_add(ad, bc);
    return r;
}

// QM31 = (a0 + u*a1) + i*(a2 + u*a3)  with i^2 = u + 2
struct QM31 {
    uint32_t a0, a1, a2, a3;
};

__device__ __forceinline__ QM31 qm31_zero() {
    QM31 r = {0, 0, 0, 0};
    return r;
}

__device__ __forceinline__ QM31 qm31_add(QM31 x, QM31 y) {
    QM31 r;
    r.a0 = m31_add(x.a0, y.a0);
    r.a1 = m31_add(x.a1, y.a1);
    r.a2 = m31_add(x.a2, y.a2);
    r.a3 = m31_add(x.a3, y.a3);
    return r;
}

__device__ __forceinline__ QM31 qm31_sub(QM31 x, QM31 y) {
    QM31 r;
    r.a0 = m31_sub(x.a0, y.a0);
    r.a1 = m31_sub(x.a1, y.a1);
    r.a2 = m31_sub(x.a2, y.a2);
    r.a3 = m31_sub(x.a3, y.a3);
    return r;
}

__device__ __forceinline__ QM31 qm31_mul(QM31 x, QM31 y) {
    CM31 x0 = {x.a0, x.a1};
    CM31 x1 = {x.a2, x.a3};
    CM31 y0 = {y.a0, y.a1};
    CM31 y1 = {y.a2, y.a3};

    CM31 x0y0 = cm31_mul(x0, y0);
    CM31 x1y1 = cm31_mul(x1, y1);
    CM31 x0y1 = cm31_mul(x0, y1);
    CM31 x1y0 = cm31_mul(x1, y0);

    // (u+2) * x1y1 = u*x1y1 + 2*x1y1
    // u * (r + u*i) = 2i + u*r
    CM31 u_x1y1 = {m31_add(x1y1.imag, x1y1.imag), x1y1.real};
    CM31 term = cm31_add(u_x1y1, cm31_add(x1y1, x1y1));

    CM31 real_part = cm31_add(x0y0, term);
    CM31 imag_part = cm31_add(x0y1, x1y0);

    QM31 r;
    r.a0 = real_part.real;
    r.a1 = real_part.imag;
    r.a2 = imag_part.real;
    r.a3 = imag_part.imag;
    return r;
}

// Shared memory layout: 3 * BLOCK_SIZE QM31 values (s0, s1, s2)
// Each QM31 is 4 u32 => 3 * 256 * 4 * 4 = 12288 bytes = 12 KB
#define BLOCK_SIZE 256

extern "C" __global__ void sumcheck_round_kernel(
    const uint32_t* __restrict__ f_a,
    const uint32_t* __restrict__ f_b,
    uint32_t* __restrict__ block_s0,
    uint32_t* __restrict__ block_s1,
    uint32_t* __restrict__ block_s2,
    uint32_t mid
) {
    __shared__ uint32_t s_s0[BLOCK_SIZE * 4];
    __shared__ uint32_t s_s1[BLOCK_SIZE * 4];
    __shared__ uint32_t s_s2[BLOCK_SIZE * 4];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize thread-local accumulators to zero
    QM31 local_s0 = qm31_zero();
    QM31 local_s1 = qm31_zero();
    QM31 local_s2 = qm31_zero();

    if (idx < mid) {
        // Load f_a[i] and f_a[mid+i]  (QM31 = 4 u32 each)
        uint32_t lo_offset = idx * 4;
        uint32_t hi_offset = (mid + idx) * 4;

        QM31 a0 = {f_a[lo_offset], f_a[lo_offset+1], f_a[lo_offset+2], f_a[lo_offset+3]};
        QM31 a1 = {f_a[hi_offset], f_a[hi_offset+1], f_a[hi_offset+2], f_a[hi_offset+3]};
        QM31 b0 = {f_b[lo_offset], f_b[lo_offset+1], f_b[lo_offset+2], f_b[lo_offset+3]};
        QM31 b1 = {f_b[hi_offset], f_b[hi_offset+1], f_b[hi_offset+2], f_b[hi_offset+3]};

        // s0 += a0 * b0
        local_s0 = qm31_mul(a0, b0);
        // s1 += a1 * b1
        local_s1 = qm31_mul(a1, b1);
        // s2 += (2*a1 - a0) * (2*b1 - b0)
        QM31 a2 = qm31_sub(qm31_add(a1, a1), a0);
        QM31 b2 = qm31_sub(qm31_add(b1, b1), b0);
        local_s2 = qm31_mul(a2, b2);
    }

    // Store to shared memory
    uint32_t base = tid * 4;
    s_s0[base+0] = local_s0.a0; s_s0[base+1] = local_s0.a1;
    s_s0[base+2] = local_s0.a2; s_s0[base+3] = local_s0.a3;
    s_s1[base+0] = local_s1.a0; s_s1[base+1] = local_s1.a1;
    s_s1[base+2] = local_s1.a2; s_s1[base+3] = local_s1.a3;
    s_s2[base+0] = local_s2.a0; s_s2[base+1] = local_s2.a1;
    s_s2[base+2] = local_s2.a2; s_s2[base+3] = local_s2.a3;

    __syncthreads();

    // Block-level tree reduction
    for (uint32_t stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            uint32_t mine = tid * 4;
            uint32_t other = (tid + stride) * 4;
            QM31 a = {s_s0[mine], s_s0[mine+1], s_s0[mine+2], s_s0[mine+3]};
            QM31 b_val = {s_s0[other], s_s0[other+1], s_s0[other+2], s_s0[other+3]};
            QM31 sum = qm31_add(a, b_val);
            s_s0[mine] = sum.a0; s_s0[mine+1] = sum.a1;
            s_s0[mine+2] = sum.a2; s_s0[mine+3] = sum.a3;

            a = (QM31){s_s1[mine], s_s1[mine+1], s_s1[mine+2], s_s1[mine+3]};
            b_val = (QM31){s_s1[other], s_s1[other+1], s_s1[other+2], s_s1[other+3]};
            sum = qm31_add(a, b_val);
            s_s1[mine] = sum.a0; s_s1[mine+1] = sum.a1;
            s_s1[mine+2] = sum.a2; s_s1[mine+3] = sum.a3;

            a = (QM31){s_s2[mine], s_s2[mine+1], s_s2[mine+2], s_s2[mine+3]};
            b_val = (QM31){s_s2[other], s_s2[other+1], s_s2[other+2], s_s2[other+3]};
            sum = qm31_add(a, b_val);
            s_s2[mine] = sum.a0; s_s2[mine+1] = sum.a1;
            s_s2[mine+2] = sum.a2; s_s2[mine+3] = sum.a3;
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        uint32_t blk = blockIdx.x * 4;
        block_s0[blk+0] = s_s0[0]; block_s0[blk+1] = s_s0[1];
        block_s0[blk+2] = s_s0[2]; block_s0[blk+3] = s_s0[3];
        block_s1[blk+0] = s_s1[0]; block_s1[blk+1] = s_s1[1];
        block_s1[blk+2] = s_s1[2]; block_s1[blk+3] = s_s1[3];
        block_s2[blk+0] = s_s2[0]; block_s2[blk+1] = s_s2[1];
        block_s2[blk+2] = s_s2[2]; block_s2[blk+3] = s_s2[3];
    }
}

// Second-pass reduction kernel: reduces per-block partial sums to a single QM31 per channel.
// Only needed when grid_dim > 1 (mid > 256).
extern "C" __global__ void sumcheck_reduce_kernel(
    const uint32_t* __restrict__ partials,
    uint32_t* __restrict__ output,
    uint32_t n_blocks
) {
    // Each block handles one of the 3 channels (s0, s1, s2)
    // partials layout: [s0_block0, s0_block1, ..., s1_block0, ..., s2_block0, ...]
    // Channel index from blockIdx.x
    uint32_t channel = blockIdx.x;
    if (channel >= 3) return;

    __shared__ uint32_t s_data[BLOCK_SIZE * 4];

    uint32_t tid = threadIdx.x;
    const uint32_t* channel_partials = partials + channel * n_blocks * 4;

    // Load partial sum for this thread's block
    QM31 val = qm31_zero();
    if (tid < n_blocks) {
        uint32_t base = tid * 4;
        val = (QM31){channel_partials[base], channel_partials[base+1],
                     channel_partials[base+2], channel_partials[base+3]};
    }

    uint32_t base = tid * 4;
    s_data[base+0] = val.a0; s_data[base+1] = val.a1;
    s_data[base+2] = val.a2; s_data[base+3] = val.a3;

    __syncthreads();

    // Tree reduction
    for (uint32_t stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < n_blocks) {
            uint32_t mine = tid * 4;
            uint32_t other = (tid + stride) * 4;
            QM31 a = {s_data[mine], s_data[mine+1], s_data[mine+2], s_data[mine+3]};
            QM31 b_val = {s_data[other], s_data[other+1], s_data[other+2], s_data[other+3]};
            QM31 sum = qm31_add(a, b_val);
            s_data[mine] = sum.a0; s_data[mine+1] = sum.a1;
            s_data[mine+2] = sum.a2; s_data[mine+3] = sum.a3;
        }
        __syncthreads();
    }

    // Thread 0 writes final result for this channel
    if (tid == 0) {
        uint32_t out_base = channel * 4;
        output[out_base+0] = s_data[0]; output[out_base+1] = s_data[1];
        output[out_base+2] = s_data[2]; output[out_base+3] = s_data[3];
    }
}
"#;

// =============================================================================
// GPU Sumcheck Executor
// =============================================================================

/// Compiled CUDA functions for sumcheck operations.
#[cfg(feature = "cuda-runtime")]
pub struct GpuSumcheckExecutor {
    device: Arc<CudaDevice>,
    sumcheck_round_fn: CudaFunction,
    sumcheck_reduce_fn: CudaFunction,
}

#[cfg(feature = "cuda-runtime")]
impl GpuSumcheckExecutor {
    /// Create a new GPU sumcheck executor by compiling kernels.
    ///
    /// Reuses the device from stwo's global `CudaFftExecutor`.
    pub fn new() -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let device = executor.device.clone();

        // Compile sumcheck kernels via NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(SUMCHECK_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("sumcheck kernel: {:?}", e)))?;

        device.load_ptx(ptx, "sumcheck", &[
            "sumcheck_round_kernel",
            "sumcheck_reduce_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("load sumcheck PTX: {:?}", e)))?;

        let sumcheck_round_fn = device.get_func("sumcheck", "sumcheck_round_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                "sumcheck_round_kernel not found".into(),
            ))?;

        let sumcheck_reduce_fn = device.get_func("sumcheck", "sumcheck_reduce_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                "sumcheck_reduce_kernel not found".into(),
            ))?;

        Ok(Self {
            device,
            sumcheck_round_fn,
            sumcheck_reduce_fn,
        })
    }

    /// Compute the sumcheck round polynomial (s0, s1, s2) on GPU.
    ///
    /// Returns three QM31 values representing the polynomial evaluated at t=0, t=1, t=2.
    pub fn compute_round_poly(
        &self,
        d_f_a: &CudaSlice<u32>,
        d_f_b: &CudaSlice<u32>,
        mid: usize,
    ) -> Result<([u32; 4], [u32; 4], [u32; 4]), CudaFftError> {
        let block_size = 256u32;
        let grid_size = ((mid as u32) + block_size - 1) / block_size;
        let n_blocks = grid_size as usize;

        // Allocate per-block partial sums
        let mut d_block_s0 = unsafe { self.device.alloc::<u32>(n_blocks * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_block_s1 = unsafe { self.device.alloc::<u32>(n_blocks * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_block_s2 = unsafe { self.device.alloc::<u32>(n_blocks * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 3 * 256 * 4 * 4, // 12 KB
        };

        unsafe {
            self.sumcheck_round_fn.clone().launch(
                cfg,
                (
                    d_f_a,
                    d_f_b,
                    &mut d_block_s0,
                    &mut d_block_s1,
                    &mut d_block_s2,
                    mid as u32,
                ),
            ).map_err(|e| CudaFftError::KernelExecution(format!("sumcheck_round: {:?}", e)))?;
        }

        if n_blocks == 1 {
            // Single block — download directly
            self.device.synchronize()
                .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;

            let mut s0 = [0u32; 4];
            let mut s1 = [0u32; 4];
            let mut s2 = [0u32; 4];
            self.device.dtoh_sync_copy_into(&d_block_s0, &mut s0)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_block_s1, &mut s1)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_block_s2, &mut s2)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            return Ok((s0, s1, s2));
        }

        // Multiple blocks — need second reduction pass
        // Pack partials into contiguous layout: [s0_blocks | s1_blocks | s2_blocks]
        let total_partials = 3 * n_blocks * 4;
        let mut d_partials = unsafe { self.device.alloc::<u32>(total_partials) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // Copy block partials into packed layout
        // cudarc dtod_copy: src -> dst
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                *d_partials.device_ptr_mut(),
                *d_block_s0.device_ptr(),
                n_blocks * 4 * std::mem::size_of::<u32>(),
                self.device.cu_stream(),
            ).map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s0: {:?}", e)))?;

            cudarc::driver::result::memcpy_dtod_async(
                d_partials.device_ptr_mut().add(n_blocks * 4),
                *d_block_s1.device_ptr(),
                n_blocks * 4 * std::mem::size_of::<u32>(),
                self.device.cu_stream(),
            ).map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s1: {:?}", e)))?;

            cudarc::driver::result::memcpy_dtod_async(
                d_partials.device_ptr_mut().add(2 * n_blocks * 4),
                *d_block_s2.device_ptr(),
                n_blocks * 4 * std::mem::size_of::<u32>(),
                self.device.cu_stream(),
            ).map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s2: {:?}", e)))?;
        }

        // Output: 3 QM31 values = 12 u32
        let mut d_output = unsafe { self.device.alloc::<u32>(12) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let reduce_cfg = LaunchConfig {
            grid_dim: (3, 1, 1), // One block per channel
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 256 * 4 * 4, // 4 KB per block
        };

        unsafe {
            self.sumcheck_reduce_fn.clone().launch(
                reduce_cfg,
                (
                    &d_partials,
                    &mut d_output,
                    n_blocks as u32,
                ),
            ).map_err(|e| CudaFftError::KernelExecution(format!("sumcheck_reduce: {:?}", e)))?;
        }

        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;

        let mut output = [0u32; 12];
        self.device.dtoh_sync_copy_into(&d_output, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        let s0 = [output[0], output[1], output[2], output[3]];
        let s1 = [output[4], output[5], output[6], output[7]];
        let s2 = [output[8], output[9], output[10], output[11]];

        Ok((s0, s1, s2))
    }

    /// MLE fold on GPU: fix first variable to `challenge`.
    ///
    /// For input of `n_points` QM31 elements (stored as `n_points * 4` u32),
    /// splits into lower/upper halves and computes:
    ///   result[i] = lhs[i] + challenge * (rhs[i] - lhs[i])
    ///
    /// Returns a new device buffer of `n_points/2 * 4` u32.
    pub fn mle_fold(
        &self,
        d_input: &CudaSlice<u32>,
        n_points: usize,
        challenge: &[u32; 4],
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let half_n = n_points / 2;

        // Upload to contiguous layout for stwo's mle_fold_secure
        // lhs = input[0..half_n*4], rhs = input[half_n*4..n_points*4]
        // We can reuse stwo's kernel directly via the CudaFftExecutor
        let executor = get_cuda_executor().map_err(|e| e.clone())?;

        // Download input to CPU, split, and call stwo's mle_fold_secure
        // (For v1, this is simpler and correct. GPU-resident fold is a future optimization.)
        let mut cpu_data = vec![0u32; n_points * 4];
        self.device.dtoh_sync_copy_into(d_input, &mut cpu_data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        let lhs = &cpu_data[..half_n * 4];
        let rhs = &cpu_data[half_n * 4..];

        let folded = executor.mle_fold_secure(lhs, rhs, challenge, half_n)?;

        // Upload result back to GPU
        let d_result = self.device.htod_sync_copy(&folded)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        Ok(d_result)
    }
}

// =============================================================================
// QM31 Conversion Helpers
// =============================================================================

#[cfg(feature = "cuda-runtime")]
#[inline]
fn secure_field_to_u32s(val: SecureField) -> [u32; 4] {
    [val.0 .0 .0, val.0 .1 .0, val.1 .0 .0, val.1 .1 .0]
}

#[cfg(feature = "cuda-runtime")]
#[inline]
fn u32s_to_secure_field(data: &[u32; 4]) -> SecureField {
    QM31(
        CM31(M31(data[0]), M31(data[1])),
        CM31(M31(data[2]), M31(data[3])),
    )
}

// =============================================================================
// GPU MultivariatePolyOracle
// =============================================================================

/// GPU-backed oracle for the matmul sumcheck protocol.
///
/// Holds `f_a` and `f_b` on the GPU device. The `sum_as_poly_in_first_variable()`
/// method runs a parallel reduction kernel, and `fix_first_variable()` folds both
/// arrays using stwo's MLE fold kernel.
#[cfg(feature = "cuda-runtime")]
pub struct GpuMatMulOracle {
    /// f_a evaluations on GPU (n_points * 4 u32)
    d_f_a: CudaSlice<u32>,
    /// f_b evaluations on GPU (n_points * 4 u32)
    d_f_b: CudaSlice<u32>,
    /// Current number of QM31 points (halved each round)
    n_points: usize,
    /// GPU executor handle
    executor: Arc<GpuSumcheckExecutor>,
}

#[cfg(feature = "cuda-runtime")]
impl MultivariatePolyOracle for GpuMatMulOracle {
    fn n_variables(&self) -> usize {
        self.n_points.ilog2() as usize
    }

    fn sum_as_poly_in_first_variable(&self, _claim: SecureField) -> UnivariatePoly<SecureField> {
        let mid = self.n_points / 2;

        // Launch GPU reduction kernel
        let (s0_u32, s1_u32, s2_u32) = self.executor
            .compute_round_poly(&self.d_f_a, &self.d_f_b, mid)
            .expect("GPU sumcheck round kernel failed");

        let s0 = u32s_to_secure_field(&s0_u32);
        let s1 = u32s_to_secure_field(&s1_u32);
        let s2 = u32s_to_secure_field(&s2_u32);

        // Lagrange interpolation on CPU (3 points → degree-2 poly)
        let two = SecureField::from(M31::from(2));
        UnivariatePoly::interpolate_lagrange(
            &[SecureField::zero(), SecureField::one(), two],
            &[s0, s1, s2],
        )
    }

    fn fix_first_variable(self, challenge: SecureField) -> Self {
        let challenge_u32 = secure_field_to_u32s(challenge);

        let new_d_f_a = self.executor
            .mle_fold(&self.d_f_a, self.n_points, &challenge_u32)
            .expect("GPU MLE fold for f_a failed");

        let new_d_f_b = self.executor
            .mle_fold(&self.d_f_b, self.n_points, &challenge_u32)
            .expect("GPU MLE fold for f_b failed");

        GpuMatMulOracle {
            d_f_a: new_d_f_a,
            d_f_b: new_d_f_b,
            n_points: self.n_points / 2,
            executor: self.executor,
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Prove C = A × B using the sumcheck protocol with GPU acceleration.
///
/// Same protocol as `prove_matmul_sumcheck` but the inner sumcheck loop
/// runs on GPU for the reduction and MLE fold operations.
///
/// The initial MLE construction and restriction happen on CPU (one-time cost),
/// then the restricted f_a, f_b are uploaded to GPU for the iterative protocol.
#[cfg(feature = "cuda-runtime")]
pub fn prove_matmul_sumcheck_gpu(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<MatMulSumcheckProof, MatMulError> {

    // Validate dimensions (same as CPU path)
    if a.cols != b.rows {
        return Err(MatMulError::DimensionMismatch(
            format!("A.cols={} != B.rows={}", a.cols, b.rows),
        ));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(MatMulError::DimensionMismatch(
            format!("C({},{}) != expected ({},{})", c.rows, c.cols, a.rows, b.cols),
        ));
    }

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    for (name, val) in [("m", m), ("k", k), ("n", n)] {
        if !val.is_power_of_two() {
            return Err(MatMulError::NonPowerOfTwo(format!("{name}={val}")));
        }
    }

    let log_m = m.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Build MLEs on CPU (one-time cost)
    let mle_a = matrix_to_mle_pub(a);
    let mle_b_t = matrix_to_mle_col_major_pub(b);
    let mle_c = matrix_to_mle_pub(c);

    // Fiat-Shamir channel (must match CPU path exactly)
    let mut channel = Blake2sChannel::default();
    channel.mix_felts(&[
        SecureField::from(M31::from(m as u32)),
        SecureField::from(M31::from(k as u32)),
        SecureField::from(M31::from(n as u32)),
    ]);

    let r_i = channel.draw_secure_felts(log_m);
    let r_j = channel.draw_secure_felts(log_n);

    // Compute claimed sum
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle_pub(&mle_c, &r_ij);

    channel.mix_felts(&[claimed_sum]);

    // Restrict MLEs on CPU
    let f_a = restrict_mle_pub(&mle_a, &r_i);
    let f_b = restrict_mle_pub(&mle_b_t, &r_j);

    assert_eq!(f_a.len(), k, "f_a should have k={k} elements");
    assert_eq!(f_b.len(), k, "f_b should have k={k} elements");

    // Initialize GPU executor
    let gpu_executor = Arc::new(
        GpuSumcheckExecutor::new()
            .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?
    );

    // Upload restricted MLEs to GPU
    let f_a_u32: Vec<u32> = f_a.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
    let f_b_u32: Vec<u32> = f_b.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();

    let d_f_a = gpu_executor.device.htod_sync_copy(&f_a_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload f_a: {:?}", e)))?;
    let d_f_b = gpu_executor.device.htod_sync_copy(&f_b_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload f_b: {:?}", e)))?;

    // Build GPU oracle and run sumcheck
    let oracle = GpuMatMulOracle {
        d_f_a,
        d_f_b,
        n_points: k,
        executor: gpu_executor,
    };

    let lambda = SecureField::one();
    let (sumcheck_proof, assignment, final_oracles, _claimed_evals) =
        sumcheck::prove_batch(vec![claimed_sum], vec![oracle], lambda, &mut channel);

    // Download final single-point evaluations from GPU
    let final_oracle = &final_oracles[0];
    assert_eq!(final_oracle.n_points, 1);

    let mut final_a_u32 = [0u32; 4];
    let mut final_b_u32 = [0u32; 4];
    final_oracle.executor.device.dtoh_sync_copy_into(&final_oracle.d_f_a, &mut final_a_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download final_a: {:?}", e)))?;
    final_oracle.executor.device.dtoh_sync_copy_into(&final_oracle.d_f_b, &mut final_b_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download final_b: {:?}", e)))?;

    let final_a_eval = u32s_to_secure_field(&final_a_u32);
    let final_b_eval = u32s_to_secure_field(&final_b_u32);

    Ok(MatMulSumcheckProof {
        sumcheck_proof,
        r_i,
        r_j,
        claimed_sum,
        final_a_eval,
        final_b_eval,
        assignment,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[cfg(feature = "cuda-runtime")]
mod tests {
    use super::*;
    use crate::components::matmul::{
        M31Matrix, matmul_m31, prove_matmul_sumcheck, verify_matmul_sumcheck,
        MatMulOracle,
    };
    use stwo::core::fields::m31::M31;

    /// Helper: create a test matrix with sequential values.
    fn test_matrix(rows: usize, cols: usize) -> M31Matrix {
        let mut m = M31Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.set(i, j, M31::from((i * cols + j + 1) as u32));
            }
        }
        m
    }

    #[test]
    fn test_gpu_sumcheck_executor_init() {
        let executor = GpuSumcheckExecutor::new();
        assert!(executor.is_ok(), "GPU sumcheck executor should initialize: {:?}", executor.err());
    }

    #[test]
    fn test_gpu_sumcheck_round_kernel() {
        // Compare GPU round polynomial to CPU for a 4×4 matmul
        let a = test_matrix(4, 4);
        let b = test_matrix(4, 4);
        let c = matmul_m31(&a, &b);

        // Build MLEs and restrict on CPU (same as prove path)
        let mle_a = matrix_to_mle(&a);
        let mle_b_t = matrix_to_mle_col_major(&b);
        let mle_c = matrix_to_mle(&c);

        let mut channel = Blake2sChannel::default();
        channel.mix_felts(&[
            SecureField::from(M31::from(4)),
            SecureField::from(M31::from(4)),
            SecureField::from(M31::from(4)),
        ]);
        let r_i = channel.draw_secure_felts(2);
        let r_j = channel.draw_secure_felts(2);

        let f_a = restrict_mle_pub(&mle_a, &r_i);
        let f_b = restrict_mle_pub(&mle_b_t, &r_j);

        // CPU oracle computation
        let cpu_oracle = MatMulOracle { f_a: f_a.clone(), f_b: f_b.clone() };
        let cpu_poly = cpu_oracle.sum_as_poly_in_first_variable(SecureField::zero());

        // GPU computation
        let executor = GpuSumcheckExecutor::new().expect("GPU init");
        let f_a_u32: Vec<u32> = f_a.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
        let f_b_u32: Vec<u32> = f_b.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();

        let d_f_a = executor.device.htod_sync_copy(&f_a_u32).unwrap();
        let d_f_b = executor.device.htod_sync_copy(&f_b_u32).unwrap();

        let mid = f_a.len() / 2;
        let (s0_u32, s1_u32, s2_u32) = executor.compute_round_poly(&d_f_a, &d_f_b, mid).unwrap();

        let gpu_s0 = u32s_to_secure_field(&s0_u32);
        let gpu_s1 = u32s_to_secure_field(&s1_u32);
        let gpu_s2 = u32s_to_secure_field(&s2_u32);

        let two = SecureField::from(M31::from(2));
        let gpu_poly = UnivariatePoly::interpolate_lagrange(
            &[SecureField::zero(), SecureField::one(), two],
            &[gpu_s0, gpu_s1, gpu_s2],
        );

        // Compare at several evaluation points
        for t in 0..5 {
            let pt = SecureField::from(M31::from(t));
            assert_eq!(
                cpu_poly.eval_at_point(pt),
                gpu_poly.eval_at_point(pt),
                "Round polynomial mismatch at t={t}"
            );
        }
    }

    #[test]
    fn test_gpu_mle_fold() {
        // Compare GPU fold to CPU fold
        let a = test_matrix(4, 4);
        let b = test_matrix(4, 4);

        let mle_a = matrix_to_mle(&a);
        let mut channel = Blake2sChannel::default();
        channel.mix_felts(&[SecureField::from(M31::from(42))]);
        let r_i = channel.draw_secure_felts(2);
        let f_a = restrict_mle_pub(&mle_a, &r_i);

        let challenge = SecureField::from(M31::from(7));
        let k = f_a.len();
        let mid = k / 2;

        // CPU fold
        let mut cpu_result = Vec::with_capacity(mid);
        for i in 0..mid {
            cpu_result.push(f_a[i] + challenge * (f_a[mid + i] - f_a[i]));
        }

        // GPU fold
        let executor = GpuSumcheckExecutor::new().expect("GPU init");
        let f_a_u32: Vec<u32> = f_a.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
        let d_f_a = executor.device.htod_sync_copy(&f_a_u32).unwrap();
        let challenge_u32 = secure_field_to_u32s(challenge);

        let d_result = executor.mle_fold(&d_f_a, k, &challenge_u32).unwrap();
        let mut gpu_result_u32 = vec![0u32; mid * 4];
        executor.device.dtoh_sync_copy_into(&d_result, &mut gpu_result_u32).unwrap();

        for i in 0..mid {
            let gpu_val = u32s_to_secure_field(&[
                gpu_result_u32[i * 4],
                gpu_result_u32[i * 4 + 1],
                gpu_result_u32[i * 4 + 2],
                gpu_result_u32[i * 4 + 3],
            ]);
            assert_eq!(cpu_result[i], gpu_val, "MLE fold mismatch at index {i}");
        }
    }

    #[test]
    fn test_prove_matmul_sumcheck_gpu() {
        // Full GPU prove + CPU verify roundtrip
        let a = test_matrix(4, 4);
        let b = test_matrix(4, 4);
        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck_gpu(&a, &b, &c)
            .expect("GPU proving should succeed");

        verify_matmul_sumcheck(&proof, &a, &b, &c)
            .expect("CPU verification of GPU proof should succeed");
    }

    #[test]
    fn test_gpu_oracle_matches_cpu() {
        // Prove same matmul with GPU and CPU, verify both produce valid proofs
        let a = test_matrix(4, 4);
        let b = test_matrix(4, 4);
        let c = matmul_m31(&a, &b);

        let cpu_proof = prove_matmul_sumcheck(&a, &b, &c)
            .expect("CPU proving should succeed");
        let gpu_proof = prove_matmul_sumcheck_gpu(&a, &b, &c)
            .expect("GPU proving should succeed");

        // Both should have the same claimed sum and Fiat-Shamir points
        assert_eq!(cpu_proof.claimed_sum, gpu_proof.claimed_sum);
        assert_eq!(cpu_proof.r_i, gpu_proof.r_i);
        assert_eq!(cpu_proof.r_j, gpu_proof.r_j);
        assert_eq!(cpu_proof.assignment.len(), gpu_proof.assignment.len());

        // Both should verify
        verify_matmul_sumcheck(&cpu_proof, &a, &b, &c)
            .expect("CPU proof verification failed");
        verify_matmul_sumcheck(&gpu_proof, &a, &b, &c)
            .expect("GPU proof verification failed");
    }

    #[test]
    fn test_prove_matmul_sumcheck_gpu_rectangular() {
        // Non-square: 2×4 × 4×2
        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..2 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                b.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck_gpu(&a, &b, &c)
            .expect("GPU rectangular proving should succeed");
        verify_matmul_sumcheck(&proof, &a, &b, &c)
            .expect("GPU rectangular verification should succeed");
    }

    // Helper: reimplementation for test access
    fn matrix_to_mle(matrix: &M31Matrix) -> Vec<SecureField> {
        let n = matrix.rows * matrix.cols;
        assert!(n.is_power_of_two());
        matrix.data.iter().map(|&v| SecureField::from(v)).collect()
    }

    fn matrix_to_mle_col_major(matrix: &M31Matrix) -> Vec<SecureField> {
        let n = matrix.rows * matrix.cols;
        assert!(n.is_power_of_two());
        let mut evals = vec![SecureField::zero(); n];
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                evals[j * matrix.rows + i] = SecureField::from(matrix.get(i, j));
            }
        }
        evals
    }
}
