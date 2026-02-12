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
use std::sync::OnceLock;

#[cfg(feature = "cuda-runtime")]
use num_traits::{One, Zero};

#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::m31::M31;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::qm31::{QM31, SecureField};
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::cm31::CM31;
#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::FieldExpOps;
#[cfg(feature = "cuda-runtime")]
use stwo::core::channel::{Blake2sChannel, Channel};
#[cfg(feature = "cuda-runtime")]
use stwo::prover::lookups::sumcheck::{self, MultivariatePolyOracle};
#[cfg(feature = "cuda-runtime")]
use stwo::prover::lookups::utils::UnivariatePoly;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::cuda_executor::{get_cuda_executor, CudaFftError};

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};

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
// NVRTC does not include <stdint.h> — define fixed-width types explicitly
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

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

            a = QM31{s_s1[mine], s_s1[mine+1], s_s1[mine+2], s_s1[mine+3]};
            b_val = QM31{s_s1[other], s_s1[other+1], s_s1[other+2], s_s1[other+3]};
            sum = qm31_add(a, b_val);
            s_s1[mine] = sum.a0; s_s1[mine+1] = sum.a1;
            s_s1[mine+2] = sum.a2; s_s1[mine+3] = sum.a3;

            a = QM31{s_s2[mine], s_s2[mine+1], s_s2[mine+2], s_s2[mine+3]};
            b_val = QM31{s_s2[other], s_s2[other+1], s_s2[other+2], s_s2[other+3]};
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

// GPU-resident MLE fold kernel: fix first variable to challenge.
// result[i] = input[i] + alpha * (input[half_n + i] - input[i])
// Operates entirely on-device — no CPU round-trip.
extern "C" __global__ void mle_fold_kernel(
    const uint32_t* __restrict__ input,
    const uint32_t* __restrict__ alpha,
    uint32_t* __restrict__ output,
    uint32_t half_n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    // Load alpha (same for all threads — broadcast from constant memory)
    QM31 a = {alpha[0], alpha[1], alpha[2], alpha[3]};

    // Load lhs = input[idx], rhs = input[half_n + idx]
    uint32_t lo = idx * 4;
    uint32_t hi = (half_n + idx) * 4;
    QM31 lhs = {input[lo], input[lo+1], input[lo+2], input[lo+3]};
    QM31 rhs = {input[hi], input[hi+1], input[hi+2], input[hi+3]};

    // result = lhs + alpha * (rhs - lhs)
    QM31 diff = qm31_sub(rhs, lhs);
    QM31 term = qm31_mul(a, diff);
    QM31 result = qm31_add(lhs, term);

    uint32_t out = idx * 4;
    output[out]   = result.a0;
    output[out+1] = result.a1;
    output[out+2] = result.a2;
    output[out+3] = result.a3;
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
        val = QM31{channel_partials[base], channel_partials[base+1],
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
// M31 GEMV CUDA Kernel (for forward pass acceleration)
// =============================================================================

/// CUDA kernel for M31 vector-matrix multiply: output[col] = Σ input[i] * weight[i * n + col] mod P.
/// One thread per output column. Uses u64 accumulation with periodic modular reduction.
#[cfg(feature = "cuda-runtime")]
const M31_GEMV_KERNEL: &str = r#"
#define P 0x7FFFFFFFu

extern "C" __global__ void m31_gemv_kernel(
    const unsigned int* input,   // k values (M31)
    const unsigned int* weight,  // k x n matrix (row-major, M31)
    unsigned int* output,        // n values (M31)
    unsigned int k,
    unsigned int n
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    unsigned long long acc = 0;
    for (unsigned int i = 0; i < k; i++) {
        unsigned long long a = (unsigned long long)input[i];
        unsigned long long b = (unsigned long long)weight[i * n + col];
        acc += a * b;
        // Reduce every 4 iterations to prevent u64 overflow:
        // max product = (2^31-2)^2 ≈ 4.6e18, sum of 4 ≈ 1.8e19 < 2^64
        if ((i & 3u) == 3u) {
            acc %= (unsigned long long)P;
        }
    }
    output[col] = (unsigned int)(acc % (unsigned long long)P);
}
"#;

/// GPU-accelerated M31 vector-matrix multiply for the forward pass.
///
/// Replaces CPU `matmul_m31` for the common case of m=1 (single-row input).
/// Falls back to CPU for multi-row inputs.
#[cfg(feature = "cuda-runtime")]
pub fn gpu_matmul_m31(
    input: &M31Matrix,
    weight: &M31Matrix,
) -> Result<M31Matrix, MatMulError> {
    use crate::components::matmul::matmul_m31;

    // Only accelerate single-row inputs (common in inference)
    if input.rows != 1 {
        return Ok(matmul_m31(input, weight));
    }

    let k = input.cols;
    let n = weight.cols;

    if k != weight.rows {
        return Err(MatMulError::SumcheckFailed(format!(
            "GPU GEMV dimension mismatch: input cols={k} != weight rows={}",
            weight.rows,
        )));
    }

    let executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

    // Compile GEMV kernel (separate from sumcheck kernels, cached in executor)
    let gemv_fn = executor.get_gemv_fn()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GEMV kernel: {e}")))?;

    // Upload input and weight as u32 arrays
    let input_u32: Vec<u32> = input.data.iter().map(|v| v.0).collect();
    let weight_u32: Vec<u32> = weight.data.iter().map(|v| v.0).collect();

    let d_input = executor.device.htod_sync_copy(&input_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload input: {:?}", e)))?;
    let d_weight = executor.device.htod_sync_copy(&weight_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload weight: {:?}", e)))?;

    let d_output: CudaSlice<u32> = executor.device.alloc_zeros(n)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU alloc output: {:?}", e)))?;

    // Launch: one thread per output column
    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;

    unsafe {
        gemv_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&d_input, &d_weight, &d_output, k as u32, n as u32),
        ).map_err(|e| MatMulError::SumcheckFailed(format!("GPU GEMV launch: {:?}", e)))?;
    }

    // Download result
    let mut output_u32 = vec![0u32; n];
    executor.device.dtoh_sync_copy_into(&d_output, &mut output_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download: {:?}", e)))?;

    let mut result = M31Matrix::new(1, n);
    for (i, &v) in output_u32.iter().enumerate() {
        result.data[i] = M31::from(v);
    }
    Ok(result)
}

// =============================================================================
// GPU Sumcheck Executor
// =============================================================================

/// Compiled CUDA functions for sumcheck operations.
#[cfg(feature = "cuda-runtime")]
pub struct GpuSumcheckExecutor {
    pub device: Arc<CudaDevice>,
    sumcheck_round_fn: CudaFunction,
    sumcheck_reduce_fn: CudaFunction,
    mle_fold_fn: CudaFunction,
    /// Lazily compiled GEMV kernel function.
    gemv_fn: std::sync::Mutex<Option<CudaFunction>>,
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
            "mle_fold_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("load sumcheck PTX: {:?}", e)))?;

        let sumcheck_round_fn = device.get_func("sumcheck", "sumcheck_round_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                "sumcheck_round_kernel not found".into(),
            ))?;

        let sumcheck_reduce_fn = device.get_func("sumcheck", "sumcheck_reduce_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                "sumcheck_reduce_kernel not found".into(),
            ))?;

        let mle_fold_fn = device.get_func("sumcheck", "mle_fold_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                "mle_fold_kernel not found".into(),
            ))?;

        Ok(Self {
            device,
            sumcheck_round_fn,
            sumcheck_reduce_fn,
            mle_fold_fn,
            gemv_fn: std::sync::Mutex::new(None),
        })
    }

    /// Get or create a cached global GPU executor.
    ///
    /// CUDA kernels are compiled via NVRTC exactly once on first call,
    /// then reused for all subsequent sumcheck proofs. This avoids
    /// recompiling ~200ms of NVRTC per matmul × 160+ matmuls = 32 seconds wasted.
    pub fn cached() -> Result<Arc<Self>, CudaFftError> {
        static EXECUTOR: OnceLock<Arc<GpuSumcheckExecutor>> = OnceLock::new();
        EXECUTOR.get_or_try_init(|| {
            eprintln!("[GPU] Compiling sumcheck CUDA kernels (one-time)...");
            let executor = GpuSumcheckExecutor::new()?;
            eprintln!("[GPU] Kernels compiled and cached.");
            Ok(Arc::new(executor))
        }).cloned()
    }

    /// Get or lazily compile the M31 GEMV kernel function.
    ///
    /// The GEMV kernel is compiled separately from the sumcheck kernels since
    /// not all code paths need it. Compilation happens once on first call.
    pub fn get_gemv_fn(&self) -> Result<CudaFunction, CudaFftError> {
        let mut guard = self.gemv_fn.lock().unwrap();
        if let Some(ref f) = *guard {
            return Ok(f.clone());
        }

        // Compile GEMV kernel via NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(M31_GEMV_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("GEMV kernel: {:?}", e)))?;

        self.device.load_ptx(ptx, "gemv", &["m31_gemv_kernel"])
            .map_err(|e| CudaFftError::KernelCompilation(format!("load GEMV PTX: {:?}", e)))?;

        let f = self.device.get_func("gemv", "m31_gemv_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                "m31_gemv_kernel not found".into(),
            ))?;

        *guard = Some(f.clone());
        Ok(f)
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

        // Copy block partials into packed layout: [s0_blocks | s1_blocks | s2_blocks]
        let s0_len = n_blocks * 4;
        let s1_offset = s0_len;
        let s2_offset = 2 * s0_len;
        self.device.dtod_copy(&d_block_s0, &mut d_partials.slice_mut(0..s0_len))
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s0: {:?}", e)))?;
        self.device.dtod_copy(&d_block_s1, &mut d_partials.slice_mut(s1_offset..s1_offset + s0_len))
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s1: {:?}", e)))?;
        self.device.dtod_copy(&d_block_s2, &mut d_partials.slice_mut(s2_offset..s2_offset + s0_len))
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s2: {:?}", e)))?;

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
    /// Runs entirely on GPU — no CPU round-trip.
    /// Returns a new device buffer of `n_points/2 * 4` u32.
    pub fn mle_fold(
        &self,
        d_input: &CudaSlice<u32>,
        n_points: usize,
        challenge: &[u32; 4],
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let half_n = n_points / 2;

        // Upload challenge (4 u32 = 16 bytes — the only transfer per round)
        let d_alpha = self.device.htod_sync_copy(challenge)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("alpha upload: {:?}", e)))?;

        // Allocate output buffer on GPU
        let mut d_output = unsafe { self.device.alloc::<u32>(half_n * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("fold output: {:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((half_n as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.mle_fold_fn.clone().launch(
                cfg,
                (
                    d_input,
                    &d_alpha,
                    &mut d_output,
                    half_n as u32,
                ),
            ).map_err(|e| CudaFftError::KernelExecution(format!("mle_fold: {:?}", e)))?;
        }

        Ok(d_output)
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

    // Auto-pad to power-of-2 dimensions (same as CPU path)
    let a = &crate::components::matmul::pad_matrix_pow2(a);
    let b = &crate::components::matmul::pad_matrix_pow2(b);
    let c = &crate::components::matmul::pad_matrix_pow2(c);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    debug_assert!(m.is_power_of_two());
    debug_assert!(k.is_power_of_two());
    debug_assert!(n.is_power_of_two());

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

    // Get cached GPU executor (kernels compiled once, reused across all matmuls)
    let gpu_executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

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
// On-Chain GPU Sumcheck (Poseidon Fiat-Shamir)
// =============================================================================

/// Prove C = A × B using GPU-accelerated sumcheck with Poseidon Fiat-Shamir
/// channel and MLE Poseidon Merkle commitments, formatted for on-chain Cairo
/// verification.
///
/// Same protocol as `prove_matmul_sumcheck_onchain` but the sumcheck inner
/// loop (s0/s1/s2 reduction + MLE fold) runs on GPU. Fiat-Shamir hashing
/// and Lagrange interpolation remain on CPU (O(1) per round, not hot).
#[cfg(feature = "cuda-runtime")]
pub fn prove_matmul_sumcheck_onchain_gpu(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<crate::components::matmul::MatMulSumcheckProofOnChain, MatMulError> {
    use crate::components::matmul::{
        MatMulSumcheckProofOnChain, RoundPoly, pad_matrix_pow2,
    };
    use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};
    use crate::crypto::mle_opening::{commit_mle, prove_mle_opening};

    // Validate dimensions
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

    // Auto-pad to power-of-2 dimensions
    let a = &pad_matrix_pow2(a);
    let b = &pad_matrix_pow2(b);
    let c = &pad_matrix_pow2(c);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let log_m = m.ilog2() as usize;
    let log_k = k.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Build MLEs on CPU (one-time cost)
    let mle_a = matrix_to_mle_pub(a);
    let mle_b_t = matrix_to_mle_col_major_pub(b);
    let mle_c = matrix_to_mle_pub(c);

    // PoseidonChannel for Fiat-Shamir (must match CPU path exactly)
    let mut channel = PoseidonChannel::new();
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);

    let r_i = channel.draw_qm31s(log_m);
    let r_j = channel.draw_qm31s(log_n);

    // Compute claimed sum: MLE_C(r_i, r_j)
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle_pub(&mle_c, &r_ij);

    channel.mix_felt(securefield_to_felt(claimed_sum));

    // Restrict MLEs on CPU
    let f_a = restrict_mle_pub(&mle_a, &r_i);
    let f_b = restrict_mle_pub(&mle_b_t, &r_j);

    assert_eq!(f_a.len(), k);
    assert_eq!(f_b.len(), k);

    // Commit to restricted MLEs
    let (a_commitment, _a_tree) = commit_mle(&f_a);
    let (b_commitment, _b_tree) = commit_mle(&f_b);

    channel.mix_felt(a_commitment);
    channel.mix_felt(b_commitment);

    // Get cached GPU executor (kernels compiled once, reused across all matmuls)
    let gpu_executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

    let f_a_u32: Vec<u32> = f_a.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
    let f_b_u32: Vec<u32> = f_b.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();

    let mut d_f_a = gpu_executor.device.htod_sync_copy(&f_a_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload f_a: {:?}", e)))?;
    let mut d_f_b = gpu_executor.device.htod_sync_copy(&f_b_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload f_b: {:?}", e)))?;

    // === Sumcheck with GPU inner loop + Poseidon channel ===
    let num_rounds = log_k;
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut assignment = Vec::with_capacity(num_rounds);
    let mut cur_n_points = k;

    for _round in 0..num_rounds {
        let mid = cur_n_points / 2;

        // GPU: compute s0, s1, s2 in parallel (only 48 bytes downloaded)
        let (s0_u32, s1_u32, s2_u32) = gpu_executor
            .compute_round_poly(&d_f_a, &d_f_b, mid)
            .map_err(|e| MatMulError::SumcheckFailed(format!("GPU round: {e}")))?;

        let s0 = u32s_to_secure_field(&s0_u32);
        let s1 = u32s_to_secure_field(&s1_u32);
        let s2 = u32s_to_secure_field(&s2_u32);

        // CPU: extract coefficients and build round polynomial
        let c0 = s0;
        let two = SecureField::from(M31::from(2));
        let c2 = (s2 - two * s1 + s0) * SecureField::from(M31::from(2)).inverse();
        let c1 = s1 - s0 - c2;

        round_polys.push(RoundPoly { c0, c1, c2 });

        // CPU: Poseidon Fiat-Shamir — mix polynomial and draw challenge
        channel.mix_poly_coeffs(c0, c1, c2);
        let r_k = channel.draw_qm31();
        assignment.push(r_k);

        // GPU: fold both MLEs with challenge (16 bytes uploaded, no download)
        let challenge_u32 = secure_field_to_u32s(r_k);

        let new_d_f_a = gpu_executor.mle_fold(&d_f_a, cur_n_points, &challenge_u32)
            .map_err(|e| MatMulError::SumcheckFailed(format!("GPU fold f_a: {e}")))?;
        let new_d_f_b = gpu_executor.mle_fold(&d_f_b, cur_n_points, &challenge_u32)
            .map_err(|e| MatMulError::SumcheckFailed(format!("GPU fold f_b: {e}")))?;

        d_f_a = new_d_f_a;
        d_f_b = new_d_f_b;
        cur_n_points = mid;
    }

    // Download final single-point evaluations (8 bytes total)
    assert_eq!(cur_n_points, 1);
    let mut final_a_u32 = [0u32; 4];
    let mut final_b_u32 = [0u32; 4];
    gpu_executor.device.dtoh_sync_copy_into(&d_f_a, &mut final_a_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download final_a: {:?}", e)))?;
    gpu_executor.device.dtoh_sync_copy_into(&d_f_b, &mut final_b_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download final_b: {:?}", e)))?;

    let final_a_eval = u32s_to_secure_field(&final_a_u32);
    let final_b_eval = u32s_to_secure_field(&final_b_u32);

    // MLE opening proofs (CPU — these use Poseidon Merkle paths)
    let a_opening = prove_mle_opening(&f_a, &assignment, &mut channel);
    let b_opening = prove_mle_opening(&f_b, &assignment, &mut channel);

    Ok(MatMulSumcheckProofOnChain {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        num_rounds: num_rounds as u32,
        claimed_sum,
        round_polys,
        final_a_eval,
        final_b_eval,
        a_commitment,
        b_commitment,
        a_opening,
        b_opening,
    })
}

// =============================================================================
// Batched Sumcheck — Multiple Matmuls in One Protocol
// =============================================================================

/// Per-matmul data for a batch: the pre-computed MLE restrictions and metadata.
#[cfg(feature = "cuda-runtime")]
pub struct BatchEntry {
    pub node_id: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub claimed_sum: SecureField,
    pub r_i: Vec<SecureField>,
    pub r_j: Vec<SecureField>,
    pub a_commitment: starknet_ff::FieldElement,
    pub b_commitment: starknet_ff::FieldElement,
    /// Restricted f_a (k elements)
    pub f_a: Vec<SecureField>,
    /// Restricted f_b (k elements)
    pub f_b: Vec<SecureField>,
}

/// Result of a batched sumcheck: shared round polys + per-matmul final evaluations.
#[cfg(feature = "cuda-runtime")]
pub struct BatchedSumcheckResult {
    pub round_polys: Vec<crate::components::matmul::RoundPoly>,
    pub assignment: Vec<SecureField>,
    pub lambda: SecureField,
    pub per_matmul: Vec<BatchedPerMatMulResult>,
}

#[cfg(feature = "cuda-runtime")]
pub struct BatchedPerMatMulResult {
    pub node_id: usize,
    pub final_a_eval: SecureField,
    pub final_b_eval: SecureField,
}

/// Prove a batch of matmuls with the same padded k dimension in one sumcheck.
///
/// Uses lambda-weighted combination: h(x) = Σ_i λ^i · f_a_i(x) · f_b_i(x)
/// This collapses N individual sumcheck protocols into ONE, sharing:
///   - Round polynomials (combined with lambda)
///   - Fiat-Shamir transcript (one PoseidonChannel)
///   - Sumcheck assignment (shared challenges)
///
/// Per-matmul data (claimed_sum, commitments, final evals) remains individual.
///
/// For Qwen3-14B: ~80 matmuls per dimension group → 2 batched sumchecks instead of 160.
#[cfg(feature = "cuda-runtime")]
pub fn prove_matmul_batch_onchain_gpu(
    entries: &[BatchEntry],
) -> Result<BatchedSumcheckResult, MatMulError> {
    use crate::components::matmul::RoundPoly;
    use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};

    if entries.is_empty() {
        return Err(MatMulError::SumcheckFailed("empty batch".into()));
    }

    let k = entries[0].k;
    let log_k = k.ilog2() as usize;
    let num_entries = entries.len();

    // All entries must have same k
    for e in entries {
        assert_eq!(e.k, k, "All entries in batch must have same padded k");
    }

    // Combined Fiat-Shamir channel
    let mut channel = PoseidonChannel::new();
    // Mix batch metadata
    channel.mix_u64(num_entries as u64);
    channel.mix_u64(k as u64);

    // Mix all per-matmul commitments and claimed sums
    for e in entries {
        channel.mix_felt(securefield_to_felt(e.claimed_sum));
        channel.mix_felt(e.a_commitment);
        channel.mix_felt(e.b_commitment);
    }

    // Draw batching lambda
    let lambda = channel.draw_qm31();

    // Get cached GPU executor
    let gpu_executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

    // Upload all f_a, f_b to GPU
    let mut d_f_a_list: Vec<CudaSlice<u32>> = Vec::with_capacity(num_entries);
    let mut d_f_b_list: Vec<CudaSlice<u32>> = Vec::with_capacity(num_entries);

    for e in entries {
        let f_a_u32: Vec<u32> = e.f_a.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
        let f_b_u32: Vec<u32> = e.f_b.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();

        let d_f_a = gpu_executor.device.htod_sync_copy(&f_a_u32)
            .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload batch f_a: {:?}", e)))?;
        let d_f_b = gpu_executor.device.htod_sync_copy(&f_b_u32)
            .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload batch f_b: {:?}", e)))?;

        d_f_a_list.push(d_f_a);
        d_f_b_list.push(d_f_b);
    }

    // Batched sumcheck: log_k rounds
    let mut round_polys = Vec::with_capacity(log_k);
    let mut assignment = Vec::with_capacity(log_k);
    let mut cur_n_points = k;

    for _round in 0..log_k {
        let mid = cur_n_points / 2;

        // GPU: compute per-matmul (s0, s1, s2) and combine with lambda
        let mut combined_s0 = SecureField::zero();
        let mut combined_s1 = SecureField::zero();
        let mut combined_s2 = SecureField::zero();
        let mut lambda_power = SecureField::one();

        for i in 0..num_entries {
            let (s0_u32, s1_u32, s2_u32) = gpu_executor
                .compute_round_poly(&d_f_a_list[i], &d_f_b_list[i], mid)
                .map_err(|e| MatMulError::SumcheckFailed(
                    format!("GPU batch round entry {i}: {e}"),
                ))?;

            let s0 = u32s_to_secure_field(&s0_u32);
            let s1 = u32s_to_secure_field(&s1_u32);
            let s2 = u32s_to_secure_field(&s2_u32);

            combined_s0 = combined_s0 + lambda_power * s0;
            combined_s1 = combined_s1 + lambda_power * s1;
            combined_s2 = combined_s2 + lambda_power * s2;

            lambda_power = lambda_power * lambda;
        }

        // Extract combined round polynomial coefficients
        let c0 = combined_s0;
        let two = SecureField::from(M31::from(2));
        let c2 = (combined_s2 - two * combined_s1 + combined_s0) * two.inverse();
        let c1 = combined_s1 - combined_s0 - c2;

        round_polys.push(RoundPoly { c0, c1, c2 });

        // Fiat-Shamir: mix combined polynomial and draw shared challenge
        channel.mix_poly_coeffs(c0, c1, c2);
        let r_k = channel.draw_qm31();
        assignment.push(r_k);

        // GPU: fold ALL MLEs with shared challenge
        let challenge_u32 = secure_field_to_u32s(r_k);

        for i in 0..num_entries {
            let new_d_f_a = gpu_executor.mle_fold(&d_f_a_list[i], cur_n_points, &challenge_u32)
                .map_err(|e| MatMulError::SumcheckFailed(
                    format!("GPU batch fold f_a entry {i}: {e}"),
                ))?;
            let new_d_f_b = gpu_executor.mle_fold(&d_f_b_list[i], cur_n_points, &challenge_u32)
                .map_err(|e| MatMulError::SumcheckFailed(
                    format!("GPU batch fold f_b entry {i}: {e}"),
                ))?;

            d_f_a_list[i] = new_d_f_a;
            d_f_b_list[i] = new_d_f_b;
        }

        cur_n_points = mid;
    }

    // Download per-matmul final evaluations
    assert_eq!(cur_n_points, 1);
    let mut per_matmul = Vec::with_capacity(num_entries);

    for i in 0..num_entries {
        let mut final_a_u32 = [0u32; 4];
        let mut final_b_u32 = [0u32; 4];
        gpu_executor.device.dtoh_sync_copy_into(&d_f_a_list[i], &mut final_a_u32)
            .map_err(|e| MatMulError::SumcheckFailed(format!("download final_a {i}: {:?}", e)))?;
        gpu_executor.device.dtoh_sync_copy_into(&d_f_b_list[i], &mut final_b_u32)
            .map_err(|e| MatMulError::SumcheckFailed(format!("download final_b {i}: {:?}", e)))?;

        per_matmul.push(BatchedPerMatMulResult {
            node_id: entries[i].node_id,
            final_a_eval: u32s_to_secure_field(&final_a_u32),
            final_b_eval: u32s_to_secure_field(&final_b_u32),
        });
    }

    Ok(BatchedSumcheckResult {
        round_polys,
        assignment,
        lambda,
        per_matmul,
    })
}

/// Prepare a BatchEntry from raw matrices (CPU work: MLE construction + restriction).
///
/// This can be called in parallel for multiple matmuls before feeding to
/// `prove_matmul_batch_onchain_gpu`.
#[cfg(feature = "cuda-runtime")]
pub fn prepare_batch_entry(
    node_id: usize,
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<BatchEntry, MatMulError> {
    use crate::components::matmul::pad_matrix_pow2;
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::crypto::mle_opening::commit_mle;

    let a = &pad_matrix_pow2(a);
    let b = &pad_matrix_pow2(b);
    let c = &pad_matrix_pow2(c);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let log_m = m.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    let mle_a = matrix_to_mle_pub(a);
    let mle_b_t = matrix_to_mle_col_major_pub(b);
    let mle_c = matrix_to_mle_pub(c);

    // Per-matmul Poseidon channel for r_i, r_j
    let mut channel = PoseidonChannel::new();
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);

    let r_i = channel.draw_qm31s(log_m);
    let r_j = channel.draw_qm31s(log_n);

    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle_pub(&mle_c, &r_ij);

    let f_a = restrict_mle_pub(&mle_a, &r_i);
    let f_b = restrict_mle_pub(&mle_b_t, &r_j);

    let (a_commitment, _) = commit_mle(&f_a);
    let (b_commitment, _) = commit_mle(&f_b);

    Ok(BatchEntry {
        node_id,
        m,
        k,
        n,
        claimed_sum,
        r_i,
        r_j,
        a_commitment,
        b_commitment,
        f_a,
        f_b,
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
