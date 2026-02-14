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
// Uses strided accumulation to handle n_blocks > BLOCK_SIZE.
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

    // Strided accumulation: each thread sums multiple blocks when n_blocks > BLOCK_SIZE
    QM31 val = qm31_zero();
    for (uint32_t i = tid; i < n_blocks; i += BLOCK_SIZE) {
        uint32_t base = i * 4;
        QM31 v = {channel_partials[base], channel_partials[base+1],
                  channel_partials[base+2], channel_partials[base+3]};
        val = qm31_add(val, v);
    }

    uint32_t base = tid * 4;
    s_data[base+0] = val.a0; s_data[base+1] = val.a1;
    s_data[base+2] = val.a2; s_data[base+3] = val.a3;

    __syncthreads();

    // Tree reduction over BLOCK_SIZE threads
    for (uint32_t stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
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
// M31 GEMV / GEMM CUDA Kernel (for forward pass acceleration)
// =============================================================================

/// CUDA kernels for M31 matrix operations: GEMV (m=1), GEMM (m>1),
/// and element-wise add/mul/relu for forward pass acceleration.
#[cfg(feature = "cuda-runtime")]
const M31_FORWARD_KERNEL: &str = r#"
#define P 0x7FFFFFFFu

// --- GEMV: single-row matmul (m=1) ---
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
        if ((i & 3u) == 3u) {
            acc %= (unsigned long long)P;
        }
    }
    output[col] = (unsigned int)(acc % (unsigned long long)P);
}

// --- GEMM: multi-row matmul (m>1) ---
// Grid: (ceil(n/16), ceil(m/16)), Block: (16, 16)
// Each thread computes one element of C[row][col] = Σ A[row][l] × B[l][col]
extern "C" __global__ void m31_gemm_kernel(
    const unsigned int* a,       // m x k matrix (row-major, M31)
    const unsigned int* b,       // k x n matrix (row-major, M31)
    unsigned int* c,             // m x n output (row-major, M31)
    unsigned int m,
    unsigned int k,
    unsigned int n
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;

    unsigned long long acc = 0;
    for (unsigned int l = 0; l < k; l++) {
        unsigned long long av = (unsigned long long)a[row * k + l];
        unsigned long long bv = (unsigned long long)b[l * n + col];
        acc += av * bv;
        if ((l & 3u) == 3u) {
            acc %= (unsigned long long)P;
        }
    }
    c[row * n + col] = (unsigned int)(acc % (unsigned long long)P);
}

// --- Element-wise add ---
extern "C" __global__ void m31_add_kernel(
    const unsigned int* lhs,
    const unsigned int* rhs,
    unsigned int* output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int sum = lhs[idx] + rhs[idx];
    output[idx] = (sum >= P) ? (sum - P) : sum;
}

// --- Element-wise mul ---
extern "C" __global__ void m31_mul_kernel(
    const unsigned int* lhs,
    const unsigned int* rhs,
    unsigned int* output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned long long prod = (unsigned long long)lhs[idx] * (unsigned long long)rhs[idx];
    unsigned int lo = (unsigned int)(prod & P);
    unsigned int hi = (unsigned int)(prod >> 31);
    unsigned int result = lo + hi;
    output[idx] = (result >= P) ? (result - P) : result;
}

// --- ReLU activation ---
// M31 convention: values in [0, P/2] are "non-negative", [P/2+1, P-1] are "negative"
extern "C" __global__ void m31_relu_kernel(
    const unsigned int* input,
    unsigned int* output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int val = input[idx];
    output[idx] = (val <= (P >> 1)) ? val : 0u;
}
"#;

// =============================================================================
// GPU Fused MLE Restrict CUDA Kernels
// =============================================================================

/// CUDA kernels for fused MLE restriction — takes an M31 matrix and QM31
/// Lagrange basis, produces the restricted QM31 vector directly on GPU.
///
/// Eliminates the CPU pipeline: pad → matrix_to_mle → restrict_mle
/// For Qwen3-14B: avoids 2 GB MLE allocation + 24s of CPU folding.
#[cfg(feature = "cuda-runtime")]
const MLE_RESTRICT_KERNEL: &str = r#"
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define M31_PRIME 0x7FFFFFFFu

__device__ __forceinline__ uint32_t m31_add_r(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return (sum >= M31_PRIME) ? (sum - M31_PRIME) : sum;
}

__device__ __forceinline__ uint32_t m31_sub_r(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (a + M31_PRIME - b);
}

__device__ __forceinline__ uint32_t m31_mul_r(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t result = lo + hi;
    return (result >= M31_PRIME) ? (result - M31_PRIME) : result;
}

struct CM31R {
    uint32_t real;
    uint32_t imag;
};

__device__ __forceinline__ CM31R cm31_add_r(CM31R a, CM31R b) {
    CM31R r;
    r.real = m31_add_r(a.real, b.real);
    r.imag = m31_add_r(a.imag, b.imag);
    return r;
}

// CM31 multiplication: (a + ub)(c + ud) = (ac + 2bd) + u(ad + bc)
__device__ __forceinline__ CM31R cm31_mul_r(CM31R a, CM31R b) {
    uint32_t ac = m31_mul_r(a.real, b.real);
    uint32_t bd = m31_mul_r(a.imag, b.imag);
    uint32_t ad = m31_mul_r(a.real, b.imag);
    uint32_t bc = m31_mul_r(a.imag, b.real);
    CM31R r;
    r.real = m31_add_r(ac, m31_add_r(bd, bd));
    r.imag = m31_add_r(ad, bc);
    return r;
}

struct QM31R {
    uint32_t a0, a1, a2, a3;
};

__device__ __forceinline__ QM31R qm31_zero_r() {
    QM31R r = {0, 0, 0, 0};
    return r;
}

__device__ __forceinline__ QM31R qm31_add_r(QM31R x, QM31R y) {
    QM31R r;
    r.a0 = m31_add_r(x.a0, y.a0);
    r.a1 = m31_add_r(x.a1, y.a1);
    r.a2 = m31_add_r(x.a2, y.a2);
    r.a3 = m31_add_r(x.a3, y.a3);
    return r;
}

__device__ __forceinline__ QM31R qm31_mul_r(QM31R x, QM31R y) {
    CM31R x0 = {x.a0, x.a1};
    CM31R x1 = {x.a2, x.a3};
    CM31R y0 = {y.a0, y.a1};
    CM31R y1 = {y.a2, y.a3};

    CM31R x0y0 = cm31_mul_r(x0, y0);
    CM31R x1y1 = cm31_mul_r(x1, y1);
    CM31R x0y1 = cm31_mul_r(x0, y1);
    CM31R x1y0 = cm31_mul_r(x1, y0);

    CM31R u_x1y1 = {m31_add_r(x1y1.imag, x1y1.imag), x1y1.real};
    CM31R term = cm31_add_r(u_x1y1, cm31_add_r(x1y1, x1y1));

    CM31R real_part = cm31_add_r(x0y0, term);
    CM31R imag_part = cm31_add_r(x0y1, x1y0);

    QM31R r;
    r.a0 = real_part.real;
    r.a1 = real_part.imag;
    r.a2 = imag_part.real;
    r.a3 = imag_part.imag;
    return r;
}

// Multiply QM31 by a scalar M31 (common case: M31→QM31 lifting)
__device__ __forceinline__ QM31R qm31_scale_m31(QM31R x, uint32_t s) {
    QM31R r;
    r.a0 = m31_mul_r(x.a0, s);
    r.a1 = m31_mul_r(x.a1, s);
    r.a2 = m31_mul_r(x.a2, s);
    r.a3 = m31_mul_r(x.a3, s);
    return r;
}

// Fused row-restrict kernel:
//   f_a[j] = Σ_{i=0}^{m_orig-1} M31_to_QM31(A[i*k_orig + j]) × lagrange[i]
// for j in 0..k_orig. Output zero-padded to k_padded.
//
// Each thread computes one output element (one column dot product).
// Lagrange basis has padded_rows entries but only m_orig are used
// (rest multiply zero-padding in the original algorithm).
extern "C" __global__ void m31_restrict_rows_kernel(
    const uint32_t* __restrict__ matrix,    // M31 matrix, m_orig × k_orig (row-major)
    const uint32_t* __restrict__ lagrange,  // QM31 Lagrange basis, m_orig entries × 4 u32 each
    uint32_t* __restrict__ output,          // QM31 output, k_padded entries × 4 u32 each
    uint32_t m_orig,
    uint32_t k_orig,
    uint32_t k_padded
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k_padded) return;

    QM31R sum = qm31_zero_r();

    if (j < k_orig) {
        for (uint32_t i = 0; i < m_orig; i++) {
            uint32_t mat_val = matrix[i * k_orig + j];
            uint32_t lb = i * 4;
            QM31R lag = {lagrange[lb], lagrange[lb+1], lagrange[lb+2], lagrange[lb+3]};
            // M31_to_QM31(mat_val) × lag = lag × mat_val (scalar multiply)
            QM31R term = qm31_scale_m31(lag, mat_val);
            sum = qm31_add_r(sum, term);
        }
    }
    // j >= k_orig: output zero (already initialized by qm31_zero_r)

    uint32_t out = j * 4;
    output[out]   = sum.a0;
    output[out+1] = sum.a1;
    output[out+2] = sum.a2;
    output[out+3] = sum.a3;
}

// Fused col-restrict kernel:
//   f_b[i] = Σ_{j=0}^{n_orig-1} M31_to_QM31(B[i*n_orig + j]) × lagrange[j]
// for i in 0..k_orig. Output zero-padded to k_padded.
//
// Same pattern: each thread computes one output element (one row dot product).
extern "C" __global__ void m31_restrict_cols_kernel(
    const uint32_t* __restrict__ matrix,    // M31 matrix, k_orig × n_orig (row-major)
    const uint32_t* __restrict__ lagrange,  // QM31 Lagrange basis, n_orig entries × 4 u32 each
    uint32_t* __restrict__ output,          // QM31 output, k_padded entries × 4 u32 each
    uint32_t k_orig,
    uint32_t n_orig,
    uint32_t k_padded
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k_padded) return;

    QM31R sum = qm31_zero_r();

    if (i < k_orig) {
        for (uint32_t j = 0; j < n_orig; j++) {
            uint32_t mat_val = matrix[i * n_orig + j];
            uint32_t lb = j * 4;
            QM31R lag = {lagrange[lb], lagrange[lb+1], lagrange[lb+2], lagrange[lb+3]};
            QM31R term = qm31_scale_m31(lag, mat_val);
            sum = qm31_add_r(sum, term);
        }
    }

    uint32_t out = i * 4;
    output[out]   = sum.a0;
    output[out+1] = sum.a1;
    output[out+2] = sum.a2;
    output[out+3] = sum.a3;
}
"#;

// =============================================================================
// LogUp CUDA Kernels
// =============================================================================

/// CUDA kernels for LogUp eq-sumcheck (activation/layernorm).
///
/// Uses `_l` suffix field ops to avoid name collisions with other kernel modules.
/// Provides:
/// - `logup_denominator_kernel`: d[i] = gamma - input[i] - beta * output[i]
/// - `logup_3way_round_kernel`: 3-factor (eq × w × d) round poly at t=0,1,2,3
/// - `logup_4way_reduce_kernel`: cross-block reduction for 4 channels
/// - `logup_3way_fold_kernel`: fold 3 MLEs simultaneously with same challenge
/// - `combine_blocks_kernel`: result[i] = Σ_b weights[b] * blocks[b][i]
#[cfg(feature = "cuda-runtime")]
const LOGUP_CUDA_KERNEL: &str = r#"
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define M31_P 0x7FFFFFFFu
#define BLOCK_SZ 256

__device__ __forceinline__ uint32_t m31_add_l(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return (s >= M31_P) ? (s - M31_P) : s;
}

__device__ __forceinline__ uint32_t m31_sub_l(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (a + M31_P - b);
}

__device__ __forceinline__ uint32_t m31_mul_l(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t r = lo + hi;
    return (r >= M31_P) ? (r - M31_P) : r;
}

struct CM31L { uint32_t real; uint32_t imag; };

__device__ __forceinline__ CM31L cm31_add_l(CM31L a, CM31L b) {
    return {m31_add_l(a.real, b.real), m31_add_l(a.imag, b.imag)};
}

__device__ __forceinline__ CM31L cm31_sub_l(CM31L a, CM31L b) {
    return {m31_sub_l(a.real, b.real), m31_sub_l(a.imag, b.imag)};
}

__device__ __forceinline__ CM31L cm31_mul_l(CM31L a, CM31L b) {
    uint32_t ac = m31_mul_l(a.real, b.real);
    uint32_t bd = m31_mul_l(a.imag, b.imag);
    uint32_t ad = m31_mul_l(a.real, b.imag);
    uint32_t bc = m31_mul_l(a.imag, b.real);
    return {m31_add_l(ac, m31_add_l(bd, bd)), m31_add_l(ad, bc)};
}

struct QM31L { uint32_t a0, a1, a2, a3; };

__device__ __forceinline__ QM31L qm31_zero_l() { return {0, 0, 0, 0}; }

__device__ __forceinline__ QM31L qm31_add_l(QM31L x, QM31L y) {
    return {m31_add_l(x.a0, y.a0), m31_add_l(x.a1, y.a1),
            m31_add_l(x.a2, y.a2), m31_add_l(x.a3, y.a3)};
}

__device__ __forceinline__ QM31L qm31_sub_l(QM31L x, QM31L y) {
    return {m31_sub_l(x.a0, y.a0), m31_sub_l(x.a1, y.a1),
            m31_sub_l(x.a2, y.a2), m31_sub_l(x.a3, y.a3)};
}

__device__ __forceinline__ QM31L qm31_mul_l(QM31L x, QM31L y) {
    CM31L x0 = {x.a0, x.a1}, x1 = {x.a2, x.a3};
    CM31L y0 = {y.a0, y.a1}, y1 = {y.a2, y.a3};
    CM31L x0y0 = cm31_mul_l(x0, y0);
    CM31L x1y1 = cm31_mul_l(x1, y1);
    CM31L x0y1 = cm31_mul_l(x0, y1);
    CM31L x1y0 = cm31_mul_l(x1, y0);
    CM31L u_x1y1 = {m31_add_l(x1y1.imag, x1y1.imag), x1y1.real};
    CM31L term = cm31_add_l(u_x1y1, cm31_add_l(x1y1, x1y1));
    CM31L rp = cm31_add_l(x0y0, term);
    CM31L ip = cm31_add_l(x0y1, x1y0);
    return {rp.real, rp.imag, ip.real, ip.imag};
}

__device__ __forceinline__ QM31L qm31_scale_l(QM31L x, uint32_t s) {
    return {m31_mul_l(x.a0, s), m31_mul_l(x.a1, s),
            m31_mul_l(x.a2, s), m31_mul_l(x.a3, s)};
}

// d[i] = gamma - input[i] - beta * output[i]
extern "C" __global__ void logup_denominator_kernel(
    const uint32_t* __restrict__ input,
    const uint32_t* __restrict__ output,
    const uint32_t* __restrict__ gamma,
    const uint32_t* __restrict__ beta,
    uint32_t* __restrict__ d_out,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    QM31L g = {gamma[0], gamma[1], gamma[2], gamma[3]};
    QM31L b = {beta[0], beta[1], beta[2], beta[3]};

    uint32_t off = idx * 4;
    QM31L inp = {input[off], input[off+1], input[off+2], input[off+3]};
    QM31L out = {output[off], output[off+1], output[off+2], output[off+3]};

    // d = gamma - input - beta * output
    QM31L beta_out = qm31_mul_l(b, out);
    QM31L result = qm31_sub_l(qm31_sub_l(g, inp), beta_out);

    d_out[off]   = result.a0;
    d_out[off+1] = result.a1;
    d_out[off+2] = result.a2;
    d_out[off+3] = result.a3;
}

// 3-way eq-sumcheck round: Σ eq_t[i] * w_t[i] * d_t[i] at t=0,1,2,3
// Each factor interpolated at t: f_t = (1-t)*f[i] + t*f[mid+i]
extern "C" __global__ void logup_3way_round_kernel(
    const uint32_t* __restrict__ eq,
    const uint32_t* __restrict__ w,
    const uint32_t* __restrict__ d,
    uint32_t* __restrict__ block_s0,
    uint32_t* __restrict__ block_s1,
    uint32_t* __restrict__ block_s2,
    uint32_t* __restrict__ block_s3,
    uint32_t mid
) {
    __shared__ uint32_t sh0[BLOCK_SZ * 4];
    __shared__ uint32_t sh1[BLOCK_SZ * 4];
    __shared__ uint32_t sh2[BLOCK_SZ * 4];
    __shared__ uint32_t sh3[BLOCK_SZ * 4];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + tid;

    QM31L ls0 = qm31_zero_l(), ls1 = qm31_zero_l();
    QM31L ls2 = qm31_zero_l(), ls3 = qm31_zero_l();

    if (idx < mid) {
        uint32_t lo = idx * 4, hi = (mid + idx) * 4;

        QM31L eq0 = {eq[lo], eq[lo+1], eq[lo+2], eq[lo+3]};
        QM31L eq1 = {eq[hi], eq[hi+1], eq[hi+2], eq[hi+3]};
        QM31L w0 = {w[lo], w[lo+1], w[lo+2], w[lo+3]};
        QM31L w1 = {w[hi], w[hi+1], w[hi+2], w[hi+3]};
        QM31L d0 = {d[lo], d[lo+1], d[lo+2], d[lo+3]};
        QM31L d1 = {d[hi], d[hi+1], d[hi+2], d[hi+3]};

        // t=0: eq0 * w0 * d0
        ls0 = qm31_mul_l(qm31_mul_l(eq0, w0), d0);

        // t=1: eq1 * w1 * d1
        ls1 = qm31_mul_l(qm31_mul_l(eq1, w1), d1);

        // t=2: (2*eq1 - eq0) * (2*w1 - w0) * (2*d1 - d0)
        QM31L eq2 = qm31_sub_l(qm31_add_l(eq1, eq1), eq0);
        QM31L w2_ = qm31_sub_l(qm31_add_l(w1, w1), w0);
        QM31L d2_ = qm31_sub_l(qm31_add_l(d1, d1), d0);
        ls2 = qm31_mul_l(qm31_mul_l(eq2, w2_), d2_);

        // t=3: (3*eq1 - 2*eq0) * (3*w1 - 2*w0) * (3*d1 - 2*d0)
        QM31L eq3 = qm31_sub_l(qm31_add_l(qm31_add_l(eq1, eq1), eq1),
                                qm31_add_l(eq0, eq0));
        QM31L w3_ = qm31_sub_l(qm31_add_l(qm31_add_l(w1, w1), w1),
                                qm31_add_l(w0, w0));
        QM31L d3_ = qm31_sub_l(qm31_add_l(qm31_add_l(d1, d1), d1),
                                qm31_add_l(d0, d0));
        ls3 = qm31_mul_l(qm31_mul_l(eq3, w3_), d3_);
    }

    // Store to shared memory
    uint32_t b = tid * 4;
    sh0[b] = ls0.a0; sh0[b+1] = ls0.a1; sh0[b+2] = ls0.a2; sh0[b+3] = ls0.a3;
    sh1[b] = ls1.a0; sh1[b+1] = ls1.a1; sh1[b+2] = ls1.a2; sh1[b+3] = ls1.a3;
    sh2[b] = ls2.a0; sh2[b+1] = ls2.a1; sh2[b+2] = ls2.a2; sh2[b+3] = ls2.a3;
    sh3[b] = ls3.a0; sh3[b+1] = ls3.a1; sh3[b+2] = ls3.a2; sh3[b+3] = ls3.a3;
    __syncthreads();

    // Block-level tree reduction
    for (uint32_t stride = BLOCK_SZ / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            uint32_t m = tid * 4, o = (tid + stride) * 4;

            QM31L a0 = {sh0[m], sh0[m+1], sh0[m+2], sh0[m+3]};
            QM31L b0 = {sh0[o], sh0[o+1], sh0[o+2], sh0[o+3]};
            QM31L s0 = qm31_add_l(a0, b0);
            sh0[m] = s0.a0; sh0[m+1] = s0.a1; sh0[m+2] = s0.a2; sh0[m+3] = s0.a3;

            QM31L a1 = {sh1[m], sh1[m+1], sh1[m+2], sh1[m+3]};
            QM31L b1 = {sh1[o], sh1[o+1], sh1[o+2], sh1[o+3]};
            QM31L s1 = qm31_add_l(a1, b1);
            sh1[m] = s1.a0; sh1[m+1] = s1.a1; sh1[m+2] = s1.a2; sh1[m+3] = s1.a3;

            QM31L a2 = {sh2[m], sh2[m+1], sh2[m+2], sh2[m+3]};
            QM31L b2 = {sh2[o], sh2[o+1], sh2[o+2], sh2[o+3]};
            QM31L s2 = qm31_add_l(a2, b2);
            sh2[m] = s2.a0; sh2[m+1] = s2.a1; sh2[m+2] = s2.a2; sh2[m+3] = s2.a3;

            QM31L a3 = {sh3[m], sh3[m+1], sh3[m+2], sh3[m+3]};
            QM31L b3 = {sh3[o], sh3[o+1], sh3[o+2], sh3[o+3]};
            QM31L s3 = qm31_add_l(a3, b3);
            sh3[m] = s3.a0; sh3[m+1] = s3.a1; sh3[m+2] = s3.a2; sh3[m+3] = s3.a3;
        }
        __syncthreads();
    }

    if (tid == 0) {
        uint32_t blk = blockIdx.x * 4;
        block_s0[blk] = sh0[0]; block_s0[blk+1] = sh0[1]; block_s0[blk+2] = sh0[2]; block_s0[blk+3] = sh0[3];
        block_s1[blk] = sh1[0]; block_s1[blk+1] = sh1[1]; block_s1[blk+2] = sh1[2]; block_s1[blk+3] = sh1[3];
        block_s2[blk] = sh2[0]; block_s2[blk+1] = sh2[1]; block_s2[blk+2] = sh2[2]; block_s2[blk+3] = sh2[3];
        block_s3[blk] = sh3[0]; block_s3[blk+1] = sh3[1]; block_s3[blk+2] = sh3[2]; block_s3[blk+3] = sh3[3];
    }
}

// 4-channel reduction kernel (for logup_3way_round_kernel's 4 outputs)
// Uses strided accumulation to handle n_blocks > BLOCK_SZ (256).
extern "C" __global__ void logup_4way_reduce_kernel(
    const uint32_t* __restrict__ partials,
    uint32_t* __restrict__ output,
    uint32_t n_blocks
) {
    uint32_t channel = blockIdx.x;
    if (channel >= 4) return;

    __shared__ uint32_t s_data[BLOCK_SZ * 4];
    uint32_t tid = threadIdx.x;
    const uint32_t* ch_partials = partials + channel * n_blocks * 4;

    // Strided accumulation: each thread sums multiple blocks
    QM31L val = qm31_zero_l();
    for (uint32_t i = tid; i < n_blocks; i += BLOCK_SZ) {
        uint32_t b = i * 4;
        QM31L v = {ch_partials[b], ch_partials[b+1], ch_partials[b+2], ch_partials[b+3]};
        val = qm31_add_l(val, v);
    }

    uint32_t b = tid * 4;
    s_data[b] = val.a0; s_data[b+1] = val.a1; s_data[b+2] = val.a2; s_data[b+3] = val.a3;
    __syncthreads();

    // Tree reduction over BLOCK_SZ threads (not n_blocks — all threads participate)
    for (uint32_t stride = BLOCK_SZ / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            uint32_t m = tid * 4, o = (tid + stride) * 4;
            QM31L a = {s_data[m], s_data[m+1], s_data[m+2], s_data[m+3]};
            QM31L bv = {s_data[o], s_data[o+1], s_data[o+2], s_data[o+3]};
            QM31L s = qm31_add_l(a, bv);
            s_data[m] = s.a0; s_data[m+1] = s.a1; s_data[m+2] = s.a2; s_data[m+3] = s.a3;
        }
        __syncthreads();
    }

    if (tid == 0) {
        uint32_t out = channel * 4;
        output[out] = s_data[0]; output[out+1] = s_data[1];
        output[out+2] = s_data[2]; output[out+3] = s_data[3];
    }
}

// Fold 3 MLEs (eq, w, d) simultaneously with the same challenge alpha
extern "C" __global__ void logup_3way_fold_kernel(
    const uint32_t* __restrict__ eq_in,
    const uint32_t* __restrict__ w_in,
    const uint32_t* __restrict__ d_in,
    const uint32_t* __restrict__ alpha,
    uint32_t* __restrict__ eq_out,
    uint32_t* __restrict__ w_out,
    uint32_t* __restrict__ d_out,
    uint32_t half_n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    QM31L a = {alpha[0], alpha[1], alpha[2], alpha[3]};
    uint32_t lo = idx * 4, hi = (half_n + idx) * 4;

    // eq fold
    QM31L eq_lo = {eq_in[lo], eq_in[lo+1], eq_in[lo+2], eq_in[lo+3]};
    QM31L eq_hi = {eq_in[hi], eq_in[hi+1], eq_in[hi+2], eq_in[hi+3]};
    QM31L eq_r = qm31_add_l(eq_lo, qm31_mul_l(a, qm31_sub_l(eq_hi, eq_lo)));
    eq_out[lo] = eq_r.a0; eq_out[lo+1] = eq_r.a1; eq_out[lo+2] = eq_r.a2; eq_out[lo+3] = eq_r.a3;

    // w fold
    QM31L w_lo = {w_in[lo], w_in[lo+1], w_in[lo+2], w_in[lo+3]};
    QM31L w_hi = {w_in[hi], w_in[hi+1], w_in[hi+2], w_in[hi+3]};
    QM31L w_r = qm31_add_l(w_lo, qm31_mul_l(a, qm31_sub_l(w_hi, w_lo)));
    w_out[lo] = w_r.a0; w_out[lo+1] = w_r.a1; w_out[lo+2] = w_r.a2; w_out[lo+3] = w_r.a3;

    // d fold
    QM31L d_lo = {d_in[lo], d_in[lo+1], d_in[lo+2], d_in[lo+3]};
    QM31L d_hi = {d_in[hi], d_in[hi+1], d_in[hi+2], d_in[hi+3]};
    QM31L d_r = qm31_add_l(d_lo, qm31_mul_l(a, qm31_sub_l(d_hi, d_lo)));
    d_out[lo] = d_r.a0; d_out[lo+1] = d_r.a1; d_out[lo+2] = d_r.a2; d_out[lo+3] = d_r.a3;
}

// Weighted block combination: result[i] = Σ_b weights[b] * blocks[b][i]
// blocks: n_blocks * n_elems * 4 u32 (blocks laid out contiguously)
// weights: n_blocks * 4 u32
extern "C" __global__ void combine_blocks_kernel(
    const uint32_t* __restrict__ blocks,
    const uint32_t* __restrict__ weights,
    uint32_t* __restrict__ output,
    uint32_t n_elems,
    uint32_t n_blocks
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;

    QM31L acc = qm31_zero_l();
    for (uint32_t b = 0; b < n_blocks; b++) {
        uint32_t wb = b * 4;
        QM31L w = {weights[wb], weights[wb+1], weights[wb+2], weights[wb+3]};
        uint32_t off = (b * n_elems + idx) * 4;
        QM31L val = {blocks[off], blocks[off+1], blocks[off+2], blocks[off+3]};
        acc = qm31_add_l(acc, qm31_mul_l(w, val));
    }

    uint32_t out = idx * 4;
    output[out] = acc.a0; output[out+1] = acc.a1; output[out+2] = acc.a2; output[out+3] = acc.a3;
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

/// Compiled CUDA functions for sumcheck and forward pass operations.
#[cfg(feature = "cuda-runtime")]
pub struct GpuSumcheckExecutor {
    pub device: Arc<CudaDevice>,
    sumcheck_round_fn: CudaFunction,
    sumcheck_reduce_fn: CudaFunction,
    mle_fold_fn: CudaFunction,
    /// Lazily compiled forward pass kernel functions.
    forward_fns: std::sync::Mutex<Option<ForwardKernels>>,
    /// Lazily compiled MLE restrict kernel functions.
    restrict_fns: std::sync::Mutex<Option<RestrictKernels>>,
    /// Lazily compiled LogUp kernel functions.
    logup_fns: std::sync::Mutex<Option<LogupKernels>>,
}

/// Forward pass CUDA kernel handles.
#[cfg(feature = "cuda-runtime")]
struct ForwardKernels {
    gemv_fn: CudaFunction,
    gemm_fn: CudaFunction,
    add_fn: CudaFunction,
    mul_fn: CudaFunction,
    relu_fn: CudaFunction,
}

/// MLE restrict CUDA kernel handles.
#[cfg(feature = "cuda-runtime")]
struct RestrictKernels {
    restrict_rows_fn: CudaFunction,
    restrict_cols_fn: CudaFunction,
}

/// LogUp CUDA kernel handles for activation/layernorm eq-sumcheck.
#[cfg(feature = "cuda-runtime")]
struct LogupKernels {
    denominator_fn: CudaFunction,
    round_3way_fn: CudaFunction,
    reduce_4way_fn: CudaFunction,
    fold_3way_fn: CudaFunction,
    combine_blocks_fn: CudaFunction,
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
            forward_fns: std::sync::Mutex::new(None),
            restrict_fns: std::sync::Mutex::new(None),
            logup_fns: std::sync::Mutex::new(None),
        })
    }

    /// Get or create a cached global GPU executor.
    ///
    /// CUDA kernels are compiled via NVRTC exactly once on first call,
    /// then reused for all subsequent sumcheck proofs. This avoids
    /// recompiling ~200ms of NVRTC per matmul × 160+ matmuls = 32 seconds wasted.
    ///
    /// When multi-GPU is enabled and a thread-local device affinity is set
    /// (via [`crate::multi_gpu::set_thread_device`]), this returns the executor
    /// for that device instead of the default device 0.
    pub fn cached() -> Result<Arc<Self>, CudaFftError> {
        #[cfg(feature = "multi-gpu")]
        if let Some(device_id) = crate::multi_gpu::get_thread_device() {
            return Self::cached_for_device(device_id);
        }

        static EXECUTOR: OnceLock<Arc<GpuSumcheckExecutor>> = OnceLock::new();
        EXECUTOR.get_or_try_init(|| {
            eprintln!("[GPU] Compiling sumcheck CUDA kernels (one-time)...");
            let executor = GpuSumcheckExecutor::new()?;
            eprintln!("[GPU] Kernels compiled and cached.");
            Ok(Arc::new(executor))
        }).cloned()
    }

    /// Create a new GPU sumcheck executor on a specific device.
    ///
    /// Uses STWO's `get_executor_for_device()` to obtain a per-device CUDA context,
    /// then compiles the sumcheck kernels on that device.
    #[cfg(feature = "multi-gpu")]
    pub fn new_on_device(device_id: usize) -> Result<Self, CudaFftError> {
        use stwo::prover::backend::gpu::cuda_executor::get_executor_for_device;

        let executor = get_executor_for_device(device_id)?;
        let device = executor.device.clone();

        // PTX is device-independent, but load_ptx must run on the target device context
        let ptx = cudarc::nvrtc::compile_ptx(SUMCHECK_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("sumcheck kernel (device {}): {:?}", device_id, e)))?;

        device.load_ptx(ptx, "sumcheck", &[
            "sumcheck_round_kernel",
            "sumcheck_reduce_kernel",
            "mle_fold_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("load sumcheck PTX (device {}): {:?}", device_id, e)))?;

        let sumcheck_round_fn = device.get_func("sumcheck", "sumcheck_round_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                format!("sumcheck_round_kernel not found on device {}", device_id),
            ))?;

        let sumcheck_reduce_fn = device.get_func("sumcheck", "sumcheck_reduce_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                format!("sumcheck_reduce_kernel not found on device {}", device_id),
            ))?;

        let mle_fold_fn = device.get_func("sumcheck", "mle_fold_kernel")
            .ok_or_else(|| CudaFftError::KernelCompilation(
                format!("mle_fold_kernel not found on device {}", device_id),
            ))?;

        Ok(Self {
            device,
            sumcheck_round_fn,
            sumcheck_reduce_fn,
            mle_fold_fn,
            forward_fns: std::sync::Mutex::new(None),
            restrict_fns: std::sync::Mutex::new(None),
            logup_fns: std::sync::Mutex::new(None),
        })
    }

    /// Get or create a cached GPU executor for a specific device.
    ///
    /// Per-device executor pool, similar to STWO's `CUDA_EXECUTOR_POOL`.
    /// Kernels are compiled once per device on first access.
    #[cfg(feature = "multi-gpu")]
    pub fn cached_for_device(device_id: usize) -> Result<Arc<Self>, CudaFftError> {
        use std::collections::HashMap;
        use std::sync::Mutex;

        static MULTI_EXECUTORS: OnceLock<Mutex<HashMap<usize, Arc<GpuSumcheckExecutor>>>> = OnceLock::new();

        let pool = MULTI_EXECUTORS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = pool.lock().map_err(|_| {
            CudaFftError::KernelCompilation("Failed to acquire multi-GPU executor pool lock".into())
        })?;

        if let Some(executor) = guard.get(&device_id) {
            return Ok(Arc::clone(executor));
        }

        eprintln!("[GPU] Compiling sumcheck CUDA kernels for device {device_id} (one-time)...");
        let executor = GpuSumcheckExecutor::new_on_device(device_id)?;
        let executor_arc = Arc::new(executor);
        guard.insert(device_id, Arc::clone(&executor_arc));
        eprintln!("[GPU] Kernels compiled and cached for device {device_id}.");

        Ok(executor_arc)
    }

    /// Get or lazily compile forward pass kernels (GEMV, GEMM, add, mul, relu).
    fn get_forward_fns(&self) -> Result<ForwardKernels, CudaFftError> {
        let mut guard = self.forward_fns.lock().unwrap();
        if let Some(ref fns) = *guard {
            return Ok(ForwardKernels {
                gemv_fn: fns.gemv_fn.clone(),
                gemm_fn: fns.gemm_fn.clone(),
                add_fn: fns.add_fn.clone(),
                mul_fn: fns.mul_fn.clone(),
                relu_fn: fns.relu_fn.clone(),
            });
        }

        let ptx = cudarc::nvrtc::compile_ptx(M31_FORWARD_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("forward kernel: {:?}", e)))?;

        self.device.load_ptx(ptx, "forward", &[
            "m31_gemv_kernel", "m31_gemm_kernel",
            "m31_add_kernel", "m31_mul_kernel", "m31_relu_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("load forward PTX: {:?}", e)))?;

        let fns = ForwardKernels {
            gemv_fn: self.device.get_func("forward", "m31_gemv_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("m31_gemv_kernel not found".into()))?,
            gemm_fn: self.device.get_func("forward", "m31_gemm_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("m31_gemm_kernel not found".into()))?,
            add_fn: self.device.get_func("forward", "m31_add_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("m31_add_kernel not found".into()))?,
            mul_fn: self.device.get_func("forward", "m31_mul_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("m31_mul_kernel not found".into()))?,
            relu_fn: self.device.get_func("forward", "m31_relu_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("m31_relu_kernel not found".into()))?,
        };

        *guard = Some(ForwardKernels {
            gemv_fn: fns.gemv_fn.clone(),
            gemm_fn: fns.gemm_fn.clone(),
            add_fn: fns.add_fn.clone(),
            mul_fn: fns.mul_fn.clone(),
            relu_fn: fns.relu_fn.clone(),
        });
        Ok(fns)
    }

    /// Get or lazily compile the GEMV kernel (backwards compat).
    pub fn get_gemv_fn(&self) -> Result<CudaFunction, CudaFftError> {
        Ok(self.get_forward_fns()?.gemv_fn)
    }

    /// Get or lazily compile MLE restrict kernels.
    fn get_restrict_fns(&self) -> Result<RestrictKernels, CudaFftError> {
        let mut guard = self.restrict_fns.lock().unwrap();
        if let Some(ref fns) = *guard {
            return Ok(RestrictKernels {
                restrict_rows_fn: fns.restrict_rows_fn.clone(),
                restrict_cols_fn: fns.restrict_cols_fn.clone(),
            });
        }

        let ptx = cudarc::nvrtc::compile_ptx(MLE_RESTRICT_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("restrict kernel: {:?}", e)))?;

        self.device.load_ptx(ptx, "restrict", &[
            "m31_restrict_rows_kernel", "m31_restrict_cols_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("load restrict PTX: {:?}", e)))?;

        let fns = RestrictKernels {
            restrict_rows_fn: self.device.get_func("restrict", "m31_restrict_rows_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("m31_restrict_rows_kernel not found".into()))?,
            restrict_cols_fn: self.device.get_func("restrict", "m31_restrict_cols_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("m31_restrict_cols_kernel not found".into()))?,
        };

        *guard = Some(RestrictKernels {
            restrict_rows_fn: fns.restrict_rows_fn.clone(),
            restrict_cols_fn: fns.restrict_cols_fn.clone(),
        });
        Ok(fns)
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

    /// GPU fused row-restrict: uploads M31 matrix + Lagrange basis, returns QM31 on GPU.
    ///
    /// Equivalent to CPU `restrict_rows_unpadded(matrix, challenges, k_padded)` but
    /// runs entirely on GPU. Avoids allocating 2 GB of intermediate MLE data.
    ///
    /// Returns a device buffer of `k_padded * 4` u32 (QM31 elements).
    pub fn restrict_rows(
        &self,
        matrix: &M31Matrix,
        challenges: &[SecureField],
        k_padded: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let fns = self.get_restrict_fns()?;
        let m_orig = matrix.rows;
        let k_orig = matrix.cols;

        // Compute Lagrange basis on CPU (fast: O(padded_rows), ~100μs)
        let lagrange = crate::components::matmul::compute_lagrange_basis_pub(challenges);
        // Only need the first m_orig entries (rest multiply zero-padding)
        let m_used = m_orig.min(lagrange.len());

        // Upload M31 matrix
        let mat_u32: Vec<u32> = matrix.data.iter().map(|v| v.0).collect();
        let d_matrix = self.device.htod_sync_copy(&mat_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("restrict_rows upload matrix: {:?}", e)))?;

        // Upload Lagrange basis (only first m_used entries as QM31)
        let lag_u32: Vec<u32> = lagrange[..m_used].iter()
            .flat_map(|sf| secure_field_to_u32s(*sf))
            .collect();
        let d_lagrange = self.device.htod_sync_copy(&lag_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("restrict_rows upload lagrange: {:?}", e)))?;

        // Allocate output
        let mut d_output = unsafe { self.device.alloc::<u32>(k_padded * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("restrict_rows output: {:?}", e)))?;

        let block_size = 256u32;
        let grid_size = (k_padded as u32 + block_size - 1) / block_size;

        unsafe {
            fns.restrict_rows_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &d_matrix,
                    &d_lagrange,
                    &mut d_output,
                    m_used as u32,
                    k_orig as u32,
                    k_padded as u32,
                ),
            ).map_err(|e| CudaFftError::KernelExecution(format!("restrict_rows: {:?}", e)))?;
        }

        Ok(d_output)
    }

    /// GPU fused col-restrict: uploads M31 matrix + Lagrange basis, returns QM31 on GPU.
    ///
    /// Equivalent to CPU `restrict_cols_unpadded(matrix, challenges, k_padded)` but
    /// runs entirely on GPU.
    ///
    /// Returns a device buffer of `k_padded * 4` u32 (QM31 elements).
    pub fn restrict_cols(
        &self,
        matrix: &M31Matrix,
        challenges: &[SecureField],
        k_padded: usize,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let fns = self.get_restrict_fns()?;
        let k_orig = matrix.rows;
        let n_orig = matrix.cols;

        // Compute Lagrange basis on CPU
        let lagrange = crate::components::matmul::compute_lagrange_basis_pub(challenges);
        let n_used = n_orig.min(lagrange.len());

        // Upload M31 matrix
        let mat_u32: Vec<u32> = matrix.data.iter().map(|v| v.0).collect();
        let d_matrix = self.device.htod_sync_copy(&mat_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("restrict_cols upload matrix: {:?}", e)))?;

        // Upload Lagrange basis (only first n_used entries)
        let lag_u32: Vec<u32> = lagrange[..n_used].iter()
            .flat_map(|sf| secure_field_to_u32s(*sf))
            .collect();
        let d_lagrange = self.device.htod_sync_copy(&lag_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("restrict_cols upload lagrange: {:?}", e)))?;

        // Allocate output
        let mut d_output = unsafe { self.device.alloc::<u32>(k_padded * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("restrict_cols output: {:?}", e)))?;

        let block_size = 256u32;
        let grid_size = (k_padded as u32 + block_size - 1) / block_size;

        unsafe {
            fns.restrict_cols_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &d_matrix,
                    &d_lagrange,
                    &mut d_output,
                    k_orig as u32,
                    n_used as u32,
                    k_padded as u32,
                ),
            ).map_err(|e| CudaFftError::KernelExecution(format!("restrict_cols: {:?}", e)))?;
        }

        Ok(d_output)
    }

    // =========================================================================
    // LogUp Kernel Management
    // =========================================================================

    /// Get or lazily compile LogUp kernels.
    fn get_logup_fns(&self) -> Result<LogupKernels, CudaFftError> {
        let mut guard = self.logup_fns.lock().unwrap();
        if let Some(ref fns) = *guard {
            return Ok(LogupKernels {
                denominator_fn: fns.denominator_fn.clone(),
                round_3way_fn: fns.round_3way_fn.clone(),
                reduce_4way_fn: fns.reduce_4way_fn.clone(),
                fold_3way_fn: fns.fold_3way_fn.clone(),
                combine_blocks_fn: fns.combine_blocks_fn.clone(),
            });
        }

        let ptx = cudarc::nvrtc::compile_ptx(LOGUP_CUDA_KERNEL)
            .map_err(|e| CudaFftError::KernelCompilation(format!("logup kernel: {:?}", e)))?;

        self.device.load_ptx(ptx, "logup", &[
            "logup_denominator_kernel",
            "logup_3way_round_kernel",
            "logup_4way_reduce_kernel",
            "logup_3way_fold_kernel",
            "combine_blocks_kernel",
        ]).map_err(|e| CudaFftError::KernelCompilation(format!("load logup PTX: {:?}", e)))?;

        let fns = LogupKernels {
            denominator_fn: self.device.get_func("logup", "logup_denominator_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("logup_denominator_kernel not found".into()))?,
            round_3way_fn: self.device.get_func("logup", "logup_3way_round_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("logup_3way_round_kernel not found".into()))?,
            reduce_4way_fn: self.device.get_func("logup", "logup_4way_reduce_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("logup_4way_reduce_kernel not found".into()))?,
            fold_3way_fn: self.device.get_func("logup", "logup_3way_fold_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("logup_3way_fold_kernel not found".into()))?,
            combine_blocks_fn: self.device.get_func("logup", "combine_blocks_kernel")
                .ok_or_else(|| CudaFftError::KernelCompilation("combine_blocks_kernel not found".into()))?,
        };

        *guard = Some(LogupKernels {
            denominator_fn: fns.denominator_fn.clone(),
            round_3way_fn: fns.round_3way_fn.clone(),
            reduce_4way_fn: fns.reduce_4way_fn.clone(),
            fold_3way_fn: fns.fold_3way_fn.clone(),
            combine_blocks_fn: fns.combine_blocks_fn.clone(),
        });
        Ok(fns)
    }

    // =========================================================================
    // Foundation GPU Methods (evaluate_mle_gpu + combine_blocks)
    // =========================================================================

    /// Evaluate an MLE at a point entirely on GPU using iterative folding.
    ///
    /// Uploads the MLE, iteratively applies `mle_fold` for each challenge in `point`,
    /// then downloads the single resulting QM31 scalar.
    pub fn evaluate_mle_gpu(
        &self,
        mle: &[SecureField],
        point: &[SecureField],
    ) -> Result<SecureField, CudaFftError> {
        let n = mle.len();
        if n == 0 {
            return Ok(SecureField::zero());
        }
        if n == 1 {
            return Ok(mle[0]);
        }

        let flat: Vec<u32> = mle.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
        let mut d_current = self.device.htod_sync_copy(&flat)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("evaluate_mle_gpu upload: {:?}", e)))?;
        let mut cur_n = n;

        for challenge in point.iter() {
            let alpha = secure_field_to_u32s(*challenge);
            d_current = self.mle_fold(&d_current, cur_n, &alpha)?;
            cur_n /= 2;
        }

        // Download final scalar (1 QM31 = 4 u32)
        let mut result = [0u32; 4];
        self.device.dtoh_sync_copy_into(&d_current, &mut result)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("evaluate_mle_gpu download: {:?}", e)))?;
        Ok(u32s_to_secure_field(&result))
    }

    /// Combine multiple equal-length blocks with per-block QM31 weights.
    ///
    /// result[i] = Σ_b weights[b] * blocks[b][i]
    ///
    /// Uses the `combine_blocks_kernel` from the LogUp kernel module.
    pub fn combine_blocks(
        &self,
        blocks: &[Vec<SecureField>],
        weights: &[SecureField],
    ) -> Result<Vec<SecureField>, CudaFftError> {
        let n_blocks = blocks.len();
        if n_blocks == 0 {
            return Ok(Vec::new());
        }
        let n_elems = blocks[0].len();
        if n_elems == 0 {
            return Ok(Vec::new());
        }

        let fns = self.get_logup_fns()?;

        // Pack blocks contiguously: [block_0_elem_0..block_0_elem_n, block_1_elem_0..]
        let blocks_flat: Vec<u32> = blocks.iter()
            .flat_map(|block| block.iter().flat_map(|sf| secure_field_to_u32s(*sf)))
            .collect();
        let weights_flat: Vec<u32> = weights.iter()
            .flat_map(|sf| secure_field_to_u32s(*sf))
            .collect();

        let d_blocks = self.device.htod_sync_copy(&blocks_flat)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("combine_blocks upload blocks: {:?}", e)))?;
        let d_weights = self.device.htod_sync_copy(&weights_flat)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("combine_blocks upload weights: {:?}", e)))?;
        let mut d_output = unsafe { self.device.alloc::<u32>(n_elems * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("combine_blocks output: {:?}", e)))?;

        let block_size = 256u32;
        let grid_size = (n_elems as u32 + block_size - 1) / block_size;

        unsafe {
            fns.combine_blocks_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_blocks, &d_weights, &mut d_output, n_elems as u32, n_blocks as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("combine_blocks: {:?}", e)))?;
        }

        // Download result
        let mut out_flat = vec![0u32; n_elems * 4];
        self.device.dtoh_sync_copy_into(&d_output, &mut out_flat)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("combine_blocks download: {:?}", e)))?;

        let result: Vec<SecureField> = out_flat.chunks_exact(4)
            .map(|c| u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(result)
    }

    // =========================================================================
    // LogUp GPU Executor Methods
    // =========================================================================

    /// Compute LogUp denominators on GPU: d[i] = gamma - input[i] - beta * output[i]
    ///
    /// Returns device buffer of n QM31 values (n * 4 u32).
    pub fn compute_logup_denominators_gpu(
        &self,
        input: &[SecureField],
        output: &[SecureField],
        gamma: SecureField,
        beta: SecureField,
    ) -> Result<CudaSlice<u32>, CudaFftError> {
        let n = input.len();
        assert_eq!(n, output.len());

        let fns = self.get_logup_fns()?;

        let input_flat: Vec<u32> = input.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
        let output_flat: Vec<u32> = output.iter().flat_map(|sf| secure_field_to_u32s(*sf)).collect();
        let gamma_u32 = secure_field_to_u32s(gamma);
        let beta_u32 = secure_field_to_u32s(beta);

        let d_input = self.device.htod_sync_copy(&input_flat)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("logup denom upload input: {:?}", e)))?;
        let d_output_mle = self.device.htod_sync_copy(&output_flat)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("logup denom upload output: {:?}", e)))?;
        let d_gamma = self.device.htod_sync_copy(&gamma_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("logup denom upload gamma: {:?}", e)))?;
        let d_beta = self.device.htod_sync_copy(&beta_u32)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("logup denom upload beta: {:?}", e)))?;
        let mut d_out = unsafe { self.device.alloc::<u32>(n * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("logup denom output: {:?}", e)))?;

        let block_size = 256u32;
        let grid_size = (n as u32 + block_size - 1) / block_size;

        unsafe {
            fns.denominator_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_input, &d_output_mle, &d_gamma, &d_beta, &mut d_out, n as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("logup_denominator: {:?}", e)))?;
        }

        Ok(d_out)
    }

    /// Compute 3-way eq-sumcheck round polynomial on GPU.
    ///
    /// Returns 4 QM31 sums: (s0, s1, s2, s3) for degree-3 interpolation.
    pub fn compute_logup_round_poly_3way(
        &self,
        d_eq: &CudaSlice<u32>,
        d_w: &CudaSlice<u32>,
        d_d: &CudaSlice<u32>,
        mid: usize,
    ) -> Result<([u32; 4], [u32; 4], [u32; 4], [u32; 4]), CudaFftError> {
        let fns = self.get_logup_fns()?;

        let block_size = 256u32;
        let grid_size = ((mid as u32) + block_size - 1) / block_size;
        let n_blocks = grid_size as usize;

        let mut d_block_s0 = unsafe { self.device.alloc::<u32>(n_blocks * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_block_s1 = unsafe { self.device.alloc::<u32>(n_blocks * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_block_s2 = unsafe { self.device.alloc::<u32>(n_blocks * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let mut d_block_s3 = unsafe { self.device.alloc::<u32>(n_blocks * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 4 * 256 * 4 * 4, // 16 KB
        };

        unsafe {
            fns.round_3way_fn.clone().launch(
                cfg,
                (d_eq, d_w, d_d,
                 &mut d_block_s0, &mut d_block_s1,
                 &mut d_block_s2, &mut d_block_s3,
                 mid as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("logup_3way_round: {:?}", e)))?;
        }

        if n_blocks == 1 {
            self.device.synchronize()
                .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;
            let mut s0 = [0u32; 4];
            let mut s1 = [0u32; 4];
            let mut s2 = [0u32; 4];
            let mut s3 = [0u32; 4];
            self.device.dtoh_sync_copy_into(&d_block_s0, &mut s0)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_block_s1, &mut s1)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_block_s2, &mut s2)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            self.device.dtoh_sync_copy_into(&d_block_s3, &mut s3)
                .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
            return Ok((s0, s1, s2, s3));
        }

        // Multi-block: second reduction pass with 4 channels
        let total_partials = 4 * n_blocks * 4;
        let mut d_partials = unsafe { self.device.alloc::<u32>(total_partials) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let chunk = n_blocks * 4;
        self.device.dtod_copy(&d_block_s0, &mut d_partials.slice_mut(0..chunk))
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s0: {:?}", e)))?;
        self.device.dtod_copy(&d_block_s1, &mut d_partials.slice_mut(chunk..2*chunk))
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s1: {:?}", e)))?;
        self.device.dtod_copy(&d_block_s2, &mut d_partials.slice_mut(2*chunk..3*chunk))
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s2: {:?}", e)))?;
        self.device.dtod_copy(&d_block_s3, &mut d_partials.slice_mut(3*chunk..4*chunk))
            .map_err(|e| CudaFftError::MemoryTransfer(format!("dtod s3: {:?}", e)))?;

        let mut d_reduced = unsafe { self.device.alloc::<u32>(16) } // 4 QM31 = 16 u32
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        unsafe {
            fns.reduce_4way_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (4, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 256 * 4 * 4,
                },
                (&d_partials, &mut d_reduced, n_blocks as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("logup_4way_reduce: {:?}", e)))?;
        }

        self.device.synchronize()
            .map_err(|e| CudaFftError::KernelExecution(format!("sync: {:?}", e)))?;

        let mut output = [0u32; 16];
        self.device.dtoh_sync_copy_into(&d_reduced, &mut output)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok((
            [output[0], output[1], output[2], output[3]],
            [output[4], output[5], output[6], output[7]],
            [output[8], output[9], output[10], output[11]],
            [output[12], output[13], output[14], output[15]],
        ))
    }

    /// Fold 3 MLEs (eq, w, d) simultaneously with the same challenge.
    ///
    /// Returns 3 new device buffers, each half the size of the input.
    pub fn logup_3way_fold(
        &self,
        d_eq: &CudaSlice<u32>,
        d_w: &CudaSlice<u32>,
        d_d: &CudaSlice<u32>,
        n_points: usize,
        challenge: &[u32; 4],
    ) -> Result<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>), CudaFftError> {
        let fns = self.get_logup_fns()?;
        let half_n = n_points / 2;

        let d_alpha = self.device.htod_sync_copy(challenge)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("3way_fold alpha: {:?}", e)))?;

        let mut d_eq_out = unsafe { self.device.alloc::<u32>(half_n * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("3way_fold eq_out: {:?}", e)))?;
        let mut d_w_out = unsafe { self.device.alloc::<u32>(half_n * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("3way_fold w_out: {:?}", e)))?;
        let mut d_d_out = unsafe { self.device.alloc::<u32>(half_n * 4) }
            .map_err(|e| CudaFftError::MemoryAllocation(format!("3way_fold d_out: {:?}", e)))?;

        let block_size = 256u32;
        let grid_size = ((half_n as u32) + block_size - 1) / block_size;

        unsafe {
            fns.fold_3way_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (d_eq, d_w, d_d, &d_alpha,
                 &mut d_eq_out, &mut d_w_out, &mut d_d_out,
                 half_n as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("logup_3way_fold: {:?}", e)))?;
        }

        Ok((d_eq_out, d_w_out, d_d_out))
    }

    /// Run a complete degree-3 3-factor sumcheck on GPU.
    ///
    /// Given three QM31 MLEs of equal length (power of 2), prove:
    ///   claim = Σ_i mle_w[i] · mle_a[i] · mle_b[i]
    ///
    /// Each round:
    ///   1. GPU kernel evaluates round poly at t=0,1,2,3 (sum reduction)
    ///   2. Newton divided differences → degree-3 coefficients (CPU, 4 field ops)
    ///   3. Fiat-Shamir: mix coefficients, draw challenge
    ///   4. GPU kernel folds all three MLEs at the challenge
    ///
    /// Returns (round_polys, challenges, final_w, final_a, final_b).
    pub fn sumcheck_3way(
        &self,
        d_w: CudaSlice<u32>,
        d_a: CudaSlice<u32>,
        d_b: CudaSlice<u32>,
        total_vars: usize,
        channel: &mut crate::crypto::poseidon_channel::PoseidonChannel,
    ) -> Result<Sumcheck3WayResult, CudaFftError> {
        use stwo::core::fields::FieldExpOps;

        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let mut cur_w = d_w;
        let mut cur_a = d_a;
        let mut cur_b = d_b;
        let mut cur_n = 1usize << total_vars;

        let mut round_polys = Vec::with_capacity(total_vars);
        let mut challenges = Vec::with_capacity(total_vars);

        for _ in 0..total_vars {
            let mid = cur_n / 2;

            // GPU: evaluate Σ w(t)·a(t)·b(t) at t=0,1,2,3
            let (s0_raw, s1_raw, s2_raw, s3_raw) =
                self.compute_logup_round_poly_3way(&cur_w, &cur_a, &cur_b, mid)?;

            let s0 = u32s_to_secure_field(&s0_raw);
            let s1 = u32s_to_secure_field(&s1_raw);
            let s2 = u32s_to_secure_field(&s2_raw);
            let s3 = u32s_to_secure_field(&s3_raw);

            // Newton divided differences → monomial coefficients
            let d1 = s1 - s0;
            let d2 = (s2 - s1 - s1 + s0) * inv2;
            let d3 = (s3 - s0 - three * (s2 - s1)) * inv6;

            let c0 = s0;
            let c1 = d1 - d2 + two * d3;
            let c2 = d2 - three * d3;
            let c3 = d3;

            round_polys.push(crate::gkr::types::RoundPolyDeg3 { c0, c1, c2, c3 });

            // Fiat-Shamir
            channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
            let challenge = channel.draw_qm31();
            challenges.push(challenge);

            // GPU: fold all three MLEs at the challenge
            let challenge_u32 = secure_field_to_u32s(challenge);
            let (new_w, new_a, new_b) =
                self.logup_3way_fold(&cur_w, &cur_a, &cur_b, cur_n, &challenge_u32)?;

            cur_w = new_w;
            cur_a = new_a;
            cur_b = new_b;
            cur_n = mid;
        }

        // Download final scalars (1 QM31 = 4 u32 each)
        let mut final_w_u32 = [0u32; 4];
        let mut final_a_u32 = [0u32; 4];
        let mut final_b_u32 = [0u32; 4];
        self.device.dtoh_sync_copy_into(&cur_w, &mut final_w_u32)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("final_w: {:?}", e)))?;
        self.device.dtoh_sync_copy_into(&cur_a, &mut final_a_u32)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("final_a: {:?}", e)))?;
        self.device.dtoh_sync_copy_into(&cur_b, &mut final_b_u32)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("final_b: {:?}", e)))?;

        Ok(Sumcheck3WayResult {
            round_polys,
            challenges,
            final_w: u32s_to_secure_field(&final_w_u32),
            final_a: u32s_to_secure_field(&final_a_u32),
            final_b: u32s_to_secure_field(&final_b_u32),
        })
    }
}

/// Result of a GPU-accelerated 3-factor sumcheck.
#[cfg(feature = "cuda-runtime")]
pub struct Sumcheck3WayResult {
    pub round_polys: Vec<crate::gkr::types::RoundPolyDeg3>,
    pub challenges: Vec<SecureField>,
    pub final_w: SecureField,
    pub final_a: SecureField,
    pub final_b: SecureField,
}

// =============================================================================
// GPU Forward Pass Operations
// =============================================================================

/// GPU-accelerated M31 matrix multiply (multi-row GEMM).
///
/// Uses GEMV kernel for m=1 (single-row), GEMM kernel for m>1.
/// Falls back to CPU only when CUDA is unavailable.
#[cfg(feature = "cuda-runtime")]
pub fn gpu_matmul_m31_full(
    input: &M31Matrix,
    weight: &M31Matrix,
) -> Result<M31Matrix, MatMulError> {
    let m = input.rows;
    let k = input.cols;
    let n = weight.cols;

    if k != weight.rows {
        return Err(MatMulError::SumcheckFailed(format!(
            "GPU GEMM dimension mismatch: input cols={k} != weight rows={}",
            weight.rows,
        )));
    }

    let executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

    let fns = executor.get_forward_fns()
        .map_err(|e| MatMulError::SumcheckFailed(format!("forward kernels: {e}")))?;

    // Upload matrices
    let input_u32: Vec<u32> = input.data.iter().map(|v| v.0).collect();
    let weight_u32: Vec<u32> = weight.data.iter().map(|v| v.0).collect();

    let d_input = executor.device.htod_sync_copy(&input_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload input: {:?}", e)))?;
    let d_weight = executor.device.htod_sync_copy(&weight_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload weight: {:?}", e)))?;
    let d_output: CudaSlice<u32> = executor.device.alloc_zeros(m * n)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU alloc output: {:?}", e)))?;

    if m == 1 {
        // GEMV path
        let block_size = 256u32;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        unsafe {
            fns.gemv_fn.clone().launch(
                LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 },
                (&d_input, &d_weight, &d_output, k as u32, n as u32),
            ).map_err(|e| MatMulError::SumcheckFailed(format!("GPU GEMV: {:?}", e)))?;
        }
    } else {
        // GEMM path: 2D grid with 16×16 blocks
        let block_x = 16u32;
        let block_y = 16u32;
        let grid_x = (n as u32 + block_x - 1) / block_x;
        let grid_y = (m as u32 + block_y - 1) / block_y;
        unsafe {
            fns.gemm_fn.clone().launch(
                LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (block_x, block_y, 1), shared_mem_bytes: 0 },
                (&d_input, &d_weight, &d_output, m as u32, k as u32, n as u32),
            ).map_err(|e| MatMulError::SumcheckFailed(format!("GPU GEMM: {:?}", e)))?;
        }
    }

    // Download result
    let mut output_u32 = vec![0u32; m * n];
    executor.device.dtoh_sync_copy_into(&d_output, &mut output_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download: {:?}", e)))?;

    let data: Vec<M31> = output_u32.iter().map(|&v| M31(v)).collect();
    Ok(M31Matrix { rows: m, cols: n, data })
}

/// GPU element-wise M31 addition.
#[cfg(feature = "cuda-runtime")]
pub fn gpu_elementwise_add(lhs: &[M31], rhs: &[M31]) -> Result<Vec<M31>, MatMulError> {
    let n = lhs.len();
    if n != rhs.len() {
        return Err(MatMulError::SumcheckFailed(format!(
            "GPU add length mismatch: {} != {}", n, rhs.len(),
        )));
    }

    let executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;
    let fns = executor.get_forward_fns()
        .map_err(|e| MatMulError::SumcheckFailed(format!("forward kernels: {e}")))?;

    let lhs_u32: Vec<u32> = lhs.iter().map(|v| v.0).collect();
    let rhs_u32: Vec<u32> = rhs.iter().map(|v| v.0).collect();

    let d_lhs = executor.device.htod_sync_copy(&lhs_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload lhs: {:?}", e)))?;
    let d_rhs = executor.device.htod_sync_copy(&rhs_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload rhs: {:?}", e)))?;
    let d_out: CudaSlice<u32> = executor.device.alloc_zeros(n)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU alloc: {:?}", e)))?;

    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;
    unsafe {
        fns.add_fn.clone().launch(
            LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 },
            (&d_lhs, &d_rhs, &d_out, n as u32),
        ).map_err(|e| MatMulError::SumcheckFailed(format!("GPU add: {:?}", e)))?;
    }

    let mut out_u32 = vec![0u32; n];
    executor.device.dtoh_sync_copy_into(&d_out, &mut out_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download: {:?}", e)))?;
    Ok(out_u32.iter().map(|&v| M31(v)).collect())
}

/// GPU element-wise M31 multiplication.
#[cfg(feature = "cuda-runtime")]
pub fn gpu_elementwise_mul(lhs: &[M31], rhs: &[M31]) -> Result<Vec<M31>, MatMulError> {
    let n = lhs.len();
    if n != rhs.len() {
        return Err(MatMulError::SumcheckFailed(format!(
            "GPU mul length mismatch: {} != {}", n, rhs.len(),
        )));
    }

    let executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;
    let fns = executor.get_forward_fns()
        .map_err(|e| MatMulError::SumcheckFailed(format!("forward kernels: {e}")))?;

    let lhs_u32: Vec<u32> = lhs.iter().map(|v| v.0).collect();
    let rhs_u32: Vec<u32> = rhs.iter().map(|v| v.0).collect();

    let d_lhs = executor.device.htod_sync_copy(&lhs_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload lhs: {:?}", e)))?;
    let d_rhs = executor.device.htod_sync_copy(&rhs_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload rhs: {:?}", e)))?;
    let d_out: CudaSlice<u32> = executor.device.alloc_zeros(n)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU alloc: {:?}", e)))?;

    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;
    unsafe {
        fns.mul_fn.clone().launch(
            LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 },
            (&d_lhs, &d_rhs, &d_out, n as u32),
        ).map_err(|e| MatMulError::SumcheckFailed(format!("GPU mul: {:?}", e)))?;
    }

    let mut out_u32 = vec![0u32; n];
    executor.device.dtoh_sync_copy_into(&d_out, &mut out_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download: {:?}", e)))?;
    Ok(out_u32.iter().map(|&v| M31(v)).collect())
}

/// GPU ReLU activation.
#[cfg(feature = "cuda-runtime")]
pub fn gpu_relu(input: &[M31]) -> Result<Vec<M31>, MatMulError> {
    let n = input.len();

    let executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;
    let fns = executor.get_forward_fns()
        .map_err(|e| MatMulError::SumcheckFailed(format!("forward kernels: {e}")))?;

    let input_u32: Vec<u32> = input.iter().map(|v| v.0).collect();
    let d_input = executor.device.htod_sync_copy(&input_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU upload: {:?}", e)))?;
    let d_out: CudaSlice<u32> = executor.device.alloc_zeros(n)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU alloc: {:?}", e)))?;

    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;
    unsafe {
        fns.relu_fn.clone().launch(
            LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 },
            (&d_input, &d_out, n as u32),
        ).map_err(|e| MatMulError::SumcheckFailed(format!("GPU relu: {:?}", e)))?;
    }

    let mut out_u32 = vec![0u32; n];
    executor.device.dtoh_sync_copy_into(&d_out, &mut out_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU download: {:?}", e)))?;
    Ok(out_u32.iter().map(|&v| M31(v)).collect())
}

// =============================================================================
// QM31 Conversion Helpers
// =============================================================================

#[cfg(feature = "cuda-runtime")]
#[inline]
pub(crate) fn secure_field_to_u32s(val: SecureField) -> [u32; 4] {
    [val.0 .0 .0, val.0 .1 .0, val.1 .0 .0, val.1 .1 .0]
}

#[cfg(feature = "cuda-runtime")]
#[inline]
pub(crate) fn u32s_to_secure_field(data: &[u32; 4]) -> SecureField {
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
/// GPU fused: restriction + sumcheck all on GPU. No intermediate MLE allocation.
///
/// Pipeline:
/// 1. Fiat-Shamir channel draws r_i, r_j (CPU, O(1))
/// 2. GPU fused restrict: upload M31 matrices + Lagrange basis, get QM31 on GPU
/// 3. GPU sumcheck inner loop (already there)
/// 4. Download final 2 QM31 evaluations (32 bytes)
///
/// For Qwen3-14B: eliminates 2 GB MLE allocation + ~150ms CPU restrict per matmul.
#[cfg(feature = "cuda-runtime")]
pub fn prove_matmul_sumcheck_gpu(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<MatMulSumcheckProof, MatMulError> {

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

    // Padded dimensions for Fiat-Shamir consistency
    let m = a.rows.next_power_of_two();
    let k = a.cols.next_power_of_two();
    let n = b.cols.next_power_of_two();
    let log_m = m.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Fiat-Shamir channel (must match CPU path exactly)
    let mut channel = Blake2sChannel::default();
    channel.mix_felts(&[
        SecureField::from(M31::from(m as u32)),
        SecureField::from(M31::from(k as u32)),
        SecureField::from(M31::from(n as u32)),
    ]);

    let r_i = channel.draw_secure_felts(log_m);
    let r_j = channel.draw_secure_felts(log_n);

    // Compute claimed sum: still need to evaluate MLE_C(r_i, r_j)
    // C is small (m×n output), so CPU pad+evaluate is fine.
    let c_padded = crate::components::matmul::pad_matrix_pow2(c);
    let mle_c = matrix_to_mle_pub(&c_padded);
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle_pub(&mle_c, &r_ij);

    channel.mix_felts(&[claimed_sum]);

    // Get cached GPU executor
    let gpu_executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

    // GPU fused restrict: upload original M31 matrices, compute QM31 restrict on GPU.
    // No matrix_to_mle, no restrict_mle, no pad_matrix_pow2 for A or B.
    let d_f_a = gpu_executor.restrict_rows(a, &r_i, k)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU restrict_rows: {e}")))?;
    let d_f_b = gpu_executor.restrict_cols(b, &r_j, k)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU restrict_cols: {e}")))?;

    // Build GPU oracle and run sumcheck (data already on GPU)
    let oracle = GpuMatMulOracle {
        d_f_a,
        d_f_b,
        n_points: k,
        executor: gpu_executor,
    };

    let lambda = SecureField::one();
    let (sumcheck_proof, assignment, final_oracles, _claimed_evals) =
        sumcheck::prove_batch(vec![claimed_sum], vec![oracle], lambda, &mut channel);

    // Download final single-point evaluations from GPU (32 bytes total)
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
    use crate::crypto::mle_opening::{commit_mle_root_only, prove_mle_opening};

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

    // Padded dimensions for Fiat-Shamir
    let m = a.rows.next_power_of_two();
    let k = a.cols.next_power_of_two();
    let n = b.cols.next_power_of_two();
    let log_m = m.ilog2() as usize;
    let log_k = k.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // PoseidonChannel for Fiat-Shamir
    let mut channel = PoseidonChannel::new();
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);

    let r_i = channel.draw_qm31s(log_m);
    let r_j = channel.draw_qm31s(log_n);

    // Compute claimed sum: MLE_C(r_i, r_j). C is small, CPU pad+eval is fine.
    let c_padded = pad_matrix_pow2(c);
    let mle_c = matrix_to_mle_pub(&c_padded);
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle_pub(&mle_c, &r_ij);

    channel.mix_felt(securefield_to_felt(claimed_sum));

    // Get cached GPU executor
    let gpu_executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

    // GPU fused restrict: original M31 matrices → QM31 on GPU
    let d_f_a_restrict = gpu_executor.restrict_rows(a, &r_i, k)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU restrict_rows: {e}")))?;
    let d_f_b_restrict = gpu_executor.restrict_cols(b, &r_j, k)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU restrict_cols: {e}")))?;

    // Download restrict results for commitment + MLE opening proofs
    let mut f_a_u32 = vec![0u32; k * 4];
    let mut f_b_u32 = vec![0u32; k * 4];
    gpu_executor.device.dtoh_sync_copy_into(&d_f_a_restrict, &mut f_a_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("download f_a: {:?}", e)))?;
    gpu_executor.device.dtoh_sync_copy_into(&d_f_b_restrict, &mut f_b_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("download f_b: {:?}", e)))?;

    let f_a: Vec<SecureField> = f_a_u32.chunks_exact(4)
        .map(|c| u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
        .collect();
    let f_b: Vec<SecureField> = f_b_u32.chunks_exact(4)
        .map(|c| u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
        .collect();

    // Commit to restricted MLEs
    let a_commitment = commit_mle_root_only(&f_a);
    let b_commitment = commit_mle_root_only(&f_b);

    channel.mix_felt(a_commitment);
    channel.mix_felt(b_commitment);

    // GPU data already on device — use directly for sumcheck
    let mut d_f_a = d_f_a_restrict;
    let mut d_f_b = d_f_b_restrict;

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
///
/// Optimized to work on ORIGINAL (unpadded) matrices:
/// - Avoids copying m×k, k×n, m×n matrices to padded pow2 dimensions
/// - Computes restrict on original dims, zero-pads only the k-element output
/// - Uses root-only parallel Merkle commit (discards tree)
///
/// For Qwen3-14B (5120→8192 padding): avoids 3× matrix copies saving ~1.5 GB/entry.
#[cfg(feature = "cuda-runtime")]
pub fn prepare_batch_entry(
    node_id: usize,
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<BatchEntry, MatMulError> {
    use crate::components::matmul::pad_matrix_pow2;
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::crypto::mle_opening::commit_mle_root_only;

    // Padded dimensions for channel draws and output sizing
    let m = a.rows.next_power_of_two();
    let k = a.cols.next_power_of_two();
    let n = b.cols.next_power_of_two();
    let log_m = m.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Per-matmul Poseidon channel for r_i, r_j
    let mut channel = PoseidonChannel::new();
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);

    let r_i = channel.draw_qm31s(log_m);
    let r_j = channel.draw_qm31s(log_n);

    // Claimed sum: evaluate C's MLE at (r_i, r_j).
    // C is small (m×n), so pad + full MLE evaluate is fine.
    let c_padded = pad_matrix_pow2(c);
    let mle_c = matrix_to_mle_pub(&c_padded);
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle_pub(&mle_c, &r_ij);

    // GPU fused restrict: upload original M31 matrices, compute QM31 on GPU, download.
    // Avoids pad_matrix_pow2 + matrix_to_mle + restrict_mle CPU pipeline.
    let gpu_executor = GpuSumcheckExecutor::cached()
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU init: {e}")))?;

    let d_f_a = gpu_executor.restrict_rows(a, &r_i, k)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU restrict_rows batch: {e}")))?;
    let d_f_b = gpu_executor.restrict_cols(b, &r_j, k)
        .map_err(|e| MatMulError::SumcheckFailed(format!("GPU restrict_cols batch: {e}")))?;

    // Download restricted MLEs for commitment and later batch upload
    let mut f_a_u32 = vec![0u32; k * 4];
    let mut f_b_u32 = vec![0u32; k * 4];
    gpu_executor.device.dtoh_sync_copy_into(&d_f_a, &mut f_a_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("download batch f_a: {:?}", e)))?;
    gpu_executor.device.dtoh_sync_copy_into(&d_f_b, &mut f_b_u32)
        .map_err(|e| MatMulError::SumcheckFailed(format!("download batch f_b: {:?}", e)))?;

    let f_a: Vec<SecureField> = f_a_u32.chunks_exact(4)
        .map(|c| u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
        .collect();
    let f_b: Vec<SecureField> = f_b_u32.chunks_exact(4)
        .map(|c| u32s_to_secure_field(&[c[0], c[1], c[2], c[3]]))
        .collect();

    // Root-only parallel commit: avoids building full tree (discarded anyway).
    let a_commitment = commit_mle_root_only(&f_a);
    let b_commitment = commit_mle_root_only(&f_b);

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
