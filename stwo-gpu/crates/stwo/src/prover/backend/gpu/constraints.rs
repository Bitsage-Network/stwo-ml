//! GPU-accelerated constraint evaluation kernels.
//!
//! This module provides CUDA kernels for evaluating AIR constraints directly on GPU,
//! bypassing the SIMD vectorization path for significantly improved performance.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐
//! │  Trace Data     │────▶│  GPU Memory     │
//! │  (Host)         │     │  (Device)       │
//! └─────────────────┘     └────────┬────────┘
//!                                  │
//!                         ┌────────▼────────┐
//!                         │  Constraint     │
//!                         │  Eval Kernel    │
//!                         └────────┬────────┘
//!                                  │
//!                         ┌────────▼────────┐
//!                         │  Accumulator    │
//!                         │  (Device)       │
//!                         └─────────────────┘
//! ```
//!
//! # M31 Field Arithmetic
//!
//! All operations are performed in the Mersenne-31 field (p = 2^31 - 1).
//! The CUDA kernels implement efficient modular arithmetic using:
//! - Lazy reduction for additions
//! - Montgomery multiplication for products
//! - Binary exponentiation for powers

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

use super::cuda_executor::CudaFftError;

/// CUDA kernel source for M31 field arithmetic operations.
///
/// The Mersenne-31 prime p = 2^31 - 1 = 0x7FFFFFFF allows for efficient
/// modular reduction using bit operations.
pub const M31_FIELD_KERNEL: &str = r#"
// Mersenne-31 prime constant
#define M31_P 0x7FFFFFFFu
#define M31_BITS 31

// M31 modular addition: (a + b) mod p
__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    // If r >= p, subtract p. Since p = 2^31 - 1, we check bit 31
    // and use the fact that r - p = r - 2^31 + 1 = (r & M31_P) + 1 when r >= 2^31
    uint32_t reduced = (r & M31_P) + (r >> 31);
    // Handle the case where reduced == p
    return reduced == M31_P ? 0 : reduced;
}

// M31 modular subtraction: (a - b) mod p
__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    uint32_t r = a - b;
    // If a < b, we wrapped around and need to add p back
    // Using conditional to avoid branch: add p if borrow occurred
    return r + (M31_P & -(r >> 31));
}

// M31 modular negation: (-a) mod p
__device__ __forceinline__ uint32_t m31_neg(uint32_t a) {
    // -a mod p = p - a for a != 0, 0 for a == 0
    return (M31_P - a) * (a != 0);
}

// M31 modular multiplication: (a * b) mod p
// Uses the identity: (a * b) mod (2^31 - 1) = lo + hi where
// a * b = hi * 2^31 + lo, and we use that 2^31 ≡ 1 (mod p)
__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    return m31_add(lo, hi);
}

// M31 modular squaring: a^2 mod p (slightly optimized)
__device__ __forceinline__ uint32_t m31_sqr(uint32_t a) {
    uint64_t prod = (uint64_t)a * (uint64_t)a;
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    return m31_add(lo, hi);
}

// M31 modular exponentiation: a^exp mod p using binary method
__device__ uint32_t m31_pow(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    uint32_t b = base;

    while (exp > 0) {
        if (exp & 1) {
            result = m31_mul(result, b);
        }
        b = m31_sqr(b);
        exp >>= 1;
    }

    return result;
}

// M31 modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
__device__ uint32_t m31_inv(uint32_t a) {
    // p - 2 = 2^31 - 3 = 0x7FFFFFFD
    return m31_pow(a, 0x7FFFFFFDu);
}

// M31 modular division: a / b mod p
__device__ __forceinline__ uint32_t m31_div(uint32_t a, uint32_t b) {
    return m31_mul(a, m31_inv(b));
}
"#;

/// CUDA kernel source for constraint evaluation.
///
/// This kernel evaluates a linear combination of constraint polynomials
/// at each point in the domain.
pub const CONSTRAINT_EVAL_KERNEL: &str = r#"
// Include M31 field operations (will be prepended)

// Evaluate a generic constraint and accumulate with random coefficient
// This is a template that can be specialized per AIR
__global__ void eval_constraints_generic(
    const uint32_t* __restrict__ trace_data,      // Flattened trace columns
    uint32_t* __restrict__ constraint_out,         // Output accumulator
    const uint32_t* __restrict__ random_coeffs,    // Random linear combination coefficients
    const uint32_t* __restrict__ column_offsets,   // Start offset for each column
    uint32_t domain_size,                          // Number of points to evaluate
    uint32_t num_columns,                          // Number of trace columns
    uint32_t num_constraints                       // Number of constraints to evaluate
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    // Initialize accumulator for this point
    uint32_t accumulator = 0;

    // Load trace values for this row into shared memory for faster access
    extern __shared__ uint32_t shared_trace[];

    // Each thread loads its trace values
    for (uint32_t col = 0; col < num_columns && col < 32; col++) {
        uint32_t offset = column_offsets[col];
        shared_trace[threadIdx.x * 32 + col] = trace_data[offset + idx];
    }
    __syncthreads();

    // Evaluate each constraint and accumulate
    // Note: This is a generic evaluator. For production AIRs, we'd generate
    // specialized kernels at compile time for each constraint type.
    for (uint32_t c = 0; c < num_constraints; c++) {
        uint32_t coeff = random_coeffs[c];

        // Placeholder constraint evaluation
        // In practice, this would be replaced with actual constraint logic
        // For example: trace[col0][idx] * trace[col1][idx] - trace[col2][idx]
        uint32_t constraint_val = shared_trace[threadIdx.x * 32];  // Simplified

        // Accumulate: acc += coeff * constraint_val
        accumulator = m31_add(accumulator, m31_mul(coeff, constraint_val));
    }

    constraint_out[idx] = accumulator;
}

// Optimized kernel for degree-2 constraints (most common in STARKs)
// a * b - c = 0 style constraints
__global__ void eval_degree2_constraints(
    const uint32_t* __restrict__ col_a,
    const uint32_t* __restrict__ col_b,
    const uint32_t* __restrict__ col_c,
    uint32_t* __restrict__ output,
    const uint32_t random_coeff,
    uint32_t domain_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    uint32_t a = col_a[idx];
    uint32_t b = col_b[idx];
    uint32_t c = col_c[idx];

    // Constraint: a * b - c
    uint32_t constraint_val = m31_sub(m31_mul(a, b), c);

    // Apply random coefficient
    output[idx] = m31_mul(random_coeff, constraint_val);
}

// Kernel for transition constraints: f(x_next) - g(x) = 0
__global__ void eval_transition_constraints(
    const uint32_t* __restrict__ trace_curr,
    const uint32_t* __restrict__ trace_next,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ coeffs,
    uint32_t domain_size,
    uint32_t num_transitions
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    uint32_t accumulator = 0;

    for (uint32_t t = 0; t < num_transitions; t++) {
        uint32_t curr = trace_curr[t * domain_size + idx];
        uint32_t next = trace_next[t * domain_size + idx];

        // Simple transition: next - curr (can be extended for complex transitions)
        uint32_t constraint_val = m31_sub(next, curr);
        accumulator = m31_add(accumulator, m31_mul(coeffs[t], constraint_val));
    }

    output[idx] = accumulator;
}

// Kernel for boundary constraints at specific indices
__global__ void eval_boundary_constraints(
    const uint32_t* __restrict__ trace,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ boundary_indices,
    const uint32_t* __restrict__ boundary_values,
    const uint32_t* __restrict__ coeffs,
    uint32_t num_boundaries,
    uint32_t domain_size
) {
    uint32_t b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_idx >= num_boundaries) return;

    uint32_t trace_idx = boundary_indices[b_idx];
    uint32_t expected = boundary_values[b_idx];
    uint32_t actual = trace[trace_idx];

    // Constraint: actual - expected = 0
    uint32_t constraint_val = m31_sub(actual, expected);
    output[b_idx] = m31_mul(coeffs[b_idx], constraint_val);
}

// Accumulate multiple constraint evaluations into a single polynomial
__global__ void accumulate_constraints(
    const uint32_t* const* __restrict__ constraint_evals,  // Array of pointers
    uint32_t* __restrict__ accumulator,
    uint32_t num_constraint_types,
    uint32_t domain_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    uint32_t sum = 0;
    for (uint32_t c = 0; c < num_constraint_types; c++) {
        sum = m31_add(sum, constraint_evals[c][idx]);
    }

    accumulator[idx] = sum;
}
"#;

/// CUDA kernel for quotient polynomial computation.
///
/// Computes the quotient q(x) = C(x) / Z(x) where:
/// - C(x) is the accumulated constraint polynomial
/// - Z(x) is the vanishing polynomial (zerofier)
pub const QUOTIENT_KERNEL: &str = r#"
// Compute quotient: constraint_eval / zerofier for each domain point
__global__ void compute_quotient(
    const uint32_t* __restrict__ constraint_eval,
    const uint32_t* __restrict__ zerofier_eval,
    uint32_t* __restrict__ quotient_out,
    uint32_t domain_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    uint32_t c = constraint_eval[idx];
    uint32_t z = zerofier_eval[idx];

    // quotient = c / z = c * z^(-1)
    quotient_out[idx] = m31_div(c, z);
}

// Batch quotient computation for multiple constraint columns
__global__ void compute_quotient_batch(
    const uint32_t* __restrict__ constraint_evals,  // num_constraints * domain_size
    const uint32_t* __restrict__ zerofier_eval,     // domain_size
    uint32_t* __restrict__ quotient_out,            // num_constraints * domain_size
    uint32_t domain_size,
    uint32_t num_constraints
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    uint32_t z = zerofier_eval[idx];
    uint32_t z_inv = m31_inv(z);

    // Compute all quotients sharing the same zerofier inverse
    for (uint32_t c = 0; c < num_constraints; c++) {
        uint32_t offset = c * domain_size + idx;
        quotient_out[offset] = m31_mul(constraint_evals[offset], z_inv);
    }
}
"#;

/// CUDA kernel source for CM31 (complex M31) and QM31 (degree-4 extension) arithmetic,
/// plus the fused PCS quotient combination kernel.
///
/// CM31 = M31[i] / (i² + 1), stored as (real, imag) pairs of M31.
/// QM31 = CM31[j] / (j² - 2 - i), stored as (a, b, c, d) where val = (a+bi) + (c+di)j.
///
/// The PCS quotient kernel fuses denominator inverse + numerator combination into a single
/// kernel launch, eliminating intermediate memory traffic.
pub const PCS_QUOTIENT_KERNEL: &str = r#"
// ============================================================
// CM31 Arithmetic (Complex M31)
// CM31 = (real, imag) where i² = -1 (mod p)
// ============================================================

// CM31 addition
__device__ __forceinline__ void cm31_add(
    uint32_t ar, uint32_t ai,
    uint32_t br, uint32_t bi,
    uint32_t* out_r, uint32_t* out_i
) {
    *out_r = m31_add(ar, br);
    *out_i = m31_add(ai, bi);
}

// CM31 subtraction
__device__ __forceinline__ void cm31_sub(
    uint32_t ar, uint32_t ai,
    uint32_t br, uint32_t bi,
    uint32_t* out_r, uint32_t* out_i
) {
    *out_r = m31_sub(ar, br);
    *out_i = m31_sub(ai, bi);
}

// CM31 multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
__device__ __forceinline__ void cm31_mul(
    uint32_t ar, uint32_t ai,
    uint32_t br, uint32_t bi,
    uint32_t* out_r, uint32_t* out_i
) {
    uint32_t ac = m31_mul(ar, br);
    uint32_t bd = m31_mul(ai, bi);
    uint32_t ad = m31_mul(ar, bi);
    uint32_t bc = m31_mul(ai, br);
    *out_r = m31_sub(ac, bd);
    *out_i = m31_add(ad, bc);
}

// CM31 inverse: (a+bi)^(-1) = (a-bi) / (a² + b²)
// norm² = a² + b² is in M31, so we only need M31 inverse.
__device__ __forceinline__ void cm31_inv(
    uint32_t ar, uint32_t ai,
    uint32_t* out_r, uint32_t* out_i
) {
    uint32_t norm_sq = m31_add(m31_mul(ar, ar), m31_mul(ai, ai));
    uint32_t norm_inv = m31_inv(norm_sq);
    *out_r = m31_mul(ar, norm_inv);
    *out_i = m31_mul(m31_neg(ai), norm_inv);
}

// ============================================================
// QM31 Arithmetic (Degree-4 Extension)
// QM31 = (a+bi) + (c+di)j where j² = 2+i
// Stored as 4 M31 values: (a, b, c, d)
// ============================================================

// QM31 addition
__device__ __forceinline__ void qm31_add(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t* o0, uint32_t* o1, uint32_t* o2, uint32_t* o3
) {
    *o0 = m31_add(a0, b0);
    *o1 = m31_add(a1, b1);
    *o2 = m31_add(a2, b2);
    *o3 = m31_add(a3, b3);
}

// QM31 subtraction
__device__ __forceinline__ void qm31_sub(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t* o0, uint32_t* o1, uint32_t* o2, uint32_t* o3
) {
    *o0 = m31_sub(a0, b0);
    *o1 = m31_sub(a1, b1);
    *o2 = m31_sub(a2, b2);
    *o3 = m31_sub(a3, b3);
}

// QM31 × M31 scalar multiplication
__device__ __forceinline__ void qm31_mul_m31(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t s,
    uint32_t* o0, uint32_t* o1, uint32_t* o2, uint32_t* o3
) {
    *o0 = m31_mul(a0, s);
    *o1 = m31_mul(a1, s);
    *o2 = m31_mul(a2, s);
    *o3 = m31_mul(a3, s);
}

// QM31 × CM31 multiplication
// (a+bi + (c+di)j) * (e+fi)
// = (a+bi)(e+fi) + ((c+di)(e+fi))j
__device__ __forceinline__ void qm31_mul_cm31(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t cr, uint32_t ci,
    uint32_t* o0, uint32_t* o1, uint32_t* o2, uint32_t* o3
) {
    // First CM31: (a0 + a1*i) * (cr + ci*i)
    cm31_mul(a0, a1, cr, ci, o0, o1);
    // Second CM31: (a2 + a3*i) * (cr + ci*i)
    cm31_mul(a2, a3, cr, ci, o2, o3);
}

// ============================================================
// Fused PCS Quotient Combination Kernel
// ============================================================
//
// For each domain point idx, computes:
//   quotient[idx] = Σ_s (full_numerator_s * denom_inv_s)
//
// where for each sample point s:
//   denom = (Pr_x - D_x) * Pi_y - (Pr_y - D_y) * Pi_x   (CM31)
//   denom_inv = 1 / denom                                  (CM31)
//   full_numerator = lifted_partial_num - first_linear_term * D_y  (QM31)
//   quotient += full_numerator * denom_inv
//
// Each thread handles one domain point.
// Domain points are pre-computed and passed as arrays.

__global__ void pcs_quotient_combine(
    // Domain points (M31 values, domain_size each)
    const uint32_t* __restrict__ domain_x_r,  // M31 real part of domain point x
    const uint32_t* __restrict__ domain_x_i,  // Always 0 for base field domain points
    const uint32_t* __restrict__ domain_y_r,  // M31 real part of domain point y
    const uint32_t* __restrict__ domain_y_i,  // Always 0 for base field domain points
    // Partial numerators: 4 columns per sample point, SoA layout
    // Layout: num_samples * 4 * numerator_size (each column contiguous)
    const uint32_t* __restrict__ partial_nums,
    // Per-sample data (packed): sample_point (8 M31s: x.r, x.i, x.j_r, x.j_i, y.r, y.i, y.j_r, y.j_i)
    // + first_linear_term (4 M31s) + log_ratio (1 u32) = 13 u32 per sample
    const uint32_t* __restrict__ sample_data,
    // Output: 4 columns, domain_size each
    uint32_t* __restrict__ out_c0,
    uint32_t* __restrict__ out_c1,
    uint32_t* __restrict__ out_c2,
    uint32_t* __restrict__ out_c3,
    uint32_t domain_size,
    uint32_t num_samples
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size) return;

    // Domain point (base field — imaginary parts are 0)
    uint32_t dx_r = domain_x_r[idx];
    uint32_t dy_r = domain_y_r[idx];

    // Accumulate quotient for this domain point
    uint32_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;

    for (uint32_t s = 0; s < num_samples; s++) {
        // Load sample data (13 u32 per sample)
        uint32_t base = s * 13;
        // Sample point x: QM31 = (sx_r + sx_i*i) + (sx_jr + sx_ji*i)*j
        uint32_t sx_r  = sample_data[base + 0];  // x.0.0 (real of CM31.0)
        uint32_t sx_i  = sample_data[base + 1];  // x.0.1 (imag of CM31.0)
        // Sample point y: QM31
        uint32_t sy_r  = sample_data[base + 4];  // y.0.0
        uint32_t sy_i  = sample_data[base + 5];  // y.0.1
        // First linear term: QM31
        uint32_t flt0  = sample_data[base + 8];
        uint32_t flt1  = sample_data[base + 9];
        uint32_t flt2  = sample_data[base + 10];
        uint32_t flt3  = sample_data[base + 11];
        // Log ratio for lifting
        uint32_t log_ratio = sample_data[base + 12];

        // Compute denominator: (Pr_x - D_x) * Pi_y - (Pr_y - D_y) * Pi_x
        // Where P = sample_point, Pr = real part (CM31.0), Pi = imag part (CM31.1)
        // Since domain point is base field: D_x = (dx_r, 0), D_y = (dy_r, 0)
        // Pr_x = (sx_r, sx_i), Pi_x = sample_data[2,3]
        uint32_t pix_r = sample_data[base + 2];
        uint32_t pix_i = sample_data[base + 3];
        uint32_t piy_r = sample_data[base + 6];
        uint32_t piy_i = sample_data[base + 7];

        // (Pr_x - D_x): CM31 - M31 = (sx_r - dx_r, sx_i)
        uint32_t diff_x_r = m31_sub(sx_r, dx_r);
        uint32_t diff_x_i = sx_i;

        // (Pr_y - D_y): CM31 - M31 = (sy_r - dy_r, sy_i)
        uint32_t diff_y_r = m31_sub(sy_r, dy_r);
        uint32_t diff_y_i = sy_i;

        // denom = diff_x * Pi_y - diff_y * Pi_x (CM31 arithmetic)
        uint32_t t1_r, t1_i, t2_r, t2_i;
        cm31_mul(diff_x_r, diff_x_i, piy_r, piy_i, &t1_r, &t1_i);
        cm31_mul(diff_y_r, diff_y_i, pix_r, pix_i, &t2_r, &t2_i);
        uint32_t den_r, den_i;
        cm31_sub(t1_r, t1_i, t2_r, t2_i, &den_r, &den_i);

        // Compute CM31 inverse of denominator
        uint32_t den_inv_r, den_inv_i;
        cm31_inv(den_r, den_i, &den_inv_r, &den_inv_i);

        // Load lifted partial numerator (with log_ratio lifting)
        uint32_t src_idx = idx >> log_ratio;
        uint32_t num_size = domain_size >> log_ratio;
        uint32_t n0 = partial_nums[s * 4 * num_size + 0 * num_size + src_idx];
        uint32_t n1 = partial_nums[s * 4 * num_size + 1 * num_size + src_idx];
        uint32_t n2 = partial_nums[s * 4 * num_size + 2 * num_size + src_idx];
        uint32_t n3 = partial_nums[s * 4 * num_size + 3 * num_size + src_idx];

        // full_numerator = lifted_num - first_linear_term * D_y
        // first_linear_term * D_y = QM31 * M31
        uint32_t flt_dy0, flt_dy1, flt_dy2, flt_dy3;
        qm31_mul_m31(flt0, flt1, flt2, flt3, dy_r, &flt_dy0, &flt_dy1, &flt_dy2, &flt_dy3);

        uint32_t fn0, fn1, fn2, fn3;
        qm31_sub(n0, n1, n2, n3, flt_dy0, flt_dy1, flt_dy2, flt_dy3, &fn0, &fn1, &fn2, &fn3);

        // quotient += full_numerator * denom_inv (QM31 × CM31)
        uint32_t prod0, prod1, prod2, prod3;
        qm31_mul_cm31(fn0, fn1, fn2, fn3, den_inv_r, den_inv_i, &prod0, &prod1, &prod2, &prod3);

        qm31_add(q0, q1, q2, q3, prod0, prod1, prod2, prod3, &q0, &q1, &q2, &q3);
    }

    out_c0[idx] = q0;
    out_c1[idx] = q1;
    out_c2[idx] = q2;
    out_c3[idx] = q3;
}
"#;

/// Combined kernel source with all M31 operations and constraint evaluation.
pub fn get_full_kernel_source() -> String {
    format!("{}\n{}\n{}", M31_FIELD_KERNEL, CONSTRAINT_EVAL_KERNEL, QUOTIENT_KERNEL)
}

/// Combined kernel source with M31 + CM31/QM31 + PCS quotient operations.
pub fn get_pcs_quotient_kernel_source() -> String {
    format!("{}\n{}", M31_FIELD_KERNEL, PCS_QUOTIENT_KERNEL)
}

/// Configuration for constraint kernel launches.
#[derive(Clone, Debug)]
pub struct ConstraintKernelConfig {
    /// Number of threads per block (typically 256 or 512).
    pub block_size: u32,
    /// Shared memory size per block in bytes.
    pub shared_mem_bytes: u32,
    /// Whether to use L1 cache preference.
    pub prefer_l1_cache: bool,
}

impl Default for ConstraintKernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            shared_mem_bytes: 0,
            prefer_l1_cache: true,
        }
    }
}

/// GPU-accelerated constraint evaluator.
///
/// This struct manages CUDA kernels for constraint evaluation and provides
/// a high-level interface for the constraint framework.
#[allow(dead_code)]
#[cfg(feature = "cuda-runtime")]
pub struct ConstraintKernel {
    device: Arc<CudaDevice>,
    generic_eval_fn: CudaFunction,
    degree2_eval_fn: CudaFunction,
    transition_eval_fn: CudaFunction,
    boundary_eval_fn: CudaFunction,
    quotient_fn: CudaFunction,
    quotient_batch_fn: CudaFunction,
    config: ConstraintKernelConfig,
}

#[cfg(feature = "cuda-runtime")]
impl ConstraintKernel {
    /// Compile and load constraint kernels onto the GPU.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaFftError> {
        Self::with_config(device, ConstraintKernelConfig::default())
    }

    /// Compile kernels with custom configuration.
    pub fn with_config(
        device: Arc<CudaDevice>,
        config: ConstraintKernelConfig,
    ) -> Result<Self, CudaFftError> {
        let full_source = get_full_kernel_source();

        // Compile the PTX
        let ptx = compile_constraint_ptx(&full_source)?;

        // Load module and get function handles
        device
            .load_ptx(ptx.clone(), "constraint_kernels", &[
                "eval_constraints_generic",
                "eval_degree2_constraints",
                "eval_transition_constraints",
                "eval_boundary_constraints",
                "compute_quotient",
                "compute_quotient_batch",
            ])
            .map_err(|e| CudaFftError::DriverInit(format!("Failed to load PTX: {}", e)))?;

        let generic_eval_fn = device
            .get_func("constraint_kernels", "eval_constraints_generic")
            .ok_or_else(|| CudaFftError::DriverInit("Missing eval_constraints_generic".into()))?;

        let degree2_eval_fn = device
            .get_func("constraint_kernels", "eval_degree2_constraints")
            .ok_or_else(|| CudaFftError::DriverInit("Missing eval_degree2_constraints".into()))?;

        let transition_eval_fn = device
            .get_func("constraint_kernels", "eval_transition_constraints")
            .ok_or_else(|| CudaFftError::DriverInit("Missing eval_transition_constraints".into()))?;

        let boundary_eval_fn = device
            .get_func("constraint_kernels", "eval_boundary_constraints")
            .ok_or_else(|| CudaFftError::DriverInit("Missing eval_boundary_constraints".into()))?;

        let quotient_fn = device
            .get_func("constraint_kernels", "compute_quotient")
            .ok_or_else(|| CudaFftError::DriverInit("Missing compute_quotient".into()))?;

        let quotient_batch_fn = device
            .get_func("constraint_kernels", "compute_quotient_batch")
            .ok_or_else(|| CudaFftError::DriverInit("Missing compute_quotient_batch".into()))?;

        Ok(Self {
            device,
            generic_eval_fn,
            degree2_eval_fn,
            transition_eval_fn,
            boundary_eval_fn,
            quotient_fn,
            quotient_batch_fn,
            config,
        })
    }

    /// Evaluate degree-2 constraints: a * b - c = 0.
    ///
    /// This is optimized for the most common constraint pattern in STARKs.
    pub fn eval_degree2(
        &self,
        col_a: &CudaSlice<u32>,
        col_b: &CudaSlice<u32>,
        col_c: &CudaSlice<u32>,
        output: &mut CudaSlice<u32>,
        random_coeff: u32,
        domain_size: u32,
    ) -> Result<(), CudaFftError> {
        let grid_size = (domain_size + self.config.block_size - 1) / self.config.block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.degree2_eval_fn
                .clone()
                .launch(launch_config, (col_a, col_b, col_c, output, random_coeff, domain_size))
                .map_err(|e| CudaFftError::KernelLaunch(format!("degree2 eval: {}", e)))?;
        }

        Ok(())
    }

    /// Evaluate transition constraints between consecutive trace rows.
    pub fn eval_transitions(
        &self,
        trace_curr: &CudaSlice<u32>,
        trace_next: &CudaSlice<u32>,
        output: &mut CudaSlice<u32>,
        coeffs: &CudaSlice<u32>,
        domain_size: u32,
        num_transitions: u32,
    ) -> Result<(), CudaFftError> {
        let grid_size = (domain_size + self.config.block_size - 1) / self.config.block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.transition_eval_fn
                .clone()
                .launch(
                    launch_config,
                    (trace_curr, trace_next, output, coeffs, domain_size, num_transitions),
                )
                .map_err(|e| CudaFftError::KernelLaunch(format!("transition eval: {}", e)))?;
        }

        Ok(())
    }

    /// Evaluate boundary constraints at specific trace indices.
    pub fn eval_boundaries(
        &self,
        trace: &CudaSlice<u32>,
        output: &mut CudaSlice<u32>,
        boundary_indices: &CudaSlice<u32>,
        boundary_values: &CudaSlice<u32>,
        coeffs: &CudaSlice<u32>,
        num_boundaries: u32,
        domain_size: u32,
    ) -> Result<(), CudaFftError> {
        let grid_size = (num_boundaries + self.config.block_size - 1) / self.config.block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.boundary_eval_fn
                .clone()
                .launch(
                    launch_config,
                    (
                        trace,
                        output,
                        boundary_indices,
                        boundary_values,
                        coeffs,
                        num_boundaries,
                        domain_size,
                    ),
                )
                .map_err(|e| CudaFftError::KernelLaunch(format!("boundary eval: {}", e)))?;
        }

        Ok(())
    }

    /// Compute quotient polynomial q(x) = C(x) / Z(x).
    pub fn compute_quotient(
        &self,
        constraint_eval: &CudaSlice<u32>,
        zerofier_eval: &CudaSlice<u32>,
        quotient_out: &mut CudaSlice<u32>,
        domain_size: u32,
    ) -> Result<(), CudaFftError> {
        let grid_size = (domain_size + self.config.block_size - 1) / self.config.block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.quotient_fn
                .clone()
                .launch(
                    launch_config,
                    (constraint_eval, zerofier_eval, quotient_out, domain_size),
                )
                .map_err(|e| CudaFftError::KernelLaunch(format!("quotient: {}", e)))?;
        }

        Ok(())
    }

    /// Batch quotient computation for multiple constraint columns.
    pub fn compute_quotient_batch(
        &self,
        constraint_evals: &CudaSlice<u32>,
        zerofier_eval: &CudaSlice<u32>,
        quotient_out: &mut CudaSlice<u32>,
        domain_size: u32,
        num_constraints: u32,
    ) -> Result<(), CudaFftError> {
        let grid_size = (domain_size + self.config.block_size - 1) / self.config.block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.quotient_batch_fn
                .clone()
                .launch(
                    launch_config,
                    (
                        constraint_evals,
                        zerofier_eval,
                        quotient_out,
                        domain_size,
                        num_constraints,
                    ),
                )
                .map_err(|e| CudaFftError::KernelLaunch(format!("quotient batch: {}", e)))?;
        }

        Ok(())
    }

    /// Get the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// GPU-accelerated PCS quotient combiner.
///
/// Compiles and caches the fused quotient combination CUDA kernel, which
/// computes denominator inverses and quotient accumulation in a single pass.
#[allow(dead_code)]
#[cfg(feature = "cuda-runtime")]
pub struct GpuQuotientExecutor {
    device: Arc<CudaDevice>,
    quotient_combine_fn: CudaFunction,
    config: ConstraintKernelConfig,
}

#[cfg(feature = "cuda-runtime")]
impl GpuQuotientExecutor {
    /// Compile and load the PCS quotient kernel.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CudaFftError> {
        Self::with_config(device, ConstraintKernelConfig::default())
    }

    /// Compile with custom configuration.
    pub fn with_config(
        device: Arc<CudaDevice>,
        config: ConstraintKernelConfig,
    ) -> Result<Self, CudaFftError> {
        let source = get_pcs_quotient_kernel_source();
        let ptx = compile_constraint_ptx(&source)?;

        device
            .load_ptx(ptx, "pcs_quotient_kernels", &["pcs_quotient_combine"])
            .map_err(|e| CudaFftError::DriverInit(format!("Failed to load PCS quotient PTX: {}", e)))?;

        let quotient_combine_fn = device
            .get_func("pcs_quotient_kernels", "pcs_quotient_combine")
            .ok_or_else(|| CudaFftError::DriverInit("Missing pcs_quotient_combine kernel".into()))?;

        Ok(Self {
            device,
            quotient_combine_fn,
            config,
        })
    }

    /// Run the fused PCS quotient combination on GPU.
    ///
    /// # Arguments
    /// * `domain_x_r`, `domain_y_r` - Real parts of domain points (M31, base field)
    /// * `partial_nums` - Flattened partial numerator columns (4 cols per sample, SoA)
    /// * `sample_data` - Packed per-sample parameters (13 u32 each)
    /// * `domain_size` - Number of domain points
    /// * `num_samples` - Number of sample points
    ///
    /// # Returns
    /// Four output columns (c0, c1, c2, c3) representing the QM31 quotient.
    pub fn compute_quotients(
        &self,
        domain_x_r: &CudaSlice<u32>,
        domain_x_i: &CudaSlice<u32>,
        domain_y_r: &CudaSlice<u32>,
        domain_y_i: &CudaSlice<u32>,
        partial_nums: &CudaSlice<u32>,
        sample_data: &CudaSlice<u32>,
        out_c0: &mut CudaSlice<u32>,
        out_c1: &mut CudaSlice<u32>,
        out_c2: &mut CudaSlice<u32>,
        out_c3: &mut CudaSlice<u32>,
        domain_size: u32,
        num_samples: u32,
    ) -> Result<(), CudaFftError> {
        let grid_size = (domain_size + self.config.block_size - 1) / self.config.block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.quotient_combine_fn
                .clone()
                .launch(
                    launch_config,
                    (
                        domain_x_r,
                        domain_x_i,
                        domain_y_r,
                        domain_y_i,
                        partial_nums,
                        sample_data,
                        out_c0,
                        out_c1,
                        out_c2,
                        out_c3,
                        domain_size,
                        num_samples,
                    ),
                )
                .map_err(|e| CudaFftError::KernelLaunch(format!("pcs_quotient_combine: {}", e)))?;
        }

        Ok(())
    }

    /// Get the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// Global cached PCS quotient executor (compiled once, reused across calls).
#[cfg(feature = "cuda-runtime")]
static GPU_QUOTIENT_EXECUTOR: std::sync::OnceLock<Result<GpuQuotientExecutor, CudaFftError>> =
    std::sync::OnceLock::new();

/// Get or initialize the global GPU quotient executor.
#[cfg(feature = "cuda-runtime")]
pub fn get_gpu_quotient_executor() -> Result<&'static GpuQuotientExecutor, &'static CudaFftError> {
    GPU_QUOTIENT_EXECUTOR
        .get_or_init(|| {
            let device = CudaDevice::new(0)
                .map_err(|e| CudaFftError::DriverInit(format!("Failed to init CUDA device: {}", e)))?;
            GpuQuotientExecutor::new(device)
        })
        .as_ref()
}

/// Minimum log domain size to dispatch to GPU for quotient operations.
/// Below this threshold, the SIMD backend is used (GPU transfer overhead dominates).
pub const GPU_QUOTIENT_THRESHOLD_LOG_SIZE: u32 = 14;

/// Compile CUDA source to PTX.
#[cfg(feature = "cuda-runtime")]
fn compile_constraint_ptx(source: &str) -> Result<cudarc::nvrtc::Ptx, CudaFftError> {
    use cudarc::nvrtc::compile_ptx_with_opts;

    let opts = cudarc::nvrtc::CompileOptions {
        ftz: Some(true),
        prec_div: Some(false),
        prec_sqrt: Some(false),
        fmad: Some(true),
        ..Default::default()
    };

    compile_ptx_with_opts(source, opts)
        .map_err(|e| CudaFftError::DriverInit(format!("PTX compilation failed: {}", e)))
}

/// Stub implementation when CUDA is not available.
#[cfg(not(feature = "cuda-runtime"))]
pub struct ConstraintKernel {
    _private: (),
}

#[cfg(not(feature = "cuda-runtime"))]
impl ConstraintKernel {
    pub fn new(_device: std::sync::Arc<()>) -> Result<Self, CudaFftError> {
        Err(CudaFftError::DriverInit(
            "CUDA runtime not available".into(),
        ))
    }
}

/// Statistics for constraint kernel execution.
#[derive(Clone, Debug, Default)]
pub struct ConstraintKernelStats {
    /// Total number of constraint evaluations.
    pub total_evaluations: u64,
    /// Total kernel execution time in microseconds.
    pub total_kernel_time_us: u64,
    /// Number of degree-2 constraint evaluations.
    pub degree2_evals: u64,
    /// Number of transition constraint evaluations.
    pub transition_evals: u64,
    /// Number of boundary constraint evaluations.
    pub boundary_evals: u64,
    /// Number of quotient computations.
    pub quotient_computations: u64,
}

impl ConstraintKernelStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get average kernel time per evaluation in microseconds.
    pub fn avg_kernel_time_us(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.total_kernel_time_us as f64 / self.total_evaluations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_source_generation() {
        let source = get_full_kernel_source();
        assert!(source.contains("m31_add"));
        assert!(source.contains("m31_mul"));
        assert!(source.contains("m31_pow"));
        assert!(source.contains("eval_constraints_generic"));
        assert!(source.contains("eval_degree2_constraints"));
        assert!(source.contains("compute_quotient"));
    }

    #[test]
    fn test_default_config() {
        let config = ConstraintKernelConfig::default();
        assert_eq!(config.block_size, 256);
        assert!(config.prefer_l1_cache);
    }

    #[test]
    fn test_stats() {
        let mut stats = ConstraintKernelStats::new();
        stats.total_evaluations = 1000;
        stats.total_kernel_time_us = 5000;
        assert!((stats.avg_kernel_time_us() - 5.0).abs() < 0.001);
    }
}
