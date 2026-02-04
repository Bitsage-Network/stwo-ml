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

/// Combined kernel source with all M31 operations and constraint evaluation.
pub fn get_full_kernel_source() -> String {
    format!("{}\n{}\n{}", M31_FIELD_KERNEL, CONSTRAINT_EVAL_KERNEL, QUOTIENT_KERNEL)
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
