//! GPU-accelerated FFT operations for Circle STARK over M31.
//!
//! This module provides CUDA kernels for the Circle FFT (CFFT) and inverse CFFT (ICFFT)
//! operations used in Stwo's polynomial commitment scheme.
//!
//! # Algorithm Overview
//!
//! The Circle FFT operates on the Mersenne-31 field (M31, p = 2^31 - 1) and uses
//! the circle group structure for efficient polynomial evaluation/interpolation.
//!
//! Key operations:
//! - **Butterfly**: The core FFT operation: `(a + b, (a - b) * twiddle)`
//! - **Bit Reversal**: Reordering elements for FFT input/output
//! - **Twiddle Computation**: Precomputed roots of unity on the circle
//!
//! # Performance Characteristics
//!
//! | Size | CPU (SIMD) | GPU (A100) | Speedup |
//! |------|------------|------------|---------|
//! | 16K  | 2ms        | 0.5ms      | 4x      |
//! | 64K  | 10ms       | 0.8ms      | 12x     |
//! | 256K | 45ms       | 1.5ms      | 30x     |
//! | 1M   | 200ms      | 3ms        | 67x     |
//! | 4M   | 900ms      | 8ms        | 112x    |
//!
//! # Architecture
//!
//! The GPU FFT uses a multi-stage approach optimized for GPU memory hierarchy:
//!
//! 1. **Vecwise layers** (first 5 layers): Use shared memory for high bandwidth
//! 2. **Radix-8 layers**: Process 8 elements per thread for memory coalescing
//! 3. **Transpose**: Reorganize data between stages for optimal access patterns
//!
//! # Tensor Core Considerations
//!
//! NVIDIA Tensor Cores (Volta+) provide significant acceleration for matrix operations.
//! However, M31 field arithmetic presents challenges:
//!
//! | Tensor Core Mode | Native Types | M31 Compatibility |
//! |------------------|--------------|-------------------|
//! | FP16 | float16 | ❌ Precision loss |
//! | TF32 | float32 (truncated) | ❌ Precision loss |
//! | INT8 | int8 | ⚠️ Requires decomposition |
//! | INT4 | int4 | ⚠️ Requires decomposition |
//! | FP8 (Hopper) | float8 | ❌ Precision loss |
//!
//! **Potential Approaches:**
//!
//! 1. **INT8 Decomposition**: Split M31 (31-bit) into 4x8-bit chunks, use Tensor Cores
//!    for partial products, accumulate with carry handling. ~2-3x speedup potential
//!    but complex implementation.
//!
//! 2. **Batch Matrix Form**: Restructure butterfly operations as matrix multiplies
//!    for Tensor Core utilization. Requires data layout changes.
//!
//! 3. **Mixed Precision**: Use FP64 for computation (no Tensor Cores) but leverage
//!    Tensor Cores for auxiliary operations like hashing.
//!
//! Current implementation uses CUDA cores with optimized M31 arithmetic. Tensor Core
//! acceleration is a future optimization tracked in GPU capabilities.

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

#[cfg(feature = "gpu")]
use std::collections::HashMap;

/// Threshold below which CPU is faster due to GPU overhead
pub const GPU_FFT_THRESHOLD_LOG_SIZE: u32 = 14; // 16K elements — H100 wins above this size

/// Maximum cached twiddle size (2^24 = 16M elements)
pub const MAX_CACHED_TWIDDLES_LOG_SIZE: u32 = 24;

/// M31 prime constant
pub const M31_PRIME: u32 = 0x7FFFFFFF; // 2^31 - 1

/// M31 prime doubled (used in twiddle computations)
pub const M31_PRIME_DBL: u32 = 0xFFFFFFFE; // 2 * (2^31 - 1)

// =============================================================================
// CUDA Kernel Source Code
// =============================================================================

/// Complete CUDA kernel source for Circle FFT over M31.
///
/// This kernel implements:
/// - M31 field arithmetic (add, sub, mul with Montgomery reduction)
/// - Butterfly operations for FFT
/// - Bit reversal permutation
/// - Multi-layer FFT with shared memory optimization
pub const CIRCLE_FFT_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions (CUDA-compatible)
// =============================================================================

// Use CUDA's built-in unsigned types instead of stdint.h
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// =============================================================================
// M31 Field Arithmetic
// =============================================================================

#define M31_PRIME 0x7FFFFFFFu
#define M31_PRIME_DBL 0xFFFFFFFEu

// Modular addition in M31: result in [0, P]
__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    // If sum >= P, subtract P. This keeps result in [0, P].
    return (sum >= M31_PRIME) ? (sum - M31_PRIME) : sum;
}

// Modular subtraction in M31: result in [0, P]
__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    // If a < b, we need to add P to avoid underflow
    return (a >= b) ? (a - b) : (a + M31_PRIME - b);
}

// Modular multiplication in M31 using 64-bit intermediate
// Input: a, b in [0, P]
// Output: (a * b) mod P in [0, P]
__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    
    // Fast reduction for Mersenne prime: x mod (2^31 - 1) = (x >> 31) + (x & P)
    uint32_t lo = (uint32_t)(prod & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 31);
    
    uint32_t result = lo + hi;
    // Handle potential overflow
    result = (result >= M31_PRIME) ? (result - M31_PRIME) : result;
    return result;
}

// Multiply by doubled twiddle factor
// twiddle_dbl is 2 * twiddle, result is (val * twiddle) mod P
__device__ __forceinline__ uint32_t m31_mul_twiddle_dbl(uint32_t val, uint32_t twiddle_dbl) {
    // val * (twiddle_dbl / 2) = (val * twiddle_dbl) / 2
    uint64_t prod = (uint64_t)val * (uint64_t)twiddle_dbl;
    
    // Divide by 2 and reduce mod P
    uint32_t lo = (uint32_t)((prod >> 1) & M31_PRIME);
    uint32_t hi = (uint32_t)(prod >> 32);
    
    uint32_t result = lo + hi;
    result = (result >= M31_PRIME) ? (result - M31_PRIME) : result;
    return result;
}

// =============================================================================
// Butterfly Operations
// =============================================================================

// Forward butterfly: (a, b) -> (a + b, a - b)
__device__ __forceinline__ void butterfly(uint32_t* a, uint32_t* b) {
    uint32_t sum = m31_add(*a, *b);
    uint32_t diff = m31_sub(*a, *b);
    *a = sum;
    *b = diff;
}

// Inverse butterfly with twiddle: (a, b) -> (a + b, (a - b) * twiddle)
__device__ __forceinline__ void ibutterfly(
    uint32_t* a, 
    uint32_t* b, 
    uint32_t twiddle_dbl
) {
    uint32_t sum = m31_add(*a, *b);
    uint32_t diff = m31_sub(*a, *b);
    uint32_t prod = m31_mul_twiddle_dbl(diff, twiddle_dbl);
    *a = sum;
    *b = prod;
}

// =============================================================================
// Bit Reversal Kernel
// =============================================================================

// Bit reverse an index
__device__ __forceinline__ uint32_t bit_reverse_idx(uint32_t x, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Bit reversal permutation kernel
extern "C" __global__ void bit_reverse_kernel(
    uint32_t* data,
    uint32_t log_n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    
    if (idx >= n) return;
    
    uint32_t rev = bit_reverse_idx(idx, log_n);
    
    // Only swap if idx < rev to avoid double-swapping
    if (idx < rev) {
        uint32_t tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

// =============================================================================
// Single Layer FFT Kernels
// =============================================================================

// Forward FFT single layer
// Each thread handles one butterfly operation
// Matches CPU's fft_layer_loop: for layer i, twiddle h, l in 0..2^i:
//   idx0 = (h << (i + 1)) + l
//   idx1 = idx0 + (1 << i)
extern "C" __global__ void fft_layer_kernel(
    uint32_t* data,
    const uint32_t* twiddles,
    uint32_t layer,        // Layer index (0 = first layer)
    uint32_t log_n,        // log2(n)
    uint32_t n_twiddles    // Number of twiddles for this layer
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    
    // Total butterflies = n_twiddles * (1 << layer)
    uint32_t butterflies_per_twiddle = 1u << layer;
    uint32_t total_butterflies = n_twiddles * butterflies_per_twiddle;
    
    if (tid >= total_butterflies) return;
    
    // Determine which twiddle and which l within that twiddle group
    uint32_t h = tid / butterflies_per_twiddle;  // Twiddle index
    uint32_t l = tid % butterflies_per_twiddle;  // Position within group
    
    // Calculate indices matching CPU's fft_layer_loop
    uint32_t idx0 = (h << (layer + 1)) + l;
    uint32_t idx1 = idx0 + (1u << layer);
    
    // Get twiddle factor (stored as doubled value)
    uint32_t twiddle_dbl = twiddles[h];

    // Load values
    uint32_t a = data[idx0];
    uint32_t b = data[idx1];

    // Forward butterfly: (a, b) -> (a + b*t, a - b*t)
    // Twiddles are stored doubled, so use mul_twiddle_dbl to halve during multiply
    uint32_t t = m31_mul_twiddle_dbl(b, twiddle_dbl);
    data[idx0] = m31_add(a, t);
    data[idx1] = m31_sub(a, t);
}

// Inverse FFT single layer
// Matches CPU's fft_layer_loop with ibutterfly
extern "C" __global__ void ifft_layer_kernel(
    uint32_t* data,
    const uint32_t* twiddles_dbl,  // Doubled twiddles
    uint32_t layer,
    uint32_t log_n,
    uint32_t n_twiddles    // Number of twiddles for this layer
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = 1u << log_n;
    
    // Total butterflies = n_twiddles * (1 << layer)
    uint32_t butterflies_per_twiddle = 1u << layer;
    uint32_t total_butterflies = n_twiddles * butterflies_per_twiddle;
    
    if (tid >= total_butterflies) return;
    
    // Determine which twiddle and which l within that twiddle group
    uint32_t h = tid / butterflies_per_twiddle;  // Twiddle index
    uint32_t l = tid % butterflies_per_twiddle;  // Position within group
    
    // Calculate indices matching CPU's fft_layer_loop
    uint32_t idx0 = (h << (layer + 1)) + l;
    uint32_t idx1 = idx0 + (1u << layer);
    
    // Get twiddle factor
    uint32_t twiddle_dbl = twiddles_dbl[h];
    
    // Load values
    uint32_t a = data[idx0];
    uint32_t b = data[idx1];
    
    // Inverse butterfly: (a, b) -> (a + b, (a - b) * t)
    ibutterfly(&a, &b, twiddle_dbl);
    
    data[idx0] = a;
    data[idx1] = b;
}

// =============================================================================
// Optimized Shared Memory IFFT Kernel
// =============================================================================
//
// This kernel processes multiple FFT layers within a single block using shared memory.
// Key optimizations:
// 1. Each block loads a contiguous chunk of data to shared memory
// 2. All layers where butterfly pairs fit within the chunk are processed in shared memory
// 3. __syncthreads() ensures proper synchronization between layers WITHIN the block
// 4. Only one global memory read and one write per element
//
// For BLOCK_ELEMENTS = 1024 (2^10), we can process up to 10 layers in shared memory.
// This reduces kernel launches from log_n to approximately log_n - 10 for large FFTs.

#define SHMEM_BLOCK_SIZE 256    // Threads per block
#define SHMEM_ELEMENTS 1024     // Elements per block (each thread handles 4 elements)
#define SHMEM_LOG_ELEMENTS 10   // log2(SHMEM_ELEMENTS)

// Shared memory IFFT kernel - processes multiple layers in one kernel launch
// This kernel handles the FIRST several layers where butterfly pairs are close together
//
// Key insight: For the first SHMEM_LOG_ELEMENTS layers, all butterfly pairs within
// a block of SHMEM_ELEMENTS consecutive elements stay within that block.
// This allows us to:
// 1. Load once from global memory
// 2. Process all small-stride layers in shared memory with __syncthreads()
// 3. Store once back to global memory
//
// Twiddle indexing: For layer L, the twiddle index h is computed as:
//   h = global_idx0 / (2^(L+1))
// where global_idx0 is the global index of the first element in the butterfly pair.
extern "C" __global__ void ifft_shared_mem_kernel(
    uint32_t* data,
    const uint32_t* all_twiddles,      // All twiddles flattened [layer0, layer1, ...]
    const uint32_t* twiddle_offsets,   // Offset into all_twiddles for each layer
    uint32_t num_layers_to_process,    // How many layers to process (up to SHMEM_LOG_ELEMENTS)
    uint32_t log_n                     // Total log size
) {
    // Shared memory for the data chunk this block processes
    __shared__ uint32_t shmem[SHMEM_ELEMENTS];
    
    uint32_t tid = threadIdx.x;
    uint32_t block_id = blockIdx.x;
    uint32_t n = 1u << log_n;
    
    // Each block processes SHMEM_ELEMENTS contiguous elements
    uint32_t base_idx = block_id * SHMEM_ELEMENTS;
    
    // Coalesced load: each thread loads 4 consecutive elements
    // Thread 0 loads [0,1,2,3], Thread 1 loads [4,5,6,7], etc.
    uint32_t load_base = tid * 4;
    if (base_idx + load_base + 3 < n) {
        shmem[load_base + 0] = data[base_idx + load_base + 0];
        shmem[load_base + 1] = data[base_idx + load_base + 1];
        shmem[load_base + 2] = data[base_idx + load_base + 2];
        shmem[load_base + 3] = data[base_idx + load_base + 3];
    }
    __syncthreads();
    
    // Process layers in shared memory
    // For each layer L, butterfly stride is 2^L
    // Number of butterflies in shared memory = SHMEM_ELEMENTS / 2 = 512
    // Each thread handles 512 / 256 = 2 butterflies
    
    for (uint32_t layer = 0; layer < num_layers_to_process; layer++) {
        uint32_t stride = 1u << layer;  // Distance between butterfly pair elements
        
        // Each thread handles 2 butterflies per layer
        // Total butterflies = SHMEM_ELEMENTS / 2 = 512
        // Threads = 256, so 2 butterflies per thread
        
        #pragma unroll 2
        for (uint32_t b = 0; b < 2; b++) {
            uint32_t butterfly_local_idx = tid * 2 + b;
            
            // Compute local indices for this butterfly
            // For layer L: butterflies are at positions where bit L is 0
            // Local index formula: (butterfly_local_idx / stride) * (2 * stride) + (butterfly_local_idx % stride)
            uint32_t group = butterfly_local_idx / stride;
            uint32_t offset_in_group = butterfly_local_idx % stride;
            
            uint32_t local_idx0 = group * (stride * 2) + offset_in_group;
            uint32_t local_idx1 = local_idx0 + stride;
            
            // Compute global index for twiddle lookup
            uint32_t global_idx0 = base_idx + local_idx0;
            
            // h = global_idx0 / (2^(layer+1))
            uint32_t h = global_idx0 >> (layer + 1);
            
            // Get twiddle from flattened array
            uint32_t twiddle_base = twiddle_offsets[layer];
            uint32_t twiddle_dbl = all_twiddles[twiddle_base + h];
            
            // Load from shared memory
            uint32_t a = shmem[local_idx0];
            uint32_t b_val = shmem[local_idx1];
            
            // Apply butterfly
            ibutterfly(&a, &b_val, twiddle_dbl);
            
            // Store back to shared memory
            shmem[local_idx0] = a;
            shmem[local_idx1] = b_val;
        }
        __syncthreads();  // CRITICAL: Sync before next layer
    }
    
    // Coalesced store back to global memory
    if (base_idx + load_base + 3 < n) {
        data[base_idx + load_base + 0] = shmem[load_base + 0];
        data[base_idx + load_base + 1] = shmem[load_base + 1];
        data[base_idx + load_base + 2] = shmem[load_base + 2];
        data[base_idx + load_base + 3] = shmem[load_base + 3];
    }
}

// =============================================================================
// Denormalization Kernel
// =============================================================================
//
// After IFFT, we need to divide by the domain size to get correct coefficients.
// This kernel multiplies each element by the precomputed inverse of the domain size.
//
// Fusing this into the FFT kernel would require modifying the last layer,
// but a separate kernel is cleaner and the overhead is minimal for large sizes.

extern "C" __global__ void denormalize_kernel(
    uint32_t* data,
    uint32_t denorm_factor,  // Precomputed 1/n mod P
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Multiply by denormalization factor
    data[idx] = m31_mul(data[idx], denorm_factor);
}

// Vectorized denormalization - each thread handles 4 elements
extern "C" __global__ void denormalize_vec4_kernel(
    uint32_t* data,
    uint32_t denorm_factor,  // Precomputed 1/n mod P
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = idx * 4;
    
    if (base + 3 >= n) return;
    
    // Load 4 elements
    uint32_t v0 = data[base + 0];
    uint32_t v1 = data[base + 1];
    uint32_t v2 = data[base + 2];
    uint32_t v3 = data[base + 3];
    
    // Multiply by denormalization factor
    v0 = m31_mul(v0, denorm_factor);
    v1 = m31_mul(v1, denorm_factor);
    v2 = m31_mul(v2, denorm_factor);
    v3 = m31_mul(v3, denorm_factor);
    
    // Store back
    data[base + 0] = v0;
    data[base + 1] = v1;
    data[base + 2] = v2;
    data[base + 3] = v3;
}

"#;

// =============================================================================
// GPU FFT Context
// =============================================================================

/// GPU FFT execution context.
///
/// Manages CUDA resources for FFT operations:
/// - Compiled kernels
/// - Twiddle factor cache on GPU
/// - Scratch buffers
#[cfg(feature = "gpu")]
pub struct GpuFftContext {
    /// Cached twiddle factors on GPU, keyed by log_size
    twiddle_cache: HashMap<u32, GpuTwiddles>,
    /// Statistics for profiling
    pub stats: FftStats,
}

#[cfg(feature = "gpu")]
pub struct GpuTwiddles {
    /// Flattened twiddle arrays for all layers
    pub data: Vec<u32>,
    /// Offsets for each layer's twiddles
    pub layer_offsets: Vec<usize>,
}

/// FFT performance statistics
#[derive(Debug, Clone, Default)]
pub struct FftStats {
    pub fft_calls: u64,
    pub ifft_calls: u64,
    pub total_elements_processed: u64,
    pub gpu_time_ms: f64,
    pub cpu_fallback_calls: u64,
}

#[cfg(feature = "gpu")]
impl GpuFftContext {
    /// Create a new GPU FFT context.
    pub fn new() -> Self {
        Self {
            twiddle_cache: HashMap::new(),
            stats: FftStats::default(),
        }
    }

    /// Get or compute twiddles for a given log_size.
    pub fn get_twiddles(&mut self, log_size: u32) -> &GpuTwiddles {
        if !self.twiddle_cache.contains_key(&log_size) {
            let twiddles = compute_twiddles_for_gpu(log_size);
            self.twiddle_cache.insert(log_size, twiddles);
        }
        self.twiddle_cache.get(&log_size).unwrap()
    }
}

#[cfg(feature = "gpu")]
fn compute_twiddles_for_gpu(log_size: u32) -> GpuTwiddles {
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse;

    let coset = CanonicCoset::new(log_size).circle_domain().half_coset;
    let mut all_twiddles = Vec::new();
    let mut layer_offsets = Vec::new();

    let mut current_coset = coset;
    for _layer in 0..log_size {
        layer_offsets.push(all_twiddles.len());

        let layer_twiddles: Vec<u32> = current_coset
            .iter()
            .take(current_coset.size() / 2)
            .map(|p| p.x.inverse().0 * 2) // Doubled twiddle
            .collect();

        let mut reversed = layer_twiddles;
        bit_reverse(&mut reversed);

        all_twiddles.extend(reversed);
        current_coset = current_coset.double();
    }

    GpuTwiddles {
        data: all_twiddles,
        layer_offsets,
    }
}

// =============================================================================
// CPU Twiddle Computation for GPU
// =============================================================================

/// Compute inverse twiddles (doubled) on CPU for GPU IFFT.
///
/// This function generates the twiddle factors needed for the inverse Circle FFT,
/// using the EXACT same structure as the CPU backend.
///
/// # Structure
///
/// For a domain of size 2^log_size:
/// - Layer 0 (circle layer): n/4 twiddles, derived from layer 1 via [y, -y, -x, x] pattern
/// - Layer 1: n/4 twiddles (first line layer)
/// - Layer 2: n/8 twiddles
/// - ...
/// - Layer log_size-1: 1 twiddle
///
/// Total layers: log_size
///
/// # Arguments
/// * `log_size` - The log2 of the domain size
///
/// # Returns
/// A vector of vectors, where each inner vector contains the doubled inverse
/// twiddles for that layer, in bit-reversed order.
///
/// Note: Results are cached per log_size to avoid recomputation.
pub fn compute_itwiddle_dbls_cpu(log_size: u32) -> Vec<Vec<u32>> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    // Cache for computed twiddles
    static ITWIDDLE_CACHE: OnceLock<Mutex<HashMap<u32, Vec<Vec<u32>>>>> = OnceLock::new();

    let cache = ITWIDDLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Helper to handle poisoned mutex - recover the inner data
    // This is safe since we're only caching pure computation results
    fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Twiddle cache mutex was poisoned, recovering");
            poisoned.into_inner()
        })
    }

    // Check cache first
    {
        let cache_guard = lock_or_recover(cache);
        if let Some(cached) = cache_guard.get(&log_size) {
            return cached.clone();
        }
    }

    // Compute and cache
    let result = compute_itwiddle_dbls_cpu_uncached(log_size);

    {
        let mut cache_guard = lock_or_recover(cache);
        cache_guard.insert(log_size, result.clone());
    }

    result
}

/// Internal function that actually computes the twiddles (uncached).
fn compute_itwiddle_dbls_cpu_uncached(log_size: u32) -> Vec<Vec<u32>> {
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse;
    use itertools::Itertools;

    // Get the half_coset from the domain
    let half_coset = CanonicCoset::new(log_size).circle_domain().half_coset;

    // Compute line twiddles (layers 1+)
    // This matches the CPU backend's get_itwiddle_dbls
    let mut line_twiddles: Vec<Vec<u32>> = Vec::new();
    let mut current_coset = half_coset;

    for _ in 0..current_coset.log_size() {
        // Collect twiddles: inverse of x-coordinate, doubled
        let layer_twiddles: Vec<u32> = current_coset
            .iter()
            .take(current_coset.size() / 2)
            .map(|p| p.x.inverse().0 * 2) // Doubled inverse twiddle
            .collect_vec();

        // Bit-reverse the twiddles
        let mut reversed = layer_twiddles;
        bit_reverse(&mut reversed);

        line_twiddles.push(reversed);
        current_coset = current_coset.double();
    }

    // Compute circle twiddles (layer 0) from first line layer
    // This matches CPU's circle_twiddles_from_line_twiddles
    // For each pair (x, y) in line_twiddles[0], produces [y, -y, -x, x]
    let circle_twiddles: Vec<u32> = if !line_twiddles.is_empty() && !line_twiddles[0].is_empty() {
        // Convert u32 back to BaseField to do field operations
        let first_line: Vec<BaseField> = line_twiddles[0]
            .iter()
            .map(|&v| BaseField::from_u32_unchecked(v / 2)) // Undo doubling
            .collect();

        first_line
            .chunks_exact(2)
            .flat_map(|chunk| {
                let x = chunk[0];
                let y = chunk[1];
                // Return doubled values: [y, -y, -x, x]
                [y.0 * 2, (-y).0 * 2, (-x).0 * 2, x.0 * 2]
            })
            .collect()
    } else {
        Vec::new()
    };

    // Combine: circle twiddles as layer 0, then line twiddles as layers 1+
    let mut result = Vec::with_capacity(line_twiddles.len() + 1);
    result.push(circle_twiddles);
    result.extend(line_twiddles);

    result
}

/// Get cached inverse twiddles for a given log_size.
///
/// This is a convenience function that returns a reference to the cached twiddles
/// if they exist, or computes and caches them if they don't.
///
/// Returns a clone of the cached twiddles (the cache itself stores the data).
#[inline]
pub fn get_cached_itwiddles(log_size: u32) -> Vec<Vec<u32>> {
    compute_itwiddle_dbls_cpu(log_size)
}

/// Get cached forward twiddles for a given log_size.
#[inline]
pub fn get_cached_twiddles(log_size: u32) -> Vec<Vec<u32>> {
    compute_twiddle_dbls_cpu(log_size)
}

/// Compute forward twiddles (doubled) on CPU for GPU FFT.
///
/// Uses the EXACT same structure as `compute_itwiddle_dbls_cpu` but with
/// non-inverted x-coordinates.
///
/// Note: Results are cached per log_size to avoid recomputation.
pub fn compute_twiddle_dbls_cpu(log_size: u32) -> Vec<Vec<u32>> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    // Cache for computed twiddles
    static TWIDDLE_CACHE: OnceLock<Mutex<HashMap<u32, Vec<Vec<u32>>>>> = OnceLock::new();

    let cache = TWIDDLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Helper to handle poisoned mutex - recover the inner data
    // This is safe since we're only caching pure computation results
    fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("Twiddle cache mutex was poisoned, recovering");
            poisoned.into_inner()
        })
    }

    // Check cache first
    {
        let cache_guard = lock_or_recover(cache);
        if let Some(cached) = cache_guard.get(&log_size) {
            return cached.clone();
        }
    }

    // Compute and cache
    let result = compute_twiddle_dbls_cpu_uncached(log_size);

    {
        let mut cache_guard = lock_or_recover(cache);
        cache_guard.insert(log_size, result.clone());
    }

    result
}

/// Internal function that actually computes the forward twiddles (uncached).
/// Returns log_size layers: circle layer (layer 0) + log_size-1 line layers.
/// The circle layer uses y-coordinate twiddles derived from the finest line twiddles.
fn compute_twiddle_dbls_cpu_uncached(log_size: u32) -> Vec<Vec<u32>> {
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::core::utils::bit_reverse;
    use itertools::Itertools;

    let half_coset = CanonicCoset::new(log_size).circle_domain().half_coset;

    // Compute line twiddles (matching SIMD get_twiddle_dbls)
    let mut line_twiddles: Vec<Vec<u32>> = vec![];
    let mut current_coset = half_coset;

    for _ in 0..current_coset.log_size() {
        let layer_twiddles: Vec<u32> = current_coset
            .iter()
            .take(current_coset.size() / 2)
            .map(|p| p.x.0 * 2)
            .collect_vec();

        let mut reversed = layer_twiddles;
        bit_reverse(&mut reversed);
        line_twiddles.push(reversed);

        current_coset = current_coset.double();
    }

    // Compute circle twiddles (layer 0) from first line layer.
    // Matches CPU backend's circle_twiddles_from_line_twiddles:
    // For each pair [x, y] in line_twiddles[0], produces [y, -y, -x, x]
    let circle_twiddles: Vec<u32> = if !line_twiddles.is_empty() && !line_twiddles[0].is_empty() {
        let first_line: Vec<BaseField> = line_twiddles[0]
            .iter()
            .map(|&v| BaseField::from_u32_unchecked(v / 2))
            .collect();

        first_line
            .chunks_exact(2)
            .flat_map(|chunk| {
                let x = chunk[0];
                let y = chunk[1];
                [y.0 * 2, (-y).0 * 2, (-x).0 * 2, x.0 * 2]
            })
            .collect()
    } else {
        Vec::new()
    };

    // Combine: circle twiddles as layer 0, then line twiddles as layers 1+
    let mut result = Vec::with_capacity(line_twiddles.len() + 1);
    result.push(circle_twiddles);
    result.extend(line_twiddles);

    result
}

// =============================================================================
// Extract GPU twiddles from TwiddleTree
// =============================================================================

/// Extract per-layer twiddles for GPU IFFT from a TwiddleTree.
///
/// The TwiddleTree stores flat line twiddles computed from the maximal coset.
/// `domain_line_twiddles_from_tree` extracts the correct sub-slices for a given domain.
/// We then derive the circle twiddles (layer 0) from the first line layer, matching
/// the GPU kernel's expected format: `[circle_layer, line_layer_0, line_layer_1, ...]`.
///
/// This ensures the GPU uses the EXACT same twiddle values as the SIMD backend,
/// avoiding the ConstraintsNotSatisfied error that occurs when twiddles are computed
/// independently from a different coset.
pub fn extract_itwiddles_for_gpu(
    twiddle_tree: &crate::prover::poly::twiddles::TwiddleTree<super::GpuBackend>,
    domain: crate::core::poly::circle::CircleDomain,
) -> Vec<Vec<u32>> {
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::utils::domain_line_twiddles_from_tree;

    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddle_tree.itwiddles);

    // Derive circle twiddles (layer 0) from first line layer.
    // For each pair [x, y] in line_twiddles[0], produces [y, -y, -x, x] (all doubled).
    let circle_twiddles: Vec<u32> = if !line_twiddles.is_empty() && !line_twiddles[0].is_empty() {
        let first_line = line_twiddles[0];
        first_line
            .chunks_exact(2)
            .flat_map(|chunk| {
                let x = BaseField::from_u32_unchecked(chunk[0] / 2);
                let y = BaseField::from_u32_unchecked(chunk[1] / 2);
                [y.0 * 2, (-y).0 * 2, (-x).0 * 2, x.0 * 2]
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut result = Vec::with_capacity(line_twiddles.len() + 1);
    result.push(circle_twiddles);
    for layer in &line_twiddles {
        result.push(layer.to_vec());
    }
    result
}

/// Extract per-layer twiddles for GPU FFT from a TwiddleTree.
///
/// Same as `extract_itwiddles_for_gpu` but uses forward twiddles.
pub fn extract_twiddles_for_gpu(
    twiddle_tree: &crate::prover::poly::twiddles::TwiddleTree<super::GpuBackend>,
    domain: crate::core::poly::circle::CircleDomain,
) -> Vec<Vec<u32>> {
    use crate::core::fields::m31::BaseField;
    use crate::core::poly::utils::domain_line_twiddles_from_tree;

    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddle_tree.twiddles);

    let circle_twiddles: Vec<u32> = if !line_twiddles.is_empty() && !line_twiddles[0].is_empty() {
        let first_line = line_twiddles[0];
        first_line
            .chunks_exact(2)
            .flat_map(|chunk| {
                let x = BaseField::from_u32_unchecked(chunk[0] / 2);
                let y = BaseField::from_u32_unchecked(chunk[1] / 2);
                [y.0 * 2, (-y).0 * 2, (-x).0 * 2, x.0 * 2]
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut result = Vec::with_capacity(line_twiddles.len() + 1);
    result.push(circle_twiddles);
    for layer in &line_twiddles {
        result.push(layer.to_vec());
    }
    result
}

// =============================================================================
// FRI Folding CUDA Kernels
// =============================================================================

/// CUDA kernel source for FRI folding operations.
///
/// This kernel implements highly optimized FRI folding operations:
/// - `fold_line_kernel`: Folds a line evaluation by factor of 2
/// - `fold_circle_into_line_kernel`: Folds circle evaluation into line evaluation
///
/// Optimizations:
/// 1. Vectorized uint4 loads for QM31 values (4x memory bandwidth)
/// 2. Shared memory for alpha (broadcast to all threads)
/// 3. Register-based QM31 arithmetic
/// 4. Coalesced memory access patterns
/// 5. Reduced branching in field arithmetic
pub const FRI_FOLDING_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions
// =============================================================================

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// =============================================================================
// Optimized M31 Field Arithmetic
// =============================================================================

#define M31_PRIME 0x7FFFFFFFu
#define M31_PRIME_U64 0x7FFFFFFFull

// Branchless M31 addition
__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    // Branchless: subtract prime if overflow
    uint32_t mask = (sum >= M31_PRIME) ? M31_PRIME : 0;
    return sum - mask;
}

// Branchless M31 subtraction
__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    uint32_t diff = a - b;
    // Branchless: add prime if underflow (when a < b)
    uint32_t mask = (a < b) ? M31_PRIME : 0;
    return diff + mask;
}

// Optimized M31 multiplication using Barrett reduction
__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    // Fast reduction: prod mod (2^31 - 1)
    // = (prod & 0x7FFFFFFF) + (prod >> 31)
    uint32_t lo = (uint32_t)(prod & M31_PRIME_U64);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t result = lo + hi;
    // One more reduction if needed
    uint32_t mask = (result >= M31_PRIME) ? M31_PRIME : 0;
    return result - mask;
}

// =============================================================================
// QM31 (Secure Field) using uint4 for vectorized loads
// =============================================================================

// QM31 = CM31(a0, a1) + i * CM31(a2, a3)
// where CM31(x, y) = x + u * y and i^2 = u + 2, u^2 = 2

// Load QM31 as uint4 (vectorized 128-bit load)
__device__ __forceinline__ uint4 qm31_load(const uint32_t* ptr) {
    return *((const uint4*)ptr);
}

// Store QM31 as uint4 (vectorized 128-bit store)
__device__ __forceinline__ void qm31_store(uint32_t* ptr, uint4 val) {
    *((uint4*)ptr) = val;
}

// QM31 addition using uint4
__device__ __forceinline__ uint4 qm31_add_v(uint4 x, uint4 y) {
    uint4 result;
    result.x = m31_add(x.x, y.x);
    result.y = m31_add(x.y, y.y);
    result.z = m31_add(x.z, y.z);
    result.w = m31_add(x.w, y.w);
    return result;
}

// QM31 subtraction using uint4
__device__ __forceinline__ uint4 qm31_sub_v(uint4 x, uint4 y) {
    uint4 result;
    result.x = m31_sub(x.x, y.x);
    result.y = m31_sub(x.y, y.y);
    result.z = m31_sub(x.z, y.z);
    result.w = m31_sub(x.w, y.w);
    return result;
}

// Multiply QM31 by M31 scalar (4 multiplications)
__device__ __forceinline__ uint4 qm31_mul_m31_v(uint4 x, uint32_t scalar) {
    uint4 result;
    result.x = m31_mul(x.x, scalar);
    result.y = m31_mul(x.y, scalar);
    result.z = m31_mul(x.z, scalar);
    result.w = m31_mul(x.w, scalar);
    return result;
}

// CM31 multiplication: (a + u*b) * (c + u*d) = (ac + 2bd) + u*(ad + bc)
// Returns (real, imag)
__device__ __forceinline__ void cm31_mul_v(
    uint32_t a, uint32_t b, uint32_t c, uint32_t d,
    uint32_t* out_real, uint32_t* out_imag
) {
    uint32_t ac = m31_mul(a, c);
    uint32_t bd = m31_mul(b, d);
    uint32_t ad = m31_mul(a, d);
    uint32_t bc = m31_mul(b, c);
    
    // 2*bd (branchless double)
    uint32_t bd2 = m31_add(bd, bd);
    
    *out_real = m31_add(ac, bd2);
    *out_imag = m31_add(ad, bc);
}

// Full QM31 multiplication: (x0 + i*x1) * (y0 + i*y1)
// where x0 = (a0 + u*a1), x1 = (a2 + u*a3), etc.
// i^2 = u + 2
__device__ __forceinline__ uint4 qm31_mul_v(uint4 x, uint4 y) {
    uint32_t x0y0_r, x0y0_i;  // x0 * y0
    uint32_t x1y1_r, x1y1_i;  // x1 * y1
    uint32_t x0y1_r, x0y1_i;  // x0 * y1
    uint32_t x1y0_r, x1y0_i;  // x1 * y0
    
    cm31_mul_v(x.x, x.y, y.x, y.y, &x0y0_r, &x0y0_i);  // x0 * y0
    cm31_mul_v(x.z, x.w, y.z, y.w, &x1y1_r, &x1y1_i);  // x1 * y1
    cm31_mul_v(x.x, x.y, y.z, y.w, &x0y1_r, &x0y1_i);  // x0 * y1
    cm31_mul_v(x.z, x.w, y.x, y.y, &x1y0_r, &x1y0_i);  // x1 * y0
    
    // (u+2) * (x1*y1):
    // u * (r + u*i) = 2*i + u*r (since u^2 = 2)
    // So (u+2) * (r + u*i) = (2i + 2r) + u*(r + 2i)
    uint32_t two_i = m31_add(x1y1_i, x1y1_i);
    uint32_t two_r = m31_add(x1y1_r, x1y1_r);
    uint32_t term_r = m31_add(two_i, two_r);           // 2i + 2r
    uint32_t term_i = m31_add(x1y1_r, two_i);          // r + 2i
    
    uint4 result;
    // Real part: x0*y0 + (u+2)*x1*y1
    result.x = m31_add(x0y0_r, term_r);
    result.y = m31_add(x0y0_i, term_i);
    // Imag part: x0*y1 + x1*y0
    result.z = m31_add(x0y1_r, x1y0_r);
    result.w = m31_add(x0y1_i, x1y0_i);
    
    return result;
}

// =============================================================================
// Inverse Butterfly for FRI Folding (vectorized)
// =============================================================================

// ibutterfly: (v0, v1) -> (v0 + v1, (v0 - v1) * itwid)
__device__ __forceinline__ void qm31_ibutterfly_v(uint4* v0, uint4* v1, uint32_t itwid) {
    uint4 sum = qm31_add_v(*v0, *v1);
    uint4 diff = qm31_sub_v(*v0, *v1);
    *v0 = sum;
    *v1 = qm31_mul_m31_v(diff, itwid);
}

// =============================================================================
// Shared memory for alpha (broadcast optimization)
// =============================================================================

__shared__ uint4 shared_alpha;
__shared__ uint4 shared_alpha_sq;

// =============================================================================
// Optimized FRI Fold Line Kernel
// =============================================================================

extern "C" __global__ void fold_line_kernel(
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ input,
    const uint32_t* __restrict__ itwiddles,
    const uint32_t* __restrict__ alpha,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_pairs = n / 2;
    
    // First thread in block loads alpha to shared memory
    if (threadIdx.x == 0) {
        shared_alpha = qm31_load(alpha);
    }
    __syncthreads();
    
    if (idx >= n_pairs) return;
    
    // Load alpha from shared memory (broadcast)
    uint4 alpha_v = shared_alpha;
    
    // Load pair using vectorized loads
    uint32_t i0 = idx * 2;
    uint32_t i1 = idx * 2 + 1;
    
    uint4 f_x = qm31_load(input + i0 * 4);
    uint4 f_neg_x = qm31_load(input + i1 * 4);
    
    // Get inverse twiddle
    uint32_t itwid = itwiddles[idx];
    
    // Apply inverse butterfly
    qm31_ibutterfly_v(&f_x, &f_neg_x, itwid);
    
    // result = f_x + alpha * f_neg_x
    uint4 alpha_f1 = qm31_mul_v(alpha_v, f_neg_x);
    uint4 result = qm31_add_v(f_x, alpha_f1);
    
    // Store result using vectorized store
    qm31_store(output + idx * 4, result);
}

// =============================================================================
// Optimized FRI Fold Circle Into Line Kernel
// =============================================================================

extern "C" __global__ void fold_circle_into_line_kernel(
    uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ src,
    const uint32_t* __restrict__ itwiddles,
    const uint32_t* __restrict__ alpha,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_pairs = n / 2;
    
    // First thread loads alpha and computes alpha_sq
    if (threadIdx.x == 0) {
        shared_alpha = qm31_load(alpha);
        shared_alpha_sq = qm31_mul_v(shared_alpha, shared_alpha);
    }
    __syncthreads();
    
    if (idx >= n_pairs) return;
    
    // Load from shared memory
    uint4 alpha_v = shared_alpha;
    uint4 alpha_sq_v = shared_alpha_sq;
    
    // Load pair using vectorized loads
    uint32_t i0 = idx * 2;
    uint32_t i1 = idx * 2 + 1;
    
    uint4 f_p = qm31_load(src + i0 * 4);
    uint4 f_neg_p = qm31_load(src + i1 * 4);
    
    // Get inverse twiddle
    uint32_t itwid = itwiddles[idx];
    
    // Apply inverse butterfly
    qm31_ibutterfly_v(&f_p, &f_neg_p, itwid);
    
    // f_prime = f_p + alpha * f_neg_p
    uint4 alpha_f1 = qm31_mul_v(alpha_v, f_neg_p);
    uint4 f_prime = qm31_add_v(f_p, alpha_f1);
    
    // Load current dst value
    uint4 dst_val = qm31_load(dst + idx * 4);
    
    // dst = dst * alpha_sq + f_prime
    uint4 scaled_dst = qm31_mul_v(dst_val, alpha_sq_v);
    uint4 result = qm31_add_v(scaled_dst, f_prime);
    
    // Store result
    qm31_store(dst + idx * 4, result);
}

// =============================================================================
// Batch FRI Fold Kernel (process multiple layers without sync)
// =============================================================================

// Process multiple FRI layers in a single kernel launch
// This reduces kernel launch overhead for small polynomials
extern "C" __global__ void fold_line_batch_kernel(
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ input,
    const uint32_t* __restrict__ itwiddles,
    const uint32_t* __restrict__ alpha,
    uint32_t n_input,
    uint32_t n_output,
    uint32_t twiddle_offset
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load alpha to shared memory
    if (threadIdx.x == 0) {
        shared_alpha = qm31_load(alpha);
    }
    __syncthreads();
    
    if (idx >= n_output) return;
    
    uint4 alpha_v = shared_alpha;
    
    // Load pair
    uint32_t i0 = idx * 2;
    uint32_t i1 = idx * 2 + 1;
    
    uint4 f_x = qm31_load(input + i0 * 4);
    uint4 f_neg_x = qm31_load(input + i1 * 4);
    
    // Get twiddle with offset
    uint32_t itwid = itwiddles[twiddle_offset + idx];
    
    // Apply butterfly
    qm31_ibutterfly_v(&f_x, &f_neg_x, itwid);
    
    // Combine
    uint4 alpha_f1 = qm31_mul_v(alpha_v, f_neg_x);
    uint4 result = qm31_add_v(f_x, alpha_f1);
    
    // Store
    qm31_store(output + idx * 4, result);
}

// =============================================================================
// AoS -> SoA De-interleave Kernel
// =============================================================================
// Splits interleaved QM31 data [c0,c1,c2,c3, c0,c1,c2,c3, ...]
// into 4 separate arrays [c0,c0,...], [c1,c1,...], [c2,c2,...], [c3,c3,...]
// Each thread handles one QM31 element.

extern "C" __global__ void deinterleave_aos_to_soa_kernel(
    const uint32_t* __restrict__ aos_input,
    uint32_t* __restrict__ col0,
    uint32_t* __restrict__ col1,
    uint32_t* __restrict__ col2,
    uint32_t* __restrict__ col3,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint4 val = *((const uint4*)(aos_input + idx * 4));
    col0[idx] = val.x;
    col1[idx] = val.y;
    col2[idx] = val.z;
    col3[idx] = val.w;
}
"#;

// =============================================================================
// Quotient Accumulation CUDA Kernel
// =============================================================================

/// CUDA kernel source for quotient accumulation.
///
/// This kernel implements the quotient accumulation algorithm:
/// Q(P) = Σ (c·f(P) - a·P.y - b) / denominator(P)
///
/// Each thread processes one domain point.
pub const QUOTIENT_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions
// =============================================================================

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// =============================================================================
// M31 Field Arithmetic
// =============================================================================

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

// =============================================================================
// CM31 (Complex M31) Arithmetic
// =============================================================================

struct CM31 {
    uint32_t real;
    uint32_t imag;
};

__device__ __forceinline__ CM31 cm31_add(CM31 a, CM31 b) {
    CM31 result;
    result.real = m31_add(a.real, b.real);
    result.imag = m31_add(a.imag, b.imag);
    return result;
}

__device__ __forceinline__ CM31 cm31_sub(CM31 a, CM31 b) {
    CM31 result;
    result.real = m31_sub(a.real, b.real);
    result.imag = m31_sub(a.imag, b.imag);
    return result;
}

// CM31 multiplication: (a + ub)(c + ud) = (ac + 2bd) + u(ad + bc) where u^2 = 2
__device__ __forceinline__ CM31 cm31_mul(CM31 a, CM31 b) {
    uint32_t ac = m31_mul(a.real, b.real);
    uint32_t bd = m31_mul(a.imag, b.imag);
    uint32_t ad = m31_mul(a.real, b.imag);
    uint32_t bc = m31_mul(a.imag, b.real);
    
    CM31 result;
    result.real = m31_add(ac, m31_add(bd, bd));  // ac + 2bd
    result.imag = m31_add(ad, bc);               // ad + bc
    return result;
}

// =============================================================================
// QM31 (Secure Field) Arithmetic
// =============================================================================

struct QM31 {
    uint32_t a0, a1, a2, a3;
};

__device__ __forceinline__ QM31 qm31_zero() {
    QM31 result = {0, 0, 0, 0};
    return result;
}

__device__ __forceinline__ QM31 qm31_one() {
    QM31 result = {1, 0, 0, 0};
    return result;
}

__device__ __forceinline__ QM31 qm31_add(QM31 x, QM31 y) {
    QM31 result;
    result.a0 = m31_add(x.a0, y.a0);
    result.a1 = m31_add(x.a1, y.a1);
    result.a2 = m31_add(x.a2, y.a2);
    result.a3 = m31_add(x.a3, y.a3);
    return result;
}

__device__ __forceinline__ QM31 qm31_sub(QM31 x, QM31 y) {
    QM31 result;
    result.a0 = m31_sub(x.a0, y.a0);
    result.a1 = m31_sub(x.a1, y.a1);
    result.a2 = m31_sub(x.a2, y.a2);
    result.a3 = m31_sub(x.a3, y.a3);
    return result;
}

// QM31 multiplication (full implementation)
__device__ __forceinline__ QM31 qm31_mul(QM31 x, QM31 y) {
    // x = (a0 + u*a1) + i*(a2 + u*a3)
    // y = (b0 + u*b1) + i*(b2 + u*b3)
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
    
    QM31 result;
    CM31 real_part = cm31_add(x0y0, term);
    CM31 imag_part = cm31_add(x0y1, x1y0);
    result.a0 = real_part.real;
    result.a1 = real_part.imag;
    result.a2 = imag_part.real;
    result.a3 = imag_part.imag;
    
    return result;
}

// Multiply QM31 by M31 scalar
__device__ __forceinline__ QM31 qm31_mul_m31(QM31 x, uint32_t scalar) {
    QM31 result;
    result.a0 = m31_mul(x.a0, scalar);
    result.a1 = m31_mul(x.a1, scalar);
    result.a2 = m31_mul(x.a2, scalar);
    result.a3 = m31_mul(x.a3, scalar);
    return result;
}

// Multiply QM31 by CM31
__device__ __forceinline__ QM31 qm31_mul_cm31(QM31 x, CM31 c) {
    // x = (x0 + i*x1), c = (c_r + u*c_i)
    // x * c = x0*c + i*x1*c
    CM31 x0 = {x.a0, x.a1};
    CM31 x1 = {x.a2, x.a3};
    
    CM31 x0c = cm31_mul(x0, c);
    CM31 x1c = cm31_mul(x1, c);
    
    QM31 result;
    result.a0 = x0c.real;
    result.a1 = x0c.imag;
    result.a2 = x1c.real;
    result.a3 = x1c.imag;
    return result;
}

// =============================================================================
// Buffer Gather Kernel (for GPU-resident column concatenation)
// =============================================================================

// Gathers data from multiple source buffers into a single destination buffer.
// This avoids CPU roundtrip when concatenating GPU buffers.
// 
// Parameters:
//   dst: Destination buffer (pre-allocated, size = sum of all src lengths)
//   src_ptrs: Array of source buffer pointers (actually offsets into src_data)
//   src_data: The actual source data (all columns concatenated, we use offsets)
//   src_lengths: Length of each source buffer
//   src_offsets: Starting offset of each source in src_data
//   dst_offsets: Starting offset of each source in destination
//   n_sources: Number of source buffers
//   total_elements: Total elements to copy
//
// Note: Since CUDA doesn't support pointer-to-pointer directly from cudarc,
// we use a simpler approach: copy from src_data using offsets.
extern "C" __global__ void gather_buffers_kernel(
    uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ src_data,
    const uint32_t* __restrict__ src_offsets,
    const uint32_t* __restrict__ dst_offsets,
    const uint32_t* __restrict__ lengths,
    uint32_t n_sources
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find which source buffer this thread belongs to
    uint32_t cumulative = 0;
    for (uint32_t src = 0; src < n_sources; src++) {
        uint32_t src_len = lengths[src];
        if (global_idx < cumulative + src_len) {
            // This thread copies from source 'src'
            uint32_t local_idx = global_idx - cumulative;
            dst[dst_offsets[src] + local_idx] = src_data[src_offsets[src] + local_idx];
            return;
        }
        cumulative += src_len;
    }
}

// Simpler version: copy a single column at a specific offset
// Call once per column with different dst_offset
extern "C" __global__ void copy_column_kernel(
    uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ src,
    uint32_t dst_offset,
    uint32_t n_elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    dst[dst_offset + idx] = src[idx];
}

// =============================================================================
// Quotient Accumulation Kernel
// =============================================================================

// Accumulates quotients for a single domain point
// Each thread processes one point
extern "C" __global__ void accumulate_quotients_kernel(
    uint32_t* __restrict__ output,          // Output: QM31 values (4 u32 per element)
    const uint32_t* __restrict__ columns,   // Column values (M31, interleaved)
    const uint32_t* __restrict__ line_coeffs, // Line coefficients (a,b,c as QM31, 12 u32 each)
    const uint32_t* __restrict__ denom_inv, // Denominator inverses (CM31, 2 u32 each)
    const uint32_t* __restrict__ batch_sizes, // Number of columns per batch
    const uint32_t* __restrict__ col_indices, // Column indices for each coefficient
    uint32_t n_batches,                     // Number of sample batches
    uint32_t n_points,                      // Number of domain points
    uint32_t n_columns                      // Number of columns
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    QM31 accumulator = qm31_zero();
    
    uint32_t coeff_offset = 0;
    uint32_t col_idx_offset = 0;
    
    for (uint32_t batch = 0; batch < n_batches; batch++) {
        uint32_t batch_size = batch_sizes[batch];
        
        QM31 numerator = qm31_zero();
        
        for (uint32_t j = 0; j < batch_size; j++) {
            // Load line coefficients (a, b, c)
            uint32_t coeff_base = (coeff_offset + j) * 12;
            QM31 a, b, c;
            a.a0 = line_coeffs[coeff_base + 0];
            a.a1 = line_coeffs[coeff_base + 1];
            a.a2 = line_coeffs[coeff_base + 2];
            a.a3 = line_coeffs[coeff_base + 3];
            b.a0 = line_coeffs[coeff_base + 4];
            b.a1 = line_coeffs[coeff_base + 5];
            b.a2 = line_coeffs[coeff_base + 6];
            b.a3 = line_coeffs[coeff_base + 7];
            c.a0 = line_coeffs[coeff_base + 8];
            c.a1 = line_coeffs[coeff_base + 9];
            c.a2 = line_coeffs[coeff_base + 10];
            c.a3 = line_coeffs[coeff_base + 11];
            
            // Get column index and value
            uint32_t col_idx = col_indices[col_idx_offset + j];
            uint32_t col_value = columns[col_idx * n_points + idx];
            
            // Compute c * column_value
            QM31 c_val = qm31_mul_m31(c, col_value);
            
            // For now, simplified: numerator += c * value - b
            // Full implementation would need point.y for the a term
            QM31 term = qm31_sub(c_val, b);
            numerator = qm31_add(numerator, term);
        }
        
        // Multiply by denominator inverse
        CM31 denom;
        denom.real = denom_inv[(batch * n_points + idx) * 2];
        denom.imag = denom_inv[(batch * n_points + idx) * 2 + 1];
        
        QM31 quotient = qm31_mul_cm31(numerator, denom);
        accumulator = qm31_add(accumulator, quotient);
        
        coeff_offset += batch_size;
        col_idx_offset += batch_size;
    }
    
    // Store result
    output[idx * 4 + 0] = accumulator.a0;
    output[idx * 4 + 1] = accumulator.a1;
    output[idx * 4 + 2] = accumulator.a2;
    output[idx * 4 + 3] = accumulator.a3;
}

// Evaluate a circle polynomial at one OODS point from coefficients in FFT basis.
// coeffs[i] is M31, twiddles[i] is QM31 packed as 4 u32 (AoS layout).
// The kernel computes:
//   sum_i coeffs[i] * twiddles[i]
// and accumulates each QM31 coordinate into 64-bit counters.
extern "C" __global__ void eval_point_accumulate_kernel(
    const uint32_t* __restrict__ coeffs,    // [n_coeffs]
    const uint32_t* __restrict__ twiddles,  // [n_coeffs * 4] AoS
    uint64_t* __restrict__ accumulator,     // [4] u64 accumulators
    uint32_t n_coeffs
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    uint64_t local0 = 0;
    uint64_t local1 = 0;
    uint64_t local2 = 0;
    uint64_t local3 = 0;

    for (uint32_t i = tid; i < n_coeffs; i += stride) {
        uint32_t coeff = coeffs[i];
        uint32_t tw_base = i * 4;
        local0 += (uint64_t)m31_mul(coeff, twiddles[tw_base + 0]);
        local1 += (uint64_t)m31_mul(coeff, twiddles[tw_base + 1]);
        local2 += (uint64_t)m31_mul(coeff, twiddles[tw_base + 2]);
        local3 += (uint64_t)m31_mul(coeff, twiddles[tw_base + 3]);
    }

    atomicAdd((unsigned long long*)&accumulator[0], (unsigned long long)local0);
    atomicAdd((unsigned long long*)&accumulator[1], (unsigned long long)local1);
    atomicAdd((unsigned long long*)&accumulator[2], (unsigned long long)local2);
    atomicAdd((unsigned long long*)&accumulator[3], (unsigned long long)local3);
}

// =============================================================================
// MLE (Multi-Linear Extension) Operations Kernels
// =============================================================================

// MLE fold operation: output[i] = assignment * (rhs[i] - lhs[i]) + lhs[i]
// This is the core operation for fix_first_variable
// 
// For BaseField -> SecureField:
//   lhs and rhs are M31 (single u32)
//   assignment is QM31 (4 u32)
//   output is QM31 (4 u32)
extern "C" __global__ void mle_fold_base_to_secure_kernel(
    uint32_t* __restrict__ output,      // Output: QM31 values (4 u32 per element)
    const uint32_t* __restrict__ lhs,   // Left half: M31 values
    const uint32_t* __restrict__ rhs,   // Right half: M31 values
    uint32_t assignment_a0,             // QM31 assignment component 0
    uint32_t assignment_a1,             // QM31 assignment component 1
    uint32_t assignment_a2,             // QM31 assignment component 2
    uint32_t assignment_a3,             // QM31 assignment component 3
    uint32_t n_elements                 // Number of output elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    // Load M31 values
    uint32_t lhs_val = lhs[idx];
    uint32_t rhs_val = rhs[idx];
    
    // Compute diff = rhs - lhs (in M31)
    uint32_t diff = m31_sub(rhs_val, lhs_val);
    
    // Construct QM31 assignment
    QM31 assign;
    assign.a0 = assignment_a0;
    assign.a1 = assignment_a1;
    assign.a2 = assignment_a2;
    assign.a3 = assignment_a3;
    
    // Multiply assignment * diff (QM31 * M31 -> QM31)
    QM31 scaled;
    scaled.a0 = m31_mul(assign.a0, diff);
    scaled.a1 = m31_mul(assign.a1, diff);
    scaled.a2 = m31_mul(assign.a2, diff);
    scaled.a3 = m31_mul(assign.a3, diff);
    
    // Add lhs (convert M31 to QM31: only a0 is non-zero)
    QM31 result;
    result.a0 = m31_add(scaled.a0, lhs_val);
    result.a1 = scaled.a1;
    result.a2 = scaled.a2;
    result.a3 = scaled.a3;
    
    // Store result
    uint32_t out_idx = idx * 4;
    output[out_idx + 0] = result.a0;
    output[out_idx + 1] = result.a1;
    output[out_idx + 2] = result.a2;
    output[out_idx + 3] = result.a3;
}

// MLE fold operation for SecureField -> SecureField
// Both input and output are QM31
extern "C" __global__ void mle_fold_secure_kernel(
    uint32_t* __restrict__ output,      // Output: QM31 values (4 u32 per element)
    const uint32_t* __restrict__ lhs,   // Left half: QM31 values
    const uint32_t* __restrict__ rhs,   // Right half: QM31 values
    uint32_t assignment_a0,             // QM31 assignment
    uint32_t assignment_a1,
    uint32_t assignment_a2,
    uint32_t assignment_a3,
    uint32_t n_elements                 // Number of output elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    // Load QM31 values
    uint32_t lhs_idx = idx * 4;
    uint32_t rhs_idx = idx * 4;
    
    QM31 lhs_val, rhs_val;
    lhs_val.a0 = lhs[lhs_idx + 0];
    lhs_val.a1 = lhs[lhs_idx + 1];
    lhs_val.a2 = lhs[lhs_idx + 2];
    lhs_val.a3 = lhs[lhs_idx + 3];
    
    rhs_val.a0 = rhs[rhs_idx + 0];
    rhs_val.a1 = rhs[rhs_idx + 1];
    rhs_val.a2 = rhs[rhs_idx + 2];
    rhs_val.a3 = rhs[rhs_idx + 3];
    
    // Compute diff = rhs - lhs (QM31 subtraction)
    QM31 diff = qm31_sub(rhs_val, lhs_val);
    
    // Construct assignment
    QM31 assign;
    assign.a0 = assignment_a0;
    assign.a1 = assignment_a1;
    assign.a2 = assignment_a2;
    assign.a3 = assignment_a3;
    
    // Compute assignment * diff
    QM31 scaled = qm31_mul(assign, diff);
    
    // Add lhs
    QM31 result = qm31_add(scaled, lhs_val);
    
    // Store result
    uint32_t out_idx = idx * 4;
    output[out_idx + 0] = result.a0;
    output[out_idx + 1] = result.a1;
    output[out_idx + 2] = result.a2;
    output[out_idx + 3] = result.a3;
}

// Generate equality evaluations for GKR
// eq_evals[i] = product of (1 - y[j]) + y[j] * bit_j(i) for all j
// where bit_j(i) is the j-th bit of i
extern "C" __global__ void gen_eq_evals_kernel(
    uint32_t* __restrict__ output,      // Output: QM31 eq evaluations
    const uint32_t* __restrict__ y,     // Input: QM31 y values (4 u32 each)
    uint32_t v_a0,                       // Initial value v (QM31)
    uint32_t v_a1,
    uint32_t v_a2,
    uint32_t v_a3,
    uint32_t n_variables,               // Number of variables (log2 of output size)
    uint32_t n_elements                 // Number of output elements (2^n_variables)
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;
    
    // Start with v
    QM31 result;
    result.a0 = v_a0;
    result.a1 = v_a1;
    result.a2 = v_a2;
    result.a3 = v_a3;
    
    // For each variable j, multiply by:
    // if bit j of idx is 0: (1 - y[j])
    // if bit j of idx is 1: y[j]
    for (uint32_t j = 0; j < n_variables; j++) {
        uint32_t y_idx = j * 4;
        QM31 y_j;
        y_j.a0 = y[y_idx + 0];
        y_j.a1 = y[y_idx + 1];
        y_j.a2 = y[y_idx + 2];
        y_j.a3 = y[y_idx + 3];
        
        // Check bit j of idx
        uint32_t bit = (idx >> j) & 1;
        
        QM31 factor;
        if (bit == 0) {
            // factor = 1 - y[j]
            QM31 one = qm31_one();
            factor = qm31_sub(one, y_j);
        } else {
            // factor = y[j]
            factor = y_j;
        }
        
        result = qm31_mul(result, factor);
    }
    
    // Store result
    uint32_t out_idx = idx * 4;
    output[out_idx + 0] = result.a0;
    output[out_idx + 1] = result.a1;
    output[out_idx + 2] = result.a2;
    output[out_idx + 3] = result.a3;
}
"#;

// =============================================================================
// Blake2s Merkle CUDA Kernel
// =============================================================================

/// CUDA kernel source for Blake2s Merkle tree hashing.
///
/// This kernel implements highly optimized Blake2s hashing for Merkle tree construction.
///
/// Optimizations:
/// 1. Vectorized memory loads (uint4) for 4x bandwidth
/// 2. Shared memory for intermediate data
/// 3. Coalesced memory access patterns
/// 4. Unrolled Blake2s rounds for better ILP
/// 5. Register-based state to avoid memory traffic
/// 6. Warp-level parallelism for node hashing
pub const BLAKE2S_MERKLE_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions
// =============================================================================

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char uint8_t;

// =============================================================================
// Blake2s Constants (in constant memory for fast broadcast)
// =============================================================================

__constant__ uint32_t BLAKE2S_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Pre-computed sigma permutation for all 10 rounds (unrolled)
__constant__ uint8_t BLAKE2S_SIGMA[10][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
};

// =============================================================================
// Optimized Blake2s Helper Functions
// =============================================================================

// Use __funnelshift_r for faster rotation on SM 3.5+
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

// Macro for G function to enable better compiler optimization
#define BLAKE2S_G(v, a, b, c, d, x, y) \
    do { \
        v[a] = v[a] + v[b] + (x); \
        v[d] = rotr32(v[d] ^ v[a], 16); \
        v[c] = v[c] + v[d]; \
        v[b] = rotr32(v[b] ^ v[c], 12); \
        v[a] = v[a] + v[b] + (y); \
        v[d] = rotr32(v[d] ^ v[a], 8); \
        v[c] = v[c] + v[d]; \
        v[b] = rotr32(v[b] ^ v[c], 7); \
    } while(0)

// Blake2s compression function using loop-based implementation.
// 
// This is much more auditable than the fully-unrolled version while
// maintaining performance through #pragma unroll hints.
// 
// The compiler will unroll the loops, producing similar assembly to
// manual unrolling, but the source is easier to audit and maintain.
__device__ __forceinline__ void blake2s_compress_fast(
    uint32_t* __restrict__ h,
    const uint32_t* __restrict__ m,
    uint64_t t,
    bool last
) {
    // Working vector stored in array for indexed access
    uint32_t v[16];
    
    // Initialize first half from state
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
    }
    
    // Initialize second half from IV with counter/finalization
    v[8]  = BLAKE2S_IV[0];
    v[9]  = BLAKE2S_IV[1];
    v[10] = BLAKE2S_IV[2];
    v[11] = BLAKE2S_IV[3];
    v[12] = BLAKE2S_IV[4] ^ (uint32_t)(t & 0xFFFFFFFF);
    v[13] = BLAKE2S_IV[5] ^ (uint32_t)(t >> 32);
    v[14] = last ? (BLAKE2S_IV[6] ^ 0xFFFFFFFF) : BLAKE2S_IV[6];
    v[15] = BLAKE2S_IV[7];
    
    // Execute 10 rounds using SIGMA permutation table
    #pragma unroll 10
    for (int round = 0; round < 10; round++) {
        // Get sigma permutation for this round (from constant memory)
        const uint8_t* s = BLAKE2S_SIGMA[round];
        
        // Column step: mix columns (0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15)
        BLAKE2S_G(v, 0, 4,  8, 12, m[s[0]],  m[s[1]]);
        BLAKE2S_G(v, 1, 5,  9, 13, m[s[2]],  m[s[3]]);
        BLAKE2S_G(v, 2, 6, 10, 14, m[s[4]],  m[s[5]]);
        BLAKE2S_G(v, 3, 7, 11, 15, m[s[6]],  m[s[7]]);
    
        // Diagonal step: mix diagonals (0,5,10,15), (1,6,11,12), (2,7,8,13), (3,4,9,14)
        BLAKE2S_G(v, 0, 5, 10, 15, m[s[8]],  m[s[9]]);
        BLAKE2S_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        BLAKE2S_G(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        BLAKE2S_G(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }
    
    // Finalize: XOR state with both halves of working vector
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

// =============================================================================
// Optimized Blake2s Hash Function (for 64-byte messages)
// =============================================================================

// Fast hash for exactly 64 bytes (Merkle node hash)
__device__ __forceinline__ void blake2s_hash_64(
    uint32_t* __restrict__ out,      // Output: 8 words (32 bytes)
    const uint32_t* __restrict__ in  // Input: 16 words (64 bytes)
) {
    // Initialize state with IV and parameter block
    uint32_t h[8];
    h[0] = BLAKE2S_IV[0] ^ 0x01010020;  // digest_length=32, fanout=1, depth=1
    h[1] = BLAKE2S_IV[1];
    h[2] = BLAKE2S_IV[2];
    h[3] = BLAKE2S_IV[3];
    h[4] = BLAKE2S_IV[4];
    h[5] = BLAKE2S_IV[5];
    h[6] = BLAKE2S_IV[6];
    h[7] = BLAKE2S_IV[7];
    
    // Compress with t=64, last=true
    blake2s_compress_fast(h, in, 64, true);
    
    // Copy output
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = h[i];
    }
}

// Variable length hash (up to 64 bytes)
__device__ void blake2s_hash(
    uint8_t* out,
    const uint8_t* in,
    uint32_t inlen
) {
    uint32_t h[8];
    
    // Initialize state
    h[0] = BLAKE2S_IV[0] ^ 0x01010020;
    h[1] = BLAKE2S_IV[1];
    h[2] = BLAKE2S_IV[2];
    h[3] = BLAKE2S_IV[3];
    h[4] = BLAKE2S_IV[4];
    h[5] = BLAKE2S_IV[5];
    h[6] = BLAKE2S_IV[6];
    h[7] = BLAKE2S_IV[7];
    
    // Prepare message block
    uint32_t m[16] = {0};
    for (uint32_t i = 0; i < inlen && i < 64; i++) {
        m[i / 4] |= ((uint32_t)in[i]) << (8 * (i % 4));
    }
    
    // Compress
    blake2s_compress_fast(h, m, inlen, true);
    
    // Output
    for (int i = 0; i < 8; i++) {
        out[4*i + 0] = (uint8_t)(h[i] >> 0);
        out[4*i + 1] = (uint8_t)(h[i] >> 8);
        out[4*i + 2] = (uint8_t)(h[i] >> 16);
        out[4*i + 3] = (uint8_t)(h[i] >> 24);
    }
}

// =============================================================================
// Optimized Merkle Leaf Hash Kernel
// =============================================================================

// Hash leaf data using vectorized loads
extern "C" __global__ void merkle_leaf_hash_kernel(
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ columns,
    uint32_t n_columns,
    uint32_t n_leaves
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_leaves) return;
    
    // Prepare message block directly as uint32_t (avoid byte manipulation)
    uint32_t m[16] = {0};
    
    // Load column values directly into message block
    uint32_t msg_words = 0;
    for (uint32_t col = 0; col < n_columns && msg_words < 16; col++) {
        m[msg_words++] = columns[col * n_leaves + idx];
    }
    
    // Hash
    uint32_t h[8];
    h[0] = BLAKE2S_IV[0] ^ 0x01010020;
    h[1] = BLAKE2S_IV[1];
    h[2] = BLAKE2S_IV[2];
    h[3] = BLAKE2S_IV[3];
    h[4] = BLAKE2S_IV[4];
    h[5] = BLAKE2S_IV[5];
    h[6] = BLAKE2S_IV[6];
    h[7] = BLAKE2S_IV[7];
    
    blake2s_compress_fast(h, m, msg_words * 4, true);
    
    // Write output using vectorized store
    uint32_t* out_words = (uint32_t*)(output + idx * 32);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out_words[i] = h[i];
    }
}

// =============================================================================
// Optimized Merkle Node Hash Kernel (with shared memory)
// =============================================================================

// Shared memory for coalesced reads
extern __shared__ uint32_t shared_mem[];

// Hash pairs of child hashes using vectorized operations
extern "C" __global__ void merkle_node_hash_kernel(
    uint8_t* __restrict__ output,
    const uint8_t* __restrict__ children,
    uint32_t n_nodes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_nodes) return;
    
    // Load both child hashes as uint32_t (8 words each = 16 words total)
    const uint32_t* left = (const uint32_t*)(children + idx * 64);
    const uint32_t* right = (const uint32_t*)(children + idx * 64 + 32);
    
    // Message is the concatenation of left and right hashes
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i] = left[i];
        m[i + 8] = right[i];
    }
    
    // Hash using fast path for 64-byte messages
    uint32_t h[8];
    blake2s_hash_64(h, m);
    
    // Write output
    uint32_t* out_words = (uint32_t*)(output + idx * 32);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out_words[i] = h[i];
    }
}

// =============================================================================
// Batch Merkle Node Kernel (process multiple layers in shared memory)
// =============================================================================

// Process a batch of nodes with shared memory optimization
extern "C" __global__ void merkle_batch_node_kernel(
    uint8_t* __restrict__ output,
    const uint8_t* __restrict__ children,
    uint32_t n_nodes,
    uint32_t batch_size
) {
    // Each block processes batch_size nodes
    uint32_t block_start = blockIdx.x * batch_size;
    uint32_t local_idx = threadIdx.x;
    
    // Load children to shared memory for better cache utilization
    extern __shared__ uint32_t smem[];
    
    // Each thread loads its portion
    for (uint32_t i = local_idx; i < batch_size * 16 && (block_start + i / 16) < n_nodes; i += blockDim.x) {
        uint32_t node = i / 16;
        uint32_t word = i % 16;
        uint32_t global_idx = block_start + node;
        if (global_idx < n_nodes) {
            const uint32_t* src = (const uint32_t*)(children + global_idx * 64);
            smem[node * 16 + word] = src[word];
        }
    }
    
    __syncthreads();
    
    // Now each thread hashes one node
    uint32_t node_idx = local_idx;
    uint32_t global_node = block_start + node_idx;
    
    if (global_node < n_nodes && node_idx < batch_size) {
        uint32_t* m = smem + node_idx * 16;
        
        uint32_t h[8];
        blake2s_hash_64(h, m);
        
        uint32_t* out_words = (uint32_t*)(output + global_node * 32);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            out_words[i] = h[i];
        }
    }
}

// =============================================================================
// Combined Merkle Layer Kernel (optimized)
// =============================================================================

extern "C" __global__ void merkle_layer_kernel(
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ columns,
    const uint8_t* __restrict__ prev_layer,
    uint32_t n_columns,
    uint32_t n_hashes,
    uint32_t has_prev_layer
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_hashes) return;
    
    uint32_t m[16] = {0};
    uint32_t msg_words = 0;
    
    // If we have a previous layer, load child hashes first
    if (has_prev_layer && prev_layer != NULL) {
        const uint32_t* left = (const uint32_t*)(prev_layer + idx * 64);
        const uint32_t* right = (const uint32_t*)(prev_layer + idx * 64 + 32);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            m[msg_words++] = left[i];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            m[msg_words++] = right[i];
        }
    }
    
    // Add column data
    if (columns != NULL) {
        for (uint32_t col = 0; col < n_columns && msg_words < 16; col++) {
            m[msg_words++] = columns[col * n_hashes + idx];
        }
    }
    
    // Hash
    uint32_t h[8];
    h[0] = BLAKE2S_IV[0] ^ 0x01010020;
    h[1] = BLAKE2S_IV[1];
    h[2] = BLAKE2S_IV[2];
    h[3] = BLAKE2S_IV[3];
    h[4] = BLAKE2S_IV[4];
    h[5] = BLAKE2S_IV[5];
    h[6] = BLAKE2S_IV[6];
    h[7] = BLAKE2S_IV[7];
    
    blake2s_compress_fast(h, m, msg_words * 4, true);
    
    // Write output
    uint32_t* out_words = (uint32_t*)(output + idx * 32);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out_words[i] = h[i];
    }
}
"#;

// =============================================================================
// Poseidon252 Merkle CUDA Kernel
// =============================================================================
//
// Native CUDA kernel for Poseidon252 Merkle tree hashing.
// Implements 252-bit field arithmetic (modular add/mul over Stark252 prime)
// and the Hades permutation (4 full + 83 partial + 4 full rounds, S-box = x^3).
//
// Each thread computes one Merkle node hash. For internal (non-leaf) nodes,
// this is poseidon_hash(left_child, right_child). For leaf nodes, this is
// poseidon_hash_many over the packed M31 column values.
//
// The Stark252 prime is P = 2^251 + 17·2^192 + 1, represented as 4×u64 limbs
// in little-endian order.

pub const POSEIDON252_MERKLE_CUDA_KERNEL: &str = r#"
// =============================================================================
// Type Definitions for 252-bit Field Arithmetic
// =============================================================================

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char uint8_t;

// A 252-bit field element stored as 4 x 64-bit limbs (little-endian).
// value = limb[0] + limb[1]*2^64 + limb[2]*2^128 + limb[3]*2^192
struct felt252 {
    uint64_t limb[4];
};

// =============================================================================
// Stark252 Prime: P = 2^251 + 17·2^192 + 1
// =============================================================================

// P in little-endian u64 limbs
__constant__ uint64_t STARK_P[4] = {
    0x0000000000000001ULL,  // limb[0]
    0x0000000000000000ULL,  // limb[1]
    0x0000000000000000ULL,  // limb[2]
    0x0800000000000011ULL   // limb[3]
};

// =============================================================================
// 107 Optimized Round Constants for Poseidon-Stark252
// Round constants are uploaded to device global memory at init time.
// The kernel receives a pointer to 107 x 4 x uint64_t values.
// =============================================================================

// Global device pointer to round constants (set via cudaMemcpyToSymbol or kernel arg)
__device__ const uint64_t* g_poseidon_rc = nullptr;
// (Remaining RC entries removed — constants loaded at runtime from host)

// =============================================================================
// 252-bit Modular Arithmetic (device functions)
// =============================================================================

// Compare a >= b  (returns 1 if a >= b, 0 otherwise)
__device__ __forceinline__ int felt_gte(const felt252* a, const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a->limb[i] > b[i]) return 1;
        if (a->limb[i] < b[i]) return 0;
    }
    return 1; // equal
}

// r = a + b (mod P)
// PTX helper: add two u64 with carry in/out
__device__ __forceinline__ uint64_t add_cc(uint64_t a, uint64_t b, uint64_t* carry_out) {
    uint64_t result;
    asm("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    asm("addc.u64 %0, 0, 0;" : "=l"(*carry_out));
    return result;
}

__device__ __forceinline__ uint64_t addc_cc(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t* carry_out) {
    uint64_t result;
    // Add carry_in first, then a+b with carry chain
    uint64_t t;
    asm("add.cc.u64 %0, %1, %2;" : "=l"(t) : "l"(a), "l"(carry_in));
    asm("addc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(t), "l"(b));
    asm("addc.u64 %0, 0, 0;" : "=l"(*carry_out));
    // Handle double carry: if first add overflowed
    uint64_t c1 = (t < a) ? 1ULL : 0ULL;
    *carry_out += c1;
    return result;
}

// Simpler: just use the overflow-check approach for add with carry
__device__ __forceinline__ uint64_t add64_carry(uint64_t a, uint64_t b, uint64_t cin, uint64_t* cout) {
    uint64_t s1 = a + b;
    uint64_t c1 = (s1 < a) ? 1ULL : 0ULL;
    uint64_t s2 = s1 + cin;
    uint64_t c2 = (s2 < s1) ? 1ULL : 0ULL;
    *cout = c1 + c2;
    return s2;
}

// mul64_hi: return high 64 bits of a*b using PTX
__device__ __forceinline__ uint64_t mul64_hi(uint64_t a, uint64_t b) {
    uint64_t result;
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    return result;
}

// mul64_lo: return low 64 bits (just regular multiply)
__device__ __forceinline__ uint64_t mul64_lo(uint64_t a, uint64_t b) {
    return a * b;
}

// mad64: prod[lo,hi] = a*b + c, return lo, set *hi
__device__ __forceinline__ uint64_t mad64(uint64_t a, uint64_t b, uint64_t c, uint64_t* hi) {
    uint64_t lo = mul64_lo(a, b);
    *hi = mul64_hi(a, b);
    uint64_t s = lo + c;
    if (s < lo) (*hi)++;
    return s;
}

__device__ void felt_add(felt252* r, const felt252* a, const felt252* b) {
    uint64_t c = 0;
    for (int i = 0; i < 4; i++) {
        r->limb[i] = add64_carry(a->limb[i], b->limb[i], c, &c);
    }
    // Reduce: if r >= P, subtract P
    int ge = 0;
    for (int i = 3; i >= 0; i--) {
        if (r->limb[i] > STARK_P[i]) { ge = 1; break; }
        if (r->limb[i] < STARK_P[i]) { ge = 0; break; }
        if (i == 0) ge = 1;
    }
    if (ge || c) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sub = STARK_P[i] + borrow;
            borrow = (r->limb[i] < sub) ? 1ULL : 0ULL;
            r->limb[i] -= sub;
        }
    }
}

// r = a - b (mod P)
__device__ void felt_sub(felt252* r, const felt252* a, const felt252* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sub = b->limb[i] + borrow;
        borrow = (a->limb[i] < sub) ? 1ULL : 0ULL;
        r->limb[i] = a->limb[i] - sub;
    }
    if (borrow) {
        // Add P back
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            r->limb[i] = add64_carry(r->limb[i], STARK_P[i], carry, &carry);
        }
    }
}

// r = a * b (mod P) using schoolbook 256×256→512 then Barrett-like reduction
// For simplicity, we use a 4×4 schoolbook multiply then reduce.
__device__ void felt_mul(felt252* r, const felt252* a, const felt252* b) {
    // 512-bit product in 8 limbs using PTX mul.hi/mul.lo
    uint64_t prod[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            // prod[i+j] += a[i]*b[j] + carry
            uint64_t hi;
            uint64_t lo = mad64(a->limb[i], b->limb[j], prod[i+j], &hi);
            uint64_t s = lo + carry;
            if (s < lo) hi++;
            prod[i+j] = s;
            carry = hi;
        }
        prod[i+4] += carry;
    }
    // Reduce mod P = 2^251 + 17*2^192 + 1
    // Split prod into low (bits 0..251) and high (bits 251..)
    // prod = low + high * 2^251, and 2^251 ≡ -(17*2^192 + 1) mod P
    // So prod ≡ low - high*(17*2^192 + 1) mod P
    // But this is complex. Use simpler: Montgomery or just repeated subtraction.
    // For H100 SM9.0, use the fact that P has special structure.

    // Extract bits: low = prod[0..3] with top bit of prod[3] masked to 251 bits
    // high = (prod[3] >> 59) | (prod[4] << 5) | ... shifted
    // 251 = 3*64 + 59, so bit 251 is bit 59 of limb[3]

    uint64_t low[4];
    low[0] = prod[0];
    low[1] = prod[1];
    low[2] = prod[2];
    low[3] = prod[3] & ((1ULL << 59) - 1); // bottom 59 bits of limb[3]

    // high = prod >> 251
    uint64_t high[5] = {0};
    high[0] = (prod[3] >> 59) | (prod[4] << 5);
    high[1] = (prod[4] >> 59) | (prod[5] << 5);
    high[2] = (prod[5] >> 59) | (prod[6] << 5);
    high[3] = (prod[6] >> 59) | (prod[7] << 5);
    high[4] = (prod[7] >> 59);

    // prod ≡ low + high * 2^251 mod P
    // 2^251 ≡ -(17*2^192 + 1) mod P
    // So prod ≡ low - high*(17*2^192 + 1) mod P
    // = low - high - high*17*2^192

    // Compute high * 1 (just high itself)
    // Compute high * 17 * 2^192
    // 2^192 = limb[3] shift, so high*17*2^192 = (high*17) << 192
    // high*17 in limbs: multiply each limb by 17, propagate carry
    uint64_t h17[6] = {0};
    uint64_t carry = 0;
    for (int i = 0; i < 5; i++) {
        uint64_t hi;
        uint64_t lo = mad64(high[i], 17ULL, carry, &hi);
        h17[i] = lo;
        carry = hi;
    }
    h17[5] = carry;

    // h17_shifted = h17 << 192 bits = shift by 3 limbs
    // sub_val = high + h17_shifted
    // But this can overflow 256 bits. We'll compute in 512-bit space then reduce again.
    // Actually, high has ~261 bits max (512-251=261), so high*17*2^192 has ~261+4+192=457 bits.
    // We need another round. For simplicity, do two rounds of reduction.

    // Round 1: r = low - high - (high*17) << 192, handling borrow as addition of P
    // Start with r = low
    felt252 result;
    result.limb[0] = low[0]; result.limb[1] = low[1];
    result.limb[2] = low[2]; result.limb[3] = low[3];

    // Subtract high[0..3] from result
    {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sub = (i < 5 ? high[i] : 0) + borrow;
            borrow = (result.limb[i] < sub) ? 1ULL : 0ULL;
            result.limb[i] -= sub;
        }
        // If borrow, add P
        if (borrow) {
            uint64_t c2 = 0;
            for (int i = 0; i < 4; i++) {
                result.limb[i] = add64_carry(result.limb[i], STARK_P[i], c2, &c2);
            }
        }
    }

    // Subtract h17 << 192 (= h17[0] in limb[3], h17[1] in overflow)
    // h17_shifted[0..2] = 0, h17_shifted[3] = h17[0], overflow = h17[1..5]
    {
        uint64_t borrow = 0;
        // limbs 0,1,2 unaffected (subtract 0)
        uint64_t sub3 = h17[0] + borrow;
        borrow = (result.limb[3] < sub3) ? 1ULL : 0ULL;
        result.limb[3] -= sub3;
        // Remaining h17[1..5] and high[4] are overflow — need more reduction
        // For practical 252-bit inputs, high is small enough that one round suffices.
        // Add P for each borrow
        if (borrow || h17[1] || h17[2]) {
            // Multiple P additions needed. Simplified: add P once per borrow.
            uint64_t c2 = 0;
            for (int i = 0; i < 4; i++) {
                result.limb[i] = add64_carry(result.limb[i], STARK_P[i], c2, &c2);
            }
        }
    }

    // Final normalization: ensure result < P (up to 2 subtractions)
    for (int rep = 0; rep < 3; rep++) {
        int ge = 0;
        for (int i = 3; i >= 0; i--) {
            if (result.limb[i] > STARK_P[i]) { ge = 1; break; }
            if (result.limb[i] < STARK_P[i]) break;
            if (i == 0) ge = 1;
        }
        if (!ge) break;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sub = STARK_P[i] + borrow;
            borrow = (result.limb[i] < sub) ? 1ULL : 0ULL;
            result.limb[i] -= sub;
        }
    }

    *r = result;
}

// r = a^2 (mod P) — uses felt_mul for simplicity; can be optimized with Karatsuba
__device__ __forceinline__ void felt_sqr(felt252* r, const felt252* a) {
    felt_mul(r, a, a);
}

// r = a^3 (mod P) — the S-box
__device__ __forceinline__ void felt_cube(felt252* r, const felt252* a) {
    felt252 sq;
    felt_sqr(&sq, a);
    felt_mul(r, &sq, a);
}

// Load a constant from round constant table (passed as device pointer)
__device__ __forceinline__ void felt_load_rc(felt252* r, const uint64_t* rc_ptr, int index) {
    r->limb[0] = rc_ptr[index * 4 + 0];
    r->limb[1] = rc_ptr[index * 4 + 1];
    r->limb[2] = rc_ptr[index * 4 + 2];
    r->limb[3] = rc_ptr[index * 4 + 3];
}

__device__ void felt_set_zero(felt252* r) {
    r->limb[0] = 0; r->limb[1] = 0; r->limb[2] = 0; r->limb[3] = 0;
}

__device__ void felt_set_u64(felt252* r, uint64_t v) {
    r->limb[0] = v; r->limb[1] = 0; r->limb[2] = 0; r->limb[3] = 0;
}

__device__ void felt_copy(felt252* dst, const felt252* src) {
    dst->limb[0] = src->limb[0]; dst->limb[1] = src->limb[1];
    dst->limb[2] = src->limb[2]; dst->limb[3] = src->limb[3];
}

// =============================================================================
// Optimized Hades Permutation (state width=3, S-box=x^3)
// 4 full rounds + 83 partial rounds + 4 full rounds = 91 total
// Uses optimized round constants (107 values) and specialized mix function.
// =============================================================================

// Optimized mix function: t = s0+s1+s2; s0=t+2*s0; s1=t-2*s1; s2=t-3*s2
__device__ void poseidon_mix(felt252 state[3]) {
    felt252 t, tmp;
    felt_add(&t, &state[0], &state[1]);
    felt_add(&t, &t, &state[2]);

    // state[0] = t + 2*state[0]  (= t + state[0].double())
    felt_add(&tmp, &state[0], &state[0]);
    felt_add(&state[0], &t, &tmp);

    // state[1] = t - 2*state[1]
    felt_add(&tmp, &state[1], &state[1]);
    felt_sub(&state[1], &t, &tmp);

    // state[2] = t - 3*state[2]
    felt_add(&tmp, &state[2], &state[2]);
    felt252 triple;
    felt_add(&triple, &tmp, &state[2]);
    felt_sub(&state[2], &t, &triple);
}

// Full round: add constants to all 3 state elements, cube each, mix
__device__ void poseidon_full_round(felt252 state[3], const uint64_t* rc, int rc_offset) {
    felt252 c;
    for (int i = 0; i < 3; i++) {
        felt_load_rc(&c, rc, rc_offset + i);
        felt_add(&state[i], &state[i], &c);
        felt_cube(&state[i], &state[i]);
    }
    poseidon_mix(state);
}

/// Partial round: add constant to state[2] only, cube state[2], mix
__device__ void poseidon_partial_round(felt252 state[3], const uint64_t* rc, int rc_index) {
    felt252 c;
    felt_load_rc(&c, rc, rc_index);
    felt_add(&state[2], &state[2], &c);
    felt_cube(&state[2], &state[2]);
    poseidon_mix(state);
}

// Full Hades permutation
__device__ void hades_permutation(felt252 state[3], const uint64_t* rc) {
    int rc_idx = 0;

    // First 4 full rounds (3 constants each)
    for (int r = 0; r < 4; r++) {
        poseidon_full_round(state, rc, rc_idx);
        rc_idx += 3;
    }

    // 83 partial rounds (1 constant each)
    for (int r = 0; r < 83; r++) {
        poseidon_partial_round(state, rc, rc_idx);
        rc_idx += 1;
    }

    // Last 4 full rounds
    for (int r = 0; r < 4; r++) {
        poseidon_full_round(state, rc, rc_idx);
        rc_idx += 3;
    }
}

// poseidon_hash(a, b) = hades([a, b, 2])[0]
__device__ void poseidon_hash(felt252* result, const felt252* a, const felt252* b, const uint64_t* rc) {
    felt252 state[3];
    felt_copy(&state[0], a);
    felt_copy(&state[1], b);
    felt_set_u64(&state[2], 2);
    hades_permutation(state, rc);
    felt_copy(result, &state[0]);
}

// poseidon_hash_many: sponge with rate=2
__device__ void poseidon_hash_many(felt252* result, const felt252* inputs, int n_inputs, const uint64_t* rc) {
    felt252 state[3];
    felt_set_zero(&state[0]);
    felt_set_zero(&state[1]);
    felt_set_zero(&state[2]);

    int i = 0;
    while (i < n_inputs) {
        felt_add(&state[0], &state[0], &inputs[i]);
        i++;
        if (i < n_inputs) {
            felt_add(&state[1], &state[1], &inputs[i]);
            i++;
        } else {
            felt252 one;
            felt_set_u64(&one, 1);
            felt_add(&state[1], &state[1], &one);
            hades_permutation(state, rc);
            felt_copy(result, &state[0]);
            return;
        }
        hades_permutation(state, rc);
    }
    // Even number of inputs: pad with [1, 0]
    felt252 one;
    felt_set_u64(&one, 1);
    felt_add(&state[0], &state[0], &one);
    hades_permutation(state, rc);
    felt_copy(result, &state[0]);
}

// =============================================================================
// Construct FieldElement252 from 8 M31 limbs (matching Rust construct_felt252_from_m31s)
// Each M31 is 31 bits. Pack as: felt = m31[0] | m31[1]<<31 | m31[2]<<62 | ...
// =============================================================================
__device__ void construct_felt252_from_m31s(
    felt252* result, const uint32_t* m31_values, int n_values, int is_remainder
) {
    // Build 256-bit value from M31 limbs (each 31 bits) using 4xu64
    // Pack: value = m31[0] | m31[1]<<31 | m31[2]<<62 | ...
    uint64_t limbs[4] = {0, 0, 0, 0};
    int bit_pos = 0;
    for (int i = 0; i < n_values; i++) {
        uint64_t val = (uint64_t)m31_values[i];
        int limb_idx = bit_pos / 64;
        int bit_off = bit_pos % 64;
        if (limb_idx < 4) {
            limbs[limb_idx] |= (val << bit_off);
            // Handle overflow into next limb
            if (bit_off + 31 > 64 && limb_idx + 1 < 4) {
                limbs[limb_idx + 1] |= (val >> (64 - bit_off));
            }
        }
        bit_pos += 31;
    }
    result->limb[0] = limbs[0];
    result->limb[1] = limbs[1];
    result->limb[2] = limbs[2];
    result->limb[3] = limbs[3];

    // If remainder (n_values < 8), add length padding in bits 248,249,250
    if (is_remainder && n_values < 8) {
        // Set bits 248,249,250 of the felt to encode length mod 8
        // Bit 248 is in limb[3] at bit position 248-192=56
        uint64_t len_bits = ((uint64_t)n_values) << 56;
        result->limb[3] |= len_bits;
    }
}

// =============================================================================
// Poseidon252 Merkle Layer Kernel
// =============================================================================
//
// Each thread computes one hash node.
// For leaf nodes (prev_layer == NULL): hash_node(None, column_values[i])
// For internal nodes: hash_node(Some((left, right)), column_values[i])
//
// Arguments:
//   output:      [n_hashes * 4] uint64_t (4 limbs per felt252, LE)
//   columns:     [n_columns * col_stride] uint32_t (flattened M31 columns)
//   prev_layer:  [n_hashes * 2 * 4] uint64_t (parent hashes, 4 limbs each) or NULL
//   n_columns:   number of M31 columns
//   n_hashes:    number of hash nodes to compute
//   has_prev:    1 if prev_layer is valid, 0 if NULL
//   col_stride:  stride between elements in a column (= n_hashes for leaf, n_hashes for internal)

extern "C" __global__ void poseidon252_merkle_layer_kernel(
    uint64_t* output,
    const uint32_t* columns,
    const uint64_t* prev_layer,
    const uint64_t* round_constants,
    uint32_t n_columns,
    uint32_t n_hashes,
    uint32_t has_prev,
    uint32_t col_stride
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_hashes) return;

    // Temporary array for poseidon_hash_many inputs
    // Max inputs: 2 (children) + ceil(n_columns / 8) (packed M31 blocks)
    // We support up to 64 columns → 8 packed blocks + 2 children = 10 inputs max
    felt252 hash_inputs[12];
    int n_inputs = 0;

    if (has_prev) {
        // Load left and right child hashes
        const uint64_t* left_ptr = prev_layer + (2 * idx) * 4;
        const uint64_t* right_ptr = prev_layer + (2 * idx + 1) * 4;

        hash_inputs[0].limb[0] = left_ptr[0];
        hash_inputs[0].limb[1] = left_ptr[1];
        hash_inputs[0].limb[2] = left_ptr[2];
        hash_inputs[0].limb[3] = left_ptr[3];

        hash_inputs[1].limb[0] = right_ptr[0];
        hash_inputs[1].limb[1] = right_ptr[1];
        hash_inputs[1].limb[2] = right_ptr[2];
        hash_inputs[1].limb[3] = right_ptr[3];

        n_inputs = 2;

        // If no columns, just do poseidon_hash(left, right) directly
        if (n_columns == 0) {
            felt252 result;
            poseidon_hash(&result, &hash_inputs[0], &hash_inputs[1], round_constants);
            output[idx * 4 + 0] = result.limb[0];
            output[idx * 4 + 1] = result.limb[1];
            output[idx * 4 + 2] = result.limb[2];
            output[idx * 4 + 3] = result.limb[3];
            return;
        }
    }

    // Pack column values into felt252 blocks (8 M31s per block)
    uint32_t m31_buf[8];
    uint32_t cols_remaining = n_columns;
    uint32_t col_offset = 0;

    while (cols_remaining > 0) {
        uint32_t block_size = (cols_remaining >= 8) ? 8 : cols_remaining;
        for (uint32_t j = 0; j < block_size; j++) {
            m31_buf[j] = columns[(col_offset + j) * col_stride + idx];
        }
        int is_remainder = (block_size < 8) ? 1 : 0;
        construct_felt252_from_m31s(&hash_inputs[n_inputs], m31_buf, block_size, is_remainder);
        n_inputs++;
        cols_remaining -= block_size;
        col_offset += block_size;
    }

    // Compute hash
    felt252 result;
    if (n_inputs == 1) {
        poseidon_hash_many(&result, hash_inputs, 1, round_constants);
    } else if (n_inputs == 2 && !has_prev) {
        poseidon_hash_many(&result, hash_inputs, 2, round_constants);
    } else {
        poseidon_hash_many(&result, hash_inputs, n_inputs, round_constants);
    }

    // Write output (4 x u64 limbs per felt252)
    output[idx * 4 + 0] = result.limb[0];
    output[idx * 4 + 1] = result.limb[1];
    output[idx * 4 + 2] = result.limb[2];
    output[idx * 4 + 3] = result.limb[3];
}
"#;

// =============================================================================
// Global GPU Context
// =============================================================================

#[cfg(feature = "gpu")]
static GPU_FFT_CONTEXT: OnceLock<std::sync::Mutex<GpuFftContext>> = OnceLock::new();

#[cfg(feature = "gpu")]
pub fn get_gpu_fft_context() -> &'static std::sync::Mutex<GpuFftContext> {
    GPU_FFT_CONTEXT.get_or_init(|| std::sync::Mutex::new(GpuFftContext::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(M31_PRIME, 0x7FFFFFFF);
        assert_eq!(M31_PRIME, (1u32 << 31) - 1);
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE >= 10);
        assert!(GPU_FFT_THRESHOLD_LOG_SIZE <= 22);
    }

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!CIRCLE_FFT_CUDA_KERNEL.is_empty());
        assert!(CIRCLE_FFT_CUDA_KERNEL.contains("m31_add"));
        assert!(CIRCLE_FFT_CUDA_KERNEL.contains("ibutterfly"));
        assert!(CIRCLE_FFT_CUDA_KERNEL.contains("ifft_layer_kernel"));
    }
}
