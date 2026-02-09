//! GPU Column implementations.
//!
//! This module provides GPU-backed column types that implement the [`Column`] trait.
//!
//! # Architecture
//!
//! The GPU backend uses the same column types as SIMD (data layout is identical),
//! but provides GPU-accelerated operations. Data is transferred to GPU only when
//! needed for bulk operations.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        BaseColumn (CPU)                         │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │  PackedM31[0]  │  PackedM31[1]  │  ...  │  PackedM31[n] │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!                   to_gpu()   │   from_gpu()
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      GpuBuffer<u32> (GPU)                       │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │   u32[0]   │   u32[1]   │   ...   │   u32[n*16]         │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! - GPU transfer overhead: ~50μs for 1MB
//! - Minimum size for GPU benefit: 16K elements (2^14)
//! - PCIe bandwidth: ~12 GB/s (Gen3 x16)

use starknet_ff::FieldElement as FieldElement252;

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::prover::backend::simd::column::{BaseColumn, SecureColumn};
use crate::prover::backend::ColumnOps;
#[cfg(feature = "cuda-runtime")]
use crate::prover::backend::Column;

use super::GpuBackend;

#[cfg(feature = "cuda-runtime")]
use super::memory::GpuBuffer;

#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::{CudaFftError, get_cuda_executor};

// =============================================================================
// Constants
// =============================================================================

/// Minimum column size (in elements) to benefit from GPU acceleration.
/// Below this threshold, CPU/SIMD is faster due to transfer overhead.
pub const GPU_THRESHOLD_ELEMENTS: usize = 1 << 14; // 16K elements

// =============================================================================
// ColumnOps Implementation for GpuBackend
// =============================================================================

impl ColumnOps<BaseField> for GpuBackend {
    type Column = BaseColumn;
    
    fn bit_reverse_column(column: &mut Self::Column) {
        // Use GPU for large columns, CPU for small ones
        #[cfg(feature = "cuda-runtime")]
        {
            let len = column.len();
            if len >= GPU_THRESHOLD_ELEMENTS {
                if let Ok(()) = gpu_bit_reverse_base_column(column) {
                    return;
                }
                // GPU failed, fall through to CPU
                tracing::warn!("GPU bit_reverse failed, using CPU");
            }
        }
        
        // CPU path (always available)
        use crate::prover::backend::simd::SimdBackend;
        <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(column);
    }
}

impl ColumnOps<SecureField> for GpuBackend {
    type Column = SecureColumn;
    
    fn bit_reverse_column(column: &mut Self::Column) {
        // SecureField is 4x BaseField, so threshold is lower
        #[cfg(feature = "cuda-runtime")]
        {
            let len = column.len();
            if len >= GPU_THRESHOLD_ELEMENTS / 4 {
                if let Ok(()) = gpu_bit_reverse_secure_column(column) {
                    return;
                }
                tracing::warn!("GPU bit_reverse (secure) failed, using CPU");
            }
        }
        
        use crate::prover::backend::simd::SimdBackend;
        <SimdBackend as ColumnOps<SecureField>>::bit_reverse_column(column);
    }
}

impl ColumnOps<FieldElement252> for GpuBackend {
    type Column = Vec<FieldElement252>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!("GPU bit_reverse for FieldElement252 not needed for Merkle")
    }
}

impl ColumnOps<Blake2sHash> for GpuBackend {
    type Column = Vec<Blake2sHash>;
    
    fn bit_reverse_column(_column: &mut Self::Column) {
        // Hash columns don't need bit reversal in typical STARK usage
        // If needed in the future, implement here
        unimplemented!("bit_reverse_column not implemented for Blake2sHash columns")
    }
}

// =============================================================================
// GPU Bit Reversal Implementation
// =============================================================================

/// Perform bit reversal on a BaseColumn using GPU.
#[cfg(feature = "cuda-runtime")]
fn gpu_bit_reverse_base_column(column: &mut BaseColumn) -> Result<(), CudaFftError> {
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    let len = column.len();
    let log_size = len.ilog2();
    
    // Convert BaseColumn to raw u32 array
    // BaseColumn stores PackedM31, which is 16 M31 values
    let raw_data: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(
            column.data.as_mut_ptr() as *mut u32,
            len
        )
    };
    
    // Execute bit reversal on GPU
    executor.bit_reverse(raw_data, log_size)?;
    
    Ok(())
}

/// Perform bit reversal on a SecureColumn using GPU.
#[cfg(feature = "cuda-runtime")]
fn gpu_bit_reverse_secure_column(column: &mut SecureColumn) -> Result<(), CudaFftError> {
    // SecureColumn has a flat data array of PackedSecureField
    // For bit reversal, we need to treat it as raw u32 data
    let executor = get_cuda_executor().map_err(|e| e.clone())?;
    let len = column.length;
    let log_size = len.ilog2();
    
    // Convert to raw u32 array (SecureField = 4 M31 values)
    let raw_data: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(
            column.data.as_mut_ptr() as *mut u32,
            len * 4  // 4 M31 values per SecureField
        )
    };
    
    // Execute bit reversal on GPU
    executor.bit_reverse(raw_data, log_size)?;
    
    Ok(())
}

// =============================================================================
// GPU Column Transfer Utilities
// =============================================================================

/// Transfer a BaseColumn to GPU memory.
#[cfg(feature = "cuda-runtime")]
pub fn base_column_to_gpu(column: &BaseColumn) -> Result<GpuBuffer<u32>, CudaFftError> {
    let raw_data: &[u32] = unsafe {
        std::slice::from_raw_parts(
            column.data.as_ptr() as *const u32,
            column.len()
        )
    };
    GpuBuffer::from_cpu(raw_data)
}

/// Transfer GPU data back to a BaseColumn.
#[cfg(feature = "cuda-runtime")]
pub fn gpu_to_base_column(buffer: &GpuBuffer<u32>, column: &mut BaseColumn) -> Result<(), CudaFftError> {
    let raw_data: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(
            column.data.as_mut_ptr() as *mut u32,
            column.len()
        )
    };
    buffer.to_cpu_into(raw_data)
}

/// Check if a column is large enough to benefit from GPU acceleration.
pub fn should_use_gpu(len: usize) -> bool {
    #[cfg(feature = "cuda-runtime")]
    {
        len >= GPU_THRESHOLD_ELEMENTS && super::cuda_executor::is_cuda_available()
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        let _ = len;
        false
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::backend::Column;
    
    #[test]
    fn test_base_column_zeros() {
        let col = <GpuBackend as ColumnOps<BaseField>>::Column::zeros(1024);
        assert_eq!(col.len(), 1024);
    }
    
    #[test]
    fn test_should_use_gpu_threshold() {
        // Without CUDA runtime, should always be false
        #[cfg(not(feature = "cuda-runtime"))]
        {
            assert!(!should_use_gpu(1_000_000));
        }
        
        // With CUDA runtime, depends on availability
        #[cfg(feature = "cuda-runtime")]
        {
            // Small columns should not use GPU
            assert!(!should_use_gpu(1000));
        }
    }
    
    #[test]
    fn test_bit_reverse_small_column() {
        // Small columns should work (uses CPU path)
        let mut col = BaseColumn::zeros(256);
        <GpuBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut col);
        // Should not panic
    }
}
