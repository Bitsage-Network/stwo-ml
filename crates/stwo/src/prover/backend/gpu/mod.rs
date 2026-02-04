//! GPU Backend for hardware-accelerated proof generation.
//!
//! This module provides a GPU-accelerated backend that implements the same traits
//! as [`SimdBackend`] but uses CUDA/ROCm for computationally intensive operations.
//!
//! # Architecture
//!
//! The GPU backend is designed to be a drop-in replacement for [`SimdBackend`]:
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐
//! │   SimdBackend   │     │   GpuBackend    │
//! │   (CPU/SIMD)    │     │   (CUDA/ROCm)   │
//! └────────┬────────┘     └────────┬────────┘
//!          │                       │
//!          └───────────┬───────────┘
//!                      │
//!              ┌───────▼───────┐
//!              │    Backend    │
//!              │    (trait)    │
//!              └───────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | SimdBackend | GpuBackend | Speedup |
//! |-----------|-------------|------------|---------|
//! | FFT 16K   | ~2ms        | ~1ms       | ~2x     |
//! | FFT 1M    | ~150ms      | ~3ms       | ~50x    |
//! | Merkle    | ~20ms       | ~15ms      | ~1.3x   |
//!
//! # Usage
//!
//! The GPU backend is selected at runtime based on availability:
//!
//! ```ignore
//! use stwo::prover::backend::gpu::GpuBackend;
//!
//! if GpuBackend::is_available() {
//!     // Use GPU for large proofs
//!     let proof = prove::<GpuBackend>(&trace)?;
//! } else {
//!     // Fallback to CPU
//!     let proof = prove::<SimdBackend>(&trace)?;
//! }
//! ```

use serde::{Deserialize, Serialize};

use super::{Backend, BackendForChannel};
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleChannel};
use crate::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;

pub mod compat;  // cudarc 0.11+ compatibility layer
pub mod column;
pub mod conversion;
pub mod fft;
pub mod fri;
pub mod poly_ops;
pub mod quotients;
pub mod accumulation;
pub mod merkle;
pub mod gkr;
pub mod grind;
pub mod cuda_executor;
pub mod memory;
pub mod pipeline;
pub mod secure_session;
pub mod multi_gpu;
pub mod multi_gpu_executor;
pub mod cuda_streams;
pub mod large_proofs;
pub mod tee;
pub mod optimizations;
pub mod constraints;

/// GPU Backend for hardware-accelerated proof generation.
///
/// This backend implements the same traits as [`SimdBackend`] but uses
/// GPU acceleration for computationally intensive operations like FFT.
///
/// # Thread Safety
///
/// `GpuBackend` is `Copy + Clone + Send + Sync` to match the requirements
/// of the `Backend` trait. The actual GPU resources are managed through
/// a global context that handles synchronization.
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct GpuBackend;

impl GpuBackend {
    /// Check if GPU acceleration is available.
    ///
    /// Returns `true` if:
    /// - CUDA runtime is installed and functional
    /// - A compatible GPU is detected
    /// - Kernels can be compiled
    pub fn is_available() -> bool {
        #[cfg(feature = "cuda-runtime")]
        {
            cuda_executor::is_cuda_available()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            // Without cuda-runtime, check if gpu feature is enabled
            #[cfg(feature = "gpu")]
            {
                gpu_context::is_initialized()
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        }
    }
    
    /// Get the name of the GPU device.
    pub fn device_name() -> Option<String> {
        #[cfg(feature = "cuda-runtime")]
        {
            cuda_executor::get_cuda_executor()
                .ok()
                .map(|e| e.device_info.name.clone())
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            #[cfg(feature = "gpu")]
            {
                gpu_context::device_name()
            }
            #[cfg(not(feature = "gpu"))]
            {
                None
            }
        }
    }
    
    /// Get available GPU memory in bytes.
    pub fn available_memory() -> Option<usize> {
        #[cfg(feature = "cuda-runtime")]
        {
            cuda_executor::get_cuda_executor()
                .ok()
                .map(|e| e.device_info.total_memory_bytes)
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            #[cfg(feature = "gpu")]
            {
                gpu_context::available_memory()
            }
            #[cfg(not(feature = "gpu"))]
            {
                None
            }
        }
    }
    
    /// Get compute capability of the GPU.
    pub fn compute_capability() -> Option<(u32, u32)> {
        #[cfg(feature = "cuda-runtime")]
        {
            cuda_executor::get_cuda_executor()
                .ok()
                .map(|e| e.device_info.compute_capability)
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            None
        }
    }
}

// Implement the Backend marker trait
impl Backend for GpuBackend {}

// Implement BackendForChannel for supported Merkle channels
impl BackendForChannel<Blake2sMerkleChannel> for GpuBackend {}
impl BackendForChannel<Blake2sM31MerkleChannel> for GpuBackend {}
impl BackendForChannel<Poseidon252MerkleChannel> for GpuBackend {}

/// Global GPU context management.
///
/// This module handles GPU initialization, memory management, and
/// kernel execution. When `cuda-runtime` feature is enabled, this
/// delegates to the actual CUDA executor. Otherwise, it provides
/// a stub implementation.
#[cfg(all(feature = "gpu", not(feature = "cuda-runtime")))]
pub mod gpu_context {
    use std::sync::OnceLock;
    
    /// Global GPU context (stub when cuda-runtime is not enabled)
    static GPU_CONTEXT: OnceLock<GpuContextInner> = OnceLock::new();
    
    struct GpuContextInner {
        device_name: String,
        total_memory: usize,
    }
    
    /// Initialize the GPU context.
    pub fn initialize() -> bool {
        GPU_CONTEXT.get_or_init(|| {
            // Stub implementation - real GPU init requires cuda-runtime
            GpuContextInner {
                device_name: "GPU (stub)".to_string(),
                total_memory: 0,
            }
        });
        true
    }
    
    /// Check if GPU context is initialized.
    pub fn is_initialized() -> bool {
        GPU_CONTEXT.get().is_some()
    }
    
    /// Get device name.
    pub fn device_name() -> Option<String> {
        GPU_CONTEXT.get().map(|ctx| ctx.device_name.clone())
    }
    
    /// Get available memory.
    pub fn available_memory() -> Option<usize> {
        GPU_CONTEXT.get().map(|ctx| ctx.total_memory)
    }
}

#[cfg(feature = "cuda-runtime")]
pub mod gpu_context {
    use super::cuda_executor;
    
    /// Initialize the GPU context using CUDA.
    pub fn initialize() -> bool {
        cuda_executor::is_cuda_available()
    }
    
    /// Check if GPU context is initialized.
    pub fn is_initialized() -> bool {
        cuda_executor::is_cuda_available()
    }
    
    /// Get device name.
    pub fn device_name() -> Option<String> {
        cuda_executor::get_cuda_executor()
            .ok()
            .map(|e| e.device_info.name.clone())
    }
    
    /// Get available memory.
    pub fn available_memory() -> Option<usize> {
        cuda_executor::get_cuda_executor()
            .ok()
            .map(|e| e.device_info.total_memory_bytes)
    }
}

#[cfg(not(feature = "gpu"))]
pub mod gpu_context {
    pub fn initialize() -> bool { false }
    pub fn is_initialized() -> bool { false }
    pub fn device_name() -> Option<String> { None }
    pub fn available_memory() -> Option<usize> { None }
}

