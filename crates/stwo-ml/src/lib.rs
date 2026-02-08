//! # stwo-ml: ML Inference Proving on Circle STARKs
//!
//! ML-specific proving circuits built on STWO — the fastest STARK prover in the world.
//!
//! ## Feature Flags
//!
//! - `std` (default): Standard library support + STWO prover.
//! - `gpu`: Enable GPU acceleration (kernel source, no runtime).
//! - `cuda-runtime`: Full CUDA execution (requires CUDA toolkit).
//! - `multi-gpu`: Multi-GPU distributed proving.
//! - `tee`: TEE attestation (requires `cuda-runtime`).
//! - `onnx`: ONNX model loading via tract-onnx.
//! - `safetensors`: SafeTensors weight loading.
//! - `model-loading`: Both ONNX + SafeTensors.

#![feature(portable_simd)]

pub mod components;
pub mod compiler;
pub mod gadgets;
pub mod backend;
pub mod gpu;
pub mod aggregation;
pub mod starknet;
pub mod tee;

/// Re-export core STWO types used throughout stwo-ml.
pub mod prelude {
    pub use stwo::core::fields::m31::M31;
    pub use stwo::core::fields::qm31::QM31;
    pub use stwo::core::pcs::PcsConfig;
    pub use stwo::prover::backend::simd::SimdBackend;
    pub use stwo::prover::backend::BackendForChannel;
    pub use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;

    pub use crate::components::matmul::M31Matrix;
    pub use crate::components::activation::ActivationType;
    pub use crate::compiler::graph::{ComputationGraph, GraphBuilder, GraphWeights};
    pub use crate::backend::BackendInfo;
    pub use crate::gpu::GpuModelProver;
}

/// Re-export GPU backend when available.
#[cfg(feature = "gpu")]
pub use stwo::prover::backend::gpu::GpuBackend;
