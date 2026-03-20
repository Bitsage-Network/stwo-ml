//! Metal GPU acceleration for Apple Silicon.
//!
//! Provides GPU-accelerated M31 matrix multiplication and Poseidon hashing
//! using Apple's Metal API via the `wgpu` crate. This enables proving on
//! M1/M2/M3/M4 Macs without CUDA.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │                    stwo-ml prover                    │
//! ├──────────────────────────────────────────────────────┤
//! │  feature = "cuda-runtime"  │  feature = "metal"      │
//! │  ┌──────────────────────┐  │  ┌───────────────────┐  │
//! │  │ CUDA kernels (.cu)   │  │  │ wgpu compute      │  │
//! │  │ cuBLAS, cuSPARSE     │  │  │ WGSL shaders      │  │
//! │  └──────────────────────┘  │  └───────────────────┘  │
//! │           ↓                │          ↓               │
//! │      NVIDIA GPU            │     Metal 3 GPU          │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! # Compute Shaders (WGSL)
//!
//! The hot kernels are:
//! - `matmul_m31.wgsl`: M31 matrix multiply (forward pass + GKR)
//! - `poseidon_merkle.wgsl`: Poseidon2-M31 Merkle tree building
//!
//! M31 arithmetic in WGSL:
//! - M31 = u32 with modular reduction: `(a + b) % P` where P = 2^31 - 1
//! - Multiply: `((a as u64) * (b as u64)) % P` — needs 64-bit intermediate
//! - WGSL supports u32 natively; u64 multiply via two u32 multiplies
//!
//! # Usage
//!
//! ```bash
//! cargo build --release --features std,metal,cli,model-loading,safetensors
//! ./target/release/prove-model --model-dir ./qwen3-14b --layers 1 --gkr --format ml_gkr
//! ```

pub mod device;
pub mod dispatch;
pub mod matmul;
pub mod sumcheck;

pub use dispatch::matmul_m31_auto;
