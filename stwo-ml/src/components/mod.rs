//! ML-specific AIR components for STWO.
//!
//! Each component defines constraints that verify a specific ML operation
//! (matrix multiplication, activation functions, attention) using STWO's
//! constraint framework with LogUp lookups and sumcheck verification.

pub mod matmul;
pub mod activation;
pub mod attention;
pub mod layernorm;
pub mod rmsnorm;
pub mod rope;
pub mod elementwise;
pub mod embedding;
pub mod conv2d;
pub mod quantize;
pub mod dequantize;
pub mod f32_ops;
pub mod tiled_matmul;
pub mod range_check;
pub mod poseidon2_air;
