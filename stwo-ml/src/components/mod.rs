//! ML-specific AIR components for STWO.
//!
//! Each component defines constraints that verify a specific ML operation
//! (matrix multiplication, activation functions, attention) using STWO's
//! constraint framework with LogUp lookups and sumcheck verification.

pub mod matmul;
pub mod activation;
pub mod attention;
pub mod layernorm;
pub mod f32_ops;
