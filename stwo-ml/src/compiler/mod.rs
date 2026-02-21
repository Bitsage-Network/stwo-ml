//! Model → Circuit compiler.
//!
//! Transforms neural network definitions into STWO proving circuits
//! by mapping each layer to the appropriate component (MatMul, Activation,
//! Attention, LayerNorm).
//!
//! # Compilation Pipeline
//!
//! ```text
//! ONNX Model (.onnx)
//!     │
//!     ▼
//! Graph Parser (onnx.rs)
//!     │  Extracts: layers, shapes, weights, activations
//!     ▼
//! Computation Graph (graph.rs)
//!     │  Assigns: components per layer, trace budget
//!     ▼
//! STWO Circuit
//!     │  Components: MatMul + Activation + Attention + LayerNorm
//!     ▼
//! Proof Generation (prove.rs)
//! ```

pub mod checkpoint;
pub mod chunked;
pub mod dual;
pub mod graph;
#[cfg(any(feature = "cli", feature = "model-loading"))]
pub mod hf_loader;
pub mod inspect;
pub mod onnx;
pub mod prove;
pub mod quantize_weights;
#[cfg(feature = "safetensors")]
pub mod safetensors;
#[cfg(feature = "safetensors")]
pub mod streaming;
