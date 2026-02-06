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
//! Proof Generation (via stwo-gpu backend)
//! ```

pub mod onnx;
pub mod graph;
