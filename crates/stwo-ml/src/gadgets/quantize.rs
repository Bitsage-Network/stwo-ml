//! Quantization gadgets for mapping floating-point model weights to M31.
//!
//! Maps INT8/FP8/FP16 weights to M31 field elements while preserving
//! the mathematical relationships needed for correct inference.
//!
//! # Quantization Scheme
//!
//! ```text
//! FP32 weight → scale + zero_point → INT8 → M31
//!
//! q = round(w / scale) + zero_point
//! M31_val = q mod (2^31 - 1)
//!
//! For INT8: q ∈ [-128, 127] → M31 range is [0, 255] (shifted)
//! For FP16: mantissa ∈ [0, 1023] → direct M31 embedding
//! ```

// TODO: Phase 1 implementation
