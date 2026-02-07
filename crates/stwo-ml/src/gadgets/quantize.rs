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
//! q = clamp(round(w / scale) + zero_point, 0, 255)
//! M31_val = q mod (2^31 - 1)
//!
//! For INT8: q ∈ [0, 255] → direct M31 embedding
//! For FP16: mantissa ∈ [0, 1023] → direct M31 embedding
//! ```
//!
//! The quantization gadget ensures all quantized values are within
//! the expected range using the range check gadget.

use stwo::core::fields::m31::M31;

/// Quantization parameters for mapping float values to field elements.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizeParams {
    /// Scale factor: real_value ≈ (quantized - zero_point) * scale.
    pub scale: f32,
    /// Zero point offset (maps 0.0 in real space to this quantized value).
    pub zero_point: u32,
    /// Number of bits for the quantized representation.
    pub bits: u32,
}

impl QuantizeParams {
    /// Standard INT8 symmetric quantization.
    pub fn int8_symmetric(scale: f32) -> Self {
        Self {
            scale,
            zero_point: 128, // Symmetric around 128
            bits: 8,
        }
    }

    /// Standard INT8 asymmetric quantization.
    ///
    /// Returns `None` if `zero_point >= 256`.
    pub fn int8_asymmetric(scale: f32, zero_point: u32) -> Option<Self> {
        if zero_point >= 256 {
            return None;
        }
        Some(Self {
            scale,
            zero_point,
            bits: 8,
        })
    }

    /// Maximum quantized value.
    pub fn max_value(&self) -> u32 {
        (1u32 << self.bits) - 1
    }

    /// Quantize a single float value to M31.
    pub fn quantize(&self, value: f32) -> M31 {
        let q = (value / self.scale).round() as i64 + self.zero_point as i64;
        let clamped = q.clamp(0, self.max_value() as i64) as u32;
        M31::from(clamped)
    }

    /// Dequantize an M31 value back to float (approximate).
    pub fn dequantize(&self, value: M31) -> f32 {
        (value.0 as f32 - self.zero_point as f32) * self.scale
    }

    /// Quantize a vector of floats to M31 values.
    pub fn quantize_vec(&self, values: &[f32]) -> Vec<M31> {
        values.iter().map(|&v| self.quantize(v)).collect()
    }
}

/// Verify that all quantized values are within the valid range.
///
/// Returns true if all values are in `[0, 2^bits)`.
pub fn validate_quantized_range(values: &[M31], bits: u32) -> bool {
    let max = 1u32 << bits;
    values.iter().all(|v| v.0 < max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_symmetric_quantize() {
        let params = QuantizeParams::int8_symmetric(0.1);
        assert_eq!(params.zero_point, 128);
        assert_eq!(params.bits, 8);
        assert_eq!(params.max_value(), 255);

        // Value 0.0 maps to zero_point (128)
        assert_eq!(params.quantize(0.0), M31::from(128));

        // Positive values shift above zero_point
        // 1.0 / 0.1 = 10, + 128 = 138
        assert_eq!(params.quantize(1.0), M31::from(138));

        // Negative values shift below zero_point
        // -1.0 / 0.1 = -10, + 128 = 118
        assert_eq!(params.quantize(-1.0), M31::from(118));
    }

    #[test]
    fn test_quantize_clamp() {
        let params = QuantizeParams::int8_symmetric(0.01);

        // Very large value clamps to 255
        assert_eq!(params.quantize(100.0), M31::from(255));

        // Very negative value clamps to 0
        assert_eq!(params.quantize(-100.0), M31::from(0));
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let params = QuantizeParams::int8_symmetric(0.1);
        let original = 0.5f32;
        let quantized = params.quantize(original);
        let recovered = params.dequantize(quantized);

        // Should be approximately equal (within quantization error)
        assert!((original - recovered).abs() < params.scale);
    }

    #[test]
    fn test_quantize_vec() {
        let params = QuantizeParams::int8_symmetric(0.1);
        let values = vec![0.0, 0.5, -0.5, 1.0];
        let quantized = params.quantize_vec(&values);

        assert_eq!(quantized.len(), 4);
        assert_eq!(quantized[0], M31::from(128)); // 0.0 → 128
        assert!(quantized[1].0 > 128); // 0.5 → 133
        assert!(quantized[2].0 < 128); // -0.5 → 123
    }

    #[test]
    fn test_validate_range() {
        let values = vec![M31::from(0), M31::from(100), M31::from(255)];
        assert!(validate_quantized_range(&values, 8)); // all < 256

        let bad = vec![M31::from(0), M31::from(256)];
        assert!(!validate_quantized_range(&bad, 8)); // 256 >= 256
    }

    #[test]
    fn test_asymmetric_quantize() {
        let params = QuantizeParams::int8_asymmetric(0.05, 50).unwrap();
        assert_eq!(params.zero_point, 50);
        assert_eq!(params.quantize(0.0), M31::from(50));
    }

    #[test]
    fn test_asymmetric_rejects_bad_zero_point() {
        assert!(QuantizeParams::int8_asymmetric(0.1, 256).is_none());
        assert!(QuantizeParams::int8_asymmetric(0.1, 1000).is_none());
        assert!(QuantizeParams::int8_asymmetric(0.1, 255).is_some());
    }
}
