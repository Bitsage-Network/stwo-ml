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

use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;

use super::range_check::{
    prove_range_check, verify_range_check, RangeCheckComponent, RangeCheckError,
};

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

// ---------------------------------------------------------------------------
// Quantize + Range Check integration
// ---------------------------------------------------------------------------

/// Error type for quantized range check operations.
#[derive(Debug, thiserror::Error)]
pub enum QuantizeError {
    #[error("range check error: {0}")]
    RangeCheck(#[from] RangeCheckError),
    #[error("quantized value {value} out of range [0, {max})")]
    ValueOutOfRange { value: u32, max: u32 },
}

/// Quantize a vector of floats and prove all quantized values are in range.
///
/// Combines `QuantizeParams::quantize_vec` with `prove_range_check` to produce
/// a STWO STARK proof that all quantized values are in `[0, 2^bits)`.
///
/// Returns the quantized values and the range check proof.
pub fn prove_quantized_range(
    values: &[f32],
    params: &QuantizeParams,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
) -> Result<
    (
        Vec<M31>,
        RangeCheckComponent,
        stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    ),
    QuantizeError,
> {
    let quantized = params.quantize_vec(values);

    // Verify all values are in range before proving.
    let max = 1u32 << params.bits;
    for &v in &quantized {
        if v.0 >= max {
            return Err(QuantizeError::ValueOutOfRange {
                value: v.0,
                max,
            });
        }
    }

    // Use the range check STARK to prove all values are in [0, 2^bits).
    // The range check table needs log_size >= bits. We use the bits value
    // directly, clamped to minimum SIMD lane size.
    let log_range = params.bits.max(stwo::prover::backend::simd::m31::LOG_N_LANES);

    let (component, proof) = prove_range_check(&quantized, log_range, config, channel)?;

    Ok((quantized, component, proof))
}

/// Verify a quantized range check proof.
pub fn verify_quantized_range(
    component: &RangeCheckComponent,
    proof: &stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    channel: &mut Blake2sChannel,
) -> Result<(), QuantizeError> {
    verify_range_check(component, proof, channel)?;
    Ok(())
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

    // -------------------------------------------------------------------
    // Quantize + Range Check STARK integration tests
    // -------------------------------------------------------------------

    #[test]
    fn test_prove_verify_quantized_range() {
        use stwo::core::channel::Blake2sChannel;
        use stwo::core::pcs::PcsConfig;

        let params = QuantizeParams::int8_symmetric(0.1);
        let values = vec![0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.3];

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (quantized, component, proof) =
            prove_quantized_range(&values, &params, config, &mut prover_channel).unwrap();

        // Verify all quantized values are in [0, 256)
        for &v in &quantized {
            assert!(v.0 < 256, "quantized value {} out of INT8 range", v.0);
        }

        let mut verifier_channel = Blake2sChannel::default();
        verify_quantized_range(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_quantized_range_batch() {
        use stwo::core::channel::Blake2sChannel;
        use stwo::core::pcs::PcsConfig;

        let params = QuantizeParams::int8_symmetric(0.05);
        // Generate a batch of 64 values in [-5, 5]
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.15).collect();

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (quantized, component, proof) =
            prove_quantized_range(&values, &params, config, &mut prover_channel).unwrap();

        assert_eq!(quantized.len(), 64);

        let mut verifier_channel = Blake2sChannel::default();
        verify_quantized_range(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_quantized_range_preserves_values() {
        use stwo::core::channel::Blake2sChannel;
        use stwo::core::pcs::PcsConfig;

        let params = QuantizeParams::int8_symmetric(0.1);
        let values = vec![0.0, 0.5];

        let config = PcsConfig::default();
        let mut channel = Blake2sChannel::default();
        let (quantized, _, _) =
            prove_quantized_range(&values, &params, config, &mut channel).unwrap();

        // 0.0 → 128 (zero_point)
        assert_eq!(quantized[0], M31::from(128));
        // 0.5 → 133 (128 + round(0.5/0.1))
        assert_eq!(quantized[1], M31::from(133));
    }
}
