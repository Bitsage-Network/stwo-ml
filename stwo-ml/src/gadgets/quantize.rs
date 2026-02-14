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

use stwo::core::fields::m31::M31;

/// Quantization strategy for converting floating-point to M31.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantStrategy {
    /// Direct clamping: clamp to [0, P-1] and convert.
    Direct,
    /// INT8 symmetric: scale to [-127, 127], shift to [0, 254].
    Symmetric8,
    /// INT8 asymmetric: scale with zero-point to [0, 255].
    Asymmetric8,
    /// INT4 symmetric: scale to [-7, 7], shift to [0, 14]. 4 bits.
    Symmetric4,
    /// INT4 asymmetric: scale with zero-point to [0, 15]. 4 bits.
    Asymmetric4,
}

/// Quantization parameters for a tensor.
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub strategy: QuantStrategy,
    /// Scale factor: real_value = (quantized - zero_point) * scale.
    pub scale: f64,
    /// Zero point offset.
    pub zero_point: i32,
    /// Number of bits.
    pub bits: u32,
}

impl QuantParams {
    /// Compute quantization parameters from a tensor's min/max values.
    pub fn from_range(min_val: f64, max_val: f64, strategy: QuantStrategy) -> Self {
        match strategy {
            QuantStrategy::Direct => {
                let p = (1u64 << 31) - 1;
                Self {
                    strategy,
                    scale: (max_val - min_val) / p as f64,
                    zero_point: 0,
                    bits: 31,
                }
            }
            QuantStrategy::Symmetric8 => {
                let abs_max = max_val.abs().max(min_val.abs());
                let scale = abs_max / 127.0;
                Self {
                    strategy,
                    scale,
                    zero_point: 127, // shift so that 0 maps to 127
                    bits: 8,
                }
            }
            QuantStrategy::Asymmetric8 => {
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale).round() as i32;
                Self {
                    strategy,
                    scale,
                    zero_point: zero_point.clamp(0, 255),
                    bits: 8,
                }
            }
            QuantStrategy::Symmetric4 => {
                let abs_max = max_val.abs().max(min_val.abs());
                let scale = abs_max / 7.0;
                Self {
                    strategy,
                    scale,
                    zero_point: 7, // shift so that 0 maps to 7
                    bits: 4,
                }
            }
            QuantStrategy::Asymmetric4 => {
                let scale = (max_val - min_val) / 15.0;
                let zero_point = (-min_val / scale).round() as i32;
                Self {
                    strategy,
                    scale,
                    zero_point: zero_point.clamp(0, 15),
                    bits: 4,
                }
            }
        }
    }
}

/// Quantize a single f32 value to M31 using the given parameters.
pub fn quantize_value(value: f32, params: &QuantParams) -> M31 {
    let q = (value as f64 / params.scale).round() as i64 + params.zero_point as i64;
    let p = (1u32 << 31) - 1;
    let clamped = q.rem_euclid(p as i64) as u32;
    M31::from(clamped)
}

/// Dequantize an M31 value back to f32.
pub fn dequantize_value(m31_val: M31, params: &QuantParams) -> f32 {
    let q = m31_val.0 as i64 - params.zero_point as i64;
    (q as f64 * params.scale) as f32
}

/// Quantize an entire tensor (flat f32 array) to Vec<M31>.
pub fn quantize_tensor(data: &[f32], strategy: QuantStrategy) -> (Vec<M31>, QuantParams) {
    if data.is_empty() {
        return (
            vec![],
            QuantParams {
                strategy,
                scale: 1.0,
                zero_point: 0,
                bits: 8,
            },
        );
    }

    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;

    let params = QuantParams::from_range(min_val, max_val, strategy);
    let quantized: Vec<M31> = data.iter().map(|&v| quantize_value(v, &params)).collect();

    (quantized, params)
}

/// Quantize a tensor using pre-computed quantization parameters.
///
/// Unlike [`quantize_tensor`], this does NOT scan the data for min/max.
/// Use this when quantizing tiles with a globally-consistent scale/zero_point
/// computed from the full weight matrix.
pub fn quantize_tensor_with_params(data: &[f32], params: &QuantParams) -> Vec<M31> {
    data.iter().map(|&v| quantize_value(v, params)).collect()
}

/// Dequantize an entire tensor back to Vec<f32>.
pub fn dequantize_tensor(data: &[M31], params: &QuantParams) -> Vec<f32> {
    data.iter().map(|&v| dequantize_value(v, params)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric8_roundtrip() {
        let values = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let (quantized, params) = quantize_tensor(&values, QuantStrategy::Symmetric8);
        let recovered = dequantize_tensor(&quantized, &params);

        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let error = (orig - recov).abs();
            assert!(
                error < 0.02,
                "Roundtrip error too large: {orig} -> {recov} (error: {error})"
            );
        }
    }

    #[test]
    fn test_asymmetric8_roundtrip() {
        let values = vec![0.0f32, 1.0, 2.0, 3.0];
        let (quantized, params) = quantize_tensor(&values, QuantStrategy::Asymmetric8);
        let recovered = dequantize_tensor(&quantized, &params);

        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let error = (orig - recov).abs();
            assert!(
                error < 0.05,
                "Roundtrip error too large: {orig} -> {recov} (error: {error})"
            );
        }
    }

    #[test]
    fn test_direct_quantization() {
        let val = quantize_value(42.0, &QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 31,
        });
        assert_eq!(val, M31::from(42));
    }

    #[test]
    fn test_empty_tensor() {
        let (q, _) = quantize_tensor(&[], QuantStrategy::Symmetric8);
        assert!(q.is_empty());
    }

    #[test]
    fn test_symmetric4_roundtrip() {
        let values = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let (quantized, params) = quantize_tensor(&values, QuantStrategy::Symmetric4);
        assert_eq!(params.bits, 4);
        let recovered = dequantize_tensor(&quantized, &params);

        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let error = (orig - recov).abs();
            assert!(
                error < 0.2,
                "Symmetric4 roundtrip error too large: {orig} -> {recov} (error: {error})"
            );
        }

        // All quantized values should be in [0, 14]
        for q in &quantized {
            assert!(q.0 <= 14, "Symmetric4 value out of range: {}", q.0);
        }
    }

    #[test]
    fn test_asymmetric4_roundtrip() {
        let values = vec![0.0f32, 1.0, 2.0, 3.0];
        let (quantized, params) = quantize_tensor(&values, QuantStrategy::Asymmetric4);
        assert_eq!(params.bits, 4);
        let recovered = dequantize_tensor(&quantized, &params);

        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let error = (orig - recov).abs();
            assert!(
                error < 0.25,
                "Asymmetric4 roundtrip error too large: {orig} -> {recov} (error: {error})"
            );
        }

        // All quantized values should be in [0, 15]
        for q in &quantized {
            assert!(q.0 <= 15, "Asymmetric4 value out of range: {}", q.0);
        }
    }

    #[test]
    fn test_symmetric4_params() {
        let params = QuantParams::from_range(-1.0, 1.0, QuantStrategy::Symmetric4);
        assert_eq!(params.bits, 4);
        assert_eq!(params.zero_point, 7);
        assert!((params.scale - 1.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric4_params() {
        let params = QuantParams::from_range(0.0, 3.0, QuantStrategy::Asymmetric4);
        assert_eq!(params.bits, 4);
        assert_eq!(params.zero_point, 0); // min=0 => zp=0
        assert!((params.scale - 0.2).abs() < 1e-10); // 3.0/15 = 0.2
    }
}
