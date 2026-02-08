//! Weight quantization for model loading.
//!
//! Converts floating-point model weights into M31 field elements
//! using configurable quantization strategies.

use stwo::core::fields::m31::M31;

use crate::gadgets::quantize::{QuantStrategy, QuantParams, quantize_tensor};
use crate::components::matmul::M31Matrix;

/// Quantize a weight matrix from f32 to M31Matrix.
pub fn quantize_weight_matrix(
    data: &[f32],
    rows: usize,
    cols: usize,
    strategy: QuantStrategy,
) -> (M31Matrix, QuantParams) {
    assert_eq!(data.len(), rows * cols, "data length mismatch");

    let (quantized, params) = quantize_tensor(data, strategy);

    let mut matrix = M31Matrix::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            matrix.set(i, j, quantized[i * cols + j]);
        }
    }

    (matrix, params)
}

/// Quantize a bias vector from f32 to Vec<M31>.
pub fn quantize_bias_vector(
    data: &[f32],
    strategy: QuantStrategy,
) -> (Vec<M31>, QuantParams) {
    quantize_tensor(data, strategy)
}

/// Error for weight loading.
#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
    #[error("IO error: {0}")]
    IoError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_weight_matrix() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let (matrix, params) = quantize_weight_matrix(&data, 3, 4, QuantStrategy::Symmetric8);

        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 4);
        assert_eq!(params.strategy, QuantStrategy::Symmetric8);
    }

    #[test]
    fn test_quantize_bias() {
        let data = vec![0.1f32, -0.2, 0.3];
        let (quantized, _params) = quantize_bias_vector(&data, QuantStrategy::Asymmetric8);
        assert_eq!(quantized.len(), 3);
    }
}
