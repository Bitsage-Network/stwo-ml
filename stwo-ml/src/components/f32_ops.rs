//! Float32 computation types and operations for dual-track inference.
//!
//! Provides an `F32Matrix` type with matmul, activations, and layernorm
//! that mirrors the M31 pipeline but operates on real floating-point values.
//! Used alongside the existing M31 proving pipeline to produce meaningful
//! inference outputs while simultaneously generating STARK proofs.

use rayon::prelude::*;

use crate::components::activation::ActivationType;
use crate::components::matmul::M31Matrix;
use crate::gadgets::quantize::{
    QuantParams, QuantStrategy, dequantize_value, quantize_tensor,
};

/// Flat matrix stored in row-major order over f32.
#[derive(Debug, Clone)]
pub struct F32Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl F32Matrix {
    /// Create a zero-initialized matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create from existing data.
    pub fn from_data(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "data length {} != rows*cols {}",
            data.len(),
            rows * cols
        );
        Self { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, val: f32) {
        self.data[i * self.cols + j] = val;
    }

    /// Dequantize an M31Matrix into f32 using the given quantization parameters.
    pub fn from_m31(m31: &M31Matrix, params: &QuantParams) -> Self {
        let data: Vec<f32> = m31.data.iter().map(|&v| dequantize_value(v, params)).collect();
        Self {
            rows: m31.rows,
            cols: m31.cols,
            data,
        }
    }

    /// Quantize this f32 matrix into M31 using the given strategy.
    /// Returns the M31Matrix and the QuantParams used.
    pub fn to_m31(&self, strategy: QuantStrategy) -> (M31Matrix, QuantParams) {
        let (quantized, params) = quantize_tensor(&self.data, strategy);
        let matrix = M31Matrix {
            rows: self.rows,
            cols: self.cols,
            data: quantized,
        };
        (matrix, params)
    }
}

// ===== Matrix Multiplication =====

/// Float32 matrix multiplication: C = A × B.
///
/// Uses rayon parallelism for matrices with 64+ rows (same threshold as matmul_m31).
pub fn matmul_f32(a: &F32Matrix, b: &F32Matrix) -> F32Matrix {
    assert_eq!(a.cols, b.rows, "A.cols ({}) must equal B.rows ({})", a.cols, b.rows);
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    if m < 64 {
        // Small matrices: simple loop
        let mut c = F32Matrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                c.data[i * n + j] = sum;
            }
        }
        c
    } else {
        // Large matrices: rayon parallel over rows
        let data: Vec<f32> = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                let mut row = vec![0.0f32; n];
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += a.data[i * k + l] * b.data[l * n + j];
                    }
                    row[j] = sum;
                }
                row
            })
            .collect();
        F32Matrix { rows: m, cols: n, data }
    }
}

// ===== Activation Functions =====

/// ReLU: max(0, x)
pub fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

/// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_f32(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax over a row.
///
/// Subtracts max for stability, then exp + normalize.
pub fn softmax_f32(row: &[f32]) -> Vec<f32> {
    if row.is_empty() {
        return vec![];
    }
    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum == 0.0 {
        return exp_vals;
    }
    exp_vals.iter().map(|&v| v / sum).collect()
}

/// Apply an activation function element-wise to a matrix.
pub fn apply_activation_f32(matrix: &F32Matrix, act_type: ActivationType) -> F32Matrix {
    let f: fn(f32) -> f32 = match act_type {
        ActivationType::ReLU => relu_f32,
        ActivationType::GELU => gelu_f32,
        ActivationType::Sigmoid => sigmoid_f32,
        ActivationType::Softmax => {
            // For softmax, apply row-wise
            let mut result = F32Matrix::new(matrix.rows, matrix.cols);
            for i in 0..matrix.rows {
                let row: Vec<f32> = (0..matrix.cols).map(|j| matrix.get(i, j)).collect();
                let soft = softmax_f32(&row);
                for (j, &v) in soft.iter().enumerate() {
                    result.set(i, j, v);
                }
            }
            return result;
        }
    };
    let data: Vec<f32> = matrix.data.iter().map(|&x| f(x)).collect();
    F32Matrix {
        rows: matrix.rows,
        cols: matrix.cols,
        data,
    }
}

// ===== Layer Normalization =====

/// Layer normalization over the last dimension.
///
/// y = (x - mean) / sqrt(var + eps)
///
/// No learnable gamma/beta (matching the M31 pipeline which also omits them).
pub fn layernorm_f32(matrix: &F32Matrix, dim: usize, eps: f32) -> F32Matrix {
    let mut output = F32Matrix::new(matrix.rows, matrix.cols);
    let n = dim.min(matrix.cols);

    for row in 0..matrix.rows {
        // Mean
        let mut sum = 0.0f32;
        for col in 0..n {
            sum += matrix.data[row * matrix.cols + col];
        }
        let mean = sum / n as f32;

        // Variance
        let mut var_sum = 0.0f32;
        for col in 0..n {
            let diff = matrix.data[row * matrix.cols + col] - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / n as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Normalize
        for col in 0..n {
            let val = matrix.data[row * matrix.cols + col];
            output.data[row * matrix.cols + col] = (val - mean) * inv_std;
        }
        // Columns beyond dim are passed through as-is (zero from initialization)
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gadgets::quantize::QuantStrategy;

    #[test]
    fn test_f32_matrix_basic() {
        let mut m = F32Matrix::new(2, 3);
        m.set(0, 0, 1.0);
        m.set(1, 2, 5.0);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 5.0);
        assert_eq!(m.get(0, 1), 0.0);
    }

    #[test]
    fn test_matmul_f32_small() {
        // [1 2] × [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        let a = F32Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = F32Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = matmul_f32(&a, &b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        assert!((c.get(0, 0) - 19.0).abs() < 1e-5);
        assert!((c.get(0, 1) - 22.0).abs() < 1e-5);
        assert!((c.get(1, 0) - 43.0).abs() < 1e-5);
        assert!((c.get(1, 1) - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_relu_f32() {
        assert_eq!(relu_f32(3.0), 3.0);
        assert_eq!(relu_f32(-2.0), 0.0);
        assert_eq!(relu_f32(0.0), 0.0);
    }

    #[test]
    fn test_gelu_f32() {
        // GELU(0) ≈ 0, GELU(1) ≈ 0.8413, GELU(-1) ≈ -0.1587
        assert!((gelu_f32(0.0)).abs() < 1e-5);
        assert!((gelu_f32(1.0) - 0.8413).abs() < 0.01);
        assert!((gelu_f32(-1.0) + 0.1587).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid_f32() {
        assert!((sigmoid_f32(0.0) - 0.5).abs() < 1e-5);
        assert!(sigmoid_f32(10.0) > 0.999);
        assert!(sigmoid_f32(-10.0) < 0.001);
    }

    #[test]
    fn test_softmax_f32_sums_to_one() {
        let row = vec![1.0, 2.0, 3.0, 4.0];
        let result = softmax_f32(&row);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {sum}");
        // Elements should be monotonically increasing
        for i in 0..result.len() - 1 {
            assert!(result[i] < result[i + 1], "softmax should preserve order");
        }
    }

    #[test]
    fn test_layernorm_f32_zero_mean_unit_var() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = F32Matrix::from_data(2, 4, data);
        let normed = layernorm_f32(&m, 4, 1e-5);

        for row in 0..normed.rows {
            // Check near-zero mean
            let mut sum = 0.0f32;
            for col in 0..normed.cols {
                sum += normed.get(row, col);
            }
            let mean = sum / normed.cols as f32;
            assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");

            // Check near-unit variance
            let mut var = 0.0f32;
            for col in 0..normed.cols {
                let diff = normed.get(row, col) - mean;
                var += diff * diff;
            }
            var /= normed.cols as f32;
            assert!(
                (var - 1.0).abs() < 0.05,
                "variance should be ~1, got {var}"
            );
        }
    }

    #[test]
    fn test_f32_m31_roundtrip() {
        let data = vec![0.1, 0.5, -0.3, 0.8, -0.7, 0.2];
        let f32_mat = F32Matrix::from_data(2, 3, data.clone());

        // f32 → M31 → f32
        let (m31_mat, params) = f32_mat.to_m31(QuantStrategy::Symmetric8);
        let recovered = F32Matrix::from_m31(&m31_mat, &params);

        for (orig, recov) in data.iter().zip(recovered.data.iter()) {
            let error = (orig - recov).abs();
            assert!(
                error < 0.02,
                "roundtrip error too large: {orig} -> {recov} (error: {error})"
            );
        }
    }

    #[test]
    fn test_apply_activation_f32_relu_matrix() {
        let m = F32Matrix::from_data(2, 3, vec![-1.0, 0.0, 1.0, -2.0, 3.0, -4.0]);
        let result = apply_activation_f32(&m, ActivationType::ReLU);
        assert_eq!(result.data, vec![0.0, 0.0, 1.0, 0.0, 3.0, 0.0]);
    }
}
