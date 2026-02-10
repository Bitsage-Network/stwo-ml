//! Conv2D verification via im2col + MatMul.
//!
//! Convolution is reduced to matrix multiplication by rearranging the input
//! tensor using the im2col transform. The resulting matmul is proven using
//! the existing sumcheck infrastructure.

use stwo::core::fields::m31::M31;
use crate::components::matmul::M31Matrix;

/// im2col configuration.
#[derive(Debug, Clone, Copy)]
pub struct Im2ColConfig {
    pub in_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    /// Input spatial height.
    pub input_h: usize,
    /// Input spatial width.
    pub input_w: usize,
}

impl Im2ColConfig {
    pub fn output_h(&self) -> usize {
        (self.input_h + 2 * self.padding - self.kernel_size) / self.stride + 1
    }

    pub fn output_w(&self) -> usize {
        (self.input_w + 2 * self.padding - self.kernel_size) / self.stride + 1
    }

    /// Number of columns in the im2col matrix (patch size).
    pub fn patch_size(&self) -> usize {
        self.in_channels * self.kernel_size * self.kernel_size
    }

    /// Number of rows in the im2col matrix (number of patches).
    pub fn num_patches(&self) -> usize {
        self.output_h() * self.output_w()
    }
}

/// Perform im2col transform: rearrange input into a matrix where each row
/// is a flattened receptive field patch.
///
/// Input shape: (in_channels, input_h, input_w) stored row-major.
/// Output shape: (num_patches, patch_size) as M31Matrix.
pub fn im2col(input: &[M31], config: &Im2ColConfig) -> M31Matrix {
    let oh = config.output_h();
    let ow = config.output_w();
    let patch_size = config.patch_size();
    let num_patches = oh * ow;

    let mut result = M31Matrix::new(num_patches, patch_size);

    for out_y in 0..oh {
        for out_x in 0..ow {
            let patch_row = out_y * ow + out_x;
            let mut col = 0;

            for c in 0..config.in_channels {
                for ky in 0..config.kernel_size {
                    for kx in 0..config.kernel_size {
                        let iy = out_y * config.stride + ky;
                        let ix = out_x * config.stride + kx;

                        let val = if iy >= config.padding
                            && iy < config.input_h + config.padding
                            && ix >= config.padding
                            && ix < config.input_w + config.padding
                        {
                            let real_y = iy - config.padding;
                            let real_x = ix - config.padding;
                            let idx = c * config.input_h * config.input_w
                                + real_y * config.input_w
                                + real_x;
                            if idx < input.len() {
                                input[idx]
                            } else {
                                M31::from(0)
                            }
                        } else {
                            M31::from(0) // padding
                        };

                        result.set(patch_row, col, val);
                        col += 1;
                    }
                }
            }
        }
    }

    result
}

/// Reshape kernel weights into the im2col-compatible format.
///
/// Kernel shape: (out_channels, in_channels, kH, kW) stored row-major.
/// Output: (patch_size, out_channels) M31Matrix for matmul with im2col output.
pub fn reshape_kernel(
    kernel: &[M31],
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
) -> M31Matrix {
    let patch_size = in_channels * kernel_size * kernel_size;
    let mut result = M31Matrix::new(patch_size, out_channels);

    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let row = ic * kernel_size * kernel_size + ky * kernel_size + kx;
                    let src_idx = oc * in_channels * kernel_size * kernel_size
                        + ic * kernel_size * kernel_size
                        + ky * kernel_size
                        + kx;
                    if src_idx < kernel.len() {
                        result.set(row, oc, kernel[src_idx]);
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im2col_config() {
        let config = Im2ColConfig {
            in_channels: 3,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_h: 4,
            input_w: 4,
        };
        assert_eq!(config.output_h(), 4);
        assert_eq!(config.output_w(), 4);
        assert_eq!(config.patch_size(), 27);
        assert_eq!(config.num_patches(), 16);
    }

    #[test]
    fn test_im2col_no_padding() {
        let config = Im2ColConfig {
            in_channels: 1,
            kernel_size: 2,
            stride: 1,
            padding: 0,
            input_h: 3,
            input_w: 3,
        };
        // 3×3 input, 2×2 kernel, stride 1 → 2×2 output = 4 patches
        assert_eq!(config.num_patches(), 4);
        assert_eq!(config.patch_size(), 4);

        let input: Vec<M31> = (1..=9).map(|i| M31::from(i as u32)).collect();
        let result = im2col(&input, &config);
        assert_eq!(result.rows, 4);
        assert_eq!(result.cols, 4);

        // First patch: top-left 2×2 = [1, 2, 4, 5]
        assert_eq!(result.get(0, 0), M31::from(1));
        assert_eq!(result.get(0, 1), M31::from(2));
        assert_eq!(result.get(0, 2), M31::from(4));
        assert_eq!(result.get(0, 3), M31::from(5));
    }

    #[test]
    fn test_reshape_kernel() {
        // 2 output channels, 1 input channel, 2×2 kernel
        let kernel: Vec<M31> = (1..=8).map(|i| M31::from(i as u32)).collect();
        let result = reshape_kernel(&kernel, 2, 1, 2);
        assert_eq!(result.rows, 4); // patch_size = 1*2*2 = 4
        assert_eq!(result.cols, 2); // out_channels
    }
}
