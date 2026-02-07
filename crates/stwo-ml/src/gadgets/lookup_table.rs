//! Precomputed function lookup tables for non-linear operations.
//!
//! Generates preprocessed columns containing (input, output) pairs for
//! activation functions, range checks, and other non-linear mappings.
//! Used with LogUp to prove correctness of function evaluations.

use stwo::core::fields::m31::M31;

/// Error type for lookup table operations.
#[derive(Debug, thiserror::Error)]
pub enum LookupTableError {
    #[error("table size mismatch: expected {expected} entries for log_size={log_size}, got {got}")]
    SizeMismatch {
        log_size: u32,
        expected: usize,
        got: usize,
    },
    #[error("inputs and outputs must have the same length (inputs={inputs_len}, outputs={outputs_len})")]
    LengthMismatch {
        inputs_len: usize,
        outputs_len: usize,
    },
}

/// A precomputed function table mapping inputs to outputs over M31.
///
/// The table stores `(input, output)` pairs for a function `f: M31 -> M31`
/// over a domain of size `2^log_size`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrecomputedTable {
    /// Log2 of the domain size.
    pub log_size: u32,
    /// Input values (domain).
    pub inputs: Vec<M31>,
    /// Output values (range), where `outputs\[i\] = f(inputs\[i\])`.
    pub outputs: Vec<M31>,
}

impl PrecomputedTable {
    /// Build a table by evaluating `f` on domain `[0, 2^log_size)`.
    pub fn new(f: impl Fn(M31) -> M31, log_size: u32) -> Self {
        let size = 1usize << log_size;
        let mut inputs = Vec::with_capacity(size);
        let mut outputs = Vec::with_capacity(size);

        for i in 0..size {
            let input = M31::from(i as u32);
            inputs.push(input);
            outputs.push(f(input));
        }

        Self {
            log_size,
            inputs,
            outputs,
        }
    }

    /// Build a table from explicit input-output pairs.
    ///
    /// Returns `Err` if `inputs.len() != outputs.len()` or if
    /// `inputs.len() != 2^log_size`.
    pub fn from_pairs(
        log_size: u32,
        inputs: Vec<M31>,
        outputs: Vec<M31>,
    ) -> Result<Self, LookupTableError> {
        if inputs.len() != outputs.len() {
            return Err(LookupTableError::LengthMismatch {
                inputs_len: inputs.len(),
                outputs_len: outputs.len(),
            });
        }
        let expected = 1usize << log_size;
        if inputs.len() != expected {
            return Err(LookupTableError::SizeMismatch {
                log_size,
                expected,
                got: inputs.len(),
            });
        }
        Ok(Self {
            log_size,
            inputs,
            outputs,
        })
    }

    /// Number of entries in the table.
    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Look up the output for a given input index.
    pub fn get(&self, index: usize) -> (M31, M31) {
        (self.inputs[index], self.outputs[index])
    }

    /// Build an identity table: f(x) = x for x in [0, 2^log_size).
    /// Used for range checks.
    pub fn identity(log_size: u32) -> Self {
        Self::new(|x| x, log_size)
    }

    /// Build a ReLU table: f(x) = max(0, x) for quantized signed inputs.
    ///
    /// Input domain [0, 2^log_size) represents signed values:
    /// - [0, 2^(log_size-1)) maps to non-negative values (output = input)
    /// - [2^(log_size-1), 2^log_size) maps to negative values (output = 0)
    pub fn relu(log_size: u32) -> Self {
        let half = 1u32 << (log_size - 1);
        Self::new(
            move |x| {
                let x_val = x.0;
                if x_val < half {
                    x // Non-negative: ReLU(x) = x
                } else {
                    M31::from(0) // Negative: ReLU(x) = 0
                }
            },
            log_size,
        )
    }

    /// Build a squaring table: f(x) = x^2 mod p for x in [0, 2^log_size).
    pub fn square(log_size: u32) -> Self {
        Self::new(|x| x * x, log_size)
    }

    /// Build a GELU table: f(x) ≈ x × Φ(x) using the fast tanh approximation.
    ///
    /// Input domain `[0, 2^log_size)` uses signed interpretation:
    /// - `[0, half)` → non-negative values
    /// - `[half, 2^log_size)` → negative values (two's complement style)
    ///
    /// The approximation: `GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`
    pub fn gelu(log_size: u32) -> Self {
        let half = 1u32 << (log_size - 1);
        let size = 1u32 << log_size;
        // Scale factor: maps the integer domain to a float range suitable for GELU.
        // We use half/4.0 so the effective range is approximately [-4, 4).
        let scale = half as f64 / 4.0;

        Self::new(
            move |x| {
                let x_signed = if x.0 < half {
                    x.0 as f64
                } else {
                    x.0 as f64 - size as f64
                };
                let x_norm = x_signed / scale;

                // Fast GELU approximation
                let sqrt_2_over_pi = std::f64::consts::FRAC_2_PI.sqrt();
                let inner = sqrt_2_over_pi * (x_norm + 0.044715 * x_norm.powi(3));
                let gelu_val = 0.5 * x_norm * (1.0 + inner.tanh());

                // Map back to unsigned domain
                let output_scaled = gelu_val * scale;
                let output_unsigned = if output_scaled >= 0.0 {
                    (output_scaled.round() as u32).min(half - 1)
                } else {
                    let wrapped = output_scaled + size as f64;
                    (wrapped.round() as u32).clamp(half, size - 1)
                };
                M31::from(output_unsigned)
            },
            log_size,
        )
    }

    /// Build a Sigmoid table: f(x) = 1 / (1 + e^(-x)).
    ///
    /// Input domain `[0, 2^log_size)` uses signed interpretation:
    /// - `[0, half)` → non-negative values → sigmoid output in `[0.5, 1.0)`
    /// - `[half, 2^log_size)` → negative values → sigmoid output in `(0.0, 0.5)`
    ///
    /// Output is scaled to `[0, 2^log_size)` where `half` represents 1.0.
    pub fn sigmoid(log_size: u32) -> Self {
        let half = 1u32 << (log_size - 1);
        let size = 1u32 << log_size;
        let scale = half as f64 / 4.0;

        Self::new(
            move |x| {
                let x_signed = if x.0 < half {
                    x.0 as f64
                } else {
                    x.0 as f64 - size as f64
                };
                let x_norm = x_signed / scale;

                // Sigmoid: 1 / (1 + e^(-x))
                let sig_val = 1.0 / (1.0 + (-x_norm).exp());

                // Scale output: sig_val ∈ (0, 1) → [0, half) where half = 1.0
                let output = (sig_val * half as f64).round() as u32;
                M31::from(output.min(half - 1))
            },
            log_size,
        )
    }

    /// Build a softmax exponential table: f(x) = e^(x / scale).
    ///
    /// Softmax = exp(x_i) / Σ exp(x_j), so we need the exp table
    /// as a building block. The normalization is handled at a higher level.
    ///
    /// Output is scaled to `[0, 2^log_size)` where `1` maps to `scale_output`.
    pub fn softmax_exp(log_size: u32) -> Self {
        let half = 1u32 << (log_size - 1);
        let size = 1u32 << log_size;
        let scale = half as f64 / 4.0;
        // Output scale: e^0 = 1.0 maps to half/2 so we have headroom for larger values
        let output_scale = half as f64 / 2.0;

        Self::new(
            move |x| {
                let x_signed = if x.0 < half {
                    x.0 as f64
                } else {
                    x.0 as f64 - size as f64
                };
                let x_norm = x_signed / scale;

                // Clamp to avoid overflow (exp(x) for x > ~10 is huge)
                let exp_val = x_norm.clamp(-8.0, 8.0).exp();

                // Scale output
                let output = (exp_val * output_scale).round() as u32;
                M31::from(output.min(size - 1))
            },
            log_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_table() {
        let table = PrecomputedTable::identity(4);
        assert_eq!(table.size(), 16);
        for i in 0..16 {
            let (input, output) = table.get(i);
            assert_eq!(input, M31::from(i as u32));
            assert_eq!(output, M31::from(i as u32));
        }
    }

    #[test]
    fn test_relu_table() {
        let table = PrecomputedTable::relu(8);
        assert_eq!(table.size(), 256);

        // Non-negative values: ReLU(x) = x
        for i in 0..128 {
            let (_, output) = table.get(i);
            assert_eq!(output, M31::from(i as u32));
        }

        // "Negative" values (high half): ReLU(x) = 0
        for i in 128..256 {
            let (_, output) = table.get(i);
            assert_eq!(output, M31::from(0));
        }
    }

    #[test]
    fn test_square_table() {
        let table = PrecomputedTable::square(4);
        assert_eq!(table.size(), 16);
        assert_eq!(table.get(0).1, M31::from(0));
        assert_eq!(table.get(1).1, M31::from(1));
        assert_eq!(table.get(3).1, M31::from(9));
        assert_eq!(table.get(15).1, M31::from(225));
    }

    #[test]
    fn test_custom_function_table() {
        // f(x) = 2x + 1
        let table = PrecomputedTable::new(|x| x + x + M31::from(1), 3);
        assert_eq!(table.size(), 8);
        assert_eq!(table.get(0).1, M31::from(1));
        assert_eq!(table.get(1).1, M31::from(3));
        assert_eq!(table.get(4).1, M31::from(9));
    }

    #[test]
    fn test_gelu_table() {
        let table = PrecomputedTable::gelu(8);
        assert_eq!(table.size(), 256);

        // GELU(0) = 0
        assert_eq!(table.get(0).1, M31::from(0));

        // GELU(x) ≈ x for large positive x
        // x=100 in domain maps to ~3.125 in real, GELU(3.125) ≈ 3.125 → ~100
        let pos_large = table.get(100).1;
        assert!(pos_large.0 > 90, "GELU of large positive should be close to identity: {}", pos_large.0);

        // GELU(x) ≈ 0 for large negative x
        // x=200 in domain → -56 → real ≈ -1.75 → GELU ≈ -0.09 → close to 0 or wrapped
        let neg_val = table.get(200).1;
        // Should be close to 0 or in the upper half (small negative)
        assert!(neg_val.0 >= 128 || neg_val.0 < 10,
            "GELU of negative should be near zero: {}", neg_val.0);
    }

    #[test]
    fn test_sigmoid_table() {
        let table = PrecomputedTable::sigmoid(8);
        assert_eq!(table.size(), 256);

        // Sigmoid(0) = 0.5 → output should be half/2 = 64
        assert_eq!(table.get(0).1, M31::from(64));

        // Sigmoid(large positive) → close to 1.0 → output close to 128
        let pos_large = table.get(100).1;
        assert!(pos_large.0 > 110, "sigmoid of large positive should approach half: {}", pos_large.0);

        // Sigmoid(large negative) → close to 0.0 → output close to 0
        let neg_large = table.get(200).1; // 200 maps to -56 → real ≈ -1.75
        assert!(neg_large.0 < 50, "sigmoid of negative should be below 0.5: {}", neg_large.0);
    }

    #[test]
    fn test_softmax_exp_table() {
        let table = PrecomputedTable::softmax_exp(8);
        assert_eq!(table.size(), 256);

        // exp(0) = 1.0 → output should be half/2 = 64
        assert_eq!(table.get(0).1, M31::from(64));

        // exp(positive) > 1.0 → output > 64
        let pos = table.get(32).1; // small positive
        assert!(pos.0 > 64, "exp of positive should be > base: {}", pos.0);

        // exp(negative) < 1.0 → output < 64
        let neg = table.get(200).1; // negative
        assert!(neg.0 < 64, "exp of negative should be < base: {}", neg.0);
    }

    #[test]
    fn test_gelu_monotonic_positive() {
        // GELU should be (approximately) monotonically increasing for positive values
        let table = PrecomputedTable::gelu(8);
        let half = 128u32;
        for i in 2..half as usize {
            let prev = table.get(i - 1).1.0;
            let curr = table.get(i).1.0;
            // Allow small non-monotonicity due to quantization
            assert!(curr + 2 >= prev,
                "GELU should be roughly monotonic for positive: f({})={} < f({})={}",
                i - 1, prev, i, curr);
        }
    }

    #[test]
    fn test_sigmoid_monotonic() {
        // Sigmoid should be monotonically increasing for positive values
        let table = PrecomputedTable::sigmoid(8);
        let half = 128u32;
        for i in 1..half as usize {
            let prev = table.get(i - 1).1.0;
            let curr = table.get(i).1.0;
            assert!(curr + 1 >= prev,
                "sigmoid should be monotonic for positive: f({})={} < f({})={}",
                i - 1, prev, i, curr);
        }
    }
}
