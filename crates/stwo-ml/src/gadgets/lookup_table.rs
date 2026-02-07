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
}
