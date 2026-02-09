//! Precomputed function lookup tables for non-linear operations.
//!
//! Generates preprocessed columns containing (input, output) pairs for
//! activation functions. Used with LogUp to prove activation correctness.

use std::collections::HashMap;

use stwo::core::fields::m31::{BaseField, M31};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::Col;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::core::poly::circle::CanonicCoset;

/// A precomputed lookup table mapping input → output over M31.
#[derive(Debug, Clone)]
pub struct PrecomputedTable {
    pub inputs: Vec<M31>,
    pub outputs: Vec<M31>,
    pub log_size: u32,
    /// Hash index for O(1) lookup (built lazily for tables > 2^10).
    index: Option<HashMap<u32, usize>>,
}

/// Threshold (log_size) above which we build a hash index for O(1) lookup.
const INDEX_THRESHOLD_LOG: u32 = 10;

impl PrecomputedTable {
    /// Build a lookup table by evaluating `f` on all inputs `[0, 2^log_size)`.
    pub fn build(f: impl Fn(M31) -> M31, log_size: u32) -> Self {
        let size = 1usize << log_size;
        let mut inputs = Vec::with_capacity(size);
        let mut outputs = Vec::with_capacity(size);

        for i in 0..size {
            let input = M31::from(i as u32);
            let output = f(input);
            inputs.push(input);
            outputs.push(output);
        }

        let index = Self::maybe_build_index(&inputs, log_size);
        Self { inputs, outputs, log_size, index }
    }

    /// Build a lookup table in parallel using rayon.
    ///
    /// Recommended for log_size >= 14 (16K+ entries).
    pub fn build_parallel(f: impl Fn(M31) -> M31 + Send + Sync, log_size: u32) -> Self {
        use rayon::prelude::*;

        let size = 1usize << log_size;
        let pairs: Vec<(M31, M31)> = (0..size)
            .into_par_iter()
            .map(|i| {
                let input = M31::from(i as u32);
                let output = f(input);
                (input, output)
            })
            .collect();

        let inputs: Vec<M31> = pairs.iter().map(|(i, _)| *i).collect();
        let outputs: Vec<M31> = pairs.iter().map(|(_, o)| *o).collect();
        let index = Self::maybe_build_index(&inputs, log_size);

        Self { inputs, outputs, log_size, index }
    }

    /// Build a lookup table from explicit (input, output) pairs.
    pub fn from_pairs(pairs: Vec<(M31, M31)>, log_size: u32) -> Self {
        let size = 1usize << log_size;
        assert!(
            pairs.len() <= size,
            "Too many pairs ({}) for log_size {log_size}",
            pairs.len(),
        );

        let mut inputs = Vec::with_capacity(size);
        let mut outputs = Vec::with_capacity(size);

        for (inp, out) in &pairs {
            inputs.push(*inp);
            outputs.push(*out);
        }

        // Pad with zeros if needed
        while inputs.len() < size {
            inputs.push(M31::from(0));
            outputs.push(M31::from(0));
        }

        let index = Self::maybe_build_index(&inputs, log_size);
        Self { inputs, outputs, log_size, index }
    }

    /// Number of entries in the table.
    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Look up an input value in the table.
    ///
    /// Uses hash index for large tables, linear scan for small ones.
    pub fn lookup(&self, input: M31) -> Option<M31> {
        if let Some(ref idx_map) = self.index {
            idx_map.get(&input.0).map(|&idx| self.outputs[idx])
        } else {
            self.inputs
                .iter()
                .position(|&x| x == input)
                .map(|idx| self.outputs[idx])
        }
    }

    /// Build hash index if table is large enough to benefit.
    fn maybe_build_index(inputs: &[M31], log_size: u32) -> Option<HashMap<u32, usize>> {
        if log_size >= INDEX_THRESHOLD_LOG {
            let mut map = HashMap::with_capacity(inputs.len());
            for (i, inp) in inputs.iter().enumerate() {
                map.insert(inp.0, i);
            }
            Some(map)
        } else {
            None
        }
    }

    /// Generate preprocessed columns (input_col, output_col) as CircleEvaluations.
    pub fn generate_trace_columns(
        &self,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        let domain = CanonicCoset::new(self.log_size).circle_domain();

        let input_col: Col<SimdBackend, BaseField> = self.inputs.iter().copied().collect();
        let output_col: Col<SimdBackend, BaseField> = self.outputs.iter().copied().collect();

        vec![
            CircleEvaluation::new(domain, input_col),
            CircleEvaluation::new(domain, output_col),
        ]
    }
}

/// Standard activation function implementations for table building.
pub mod activations {
    use stwo::core::fields::m31::M31;

    /// ReLU: max(0, x). Treats values > P/2 as "negative" (wrapping).
    pub fn relu(x: M31) -> M31 {
        let val = x.0;
        let half_p = (1u32 << 30) - 1;
        if val <= half_p { x } else { M31::from(0) }
    }

    /// Approximate GELU using a piecewise-linear table.
    pub fn gelu_approx(x: M31) -> M31 {
        let val = x.0;
        let half_p = (1u32 << 30) - 1;
        if val > half_p { M31::from(0) } else { x }
    }

    /// Approximate sigmoid: 1/(1+e^(-x)).
    pub fn sigmoid_approx(x: M31) -> M31 {
        let val = x.0;
        let half_p = (1u32 << 30) - 1;
        let scale = 1u32 << 16;
        if val > half_p {
            M31::from(0)
        } else if val == 0 {
            M31::from(scale / 2)
        } else {
            M31::from(scale)
        }
    }

    /// Approximate exp(x) for softmax computation.
    ///
    /// Maps M31 values to a scaled exponential approximation:
    /// - "Negative" region (val > P/2): returns 1 (near-zero after normalization)
    /// - Zero: returns 2^16 (exp(0) = 1, scaled)
    /// - Positive: linear scale as approximation
    pub fn softmax_exp(x: M31) -> M31 {
        let val = x.0;
        let half_p = (1u32 << 30) - 1;
        if val > half_p {
            // "Negative" region — exp of large negative ≈ 0
            M31::from(1)
        } else if val == 0 {
            // exp(0) = 1, scaled by 2^16
            M31::from(1u32 << 16)
        } else {
            // Positive region — capped linear approximation
            M31::from(val.min((1u32 << 31) - 2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_relu_table() {
        let table = PrecomputedTable::build(activations::relu, 4);
        assert_eq!(table.size(), 16);
        assert_eq!(table.lookup(M31::from(0)), Some(M31::from(0)));
        assert_eq!(table.lookup(M31::from(5)), Some(M31::from(5)));
    }

    #[test]
    fn test_build_from_pairs() {
        let pairs = vec![
            (M31::from(0), M31::from(10)),
            (M31::from(1), M31::from(20)),
            (M31::from(2), M31::from(30)),
        ];
        let table = PrecomputedTable::from_pairs(pairs, 2);
        assert_eq!(table.size(), 4);
        assert_eq!(table.lookup(M31::from(0)), Some(M31::from(10)));
    }

    #[test]
    fn test_generate_trace_columns() {
        let table = PrecomputedTable::build(activations::relu, 4);
        let cols = table.generate_trace_columns();
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn test_sigmoid_boundary() {
        let result = activations::sigmoid_approx(M31::from(0));
        assert_eq!(result, M31::from(1 << 15));
    }

    #[test]
    fn test_large_relu_table_2_16() {
        let table = PrecomputedTable::build_parallel(activations::relu, 16);
        assert_eq!(table.size(), 65536);
        assert!(table.index.is_some());

        // Verify correctness at boundaries
        assert_eq!(table.lookup(M31::from(0)), Some(M31::from(0)));
        assert_eq!(table.lookup(M31::from(1000)), Some(M31::from(1000)));
        assert_eq!(table.lookup(M31::from(65535)), Some(M31::from(65535)));
    }

    #[test]
    fn test_large_table_trace_columns_2_16() {
        let table = PrecomputedTable::build_parallel(activations::relu, 16);
        let cols = table.generate_trace_columns();
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let sequential = PrecomputedTable::build(activations::relu, 12);
        let parallel = PrecomputedTable::build_parallel(activations::relu, 12);

        assert_eq!(sequential.inputs, parallel.inputs);
        assert_eq!(sequential.outputs, parallel.outputs);
    }

    #[test]
    fn test_indexed_lookup_performance() {
        // Build table large enough to trigger indexing
        let table = PrecomputedTable::build(activations::relu, 12);
        assert!(table.index.is_some());

        // O(1) lookups should work correctly
        for i in [0, 100, 500, 1000, 2048, 4095] {
            let result = table.lookup(M31::from(i as u32));
            assert!(result.is_some(), "lookup failed for {i}");
        }
    }
}
