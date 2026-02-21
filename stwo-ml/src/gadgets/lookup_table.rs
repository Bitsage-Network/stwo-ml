//! Precomputed function lookup tables for non-linear operations.
//!
//! Generates preprocessed columns containing (input, output) pairs for
//! activation functions. Used with LogUp to prove activation correctness.

use std::collections::HashMap;

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

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
        Self {
            inputs,
            outputs,
            log_size,
            index,
        }
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

        Self {
            inputs,
            outputs,
            log_size,
            index,
        }
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
        Self {
            inputs,
            outputs,
            log_size,
            index,
        }
    }

    /// Number of entries in the table.
    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Look up an input value in the table and return its output.
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

    /// Look up an input value and return its index in the table.
    ///
    /// Uses hash index for O(1) lookup on large tables.
    pub fn lookup_index(&self, input: M31) -> Option<usize> {
        if let Some(ref idx_map) = self.index {
            idx_map.get(&input.0).copied()
        } else {
            self.inputs.iter().position(|&x| x == input)
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

    /// Flat u32 array indexed by input value, for GPU texture lookup.
    ///
    /// Returns a `Vec<u32>` of size `2^log_size` where `table[inp.0 % size] = out.0`.
    /// Used by GPU LayerNorm/RMSNorm kernels to do direct rsqrt lookups.
    pub fn as_gpu_lookup_table(&self) -> Vec<u32> {
        let size = 1usize << self.log_size;
        let mut table = vec![0u32; size];
        for (inp, out) in self.inputs.iter().zip(self.outputs.iter()) {
            table[inp.0 as usize % size] = out.0;
        }
        table
    }

    /// Generate preprocessed columns (input_col, output_col) as CircleEvaluations.
    ///
    /// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
    pub fn generate_trace_columns<B: ColumnOps<BaseField>>(
        &self,
    ) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
        let domain = CanonicCoset::new(self.log_size).circle_domain();

        let input_col: Col<B, BaseField> = self.inputs.iter().copied().collect();
        let output_col: Col<B, BaseField> = self.outputs.iter().copied().collect();

        vec![
            CircleEvaluation::new(domain, input_col),
            CircleEvaluation::new(domain, output_col),
        ]
    }

    /// Generate preprocessed columns using `SimdBackend` (convenience wrapper).
    pub fn generate_trace_columns_simd(
        &self,
    ) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
        self.generate_trace_columns::<SimdBackend>()
    }
}

/// Standard activation function implementations for table building.
pub mod activations {
    use stwo::core::fields::m31::M31;

    /// ReLU: max(0, x). Treats values > P/2 as "negative" (wrapping).
    pub fn relu(x: M31) -> M31 {
        let val = x.0;
        let half_p = (1u32 << 30) - 1;
        if val <= half_p {
            x
        } else {
            M31::from(0)
        }
    }

    /// Fixed-point scale for GELU (and other scaled activations).
    ///
    /// M31 values are interpreted as x_real = val / GELU_SCALE.
    /// Output is round(gelu(x_real) * GELU_SCALE).
    /// Matches the scale convention used by sigmoid_approx and softmax_exp.
    pub const GELU_FIXED_POINT_SCALE: u32 = 1 << 16;

    /// GELU: x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    ///
    /// Uses fixed-point arithmetic with scale = GELU_FIXED_POINT_SCALE (2^16).
    /// Interprets M31 values using the standard sign convention:
    ///   val <= P/2  → positive: x_real = val / scale
    ///   val >  P/2  → negative: x_real = -(P - val) / scale
    ///
    /// For table inputs [0, 2^log_size), all values are positive (since 2^18 << P/2).
    /// The f64 computation matches the reference implementation in f32_ops::gelu_f32.
    pub fn gelu_approx(x: M31) -> M31 {
        let val = x.0;
        let p = (1u64 << 31) - 1; // M31 prime
        let half_p = (1u32 << 30) - 1;
        let scale = GELU_FIXED_POINT_SCALE as f64;

        if val == 0 {
            return M31::from(0);
        }

        // Convert M31 to signed real number
        let x_real = if val <= half_p {
            val as f64 / scale
        } else {
            -((p - val as u64) as f64) / scale
        };

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi: f64 = (2.0_f64 / std::f64::consts::PI).sqrt();
        let inner = sqrt_2_over_pi * (x_real + 0.044715 * x_real * x_real * x_real);
        let gelu = 0.5 * x_real * (1.0 + inner.tanh());

        // Convert back to M31 fixed-point
        if gelu >= 0.0 {
            let result = (gelu * scale).round();
            let clamped = (result as u64).min(p - 1) as u32;
            M31::from(clamped)
        } else {
            // Negative result wraps around in M31
            let neg = (-gelu * scale).round() as u64;
            let wrapped = (p - neg.min(p - 1)) as u32;
            M31::from(wrapped)
        }
    }

    /// Approximate sigmoid: 1/(1+e^(-x)).
    ///
    /// Maps M31 values to a fixed-point sigmoid approximation (scale = 2^16).
    /// Treats values > P/2 as negative (field wrapping convention).
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

    /// Softmax element-wise component: approximate exp(x).
    ///
    /// Softmax = exp(x_i) / sum(exp(x_j)). This function computes the
    /// per-element exp(x) part. Normalization (division by sum) is handled
    /// separately in the attention pipeline.
    ///
    /// Fixed-point with scale = 2^16. Treats values > P/2 as negative.
    pub fn softmax_exp(x: M31) -> M31 {
        let val = x.0;
        let half_p = (1u32 << 30) - 1;
        let scale = 1u32 << 16;
        if val > half_p {
            // "Negative" value → exp approaches 0
            // exp(-(P - val) / scale), clamped to minimum 1
            let neg = ((1u32 << 31) - 2) - val;
            let decay = (neg / scale).min(20);
            let result = scale >> decay;
            M31::from(result.max(1))
        } else if val == 0 {
            // exp(0) = 1.0 → scale
            M31::from(scale)
        } else {
            // Positive value → exp grows, clamped to field max
            let growth = (val / scale).min(10);
            let result = scale.checked_shl(growth).unwrap_or((1u32 << 31) - 2);
            M31::from(result.min((1u32 << 31) - 2))
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
        let cols = table.generate_trace_columns::<SimdBackend>();
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
        let cols = table.generate_trace_columns::<SimdBackend>();
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

    // ===== GELU-specific tests =====

    #[test]
    fn test_gelu_zero() {
        assert_eq!(activations::gelu_approx(M31::from(0)), M31::from(0));
    }

    #[test]
    fn test_gelu_differs_from_relu() {
        // The core bug: GELU was identical to ReLU. Verify they now differ.
        let scale = activations::GELU_FIXED_POINT_SCALE;

        // At x_real = 0.5 (val = scale/2), GELU ≈ 0.346, ReLU = 0.5
        let val_half = M31::from(scale / 2);
        let gelu_out = activations::gelu_approx(val_half);
        let relu_out = activations::relu(val_half);
        assert_ne!(
            gelu_out,
            relu_out,
            "GELU and ReLU must differ at x=0.5 (scale/2={}, gelu={}, relu={})",
            scale / 2,
            gelu_out.0,
            relu_out.0,
        );

        // GELU output should be less than ReLU output for moderate positive values
        assert!(
            gelu_out.0 < relu_out.0,
            "GELU({}) = {} should be < ReLU({}) = {} for moderate x",
            val_half.0,
            gelu_out.0,
            val_half.0,
            relu_out.0,
        );

        // At x_real = 1.0 (val = scale), GELU ≈ 0.841, ReLU = 1.0
        let val_one = M31::from(scale);
        let gelu_one = activations::gelu_approx(val_one);
        let relu_one = activations::relu(val_one);
        assert_ne!(gelu_one, relu_one, "GELU and ReLU must differ at x=1.0");
        assert!(gelu_one.0 < relu_one.0, "GELU(1.0) < ReLU(1.0)");
    }

    #[test]
    fn test_gelu_known_values() {
        let scale = activations::GELU_FIXED_POINT_SCALE as f64;

        // Test against reference gelu_f32 at known points
        let test_points: Vec<(f64, f64)> = vec![
            (0.0, 0.0),
            (0.5, 0.3457), // GELU(0.5) ≈ 0.3457
            (1.0, 0.8412), // GELU(1.0) ≈ 0.8412
            (2.0, 1.9545), // GELU(2.0) ≈ 1.9545
            (3.0, 2.9964), // GELU(3.0) ≈ 2.9964
        ];

        for (x_real, expected_gelu) in test_points {
            let val = (x_real * scale).round() as u32;
            let output = activations::gelu_approx(M31::from(val));
            let expected_val = (expected_gelu * scale).round() as u32;
            let tolerance = (0.01 * scale).round() as u32; // 1% tolerance

            let diff = if output.0 > expected_val {
                output.0 - expected_val
            } else {
                expected_val - output.0
            };
            assert!(
                diff <= tolerance,
                "GELU({}) = {} (expected ~{}, diff={})",
                x_real,
                output.0,
                expected_val,
                diff,
            );
        }
    }

    #[test]
    fn test_gelu_converges_to_identity_for_large_x() {
        let scale = activations::GELU_FIXED_POINT_SCALE;

        // For large x_real (> 4), GELU(x) ≈ x
        // x_real = 5.0 → val = 5 * scale
        let val_large = 5 * scale;
        let gelu_large = activations::gelu_approx(M31::from(val_large));
        let tolerance = scale / 100; // 1% of scale

        let diff = if gelu_large.0 > val_large {
            gelu_large.0 - val_large
        } else {
            val_large - gelu_large.0
        };
        assert!(
            diff <= tolerance,
            "GELU(5.0) should be ≈ 5.0: got {} (expected ~{}, diff={})",
            gelu_large.0,
            val_large,
            diff,
        );
    }

    #[test]
    fn test_gelu_negative_returns_near_zero() {
        let scale = activations::GELU_FIXED_POINT_SCALE;
        let p = (1u64 << 31) - 1;

        // x_real = -5.0 → val = P - 5*scale
        // GELU(-5.0) ≈ -5 * Φ(-5) ≈ -5 * 0.0000003 ≈ 0 (within rounding)
        let val_neg5 = (p - 5 * scale as u64) as u32;
        let result = activations::gelu_approx(M31::from(val_neg5));
        assert_eq!(
            result,
            M31::from(0),
            "GELU(-5.0) should be 0: got {}",
            result.0,
        );

        // x_real = -3.0 → GELU(-3.0) ≈ -0.00363 → rounds to 0 in fixed-point
        let val_neg3 = (p - 3 * scale as u64) as u32;
        let result_neg3 = activations::gelu_approx(M31::from(val_neg3));
        // For x=-3, GELU is very small negative. In fixed-point it may round to 0
        // or wrap to a value very close to P. Either way, the M31 value should be
        // 0 or very close to P (representing a tiny negative).
        let is_near_zero = result_neg3.0 == 0 || result_neg3.0 > p as u32 - scale;
        assert!(
            is_near_zero,
            "GELU(-3.0) should be near zero: got {} (P={})",
            result_neg3.0, p,
        );
    }

    #[test]
    fn test_gelu_monotonic_positive() {
        // GELU should be monotonically increasing for positive x
        let scale = activations::GELU_FIXED_POINT_SCALE;
        let mut prev = 0u32;
        for i in 0..100 {
            let val = i * (scale / 10); // x_real = 0.0, 0.1, 0.2, ..., 9.9
            let out = activations::gelu_approx(M31::from(val));
            assert!(
                out.0 >= prev,
                "GELU should be monotonic: GELU({}) = {} < prev = {}",
                val,
                out.0,
                prev,
            );
            prev = out.0;
        }
    }

    #[test]
    fn test_gelu_table_builds_correctly() {
        let table = PrecomputedTable::build(activations::gelu_approx, 8);
        assert_eq!(table.size(), 256);

        // GELU(0) = 0
        assert_eq!(table.lookup(M31::from(0)), Some(M31::from(0)));

        // All entries should be valid M31 values (no panics during build)
        for i in 0..256 {
            let result = table.lookup(M31::from(i as u32));
            assert!(result.is_some(), "table should contain entry for {i}");
        }
    }

    #[test]
    fn test_gelu_matches_f64_reference() {
        // Cross-check our M31 GELU against an f64 reference implementation
        let scale = activations::GELU_FIXED_POINT_SCALE as f64;
        let sqrt_2_over_pi: f64 = (2.0_f64 / std::f64::consts::PI).sqrt();

        for i in (0..200).map(|x| x * 500) {
            let x_real = i as f64 / scale;
            let inner = sqrt_2_over_pi * (x_real + 0.044715 * x_real.powi(3));
            let expected = 0.5 * x_real * (1.0 + inner.tanh());
            let expected_val = (expected * scale).round() as u32;

            let actual = activations::gelu_approx(M31::from(i as u32));
            assert_eq!(
                actual.0, expected_val,
                "M31 GELU({}) = {} but f64 reference = {} (x_real={})",
                i, actual.0, expected_val, x_real,
            );
        }
    }
}
