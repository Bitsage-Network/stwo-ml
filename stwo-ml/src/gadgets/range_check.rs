//! Range check gadgets for bounding ML values within valid ranges.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

use super::lookup_table::PrecomputedTable;

/// Range check configuration.
#[derive(Debug, Clone, Copy)]
pub struct RangeCheckConfig {
    pub min: u32,
    pub max: u32,
    pub log_size: u32,
}

impl RangeCheckConfig {
    pub fn uint8() -> Self {
        Self {
            min: 0,
            max: 255,
            log_size: 8,
        }
    }

    pub fn int8_unsigned() -> Self {
        Self {
            min: 0,
            max: 255,
            log_size: 8,
        }
    }

    pub fn uint16() -> Self {
        Self {
            min: 0,
            max: 65535,
            log_size: 16,
        }
    }

    pub fn custom(min: u32, max: u32) -> Self {
        let range = max - min + 1;
        let log_size = (range as f64).log2().ceil() as u32;
        Self { min, max, log_size }
    }

    pub fn range_size(&self) -> u32 {
        self.max - self.min + 1
    }
}

/// Generate a range table as a `PrecomputedTable`.
pub fn generate_range_table(config: &RangeCheckConfig) -> PrecomputedTable {
    PrecomputedTable::build(|x| x, config.log_size)
}

/// Generate a range check trace for a set of values.
///
/// Generic over backend `B` — works with `SimdBackend`, `CpuBackend`, or `GpuBackend`.
pub fn generate_range_trace<B: ColumnOps<BaseField>>(
    values: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    assert!(values.len() <= size);

    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut col = Col::<B, BaseField>::zeros(size);
    for (i, &val) in values.iter().enumerate() {
        col.set(i, val);
    }

    vec![CircleEvaluation::new(domain, col)]
}

/// Generate a range check trace using `SimdBackend` (convenience wrapper).
pub fn generate_range_trace_simd(
    values: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    generate_range_trace::<SimdBackend>(values, log_size)
}

/// Compute multiplicities: count how many times each table entry appears in `values`.
///
/// Returns a vector of length `config.range_size()` where entry `i` holds
/// the number of occurrences of `(config.min + i)` in `values`.
pub fn compute_range_multiplicities(values: &[M31], config: &RangeCheckConfig) -> Vec<M31> {
    let table_size = config.range_size() as usize;
    let mut counts = vec![0u32; table_size];

    for v in values {
        let idx = v.0.wrapping_sub(config.min) as usize;
        if idx < table_size {
            counts[idx] += 1;
        }
    }

    counts.into_iter().map(M31::from).collect()
}

/// Generate the execution trace for a range check: 2 columns (value, multiplicity).
///
/// The table column has `2^log_size` entries: `[min, min+1, ..., max]` padded with
/// the first table entry. Multiplicities correspond 1:1 with table entries.
/// Padding rows use multiplicity 0.
pub fn generate_range_execution_trace<B: ColumnOps<BaseField>>(
    values: &[M31],
    multiplicities: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<B, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    assert!(
        values.len() <= size,
        "values ({}) exceed trace size ({})",
        values.len(),
        size
    );
    assert!(multiplicities.len() <= size);

    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut val_col = Col::<B, BaseField>::zeros(size);
    let mut mult_col = Col::<B, BaseField>::zeros(size);

    for (i, &v) in values.iter().enumerate() {
        val_col.set(i, v);
    }
    // Padding rows use value from the first table entry (already zero-initialized for min=0)
    for (i, &m) in multiplicities.iter().enumerate() {
        mult_col.set(i, m);
    }

    vec![
        CircleEvaluation::new(domain, val_col),
        CircleEvaluation::new(domain, mult_col),
    ]
}

/// Verify that all values are within the range (CPU-side validation).
pub fn check_range(values: &[M31], config: &RangeCheckConfig) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.0 < config.min || v.0 > config.max)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_check_configs() {
        let uint8 = RangeCheckConfig::uint8();
        assert_eq!(uint8.range_size(), 256);
        assert_eq!(uint8.log_size, 8);

        let custom = RangeCheckConfig::custom(10, 100);
        assert_eq!(custom.range_size(), 91);
    }

    #[test]
    fn test_generate_range_table() {
        let config = RangeCheckConfig::uint8();
        let table = generate_range_table(&config);
        assert_eq!(table.size(), 256);
        assert_eq!(table.lookup(M31::from(42)), Some(M31::from(42)));
    }

    #[test]
    fn test_check_range() {
        let config = RangeCheckConfig::custom(0, 10);
        let values = vec![M31::from(5), M31::from(11), M31::from(0), M31::from(100)];
        let oob = check_range(&values, &config);
        assert_eq!(oob, vec![1, 3]);
    }

    #[test]
    fn test_generate_range_trace() {
        let values = vec![M31::from(1), M31::from(2), M31::from(3)];
        let trace = generate_range_trace::<SimdBackend>(&values, 4);
        assert_eq!(trace.len(), 1);
    }

    #[test]
    fn test_compute_range_multiplicities() {
        let config = RangeCheckConfig::custom(0, 7);
        let values = vec![
            M31::from(0),
            M31::from(0),
            M31::from(3),
            M31::from(7),
            M31::from(3),
            M31::from(5),
        ];
        let mults = compute_range_multiplicities(&values, &config);
        assert_eq!(mults.len(), 8);
        assert_eq!(mults[0], M31::from(2)); // value 0 appears twice
        assert_eq!(mults[1], M31::from(0)); // value 1 not present
        assert_eq!(mults[3], M31::from(2)); // value 3 appears twice
        assert_eq!(mults[5], M31::from(1)); // value 5 appears once
        assert_eq!(mults[7], M31::from(1)); // value 7 appears once
    }

    #[test]
    fn test_generate_range_execution_trace() {
        let values = vec![M31::from(0), M31::from(1), M31::from(2)];
        let mults = vec![M31::from(1), M31::from(1), M31::from(1)];
        let trace = generate_range_execution_trace::<SimdBackend>(&values, &mults, 4);
        assert_eq!(trace.len(), 2); // value col + multiplicity col
    }
}
