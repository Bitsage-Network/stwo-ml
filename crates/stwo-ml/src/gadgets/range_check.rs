//! Range check gadgets for bounding ML values within valid ranges.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::core::poly::circle::CanonicCoset;

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
        Self { min: 0, max: 255, log_size: 8 }
    }

    pub fn int8_unsigned() -> Self {
        Self { min: 0, max: 255, log_size: 8 }
    }

    pub fn uint16() -> Self {
        Self { min: 0, max: 65535, log_size: 16 }
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
pub fn generate_range_trace(
    values: &[M31],
    log_size: u32,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let size = 1usize << log_size;
    assert!(values.len() <= size);

    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut col = Col::<SimdBackend, BaseField>::zeros(size);
    for (i, &val) in values.iter().enumerate() {
        col.set(i, val);
    }

    vec![CircleEvaluation::new(domain, col)]
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
        let trace = generate_range_trace(&values, 4);
        assert_eq!(trace.len(), 1);
    }
}
