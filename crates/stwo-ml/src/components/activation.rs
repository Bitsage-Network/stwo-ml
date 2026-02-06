//! LogUp-based activation function verification.
//!
//! Non-linear operations (ReLU, GELU, sigmoid, softmax) are prohibitively
//! expensive to arithmetize directly. Instead, we precompute lookup tables
//! and use STWO's LogUp protocol to prove each activation value exists in
//! the table.
//!
//! # Supported Activations
//!
//! | Function | Table Size | Precision | Use Case |
//! |----------|-----------|-----------|----------|
//! | ReLU     | 2^16      | Exact     | Most layers |
//! | GELU     | 2^16      | ~0.001%   | Transformers |
//! | Sigmoid  | 2^16      | ~0.001%   | Classification |
//! | Softmax  | 2^20      | ~0.01%    | Attention weights |
//! | LayerNorm| 2^16      | ~0.01%    | Normalization |
//!
//! # How It Works
//!
//! ```text
//! Preprocessed Column (read-only):
//!   table[0]     = (input_0, relu(input_0))
//!   table[1]     = (input_1, relu(input_1))
//!   ...
//!   table[65535] = (input_65535, relu(input_65535))
//!
//! Execution Trace:
//!   row 0: (x=1234, y=1234)   → LogUp proves (1234, 1234) ∈ table
//!   row 1: (x=-500, y=0)      → LogUp proves (-500, 0) ∈ table
//!   ...
//!
//! LogUp Constraint:
//!   Σ multiplicity_i / (α - table_i) = Σ 1 / (α - trace_i)
//!   (batched across all rows using random challenge α)
//! ```

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// max(0, x) — exact, no approximation needed.
    ReLU,
    /// x × Φ(x) — Gaussian error linear unit, table-approximated.
    GELU,
    /// 1 / (1 + e^(-x)) — table-approximated.
    Sigmoid,
    /// e^(x_i) / Σ e^(x_j) — requires normalization gadget + table.
    Softmax,
    /// (x - μ) / σ — requires running stats + table for reciprocal sqrt.
    LayerNorm,
}

impl ActivationType {
    /// Recommended lookup table size (log2) for this activation.
    pub fn recommended_table_log_size(&self) -> u32 {
        match self {
            ActivationType::ReLU => 16,     // 65K entries, exact
            ActivationType::GELU => 16,     // 65K entries, ~0.001% error
            ActivationType::Sigmoid => 16,  // 65K entries, ~0.001% error
            ActivationType::Softmax => 20,  // 1M entries, ~0.01% error
            ActivationType::LayerNorm => 16, // 65K entries for rsqrt table
        }
    }

    /// Whether this activation can be computed exactly (no approximation).
    pub fn is_exact(&self) -> bool {
        matches!(self, ActivationType::ReLU)
    }
}

// TODO: Phase 1 implementation
// - Generate preprocessed columns with precomputed activation tables
// - Implement LogUp interaction for activation lookups
// - Use STWO's LogUpSingles variant (implicit numerator = 1)
// - Support quantized inputs (INT8, FP16 mapped to M31 range)
// - GPU-accelerated table generation for large tables (2^20)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_sizes() {
        assert_eq!(ActivationType::ReLU.recommended_table_log_size(), 16);
        assert_eq!(ActivationType::Softmax.recommended_table_log_size(), 20);
        assert!(ActivationType::ReLU.is_exact());
        assert!(!ActivationType::GELU.is_exact());
    }
}
