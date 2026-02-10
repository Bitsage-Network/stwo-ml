/// ML-specific STARK components.
///
/// Each component corresponds to a provable ML operation:
///
/// - `matmul`: Sumcheck round polynomial consistency (not a STARK component;
///             verified via `sumcheck::verify_matmul_sumcheck`)
/// - `activation`: LogUp table lookups for ReLU/GELU/Sigmoid
/// - `layernorm`: LayerNorm constraint verification (mean, variance, rsqrt)
///
/// Components implement the constraint evaluation pattern from the
/// `constraint_framework` crate, making them compatible with the generic
/// STARK verifier.

pub mod activation;
pub mod matmul;
