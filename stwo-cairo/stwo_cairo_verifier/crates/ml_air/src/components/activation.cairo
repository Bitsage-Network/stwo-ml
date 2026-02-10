/// Activation component for LogUp-based activation function verification.
///
/// Verifies that each (input, output) pair exists in a precomputed activation
/// table using the LogUp protocol. The activation table is committed as a
/// preprocessed column.
///
/// Supported activations: ReLU (exact), GELU (approx), Sigmoid (approx).
///
/// The LogUp relation checks:
///   Σ_{row} 1/(z - combine(input, output)) = claimed_sum
///
/// where combine uses random linear combination with alpha from the channel.

use stwo_verifier_core::fields::qm31::{QM31, QM31Serde};

/// Claim about a single activation layer.
#[derive(Drop, Serde, Copy)]
pub struct ActivationClaim {
    /// Layer index in the computation graph.
    pub layer_index: u32,
    /// Log2 of the trace size (number of activation evaluations).
    pub log_size: u32,
    /// Activation type: 0=ReLU, 1=GELU, 2=Sigmoid.
    pub activation_type: u8,
}

/// Interaction claim for activation layer (LogUp sum).
#[derive(Drop, Serde, Copy)]
pub struct ActivationInteractionClaim {
    /// Claimed LogUp sum for this activation layer.
    pub claimed_sum: QM31,
}

/// Number of trace columns for the activation component.
///
/// - Column 0: input values
/// - Column 1: output values
/// - Column 2: multiplicities (how many times each table entry is used)
pub const N_ACTIVATION_TRACE_COLUMNS: u32 = 3;

/// Number of preprocessed columns (the activation table).
///
/// - Column 0: table input values
/// - Column 1: table output values
pub const N_ACTIVATION_PREPROCESSED_COLUMNS: u32 = 2;
