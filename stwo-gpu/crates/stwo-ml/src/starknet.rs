//! Starknet on-chain verification types and calldata generation.


use crate::aggregation::LayerClaim;

/// A proof formatted for Starknet on-chain verification.
#[derive(Debug)]
pub struct StarknetModelProof {
    pub proof_bytes: Vec<u8>,
    pub layer_claims: Vec<LayerClaim>,
    pub model_commitment: [u8; 32],
    pub input_commitment: [u8; 32],
}

/// Error type for Starknet proof generation.
#[derive(Debug, thiserror::Error)]
pub enum StarknetModelError {
    #[error("Proving error: {0}")]
    ProvingError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Aggregation error: {0}")]
    AggregationError(String),
}

/// Estimate gas cost for verifying the proof on-chain.
pub fn estimate_verification_gas(num_layers: usize, total_trace_rows: usize) -> u64 {
    let base_cost: u64 = 50_000;
    let per_layer: u64 = 10_000;
    let per_row_log: u64 = 100;

    let log_rows = (total_trace_rows as f64).log2().ceil() as u64;
    base_cost + (num_layers as u64) * per_layer + log_rows * per_row_log
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gas_estimation() {
        let gas = estimate_verification_gas(3, 10_000);
        assert!(gas > 50_000);
        assert!(gas < 200_000);

        let gas = estimate_verification_gas(32, 10_000_000);
        assert!(gas > 300_000);
    }
}
