//! Transaction feature encoder: converts raw TX data to M31 input vector.
//!
//! Produces a `(1, 64)` M31Matrix from [`TransactionFeatures`]. The first 48
//! elements carry transaction data; the remaining 16 are zero-padded for
//! power-of-2 MLE alignment required by the GKR prover.

use stwo::core::fields::m31::M31;

use crate::components::matmul::M31Matrix;

use super::types::*;

/// Encode raw transaction features into a `(1, 64)` M31 matrix.
///
/// The encoding is deterministic: identical features always produce the
/// identical input vector, which produces an identical IO commitment hash.
pub fn encode_transaction(tx: &TransactionFeatures) -> M31Matrix {
    let mut input = M31Matrix::new(1, INPUT_DIM);
    let mut idx = 0;

    // Features 0-7: Target address (felt252 → 8 × 31-bit chunks)
    let target_bytes = tx.target.to_bytes_be();
    for chunk in 0..8 {
        let offset = 28 - chunk * 4; // big-endian, 4 bytes per chunk
        let val = if offset < 32 {
            u32::from_be_bytes([
                target_bytes.get(offset).copied().unwrap_or(0),
                target_bytes.get(offset + 1).copied().unwrap_or(0),
                target_bytes.get(offset + 2).copied().unwrap_or(0),
                target_bytes.get(offset + 3).copied().unwrap_or(0),
            ]) & 0x7FFF_FFFF // mask to 31 bits for M31
        } else {
            0
        };
        input.set(0, idx, M31::from(val));
        idx += 1;
    }

    // Features 8-15: Value (u256 → 8 × 31-bit chunks)
    let [high, low] = tx.value;
    for i in 0..4 {
        let shift = i * 31;
        let val = ((low >> shift) & 0x7FFF_FFFF) as u32;
        input.set(0, idx, M31::from(val));
        idx += 1;
    }
    for i in 0..4 {
        let shift = i * 31;
        let val = ((high >> shift) & 0x7FFF_FFFF) as u32;
        input.set(0, idx, M31::from(val));
        idx += 1;
    }

    // Feature 16: Function selector
    input.set(0, idx, M31::from(tx.selector & 0x7FFF_FFFF));
    idx += 1;

    // Features 17-24: Calldata prefix (8 words)
    for &word in &tx.calldata_prefix {
        input.set(0, idx, M31::from(word & 0x7FFF_FFFF));
        idx += 1;
    }

    // Feature 25: Calldata length
    input.set(0, idx, M31::from(tx.calldata_len.min(0x7FFF_FFFF)));
    idx += 1;

    // Feature 26: Agent trust score
    input.set(0, idx, M31::from(tx.agent_trust_score.min(100_000)));
    idx += 1;

    // Feature 27: Agent strike count
    input.set(0, idx, M31::from(tx.agent_strikes));
    idx += 1;

    // Feature 28: Agent age (blocks)
    input.set(0, idx, M31::from(tx.agent_age_blocks & 0x7FFF_FFFF));
    idx += 1;

    // Features 29-32: Target flags
    input.set(0, idx, M31::from(tx.target_flags.is_verified as u32));
    idx += 1;
    input.set(0, idx, M31::from(tx.target_flags.is_proxy as u32));
    idx += 1;
    input.set(0, idx, M31::from(tx.target_flags.has_source as u32));
    idx += 1;
    input.set(0, idx, M31::from(tx.target_flags.interaction_count.min(0x7FFF_FFFF)));
    idx += 1;

    // Features 33-36: Value features
    input.set(0, idx, M31::from(tx.value_features.log2_value));
    idx += 1;
    input.set(0, idx, M31::from(tx.value_features.value_balance_ratio.min(100_000)));
    idx += 1;
    input.set(0, idx, M31::from(tx.value_features.is_max_approval as u32));
    idx += 1;
    input.set(0, idx, M31::from(tx.value_features.is_zero_value as u32));
    idx += 1;

    // Features 37-40: Selector features
    input.set(0, idx, M31::from(tx.selector_features.is_transfer as u32));
    idx += 1;
    input.set(0, idx, M31::from(tx.selector_features.is_approve as u32));
    idx += 1;
    input.set(0, idx, M31::from(tx.selector_features.is_swap as u32));
    idx += 1;
    input.set(0, idx, M31::from(tx.selector_features.is_unknown as u32));
    idx += 1;

    // Features 41-44: Behavioral features
    input.set(0, idx, M31::from(tx.behavioral.tx_frequency.min(0x7FFF_FFFF)));
    idx += 1;
    input.set(0, idx, M31::from(tx.behavioral.unique_targets_24h.min(0x7FFF_FFFF)));
    idx += 1;
    input.set(0, idx, M31::from(tx.behavioral.avg_value_24h & 0x7FFF_FFFF));
    idx += 1;
    input.set(0, idx, M31::from(tx.behavioral.max_value_24h & 0x7FFF_FFFF));
    idx += 1;

    // Features 45-47: Reserved
    // idx += 3; (already zero from M31Matrix::new)

    // Features 48-63: Zero padding (power-of-2 alignment for MLE)
    // Already zero from M31Matrix::new

    debug_assert_eq!(idx, NUM_FEATURES - 3); // 45 features set explicitly
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use starknet_ff::FieldElement;

    fn sample_tx() -> TransactionFeatures {
        TransactionFeatures {
            target: FieldElement::from(0xDEADBEEFu64),
            value: [0, 1_000_000_000],
            selector: 0xa9059cbb, // ERC20 transfer
            calldata_prefix: [0x1234, 0x5678, 0, 0, 0, 0, 0, 0],
            calldata_len: 68,
            agent_trust_score: 15000,
            agent_strikes: 0,
            agent_age_blocks: 1000,
            target_flags: TargetFlags {
                is_verified: true,
                is_proxy: false,
                has_source: true,
                interaction_count: 50,
            },
            value_features: ValueFeatures {
                log2_value: 30,
                value_balance_ratio: 5000,
                is_max_approval: false,
                is_zero_value: false,
            },
            selector_features: SelectorFeatures {
                is_transfer: true,
                is_approve: false,
                is_swap: false,
                is_unknown: false,
            },
            behavioral: BehavioralFeatures {
                tx_frequency: 10,
                unique_targets_24h: 5,
                avg_value_24h: 500000,
                max_value_24h: 2000000,
            },
        }
    }

    #[test]
    fn test_encode_produces_correct_dimensions() {
        let tx = sample_tx();
        let encoded = encode_transaction(&tx);
        assert_eq!(encoded.rows, 1);
        assert_eq!(encoded.cols, INPUT_DIM);
        assert_eq!(encoded.data.len(), INPUT_DIM);
    }

    #[test]
    fn test_encode_is_deterministic() {
        let tx = sample_tx();
        let a = encode_transaction(&tx);
        let b = encode_transaction(&tx);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn test_encode_padding_is_zero() {
        let tx = sample_tx();
        let encoded = encode_transaction(&tx);
        // Features 48-63 should be zero (padding)
        for i in NUM_FEATURES..INPUT_DIM {
            assert_eq!(encoded.data[i], M31::from(0u32), "padding at index {i} should be zero");
        }
    }

    #[test]
    fn test_encode_selector_feature() {
        let tx = sample_tx();
        let encoded = encode_transaction(&tx);
        // Feature 16 = selector (0xa9059cbb masked to 31 bits)
        let expected = 0xa9059cbb_u32 & 0x7FFF_FFFF;
        assert_eq!(encoded.data[16], M31::from(expected));
    }

    #[test]
    fn test_encode_trust_score() {
        let tx = sample_tx();
        let encoded = encode_transaction(&tx);
        // Feature 26 = agent_trust_score
        assert_eq!(encoded.data[26], M31::from(15000u32));
    }
}
