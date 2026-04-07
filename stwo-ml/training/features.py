"""
Feature engineering for the transaction classifier.

Must match encoder.rs EXACTLY — same 48 features in same order.
Features 48-63 are zero padding for MLE alignment.
"""

import numpy as np
from dataclasses import dataclass, field

M31_MASK = 0x7FFF_FFFF  # 2^31 - 1


@dataclass
class TransactionFeatures:
    """Raw transaction features before encoding."""

    # Core transaction data
    target: str = "0x0"  # hex address
    value: int = 0  # u256 as integer
    selector: int = 0  # u32 function selector
    calldata_prefix: list[int] = field(default_factory=lambda: [0] * 8)
    calldata_len: int = 0

    # Agent metadata
    agent_trust_score: int = 0  # 0-100000
    agent_strikes: int = 0
    agent_age_blocks: int = 0

    # Target flags
    is_verified: bool = False
    is_proxy: bool = False
    has_source: bool = False
    interaction_count: int = 0

    # Value features
    log2_value: int = 0
    value_balance_ratio: int = 0  # 0-100000 fixed point
    is_max_approval: bool = False
    is_zero_value: bool = True

    # Selector features
    is_transfer: bool = False
    is_approve: bool = False
    is_swap: bool = False
    is_unknown: bool = True

    # Behavioral features
    tx_frequency: int = 0  # txs per hour
    unique_targets_24h: int = 0
    avg_value_24h: int = 0
    max_value_24h: int = 0


# Known selectors for feature flags
TRANSFER_SELECTORS = {0xA9059CBB, 0x23B872DD}  # transfer, transferFrom
APPROVE_SELECTORS = {0x095EA7B3}  # approve
SWAP_SELECTORS = {0x38ED1739, 0x7FF36AB5, 0x18CBAFE5}  # various Uniswap


def compute_selector_features(selector: int) -> dict:
    """Derive selector boolean features."""
    return {
        "is_transfer": selector in TRANSFER_SELECTORS,
        "is_approve": selector in APPROVE_SELECTORS,
        "is_swap": selector in SWAP_SELECTORS,
        "is_unknown": selector == 0,
    }


def compute_value_features(value: int, balance: int = 0) -> dict:
    """Derive value-based features."""
    log2_val = max(0, value.bit_length() - 1) if value > 0 else 0
    ratio = int((value / balance) * 100_000) if balance > 0 else 0
    ratio = min(ratio, 100_000)

    return {
        "log2_value": log2_val,
        "value_balance_ratio": ratio,
        "is_max_approval": value >= (2**128 - 1),
        "is_zero_value": value == 0,
    }


def encode_features(tx: TransactionFeatures) -> np.ndarray:
    """
    Encode transaction features into a 64-element M31 array.
    Matches encoder.rs encode_transaction() EXACTLY.
    """
    features = np.zeros(64, dtype=np.int64)

    # Features 0-7: target address as 8 × 31-bit chunks
    target_int = int(tx.target, 16) if isinstance(tx.target, str) else tx.target
    for i in range(8):
        shift = (7 - i) * 31
        features[i] = (target_int >> shift) & M31_MASK

    # Features 8-11: value_low (lower 128 bits) as 4 × 31-bit chunks
    value_low = tx.value & ((1 << 128) - 1)
    for i in range(4):
        features[8 + i] = (value_low >> (i * 31)) & M31_MASK

    # Features 12-15: value_high (upper 128 bits) as 4 × 31-bit chunks
    value_high = (tx.value >> 128) & ((1 << 128) - 1)
    for i in range(4):
        features[12 + i] = (value_high >> (i * 31)) & M31_MASK

    # Feature 16: selector
    features[16] = tx.selector & M31_MASK

    # Features 17-24: calldata prefix
    for i in range(8):
        features[17 + i] = tx.calldata_prefix[i] & M31_MASK if i < len(tx.calldata_prefix) else 0

    # Feature 25: calldata length
    features[25] = min(tx.calldata_len, M31_MASK)

    # Features 26-28: agent metadata
    features[26] = min(tx.agent_trust_score, 100_000)
    features[27] = tx.agent_strikes
    features[28] = tx.agent_age_blocks & M31_MASK

    # Features 29-32: target flags
    features[29] = int(tx.is_verified)
    features[30] = int(tx.is_proxy)
    features[31] = int(tx.has_source)
    features[32] = min(tx.interaction_count, M31_MASK)

    # Features 33-36: value features
    features[33] = tx.log2_value
    features[34] = min(tx.value_balance_ratio, 100_000)
    features[35] = int(tx.is_max_approval)
    features[36] = int(tx.is_zero_value)

    # Features 37-40: selector features
    features[37] = int(tx.is_transfer)
    features[38] = int(tx.is_approve)
    features[39] = int(tx.is_swap)
    features[40] = int(tx.is_unknown)

    # Features 41-44: behavioral
    features[41] = min(tx.tx_frequency, M31_MASK)
    features[42] = min(tx.unique_targets_24h, M31_MASK)
    features[43] = tx.avg_value_24h & M31_MASK
    features[44] = tx.max_value_24h & M31_MASK

    # Features 45-47: reserved (zero)
    # Features 48-63: zero padding (MLE alignment)

    return features
