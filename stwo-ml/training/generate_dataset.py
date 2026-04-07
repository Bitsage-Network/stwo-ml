"""
Synthetic transaction dataset generator.

Generates labeled transactions using heuristic rules based on known
attack patterns. This is a v1 baseline — replace with real labeled
data from Etherscan/Forta/Chainalysis for production.

Labels:
  0 = safe (routine transactions)
  1 = suspicious (unusual but not clearly malicious)
  2 = malicious (known attack patterns)

Target: 60K samples (40K safe, 12K suspicious, 8K malicious)
"""

import numpy as np
import json
import argparse
from pathlib import Path
from features import TransactionFeatures, encode_features, compute_selector_features

# Reproducible
RNG = np.random.default_rng(42)

# Known contract addresses (Starknet mainnet, for realistic feature encoding)
KNOWN_DEFI = [
    0x049D36570D4E46F48E99674BD3FCC84644DDD6B96F7C741B1562B82F9E004DC7,  # ETH
    0x04718F5A0FC34CC1AF16A1CDEE98FFB20C31F5CD61D6AB07201858F4287C938D,  # STRK
    0x053C91253BC9682C04929CA02ED00B3E423F6710D2EE7E0D5EBB06F3ECF368A8,  # USDC
]

KNOWN_MALICIOUS_PATTERNS = [
    # Rug pull: max approval to unverified contract
    {"is_approve": True, "is_max_approval": True, "is_verified": False},
    # Flash loan attack: high value, new target, high frequency
    {"log2_value": 60, "interaction_count": 0, "tx_frequency": 50},
    # Drain: transfer to fresh address with max value
    {"is_transfer": True, "value_balance_ratio": 99000, "interaction_count": 0},
]


def generate_safe_tx() -> tuple[TransactionFeatures, int]:
    """Generate a safe/routine transaction."""
    target = RNG.choice(KNOWN_DEFI)
    value = int(RNG.exponential(scale=1e18))  # typical ETH value
    selector = int(RNG.choice([0xA9059CBB, 0x095EA7B3, 0x38ED1739, 0x0]))

    sel_features = compute_selector_features(selector)
    log2_val = max(0, value.bit_length() - 1) if value > 0 else 0

    tx = TransactionFeatures(
        target=hex(target),
        value=value,
        selector=selector,
        calldata_len=int(RNG.integers(0, 256)),
        agent_trust_score=int(RNG.integers(0, 20000)),  # low trust score = safe history
        agent_strikes=0,
        agent_age_blocks=int(RNG.integers(1000, 100000)),  # established agent
        is_verified=True,
        is_proxy=bool(RNG.random() < 0.3),
        has_source=True,
        interaction_count=int(RNG.integers(5, 500)),  # many prior interactions
        log2_value=log2_val,
        value_balance_ratio=int(RNG.integers(0, 30000)),  # <30% of balance
        is_max_approval=False,
        is_zero_value=value == 0,
        tx_frequency=int(RNG.integers(1, 10)),
        unique_targets_24h=int(RNG.integers(1, 5)),
        avg_value_24h=int(RNG.integers(1000, 100000)),
        max_value_24h=int(RNG.integers(10000, 500000)),
        **sel_features,
    )
    return tx, 0  # label = safe


def generate_suspicious_tx() -> tuple[TransactionFeatures, int]:
    """Generate a suspicious but not clearly malicious transaction."""
    target = int.from_bytes(RNG.bytes(20), "big")  # random 160-bit address
    value = int(RNG.exponential(scale=1e20))  # higher than average
    selector = int(RNG.choice([0xA9059CBB, 0x095EA7B3, 0x0, 0x12345678]))

    sel_features = compute_selector_features(selector)
    log2_val = max(0, value.bit_length() - 1) if value > 0 else 0

    tx = TransactionFeatures(
        target=hex(target),
        value=value,
        selector=selector,
        calldata_len=int(RNG.integers(0, 1024)),
        agent_trust_score=int(RNG.integers(20000, 50000)),  # elevated trust score
        agent_strikes=int(RNG.integers(0, 2)),
        agent_age_blocks=int(RNG.integers(100, 5000)),  # relatively new
        is_verified=bool(RNG.random() < 0.5),
        is_proxy=bool(RNG.random() < 0.5),
        has_source=bool(RNG.random() < 0.5),
        interaction_count=int(RNG.integers(0, 20)),  # few prior interactions
        log2_value=log2_val,
        value_balance_ratio=int(RNG.integers(30000, 70000)),  # 30-70% of balance
        is_max_approval=bool(RNG.random() < 0.2),
        is_zero_value=False,
        tx_frequency=int(RNG.integers(5, 30)),
        unique_targets_24h=int(RNG.integers(3, 15)),
        avg_value_24h=int(RNG.integers(50000, 500000)),
        max_value_24h=int(RNG.integers(200000, 2000000)),
        **sel_features,
    )
    return tx, 1  # label = suspicious


def generate_malicious_tx() -> tuple[TransactionFeatures, int]:
    """Generate a clearly malicious transaction (known attack pattern)."""
    pattern = RNG.choice(3)  # pick attack type
    target = int.from_bytes(RNG.bytes(20), "big")

    if pattern == 0:
        # Rug pull: max approval to unverified contract
        value = 2**128 - 1
        selector = 0x095EA7B3
        tx = TransactionFeatures(
            target=hex(target),
            value=value,
            selector=selector,
            calldata_len=68,
            agent_trust_score=int(RNG.integers(50000, 90000)),
            agent_strikes=int(RNG.integers(2, 5)),
            agent_age_blocks=int(RNG.integers(0, 100)),  # brand new
            is_verified=False,
            is_proxy=bool(RNG.random() < 0.7),
            has_source=False,
            interaction_count=0,
            log2_value=127,
            value_balance_ratio=100000,  # 100% of balance
            is_max_approval=True,
            is_zero_value=False,
            is_transfer=False,
            is_approve=True,
            is_swap=False,
            is_unknown=False,
            tx_frequency=int(RNG.integers(20, 60)),
            unique_targets_24h=int(RNG.integers(10, 50)),
            avg_value_24h=int(RNG.integers(500000, 5000000)),
            max_value_24h=int(RNG.integers(2000000, 10000000)),
        )

    elif pattern == 1:
        # Drain: transfer full balance to fresh address
        value = int(RNG.exponential(scale=1e22))
        selector = 0xA9059CBB
        tx = TransactionFeatures(
            target=hex(target),
            value=value,
            selector=selector,
            calldata_len=68,
            agent_trust_score=int(RNG.integers(60000, 100000)),
            agent_strikes=int(RNG.integers(3, 5)),
            agent_age_blocks=int(RNG.integers(0, 50)),
            is_verified=False,
            is_proxy=False,
            has_source=False,
            interaction_count=0,
            log2_value=max(0, value.bit_length() - 1),
            value_balance_ratio=int(RNG.integers(90000, 100000)),
            is_max_approval=False,
            is_zero_value=False,
            is_transfer=True,
            is_approve=False,
            is_swap=False,
            is_unknown=False,
            tx_frequency=int(RNG.integers(30, 100)),
            unique_targets_24h=int(RNG.integers(20, 100)),
            avg_value_24h=int(RNG.integers(1000000, 10000000)),
            max_value_24h=int(RNG.integers(5000000, 50000000)),
        )

    else:
        # Flash loan / manipulation: unusual selector, high frequency, new target
        selector = int(RNG.integers(1, 2**32))
        value = int(RNG.exponential(scale=1e21))
        tx = TransactionFeatures(
            target=hex(target),
            value=value,
            selector=selector,
            calldata_len=int(RNG.integers(256, 4096)),  # complex calldata
            agent_trust_score=int(RNG.integers(40000, 80000)),
            agent_strikes=int(RNG.integers(1, 4)),
            agent_age_blocks=int(RNG.integers(0, 200)),
            is_verified=False,
            is_proxy=bool(RNG.random() < 0.5),
            has_source=False,
            interaction_count=0,
            log2_value=max(0, value.bit_length() - 1),
            value_balance_ratio=int(RNG.integers(50000, 100000)),
            is_max_approval=False,
            is_zero_value=False,
            is_transfer=False,
            is_approve=False,
            is_swap=False,
            is_unknown=True,
            tx_frequency=int(RNG.integers(40, 200)),  # very high frequency
            unique_targets_24h=int(RNG.integers(30, 200)),
            avg_value_24h=int(RNG.integers(2000000, 20000000)),
            max_value_24h=int(RNG.integers(10000000, 100000000)),
        )

    return tx, 2  # label = malicious


def generate_dataset(
    n_safe: int = 40000,
    n_suspicious: int = 12000,
    n_malicious: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the full labeled dataset.
    Returns (features, labels) where features is (N, 64) and labels is (N,).
    """
    features_list = []
    labels_list = []

    generators = [
        (generate_safe_tx, n_safe),
        (generate_suspicious_tx, n_suspicious),
        (generate_malicious_tx, n_malicious),
    ]

    for gen_fn, count in generators:
        for _ in range(count):
            tx, label = gen_fn()
            encoded = encode_features(tx)
            features_list.append(encoded)
            labels_list.append(label)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    # Shuffle
    indices = RNG.permutation(len(labels))
    return features[indices], labels[indices]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction dataset")
    parser.add_argument("--output", default="dataset.npz", help="Output file path")
    parser.add_argument("--safe", type=int, default=40000, help="Number of safe samples")
    parser.add_argument("--suspicious", type=int, default=12000)
    parser.add_argument("--malicious", type=int, default=8000)
    args = parser.parse_args()

    print(f"Generating dataset: {args.safe} safe, {args.suspicious} suspicious, {args.malicious} malicious")
    features, labels = generate_dataset(args.safe, args.suspicious, args.malicious)

    print(f"Dataset shape: features={features.shape}, labels={labels.shape}")
    print(f"Label distribution: safe={np.sum(labels==0)}, suspicious={np.sum(labels==1)}, malicious={np.sum(labels==2)}")

    np.savez(args.output, features=features, labels=labels)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
