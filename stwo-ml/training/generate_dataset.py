"""
Synthetic transaction dataset generator — v2 (production-grade).

Generates labeled transactions using a realistic attack taxonomy with:
- 12 attack patterns across 3 severity tiers
- Gaussian noise on all continuous features
- Borderline cases designed to be hard to classify
- Evasion attempts (malicious txs disguised as safe)
- Feature interactions that require nonlinear decision boundaries
- Class overlap to prevent trivial separation

Labels:
  0 = safe        (routine, known counterparties, normal values)
  1 = suspicious  (anomalous but not conclusively malicious)
  2 = malicious   (known attack patterns, high confidence)

Target: 100K samples with realistic class overlap.
"""

import numpy as np
from dataclasses import dataclass, field
from features import TransactionFeatures, encode_features

RNG = np.random.default_rng(2026)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

TRANSFER = 0xA9059CBB
TRANSFER_FROM = 0x23B872DD
APPROVE = 0x095EA7B3
SWAP_EXACT = 0x38ED1739
SWAP_ETH = 0x7FF36AB5
MULTICALL = 0xAC9650D8
FLASH_LOAN = 0x5CFFE9DE

KNOWN_VERIFIED = [
    0x049D36570D4E46F48E99674BD3FCC84644DDD6B96F7C741B1562B82F9E004DC7,
    0x04718F5A0FC34CC1AF16A1CDEE98FFB20C31F5CD61D6AB07201858F4287C938D,
    0x053C91253BC9682C04929CA02ED00B3E423F6710D2EE7E0D5EBB06F3ECF368A8,
    0x03FE2B97C1FD336E750087D68B9B867997FD64A2661FF3CA5A7C771641E8E7AC,
]


def _noise(base: float, std_pct: float = 0.3) -> int:
    """Add Gaussian noise to a base value. Clamp to non-negative."""
    noisy = base * (1.0 + RNG.normal(0, std_pct))
    return max(0, int(noisy))


def _rand_address() -> int:
    return int.from_bytes(RNG.bytes(20), "big")


def _rand_bool(p: float = 0.5) -> bool:
    return bool(RNG.random() < p)


# ═══════════════════════════════════════════════════════════════════════
# SAFE patterns (label=0)
# ═══════════════════════════════════════════════════════════════════════

def safe_routine_transfer() -> TransactionFeatures:
    """Normal token transfer to a known verified contract."""
    return TransactionFeatures(
        target=hex(RNG.choice(KNOWN_VERIFIED)),
        value=_noise(1e18, 0.5),
        selector=TRANSFER,
        calldata_len=_noise(68, 0.1),
        agent_trust_score=_noise(5000, 0.5),
        agent_strikes=0,
        agent_age_blocks=_noise(50000, 0.3),
        is_verified=True,
        is_proxy=_rand_bool(0.2),
        has_source=True,
        interaction_count=_noise(100, 0.5),
        log2_value=60,
        value_balance_ratio=_noise(5000, 0.5),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=True, is_approve=False, is_swap=False, is_unknown=False,
        tx_frequency=_noise(3, 0.5),
        unique_targets_24h=_noise(2, 0.5),
        avg_value_24h=_noise(50000, 0.4),
        max_value_24h=_noise(200000, 0.4),
    )


def safe_small_approve() -> TransactionFeatures:
    """Small approval to a verified DEX."""
    return TransactionFeatures(
        target=hex(RNG.choice(KNOWN_VERIFIED)),
        value=_noise(1e17, 0.5),
        selector=APPROVE,
        calldata_len=68,
        agent_trust_score=_noise(3000, 0.5),
        agent_strikes=0,
        agent_age_blocks=_noise(80000, 0.3),
        is_verified=True,
        is_proxy=_rand_bool(0.3),
        has_source=True,
        interaction_count=_noise(200, 0.5),
        log2_value=57,
        value_balance_ratio=_noise(2000, 0.5),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=False, is_approve=True, is_swap=False, is_unknown=False,
        tx_frequency=_noise(2, 0.5),
        unique_targets_24h=_noise(1, 0.5),
        avg_value_24h=_noise(30000, 0.4),
        max_value_24h=_noise(100000, 0.4),
    )


def safe_regular_swap() -> TransactionFeatures:
    """Normal DEX swap at moderate value."""
    return TransactionFeatures(
        target=hex(RNG.choice(KNOWN_VERIFIED)),
        value=_noise(5e18, 0.6),
        selector=RNG.choice([SWAP_EXACT, SWAP_ETH]),
        calldata_len=_noise(228, 0.2),
        agent_trust_score=_noise(8000, 0.5),
        agent_strikes=0,
        agent_age_blocks=_noise(30000, 0.4),
        is_verified=True,
        is_proxy=True,
        has_source=True,
        interaction_count=_noise(50, 0.5),
        log2_value=62,
        value_balance_ratio=_noise(10000, 0.5),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=False, is_approve=False, is_swap=True, is_unknown=False,
        tx_frequency=_noise(5, 0.5),
        unique_targets_24h=_noise(3, 0.5),
        avg_value_24h=_noise(100000, 0.4),
        max_value_24h=_noise(500000, 0.4),
    )


def safe_zero_value_call() -> TransactionFeatures:
    """Zero-value function call (view, claim, etc.)."""
    return TransactionFeatures(
        target=hex(RNG.choice(KNOWN_VERIFIED)),
        value=0,
        selector=int.from_bytes(RNG.bytes(4), "big"),
        calldata_len=_noise(100, 0.5),
        agent_trust_score=_noise(2000, 0.5),
        agent_strikes=0,
        agent_age_blocks=_noise(60000, 0.3),
        is_verified=True,
        is_proxy=_rand_bool(0.4),
        has_source=True,
        interaction_count=_noise(300, 0.5),
        log2_value=0,
        value_balance_ratio=0,
        is_max_approval=False,
        is_zero_value=True,
        is_transfer=False, is_approve=False, is_swap=False, is_unknown=True,
        tx_frequency=_noise(4, 0.5),
        unique_targets_24h=_noise(2, 0.5),
        avg_value_24h=_noise(20000, 0.4),
        max_value_24h=_noise(80000, 0.4),
    )


# ═══════════════════════════════════════════════════════════════════════
# SUSPICIOUS patterns (label=1)
# ═══════════════════════════════════════════════════════════════════════

def suspicious_new_target_high_value() -> TransactionFeatures:
    """First interaction with unknown contract at above-average value."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(1e20, 0.6),
        selector=RNG.choice([TRANSFER, APPROVE, SWAP_EXACT]),
        calldata_len=_noise(100, 0.5),
        agent_trust_score=_noise(25000, 0.4),
        agent_strikes=_noise(1, 0.5),
        agent_age_blocks=_noise(5000, 0.5),
        is_verified=_rand_bool(0.4),
        is_proxy=_rand_bool(0.5),
        has_source=_rand_bool(0.3),
        interaction_count=0,
        log2_value=66,
        value_balance_ratio=_noise(40000, 0.3),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=_rand_bool(0.3), is_approve=_rand_bool(0.3),
        is_swap=_rand_bool(0.2), is_unknown=_rand_bool(0.2),
        tx_frequency=_noise(15, 0.5),
        unique_targets_24h=_noise(8, 0.5),
        avg_value_24h=_noise(300000, 0.4),
        max_value_24h=_noise(1500000, 0.4),
    )


def suspicious_large_approve_unverified() -> TransactionFeatures:
    """Large (but not max) approval to an unverified contract."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(1e21, 0.5),
        selector=APPROVE,
        calldata_len=68,
        agent_trust_score=_noise(30000, 0.4),
        agent_strikes=_noise(1, 0.5),
        agent_age_blocks=_noise(3000, 0.5),
        is_verified=False,
        is_proxy=_rand_bool(0.6),
        has_source=_rand_bool(0.3),
        interaction_count=_noise(3, 0.5),
        log2_value=70,
        value_balance_ratio=_noise(50000, 0.3),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=False, is_approve=True, is_swap=False, is_unknown=False,
        tx_frequency=_noise(10, 0.5),
        unique_targets_24h=_noise(6, 0.5),
        avg_value_24h=_noise(200000, 0.4),
        max_value_24h=_noise(800000, 0.4),
    )


def suspicious_high_frequency_burst() -> TransactionFeatures:
    """Sudden burst of transactions from normally quiet agent."""
    return TransactionFeatures(
        target=hex(RNG.choice(KNOWN_VERIFIED) if _rand_bool(0.5) else _rand_address()),
        value=_noise(5e18, 0.6),
        selector=RNG.choice([TRANSFER, SWAP_EXACT, MULTICALL]),
        calldata_len=_noise(200, 0.5),
        agent_trust_score=_noise(35000, 0.4),
        agent_strikes=_noise(1, 0.5),
        agent_age_blocks=_noise(10000, 0.4),
        is_verified=_rand_bool(0.6),
        is_proxy=_rand_bool(0.4),
        has_source=_rand_bool(0.6),
        interaction_count=_noise(20, 0.5),
        log2_value=62,
        value_balance_ratio=_noise(25000, 0.4),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=_rand_bool(0.3), is_approve=False,
        is_swap=_rand_bool(0.3), is_unknown=_rand_bool(0.2),
        tx_frequency=_noise(40, 0.3),  # high frequency is the signal
        unique_targets_24h=_noise(15, 0.4),
        avg_value_24h=_noise(400000, 0.4),
        max_value_24h=_noise(2000000, 0.4),
    )


def suspicious_proxy_no_source() -> TransactionFeatures:
    """Interaction with a proxy contract that has no published source."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(2e19, 0.5),
        selector=int.from_bytes(RNG.bytes(4), "big"),
        calldata_len=_noise(300, 0.5),
        agent_trust_score=_noise(20000, 0.4),
        agent_strikes=0,
        agent_age_blocks=_noise(15000, 0.4),
        is_verified=False,
        is_proxy=True,
        has_source=False,
        interaction_count=_noise(5, 0.5),
        log2_value=64,
        value_balance_ratio=_noise(15000, 0.4),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=False, is_approve=False, is_swap=False, is_unknown=True,
        tx_frequency=_noise(8, 0.5),
        unique_targets_24h=_noise(4, 0.5),
        avg_value_24h=_noise(150000, 0.4),
        max_value_24h=_noise(600000, 0.4),
    )


# ═══════════════════════════════════════════════════════════════════════
# MALICIOUS patterns (label=2)
# ═══════════════════════════════════════════════════════════════════════

def malicious_infinite_approve() -> TransactionFeatures:
    """Max uint approval to unverified contract (rug pull setup)."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=2**128 - 1,
        selector=APPROVE,
        calldata_len=68,
        agent_trust_score=_noise(70000, 0.3),
        agent_strikes=_noise(3, 0.3),
        agent_age_blocks=_noise(200, 0.5),
        is_verified=False,
        is_proxy=_rand_bool(0.7),
        has_source=False,
        interaction_count=0,
        log2_value=127,
        value_balance_ratio=100000,
        is_max_approval=True,
        is_zero_value=False,
        is_transfer=False, is_approve=True, is_swap=False, is_unknown=False,
        tx_frequency=_noise(25, 0.4),
        unique_targets_24h=_noise(12, 0.4),
        avg_value_24h=_noise(1000000, 0.4),
        max_value_24h=_noise(5000000, 0.4),
    )


def malicious_full_balance_drain() -> TransactionFeatures:
    """Transfer 90-100% of balance to a fresh address."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(1e22, 0.3),
        selector=TRANSFER,
        calldata_len=68,
        agent_trust_score=_noise(75000, 0.3),
        agent_strikes=_noise(4, 0.2),
        agent_age_blocks=_noise(100, 0.5),
        is_verified=False,
        is_proxy=False,
        has_source=False,
        interaction_count=0,
        log2_value=73,
        value_balance_ratio=_noise(95000, 0.05),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=True, is_approve=False, is_swap=False, is_unknown=False,
        tx_frequency=_noise(50, 0.3),
        unique_targets_24h=_noise(25, 0.3),
        avg_value_24h=_noise(3000000, 0.3),
        max_value_24h=_noise(15000000, 0.3),
    )


def malicious_flash_loan_attack() -> TransactionFeatures:
    """Flash loan with complex calldata to unverified contract."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(1e23, 0.4),
        selector=FLASH_LOAN,
        calldata_len=_noise(2000, 0.3),
        agent_trust_score=_noise(60000, 0.3),
        agent_strikes=_noise(2, 0.3),
        agent_age_blocks=_noise(300, 0.5),
        is_verified=False,
        is_proxy=_rand_bool(0.5),
        has_source=False,
        interaction_count=0,
        log2_value=76,
        value_balance_ratio=_noise(80000, 0.2),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=False, is_approve=False, is_swap=False, is_unknown=True,
        tx_frequency=_noise(80, 0.3),
        unique_targets_24h=_noise(40, 0.3),
        avg_value_24h=_noise(5000000, 0.3),
        max_value_24h=_noise(20000000, 0.3),
    )


def malicious_multicall_drain() -> TransactionFeatures:
    """Multicall bundling approve+transferFrom in one tx."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(5e21, 0.4),
        selector=MULTICALL,
        calldata_len=_noise(500, 0.3),
        agent_trust_score=_noise(65000, 0.3),
        agent_strikes=_noise(3, 0.3),
        agent_age_blocks=_noise(150, 0.5),
        is_verified=False,
        is_proxy=True,
        has_source=False,
        interaction_count=_noise(1, 0.5),
        log2_value=72,
        value_balance_ratio=_noise(85000, 0.15),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=False, is_approve=False, is_swap=False, is_unknown=True,
        tx_frequency=_noise(60, 0.3),
        unique_targets_24h=_noise(30, 0.3),
        avg_value_24h=_noise(4000000, 0.3),
        max_value_24h=_noise(10000000, 0.3),
    )


# ═══════════════════════════════════════════════════════════════════════
# EVASION patterns (malicious disguised as safe — hardest to classify)
# ═══════════════════════════════════════════════════════════════════════

def evasion_slow_drain() -> TransactionFeatures:
    """Malicious: small transfers over time that sum to drain (looks safe individually)."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(5e17, 0.5),  # small value per tx
        selector=TRANSFER,
        calldata_len=68,
        agent_trust_score=_noise(40000, 0.4),  # moderate — not obviously bad
        agent_strikes=_noise(1, 0.5),
        agent_age_blocks=_noise(2000, 0.5),
        is_verified=_rand_bool(0.3),  # sometimes targets verified contracts
        is_proxy=_rand_bool(0.3),
        has_source=_rand_bool(0.3),
        interaction_count=_noise(2, 0.5),
        log2_value=59,
        value_balance_ratio=_noise(3000, 0.5),  # looks small per tx
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=True, is_approve=False, is_swap=False, is_unknown=False,
        # KEY SIGNAL: high frequency + many unique targets = slow drain
        tx_frequency=_noise(30, 0.3),
        unique_targets_24h=_noise(20, 0.3),
        avg_value_24h=_noise(500000, 0.3),
        max_value_24h=_noise(2000000, 0.3),
    )


def evasion_verified_proxy_exploit() -> TransactionFeatures:
    """Malicious: exploit through a verified proxy (looks safe because verified)."""
    return TransactionFeatures(
        target=hex(RNG.choice(KNOWN_VERIFIED)),  # targets a REAL verified contract
        value=_noise(1e21, 0.4),
        selector=int.from_bytes(RNG.bytes(4), "big"),  # unknown selector
        calldata_len=_noise(1000, 0.3),  # complex calldata
        agent_trust_score=_noise(55000, 0.3),
        agent_strikes=_noise(2, 0.3),
        agent_age_blocks=_noise(500, 0.5),
        is_verified=True,  # the target IS verified — that's the evasion
        is_proxy=True,
        has_source=True,
        interaction_count=_noise(10, 0.5),
        log2_value=70,
        value_balance_ratio=_noise(70000, 0.2),  # but value ratio is high
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=False, is_approve=False, is_swap=False, is_unknown=True,
        tx_frequency=_noise(45, 0.3),
        unique_targets_24h=_noise(15, 0.3),
        avg_value_24h=_noise(2000000, 0.3),
        max_value_24h=_noise(8000000, 0.3),
    )


# ═══════════════════════════════════════════════════════════════════════
# BORDERLINE patterns (deliberately ambiguous — tests decision boundary)
# ═══════════════════════════════════════════════════════════════════════

def borderline_safe_ish() -> TransactionFeatures:
    """Borderline safe: new target but small value, verified, some history."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(1e18, 0.6),
        selector=RNG.choice([TRANSFER, SWAP_EXACT]),
        calldata_len=_noise(100, 0.5),
        agent_trust_score=_noise(15000, 0.5),
        agent_strikes=0,
        agent_age_blocks=_noise(20000, 0.4),
        is_verified=_rand_bool(0.5),
        is_proxy=_rand_bool(0.3),
        has_source=_rand_bool(0.5),
        interaction_count=_noise(5, 0.5),
        log2_value=60,
        value_balance_ratio=_noise(10000, 0.5),
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=_rand_bool(0.5), is_approve=False,
        is_swap=_rand_bool(0.3), is_unknown=_rand_bool(0.2),
        tx_frequency=_noise(7, 0.5),
        unique_targets_24h=_noise(4, 0.5),
        avg_value_24h=_noise(80000, 0.5),
        max_value_24h=_noise(300000, 0.5),
    )


def borderline_suspicious_ish() -> TransactionFeatures:
    """Borderline suspicious: moderate signals in multiple directions."""
    return TransactionFeatures(
        target=hex(_rand_address()),
        value=_noise(5e19, 0.5),
        selector=RNG.choice([APPROVE, MULTICALL, TRANSFER]),
        calldata_len=_noise(200, 0.5),
        agent_trust_score=_noise(35000, 0.3),
        agent_strikes=_noise(1, 0.5),
        agent_age_blocks=_noise(5000, 0.5),
        is_verified=_rand_bool(0.4),
        is_proxy=_rand_bool(0.5),
        has_source=_rand_bool(0.4),
        interaction_count=_noise(8, 0.5),
        log2_value=66,
        value_balance_ratio=_noise(35000, 0.3),
        is_max_approval=_rand_bool(0.1),
        is_zero_value=False,
        is_transfer=_rand_bool(0.3), is_approve=_rand_bool(0.3),
        is_swap=_rand_bool(0.2), is_unknown=_rand_bool(0.2),
        tx_frequency=_noise(20, 0.4),
        unique_targets_24h=_noise(10, 0.4),
        avg_value_24h=_noise(500000, 0.4),
        max_value_24h=_noise(2000000, 0.4),
    )


# ═══════════════════════════════════════════════════════════════════════
# Dataset generation
# ═══════════════════════════════════════════════════════════════════════

GENERATORS = {
    # Safe patterns (label=0)
    0: [
        (safe_routine_transfer, 15000),
        (safe_small_approve, 10000),
        (safe_regular_swap, 10000),
        (safe_zero_value_call, 8000),
        (borderline_safe_ish, 7000),  # borderline → labeled safe
    ],
    # Suspicious patterns (label=1)
    1: [
        (suspicious_new_target_high_value, 5000),
        (suspicious_large_approve_unverified, 4000),
        (suspicious_high_frequency_burst, 4000),
        (suspicious_proxy_no_source, 3000),
        (borderline_suspicious_ish, 4000),
    ],
    # Malicious patterns (label=2)
    2: [
        (malicious_infinite_approve, 3000),
        (malicious_full_balance_drain, 3000),
        (malicious_flash_loan_attack, 2000),
        (malicious_multicall_drain, 2000),
        (evasion_slow_drain, 3000),  # evasion → still malicious
        (evasion_verified_proxy_exploit, 2000),
    ],
}


def generate_dataset(
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the full labeled dataset.

    Args:
        scale: Multiplier for sample counts (0.1 = 10% for quick testing)

    Returns:
        (features, labels) where features is (N, 64) and labels is (N,)
    """
    features_list = []
    labels_list = []

    for label, generators in GENERATORS.items():
        for gen_fn, count in generators:
            n = max(1, int(count * scale))
            for _ in range(n):
                tx = gen_fn()
                encoded = encode_features(tx)
                features_list.append(encoded)
                labels_list.append(label)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    # Shuffle
    indices = RNG.permutation(len(labels))
    return features[indices], labels[indices]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic transaction dataset v2")
    parser.add_argument("--output", default="dataset.npz", help="Output file path")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for sample counts")
    args = parser.parse_args()

    print(f"Generating dataset (scale={args.scale})...")
    features, labels = generate_dataset(args.scale)

    n_safe = np.sum(labels == 0)
    n_susp = np.sum(labels == 1)
    n_mal = np.sum(labels == 2)
    total = len(labels)

    print(f"Dataset: {total} samples, {features.shape[1]} features")
    print(f"  safe:       {n_safe:6d} ({100*n_safe/total:.1f}%)")
    print(f"  suspicious: {n_susp:6d} ({100*n_susp/total:.1f}%)")
    print(f"  malicious:  {n_mal:6d} ({100*n_mal/total:.1f}%)")

    np.savez(args.output, features=features, labels=labels)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
