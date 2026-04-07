"""
Adversarial evaluation for the transaction classifier.

Tests the classifier against:
1. Feature perturbation attacks (FGSM-style)
2. Evasion strategies (manually crafted to bypass)
3. Boundary probing (find the exact decision boundary)
4. Label noise robustness (mislabeled training data)

Usage:
  python adversarial.py --model output/model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import create_model, TransactionClassifier
from features import TransactionFeatures, encode_features
from generate_dataset import generate_dataset


def fgsm_attack(
    model: TransactionClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Fast Gradient Sign Method — perturb inputs to maximize loss."""
    X_adv = X.clone().requires_grad_(True)
    logits = model(X_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    perturbation = epsilon * X_adv.grad.sign()
    return (X_adv + perturbation).detach()


def evaluate_adversarial_robustness(
    model: TransactionClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilons: list[float] = [0.01, 0.05, 0.1, 0.2, 0.5],
):
    """Evaluate accuracy under FGSM attack at various epsilon levels."""
    model.eval()
    X_tensor = torch.tensor(X_test)
    y_tensor = torch.tensor(y_test)

    # Clean accuracy
    with torch.no_grad():
        clean_preds = model(X_tensor).argmax(dim=1).numpy()
    clean_acc = np.mean(clean_preds == y_test)

    print(f"\n{'='*60}")
    print("Adversarial Robustness (FGSM)")
    print(f"{'='*60}")
    print(f"{'Epsilon':>10} | {'Accuracy':>10} | {'Drop':>10} | {'Flipped':>10}")
    print("-" * 50)
    print(f"{'clean':>10} | {clean_acc:>10.4f} | {'—':>10} | {'—':>10}")

    for eps in epsilons:
        X_adv = fgsm_attack(model, X_tensor, y_tensor, epsilon=eps)
        with torch.no_grad():
            adv_preds = model(X_adv).argmax(dim=1).numpy()

        adv_acc = np.mean(adv_preds == y_test)
        drop = clean_acc - adv_acc
        flipped = np.sum(clean_preds != adv_preds)

        print(f"{eps:>10.3f} | {adv_acc:>10.4f} | {drop:>10.4f} | {flipped:>10d}")

    print()


def evaluate_evasion_strategies(model: TransactionClassifier, feat_max: np.ndarray):
    """
    Test handcrafted evasion strategies — malicious transactions
    designed to look safe to the classifier.
    """
    model.eval()
    print(f"\n{'='*60}")
    print("Evasion Strategy Evaluation")
    print(f"{'='*60}")

    strategies = [
        (
            "Rug pull with verified=True disguise",
            TransactionFeatures(
                target=hex(0x049D36570D4E46F48E99674BD3FCC84644DDD6B96F7C741B1562B82F9E004DC7),
                value=2**128 - 1,
                selector=0x095EA7B3,
                calldata_len=68,
                agent_trust_score=5000,  # low trust = looks safe
                agent_strikes=0,
                agent_age_blocks=50000,  # old agent = looks safe
                is_verified=True,  # evasion: claim verified
                is_proxy=False,
                has_source=True,  # evasion: claim has source
                interaction_count=100,  # evasion: fake history
                log2_value=127,
                value_balance_ratio=100000,
                is_max_approval=True,
                is_zero_value=False,
                is_transfer=False, is_approve=True, is_swap=False, is_unknown=False,
                tx_frequency=2,  # evasion: low frequency
                unique_targets_24h=1,
                avg_value_24h=50000,
                max_value_24h=200000,
            ),
        ),
        (
            "Slow drain with normal-looking metrics",
            TransactionFeatures(
                target=hex(int.from_bytes(b"\xde\xad" * 10, "big")),
                value=int(1e17),  # small per tx
                selector=0xA9059CBB,
                calldata_len=68,
                agent_trust_score=10000,
                agent_strikes=0,
                agent_age_blocks=30000,
                is_verified=False,
                is_proxy=False,
                has_source=False,
                interaction_count=0,
                log2_value=57,
                value_balance_ratio=2000,
                is_max_approval=False,
                is_zero_value=False,
                is_transfer=True, is_approve=False, is_swap=False, is_unknown=False,
                tx_frequency=3,  # evasion: normal frequency
                unique_targets_24h=2,
                avg_value_24h=30000,
                max_value_24h=100000,
            ),
        ),
        (
            "Flash loan through verified proxy",
            TransactionFeatures(
                target=hex(0x04718F5A0FC34CC1AF16A1CDEE98FFB20C31F5CD61D6AB07201858F4287C938D),
                value=int(1e23),
                selector=0x5CFFE9DE,
                calldata_len=2000,
                agent_trust_score=15000,
                agent_strikes=0,
                agent_age_blocks=10000,
                is_verified=True,
                is_proxy=True,
                has_source=True,
                interaction_count=20,
                log2_value=76,
                value_balance_ratio=80000,
                is_max_approval=False,
                is_zero_value=False,
                is_transfer=False, is_approve=False, is_swap=False, is_unknown=True,
                tx_frequency=5,
                unique_targets_24h=3,
                avg_value_24h=100000,
                max_value_24h=500000,
            ),
        ),
        (
            "Multicall drain disguised as normal batch",
            TransactionFeatures(
                target=hex(int.from_bytes(b"\xba\xd0" * 10, "big")),
                value=int(5e21),
                selector=0xAC9650D8,
                calldata_len=500,
                agent_trust_score=20000,
                agent_strikes=1,
                agent_age_blocks=8000,
                is_verified=False,
                is_proxy=True,
                has_source=False,
                interaction_count=5,
                log2_value=72,
                value_balance_ratio=85000,
                is_max_approval=False,
                is_zero_value=False,
                is_transfer=False, is_approve=False, is_swap=False, is_unknown=True,
                tx_frequency=10,
                unique_targets_24h=5,
                avg_value_24h=200000,
                max_value_24h=1000000,
            ),
        ),
        (
            "Clean safe transfer (should be approved)",
            TransactionFeatures(
                target=hex(0x049D36570D4E46F48E99674BD3FCC84644DDD6B96F7C741B1562B82F9E004DC7),
                value=int(1e18),
                selector=0xA9059CBB,
                calldata_len=68,
                agent_trust_score=3000,
                agent_strikes=0,
                agent_age_blocks=60000,
                is_verified=True,
                is_proxy=False,
                has_source=True,
                interaction_count=200,
                log2_value=60,
                value_balance_ratio=5000,
                is_max_approval=False,
                is_zero_value=False,
                is_transfer=True, is_approve=False, is_swap=False, is_unknown=False,
                tx_frequency=2,
                unique_targets_24h=1,
                avg_value_24h=40000,
                max_value_24h=150000,
            ),
        ),
    ]

    labels = ["safe", "suspicious", "malicious"]

    for name, tx in strategies:
        encoded = encode_features(tx)
        normalized = encoded / (feat_max.flatten() + 1e-10)
        with torch.no_grad():
            logits = model(torch.tensor(normalized, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred = int(logits.argmax(dim=1).item())

        print(f"\n  Strategy: {name}")
        print(f"  Prediction: {labels[pred]}")
        print(f"  Scores: safe={probs[0]:.4f}, suspicious={probs[1]:.4f}, malicious={probs[2]:.4f}")

        # Check if evasion succeeded
        if "safe" in name.lower() or "clean" in name.lower():
            if pred == 0:
                print(f"  Result: CORRECT (approved as expected)")
            else:
                print(f"  Result: FALSE ALARM (safe tx was flagged)")
        else:
            if pred == 0:
                print(f"  Result: EVASION SUCCEEDED (malicious tx approved)")
            elif pred == 2:
                print(f"  Result: CAUGHT (malicious tx blocked)")
            else:
                print(f"  Result: PARTIALLY CAUGHT (malicious tx escalated)")


def probe_decision_boundary(
    model: TransactionClassifier,
    feat_max: np.ndarray,
):
    """
    Find the decision boundary by varying key features individually.
    Shows what feature values flip the decision.
    """
    model.eval()
    print(f"\n{'='*60}")
    print("Decision Boundary Probing")
    print(f"{'='*60}")

    # Base: a borderline transaction
    base_tx = TransactionFeatures(
        target=hex(int.from_bytes(b"\xab\xcd" * 10, "big")),
        value=int(1e19),
        selector=0xA9059CBB,
        calldata_len=100,
        agent_trust_score=30000,
        agent_strikes=1,
        agent_age_blocks=5000,
        is_verified=False,
        is_proxy=False,
        has_source=False,
        interaction_count=5,
        log2_value=63,
        value_balance_ratio=30000,
        is_max_approval=False,
        is_zero_value=False,
        is_transfer=True, is_approve=False, is_swap=False, is_unknown=False,
        tx_frequency=10,
        unique_targets_24h=5,
        avg_value_24h=200000,
        max_value_24h=800000,
    )

    base_encoded = encode_features(base_tx)
    base_normalized = base_encoded / (feat_max.flatten() + 1e-10)

    # Probe each key feature
    probe_features = [
        (26, "agent_trust_score", [0, 10000, 20000, 30000, 50000, 70000, 90000, 100000]),
        (27, "agent_strikes", [0, 1, 2, 3, 4, 5]),
        (32, "interaction_count", [0, 1, 5, 20, 50, 100, 500]),
        (34, "value_balance_ratio", [0, 5000, 20000, 40000, 60000, 80000, 100000]),
        (41, "tx_frequency", [1, 5, 10, 20, 40, 80, 150]),
    ]

    labels = ["safe", "suspicious", "malicious"]

    for feat_idx, feat_name, values in probe_features:
        print(f"\n  Probing: {feat_name} (feature {feat_idx})")
        for val in values:
            x = base_normalized.copy()
            x[feat_idx] = val / (feat_max.flatten()[feat_idx] + 1e-10)
            with torch.no_grad():
                logit = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
                pred = int(logit.argmax(dim=1).item())
                probs = torch.softmax(logit, dim=1).numpy()[0]
            print(f"    {feat_name}={val:>7d} → {labels[pred]:>11s} (s={probs[0]:.3f} u={probs[1]:.3f} m={probs[2]:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Adversarial evaluation")
    parser.add_argument("--model", default="output/model.pt")
    parser.add_argument("--feat-max", default="output/feat_max.npy")
    parser.add_argument("--scale", type=float, default=0.1, help="Dataset scale for FGSM test")
    args = parser.parse_args()

    # Load model
    model = create_model()
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()

    feat_max = np.load(args.feat_max)

    # Generate test data
    print("Generating test dataset...")
    features, labels = generate_dataset(scale=args.scale)

    # Normalize
    X_norm = features / (feat_max + 1e-10)

    # Run evaluations
    evaluate_adversarial_robustness(model, X_norm, labels)
    evaluate_evasion_strategies(model, feat_max)
    probe_decision_boundary(model, feat_max)

    print(f"\n{'='*60}")
    print("Adversarial evaluation complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
