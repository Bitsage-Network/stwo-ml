"""
ObelyZK Transaction Classifier — Training Script

Trains the MLP classifier on synthetic (or real) labeled transaction data.
Exports trained weights in both PyTorch and M31-quantized format.

Usage:
  python train.py                          # Train on synthetic data
  python train.py --dataset real_data.npz  # Train on real labeled data
  python train.py --epochs 200 --lr 0.001  # Custom hyperparameters
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from model import (
    create_model,
    quantize_to_m31,
    load_from_m31,
    TransactionClassifier,
    TOTAL_WEIGHTS,
    M31_MODULUS,
)
from generate_dataset import generate_dataset


def train(
    model: TransactionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.003,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Train the classifier. Returns training history."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights to handle imbalance (safe=40K, suspicious=12K, malicious=8K)
    class_weights = torch.tensor([1.0, 3.3, 5.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_f1={val_f1:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    model = model.cpu()

    print(f"\nBest validation F1: {best_f1:.4f}")
    return history


def evaluate(model: TransactionClassifier, X_test: np.ndarray, y_test: np.ndarray):
    """Full evaluation with classification report and confusion matrix."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        preds = logits.argmax(dim=1).numpy()

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(
        y_test, preds,
        target_names=["safe", "suspicious", "malicious"],
        digits=4,
    ))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)

    f1 = f1_score(y_test, preds, average="macro")
    return f1, preds


def verify_quantization(
    model: TransactionClassifier,
    quantized: dict[str, list[int]],
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Verify quantized weights via M31 modular arithmetic simulation.

    NOTE: Python float32 matmul != M31 modular matmul. This function
    simulates M31 arithmetic to approximate what the Rust prover does.
    The definitive test is running the weights through the Rust prover.
    """
    M31 = (1 << 31) - 1

    # Convert inputs to M31 integers
    X_m31 = (X_test * M31 / (X_test.max() + 1e-10)).astype(np.int64) & M31

    w0 = np.array(quantized["layer0"], dtype=np.int64).reshape(64, 64)
    w2 = np.array(quantized["layer2"], dtype=np.int64).reshape(32, 64)
    w4 = np.array(quantized["layer4"], dtype=np.int64).reshape(3, 32)

    correct = 0
    for i in range(len(X_m31)):
        # Layer 0: matmul + ReLU (in M31)
        h1 = (X_m31[i] @ w0.T) % M31
        h1 = np.where(h1 > M31 // 2, 0, h1)  # ReLU in M31: treat > M31/2 as negative

        # Layer 2: matmul + ReLU
        h2 = (h1 @ w2.T) % M31
        h2 = np.where(h2 > M31 // 2, 0, h2)

        # Layer 4: matmul (no activation)
        out = (h2 @ w4.T) % M31

        pred = np.argmax(out)
        if pred == y_test[i]:
            correct += 1

    agreement = correct / len(y_test)
    print(f"\nM31 quantization accuracy: {agreement:.4f} ({correct}/{len(y_test)})")

    if agreement < 0.70:
        print("NOTE: M31 accuracy is approximate. Definitive test requires the Rust prover.")
        print("      The synthetic dataset separates cleanly — real data will be harder.")
    else:
        print("Quantization looks viable for M31 field arithmetic.")

    return agreement


def export_weights_rust(quantized: dict[str, list[int]], output_path: str):
    """Export M31 weights as a Rust source file for embedding."""
    with open(output_path, "w") as f:
        f.write("// Auto-generated by training/train.py — do not edit manually.\n")
        f.write("// M31-quantized weights for the transaction classifier.\n\n")

        for name, values in quantized.items():
            f.write(f"pub const {name.upper()}_WEIGHTS: [u32; {len(values)}] = [\n")
            for i in range(0, len(values), 16):
                chunk = values[i : i + 16]
                line = ", ".join(str(v) for v in chunk)
                f.write(f"    {line},\n")
            f.write("];\n\n")

        total = sum(len(v) for v in quantized.values())
        f.write(f"// Total weights: {total}\n")

    print(f"Rust weights written to {output_path}")


def export_weights_json(quantized: dict[str, list[int]], output_path: str):
    """Export M31 weights as JSON for HuggingFace / JavaScript consumption."""
    metadata = {
        "model": "obelyzk-transaction-classifier",
        "version": "1.0.0",
        "architecture": "MLP(64→64→32→3)",
        "activation": "ReLU",
        "bias": False,
        "field": "M31 (2^31 - 1)",
        "total_weights": sum(len(v) for v in quantized.values()),
        "layers": {
            "layer0": {"shape": [64, 64], "count": len(quantized["layer0"])},
            "layer2": {"shape": [64, 32], "count": len(quantized["layer2"])},
            "layer4": {"shape": [32, 3], "count": len(quantized["layer4"])},
        },
    }

    output = {"metadata": metadata, "weights": quantized}

    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"JSON weights written to {output_path} ({Path(output_path).stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train ObelyZK transaction classifier")
    parser.add_argument("--dataset", default=None, help="Path to .npz dataset (default: generate synthetic)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="output", help="Directory for trained weights")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load or generate dataset
    if args.dataset:
        print(f"Loading dataset from {args.dataset}")
        data = np.load(args.dataset)
        features, labels = data["features"], data["labels"]
    else:
        print("Generating synthetic dataset (60K samples)...")
        features, labels = generate_dataset(40000, 12000, 8000)

    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Labels: safe={np.sum(labels==0)}, suspicious={np.sum(labels==1)}, malicious={np.sum(labels==2)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42,
    )

    print(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # Normalize features to [0, 1] for training stability
    # (The M31 values can be very large — normalize for gradient descent)
    feat_max = X_train.max(axis=0, keepdims=True)
    feat_max = np.where(feat_max == 0, 1.0, feat_max)  # avoid div by zero
    X_train_norm = X_train / feat_max
    X_val_norm = X_val / feat_max
    X_test_norm = X_test / feat_max

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val_norm), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Create and train model
    model = create_model()
    print(f"\nModel: {model.count_parameters()} parameters")

    start = time.time()
    history = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f}s")

    # Evaluate on test set
    f1, preds = evaluate(model, X_test_norm, y_test)

    if f1 < 0.85:
        print(f"\nWARNING: F1 score {f1:.4f} is below 0.85 target. Consider:")
        print("  - More training data")
        print("  - Longer training (--epochs 200)")
        print("  - Lower learning rate (--lr 0.001)")
    else:
        print(f"\nF1 score {f1:.4f} meets target (>0.85)")

    # Save PyTorch model
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"\nPyTorch weights saved to {output_dir / 'model.pt'}")

    # Quantize to M31
    quantized = quantize_to_m31(model)
    total = sum(len(v) for v in quantized.values())
    print(f"Quantized: {total} M31 values")

    # Verify quantization
    verify_quantization(model, quantized, X_test_norm, y_test)

    # Export weights
    export_weights_rust(quantized, str(output_dir / "trained_weights.rs"))
    export_weights_json(quantized, str(output_dir / "weights_m31.json"))

    # Save normalization constants (needed for inference)
    np.save(output_dir / "feat_max.npy", feat_max)

    # Save training metadata
    metadata = {
        "dataset_size": len(labels),
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_val_f1": max(history["val_f1"]),
        "test_f1": float(f1),
        "training_time_s": elapsed,
        "total_weights": total,
        "quantization_agreement": float(verify_quantization(model, quantized, X_test_norm, y_test)),
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll outputs in {output_dir}/")
    print("  model.pt              — PyTorch weights")
    print("  trained_weights.rs    — Rust const arrays (embed in prover)")
    print("  weights_m31.json      — JSON for HuggingFace / JS SDK")
    print("  feat_max.npy          — Feature normalization constants")
    print("  training_metadata.json — Training stats")


if __name__ == "__main__":
    main()
