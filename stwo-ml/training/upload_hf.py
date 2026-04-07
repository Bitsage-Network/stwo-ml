"""
Upload trained classifier to HuggingFace Hub.

Creates a model repository with:
  - Model card (README.md)
  - SafeTensors weights (float32)
  - M31-quantized weights (JSON)
  - Training metadata

Usage:
  python upload_hf.py --repo obelyzk/transaction-classifier-v1
  python upload_hf.py --repo obelyzk/transaction-classifier-v1 --private
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo

from model import create_model, TOTAL_WEIGHTS


MODEL_CARD = """---
library_name: obelyzk
tags:
  - zkml
  - transaction-classification
  - starknet
  - security
  - gkr-proof
  - stark-proof
license: mit
datasets:
  - synthetic
metrics:
  - f1
  - accuracy
---

# ObelyZK Transaction Classifier

A small MLP (64→64→32→3) that classifies Starknet transactions as **safe**, **suspicious**, or **malicious**.

## What makes this different

Every inference through this model can be **cryptographically proven** using GKR+STARK proofs and **verified on-chain** on Starknet. The proof guarantees that the classifier actually ran with these exact weights on the exact input — no trust required.

## Architecture

```
Input(64) → Linear(64,64) → ReLU → Linear(64,32) → ReLU → Linear(32,3)
```

- **Parameters:** 6,240 (no bias terms)
- **Field:** M31 (Mersenne-31, 2^31 - 1) for verifiable arithmetic
- **Activation:** ReLU (provable via GKR sumcheck)

## Input Features (48 + 16 zero-padding)

| Features | Description |
|----------|-------------|
| 0-7 | Target contract address (8 × 31-bit chunks) |
| 8-15 | Transaction value u256 (8 × 31-bit chunks) |
| 16 | Function selector |
| 17-24 | Calldata prefix (8 words) |
| 25 | Calldata length |
| 26-28 | Agent metadata (trust score, strikes, age) |
| 29-32 | Target flags (verified, proxy, source, interaction count) |
| 33-36 | Value features (log2, ratio, max approval, zero) |
| 37-40 | Selector features (transfer, approve, swap, unknown) |
| 41-44 | Behavioral (frequency, unique targets, avg/max value) |
| 45-63 | Reserved + zero padding |

## Output

Three scores: `[safe, suspicious, malicious]`

Threat score = `malicious / (safe + suspicious + malicious) × 100,000`

| Score Range | Decision |
|-------------|----------|
| 0 - 40,000 | Approve |
| 40,001 - 70,000 | Escalate (human review) |
| 70,001 - 100,000 | Block |

## Usage with ObelyZK

```bash
# Prove inference with this classifier
prove-model --classifier transaction-classifier-v1 --input tx_features.json --gkr

# Verify on-chain (Starknet Sepolia)
# The proof is submitted to the AgentFirewallZK contract
```

```typescript
// Via SDK
import { AgentFirewallSDK } from '@obelyzk/sdk/firewall';

const firewall = new AgentFirewallSDK({
  proverUrl: 'http://localhost:8080',
  firewallContract: '0x...',
  verifierContract: '0x...',
  rpcUrl: process.env.STARKNET_RPC,
});

const result = await firewall.classify({
  target: '0x049d365...',
  value: '1000000000000000000',
  selector: '0xa9059cbb',
});

console.log(result.decision);     // "approve" | "escalate" | "block"
console.log(result.threatScore);  // 0-100000
console.log(result.ioCommitment); // cryptographic proof hash
```

## Training

Trained on synthetic labeled transaction data (60K samples). See the training pipeline at `github.com/bitsage-network/libs/stwo-ml/training/`.

## Files

- `model.safetensors` — Float32 PyTorch weights
- `weights_m31.json` — M31-quantized weights for the ZKML prover
- `config.json` — Model architecture and metadata
- `training_metadata.json` — Training hyperparameters and metrics

## License

MIT
"""


def main():
    parser = argparse.ArgumentParser(description="Upload classifier to HuggingFace")
    parser.add_argument("--repo", required=True, help="HF repo name (e.g., obelyzk/transaction-classifier-v1)")
    parser.add_argument("--output-dir", default="output", help="Directory with trained weights")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load model
    model = create_model()
    state_dict = torch.load(output_dir / "model.pt", weights_only=True)
    model.load_state_dict(state_dict)

    # Create repo
    api = HfApi()
    create_repo(args.repo, private=args.private, exist_ok=True)
    print(f"Repository: https://huggingface.co/{args.repo}")

    # Save as SafeTensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(model.state_dict(), str(safetensors_path))
    print(f"SafeTensors: {safetensors_path}")

    # Config
    config = {
        "model_type": "obelyzk-transaction-classifier",
        "architecture": "MLP",
        "layers": [
            {"type": "Linear", "in": 64, "out": 64, "bias": False},
            {"type": "ReLU"},
            {"type": "Linear", "in": 64, "out": 32, "bias": False},
            {"type": "ReLU"},
            {"type": "Linear", "in": 32, "out": 3, "bias": False},
        ],
        "input_dim": 64,
        "output_dim": 3,
        "total_parameters": TOTAL_WEIGHTS,
        "field": "M31",
        "output_labels": ["safe", "suspicious", "malicious"],
        "thresholds": {
            "approve": 40000,
            "escalate": 70000,
            "block": 100000,
        },
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Write model card
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD)

    # Upload files
    files_to_upload = [
        (str(safetensors_path), "model.safetensors"),
        (str(config_path), "config.json"),
        (str(readme_path), "README.md"),
    ]

    # Add M31 weights if they exist
    m31_path = output_dir / "weights_m31.json"
    if m31_path.exists():
        files_to_upload.append((str(m31_path), "weights_m31.json"))

    # Add training metadata
    meta_path = output_dir / "training_metadata.json"
    if meta_path.exists():
        files_to_upload.append((str(meta_path), "training_metadata.json"))

    for local_path, repo_path in files_to_upload:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=args.repo,
        )
        print(f"Uploaded: {repo_path}")

    print(f"\nModel published at: https://huggingface.co/{args.repo}")
    print("To use: download weights_m31.json and load into the prover")


if __name__ == "__main__":
    main()
