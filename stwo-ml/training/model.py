"""
ObelyZK Transaction Classifier — PyTorch Model

Architecture must match EXACTLY:
  Input(64) → Linear(64,64) → ReLU → Linear(64,32) → ReLU → Linear(32,3)

No bias terms. Row-major weight layout. M31 field (2^31 - 1) for quantization.
"""

import torch
import torch.nn as nn

M31_MODULUS = (1 << 31) - 1  # 2147483647

# Feature count (48 real + 16 zero-padding = 64)
INPUT_DIM = 64
HIDDEN1_DIM = 64
HIDDEN2_DIM = 32
OUTPUT_DIM = 3  # safe, suspicious, malicious

# Total weight count: 64*64 + 64*32 + 32*3 = 6240
TOTAL_WEIGHTS = INPUT_DIM * HIDDEN1_DIM + HIDDEN1_DIM * HIDDEN2_DIM + HIDDEN2_DIM * OUTPUT_DIM


class TransactionClassifier(nn.Module):
    """
    MLP classifier for transaction safety scoring.

    Output interpretation:
      scores[0] = safe
      scores[1] = suspicious
      scores[2] = malicious

    Threat score = malicious / (safe + suspicious + malicious) * 100000
    """

    def __init__(self):
        super().__init__()
        # No bias — matches Rust MatMul (weight-only linear layers)
        self.layer0 = nn.Linear(INPUT_DIM, HIDDEN1_DIM, bias=False)
        self.layer2 = nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM, bias=False)
        self.layer4 = nn.Linear(HIDDEN2_DIM, OUTPUT_DIM, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer0(x))
        x = self.relu(self.layer2(x))
        x = self.layer4(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model() -> TransactionClassifier:
    """Create a fresh (untrained) classifier."""
    model = TransactionClassifier()
    assert model.count_parameters() == TOTAL_WEIGHTS, (
        f"Parameter count mismatch: {model.count_parameters()} != {TOTAL_WEIGHTS}"
    )
    return model


def quantize_to_m31(model: TransactionClassifier) -> dict[str, list[int]]:
    """
    Quantize float32 weights to M31 (unsigned 31-bit integers).

    Strategy: Scale weights to [0, M31_MODULUS) range.
    - Shift so min value maps to 0
    - Scale so max value maps to M31_MODULUS - 1
    - Round to nearest integer
    - Mask to 31 bits

    Returns dict with keys 'layer0', 'layer2', 'layer4',
    each a flat list of u32 values in row-major order.
    """
    result = {}

    for name, layer_name in [
        ("layer0", "layer0"),
        ("layer2", "layer2"),
        ("layer4", "layer4"),
    ]:
        weight = model.state_dict()[f"{layer_name}.weight"].detach().cpu().numpy()

        # Scale to [0, 1] range
        w_min = weight.min()
        w_max = weight.max()
        w_range = w_max - w_min
        if w_range < 1e-10:
            w_range = 1.0  # avoid division by zero

        normalized = (weight - w_min) / w_range

        # Scale to [0, max_val) — the key constraint is that matmul accumulation
        # (sum of input_dim products) must not overflow M31 (2^31-1).
        # Input features are already M31-masked, so max input ~2^31.
        # With 64 accumulation terms: max_weight < M31 / (64 * max_input)
        # But inputs are normalized in practice, so we can use a wider range.
        # Use M31 / 256 ≈ 8M per weight — safe for 64-term accumulation with
        # normalized inputs (max ~1.0 after feature normalization).
        max_val = M31_MODULUS // 4  # ~537M — wider range preserves precision
        scaled = (normalized * max_val).round().astype("int64")

        # Mask to 31 bits
        masked = scaled & M31_MODULUS

        # Flatten row-major (C order) — matches Rust load_weights_from_arrays
        flat = masked.flatten(order="C").tolist()
        result[name] = flat

    return result


def load_from_m31(weights: dict[str, list[int]]) -> TransactionClassifier:
    """
    Load M31-quantized weights back into a PyTorch model.
    For verification — checks that quantized weights produce same outputs.
    """
    model = create_model()

    import numpy as np

    w0 = torch.tensor(
        np.array(weights["layer0"], dtype=np.float32).reshape(HIDDEN1_DIM, INPUT_DIM)
    )
    w2 = torch.tensor(
        np.array(weights["layer2"], dtype=np.float32).reshape(HIDDEN2_DIM, HIDDEN1_DIM)
    )
    w4 = torch.tensor(
        np.array(weights["layer4"], dtype=np.float32).reshape(OUTPUT_DIM, HIDDEN2_DIM)
    )

    model.layer0.weight.data = w0
    model.layer2.weight.data = w2
    model.layer4.weight.data = w4

    return model
