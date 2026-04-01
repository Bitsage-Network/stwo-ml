# ObelyZK Python SDK

Python client for the ObelyZK provable inference API.

## Install

```bash
pip install obelyzk
# or from source:
pip install -e sdk/python/
```

## Quick Start

```python
import obelyzk

# Connect to prove-server
client = obelyzk.Client("http://localhost:8080")

# Load a model
model = client.load_model("/path/to/model.onnx")
print(f"Model ID: {model.model_id}")
print(f"Weight commitment: {model.weight_commitment}")

# Run provable inference
result = client.infer(
    model_id=model.model_id,
    input_data=[0.1, 0.2, 0.3, 0.4],  # must match model input shape
)

print(f"Output: {result.output}")
print(f"Proof hash: {result.proof_hash}")
print(f"Proved in {result.prove_time_seconds:.1f}s")
print(f"Calldata size: {result.calldata_size} felts")

# Verify the proof
verification = client.verify(result.proof_hash)
assert verification.valid
print(f"Verified: {verification.valid} (method: {verification.method})")

# List all proven inferences
proofs = client.list_proofs()
for p in proofs:
    print(f"  {p.proof_hash[:16]}... model={p.model_id[:16]}... layers={p.num_proven_layers}")
```

## API Reference

### `Client(base_url, api_key=None, timeout=300)`

Create a client connected to an ObelyZK prove-server.

### `client.load_model(model_path)` → `ModelInfo`

Load an ONNX model on the server.

### `client.load_hf_model(model_dir, num_layers=None)` → `ModelInfo`

Load a HuggingFace model directory.

### `client.infer(model_id, input_data, ...)` → `InferResult`

Run provable inference. Returns output + proof.

### `client.verify(proof_hash)` → `VerifyResult`

Verify a proof by hash.

### `client.list_proofs()` → `list[StoredProof]`

List all proven inferences.

## What This Proves

Every call to `client.infer()` generates a cryptographic proof (GKR sumcheck over M31 field arithmetic) that:

1. The **output** was computed by running the **committed model weights** on the **given input**
2. Every matmul, activation, normalization, and attention operation was executed correctly
3. The proof is verifiable on Starknet (on-chain) or locally (off-chain)

No IEEE 754 floating-point is used in the proving path. All arithmetic is native M31 field operations — deterministic across every platform.
