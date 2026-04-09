"""ObelyZK Python SDK — Provable ML Inference Client.

Wraps the ObelyZK prove-server REST API for easy integration.

Usage:
    import obelyzk

    client = obelyzk.Client("http://localhost:8080")

    # Load a model
    model = client.load_model("/path/to/model.onnx")

    # Run provable inference
    result = client.infer(model.model_id, input_data=[0.1, 0.2, ...])
    print(result.output)
    print(result.proof_hash)

    # Verify
    verification = client.verify(result.proof_hash)
    assert verification.valid

    # List all proofs
    proofs = client.list_proofs()
"""

from .client import Client, InferResult, VerifyResult, ModelInfo, StoredProof

__version__ = "0.1.0"
__all__ = ["Client", "InferResult", "VerifyResult", "ModelInfo", "StoredProof"]
