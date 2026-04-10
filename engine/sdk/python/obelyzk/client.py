"""ObelyZK REST API client for provable ML inference."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import requests
except ImportError:
    raise ImportError(
        "The 'requests' package is required. Install it with: pip install requests"
    )


@dataclass
class ModelInfo:
    """Loaded model information."""

    model_id: str
    weight_commitment: str
    num_layers: int
    input_shape: tuple[int, int]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelInfo:
        return cls(
            model_id=d["model_id"],
            weight_commitment=d["weight_commitment"],
            num_layers=d["num_layers"],
            input_shape=tuple(d["input_shape"]),
        )


@dataclass
class InferResult:
    """Result of a provable inference call."""

    proof_id: str
    output: Optional[list[float]]
    output_shape: tuple[int, int]
    io_commitment: str
    weight_commitment: str
    proof_hash: str
    verify_url: str
    num_proven_layers: int
    prove_time_ms: int
    estimated_gas: int
    calldata: Optional[list[str]]
    calldata_size: int

    @property
    def prove_time_seconds(self) -> float:
        return self.prove_time_ms / 1000.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InferResult:
        return cls(
            proof_id=d["proof_id"],
            output=d.get("output"),
            output_shape=tuple(d["output_shape"]),
            io_commitment=d["io_commitment"],
            weight_commitment=d["weight_commitment"],
            proof_hash=d["proof_hash"],
            verify_url=d["verify_url"],
            num_proven_layers=d["num_proven_layers"],
            prove_time_ms=d["prove_time_ms"],
            estimated_gas=d["estimated_gas"],
            calldata=d.get("calldata"),
            calldata_size=d["calldata_size"],
        )


@dataclass
class VerifyResult:
    """Result of a proof verification."""

    valid: bool
    proof_hash: str
    model_id: Optional[str]
    io_commitment: Optional[str]
    method: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerifyResult:
        return cls(
            valid=d["valid"],
            proof_hash=d["proof_hash"],
            model_id=d.get("model_id"),
            io_commitment=d.get("io_commitment"),
            method=d["method"],
        )


@dataclass
class StoredProof:
    """A stored proof record from the server."""

    proof_hash: str
    model_id: str
    io_commitment: str
    weight_commitment: str
    num_proven_layers: int
    prove_time_ms: int
    calldata_size: int
    created_at_epoch_ms: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoredProof:
        return cls(
            proof_hash=d["proof_hash"],
            model_id=d["model_id"],
            io_commitment=d["io_commitment"],
            weight_commitment=d["weight_commitment"],
            num_proven_layers=d["num_proven_layers"],
            prove_time_ms=d["prove_time_ms"],
            calldata_size=d["calldata_size"],
            created_at_epoch_ms=d["created_at_epoch_ms"],
        )


class ObelyZKError(Exception):
    """Error from the ObelyZK API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class Client:
    """ObelyZK prove-server client.

    Example:
        >>> client = Client("http://localhost:8080")
        >>> model = client.load_model("/path/to/model.onnx")
        >>> result = client.infer(model.model_id, [0.1, 0.2, 0.3])
        >>> print(f"Proof: {result.proof_hash}")
        >>> assert client.verify(result.proof_hash).valid
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: int = 300,
    ):
        """Initialize the client.

        Args:
            base_url: URL of the prove-server (default: http://localhost:8080).
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds (default: 300 for proving).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.request(method, url, **kwargs)
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", resp.text)
            except Exception:
                err = resp.text
            raise ObelyZKError(resp.status_code, err)
        return resp.json()

    # ── Health ────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Check server health."""
        return self._request("GET", "/health")

    # ── Models ───────────────────────────────────────────────────────────

    def load_model(self, model_path: str, description: str = "") -> ModelInfo:
        """Load an ONNX model on the server.

        Args:
            model_path: Path to the ONNX model file on the server filesystem.
            description: Optional description for model registration.

        Returns:
            ModelInfo with model_id, weight_commitment, etc.
        """
        data = self._request(
            "POST",
            "/api/v1/models",
            json={"model_path": model_path, "description": description},
        )
        return ModelInfo.from_dict(data)

    def load_hf_model(
        self, model_dir: str, num_layers: Optional[int] = None
    ) -> ModelInfo:
        """Load a HuggingFace model directory on the server.

        Args:
            model_dir: Path to the HuggingFace model directory (with config.json + safetensors).
            num_layers: Optional number of layers to load (default: all).

        Returns:
            ModelInfo with model_id, weight_commitment, etc.
        """
        payload: dict[str, Any] = {"model_dir": model_dir}
        if num_layers is not None:
            payload["num_layers"] = num_layers
        data = self._request("POST", "/api/v1/models/hf", json=payload)
        return ModelInfo.from_dict(data)

    def get_model(self, model_id: str) -> dict[str, Any]:
        """Get model info by ID."""
        return self._request("GET", f"/api/v1/models/{model_id}")

    # ── Provable Inference ───────────────────────────────────────────────

    def infer(
        self,
        model_id: str,
        input_data: list[float],
        gpu: bool = True,
        include_output: bool = True,
        include_calldata: bool = False,
    ) -> InferResult:
        """Run provable inference: input → model → output + proof.

        This is the primary API. It runs the model forward pass and generates
        a cryptographic proof (GKR sumcheck) that the output was computed
        correctly from the input using the committed model weights.

        Args:
            model_id: Model ID (from load_model or load_hf_model).
            input_data: Flat array of f32 input values matching model's input shape.
            gpu: Whether to use GPU for proving (default: True).
            include_output: Include raw output values in response (default: True).
            include_calldata: Include full Starknet calldata in response (default: False).

        Returns:
            InferResult with output, proof_hash, io_commitment, etc.

        Example:
            >>> result = client.infer("0xabc...", [0.1, 0.2, 0.3, 0.4])
            >>> print(f"Output: {result.output}")
            >>> print(f"Proof: {result.proof_hash}")
            >>> print(f"Proved in {result.prove_time_seconds:.1f}s")
        """
        data = self._request(
            "POST",
            "/api/v1/infer",
            json={
                "model_id": model_id,
                "input": input_data,
                "gpu": gpu,
                "include_output": include_output,
                "include_calldata": include_calldata,
            },
        )
        return InferResult.from_dict(data)

    # ── Chat (verifiable inference from text) ──────────────────────────

    def chat(
        self,
        prompt: str,
        model: str = "local",
        max_tokens: int = 1,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Verifiable chat inference — send a prompt, get a response with on-chain proof.

        Runs the full pipeline: tokenize → forward pass → GKR proof →
        recursive STARK → on-chain submission (if STARKNET_PRIVATE_KEY is set
        on the server).

        Args:
            prompt: User message text.
            model: Model name (default: "local" for the loaded model).
            max_tokens: Maximum tokens to generate (default: 1).
            session_id: Optional session ID for multi-turn conversations.

        Returns:
            Dict with keys: text, tx_hash, proof_hash, model_id, io_commitment,
            calldata_felts, explorer_url, prove_time_secs, trust_model.

        Example:
            >>> result = client.chat("What is 2+2?")
            >>> print(result["text"])
            >>> print(result["tx_hash"])       # Starknet TX hash
            >>> print(result["explorer_url"])   # Starkscan link
        """
        data = self._request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
                **({"session_id": session_id} if session_id else {}),
            },
        )
        choice = data.get("choices", [{}])[0]
        meta = data.get("obelyzk", {})
        return {
            "text": choice.get("message", {}).get("content", ""),
            "tx_hash": meta.get("tx_hash"),
            "proof_hash": meta.get("proof_hash"),
            "model_id": meta.get("model_id"),
            "io_commitment": meta.get("io_commitment"),
            "calldata_felts": meta.get("calldata_felts"),
            "explorer_url": meta.get("explorer_url"),
            "prove_time_secs": meta.get("prove_time_secs"),
            "trust_model": meta.get("trust_model", "unknown"),
        }

    # ── Verification ─────────────────────────────────────────────────────

    def verify(self, proof_hash: str) -> VerifyResult:
        """Verify a proof by its hash.

        Checks the prove-server's local proof store. Future: also checks
        Starknet on-chain verification status.

        Args:
            proof_hash: The proof hash (from InferResult.proof_hash).

        Returns:
            VerifyResult with valid, model_id, io_commitment, method.
        """
        data = self._request("GET", f"/api/v1/verify/{proof_hash}")
        return VerifyResult.from_dict(data)

    def list_proofs(self) -> list[StoredProof]:
        """List all proven inferences, most recent first."""
        data = self._request("GET", "/api/v1/proofs")
        return [StoredProof.from_dict(p) for p in data]

    # ── Async Proving (lower-level) ──────────────────────────────────────

    def submit_prove(
        self,
        model_id: str,
        input_data: Optional[list[float]] = None,
        gpu: bool = False,
    ) -> str:
        """Submit an async proving job. Returns job_id.

        Use get_prove_status() to poll for completion, then
        get_prove_result() to retrieve the proof.
        """
        data = self._request(
            "POST",
            "/api/v1/prove",
            json={"model_id": model_id, "input": input_data, "gpu": gpu},
        )
        return data["job_id"]

    def get_prove_status(self, job_id: str) -> dict[str, Any]:
        """Get status of an async proving job."""
        return self._request("GET", f"/api/v1/prove/{job_id}")

    def get_prove_result(self, job_id: str) -> dict[str, Any]:
        """Get the result of a completed proving job."""
        return self._request("GET", f"/api/v1/prove/{job_id}/result")

    def wait_for_proof(
        self, job_id: str, poll_interval: float = 2.0, timeout: float = 600.0
    ) -> dict[str, Any]:
        """Wait for an async proving job to complete, then return the result.

        Args:
            job_id: Job ID from submit_prove().
            poll_interval: Seconds between status polls.
            timeout: Maximum seconds to wait.

        Returns:
            The proof result dict.

        Raises:
            TimeoutError: If the job doesn't complete within timeout.
            ObelyZKError: If the job fails.
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_prove_status(job_id)
            if status["status"] == "completed":
                return self.get_prove_result(job_id)
            if status["status"] == "failed":
                raise ObelyZKError(500, f"Job failed: {status.get('error', 'unknown')}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
