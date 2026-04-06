# ObelyZK: Getting Started

Prove that a machine learning model ran correctly, with a verifiable ZK proof on Starknet.
ObelyZK generates STARK proofs of LLM inference and submits them on-chain in a single
recursive transaction.

---

## Prerequisites

You need one of the following, depending on your integration path:

| Runtime | Version | Required for |
|---------|---------|--------------|
| Node.js | 18+ | SDK (TypeScript), CLI install script |
| Python | 3.9+ | SDK (Python) |
| Rust | nightly-2025-07-14 | Self-hosted prover, building from source |
| CUDA | 12+ | Self-hosted GPU proving (optional) |

---

## Option 1: Use the Hosted Prover (Easiest)

The hosted GPU fleet at `https://api.obelysk.com` handles proving and on-chain submission.
No GPU, no Rust toolchain, no model downloads.

### TypeScript

```bash
npm install @obelyzk/sdk
```

```typescript
import { createClient } from "@obelyzk/sdk";

const client = createClient({ apiKey: process.env.OBELYZK_API_KEY });

const result = await client.prove({
  model: "smollm2-135m",
  input: "The capital of France is",
  recursive: true,
});

console.log("Output:", result.output);
console.log("Proof hash:", result.proofHash);
console.log("TX hash:", result.txHash);
console.log("Verify:", `https://sepolia.starkscan.co/tx/${result.txHash}`);
```

### Python

```bash
pip install obelyzk
```

```python
import obelyzk

client = obelyzk.Client(api_key=os.environ["OBELYZK_API_KEY"])

result = client.prove(
    model="smollm2-135m",
    input="The capital of France is",
    recursive=True,
)

print(f"Output: {result.output}")
print(f"Proof hash: {result.proof_hash}")
print(f"TX hash: {result.tx_hash}")
print(f"Verify: https://sepolia.starkscan.co/tx/{result.tx_hash}")
```

The SDK sends your input to the hosted GPU prover, which runs inference, generates a
STARK proof, and submits a recursive verification transaction on Starknet. You get back
the model output, proof hash, and on-chain TX hash.

---

## Option 2: CLI Proving

Install the `obelysk` CLI and prove models from your terminal. The prover runs locally
(CPU or GPU) and can submit proofs on-chain.

### Install

```bash
curl -sSf https://raw.githubusercontent.com/obelyzk/stwo-ml/main/install.sh | sh
```

### Prove a model

```bash
obelysk prove --model smollm2-135m --input "test" --recursive
```

This downloads the model weights on first run, executes inference, generates a recursive
STARK proof, and writes `proof.json` to the current directory.

### Submit on-chain

```bash
obelysk submit --proof proof.json --network sepolia
```

### List available models

```bash
obelysk models
```

### Download a model

```bash
obelysk models --download smollm2-135m
```

---

## Option 3: Self-Host a GPU Prover

Run your own prover server on a GPU machine. This gives you full control over the proving
pipeline and lets you point the SDK or CLI at your own endpoint.

### Requirements

- NVIDIA GPU: A10G, A100, or H100 (RTX 4090 also works)
- Ubuntu 22.04 or later
- CUDA 12+ with `nvcc` in PATH
- 50 GB free disk space (model weights + build artifacts)

### Setup

Clone the repository and run the interactive setup script:

```bash
git clone https://github.com/obelyzk/stwo-ml.git
cd stwo-ml
./scripts/setup.sh
```

The setup script detects your GPU, selects the appropriate build features, downloads
model weights, and compiles the prover binary.

### Run the prover server

```bash
./target/release/prove-server --port 8080
```

The server exposes a REST API for proving and attestation. It auto-detects CUDA and uses
GPU acceleration when available.

### Point the SDK at your server

```typescript
import { createClient } from "@obelyzk/sdk";

const client = createClient({
  url: "http://your-gpu-server:8080",
  apiKey: "your-local-key",
});

const result = await client.prove({
  model: "smollm2-135m",
  input: "Hello world",
  recursive: true,
});
```

```python
import obelyzk

client = obelyzk.Client(
    url="http://your-gpu-server:8080",
    api_key="your-local-key",
)

result = client.prove(model="smollm2-135m", input="Hello world", recursive=True)
```

---

## On-Chain Verification

ObelyZK generates recursive STARK proofs that verify in a single Starknet transaction.

| Property | Value |
|----------|-------|
| Proof type | Recursive STARK (default) |
| Calldata size | ~981 felts |
| Verification cost | ~$0.02 on Sepolia |
| Transaction count | 1 (recursive path) |

### Check a proof on-chain

Visit `https://sepolia.starkscan.co/tx/<tx_hash>` to inspect the verification transaction.

### Verifier contract (Sepolia)

```
0x16919296b3990c10db6d714a04d2b6a1f62f007ed93e1b5816de1033beb248c
```

### Query verification status

Call `is_recursive_proof_verified` on the contract with the proof hash:

```typescript
import { Contract, RpcProvider } from "starknet";

const provider = new RpcProvider({ nodeUrl: "https://starknet-sepolia.public.blastapi.io" });
const contract = new Contract(abi, "0x16919296b3990c10db6d714a04d2b6a1f62f007ed93e1b5816de1033beb248c", provider);

const verified = await contract.is_recursive_proof_verified(proofHash);
console.log("Verified:", verified);
```

---

## Supported Models

| Model | HuggingFace ID | Params | Hidden Size | Layers | Prove Time (A10G) |
|-------|----------------|--------|-------------|--------|-------------------|
| SmolLM2-135M | HuggingFaceTB/SmolLM2-135M | 135M | 576 | 30 | ~95s |
| Qwen2-0.5B | Qwen/Qwen2-0.5B | 500M | 896 | 24 | ~45s |
| Phi-3-mini | microsoft/Phi-3-mini-4k-instruct | 3.8B | 3072 | 32 | ~180s |
| Custom | Any LLaMA/Qwen/Phi architecture | -- | -- | -- | Varies |

Custom models can be loaded from any HuggingFace repository that uses the LLaMA, Qwen,
or Phi architecture. Point the prover at the model directory or HuggingFace ID and it
will auto-detect the architecture from `config.json`.

```bash
obelysk prove --model Qwen/Qwen2-0.5B --input "test" --recursive
```

---

## Environment Variables

These variables control proof generation and on-chain submission. Set them before running
the prover or prove-server.

| Variable | Description | Default |
|----------|-------------|---------|
| `STWO_WEIGHT_BINDING` | Weight binding mode: `aggregated`, `individual`, `sequential` | `aggregated` |
| `STWO_AGGREGATED_FULL_BINDING` | Enable full aggregated binding proof (`1` or `0`) | `1` |
| `STWO_GPU_MERKLE_THRESHOLD` | Minimum tree size for GPU Merkle acceleration | `4096` |
| `STWO_SKIP_POLICY_COMMITMENT` | Skip policy commitment check (`1` for development) | `0` |
| `STWO_ALLOW_MISSING_ACTIVATION_PROOF` | Accept proofs without activation LogUp (`1` or `0`) | `0` |
| `STWO_PROFILE` | Enable phase profiling output (`1` or `0`) | `0` |
| `STARKNET_PRIVATE_KEY` | Account private key for on-chain submission | -- |
| `STARKNET_RPC` | Starknet RPC endpoint URL | Sepolia default |

---

## Troubleshooting

**"Model not found"**
Download the model first:
```bash
obelysk models --download smollm2-135m
```

**"CUDA not detected"**
The prover falls back to CPU automatically. CPU proving works but is roughly 10x slower
than GPU. Ensure `nvcc` is in your PATH if you have a GPU:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

**"Transaction reverted"**
Check that the gas bounds are sufficient. Use the recursive proof path (default) which
fits in a single transaction. For streaming multi-TX proofs, ensure your account has at
least 5 STRK balance.

**"self-verification failed"**
Set `STWO_SKIP_POLICY_COMMITMENT=1` during development. In production, ensure the policy
commitment matches the on-chain verifier expectation.

**Out of GPU memory**
Reduce the number of layers with `--layers 2` or lower the GPU Merkle threshold:
```bash
export STWO_GPU_MERKLE_THRESHOLD=2048
```

**Weight commitment is slow on first run**
Pre-compute the weight cache so subsequent proves skip this step:
```bash
obelysk prove --model smollm2-135m --generate-cache
```

**Build fails on CUDA**
Make sure `nvcc` is in your PATH and that your CUDA version is 12 or later.

---

## API Reference

The prove-server exposes the following REST endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/prove` | Prove a model inference. Body: `{ model, input, recursive?, policy? }` |
| POST | `/api/v1/attest` | Prove and submit on-chain. Returns TX hash. |
| GET | `/health` | Server status, GPU availability, loaded models |
| GET | `/api/v1/models` | List available models with metadata |

### Example: prove via curl

```bash
curl -X POST http://localhost:8080/api/v1/prove \
  -H "Content-Type: application/json" \
  -d '{"model": "smollm2-135m", "input": "Hello", "recursive": true}'
```

### Example: attest (prove + submit on-chain)

```bash
curl -X POST http://localhost:8080/api/v1/attest \
  -H "Content-Type: application/json" \
  -d '{"model": "smollm2-135m", "input": "Hello", "recursive": true}'
```

Response:

```json
{
  "output": "Hello world ...",
  "proof_hash": "0x03af8b...",
  "tx_hash": "0x07c1a2...",
  "prove_time_ms": 9500,
  "calldata_felts": 981
}
```

---

## Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/obelyzk/stwo-ml |
| npm | https://www.npmjs.com/package/@obelyzk/sdk |
| PyPI | https://pypi.org/project/obelyzk |
| Documentation | https://docs.obelysk.com |
| Paper | https://arxiv.org/abs/2026.obelyzk |
| Starknet Contract | https://sepolia.starkscan.co/contract/0x16919296b3990c10db6d714a04d2b6a1f62f007ed93e1b5816de1033beb248c |
