# Obelysk Getting Started

Prove that an ML model ran correctly with a cryptographic STARK proof, verified on Starknet in a single transaction.

There are three ways to get started, from fastest to most flexible:

| Path | What you need | Time to first proof |
|------|---------------|---------------------|
| **Hosted API** | An API key | 2 minutes |
| **CLI** | Rust nightly, 50 GB disk | 15 minutes |
| **Self-hosted GPU prover** | NVIDIA GPU + CUDA 12+ | 30 minutes |

---

## 1. Prerequisites

All three paths submit proofs to the same on-chain verifier:

| Property | Value |
|----------|-------|
| Network | Starknet Sepolia |
| Contract | `0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7` |
| Proof type | Recursive STARK (single transaction) |

For on-chain submission or verification queries you also need:

- **Node.js 20+** with the `starknet` package
- A funded Starknet Sepolia account (private key)

---

## 2. Option 1: Hosted API (Fastest)

The hosted fleet at **https://api.bitsage.network** handles model loading, GPU proving, and on-chain submission. No local GPU or Rust toolchain required.

### Prove with curl

```bash
curl -X POST https://api.bitsage.network/api/v1/prove \
  -H "Authorization: Bearer $OBELYZK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2-135m",
    "input": "The capital of France is",
    "recursive": true
  }'
```

Response:

```json
{
  "output": "The capital of France is Paris ...",
  "proof_hash": "0x03af8b...",
  "tx_hash": "0x07c1a2...",
  "prove_time_ms": 9500,
  "calldata_felts": 981
}
```

### Prove with the TypeScript SDK

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

### Prove with the Python SDK

```bash
pip install obelyzk
```

```python
import os
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

---

## 3. Option 2: CLI

Run the prover locally from your terminal. Works on CPU (slower) or GPU (recommended).

### Install Rust nightly

```bash
rustup install nightly-2025-07-14
rustup default nightly-2025-07-14
```

### Build the CLI binary

GPU build (requires CUDA 12+):

```bash
cargo +nightly-2025-07-14 build --release \
  --bin prove-model \
  --features "std,gpu,cuda-runtime,model-loading,safetensors,cli"
```

CPU-only build (no CUDA needed):

```bash
cargo +nightly-2025-07-14 build --release \
  --bin prove-model \
  --features "std,model-loading,safetensors,cli"
```

### Download a model

```bash
mkdir -p ~/.obelysk/models/smollm2-135m && cd $_
for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  curl -sL "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/$f" -o $f
done
```

### Generate a recursive proof

```bash
export STWO_SKIP_RMS_SQ_PROOF=1
export STWO_ALLOW_MISSING_NORM_PROOF=1
export STWO_PIECEWISE_ACTIVATION=0
export STWO_ALLOW_LOGUP_ACTIVATION=1
export STWO_AGGREGATED_FULL_BINDING=1
export STWO_SKIP_BATCH_TOKENS=1

./target/release/prove-model \
  --model-dir ~/.obelysk/models/smollm2-135m \
  --gkr --format ml_gkr --recursive \
  --output proof.json
```

The prover runs inference, generates a GKR proof over every layer, wraps it in a recursive STARK, and writes `proof.json`.

### Submit the proof on-chain

```bash
cd stwo-ml
npm install starknet

STARKNET_PRIVATE_KEY=0x... \
RECURSIVE_CONTRACT=0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7 \
node scripts/submit_recursive.mjs proof.json
```

The script reads `proof.json`, encodes the recursive STARK as calldata, and sends a single transaction to the verifier contract.

---

## 4. Option 3: Self-Host a GPU Prover

Run your own `prove-server` that exposes a REST API identical to the hosted service. This gives you full control over hardware, model selection, and network configuration.

### Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA A10G or RTX 4090 | NVIDIA H100 |
| CUDA | 12.0+ | 12.4+ |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB | 100 GB |
| Rust | nightly-2025-07-14 | nightly-2025-07-14 |
| Node.js | 20+ | 22 LTS |

### Build

```bash
git clone https://github.com/obelyzk/stwo-ml.git
cd stwo-ml

rustup install nightly-2025-07-14

cargo +nightly-2025-07-14 build --release \
  --bin prove-model --bin prove-server \
  --features "std,gpu,cuda-runtime,model-loading,safetensors,cli"
```

### Download model weights

```bash
mkdir -p ~/.obelysk/models/smollm2-135m && cd $_
for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  curl -sL "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/$f" -o $f
done
```

### Start the server

```bash
./target/release/prove-server --port 8080 \
  --model-dir ~/.obelysk/models/smollm2-135m
```

### Verify the server is running

```bash
curl http://localhost:8080/health
```

### Submit a proof request

```bash
curl -X POST http://localhost:8080/api/v1/prove \
  -H 'Content-Type: application/json' \
  -d '{"model":"smollm2-135m","input":[1.0,2.0,3.0]}'
```

### Point the SDK at your server

TypeScript:

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

Python:

```python
import obelyzk

client = obelyzk.Client(
    url="http://your-gpu-server:8080",
    api_key="your-local-key",
)

result = client.prove(model="smollm2-135m", input="Hello world", recursive=True)
```

---

## 5. On-Chain Verification

Every recursive proof lands on Starknet as a single transaction. You can query the contract to check verification status.

### Submit a proof

If you generated `proof.json` locally (Option 2 or 3), submit it:

```bash
cd stwo-ml
npm install starknet

STARKNET_PRIVATE_KEY=0x... \
RECURSIVE_CONTRACT=0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7 \
node scripts/submit_recursive.mjs proof.json
```

### Query verification count

```javascript
const { RpcProvider } = require("starknet");

const provider = new RpcProvider({
  nodeUrl: "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo",
});

const count = await provider.callContract({
  contractAddress:
    "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7",
  entrypoint: "get_recursive_verification_count",
  calldata: [modelId],
});

console.log("Verification count:", count);
```

### Query whether a specific proof is verified

```javascript
const { Contract, RpcProvider } = require("starknet");

const provider = new RpcProvider({
  nodeUrl: "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo",
});

const contract = new Contract(
  abi,
  "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7",
  provider
);

const verified = await contract.is_recursive_proof_verified(proofHash);
console.log("Verified:", verified);
```

### Inspect on a block explorer

```
https://sepolia.starkscan.co/tx/<tx_hash>
```

---

## 6. Automated Node Deployment

The `deploy_node.sh` script sets up both a Juno Starknet node and a `prove-server` instance on a single machine. It handles CUDA detection, model downloads, systemd service files, and on-chain registration.

```bash
cd stwo-ml

STARKNET_PRIVATE_KEY=0x... ./scripts/deploy_node.sh
```

The script will:

1. Detect your GPU and CUDA version
2. Build `prove-model` and `prove-server` with appropriate features
3. Download SmolLM2-135M weights
4. Start `prove-server` on port 8080
5. Register the node on-chain if not already registered

### Production API Deployment

For a production-grade deployment with HTTPS, use `deploy_api.sh`:

```bash
cd stwo-ml

DOMAIN=api.example.com \
STARKNET_PRIVATE_KEY=0x... \
./scripts/deploy_api.sh
```

This script automates the full deployment stack:

1. Builds `prove-model` and `prove-server` with GPU features (auto-detects CUDA)
2. Creates a systemd service for `prove-server`
3. Configures nginx as a reverse proxy
4. Provisions HTTPS certificates via certbot (Let's Encrypt)
5. Sets up Bearer token authentication and rate limiting (60 req/min)

The hosted API at `https://api.bitsage.network` and `https://prover.bitsage.network` was deployed using this script.

---

## 7. Contract Interface Reference

The recursive verifier contract at `0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7` exposes the following entrypoints:

| Entrypoint | Type | Description |
|------------|------|-------------|
| `verify_recursive_proof` | External | Verify a recursive STARK proof. Called by `submit_recursive.mjs`. |
| `register_model_recursive` | External | Register a new model ID for recursive verification. Must be called before the first proof submission for that model. |
| `is_recursive_proof_verified` | View | Returns `true` if the given proof hash has been verified. |
| `get_recursive_verification_count` | View | Returns the total number of verified proofs for a model ID. |
| `get_proof_details` | View | Returns proof metadata (timestamp, prover address, model ID) for a verified proof hash. |
| `propose_upgrade` | External | Owner-only. Propose a contract class upgrade with a 5-minute timelock. |
| `execute_upgrade` | External | Owner-only. Execute a proposed upgrade after the timelock expires. |

---

## 8. Troubleshooting

### "Model not found"

Download the model files to the expected directory:

```bash
mkdir -p ~/.obelysk/models/smollm2-135m && cd $_
for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  curl -sL "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/$f" -o $f
done
```

### "CUDA not detected"

Install CUDA 12+ and ensure `nvcc` is in your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

The prover falls back to CPU automatically, but proving is roughly 10x slower without a GPU.

### "self-verification failed"

Set the required environment variables before running the prover:

```bash
export STWO_SKIP_RMS_SQ_PROOF=1
export STWO_ALLOW_MISSING_NORM_PROOF=1
export STWO_PIECEWISE_ACTIVATION=0
export STWO_ALLOW_LOGUP_ACTIVATION=1
export STWO_AGGREGATED_FULL_BINDING=1
export STWO_SKIP_BATCH_TOKENS=1
```

### "Transaction reverted: Model not registered"

You must register the model on-chain before submitting proofs for it. Call `register_model_recursive` on the verifier contract with the model ID first.

### "compiled_class_hash mismatch"

When declaring a new contract class, use `starkli --casm-hash` with the sequencer's expected hash. The Sepolia sequencer recompiles Sierra and may produce a different CASM hash than your local toolchain.

### Build fails on CUDA

Verify your CUDA installation:

```bash
nvcc --version   # must be 12.0 or later
nvidia-smi       # check driver version
```

If `nvcc` is present but the build still fails, confirm that `CUDA_HOME` or `CUDA_PATH` is set:

```bash
export CUDA_HOME=/usr/local/cuda
```

### Out of GPU memory

Reduce the layer count or lower the GPU Merkle threshold:

```bash
./target/release/prove-model \
  --model-dir ~/.obelysk/models/smollm2-135m \
  --gkr --format ml_gkr --recursive --layers 2 \
  --output proof.json

# Or lower the Merkle threshold
export STWO_GPU_MERKLE_THRESHOLD=2048
```

---

## 9. Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/obelyzk/stwo-ml |
| Hosted API | https://api.bitsage.network |
| npm SDK | https://www.npmjs.com/package/@obelyzk/sdk |
| PyPI SDK | https://pypi.org/project/obelyzk |
| Documentation | https://docs.obelysk.xyz |
| Paper | https://arxiv.org/abs/2026.obelyzk |
| Contract (Sepolia) | https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7 |
