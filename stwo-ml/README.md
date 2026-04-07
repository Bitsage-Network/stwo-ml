# ObelyZK

**Cryptographic proof of ML inference, verified on Starknet in a single transaction.**

[![Rust](https://img.shields.io/badge/rust-nightly--2025--07--14-orange)](https://rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Cairo](https://img.shields.io/badge/Cairo-2.12.2-blue)](https://www.cairo-lang.org/)
[![npm](https://img.shields.io/npm/v/@obelyzk/sdk)](https://www.npmjs.com/package/@obelyzk/sdk)
[![PyPI](https://img.shields.io/pypi/v/obelyzk)](https://pypi.org/project/obelyzk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

ObelyZK proves that a transformer model (LLaMA, Qwen, Phi, SmolLM, Mistral) produced a specific output for a given input. The proof compresses into ~942 felts via recursive STARK and verifies in one Starknet transaction. No trusted setup. No multi-step coordination.

---

## Quick Start

### Path 1: TypeScript SDK

```bash
npm install @obelyzk/sdk
```

```typescript
import { createObelyzkClient } from "@obelyzk/sdk";

const client = createObelyzkClient(); // defaults to https://api.bitsage.network
const proof = await client.prove("smollm2-135m", {
  input: "Hello world",
  gpu: true,
});

console.log(proof.txHash);        // Starknet verification TX hash
console.log(proof.ioCommitment);  // Poseidon commitment over packed IO
console.log(proof.calldata);      // raw felts submitted on-chain
```

### Path 2: Python SDK

```bash
pip install obelyzk
```

```python
from obelyzk import ObelyzkClient

client = ObelyzkClient()  # defaults to https://api.bitsage.network
result = client.prove("smollm2-135m", input="Hello world", gpu=True)

print(result.tx_hash)
print(result.io_commitment)
print(result.calldata)
```

### Path 3: CLI

```bash
npm install -g @obelyzk/cli
```

```bash
obelysk prove --model smollm2-135m --input "Hello world" --on-chain
```

```
Model:     SmolLM2-135M (30 layers)
GKR proof: 102.0s
Recursive: 3.55s (942 felts)
TX hash:   0x276c6a44...
Status:    VERIFIED
```

Point any SDK or CLI at a self-hosted prover by setting `OBELYZK_API_URL`:

```bash
export OBELYZK_API_URL=http://localhost:8080
```

---

## How It Works

```
 +-----------------+     +-----------------+     +------------------+     +------------------+
 |  Model Weights  |     |  M31 Forward    |     |  GKR Sumcheck    |     |  Recursive STARK |
 |  (SafeTensors)  | --> |  Pass Execution | --> |  Prover (GPU)    | --> |  Compression     |
 |  HuggingFace    |     |  Mersenne-31    |     |  per-layer proof |     |  ~942 felts      |
 +-----------------+     +-----------------+     +------------------+     +--------+---------+
                                                                                   |
                                                                                   v
                                                                          +------------------+
                                                                          |  Starknet        |
                                                                          |  On-Chain Verify |
                                                                          |  OODS + Merkle   |
                                                                          |  FRI + PoW       |
                                                                          |  1 TX            |
                                                                          +------------------+
```

**Step 1 -- Load.** Model weights are loaded from HuggingFace SafeTensors format and quantized into the M31 (Mersenne-31) prime field.

**Step 2 -- Execute.** The full forward pass runs over M31 arithmetic: MatMul, SiLU, GELU, Softmax, RMSNorm, LayerNorm, RoPE, GQA attention, KV-cache.

**Step 3 -- Prove.** Each operation becomes a GKR sumcheck circuit. Weight commitments are bound via aggregated Poseidon2 Merkle roots through an oracle sumcheck. The prover generates an interactive proof for every layer in the computation graph.

**Step 4 -- Compress.** The GKR proof tree (~46,148 felts for 30 layers) is wrapped in a STWO Circle STARK. The recursive verifier circuit checks the entire GKR proof inside the STARK, producing ~942 felts of calldata.

**Step 5 -- Verify.** A Cairo contract on Starknet performs full STARK verification: OODS sampling, Merkle decommitment, FRI layer folding, and proof-of-work validation. One transaction. No trust assumptions.

**Security:** M31 base field with QM31 quartic extension provides 128-bit security. Poseidon2 Merkle trees commit to weights, IO, and proof data.

---

## Performance

Benchmarked on real hardware. All times include full GKR sumcheck + recursive STARK compression.

| Model | Layers | Params | GPU | GKR Prove | Recursive | Total | Calldata |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 1 | 135M | A10G | 3.3s | 0.18s | 3.5s | 942 felts |
| SmolLM2-135M | 30 | 135M | A10G | 102s | 3.55s | ~106s | 942 felts |
| Qwen2-0.5B | 1 | 494M | A10G | 3.0s | 0.20s | 3.2s | 942 felts |
| Qwen3-14B | 40 | 14B | H100 | 103s | 3.6s | ~107s | 942 felts |

Key observations:
- Recursive STARK adds 0.18--3.6s depending on model depth
- Output calldata is ~942 felts regardless of model size (49x compression from GKR)
- GPU acceleration uses CUDA for Merkle commits, MLE restriction, and FRI; CPU SIMD for FFT

---

## On-Chain Verification

Every proof is verified trustlessly on Starknet. The Cairo contract performs complete cryptographic STARK verification -- no optimistic assumptions, no fraud proofs, no committee.

### Deployed Contracts (Starknet Sepolia)

| Contract | Address |
|---|---|
| Recursive Verifier | [`0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`](https://sepolia.voyager.online/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) |
| Class Hash | `0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5` |
| Deployer | [`0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344`](https://sepolia.voyager.online/contract/0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344) |

### Verified Transaction

The SmolLM2-135M (30-layer) recursive proof was verified on-chain:

- **TX:** [`0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc`](https://sepolia.starkscan.co/tx/0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc)
- **Calldata:** 942 felts (compressed from 46,148 GKR felts)
- **Verification:** OODS + Merkle decommitment + FRI folding + PoW

### Verify a Proof Programmatically

```typescript
import { createObelyzkClient } from "@obelyzk/sdk";

const client = createObelyzkClient();
const status = await client.verify("0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc");
console.log(status.verified); // true
```

Or query the contract directly:

```bash
starkli call 0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7 \
  is_verified 0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc
```

See [docs/ON_CHAIN_VERIFICATION.md](docs/ON_CHAIN_VERIFICATION.md) for the full protocol specification, calldata encoding, and contract ABI.

---

## Supported Models

| Model | Architecture | Params | Hidden Size | Status |
|---|---|---|---|---|
| SmolLM2-135M | LLaMA-style | 135M | 576 | Fully verified on-chain |
| Qwen2-0.5B | Qwen | 494M | 896 | Proven, on-chain ready |
| Llama-3.2-3B | LLaMA | 3B | 3,072 | Proven, on-chain ready |
| Phi-3-mini | Phi | 3.8B | 3,072 | Proven, on-chain ready |
| Mistral-7B | Mistral | 7B | 4,096 | Proven, on-chain ready |
| Mixtral-8x7B | MoE | 47B | 4,096 | MoE routing proven |
| Qwen3-14B | Qwen | 14B | 5,120 | Proven on H100 |

Any HuggingFace transformer with SafeTensors weights is supported. The HF loader auto-detects architecture, tensor naming conventions, and MoE routing.

---

## SDKs

| Package | Install | Docs |
|---|---|---|
| `@obelyzk/sdk` (TypeScript) | `npm install @obelyzk/sdk` | [sdk/typescript/README.md](sdk/typescript/README.md) |
| `obelyzk` (Python) | `pip install obelyzk` | [sdk/python/README.md](sdk/python/README.md) |
| `@obelyzk/cli` | `npm install -g @obelyzk/cli` | [sdk/cli/README.md](sdk/cli/README.md) |

All SDKs default to `https://api.bitsage.network`. Set `OBELYZK_API_URL` to point at a self-hosted prover.

The hosted API is live at `https://api.bitsage.network` and `https://prover.bitsage.network` with HTTPS, Bearer token authentication, and a rate limit of 60 requests/minute.

---

## Self-Hosting

Run your own GPU prover. Requires NVIDIA GPU with CUDA 12+.

### Setup

```bash
git clone https://github.com/obelyzk/stwo-ml.git
cd stwo-ml
./scripts/setup.sh   # installs Rust nightly, checks CUDA, builds release
```

### Prove a Model

```bash
# Download model and generate recursive proof
obelysk prove --model smollm2-135m --gkr --recursive --on-chain

# Or specify a local model directory
obelysk prove --model-dir ./models/smollm2-135m --gkr --recursive --on-chain
```

### Run as a Server

```bash
obelysk serve --port 8080 --gpu
```

Exposes:
- `POST /prove` -- submit a proving request
- `GET /ws` -- WebSocket endpoint for real-time proof streaming
- `GET /` -- web dashboard with live proof visualization

For production deployment with HTTPS, nginx, and certbot:

```bash
DOMAIN=api.example.com ./scripts/deploy_api.sh
```

See [scripts/pipeline/GETTING_STARTED.md](scripts/pipeline/GETTING_STARTED.md) for VM deployment scripts, Docker setup, and multi-GPU configuration.

---

## Environment Variables

The most important configuration options. Full reference: [docs/ENV_VARS.md](docs/ENV_VARS.md).

| Variable | Default | Description |
|---|---|---|
| `OBELYZK_API_URL` | `https://api.bitsage.network` | Prover API endpoint for SDKs and CLI |
| `STARKNET_PRIVATE_KEY` | -- | Account private key for on-chain TX submission |
| `STARKNET_RPC` | Alchemy Sepolia | Starknet JSON-RPC endpoint |
| `RECURSIVE_CONTRACT` | Deployed address | Recursive verifier contract address |
| `STWO_GPU_MERKLE_THRESHOLD` | `4096` | Leaf count threshold for GPU Merkle acceleration |
| `STWO_WEIGHT_BINDING` | `aggregated` | Weight binding mode: `aggregated`, `individual`, `sequential` |
| `STWO_PROFILE` | `0` | Enable phase profiling (`1` to activate) |

---

## Development

### Prerequisites

- Rust nightly-2025-07-14 (`rustup toolchain install nightly-2025-07-14`)
- CUDA 12+ (for GPU features)
- Scarb 2.12.2 (for Cairo contract compilation)

### Build

```bash
# CPU-only build
cargo build --release

# GPU build (requires CUDA)
cargo build --release --features gpu,cuda-runtime

# Full feature set
cargo build --release --features std,gpu,cuda-runtime,multi-gpu,onnx,safetensors,model-loading,audit,proof-stream
```

### Test

```bash
# Core tests (935+)
cargo test --features std

# End-to-end audit tests
cargo test --features std,audit -- e2e_audit

# GPU tests (requires CUDA)
cargo test --features std,gpu,cuda-runtime -- gpu
```

### Project Structure

```
stwo-ml/
  src/
    bin/obelysk.rs       # CLI entrypoint
    gkr/                 # GKR sumcheck prover + verifier
    compiler/            # Model compiler (HF -> GKR circuit)
    gpu/                 # CUDA kernels + GPU sumcheck
    recursive/           # Recursive STARK wrapper
    starknet.rs          # On-chain calldata builder
    aggregation.rs       # Proof aggregation + weight binding
    tui/                 # Terminal dashboard
  contracts/             # Cairo verifier contracts
  sdk/                   # TypeScript + Python SDKs
  scripts/               # Deployment + benchmarking
  docs/                  # Protocol specs + guides
```

### Contributing

1. Fork the repo
2. Create a feature branch from `development`
3. Run `cargo test --features std` and ensure all tests pass
4. Submit a PR against `development`

---

## Links

| Resource | URL |
|---|---|
| Paper | [obelyzk-paper.pdf](obelyzk-paper.pdf) |
| Recursive STARK Spec | [docs/RECURSIVE_STARK.md](docs/RECURSIVE_STARK.md) |
| On-Chain Verification | [docs/ON_CHAIN_VERIFICATION.md](docs/ON_CHAIN_VERIFICATION.md) |
| GKR Protocol | [docs/gkr-protocol.md](docs/gkr-protocol.md) |
| GPU Acceleration | [docs/gpu-acceleration.md](docs/gpu-acceleration.md) |
| Transformer Architecture | [docs/transformer-architecture.md](docs/transformer-architecture.md) |
| Sepolia Explorer | [Starkscan](https://sepolia.starkscan.co/tx/0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc) |
| npm | [@obelyzk/sdk](https://www.npmjs.com/package/@obelyzk/sdk) |
| PyPI | [obelyzk](https://pypi.org/project/obelyzk/) |

---

## License

Apache-2.0
