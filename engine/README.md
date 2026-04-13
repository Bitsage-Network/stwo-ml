# ObelyZK

**Cryptographic proof of ML inference, verified on Starknet in a single transaction.**

[![Rust](https://img.shields.io/badge/rust-nightly--2025--07--14-orange)](https://rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Cairo](https://img.shields.io/badge/Cairo-2.12.2-blue)](https://www.cairo-lang.org/)
[![npm](https://img.shields.io/npm/v/@obelyzk/sdk)](https://www.npmjs.com/package/@obelyzk/sdk)
[![PyPI](https://img.shields.io/pypi/v/obelyzk)](https://pypi.org/project/obelyzk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

ObelyZK proves that a transformer model (LLaMA, Qwen, Phi, SmolLM, Mistral) produced a specific output for a given input. The proof compresses into ~4,934 felts via two-level recursive STARK and verifies in one Starknet transaction. No trusted setup. No multi-step coordination.

The recursive STARK system provides 160-bit cryptographic security (pow_bits=20 + log_blowup=5 x n_queries=28 = 20+140 = 160) through a 48-column chain AIR with 38 constraints, including an amortized accumulator, carry-chain modular addition, and boundary constraints. Nine security layers protect against proof forgery, trace miniaturization, and selector manipulation. The two-level recursion architecture uses cairo-prove to verify 145 Hades permutations (Level 1, off-chain) and a chain STARK bound to the Level 1 commitment (Level 2, on-chain). Security exceeds AES-256 target and is quantum-resistant (Grover reduces to 2^80).

**Live on Starknet Sepolia** — multiple architectures verified on-chain:

| Model | Params | GKR | STARK | Felts | Verified TX |
|-------|--------|-----|-------|-------|-------------|
| SmolLM2-135M | 135M | 102s | 6.5s | ~4,934 | [`0x021512dd...`](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8) |
| Qwen2.5-14B | 14B | 46s | 1.2s | 946 | [`0x5ce1b4...`](https://sepolia.starkscan.co/tx/0x5ce1b41815e29a7b3dd03b77187cf32c8c5f0e2607960303174cbea303edfd3) |
| GLM-4-9B | 9B | 201s | 1.1s | 929 | [`0x542960...`](https://sepolia.starkscan.co/tx/0x542960d703a62d4beaacf0d9094ea92dc86bf326cd917c533039f4dd1eb4a30) |

> **[Full Getting Started Guide →](scripts/pipeline/GETTING_STARTED.md)**

---

## Try It Now

No SDK required. The hosted API is live at `https://api.bitsage.network`.

```bash
# Health check (no auth)
curl https://api.bitsage.network/health

# Prove from a text prompt — server tokenizes and embeds automatically
curl -X POST https://api.bitsage.network/api/v1/chat \
  -H "Authorization: Bearer $OBELYSK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"smollm2-135m","prompt":"Hello world"}'

# Or prove from raw f32 input (advanced)
curl -X POST https://api.bitsage.network/api/v1/infer \
  -H "Authorization: Bearer $OBELYSK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"smollm2-135m","input":[1.0,2.0,3.0],"gpu":true}'
```

Auth: pass `Authorization: Bearer <API_KEY>` on every request (env: `OBELYSK_API_KEY`).

Recursive Verifier Contract: [`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) (Starknet Sepolia)

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
Recursive: 6.5s (~4,934 felts)
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
 |  HuggingFace    |     |  Mersenne-31    |     |  per-layer proof |     |  ~4,934 felts    |
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

**Step 4 -- Compress.** The GKR proof tree (~46,148 felts for 30 layers) is wrapped in a two-level recursive STARK via a 48-column chain AIR with 38 constraints. Level 1 (off-chain): cairo-prove verifies 145 Hades permutations (~10s, 278K felts). Level 2 (on-chain): the chain STARK binds to the Level 1 commitment (~6.5s, ~4,934 felts). The chain AIR includes HadesPerm-level rows with carry-chain modular addition and an amortized accumulator constraint that blocks all-zeros-selector attacks, plus a hades_commitment binding layer for two-level recursion.

**Step 5 -- Verify.** A Cairo contract on Starknet performs full STARK verification: OODS sampling, Merkle decommitment, FRI layer folding, and proof-of-work validation. One transaction. No trust assumptions. The Cairo verifier at `recursive_air.cairo` implements all 38 constraints matching the Rust AIR exactly.

**Security:** 160-bit cryptographic security via PcsConfig (pow_bits=20, log_blowup=5, n_queries=28, log_last_layer_deg=0). M31 base field with QM31 quartic extension. Poseidon2 Merkle trees commit to weights, IO, and proof data. Nine security layers protect proof integrity (see below).

---

## Performance

Benchmarked on real hardware. All times include full GKR sumcheck + recursive STARK compression.

| Model | Layers | Params | GPU | GKR Prove | Recursive | Total | Calldata |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 1 | 135M | A10G | 3.3s | 0.18s | 3.5s | ~4,934 felts |
| SmolLM2-135M | 30 | 135M | A10G | 102s | 3.55s | ~106s | ~4,934 felts |
| Qwen2-0.5B | 1 | 494M | A10G | 3.0s | 0.20s | 3.2s | ~4,934 felts |
| Qwen3-14B | 40 | 14B | H100 | 103s | 3.6s | ~107s | ~4,934 felts |

Key observations:
- Recursive STARK adds 0.18--3.6s depending on model depth
- Output calldata is ~4,934 felts regardless of model size (compression from GKR)
- 160-bit security via PcsConfig: pow_bits=20, log_blowup=5, n_queries=28
- GPU acceleration uses CUDA for Merkle commits, MLE restriction, and FRI; CPU SIMD for FFT

---

## On-Chain Verification

Every proof is verified trustlessly on Starknet. The Cairo contract performs complete cryptographic STARK verification -- no optimistic assumptions, no fraud proofs, no committee.

### Deployed Contracts (Starknet Sepolia)

| Contract | Address |
|---|---|
| Recursive Verifier | [`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) |
| Class Hash | `0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5` |
| Deployer | [`0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344`](https://sepolia.voyager.online/contract/0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344) |

### Verified Transactions

4 verified TXs on Sepolia with the production 48-column chain AIR:

- **Latest TX:** [`0x021512dd...`](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8)
- **Contract:** `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`
- **Calldata:** ~4,934 felts (under 5,000 Starknet limit)
- **Security:** 160-bit (pow_bits=20, log_blowup=5, n_queries=28)
- **AIR:** 48 columns, 38 constraints (chain + Hades components)
- **Two-level recursion:** Level 1 cairo-prove (145 Hades perms, 10s, off-chain) + Level 2 chain STARK (6.5s, on-chain)
- **Verification:** OODS + Merkle decommitment + FRI folding + PoW

### Recursive STARK Security Layers (9 total)

The recursive verification system is protected by 9 independent security layers:

1. **Fiat-Shamir channel binding** -- all public inputs mixed before tree commits
2. **Amortized accumulator** -- unconditional constraint, blocks all-zeros-selector attack
3. **n_poseidon_perms on-chain validation** -- prevents trace miniaturization
4. **seed_digest checkpoint** -- binds chain to model dimensions
5. **pass1_final_digest binding** -- proves full GKR verification ran
6. **Carry-chain modular addition** -- HadesPerm-level chain integrity
7. **hades_commitment binding** -- two-level recursion (Level 2 chain STARK binds to Level 1 cairo-prove commitment)
8. **Boundary constraints** -- initial/final digest enforcement
9. **160-bit STARK security** -- pow=20, blowup=5, queries=28

### Verify a Proof Programmatically

```typescript
import { createObelyzkClient } from "@obelyzk/sdk";

const client = createObelyzkClient();
const status = await client.verify("0x055c2bf89f43d9b65580862e0b81e6b47842b9dda3b862c134f35b61b0ae620f");
console.log(status.verified); // true
```

Or query the contract directly:

```bash
starkli call 0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005 \
  is_verified 0x055c2bf89f43d9b65580862e0b81e6b47842b9dda3b862c134f35b61b0ae620f
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

**Decode (multi-turn).** Prefill + N decode steps with KV-cache commitment chaining. Each decode step produces a GKR proof bound to the previous KV-cache Poseidon root, enabling verifiable multi-turn conversations.

**MoE (Mixture of Experts).** TopK routing with dynamic expert weight binding. The prover detects MoE architecture automatically and binds per-expert weight commitments through the GKR channel (17 tests pass).

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
- `POST /api/v1/chat` -- prove from a text prompt (tokenize + embed + GKR proof)
- `POST /api/v1/infer` -- prove from raw input or text prompt
- `POST /api/v1/prove` -- async proving (returns job ID)
- `POST /api/v1/attest` -- prove + submit to Starknet
- `GET /api/v1/models` -- list loaded models
- `GET /api/v1/verify/:hash` -- verify a proof by hash
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

## API Reference

All authenticated endpoints require `Authorization: Bearer <API_KEY>`. Rate limit: 60 req/min.

### `POST /api/v1/chat` — Prove from Text

Tokenizes the prompt, embeds via the model's embedding table, runs the full GKR-proven forward pass, and predicts the next token via lm_head projection.

```bash
curl -X POST https://api.bitsage.network/api/v1/chat \
  -H "Authorization: Bearer $OBELYSK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "smollm2-135m",
    "prompt": "Hello world",
    "gpu": true,
    "include_calldata": false
  }'
```

**Request:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | yes | Model name or hex ID |
| `prompt` | string | yes | Text to tokenize and prove |
| `gpu` | bool | no | GPU acceleration (default: true) |
| `include_calldata` | bool | no | Return full proof calldata (default: false) |

**Response:**

```json
{
  "proof_id": "proof-174605e2-...",
  "token_ids": [19556, 905],
  "num_tokens": 2,
  "output": [1214124500.0, ...],
  "output_shape": [1, 576],
  "predicted_token_id": 19969,
  "predicted_text": "the",
  "io_commitment": "0x312c67eb...",
  "weight_commitment": "0x...",
  "proof_hash": "0x3d567aa9...",
  "prove_time_ms": 95523,
  "calldata_size": 46148,
  "calldata": null
}
```

### `POST /api/v1/infer` — Prove from Text or Raw Input

Synchronous provable inference. Provide either `prompt` (text, server tokenizes) or `input` (raw f32 array).

```bash
# From text prompt
curl -X POST https://api.bitsage.network/api/v1/infer \
  -H "Authorization: Bearer $OBELYSK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"smollm2-135m","prompt":"What is zero knowledge?"}'

# From raw input (576-dim embedding for SmolLM2)
curl -X POST https://api.bitsage.network/api/v1/infer \
  -H "Authorization: Bearer $OBELYSK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"smollm2-135m","input":[1.0, 2.0, ...]}'
```

**Request:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | yes | Model name or hex ID |
| `prompt` | string | one of | Text prompt (mutually exclusive with `input`) |
| `input` | float[] | one of | Raw f32 array matching model input shape |
| `gpu` | bool | no | GPU acceleration (default: true) |
| `include_output` | bool | no | Include raw output values (default: true) |
| `include_calldata` | bool | no | Return full proof calldata (default: false) |

**Response:**

```json
{
  "proof_id": "proof-292699ba-...",
  "output": [1214124500.0, ...],
  "output_shape": [1, 576],
  "io_commitment": "0x3186b7aa...",
  "weight_commitment": "0x...",
  "proof_hash": "0x20d2d7fd...",
  "verify_url": "/api/v1/verify/:proof_hash",
  "num_proven_layers": 211,
  "prove_time_ms": 96526,
  "estimated_gas": 9729600,
  "calldata_size": 46148
}
```

### `POST /api/v1/attest` — Prove + Submit On-Chain

Same as `/infer` but submits the recursive STARK proof to Starknet after proving. Returns TX hash.

### `GET /api/v1/models` — List Models

```bash
curl https://api.bitsage.network/api/v1/models \
  -H "Authorization: Bearer $OBELYSK_API_KEY"
```

### `GET /api/v1/verify/:proof_hash` — Verify Proof

```bash
curl https://api.bitsage.network/api/v1/verify/0x3d567aa9... \
  -H "Authorization: Bearer $OBELYSK_API_KEY"
```

### `GET /health` — Health Check (No Auth)

```bash
curl https://api.bitsage.network/health
```

### Per-Token Proving Time

Benchmarked end-to-end on the live hosted API (April 2026):

| Endpoint | Prompt | Model | GPU | Prove Time |
|----------|--------|-------|-----|------------|
| `/api/v1/chat` | "Hello world" | SmolLM2-135M (30L) | A10G | **95.5s** |
| `/api/v1/infer` | "What is zero knowledge?" | SmolLM2-135M (30L) | A10G | **96.5s** |

The proving time covers: tokenization + embedding extraction + full 30-layer forward pass + GKR sumcheck proof generation + unified STARK. The recursive STARK compression (~3.5s additional) is not included in these measurements.

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
    recursive/           # Recursive STARK wrapper (48-col chain AIR, 38 constraints)
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
| Sepolia Explorer | [Starkscan](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8) |
| npm | [@obelyzk/sdk](https://www.npmjs.com/package/@obelyzk/sdk) |
| PyPI | [obelyzk](https://pypi.org/project/obelyzk/) |

---

## License

Apache-2.0
