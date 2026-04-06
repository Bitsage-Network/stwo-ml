# ObelyZK: Verifiable ML Inference on Starknet

**Prove that any ML model ran correctly. Verify it on-chain in a single transaction.**
Built on STWO Circle STARKs and GKR sumcheck proofs over the Mersenne-31 field.

---

## Quick Start

### 1. NPM SDK (simplest)

```bash
npm install @obelyzk/sdk
```

```typescript
import { createObelyzkClient } from "@obelyzk/sdk";

const client = createObelyzkClient(); // defaults to https://api.obelysk.com
const proof = await client.prove("smollm2-135m", { input: "Hello world", gpu: true });
console.log(proof.txHash);             // Starknet verification TX
console.log(proof.ioCommitment);       // on-chain commitment
```

### 2. Python SDK

```bash
pip install obelyzk
```

```python
from obelyzk import ObelyzkClient

client = ObelyzkClient()  # defaults to https://api.obelysk.com
result = client.prove("smollm2-135m", input="Hello world", gpu=True)

print(result.proof_hash)
print(result.io_commitment)
print(result.calldata)  # felts for Starknet verification
```

### 3. CLI

```bash
curl -sSf https://raw.githubusercontent.com/obelyzk/stwo-ml/main/install.sh | sh
obelysk prove --model smollm2-135m --input "Hello world" --on-chain
```

Output:

```
Model:     SmolLM2-135M (30 layers)
GKR proof: 95.2s
Recursive: 0.42s (981 felts)
TX hash:   0x04a1...
Status:    VERIFIED
```

---

## How It Works

```
1. LOAD MODEL          HuggingFace SafeTensors -> M31 quantization
       |
2. FORWARD PASS        Execute inference over the Mersenne-31 prime field
       |
3. GKR SUMCHECK        Interactive oracle proof over every MatMul, activation, norm
       |
4. RECURSIVE STARK     Compress to ~981 felts -> verify in 1 Starknet transaction
```

Every operation in the model -- MatMul, SiLU, GELU, Softmax, RMSNorm, LayerNorm -- becomes a GKR sumcheck circuit. Weight commitments are bound via Poseidon Merkle roots. The full computation graph compresses into a single recursive STARK proof that fits in one Starknet transaction.

---

## Supported Models

| Model | Params | Hidden Size | Prove Time (GPU) | On-Chain Felts |
|---|---|---|---|---|
| SmolLM2-135M (1 layer) | 135M | 576 | 3.3s | 981 |
| SmolLM2-135M (30 layers) | 135M | 576 | 95s | 981 |
| Qwen2-0.5B | 494M | 896 | ~20s | 981 |
| Llama-3.2-3B | 3B | 3072 | ~35s | 981 |
| Phi-3-mini | 3.8B | 3072 | ~45s | 981 |
| Mistral-7B | 7B | 4096 | ~90s | 981 |
| Qwen3-14B (H100) | 14B | 5120 | 103s | 981 |

Any HuggingFace transformer with SafeTensors weights is supported. GPU proving uses CUDA on A10G, RTX 4090, or H100.

---

## On-Chain Verification

ObelyZK produces a **fully trustless recursive STARK proof** that compresses the entire GKR computation into approximately 981 felts -- a 260x reduction from the streaming format. This means a full 30-layer SmolLM2-135M proof verifies in a single Starknet transaction, with no multi-step coordination needed. The recursive verifier performs complete cryptographic verification on-chain: OODS sampling, Merkle decommitment, FRI layer folding, and proof-of-work.

For larger models or when finer-grained verification is preferred, a **streaming verifier** breaks the proof into sequential steps that can be submitted independently. Both paths produce the same cryptographic guarantee: the model inference is correct and the weights match their committed Poseidon roots.

**Contracts on Starknet Sepolia:**

| Contract | Address |
|---|---|
| Recursive Verifier (Phase 1) | `0x707819dea6210ab58b358151419a604ffdb16809b568bf6f8933067c2a28715` |
| Streaming Verifier | `0x376fa0c4a9cf3d069e6a5b91bad6e131e7a800f9fced49bd72253a0b0983039` |

The fully trustless recursive class (`0x006d4ff233...ce820`) has been verified on devnet and is pending Sepolia deployment.

See [docs/ON_CHAIN_VERIFICATION.md](docs/ON_CHAIN_VERIFICATION.md) for the full protocol specification, calldata encoding, and contract ABI.

---

## Self-Hosting

Run your own prover on any machine with a CUDA-capable GPU:

```bash
git clone https://github.com/obelyzk/stwo-ml.git
cd stwo-ml
./scripts/setup.sh
```

Generate a proof and submit on-chain:

```bash
obelysk prove --model-dir ./models/smollm2-135m --gkr --recursive --on-chain
```

Or start a prove server that exposes an HTTP/WebSocket API:

```bash
obelysk serve --port 8080 --gpu
```

The prove server provides a `/ws` endpoint for real-time proof streaming and a web dashboard at `/`.

---

## SDK Reference

| Package | Install | Docs |
|---|---|---|
| `@obelyzk/sdk` | `npm install @obelyzk/sdk` | [sdk/typescript/README.md](sdk/typescript/README.md) |
| `obelyzk` (Python) | `pip install obelyzk` | [sdk/python/README.md](sdk/python/README.md) |
| `@obelyzk/cli` | `npm install -g @obelyzk/cli` | [sdk/cli/README.md](sdk/cli/README.md) |

All SDKs default to `https://api.obelysk.com` and can be pointed at a self-hosted prover via the `OBELYZK_API_URL` environment variable.

---

## Architecture

```
                    +------------------+
                    |  Model Weights   |
                    |  (SafeTensors)   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  M31 Forward     |
                    |  Pass Execution  |
                    +--------+---------+
                             |
                    +--------+---------+
                    |  GKR Sumcheck    |
                    |  Prover (GPU)    |
                    +--------+---------+
                             |
                    +--------+---------+
                    |  Recursive STARK |
                    |  Compression     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Starknet        |
                    |  On-Chain STARK  |
                    |  Verification    |
                    |  (OODS + Merkle  |
                    |   + FRI + PoW)   |
                    |  1 TX, ~981 felts|
                    +------------------+
```

The GKR prover generates a sumcheck proof for every layer in the computation graph. Weight Poseidon commitments are aggregated via an oracle sumcheck. The entire proof tree is then compressed into a recursive STARK -- a single proof that attests to the validity of all inner proofs. The on-chain verifier performs full cryptographic STARK verification: OODS sampling, Merkle decommitment, FRI layer folding, and proof-of-work validation.

---

## Performance

| Model | GPU | GKR Prove | Recursive | Total | On-Chain Felts |
|---|---|---|---|---|---|
| SmolLM2-135M (1 layer) | A10G | 3.3s | 0.18s | 3.5s | 981 |
| SmolLM2-135M (30 layers) | A10G | 95s | 0.42s | 95.4s | 981 |
| Qwen2-0.5B (1 layer) | A10G | 3.0s | 0.20s | 3.2s | 981 |
| Qwen3-14B (40 layers) | H100 | 103s | 3.6s | 106.6s | 981 |

Recursive STARK adds 0.18--3.6s depending on the number of layers. The output is always ~981 felts regardless of model size.

---

## Environment Variables

See [docs/ENV_VARS.md](docs/ENV_VARS.md) for the full reference. The most important ones:

| Variable | Default | Description |
|---|---|---|
| `OBELYZK_API_URL` | `https://api.obelysk.com` | Prover API endpoint (SDKs and CLI) |
| `STARKNET_PRIVATE_KEY` | -- | Account private key for on-chain submission |
| `STARKNET_RPC` | Alchemy Sepolia | Starknet RPC endpoint URL |
| `RECURSIVE_CONTRACT` | Phase 1 address | Recursive verifier contract address |
| `STWO_GPU_MERKLE_THRESHOLD` | `4096` | Minimum leaf count before GPU Merkle kicks in |
| `STWO_WEIGHT_BINDING` | `aggregated` | Weight binding mode: `aggregated` (default), `individual`, or `sequential` |

---

## Links

| Resource | URL |
|---|---|
| Documentation | [docs/](docs/) |
| Paper | [obelyzk-paper.pdf](obelyzk-paper.pdf) |
| Recursive STARK Spec | [docs/RECURSIVE_STARK.md](docs/RECURSIVE_STARK.md) |
| On-Chain Verification | [docs/ON_CHAIN_VERIFICATION.md](docs/ON_CHAIN_VERIFICATION.md) |
| Starknet Explorer (Recursive) | [Voyager](https://sepolia.voyager.online/contract/0x707819dea6210ab58b358151419a604ffdb16809b568bf6f8933067c2a28715) |
| Starknet Explorer (Streaming) | [Voyager](https://sepolia.voyager.online/contract/0x376fa0c4a9cf3d069e6a5b91bad6e131e7a800f9fced49bd72253a0b0983039) |
| npm | [@obelyzk/sdk](https://www.npmjs.com/package/@obelyzk/sdk) |
| PyPI | [obelyzk](https://pypi.org/project/obelyzk/) |

---

## License

Apache-2.0
