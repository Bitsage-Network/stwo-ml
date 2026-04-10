<div align="center">

<br/>

```
                    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ██╗
                   ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝╚══███╔╝██║ ██╔╝
                   ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝   ███╔╝ █████╔╝
                   ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝   ███╔╝  ██╔═██╗
                   ╚██████╔╝██████╔╝███████╗███████╗██║   ███████╗██║  ██╗
                    ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝
                                          ·  r s  ·
```

<br/>

### Verifiable AI Engine Written in Rust

**Provable inference for every model. Qwen2.5-14B + GLM-4-9B proven on-chain.**

[Qwen2.5-14B → `0x5ce1b4...`](https://sepolia.starkscan.co/tx/0x5ce1b41815e29a7b3dd03b77187cf32c8c5f0e2607960303174cbea303edfd3) · [GLM-4-9B → `0x542960...`](https://sepolia.starkscan.co/tx/0x542960d703a62d4beaacf0d9094ea92dc86bf326cd917c533039f4dd1eb4a30)

<br/>

[![crates.io](https://img.shields.io/crates/v/obelyzk?style=for-the-badge&color=orange)](https://crates.io/crates/obelyzk)
[![PyPI](https://img.shields.io/pypi/v/obelyzk?style=for-the-badge&color=blue)](https://pypi.org/project/obelyzk/)
[![npm](https://img.shields.io/npm/v/@obelyzk/sdk?style=for-the-badge&color=red)](https://www.npmjs.com/package/@obelyzk/sdk)
[![Rust](https://img.shields.io/badge/rust-nightly--2025--07--14-orange?style=for-the-badge)](https://rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-76B900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Starknet](https://img.shields.io/badge/Starknet-Sepolia-29296E?style=for-the-badge)](https://starknet.io)
[![Tests](https://img.shields.io/badge/tests-950_passing-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/Apache--2.0-grey?style=for-the-badge)](LICENSE)

<br/>

[Install](#install) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Performance](#performance) · [Models](#supported-models) · [On-Chain](#on-chain-verification) · [SDKs](#sdks) · [Contributing](CONTRIBUTING.md)

<br/>

---

</div>

<br/>

> **obelyzk.rs** is a purpose-built execution environment where every AI computation produces a cryptographic proof — verified on Starknet in a single transaction. Open-weight models get full ZK proofs. Closed-source APIs get TLS attestation. Every model, every provider, every response — verifiable.

<br/>

## Install

```bash
# Rust (crates.io)
cargo add obelyzk

# Python (PyPI)
pip install obelyzk

# TypeScript (npm)
npm install @obelyzk/sdk

# CLI
npm install -g @obelyzk/cli

# From source (GPU support)
git clone https://github.com/Bitsage-Network/obelyzk.rs.git
cd obelyzk.rs/engine
cargo build --release --bin obelyzk --features "server,cuda-runtime"
```

<br/>

## Quick Start

```bash
# Prove a local model (full ZK proof — GKR + recursive STARK)
OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk chat

# Chat with Claude (TLS-attested, every response verified)
ANTHROPIC_API_KEY=sk-ant-... obelyzk chat --model claude-sonnet

# Serve OpenAI-compatible API with built-in proving
OBELYSK_MODEL_DIR=./models/qwen2.5-14b obelyzk serve --port 8080

# Benchmark proving throughput
obelyzk bench --tokens 64

# Live dashboard (Cipher Noir TUI)
obelyzk dashboard

# Load any llama.cpp GGUF model
OBELYSK_MODEL_DIR=./model.Q8_0.gguf obelyzk serve
```

**OpenAI-compatible API** — drop-in replacement:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-14b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Every response includes `obelyzk.proof_id` and `obelyzk.trust_model` — poll `/v1/proofs/:id` for proof status.

**Python SDK:**

```python
from obelyzk import ObelyzkClient

client = ObelyzkClient()
result = await client.chat("smollm2-135m", "What is ZKML?")
print(result.predicted_text)   # model response
print(result.proof_id)         # cryptographic proof
print(result.io_commitment)    # Poseidon commitment
```

<br/>

## Architecture

```
obelyzk.rs/
├── engine/              The ObelyZK proving engine (crate: obelyzk)
│   └── src/
│       ├── gkr/                  GKR sumcheck prover · CPU + 19 CUDA kernels
│       ├── vm/                   VM runtime · trace, executor, queue, providers
│       ├── providers/            Local (ZK) · OpenAI · Anthropic (TLS attestation)
│       ├── components/           MatMul · Attention · Norm · Embedding · RoPE · TopK
│       ├── compiler/             HuggingFace + GGUF model loader · graph compiler
│       ├── recursive/            Recursive STARK compression (46K → 942 felts)
│       ├── crypto/               Poseidon channel · GPU Poseidon · Merkle trees
│       ├── gpu_forward.rs        GPU-resident forward pass (GpuForwardExecutor)
│       ├── gpu_sumcheck.rs       19 CUDA kernels (sumcheck, MLE, LogUp, forward)
│       ├── cuda/                 CUDA kernel sources (Poseidon Hades, felt252)
│       └── tui/                  Cipher Noir dashboards (ratatui)
│
├── stwo-gpu/            STWO Circle STARK prover + our GPU backend (crate: stwo-gpu)
│                        CUDA kernel dispatch · GPU FRI · GPU quotient eval
│
├── elo-cairo-verifier/  Recursive STARK verifier on Starknet (Cairo)
├── stark-cairo/         STWO STARK verifier in Cairo
├── verifier/            ML proof verifier
├── proof-stream/        Real-time proof visualization (WebSocket) (crate: proof-stream)
│
├── sdk/
│   ├── python/          pip install obelyzk
│   ├── typescript/      npm install @obelyzk/sdk
│   └── cli/             npm install -g @obelyzk/cli
│
└── scripts/             Provisioning (H100, A10G), deployment, benchmarking
```

<details>
<summary><b>STWO Foundation</b></summary>
<br/>

Built on **[STWO](https://github.com/starkware-libs/stwo)** by StarkWare — the Circle STARK prover. Our fork ([`stwo-gpu`](https://crates.io/crates/stwo-gpu)) adds:

- Full GPU proving backend (CUDA kernel dispatch, CudaStream graph execution)
- GPU FRI layer folding + GPU quotient evaluation
- Felt252Dict removal for Starknet Sierra 1.7 compatibility
- Preprocessed column deduplication fixes

The GKR protocol, ML inference pipeline, VM runtime, GGUF loader, TLS attestation, and on-chain verification are original work by [Bitsage Network](https://bitsage.network).

</details>

<br/>

## Performance

Benchmarked on real hardware. All times are warm-cache (weight Merkle roots cached to disk).

<div align="center">

| Model | Hardware | Tokens | Prove Time | Per Token |
|:------|:---------|-------:|----------:|----------:|
| SmolLM2-135M | A10G | 1 | **4.3s** | 4.3s |
| SmolLM2-135M | H100 | 1 | **3.8s** | 3.8s |
| SmolLM2-135M | H100 | 8 | **17.1s** | 2.1s |
| SmolLM2-135M | H100 | 16 | **32.3s** | 2.0s |
| Qwen2.5-14B | H100 | 1 | **41.4s** | 41.4s |
| Claude / GPT / Grok | Any | Stream | **0.5s** | instant |

</div>

> **SSE streaming**: user sees response in **0.5s**. Proof generates asynchronously in background. Poll `/v1/proofs/:id`.

> **Recursive STARK**: 46,148 GKR felts → 942 felts. Verified in one Starknet transaction.

**Optimization journey (this build session):**
```
95.5s → 19.7s → 11.5s → 4.3s  (22x speedup)
```
Key: skip redundant unified STARK + weight commitment disk cache + batched sumcheck.

<br/>

## On-Chain Verification

Every proof is verified trustlessly on Starknet. No optimistic assumptions. No fraud proofs. No committee.

<div align="center">

| Contract | Address |
|:---------|:--------|
| **Recursive Verifier** | [`0x1c208a...0c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) |
| **Verified TX** | [`0x276c6a...ddc`](https://sepolia.starkscan.co/tx/0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc) |
| **Network** | Starknet Sepolia |
| **Verification** | OODS + Merkle + FRI + PoW |

</div>

<br/>

## Supported Models

> Every model. Every provider. Every trust level.

### Open-Weight Models — Full ZK Proof

GKR sumcheck over every operation. Recursive STARK compression. Weight binding via Poseidon Merkle roots.

<div align="center">

| Architecture | Models | Format | Status |
|:------------|:-------|:-------|:------:|
| **LLaMA** | Llama-3.x, SmolLM2, CodeLlama | SafeTensors + GGUF | ✅ Proven |
| **Qwen** | Qwen2, Qwen2.5-14B | SafeTensors + GGUF | ✅ Proven on H100 |
| **Phi** | Phi-3, Phi-4 | SafeTensors | ✅ Proven |
| **Mistral** | Mistral-7B, Mixtral-8x7B (MoE) | SafeTensors | ✅ Proven |
| **Yi** | Yi-1.5-6B | SafeTensors | ✅ Proven |
| **Gemma** | Gemma-2 | SafeTensors | ✅ Auto-detect |
| **DeepSeek** | DeepSeek-V2, DeepSeek-R1 | SafeTensors | ✅ Auto-detect |
| **GLM** | ChatGLM, GLM-4 | SafeTensors | ✅ Auto-detect |
| **MiniMax** | MiniMax-01, MiniMax-Text | SafeTensors | ✅ Auto-detect |
| **Falcon** | Falcon-7B, Falcon-40B | SafeTensors | ✅ Auto-detect |
| **Any HuggingFace** | SafeTensors or GGUF format | Both | ✅ Auto-detect |

</div>

### Closed-Source APIs — TLS Attestation

Cryptographic proof the API call happened. Certificate-verified commitment over request + response.

<div align="center">

| Provider | Models | Trust |
|:---------|:-------|:-----:|
| **Anthropic** | Claude Opus, Sonnet, Haiku | 🔒 TLS |
| **OpenAI** | GPT-4o, o1, o3, GPT-4 | 🔒 TLS |
| **Google** | Gemini Pro, Ultra, Flash | 🔒 TLS |
| **xAI** | Grok-2, Grok-3 | 🔒 TLS |
| **DeepSeek** | DeepSeek-Chat API | 🔒 TLS |
| **MiniMax** | abab-7B-chat | 🔒 TLS |
| **Any OpenAI-compatible** | vLLM, TGI, Ollama, LM Studio | 📋 Commitment |

</div>

<br/>

## API Reference

### `POST /v1/chat/completions` — OpenAI-compatible

```bash
# Non-streaming (blocks until proof is done)
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"smollm2-135m","messages":[{"role":"user","content":"Hello"}]}'

# Streaming (tokens instant, proof in background)
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"smollm2-135m","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Response includes `obelyzk` extension:
```json
{
  "obelyzk": {
    "proof_id": "proof-97edf02f-...",
    "proof_status": "complete",
    "trust_model": "zk_proof",
    "io_commitment": "0x6a70ab3c...",
    "session_id": "ses-7e49..."
  }
}
```

### `GET /v1/proofs/:id` — Proof status

```json
{"proof_id": "proof-...", "status": "complete", "prove_time_ms": 3810, "io_commitment": "0x..."}
```

### `GET /v1/sessions` — Active conversations

### `GET /v1/models` — Available models with trust model

### `GET /health` — Server health

### `POST /v1/prove/batch` — Batched throughput proving

<br/>

## SDKs

<div align="center">

| Package | Registry | Install |
|:--------|:---------|:--------|
| `obelyzk` | [crates.io](https://crates.io/crates/obelyzk) | `cargo add obelyzk` |
| `stwo-gpu` | [crates.io](https://crates.io/crates/stwo-gpu) | `cargo add stwo-gpu` |
| `obelyzk` | [PyPI](https://pypi.org/project/obelyzk/) | `pip install obelyzk` |
| `@obelyzk/sdk` | [npm](https://www.npmjs.com/package/@obelyzk/sdk) | `npm install @obelyzk/sdk` |
| `@obelyzk/cli` | [npm](https://www.npmjs.com/package/@obelyzk/cli) | `npm install -g @obelyzk/cli` |
| `proof-stream` | [crates.io](https://crates.io/crates/proof-stream) | `cargo add proof-stream` |

</div>

**Rust:**
```rust
use obelyzk::aggregation::prove_model_pure_gkr_auto_with_cache;
use obelyzk::compiler::hf_loader::load_hf_model;
use obelyzk::vm::trace::ExecutionTrace;
```

**Python:**
```python
from obelyzk import ObelyzkClient
client = ObelyzkClient("https://api.obelyzk.xyz")
result = await client.chat("qwen2.5-14b", "Explain zero-knowledge proofs")
```

**TypeScript:**
```typescript
import { createObelyzkClient } from "@obelyzk/sdk";
const client = createObelyzkClient();
const proof = await client.prove("smollm2-135m", { input: "Hello" });
```

**CLI:**
```bash
obelyzk prove smollm2-135m --prompt "Hello world"
obelyzk submit --proof proof.json --network sepolia
```

<br/>

## Deployment

### One-command H100 provisioning:

```bash
# On a fresh H100 (Ubuntu 22.04):
git clone https://github.com/Bitsage-Network/obelyzk.rs.git
cd obelyzk.rs && bash engine/scripts/provision_h100.sh
```

Installs Rust, CUDA, builds engine, downloads model, sets up systemd service.

### Manual:

```bash
# Build
rustup toolchain install nightly-2025-07-14
cd engine && cargo build --release --bin obelyzk --features "server,cuda-runtime"

# Run
OBELYSK_MODEL_DIR=./models/qwen2.5-14b obelyzk serve --port 8080

# Service
sudo systemctl start obelyzk
```

### Live infrastructure:

| Service | URL | Model |
|---------|-----|-------|
| H100 Prover | `http://62.169.159.231:8080` | Qwen2.5-14B |
| Starknet Contract | [`0x1c208a...0c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) | Recursive verifier |

<br/>

## Building

```bash
# Prerequisites: Rust nightly-2025-07-14, CUDA 12+ (optional)
rustup toolchain install nightly-2025-07-14

# CPU build
cd engine && cargo build --release --bin obelyzk --features server

# GPU build (CUDA)
cargo build --release --bin obelyzk --features "server,cuda-runtime"

# With TUI dashboard
cargo build --release --bin obelyzk --features "server,cuda-runtime,tui"

# Run tests (950 passing)
cargo test --lib --features std
```

<br/>

---

<div align="center">

**950 tests · 1,750+ commits · 7 published packages · Verifiable AI for every model**

[obelysk.xyz](https://obelysk.xyz) · [bitsage.network](https://bitsage.network) · [Starknet Sepolia](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) · [crates.io](https://crates.io/crates/obelyzk) · [PyPI](https://pypi.org/project/obelyzk/) · [npm](https://www.npmjs.com/package/@obelyzk/sdk)

Apache-2.0

</div>
