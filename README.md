<p align="center">

```
 ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═  ┏━┓┏━┓
 ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗  ┣┳┛┗━┓
 ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩  ╹┗╸┗━┛
```

</p>

# obelyzk.rs

**Verifiable AI engine written in Rust.** Provable inference for every model.

[![Rust](https://img.shields.io/badge/rust-nightly--2025--07--14-orange)](https://rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![Cairo](https://img.shields.io/badge/Cairo-2.12.2-blue)](https://www.cairo-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

---

## What is obelyzk.rs?

A purpose-built execution environment where every AI computation produces a cryptographic proof — verified on Starknet in a single transaction.

```bash
obelyzk chat       # Interactive verified chat with any model
obelyzk serve      # OpenAI-compatible API server with proving
obelyzk bench      # Throughput benchmark (80-322 tok/s on H100)
obelyzk prove      # Prove a model inference
obelyzk verify     # Verify a proof
```

**Two trust models, one interface:**
- **Open-weight models** (Llama, Qwen, Mistral, Phi, SmolLM): Full ZK proof — GKR sumcheck + recursive STARK
- **Closed-source APIs** (Claude, GPT, Grok, Gemini): TLS attestation — cryptographic proof the API call happened

---

## Architecture

```
obelyzk.rs/
├── engine/          — The ObelyZK proving engine (crate: obelyzk)
│   └── src/
│       ├── gkr/                  GKR sumcheck prover (CPU + 19 CUDA kernels)
│       ├── vm/                   VM runtime (trace, executor, queue, providers)
│       ├── providers/            Local (ZK), OpenAI, Anthropic (TLS attestation)
│       ├── components/           MatMul, Attention, Norm, Embedding, RoPE, TopK
│       ├── compiler/             HF model loader, graph compiler
│       ├── recursive/            Recursive STARK compression (46K → 942 felts)
│       └── aggregation.rs        Proving pipeline + batched throughput
│
├── stwo-gpu/        — STWO Circle STARK prover + our GPU backend
│
├── elo-cairo-verifier/           Recursive STARK verifier on Starknet
├── stark-cairo/                  STWO STARK verifier in Cairo
├── verifier/                     ML proof verifier
├── proof-stream/                 Real-time proof visualization (WebSocket)
└── sdk/                          Python, TypeScript, CLI SDKs
```

**Built on [STWO](https://github.com/starkware-libs/stwo)** by StarkWare. Our fork adds the full GPU proving backend (CUDA kernel dispatch, GPU FRI, GPU quotient evaluation, CudaStream optimization), Felt252Dict removal for Starknet Sierra 1.7 compatibility, and preprocessed column fixes. The GKR protocol, ML inference pipeline, VM runtime, TLS attestation, and on-chain verification are original work by [Bitsage Network](https://bitsage.network).

---

## Quick Start

```bash
# Chat with Claude (TLS-attested, every response verified)
ANTHROPIC_API_KEY=sk-ant-... obelyzk chat --model claude-sonnet

# Chat with local model (full ZK proof)
OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk chat

# Serve OpenAI-compatible API with proving
OBELYSK_MODEL_DIR=./models/qwen3-14b obelyzk serve --port 8080

# Benchmark throughput
OBELYSK_MODEL_DIR=./models/qwen3-14b obelyzk bench --tokens 10000
```

---

## Performance

| Model | Hardware | Batch | Throughput | Proof |
|-------|----------|-------|------------|-------|
| SmolLM2-135M | A10G | 1 | 0.05 tok/s | GKR + STARK |
| SmolLM2-135M | A10G | 8 | 0.23 tok/s | GKR + STARK |
| Qwen3-14B | H100 | 10K | **80-322 tok/s** | GKR + STARK |
| Claude/GPT | Any | Stream | **Instant** | TLS attestation |

Recursive STARK: 46,148 felts → 942 felts. One Starknet transaction.

---

## On-Chain Verification (Starknet Sepolia)

| Contract | Address |
|----------|---------|
| Recursive Verifier | [`0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) |
| Verified TX | [`0x276c6a44...`](https://sepolia.starkscan.co/tx/0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc) |

---

## Supported Models

Every model. Every provider. Every trust level.

**Open-weight models — full ZK proof (GKR + recursive STARK):**

| Architecture | Models | Status |
|-------------|--------|--------|
| LLaMA | Llama-3.x, SmolLM2, CodeLlama | Proven |
| Qwen | Qwen2, Qwen3-14B | Proven |
| Phi | Phi-3, Phi-4 | Proven |
| Mistral | Mistral-7B, Mixtral-8x7B (MoE) | Proven |
| Yi | Yi-1.5-6B | Proven |
| Gemma | Gemma-2 | HF auto-detect |
| MiniMax | MiniMax-01, MiniMax-Text | HF auto-detect |
| GLM | ChatGLM, GLM-4 | HF auto-detect |
| DeepSeek | DeepSeek-V2, DeepSeek-R1 | HF auto-detect |
| Falcon | Falcon-7B, Falcon-40B | HF auto-detect |
| MPT | MPT-7B, MPT-30B | HF auto-detect |
| RWKV | RWKV-6 | HF auto-detect |
| Any HuggingFace | SafeTensors format | Auto-detect architecture |

**Closed-source APIs — TLS attestation (cryptographic proof of API call):**

| Provider | Models | Trust |
|----------|--------|-------|
| Anthropic | Claude Opus, Sonnet, Haiku | TLS attestation |
| OpenAI | GPT-4o, o1, o3, GPT-4 | TLS attestation |
| Google | Gemini Pro, Ultra, Flash | TLS attestation |
| xAI | Grok-2, Grok-3 | TLS attestation |
| MiniMax | abab-7B-chat | TLS attestation |
| DeepSeek | DeepSeek-Chat API | TLS attestation |
| Any OpenAI-compatible | vLLM, TGI, Ollama, LM Studio | IO commitment |

---

## SDKs

```bash
pip install obelyzk          # Python
npm install @obelyzk/sdk     # TypeScript
npm install -g @obelyzk/cli  # CLI
```

---

## Building

```bash
rustup toolchain install nightly-2025-07-14
cd engine && cargo build --release --bin obelyzk --features "server,cuda-runtime"
```

---

## License

Apache-2.0

**950 tests. 1,750+ commits. Verifiable AI for every model.**
