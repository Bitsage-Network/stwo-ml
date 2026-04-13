# obelyzk.rs — VM Usage Manual

Complete guide to running the ObelyZK Verifiable AI Engine.

---

## Table of Contents

1. [Installation](#installation)
2. [Commands](#commands)
3. [Serving the API](#serving-the-api)
4. [Interactive Chat](#interactive-chat)
5. [Benchmarking](#benchmarking)
6. [Dashboard](#dashboard)
7. [Model Loading](#model-loading)
8. [Providers](#providers)
9. [Sessions & Multi-Turn](#sessions--multi-turn)
10. [Proof Lifecycle](#proof-lifecycle)
11. [Configuration](#configuration)
12. [SDKs](#sdks)
13. [Deployment](#deployment)

---

## Installation

### From Source (recommended for GPU)

```bash
git clone https://github.com/Bitsage-Network/obelyzk.rs.git
cd obelyzk.rs/engine

# CPU only
cargo build --release --bin obelyzk --features server

# GPU (CUDA 12+)
cargo build --release --bin obelyzk --features "server,cuda-runtime"

# GPU + TUI dashboard
cargo build --release --bin obelyzk --features "server,cuda-runtime,tui"
```

### From Package Registries

```bash
# Rust library
cargo add obelyzk

# Python SDK
pip install obelyzk

# TypeScript SDK
npm install @obelyzk/sdk

# CLI tool
npm install -g @obelyzk/cli
```

### One-Command H100 Setup

```bash
git clone https://github.com/Bitsage-Network/obelyzk.rs.git
cd obelyzk.rs && bash engine/scripts/provision_h100.sh
```

---

## Commands

```
obelyzk serve             Start the API server
obelyzk chat              Interactive terminal chat
obelyzk bench             Throughput benchmark
obelyzk dashboard         Live Cipher Noir TUI monitor
```

---

## Serving the API

```bash
# Load a HuggingFace model and serve OpenAI-compatible API
OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk serve --port 8080

# Load a GGUF model (llama.cpp format)
OBELYSK_MODEL_DIR=./model.Q8_0.gguf obelyzk serve --port 8080

# With Claude/GPT TLS attestation
ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... obelyzk serve
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (stream + non-stream) |
| `GET` | `/v1/models` | List available models with trust model |
| `GET` | `/v1/proofs/:id` | Proof status and result |
| `GET` | `/v1/sessions` | Active conversation sessions |
| `DELETE` | `/v1/sessions/:id` | Delete a session |
| `GET` | `/v1/attestations/:id` | TLS attestation record |
| `POST` | `/v1/prove/batch` | Batch proving for throughput |
| `GET` | `/health` | Server health check |

### Chat Completions

**Non-streaming** (blocks until proof is done):
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2-135m",
    "messages": [{"role": "user", "content": "What is ZKML?"}]
  }'
```

**Streaming** (tokens in ~0.5s, proof in background):
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm2-135m",
    "messages": [{"role": "user", "content": "What is ZKML?"}],
    "stream": true
  }'
```

SSE events include the model response immediately. The final event contains:
```json
{
  "obelyzk": {
    "proof_id": "proof-97edf02f-...",
    "proof_status": "proving",
    "trust_model": "zk_proof",
    "session_id": "ses-7e49...",
    "inference_time_ms": 521
  }
}
```

**Multi-turn** (pass `session_id` from previous response):
```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{
    "model": "smollm2-135m",
    "messages": [{"role": "user", "content": "Tell me more"}],
    "session_id": "ses-7e49...",
    "stream": true
  }'
```

### Model Routing

The server auto-routes based on model name:

| Model Name | Provider | Trust Model |
|------------|----------|-------------|
| `smollm2-135m`, `qwen2.5-14b` | Local (loaded model) | ZK proof |
| `claude-*` | Anthropic API | TLS attestation |
| `gpt-*`, `o1-*`, `o3-*` | OpenAI API | TLS attestation |
| Other | Upstream (if configured) | IO commitment |

---

## Interactive Chat

```bash
# Chat with local model (ZK-proved)
OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk chat

# Chat with Claude (TLS-attested)
ANTHROPIC_API_KEY=sk-ant-... obelyzk chat --model claude-sonnet

# Chat with GPT
OPENAI_API_KEY=sk-... obelyzk chat --model gpt-4o
```

Every response shows:
```
  AI   The answer is...

  ✓ TLS attested  att-a4f2...  0x3a8f...c2e1  1203ms
```

Type `exit` to quit.

---

## Benchmarking

```bash
# Single token (measures cold/warm proving time)
obelyzk bench --tokens 1

# Batched (measures throughput amortization)
obelyzk bench --tokens 8
obelyzk bench --tokens 64

# With specific model
OBELYSK_MODEL_DIR=./models/qwen2.5-14b obelyzk bench --tokens 1
```

**Expected results (warm cache):**

| Model | Hardware | 1 token | 8 tokens | 16 tokens |
|-------|----------|---------|----------|-----------|
| SmolLM2-135M | A10G | 4.3s | 35.3s | — |
| SmolLM2-135M | H100 | 3.8s | 17.1s | 32.3s |
| Qwen2.5-14B | H100 | 41.4s | — | — |

---

## Dashboard

```bash
# Interactive ratatui TUI (requires 'tui' feature)
obelyzk dashboard

# Build with TUI support
cargo build --release --bin obelyzk --features "server,cuda-runtime,tui"
```

Shows: GPU workers, proving queue, sessions, conversation stream, throughput.
Press `q` or `Ctrl+C` to exit.

---

## Model Loading

### HuggingFace SafeTensors (directory)

```bash
# Auto-detects architecture from config.json
OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk serve
```

Supports: LLaMA, Qwen, Phi, Mistral, Yi, Gemma, DeepSeek, GLM, Falcon, MPT, and any transformer with SafeTensors weights.

### GGUF (llama.cpp format)

```bash
# Single .gguf file — auto-detected by extension
OBELYSK_MODEL_DIR=./model.Q8_0.gguf obelyzk serve
```

Supported quantizations: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q6_K.

### Tokenizer

The tokenizer is loaded automatically from `tokenizer.json` in the model directory. For GGUF files, the tokenizer is searched in the parent directory.

---

## Providers

### Local Provider (ZK Proof)

Full cryptographic proof of computation. Every MatMul, attention, norm, activation is provable via GKR sumcheck + recursive STARK.

```bash
OBELYSK_MODEL_DIR=./models/smollm2-135m obelyzk serve
```

### Anthropic Provider (TLS Attestation)

```bash
ANTHROPIC_API_KEY=sk-ant-... obelyzk serve
# Then: curl .../v1/chat/completions -d '{"model":"claude-sonnet",...}'
```

### OpenAI Provider (TLS Attestation)

```bash
OPENAI_API_KEY=sk-... obelyzk serve
# Then: curl .../v1/chat/completions -d '{"model":"gpt-4o",...}'
```

### Upstream Provider (IO Commitment)

Forward to any OpenAI-compatible API (vLLM, Ollama, TGI):

```bash
UPSTREAM_URL=http://vllm-server:8000/v1 UPSTREAM_MODEL=llama-3.1-8b obelyzk serve
```

---

## Sessions & Multi-Turn

Each conversation creates a session with KV-cache for efficient multi-turn proving.

- First request: creates session, returns `session_id`
- Subsequent requests: pass `session_id` to continue conversation
- KV-cache persists between turns (Poseidon commitment chain)
- Sessions expire after 5 minutes of inactivity
- Delete explicitly: `DELETE /v1/sessions/:id`
- List active: `GET /v1/sessions`

---

## Proof Lifecycle

1. **Request** → `POST /v1/chat/completions` with `stream: true`
2. **SSE response** → tokens stream in ~0.5s
3. **Background proof** → GKR sumcheck + weight binding (async)
4. **Poll** → `GET /v1/proofs/:proof_id`
5. **Complete** → `io_commitment`, `proof_hash`, `prove_time_ms`

Status values: `queued` → `proving` → `complete` (or `failed`)

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OBELYSK_MODEL_DIR` | — | Model directory or .gguf file path |
| `PORT` | `8080` | API server port |
| `PROVE_WORKERS` | `1` | Number of GPU proving workers |
| `PROVE_SERVER_API_KEY` | — | API key for authenticated access |
| `PROVE_SERVER_RATE_LIMIT` | `60` | Requests per minute |
| `ANTHROPIC_API_KEY` | — | Enables Claude provider |
| `OPENAI_API_KEY` | — | Enables GPT provider |
| `UPSTREAM_URL` | — | OpenAI-compatible upstream URL |
| `UPSTREAM_MODEL` | `default` | Model name for upstream |
| `OBELYZK_BATCH_SIZE` | `1` | Tokens per proof chunk |
| `OBELYZK_GPU_FORWARD` | `1` | Enable GPU forward pass (`0` to disable) |

---

## SDKs

### Python

```bash
pip install obelyzk
```

```python
from obelyzk import ObelyzkClient

client = ObelyzkClient(url="http://localhost:8080")

# Text-based proving (chat endpoint)
result = await client.chat("smollm2-135m", "What is ZKML?")
print(result.predicted_text)
print(result.proof_id)
print(result.io_commitment)

# Synchronous inference (blocks until proof done)
result = await client.infer_sync("smollm2-135m", prompt="Hello world")
print(result.output)
print(result.prove_time_ms)

# Async proving (poll for result)
result = await client.prove("smollm2-135m", input=[1.0, 2.0, 3.0])
print(result.calldata)
```

### TypeScript

```bash
npm install @obelyzk/sdk
```

```typescript
import { createProverClient } from "@obelyzk/sdk";

const client = createProverClient({ url: "http://localhost:8080" });

// Prove a model
const result = await client.prove("smollm2-135m", {
  input: [1.0, 2.0, 3.0],
  gpu: true,
});

console.log(result.proofHash);
console.log(result.ioCommitment);
```

### CLI

```bash
npm install -g @obelyzk/cli
```

```bash
# Prove from text prompt
obelysk prove smollm2-135m --prompt "Hello world"

# Prove from raw input
obelysk prove smollm2-135m --input "[1.0, 2.0, 3.0]"

# Submit proof on-chain
obelysk submit --proof proof.json --network sepolia

# List models
obelysk models --prover-url http://localhost:8080
```

### Rust

```toml
[dependencies]
obelyzk = "0.3.0"
```

```rust
use obelyzk::compiler::hf_loader::load_hf_model;
use obelyzk::aggregation::prove_model_pure_gkr_auto_with_cache;

let model = load_hf_model("./models/smollm2-135m", None)?;
let proof = prove_model_pure_gkr_auto_with_cache(
    &model.graph, &input, &model.weights, None, None,
)?;
println!("IO commitment: 0x{:x}", proof.io_commitment);
```

---

## Deployment

### Systemd Service

```bash
sudo systemctl start obelyzk
sudo systemctl status obelyzk
journalctl -u obelyzk -f
```

### Docker (coming soon)

```bash
docker run -p 8080:8080 --gpus all \
  -v ./models:/models \
  -e OBELYSK_MODEL_DIR=/models/smollm2-135m \
  obelyzk/obelyzk:latest
```

### Cloud Provisioning

```bash
# H100 (NVIDIA DGX / Shadeform / Lambda)
bash engine/scripts/provision_h100.sh

# A10G (AWS)
bash scripts/deploy_vm.sh
```

---

## On-Chain Verification

Proofs are verified on Starknet Sepolia via recursive STARK (~4,934 felts, 1 TX, 160-bit security).

The production v2 recursive verifier uses a 48-column chain AIR (was 89 -- 41 unused columns removed) with 38 constraints, including an amortized accumulator, carry-chain modular addition, and hades_commitment binding for two-level recursion. PcsConfig: pow_bits=20, log_blowup=5, n_queries=28. Two-level recursion: Level 1 cairo-prove (145 Hades perms, off-chain) + Level 2 chain STARK (on-chain). 9 security layers.

**Contract (v2, preferred):** [`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005)

**Contract (v1):** [`0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7)

```bash
# Submit a recursive proof
node scripts/submit_recursive.mjs /tmp/proof.json

# Check verification count (v2 contract)
starkli call 0x0121d1...8c005 get_recursive_verification_count 0x_model_id
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA not found` | Install CUDA toolkit 12+ or build with `--features server` (CPU only) |
| `tokenizer not found` | Ensure `tokenizer.json` is in the model directory |
| `weight dimension mismatch` | Model config.json doesn't match weight shapes |
| `port already in use` | Change `PORT=8081` or kill existing process |
| `proof timeout` | Increase `PROVE_WORKERS` or use streaming mode |
| `OOM on GPU` | Use a smaller model or reduce batch size |

---

*950 tests · 1,750+ commits · 7 published packages · Apache-2.0*
