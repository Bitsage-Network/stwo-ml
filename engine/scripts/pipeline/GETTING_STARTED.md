# ObelyZK — Getting Started

Prove that an ML model ran correctly with a cryptographic proof, verified on Starknet in a single transaction.

**Live proofs on Starknet Sepolia:**

| Model | Params | TX | Felts | GKR | STARK |
|-------|--------|-----|-------|-----|-------|
| **Qwen2.5-14B** | 14B | [`0x5ce1b4...edfd3`](https://sepolia.starkscan.co/tx/0x5ce1b41815e29a7b3dd03b77187cf32c8c5f0e2607960303174cbea303edfd3) | 946 | 46s | 1.2s |
| **GLM-4-9B** | 9B | [`0x542960...4dd1e`](https://sepolia.starkscan.co/tx/0x542960d703a62d4beaacf0d9094ea92dc86bf326cd917c533039f4dd1eb4a30) | 929 | 201s | 1.1s |

## Quick Overview

```
Input text → Tokenize → Forward pass (GPU) → GKR proof (GPU) → Recursive STARK → On-chain TX
                                                   46s              1.2s            ~15s
```

| Component | Description |
|-----------|-------------|
| **GKR Sumcheck** | Layer-by-layer interactive proof over M31/QM31 fields (192 matmuls, 337 layers) |
| **Recursive STARK** | Compresses GKR proof (~46K felts) into ~950 felts via STWO Circle STARK |
| **On-chain Verifier** | Cairo contract replays Fiat-Shamir transcript, checks FRI + Merkle proofs |
| **Weight Binding** | Poseidon Merkle roots bind each weight matrix — swap detection is instant |

---

## 1. Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | NVIDIA A10G / RTX 4090 | NVIDIA H100 |
| CUDA | 12.0+ | 12.4+ |
| RAM | 32 GB | 64 GB+ |
| Disk | 60 GB | 100 GB |
| Rust | nightly-2025-07-14 | nightly-2025-07-14 |
| Node.js | 18+ | 20 LTS |

```bash
# Rust nightly (required — STWO uses nightly features)
rustup install nightly-2025-07-14
rustup default nightly-2025-07-14

# Node.js (for on-chain submission)
# Ubuntu: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash - && sudo apt install -y nodejs
# macOS:  brew install node
```

---

## 2. Instant Access (no build required)

Three ways to run verifiable inference right now, from fastest to most independent:

### Option A: SSH into the H100 (branded CLI)

```bash
npm i -g @bitsagecli/cli
bitsage login --api-key <your-key>
bitsage shell h100-prover
```

You land in a branded H100 environment with Qwen2.5-14B + GLM-4-9B loaded. Type `prove` to generate an on-chain proof. The server is already running — no setup needed.

### Option B: Use the Python/TypeScript SDK (no SSH)

```bash
pip install obelyzk

python3 -c "
from obelyzk import Client
c = Client('http://62.169.159.231:8080', timeout=600)
r = c.chat('What is AI?')
print(r['text'])
print(r['tx_hash'])
print(r['explorer_url'])
"
```

TypeScript:
```bash
npm install @obelyzk/sdk

node -e "
import { createStwoProverClient } from '@obelyzk/sdk';
const p = createStwoProverClient({ baseUrl: 'http://62.169.159.231:8080' });
const r = await p.chat('What is AI?');
console.log(r.text, r.txHash, r.explorerUrl);
"
```

### Option C: curl (raw API, zero dependencies)

```bash
curl -s -X POST http://62.169.159.231:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"What is AI?"}],"max_tokens":1}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('obelyzk',{}); print(f'Text: {d[\"choices\"][0][\"message\"][\"content\"]}'); print(f'TX: {m.get(\"tx_hash\")}'); print(f'Explorer: {m.get(\"explorer_url\")}')"
```

All three paths hit the same proving pipeline: GKR sumcheck → recursive STARK → Starknet verification.

---

## 3. Build from Source (fully independent)

```bash
git clone https://github.com/Bitsage-Network/obelyzk.rs.git
cd obelyzk.rs/engine

# GPU build (CUDA 12+ required)
cargo build --release --bin obelyzk \
  --features "server,server-stream,cuda-runtime,cli"

# CPU-only build (slower, no GPU needed)
cargo build --release --bin obelyzk \
  --features "server,cli"
```

The binary is at `target/release/obelyzk`.

---

## 4. Supported Models

### Full ZK Proof (open weights — every operation cryptographically proven)

| Model | Params | Architecture | Status | Download |
|-------|--------|-------------|--------|----------|
| **Qwen2.5-14B** | 14B | Qwen2 (GQA) | Verified on-chain | `huggingface-cli download Qwen/Qwen2.5-14B` |
| **Qwen2.5-7B** | 7B | Qwen2 (GQA) | Supported | `huggingface-cli download Qwen/Qwen2.5-7B` |
| **LLaMA-3.1-8B** | 8B | LLaMA (GQA) | Supported | `huggingface-cli download meta-llama/Llama-3.1-8B` |
| **Mistral-7B** | 7B | Mistral (GQA+SWA) | Supported | `huggingface-cli download mistralai/Mistral-7B-v0.3` |
| **Mixtral-8x7B** | 47B | MoE (8 experts) | Supported | `huggingface-cli download mistralai/Mixtral-8x7B-v0.1` |
| **Phi-3-mini** | 3.8B | Phi (fused QKV) | Supported | `huggingface-cli download microsoft/Phi-3-mini-4k-instruct` |
| **GLM-4-9B** | 9B | ChatGLM (fused QKV+GQA) | Supported (v0.4.0+) | `huggingface-cli download THUDM/glm-4-9b` |
| **SmolLM2-135M** | 135M | SmolLM | Supported | See below |
| **Gemma-2B** | 2B | Gemma | Supported | `huggingface-cli download google/gemma-2b` |
| **MiniMax-M2.5** | 256B MoE | 256 experts, sigmoid | Config supported, needs FP8 dequant | — |
| **Kimi-K2.5** | 1T MoE | MLA + 384 experts | Config supported, needs MLA attention | — |

Any HuggingFace SafeTensors transformer with standard Q/K/V/O + SwiGLU FFN is auto-detected.
GGUF format (Q4_K, Q8_0, F16, BF16) is also supported.

### TLS Attestation (closed APIs — certificate-verified commitments)

| Provider | Models | Status |
|----------|--------|--------|
| Anthropic | Claude 3.5/4 | Supported |
| OpenAI | GPT-4o, o1, o3 | Supported |
| MiniMax API | MiniMax-M2.5 | Planned |
| Moonshot API | Kimi-K2.5 | Planned |

### Download a Model

**SmolLM2-135M** (small, fast — good for testing):
```bash
mkdir -p ~/.obelyzk/models/smollm2-135m && cd $_
for f in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  curl -sL "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/$f" -o $f
done
```

**Qwen2.5-14B** (production — 14B params, verified on Starknet):
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-14B --local-dir ~/.obelyzk/models/qwen2.5-14b
```

**GLM-4-9B** (ChatGLM — fused QKV attention):
```bash
huggingface-cli download THUDM/glm-4-9b --local-dir ~/.obelyzk/models/glm-4-9b
```

> Large models (7B+) are ~15-30 GB. Weight loading takes ~125s on first run (cached after).

---

## 4. Prove a Single Inference

### One command — prove + recursive STARK + on-chain submission

```bash
echo "What is 2+2?" | \
  OBELYSK_MODEL_DIR=~/.obelyzk/models/qwen2.5-14b \
  OBELYZK_MAX_TOKENS=1 \
  STARKNET_PRIVATE_KEY=0x<your-deployer-key> \
  STARKNET_ACCOUNT=0x<your-account-address> \
  RUST_MIN_STACK=16777216 \
  ./target/release/obelyzk chat --model local
```

This runs the full pipeline automatically:

1. **Tokenize** input text
2. **Forward pass** through all 48 transformer layers (GPU-accelerated)
3. **GKR proof** — 192 matmul sumcheck reductions, 337 layer proofs (~46s on H100)
4. **Recursive STARK** — compress to ~950 felts (~1.2s)
5. **Register model** on Starknet (if not already registered)
6. **Submit `verify_recursive` TX** — single transaction, full STARK verification on-chain
7. **Report TX hash** and explorer link

### Without on-chain submission (local proof only)

Omit `STARKNET_PRIVATE_KEY`:

```bash
echo "Hello" | \
  OBELYSK_MODEL_DIR=~/.obelyzk/models/qwen2.5-14b \
  OBELYZK_MAX_TOKENS=1 \
  RUST_MIN_STACK=16777216 \
  ./target/release/obelyzk chat --model local
```

The proof is saved to `/tmp/obelyzk_recursive_prefill.json`. Submit manually later:

```bash
cd obelyzk.rs/engine/scripts && npm install starknet

STARKNET_PRIVATE_KEY=0x... \
STARKNET_ACCOUNT=0x... \
node submit_recursive.mjs /tmp/obelyzk_recursive_prefill.json
```

---

## 5. What's in the Proof

Each on-chain transaction contains these **named, decoded fields** visible in any block explorer:

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `model_id` | felt252 | `0x05e8dc...6eaf` | Poseidon hash of all weight commitments (unique per model) |
| `io_commitment` | felt252 | `0x06cf7d...d450` | Poseidon hash of input tokens + output logits |
| `circuit_hash` | felt252 | `0x0998f3...6fa2` | Model architecture fingerprint |
| `weight_super_root` | felt252 | `0x05cc23...2a82` | Poseidon Merkle root of all 192 weight matrices |
| `n_layers` | u32 | 337 | Total GKR layers proven |
| `n_matmuls` | u32 | 192 | Number of matmul sumcheck reductions |
| `hidden_size` | u32 | 5120 | Model hidden dimension |
| `num_transformer_blocks` | u32 | 48 | Transformer block count |
| `policy_commitment` | felt252 | `0x0178ed...6162` | Proving policy hash |
| `trace_log_size` | u32 | 15 | STARK trace: 2^15 = 32,768 rows |
| `stark_proof_data` | Array | ~950 felts | FRI + Merkle decommitments |

The contract performs **full cryptographic STARK verification** on-chain:
- AIR boundary constraints (Poseidon digest chain)
- Merkle tree decommitments (28 trace columns)
- FRI low-degree proof (polynomial proximity)
- Proof of work (grinding resistance)

---

## 6. Verify a Proof Independently

### On the block explorer

```
https://sepolia.starkscan.co/tx/<tx_hash>
```

Click **Events** to see the `RecursiveProofVerified` event with all metadata.

### Query the contract

```javascript
import { RpcProvider } from "starknet";

const provider = new RpcProvider({
  nodeUrl: "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo",
});

const CONTRACT = "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7";

// Check if a specific proof was verified
const verified = await provider.callContract({
  contractAddress: CONTRACT,
  entrypoint: "is_recursive_proof_verified",
  calldata: ["<proof_hash>"],
});
console.log("Verified:", verified[0] === "0x1");

// Get verification count for a model
const count = await provider.callContract({
  contractAddress: CONTRACT,
  entrypoint: "get_recursive_verification_count",
  calldata: ["<model_id>"],
});
console.log("Total verifications:", Number(BigInt(count[0])));

// Get last verification details
const details = await provider.callContract({
  contractAddress: CONTRACT,
  entrypoint: "get_last_verification",
  calldata: ["<model_id>"],
});
console.log("Last IO commitment:", details[0]);
console.log("Last proof hash:", details[1]);
console.log("Timestamp:", Number(BigInt(details[2])));
console.log("Proof felts:", Number(BigInt(details[3])));
console.log("Layers:", Number(BigInt(details[4])));
console.log("Trace log_size:", Number(BigInt(details[5])));
console.log("Verification #:", Number(BigInt(details[6])));
```

### Reproduce the proof locally

Anyone with the same model weights can independently generate the same proof:

```bash
# 1. Download the same model
huggingface-cli download Qwen/Qwen2.5-14B --local-dir /tmp/qwen2.5-14b

# 2. Run the same inference with the same input
echo "What is 2+2?" | \
  OBELYSK_MODEL_DIR=/tmp/qwen2.5-14b \
  OBELYZK_MAX_TOKENS=1 \
  RUST_MIN_STACK=16777216 \
  ./target/release/obelyzk chat --model local

# 3. Compare the io_commitment — it must match the on-chain value
# The weight_super_root will also match (same weights → same Merkle root)
```

---

## 7. Contract Reference

**Verifier contract**: [`0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7)

| Entrypoint | Type | Description |
|------------|------|-------------|
| `verify_recursive` | External | Full STARK verification + record proof fact |
| `register_model_recursive` | External | Register model (circuit_hash + weight_root) |
| `is_recursive_proof_verified` | View | Check if a proof_hash was verified |
| `get_recursive_verification_count` | View | Total verifications for a model |
| `get_recursive_model_info` | View | Registered model details |
| `get_last_verification` | View | Last proof details (io, hash, timestamp, felts, layers) |
| `get_model_policy` | View | Registered policy commitment |
| `propose_upgrade` | External | Owner-only timelock upgrade (5 min delay) |
| `execute_upgrade` | External | Execute pending upgrade |

---

## 8. Performance Benchmarks (H100 PCIe)

### Qwen2.5-14B (14 billion parameters, 48 layers, 192 matmuls)

| Phase | Time | Notes |
|-------|------|-------|
| Weight loading | 125s | One-time per model (8 SafeTensor shards, 30 GB) |
| GPU forward pass | ~10s | 192 matmuls across 48 transformer blocks |
| GKR proof (GPU sumcheck) | 46s | 337 layer proofs, 192 matmul reductions |
| Weight commitments | 0s | Warm cache (auto-saved to disk after first run) |
| Recursive STARK | 1.2s | 22,771 Poseidon perms, log_size=15 |
| On-chain submission | ~15s | Model registration (once) + verify TX |
| **Total per token (warm)** | **~72s** | Includes on-chain confirmation |

### GLM-4-9B (9 billion parameters, 40 layers, 160 matmuls)

| Phase | Time | Notes |
|-------|------|-------|
| Weight loading | ~80s | One-time (10 SafeTensor shards, 18 GB) |
| Forward pass | ~56s | 160 matmuls, fused QKV auto-split |
| GKR proof | 201s | 281 layer proofs, 160 matmul reductions |
| Weight commitments | 0s | Warm cache |
| Recursive STARK | 1.1s | 18,276 Poseidon perms, log_size=15 |
| On-chain submission | ~15s | Auto-registration + verify TX |
| **Total per token (warm)** | **~273s** | GLM uses CPU forward (GPU path coming) |

### MiniMax-M2.5 (estimated — 256-expert MoE, requires 8× H100)

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Weight loading | ~10 min | 400 GB FP8, 8-way parallel shard loading |
| Forward pass | ~5s | Only 8/256 experts active per token |
| GKR proof | ~15s | ~496 matmuls (8 active experts × 62 layers) |
| Weight commitments (cold) | ~30 min | 15,872 matrices, 8-way parallel |
| Weight commitments (warm) | 0s | Cached after first run |
| Recursive STARK | ~2s | Same constant-size compression |
| **Total per token (warm)** | **~22s** | Requires 640 GB HBM3 (8× H100) |

---

## 9. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    obelyzk chat                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Tokenize (tokenizers crate)                         │
│     "What is 2+2?" → [1024, 318, 220, 17, 10, 17, 30]  │
│                                                         │
│  2. Embed (load_embedding_row)                          │
│     token_id → (1, 5120) M31 matrix                     │
│                                                         │
│  3. Forward Pass (GPU GEMV, 48 transformer blocks)      │
│     ┌──────────────────────────────────────┐             │
│     │  RMSNorm → Q/K/V matmul → FFN       │ × 48       │
│     │  gate×up → down_proj → Add(residual) │             │
│     └──────────────────────────────────────┘             │
│     → (1, 5120) output hidden state                     │
│                                                         │
│  4. GKR Proof (GPU sumcheck reductions)                 │
│     192 matmul sumchecks over M31/QM31 fields           │
│     Poseidon Fiat-Shamir channel (deterministic)        │
│     Weight binding: Poseidon Merkle roots per matrix    │
│     → GKRProof (337 layer proofs, ~46K felts)           │
│                                                         │
│  5. Recursive STARK (STWO Circle STARK prover)          │
│     GKR verifier witness → execution trace              │
│     → STARK proof (~950 calldata felts)                 │
│                                                         │
│  6. On-Chain (Starknet Sepolia)                         │
│     register_model_recursive() — bind model identity    │
│     verify_recursive() — full STARK verification        │
│     → RecursiveProofVerified event (rich metadata)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OBELYSK_MODEL_DIR` | — | Path to model directory (SafeTensors or GGUF) |
| `OBELYZK_MAX_TOKENS` | 1 | Max tokens to generate per inference |
| `STARKNET_PRIVATE_KEY` | — | Deployer private key (enables auto on-chain submission) |
| `STARKNET_ACCOUNT` | — | Deployer account address |
| `OBELYZK_TRUST_WEIGHT_CLAIMS` | — | Set to `1` for fast recursive STARK (skips MLE re-eval) |
| `OBELYZK_GPU_FORWARD` | `1` | Set to `0` to force CPU forward pass |
| `OBELYZK_BATCH_SIZE` | 1 | Tokens per proof batch |
| `RUST_MIN_STACK` | — | Set to `16777216` (16 MB) for large models |

---

## 11. Verified Proofs on Sepolia

| TX | Model | Params | Felts | GKR | STARK |
|----|-------|--------|-------|-----|-------|
| [`0x5ce1b41...`](https://sepolia.starkscan.co/tx/0x5ce1b41815e29a7b3dd03b77187cf32c8c5f0e2607960303174cbea303edfd3) | Qwen2.5-14B | 14B | 946 | 46s | 1.2s |
| [`0x542960d...`](https://sepolia.starkscan.co/tx/0x542960d703a62d4beaacf0d9094ea92dc86bf326cd917c533039f4dd1eb4a30) | GLM-4-9B | 9B | 929 | 201s | 1.1s |
| [`0x16c9fa1...`](https://sepolia.starkscan.co/tx/0x16c9fa1a9da0a388125e4d27e11b8eff6dd663f911b38e0f799d12e4cf15feb) | GLM-4-9B | 9B | 970 | — | — |
| [`0x67a7b92...`](https://sepolia.starkscan.co/tx/0x67a7b9259d874aa40d593ac55fa47f3c4db6836f20893db718334d56ac0f0d9) | Qwen2.5-14B | 14B | 927 | — | — |
| [`0x38a156d...`](https://sepolia.starkscan.co/tx/0x38a156d972cdc111f40bca7dedf056f42031088daf434d3849a1352da713317) | Qwen2.5-14B | 14B | 1,007 | — | — |

Two different model architectures (Qwen2 + ChatGLM), both verified on-chain in single transactions.

### SDK-submitted proofs

| SDK | TX | Felts |
|-----|-----|-------|
| Python (`pip install obelyzk`) | [`0x677694b...`](https://sepolia.starkscan.co/tx/0x677694b934d9bd6d8d2f984acb26b7aff8204d162e2bd929c729fee060fa890) | 892 |
| TypeScript (`npm install @obelyzk/sdk`) | [`0x534424...`](https://sepolia.starkscan.co/tx/0x53442404ace74ca5391f90214ff5d4695c7ca4eefc5ea053a28464bceeecf42) | 1,007 |
| curl (raw API) | [`0x4d868f...`](https://sepolia.starkscan.co/tx/0x4d868fe123f821af0b09c506f7307056ac23afe53f3d02a6d44ea6089f4b790) | 1,007 |

All SDKs call `/v1/chat/completions` on the `obelyzk serve` instance. The server runs the full pipeline (GKR → STARK → on-chain) and returns `tx_hash` + `explorer_url` in the response.

---

## 12. Troubleshooting

### Model download script

Use the interactive download script for any supported model:

```bash
./scripts/download_model.sh           # show all models
./scripts/download_model.sh glm-4-9b  # download GLM-4
./scripts/download_model.sh qwen2.5-14b
```

Or models auto-download when you point `OBELYSK_MODEL_DIR` at a model name:
```bash
# Auto-downloads THUDM/glm-4-9b if not present
OBELYSK_MODEL_DIR=~/.obelyzk/models/glm-4-9b obelyzk chat --model local
```

### GLM-4 tokenizer

GLM-4 uses a tiktoken-based tokenizer that needs conversion. If you see `tokenizer.json not found`:

```bash
pip install transformers tiktoken
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('path/to/glm-4-9b', trust_remote_code=True)
import json
vocab = {tok.decode([i]): i for i in range(tok.vocab_size)}
with open('path/to/glm-4-9b/tokenizer.json', 'w') as f:
    json.dump({'version':'1.0','model':{'type':'BPE','vocab':vocab,'merges':[]}}, f)
"
```

### "CUDA not detected"

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version  # must be 12.0+
```

The prover falls back to CPU automatically but proving is ~10x slower.

### "Model not registered" (on-chain)

The model must be registered before the first proof. The pipeline does this automatically when `STARKNET_PRIVATE_KEY` is set. To register manually:

```bash
STARKNET_PRIVATE_KEY=0x... node scripts/submit_recursive.mjs proof.json
```

### "Already verified"

The contract deduplicates proofs by `(model_id, io_commitment)`. Run a different inference (different input) to generate a new proof.

### Out of GPU memory

Stop any running `obelyzk serve` process first:
```bash
sudo systemctl stop obelyzk.service
nvidia-smi  # verify GPU is free
```

### Stack overflow on large models

```bash
export RUST_MIN_STACK=16777216  # 16 MB stack
```

---

## 13. SDKs

### Rust (crates.io)

```bash
cargo add obelyzk  # v0.4.0
```

```rust
use obelyzk::providers::local::LocalProvider;

let provider = LocalProvider::load(Path::new("path/to/model"), None)?;
let (text, ids, proof_meta) = provider.generate_with_proof("What is AI?", 1, |_, _, _| {})?;
println!("TX: {:?}", proof_meta.tx_hash);
println!("Explorer: {:?}", proof_meta.explorer_url);
```

### TypeScript (npm)

```bash
npm install @obelyzk/sdk  # v1.6.0
```

```typescript
import { createStwoProverClient } from "@obelyzk/sdk";

const prover = createStwoProverClient({ baseUrl: "http://localhost:9090" });
const result = await prover.chat("What is AI?", { model: "local" });
console.log(result.text);         // model output
console.log(result.txHash);       // Starknet TX hash
console.log(result.explorerUrl);  // Starkscan link
console.log(result.calldataFelts); // ~950
```

### Python (PyPI)

```bash
pip install obelyzk  # v0.4.0
```

```python
from obelyzk import Client

client = Client("http://localhost:9090")
result = client.chat("What is AI?")
print(result["text"])           # model output
print(result["tx_hash"])        # Starknet TX hash
print(result["explorer_url"])   # Starkscan link
print(result["calldata_felts"]) # ~950
```

---

## 14. Links

| Resource | URL |
|----------|-----|
| GitHub | https://github.com/Bitsage-Network/obelyzk.rs |
| Contract (Sepolia) | https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7 |
| crates.io | https://crates.io/crates/obelyzk |
| npm SDK | https://www.npmjs.com/package/@obelyzk/sdk |
| PyPI SDK | https://pypi.org/project/obelyzk |
| Download models | `./scripts/download_model.sh` |
