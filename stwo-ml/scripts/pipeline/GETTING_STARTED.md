# Obelysk: Prove Your ML Model in One Command

Prove that a machine learning model ran correctly on GPU, with a verifiable ZK proof submitted on-chain.

## What You Need

- NVIDIA GPU (H100, A100, RTX 4090, or similar)
- Ubuntu/Debian Linux
- ~50 GB free disk space

## Quick Start

```bash
git clone https://github.com/Bitsage-Network/stwo-ml.git
cd stwo-ml

# Build + capture + prove + submit (all in one)
./scripts/pipeline/run_e2e.sh --preset qwen3-14b --gpu --submit
```

That single command will:

1. **Build** the prover binary (first run only, ~2 min)
2. **Run** 3 inference forward passes and log them
3. **Prove** each inference with ZK proofs (~8s each on H100)
4. **Generate** a full audit report
5. **Submit** the proof on-chain to Starknet

## Step by Step

If you prefer to run each stage separately:

### 1. Build

```bash
cargo build --release --features std,gpu,cuda-runtime,onnx,safetensors,model-loading,cli,audit
```

### 2. Download a model

```bash
# Qwen3-14B (recommended, 28 GB, no auth needed)
huggingface-cli download Qwen/Qwen3-14B --local-dir ~/.obelysk/models/qwen3-14b
```

### 3. Capture inference logs

```bash
./target/release/prove-model capture \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 5 \
  --log-dir /tmp/inference_logs \
  --count 3
```

### 4. Run the audit

```bash
# Dry run (no on-chain submission)
./target/release/prove-model audit \
  --log-dir /tmp/inference_logs \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 5 \
  --gpu \
  --evaluate \
  --dry-run \
  --output audit_report.json

# Full run (submits proof on-chain)
./target/release/prove-model audit \
  --log-dir /tmp/inference_logs \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 5 \
  --gpu \
  --evaluate \
  --submit \
  --output audit_report.json
```

## Standalone GKR Proving

For direct model proving without the audit pipeline, use `--format ml_gkr`:

### Prove

```bash
./target/release/prove-model \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 1 \
  --gkr \
  --format ml_gkr \
  --output proof.json
```

### Dry-run (health check + step estimation, no submission)

```bash
./target/release/prove-model \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 1 \
  --gkr \
  --format ml_gkr \
  --dry-run \
  --output proof.json
```

### Verify an existing proof

```bash
./target/release/prove-model \
  --verify-proof proof.json \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 1
```

### Submit on-chain (gasless via AVNU paymaster)

```bash
./target/release/prove-model \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 1 \
  --gkr \
  --format ml_gkr \
  --submit-paymaster \
  --output proof.json
```

### Submit on-chain (streaming multi-TX)

For proofs that exceed single-TX calldata limits, use the streaming pipeline.
This is the primary submission path for production proofs.

```bash
# 1. Generate the proof (on GPU)
./target/release/prove-model \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --layers 1 \
  --gkr \
  --format ml_gkr \
  --output proof.json

# 2. Register model + open session + upload + submit all verification steps
node scripts/pipeline/register_and_submit.mjs proof.json
```

On subsequent runs with the same model, skip registration:

```bash
node scripts/pipeline/register_and_submit.mjs proof.json --skip-register
```

The script handles 18 transactions in sequence:

| Phase | TXs | Description |
|-------|-----|-------------|
| Session management | 4 | open_gkr_session, upload 2 chunks, seal |
| Verification | 14 | init, output_mle x5, layers x1, weight_binding, input_mle x5, finalize |

**Step ordering is critical.** The protocol requires:
`init` -> `output_mle` (all chunks) -> `layers` (all batches) -> `weight_binding` -> `input_mle` (all chunks) -> `finalize`

Output MLE **must** come before layers due to channel state dependencies.

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `STARKNET_ACCOUNT` | Account address (defaults to deployer) |
| `STARKNET_PRIVATE_KEY` | Account private key |
| `STARKNET_RPC` | RPC endpoint URL |
| `CONTRACT_ADDRESS` | Verifier contract address |

**Contract info (Sepolia):**
- Contract: `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`
- Current class (v31): `0x6a6b7a75d5ec1f63d715617d352bc0d353042b2a033d98fa28ffbaf6c5b5439`

### Pre-warm weight cache

First-time proving computes Merkle roots for all weight matrices (~60s for Qwen3-14B).
Pre-warm the cache to make subsequent proves instant:

```bash
./target/release/prove-model \
  --model-dir ~/.obelysk/models/qwen3-14b \
  --generate-cache
```

### Useful flags

| Flag | Description |
|------|-------------|
| `--format ml_gkr` | Recommended format for GKR proofs |
| `--gkr` | Enable GKR proving (required for `ml_gkr`) |
| `--dry-run` | Prove + health check + step estimation, skip submission |
| `--health-check` | Run structural health check after proving |
| `--verify-proof <path>` | Verify an existing proof file |
| `--generate-cache` | Pre-compute weight cache and exit |
| `--quiet` | Suppress verbose diagnostic output |
| `--submit-paymaster` | Submit via AVNU gasless paymaster |
| `--submit-gkr` | Submit via sncast (requires funded account) |

## Available Models

| Model | Size | Auth | Preset |
|-------|------|------|--------|
| Qwen3-14B | 28 GB | None | `qwen3-14b` |
| Phi-3 Mini | 7 GB | None | `phi3-mini` |
| Llama 3 8B | 16 GB | HF Token | `llama3-8b` |

## What the Audit Report Contains

- **Proof mode**: `direct` (aggregated STARK, ~8s/inference on H100)
- **Commitments**: IO Merkle root, weight commitment, inference log root
- **Proof calldata**: Ready for on-chain verification
- **Semantic evaluation**: Deterministic correctness checks

## Performance (H100)

| Phase | First Run | Cached |
|-------|-----------|--------|
| Model loading | ~5s | ~5s |
| Weight cache pre-warm | ~60s | 0s (cached) |
| Forward pass (1 layer) | ~1s | ~1s |
| GKR prove (1 layer) | ~3s | ~3s |
| Serialization | ~2.5s | ~2.5s |
| **Total (1 layer, first run)** | **~140s** | **~8s** |
| **Total (1 layer, cached)** | — | **~8s** |

## Troubleshooting

**`nvidia-smi` not found**: Install NVIDIA drivers first.

**Build fails on CUDA**: Make sure `nvcc` is in your PATH. Run `export PATH=/usr/local/cuda/bin:$PATH`.

**Model download slow**: Use `--local-dir` to point to an existing download.

**Out of memory**: Reduce `--layers` (try 2 instead of 5) or use a smaller model preset.
Lower `STWO_GPU_MERKLE_THRESHOLD` (default 4096) to reduce GPU memory pressure.

**Health check shows FAIL**: Update to the latest version and re-prove. Old proof files
may use a different calldata layout. Use `--format ml_gkr` for the current format.

**Soundness gate rejected**: Check environment variables. Default settings should work.
See `docs/ENV_VARS.md` for a full reference.

**Weight commitment is slow**: Use `--generate-cache` to pre-compute the cache once.
Subsequent runs will load cached roots in <1ms.

**GPU OOM during proving**: Lower `STWO_GPU_MERKLE_THRESHOLD` (e.g., `export STWO_GPU_MERKLE_THRESHOLD=2048`).

**Streaming submission fails**: Ensure `STARKNET_ACCOUNT`, `STARKNET_PRIVATE_KEY` are set,
and `node` (v18+) is available for the paymaster scripts. Check that the account has
sufficient STRK balance (~5-8 STRK for the 18 TXs). Install starknet.js: `npm install starknet`.

**BINDING_SUPER_ROOT_FAILED**: The contract class is outdated. Upgrade to v31+
(`0x6a6b7a75d5ec1f63d715617d352bc0d353042b2a033d98fa28ffbaf6c5b5439`).

**Streaming steps fail mid-way**: You can resume by passing `--skip-register --skip-session`
with the `SESSION_ID` env var set to the session ID from the open_gkr_session TX.

## Environment Variables

See `docs/ENV_VARS.md` for a full categorized reference of all `STWO_*` environment variables.
