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

| Phase | Time |
|-------|------|
| Model loading | ~5s |
| Forward pass (per inference) | ~3s |
| ZK proof (per inference) | ~8s |
| **3 inferences end-to-end** | **~37s** |

## Troubleshooting

**`nvidia-smi` not found**: Install NVIDIA drivers first.

**Build fails on CUDA**: Make sure `nvcc` is in your PATH. Run `export PATH=/usr/local/cuda/bin:$PATH`.

**Model download slow**: Use `--local-dir` to point to an existing download.

**Out of memory**: Reduce `--layers` (try 2 instead of 5) or use a smaller model preset.
