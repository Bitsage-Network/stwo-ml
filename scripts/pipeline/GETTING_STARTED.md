# Getting Started with Obelysk Pipeline

Prove that an ML model ran correctly, and verify the proof on-chain. Works on any NVIDIA GPU.

---

## What You Need

- A machine with an NVIDIA GPU (RTX 4090, A100, H100, H200, B200, or B300)
- Ubuntu/Debian or RHEL/Rocky Linux
- Internet connection
- ~50GB free disk space (for model + build + proof)

**Optional (for on-chain verification):**
- A Starknet wallet private key
- A HuggingFace token (only for gated models like Llama or Gemma)

---

## The Fastest Way (One Command)

SSH into your GPU machine and run:

```bash
git clone https://github.com/Bitsage-Network/stwo-ml.git
cd stwo-ml/scripts/pipeline

# Dry run (no on-chain submission) — good for first try
./run_e2e.sh --preset phi3-mini --gpu --dry-run
```

This does everything: installs drivers, downloads the model, tests it, generates a proof, and verifies it locally.

To also submit the proof on-chain:

```bash
STARKNET_PRIVATE_KEY=0x_your_key_here ./run_e2e.sh --preset qwen3-14b --gpu --submit
```

That's it. The rest of this guide explains what each step does and how to run them individually.

---

## Step-by-Step Guide

### Step 0 — Set Up the GPU Machine

**What it does:** Installs system packages, NVIDIA drivers, CUDA, Rust, and builds the prover binary. Also builds llama.cpp for inference testing.

```bash
./00_setup_gpu.sh
```

When it finishes, your GPU is detected and ready. The script saves its config to `~/.obelysk/gpu_config.env`.

**Common flags:**
| Flag | What it does |
|------|--------------|
| `--install-drivers` | Force install NVIDIA drivers + CUDA |
| `--skip-drivers` | Skip driver install (already installed on cloud instances) |
| `--skip-build` | Skip building Rust binaries (already built) |
| `--skip-llama` | Skip building llama.cpp |

**Example — cloud instance with pre-installed drivers:**
```bash
./00_setup_gpu.sh --skip-drivers
```

---

### Step 1 — Download a Model

**What it does:** Downloads a model from HuggingFace and saves it to `~/.obelysk/models/`.

```bash
# Pick a preset
./01_setup_model.sh --preset phi3-mini        # 7GB, fastest for testing
./01_setup_model.sh --preset qwen3-14b        # 28GB, production default
./01_setup_model.sh --preset mistral-7b       # 15GB

# See all presets
./01_setup_model.sh --list
```

**Available presets:**

| Preset | Size | Needs HF Token? |
|--------|------|-----------------|
| `phi3-mini` | 7GB | No |
| `mistral-7b` | 15GB | No |
| `llama3-8b` | 16GB | Yes |
| `gemma2-9b` | 18GB | Yes |
| `qwen3-14b` | 28GB | No |
| `llama3-70b` | 140GB | Yes |

**For gated models (Llama, Gemma):**
1. Go to https://huggingface.co/settings/tokens and create a token
2. Accept the model's license on its HuggingFace page
3. Run:
```bash
HF_TOKEN=hf_your_token ./01_setup_model.sh --preset llama3-8b
```

**Custom model (not in presets):**
```bash
./01_setup_model.sh --hf-model Qwen/Qwen3-0.5B --layers 24
```

---

### Step 2 — Validate the Model

**What it does:** Checks that all model files downloaded correctly, dimensions match, and weights are valid.

```bash
./02_validate_model.sh
```

This runs automatically. No flags needed.

---

### Step 2a — Test Inference (Optional)

**What it does:** Converts the model to GGUF format and runs it through llama.cpp to confirm it produces real output. This is optional but recommended to verify the model works before proving.

```bash
# Quick test — ask it a question
./02a_test_inference.sh --model-name phi3-mini

# Custom prompt
./02a_test_inference.sh --model-name phi3-mini --prompt "Explain gravity in one sentence"

# Interactive chat
./02a_test_inference.sh --model-name phi3-mini --chat

# Speed benchmark
./02a_test_inference.sh --model-name phi3-mini --benchmark
```

---

### Step 3 — Generate the Proof

**What it does:** Runs the model through the prover, which generates a cryptographic proof that the inference was computed correctly. Then verifies the proof locally before saving it.

```bash
# Default mode (GKR) — fastest
./03_prove.sh --model-name qwen3-14b --gpu

# Prove just 1 layer (faster, good for testing)
./03_prove.sh --model-name qwen3-14b --layers 1 --gpu

# Multi-GPU (if you have multiple GPUs)
./03_prove.sh --model-name qwen3-14b --gpu --multi-gpu
```

**Proof modes:**

| Mode | Speed | Security | Command |
|------|-------|----------|---------|
| `gkr` | Fastest | High | `--mode gkr` (default) |
| `direct` | Moderate | High | `--mode direct` |
| `recursive` | Slowest | Highest | `--mode recursive` |

The proof is saved to `~/.obelysk/proofs/`.

---

### Step 4 — Verify On-Chain

**What it does:** Submits the proof to the Starknet smart contract, waits for the transaction to confirm, and checks that `is_verified` returns true.

**You need a Starknet private key for this step.**

```bash
# Dry run first — shows what will be submitted without sending anything
./04_verify_onchain.sh --dry-run

# Submit for real
STARKNET_PRIVATE_KEY=0x_your_key ./04_verify_onchain.sh --submit
```

The script will:
1. Auto-create a Starknet account (if needed)
2. Submit the proof transaction
3. Wait for confirmation (~30 seconds)
4. Check `is_verified()` on the contract
5. Print the explorer link

---

## Quick Reference

### One-Command Examples

```bash
# Test everything locally (no on-chain, smallest model)
./run_e2e.sh --preset phi3-mini --gpu --dry-run

# Full pipeline with on-chain verification
STARKNET_PRIVATE_KEY=0x... ./run_e2e.sh --preset qwen3-14b --gpu --submit

# Gated model with HF auth + on-chain
HF_TOKEN=hf_xxx STARKNET_PRIVATE_KEY=0x... ./run_e2e.sh --preset llama3-8b --gpu --submit

# Chat with the model before proving
./run_e2e.sh --preset phi3-mini --gpu --chat --dry-run

# Resume from a failed step
./run_e2e.sh --preset qwen3-14b --resume-from prove --gpu --submit

# Skip setup (machine already configured)
./run_e2e.sh --preset qwen3-14b --skip-setup --gpu --submit
```

### Environment Variables

| Variable | What it does |
|----------|--------------|
| `STARKNET_PRIVATE_KEY` | Your Starknet wallet key (for on-chain) |
| `HF_TOKEN` | HuggingFace token (for gated models) |
| `DRY_RUN=1` | Print commands without running them |
| `OBELYSK_DEBUG=1` | Show verbose debug output |

### Where Things Are Saved

```
~/.obelysk/
  models/phi3-mini/       <- Downloaded model files
  proofs/phi3-mini_.../   <- Generated proofs
  llama.cpp/              <- Built llama.cpp (for inference testing)
  gpu_config.env          <- Detected GPU info
```

---

## GPU Compatibility

| GPU | VRAM | Works? |
|-----|------|--------|
| RTX 4090 | 24GB | Yes (small models or few layers) |
| A100 | 40-80GB | Yes |
| H100 | 80GB | Yes |
| H200 | 141GB | Yes |
| B200 | 192GB | Yes |
| B300 | 288GB | Yes (can prove 70B+ models) |

---

## Troubleshooting

**"nvidia-smi not found"**
```bash
./00_setup_gpu.sh --install-drivers
# Reboot if needed: sudo reboot
```

**"CUDA not found"**
```bash
./00_setup_gpu.sh --install-drivers
```

**"Permission denied" on model download**
The model is gated. Get a HuggingFace token and set `HF_TOKEN`.

**"Not enough disk space"**
You need model_size x 1.5 free. For Qwen3-14B, that's ~50GB.

**Proof fails on-chain but passes locally**
```bash
# Check TX status
sncast tx-status 0xYOUR_TX_HASH
```

**Want to start over?**
```bash
rm -rf ~/.obelysk
```
