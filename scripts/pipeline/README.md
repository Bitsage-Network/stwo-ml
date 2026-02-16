# Obelysk Pipeline

> Note: `scripts/pipeline` is the canonical implementation. Files in `libs/scripts/pipeline` are compatibility wrappers that delegate to `scripts/pipeline` to prevent drift.

End-to-end pipeline for proving ML model inference and verifying on-chain. Works on any NVIDIA GPU from RTX 4090 to B300.

## Quick Start

```bash
# Full pipeline: setup -> download -> validate -> prove -> verify
./run_e2e.sh --preset qwen3-14b --gpu --submit

# Dry run (no on-chain submission)
./run_e2e.sh --preset phi3-mini --gpu --dry-run

# With HuggingFace auth (for gated models like Llama)
HF_TOKEN=hf_xxx ./run_e2e.sh --preset llama3-8b --gpu --submit

# With on-chain submission
STARKNET_PRIVATE_KEY=0x... ./run_e2e.sh --preset qwen3-14b --gpu --submit
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 0 | `00_setup_gpu.sh` | Install deps, drivers, Rust, detect GPU/CUDA, build binaries, build llama.cpp |
| 1 | `01_setup_model.sh` | Download model from HuggingFace (with auth), verify integrity |
| 2 | `02_validate_model.sh` | Validate model files, dimensions, weights |
| 2a | `02a_test_inference.sh` | Test inference via llama.cpp (single prompt, chat, benchmark) |
| 3 | `03_prove.sh` | Generate cryptographic proof with local verification |
| 4 | `04_verify_onchain.sh` | Submit proof to Starknet with TX confirmation + acceptance and assurance classification (`accepted_onchain`, `full_gkr_verified`) |
| E2E | `run_e2e.sh` | Run all steps in sequence with resume support |

## Environment Variables

| Variable | Required | Default | Used By |
|----------|----------|---------|---------|
| `HF_TOKEN` | For gated models | `~/.huggingface/token` | `01_setup_model.sh` |
| `STARKNET_PRIVATE_KEY` | For on-chain | -- | `04_verify_onchain.sh`, `lib/starknet_utils.sh` |
| `STARKNET_RPC` | No | Alchemy/Nethermind | `lib/contract_addresses.sh` |
| `STARKNET_ACCOUNT_ADDRESS` | For auto-account | Derived from key | `lib/starknet_utils.sh` |
| `ALCHEMY_KEY` | No | -- | `lib/contract_addresses.sh` |
| `OBELYSK_DIR` | No | `~/.obelysk` | All scripts |
| `OBELYSK_DEBUG` | No | `0` | `lib/common.sh` |
| `DRY_RUN` | No | `0` | All scripts |
| `REPO_URL` | No | GitHub | `00_setup_gpu.sh` |
| `MAX_FEE` | No | `0.05` ETH | `04_verify_onchain.sh` |

## Model Presets

| Preset | Model | Layers | Size | Auth | Notes |
|--------|-------|--------|------|------|-------|
| `qwen3-14b` | Qwen/Qwen3-14B | 40 | 28GB | No | Default |
| `llama3-8b` | meta-llama/Llama-3.1-8B | 32 | 16GB | Yes | HF_TOKEN required |
| `llama3-70b` | meta-llama/Llama-3.1-70B | 80 | 140GB | Yes | Needs 192GB+ VRAM |
| `mistral-7b` | mistralai/Mistral-7B-v0.3 | 32 | 15GB | No | |
| `phi3-mini` | microsoft/Phi-3-mini-4k | 32 | 7GB | No | Small, fast for testing |
| `gemma2-9b` | google/gemma-2-9b | 42 | 18GB | Yes | HF_TOKEN required |

Custom: `--hf-model Qwen/Qwen3-0.5B --layers 24`

## GPU Compatibility

| GPU | VRAM | Compute | Max Layers | Chunk Budget | CC Support |
|-----|------|---------|------------|--------------|------------|
| RTX 3090 | 24GB | 8.6 | ~5 | 8GB | No |
| RTX 4090 | 24GB | 8.9 | ~10 | 8GB | No |
| A100 40GB | 40GB | 8.0 | All | 16GB | No |
| A100 80GB | 80GB | 8.0 | All | 24GB | No |
| H100 | 80GB | 9.0 | All | 24GB | Yes |
| H200 | 141GB | 9.0 | All | 32GB | Yes |
| B200 | 192GB | 10.0 | All | 48GB | Yes |
| B300 | 288GB | 10.0 | All | 64GB | Yes |

GPU presets in `configs/4090.env`, `configs/b200.env`, `configs/b300.env`.

## Proof Modes

| Mode | Pipeline | On-Chain Function | Notes |
|------|----------|-------------------|-------|
| `gkr` | prove-model (GKR sumcheck) | `verify_model_gkr()` | Default. Fastest |
| `direct` | prove-model -> chunked calldata | `verify_model_direct()` | Partial on-chain cryptographic coverage (not full GKR assurance) |
| `recursive` | prove-model -> cairo-prove -> Circle STARK | Multi-step (9+ TXs) | Most secure |

## Per-Script Usage

### 00_setup_gpu.sh

```bash
./00_setup_gpu.sh                    # Auto-detect and install everything
./00_setup_gpu.sh --install-drivers  # Force driver + CUDA install
./00_setup_gpu.sh --skip-drivers     # Skip driver install (cloud instances)
./00_setup_gpu.sh --skip-deps        # Skip apt/yum packages
./00_setup_gpu.sh --skip-build       # Skip building Rust binaries
./00_setup_gpu.sh --skip-llama       # Skip llama.cpp build
./00_setup_gpu.sh --cuda-path /usr/local/cuda-12.6
```

### 01_setup_model.sh

```bash
./01_setup_model.sh --list                             # Show presets
./01_setup_model.sh --preset qwen3-14b                 # Built-in preset
./01_setup_model.sh --hf-model Qwen/Qwen3-0.5B --layers 24
./01_setup_model.sh --onnx /path/to/model.onnx
./01_setup_model.sh --preset llama3-8b --hf-token hf_xxx
```

### 02a_test_inference.sh

```bash
./02a_test_inference.sh --model-name phi3-mini --prompt "What is 1+1?"
./02a_test_inference.sh --model-name phi3-mini --chat      # Interactive
./02a_test_inference.sh --model-name phi3-mini --benchmark # Speed test
```

### 03_prove.sh

```bash
./03_prove.sh --model-name qwen3-14b --layers 1 --mode gkr --gpu
./03_prove.sh --model-name qwen3-14b --mode recursive --gpu
./03_prove.sh --model-name qwen3-14b --mode gkr --multi-gpu
./03_prove.sh --model-name qwen3-14b --server http://prover:8080
```

### 04_verify_onchain.sh

```bash
./04_verify_onchain.sh --dry-run                        # Print commands
STARKNET_PRIVATE_KEY=0x... ./04_verify_onchain.sh --submit
./04_verify_onchain.sh --submit --max-fee 0.1           # Custom fee
./04_verify_onchain.sh --submit --contract 0x123...     # Custom contract
```

### run_e2e.sh

```bash
./run_e2e.sh --preset qwen3-14b --gpu --submit
./run_e2e.sh --preset phi3-mini --gpu --dry-run
./run_e2e.sh --preset qwen3-14b --resume-from prove --submit
./run_e2e.sh --preset llama3-8b --chat --submit         # Pause for chat
./run_e2e.sh --preset qwen3-14b --skip-setup --skip-inference --submit
```

## Directory Structure

```
~/.obelysk/
  gpu_config.env          # Detected GPU configuration
  cuda_env.sh             # CUDA environment exports
  setup_state.env         # Setup step state
  model_state.env         # Current model state
  prove_state.env         # Last proof state
  inference_state.env     # Inference test state
  models/
    qwen3-14b/
      config.env          # Model configuration
      *.safetensors       # Model weights
      config.json         # HF config
  proofs/
    qwen3-14b_20260214/
      ml_proof.json       # Proof file
      metadata.json       # Proof metadata
      verify_receipt.json # On-chain receipt
  llama.cpp/              # llama.cpp build (for inference testing)
  starknet/
    accounts.json         # Auto-created sncast account
  runs/
    20260214_120000_qwen3-14b/
      run_summary.json    # E2E run summary
```

```
scripts/pipeline/
  lib/
    common.sh               Shared logging, state, prereq checks
    gpu_detect.sh            GPU/CUDA/CC detection + driver install
    model_registry.sh        Model preset definitions
    contract_addresses.sh    Starknet contract/RPC config
    starknet_utils.sh        sncast account, TX wait, verification check
  configs/
    qwen3-14b.env            Model presets
    llama3-8b.env
    llama3-70b.env
    mistral-7b.env
    phi3-mini.env
    gemma2-9b.env
    4090.env                 GPU config presets
    b200.env
    b300.env
    custom.env.template      Template for custom models
  00_setup_gpu.sh            GPU environment + driver setup
  01_setup_model.sh          Model download + HF auth
  02_validate_model.sh       Model validation
  02a_test_inference.sh      Inference testing (llama.cpp)
  03_prove.sh                Proof generation + local verification
  04_verify_onchain.sh       On-chain submission + TX confirmation
  run_e2e.sh                 End-to-end runner
  README.md                  This file
```

## Troubleshooting

**nvidia-smi not found:**
```bash
./00_setup_gpu.sh --install-drivers
sudo reboot  # May be needed after first driver install
```

**CUDA toolkit not found:**
```bash
./00_setup_gpu.sh --install-drivers  # Installs both driver + CUDA
# Or manually: sudo apt install cuda-toolkit-12-8
```

**HuggingFace download fails (gated model):**
```bash
# 1. Get token: https://huggingface.co/settings/tokens
# 2. Accept model license on the model's HF page
export HF_TOKEN=hf_xxx
./01_setup_model.sh --preset llama3-8b
```

**Proof verification fails on-chain:**
```bash
# Verify locally first
prove-model --verify-proof ~/.obelysk/proofs/latest/ml_proof.json
# Check TX status
sncast tx-status 0xTX_HASH
```

**Insufficient disk space:**
Pipeline needs: model_size x 1.5 (download) + ~10GB (build) + proof output.
For Qwen3-14B: ~50GB minimum free.

## Legacy Scripts

The following scripts in `scripts/` are deprecated in favor of the pipeline:
- `scripts/h200_setup.sh` -> `pipeline/run_e2e.sh`
- `scripts/gpu-testing/brev_setup.sh` -> `pipeline/run_e2e.sh`

The original `scripts/full_stark_verify.py` remains for multi-step STARK verification flows.
