# Obelysk Pipeline

End-to-end pipeline for proving ML model inference and verifying on-chain. Works on any NVIDIA GPU from RTX 4090 to B300.

## Prerequisites

Before running the pipeline, ensure the following are available on your machine:

| Requirement | Notes |
|-------------|-------|
| **NVIDIA GPU** | RTX 3090+ recommended; see GPU Compatibility table below |
| **NVIDIA Driver + CUDA** | Auto-installed by `00_setup_gpu.sh --install-drivers`, or install manually |
| **Node.js 18+** | Auto-installed via nvm if missing (needed for paymaster submission) |
| **Python 3.8+** | Used for HuggingFace downloads, config parsing, verification receipts |
| **git-lfs** | Installed by `00_setup_gpu.sh`; needed for large model downloads |
| **Rust nightly-2025-07-14** | Pinned toolchain matching STWO; installed by `00_setup_gpu.sh` |
| **`OBELYSK_SECRETS_KEY`** | Single passphrase to unlock all pipeline secrets (see below) |

### Zero-Config Storage (Marketplace)

Audit reports and proofs are stored automatically through the **BitSage Marketplace**:

1. **`00_setup_gpu.sh`** auto-registers your machine with the marketplace
2. An org + API key are provisioned for your GPU device
3. Credentials are cached in `~/.obelysk/marketplace.env`
4. At audit time, reports are encrypted and uploaded to Arweave via the marketplace
5. **View your proofs** at the marketplace dashboard: `/storage`

No tokens, wallets, or API keys needed from the user.

**Storage priority chain** (first available wins):
1. `IRYS_TOKEN` — Direct Irys upload (for advanced users with their own Arweave wallet)
2. `MARKETPLACE_API_KEY` — Auto-provisioned marketplace (default)
3. Relay fallback — Coordinator EC2 proxy

### Secrets & Tokens

API tokens are managed automatically:

- **`IRYS_TOKEN`** — **Not needed.** Marketplace or relay handles Arweave uploads.
- **`STARKNET_PRIVATE_KEY`** — **Not needed on Sepolia.** The AVNU paymaster handles gas-free submission.
- **`HF_TOKEN`** — **Not needed for default models** (Qwen, Phi, Mistral). Only required for gated models (Llama, Gemma) where HuggingFace requires license acceptance.

For teams that want to self-host or override: tokens can be shipped encrypted in `configs/.secrets.env.enc`. The pipeline auto-decrypts at startup with a single passphrase.

```bash
# Option 1: Set env var (non-interactive, CI-friendly)
export OBELYSK_SECRETS_KEY="your-passphrase"
./run_e2e.sh --preset qwen3-14b --gpu --submit

# Option 2: Override individual tokens (always takes priority)
HF_TOKEN=hf_xxx IRYS_TOKEN=irys_xxx ./run_e2e.sh --preset qwen3-14b --gpu --submit

# Option 3: Custom marketplace URL
MARKETPLACE_URL=https://my-instance.example.com ./run_e2e.sh --preset qwen3-14b --gpu --submit
```

**For pipeline administrators** — create or update the encrypted secrets file:

```bash
# Interactive: prompts for each token + passphrase
./manage_secrets.sh --encrypt

# From existing .env file
./manage_secrets.sh --encrypt --from .env

# Rotate passphrase
./manage_secrets.sh --rotate

# View current secrets (debugging)
./manage_secrets.sh --decrypt
```

The encrypted file (`configs/.secrets.env.enc`) is safe to commit. The decrypted cache (`~/.obelysk/secrets.env`) is gitignored and restricted to owner-only permissions.

## Quick Start

**Fresh GPU machine (one-liner bootstrap):**

```bash
# This clones the repo, installs all deps, builds binaries, and runs the full pipeline
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/stwo-ml/main/scripts/pipeline/bootstrap.sh | bash -s -- --preset qwen3-14b --gpu --submit

# Or just set up the environment (no prove/verify):
curl -fsSL https://raw.githubusercontent.com/Bitsage-Network/stwo-ml/main/scripts/pipeline/bootstrap.sh | bash
```

The bootstrap script clones the repo to `~/obelysk/`, installs dependencies, detects your GPU, builds the proving stack, and optionally runs the full pipeline. During setup you'll be prompted for your email to link proofs to your [marketplace dashboard](https://marketplace.bitsage.network).

**Already have the repo cloned:**

```bash
cd scripts/pipeline

# Full pipeline: setup -> download -> validate -> capture -> prove -> verify -> audit
./run_e2e.sh --preset qwen3-14b --gpu --submit

# Dry run (no on-chain submission)
./run_e2e.sh --preset phi3-mini --gpu --dry-run

# Enforce GPU-only proving (fail if critical proving paths fallback to CPU)
./run_e2e.sh --preset qwen3-14b --gpu --gpu-only --dry-run

# With HuggingFace auth (for gated models like Llama)
HF_TOKEN=hf_xxx ./run_e2e.sh --preset llama3-8b --gpu --submit

# With on-chain submission
STARKNET_PRIVATE_KEY=0x... ./run_e2e.sh --preset qwen3-14b --gpu --submit
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 0 | `00_setup_gpu.sh` | Install deps, drivers, Rust, detect GPU/CUDA, build binaries, build llama.cpp, register with marketplace |
| 1 | `01_setup_model.sh` | Download model from HuggingFace (with auth), verify integrity |
| 2 | `02_validate_model.sh` | Validate model files, dimensions, weights |
| 2a | `02a_test_inference.sh` | Test inference via llama.cpp (single prompt, chat, benchmark) |
| 2b | `02b_capture_inference.sh` | Capture inference log via prover forward pass (required for audit) |
| 3 | `03_prove.sh` | Generate cryptographic proof with local verification |
| 4 | `04_verify_onchain.sh` | Submit proof to Starknet with TX confirmation + acceptance and assurance classification (`accepted_onchain`, `full_gkr_verified`) |
| 5 | `05_audit.sh` | Run verifiable inference audit with marketplace storage (encrypt → Arweave → index) |
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
| `MARKETPLACE_URL` | No | `https://marketplace.bitsage.network` | `05_audit.sh`, `lib/common.sh` |
| `MARKETPLACE_API_KEY` | No | Auto-provisioned | `05_audit.sh` |
| `STWO_GPU_COMMIT_STRICT` | No | Off | `03_prove.sh`, `prove-model` |
| `STWO_GPU_COMMIT_HARDEN` | No | Off | `03_prove.sh`, `prove-model` |
| `STWO_GPU_POLY_STRICT` | No | Off | STWO GPU poly backend |
| `STWO_GPU_POLY_HARDEN` | No | Off | STWO GPU poly backend |
| `STWO_UNIFIED_STARK_NO_FALLBACK` | No | Off | `prove-model` unified STARK (disable GPU→SIMD retry on `ConstraintsNotSatisfied`) |
| `STWO_PURE_GKR_SKIP_UNIFIED_STARK` | No | `on` in `03_prove.sh` | Skip Phase 3 unified STARK in pure `ml_gkr` runs when GKR already covers non-matmul ops |
| `STWO_PARALLEL_GPU_COMMIT` | No | Off (single GPU default) | `03_prove.sh`, `prove-model` |
| `STWO_WEIGHT_PROGRESS_EVERY` | No | `1` | Weight commitment progress cadence |
| `STWO_GKR_OPENINGS_PROGRESS_EVERY` | No | `1` | Weight-opening progress cadence |
| `STWO_GKR_OPENING_HEARTBEAT_SEC` | No | `15` | Per-opening heartbeat seconds |
| `STWO_GPU_MLE_FOLD` | No | `1` (pipeline default) | GPU fold for MLE openings |
| `STWO_GPU_MLE_FOLD_MIN_POINTS` | No | `1048576` (pipeline default) | Min MLE size to start GPU fold |
| `STWO_GPU_MLE_MERKLE_REQUIRE` | No | Off | Fail if MLE Merkle falls back to CPU |
| `STWO_GPU_MLE_FOLD_REQUIRE` | No | Off | Fail if MLE fold falls back to CPU |
| `STWO_GPU_MLE_OPENING_TREE_REQUIRE` | No | Off | Fail if GPU-resident opening-tree path fails |
| `STWO_GPU_MLE_OPENING_TIMING` | No | Off | Print per-opening tree/query timing breakdown |
| `STWO_GKR_AGGREGATE_WEIGHT_BINDING` | No | `on` in `03_prove.sh` fast mode, auto-`off` for `run_e2e.sh --submit` | Batched RLC weight-binding mode (serializable artifact, not submit-ready for Starknet `verify_model_gkr`) |

Notes:
- The opening path now packs QM31 leaves to felt252 on GPU (no per-round CPU repack/upload), which reduces weight-opening overhead on large models.
- Query extraction now replays folds on GPU and downloads only queried leaf pairs (instead of full folded layers), reducing opening-phase host transfer pressure.
- `03_prove.sh` defaults to aggregated RLC weight binding for faster off-chain proving.
- `run_e2e.sh --submit` auto-adds `--starknet-ready`, which forces sequential openings.
- Unified STARK now retries once on SIMD if GPU path hits `ConstraintsNotSatisfied` (soundness-preserving fallback). Set `--gpu-only` or `STWO_UNIFIED_STARK_NO_FALLBACK=1` to fail closed instead.
- `03_prove.sh` defaults `STWO_PURE_GKR_SKIP_UNIFIED_STARK=1` for `ml_gkr`, which bypasses Phase 3 when GKR already covers activation/add/mul/layernorm/rmsnorm/dequantize.
- In aggregated weight-binding mode, `ml_gkr` output still serializes full proof artifacts with `submission_ready=false`, `weight_opening_mode`, and `weight_claim_calldata`.

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
| `gkr` | prove-model (GKR sumcheck) | `verify_model_gkr()` | Hardened production mode. Full on-chain GKR assurance. |

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

### 02b_capture_inference.sh

```bash
./02b_capture_inference.sh                                         # Uses model from pipeline state
./02b_capture_inference.sh --model-name phi3-mini --count 5        # 5 captures
./02b_capture_inference.sh --model-dir ~/models/qwen3-14b --layers 1
./02b_capture_inference.sh --skip-commitment                       # Faster, weaker audit
```

### 03_prove.sh

```bash
./03_prove.sh --model-name qwen3-14b --layers 1 --mode gkr --gpu
./03_prove.sh --model-name qwen3-14b --mode gkr --multi-gpu
./03_prove.sh --model-name qwen3-14b --mode gkr --gpu --gpu-only
./03_prove.sh --model-name qwen3-14b --server http://prover:8080
```

### 04_verify_onchain.sh

```bash
./04_verify_onchain.sh --dry-run                        # Print commands
STARKNET_PRIVATE_KEY=0x... ./04_verify_onchain.sh --submit
./04_verify_onchain.sh --submit --max-fee 0.1           # Custom fee
./04_verify_onchain.sh --submit --contract 0x123...     # Custom contract
```

Notes:
- If `verify_calldata.entrypoint` is `unsupported` (e.g. aggregated RLC
  weight-binding mode), the script prints `weight_opening_mode` + gate reason.
- In `--dry-run`, unsupported artifacts are reported and skipped cleanly.
- In `--submit`, unsupported artifacts fail fast before submitting any tx.

### 05_audit.sh

```bash
./05_audit.sh --evaluate                                # Audit with semantic evaluation
./05_audit.sh --evaluate --submit                       # Audit + on-chain submission
./05_audit.sh --evaluate --submit --privacy private     # Encrypted audit
./05_audit.sh --log-dir /path/to/logs --dry-run         # Custom log dir, dry run
./05_audit.sh --prove-evals --submit                    # Prove evaluation forward passes
```

### run_e2e.sh

```bash
./run_e2e.sh --preset qwen3-14b --gpu --submit
./run_e2e.sh --preset phi3-mini --gpu --dry-run
./run_e2e.sh --preset qwen3-14b --gpu --gpu-only --dry-run
./run_e2e.sh --preset qwen3-14b --resume-from prove --submit
./run_e2e.sh --preset llama3-8b --chat --submit         # Pause for chat
./run_e2e.sh --preset qwen3-14b --resume-from capture --submit  # Resume from capture
./run_e2e.sh --preset qwen3-14b --skip-setup --skip-inference --submit
./run_e2e.sh --preset phi3-mini --gpu --dry-run --no-audit     # Skip audit
```

## Directory Structure

```
~/.obelysk/
  gpu_config.env          # Detected GPU configuration
  cuda_env.sh             # CUDA environment exports
  setup_state.env         # Setup step state
  model_state.env         # Current model state
  prove_state.env         # Last proof state
  capture_state.env       # Inference capture state (02b)
  inference_state.env     # Inference test state
  models/
    qwen3-14b/
      config.env          # Model configuration
      *.safetensors       # Model weights
      config.json         # HF config
  logs/
    qwen3-14b/
      meta.json           # Capture session metadata
      log.jsonl           # Chain-linked inference entries
      matrices.bin        # M31 matrix sidecar
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
  02b_capture_inference.sh   Inference capture via prover (required for audit)
  03_prove.sh                Proof generation + local verification
  04_verify_onchain.sh       On-chain submission + TX confirmation
  run_e2e.sh                 End-to-end runner
  README.md                  This file
```

## VM31 Privacy Pool

The pipeline includes a deployed privacy pool contract for shielded transactions.

### Contract Addresses

| Network | Contract | Address |
|---------|----------|---------|
| Sepolia | VM31Pool | `0x07cf94e27a60b94658ec908a00a9bb6dfff03358e952d9d48a8ed0be080ce1f9` |
| Sepolia | EloVerifier | `0x00c7845a80d01927826b17032a432ad9cd36ea61be17fe8cc089d9b68c57e710` |

Configured in `lib/contract_addresses.sh`. Override with `VM31_POOL_ADDRESS` env var.

### Privacy CLI Commands

```bash
# Wallet
prove-model wallet --create
prove-model wallet --info

# Deposit, withdraw, transfer
prove-model deposit --amount 1000 --asset 0
prove-model withdraw --amount 500 --asset 0
prove-model transfer --amount 300 --to 0x<pubkey> --to-viewing-key 0x<vk>

# Pool queries
prove-model pool-status
prove-model scan
```

### Deploying a New Pool

```bash
cd libs/elo-cairo-verifier
./scripts/deploy.sh \
  --contract vm31-pool \
  --relayer 0x<relayer> \
  --verifier 0x<elo_verifier> \
  0x<owner>
```

The deploy script auto-updates `lib/contract_addresses.sh` with the new address.

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

**Looks stuck after `layer reductions complete ... entering opening phase`:**
```bash
# Tail raw prover log (full, unfiltered)
LATEST=$(ls -td ~/.obelysk/proofs/* | head -1)
tail -f "$LATEST/prove_model.raw.log"
```
Weight-opening proofs can be the longest part for large models. Progress + heartbeat logs should continue during this phase.

**Insufficient disk space:**
Pipeline needs: model_size x 1.5 (download) + ~10GB (build) + proof output.
For Qwen3-14B: ~50GB minimum free.

## Legacy Scripts

The following scripts in `scripts/` are deprecated in favor of the pipeline:
- `scripts/h200_setup.sh` -> `pipeline/run_e2e.sh`
- `scripts/gpu-testing/brev_setup.sh` -> `pipeline/run_e2e.sh`

The original `scripts/full_stark_verify.py` remains for multi-step STARK verification flows.
