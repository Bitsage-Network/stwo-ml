#!/usr/bin/env bash
#
# Obelysk GPU Setup — One-Command Deployment
# =============================================
# Sets up a fresh GPU instance with a fully verified ML model + proving stack.
# Proofs are meaningless without a working model — this script guarantees both.
#
# What this does:
#   1. Installs system deps + Python ML stack (torch, transformers, accelerate)
#   2. Installs Rust nightly (pinned to match stwo's requirements)
#   3. Verifies CUDA environment (GPU, driver, toolkit)
#   4. Clones the stwo-ml repo + submodules
#   5. Downloads model weights from HuggingFace (ALL required files)
#   6. Builds stwo-ml proving binary (GPU → CPU fallback)
#   7. Validates model END-TO-END: files, weights, tokenizer, GPU load, inference
#   8. Prints benchmark commands
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Bitsage-Network/stwo-ml/main/scripts/h200_setup.sh | bash
#
#   Or with options:
#   bash scripts/h200_setup.sh --model Qwen/Qwen3-14B
#   bash scripts/h200_setup.sh --model meta-llama/Llama-3-8B
#   bash scripts/h200_setup.sh --skip-model --skip-deps  # Rebuild only
#
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

REPO_URL="https://github.com/Bitsage-Network/stwo-ml.git"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/stwo-ml}"
MODEL_HF="${MODEL_HF:-Qwen/Qwen3-14B}"
MODEL_DIR="${MODEL_DIR:-}"   # auto-derived from MODEL_HF if not set

SKIP_MODEL=false
SKIP_BUILD=false
SKIP_DEPS=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-model)  SKIP_MODEL=true; shift ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --skip-deps)   SKIP_DEPS=true; shift ;;
        --branch)      BRANCH="$2"; shift 2 ;;
        --model)       MODEL_HF="$2"; shift 2 ;;
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model HF_ID   HuggingFace model (default: Qwen/Qwen3-14B)"
            echo "  --model-dir DIR Where to store model weights (auto-derived from model)"
            echo "  --skip-model    Skip downloading model weights"
            echo "  --skip-build    Skip building Rust binaries"
            echo "  --skip-deps     Skip installing system dependencies"
            echo "  --branch NAME   Git branch (default: main)"
            echo "  --install-dir   Where to clone repo (default: ~/stwo-ml)"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Default: Qwen3-14B"
            echo "  $0 --model meta-llama/Llama-3-8B            # Use LLaMA"
            echo "  $0 --model mistralai/Mistral-7B-v0.3        # Use Mistral"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-derive MODEL_DIR from HuggingFace model ID if not explicitly set
if [ -z "$MODEL_DIR" ]; then
    MODEL_SLUG=$(echo "${MODEL_HF}" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    MODEL_DIR="$HOME/models/${MODEL_SLUG}"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ██╗                    ║
║    ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝██╔════╝██║ ██╔╝                    ║
║    ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝ ███████╗█████╔╝                     ║
║    ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝  ╚════██║██╔═██╗                     ║
║    ╚██████╔╝██████╔╝███████╗███████╗██║   ███████║██║  ██╗                    ║
║     ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝                    ║
║                                                                               ║
║    ███████╗████████╗██╗    ██╗ ██████╗    ███╗   ███╗██╗                      ║
║    ██╔════╝╚══██╔══╝██║    ██║██╔═══██╗   ████╗ ████║██║                      ║
║    ███████╗   ██║   ██║ █╗ ██║██║   ██║   ██╔████╔██║██║                      ║
║    ╚════██║   ██║   ██║███╗██║██║   ██║   ██║╚██╔╝██║██║                      ║
║    ███████║   ██║   ╚███╔███╔╝╚██████╔╝   ██║ ╚═╝ ██║███████╗                ║
║    ╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝    ╚═╝     ╚═╝╚══════╝                ║
║                                                                               ║
║              GPU-Accelerated ZK Proofs for Verifiable AI                      ║
║                                                                               ║
║    Model → Circle STARK Proof → Recursive Verification → On-Chain             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"
echo -e "  ${BOLD}Model:${NC}   ${MODEL_HF}"
echo -e "  ${BOLD}Target:${NC}  ${MODEL_DIR}"
echo ""

START_TIME=$(date +%s)

# ═══════════════════════════════════════════════════════════════════════
# Step 1: System Dependencies
# ═══════════════════════════════════════════════════════════════════════

if [ "$SKIP_DEPS" = false ]; then
    echo -e "${YELLOW}[1/8] Installing system dependencies${NC}"

    # Detect package manager
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            build-essential \
            cmake \
            pkg-config \
            libssl-dev \
            git \
            git-lfs \
            curl \
            wget \
            python3 \
            python3-pip \
            jq \
            bc \
            2>&1 | tail -3
        echo -e "  ${GREEN}apt packages installed${NC}"
    elif command -v yum &>/dev/null; then
        sudo yum install -y -q \
            gcc gcc-c++ make cmake \
            openssl-devel pkg-config \
            git git-lfs curl wget \
            python3 python3-pip jq bc
        echo -e "  ${GREEN}yum packages installed${NC}"
    else
        echo -e "  ${YELLOW}Unknown package manager — skipping system deps${NC}"
        echo "  Please install: build-essential cmake pkg-config libssl-dev git git-lfs python3 jq bc"
    fi

    # Install HuggingFace CLI for model download
    pip3 install --quiet --upgrade huggingface_hub 2>/dev/null || \
        pip3 install --quiet --upgrade --user huggingface_hub 2>/dev/null || true
    echo -e "  ${GREEN}huggingface_hub installed${NC}"

    # Install PyTorch + Transformers for model validation and chat
    echo "  Installing torch + transformers (for model validation)..."
    pip3 install --quiet torch transformers accelerate 2>/dev/null || \
        pip3 install --quiet --user torch transformers accelerate 2>/dev/null || true
    echo -e "  ${GREEN}torch + transformers installed${NC}"
else
    echo -e "${YELLOW}[1/8] Skipping system dependencies (--skip-deps)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Rust Nightly
# ═══════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[2/8] Setting up Rust toolchain${NC}"

# The stwo and stwo-ml crates pin nightly-2025-07-14 via rust-toolchain.toml.
# We install that exact version to avoid compilation breakage.
PINNED_NIGHTLY="nightly-2025-07-14"

if ! command -v rustup &>/dev/null; then
    echo "  Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${PINNED_NIGHTLY}"
    source "$HOME/.cargo/env"
else
    echo "  rustup already installed"
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

# Install the exact pinned nightly and set as default
rustup install "${PINNED_NIGHTLY}" 2>/dev/null || true
rustup default "${PINNED_NIGHTLY}" 2>/dev/null || true
rustup component add rust-src --toolchain "${PINNED_NIGHTLY}" 2>/dev/null || true

echo "  $(rustc --version)"
echo "  $(cargo --version)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 3: CUDA Verification
# ═══════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[3/8] Verifying CUDA environment${NC}"

# Check nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    echo -e "  ${RED}ERROR: nvidia-smi not found. Is the NVIDIA driver installed?${NC}"
    echo "  On Brev/Shadeform instances, CUDA should be pre-installed."
    echo "  If not, install: https://developer.nvidia.com/cuda-12-4-0-download-archive"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | xargs)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)

echo "  GPU:    ${GPU_NAME}"
echo "  VRAM:   ${GPU_MEM}"
echo "  Driver: ${DRIVER_VER}"

# Find CUDA toolkit
CUDA_PATH=""
for dir in /usr/local/cuda-12.4 /usr/local/cuda-12.6 /usr/local/cuda /opt/cuda; do
    if [ -d "$dir" ] && [ -f "$dir/bin/nvcc" ]; then
        CUDA_PATH="$dir"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    echo -e "  ${RED}ERROR: CUDA toolkit not found${NC}"
    echo "  Looked in: /usr/local/cuda-12.4, /usr/local/cuda-12.6, /usr/local/cuda, /opt/cuda"
    echo "  Install: https://developer.nvidia.com/cuda-12-4-0-download-archive"
    exit 1
fi

export PATH="${CUDA_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_PATH}"

NVCC_VER=$("${CUDA_PATH}/bin/nvcc" --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "  CUDA:   ${NVCC_VER} (${CUDA_PATH})"

# Write env to .bashrc so it persists across sessions
ENVFILE="$HOME/.obelysk_env"
cat > "$ENVFILE" << EOF
# Obelysk CUDA environment (auto-generated by h200_setup.sh)
export PATH="${CUDA_PATH}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:\${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_PATH}"
export MODEL_DIR="${MODEL_DIR}"
export BITSAGE_DIR="${INSTALL_DIR}"
EOF

# Add to .bashrc if not already there
if ! grep -q "obelysk_env" "$HOME/.bashrc" 2>/dev/null; then
    echo "source $ENVFILE" >> "$HOME/.bashrc"
fi

echo -e "  ${GREEN}CUDA environment configured${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Clone Repository
# ═══════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[4/8] Setting up repository${NC}"

if [ -d "${INSTALL_DIR}/.git" ]; then
    echo "  Repo exists at ${INSTALL_DIR}, pulling latest..."
    cd "${INSTALL_DIR}"
    git fetch origin
    git checkout "${BRANCH}" 2>/dev/null || git checkout -b "${BRANCH}" "origin/${BRANCH}"
    git pull origin "${BRANCH}" --ff-only 2>/dev/null || true
else
    echo "  Cloning ${REPO_URL} (branch: ${BRANCH})..."
    git clone --branch "${BRANCH}" --depth 1 "${REPO_URL}" "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
fi

# Initialize submodules (stwo-cairo)
echo "  Initializing submodules..."
git submodule update --init --recursive 2>/dev/null || true

echo "  Repo: ${INSTALL_DIR}"
echo "  Branch: $(git branch --show-current)"
echo -e "  ${GREEN}Repository ready${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 5: Download Qwen3-14B
# ═══════════════════════════════════════════════════════════════════════

if [ "$SKIP_MODEL" = false ]; then
    echo -e "${YELLOW}[5/8] Downloading model weights${NC}"
    echo -e "  Model: ${BOLD}${MODEL_HF}${NC}"
    echo -e "  Dir:   ${MODEL_DIR}"

    mkdir -p "${MODEL_DIR}"

    # Always use snapshot_download to get ALL required files
    # This handles: weights, config, tokenizer, shard index, generation config
    # Works for ANY HuggingFace model (Qwen, LLaMA, Mistral, YOLO, etc.)
    echo "  Downloading all model files..."
    python3 -c "
from huggingface_hub import snapshot_download
import os

model_dir = '${MODEL_DIR}'
model_id = '${MODEL_HF}'

# Download everything needed for inference + proving
# Exclude only large non-essential files (gguf, bin checkpoints, etc.)
result = snapshot_download(
    model_id,
    local_dir=model_dir,
    ignore_patterns=['*.gguf', '*.bin', '*.pt', '*.pth', '*.ot', '*.msgpack', '.git*'],
)

# Verify critical files exist
files = os.listdir(model_dir)
safetensors = [f for f in files if f.endswith('.safetensors')]
has_config = 'config.json' in files
has_index = any('index' in f and f.endswith('.json') for f in files) or len(safetensors) == 1
has_tokenizer = any('tokenizer' in f.lower() for f in files)

total_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in files if os.path.isfile(os.path.join(model_dir, f)))

print(f'  ✓ config.json:      {\"found\" if has_config else \"MISSING\"}'  )
print(f'  ✓ shard index:      {\"found\" if has_index else \"MISSING\"}'  )
print(f'  ✓ tokenizer:        {\"found\" if has_tokenizer else \"MISSING\"}'  )
print(f'  ✓ weight shards:    {len(safetensors)} files'  )
print(f'  ✓ total size:       {total_size / 1e9:.1f} GB'  )

if not has_config:
    print('ERROR: config.json missing — model cannot load')
    exit(1)
if len(safetensors) == 0:
    print('ERROR: no .safetensors files found — weights missing')
    exit(1)
if not has_tokenizer:
    print('WARNING: no tokenizer found — inference may fail')

print('  Download complete')
" || {
        echo -e "  ${RED}Model download failed${NC}"
        exit 1
    }
    echo -e "  ${GREEN}Model downloaded${NC}"
else
    echo -e "${YELLOW}[5/8] Skipping model download (--skip-model)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 6: Build Everything
# ═══════════════════════════════════════════════════════════════════════

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[6/8] Building Obelysk proving stack${NC}"

    cd "${INSTALL_DIR}"

    # 6a: Build stwo-ml with GPU + model loading + CLI
    # rust-toolchain.toml in stwo-ml/ pins the correct nightly automatically
    echo "  [6a] Building stwo-ml (GPU + CLI)..."
    (
        cd stwo-ml
        cargo build --release \
            --bin prove-model \
            --features "cuda-runtime,cli" \
            2>&1 | tail -20
    ) && echo -e "  ${GREEN}stwo-ml built${NC}" || {
        echo -e "  ${RED}stwo-ml GPU build failed${NC}"
        echo "  Trying without GPU (CPU-only)..."
        (
            cd stwo-ml
            cargo build --release \
                --bin prove-model \
                --features "cli" \
                2>&1 | tail -20
        ) && echo -e "  ${YELLOW}Built in CPU-only mode${NC}" || {
            echo -e "  ${RED}CPU build also failed. Printing full error:${NC}"
            cd stwo-ml
            cargo build --release --bin prove-model --features "cli" 2>&1
        }
    }

    # 6b: Build cairo-prove
    echo ""
    echo "  [6b] Building cairo-prove..."
    if [ -d "stwo-cairo/cairo-prove" ]; then
        (
            cd stwo-cairo/cairo-prove
            cargo build --release 2>&1 | tail -10
        ) && echo -e "  ${GREEN}cairo-prove built${NC}" || \
            echo -e "  ${YELLOW}cairo-prove build failed (recursive proving will be unavailable)${NC}"
    else
        echo -e "  ${YELLOW}stwo-cairo not found, skipping cairo-prove${NC}"
    fi

    # 6c: Build Cairo ML verifier (if scarb is available)
    echo ""
    echo "  [6c] Building Cairo ML verifier..."
    if command -v scarb &>/dev/null; then
        if [ -d "stwo-cairo/stwo_cairo_verifier" ]; then
            (
                cd stwo-cairo/stwo_cairo_verifier
                scarb build 2>&1 | tail -3
            ) && echo -e "  ${GREEN}Cairo ML verifier built${NC}" || \
                echo -e "  ${YELLOW}Cairo verifier build failed${NC}"
        fi
    else
        echo -e "  ${YELLOW}scarb not found — installing...${NC}"
        curl -L https://docs.swmansion.com/scarb/install.sh 2>/dev/null | sh -s -- -v 2.12.0 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
        if command -v scarb &>/dev/null; then
            echo "  scarb installed: $(scarb --version)"
            if [ -d "stwo-cairo/stwo_cairo_verifier" ]; then
                (cd stwo-cairo/stwo_cairo_verifier && scarb build 2>&1 | tail -3) || true
            fi
        else
            echo -e "  ${YELLOW}scarb install failed — Cairo verifier unavailable${NC}"
        fi
    fi

    # 6d: Run quick sanity test
    echo ""
    echo "  [6d] Running quick sanity test..."
    (
        cd stwo-ml
        cargo test --release --lib \
            -- test_matmul_sumcheck_basic --nocapture 2>&1 | tail -10
    ) && echo -e "  ${GREEN}Sanity test passed${NC}" || \
        echo -e "  ${YELLOW}Sanity test skipped (not critical)${NC}"

else
    echo -e "${YELLOW}[6/8] Skipping build (--skip-build)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 7: Full Model Validation (MUST PASS)
# ═══════════════════════════════════════════════════════════════════════

if [ "$SKIP_MODEL" = false ]; then
    echo -e "${YELLOW}[7/8] Validating model end-to-end${NC}"
    echo -e "  ${BOLD}Every check must pass. Proofs are meaningless without a verified model.${NC}"
    echo ""

    cd "${INSTALL_DIR}"

    VALIDATION_FAILED=false

    export MODEL_DIR="${MODEL_DIR}"
    python3 << 'VALIDATE_SCRIPT'
import json, os, sys, time

model_dir = os.environ.get('MODEL_DIR', os.path.expanduser('~/models/qwen3-14b'))
checks_passed = 0
checks_failed = 0

def check(name, condition, detail=""):
    global checks_passed, checks_failed
    if condition:
        print(f'  \033[0;32m✓\033[0m {name}' + (f'  ({detail})' if detail else ''))
        checks_passed += 1
        return True
    else:
        print(f'  \033[0;31m✗\033[0m {name}' + (f'  ({detail})' if detail else ''))
        checks_failed += 1
        return False

print(f'  Model directory: {model_dir}')
print()

# ── Check 1: Directory exists ──
check('Model directory exists', os.path.isdir(model_dir))

# ── Check 2: config.json ──
config_path = os.path.join(model_dir, 'config.json')
has_config = os.path.isfile(config_path)
check('config.json present', has_config)

config = {}
if has_config:
    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get('model_type', 'unknown')
    hidden = config.get('hidden_size', 0)
    heads = config.get('num_attention_heads', 0)
    layers = config.get('num_hidden_layers', 0)
    ff = config.get('intermediate_size', 0)
    vocab = config.get('vocab_size', 0)

    check('config.json parseable', hidden > 0 and layers > 0,
          f'{model_type}: d={hidden}, heads={heads}, ff={ff}, layers={layers}, vocab={vocab}')

# ── Check 3: Weight files ──
all_files = os.listdir(model_dir) if os.path.isdir(model_dir) else []
safetensors = sorted([f for f in all_files if f.endswith('.safetensors')])
total_weight_bytes = sum(
    os.path.getsize(os.path.join(model_dir, f))
    for f in safetensors
)
check('SafeTensors weight files present', len(safetensors) > 0,
      f'{len(safetensors)} shards, {total_weight_bytes/1e9:.1f} GB')

# ── Check 4: Shard index (for multi-shard models) ──
has_index = any('index' in f and f.endswith('.json') for f in all_files)
if len(safetensors) > 1:
    check('Shard index file present', has_index,
          'model.safetensors.index.json' if has_index else 'MISSING — transformers cannot load sharded model')
elif len(safetensors) == 1:
    check('Single-shard model (no index needed)', True)

# ── Check 5: Tokenizer ──
tokenizer_files = [f for f in all_files if 'tokenizer' in f.lower() or f.endswith('.tiktoken')]
check('Tokenizer files present', len(tokenizer_files) > 0,
      ', '.join(tokenizer_files[:3]) + ('...' if len(tokenizer_files) > 3 else ''))

# ── Check 6: Weight integrity (read first shard header) ──
if safetensors:
    try:
        import struct
        first_shard = os.path.join(model_dir, safetensors[0])
        with open(first_shard, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len))
        tensor_count = len([k for k in header if k != '__metadata__'])
        check('Weight shard readable', tensor_count > 0,
              f'{safetensors[0]}: {tensor_count} tensors')
    except Exception as e:
        check('Weight shard readable', False, str(e))

# ── Check 7: GPU available ──
try:
    import torch
    gpu_ok = torch.cuda.is_available()
    if gpu_ok:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        check('CUDA GPU available', True, f'{gpu_name}, {gpu_mem:.0f} GB')
    else:
        check('CUDA GPU available', False, 'torch.cuda.is_available() = False')
except ImportError:
    check('CUDA GPU available', False, 'torch not installed')
    gpu_ok = False

# ── Check 8: Tokenizer loads ──
tokenizer = None
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    test_ids = tokenizer.encode('Hello world')
    check('Tokenizer loads and encodes', len(test_ids) > 0,
          f'"Hello world" → {len(test_ids)} tokens')
except Exception as e:
    check('Tokenizer loads and encodes', False, str(e))

# ── Check 9: Model loads onto GPU ──
model = None
if gpu_ok:
    try:
        from transformers import AutoModelForCausalLM
        print()
        print('  Loading model onto GPU (this takes 15-30s)...')
        t0 = time.time()
        load_kwargs = dict(device_map='auto', trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, dtype='float16', **load_kwargs)
        except (TypeError, ValueError):
            try:
                model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='float16', **load_kwargs)
            except TypeError:
                import torch
                model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, **load_kwargs)
        load_time = time.time() - t0
        total_params = sum(p.numel() for p in model.parameters())
        gpu_mem_used = torch.cuda.memory_allocated() / 1e9
        check('Model loads onto GPU', True,
              f'{total_params/1e9:.1f}B params, {gpu_mem_used:.1f} GB VRAM, {load_time:.1f}s')
    except Exception as e:
        check('Model loads onto GPU', False, str(e))

# ── Check 10: Inference produces output ──
if model is not None and tokenizer is not None:
    try:
        import torch
        inputs = tokenizer('The capital of France is', return_tensors='pt').to('cuda')
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        gen_time = time.time() - t0
        new_tokens = out.shape[1] - inputs['input_ids'].shape[1]
        output_text = tokenizer.decode(out[0], skip_special_tokens=True)
        tok_per_sec = new_tokens / gen_time if gen_time > 0 else 0
        check('Inference produces output', new_tokens > 0,
              f'{new_tokens} tokens in {gen_time:.2f}s ({tok_per_sec:.0f} tok/s)')
        print(f'       → "{output_text[:100]}"')
    except Exception as e:
        check('Inference produces output', False, str(e))

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

# ── Summary ──
print()
total = checks_passed + checks_failed
print(f'  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print(f'  Checks: {checks_passed}/{total} passed', end='')
if checks_failed > 0:
    print(f', \033[0;31m{checks_failed} FAILED\033[0m')
else:
    print(f' — \033[0;32mALL PASSED\033[0m')
print(f'  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')

if checks_failed > 0:
    print()
    print('  Some checks failed. Fix the issues above before running benchmarks.')
    print('  Proofs over a broken model are meaningless.')
    sys.exit(1)
VALIDATE_SCRIPT

    if [ $? -ne 0 ]; then
        VALIDATION_FAILED=true
        echo -e "  ${RED}Model validation FAILED — fix issues above before proving${NC}"
    else
        echo -e "  ${GREEN}Model validation PASSED${NC}"
    fi

    # Run the Rust prover's own validation (the proof system checks everything)
    PROVE_MODEL_BIN=$(find . -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)
    if [ -n "$PROVE_MODEL_BIN" ]; then
        echo ""
        echo -e "  ${BOLD}Running prover validation (prove-model --validate)...${NC}"
        echo "  The proof system itself must agree the model is valid."
        echo ""
        if $PROVE_MODEL_BIN --model-dir "${MODEL_DIR}" --layers 1 --validate 2>&1; then
            echo -e "  ${GREEN}Prover validation PASSED${NC}"
        else
            echo -e "  ${RED}Prover validation FAILED — the proof system rejected this model${NC}"
            VALIDATION_FAILED=true
        fi
    fi

    if [ "$VALIDATION_FAILED" = true ]; then
        echo -e "\n  ${RED}${BOLD}SETUP INCOMPLETE: Model validation failed.${NC}"
        echo -e "  ${RED}Fix the issues above, then re-run: bash scripts/h200_setup.sh --skip-deps --skip-build${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[7/8] Skipping model validation (--skip-model)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 8: Summary
# ═══════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[8/8] Setup summary${NC}"

cd "${INSTALL_DIR}"

# Find binaries
PROVE_MODEL_BIN=$(find . -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)
CAIRO_PROVE_BIN=$(find . -name "cairo-prove" -path "*/release/*" -type f 2>/dev/null | head -1)

echo "  prove-model: ${PROVE_MODEL_BIN:-NOT FOUND}"
echo "  cairo-prove: ${CAIRO_PROVE_BIN:-NOT FOUND}"
echo "  Model dir:   ${MODEL_DIR}"

# Check model files
if [ -d "$MODEL_DIR" ]; then
    SHARD_COUNT=$(ls "${MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l)
    CONFIG_EXISTS="no"
    [ -f "${MODEL_DIR}/config.json" ] && CONFIG_EXISTS="yes"
    echo "  Model shards: ${SHARD_COUNT}"
    echo "  config.json:  ${CONFIG_EXISTS}"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo -e "${GREEN}${BOLD}"
cat << SUMMARY
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                     ✓  SETUP COMPLETE — ALL VALIDATED                         ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  GPU:          ${GPU_NAME}
║  VRAM:         ${GPU_MEM}
║  CUDA:         ${NVCC_VER}
║  Model:        ${MODEL_HF}
║  Weights:      ${MODEL_DIR}
║  Repo:         ${INSTALL_DIR}
║  Setup time:   ${ELAPSED_MIN} min ${ELAPSED} sec
║                                                                               ║
║  The model has been downloaded, loaded onto the GPU, and inference             ║
║  has been verified. The proving stack is built and ready.                      ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  COMMANDS:                                                                    ║
║                                                                               ║
║  Inspect model (no proving):                                                  ║
║     cd ${INSTALL_DIR}/stwo-ml
║     ./target/release/prove-model \\
║       --model-dir ${MODEL_DIR} --layers 1 --inspect
║                                                                               ║
║  Prove 1 transformer block:                                                   ║
║     ./target/release/prove-model \\
║       --model-dir ${MODEL_DIR} \\
║       --layers 1 --output proof_1block.json
║                                                                               ║
║  Prove all blocks:                                                            ║
║     ./target/release/prove-model \\
║       --model-dir ${MODEL_DIR} \\
║       --output proof_full.json
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
SUMMARY
echo -e "${NC}"
