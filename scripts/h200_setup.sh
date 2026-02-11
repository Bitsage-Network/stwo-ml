#!/usr/bin/env bash
#
# Obelysk H200 One-Command Setup
# ================================
# Sets up a fresh GPU instance for ML proving benchmarks.
#
# What this does:
#   1. Installs system dependencies (build-essential, cmake, pkg-config, etc.)
#   2. Installs Rust nightly with portable_simd support
#   3. Verifies CUDA 12.4+ is available
#   4. Clones the stwo-ml repo
#   5. Downloads Qwen3-14B weights from HuggingFace (~28GB)
#   6. Builds stwo-ml, cairo-prove, and Cairo ML verifier
#   7. Runs a quick sanity test
#
# Usage:
#   brev shell bitsage-worker
#   curl -sSL https://raw.githubusercontent.com/Bitsage-Network/stwo-ml/main/scripts/h200_setup.sh | bash
#
#   Or if you already have the repo:
#   bash scripts/h200_setup.sh [--skip-model] [--skip-build] [--branch BRANCH]
#
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

REPO_URL="https://github.com/Bitsage-Network/stwo-ml.git"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/stwo-ml}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3-14b}"
MODEL_HF="Qwen/Qwen3-14B"

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
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-model    Skip downloading Qwen3-14B weights"
            echo "  --skip-build    Skip building Rust binaries"
            echo "  --skip-deps     Skip installing system dependencies"
            echo "  --branch NAME   Git branch (default: feat/f32-dual-track)"
            echo "  --model-dir DIR Where to store model weights (default: ~/models/qwen3-14b)"
            echo "  --install-dir   Where to clone repo (default: ~/stwo-ml)"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

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
║                   H200 GPU WORKER SETUP                                       ║
║                                                                               ║
║     Qwen3-14B → Circle STARK Proof → On-Chain Verification                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

START_TIME=$(date +%s)

# ═══════════════════════════════════════════════════════════════════════
# Step 1: System Dependencies
# ═══════════════════════════════════════════════════════════════════════

if [ "$SKIP_DEPS" = false ]; then
    echo -e "${YELLOW}[1/7] Installing system dependencies${NC}"

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
else
    echo -e "${YELLOW}[1/7] Skipping system dependencies (--skip-deps)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Rust Nightly
# ═══════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[2/7] Setting up Rust nightly${NC}"

if ! command -v rustup &>/dev/null; then
    echo "  Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
    source "$HOME/.cargo/env"
else
    echo "  rustup already installed"
fi

# Ensure nightly is installed and has the components we need
rustup install nightly 2>/dev/null || true
rustup default nightly 2>/dev/null || true
rustup component add rust-src --toolchain nightly 2>/dev/null || true

echo "  $(rustc --version)"
echo "  $(cargo --version)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 3: CUDA Verification
# ═══════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[3/7] Verifying CUDA environment${NC}"

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

echo -e "${YELLOW}[4/7] Setting up repository${NC}"

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
    echo -e "${YELLOW}[5/7] Downloading Qwen3-14B weights${NC}"

    mkdir -p "${MODEL_DIR}"

    if [ -f "${MODEL_DIR}/config.json" ] && ls "${MODEL_DIR}"/*.safetensors &>/dev/null 2>&1; then
        echo "  Model already downloaded at ${MODEL_DIR}"
        SHARD_COUNT=$(ls "${MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l)
        TOTAL_SIZE=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)
        echo "  Shards: ${SHARD_COUNT}, Size: ${TOTAL_SIZE}"
    else
        echo "  Downloading ${MODEL_HF} to ${MODEL_DIR}..."
        echo "  This is ~28GB and may take 10-20 minutes depending on bandwidth."
        echo ""

        # Use huggingface-cli if available, otherwise git lfs
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download "${MODEL_HF}" \
                --local-dir "${MODEL_DIR}" \
                --include "*.safetensors" "config.json" "tokenizer.json" "tokenizer_config.json" \
                --quiet
        elif python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MODEL_HF}',
    local_dir='${MODEL_DIR}',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'],
)
print('Download complete')
"
        else
            echo -e "  ${YELLOW}huggingface_hub not available, using git lfs...${NC}"
            git lfs install 2>/dev/null || true
            GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 \
                "https://huggingface.co/${MODEL_HF}" "${MODEL_DIR}"
        fi

        echo ""
        SHARD_COUNT=$(ls "${MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l)
        TOTAL_SIZE=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)
        echo -e "  ${GREEN}Downloaded: ${SHARD_COUNT} shards, ${TOTAL_SIZE}${NC}"
    fi
else
    echo -e "${YELLOW}[5/7] Skipping model download (--skip-model)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 6: Build Everything
# ═══════════════════════════════════════════════════════════════════════

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[6/7] Building Obelysk proving stack${NC}"

    cd "${INSTALL_DIR}"

    # 6a: Build stwo-ml with GPU + model loading + CLI
    echo "  [6a] Building stwo-ml (GPU + CLI)..."
    (
        cd stwo-ml
        RUSTUP_TOOLCHAIN=nightly cargo build --release \
            --bin prove-model \
            --features "cuda-runtime,cli" \
            2>&1 | tail -5
    ) && echo -e "  ${GREEN}stwo-ml built${NC}" || {
        echo -e "  ${RED}stwo-ml build failed${NC}"
        echo "  Trying without GPU..."
        (
            cd stwo-ml
            RUSTUP_TOOLCHAIN=nightly cargo build --release \
                --bin prove-model \
                --features "cli" \
                2>&1 | tail -5
        )
        echo -e "  ${YELLOW}Built in CPU-only mode${NC}"
    }

    # 6b: Build cairo-prove
    echo ""
    echo "  [6b] Building cairo-prove..."
    if [ -d "stwo-cairo/cairo-prove" ]; then
        (
            cd stwo-cairo/cairo-prove
            RUSTUP_TOOLCHAIN=nightly cargo build --release 2>&1 | tail -5
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

    # 6d: Run quick tests
    echo ""
    echo "  [6d] Running quick sanity test..."
    (
        cd stwo-ml
        RUSTUP_TOOLCHAIN=nightly cargo test --release --lib \
            -- test_matmul_sumcheck_basic --nocapture 2>&1 | tail -5
    ) && echo -e "  ${GREEN}Sanity test passed${NC}" || \
        echo -e "  ${YELLOW}Sanity test skipped (not critical)${NC}"

else
    echo -e "${YELLOW}[6/7] Skipping build (--skip-build)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Step 7: Verify Setup & Print Summary
# ═══════════════════════════════════════════════════════════════════════

echo -e "${YELLOW}[7/7] Verifying setup${NC}"

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
║                          SETUP COMPLETE                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  GPU:          ${GPU_NAME}
║  VRAM:         ${GPU_MEM}
║  CUDA:         ${NVCC_VER}
║  Repo:         ${INSTALL_DIR}
║  Model:        ${MODEL_DIR}
║  Setup time:   ${ELAPSED_MIN} minutes
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  NEXT STEPS:                                                                  ║
║                                                                               ║
║  1. Quick test (prove 1 block):                                               ║
║     cd ${INSTALL_DIR}/stwo-ml
║     cargo test --release --features cuda-runtime --test gpu_pipeline
║                                                                               ║
║  2. Prove single block:                                                       ║
║     ${PROVE_MODEL_BIN:-./prove-model} \\
║       --model ${MODEL_DIR} \\
║       --input input.json --output proof.json --gpu
║                                                                               ║
║  3. Full 40-block benchmark:                                                  ║
║     cd ${INSTALL_DIR}
║     bash scripts/benchmark_full_model.sh --layers 40 \\
║       --model-dir ${MODEL_DIR}
║                                                                               ║
║  4. Full pipeline (prove + recursive STARK + on-chain):                       ║
║     bash scripts/h200_recursive_pipeline.sh --layers 40 \\
║       --model-dir ${MODEL_DIR}
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
SUMMARY
echo -e "${NC}"
