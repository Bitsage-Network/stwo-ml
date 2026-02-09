#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Obelysk Protocol — One-Command Qwen3-14B Deployment
# Target: NVIDIA H200 (143GB HBM3e)
# ============================================================================

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Obelysk Protocol — H200 Deployment                ║"
echo "║  Qwen3-14B ML Inference Proving Pipeline            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo

MODEL_DIR="${HOME}/models/qwen3-14b"
REPO_DIR="${HOME}/stwo-ml"
LAYERS="${LAYERS:-1}"
SEQ_LEN="${SEQ_LEN:-1}"

# ============================================================================
# Phase 1: System Dependencies
# ============================================================================
echo "[1/6] Checking system dependencies..."

# CUDA toolkit
if ! command -v nvcc &>/dev/null; then
    echo "  Installing CUDA toolkit 12.6..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-6
    export PATH="/usr/local/cuda-12.6/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"
    echo 'export PATH="/usr/local/cuda-12.6/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
fi
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'not found')"

# Rust nightly (required for portable_simd)
if ! command -v rustc &>/dev/null; then
    echo "  Installing Rust nightly..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
    source "$HOME/.cargo/env"
fi
echo "  Rust: $(rustc --version)"

# Python + HuggingFace CLI
if ! command -v huggingface-cli &>/dev/null; then
    echo "  Installing HuggingFace CLI..."
    pip3 install -q huggingface_hub[cli] 2>/dev/null || {
        sudo apt-get install -y -qq python3-pip
        pip3 install -q huggingface_hub[cli]
    }
fi

# Git LFS for large files
if ! git lfs version &>/dev/null; then
    sudo apt-get install -y -qq git-lfs
    git lfs install
fi

echo "  All dependencies ready."
echo

# ============================================================================
# Phase 2: Clone Repository
# ============================================================================
echo "[2/6] Setting up repository..."
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/Bitsage-Network/stwo-ml.git "$REPO_DIR"
else
    cd "$REPO_DIR" && git pull origin main
fi
echo "  Repository: $REPO_DIR"
echo

# ============================================================================
# Phase 3: Download Qwen3-14B
# ============================================================================
echo "[3/6] Downloading Qwen3-14B model weights..."
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR"/*.safetensors 2>/dev/null)" ]; then
    mkdir -p "$MODEL_DIR"
    huggingface-cli download Qwen/Qwen3-14B \
        --include "*.safetensors" "config.json" "tokenizer.json" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    echo "  Downloaded to: $MODEL_DIR"
    echo "  Size: $(du -sh "$MODEL_DIR" | cut -f1)"
else
    echo "  Model already downloaded: $MODEL_DIR"
    echo "  Size: $(du -sh "$MODEL_DIR" | cut -f1)"
fi
echo

# ============================================================================
# Phase 4: Build Prover Binary
# ============================================================================
echo "[4/6] Building prove-qwen binary (release + GPU)..."
cd "$REPO_DIR"

# Ensure CUDA paths are set
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"

# Build with GPU + SafeTensors features
cargo build --release -p stwo-ml --bin prove-qwen \
    --features "safetensors" \
    2>&1 | tail -5

echo "  Binary: target/release/prove-qwen"
echo "  Size: $(ls -lh target/release/prove-qwen 2>/dev/null | awk '{print $5}' || echo 'N/A')"
echo

# ============================================================================
# Phase 5: GPU Check
# ============================================================================
echo "[5/6] GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader
echo

# ============================================================================
# Phase 6: Run Proof Generation
# ============================================================================
echo "[6/6] Running Qwen3-14B proof generation..."
echo "  Layers: $LAYERS"
echo "  Seq len: $SEQ_LEN"
echo

./target/release/prove-qwen \
    --model-dir "$MODEL_DIR" \
    --layers "$LAYERS" \
    --seq-len "$SEQ_LEN"

echo
echo "Deployment complete."
