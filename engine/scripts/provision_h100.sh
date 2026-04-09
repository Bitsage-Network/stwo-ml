#!/bin/bash
# obelyzk.rs H100 Provisioning Script
# Sets up the complete stack: Rust, CUDA, obelyzk engine, model, server
#
# Usage: curl -sSf https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/main/engine/scripts/provision_h100.sh | bash
# Or:    bash provision_h100.sh

set -euo pipefail

echo "╔═══════════════════════════════════════════╗"
echo "║  obelyzk.rs — H100 Provisioning           ║"
echo "║  Verifiable AI Engine Setup                ║"
echo "╚═══════════════════════════════════════════╝"
echo ""

# ── Check GPU ────────────────────────────────────────────────
echo "==> Checking GPU..."
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "  WARNING: nvidia-smi not found. CUDA builds may fail."
fi

# ── Install Rust ─────────────────────────────────────────────
echo ""
echo "==> Installing Rust nightly-2025-07-14..."
if command -v rustup &>/dev/null; then
    echo "  rustup already installed"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source "$HOME/.cargo/env"
fi
rustup toolchain install nightly-2025-07-14 --profile minimal
rustup default nightly-2025-07-14
echo "  $(rustc --version)"

# ── Install CUDA toolkit (if needed) ────────────────────────
echo ""
echo "==> Checking CUDA toolkit..."
if command -v nvcc &>/dev/null; then
    echo "  nvcc: $(nvcc --version | grep release | awk '{print $6}')"
else
    echo "  Installing CUDA toolkit 12.5..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq cuda-toolkit-12-5 2>/dev/null || {
        # Fallback: install from NVIDIA repo
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
        sudo apt-get install -y -qq cuda-toolkit-12-5
        rm -f cuda-keyring_1.1-1_all.deb
    }
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
fi

# ── Install build dependencies ───────────────────────────────
echo ""
echo "==> Installing build dependencies..."
sudo apt-get install -y -qq build-essential pkg-config libssl-dev git 2>/dev/null

# ── Clone obelyzk.rs ─────────────────────────────────────────
echo ""
REPO_DIR="$HOME/obelyzk.rs"
if [ -d "$REPO_DIR" ]; then
    echo "==> Updating obelyzk.rs..."
    cd "$REPO_DIR"
    git pull origin development 2>/dev/null || git pull origin main
else
    echo "==> Cloning obelyzk.rs..."
    git clone https://github.com/Bitsage-Network/obelyzk.rs.git "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout development 2>/dev/null || true
fi

# ── Build engine ─────────────────────────────────────────────
echo ""
echo "==> Building obelyzk engine (release, CUDA)..."
cd "$REPO_DIR/engine"
cargo build --release --bin obelyzk --features "server,server-stream,cuda-runtime,tui" 2>&1 | tail -5
echo "  Binary: $(ls -lh target/release/obelyzk | awk '{print $5}')"

# ── Download model ───────────────────────────────────────────
echo ""
MODEL_DIR="$HOME/.obelyzk/models"
mkdir -p "$MODEL_DIR"

# SmolLM2-135M for quick testing
SMOL_DIR="$MODEL_DIR/smollm2-135m"
if [ -d "$SMOL_DIR" ] && [ -f "$SMOL_DIR/config.json" ]; then
    echo "==> SmolLM2-135M already downloaded"
else
    echo "==> Downloading SmolLM2-135M (quick test model)..."
    pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM2-135M', local_dir='$SMOL_DIR',
                  ignore_patterns=['*.bin', '*.ot', '*.msgpack'])
print('Downloaded SmolLM2-135M')
"
fi

echo ""
echo "==> Quick benchmark (SmolLM2-135M, 1 token)..."
OBELYSK_MODEL_DIR="$SMOL_DIR" "$REPO_DIR/engine/target/release/obelyzk" bench --tokens 1 2>&1 | grep -E "Total time|Throughput|GPU"

echo ""
echo "==> Quick benchmark (SmolLM2-135M, 8 tokens)..."
OBELYSK_MODEL_DIR="$SMOL_DIR" "$REPO_DIR/engine/target/release/obelyzk" bench --tokens 8 2>&1 | grep -E "Total time|Throughput|GPU"

# ── Setup systemd service ────────────────────────────────────
echo ""
echo "==> Setting up obelyzk service..."
sudo tee /etc/systemd/system/obelyzk.service > /dev/null << EOF
[Unit]
Description=ObelyZK Verifiable AI Engine
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR/engine
Environment="PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
Environment="OBELYSK_MODEL_DIR=$SMOL_DIR"
Environment="PORT=8080"
Environment="PROVE_WORKERS=1"
Environment="PROVE_SERVER_API_KEY=obelyzk_h100_$(openssl rand -hex 8)"
ExecStart=$REPO_DIR/engine/target/release/obelyzk serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable obelyzk

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  obelyzk.rs — H100 Provisioning Complete                     ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║                                                               ║"
echo "║  Binary:    $REPO_DIR/engine/target/release/obelyzk          ║"
echo "║  Model:     $SMOL_DIR                                        ║"
echo "║                                                               ║"
echo "║  Commands:                                                    ║"
echo "║    obelyzk serve          # Start API server (port 8080)     ║"
echo "║    obelyzk chat           # Interactive verified chat        ║"
echo "║    obelyzk bench --tokens 64  # Throughput benchmark         ║"
echo "║    obelyzk dashboard      # Live Cipher Noir TUI             ║"
echo "║                                                               ║"
echo "║  Service:                                                     ║"
echo "║    sudo systemctl start obelyzk                              ║"
echo "║    sudo systemctl status obelyzk                             ║"
echo "║    journalctl -u obelyzk -f                                  ║"
echo "║                                                               ║"
echo "║  To download Qwen3-14B (16GB, for real benchmarks):          ║"
echo "║    python3 -c \"from huggingface_hub import snapshot_download;║"
echo "║    snapshot_download('Qwen/Qwen2.5-14B',                     ║"
echo "║      local_dir='$MODEL_DIR/qwen3-14b')\"                     ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Add obelyzk to PATH
echo "export PATH=\"$REPO_DIR/engine/target/release:\$PATH\"" >> ~/.bashrc
echo "export OBELYSK_MODEL_DIR=\"$SMOL_DIR\"" >> ~/.bashrc
source ~/.bashrc 2>/dev/null || true
