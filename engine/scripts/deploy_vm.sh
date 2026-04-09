#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — Cloud VM Deployment
#
# Deploy a production proving node on any cloud VM.
# Supports: AWS EC2, GCP, bare metal — any Linux with SSH access.
#
# Usage:
#   # Deploy to a remote VM via SSH
#   ./scripts/deploy_vm.sh --host user@gpu-vm.example.com
#
#   # Deploy locally (bare metal)
#   ./scripts/deploy_vm.sh --local
#
#   # Deploy with prove-server (API mode)
#   ./scripts/deploy_vm.sh --host user@vm --mode server
#
#   # Deploy with TUI (interactive mode)
#   ./scripts/deploy_vm.sh --host user@vm --mode tui
#
# Options:
#   --host HOST     SSH target (user@host)
#   --local         Deploy on this machine
#   --mode MODE     server | tui | full (default: full)
#   --model MODEL   Model to provision (default: qwen2-0.5b)
#   --port PORT     Prove-server port (default: 8080)
#   --branch BRANCH Git branch to deploy (default: current)
#   --key FILE      SSH key file
#   --no-model      Skip model download
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── ANSI 256-color palette (Cipher Noir) ─────────────────────────────
LIME='\033[38;5;118m'        # primary brand
LIME_DIM='\033[38;5;70m'     # secondary
EMERALD='\033[38;5;48m'      # success
VIOLET='\033[38;5;73m'       # hashes
ORANGE='\033[38;5;208m'      # warnings
LILAC='\033[38;5;141m'       # labels
WHITE='\033[38;5;255m'       # bright white
SILVER='\033[38;5;249m'      # light gray
SLATE='\033[38;5;245m'       # medium gray
GHOST='\033[38;5;240m'       # dark gray
RED='\033[38;5;178m'         # errors
BOLD='\033[1m'; DIM='\033[2m'; X='\033[0m'
H="─"; V="│"; CHECK="✓"; CROSS="✗"; ARROW="▸"; DOT="·"; DIAMOND="◆"

info()  { echo -e "  ${LIME}${ARROW}${X} ${SILVER}$*${X}"; }
warn()  { echo -e "  ${ORANGE}!${X} $*"; }
error() { echo -e "  ${RED}${CROSS}${X} $*"; exit 1; }
ok()    { echo -e "  ${EMERALD}${CHECK}${X} $*"; }
step()  { echo -e "\n  ${LIME}┌${H}${X} ${WHITE}${BOLD}$2${X} ${GHOST}$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H$H${X}"; echo -e "  ${GHOST}${V}${X}"; }

# ── Parse args ───────────────────────────────────────────────────────
HOST=""
LOCAL=false
MODE="full"
MODEL="qwen2-0.5b"
PORT=8080
BRANCH=""
SSH_KEY=""
NO_MODEL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)     HOST="$2"; shift 2;;
        --local)    LOCAL=true; shift;;
        --mode)     MODE="$2"; shift 2;;
        --model)    MODEL="$2"; shift 2;;
        --port)     PORT="$2"; shift 2;;
        --branch)   BRANCH="$2"; shift 2;;
        --key)      SSH_KEY="-i $2"; shift 2;;
        --no-model) NO_MODEL=true; shift;;
        *)          error "Unknown option: $1";;
    esac
done

if [[ -z "$HOST" ]] && ! $LOCAL; then
    echo "Usage: deploy_vm.sh --host user@host [--mode server|tui|full]"
    echo "       deploy_vm.sh --local [--mode server|tui|full]"
    exit 1
fi

[[ -z "$BRANCH" ]] && BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# ── Banner ───────────────────────────────────────────────────────────
clear 2>/dev/null || true
echo ""
echo -e "${LIME}  ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═${X}"
echo -e "${LIME}  ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗${X}"
echo -e "${LIME_DIM}  ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩${X}"
echo -e "  ${SILVER}V E R I F I A B L E   M L${X}  ${GHOST}${DOT}${X}  ${SLATE}Cloud VM Deployment${X}"
echo ""
echo -e "  ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"

TARGET_DESC=$($LOCAL && echo "localhost" || echo "$HOST")
echo ""
echo -e "  ${SLATE}Target${X}   ${WHITE}$TARGET_DESC${X}"
echo -e "  ${SLATE}Mode${X}     ${LILAC}$MODE${X}"
echo -e "  ${SLATE}Model${X}    ${LILAC}$MODEL${X}"
echo -e "  ${SLATE}Branch${X}   ${VIOLET}$BRANCH${X}"
echo ""

# ── Remote execution helper ──────────────────────────────────────────

run_remote() {
    if $LOCAL; then
        bash -c "$1"
    else
        ssh $SSH_KEY -o StrictHostKeyChecking=no "$HOST" "$1"
    fi
}

run_remote_script() {
    if $LOCAL; then
        bash -s <<< "$1"
    else
        ssh $SSH_KEY -o StrictHostKeyChecking=no "$HOST" bash -s <<< "$1"
    fi
}

# ── Step 1: Probe VM ────────────────────────────────────────────────

step "1/7" "Probing VM environment"

VM_INFO=$(run_remote "
    echo \"OS=\$(uname -s) \$(uname -r)\"
    echo \"ARCH=\$(uname -m)\"
    echo \"CPU=\$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)\"
    echo \"MEM=\$(free -g 2>/dev/null | awk '/Mem:/{print \$2}' || echo '?')G\"
    if command -v nvidia-smi &>/dev/null; then
        echo \"GPU=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\"
        echo \"VRAM=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)\"
    else
        echo 'GPU=none'
        echo 'VRAM=0'
    fi
")

echo "$VM_INFO" | while read -r line; do info "$line"; done

HAS_GPU=$(echo "$VM_INFO" | grep "GPU=" | grep -v "none" && echo true || echo false)

# ── Step 2: Install system dependencies ─────────────────────────────

step "2/7" "Installing system dependencies"

run_remote_script "
set -e

# Detect package manager
if command -v apt-get &>/dev/null; then
    PKG='apt-get'
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential pkg-config libssl-dev git curl python3 python3-pip
elif command -v yum &>/dev/null; then
    PKG='yum'
    sudo yum install -y gcc gcc-c++ openssl-devel git curl python3 python3-pip
elif command -v brew &>/dev/null; then
    PKG='brew'
    brew install openssl git python3
else
    echo 'Unknown package manager'
    exit 1
fi

# Node.js (for on-chain scripts)
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - 2>/dev/null || true
    sudo \$PKG install -y nodejs 2>/dev/null || true
fi

echo 'DEPS_OK'
"
ok "System dependencies"

# ── Step 3: Install Rust ────────────────────────────────────────────

step "3/7" "Installing Rust toolchain"

run_remote_script "
set -e
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source \$HOME/.cargo/env
fi
source \$HOME/.cargo/env

TOOLCHAIN='nightly-2025-07-14'
if ! rustup toolchain list | grep -q \"\$TOOLCHAIN\"; then
    rustup toolchain install \$TOOLCHAIN --profile minimal
fi
echo \"RUST_OK \$(rustc +\$TOOLCHAIN --version 2>/dev/null | head -1)\"
"
ok "Rust nightly-2025-07-14"

# ── Step 4: Clone and build ─────────────────────────────────────────

step "4/7" "Building ObelyZK (branch: $BRANCH)"

# Determine features based on mode + GPU
FEATURES="std"
if [[ "$HAS_GPU" == "true" ]]; then
    FEATURES="$FEATURES,gpu,cuda-runtime"
fi

BUILD_TARGETS=""
case "$MODE" in
    server)
        FEATURES="$FEATURES,server,model-loading,safetensors,binary-proof"
        BUILD_TARGETS="--bin prove-server"
        ;;
    tui)
        FEATURES="$FEATURES,tui,cli,model-loading,safetensors"
        BUILD_TARGETS="--bin obelysk"
        ;;
    full)
        FEATURES="$FEATURES,cli,model-loading,safetensors,audit,parallel-audit,tui,server,binary-proof"
        BUILD_TARGETS="--bin prove-model --bin obelysk --bin prove-server"
        ;;
esac

info "Features: $FEATURES"

run_remote_script "
set -e
source \$HOME/.cargo/env

# Clone or update
if [[ -d ~/obelysk ]]; then
    cd ~/obelysk
    git fetch origin
    git checkout $BRANCH 2>/dev/null || git checkout -b $BRANCH origin/$BRANCH
    git pull origin $BRANCH 2>/dev/null || true
else
    git clone --branch $BRANCH https://github.com/Bitsage-Network/stwo-ml.git ~/obelysk 2>/dev/null || \
    git clone https://github.com/Bitsage-Network/stwo-ml.git ~/obelysk
    cd ~/obelysk
fi

# Build
cargo +nightly-2025-07-14 build --release $BUILD_TARGETS --features '$FEATURES' 2>&1 | tail -5

echo 'BUILD_OK'
"
ok "Build complete"

# ── Step 5: Download model ──────────────────────────────────────────

if ! $NO_MODEL; then
    step "5/7" "Downloading model: $MODEL"

    run_remote_script "
set -e
pip3 install -q huggingface-hub 2>/dev/null || true

MODEL_DIR=\$HOME/.obelysk/models/$MODEL
mkdir -p \"\$MODEL_DIR\"

if [[ ! -f \"\$MODEL_DIR/config.json\" ]]; then
    python3 -c \"
from huggingface_hub import snapshot_download
import sys
models = {
    'qwen2-0.5b': 'Qwen/Qwen2-0.5B',
    'phi3-mini': 'microsoft/Phi-3-mini-4k-instruct',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
    'qwen3-14b': 'Qwen/Qwen3-14B',
}
repo = models.get('$MODEL', 'Qwen/Qwen2-0.5B')
snapshot_download(repo, local_dir='\$MODEL_DIR',
    allow_patterns=['*.safetensors','config.json','tokenizer*','*.json'])
\" 2>&1 | tail -3
    echo 'MODEL_OK'
else
    echo 'MODEL_CACHED'
fi
"
    ok "Model ready"
else
    step "5/7" "Skipping model download"
fi

# ── Step 6: Configure systemd service ───────────────────────────────

step "6/7" "Configuring service"

if [[ "$MODE" == "server" ]] || [[ "$MODE" == "full" ]]; then
    run_remote_script "
set -e
mkdir -p ~/.obelysk/proofs

# Create systemd service for prove-server
sudo tee /etc/systemd/system/obelysk-server.service > /dev/null << 'SERVICE'
[Unit]
Description=ObelyZK Prove Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/root/obelysk
ExecStart=/root/obelysk/target/release/prove-server
Environment=BIND_ADDR=0.0.0.0:$PORT
Environment=RUST_LOG=info
Environment=OBELYSK_MODEL_DIR=/root/.obelysk/models/$MODEL
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable obelysk-server
sudo systemctl restart obelysk-server

echo 'SERVICE_OK'
" 2>/dev/null || warn "systemd setup skipped (non-root or macOS)"
    ok "prove-server configured on port $PORT"
fi

# Write env config
run_remote_script "
cat > ~/.obelysk/config.env << CONF
export OBELYSK_MODEL_DIR=\$HOME/.obelysk/models/$MODEL
export OBELYSK_PORT=$PORT
export OBELYSK_NETWORK='Starknet Sepolia'
CONF
echo 'CONFIG_OK'
"
ok "Config written"

# ── Step 7: Validate ────────────────────────────────────────────────

step "7/7" "Validating deployment"

VALIDATION=$(run_remote "
source ~/.cargo/env
cd ~/obelysk

# Test binary
if [[ -f target/release/prove-model ]]; then
    echo 'BINARY prove-model OK'
fi
if [[ -f target/release/obelysk ]]; then
    echo 'BINARY obelysk OK'
fi
if [[ -f target/release/prove-server ]]; then
    echo 'BINARY prove-server OK'
fi

# Test model
if [[ -f ~/.obelysk/models/$MODEL/config.json ]]; then
    echo 'MODEL $MODEL OK'
fi

# Test GPU
if command -v nvidia-smi &>/dev/null; then
    echo \"GPU \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"
fi

# Test server (if running)
if curl -sf http://localhost:$PORT/health &>/dev/null; then
    echo 'SERVER healthy'
fi
")

echo "$VALIDATION" | while read -r line; do ok "$line"; done

# ── Done ─────────────────────────────────────────────────────────────

echo ""
echo ""
echo -e "  ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
echo ""
echo -e "  ${EMERALD}${BOLD}${CHECK} Deployment Complete${X}"
echo ""
echo -e "  ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
echo ""

if [[ "$MODE" == "server" ]] || [[ "$MODE" == "full" ]]; then
    echo -e "  ${LIME}${ARROW}${X} ${WHITE}API Server${X}"
    if $LOCAL; then
        echo -e "    ${LIME}curl http://localhost:$PORT/health${X}"
    else
        echo -e "    ${LIME}curl http://$HOST:$PORT/health${X}"
    fi
    echo ""
fi

if [[ "$MODE" == "tui" ]] || [[ "$MODE" == "full" ]]; then
    echo -e "  ${LIME}${ARROW}${X} ${WHITE}Interactive TUI${X}"
    if $LOCAL; then
        echo -e "    ${LIME}~/obelysk/target/release/obelysk${X}"
    else
        echo -e "    ${LIME}ssh $HOST '~/obelysk/target/release/obelysk'${X}"
    fi
    echo ""
fi

echo -e "  ${LIME}${ARROW}${X} ${WHITE}Prove a model${X}"
if $LOCAL; then
    echo -e "    ${LIME}prove-model --model-dir ~/.obelysk/models/$MODEL --layers 1 --gkr${X}"
else
    echo -e "    ${LIME}ssh $HOST 'prove-model --model-dir ~/.obelysk/models/$MODEL --layers 1 --gkr'${X}"
fi
echo ""
echo -e "  ${GHOST}Logs    journalctl -u obelysk-server -f${X}"
echo -e "  ${GHOST}Config  ~/.obelysk/config.env${X}"
echo ""
