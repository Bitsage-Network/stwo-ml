#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — Worker Provisioning
#
# Provisions a GPU machine as a prover worker:
#   1. Installs prove-server with CUDA
#   2. Sets up nginx + HTTPS (Let's Encrypt)
#   3. Registers with coordinator
#   4. Installs branded shell (MOTD + prompt)
#   5. Starts as systemd service
#
# Usage:
#   # On the GPU machine itself:
#   curl -sSf https://raw.githubusercontent.com/bitsage-network/stwo-ml/main/scripts/provision_worker.sh | bash -s -- \
#     --domain my-prover.example.com \
#     --wallet 0x... \
#     --email admin@example.com
#
#   # Or via SSH from your laptop:
#   ssh user@gpu-vm 'bash -s' < scripts/provision_worker.sh -- \
#     --domain my-prover.example.com \
#     --wallet 0x...
#
# Options:
#   --domain DOMAIN     Domain pointing to this machine (for SSL)
#   --wallet ADDRESS    Starknet wallet for earnings
#   --email EMAIL       Email for Let's Encrypt (default: admin@bitsage.network)
#   --name NAME         Worker name (default: derived from domain)
#   --region REGION     Region hint (default: auto-detect)
#   --coordinator URL   Coordinator URL (default: https://api.bitsage.network)
#   --no-ssl            Skip HTTPS setup
#   --no-brand          Skip branded shell
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Cipher Noir palette ─────────────────────────────────────────────
LIME='\033[38;5;118m'
LIME_DIM='\033[38;5;70m'
EMERALD='\033[38;5;48m'
VIOLET='\033[38;5;73m'
ORANGE='\033[38;5;208m'
WHITE='\033[38;5;255m'
SILVER='\033[38;5;249m'
SLATE='\033[38;5;245m'
GHOST='\033[38;5;240m'
RED='\033[38;5;178m'
BOLD='\033[1m'
X='\033[0m'
BG_LIME='\033[48;5;118m'
FG_BLACK='\033[38;5;0m'

H="─"; CHECK="✓"; CROSS="✗"; ARROW="▸"; DOT="·"

step_ok()   { echo -e "  ${EMERALD}${CHECK}${X} $*"; }
step_info() { echo -e "  ${LIME}${ARROW}${X} ${SILVER}$*${X}"; }
step_warn() { echo -e "  ${ORANGE}!${X} $*"; }
step_fail() { echo -e "  ${RED}${CROSS}${X} $*"; exit 1; }

section() {
    echo ""
    echo -e "  ${BG_LIME}${FG_BLACK}${BOLD} $1 ${X} ${WHITE}${BOLD}$2${X} ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
    echo ""
}

# ── Parse args ──────────────────────────────────────────────────────
DOMAIN=""
WALLET=""
EMAIL="admin@bitsage.network"
WORKER_NAME=""
REGION=""
COORDINATOR="https://api.bitsage.network"
SETUP_SSL=true
SETUP_BRAND=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)      DOMAIN="$2"; shift 2;;
        --wallet)      WALLET="$2"; shift 2;;
        --email)       EMAIL="$2"; shift 2;;
        --name)        WORKER_NAME="$2"; shift 2;;
        --region)      REGION="$2"; shift 2;;
        --coordinator) COORDINATOR="$2"; shift 2;;
        --no-ssl)      SETUP_SSL=false; shift;;
        --no-brand)    SETUP_BRAND=false; shift;;
        *)             echo "Unknown: $1"; exit 1;;
    esac
done

[[ -z "$WALLET" ]] && step_fail "Missing --wallet ADDRESS"
[[ -z "$DOMAIN" ]] && [[ "$SETUP_SSL" == "true" ]] && step_fail "Missing --domain (or use --no-ssl)"
[[ -z "$WORKER_NAME" ]] && WORKER_NAME="${DOMAIN%%.*}"

# ── Banner ──────────────────────────────────────────────────────────
echo ""
echo -e "  ${LIME}${BOLD}╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═${X}"
echo -e "  ${LIME}║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗${X}"
echo -e "  ${LIME_DIM}╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩${X}"
echo -e "  ${SILVER}Worker Provisioning${X}"
echo ""

# ── Detect GPU ──────────────────────────────────────────────────────
section "1" "GPU DETECTION"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%.0f", $1/1024}')
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    step_ok "${WHITE}${GPU_NAME}${X} ${GHOST}${GPU_VRAM}GB VRAM${X} ${GHOST}driver ${GPU_DRIVER}${X}"
else
    step_warn "No NVIDIA GPU detected — running in CPU mode"
    GPU_NAME="CPU"
    GPU_VRAM=0
fi

# Auto-detect region
if [[ -z "$REGION" ]]; then
    REGION=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "unknown")
fi

echo ""
echo -e "  ${SLATE}Domain${X}       ${WHITE}${DOMAIN:-none}${X}"
echo -e "  ${SLATE}Wallet${X}       ${VIOLET}${WALLET}${X}"
echo -e "  ${SLATE}Worker${X}       ${WHITE}${WORKER_NAME}${X}"
echo -e "  ${SLATE}Region${X}       ${WHITE}${REGION}${X}"
echo -e "  ${SLATE}Coordinator${X}  ${WHITE}${COORDINATOR}${X}"

# ── Install dependencies ────────────────────────────────────────────
section "2" "DEPENDENCIES"

if ! command -v rustup &>/dev/null; then
    step_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none 2>/dev/null
fi
source "$HOME/.cargo/env" 2>/dev/null || true

TOOLCHAIN="nightly-2025-07-14"
if ! rustup toolchain list | grep -q "$TOOLCHAIN" 2>/dev/null; then
    step_info "Installing Rust ${TOOLCHAIN}..."
    rustup toolchain install "$TOOLCHAIN" --profile minimal 2>/dev/null
fi
step_ok "${WHITE}Rust${X} ${SLATE}${TOOLCHAIN}${X}"

if ! command -v nginx &>/dev/null && $SETUP_SSL; then
    step_info "Installing nginx + certbot..."
    sudo apt-get update -qq 2>/dev/null
    sudo apt-get install -y -qq nginx certbot python3-certbot-nginx 2>/dev/null
    step_ok "${WHITE}nginx + certbot${X}"
fi

sudo apt-get install -y -qq pkg-config libssl-dev git python3 python3-pip 2>/dev/null
step_ok "${WHITE}Build dependencies${X}"

# ── Clone and build ─────────────────────────────────────────────────
section "3" "BUILD"

if [[ -d ~/stwo-ml/stwo-ml/Cargo.toml ]] 2>/dev/null; then
    cd ~/stwo-ml && git pull origin main 2>/dev/null || true
    step_ok "Source updated"
else
    step_info "Cloning repository..."
    git clone --depth 1 https://github.com/bitsage-network/stwo-ml.git ~/stwo-ml 2>/dev/null
    step_ok "Source cloned"
fi

cd ~/stwo-ml/stwo-ml

FEATURES="std,server,model-loading,safetensors"
if [[ "$GPU_NAME" != "CPU" ]]; then
    FEATURES="$FEATURES,gpu,cuda-runtime"
fi

step_info "Building prove-server ${GHOST}(features: ${FEATURES})${X}"
cargo +$TOOLCHAIN build --release --bin prove-server --features "$FEATURES" 2>&1 | tail -3
step_ok "${WHITE}prove-server${X} built"

# ── Download model ──────────────────────────────────────────────────
section "4" "MODEL"

mkdir -p ~/.obelysk/models/smollm2-135m
if [[ ! -f ~/.obelysk/models/smollm2-135m/config.json ]]; then
    step_info "Downloading SmolLM2-135M..."
    pip3 install -q huggingface-hub 2>/dev/null
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM2-135M', local_dir='$HOME/.obelysk/models/smollm2-135m',
    allow_patterns=['*.safetensors','config.json','tokenizer*','*.json'])" 2>&1 | tail -3
    step_ok "SmolLM2-135M downloaded"
else
    step_ok "SmolLM2-135M ${SLATE}cached${X}"
fi

# ── Configure systemd ───────────────────────────────────────────────
section "5" "SERVICE"

WORKER_ADDRESS="http://$(curl -s --max-time 3 http://checkip.amazonaws.com 2>/dev/null || hostname -I | awk '{print $1}'):8080"
if [[ -n "$DOMAIN" ]]; then
    if $SETUP_SSL; then
        WORKER_ADDRESS="https://${DOMAIN}"
    else
        WORKER_ADDRESS="http://${DOMAIN}"
    fi
fi

sudo tee /etc/systemd/system/bitsage-prover.service > /dev/null << SERVICE
[Unit]
Description=ObelyZK Prove Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/stwo-ml/stwo-ml
ExecStart=$HOME/stwo-ml/stwo-ml/target/release/prove-server
Environment=BIND_ADDR=0.0.0.0:8080
Environment=RUST_LOG=info
Environment=OBELYSK_MODEL_DIR=$HOME/.obelysk/models/smollm2-135m
Environment=MAX_QUEUE_DEPTH=64
Environment=COORDINATOR_URL=${COORDINATOR}
Environment=WORKER_WALLET=${WALLET}
Environment=WORKER_ADDRESS=${WORKER_ADDRESS}
Environment=WORKER_NAME=${WORKER_NAME}
Environment=WORKER_REGION=${REGION}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable bitsage-prover
sudo systemctl start bitsage-prover
step_ok "prove-server service started"

# ── HTTPS (nginx + certbot) ─────────────────────────────────────────
if $SETUP_SSL && [[ -n "$DOMAIN" ]]; then
    section "6" "HTTPS"

    sudo tee /etc/nginx/sites-available/${DOMAIN} > /dev/null << NGINX
server {
    listen 80;
    server_name ${DOMAIN};

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        client_max_body_size 50M;
    }
}
NGINX

    sudo ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t 2>/dev/null && sudo systemctl reload nginx

    step_info "Getting SSL certificate..."
    sudo certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --email ${EMAIL} --redirect 2>/dev/null
    step_ok "HTTPS enabled: ${WHITE}https://${DOMAIN}${X}"
fi

# ── Branded shell ───────────────────────────────────────────────────
if $SETUP_BRAND; then
    section "7" "BRANDED SHELL"

    # MOTD
    sudo tee /etc/motd > /dev/null << 'MOTD'

  ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═
  ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗
  ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩

  ObelyZK Prover Node
  ─────────────────────────────────────

MOTD

    # Dynamic MOTD script
    sudo tee /etc/update-motd.d/99-obelysk > /dev/null << 'DYNMOTD'
#!/bin/bash
LIME='\033[38;5;118m'
EMERALD='\033[38;5;48m'
GHOST='\033[38;5;240m'
WHITE='\033[38;5;255m'
SLATE='\033[38;5;245m'
VIOLET='\033[38;5;73m'
X='\033[0m'

# GPU info
GPU=$(nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ -n "$GPU" ]]; then
    IFS=',' read -r GPU_NAME MEM_USED MEM_TOTAL TEMP <<< "$GPU"
    printf "  ${SLATE}GPU${X}       ${WHITE}${GPU_NAME}${X}  ${GHOST}${MEM_USED}/${MEM_TOTAL} MiB  ${TEMP}°C${X}\n"
fi

# Service status
if systemctl is-active --quiet bitsage-prover 2>/dev/null; then
    HEALTH=$(curl -s --max-time 2 http://localhost:8080/health 2>/dev/null)
    if [[ -n "$HEALTH" ]]; then
        MODELS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('loaded_models',0))" 2>/dev/null)
        JOBS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_jobs',0))" 2>/dev/null)
        printf "  ${SLATE}Prover${X}    ${EMERALD}online${X}  ${GHOST}models: ${MODELS}  jobs: ${JOBS}${X}\n"
    else
        printf "  ${SLATE}Prover${X}    ${LIME}starting...${X}\n"
    fi
else
    printf "  ${SLATE}Prover${X}    ${GHOST}offline${X}\n"
fi

# Uptime
printf "  ${SLATE}Uptime${X}    ${WHITE}$(uptime -p | sed 's/up //')${X}\n"
printf "\n"
printf "  ${GHOST}Logs:    journalctl -u bitsage-prover -f${X}\n"
printf "  ${GHOST}Health:  curl localhost:8080/health${X}\n"
printf "\n"
DYNMOTD
    sudo chmod +x /etc/update-motd.d/99-obelysk

    # Disable default MOTD scripts
    sudo chmod -x /etc/update-motd.d/10-help-text 2>/dev/null || true
    sudo chmod -x /etc/update-motd.d/50-motd-news 2>/dev/null || true
    sudo chmod -x /etc/update-motd.d/50-landscape-sysinfo 2>/dev/null || true

    # Branded bash prompt
    cat >> ~/.bashrc << 'BASHRC'

# ObelyZK branded prompt
PS1='\[\033[38;5;118m\]obelysk\[\033[0m\]:\[\033[38;5;73m\]\w\[\033[0m\]\$ '
alias prover-logs='journalctl -u bitsage-prover -f'
alias prover-health='curl -s localhost:8080/health | python3 -m json.tool'
alias prover-restart='sudo systemctl restart bitsage-prover'
alias gpu='nvidia-smi'
BASHRC

    step_ok "Branded shell installed"
    step_ok "SSH prompt: ${LIME}obelysk${X}:${VIOLET}/path${X}\$"
    step_ok "Aliases: ${WHITE}prover-logs${X}, ${WHITE}prover-health${X}, ${WHITE}prover-restart${X}, ${WHITE}gpu${X}"
fi

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo -e "  ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
echo ""
echo -e "  ${EMERALD}${BOLD}${CHECK} Worker Provisioned${X}"
echo ""
echo -e "  ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
echo ""
echo -e "  ${SLATE}GPU${X}          ${WHITE}${GPU_NAME}${X}"
echo -e "  ${SLATE}Worker${X}       ${WHITE}${WORKER_NAME}${X}"
echo -e "  ${SLATE}Wallet${X}       ${VIOLET}${WALLET}${X}"
echo -e "  ${SLATE}Coordinator${X}  ${WHITE}${COORDINATOR}${X}"
if [[ -n "$DOMAIN" ]]; then
    echo -e "  ${SLATE}URL${X}          ${EMERALD}https://${DOMAIN}${X}"
fi
echo ""
echo -e "  ${LIME}${ARROW}${X} ${WHITE}Check health${X}"
echo -e "    curl https://${DOMAIN:-localhost:8080}/health"
echo ""
echo -e "  ${LIME}${ARROW}${X} ${WHITE}View logs${X}"
echo -e "    journalctl -u bitsage-prover -f"
echo ""
echo -e "  ${LIME}${ARROW}${X} ${WHITE}Earnings${X}"
echo -e "    Proofs generated earn SAGE tokens to ${VIOLET}${WALLET}${X}"
echo ""
