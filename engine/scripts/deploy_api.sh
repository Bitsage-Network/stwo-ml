#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — Deploy Prove-Server API
#
# Deploy the hosted prover that SDKs connect to.
# This is the service behind api.bitsage.network/api/v1/prove
#
# Usage:
#   ./scripts/deploy_api.sh --local                     # Run locally
#   ./scripts/deploy_api.sh --docker                    # Docker container
#   ./scripts/deploy_api.sh --host user@gpu-vm          # Remote VM via SSH
#   ./scripts/deploy_api.sh --local --api-key sk-test   # With auth
#   ./scripts/deploy_api.sh --local --gpu               # With GPU
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Cipher Noir palette ─────────────────────────────────────────────
LIME='\033[38;5;118m'
LIME_DIM='\033[38;5;70m'
EMERALD='\033[38;5;48m'
VIOLET='\033[38;5;73m'
ORANGE='\033[38;5;208m'
LILAC='\033[38;5;141m'
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
step_warn() { echo -e "  ${ORANGE}!${X} $*"; }
step_fail() { echo -e "  ${RED}${CROSS}${X} $*"; exit 1; }
step_info() { echo -e "  ${LIME}${ARROW}${X} ${SILVER}$*${X}"; }

section() {
    echo ""
    echo -e "  ${BG_LIME}${FG_BLACK}${BOLD} $1 ${X} ${WHITE}${BOLD}$2${X} ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
    echo ""
}

# ── Parse args ──────────────────────────────────────────────────────
MODE=""
HOST=""
PORT=8080
API_KEY=""
GPU=false
MODEL="qwen2-0.5b"
SSH_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)    MODE="local"; shift;;
        --docker)   MODE="docker"; shift;;
        --host)     MODE="remote"; HOST="$2"; shift 2;;
        --port)     PORT="$2"; shift 2;;
        --api-key)  API_KEY="$2"; shift 2;;
        --gpu)      GPU=true; shift;;
        --model)    MODEL="$2"; shift 2;;
        --key)      SSH_KEY="-i $2"; shift 2;;
        -h|--help)
            echo ""
            echo -e "  ${LIME}Deploy Prove-Server API${X}"
            echo ""
            echo -e "  ${WHITE}Usage:${X}"
            echo -e "    ${LIME}--local${X}           Run on this machine"
            echo -e "    ${LIME}--docker${X}          Build + run in Docker"
            echo -e "    ${LIME}--host${X} USER@HOST  Deploy to remote VM via SSH"
            echo -e "    ${LIME}--port${X} PORT       API port (default: 8080)"
            echo -e "    ${LIME}--api-key${X} KEY     Enable API key auth"
            echo -e "    ${LIME}--gpu${X}             Enable GPU acceleration"
            echo -e "    ${LIME}--model${X} MODEL     Pre-load model (default: qwen2-0.5b)"
            echo ""
            exit 0;;
        *)  step_fail "Unknown option: $1";;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: deploy_api.sh --local | --docker | --host user@host"
    exit 1
fi

# ── Find project ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Banner ──────────────────────────────────────────────────────────
echo ""
echo -e "  ${LIME}${BOLD}╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═${X}"
echo -e "  ${LIME}║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗${X}"
echo -e "  ${LIME_DIM}╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩${X}"
echo -e "  ${SILVER}Prove-Server API Deployment${X}"
echo ""
echo -e "  ${SLATE}Mode${X}      ${WHITE}${MODE}${X}"
echo -e "  ${SLATE}Port${X}      ${WHITE}${PORT}${X}"
echo -e "  ${SLATE}GPU${X}       ${WHITE}${GPU}${X}"
echo -e "  ${SLATE}Model${X}     ${LILAC}${MODEL}${X}"
[[ -n "$API_KEY" ]] && echo -e "  ${SLATE}Auth${X}      ${EMERALD}enabled${X}"
[[ -n "$HOST" ]] && echo -e "  ${SLATE}Host${X}      ${WHITE}${HOST}${X}"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# LOCAL MODE
# ═══════════════════════════════════════════════════════════════════════

if [[ "$MODE" == "local" ]]; then

    section "1" "BUILD"

    TOOLCHAIN="nightly-2025-07-14"
    FEATURES="std,server,model-loading,safetensors,binary-proof"
    if $GPU; then
        if command -v nvidia-smi &>/dev/null; then
            FEATURES="$FEATURES,gpu,cuda-runtime"
            step_ok "CUDA detected"
        elif [[ "$(uname)" == "Darwin" ]]; then
            FEATURES="$FEATURES,metal"
            step_ok "Metal GPU"
        fi
    fi

    step_info "Building prove-server ${GHOST}(features: ${FEATURES})${X}"
    cd "$PROJECT_DIR"
    cargo +$TOOLCHAIN build --release --bin prove-server --features "$FEATURES" 2>&1 | tail -3
    step_ok "prove-server built"

    section "2" "START"

    export BIND_ADDR="0.0.0.0:${PORT}"
    export RUST_LOG="${RUST_LOG:-info}"
    [[ -n "$API_KEY" ]] && export API_KEY="$API_KEY"
    export OBELYSK_MODEL_DIR="$HOME/.obelysk/models/$MODEL"

    BINARY="$PROJECT_DIR/target/release/prove-server"

    step_info "Starting on ${WHITE}http://0.0.0.0:${PORT}${X}"

    # Start in background, wait for health
    "$BINARY" &
    SERVER_PID=$!
    echo "$SERVER_PID" > /tmp/obelysk-prove-server.pid
    step_info "PID: ${WHITE}${SERVER_PID}${X}"

    # Wait for health
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
            HEALTH=$(curl -s "http://localhost:${PORT}/health")
            step_ok "Server healthy"
            break
        fi
        sleep 1
    done

# ═══════════════════════════════════════════════════════════════════════
# DOCKER MODE
# ═══════════════════════════════════════════════════════════════════════

elif [[ "$MODE" == "docker" ]]; then

    section "1" "DOCKERFILE"

    DOCKERFILE="$PROJECT_DIR/Dockerfile"
    if [[ ! -f "$DOCKERFILE" ]]; then
        step_info "Creating Dockerfile..."
        cat > "$DOCKERFILE" << 'DOCKER'
# ── Build stage ─────────────────────────────────────────────────────
FROM rust:1.90-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN rustup toolchain install nightly-2025-07-14 --profile minimal

WORKDIR /app
COPY . .

RUN cargo +nightly-2025-07-14 build --release \
    --bin prove-server \
    --features "std,server,model-loading,safetensors,binary-proof"

# ── Runtime stage ───────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/prove-server /usr/local/bin/

ENV BIND_ADDR=0.0.0.0:8080
ENV RUST_LOG=info

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["prove-server"]
DOCKER
        step_ok "Dockerfile created"
    else
        step_ok "Dockerfile exists"
    fi

    section "2" "BUILD IMAGE"

    cd "$PROJECT_DIR"
    step_info "Building Docker image..."

    DOCKER_ARGS=""
    if $GPU; then
        DOCKER_ARGS="--build-arg GPU=1"
    fi

    docker build -t obelysk/prove-server:latest $DOCKER_ARGS . 2>&1 | tail -5
    step_ok "Image built: ${WHITE}obelysk/prove-server:latest${X}"

    section "3" "RUN CONTAINER"

    DOCKER_RUN="docker run -d --name obelysk-prover -p ${PORT}:8080"
    DOCKER_RUN="$DOCKER_RUN -e RUST_LOG=info"
    [[ -n "$API_KEY" ]] && DOCKER_RUN="$DOCKER_RUN -e API_KEY=$API_KEY"
    DOCKER_RUN="$DOCKER_RUN -v $HOME/.obelysk/models:/models"
    DOCKER_RUN="$DOCKER_RUN -e OBELYSK_MODEL_DIR=/models/$MODEL"

    if $GPU; then
        DOCKER_RUN="$DOCKER_RUN --gpus all"
    fi

    DOCKER_RUN="$DOCKER_RUN obelysk/prove-server:latest"

    step_info "Running: ${GHOST}${DOCKER_RUN}${X}"
    CONTAINER_ID=$(eval "$DOCKER_RUN" 2>/dev/null)
    step_ok "Container: ${WHITE}${CONTAINER_ID:0:12}${X}"

    # Wait for health
    sleep 3
    if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
        step_ok "Container healthy"
    else
        step_warn "Waiting for container to start..."
        sleep 5
    fi

# ═══════════════════════════════════════════════════════════════════════
# REMOTE MODE
# ═══════════════════════════════════════════════════════════════════════

elif [[ "$MODE" == "remote" ]]; then

    section "1" "CONNECT"
    step_info "Target: ${WHITE}${HOST}${X}"

    # Test SSH
    if ! ssh $SSH_KEY -o ConnectTimeout=5 "$HOST" "echo ok" &>/dev/null; then
        step_fail "Cannot SSH to $HOST"
    fi
    step_ok "SSH connected"

    section "2" "DEPLOY"

    step_info "Running setup on remote host..."

    ssh $SSH_KEY "$HOST" bash -s << REMOTE
set -e

# Install Rust if needed
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
fi
source \$HOME/.cargo/env
rustup toolchain install nightly-2025-07-14 --profile minimal 2>/dev/null || true

# Clone or update
if [[ -d ~/obelysk ]]; then
    cd ~/obelysk && git pull origin main 2>/dev/null || true
else
    git clone --depth 1 https://github.com/bitsage-network/stwo-ml.git ~/obelysk
fi

cd ~/obelysk

# Detect GPU
FEATURES="std,server,model-loading,safetensors,binary-proof"
if command -v nvidia-smi &>/dev/null; then
    FEATURES="\$FEATURES,gpu,cuda-runtime"
    echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

# Build
cargo +nightly-2025-07-14 build --release --bin prove-server --features "\$FEATURES" 2>&1 | tail -3

# Configure systemd
sudo tee /etc/systemd/system/obelysk-prover.service > /dev/null << SERVICE
[Unit]
Description=ObelyZK Prove Server
After=network.target

[Service]
Type=simple
User=\$USER
WorkingDirectory=\$HOME/obelysk
ExecStart=\$HOME/obelysk/target/release/prove-server
Environment=BIND_ADDR=0.0.0.0:${PORT}
Environment=RUST_LOG=info
${API_KEY:+Environment=API_KEY=${API_KEY}}
Environment=OBELYSK_MODEL_DIR=\$HOME/.obelysk/models/${MODEL}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable obelysk-prover
sudo systemctl restart obelysk-prover

echo "DEPLOYED"
REMOTE

    step_ok "Deployed to ${WHITE}${HOST}${X}"

    # Test health
    sleep 3
    if curl -sf "http://${HOST%%@*}:${PORT}/health" &>/dev/null 2>/dev/null; then
        step_ok "Remote server healthy"
    else
        step_warn "Server may still be starting — check: ${WHITE}curl http://${HOST}:${PORT}/health${X}"
    fi

    section "3" "NGINX + TLS"
    step_info "Setting up HTTPS via nginx + certbot..."

    # Extract hostname for DNS (user@host → host)
    REMOTE_HOST="${HOST#*@}"
    DOMAIN="${DOMAIN:-api.bitsage.network}"

    ssh $SSH_KEY "$HOST" bash -s << NGINX_SETUP
set -e

# Install nginx + certbot if needed
if ! command -v nginx &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq nginx certbot python3-certbot-nginx
fi

# Create nginx site config
sudo tee /etc/nginx/sites-enabled/${DOMAIN} > /dev/null << SITE
server {
    server_name ${DOMAIN};

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \\\$host;
        proxy_set_header X-Real-IP \\\$remote_addr;
        proxy_set_header X-Forwarded-For \\\$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \\\$scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \\\$http_upgrade;
        proxy_set_header Connection "upgrade";
        client_max_body_size 50M;
    }

    listen 80;
}
SITE

sudo nginx -t && sudo systemctl reload nginx

# Get TLS certificate
if [ ! -d "/etc/letsencrypt/live/${DOMAIN}" ]; then
    sudo certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --email dev@bitsage.network 2>&1 | tail -5
fi

echo "NGINX_DONE"
NGINX_SETUP

    if curl -sf "https://${DOMAIN}/health" &>/dev/null; then
        step_ok "HTTPS live at ${WHITE}https://${DOMAIN}${X}"
    else
        step_warn "HTTPS may need DNS config: ${DOMAIN} → ${REMOTE_HOST}"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# API Guide
# ═══════════════════════════════════════════════════════════════════════

BASE_URL="http://localhost:${PORT}"
[[ "$MODE" == "remote" ]] && BASE_URL="http://${HOST}:${PORT}"

echo ""
echo ""
echo -e "  ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
echo ""
echo -e "  ${EMERALD}${BOLD}${CHECK} Prove-Server Running${X}"
echo -e "  ${GHOST}${BASE_URL}${X}"
echo ""
echo -e "  ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
echo ""
echo ""
echo -e "  ${WHITE}${BOLD}Test Your API${X}"
echo ""

echo -e "  ${LIME}${ARROW}${X} ${WHITE}Health check${X}"
echo ""
echo -e "    ${LIME}curl ${BASE_URL}/health${X}"
echo ""

echo -e "  ${LIME}${ARROW}${X} ${WHITE}Load a HuggingFace model${X}"
echo ""
AUTH_HEADER=""
[[ -n "$API_KEY" ]] && AUTH_HEADER=" -H 'Authorization: Bearer ${API_KEY}'"
echo -e "    ${LIME}curl -X POST ${BASE_URL}/api/v1/models/hf \\\\${X}"
echo -e "    ${LIME}  -H 'Content-Type: application/json'${AUTH_HEADER} \\\\${X}"
echo -e "    ${LIME}  -d '{\"huggingface_id\": \"Qwen/Qwen2-0.5B\"}'${X}"
echo ""

echo -e "  ${LIME}${ARROW}${X} ${WHITE}Generate a proof${X}"
echo ""
echo -e "    ${LIME}curl -X POST ${BASE_URL}/api/v1/infer \\\\${X}"
echo -e "    ${LIME}  -H 'Content-Type: application/json'${AUTH_HEADER} \\\\${X}"
echo -e "    ${LIME}  -d '{\"model_id\": \"MODEL_ID\", \"input\": [0.1, 0.2], \"gpu\": true}'${X}"
echo ""

echo -e "  ${LIME}${ARROW}${X} ${WHITE}Use from SDK${X}"
echo ""
echo -e "    ${GHOST}Python:${X}  ${LIME}pip install obelyzk${X}"
echo -e "    ${GHOST}TS:${X}      ${LIME}npm install @obelyzk/sdk${X}"
echo -e "    ${GHOST}CLI:${X}     ${LIME}npm install -g @obelyzk/cli${X}"
echo ""

if [[ "$MODE" == "local" ]]; then
    echo -e "  ${GHOST}Stop: kill \$(cat /tmp/obelysk-prove-server.pid)${X}"
elif [[ "$MODE" == "docker" ]]; then
    echo -e "  ${GHOST}Stop: docker stop obelysk-prover${X}"
    echo -e "  ${GHOST}Logs: docker logs -f obelysk-prover${X}"
elif [[ "$MODE" == "remote" ]]; then
    echo -e "  ${GHOST}Stop: ssh ${HOST} 'sudo systemctl stop obelysk-prover'${X}"
    echo -e "  ${GHOST}Logs: ssh ${HOST} 'journalctl -u obelysk-prover -f'${X}"
fi
echo ""
