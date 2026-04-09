#!/usr/bin/env bash
set -euo pipefail

# ObelyZK Node Deployment Script
# Automates: Juno sync + contract declaration + prove-server restart
#
# Usage:
#   STARKNET_PRIVATE_KEY=0x... ./scripts/deploy_node.sh
#
# Optional env vars:
#   STARKNET_RPC          — Override RPC (default: local Juno at :6060)
#   SKIP_JUNO             — Set to 1 to skip Juno setup
#   SKIP_BUILD            — Set to 1 to skip cargo build
#   SKIP_DECLARE          — Set to 1 to skip contract declaration
#   RECURSIVE_CLASS_JSON  — Path to Sierra class (default: auto-build)
#   RECURSIVE_CASM_JSON   — Path to CASM class (default: auto-build)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CAIRO_DIR="$(cd "$ROOT_DIR/../elo-cairo-verifier" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy]${NC} $*"; }
err() { echo -e "${RED}[deploy]${NC} $*" >&2; }

# ── Validate prerequisites ──────────────────────────────────────────

if [ -z "${STARKNET_PRIVATE_KEY:-}" ]; then
    err "STARKNET_PRIVATE_KEY not set"
    exit 1
fi

command -v docker >/dev/null 2>&1 || { err "docker not found"; exit 1; }
command -v cargo >/dev/null 2>&1 || { err "cargo not found — install Rust"; exit 1; }

DEPLOYER_ACCOUNT="0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f"
DEPLOYER_CLASS="0x061dac032f228abef9c6626f995015233097ae253a7f72d68552db02f2971b8f"

# ── Step 1: Start Juno node ─────────────────────────────────────────

JUNO_RPC="http://localhost:6060"

if [ "${SKIP_JUNO:-0}" != "1" ]; then
    log "Starting Juno Sepolia node..."

    docker rm -f juno-sepolia 2>/dev/null || true
    docker run -d --name juno-sepolia \
        -p 6060:6060 \
        --memory=16g \
        --restart unless-stopped \
        nethermind/juno:latest \
        --network sepolia \
        --http --http-host 0.0.0.0 --http-port 6060 \
        --db-path /var/lib/juno \
        --snap-sync \
        --p2p \
        -v juno-sepolia-data:/var/lib/juno

    log "Juno started. Waiting for sync..."

    # Wait for Juno to reach a recent block
    TARGET_BLOCK=8450000
    while true; do
        CURRENT=$(curl -s "$JUNO_RPC" -X POST \
            -H 'Content-Type: application/json' \
            -d '{"jsonrpc":"2.0","id":1,"method":"starknet_blockNumber"}' 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',0))" 2>/dev/null || echo "0")

        if [ "$CURRENT" -ge "$TARGET_BLOCK" ] 2>/dev/null; then
            log "Juno synced to block $CURRENT (target: $TARGET_BLOCK)"
            break
        fi

        REMAINING=$((TARGET_BLOCK - CURRENT))
        warn "Juno at block $CURRENT / $TARGET_BLOCK ($REMAINING remaining)..."
        sleep 60
    done
else
    log "Skipping Juno setup (SKIP_JUNO=1)"
    JUNO_RPC="${STARKNET_RPC:-https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo}"
fi

# ── Step 2: Build prove-model + prove-server ────────────────────────

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    log "Building prove-model and prove-server..."
    cd "$ROOT_DIR"

    FEATURES="std,gpu,cuda-runtime,model-loading,safetensors,cli"
    if command -v nvcc >/dev/null 2>&1; then
        log "CUDA detected"
    else
        FEATURES="std,model-loading,safetensors,cli"
        warn "No CUDA — building CPU-only"
    fi

    cargo +nightly-2025-07-14 build --release \
        --bin prove-model --bin prove-server \
        --features "$FEATURES" 2>&1 | tail -5

    log "Build complete"
else
    log "Skipping build (SKIP_BUILD=1)"
fi

# ── Step 3: Build Cairo contract ────────────────────────────────────

SIERRA="${RECURSIVE_CLASS_JSON:-}"
CASM="${RECURSIVE_CASM_JSON:-}"

if [ -z "$SIERRA" ] && [ -d "$CAIRO_DIR" ]; then
    log "Building Cairo recursive verifier..."
    cd "$CAIRO_DIR"

    if command -v scarb >/dev/null 2>&1; then
        scarb build 2>&1 | tail -3
        SIERRA="$CAIRO_DIR/target/dev/elo_cairo_verifier_RecursiveVerifierContract.contract_class.json"
        CASM="$CAIRO_DIR/target/dev/elo_cairo_verifier_RecursiveVerifierContract.compiled_contract_class.json"
        log "Cairo build complete: $(du -h "$SIERRA" | cut -f1)"
    else
        warn "scarb not found — skipping Cairo build"
    fi
fi

# ── Step 4: Declare + Deploy contract ───────────────────────────────

if [ "${SKIP_DECLARE:-0}" != "1" ] && [ -n "$SIERRA" ] && [ -f "$SIERRA" ]; then
    log "Declaring recursive verifier on Sepolia via $JUNO_RPC..."

    # Install starkli if needed
    if ! command -v starkli >/dev/null 2>&1; then
        if [ -f "$HOME/.starkli/bin/starkli" ]; then
            export PATH="$HOME/.starkli/bin:$PATH"
        else
            warn "starkli not found — installing..."
            curl -L https://raw.githubusercontent.com/foundry-rs/starknet-foundry/master/scripts/install.sh | sh
            source "$HOME/.starkli/env" 2>/dev/null || true
            starkliup
        fi
    fi

    # Create account descriptor
    ACCOUNT_FILE=$(mktemp /tmp/starkli_account_XXXX.json)
    cat > "$ACCOUNT_FILE" << ACCEOF
{
  "version": 1,
  "variant": {
    "type": "open_zeppelin",
    "version": 1,
    "public_key": "0x1b4fc4a44546eecebc2339b6e6c5ddcbd3f3cdbc5fa23eab2bd800d831a1b2d"
  },
  "deployment": {
    "status": "deployed",
    "class_hash": "$DEPLOYER_CLASS",
    "address": "$DEPLOYER_ACCOUNT"
  }
}
ACCEOF

    # Declare
    DECLARE_OUTPUT=$(starkli declare \
        --rpc "$JUNO_RPC" \
        --account "$ACCOUNT_FILE" \
        --private-key "$STARKNET_PRIVATE_KEY" \
        --l2-gas 500000000 --l2-gas-price 0.000000012 \
        --l1-data-gas 500000 --l1-data-gas-price 0.000000035 \
        "$SIERRA" \
        --casm-file "$CASM" 2>&1) || true

    CLASS_HASH=$(echo "$DECLARE_OUTPUT" | grep -o '0x[0-9a-f]\{60,66\}' | head -1)

    if [ -n "$CLASS_HASH" ]; then
        log "Declared class: $CLASS_HASH"

        # Deploy
        log "Deploying contract..."
        DEPLOY_OUTPUT=$(starkli deploy \
            --rpc "$JUNO_RPC" \
            --account "$ACCOUNT_FILE" \
            --private-key "$STARKNET_PRIVATE_KEY" \
            "$CLASS_HASH" "$DEPLOYER_ACCOUNT" 2>&1) || true

        CONTRACT=$(echo "$DEPLOY_OUTPUT" | grep -o '0x[0-9a-f]\{60,66\}' | tail -1)

        if [ -n "$CONTRACT" ]; then
            log "Deployed contract: $CONTRACT"
            export RECURSIVE_CONTRACT="$CONTRACT"
        else
            warn "Deploy may have failed: $DEPLOY_OUTPUT"
        fi
    else
        if echo "$DECLARE_OUTPUT" | grep -q "already declared"; then
            log "Class already declared"
        else
            warn "Declare output: $DECLARE_OUTPUT"
        fi
    fi

    rm -f "$ACCOUNT_FILE"
else
    log "Skipping contract declaration"
fi

# ── Step 5: Restart prove-server ────────────────────────────────────

log "Restarting prove-server..."

# Kill existing prove-server
pkill -f 'prove-server' 2>/dev/null || true
sleep 2

cd "$ROOT_DIR"

# On-chain compatible env vars — all soundness gates CLOSED (verified Apr 7 2026)
# All 5 gates verified safe to close on A10G with recursive STARK path.
# The streaming calldata path still has a MATMUL_FINAL_MISMATCH but recursive is unaffected.
export STWO_PIECEWISE_ACTIVATION=1
export STWO_AGGREGATED_FULL_BINDING=1
export STWO_PURE_GKR_SKIP_UNIFIED_STARK=1
export STARKNET_RPC="${STARKNET_RPC:-$JUNO_RPC}"
export RECURSIVE_CONTRACT="${RECURSIVE_CONTRACT:-0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7}"

# Start prove-server in background
nohup ./target/release/prove-server \
    --port 8080 \
    --model-dir "${MODEL_DIR:-$HOME/.obelysk/models/smollm2-135m}" \
    > /var/log/prove-server.log 2>&1 &

PROVE_PID=$!
sleep 3

if kill -0 "$PROVE_PID" 2>/dev/null; then
    log "prove-server running (PID $PROVE_PID)"
else
    err "prove-server failed to start — check /var/log/prove-server.log"
    exit 1
fi

# ── Step 6: Health check ────────────────────────────────────────────

log "Running health check..."
sleep 5

HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo "failed")
if echo "$HEALTH" | grep -q "healthy\|ok\|ready"; then
    log "prove-server healthy"
else
    warn "Health check: $HEALTH"
fi

# ── Summary ─────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  ObelyZK Node Deployment Complete"
echo "================================================================"
echo "  Juno RPC:        $JUNO_RPC"
echo "  prove-server:    http://localhost:8080"
echo "  Contract:        ${RECURSIVE_CONTRACT}"
echo "  Deployer:        $DEPLOYER_ACCOUNT"
echo ""
echo "  Test:"
echo "    curl http://localhost:8080/health"
echo ""
echo "    STARKNET_PRIVATE_KEY=\$KEY prove-model \\"
echo "      --model-dir ~/.obelysk/models/smollm2-135m \\"
echo "      --gkr --format ml_gkr --recursive --on-chain"
echo "================================================================"
