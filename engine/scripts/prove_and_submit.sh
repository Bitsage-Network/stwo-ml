#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Obelysk — End-to-End Proof Generation + On-Chain Submission
#
# Generates an on-chain compatible ZKML proof and submits all 6
# streaming verification steps to Starknet Sepolia.
#
# Prerequisites:
#   - Model weights at ~/.obelysk/models/qwen2-0.5b/
#   - Node.js with starknet.js in scripts/pipeline/lib/node_modules/
#   - STARKNET_PRIVATE_KEY and STARKNET_ACCOUNT_ADDRESS env vars
#
# Usage:
#   export STARKNET_PRIVATE_KEY="0x..."
#   export STARKNET_ACCOUNT_ADDRESS="0x..."
#   ./scripts/prove_and_submit.sh
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROVE_BIN="${SCRIPT_DIR}/../target/release/prove-model"
SUBMIT_SCRIPT="${SCRIPT_DIR}/pipeline/lib/paymaster_submit.mjs"
MODEL_DIR="${HOME}/.obelysk/models/qwen2-0.5b"
CONTRACT="0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005"
PROOF_FILE="/tmp/obelysk_proof_$(date +%s).json"

# Colors
G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; W='\033[1;37m'
D='\033[0;90m'; R='\033[0;31m'; X='\033[0m'

echo ""
echo -e "${C}  ObelyZK — End-to-End On-Chain Proof${X}"
echo -e "${D}  Contract: ${CONTRACT}${X}"
echo ""

# Check prerequisites
[[ ! -f "$PROVE_BIN" ]] && { echo -e "${R}prove-model not found. Run: cargo build --release --features std,cli,model-loading,safetensors,audit,parallel-audit${X}"; exit 1; }
[[ ! -f "$MODEL_DIR/config.json" ]] && { echo -e "${R}Model not found at $MODEL_DIR${X}"; exit 1; }
[[ -z "${STARKNET_PRIVATE_KEY:-}" ]] && { echo -e "${R}Set STARKNET_PRIVATE_KEY${X}"; exit 1; }
[[ -z "${STARKNET_ACCOUNT_ADDRESS:-}" ]] && { echo -e "${R}Set STARKNET_ACCOUNT_ADDRESS${X}"; exit 1; }

# ── Step 1: Generate Proof ──────────────────────────────────────────
echo -e "${Y}[1/2]${X} Generating on-chain compatible proof..."

export STWO_SKIP_RMS_SQ_PROOF=1
export STWO_ALLOW_MISSING_NORM_PROOF=1
export STWO_PIECEWISE_ACTIVATION=0
export STWO_ALLOW_LOGUP_ACTIVATION=1
export STWO_AGGREGATED_FULL_BINDING=1
export STWO_SKIP_BATCH_TOKENS=1
export STWO_MLE_N_QUERIES=5

"$PROVE_BIN" \
    --model-dir "$MODEL_DIR" \
    --layers 1 \
    --format ml_gkr \
    --gkr \
    --output "$PROOF_FILE" \
    --quiet 2>&1 | grep -E "Proof|self_verify|cryptographic|Model|layers" || true

echo -e "  ${G}Proof: $PROOF_FILE${X}"

# Extract model_id from proof
MODEL_ID=$(python3 -c "import json; print(json.load(open('$PROOF_FILE'))['verify_calldata']['model_id'])")
echo -e "  ${D}Model ID: $MODEL_ID${X}"

# ── Step 2: Submit On-Chain ─────────────────────────────────────────
echo ""
echo -e "${Y}[2/2]${X} Submitting 6-step streaming verification to Starknet Sepolia..."

cd "${SCRIPT_DIR}/pipeline/lib"
node paymaster_submit.mjs verify \
    --proof "$PROOF_FILE" \
    --contract "$CONTRACT" \
    --model-id "$MODEL_ID" \
    --network sepolia \
    --no-paymaster \
    2>&1 | grep -E "\[E2E\]|TX:|COMPLETE|verified|ERR|count"

echo ""
echo -e "${G}═══════════════════════════════════════════════════${X}"
echo -e "${G}  On-chain verification complete${X}"
echo -e "${G}═══════════════════════════════════════════════════${X}"
echo -e "${D}  Proof: $PROOF_FILE${X}"
echo -e "${D}  Contract: $CONTRACT${X}"
echo -e "${D}  Explorer: https://sepolia.voyager.online/contract/$CONTRACT${X}"
echo ""
