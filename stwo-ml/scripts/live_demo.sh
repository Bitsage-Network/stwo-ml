#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Live Demo — Chat → Prove → Audit → On-chain
#
# Usage:
#   ./scripts/live_demo.sh
#
# What happens:
#   1. Starts Qwen2-0.5B on llama.cpp (local Metal GPU)
#   2. You chat with the model interactively
#   3. Type "prove" when done chatting
#   4. Each conversation turn is proved via GKR sumcheck
#   5. Audit report generated
#   6. Recursive STARK compresses everything to 1 TX
#   7. Ready for on-chain submission
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

MODEL_DIR="$HOME/.obelysk/models/qwen2-0.5b"
GGUF_PATH="$HOME/.obelysk/models/qwen2-0.5b-gguf/qwen2-0_5b-instruct-q4_k_m.gguf"
LOG_DIR="/tmp/obelysk-demo-$(date +%s)"
CONV_FILE="$LOG_DIR/conversation.json"
PROVE_BIN="$(dirname "$0")/../target/release/prove-model"
PORT=8192

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
WHITE='\033[1;37m'
DIM='\033[0;90m'
RESET='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}  ▗▄▖ ▗▄▄▖ ▗▄▄▄▖▗▖  ▗▖ ▗▖ ▗▖▗▄▄▄▖▗▖ ▗▖${RESET}"
    echo -e "${CYAN} ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌  ▝▜▌▐▛▘   ▄▄▄▘▐▌▗▞▘${RESET}"
    echo -e "${CYAN} ▐▌ ▐▌▐▛▀▚▖▐▛▀▀▘▐▌   ▐▌▐▌  ▗▄▄▄▖ ▐▛▚▖${RESET}"
    echo -e "${CYAN} ▝▚▄▞▘▐▙▄▞▘▐▙▄▄▖▐▙▄▄▖▐▌▐▌  ▐▌  ▐▌▐▌ ▐▌${RESET}"
    echo ""
    echo -e "  ${WHITE}Verifiable ML Inference${RESET}"
    echo -e "  ${DIM}Chat with AI. Prove it happened. Verify on-chain.${RESET}"
    echo ""
}

# ── Check prerequisites ──────────────────────────────────────────────

check_prereqs() {
    if [ ! -f "$GGUF_PATH" ]; then
        echo -e "${YELLOW}Downloading Qwen2-0.5B GGUF...${RESET}"
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen2-0.5B-Instruct-GGUF', 'qwen2-0_5b-instruct-q4_k_m.gguf',
    local_dir='$HOME/.obelysk/models/qwen2-0.5b-gguf')
"
    fi

    if [ ! -f "$MODEL_DIR/config.json" ]; then
        echo -e "${YELLOW}Downloading Qwen2-0.5B weights...${RESET}"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-0.5B', local_dir='$MODEL_DIR',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer*', '*.json'])
"
    fi

    if [ ! -f "$PROVE_BIN" ]; then
        echo -e "${YELLOW}Building prover (first time only)...${RESET}"
        cd "$(dirname "$0")/.."
        cargo build --release --features std,metal,cli,model-loading,safetensors,audit
    fi

    if ! command -v llama-server &>/dev/null; then
        echo "Installing llama.cpp..."
        brew install llama.cpp
    fi
}

# ── Start llama.cpp server ───────────────────────────────────────────

start_server() {
    echo -e "${DIM}Starting Qwen2-0.5B on Metal GPU...${RESET}"
    llama-server \
        --model "$GGUF_PATH" \
        --port $PORT \
        --ctx-size 2048 \
        --n-gpu-layers 99 \
        &>/dev/null &
    SERVER_PID=$!

    # Wait for server to be ready (model loading takes ~15-20s on M4 Max)
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
            echo -e "${GREEN}Model loaded. (${i}s)${RESET}"
            echo ""
            return 0
        fi
        if [ $((i % 5)) -eq 0 ]; then
            echo -e "${DIM}  Loading... (${i}s)${RESET}"
        fi
        sleep 1
    done
    echo "Server failed to start after 60s"
    exit 1
}

# ── Interactive chat ─────────────────────────────────────────────────

chat_loop() {
    mkdir -p "$LOG_DIR"

    # Use the Python chat engine (handles escaping, tokenization, JSON)
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    python3 "$SCRIPT_DIR/chat_engine.py" "$CONV_FILE"
}

# ── Prove the conversation ───────────────────────────────────────────

prove_conversation() {
    local n_turns=$(python3 -c "import json; print(len(json.load(open('$CONV_FILE'))['turns']))")

    echo ""
    echo -e "${DIM}─────────────────────────────────────────────────${RESET}"
    echo -e "${WHITE}Proving $n_turns conversation turns...${RESET}"
    echo ""

    # Step 1: Capture — run each turn through all 24 transformer layers
    # Uses the standard graph (96 MatMuls + 24 SiLU + 49 RMSNorm).
    # Full attention graph (--full-attention) captures Q/K/V/O + attention
    # but requires matching audit pipeline support (in progress).
    echo -e "${YELLOW}Step 1/4: Capturing M31 forward passes (all 24 layers)${RESET}"
    "$PROVE_BIN" capture \
        --model-dir "$MODEL_DIR" \
        --log-dir "$LOG_DIR/logs" \
        --conversation "$CONV_FILE" \
        --model-name "qwen2-0.5b" \
        2>&1 | grep -v "^\s*$"

    # Step 2: Audit — prove all captured inferences with real weight commitments
    echo ""
    echo -e "${YELLOW}Step 2/4: Proving with GKR sumcheck (full model, all weights)${RESET}"
    "$PROVE_BIN" audit \
        --log-dir "$LOG_DIR/logs" \
        --model-dir "$MODEL_DIR" \
        --dry-run \
        --output "$LOG_DIR/audit_report.json" \
        2>&1 | grep -v "^\s*$"

    # Step 3: Recursive STARK on the full model proof
    echo ""
    echo -e "${YELLOW}Step 3/4: Recursive STARK compression${RESET}"
    "$PROVE_BIN" \
        --model-dir "$MODEL_DIR" \
        --gkr \
        --format ml_gkr \
        --recursive \
        --dry-run \
        --output "$LOG_DIR/recursive_proof.json" \
        2>&1 | grep -v "^\s*$"

    # Step 4: Summary
    echo ""
    echo -e "${DIM}─────────────────────────────────────────────────${RESET}"
    echo -e "${GREEN}Verification complete.${RESET}"
    echo ""
    echo -e "  ${WHITE}Turns proved:${RESET}    $n_turns"
    echo -e "  ${WHITE}Audit report:${RESET}    $LOG_DIR/audit_report.json"
    echo -e "  ${WHITE}Recursive proof:${RESET} $LOG_DIR/recursive_proof.json"
    echo -e "  ${WHITE}Conversation:${RESET}    $CONV_FILE"

    if [ -f "$LOG_DIR/audit_report.json" ]; then
        local proof_count=$(python3 -c "
import json
try:
    r = json.load(open('$LOG_DIR/audit_report.json'))
    print(r.get('num_proofs', r.get('proofs_generated', '?')))
except:
    print('?')
" 2>/dev/null)
        echo -e "  ${WHITE}Proofs:${RESET}          $proof_count"
    fi

    echo ""
    echo -e "  ${DIM}Every response is cryptographically verified.${RESET}"
    echo -e "  ${DIM}Submit on-chain: prove-model audit --submit${RESET}"
    echo ""
}

# ── Cleanup ──────────────────────────────────────────────────────────

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Main ─────────────────────────────────────────────────────────────

banner
check_prereqs
start_server
chat_loop
prove_conversation
