#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 2c: Multi-Turn Conversation Capture
# ═══════════════════════════════════════════════════════════════════════
#
# Generates a multi-turn conversation via HuggingFace float inference,
# then proves each turn through the M31 forward pass.
#
# Two-phase approach:
#   Phase 1: Python (float16) generates real text responses
#   Phase 2: Rust (M31) proves each turn's forward pass
#
# Usage:
#   bash scripts/pipeline/02c_conversation.sh --topic "quantum computing" --turns 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────

MODEL_NAME=""
MODEL_DIR=""
NUM_LAYERS=""
TOPIC=""
TURNS=3
TEMPERATURE=0.7
MAX_TOKENS=512
MODEL_ID="0x1"
SKIP_COMMITMENT=true
SKIP_BUILD=false
LOG_DIR_OVERRIDE=""
CONVERSATION_FILE=""
CAPTURE_TIMEOUT_SEC="${CAPTURE_TIMEOUT_SEC:-3600}"

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)       MODEL_NAME="$2"; shift 2 ;;
        --model-dir)        MODEL_DIR="$2"; shift 2 ;;
        --layers)           NUM_LAYERS="$2"; shift 2 ;;
        --topic)            TOPIC="$2"; shift 2 ;;
        --turns)            TURNS="$2"; shift 2 ;;
        --temperature)      TEMPERATURE="$2"; shift 2 ;;
        --max-tokens)       MAX_TOKENS="$2"; shift 2 ;;
        --model-id)         MODEL_ID="$2"; shift 2 ;;
        --skip-commitment)  SKIP_COMMITMENT=true; shift ;;
        --no-skip-commitment) SKIP_COMMITMENT=false; shift ;;
        --skip-build)       SKIP_BUILD=true; shift ;;
        --log-dir)          LOG_DIR_OVERRIDE="$2"; shift 2 ;;
        --conversation-file) CONVERSATION_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Generate a multi-turn conversation and prove each turn."
            echo ""
            echo "Model source (pick one):"
            echo "  --model-name NAME   Load from ~/.obelysk/models/NAME/"
            echo "  --model-dir DIR     Path to model directory"
            echo ""
            echo "Conversation:"
            echo "  --topic TEXT         Conversation topic (required unless --conversation-file)"
            echo "  --turns N            Number of Q&A turns (default: 3)"
            echo "  --temperature F      Sampling temperature (default: 0.7)"
            echo "  --max-tokens N       Max new tokens per response (default: 512)"
            echo "  --conversation-file  Skip generation, use existing conversation.json"
            echo ""
            echo "Options:"
            echo "  --layers N           Number of transformer layers"
            echo "  --model-id ID        Model ID for log metadata (default: 0x1)"
            echo "  --skip-commitment    Skip weight commitment (faster)"
            echo "  --skip-build         Skip rebuilding prove-model"
            echo "  --log-dir DIR        Override log directory"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────

if [[ -z "$CONVERSATION_FILE" ]] && [[ -z "$TOPIC" ]]; then
    err "Specify --topic or --conversation-file"
    exit 1
fi

# ─── Resolve Model ───────────────────────────────────────────────────

init_obelysk_dir

if [[ -z "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_NAME=$(get_state "model_state.env" "CURRENT_MODEL" 2>/dev/null || echo "")
    if [[ -z "$MODEL_NAME" ]]; then
        err "No model specified. Use --model-name or --model-dir, or run 01_setup_model.sh first."
        exit 1
    fi
fi

if [[ -n "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_CONFIG="${OBELYSK_DIR}/models/${MODEL_NAME}/config.env"
    if [[ -f "$MODEL_CONFIG" ]]; then
        set -a; source "$MODEL_CONFIG"; set +a
    else
        err "Config not found: ${MODEL_CONFIG}"
        exit 1
    fi
fi

[[ -n "$NUM_LAYERS" ]] && MODEL_LAYERS="$NUM_LAYERS"

check_dir "$MODEL_DIR" "Model directory not found" || exit 1

# ─── Log Directory ───────────────────────────────────────────────────

if [[ -n "$LOG_DIR_OVERRIDE" ]]; then
    LOG_DIR="$LOG_DIR_OVERRIDE"
else
    LOG_DIR="${OBELYSK_DIR}/logs/${MODEL_NAME:-$(basename "$MODEL_DIR")}"
fi
mkdir -p "$LOG_DIR"

# ─── Display Config ──────────────────────────────────────────────────

banner
echo -e "${BOLD}  Multi-Turn Conversation Capture${NC}"
echo ""
log "Model:          ${MODEL_NAME:-$(basename "$MODEL_DIR")}"
log "Model dir:      ${MODEL_DIR}"
log "Layers:         ${MODEL_LAYERS:-all}"
if [[ -n "$CONVERSATION_FILE" ]]; then
    log "Conversation:   ${CONVERSATION_FILE} (pre-generated)"
else
    log "Topic:          ${TOPIC}"
    log "Turns:          ${TURNS}"
    log "Temperature:    ${TEMPERATURE}"
    log "Max tokens:     ${MAX_TOKENS}"
fi
log "Log dir:        ${LOG_DIR}"
echo ""

timer_start "conversation"

# ═════════════════════════════════════════════════════════════════════
# Phase 1: Generate conversation (Python float inference)
# ═════════════════════════════════════════════════════════════════════

if [[ -z "$CONVERSATION_FILE" ]]; then
    step "2c.1" "Generating conversation (float inference)..."

    # Check Python + torch
    if ! command -v python3 &>/dev/null; then
        err "python3 not found"
        exit 1
    fi

    CONV_OUTPUT="/tmp/obelysk_conversation_$(date +%s).json"
    GEN_SCRIPT="${SCRIPT_DIR}/lib/generate_conversation.py"

    if [[ ! -f "$GEN_SCRIPT" ]]; then
        err "generate_conversation.py not found at ${GEN_SCRIPT}"
        exit 1
    fi

    GEN_CMD=(python3 "$GEN_SCRIPT"
        --model-dir "$MODEL_DIR"
        --topic "$TOPIC"
        --turns "$TURNS"
        --temperature "$TEMPERATURE"
        --max-tokens "$MAX_TOKENS"
        --output "$CONV_OUTPUT"
    )

    log "Command: ${GEN_CMD[*]}"
    echo ""

    # stderr streams to terminal in real time (progress), stdout captured for parsing
    GEN_STDOUT="/tmp/obelysk_gen_conv_stdout_$(date +%s).log"
    set +e
    "${GEN_CMD[@]}" >"$GEN_STDOUT"
    GEN_RC=$?
    set -e
    if (( GEN_RC == 0 )); then
        CONVERSATION_FILE=$(grep '^CONVERSATION_FILE=' "$GEN_STDOUT" | head -1 | cut -d= -f2-)
        CONV_ID=$(grep '^CONVERSATION_ID=' "$GEN_STDOUT" | head -1 | cut -d= -f2-)
        CONV_TURNS=$(grep '^CONVERSATION_TURNS=' "$GEN_STDOUT" | head -1 | cut -d= -f2-)
        rm -f "$GEN_STDOUT" 2>/dev/null || true
        ok "Conversation generated: ${CONV_TURNS} turns (id: ${CONV_ID})"
    else
        err "Conversation generation failed (exit code: ${GEN_RC})"
        cat "$GEN_STDOUT" >&2 || true
        rm -f "$GEN_STDOUT" 2>/dev/null || true
        exit 1
    fi

    if [[ -z "$CONVERSATION_FILE" ]]; then
        CONVERSATION_FILE="$CONV_OUTPUT"
    fi
else
    log "Using pre-generated conversation: ${CONVERSATION_FILE}"
    CONV_ID=$(python3 -c "import json; print(json.load(open('${CONVERSATION_FILE}'))['conversation_id'])" 2>/dev/null || echo "unknown")
    CONV_TURNS=$(python3 -c "import json; print(len(json.load(open('${CONVERSATION_FILE}'))['turns']))" 2>/dev/null || echo "?")
fi

if [[ ! -f "$CONVERSATION_FILE" ]]; then
    err "Conversation file not found: ${CONVERSATION_FILE}"
    exit 1
fi

echo ""
ok "Conversation: ${CONVERSATION_FILE} (${CONV_TURNS} turns)"

# ═════════════════════════════════════════════════════════════════════
# Phase 2: Prove conversation (Rust M31 forward pass)
# ═════════════════════════════════════════════════════════════════════

step "2c.2" "Proving conversation (M31 forward pass)..."

# ─── Find / Build Binary ────────────────────────────────────────────

INSTALL_DIR=$(get_state "setup_state.env" "INSTALL_DIR" 2>/dev/null || echo "$HOME/obelysk")
LIBS_DIR="${INSTALL_DIR}"

PROVE_BIN=$(get_state "setup_state.env" "PROVE_MODEL_BIN" 2>/dev/null || echo "")
if [[ -z "$PROVE_BIN" ]] || [[ ! -f "$PROVE_BIN" ]]; then
    PROVE_BIN=$(find_binary "prove-model" "$LIBS_DIR" 2>/dev/null || echo "")
fi
if [[ -z "$PROVE_BIN" ]] || [[ ! -f "$PROVE_BIN" ]]; then
    PROVE_BIN=$(command -v prove-model 2>/dev/null || echo "")
fi

if [[ "$SKIP_BUILD" == "false" ]] && [[ -d "${LIBS_DIR}/stwo-ml" ]]; then
    FEATURES="cli,audit,model-loading,safetensors"
    if command -v nvcc &>/dev/null || [[ -f /usr/local/cuda/bin/nvcc ]]; then
        FEATURES="cli,audit,model-loading,safetensors,cuda-runtime"
    fi
    log "Rebuilding prove-model (features: ${FEATURES})..."
    if (export PATH="$HOME/.cargo/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"; cd "${LIBS_DIR}/stwo-ml" && cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -3); then
        PROVE_BIN=$(find_binary "prove-model" "$LIBS_DIR" 2>/dev/null || echo "$PROVE_BIN")
        ok "prove-model rebuilt (features: ${FEATURES})"
    else
        warn "Rebuild failed, using existing binary"
    fi
fi

if [[ -z "$PROVE_BIN" ]]; then
    err "prove-model not found. Run 00_setup_gpu.sh first."
    exit 1
fi

# ─── Build Capture Command ──────────────────────────────────────────

CAPTURE_CMD=("${PROVE_BIN}" "capture")
CAPTURE_CMD+=("--model-dir" "$MODEL_DIR")
CAPTURE_CMD+=("--log-dir" "$LOG_DIR")
CAPTURE_CMD+=("--conversation" "$CONVERSATION_FILE")
CAPTURE_CMD+=("--model-id" "$MODEL_ID")
CAPTURE_CMD+=("--model-name" "${MODEL_NAME:-$(basename "$MODEL_DIR")}")

if [[ -n "${MODEL_LAYERS:-}" ]]; then
    CAPTURE_CMD+=("--layers" "$MODEL_LAYERS")
fi

if [[ "$SKIP_COMMITMENT" == "true" ]]; then
    CAPTURE_CMD+=("--skip-commitment")
fi

log "Command: ${CAPTURE_CMD[*]}"
log "Capture timeout: ${CAPTURE_TIMEOUT_SEC}s"
echo ""

# ─── Run Capture ─────────────────────────────────────────────────────

CAPTURE_RUN_LOG="/tmp/obelysk_conv_capture_$(date +%s).log"
: > "$CAPTURE_RUN_LOG"

"${CAPTURE_CMD[@]}" >"$CAPTURE_RUN_LOG" 2>&1 &
CAPTURE_PID=$!
tail -n +1 -f "$CAPTURE_RUN_LOG" &
TAIL_PID=$!
CAPTURE_STARTED_AT=$(date +%s)

_stop_capture_process() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
}

trap '_stop_capture_process "$CAPTURE_PID"; kill "$TAIL_PID" 2>/dev/null || true; trap - INT TERM; exit 130' INT TERM

while kill -0 "$CAPTURE_PID" 2>/dev/null; do
    sleep 15
    if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then
        break
    fi
    ELAPSED_CAPTURE=$(( $(date +%s) - CAPTURE_STARTED_AT ))
    log "Capture still running... ${ELAPSED_CAPTURE}s elapsed"
    if (( ELAPSED_CAPTURE >= CAPTURE_TIMEOUT_SEC )); then
        warn "Capture timeout (${CAPTURE_TIMEOUT_SEC}s) reached, terminating..."
        _stop_capture_process "$CAPTURE_PID"
        break
    fi
done

wait "$CAPTURE_PID" 2>/dev/null
CAPTURE_RC=$?
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true
trap - INT TERM

if (( CAPTURE_RC != 0 )); then
    err "Conversation capture failed"
    tail -n 200 "$CAPTURE_RUN_LOG" 2>/dev/null || true
    exit 1
fi

# Parse machine-readable lines
CAPTURED_LOG_DIR=$(grep '^CAPTURE_LOG_DIR=' "$CAPTURE_RUN_LOG" | head -1 | cut -d= -f2-)
CAPTURED_COUNT=$(grep '^CAPTURE_COUNT=' "$CAPTURE_RUN_LOG" | head -1 | cut -d= -f2-)
CAPTURED_MODEL=$(grep '^CAPTURE_MODEL=' "$CAPTURE_RUN_LOG" | head -1 | cut -d= -f2-)
rm -f "$CAPTURE_RUN_LOG" 2>/dev/null || true

# ─── Verify Output ───────────────────────────────────────────────────

step "2c.3" "Verifying capture output..."

if [[ ! -f "${LOG_DIR}/meta.json" ]]; then
    err "meta.json not found in ${LOG_DIR}"
    exit 1
fi
ok "meta.json exists"

if [[ ! -f "${LOG_DIR}/log.jsonl" ]]; then
    err "log.jsonl not found in ${LOG_DIR}"
    exit 1
fi

LOG_LINES=$(wc -l < "${LOG_DIR}/log.jsonl" | tr -d ' ')
if (( LOG_LINES < 1 )); then
    err "log.jsonl is empty"
    exit 1
fi
ok "log.jsonl has ${LOG_LINES} entries"

if [[ ! -f "${LOG_DIR}/matrices.bin" ]]; then
    err "matrices.bin not found in ${LOG_DIR}"
    exit 1
fi
ok "matrices.bin exists ($(du -h "${LOG_DIR}/matrices.bin" | cut -f1))"

echo ""

# ─── Save State ──────────────────────────────────────────────────────

ELAPSED=$(timer_elapsed "conversation")

save_state "capture_state.env" \
    "CAPTURE_COMPLETED=true" \
    "AUDIT_LOG_DIR=${LOG_DIR}" \
    "CAPTURE_COUNT=${CAPTURED_COUNT:-${LOG_LINES}}" \
    "CAPTURE_MODEL=${CAPTURED_MODEL:-${MODEL_NAME:-$(basename "$MODEL_DIR")}}" \
    "CAPTURE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "CAPTURE_DURATION_SEC=${ELAPSED}" \
    "CAPTURE_MODE=conversation" \
    "CONVERSATION_ID=${CONV_ID:-}" \
    "CONVERSATION_TOPIC=${TOPIC:-}" \
    "CONVERSATION_TURNS=${CONV_TURNS:-}"

save_state "conversation_state.env" \
    "CONVERSATION_COMPLETED=true" \
    "CONVERSATION_FILE=${CONVERSATION_FILE}" \
    "CONVERSATION_ID=${CONV_ID:-}" \
    "CONVERSATION_TURNS=${CONV_TURNS:-}" \
    "CONVERSATION_TOPIC=${TOPIC:-}" \
    "CONVERSATION_LOG_DIR=${LOG_DIR}" \
    "CONVERSATION_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "CONVERSATION_DURATION_SEC=${ELAPSED}"

# ─── Summary ─────────────────────────────────────────────────────────

echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  CONVERSATION CAPTURE COMPLETE                       ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Model:       %-36s ║\n" "${MODEL_NAME:-$(basename "$MODEL_DIR")}"
printf "  ║  Topic:       %-36s ║\n" "${TOPIC:0:36}"
printf "  ║  Conv ID:     %-36s ║\n" "${CONV_ID:-unknown}"
printf "  ║  Turns:       %-36s ║\n" "${CONV_TURNS:-?}"
printf "  ║  Log entries: %-36s ║\n" "${CAPTURED_COUNT:-${LOG_LINES}}"
printf "  ║  Log dir:     %-36s ║\n" "${LOG_DIR}"
printf "  ║  Duration:    %-36s ║\n" "$(format_duration $ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next: ./03_prove.sh                                 ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
