#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 2b: Capture Inference Log
# ═══════════════════════════════════════════════════════════════════════
#
# Runs N forward passes through the prover's execute_forward_pass()
# and records each via the CaptureHook. This produces a chain-linked
# inference log (meta.json + log.jsonl + matrices.bin) that the audit
# pipeline can verify.
#
# This is the MANDATORY bridge between model download and audit:
# it ensures real inference through the prover code path exists before
# anyone attempts to prove or audit.
#
# Usage:
#   bash scripts/pipeline/02b_capture_inference.sh
#   bash scripts/pipeline/02b_capture_inference.sh --model-dir ~/models/phi3-mini --count 5
#   bash scripts/pipeline/02b_capture_inference.sh --model-name qwen3-14b --layers 1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────

MODEL_NAME=""
MODEL_DIR=""
NUM_LAYERS=""
CAPTURE_COUNT=3
MODEL_ID="0x1"
INPUT_FILE=""
SKIP_COMMITMENT=true   # Legacy packed commitment not used by GKR pipeline; use --no-skip-commitment to force
SKIP_BUILD=false
LOG_DIR_OVERRIDE=""
CAPTURE_TIMEOUT_SEC="${CAPTURE_TIMEOUT_SEC:-3600}"

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)       MODEL_NAME="$2"; shift 2 ;;
        --model-dir)        MODEL_DIR="$2"; shift 2 ;;
        --layers)           NUM_LAYERS="$2"; shift 2 ;;
        --count)            CAPTURE_COUNT="$2"; shift 2 ;;
        --model-id)         MODEL_ID="$2"; shift 2 ;;
        --input)            INPUT_FILE="$2"; shift 2 ;;
        --skip-commitment)  SKIP_COMMITMENT=true; shift ;;
        --no-skip-commitment) SKIP_COMMITMENT=false; shift ;;
        --skip-build)       SKIP_BUILD=true; shift ;;
        --log-dir)          LOG_DIR_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Capture inference log via prover forward pass."
            echo ""
            echo "Model source (pick one):"
            echo "  --model-name NAME   Load from ~/.obelysk/models/NAME/"
            echo "  --model-dir DIR     Path to model directory"
            echo ""
            echo "Options:"
            echo "  --layers N           Number of transformer layers (default: from config)"
            echo "  --count N            Number of forward passes to capture (default: 3)"
            echo "  --model-id ID        Model ID for log metadata (default: 0x1)"
            echo "  --input FILE         JSON input file (default: diverse generated inputs)"
            echo "  --skip-commitment    Skip weight commitment (faster, weaker audit)"
            echo "  --skip-build         Skip rebuilding prove-model"
            echo "  --log-dir DIR        Override log directory"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

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

# Optional rebuild with audit feature
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

# ─── Display Config ──────────────────────────────────────────────────

banner
echo -e "${BOLD}  Inference Capture${NC}"
echo ""
log "Model:          ${MODEL_NAME:-$(basename "$MODEL_DIR")}"
log "Model dir:      ${MODEL_DIR}"
log "Layers:         ${MODEL_LAYERS:-all}"
log "Count:          ${CAPTURE_COUNT}"
log "Log dir:        ${LOG_DIR}"
log "prove-model:    ${PROVE_BIN}"
echo ""

timer_start "capture"

# ─── Build Capture Command ───────────────────────────────────────────

CAPTURE_CMD=("${PROVE_BIN}" "capture")
CAPTURE_CMD+=("--model-dir" "$MODEL_DIR")
CAPTURE_CMD+=("--log-dir" "$LOG_DIR")
CAPTURE_CMD+=("--count" "$CAPTURE_COUNT")
CAPTURE_CMD+=("--model-id" "$MODEL_ID")
CAPTURE_CMD+=("--model-name" "${MODEL_NAME:-$(basename "$MODEL_DIR")}")

if [[ -n "${MODEL_LAYERS:-}" ]]; then
    CAPTURE_CMD+=("--layers" "$MODEL_LAYERS")
fi

if [[ -n "$INPUT_FILE" ]]; then
    CAPTURE_CMD+=("--input" "$INPUT_FILE")
fi

if [[ "$SKIP_COMMITMENT" == "true" ]]; then
    CAPTURE_CMD+=("--skip-commitment")
fi

log "Command: ${CAPTURE_CMD[*]}"
log "Capture timeout: ${CAPTURE_TIMEOUT_SEC}s"
echo ""

# ─── Run Capture ─────────────────────────────────────────────────────

step "2b.1" "Running inference capture..."
CAPTURE_RUN_LOG="/tmp/obelysk_capture_$(date +%s).log"
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
    err "Inference capture failed"
    tail -n 200 "$CAPTURE_RUN_LOG" 2>/dev/null || true
    exit 1
fi

# Parse machine-readable lines
CAPTURED_LOG_DIR=$(grep '^CAPTURE_LOG_DIR=' "$CAPTURE_RUN_LOG" | head -1 | cut -d= -f2-)
CAPTURED_COUNT=$(grep '^CAPTURE_COUNT=' "$CAPTURE_RUN_LOG" | head -1 | cut -d= -f2-)
CAPTURED_MODEL=$(grep '^CAPTURE_MODEL=' "$CAPTURE_RUN_LOG" | head -1 | cut -d= -f2-)
rm -f "$CAPTURE_RUN_LOG" 2>/dev/null || true

# ─── Verify Output ───────────────────────────────────────────────────

step "2b.2" "Verifying capture output..."

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

ELAPSED=$(timer_elapsed "capture")

save_state "capture_state.env" \
    "CAPTURE_COMPLETED=true" \
    "AUDIT_LOG_DIR=${LOG_DIR}" \
    "CAPTURE_COUNT=${CAPTURED_COUNT:-${LOG_LINES}}" \
    "CAPTURE_MODEL=${CAPTURED_MODEL:-${MODEL_NAME:-$(basename "$MODEL_DIR")}}" \
    "CAPTURE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "CAPTURE_DURATION_SEC=${ELAPSED}"

# ─── Summary ─────────────────────────────────────────────────────────

echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  INFERENCE CAPTURE COMPLETE                           ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Model:       %-36s ║\n" "${MODEL_NAME:-$(basename "$MODEL_DIR")}"
printf "  ║  Entries:     %-36s ║\n" "${CAPTURED_COUNT:-${LOG_LINES}}"
printf "  ║  Log dir:     %-36s ║\n" "${LOG_DIR}"
printf "  ║  Duration:    %-36s ║\n" "$(format_duration $ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next: ./03_prove.sh                                 ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
