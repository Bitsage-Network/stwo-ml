#!/usr/bin/env bash
set -euo pipefail

# ─── Obelysk E2E Pipeline ──────────────────────────────────────────
# Captures inference logs and runs a full audit with ZK proof generation.
#
# Usage:
#   ./run_e2e.sh --model-dir /path/to/qwen3-14b --layers 5 --gpu
#   ./run_e2e.sh --preset qwen3-14b --gpu --submit
#   ./run_e2e.sh --preset qwen3-14b --gpu --dry-run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_DIR/target/release/prove-model"

# ─── Defaults ────────────────────────────────────────────────────────
MODEL_DIR=""
LAYERS=""
PRESET=""
GPU_FLAG=""
DRY_RUN=""
SUBMIT=""
COUNT=3
OUTPUT_DIR="$HOME/.obelysk/audits"

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --preset NAME       Model preset (qwen3-14b, phi3-mini, llama3-8b)"
  echo "  --model-dir PATH    Path to HuggingFace model directory"
  echo "  --layers N          Number of transformer layers to prove"
  echo "  --gpu               Use GPU acceleration"
  echo "  --count N           Number of inferences to capture (default: 3)"
  echo "  --submit            Submit proof on-chain after audit"
  echo "  --dry-run           Prove + report, skip on-chain submission"
  echo "  --output-dir PATH   Where to save reports (default: ~/.obelysk/audits)"
  echo "  -h, --help          Show this help"
  exit 0
}

# ─── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)     PRESET="$2"; shift 2 ;;
    --model-dir)  MODEL_DIR="$2"; shift 2 ;;
    --layers)     LAYERS="$2"; shift 2 ;;
    --gpu)        GPU_FLAG="--gpu"; shift ;;
    --count)      COUNT="$2"; shift 2 ;;
    --submit)     SUBMIT="--submit"; shift ;;
    --dry-run)    DRY_RUN="--dry-run"; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            echo "Unknown option: $1"; usage ;;
  esac
done

# ─── Resolve preset ─────────────────────────────────────────────────
if [[ -n "$PRESET" ]]; then
  case "$PRESET" in
    qwen3-14b)
      MODEL_DIR="${MODEL_DIR:-$HOME/.obelysk/models/qwen3-14b}"
      LAYERS="${LAYERS:-5}"
      ;;
    phi3-mini)
      MODEL_DIR="${MODEL_DIR:-$HOME/.obelysk/models/phi3-mini}"
      LAYERS="${LAYERS:-2}"
      ;;
    llama3-8b)
      MODEL_DIR="${MODEL_DIR:-$HOME/.obelysk/models/llama3-8b}"
      LAYERS="${LAYERS:-5}"
      ;;
    *)
      echo "Error: unknown preset '$PRESET'"
      echo "Available: qwen3-14b, phi3-mini, llama3-8b"
      exit 1
      ;;
  esac
fi

if [[ -z "$MODEL_DIR" ]]; then
  echo "Error: specify --model-dir or --preset"
  usage
fi

if [[ -z "$LAYERS" ]]; then
  echo "Error: specify --layers N"
  exit 1
fi

# ─── Check binary ───────────────────────────────────────────────────
if [[ ! -f "$BINARY" ]]; then
  echo "Binary not found at $BINARY"
  echo "Building..."
  FEATURES="std,gpu,onnx,safetensors,model-loading,cli,audit"
  if command -v nvidia-smi &>/dev/null; then
    FEATURES="$FEATURES,cuda-runtime"
  fi
  (cd "$REPO_DIR" && cargo build --release --features "$FEATURES")
fi

# ─── Setup ───────────────────────────────────────────────────────────
LOG_DIR=$(mktemp -d /tmp/obelysk_e2e_XXXXXX)
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$OUTPUT_DIR/audit_${TIMESTAMP}.json"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║         Obelysk E2E Pipeline                 ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Model:      $MODEL_DIR"
echo "  Layers:     $LAYERS"
echo "  Inferences: $COUNT"
echo "  GPU:        ${GPU_FLAG:-off}"
echo "  Output:     $REPORT"
echo ""

# ─── Step 1: Capture inference logs ──────────────────────────────────
echo "━━━ Step 1/2: Capturing $COUNT inference logs ━━━"
"$BINARY" capture \
  --model-dir "$MODEL_DIR" \
  --layers "$LAYERS" \
  --log-dir "$LOG_DIR" \
  --count "$COUNT"

echo ""

# ─── Step 2: Run audit ──────────────────────────────────────────────
echo "━━━ Step 2/2: Running audit (prove + report) ━━━"

AUDIT_ARGS=(
  --log-dir "$LOG_DIR"
  --model-dir "$MODEL_DIR"
  --layers "$LAYERS"
  --evaluate
  --output "$REPORT"
)

[[ -n "$GPU_FLAG" ]] && AUDIT_ARGS+=($GPU_FLAG)
[[ -n "$DRY_RUN" ]] && AUDIT_ARGS+=($DRY_RUN)
[[ -n "$SUBMIT" ]] && AUDIT_ARGS+=($SUBMIT)

"$BINARY" audit "${AUDIT_ARGS[@]}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pipeline complete!"
echo "  Report: $REPORT"
echo "  Logs:   $LOG_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
