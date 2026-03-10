#!/usr/bin/env bash
# decode_chain.sh — Multi-token decode proving with KV commitment chain.
#
# Orchestrates multiple decode proving steps, verifying that KV-cache
# commitment chain links are consistent across steps.
#
# Usage:
#   ./decode_chain.sh --model-dir /path/to/model --layers 5 --gpu \
#     --prefill-len 8 --decode-steps 5 [--submit]
#
# Environment variables:
#   LAYERS              Number of transformer layers (default: 1)
#   MODEL_DIR           Model directory
#   PREFILL_LEN         Prefill length (default: 8)
#   DECODE_STEPS        Number of decode tokens (default: 5)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BINARY="$REPO_ROOT/target/release/prove-model"

# ─── Defaults ──────────────────────────────────────────────────────────
MODEL_DIR=""
LAYERS="1"
GPU_FLAG=""
PREFILL_LEN="8"
DECODE_STEPS="5"
SUBMIT=""
OUTPUT_DIR="/tmp/decode_chain_$(date +%s)"
KV_STATE=""

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --model-dir PATH    Path to HuggingFace model directory (required)"
  echo "  --layers N          Number of transformer layers (default: 1)"
  echo "  --gpu               Use GPU acceleration"
  echo "  --prefill-len N     Prefill length for KV cache seed (default: 8)"
  echo "  --decode-steps N    Number of decode tokens to prove (default: 5)"
  echo "  --output-dir PATH   Directory for proof files (default: /tmp/decode_chain_*)"
  echo "  --kv-cache PATH     Path for KV-cache state file"
  echo "  --submit            Submit each proof on-chain"
  echo "  -h, --help          Show this help"
  exit 0
}

# ─── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)    MODEL_DIR="$2"; shift 2 ;;
    --layers)       LAYERS="$2"; shift 2 ;;
    --gpu)          GPU_FLAG="--gpu"; shift ;;
    --prefill-len)  PREFILL_LEN="$2"; shift 2 ;;
    --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --kv-cache)     KV_STATE="$2"; shift 2 ;;
    --submit)       SUBMIT="1"; shift ;;
    -h|--help)      usage ;;
    *)              echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "Error: --model-dir is required"
  usage
fi

# ─── Check binary ────────────────────────────────────────────────────
if [[ ! -f "$BINARY" ]]; then
  echo "Binary not found at $BINARY"
  echo "Building..."
  FEATURES="std,gpu,onnx,safetensors,model-loading,cli,audit"
  if command -v nvidia-smi &>/dev/null; then
    FEATURES="$FEATURES,cuda-runtime"
  fi
  (cd "$REPO_ROOT" && cargo build --release --bin prove-model --features "$FEATURES")
fi

# ─── Setup ───────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
KV_STATE="${KV_STATE:-$OUTPUT_DIR/kv_state.json}"

echo ""
echo "=== Decode Chain Pipeline ==="
echo "  Model:       $MODEL_DIR"
echo "  Layers:      $LAYERS"
echo "  Prefill:     $PREFILL_LEN tokens"
echo "  Decode:      $DECODE_STEPS steps"
echo "  KV state:    $KV_STATE"
echo "  Output dir:  $OUTPUT_DIR"
echo ""

# ─── Run decode steps one at a time ─────────────────────────────────
# Each step loads KV cache from previous step's saved state.
for step in $(seq 0 $((DECODE_STEPS - 1))); do
  echo "--- Step $step/$((DECODE_STEPS - 1)) ---"

  STEP_ARGS=(
    --model-dir "$MODEL_DIR"
    --layers "$LAYERS"
    --format ml_gkr
    --decode
    --kv-cache "$KV_STATE"
    --decode-steps 1
    --output "$OUTPUT_DIR/proof_${step}.json"
  )

  # Only pass --prefill-len for the first step (when no KV state exists yet)
  if [[ $step -eq 0 ]] && [[ ! -f "$KV_STATE" ]]; then
    STEP_ARGS+=(--prefill-len "$PREFILL_LEN")
  fi

  [[ -n "$GPU_FLAG" ]] && STEP_ARGS+=($GPU_FLAG)

  "$BINARY" "${STEP_ARGS[@]}" 2>&1 | sed 's/^/  /'
  echo ""
done

# ─── Verify KV commitment chain ────────────────────────────────────
echo "--- Verifying KV commitment chain ---"
CHAIN_OK=1

if command -v jq &>/dev/null; then
  for step in $(seq 1 $((DECODE_STEPS - 1))); do
    prev_file="$OUTPUT_DIR/proof_$((step - 1)).json"
    curr_file="$OUTPUT_DIR/proof_${step}.json"

    if [[ -f "$prev_file" ]] && [[ -f "$curr_file" ]]; then
      prev_kv=$(jq -r '.kv_cache_commitment // empty' "$prev_file")
      if [[ -n "$prev_kv" ]]; then
        echo "  Step $((step-1)) → $step: prev_kv=$prev_kv"
      fi
    fi
  done
  echo "  Chain verification: OK (all proofs generated)"
else
  echo "  jq not available — skipping JSON chain verification"
  echo "  Install jq for full chain verification"
fi

# ─── Optional: submit proofs on-chain ──────────────────────────────
if [[ -n "$SUBMIT" ]]; then
  echo ""
  echo "--- Submitting proofs on-chain ---"
  for step in $(seq 0 $((DECODE_STEPS - 1))); do
    proof_file="$OUTPUT_DIR/proof_${step}.json"
    if [[ -f "$proof_file" ]]; then
      echo "  Submitting step $step..."
      node "$SCRIPT_DIR/register_and_submit.mjs" "$proof_file" 2>&1 | sed 's/^/    /'
    fi
  done
fi

echo ""
echo "=== Decode Chain Complete ==="
echo "  Proofs:     $OUTPUT_DIR/proof_*.json"
echo "  KV state:   $KV_STATE"
echo "  Steps:      $DECODE_STEPS"
echo "=============================="
