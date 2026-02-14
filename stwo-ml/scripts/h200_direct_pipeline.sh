#!/bin/bash
# Phase 2: Direct On-Chain ML Verification Pipeline
#
# Eliminates Stage 2 (Cairo VM recursive proving, 46.8s) entirely.
#
# BEFORE (3-stage):
#   Stage 1: GPU ML Proof          (~40s)
#   Stage 2: Cairo VM recursion    (~46.8s)  <-- ELIMINATED
#   Stage 3: On-chain verify       (~10s network)
#
# AFTER (2-stage):
#   Stage 1: GPU ML Proof          (~40s)
#   Stage 2: On-chain verify_model_direct()  (~10s network, 0s proving)
#
# Usage:
#   ./h200_direct_pipeline.sh \
#     --model /path/to/model.onnx \
#     --input /path/to/input.json \
#     --contract 0x<elo-cairo-verifier-address> \
#     --model-id 0x1234 \
#     --account deployer

set -euo pipefail

# Default values
CONTRACT=""
MODEL=""
INPUT=""
MODEL_ID="0x1"
ACCOUNT="deployer"
GPU="--gpu"
OUTPUT_DIR="/tmp/direct_proof"
NETWORK="sepolia"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --model-dir) MODEL="--model-dir $2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --contract) CONTRACT="$2"; shift 2 ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --no-gpu) GPU=""; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --network) NETWORK="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "$CONTRACT" ]]; then
    echo "ERROR: --contract is required (elo-cairo-verifier address)"
    exit 1
fi
if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model or --model-dir is required"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Direct On-Chain ML Verification Pipeline (Phase 2)"
echo "============================================================"
echo "  Model:     $MODEL"
echo "  Input:     ${INPUT:-<generated>}"
echo "  Contract:  $CONTRACT"
echo "  Model ID:  $MODEL_ID"
echo "  Network:   $NETWORK"
echo "  GPU:       ${GPU:-disabled}"
echo "============================================================"
echo ""

# ============================================================
# Stage 1: GPU ML Proof (same as before)
# ============================================================
echo "[Stage 1] Generating ML proof with prove-model..."
START_TIME=$(date +%s)

PROVE_CMD="prove-model $MODEL"
if [[ -n "$INPUT" ]]; then
    PROVE_CMD="$PROVE_CMD --input $INPUT"
fi
PROVE_CMD="$PROVE_CMD --output $OUTPUT_DIR/proof.json --format direct --model-id $MODEL_ID $GPU"

echo "  Command: $PROVE_CMD"
eval "$PROVE_CMD"

PROVE_TIME=$(($(date +%s) - START_TIME))
echo "[Stage 1] Proof generated in ${PROVE_TIME}s"
echo ""

# ============================================================
# Stage 2: ELIMINATED — No Cairo VM step!
# ============================================================
echo "[Stage 2] ELIMINATED — No Cairo VM recursion needed!"
echo "  (Previously took 46.8s for Qwen3-14B)"
echo ""

# ============================================================
# Stage 3: Upload chunks + verify on-chain
# ============================================================
echo "[Stage 3] Uploading proof chunks and verifying on-chain..."

# Generate a unique session ID based on timestamp
SESSION_ID="0x$(date +%s | xxd -p | head -c 16)"
echo "  Session ID: $SESSION_ID"

# Upload STARK chunks (if any)
CHUNK_DIR="$OUTPUT_DIR/chunks"
if [[ -d "$CHUNK_DIR" ]]; then
    CHUNK_COUNT=$(ls "$CHUNK_DIR"/chunk_*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  Uploading $CHUNK_COUNT STARK chunks..."

    for i in $(seq 0 $((CHUNK_COUNT - 1))); do
        CHUNK_FILE="$CHUNK_DIR/chunk_${i}.json"
        if [[ -f "$CHUNK_FILE" ]]; then
            CHUNK_DATA=$(cat "$CHUNK_FILE")
            echo "    Chunk $i: $(echo "$CHUNK_DATA" | wc -c | tr -d ' ') bytes"

            sncast --account "$ACCOUNT" --network "$NETWORK" \
                invoke --contract-address "$CONTRACT" \
                --function upload_proof_chunk \
                --calldata "$SESSION_ID" "$i" $CHUNK_DATA \
                --max-fee 0.01

            echo "    Chunk $i uploaded"
        fi
    done
else
    echo "  No STARK chunks to upload"
fi

# Call verify_model_direct
echo "  Calling verify_model_direct..."

BATCHED_DATA="$OUTPUT_DIR/batched_calldata.json"
if [[ -f "$BATCHED_DATA" ]]; then
    CALLDATA=$(cat "$BATCHED_DATA")
else
    CALLDATA=""
fi

sncast --account "$ACCOUNT" --network "$NETWORK" \
    invoke --contract-address "$CONTRACT" \
    --function verify_model_direct \
    --calldata "$MODEL_ID" "$SESSION_ID" $CALLDATA \
    --max-fee 0.05

TOTAL_TIME=$(($(date +%s) - START_TIME))

echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo "  Stage 1 (GPU prove):    ${PROVE_TIME}s"
echo "  Stage 2 (eliminated):   0s"
echo "  Total time:             ${TOTAL_TIME}s"
echo "  Savings vs old:         ~46.8s"
echo "============================================================"
