#!/usr/bin/env bash
# benchmark_gpu.sh — Run GPU proving benchmarks on H100 and report timing.
#
# Usage:
#   bash scripts/benchmark_gpu.sh [LAYERS] [MODEL_DIR]
#
# Examples:
#   bash scripts/benchmark_gpu.sh 1
#   bash scripts/benchmark_gpu.sh 5 /path/to/qwen3-14b
#   STWO_WEIGHT_BINDING=aggregated bash scripts/benchmark_gpu.sh 1
#
# Environment variables:
#   LAYERS              Number of transformer layers (default: 1)
#   MODEL_DIR           Model directory (default: auto-detect)
#   FORMAT              Output format: ml_gkr or cairo_serde (default: ml_gkr)
#   STWO_WEIGHT_BINDING Weight binding mode: aggregated or unset (default: unset)
#   SKIP_BUILD          Set to 1 to skip cargo build (default: 0)
set -euo pipefail

LAYERS="${1:-${LAYERS:-1}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FORMAT="${FORMAT:-ml_gkr}"
SKIP_BUILD="${SKIP_BUILD:-0}"

# Auto-detect model directory
if [[ -n "${2:-}" ]]; then
    MODEL_DIR="$2"
elif [[ -n "${MODEL_DIR:-}" ]]; then
    : # already set
elif [[ -d "/home/shadeform/.obelysk/models/qwen3-14b" ]]; then
    MODEL_DIR="/home/shadeform/.obelysk/models/qwen3-14b"
elif [[ -d "$HOME/.obelysk/models/qwen3-14b" ]]; then
    MODEL_DIR="$HOME/.obelysk/models/qwen3-14b"
else
    echo "ERROR: No model directory found. Set MODEL_DIR or pass as arg."
    exit 1
fi

# Determine features
FEATURES="std,gpu,onnx,safetensors,model-loading,cli,audit"
if command -v nvidia-smi &>/dev/null; then
    FEATURES="${FEATURES},cuda-runtime"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'detected')"
    GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')MiB"
else
    GPU_NAME="none (CPU only)"
    GPU_MEM="n/a"
fi

echo "=== ZKML GPU Benchmark ==="
echo "  Model     : ${MODEL_DIR}"
echo "  Layers    : ${LAYERS}"
echo "  Format    : ${FORMAT}"
echo "  GPU       : ${GPU_NAME} (${GPU_MEM})"
echo "  Features  : ${FEATURES}"
echo "  Binding   : ${STWO_WEIGHT_BINDING:-default (batched openings)}"
echo ""

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo "Building prove-model (release)..."
    cd "${REPO_ROOT}"
    cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -3
    echo ""
fi

OUTPUT="/tmp/benchmark_qwen3_${LAYERS}L_$(date +%s).json"

echo "Starting prove (${LAYERS} layers, format=${FORMAT})..."
echo "Output: ${OUTPUT}"
echo ""

cd "${REPO_ROOT}"
time target/release/prove-model \
    --model-dir "${MODEL_DIR}" \
    --layers "${LAYERS}" \
    --gpu \
    --format "${FORMAT}" \
    --output "${OUTPUT}" \
    2>&1 | tee /tmp/benchmark_latest.log

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "=== Benchmark Summary ==="

# Extract key timings from log
if [[ -f /tmp/benchmark_latest.log ]]; then
    echo "Key timings:"
    grep -E '(Forward pass complete|layer reductions complete|weight commitments:.*done|weight openings:.*elapsed|GKR proof:.*layer proofs|Unified STARK|Proving completed|Phase [0-9]+ complete)' /tmp/benchmark_latest.log | sed 's/^/  /'
    echo ""
    grep -E '(GPU|CPU|backend|MLE.*backend|OOM|fallback)' /tmp/benchmark_latest.log | sort -u | sed 's/^/  /'
fi

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "Result: SUCCESS"
    echo "Proof: ${OUTPUT} ($(du -h "${OUTPUT}" 2>/dev/null | cut -f1 || echo '?') bytes)"
else
    echo "Result: FAILED (exit code ${EXIT_CODE})"
    # Check for known issues
    if grep -q 'ConstraintsNotSatisfied' /tmp/benchmark_latest.log 2>/dev/null; then
        echo "  Known issue: Phase 3 STARK ConstraintsNotSatisfied (pre-existing)"
        echo "  GKR/commitment timing above is still valid."
    fi
fi
