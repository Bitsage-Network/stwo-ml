#!/usr/bin/env bash
# ec2_stream.sh — Build and launch prove-server with GKR proof streaming on EC2.
#
# Usage:
#   FEATURES=server-stream,cuda-runtime PORT=8080 bash scripts/ec2_stream.sh
#
# Environment variables:
#   FEATURES        Cargo feature flags (default: server-stream,cuda-runtime)
#   PORT            Server port (default: 8080)
#   VALIDATOR_URL   Optional validator endpoint to receive proof results
#   BIND_ADDR       Full bind address override (default: 0.0.0.0:$PORT)
#   SKIP_BUILD      Set to 1 to skip cargo build (use existing binary)
set -euo pipefail

FEATURES="${FEATURES:-server-stream,cuda-runtime}"
PORT="${PORT:-8080}"
VALIDATOR_URL="${VALIDATOR_URL:-}"
BIND_ADDR="${BIND_ADDR:-0.0.0.0:${PORT}}"
SKIP_BUILD="${SKIP_BUILD:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== ZKML GKR Proof Stream Server ==="
echo "  Features : ${FEATURES}"
echo "  Bind     : ${BIND_ADDR}"
echo "  Validator: ${VALIDATOR_URL:-<none>}"
echo ""

# Check for CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'detected')"
    # Ensure cuda-runtime is included
    if [[ "${FEATURES}" != *cuda-runtime* ]]; then
        FEATURES="${FEATURES},cuda-runtime"
        echo "  Note     : cuda-runtime auto-added to features"
    fi
else
    echo "  GPU      : not detected (CPU prover path)"
    FEATURES="${FEATURES/,cuda-runtime/}"
    FEATURES="${FEATURES/cuda-runtime,/}"
    FEATURES="${FEATURES/cuda-runtime/}"
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo ""
    echo "Building prove-server (features=${FEATURES})…"
    cd "${REPO_ROOT}"
    cargo build --release --bin prove-server --features "${FEATURES}"
    echo "Build complete."
fi

# Resolve public IP for display
PUBLIC_IP="$(curl -sf --max-time 3 http://ifconfig.me 2>/dev/null || curl -sf --max-time 3 http://ipecho.net/plain 2>/dev/null || echo "localhost")"

echo ""
echo "=== Server starting ==="
echo "  Dashboard : http://${PUBLIC_IP}:${PORT}/"
echo "  WebSocket : ws://${PUBLIC_IP}:${PORT}/ws"
echo "  REST API  : http://${PUBLIC_IP}:${PORT}/api/v1/"
echo "  Health    : http://${PUBLIC_IP}:${PORT}/health"
[[ -n "${VALIDATOR_URL}" ]] && echo "  Validator : ${VALIDATOR_URL} (bridge active)"
echo ""

exec env \
    BIND_ADDR="${BIND_ADDR}" \
    VALIDATOR_URL="${VALIDATOR_URL}" \
    "${REPO_ROOT}/target/release/prove-server"
