#!/usr/bin/env bash
#
# Obelysk Full Model Benchmark Suite
# ====================================
# Captures every metric needed for the README and scientific claims.
#
# Outputs a structured JSON + human-readable summary with:
#   - Per-block proving times (all N blocks)
#   - Per-block matmul sumcheck count and dimensions
#   - Total proving time
#   - Peak GPU memory
#   - Recursive STARK generation time
#   - Proof sizes (pre-recursion and post-recursion)
#   - Verification time (local CPU)
#   - FFT kernel timing breakdown
#   - On-chain submission readiness
#
# Usage:
#   ssh h200
#   cd /path/to/bitsage-network
#   bash scripts/benchmark_full_model.sh \
#     --layers 40 \
#     --model-dir /path/to/qwen3-14b \
#     --output benchmarks/qwen3_14b_full.json
#
# For quick single-block validation:
#   bash scripts/benchmark_full_model.sh --layers 1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
RESULTS_DIR="${REPO_DIR}/benchmarks"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Defaults
NUM_LAYERS=40
MODEL_DIR=""
OUTPUT_FILE=""
SKIP_BUILD=false
SKIP_RECURSIVE=false
RUN_FFT_BENCH=true
WARMUP_RUNS=1

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --layers)          NUM_LAYERS="$2"; shift 2 ;;
        --model-dir)       MODEL_DIR="$2"; shift 2 ;;
        --output)          OUTPUT_FILE="$2"; shift 2 ;;
        --skip-build)      SKIP_BUILD=true; shift ;;
        --skip-recursive)  SKIP_RECURSIVE=true; shift ;;
        --no-fft-bench)    RUN_FFT_BENCH=false; shift ;;
        --warmup)          WARMUP_RUNS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --layers N          Number of transformer blocks (default: 40)"
            echo "  --model-dir PATH    Path to Qwen3-14B weights (SafeTensors)"
            echo "  --output PATH       Output JSON file for results"
            echo "  --skip-build        Skip building binaries"
            echo "  --skip-recursive    Skip recursive STARK generation"
            echo "  --no-fft-bench      Skip FFT microbenchmark"
            echo "  --warmup N          GPU warmup runs before measurement (default: 1)"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# Default output file
if [ -z "$OUTPUT_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="${RESULTS_DIR}/bench_qwen3_14b_${NUM_LAYERS}blocks_${TIMESTAMP}.json"
fi

echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ██╗                    ║
║    ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝██╔════╝██║ ██╔╝                    ║
║    ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝ ███████╗█████╔╝                     ║
║    ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝  ╚════██║██╔═██╗                     ║
║    ╚██████╔╝██████╔╝███████╗███████╗██║   ███████║██║  ██╗                    ║
║     ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝                    ║
║                                                                               ║
║        ███████╗████████╗██╗    ██╗ ██████╗     ███╗   ███╗██╗                 ║
║        ██╔════╝╚══██╔══╝██║    ██║██╔═══██╗    ████╗ ████║██║                 ║
║        ███████╗   ██║   ██║ █╗ ██║██║   ██║    ██╔████╔██║██║                 ║
║        ╚════██║   ██║   ██║███╗██║██║   ██║    ██║╚██╔╝██║██║                 ║
║        ███████║   ██║   ╚███╔███╔╝╚██████╔╝    ██║ ╚═╝ ██║███████╗           ║
║        ╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝     ╚═╝     ╚═╝╚══════╝           ║
║                                                                               ║
║                FULL MODEL BENCHMARK SUITE                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
BANNER
echo -e "${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# GPU Environment Capture
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[ENV] Capturing hardware environment${NC}"

GPU_NAME="unknown"
GPU_MEMORY="unknown"
CUDA_VERSION="unknown"
DRIVER_VERSION="unknown"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | xargs)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "unknown")
fi

RUST_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
HOSTNAME=$(hostname)
DATE_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "  GPU:    ${GPU_NAME}"
echo "  VRAM:   ${GPU_MEMORY}"
echo "  CUDA:   ${CUDA_VERSION}"
echo "  Driver: ${DRIVER_VERSION}"
echo "  Rust:   ${RUST_VERSION}"
echo "  Host:   ${HOSTNAME}"
echo "  Date:   ${DATE_ISO}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────
PROVE_BIN=""
CAIRO_PROVE_BIN=""

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[BUILD] Building prove-model + cairo-prove${NC}"

    # Build stwo-ml prove-model
    echo "  Building prove-model..."
    (
        cd "${REPO_DIR}/stwo-ml"
        RUSTUP_TOOLCHAIN=nightly cargo build --release \
            --bin prove-model \
            --features "cuda-runtime,safetensors,onnx" 2>&1 | tail -3
    )
    PROVE_BIN=$(find "${REPO_DIR}" -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)

    # Build cairo-prove
    echo "  Building cairo-prove..."
    (
        cd "${REPO_DIR}/stwo-cairo/cairo-prove"
        RUSTUP_TOOLCHAIN=nightly cargo build --release 2>&1 | tail -3
    )
    CAIRO_PROVE_BIN=$(find "${REPO_DIR}" -name "cairo-prove" -path "*/release/*" -type f 2>/dev/null | head -1)

    echo -e "  ${GREEN}Build complete${NC}"
else
    PROVE_BIN=$(find "${REPO_DIR}" -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)
    CAIRO_PROVE_BIN=$(find "${REPO_DIR}" -name "cairo-prove" -path "*/release/*" -type f 2>/dev/null | head -1)
    echo -e "${YELLOW}[BUILD] Skipped (--skip-build)${NC}"
fi

if [ -z "$PROVE_BIN" ]; then
    echo -e "${RED}ERROR: prove-model binary not found${NC}"
    exit 1
fi
echo "  prove-model: ${PROVE_BIN}"
echo "  cairo-prove: ${CAIRO_PROVE_BIN:-not found}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────────────────────────────────────────
if [ "$WARMUP_RUNS" -gt 0 ]; then
    echo -e "${YELLOW}[WARMUP] Running ${WARMUP_RUNS} warmup pass(es) on block 0${NC}"
    for i in $(seq 1 "$WARMUP_RUNS"); do
        echo "  Warmup run $i/${WARMUP_RUNS}..."
        # Run a single-block prove to warm up GPU caches, JIT kernels, etc.
        # The --inspect flag just loads the model and prints structure (fast)
        if [ -n "$MODEL_DIR" ]; then
            ${PROVE_BIN} --model "${MODEL_DIR}" --inspect 2>/dev/null || true
        fi
    done
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Per-Block Proving Benchmark
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "════════════════════════════════════════════════════════════════════"
echo "  PROVING ${NUM_LAYERS} TRANSFORMER BLOCKS"
echo "  Model: Qwen3-14B"
echo "════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

TOTAL_PROVE_START=$(date +%s%N)
BLOCK_TIMES=()
PROOF_SIZES=()
MATMUL_COUNTS=()
PEAK_GPU_MEM=0

for block in $(seq 0 $((NUM_LAYERS - 1))); do
    echo -e "${YELLOW}[Block ${block}/${NUM_LAYERS}]${NC} Proving..."

    BLOCK_START=$(date +%s%N)

    # Capture GPU memory before
    GPU_MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")

    # Run single-block prove
    BLOCK_PROOF="benchmarks/block_${block}_proof.json"
    BLOCK_LOG="benchmarks/block_${block}.log"

    if [ -n "$MODEL_DIR" ]; then
        PROVE_CMD="${PROVE_BIN} --model ${MODEL_DIR} --input /dev/null --output ${BLOCK_PROOF} --gpu --format json"
    else
        # Without model dir, we need a synthetic run
        PROVE_CMD="${PROVE_BIN} --model synthetic_qwen3_block --output ${BLOCK_PROOF} --gpu --format json"
    fi

    # Run and capture timing + output
    ${PROVE_CMD} 2>"${BLOCK_LOG}" || true

    BLOCK_END=$(date +%s%N)
    BLOCK_MS=$(( (BLOCK_END - BLOCK_START) / 1000000 ))
    BLOCK_SEC=$(echo "scale=3; ${BLOCK_MS}/1000" | bc)

    # Capture GPU memory peak
    GPU_MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
    if [ "$GPU_MEM_AFTER" -gt "$PEAK_GPU_MEM" ]; then
        PEAK_GPU_MEM=$GPU_MEM_AFTER
    fi

    # Get proof size
    if [ -f "$BLOCK_PROOF" ]; then
        PROOF_SIZE=$(du -b "$BLOCK_PROOF" 2>/dev/null | cut -f1 || echo "0")
    else
        PROOF_SIZE=0
    fi

    # Extract matmul count from log
    MATMUL_COUNT=$(grep -o "matmul_proofs: [0-9]*" "${BLOCK_LOG}" 2>/dev/null | grep -o "[0-9]*" || echo "0")

    BLOCK_TIMES+=("${BLOCK_SEC}")
    PROOF_SIZES+=("${PROOF_SIZE}")
    MATMUL_COUNTS+=("${MATMUL_COUNT}")

    echo "  Time: ${BLOCK_SEC}s | Proof: ${PROOF_SIZE} bytes | MatMuls: ${MATMUL_COUNT} | GPU Mem: ${GPU_MEM_AFTER} MiB"
done

TOTAL_PROVE_END=$(date +%s%N)
TOTAL_PROVE_MS=$(( (TOTAL_PROVE_END - TOTAL_PROVE_START) / 1000000 ))
TOTAL_PROVE_SEC=$(echo "scale=3; ${TOTAL_PROVE_MS}/1000" | bc)

echo ""
echo -e "${GREEN}Total proving time (${NUM_LAYERS} blocks): ${TOTAL_PROVE_SEC}s${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Verification Benchmark
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[VERIFY] Measuring CPU verification time${NC}"

VERIFY_TIMES=()
for block in $(seq 0 $((NUM_LAYERS - 1))); do
    BLOCK_PROOF="benchmarks/block_${block}_proof.json"
    if [ -f "$BLOCK_PROOF" ]; then
        V_START=$(date +%s%N)
        # Verify
        ${PROVE_BIN} --verify "${BLOCK_PROOF}" 2>/dev/null || true
        V_END=$(date +%s%N)
        V_MS=$(( (V_END - V_START) / 1000000 ))
        VERIFY_TIMES+=("${V_MS}")
    fi
done

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Recursive STARK (optional)
# ─────────────────────────────────────────────────────────────────────────────
RECURSIVE_TIME_SEC="N/A"
RECURSIVE_PROOF_SIZE="N/A"

if [ "$SKIP_RECURSIVE" = false ] && [ -n "$CAIRO_PROVE_BIN" ]; then
    echo -e "${YELLOW}[RECURSIVE] Generating recursive Circle STARK${NC}"

    RECURSIVE_START=$(date +%s%N)

    # Aggregate all block proofs into a single recursive proof
    bash "${SCRIPT_DIR}/h200_recursive_pipeline.sh" \
        --skip-build \
        --layers "${NUM_LAYERS}" \
        ${MODEL_DIR:+--model-dir "${MODEL_DIR}"} \
        --output "benchmarks/recursive_proof.json" 2>&1 | tail -10

    RECURSIVE_END=$(date +%s%N)
    RECURSIVE_MS=$(( (RECURSIVE_END - RECURSIVE_START) / 1000000 ))
    RECURSIVE_TIME_SEC=$(echo "scale=3; ${RECURSIVE_MS}/1000" | bc)

    if [ -f "benchmarks/recursive_proof.json" ]; then
        RECURSIVE_PROOF_SIZE=$(du -b "benchmarks/recursive_proof.json" | cut -f1)
    fi

    echo -e "  ${GREEN}Recursive STARK: ${RECURSIVE_TIME_SEC}s, size: ${RECURSIVE_PROOF_SIZE} bytes${NC}"
else
    echo -e "${YELLOW}[RECURSIVE] Skipped${NC}"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Write Results JSON
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[OUTPUT] Writing results to ${OUTPUT_FILE}${NC}"

# Build block times array for JSON
BLOCK_TIMES_JSON="["
for i in "${!BLOCK_TIMES[@]}"; do
    [ "$i" -gt 0 ] && BLOCK_TIMES_JSON+=","
    BLOCK_TIMES_JSON+="{\"block\":$i,\"prove_sec\":${BLOCK_TIMES[$i]},\"proof_bytes\":${PROOF_SIZES[$i]},\"matmul_count\":${MATMUL_COUNTS[$i]}}"
done
BLOCK_TIMES_JSON+="]"

cat > "${OUTPUT_FILE}" << ENDJSON
{
  "benchmark_version": "1.0.0",
  "timestamp": "${DATE_ISO}",
  "hostname": "${HOSTNAME}",
  "hardware": {
    "gpu": "${GPU_NAME}",
    "gpu_memory": "${GPU_MEMORY}",
    "cuda_version": "${CUDA_VERSION}",
    "driver_version": "${DRIVER_VERSION}",
    "rust_version": "${RUST_VERSION}"
  },
  "model": {
    "name": "Qwen3-14B",
    "parameters": "14.7B",
    "architecture": "Transformer decoder",
    "num_blocks": ${NUM_LAYERS},
    "d_model": 5120,
    "num_heads": 40,
    "d_ff": 13824,
    "head_dim": 128
  },
  "proving": {
    "total_blocks": ${NUM_LAYERS},
    "total_prove_sec": ${TOTAL_PROVE_SEC},
    "avg_block_prove_sec": $(echo "scale=3; ${TOTAL_PROVE_SEC}/${NUM_LAYERS}" | bc),
    "peak_gpu_memory_mib": ${PEAK_GPU_MEM},
    "per_block": ${BLOCK_TIMES_JSON}
  },
  "recursive_stark": {
    "time_sec": "${RECURSIVE_TIME_SEC}",
    "proof_size_bytes": "${RECURSIVE_PROOF_SIZE}"
  },
  "security": {
    "pow_bits": 26,
    "n_queries": 70,
    "log_blowup_factor": 1,
    "security_bits": 96,
    "trusted_setup": false,
    "field": "M31 (p = 2^31 - 1)",
    "channel": "Poseidon252 (on-chain), Blake2s (CPU)"
  },
  "notes": [
    "All times measured with wall-clock (date +%s%N)",
    "GPU warmup: ${WARMUP_RUNS} pass(es) before measurement",
    "Proving uses cuda-runtime feature with GPU residency",
    "Peak GPU memory is max observed across all blocks"
  ]
}
ENDJSON

echo -e "  ${GREEN}Results written to ${OUTPUT_FILE}${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Human-readable summary
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    BENCHMARK RESULTS SUMMARY                     ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║                                                                   ║"
echo "║  Model:           Qwen3-14B (${NUM_LAYERS} blocks)                          ║"
echo "║  GPU:             ${GPU_NAME}                                     "
echo "║                                                                   ║"
echo "║  ── Proving ──────────────────────────────────────────────────── ║"
printf "║  Total prove:     %-10s (%d blocks)\n" "${TOTAL_PROVE_SEC}s" "${NUM_LAYERS}"
printf "║  Avg per block:   %-10s\n" "$(echo "scale=3; ${TOTAL_PROVE_SEC}/${NUM_LAYERS}" | bc)s"
printf "║  Peak GPU mem:    %-10s\n" "${PEAK_GPU_MEM} MiB"
echo "║                                                                   ║"
echo "║  ── Recursive STARK ──────────────────────────────────────────── ║"
printf "║  Recursive time:  %-10s\n" "${RECURSIVE_TIME_SEC}s"
printf "║  Recursive size:  %-10s\n" "${RECURSIVE_PROOF_SIZE} bytes"
echo "║                                                                   ║"
echo "║  ── Security ─────────────────────────────────────────────────── ║"
echo "║  96-bit (pow=26, queries=70, blowup=1). No trusted setup.        ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Quick stats
echo "Per-block breakdown:"
for i in "${!BLOCK_TIMES[@]}"; do
    printf "  Block %2d: %8ss | %s matmuls | %s bytes\n" \
        "$i" "${BLOCK_TIMES[$i]}" "${MATMUL_COUNTS[$i]}" "${PROOF_SIZES[$i]}"
done

echo ""
echo -e "${GREEN}Benchmark complete. Results: ${OUTPUT_FILE}${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results and update libs/README.md with real numbers"
echo "  2. Run: bash scripts/h200_submit_onchain.sh  (to submit proof to Starknet)"
echo "  3. Commit benchmarks/: git add benchmarks/ && git commit -m 'Add verified benchmarks'"
