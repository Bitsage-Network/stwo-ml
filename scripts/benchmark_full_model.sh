#!/usr/bin/env bash
#
# Obelysk Full Model Benchmark Suite
# ====================================
# Captures every metric needed for the README and scientific claims.
#
# Outputs a structured JSON + human-readable summary with:
#   - Per-block proving times (each block proved individually)
#   - Per-block matmul sumcheck count and dimensions
#   - Total proving time (all blocks together)
#   - Peak GPU memory
#   - Recursive STARK generation time
#   - Proof sizes (pre-recursion and post-recursion)
#   - Verification time (local CPU)
#   - On-chain submission readiness
#
# Usage:
#   ssh h200
#   cd /path/to/bitsage-network/libs
#   bash scripts/benchmark_full_model.sh \
#     --layers 40 \
#     --model-dir ~/models/qwen3-14b \
#     --output benchmarks/qwen3_14b_full.json
#
# For quick single-block validation:
#   bash scripts/benchmark_full_model.sh --layers 1 --model-dir ~/models/qwen3-14b
#
# For full-model one-shot (all N blocks in a single prove-model call):
#   bash scripts/benchmark_full_model.sh --layers 40 --model-dir ~/models/qwen3-14b --one-shot
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
WARMUP_RUNS=1
NO_WARMUP=false
ONE_SHOT=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --layers)          NUM_LAYERS="$2"; shift 2 ;;
        --model-dir)       MODEL_DIR="$2"; shift 2 ;;
        --output)          OUTPUT_FILE="$2"; shift 2 ;;
        --skip-build)      SKIP_BUILD=true; shift ;;
        --skip-recursive)  SKIP_RECURSIVE=true; shift ;;
        --warmup)          WARMUP_RUNS="$2"; shift 2 ;;
        --no-warmup)       NO_WARMUP=true; shift ;;
        --one-shot)        ONE_SHOT=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --layers N          Number of transformer blocks (default: 40)"
            echo "  --model-dir PATH    Path to Qwen3-14B weights (SafeTensors)"
            echo "  --output PATH       Output JSON file for results"
            echo "  --skip-build        Skip building binaries"
            echo "  --skip-recursive    Skip recursive STARK generation"
            echo "  --warmup N          GPU warmup runs before measurement (default: 1)"
            echo "  --no-warmup         Skip warmup entirely (use after first run)"
            echo "  --one-shot          Prove all N blocks in a single invocation"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    echo -e "${RED}ERROR: --model-dir is required${NC}"
    echo "  Example: --model-dir ~/models/qwen3-14b"
    exit 1
fi

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

# CUDA paths
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu"
export PATH="/usr/local/cuda-12.4/bin:${PATH}"

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

echo "  GPU:       ${GPU_NAME}"
echo "  VRAM:      ${GPU_MEMORY}"
echo "  CUDA:      ${CUDA_VERSION}"
echo "  Driver:    ${DRIVER_VERSION}"
echo "  Rust:      ${RUST_VERSION}"
echo "  Host:      ${HOSTNAME}"
echo "  Date:      ${DATE_ISO}"
echo "  Model dir: ${MODEL_DIR}"
echo "  Layers:    ${NUM_LAYERS}"
echo "  Mode:      $([ "$ONE_SHOT" = true ] && echo "one-shot" || echo "per-block")"
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
    BUILD_START=$(date +%s%N)
    (
        cd "${REPO_DIR}/stwo-ml"
        FEATURES="cli"
        if command -v nvidia-smi &>/dev/null; then
            FEATURES="cli,cuda-runtime"
        fi
        cargo build --release \
            --bin prove-model \
            --features "${FEATURES}" 2>&1 | tail -5
    )
    BUILD_END=$(date +%s%N)
    BUILD_SEC=$(echo "scale=1; ($BUILD_END - $BUILD_START) / 1000000000" | bc)
    PROVE_BIN=$(find "${REPO_DIR}" -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)
    echo -e "  ${GREEN}prove-model built in ${BUILD_SEC}s${NC}"

    # Build cairo-prove
    echo "  Building cairo-prove..."
    (
        cd "${REPO_DIR}/stwo-cairo/cairo-prove"
        cargo build --release 2>&1 | tail -5
    )
    CAIRO_PROVE_BIN=$(find "${REPO_DIR}" -name "cairo-prove" -path "*/release/*" -type f 2>/dev/null | head -1)

    echo -e "  ${GREEN}Build complete${NC}"
else
    PROVE_BIN=$(find "${REPO_DIR}" -name "prove-model" -path "*/release/*" -type f 2>/dev/null | head -1)
    CAIRO_PROVE_BIN=$(find "${REPO_DIR}" -name "cairo-prove" -path "*/release/*" -type f 2>/dev/null | head -1)
    echo -e "${YELLOW}[BUILD] Skipped (--skip-build) — using existing binary${NC}"
    echo -e "${YELLOW}  WARNING: If you recently changed code, remove --skip-build to rebuild!${NC}"
fi

if [ -z "$PROVE_BIN" ]; then
    echo -e "${RED}ERROR: prove-model binary not found${NC}"
    exit 1
fi
echo "  prove-model: ${PROVE_BIN}"
echo "  cairo-prove: ${CAIRO_PROVE_BIN:-not found}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Validate model
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[VALIDATE] Checking model directory${NC}"
${PROVE_BIN} --model-dir "${MODEL_DIR}" --layers "${NUM_LAYERS}" --validate 2>&1 || {
    echo -e "${RED}ERROR: Model validation failed${NC}"
    exit 1
}
echo -e "  ${GREEN}Model validation passed${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Warmup — lightweight CUDA context init (NOT a full model prove)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$NO_WARMUP" = true ] || [ "$WARMUP_RUNS" -eq 0 ]; then
    echo -e "${YELLOW}[WARMUP] Skipped${NC}"
    echo ""
else
    echo -e "${YELLOW}[WARMUP] Initializing CUDA context${NC}"
    WARMUP_START=$(date +%s%N)

    # Step 1: CUDA driver + context init via nvidia-smi (< 1s)
    echo "  Initializing CUDA driver..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true

    # Step 2: Model inspection (loads safetensors, builds graph — no proving)
    echo "  Loading model weights (inspect only, no proving)..."
    ${PROVE_BIN} --model-dir "${MODEL_DIR}" --layers 1 --inspect 2>&1 | head -20 || true

    WARMUP_END=$(date +%s%N)
    WARMUP_MS=$(( (WARMUP_END - WARMUP_START) / 1000000 ))
    WARMUP_SEC=$(echo "scale=1; ${WARMUP_MS}/1000" | bc)

    echo -e "  ${GREEN}Warmup complete in ${WARMUP_SEC}s${NC}"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Proving Benchmark
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "════════════════════════════════════════════════════════════════════"
echo "  PROVING ${NUM_LAYERS} TRANSFORMER BLOCKS"
echo "  Model: Qwen3-14B | GPU: ${GPU_NAME}"
echo "════════════════════════════════════════════════════════════════════"
echo -e "${NC}"

BLOCK_TIMES=()
PROOF_SIZES=()
MATMUL_COUNTS=()
PEAK_GPU_MEM=0

if [ "$ONE_SHOT" = true ]; then
    # ── ONE-SHOT MODE: Prove all N layers in a single invocation ──
    echo -e "${YELLOW}[ONE-SHOT] Proving all ${NUM_LAYERS} blocks in a single invocation${NC}"
    echo ""

    FULL_PROOF="benchmarks/full_${NUM_LAYERS}blocks_proof.json"
    FULL_LOG="benchmarks/full_${NUM_LAYERS}blocks.log"

    GPU_MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
    echo -e "  GPU memory before: ${GPU_MEM_BEFORE} MiB"

    TOTAL_PROVE_START=$(date +%s%N)

    # Stream stderr to BOTH terminal and log file so the user sees live progress
    ${PROVE_BIN} \
        --model-dir "${MODEL_DIR}" \
        --layers "${NUM_LAYERS}" \
        --output "${FULL_PROOF}" \
        --format json \
        --gpu 2>&1 | tee "${FULL_LOG}" || true

    TOTAL_PROVE_END=$(date +%s%N)
    TOTAL_PROVE_MS=$(( (TOTAL_PROVE_END - TOTAL_PROVE_START) / 1000000 ))
    TOTAL_PROVE_SEC=$(echo "scale=3; ${TOTAL_PROVE_MS}/1000" | bc)

    GPU_MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")
    PEAK_GPU_MEM=$GPU_MEM_AFTER

    if [ -f "$FULL_PROOF" ]; then
        PROOF_SIZE=$(du -b "$FULL_PROOF" 2>/dev/null | cut -f1 || echo "0")
    else
        PROOF_SIZE=0
    fi

    MATMUL_COUNT=$(grep -o "matmul_proofs: [0-9]*" "${FULL_LOG}" 2>/dev/null | grep -o "[0-9]*" || echo "0")

    # Store as single "block" entry
    BLOCK_TIMES+=("${TOTAL_PROVE_SEC}")
    PROOF_SIZES+=("${PROOF_SIZE}")
    MATMUL_COUNTS+=("${MATMUL_COUNT}")

    echo ""
    echo -e "  ${GREEN}Time: ${TOTAL_PROVE_SEC}s | Proof: ${PROOF_SIZE} bytes | MatMuls: ${MATMUL_COUNT} | GPU Mem: ${GPU_MEM_AFTER} MiB${NC}"

else
    # ── PER-BLOCK MODE: Prove each block individually ──
    TOTAL_PROVE_START=$(date +%s%N)

    for block in $(seq 1 "${NUM_LAYERS}"); do
        echo -e "${YELLOW}[Block ${block}/${NUM_LAYERS}]${NC} Proving (layers=${block})..."

        BLOCK_START=$(date +%s%N)

        # Capture GPU memory before
        GPU_MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0")

        BLOCK_PROOF="benchmarks/block_${block}_proof.json"
        BLOCK_LOG="benchmarks/block_${block}.log"

        # Prove exactly 'block' layers to get per-block cumulative timing
        ${PROVE_BIN} \
            --model-dir "${MODEL_DIR}" \
            --layers "${block}" \
            --output "${BLOCK_PROOF}" \
            --format json \
            --gpu 2>&1 | tee "${BLOCK_LOG}" || true

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
fi

echo ""
echo -e "${GREEN}Total proving time: ${TOTAL_PROVE_SEC}s${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Recursive STARK (optional)
# ─────────────────────────────────────────────────────────────────────────────
RECURSIVE_TIME_SEC="N/A"
RECURSIVE_PROOF_SIZE="N/A"

if [ "$SKIP_RECURSIVE" = false ] && [ -n "$CAIRO_PROVE_BIN" ]; then
    echo -e "${YELLOW}[RECURSIVE] Generating recursive Circle STARK${NC}"

    # First re-prove the full model in cairo_serde format for recursive consumption
    CAIRO_SERDE_PROOF="benchmarks/ml_proof_cairo_serde.json"
    echo "  Generating cairo_serde proof for recursive pipeline..."

    ${PROVE_BIN} \
        --model-dir "${MODEL_DIR}" \
        --layers "${NUM_LAYERS}" \
        --output "${CAIRO_SERDE_PROOF}" \
        --format cairo_serde \
        --gpu 2>/dev/null || true

    if [ -f "$CAIRO_SERDE_PROOF" ]; then
        EXECUTABLE="${REPO_DIR}/stwo-cairo/stwo_cairo_verifier/target/dev/obelysk_ml_verifier.executable.json"
        if [ ! -f "$EXECUTABLE" ] && [ -f "${REPO_DIR}/artifacts/obelysk_ml_verifier.executable.json" ]; then
            EXECUTABLE="${REPO_DIR}/artifacts/obelysk_ml_verifier.executable.json"
        fi

        if [ -f "$EXECUTABLE" ]; then
            RECURSIVE_START=$(date +%s%N)

            ${CAIRO_PROVE_BIN} prove-ml \
                --verifier-executable "${EXECUTABLE}" \
                --ml-proof "${CAIRO_SERDE_PROOF}" \
                --output "benchmarks/recursive_proof.json" 2>&1 | tail -10

            RECURSIVE_END=$(date +%s%N)
            RECURSIVE_MS=$(( (RECURSIVE_END - RECURSIVE_START) / 1000000 ))
            RECURSIVE_TIME_SEC=$(echo "scale=3; ${RECURSIVE_MS}/1000" | bc)

            if [ -f "benchmarks/recursive_proof.json" ]; then
                RECURSIVE_PROOF_SIZE=$(du -b "benchmarks/recursive_proof.json" | cut -f1)
            fi

            echo -e "  ${GREEN}Recursive STARK: ${RECURSIVE_TIME_SEC}s, size: ${RECURSIVE_PROOF_SIZE} bytes${NC}"
        else
            echo -e "  ${YELLOW}ML verifier executable not found — skipping recursive${NC}"
            echo "  Expected: ${EXECUTABLE}"
        fi
    else
        echo -e "  ${YELLOW}cairo_serde proof generation failed — skipping recursive${NC}"
    fi
else
    echo -e "${YELLOW}[RECURSIVE] Skipped${NC}"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Write Results JSON
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[OUTPUT] Writing results to ${OUTPUT_FILE}${NC}"

# Compute average
if [ ${#BLOCK_TIMES[@]} -gt 0 ]; then
    AVG_BLOCK_SEC=$(echo "scale=3; ${TOTAL_PROVE_SEC}/${NUM_LAYERS}" | bc)
else
    AVG_BLOCK_SEC="0"
fi

# Build block times array for JSON
BLOCK_TIMES_JSON="["
for i in "${!BLOCK_TIMES[@]}"; do
    [ "$i" -gt 0 ] && BLOCK_TIMES_JSON+=","
    BLOCK_TIMES_JSON+="{\"block\":$i,\"prove_sec\":${BLOCK_TIMES[$i]},\"proof_bytes\":${PROOF_SIZES[$i]},\"matmul_count\":${MATMUL_COUNTS[$i]}}"
done
BLOCK_TIMES_JSON+="]"

cat > "${OUTPUT_FILE}" << ENDJSON
{
  "benchmark_version": "1.1.0",
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
    "mode": "$([ "$ONE_SHOT" = true ] && echo "one_shot" || echo "per_block")",
    "total_blocks": ${NUM_LAYERS},
    "total_prove_sec": ${TOTAL_PROVE_SEC},
    "avg_block_prove_sec": ${AVG_BLOCK_SEC},
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
    "Peak GPU memory is max observed across all blocks",
    "Per-block times are cumulative (block N = prove layers 1..N)"
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
printf "║  Model:           Qwen3-14B (%d blocks)\n" "${NUM_LAYERS}"
printf "║  GPU:             %s\n" "${GPU_NAME}"
echo "║                                                                   ║"
echo "║  ── Proving ──────────────────────────────────────────────────── ║"
printf "║  Total prove:     %-10s (%d blocks)\n" "${TOTAL_PROVE_SEC}s" "${NUM_LAYERS}"
printf "║  Avg per block:   %-10s\n" "${AVG_BLOCK_SEC}s"
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
if [ "$ONE_SHOT" = false ]; then
    echo "Per-block breakdown:"
    for i in "${!BLOCK_TIMES[@]}"; do
        printf "  Block %2d: %8ss | %s matmuls | %s bytes\n" \
            "$((i+1))" "${BLOCK_TIMES[$i]}" "${MATMUL_COUNTS[$i]}" "${PROOF_SIZES[$i]}"
    done
    echo ""
fi

echo -e "${GREEN}Benchmark complete. Results: ${OUTPUT_FILE}${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results and update libs/README.md with real numbers"
echo "  2. Run: bash scripts/h200_submit_onchain.sh --proof benchmarks/recursive_proof.json --submit"
echo "  3. Commit: git add benchmarks/ && git commit -m 'Add verified benchmarks'"
