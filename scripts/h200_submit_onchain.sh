#!/usr/bin/env bash
#
# Obelysk Protocol — H200 On-Chain STARK Proof Submission
# ========================================================
#
# Complete pipeline for submitting the recursive STARK proof on-chain
# from the H200 GPU worker (bitsage-worker).
#
# This script:
#   1. Sets up the H200 environment (CUDA, sncast)
#   2. Runs the full recursive pipeline (prove-qwen → cairo-prove) if --prove
#   3. Parses the proof and submits verify_recursive_output() on-chain
#
# Usage:
#   brev shell bitsage-worker
#   cd ~/stwo-ml   # or wherever the repo lives
#   bash scripts/h200_submit_onchain.sh --submit
#   bash scripts/h200_submit_onchain.sh --dry-run
#   bash scripts/h200_submit_onchain.sh --prove --submit  # full pipeline
#
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# H200 paths (Brev bitsage-worker)
STWO_ML_DIR="${STWO_ML_DIR:-$(pwd)}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3-14b}"
CAIRO_PROVE="${CAIRO_PROVE:-$HOME/stwo-cairo/cairo-prove/target/release/cairo-prove}"
PROVE_QWEN="${PROVE_QWEN:-./target/release/prove-qwen}"
EXECUTABLE="${EXECUTABLE:-cairo/stwo-ml-recursive/target/release/stwo_ml_recursive.executable.json}"
PROOF_FILE="${PROOF_FILE:-recursive_proof.json}"

# Starknet
ACCOUNT="${SNCAST_ACCOUNT:-deployer}"

# CUDA (critical for H200 — driver 550 needs 12.4 NVRTC)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu}"
export PATH="/usr/local/cuda-12.4/bin:$PATH"

# ═══════════════════════════════════════════════════════════════
# Parse arguments
# ═══════════════════════════════════════════════════════════════

DO_PROVE=false
DRY_RUN=false
DO_SUBMIT=false
SKIP_BUILD=false
NUM_LAYERS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --prove)       DO_PROVE=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --submit)      DO_SUBMIT=true; shift ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --layers)      NUM_LAYERS="$2"; shift 2 ;;
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --proof)       PROOF_FILE="$2"; shift 2 ;;
        --account)     ACCOUNT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--prove] [--submit|--dry-run] [--skip-build] [--layers N]"
            echo ""
            echo "Options:"
            echo "  --prove       Run GPU proving pipeline first (prove-qwen → cairo-prove)"
            echo "  --submit      Submit transactions on-chain (required for actual submission)"
            echo "  --dry-run     Print commands without executing"
            echo "  --skip-build  Skip building prove-qwen and cairo-prove"
            echo "  --layers N    Number of transformer layers to prove (default: 1)"
            echo "  --model-dir   Path to Qwen3-14B model directory"
            echo "  --proof       Path to recursive_proof.json (default: recursive_proof.json)"
            echo "  --account     sncast account name (default: deployer)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$DRY_RUN" = false ] && [ "$DO_SUBMIT" = false ]; then
    echo "ERROR: Must specify --dry-run or --submit"
    echo "  --dry-run: Print commands without executing"
    echo "  --submit:  Actually submit on-chain"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════════════════════════

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Obelysk Protocol — H200 On-Chain STARK Submission Pipeline  ║"
echo "║                                                               ║"
echo "║  GPU Prover → Circle STARK → verify_recursive_output()       ║"
echo "║  Qwen3-14B → Starknet Sepolia (1 tx, ~20K gas)              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ═══════════════════════════════════════════════════════════════
# Step 0: Environment
# ═══════════════════════════════════════════════════════════════

echo -e "${YELLOW}[Step 0] Environment${NC}"

# GPU check
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:    ${GPU_NAME} (${GPU_MEM})"
    echo "  Driver: ${DRIVER}"
    echo "  CUDA:   $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')"
else
    echo "  GPU: Not detected (CPU mode)"
fi

echo "  Rust:   $(rustc --version 2>/dev/null || echo 'not found')"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

# sncast check
if ! command -v sncast &>/dev/null; then
    echo -e "${YELLOW}  sncast not found. Installing starknet-foundry...${NC}"
    curl -L https://raw.githubusercontent.com/foundry-rs/starknet-foundry/master/scripts/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v sncast &>/dev/null; then
        echo -e "${RED}  ERROR: sncast still not found after install${NC}"
        echo "  Try: cargo install starknet-foundry"
        exit 1
    fi
fi
echo "  sncast: $(sncast --version 2>/dev/null || echo 'available')"
echo ""

# ═══════════════════════════════════════════════════════════════
# Step 1 (Optional): GPU Proving Pipeline
# ═══════════════════════════════════════════════════════════════

if [ "$DO_PROVE" = true ]; then
    echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  PHASE 1: GPU PROVING PIPELINE${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
    echo ""

    # Build prove-qwen
    if [ "$SKIP_BUILD" = false ]; then
        echo -e "${YELLOW}[1a] Building prove-qwen with GPU features${NC}"
        FEATURES="safetensors"
        if command -v nvidia-smi &>/dev/null; then
            FEATURES="safetensors,cuda-runtime"
        fi
        cargo build --release --bin prove-qwen --features "${FEATURES}" 2>&1 | tail -5
        echo -e "  ${GREEN}Built prove-qwen${NC}"
        echo ""
    fi

    # Run prove-qwen --recursive
    echo -e "${YELLOW}[1b] Running recursive proving pipeline${NC}"
    echo "  Model:  ${MODEL_DIR}"
    echo "  Layers: ${NUM_LAYERS}"
    echo "  Output: ${PROOF_FILE}"
    echo ""

    PROVE_CMD="${PROVE_QWEN} --model-dir ${MODEL_DIR} --layers ${NUM_LAYERS} --recursive"
    PROVE_CMD+=" --cairo-prove-bin ${CAIRO_PROVE}"
    PROVE_CMD+=" --executable ${EXECUTABLE}"
    PROVE_CMD+=" --proof-output ${PROOF_FILE}"

    echo "  $ ${PROVE_CMD}"
    time eval ${PROVE_CMD}

    echo ""

    # Verify locally
    if [ -f "$PROOF_FILE" ]; then
        echo -e "${YELLOW}[1c] Local STARK verification${NC}"
        ${CAIRO_PROVE} verify "${PROOF_FILE}" && \
            echo -e "  ${GREEN}LOCAL VERIFICATION: PASS${NC}" || \
            { echo -e "  ${RED}LOCAL VERIFICATION: FAIL${NC}"; exit 1; }
        echo ""
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Step 2: Parse Proof and Submit On-Chain
# ═══════════════════════════════════════════════════════════════

echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  PHASE 2: ON-CHAIN SUBMISSION${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

if [ ! -f "$PROOF_FILE" ]; then
    echo -e "${RED}ERROR: Proof file not found: ${PROOF_FILE}${NC}"
    echo "  Run with --prove to generate it, or specify --proof PATH"
    exit 1
fi

echo "  Proof file: ${PROOF_FILE} ($(du -h "${PROOF_FILE}" | cut -f1))"
echo ""

# Determine mode flag
MODE_FLAG="--dry-run"
if [ "$DO_SUBMIT" = true ]; then
    MODE_FLAG="--submit"
fi

# Find the submit script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_recursive_proof.py"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    # Try relative to current dir
    SUBMIT_SCRIPT="scripts/submit_recursive_proof.py"
fi

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo -e "${RED}ERROR: submit_recursive_proof.py not found${NC}"
    exit 1
fi

python3 "${SUBMIT_SCRIPT}" \
    --proof "${PROOF_FILE}" \
    --account "${ACCOUNT}" \
    ${MODE_FLAG}

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE                                           ║"
echo "║                                                               ║"
if [ "$DO_SUBMIT" = true ]; then
echo "║  Qwen3-14B inference verified on Starknet Sepolia            ║"
echo "║  via verify_recursive_output() — 1 tx, permanent record     ║"
else
echo "║  Dry run complete — re-run with --submit to execute          ║"
fi
echo "║                                                               ║"
echo "║  Contract: ${STARK_VERIFIER:0:20}...  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
