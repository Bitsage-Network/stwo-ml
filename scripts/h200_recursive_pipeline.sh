#!/usr/bin/env bash
#
# H200 Recursive ML Proof Pipeline
# ==================================
# Full pipeline: GPU proof gen → Cairo recursive verification → Circle STARK proof
#
# Pipeline:
#   prove_qwen (GPU, Rust)
#     → 4 matmul sumcheck proofs (Qwen3-14B block)
#     → serialize as RecursiveInput
#     → cairo-prove prove (Circle STARK of verification)
#     → recursive_proof.json
#
# Usage:
#   ssh h200
#   cd /path/to/bitsage-network/libs
#   bash ../scripts/h200_recursive_pipeline.sh [--skip-build] [--layers N] [--model-dir PATH]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
SKIP_BUILD=false
NUM_LAYERS=1
MODEL_DIR=""
PROOF_OUTPUT="recursive_proof.json"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)  SKIP_BUILD=true; shift ;;
        --layers)      NUM_LAYERS="$2"; shift 2 ;;
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --output)      PROOF_OUTPUT="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Obelysk Protocol — H200 Recursive STARK Pipeline   ║"
echo "║  Qwen3-14B → Circle STARK Proof                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ----------------------------------------------------------------
# Step 0: Environment checks
# ----------------------------------------------------------------
echo -e "${YELLOW}[Step 0] Environment${NC}"

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "  CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "  Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
else
    echo -e "  ${RED}WARNING: nvidia-smi not found. GPU features will be disabled.${NC}"
fi

# Check Rust
if ! command -v rustup &>/dev/null; then
    echo -e "${RED}ERROR: rustup not found. Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh${NC}"
    exit 1
fi
echo "  Rust: $(rustc --version 2>/dev/null || echo 'not found')"

# Check Scarb (for Cairo executable compilation)
if command -v scarb &>/dev/null; then
    echo "  Scarb: $(scarb --version 2>/dev/null)"
else
    echo -e "  ${YELLOW}Scarb not found — using pre-compiled executable if available${NC}"
fi

echo ""

# ----------------------------------------------------------------
# Step 1: Build cairo-prove
# ----------------------------------------------------------------
CAIRO_PROVE_BIN="${REPO_DIR}/stwo-cairo/cairo-prove/target/release/cairo-prove"

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[Step 1] Building cairo-prove${NC}"
    echo "  Directory: ${REPO_DIR}/stwo-cairo/cairo-prove"

    # cairo-prove needs a recent nightly (Rust 1.89+ for proc_macro_span)
    # Override the pinned toolchain if it's too old
    PINNED_TOOLCHAIN=""
    if [ -f "${REPO_DIR}/stwo-cairo/cairo-prove/rust-toolchain.toml" ]; then
        PINNED_TOOLCHAIN=$(grep channel "${REPO_DIR}/stwo-cairo/cairo-prove/rust-toolchain.toml" | sed 's/.*= "//;s/"//')
        echo "  Pinned toolchain: ${PINNED_TOOLCHAIN}"
    fi

    # Try building with pinned toolchain first, fallback to default nightly
    echo "  Building with RUSTUP_TOOLCHAIN=nightly..."
    (
        cd "${REPO_DIR}/stwo-cairo/cairo-prove"
        RUSTUP_TOOLCHAIN=nightly cargo build --release 2>&1 | tail -5
    ) || {
        echo -e "${RED}  Build with nightly failed. Trying with pinned toolchain...${NC}"
        (
            cd "${REPO_DIR}/stwo-cairo/cairo-prove"
            cargo build --release 2>&1 | tail -5
        )
    }

    if [ -f "$CAIRO_PROVE_BIN" ]; then
        echo -e "  ${GREEN}cairo-prove built successfully${NC}"
        echo "  Binary: ${CAIRO_PROVE_BIN}"
        echo "  Size: $(du -h "$CAIRO_PROVE_BIN" | cut -f1)"
    else
        echo -e "${RED}  ERROR: cairo-prove binary not found after build${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[Step 1] Skipping build (--skip-build)${NC}"
    if [ ! -f "$CAIRO_PROVE_BIN" ]; then
        echo -e "${RED}  ERROR: cairo-prove not found at ${CAIRO_PROVE_BIN}${NC}"
        echo "  Run without --skip-build to build it"
        exit 1
    fi
    echo "  Using existing: ${CAIRO_PROVE_BIN}"
fi
echo ""

# ----------------------------------------------------------------
# Step 2: Build stwo-ml with GPU features
# ----------------------------------------------------------------
PROVE_QWEN_BIN="${REPO_DIR}/stwo-ml-repo/crates/stwo-ml/target/release/prove-qwen"

if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}[Step 2] Building prove-qwen (stwo-ml + GPU)${NC}"
    echo "  Directory: ${REPO_DIR}/stwo-ml-repo/crates/stwo-ml"

    FEATURES="safetensors"
    if command -v nvidia-smi &>/dev/null; then
        FEATURES="safetensors,cuda-runtime"
        echo "  Features: ${FEATURES} (GPU enabled)"
    else
        echo "  Features: ${FEATURES} (CPU only)"
    fi

    (
        cd "${REPO_DIR}/stwo-ml-repo/crates/stwo-ml"
        RUSTUP_TOOLCHAIN=nightly cargo build --release --bin prove-qwen --features "${FEATURES}" 2>&1 | tail -5
    )

    if [ -f "$PROVE_QWEN_BIN" ]; then
        echo -e "  ${GREEN}prove-qwen built successfully${NC}"
    else
        # Check alternate target location
        ALT_BIN="${REPO_DIR}/stwo-ml-repo/target/release/prove-qwen"
        if [ -f "$ALT_BIN" ]; then
            PROVE_QWEN_BIN="$ALT_BIN"
            echo -e "  ${GREEN}prove-qwen built at ${ALT_BIN}${NC}"
        else
            echo -e "${RED}  ERROR: prove-qwen not found${NC}"
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}[Step 2] Skipping build (--skip-build)${NC}"
    # Check both possible locations
    if [ ! -f "$PROVE_QWEN_BIN" ]; then
        ALT_BIN="${REPO_DIR}/stwo-ml-repo/target/release/prove-qwen"
        if [ -f "$ALT_BIN" ]; then
            PROVE_QWEN_BIN="$ALT_BIN"
        else
            echo -e "${RED}  ERROR: prove-qwen not found${NC}"
            exit 1
        fi
    fi
    echo "  Using existing: ${PROVE_QWEN_BIN}"
fi
echo ""

# ----------------------------------------------------------------
# Step 3: Check Cairo recursive executable
# ----------------------------------------------------------------
EXECUTABLE="${REPO_DIR}/stwo-ml-repo/cairo/stwo-ml-recursive/target/release/stwo_ml_recursive.executable.json"

echo -e "${YELLOW}[Step 3] Cairo recursive executable${NC}"
if [ -f "$EXECUTABLE" ]; then
    echo "  Found: ${EXECUTABLE}"
    echo "  Size: $(du -h "$EXECUTABLE" | cut -f1)"
else
    echo "  Not found at expected path."
    if command -v scarb &>/dev/null; then
        echo "  Building with scarb..."
        (
            cd "${REPO_DIR}/stwo-ml-repo/cairo/stwo-ml-recursive"
            scarb build --release 2>&1 | tail -3
        )
        if [ -f "$EXECUTABLE" ]; then
            echo -e "  ${GREEN}Built successfully${NC}"
        else
            echo -e "${RED}  ERROR: Executable not found after build${NC}"
            exit 1
        fi
    else
        echo -e "${RED}  ERROR: scarb not available. Install: curl -L https://docs.swmansion.com/scarb/install.sh | sh${NC}"
        exit 1
    fi
fi
echo ""

# ----------------------------------------------------------------
# Step 4: Run the full pipeline
# ----------------------------------------------------------------
echo -e "${CYAN}"
echo "════════════════════════════════════════════════════════"
echo "  RUNNING FULL RECURSIVE PIPELINE"
echo "  Layers: ${NUM_LAYERS} | Output: ${PROOF_OUTPUT}"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

PROVE_CMD="${PROVE_QWEN_BIN} --layers ${NUM_LAYERS} --recursive"
PROVE_CMD+=" --cairo-prove-bin ${CAIRO_PROVE_BIN}"
PROVE_CMD+=" --executable ${EXECUTABLE}"
PROVE_CMD+=" --proof-output ${PROOF_OUTPUT}"

if [ -n "$MODEL_DIR" ]; then
    PROVE_CMD+=" --model-dir ${MODEL_DIR}"
fi

echo "  Command: ${PROVE_CMD}"
echo ""

eval ${PROVE_CMD}

echo ""

# ----------------------------------------------------------------
# Step 5: Verify the STARK proof locally
# ----------------------------------------------------------------
if [ -f "$PROOF_OUTPUT" ]; then
    echo -e "${YELLOW}[Step 5] Verifying Circle STARK proof${NC}"
    echo "  Proof file: ${PROOF_OUTPUT}"
    echo "  Size: $(du -h "$PROOF_OUTPUT" | cut -f1)"

    ${CAIRO_PROVE_BIN} verify "${PROOF_OUTPUT}" && {
        echo -e "  ${GREEN}VERIFICATION PASSED${NC}"
    } || {
        echo -e "  ${RED}VERIFICATION FAILED${NC}"
        exit 1
    }
else
    echo -e "${YELLOW}[Step 5] No proof file found at ${PROOF_OUTPUT}${NC}"
    echo "  Check the output above for errors."
fi

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE                                   ║"
echo "║                                                      ║"
echo "║  Circle STARK proof: ${PROOF_OUTPUT}"
echo "║                                                      ║"
echo "║  Next: Upload proof to Starknet for on-chain         ║"
echo "║  verification via the STWO verifier contract.        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
