#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Generate Weight Commitment Cache
# ═══════════════════════════════════════════════════════════════════════
#
# Pre-computes Poseidon Merkle roots for all weight matrices and saves
# them to .stwo_weight_cache.swcf. Ship this file with model presets
# so subsequent proofs skip the 20-40 minute commitment phase.
#
# Usage:
#   bash scripts/pipeline/generate_weight_cache.sh --model-name qwen3-14b
#   bash scripts/pipeline/generate_weight_cache.sh --model-dir /path/to/model
#   bash scripts/pipeline/generate_weight_cache.sh --model-name qwen3-14b --upload
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────

MODEL_NAME=""
MODEL_DIR=""
NUM_LAYERS=""
MODEL_ID="0x1"
USE_GPU=true
SKIP_BUILD=false
UPLOAD=false

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)   MODEL_NAME="$2"; shift 2 ;;
        --model-dir)    MODEL_DIR="$2"; shift 2 ;;
        --layers)       NUM_LAYERS="$2"; shift 2 ;;
        --model-id)     MODEL_ID="$2"; shift 2 ;;
        --no-gpu)       USE_GPU=false; shift ;;
        --skip-build)   SKIP_BUILD=true; shift ;;
        --upload)       UPLOAD=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Pre-compute weight commitment cache for a model."
            echo ""
            echo "Model source (pick one):"
            echo "  --model-name NAME  Load from ~/.obelysk/models/NAME/"
            echo "  --model-dir DIR    Path to model directory"
            echo ""
            echo "Options:"
            echo "  --layers N         Number of transformer layers (default: from config)"
            echo "  --model-id ID      Model ID (default: 0x1)"
            echo "  --no-gpu           Use CPU for Merkle root computation"
            echo "  --skip-build       Skip rebuilding prove-model binary"
            echo "  --upload           Upload cache to release artifacts (requires gh CLI)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Output:"
            echo "  <model_dir>/.stwo_weight_cache.swcf"
            echo ""
            echo "This cache file should be shipped alongside model presets so that"
            echo "first-time provers skip the ~20-40 minute weight commitment phase."
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Resolve Model ──────────────────────────────────────────────────

init_obelysk_dir

if [[ -z "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_NAME=$(get_state "model_state.env" "CURRENT_MODEL" 2>/dev/null || echo "")
    if [[ -z "$MODEL_NAME" ]]; then
        err "No model specified. Use --model-name or --model-dir."
        exit 1
    fi
fi

if [[ -n "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_CONFIG="${OBELYSK_DIR}/models/${MODEL_NAME}/config.env"
    if [[ -f "$MODEL_CONFIG" ]]; then
        set -a; source "$MODEL_CONFIG"; set +a
    else
        err "Config not found: ${MODEL_CONFIG}"
        err "Run 01_setup_model.sh --preset ${MODEL_NAME} first."
        exit 1
    fi
fi

[[ -n "$NUM_LAYERS" ]] && MODEL_LAYERS="$NUM_LAYERS"

check_dir "$MODEL_DIR" "Model directory not found: ${MODEL_DIR}" || exit 1

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

if [[ "$SKIP_BUILD" == "false" ]] && [[ -d "${LIBS_DIR}/stwo-ml" ]]; then
    FEATURES="cli,audit,model-loading,safetensors"
    if [[ "$USE_GPU" == "true" ]] && { command -v nvcc &>/dev/null || [[ -f /usr/local/cuda/bin/nvcc ]]; }; then
        FEATURES="cli,audit,model-loading,safetensors,cuda-runtime"
    fi
    log "Building prove-model (features: ${FEATURES})..."
    (export PATH="$HOME/.cargo/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"; \
     cd "${LIBS_DIR}/stwo-ml" && \
     cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -3) || true
    # Update binary path after build
    _BUILT_BIN="${LIBS_DIR}/stwo-ml/target/release/prove-model"
    [[ -f "$_BUILT_BIN" ]] && PROVE_BIN="$_BUILT_BIN"
fi

if [[ -z "$PROVE_BIN" ]] || [[ ! -f "$PROVE_BIN" ]]; then
    err "prove-model binary not found. Run 00_setup_gpu.sh first."
    exit 1
fi

# ─── Display Config ──────────────────────────────────────────────────

CACHE_FILE="${MODEL_DIR}/.stwo_weight_cache.swcf"

banner
echo -e "${BOLD}  Weight Cache Generation${NC}"
echo ""
log "Model:       ${MODEL_NAME:-$(basename "$MODEL_DIR")}"
log "Model dir:   ${MODEL_DIR}"
log "Layers:      ${MODEL_LAYERS:-all}"
log "GPU:         ${USE_GPU}"
log "prove-model: ${PROVE_BIN}"
log "Cache file:  ${CACHE_FILE}"
echo ""

# Check existing cache
if [[ -f "$CACHE_FILE" ]]; then
    CACHE_SIZE=$(du -h "$CACHE_FILE" | cut -f1)
    warn "Existing cache found (${CACHE_SIZE}). Will be updated with any missing entries."
fi

# ─── Generate Cache ──────────────────────────────────────────────────

header "Generating weight commitment cache"
timer_start "cache_gen"

CACHE_CMD=("${PROVE_BIN}" "--generate-cache" "--model-dir" "$MODEL_DIR" "--model-id" "$MODEL_ID")
if [[ -n "${MODEL_LAYERS:-}" ]]; then
    CACHE_CMD+=("--layers" "$MODEL_LAYERS")
fi
if [[ "$USE_GPU" == "true" ]]; then
    CACHE_CMD+=("--gpu")
fi

log "Command: ${CACHE_CMD[*]}"
echo ""

"${CACHE_CMD[@]}" 2>&1 | while IFS= read -r line; do
    log "$line"
done
CACHE_RC=${PIPESTATUS[0]}

if [[ ${CACHE_RC} -ne 0 ]]; then
    err "Cache generation failed (exit ${CACHE_RC})"
    exit ${CACHE_RC}
fi

ELAPSED=$(timer_elapsed "cache_gen")

# ─── Verify Cache ────────────────────────────────────────────────────

if [[ -f "$CACHE_FILE" ]]; then
    CACHE_SIZE=$(du -h "$CACHE_FILE" | cut -f1)
    ok "Cache generated: ${CACHE_FILE} (${CACHE_SIZE})"
else
    err "Cache file not found after generation: ${CACHE_FILE}"
    exit 1
fi

# ─── Upload (optional) ───────────────────────────────────────────────

if [[ "$UPLOAD" == "true" ]]; then
    header "Uploading cache to release artifacts"

    if ! command -v gh &>/dev/null; then
        err "gh CLI not found. Install it: https://cli.github.com/"
        exit 1
    fi

    CACHE_ARTIFACT="${MODEL_NAME:-custom}.stwo_weight_cache.swcf"
    cp "$CACHE_FILE" "/tmp/${CACHE_ARTIFACT}"

    # Upload as release asset to the latest release
    LATEST_TAG=$(gh release list --limit 1 --json tagName -q '.[0].tagName' 2>/dev/null || echo "")
    if [[ -n "$LATEST_TAG" ]]; then
        log "Uploading to release ${LATEST_TAG}..."
        gh release upload "$LATEST_TAG" "/tmp/${CACHE_ARTIFACT}" --clobber 2>&1 || {
            warn "Release upload failed. Creating a new release..."
            gh release create "cache-${MODEL_NAME:-v0}" "/tmp/${CACHE_ARTIFACT}" \
                --title "Weight Cache: ${MODEL_NAME:-custom}" \
                --notes "Pre-computed weight commitment cache for ${MODEL_NAME:-custom}." 2>&1
        }
        ok "Cache uploaded as release asset: ${CACHE_ARTIFACT}"
    else
        log "No existing release found. Creating cache release..."
        gh release create "cache-${MODEL_NAME:-v0}" "/tmp/${CACHE_ARTIFACT}" \
            --title "Weight Cache: ${MODEL_NAME:-custom}" \
            --notes "Pre-computed weight commitment cache for ${MODEL_NAME:-custom}." 2>&1
        ok "Cache release created with asset: ${CACHE_ARTIFACT}"
    fi

    rm -f "/tmp/${CACHE_ARTIFACT}"
fi

# ─── Summary ─────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  WEIGHT CACHE GENERATION COMPLETE                    ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Model:     %-39s ║\n" "${MODEL_NAME:-custom}"
printf "  ║  Cache:     %-39s ║\n" "${CACHE_FILE}"
printf "  ║  Size:      %-39s ║\n" "${CACHE_SIZE:-?}"
printf "  ║  Duration:  %-39s ║\n" "$(format_duration $ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next steps:                                         ║"
echo "  ║  1. Ship .swcf file with model presets               ║"
echo "  ║  2. Run 03_prove.sh — commitments will be instant    ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
