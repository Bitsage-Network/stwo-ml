#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 3: Proof Generation
# ═══════════════════════════════════════════════════════════════════════
#
# Generates a cryptographic proof of ML model inference.
# Production-hardened mode: GKR only.
#
# Usage:
#   bash scripts/pipeline/03_prove.sh --model-dir ~/models/qwen3-14b --layers 1 --mode gkr
#   bash scripts/pipeline/03_prove.sh --model-name qwen3-14b --mode gkr --gpu --multi-gpu
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────

MODEL_NAME=""
MODEL_DIR=""
NUM_LAYERS=""
MODE="gkr"
INPUT_FILE=""
OUTPUT_DIR=""
MODEL_ID="0x1"
USE_GPU=true
MULTI_GPU=false
GPU_ONLY=false
CHUNK_BUDGET_GB=16
SECURITY="auto"
SKIP_BUILD=false
SKIP_COMMITMENT=false
GKR_FLAG=false
SALT=""
SERVER_URL=""
STARKNET_READY=false
GKR_V2=false
GKR_V3=false
GKR_V3_MODE2=false
GKR_V4=false
GKR_V4_MODE3=false
LEGACY_GKR_V1=false
SUBMIT=false

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)       MODEL_NAME="$2"; shift 2 ;;
        --model-dir)        MODEL_DIR="$2"; shift 2 ;;
        --layers)           NUM_LAYERS="$2"; shift 2 ;;
        --mode)             MODE="$2"; shift 2 ;;
        --input)            INPUT_FILE="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --model-id)         MODEL_ID="$2"; shift 2 ;;
        --gpu)              USE_GPU=true; shift ;;
        --no-gpu)           USE_GPU=false; shift ;;
        --multi-gpu)        MULTI_GPU=true; USE_GPU=true; shift ;;
        --gpu-only)         GPU_ONLY=true; USE_GPU=true; shift ;;
        --chunk-budget-gb)  CHUNK_BUDGET_GB="$2"; shift 2 ;;
        --security)         SECURITY="$2"; shift 2 ;;
        --skip-build)       SKIP_BUILD=true; shift ;;
        --skip-commitment)  SKIP_COMMITMENT=true; shift ;;
        --gkr)              GKR_FLAG=true; shift ;;
        --starknet-ready)   STARKNET_READY=true; shift ;;
        --gkr-v2)           GKR_V2=true; shift ;;
        --gkr-v3)           GKR_V3=true; shift ;;
        --gkr-v3-mode2)     GKR_V3_MODE2=true; GKR_V3=true; shift ;;
        --gkr-v4)           GKR_V4=true; shift ;;
        --gkr-v4-mode3)     GKR_V4_MODE3=true; GKR_V4=true; shift ;;
        --legacy-gkr-v1)    LEGACY_GKR_V1=true; shift ;;
        --submit)           SUBMIT=true; shift ;;
        --salt)             SALT="$2"; shift 2 ;;
        --server)           SERVER_URL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Generate a cryptographic proof of ML inference."
            echo ""
            echo "Model source (pick one):"
            echo "  --model-name NAME  Load from ~/.obelysk/models/NAME/"
            echo "  --model-dir DIR    Path to model directory"
            echo ""
            echo "Proof mode:"
            echo "  --mode MODE        gkr (default: gkr)"
            echo "    gkr:        prove-model → GKR calldata for verify_model_gkr()/v2()/v3()/v4()"
            echo ""
            echo "Options:"
            echo "  --layers N           Number of transformer layers (default: from config)"
            echo "  --input FILE         JSON input file (default: random input)"
            echo "  --output-dir DIR     Output directory (default: ~/.obelysk/proofs/<timestamp>)"
            echo "  --model-id ID        Model ID for on-chain claim (default: 0x1)"
            echo "  --gpu                Enable GPU acceleration (default)"
            echo "  --no-gpu             Disable GPU"
            echo "  --multi-gpu          Distribute across all GPUs"
            echo "  --gpu-only           Fail if any critical proving path falls back to CPU"
            echo "  --chunk-budget-gb N  Memory per chunk in GB (default: 16)"
            echo "  --security LEVEL     auto | tee | zk-only (default: auto)"
            echo "  --skip-build         Skip building binaries"
            echo "  --skip-commitment    Skip weight commitment (faster, can't submit on-chain)"
            echo "  --gkr                Enable GKR for LogUp verification"
            echo "  --submit             Enable on-chain submission mode (aggregated oracle sumcheck, ~17K felts)"
            echo "  --starknet-ready     (Deprecated) Legacy submit-ready gate mode"
            echo "  --gkr-v2             (Legacy) Emit verify_model_gkr_v2 calldata"
            echo "  --gkr-v3             (Legacy) Emit verify_model_gkr_v3 calldata"
            echo "  --gkr-v3-mode2       (Legacy) Enable verify_model_gkr_v3 mode=2"
            echo "  --gkr-v4             (Legacy) Emit verify_model_gkr_v4 calldata"
            echo "  --gkr-v4-mode3       (Legacy) Enable verify_model_gkr_v4 mode=3"
            echo "  --legacy-gkr-v1      (Legacy) Keep verify_model_gkr v1 sequential-opening submit path"
            echo "  --salt N             Fiat-Shamir channel salt"
            echo "  --server URL         Submit to remote prove-server instead of local binary"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate mode (fail-closed to full on-chain GKR verification).
if [[ "$MODE" != "gkr" ]]; then
    err "Only --mode gkr is supported in the hardened pipeline (got: ${MODE})"
    err "Use GKR mode for full on-chain cryptographic verification."
    exit 1
fi

# Deprecation path: bare --starknet-ready now maps to --submit (mode 4 aggregated).
# Legacy selectors still work when explicitly requested.
if [[ "$STARKNET_READY" == "true" ]] && [[ "$SUBMIT" != "true" ]]; then
    if [[ "$LEGACY_GKR_V1" != "true" ]] && [[ "$GKR_V2" != "true" ]] && [[ "$GKR_V3" != "true" ]] && [[ "$GKR_V3_MODE2" != "true" ]] && [[ "$GKR_V4" != "true" ]] && [[ "$GKR_V4_MODE3" != "true" ]]; then
        warn "--starknet-ready is deprecated; using --submit (verify_model_gkr_v4 mode 4, aggregated oracle sumcheck)."
        SUBMIT=true
    else
        warn "--starknet-ready is deprecated. Prefer --submit for the default path; legacy selectors remain opt-in."
    fi
fi

# ─── --submit: Aggregated Oracle Sumcheck (mode 4) ──────────────────
# --submit is the recommended path for on-chain submission.
# Forces aggregated weight binding (unified oracle mismatch sumcheck),
# producing ~17K felts calldata instead of ~2.4M (160 separate openings).
if [[ "$SUBMIT" == "true" ]]; then
    STARKNET_READY=true
    GKR_V4=true
    export STWO_WEIGHT_BINDING=aggregated
    export STWO_STARKNET_GKR_V4=1
    # Disable legacy modes that conflict with aggregated oracle sumcheck
    GKR_V2=false
    GKR_V3=false
    GKR_V3_MODE2=false
    GKR_V4_MODE3=false
    LEGACY_GKR_V1=false
    log "Aggregated oracle sumcheck mode enabled (--submit)"
    log "Estimated calldata: ~17K felts (vs ~2.4M for legacy per-opening mode)"
fi

# Safety: v2/v3 submission paths require Starknet-ready artifact gating.
if [[ "$GKR_V2" == "true" ]] && [[ "$STARKNET_READY" != "true" ]]; then
    warn "--gkr-v2 requested; enabling --starknet-ready for submit-ready artifact checks"
    STARKNET_READY=true
fi
if [[ "$GKR_V3" == "true" ]] && [[ "$STARKNET_READY" != "true" ]]; then
    warn "--gkr-v3 requested; enabling --starknet-ready for submit-ready artifact checks"
    STARKNET_READY=true
fi
if [[ "$GKR_V3_MODE2" == "true" ]] && [[ "$STARKNET_READY" != "true" ]]; then
    warn "--gkr-v3-mode2 requested; enabling --starknet-ready for submit-ready artifact checks"
    STARKNET_READY=true
fi
if [[ "$GKR_V4" == "true" ]] && [[ "$STARKNET_READY" != "true" ]]; then
    warn "--gkr-v4 requested; enabling --starknet-ready for submit-ready artifact checks"
    STARKNET_READY=true
fi
if [[ "$GKR_V4_MODE3" == "true" ]] && [[ "$STARKNET_READY" != "true" ]]; then
    warn "--gkr-v4-mode3 requested; enabling --starknet-ready for submit-ready artifact checks"
    STARKNET_READY=true
fi
if [[ "$GKR_V2" == "true" && "$GKR_V3" == "true" ]]; then
    warn "Both --gkr-v2 and --gkr-v3 were set; preferring higher entrypoint version."
    GKR_V2=false
fi
if [[ "$GKR_V4" == "true" ]] && [[ "$GKR_V3" == "true" ]]; then
    warn "Both --gkr-v3 and --gkr-v4 were set; preferring v4 entrypoint."
    GKR_V3=false
    GKR_V3_MODE2=false
fi

if [[ "$LEGACY_GKR_V1" == "true" ]] && [[ "$GKR_V2" == "true" || "$GKR_V3" == "true" || "$GKR_V3_MODE2" == "true" || "$GKR_V4" == "true" || "$GKR_V4_MODE3" == "true" ]]; then
    warn "--legacy-gkr-v1 ignored because a v2/v3/v4 entrypoint was explicitly selected."
    LEGACY_GKR_V1=false
fi

if [[ "$STARKNET_READY" == "true" ]] && [[ "$GKR_V2" != "true" ]] && [[ "$GKR_V3" != "true" ]] && [[ "$GKR_V4" != "true" ]]; then
    if [[ "${LEGACY_GKR_V1}" == "true" ]]; then
        warn "--starknet-ready + --legacy-gkr-v1: using verify_model_gkr (v1 sequential openings)."
    else
        err "--starknet-ready requested without explicit legacy selector."
        err "Use --submit for the default mode-4 aggregated path, or pass a legacy selector explicitly."
        exit 1
    fi
fi

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
        exit 1
    fi
fi

[[ -n "$NUM_LAYERS" ]] && MODEL_LAYERS="$NUM_LAYERS"

check_dir "$MODEL_DIR" "Model directory not found" || exit 1

# ─── Performance Defaults (Safe / Soundness-Preserving) ─────────────

# Fast defaults: leave strict/hardening OFF unless caller explicitly sets them.
GPU_COMMIT_STRICT="${STWO_GPU_COMMIT_STRICT:-off}"
GPU_COMMIT_HARDEN="${STWO_GPU_COMMIT_HARDEN:-off}"
GPU_POLY_STRICT="${STWO_GPU_POLY_STRICT:-off}"
GPU_POLY_HARDEN="${STWO_GPU_POLY_HARDEN:-off}"
GPU_MLE_MERKLE_REQUIRE="${STWO_GPU_MLE_MERKLE_REQUIRE:-off}"
GPU_MLE_FOLD_REQUIRE="${STWO_GPU_MLE_FOLD_REQUIRE:-off}"
GPU_MLE_OPENING_TREE="${STWO_GPU_MLE_OPENING_TREE:-on}"
GPU_MLE_OPENING_TREE_REQUIRE="${STWO_GPU_MLE_OPENING_TREE_REQUIRE:-off}"
GPU_MLE_OPENING_TIMING="${STWO_GPU_MLE_OPENING_TIMING:-off}"
UNIFIED_STARK_NO_FALLBACK="${STWO_UNIFIED_STARK_NO_FALLBACK:-off}"

# In pure ml_gkr mode, unified STARK is redundant for Starknet GKR calldata and
# can be skipped for a major speed/stability win. Caller can override to 0/off.
if [[ -z "${STWO_PURE_GKR_SKIP_UNIFIED_STARK:-}" ]]; then
    export STWO_PURE_GKR_SKIP_UNIFIED_STARK=1
fi
PURE_GKR_SKIP_UNIFIED_STARK="${STWO_PURE_GKR_SKIP_UNIFIED_STARK:-off}"
case "${PURE_GKR_SKIP_UNIFIED_STARK,,}" in
    1|true|on|yes) PURE_GKR_SKIP_UNIFIED_STARK="on" ;;
    *) PURE_GKR_SKIP_UNIFIED_STARK="off" ;;
esac

# Default to fast aggregated weight binding for off-chain proving.
# For submit-ready Starknet calldata, caller can pass --starknet-ready or --submit.
# --submit uses aggregated oracle sumcheck (mode 4) which IS submit-ready.
if [[ "$SUBMIT" == "true" ]]; then
    # Aggregated oracle sumcheck handles weight binding via unified oracle.
    # Disable old RLC mode — it conflicts with the new protocol.
    GKR_AGG_WEIGHT_BINDING="off"
    export STWO_GKR_AGGREGATE_WEIGHT_BINDING="off"
else
    GKR_AGG_WEIGHT_BINDING_DEFAULT="on"
    if [[ "$STARKNET_READY" == "true" ]]; then
        GKR_AGG_WEIGHT_BINDING_DEFAULT="off"
    fi
    GKR_AGG_WEIGHT_BINDING="${STWO_GKR_AGGREGATE_WEIGHT_BINDING:-${GKR_AGG_WEIGHT_BINDING_DEFAULT}}"

    case "${GKR_AGG_WEIGHT_BINDING,,}" in
        1|true|on|yes) GKR_AGG_WEIGHT_BINDING="on" ;;
        *) GKR_AGG_WEIGHT_BINDING="off" ;;
    esac

    if [[ "$STARKNET_READY" == "true" ]] && [[ "${GKR_AGG_WEIGHT_BINDING}" == "on" ]]; then
        warn "Overriding STWO_GKR_AGGREGATE_WEIGHT_BINDING=on -> off (--starknet-ready requested)"
        GKR_AGG_WEIGHT_BINDING="off"
    fi
    export STWO_GKR_AGGREGATE_WEIGHT_BINDING="${GKR_AGG_WEIGHT_BINDING}"
fi

if [[ "$GKR_V2" == "true" ]]; then
    export STWO_STARKNET_GKR_V2=1
fi
if [[ "$GKR_V3" == "true" ]]; then
    export STWO_STARKNET_GKR_V3=1
fi
if [[ "$GKR_V3_MODE2" == "true" ]]; then
    export STWO_GKR_TRUSTLESS_MODE2=1
fi
if [[ "$GKR_V4" == "true" ]]; then
    export STWO_STARKNET_GKR_V4=1
fi
if [[ "$GKR_V4_MODE3" == "true" ]]; then
    export STWO_GKR_TRUSTLESS_MODE3=1
fi

# Batched sub-channel weight openings:
# - Safe and submit-ready with verify_model_gkr_v2/v3 (weight_binding_mode=1)
# - Also used by v3 mode2 trustless path for opening transcript derivation.
# - Keep v1 (`verify_model_gkr`) on sequential openings only.
GKR_BATCH_WEIGHT_OPENINGS_DEFAULT="off"
if [[ "$USE_GPU" == "true" ]] && [[ "$STARKNET_READY" == "true" ]] && { [[ "$GKR_V2" == "true" ]] || [[ "$GKR_V3" == "true" ]] || [[ "$GKR_V4" == "true" ]]; }; then
    GKR_BATCH_WEIGHT_OPENINGS_DEFAULT="on"
fi
GKR_BATCH_WEIGHT_OPENINGS="${STWO_GKR_BATCH_WEIGHT_OPENINGS:-${GKR_BATCH_WEIGHT_OPENINGS_DEFAULT}}"
case "${GKR_BATCH_WEIGHT_OPENINGS,,}" in
    1|true|on|yes) GKR_BATCH_WEIGHT_OPENINGS="on" ;;
    *) GKR_BATCH_WEIGHT_OPENINGS="off" ;;
esac

if [[ "$STARKNET_READY" == "true" ]] && [[ "$GKR_V2" != "true" ]] && [[ "$GKR_V3" != "true" ]] && [[ "$GKR_V4" != "true" ]] && [[ "${GKR_BATCH_WEIGHT_OPENINGS}" == "on" ]]; then
    warn "Overriding STWO_GKR_BATCH_WEIGHT_OPENINGS=on -> off (verify_model_gkr v1 requires Sequential openings)"
    GKR_BATCH_WEIGHT_OPENINGS="off"
fi
if [[ "${GKR_AGG_WEIGHT_BINDING}" == "on" ]] && [[ "${GKR_BATCH_WEIGHT_OPENINGS}" == "on" ]]; then
    warn "Disabling STWO_GKR_BATCH_WEIGHT_OPENINGS because aggregated RLC mode eliminates opening proofs."
    GKR_BATCH_WEIGHT_OPENINGS="off"
fi
export STWO_GKR_BATCH_WEIGHT_OPENINGS="${GKR_BATCH_WEIGHT_OPENINGS}"

# Favor GPU fold for heavy weight-opening phases unless caller overrides.
if [[ "$USE_GPU" == "true" ]] && [[ -z "${STWO_GPU_MLE_FOLD:-}" ]]; then
    export STWO_GPU_MLE_FOLD=1
fi
if [[ "$USE_GPU" == "true" ]] && [[ -z "${STWO_GPU_MLE_FOLD_MIN_POINTS:-}" ]]; then
    export STWO_GPU_MLE_FOLD_MIN_POINTS=1048576
fi
if [[ "$USE_GPU" == "true" ]] && [[ -z "${STWO_GPU_MLE_OPENING_TREE:-}" ]]; then
    export STWO_GPU_MLE_OPENING_TREE=1
fi
GPU_MLE_FOLD="${STWO_GPU_MLE_FOLD:-auto}"
GPU_MLE_FOLD_MIN_POINTS="${STWO_GPU_MLE_FOLD_MIN_POINTS:-default}"
GPU_MLE_OPENING_TREE="${STWO_GPU_MLE_OPENING_TREE:-auto}"

if [[ "$GPU_ONLY" == "true" ]]; then
    export STWO_GPU_ONLY=1
    export STWO_UNIFIED_STARK_NO_FALLBACK=1
    export STWO_GPU_MLE_MERKLE_REQUIRE=1
    export STWO_GPU_MLE_FOLD_REQUIRE=1
    export STWO_GPU_MLE_OPENING_TREE=1
    export STWO_GPU_MLE_OPENING_TREE_REQUIRE=1
    export STWO_GPU_MLE_FOLD=1
    export STWO_GPU_POLY_STRICT=1
    UNIFIED_STARK_NO_FALLBACK="on"
    GPU_MLE_MERKLE_REQUIRE="on"
    GPU_MLE_FOLD_REQUIRE="on"
    GPU_MLE_OPENING_TREE_REQUIRE="on"
    GPU_POLY_STRICT="on"
fi

# Single-GPU default: keep commitment/proving serialized to avoid GPU contention.
if [[ "$USE_GPU" == "true" ]] && [[ "$MULTI_GPU" != "true" ]]; then
    if [[ -n "${STWO_PARALLEL_GPU_COMMIT:-}" ]]; then
        if [[ "${STWO_PARALLEL_GPU_COMMIT}" == "1" ]]; then
            warn "STWO_PARALLEL_GPU_COMMIT=1 enabled: overlap can increase contention on single GPU."
        fi
    else
        unset STWO_PARALLEL_GPU_COMMIT
    fi
fi

# Thread defaults for CPU fallback sections.
if [[ -z "${RAYON_NUM_THREADS:-}" ]]; then
    export RAYON_NUM_THREADS="$(nproc 2>/dev/null || echo 4)"
fi
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    export OMP_NUM_THREADS="${RAYON_NUM_THREADS}"
fi

# Keep progress updates dense so long phases are visibly alive.
if [[ -z "${STWO_WEIGHT_PROGRESS_EVERY:-}" ]]; then
    export STWO_WEIGHT_PROGRESS_EVERY=1
fi
if [[ -z "${STWO_GKR_OPENINGS_PROGRESS_EVERY:-}" ]]; then
    export STWO_GKR_OPENINGS_PROGRESS_EVERY=1
fi
if [[ -z "${STWO_GKR_OPENING_HEARTBEAT_SEC:-}" ]]; then
    export STWO_GKR_OPENING_HEARTBEAT_SEC=15
fi

# Optional protocol-level speed mode (off by default).
# This removes per-weight Merkle openings and uses a batched RLC weight-binding check.
# Proof artifacts remain serializable, but Starknet submission is disabled by soundness gates.
if [[ "${GKR_AGG_WEIGHT_BINDING,,}" == "1" || "${GKR_AGG_WEIGHT_BINDING,,}" == "true" || "${GKR_AGG_WEIGHT_BINDING,,}" == "on" ]]; then
    warn "STWO_GKR_AGGREGATE_WEIGHT_BINDING is enabled."
    warn "This mode is off-chain only today: artifact is serialized, but Starknet submission is rejected by soundness gates."
fi

# ─── Output Directory ───────────────────────────────────────────────

if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OBELYSK_DIR}/proofs/${MODEL_NAME:-proof}_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

# ─── Find Binaries ──────────────────────────────────────────────────

INSTALL_DIR=$(get_state "setup_state.env" "INSTALL_DIR" 2>/dev/null || echo "$HOME/obelysk")
LIBS_DIR="${INSTALL_DIR}"

# prove-model
PROVE_BIN=$(get_state "setup_state.env" "PROVE_MODEL_BIN" 2>/dev/null || echo "")
if [[ -z "$PROVE_BIN" ]] || [[ ! -f "$PROVE_BIN" ]]; then
    PROVE_BIN=$(find_binary "prove-model" "$LIBS_DIR" 2>/dev/null || echo "")
fi
if [[ -z "$PROVE_BIN" ]] || [[ ! -f "$PROVE_BIN" ]]; then
    PROVE_BIN=$(command -v prove-model 2>/dev/null || echo "")
fi
if [[ -z "$PROVE_BIN" ]]; then
    err "prove-model not found. Run 00_setup_gpu.sh first."
    exit 1
fi

# Optional rebuild
if [[ "$SKIP_BUILD" == "false" ]] && [[ -d "${LIBS_DIR}/stwo-ml" ]]; then
    FEATURES="cli,audit,model-loading,safetensors"
    if [[ "$USE_GPU" == "true" ]] && { command -v nvcc &>/dev/null || [[ -f /usr/local/cuda/bin/nvcc ]]; }; then
        FEATURES="cli,audit,model-loading,safetensors,cuda-runtime"
    fi
    log "Rebuilding prove-model (features: ${FEATURES})..."
    (export PATH="$HOME/.cargo/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"; cd "${LIBS_DIR}/stwo-ml" && cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -3) || true
fi

# ─── Display Config ─────────────────────────────────────────────────

banner
echo -e "${BOLD}  Proof Generation${NC}"
echo ""
GPU_COMMIT_PARALLEL="off"
if [[ -n "${STWO_PARALLEL_GPU_COMMIT:-}" ]]; then
    GPU_COMMIT_PARALLEL="on"
fi
UNIFIED_STARK_FALLBACK="on"
if [[ "${UNIFIED_STARK_NO_FALLBACK,,}" =~ ^(1|true|on|yes)$ ]]; then
    UNIFIED_STARK_FALLBACK="off"
fi
log "Mode:           ${MODE}"
log "Model:          ${MODEL_NAME:-$(basename "$MODEL_DIR")}"
log "Model dir:      ${MODEL_DIR}"
log "Layers:         ${MODEL_LAYERS:-all}"
log "GPU:            ${USE_GPU} (multi: ${MULTI_GPU})"
log "Output:         ${OUTPUT_DIR}"
log "prove-model:    ${PROVE_BIN}"
log "Threads:        RAYON=${RAYON_NUM_THREADS} OMP=${OMP_NUM_THREADS}"
log "Submit mode:    ${SUBMIT}"
log "Starknet ready: ${STARKNET_READY}"
if [[ "$SUBMIT" == "true" ]]; then
    log "Weight binding: aggregated_oracle_sumcheck (mode 4)"
fi
if [[ "$GKR_V4" == "true" ]]; then
    _GKR_ENTRYPOINT="verify_model_gkr_v4"
elif [[ "$GKR_V3" == "true" ]]; then
    _GKR_ENTRYPOINT="verify_model_gkr_v3"
elif [[ "$GKR_V2" == "true" ]]; then
    _GKR_ENTRYPOINT="verify_model_gkr_v2"
else
    _GKR_ENTRYPOINT="verify_model_gkr"
fi
log "GKR entrypoint: ${_GKR_ENTRYPOINT}"
log "Legacy v1 path: ${LEGACY_GKR_V1}"
log "GPU only mode:  ${GPU_ONLY}"
log "GPU commit:     strict=${GPU_COMMIT_STRICT} harden=${GPU_COMMIT_HARDEN} parallel=${GPU_COMMIT_PARALLEL}"
log "GPU poly path:  strict=${GPU_POLY_STRICT} harden=${GPU_POLY_HARDEN}"
log "Unified STARK:  gpu_constraints_fallback=${UNIFIED_STARK_FALLBACK}"
log "Pure GKR:       skip_unified_stark=${PURE_GKR_SKIP_UNIFIED_STARK}"
log "GPU MLE path:   fold=${GPU_MLE_FOLD} fold_min_points=${GPU_MLE_FOLD_MIN_POINTS} opening_tree=${GPU_MLE_OPENING_TREE} merkle_require=${GPU_MLE_MERKLE_REQUIRE} fold_require=${GPU_MLE_FOLD_REQUIRE} opening_tree_require=${GPU_MLE_OPENING_TREE_REQUIRE}"
log "GPU opening:    qm31_pack=device (enabled by default)"
log "GPU opening dbg: timing=${GPU_MLE_OPENING_TIMING}"
log "Weight binding: aggregate_rlc=${GKR_AGG_WEIGHT_BINDING}"
log "Weight binding: trustless_mode2=${STWO_GKR_TRUSTLESS_MODE2:-off}"
log "Weight binding: trustless_mode3=${STWO_GKR_TRUSTLESS_MODE3:-off}"
log "Weight openings: batch_subchannel=${GKR_BATCH_WEIGHT_OPENINGS} jobs=${STWO_GKR_BATCH_WEIGHT_OPENING_JOBS:-auto}"
log "Progress:       weight_every=${STWO_WEIGHT_PROGRESS_EVERY} opening_every=${STWO_GKR_OPENINGS_PROGRESS_EVERY} opening_heartbeat=${STWO_GKR_OPENING_HEARTBEAT_SEC}s"
echo ""

timer_start "prove_total"

# ─── Format Mapping ─────────────────────────────────────────────────

FORMAT="ml_gkr"

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: GPU ML Proof (all modes start here)
# ═══════════════════════════════════════════════════════════════════════

header "Phase 1: GPU ML Proof Generation"
timer_start "phase1"

ML_PROOF="${OUTPUT_DIR}/ml_proof.json"

# ── Remote server mode ──────────────────────────────────────────────
if [[ -n "$SERVER_URL" ]]; then
    log "Submitting to remote prove-server: ${SERVER_URL}"

    JOB_ID=$(submit_prove_job "$SERVER_URL" "$MODEL_ID" "$USE_GPU" "$SECURITY")
    if [[ -z "$JOB_ID" ]]; then
        err "Failed to submit prove job to ${SERVER_URL}"
        exit 1
    fi
    ok "Job submitted: ${JOB_ID}"

    POLL_STATUS=$(poll_prove_job "$SERVER_URL" "$JOB_ID" 5 3600)
    if [[ "$POLL_STATUS" != "completed" ]]; then
        err "Prove job did not complete successfully"
        exit 1
    fi

    fetch_prove_result "$SERVER_URL" "$JOB_ID" "$ML_PROOF"

    PHASE1_SEC=$(timer_elapsed "phase1")
    if [[ -f "$ML_PROOF" ]]; then
        ML_PROOF_SIZE=$(du -h "$ML_PROOF" | cut -f1)
        ok "Remote proof received in $(format_duration $PHASE1_SEC)"
        ok "Output: ${ML_PROOF} (${ML_PROOF_SIZE})"
    else
        err "Failed to fetch proof result"
        exit 1
    fi
    echo ""

    # Skip to Phase 2 (remote server handles proving)
    PHASE2_SEC=0
    FINAL_PROOF="$ML_PROOF"

    # Jump directly to metadata save
    TOTAL_SEC=$(timer_elapsed "prove_total")
    cat > "${OUTPUT_DIR}/metadata.json" << METAEOF
{
    "model_name": "${MODEL_NAME:-custom}",
    "model_dir": "${MODEL_DIR}",
    "mode": "remote",
    "server_url": "${SERVER_URL}",
    "job_id": "${JOB_ID}",
    "model_id": "${MODEL_ID}",
    "gpu": ${USE_GPU},
    "security": "${SECURITY}",
    "total_seconds": ${TOTAL_SEC},
    "ml_proof": "ml_proof.json",
    "ml_proof_size": "${ML_PROOF_SIZE:-?}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
METAEOF
    ok "Metadata saved to ${OUTPUT_DIR}/metadata.json"
    save_state "prove_state.env" \
        "LAST_PROOF_DIR=${OUTPUT_DIR}" \
        "LAST_PROOF_MODE=remote" \
        "LAST_PROOF_FILE=${FINAL_PROOF}" \
        "LAST_PROOF_MODEL=${MODEL_NAME:-custom}"
    echo ""
    echo -e "${GREEN}${BOLD}  Remote proof generation complete in $(format_duration $TOTAL_SEC)${NC}"
    exit 0
fi

# ── Local binary mode ───────────────────────────────────────────────

PROVE_CMD=("${PROVE_BIN}")
PROVE_CMD+=("--output" "$ML_PROOF")
PROVE_CMD+=("--format" "$FORMAT")
PROVE_CMD+=("--model-id" "$MODEL_ID")
PROVE_CMD+=("--security" "$SECURITY")

# Model source
if [[ -f "${MODEL_ONNX:-}" ]]; then
    PROVE_CMD+=("--model" "$MODEL_ONNX")
else
    PROVE_CMD+=("--model-dir" "$MODEL_DIR")
fi

# Layers
if [[ -n "${MODEL_LAYERS:-}" ]]; then
    PROVE_CMD+=("--layers" "$MODEL_LAYERS")
fi

# Input
if [[ -n "$INPUT_FILE" ]]; then
    PROVE_CMD+=("--input" "$INPUT_FILE")
fi

# GPU options
if [[ "$USE_GPU" == "true" ]]; then
    PROVE_CMD+=("--gpu")
fi
if [[ "$MULTI_GPU" == "true" ]]; then
    PROVE_CMD+=("--multi-gpu" "--chunk-budget-gb" "$CHUNK_BUDGET_GB")
fi

# Extra flags
if [[ "$SKIP_COMMITMENT" == "true" ]]; then
    PROVE_CMD+=("--skip-commitment")
fi
if [[ "$GKR_FLAG" == "true" ]]; then
    PROVE_CMD+=("--gkr")
fi
if [[ -n "$SALT" ]]; then
    PROVE_CMD+=("--salt" "$SALT")
fi

log "Command: ${PROVE_CMD[*]}"
echo ""

# Run with progress monitoring
if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] Would execute: ${PROVE_CMD[*]}"
    log "[DRY RUN] Skipping actual proof generation."
    # Create a minimal placeholder so downstream steps know dry-run happened
    mkdir -p "$OUTPUT_DIR"
    python3 -c "
import json, sys
json.dump({
    'dry_run': True,
    'command': sys.argv[1:],
    'verify_calldata': None,
}, open('${OUTPUT_DIR}/ml_proof.json', 'w'), indent=2)
" "${PROVE_CMD[@]}"
    ok "[DRY RUN] Placeholder proof written to ${OUTPUT_DIR}/ml_proof.json"
else
    RAW_LOG="${OUTPUT_DIR}/prove_model.raw.log"
    log "Raw prover log: ${RAW_LOG}"

    # Run proving, keep a full raw log, and stream key lines live.
    # We intentionally disable `set -e` around the pipeline so we can surface
    # the real prover exit code + error tail instead of silently exiting.
    set +e
    "${PROVE_CMD[@]}" 2>&1 | tee "$RAW_LOG" | while IFS= read -r line; do
        # Show progress + fatal diagnostics lines
        # Keep this broad so long-running phases don't look "stuck".
        if echo "$line" | grep -qE '\[layer|\[BG\]|Proving|Weight commit|Phase|Forward pass|GKR|Unified STARK|\[[0-9]+/[0-9]+\]|\[prewarm\]|\[CACHE\]|\[commit\]|aggregated|cache|opening|Merkle|root|TEE:|Security:|Error:|error:|panic|thread .+ panicked|Segmentation fault|completed|elapsed'; then
            log "$line"
        fi
    done
    PROVE_RC=${PIPESTATUS[0]}
    set -e

    if [[ ${PROVE_RC} -ne 0 ]]; then
        err "prove-model failed (exit ${PROVE_RC})"
        err "Last 120 lines from raw log:"
        tail -120 "$RAW_LOG" | sed 's/^/[prove-model] /'
        exit ${PROVE_RC}
    fi
fi

PHASE1_SEC=$(timer_elapsed "phase1")

if [[ -f "$ML_PROOF" ]]; then
    ML_PROOF_SIZE=$(du -h "$ML_PROOF" | cut -f1)
    ok "ML proof generated in $(format_duration $PHASE1_SEC)"
    ok "Output: ${ML_PROOF} (${ML_PROOF_SIZE})"
else
    err "ML proof not generated"
    exit 1
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Mode-specific processing
# ═══════════════════════════════════════════════════════════════════════

PHASE2_SEC=0
FINAL_PROOF="$ML_PROOF"

case "$MODE" in
    # ─── Recursive: ML proof → Cairo VM → Circle STARK ───────────────
    recursive)
        header "Phase 2: Recursive Circle STARK"
        timer_start "phase2"

        RECURSIVE_PROOF="${OUTPUT_DIR}/recursive_proof.json"

        # Find Cairo verifier executable
        EXECUTABLE=""
        for path in \
            "${LIBS_DIR}/stwo-cairo/stwo_cairo_verifier/target/dev/obelysk_ml_verifier.executable.json" \
            "${INSTALL_DIR}/artifacts/obelysk_ml_verifier.executable.json"; do
            if [[ -f "$path" ]]; then
                EXECUTABLE="$path"
                break
            fi
        done

        if [[ -z "$EXECUTABLE" ]]; then
            err "Cairo verifier executable not found."
            err "Build it: cd stwo-cairo/stwo_cairo_verifier && scarb build"
            exit 1
        fi

        RECURSIVE_CMD=("${CAIRO_BIN}" "prove-ml")
        RECURSIVE_CMD+=("--verifier-executable" "$EXECUTABLE")
        RECURSIVE_CMD+=("--ml-proof" "$ML_PROOF")
        RECURSIVE_CMD+=("--output" "$RECURSIVE_PROOF")

        log "Command: ${RECURSIVE_CMD[*]}"
        echo ""

        run_cmd "${RECURSIVE_CMD[@]}"

        PHASE2_SEC=$(timer_elapsed "phase2")

        if [[ -f "$RECURSIVE_PROOF" ]]; then
            RECURSIVE_SIZE=$(du -h "$RECURSIVE_PROOF" | cut -f1)
            ok "Recursive STARK generated in $(format_duration $PHASE2_SEC)"
            ok "Output: ${RECURSIVE_PROOF} (${RECURSIVE_SIZE})"
            FINAL_PROOF="$RECURSIVE_PROOF"

            # Verify locally
            log "Verifying STARK proof locally..."
            if run_cmd "${CAIRO_BIN}" verify "$RECURSIVE_PROOF" 2>&1; then
                ok "Local verification: PASS"
            else
                err "Local verification: FAIL"
                exit 1
            fi
        else
            err "Recursive proof not generated"
            exit 1
        fi
        ;;

    # ─── GKR: proof ready as-is + local verification ────────────────
    gkr)
        header "Phase 2: GKR Mode — Local Verification"
        log "Verifying GKR proof locally before saving..."

        if "$PROVE_BIN" --verify-proof "$ML_PROOF" 2>&1 | tail -5; then
            ok "Local GKR proof verification: PASS"
        else
            warn "Local verification not available (--verify-proof may not be supported)"
            log "Checking proof structure..."
            # Fallback: verify JSON structure
            if python3 -c "
import json, sys
with open('${ML_PROOF}') as f:
    proof = json.load(f)
fmt = proof.get('format', '')
if fmt != 'ml_gkr':
    print(f'Warning: format is {fmt}, expected ml_gkr', file=sys.stderr)
vc = proof.get('verify_calldata')
assert isinstance(vc, dict), 'Missing verify_calldata object'
sv = vc.get('schema_version')
assert sv in (1, 2), f'verify_calldata.schema_version must be 1 or 2 (got {sv})'
entrypoint = vc.get('entrypoint')
assert isinstance(entrypoint, str) and len(entrypoint) > 0, 'verify_calldata.entrypoint must be non-empty string'
ready = bool(proof.get('submission_ready', False))

def parse_nat(tok, label):
    s = str(tok)
    try:
        v = int(s, 0)
    except Exception as e:
        raise AssertionError(f'invalid {label}: {s} ({e})')
    if v < 0:
        raise AssertionError(f'{label} must be >= 0 (got {v})')
    return v

if sv == 2:
    # Schema v2: chunked session protocol
    assert entrypoint == 'verify_gkr_from_session', \
        f'schema_version=2 requires entrypoint=verify_gkr_from_session (got {entrypoint})'
    assert vc.get('mode') == 'chunked', f'schema_version=2 requires mode=chunked (got {vc.get(\"mode\")})'
    chunks = vc.get('chunks')
    assert isinstance(chunks, list) and len(chunks) > 0, 'schema_version=2 requires non-empty chunks array'
    total_felts = vc.get('total_felts')
    assert isinstance(total_felts, int) and total_felts > 0, f'schema_version=2 requires total_felts > 0 (got {total_felts})'
    circuit_depth = vc.get('circuit_depth')
    assert isinstance(circuit_depth, int) and circuit_depth > 0, f'schema_version=2 requires circuit_depth > 0 (got {circuit_depth})'
    num_layers = vc.get('num_layers')
    assert isinstance(num_layers, int) and num_layers > 0, f'schema_version=2 requires num_layers > 0 (got {num_layers})'
    wb_mode = vc.get('weight_binding_mode')
    assert wb_mode in (3, 4), f'schema_version=2 requires weight_binding_mode in (3,4) (got {wb_mode})'
    packed = vc.get('packed')
    assert isinstance(packed, bool), f'schema_version=2 requires packed to be boolean (got {packed})'
    print(f'  verify_calldata: schema_version=2, {total_felts} felts in {len(chunks)} chunks (packed={packed})', file=sys.stderr)
elif entrypoint in ('verify_model_gkr', 'verify_model_gkr_v2', 'verify_model_gkr_v3', 'verify_model_gkr_v4', 'verify_model_gkr_v4_packed'):
    calldata = vc.get('calldata')
    chunks = vc.get('upload_chunks', [])
    assert isinstance(chunks, list), 'verify_calldata.upload_chunks must be an array'
    assert ready is True, f'{entrypoint} requires submission_ready=true'
    assert isinstance(calldata, list) and len(calldata) > 0, 'verify_calldata.calldata must be non-empty array'
    assert len(chunks) == 0, 'verify_model_gkr(*) should not include upload chunks'
    assert all(str(v) != '__SESSION_ID__' for v in calldata), 'verify_model_gkr(*) calldata must not include __SESSION_ID__ placeholder'
    mode = proof.get('weight_opening_mode')
    mode_s = None if mode is None else str(mode)
    if entrypoint == 'verify_model_gkr':
        if mode_s is not None:
            assert mode_s == 'Sequential', f'{entrypoint} requires weight_opening_mode=Sequential (got {mode})'
    elif entrypoint in ('verify_model_gkr_v2', 'verify_model_gkr_v3'):
        if mode_s is not None:
            assert mode_s in ('Sequential', 'BatchedSubchannelV1'), \
                f'{entrypoint} requires weight_opening_mode in (Sequential, BatchedSubchannelV1) (got {mode})'
    elif entrypoint in ('verify_model_gkr_v4', 'verify_model_gkr_v4_packed'):
        if mode_s is not None:
            allowed_v4 = ('AggregatedOpeningsV4Experimental', 'AggregatedOracleSumcheck')
            assert mode_s in allowed_v4, \
                f'{entrypoint} requires weight_opening_mode in {allowed_v4} (got {mode})'
    if entrypoint in ('verify_model_gkr_v2', 'verify_model_gkr_v3', 'verify_model_gkr_v4', 'verify_model_gkr_v4_packed'):
        # v2/v3 calldata inserts weight_binding_mode after weight_commitments array.
        idx = 0
        idx += 1  # model_id
        assert idx < len(calldata), 'calldata truncated before raw_io length'
        raw_io_len = parse_nat(calldata[idx], 'raw_io_data length')
        idx += 1 + raw_io_len
        idx += 2  # circuit_depth, num_layers
        assert idx < len(calldata), 'calldata truncated before matmul_dims length'
        matmul_len = parse_nat(calldata[idx], 'matmul_dims length')
        idx += 1 + matmul_len
        assert idx < len(calldata), 'calldata truncated before dequantize_bits length'
        deq_len = parse_nat(calldata[idx], 'dequantize_bits length')
        idx += 1 + deq_len
        assert idx < len(calldata), 'calldata truncated before proof_data length'
        proof_data_len = parse_nat(calldata[idx], 'proof_data length')
        idx += 1 + proof_data_len
        assert idx < len(calldata), 'calldata truncated before weight_commitments length'
        wc_len = parse_nat(calldata[idx], 'weight_commitments length')
        idx += 1 + wc_len
        assert idx < len(calldata), 'calldata truncated before weight_binding_mode'
        wb_mode = parse_nat(calldata[idx], 'weight_binding_mode')
        if entrypoint == 'verify_model_gkr_v2':
            assert wb_mode in (0, 1), \
                f'{entrypoint} requires weight_binding_mode in (0,1) (got {wb_mode})'
        if entrypoint in ('verify_model_gkr_v4', 'verify_model_gkr_v4_packed'):
            assert wb_mode in (3, 4), \
                f'{entrypoint} requires weight_binding_mode in (3, 4) (got {wb_mode})'
        expected_mode = None
        if mode_s == 'Sequential':
            expected_mode = 0
        elif mode_s == 'BatchedSubchannelV1':
            expected_mode = 1
        elif mode_s == 'AggregatedTrustlessV2':
            expected_mode = 2
        elif mode_s == 'AggregatedOpeningsV4Experimental':
            expected_mode = 3
        elif mode_s == 'AggregatedOracleSumcheck':
            expected_mode = 4
        if expected_mode is not None:
            assert wb_mode == expected_mode, \
                f'{entrypoint} expected weight_binding_mode={expected_mode} for weight_opening_mode={mode_s} (got {wb_mode})'
        else:
            allowed_modes = (0, 1, 2) if entrypoint == 'verify_model_gkr_v3' else (3, 4) if entrypoint in ('verify_model_gkr_v4', 'verify_model_gkr_v4_packed') else (0, 1)
            assert wb_mode in allowed_modes, \
                f'{entrypoint} requires weight_binding_mode in {allowed_modes} (got {wb_mode})'
        artifact_mode_id = proof.get('weight_binding_mode_id')
        if artifact_mode_id is not None:
            artifact_mode_id = int(str(artifact_mode_id), 0)
            assert artifact_mode_id == wb_mode, \
                f'weight_binding_mode_id mismatch: artifact={artifact_mode_id} calldata={wb_mode}'
        if entrypoint in ('verify_model_gkr_v3', 'verify_model_gkr_v4', 'verify_model_gkr_v4_packed'):
            idx += 1  # consume weight_binding_mode
            assert idx < len(calldata), f'{entrypoint} calldata truncated before weight_binding_data length'
            binding_data_len = parse_nat(calldata[idx], 'weight_binding_data length')
            idx += 1 + binding_data_len
            if wb_mode in (0, 1):
                assert binding_data_len == 0, \
                    f'{entrypoint} mode {wb_mode} requires empty weight_binding_data (got len={binding_data_len})'
            elif wb_mode == 2:
                assert binding_data_len > 0, \
                    f'{entrypoint} mode 2 requires non-empty weight_binding_data'
            elif wb_mode == 3:
                assert binding_data_len > 0, \
                    f'{entrypoint} mode 3 requires non-empty weight_binding_data'
            elif wb_mode == 4:
                assert binding_data_len > 0, \
                    f'{entrypoint} mode 4 (aggregated oracle sumcheck) requires non-empty weight_binding_data'
            artifact_binding_data = proof.get('weight_binding_data_calldata')
            if isinstance(artifact_binding_data, list):
                assert len(artifact_binding_data) == binding_data_len, \
                    f'weight_binding_data_calldata length mismatch: artifact={len(artifact_binding_data)} calldata={binding_data_len}'
    print(f'  verify_calldata: {len(calldata)} felts', file=sys.stderr)
else:
    # Off-chain / experimental transcript modes are serializable but not submit-ready.
    assert entrypoint == 'unsupported', f'unsupported verify_calldata.entrypoint: {entrypoint}'
    calldata = vc.get('calldata')
    assert isinstance(calldata, list), 'verify_calldata.calldata must be an array'
    mode = proof.get('weight_opening_mode', 'unknown')
    reason = vc.get('reason') or proof.get('soundness_gate_error') or 'unspecified'
    assert ready is False, 'unsupported verify_calldata must correspond to submission_ready=false'
    claims = proof.get('weight_claim_calldata', [])
    assert isinstance(claims, list) and len(claims) > 0, 'weight_claim_calldata must be present in unsupported mode'
    print(f'  verify_calldata unavailable for submission (mode={mode}): {reason}', file=sys.stderr)
print(f'  model_id: {proof.get("model_id", "?")}', file=sys.stderr)
" 2>&1; then
                ok "GKR proof structure valid"
            else
                err "GKR proof structure invalid"
                exit 1
            fi
        fi
        ok "GKR proof ready"
        ;;
esac
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Save Metadata
# ═══════════════════════════════════════════════════════════════════════

TOTAL_SEC=$(timer_elapsed "prove_total")

cat > "${OUTPUT_DIR}/metadata.json" << METAEOF
{
    "model_name": "${MODEL_NAME:-custom}",
    "model_dir": "${MODEL_DIR}",
    "mode": "${MODE}",
    "layers": "${MODEL_LAYERS:-all}",
    "model_id": "${MODEL_ID}",
    "gpu": ${USE_GPU},
    "multi_gpu": ${MULTI_GPU},
    "gkr_v2": ${GKR_V2},
    "gkr_v3": ${GKR_V3},
    "gkr_v3_mode2": ${GKR_V3_MODE2},
    "gkr_v4": ${GKR_V4},
    "gkr_v4_mode3": ${GKR_V4_MODE3},
    "legacy_gkr_v1": ${LEGACY_GKR_V1},
    "submit": ${SUBMIT},
    "weight_binding": "${STWO_WEIGHT_BINDING:-legacy}",
    "security": "${SECURITY}",
    "phase1_seconds": ${PHASE1_SEC},
    "phase2_seconds": ${PHASE2_SEC},
    "total_seconds": ${TOTAL_SEC},
    "ml_proof": "ml_proof.json",
    "ml_proof_size": "${ML_PROOF_SIZE:-?}",
    "final_proof": "$(basename "$FINAL_PROOF")",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "prove_model_bin": "${PROVE_BIN}",
    "gpu_name": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
}
METAEOF

ok "Metadata saved to ${OUTPUT_DIR}/metadata.json"

# Save to global state
save_state "prove_state.env" \
    "LAST_PROOF_DIR=${OUTPUT_DIR}" \
    "LAST_PROOF_MODE=${MODE}" \
    "LAST_PROOF_FILE=${FINAL_PROOF}" \
    "LAST_PROOF_MODEL=${MODEL_NAME:-custom}"

# ─── Summary ─────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  PROOF GENERATION COMPLETE                           ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Model:        %-36s ║\n" "${MODEL_NAME:-custom}"
printf "  ║  Mode:         %-36s ║\n" "${MODE}"
printf "  ║  Phase 1 (ML): %-36s ║\n" "$(format_duration $PHASE1_SEC)"
if [[ "$MODE" == "recursive" ]]; then
printf "  ║  Phase 2 (STARK): %-33s ║\n" "$(format_duration $PHASE2_SEC)"
fi
printf "  ║  Total:        %-36s ║\n" "$(format_duration $TOTAL_SEC)"
printf "  ║  Output:       %-36s ║\n" "${OUTPUT_DIR}"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next: ./04_verify_onchain.sh                        ║"
echo "  ║    --proof-dir ${OUTPUT_DIR}"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
