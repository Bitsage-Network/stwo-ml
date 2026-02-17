#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 1: Model Setup
# ═══════════════════════════════════════════════════════════════════════
#
# Downloads and configures a model for proving.
#
# Usage:
#   bash scripts/pipeline/01_setup_model.sh --preset qwen3-14b
#   bash scripts/pipeline/01_setup_model.sh --hf-model Qwen/Qwen3-0.5B --layers 24
#   bash scripts/pipeline/01_setup_model.sh --onnx /path/to/model.onnx
#   bash scripts/pipeline/01_setup_model.sh --list
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/model_registry.sh"

# ─── Defaults ────────────────────────────────────────────────────────

PRESET=""
HF_MODEL=""
ONNX_PATH=""
MODEL_DIR_OVERRIDE=""
LAYERS_OVERRIDE=""
QUANT_OVERRIDE=""
SKIP_DOWNLOAD=false
LIST_ONLY=false
CONFIG_FILE=""
HF_TOKEN_ARG=""

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)         PRESET="$2"; shift 2 ;;
        --hf-model)       HF_MODEL="$2"; shift 2 ;;
        --onnx)           ONNX_PATH="$2"; shift 2 ;;
        --model-dir)      MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
        --layers)         LAYERS_OVERRIDE="$2"; shift 2 ;;
        --quantize)       QUANT_OVERRIDE="$2"; shift 2 ;;
        --config)         CONFIG_FILE="$2"; shift 2 ;;
        --hf-token)       HF_TOKEN_ARG="$2"; shift 2 ;;
        --skip-download)  SKIP_DOWNLOAD=true; shift ;;
        --list)           LIST_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Download and configure a model for proving."
            echo ""
            echo "Model source (pick one):"
            echo "  --preset NAME      Load a built-in preset (e.g., qwen3-14b)"
            echo "  --hf-model REPO    Custom HuggingFace repo (e.g., Qwen/Qwen3-0.5B)"
            echo "  --onnx PATH        Local ONNX model file"
            echo "  --config FILE      Custom .env config file"
            echo ""
            echo "Options:"
            echo "  --list             Print available presets and exit"
            echo "  --layers N         Override number of layers to prove"
            echo "  --quantize STRAT   Quantization: symmetric8, asymmetric8, direct, int4"
            echo "  --model-dir DIR    Override download location"
            echo "  --hf-token TOKEN   HuggingFace API token (for gated models like Llama)"
            echo "  --skip-download    Model already exists locally"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Environment variables:"
            echo "  HF_TOKEN=...       HuggingFace token (alternative to --hf-token)"
            echo ""
            echo "Examples:"
            echo "  $0 --preset qwen3-14b"
            echo "  $0 --hf-model Qwen/Qwen3-0.5B --layers 24"
            echo "  $0 --list"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── List Mode ───────────────────────────────────────────────────────

if [[ "$LIST_ONLY" == "true" ]]; then
    list_presets
    exit 0
fi

# ─── Load Configuration ─────────────────────────────────────────────

init_obelysk_dir

if [[ -n "$CONFIG_FILE" ]]; then
    # Custom config file
    log "Loading config from ${CONFIG_FILE}..."
    load_config_env "$CONFIG_FILE"
elif [[ -n "$PRESET" ]]; then
    # Built-in preset
    log "Loading preset: ${PRESET}..."

    # Try configs/ directory first, then built-in
    config_path=$(find_config "$PRESET" "$SCRIPT_DIR" 2>/dev/null || echo "")
    if [[ -n "$config_path" ]]; then
        load_config_env "$config_path"
    else
        get_preset "$PRESET" || exit 1
    fi
elif [[ -n "$HF_MODEL" ]]; then
    # Custom HF model
    MODEL_HF="$HF_MODEL"
    # Derive name from repo
    MODEL_NAME=$(echo "$HF_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    MODEL_QUANT="${QUANT_OVERRIDE:-symmetric8}"
    MODEL_SIZE_GB="?"
    MODEL_DESCRIPTION="Custom: ${HF_MODEL}"
elif [[ -n "$ONNX_PATH" ]]; then
    # ONNX model
    MODEL_ONNX="$ONNX_PATH"
    MODEL_NAME=$(basename "$ONNX_PATH" .onnx)
    MODEL_QUANT="${QUANT_OVERRIDE:-symmetric8}"
    MODEL_DESCRIPTION="ONNX: ${ONNX_PATH}"
else
    err "No model source specified."
    err "Use --preset, --hf-model, --onnx, or --config."
    echo ""
    list_presets
    exit 1
fi

# Apply overrides
[[ -n "$LAYERS_OVERRIDE" ]] && MODEL_LAYERS="$LAYERS_OVERRIDE"
[[ -n "$QUANT_OVERRIDE" ]] && MODEL_QUANT="$QUANT_OVERRIDE"

# ─── Determine Model Directory ───────────────────────────────────────

if [[ -n "$MODEL_DIR_OVERRIDE" ]]; then
    MODEL_DIR="$MODEL_DIR_OVERRIDE"
elif [[ -n "${MODEL_ONNX:-}" ]]; then
    MODEL_DIR="$(dirname "${MODEL_ONNX}")"
else
    MODEL_DIR="${OBELYSK_DIR}/models/${MODEL_NAME}"
fi

# ─── Display Configuration ───────────────────────────────────────────

banner
echo -e "${BOLD}  Model Setup: ${MODEL_NAME:-custom}${NC}"
echo ""
print_model_config
log "Model directory: ${MODEL_DIR}"
echo ""

timer_start "model_setup"

# ─── HuggingFace Authentication ──────────────────────────────────────

_HF_TOKEN="${HF_TOKEN_ARG:-${HF_TOKEN:-}}"

# Fallback: read from ~/.huggingface/token
if [[ -z "$_HF_TOKEN" ]] && [[ -f "$HOME/.huggingface/token" ]]; then
    _HF_TOKEN=$(cat "$HOME/.huggingface/token" 2>/dev/null | tr -d '[:space:]')
    debug "HF token loaded from ~/.huggingface/token"
fi

if [[ -n "$_HF_TOKEN" ]]; then
    log "HuggingFace authentication: token found"
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli login --token "$_HF_TOKEN" 2>/dev/null || true
        ok "HuggingFace CLI logged in"
    fi
    export HF_TOKEN="$_HF_TOKEN"
else
    # Check if model is likely gated (Llama, etc.)
    case "${MODEL_HF:-}" in
        *llama*|*Llama*|*gemma*|*Gemma*)
            warn "No HF token found. This model may be gated and require authentication."
            warn "  Set HF_TOKEN env var or use --hf-token TOKEN"
            warn "  Get a token at: https://huggingface.co/settings/tokens"
            ;;
    esac
fi

# ─── Disk Space Check ───────────────────────────────────────────────

if [[ -n "${MODEL_HF:-}" ]] && [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    _REQUIRED_GB="${MODEL_SIZE_GB:-10}"
    if [[ "$_REQUIRED_GB" != "?" ]]; then
        _REQUIRED=$(( _REQUIRED_GB * 3 / 2 ))  # 1.5x for download + extraction
        # Use parent dir or $HOME if MODEL_DIR doesn't exist yet
        _DF_TARGET="${MODEL_DIR:-$HOME}"
        [[ -d "$_DF_TARGET" ]] || _DF_TARGET="$HOME"
        _AVAILABLE_GB=$(df -BG "$_DF_TARGET" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo "")
        if [[ -n "$_AVAILABLE_GB" ]] && (( _AVAILABLE_GB < _REQUIRED )); then
            err "Insufficient disk space: need ~${_REQUIRED}GB, only ${_AVAILABLE_GB}GB available."
            err "Free up space or use a larger disk."
            exit 1
        fi
    fi
fi

# ─── Download (HuggingFace Models) ───────────────────────────────────

if [[ -n "${MODEL_HF:-}" ]] && [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    header "Downloading ${MODEL_HF}"

    mkdir -p "${MODEL_DIR}"

    # Check if already downloaded
    if [[ -f "${MODEL_DIR}/config.json" ]] && ls "${MODEL_DIR}"/*.safetensors &>/dev/null 2>&1; then
        SHARD_COUNT=$(ls "${MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
        TOTAL_SIZE=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)
        ok "Model already downloaded (${SHARD_COUNT} shards, ${TOTAL_SIZE})"
    else
        log "Downloading ${MODEL_HF} to ${MODEL_DIR}..."
        if [[ "${MODEL_SIZE_GB:-?}" != "?" ]]; then
            log "Expected size: ~${MODEL_SIZE_GB}GB"
        fi
        echo ""

        # Method 1: huggingface-cli
        if command -v huggingface-cli &>/dev/null; then
            run_cmd huggingface-cli download "${MODEL_HF}" \
                --local-dir "${MODEL_DIR}" \
                --include "*.safetensors" "config.json" "tokenizer.json" "tokenizer_config.json" \
                --quiet
        # Method 2: Python API
        elif python3 -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MODEL_HF}',
    local_dir='${MODEL_DIR}',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'],
)
print('Download complete')
"
        # Method 3: git lfs
        else
            if ! git lfs version &>/dev/null; then
                err "huggingface_hub not available and git-lfs is not installed."
                err "Install git-lfs: sudo apt-get install git-lfs  (or run 00_setup_gpu.sh first)"
                exit 1
            fi
            warn "huggingface_hub not available, falling back to git lfs..."
            run_cmd git lfs install 2>/dev/null || true
            GIT_LFS_SKIP_SMUDGE=0 run_cmd git clone --depth 1 \
                "https://huggingface.co/${MODEL_HF}" "${MODEL_DIR}"
        fi

        SHARD_COUNT=$(ls "${MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
        TOTAL_SIZE=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)
        ok "Downloaded: ${SHARD_COUNT} shards, ${TOTAL_SIZE}"

        # Integrity verification
        log "Verifying download integrity..."
        _INTEGRITY_OK=true

        # Check all safetensors are non-zero
        for st_file in "${MODEL_DIR}"/*.safetensors; do
            if [[ -f "$st_file" ]] && [[ ! -s "$st_file" ]]; then
                err "Empty safetensors file: $(basename "$st_file")"
                _INTEGRITY_OK=false
            fi
        done

        # Check shard count matches index if present
        if [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
            _EXPECTED_SHARDS=$(python3 -c "
import json
with open('${MODEL_DIR}/model.safetensors.index.json') as f:
    idx = json.load(f)
files = set(idx.get('weight_map', {}).values())
print(len(files))
" 2>/dev/null || echo "0")
            if [[ "$_EXPECTED_SHARDS" != "0" ]] && [[ "$SHARD_COUNT" != "$_EXPECTED_SHARDS" ]]; then
                err "Shard count mismatch: found ${SHARD_COUNT}, expected ${_EXPECTED_SHARDS}"
                _INTEGRITY_OK=false
            fi
        fi

        # Verify config.json is valid JSON
        if [[ -f "${MODEL_DIR}/config.json" ]]; then
            if ! python3 -c "import json; json.load(open('${MODEL_DIR}/config.json'))" 2>/dev/null; then
                err "config.json is not valid JSON"
                _INTEGRITY_OK=false
            fi
        fi

        if [[ "$_INTEGRITY_OK" == "true" ]]; then
            ok "Download integrity verified"
        else
            err "Download integrity check FAILED. Re-download with: $0 --preset ${PRESET:-custom}"
            exit 1
        fi
    fi

elif [[ -n "${MODEL_ONNX:-}" ]]; then
    # ONNX mode — just verify the file exists
    check_file "${MODEL_ONNX}" "ONNX model not found: ${MODEL_ONNX}" || exit 1
    ok "ONNX model: ${MODEL_ONNX}"

elif [[ "$SKIP_DOWNLOAD" == "true" ]]; then
    log "Skipping download (--skip-download)"
    check_dir "${MODEL_DIR}" "Model directory not found: ${MODEL_DIR}" || exit 1
fi
echo ""

# ─── Verify Model Files ─────────────────────────────────────────────

header "Verifying model files"

if [[ -n "${MODEL_HF:-}" ]]; then
    # HF model checks
    check_file "${MODEL_DIR}/config.json" "config.json not found in ${MODEL_DIR}" || exit 1

    SHARD_COUNT=$(ls "${MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    if (( SHARD_COUNT == 0 )); then
        err "No .safetensors files found in ${MODEL_DIR}"
        exit 1
    fi

    ok "config.json present"
    ok "${SHARD_COUNT} safetensors shards"

    # Extract key info from config.json
    if command -v python3 &>/dev/null; then
        DETECTED_LAYERS=$(parse_json_field "${MODEL_DIR}/config.json" "num_hidden_layers")
        DETECTED_HIDDEN=$(parse_json_field "${MODEL_DIR}/config.json" "hidden_size")
        DETECTED_HEADS=$(parse_json_field "${MODEL_DIR}/config.json" "num_attention_heads")
        DETECTED_KV_HEADS=$(parse_json_field "${MODEL_DIR}/config.json" "num_key_value_heads")
        DETECTED_ARCH=$(parse_json_field "${MODEL_DIR}/config.json" "architectures.0")

        log "Architecture:  ${DETECTED_ARCH:-unknown}"
        log "Hidden size:   ${DETECTED_HIDDEN:-?}"
        log "Layers:        ${DETECTED_LAYERS:-?}"
        log "Heads:         ${DETECTED_HEADS:-?} (KV: ${DETECTED_KV_HEADS:-?})"

        # Auto-fill missing values
        [[ -z "${MODEL_LAYERS:-}" ]] && MODEL_LAYERS="${DETECTED_LAYERS}"
        [[ -z "${MODEL_HIDDEN:-}" ]] && MODEL_HIDDEN="${DETECTED_HIDDEN}"
        [[ -z "${MODEL_HEADS:-}" ]] && MODEL_HEADS="${DETECTED_HEADS}"
        [[ -z "${MODEL_KV_HEADS:-}" ]] && MODEL_KV_HEADS="${DETECTED_KV_HEADS}"
    fi
fi
echo ""

# ─── Save Model Config ──────────────────────────────────────────────

header "Saving configuration"

MODEL_CONFIG_DIR="${OBELYSK_DIR}/models/${MODEL_NAME:-custom}"
mkdir -p "${MODEL_CONFIG_DIR}"

cat > "${MODEL_CONFIG_DIR}/config.env" << CFGEOF
# Obelysk model config (auto-generated at $(date -u +%Y-%m-%dT%H:%M:%SZ))
MODEL_NAME=${MODEL_NAME:-custom}
MODEL_HF=${MODEL_HF:-}
MODEL_ONNX=${MODEL_ONNX:-}
MODEL_DIR=${MODEL_DIR}
MODEL_LAYERS=${MODEL_LAYERS:-}
MODEL_QUANT=${MODEL_QUANT:-symmetric8}
MODEL_SIZE_GB=${MODEL_SIZE_GB:-?}
MODEL_HIDDEN=${MODEL_HIDDEN:-}
MODEL_HEADS=${MODEL_HEADS:-}
MODEL_KV_HEADS=${MODEL_KV_HEADS:-}
MODEL_ACTIVATION=${MODEL_ACTIVATION:-}
MODEL_DESCRIPTION=${MODEL_DESCRIPTION:-}
VALIDATED=false
CFGEOF

ok "Config saved to ${MODEL_CONFIG_DIR}/config.env"

# Also save to global state
save_state "model_state.env" \
    "CURRENT_MODEL=${MODEL_NAME:-custom}" \
    "MODEL_DIR=${MODEL_DIR}" \
    "MODEL_CONFIG=${MODEL_CONFIG_DIR}/config.env"

# ─── Summary ─────────────────────────────────────────────────────────

ELAPSED=$(timer_elapsed "model_setup")

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  MODEL SETUP COMPLETE                                ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Model:       %-37s ║\n" "${MODEL_NAME:-custom}"
printf "  ║  Source:      %-37s ║\n" "${MODEL_HF:-${MODEL_ONNX:-N/A}}"
printf "  ║  Directory:   %-37s ║\n" "${MODEL_DIR}"
printf "  ║  Layers:      %-37s ║\n" "${MODEL_LAYERS:-all}"
printf "  ║  Quant:       %-37s ║\n" "${MODEL_QUANT:-symmetric8}"
printf "  ║  Duration:    %-37s ║\n" "$(format_duration $ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next: ./02_validate_model.sh                        ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
