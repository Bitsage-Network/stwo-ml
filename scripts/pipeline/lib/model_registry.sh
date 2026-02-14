#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Model Registry
# ═══════════════════════════════════════════════════════════════════════
#
# Source this after common.sh:
#   source "${SCRIPT_DIR}/lib/common.sh"
#   source "${SCRIPT_DIR}/lib/model_registry.sh"
#
# Provides:
#   - list_presets    → print available model presets
#   - get_preset      → load a preset into MODEL_* variables
#   - load_config_env → load a configs/*.env file
#   - validate_model_vars → check all required MODEL_* vars are set

[[ -n "${_OBELYSK_MODEL_REGISTRY_LOADED:-}" ]] && return 0
_OBELYSK_MODEL_REGISTRY_LOADED=1

# ─── Built-in Presets ────────────────────────────────────────────────
#
# Format: PRESET_<NAME>="hf_repo|layers|size_gb|quant|shards|hidden|heads|kv_heads|activation|desc"

PRESET_qwen3_14b="Qwen/Qwen3-14B|40|28|symmetric8|6|5120|40|8|swiglu|Qwen3 14B (40 layers, GQA)"
PRESET_llama3_8b="meta-llama/Llama-3.1-8B|32|16|symmetric8|4|4096|32|8|swiglu|Llama 3.1 8B (32 layers, GQA)"
PRESET_llama3_70b="meta-llama/Llama-3.1-70B|80|140|symmetric8|16|8192|64|8|swiglu|Llama 3.1 70B (80 layers, GQA)"
PRESET_mistral_7b="mistralai/Mistral-7B-v0.3|32|15|symmetric8|3|4096|32|8|swiglu|Mistral 7B v0.3 (32 layers, GQA)"
PRESET_phi3_mini="microsoft/Phi-3-mini-4k-instruct|32|7|symmetric8|2|3072|32|32|swiglu|Phi-3 Mini 3.8B (32 layers, fast)"
PRESET_gemma2_9b="google/gemma-2-9b|42|18|symmetric8|4|3584|16|8|gelu|Gemma 2 9B (42 layers, GQA)"

# All known preset names
_PRESETS=("qwen3-14b" "llama3-8b" "llama3-70b" "mistral-7b" "phi3-mini" "gemma2-9b")

# ─── Preset Functions ────────────────────────────────────────────────

list_presets() {
    echo ""
    echo "  Available Model Presets"
    echo "  ─────────────────────────────────────────────────────────"
    printf "  %-15s %-28s %-6s %-6s %s\n" "PRESET" "HF REPO" "LAYERS" "SIZE" "DESCRIPTION"
    echo "  ─────────────────────────────────────────────────────────"

    for name in "${_PRESETS[@]}"; do
        local var_name="PRESET_${name//-/_}"
        local data="${!var_name}"
        if [[ -n "$data" ]]; then
            IFS='|' read -r hf layers size quant shards hidden heads kv act desc <<< "$data"
            printf "  %-15s %-28s %-6s %-6s %s\n" "$name" "$hf" "$layers" "${size}GB" "$desc"
        fi
    done

    echo "  ─────────────────────────────────────────────────────────"
    echo ""
    echo "  Usage: $0 --preset qwen3-14b"
    echo "  Custom: $0 --hf-model Qwen/Qwen3-0.5B --layers 24"
    echo ""
}

get_preset() {
    local name="$1"
    local var_name="PRESET_${name//-/_}"
    local data="${!var_name}"

    if [[ -z "$data" ]]; then
        err "Unknown preset: ${name}"
        err "Available: ${_PRESETS[*]}"
        return 1
    fi

    IFS='|' read -r MODEL_HF MODEL_LAYERS MODEL_SIZE_GB MODEL_QUANT MODEL_SHARDS \
                    MODEL_HIDDEN MODEL_HEADS MODEL_KV_HEADS MODEL_ACTIVATION \
                    MODEL_DESCRIPTION <<< "$data"

    MODEL_NAME="$name"
    # Aliases for config file compatibility (configs use MODEL_NUM_ATTENTION_HEADS)
    MODEL_NUM_ATTENTION_HEADS="$MODEL_HEADS"
    MODEL_NUM_KEY_VALUE_HEADS="$MODEL_KV_HEADS"
    export MODEL_NAME MODEL_HF MODEL_LAYERS MODEL_SIZE_GB MODEL_QUANT MODEL_SHARDS
    export MODEL_HIDDEN MODEL_HEADS MODEL_KV_HEADS MODEL_ACTIVATION MODEL_DESCRIPTION
    export MODEL_NUM_ATTENTION_HEADS MODEL_NUM_KEY_VALUE_HEADS

    debug "Loaded preset: ${name} → ${MODEL_HF} (${MODEL_LAYERS} layers, ${MODEL_SIZE_GB}GB)"
    return 0
}

# ─── Config File Loading ─────────────────────────────────────────────

load_config_env() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        err "Config file not found: ${config_file}"
        return 1
    fi

    # Source the config, exporting all variables
    set -a
    # shellcheck disable=SC1090
    source "$config_file"
    set +a

    debug "Loaded config from ${config_file}"
    return 0
}

# Find a config file by preset name
find_config() {
    local name="$1"
    local script_dir="${2:-}"

    # Check pipeline configs directory
    if [[ -n "$script_dir" ]] && [[ -f "${script_dir}/configs/${name}.env" ]]; then
        echo "${script_dir}/configs/${name}.env"
        return 0
    fi

    # Check obelysk dir
    if [[ -f "${OBELYSK_DIR}/configs/${name}.env" ]]; then
        echo "${OBELYSK_DIR}/configs/${name}.env"
        return 0
    fi

    return 1
}

# ─── Validation ──────────────────────────────────────────────────────

validate_model_vars() {
    local errors=0

    if [[ -z "${MODEL_HF:-}" ]] && [[ -z "${MODEL_ONNX:-}" ]]; then
        err "Neither MODEL_HF nor MODEL_ONNX is set"
        (( errors++ ))
    fi

    if [[ -z "${MODEL_LAYERS:-}" ]]; then
        warn "MODEL_LAYERS not set (will use model default)"
    fi

    if [[ -z "${MODEL_QUANT:-}" ]]; then
        warn "MODEL_QUANT not set (defaulting to symmetric8)"
        MODEL_QUANT="symmetric8"
    fi

    # Normalize variable names: config files may use either convention
    if [[ -n "${MODEL_NUM_ATTENTION_HEADS:-}" ]] && [[ -z "${MODEL_HEADS:-}" ]]; then
        MODEL_HEADS="$MODEL_NUM_ATTENTION_HEADS"
        export MODEL_HEADS
    fi
    if [[ -n "${MODEL_NUM_KEY_VALUE_HEADS:-}" ]] && [[ -z "${MODEL_KV_HEADS:-}" ]]; then
        MODEL_KV_HEADS="$MODEL_NUM_KEY_VALUE_HEADS"
        export MODEL_KV_HEADS
    fi
    # And vice versa
    if [[ -n "${MODEL_HEADS:-}" ]] && [[ -z "${MODEL_NUM_ATTENTION_HEADS:-}" ]]; then
        MODEL_NUM_ATTENTION_HEADS="$MODEL_HEADS"
        export MODEL_NUM_ATTENTION_HEADS
    fi
    if [[ -n "${MODEL_KV_HEADS:-}" ]] && [[ -z "${MODEL_NUM_KEY_VALUE_HEADS:-}" ]]; then
        MODEL_NUM_KEY_VALUE_HEADS="$MODEL_KV_HEADS"
        export MODEL_NUM_KEY_VALUE_HEADS
    fi

    return $errors
}

# ─── Print Model Config ─────────────────────────────────────────────

print_model_config() {
    echo "" >&2
    echo -e "${BOLD}  Model Configuration${NC}" >&2
    echo "  ──────────────────────────────────────────" >&2
    echo "  Name:         ${MODEL_NAME:-custom}" >&2
    echo "  HF Repo:      ${MODEL_HF:-N/A}" >&2
    echo "  ONNX:         ${MODEL_ONNX:-N/A}" >&2
    echo "  Layers:       ${MODEL_LAYERS:-all}" >&2
    echo "  Hidden Size:  ${MODEL_HIDDEN:-N/A}" >&2
    echo "  Heads:        ${MODEL_HEADS:-N/A} (KV: ${MODEL_KV_HEADS:-N/A})" >&2
    echo "  Quant:        ${MODEL_QUANT:-symmetric8}" >&2
    echo "  Size:         ${MODEL_SIZE_GB:-?}GB" >&2
    echo "  ──────────────────────────────────────────" >&2
    echo "" >&2
}
