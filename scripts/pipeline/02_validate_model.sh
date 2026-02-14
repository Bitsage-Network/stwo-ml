#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 2: Model Validation
# ═══════════════════════════════════════════════════════════════════════
#
# Validates a downloaded model before proving:
#   1. Model inspection (architecture summary)
#   2. 8-check validation suite (weights, dims, config)
#   3. Optional: 1-layer test proof to verify GPU pipeline
#   4. Resource estimation
#
# Usage:
#   bash scripts/pipeline/02_validate_model.sh
#   bash scripts/pipeline/02_validate_model.sh --model-name qwen3-14b
#   bash scripts/pipeline/02_validate_model.sh --model-dir ~/models/qwen3-14b --full
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────

MODEL_NAME=""
MODEL_DIR=""
NUM_LAYERS=""
QUICK=false
FULL=false

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)  MODEL_NAME="$2"; shift 2 ;;
        --model-dir)   MODEL_DIR="$2"; shift 2 ;;
        --layers)      NUM_LAYERS="$2"; shift 2 ;;
        --quick)       QUICK=true; shift ;;
        --full)        FULL=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Validate a model before proving."
            echo ""
            echo "Options:"
            echo "  --model-name NAME  Load model config from ~/.obelysk/models/NAME/"
            echo "  --model-dir DIR    Path to model directory (overrides --model-name)"
            echo "  --layers N         Number of layers to validate (default: from config)"
            echo "  --quick            Inspect only, skip validation checks"
            echo "  --full             Also run a 1-layer test proof"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Resolve Model Config ───────────────────────────────────────────

init_obelysk_dir

# If no model specified, try current model from state
if [[ -z "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_NAME=$(get_state "model_state.env" "CURRENT_MODEL" 2>/dev/null || echo "")
    if [[ -z "$MODEL_NAME" ]]; then
        err "No model specified. Use --model-name or --model-dir."
        err "Or run 01_setup_model.sh first."
        exit 1
    fi
    log "Using current model: ${MODEL_NAME}"
fi

# Load model config
MODEL_CONFIG=""
if [[ -n "$MODEL_NAME" ]] && [[ -z "$MODEL_DIR" ]]; then
    MODEL_CONFIG="${OBELYSK_DIR}/models/${MODEL_NAME}/config.env"
    if [[ -f "$MODEL_CONFIG" ]]; then
        log "Loading config: ${MODEL_CONFIG}"
        set -a; source "$MODEL_CONFIG"; set +a
    else
        err "Model config not found: ${MODEL_CONFIG}"
        err "Run 01_setup_model.sh --preset ${MODEL_NAME} first."
        exit 1
    fi
fi

# MODEL_DIR must be set at this point
if [[ -z "$MODEL_DIR" ]]; then
    err "MODEL_DIR not set. Specify --model-dir or configure via 01_setup_model.sh."
    exit 1
fi

# Apply overrides
[[ -n "$NUM_LAYERS" ]] && MODEL_LAYERS="$NUM_LAYERS"

# ─── Find prove-model binary ────────────────────────────────────────

PROVE_BIN=""

# Check state
PROVE_BIN=$(get_state "setup_state.env" "PROVE_MODEL_BIN" 2>/dev/null || echo "")

# Search common locations
if [[ -z "$PROVE_BIN" ]] || [[ ! -f "$PROVE_BIN" ]]; then
    INSTALL_DIR=$(get_state "setup_state.env" "INSTALL_DIR" 2>/dev/null || echo "$HOME/bitsage-network")
    PROVE_BIN=$(find_binary "prove-model" "${INSTALL_DIR}/libs" 2>/dev/null || echo "")
fi

# PATH fallback
if [[ -z "$PROVE_BIN" ]] || [[ ! -f "$PROVE_BIN" ]]; then
    PROVE_BIN=$(command -v prove-model 2>/dev/null || echo "")
fi

if [[ -z "$PROVE_BIN" ]]; then
    err "prove-model binary not found."
    err "Run 00_setup_gpu.sh first, or ensure prove-model is in PATH."
    exit 1
fi

# ─── Start Validation ───────────────────────────────────────────────

banner
echo -e "${BOLD}  Model Validation: ${MODEL_NAME:-$(basename "$MODEL_DIR")}${NC}"
echo ""
log "Model directory: ${MODEL_DIR}"
log "prove-model:     ${PROVE_BIN}"
log "Layers:          ${MODEL_LAYERS:-all}"
echo ""

timer_start "validate"

# ─── Step 1: Inspect ─────────────────────────────────────────────────

header "Architecture Inspection"

INSPECT_ARGS=("--inspect")
if [[ -d "$MODEL_DIR" ]] && [[ -f "${MODEL_DIR}/config.json" ]]; then
    INSPECT_ARGS+=("--model-dir" "$MODEL_DIR")
elif [[ -f "${MODEL_ONNX:-}" ]]; then
    INSPECT_ARGS+=("--model" "$MODEL_ONNX")
else
    INSPECT_ARGS+=("--model-dir" "$MODEL_DIR")
fi

if [[ -n "${MODEL_LAYERS:-}" ]]; then
    INSPECT_ARGS+=("--layers" "$MODEL_LAYERS")
fi

log "Running: ${PROVE_BIN} ${INSPECT_ARGS[*]}"
echo ""

if run_cmd "${PROVE_BIN}" "${INSPECT_ARGS[@]}" 2>&1; then
    ok "Inspection complete"
else
    warn "Inspection failed (non-fatal)"
fi
echo ""

if [[ "$QUICK" == "true" ]]; then
    ok "Quick mode — skipping validation checks"
    ELAPSED=$(timer_elapsed "validate")
    echo ""
    echo -e "${GREEN}Inspection complete in $(format_duration $ELAPSED)${NC}"
    exit 0
fi

# ─── Step 2: Validate ───────────────────────────────────────────────

header "Validation Suite (8 checks)"

VALIDATE_ARGS=("--validate")
if [[ -d "$MODEL_DIR" ]] && [[ -f "${MODEL_DIR}/config.json" ]]; then
    VALIDATE_ARGS+=("--model-dir" "$MODEL_DIR")
elif [[ -f "${MODEL_ONNX:-}" ]]; then
    VALIDATE_ARGS+=("--model" "$MODEL_ONNX")
else
    VALIDATE_ARGS+=("--model-dir" "$MODEL_DIR")
fi

if [[ -n "${MODEL_LAYERS:-}" ]]; then
    VALIDATE_ARGS+=("--layers" "$MODEL_LAYERS")
fi

log "Running: ${PROVE_BIN} ${VALIDATE_ARGS[*]}"
echo ""

if run_cmd "${PROVE_BIN}" "${VALIDATE_ARGS[@]}" 2>&1; then
    ok "All validation checks passed"
    VALIDATE_OK=true
else
    err "Validation failed"
    VALIDATE_OK=false
fi
echo ""

# ─── Step 3: Full Test Proof (optional) ──────────────────────────────

TEST_PROOF_OK=""
if [[ "$FULL" == "true" ]]; then
    header "Test Proof (1 layer)"

    TEST_OUTPUT="/tmp/obelysk_test_proof_$(date +%s).json"
    TEST_ARGS=("--model-dir" "$MODEL_DIR" "--layers" "1" "--output" "$TEST_OUTPUT" "--format" "json")

    # Enable GPU if available
    if [[ "$(get_state "gpu_config.env" "GPU_AVAILABLE" 2>/dev/null)" == "true" ]]; then
        TEST_ARGS+=("--gpu")
    fi

    log "Running 1-layer test proof..."
    log "Command: ${PROVE_BIN} ${TEST_ARGS[*]}"
    echo ""

    if run_cmd "${PROVE_BIN}" "${TEST_ARGS[@]}" 2>&1; then
        if [[ -f "$TEST_OUTPUT" ]]; then
            TEST_SIZE=$(du -h "$TEST_OUTPUT" | cut -f1)
            ok "Test proof generated: ${TEST_OUTPUT} (${TEST_SIZE})"
            TEST_PROOF_OK="PASS"
            rm -f "$TEST_OUTPUT"
        else
            warn "Proof command succeeded but output file not found"
            TEST_PROOF_OK="WARN"
        fi
    else
        err "Test proof failed"
        TEST_PROOF_OK="FAIL"
    fi
    echo ""
fi

# ─── Step 4: Resource Estimation ────────────────────────────────────

header "Resource Estimation"

LAYERS="${MODEL_LAYERS:-32}"
HIDDEN="${MODEL_HIDDEN:-4096}"

if [[ -n "$HIDDEN" ]] && [[ "$HIDDEN" != "?" ]]; then
    # Rough estimates based on model architecture
    WEIGHT_ELEMENTS=$(( HIDDEN * HIDDEN * 4 * LAYERS ))  # 4 weight matrices per layer
    TRACE_ROWS=$(( HIDDEN * HIDDEN ))
    EST_PROOF_MB=$(( WEIGHT_ELEMENTS / 100000 ))  # rough: ~10 bytes per 1M elements
    EST_MEMORY_GB=$(( WEIGHT_ELEMENTS * 4 / 1073741824 + 2 ))  # 4 bytes/element + overhead

    log "Estimated per layer:"
    log "  Weight elements:  ~$(( HIDDEN * HIDDEN * 4 )) per layer"
    log "  Trace rows:       ~${TRACE_ROWS}"
    log ""
    log "Estimated total (${LAYERS} layers):"
    log "  Total elements:   ~${WEIGHT_ELEMENTS}"
    log "  Proof size:       ~${EST_PROOF_MB}MB (cairo_serde)"
    log "  Peak memory:      ~${EST_MEMORY_GB}GB"
else
    warn "Cannot estimate resources: MODEL_HIDDEN not set"
fi
echo ""

# ─── Update Config ──────────────────────────────────────────────────

if [[ -n "$MODEL_CONFIG" ]] && [[ -f "$MODEL_CONFIG" ]]; then
    if [[ "$VALIDATE_OK" == "true" ]]; then
        # Mark as validated
        if grep -q "^VALIDATED=" "$MODEL_CONFIG" 2>/dev/null; then
            sed -i.bak "s/^VALIDATED=.*/VALIDATED=true/" "$MODEL_CONFIG"
            rm -f "${MODEL_CONFIG}.bak"
        else
            echo "VALIDATED=true" >> "$MODEL_CONFIG"
        fi
        ok "Config updated: VALIDATED=true"
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────

ELAPSED=$(timer_elapsed "validate")

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  VALIDATION COMPLETE                                 ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Model:        %-36s ║\n" "${MODEL_NAME:-$(basename "$MODEL_DIR")}"
printf "  ║  Validation:   %-36s ║\n" "${VALIDATE_OK:-skipped}"
if [[ -n "$TEST_PROOF_OK" ]]; then
printf "  ║  Test proof:   %-36s ║\n" "$TEST_PROOF_OK"
fi
printf "  ║  Duration:     %-36s ║\n" "$(format_duration $ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next: ./03_prove.sh --model-name ${MODEL_NAME:-mymodel}  ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
