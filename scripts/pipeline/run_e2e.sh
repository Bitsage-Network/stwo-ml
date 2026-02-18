#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — End-to-End Runner
# ═══════════════════════════════════════════════════════════════════════
#
# Single-command execution of the full pipeline:
#   Setup GPU → Download Model → Validate → [Test Inference] → Capture → Prove → Verify On-Chain → Audit
#
# Usage:
#   ./run_e2e.sh --preset qwen3-14b --gpu --submit
#   ./run_e2e.sh --hf-model Qwen/Qwen3-14B --layers 1 --gpu --dry-run
#   ./run_e2e.sh --preset phi3-mini --hf-token $HF_TOKEN --gpu --chat --submit
#   ./run_e2e.sh --preset llama3-8b --resume-from prove --submit
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

# ─── Defaults ────────────────────────────────────────────────────────

PRESET=""
HF_MODEL=""
LAYERS=""
MODE="gkr"
DO_SUBMIT=false
DO_DRY_RUN=false
DO_CHAT=false
DO_GPU=true
DO_MULTI_GPU=false
DO_GPU_ONLY=false
DO_AUDIT=true
SKIP_SETUP=false
SKIP_INFERENCE=false
RESUME_FROM=""
HF_TOKEN_ARG=""
MAX_FEE="0.05"
MODEL_ID="0x1"
FORCE_PAYMASTER=false
FORCE_NO_PAYMASTER=false
GKR_V2=false
GKR_V3=false
GKR_V4=false
GKR_V2_MODE="auto"  # auto|sequential|batched|mode2|mode3
LEGACY_GKR_V1=false

# Passthrough arrays for sub-scripts
SETUP_ARGS=()
MODEL_ARGS=()
PROVE_ARGS=()
VERIFY_ARGS=()

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)          PRESET="$2"; shift 2 ;;
        --hf-model)        HF_MODEL="$2"; shift 2 ;;
        --layers)          LAYERS="$2"; shift 2 ;;
        --mode)            MODE="$2"; shift 2 ;;
        --submit)          DO_SUBMIT=true; shift ;;
        --dry-run)         DO_DRY_RUN=true; shift ;;
        --chat)            DO_CHAT=true; shift ;;
        --gpu)             DO_GPU=true; shift ;;
        --no-gpu)          DO_GPU=false; shift ;;
        --multi-gpu)       DO_MULTI_GPU=true; DO_GPU=true; shift ;;
        --gpu-only)        DO_GPU_ONLY=true; DO_GPU=true; shift ;;
        --no-audit)        DO_AUDIT=false; shift ;;
        --skip-setup)      SKIP_SETUP=true; shift ;;
        --skip-inference)  SKIP_INFERENCE=true; shift ;;
        --paymaster)       FORCE_PAYMASTER=true; shift ;;
        --no-paymaster)    FORCE_NO_PAYMASTER=true; shift ;;
        --resume-from)     RESUME_FROM="$2"; shift 2 ;;
        --hf-token)        HF_TOKEN_ARG="$2"; shift 2 ;;
        --max-fee)         MAX_FEE="$2"; shift 2 ;;
        --model-id)        MODEL_ID="$2"; shift 2 ;;
        --gkr-v2)          GKR_V2=true; shift ;;
        --gkr-v3)          GKR_V3=true; shift ;;
        --gkr-v4)          GKR_V4=true; shift ;;
        --gkr-v2-mode)     GKR_V2_MODE="$2"; shift 2 ;;
        --legacy-gkr-v1)   LEGACY_GKR_V1=true; shift ;;
        --install-drivers) SETUP_ARGS+=("--install-drivers"); shift ;;
        --skip-drivers)    SETUP_ARGS+=("--skip-drivers"); shift ;;
        --branch)          SETUP_ARGS+=("--branch" "$2"); shift 2 ;;
        --cuda-path)       SETUP_ARGS+=("--cuda-path" "$2"); shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run the full Obelysk pipeline end-to-end."
            echo ""
            echo "Model (pick one):"
            echo "  --preset NAME        Built-in preset (qwen3-14b, phi3-mini, etc.)"
            echo "  --hf-model REPO      HuggingFace model repo"
            echo ""
            echo "Pipeline control:"
            echo "  --submit             Submit proof on-chain"
            echo "  --dry-run            Run without on-chain submission"
            echo "  --chat               Pause for interactive chat after inference test"
            echo "  --skip-setup         Skip GPU setup (machine already configured)"
            echo "  --skip-inference     Skip inference testing"
            echo "  --no-audit           Skip inference audit (audit is on by default)"
            echo "  --resume-from STEP   Resume from: model, validate, inference, capture, prove, verify, audit"
            echo "  --paymaster          Force AVNU paymaster (gasless, sponsored)"
            echo "  --no-paymaster       Force legacy sncast submission (you pay gas)"
            echo ""
            echo "Options:"
            echo "  --layers N           Number of layers to prove"
            echo "  --mode MODE          Proof mode: gkr (default: gkr)"
            echo "  --gpu / --no-gpu     Enable/disable GPU (default: on)"
            echo "  --multi-gpu          Use all GPUs"
            echo "  --gpu-only           Fail if any critical proving path falls back to CPU"
            echo "                       (submit path auto-forces --starknet-ready in step 6)"
            echo "  --gkr-v2             Use verify_model_gkr_v2 calldata"
            echo "  --gkr-v3             Use verify_model_gkr_v3 calldata (v3 envelope)"
            echo "  --gkr-v4             Use verify_model_gkr_v4 calldata (experimental mode-3 envelope)"
            echo "  --gkr-v2-mode MODE   v2/v3/v4 weight-binding profile: auto|sequential|batched|mode2|mode3"
            echo "                       auto(default): submit path prefers mode2 on v3, mode3 on v4"
            echo "  --legacy-gkr-v1      Keep legacy verify_model_gkr (v1 sequential openings)"
            echo "  --hf-token TOKEN     HuggingFace API token"
            echo "  --max-fee ETH        Max TX fee (default: 0.05)"
            echo "  --model-id ID        On-chain model ID (default: 0x1)"
            echo "  --install-drivers    Install NVIDIA driver if missing"
            echo "  --skip-drivers       Skip driver installation"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Environment variables:"
            echo "  HF_TOKEN                  HuggingFace token"
            echo "  STARKNET_PRIVATE_KEY      For on-chain submission (optional on Sepolia)"
            echo "  STARKNET_ACCOUNT_ADDRESS  Account address (when using own key with paymaster)"
            echo "  STARKNET_RPC              Override RPC endpoint"
            echo "  OBELYSK_DEPLOYER_KEY      Deployer key for factory account creation"
            echo "  OBELYSK_DEPLOYER_ADDRESS  Deployer address for factory account creation"
            echo "  OBELYSK_DEBUG=1           Debug logging"
            echo ""
            echo "Examples:"
            echo "  $0 --preset phi3-mini --gpu --dry-run"
            echo "  $0 --preset qwen3-14b --gpu --submit          # zero-config on Sepolia"
            echo "  $0 --preset qwen3-14b --gpu --submit --no-paymaster  # legacy sncast"
            echo "  HF_TOKEN=hf_xxx $0 --preset llama3-8b --chat --submit"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────

if [[ -z "$PRESET" ]] && [[ -z "$HF_MODEL" ]]; then
    err "Specify a model with --preset or --hf-model"
    exit 1
fi

if [[ "$MODE" != "gkr" ]]; then
    err "Only --mode gkr is supported in the hardened pipeline (got: ${MODE})"
    exit 1
fi

case "${GKR_V2_MODE,,}" in
    auto|sequential|batched|mode2|mode3) ;;
    *)
        err "Invalid --gkr-v2-mode: ${GKR_V2_MODE} (expected: auto|sequential|batched|mode2|mode3)"
        exit 1
        ;;
esac
if [[ "$GKR_V2" == "true" && "$GKR_V3" == "true" ]]; then
    warn "Both --gkr-v2 and --gkr-v3 were set; preferring v3 entrypoint."
    GKR_V2=false
fi
if [[ "$GKR_V4" == "true" && "$GKR_V3" == "true" ]]; then
    warn "Both --gkr-v3 and --gkr-v4 were set; preferring v4 entrypoint."
    GKR_V3=false
fi
if [[ "$GKR_V4" == "true" && "$GKR_V2" == "true" ]]; then
    warn "Both --gkr-v2 and --gkr-v4 were set; preferring v4 entrypoint."
    GKR_V2=false
fi

if [[ "${GKR_V2_MODE,,}" != "auto" ]] && [[ "$GKR_V2" != "true" ]] && [[ "$GKR_V3" != "true" ]] && [[ "$GKR_V4" != "true" ]]; then
    err "--gkr-v2-mode requires --gkr-v2, --gkr-v3, or --gkr-v4"
    exit 1
fi
if [[ "${GKR_V2_MODE,,}" == "mode2" ]] && [[ "$GKR_V3" != "true" ]]; then
    err "--gkr-v2-mode mode2 requires --gkr-v3"
    exit 1
fi
if [[ "${GKR_V2_MODE,,}" == "mode3" ]] && [[ "$GKR_V4" != "true" ]]; then
    err "--gkr-v2-mode mode3 requires --gkr-v4"
    exit 1
fi
if [[ "$GKR_V4" == "true" ]] && [[ "${GKR_V2_MODE,,}" == "sequential" || "${GKR_V2_MODE,,}" == "batched" || "${GKR_V2_MODE,,}" == "mode2" ]]; then
    err "--gkr-v4 only supports --gkr-v2-mode auto|mode3"
    exit 1
fi

if [[ "$LEGACY_GKR_V1" == "true" ]] && [[ "$GKR_V2" == "true" || "$GKR_V3" == "true" || "$GKR_V4" == "true" ]]; then
    warn "--legacy-gkr-v1 ignored because a v2/v3/v4 entrypoint was explicitly selected."
    LEGACY_GKR_V1=false
fi

if [[ "$DO_SUBMIT" == "false" ]] && [[ "$DO_DRY_RUN" == "false" ]]; then
    warn "Neither --submit nor --dry-run specified. Defaulting to --dry-run."
    DO_DRY_RUN=true
fi

if [[ "$DO_SUBMIT" == "true" ]] && [[ "$MODE" == "gkr" ]] && [[ "$GKR_V2" != "true" ]] && [[ "$GKR_V3" != "true" ]] && [[ "$GKR_V4" != "true" ]]; then
    if [[ "$LEGACY_GKR_V1" == "true" ]]; then
        warn "--submit + --legacy-gkr-v1: keeping verify_model_gkr (v1 sequential openings)."
    else
        warn "--submit requested without --gkr-v2/--gkr-v3/--gkr-v4; defaulting to verify_model_gkr_v3 mode2 (fast trustless submit path)."
        GKR_V3=true
        if [[ "${GKR_V2_MODE,,}" == "auto" ]]; then
            GKR_V2_MODE="mode2"
        fi
    fi
fi

if [[ "$DO_SUBMIT" == "true" ]] && [[ "$GKR_V3" == "true" ]] && [[ "${GKR_V2_MODE,,}" == "auto" ]] && [[ "$LEGACY_GKR_V1" != "true" ]]; then
    warn "--submit + --gkr-v3 with auto mode: defaulting to mode2 trustless binding."
    GKR_V2_MODE="mode2"
fi
if [[ "$DO_SUBMIT" == "true" ]] && [[ "$GKR_V4" == "true" ]] && [[ "${GKR_V2_MODE,,}" == "auto" ]] && [[ "$LEGACY_GKR_V1" != "true" ]]; then
    warn "--submit + --gkr-v4 with auto mode: defaulting to mode3 experimental binding envelope."
    GKR_V2_MODE="mode3"
fi

# ─── Start ───────────────────────────────────────────────────────────

banner
echo -e "${BOLD}  End-to-End Pipeline${NC}"
echo ""

init_obelysk_dir
timer_start "e2e"

# Generate run ID
RUN_ID="$(date +%Y%m%d_%H%M%S)_${PRESET:-custom}"
RUN_DIR="${OBELYSK_DIR}/runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

log "Run ID:      ${RUN_ID}"
log "Model:       ${PRESET:-${HF_MODEL}}"
log "Mode:        ${MODE}"
log "GPU:         ${DO_GPU} (multi: ${DO_MULTI_GPU})"
log "GPU only:    ${DO_GPU_ONLY}"
log "GKR v2:      ${GKR_V2}"
log "GKR v3:      ${GKR_V3}"
log "GKR v4:      ${GKR_V4}"
log "GKR v2 mode: ${GKR_V2_MODE}"
log "Legacy v1:   ${LEGACY_GKR_V1}"
log "Layers:      ${LAYERS:-all}"
log "Action:      $(if [[ "$DO_SUBMIT" == "true" ]]; then echo "SUBMIT"; else echo "DRY RUN"; fi)"
log "Run dir:     ${RUN_DIR}"
echo ""

# Determine starting step
STEPS=("setup" "model" "validate" "inference" "capture" "prove" "verify" "audit")
START_IDX=0

case "${RESUME_FROM}" in
    "")        START_IDX=0 ;;
    setup)     START_IDX=0 ;;
    model)     START_IDX=1 ;;
    validate)  START_IDX=2 ;;
    inference) START_IDX=3 ;;
    capture)   START_IDX=4 ;;
    prove)     START_IDX=5 ;;
    verify)    START_IDX=6 ;;
    audit)     START_IDX=7 ;;
    *) err "Unknown step: ${RESUME_FROM}. Expected: setup, model, validate, inference, capture, prove, verify, audit"; exit 1 ;;
esac

if [[ "$SKIP_SETUP" == "true" ]] && (( START_IDX == 0 )); then
    START_IDX=1
fi

# ═══════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════

STEP_RESULTS=()

run_step() {
    local name="$1"
    local num="$2"
    local total="$3"
    shift 3

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  [${num}/${total}] ${name}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if "$@"; then
        STEP_RESULTS+=("${name}: PASS")
        ok "${name} completed"
    else
        STEP_RESULTS+=("${name}: FAIL")
        err "${name} failed"
        return 1
    fi
}

TOTAL_STEPS=$(( 7 - START_IDX ))
if [[ "$DO_AUDIT" == "true" ]]; then
    (( TOTAL_STEPS++ )) || true
fi
if [[ "$SKIP_INFERENCE" == "true" ]]; then
    (( TOTAL_STEPS-- )) || true
fi
CURRENT=1

# ─── Step 1: GPU Setup ──────────────────────────────────────────────

if (( START_IDX <= 0 )); then
    _SETUP_ARGS=("${SETUP_ARGS[@]}")
    if [[ "$SKIP_INFERENCE" == "true" ]]; then
        _SETUP_ARGS+=("--skip-llama")
        log "Skipping llama.cpp build in setup (--skip-inference enabled)."
    fi
    run_step "GPU Setup" "$CURRENT" "$TOTAL_STEPS" \
        env OBELYSK_REQUIRE_GPU="${DO_GPU}" \
        bash "${SCRIPT_DIR}/00_setup_gpu.sh" "${_SETUP_ARGS[@]}" || exit 1
    (( CURRENT++ ))
fi

# ─── Step 2: Model Download ─────────────────────────────────────────

if (( START_IDX <= 1 )); then
    _MODEL_ARGS=()
    if [[ -n "$PRESET" ]]; then
        _MODEL_ARGS+=("--preset" "$PRESET")
    elif [[ -n "$HF_MODEL" ]]; then
        _MODEL_ARGS+=("--hf-model" "$HF_MODEL")
    fi
    [[ -n "$LAYERS" ]] && _MODEL_ARGS+=("--layers" "$LAYERS")
    [[ -n "$HF_TOKEN_ARG" ]] && _MODEL_ARGS+=("--hf-token" "$HF_TOKEN_ARG")

    run_step "Model Download" "$CURRENT" "$TOTAL_STEPS" \
        bash "${SCRIPT_DIR}/01_setup_model.sh" "${_MODEL_ARGS[@]}" || exit 1
    (( CURRENT++ ))
fi

# ─── Step 3: Model Validation ───────────────────────────────────────

if (( START_IDX <= 2 )); then
    _VALIDATE_SCRIPT="${SCRIPT_DIR}/02_validate_model.sh"
    if [[ -f "$_VALIDATE_SCRIPT" ]]; then
        run_step "Model Validation" "$CURRENT" "$TOTAL_STEPS" \
            bash "$_VALIDATE_SCRIPT" --full || exit 1
    else
        log "02_validate_model.sh not found, skipping validation"
        STEP_RESULTS+=("Model Validation: SKIP")
    fi
    (( CURRENT++ ))
fi

# ─── Step 4: Inference Testing (optional, llama.cpp) ─────────────────

if (( START_IDX <= 3 )) && [[ "$SKIP_INFERENCE" != "true" ]]; then
    _INFER_ARGS=()
    if [[ "$DO_CHAT" == "true" ]]; then
        _INFER_ARGS+=("--chat")
    fi

    run_step "Inference Test" "$CURRENT" "$TOTAL_STEPS" \
        bash "${SCRIPT_DIR}/02a_test_inference.sh" "${_INFER_ARGS[@]}" || {
        warn "Inference test failed (non-critical, continuing)"
        STEP_RESULTS+=("Inference Test: WARN")
    }
    (( CURRENT++ ))
fi

# ─── Step 5: Inference Capture (mandatory for audit) ─────────────────

if (( START_IDX <= 4 )); then
    _CAPTURE_ARGS=()
    [[ -n "$LAYERS" ]] && _CAPTURE_ARGS+=("--layers" "$LAYERS")
    [[ -n "$MODEL_ID" ]] && _CAPTURE_ARGS+=("--model-id" "$MODEL_ID")

    run_step "Inference Capture" "$CURRENT" "$TOTAL_STEPS" \
        bash "${SCRIPT_DIR}/02b_capture_inference.sh" "${_CAPTURE_ARGS[@]}" || exit 1
    (( CURRENT++ ))
fi

# ─── Step 6: Proof Generation ───────────────────────────────────────

if (( START_IDX <= 5 )); then
    _PROVE_ARGS=("--mode" "$MODE" "--model-id" "$MODEL_ID")
    [[ -n "$LAYERS" ]] && _PROVE_ARGS+=("--layers" "$LAYERS")
    [[ "$DO_GPU" == "true" ]] && _PROVE_ARGS+=("--gpu")
    [[ "$DO_MULTI_GPU" == "true" ]] && _PROVE_ARGS+=("--multi-gpu")
    [[ "$DO_GPU_ONLY" == "true" ]] && _PROVE_ARGS+=("--gpu-only")
    [[ "$DO_SUBMIT" == "true" ]] && _PROVE_ARGS+=("--starknet-ready")
    [[ "$GKR_V2" == "true" ]] && _PROVE_ARGS+=("--gkr-v2")
    [[ "$GKR_V3" == "true" ]] && _PROVE_ARGS+=("--gkr-v3")
    [[ "$GKR_V4" == "true" ]] && _PROVE_ARGS+=("--gkr-v4")
    [[ "$LEGACY_GKR_V1" == "true" ]] && _PROVE_ARGS+=("--legacy-gkr-v1")

    _PROVE_ENV=()
    if [[ "$GKR_V2" == "true" || "$GKR_V3" == "true" || "$GKR_V4" == "true" ]]; then
        case "${GKR_V2_MODE,,}" in
            sequential)
                _PROVE_ENV+=("STWO_GKR_BATCH_WEIGHT_OPENINGS=off")
                ;;
            batched)
                _PROVE_ENV+=("STWO_GKR_BATCH_WEIGHT_OPENINGS=on")
                ;;
            mode2)
                _PROVE_ARGS+=("--gkr-v3-mode2")
                _PROVE_ENV+=("STWO_GKR_BATCH_WEIGHT_OPENINGS=on")
                ;;
            mode3)
                _PROVE_ARGS+=("--gkr-v4-mode3")
                _PROVE_ENV+=("STWO_GKR_BATCH_WEIGHT_OPENINGS=on")
                ;;
            auto)
                # Keep 03_prove.sh defaults:
                # - on for --starknet-ready --gkr-v2/--gkr-v3/--gkr-v4 --gpu
                # - off otherwise
                ;;
        esac
    fi

    run_step "Proof Generation" "$CURRENT" "$TOTAL_STEPS" \
        env "${_PROVE_ENV[@]}" bash "${SCRIPT_DIR}/03_prove.sh" "${_PROVE_ARGS[@]}" || exit 1
    (( CURRENT++ ))
fi

# ─── Step 7: On-Chain Verification ──────────────────────────────────

if (( START_IDX <= 6 )); then
    _VERIFY_ARGS=("--max-fee" "$MAX_FEE")
    if [[ "$DO_SUBMIT" == "true" ]]; then
        _VERIFY_ARGS+=("--submit")
    else
        _VERIFY_ARGS+=("--dry-run")
    fi
    [[ "$FORCE_PAYMASTER" == "true" ]] && _VERIFY_ARGS+=("--paymaster")
    [[ "$FORCE_NO_PAYMASTER" == "true" ]] && _VERIFY_ARGS+=("--no-paymaster")

    run_step "On-Chain Verification" "$CURRENT" "$TOTAL_STEPS" \
        bash "${SCRIPT_DIR}/04_verify_onchain.sh" "${_VERIFY_ARGS[@]}" || {
        if [[ "$DO_DRY_RUN" == "true" ]]; then
            warn "Dry run mode — on-chain step shows commands only"
        else
            err "On-chain verification failed"
            exit 1
        fi
    }
    (( CURRENT++ ))
fi

# ─── Step 8: Audit ─────────────────────────────────────────────────

if (( START_IDX <= 7 )) && [[ "$DO_AUDIT" == "true" ]]; then
    _AUDIT_ARGS=("--evaluate")
    if [[ "$DO_SUBMIT" == "true" ]]; then
        _AUDIT_ARGS+=("--submit")
    else
        _AUDIT_ARGS+=("--dry-run")
    fi

    if [[ "$DO_SUBMIT" == "true" ]]; then
        run_step "Inference Audit" "$CURRENT" "$TOTAL_STEPS" \
            bash "${SCRIPT_DIR}/05_audit.sh" "${_AUDIT_ARGS[@]}" || {
            err "Audit step failed during submission — aborting"
            exit 1
        }
    else
        run_step "Inference Audit" "$CURRENT" "$TOTAL_STEPS" \
            bash "${SCRIPT_DIR}/05_audit.sh" "${_AUDIT_ARGS[@]}" || {
            warn "Audit step failed (non-critical in dry-run mode)"
            STEP_RESULTS+=("Inference Audit: WARN")
        }
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════════════

TOTAL_ELAPSED=$(timer_elapsed "e2e")

# Collect results from state files
_PROOF_DIR=$(get_state "prove_state.env" "LAST_PROOF_DIR" 2>/dev/null || echo "N/A")
_PROOF_MODE=$(get_state "prove_state.env" "LAST_PROOF_MODE" 2>/dev/null || echo "N/A")
_IS_ACCEPTED="N/A"
_FULL_GKR_VERIFIED="N/A"
_ASSURANCE_LEVEL="N/A"
if [[ -f "${_PROOF_DIR}/verify_receipt.json" ]] 2>/dev/null; then
    _IS_ACCEPTED=$(python3 -c "
import json
with open('${_PROOF_DIR}/verify_receipt.json') as f:
    d = json.load(f)
    print(d.get('accepted_onchain', d.get('is_verified', 'N/A')))
" 2>/dev/null || echo "N/A")
    _FULL_GKR_VERIFIED=$(python3 -c "
import json
with open('${_PROOF_DIR}/verify_receipt.json') as f:
    d = json.load(f)
    print(d.get('full_gkr_verified', d.get('is_verified', 'N/A')))
" 2>/dev/null || echo "N/A")
    _ASSURANCE_LEVEL=$(python3 -c "
import json
with open('${_PROOF_DIR}/verify_receipt.json') as f:
    d = json.load(f)
    print(d.get('assurance_level', 'N/A'))
" 2>/dev/null || echo "N/A")
fi

# Save run state
cat > "${RUN_DIR}/run_summary.json" << RUNEOF
{
    "run_id": "${RUN_ID}",
    "preset": "${PRESET:-}",
    "hf_model": "${HF_MODEL:-}",
    "mode": "${MODE}",
    "layers": "${LAYERS:-all}",
    "gpu": ${DO_GPU},
    "submitted": ${DO_SUBMIT},
    "total_seconds": ${TOTAL_ELAPSED},
    "proof_dir": "${_PROOF_DIR}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
RUNEOF

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  END-TO-END PIPELINE COMPLETE                        ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Run ID:      %-36s ║\n" "${RUN_ID}"
printf "  ║  Model:       %-36s ║\n" "${PRESET:-${HF_MODEL:-custom}}"
printf "  ║  Mode:        %-36s ║\n" "${MODE}"
printf "  ║  Duration:    %-36s ║\n" "$(format_duration $TOTAL_ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
for result in "${STEP_RESULTS[@]}"; do
    printf "  ║  %-49s ║\n" "$result"
done
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Proof dir:   %-36s ║\n" "${_PROOF_DIR}"
printf "  ║  Accepted:    %-36s ║\n" "${_IS_ACCEPTED}"
printf "  ║  Assurance:   %-36s ║\n" "${_ASSURANCE_LEVEL}"
printf "  ║  Full GKR:    %-36s ║\n" "${_FULL_GKR_VERIFIED}"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
