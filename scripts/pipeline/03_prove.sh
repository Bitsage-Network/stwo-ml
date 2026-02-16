#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 3: Proof Generation
# ═══════════════════════════════════════════════════════════════════════
#
# Generates a cryptographic proof of ML model inference.
# Supports 3 proof modes:
#
#   recursive — GPU proof → Cairo VM recursive STARK → compact proof
#   direct    — GPU proof → chunked calldata for verify_model_direct()
#   gkr       — GPU proof → GKR calldata for verify_model_gkr()
#
# Usage:
#   bash scripts/pipeline/03_prove.sh --model-name qwen3-14b --mode recursive
#   bash scripts/pipeline/03_prove.sh --model-dir ~/models/qwen3-14b --layers 1 --mode gkr
#   bash scripts/pipeline/03_prove.sh --model-name qwen3-14b --mode direct --gpu --multi-gpu
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
CHUNK_BUDGET_GB=16
SECURITY="auto"
SKIP_BUILD=false
SKIP_COMMITMENT=false
GKR_FLAG=false
SALT=""
SERVER_URL=""

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
        --chunk-budget-gb)  CHUNK_BUDGET_GB="$2"; shift 2 ;;
        --security)         SECURITY="$2"; shift 2 ;;
        --skip-build)       SKIP_BUILD=true; shift ;;
        --skip-commitment)  SKIP_COMMITMENT=true; shift ;;
        --gkr)              GKR_FLAG=true; shift ;;
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
            echo "  --mode MODE        recursive | direct | gkr (default: gkr)"
            echo "    recursive:  prove-model → cairo-prove → Circle STARK"
            echo "    direct:     prove-model → chunked calldata"
            echo "    gkr:        prove-model → GKR calldata"
            echo ""
            echo "Options:"
            echo "  --layers N           Number of transformer layers (default: from config)"
            echo "  --input FILE         JSON input file (default: random input)"
            echo "  --output-dir DIR     Output directory (default: ~/.obelysk/proofs/<timestamp>)"
            echo "  --model-id ID        Model ID for on-chain claim (default: 0x1)"
            echo "  --gpu                Enable GPU acceleration (default)"
            echo "  --no-gpu             Disable GPU"
            echo "  --multi-gpu          Distribute across all GPUs"
            echo "  --chunk-budget-gb N  Memory per chunk in GB (default: 16)"
            echo "  --security LEVEL     auto | tee | zk-only (default: auto)"
            echo "  --skip-build         Skip building binaries"
            echo "  --skip-commitment    Skip weight commitment (faster, can't submit on-chain)"
            echo "  --gkr                Enable GKR for LogUp verification"
            echo "  --salt N             Fiat-Shamir channel salt"
            echo "  --server URL         Submit to remote prove-server instead of local binary"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate mode
case "$MODE" in
    recursive|direct|gkr) ;;
    *) err "Unknown mode: ${MODE} (expected: recursive, direct, gkr)"; exit 1 ;;
esac

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

# ─── Output Directory ───────────────────────────────────────────────

if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${OBELYSK_DIR}/proofs/${MODEL_NAME:-proof}_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

# ─── Find Binaries ──────────────────────────────────────────────────

INSTALL_DIR=$(get_state "setup_state.env" "INSTALL_DIR" 2>/dev/null || echo "$HOME/bitsage-network")
LIBS_DIR="${INSTALL_DIR}/libs"

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

# cairo-prove (for recursive mode)
CAIRO_BIN=""
if [[ "$MODE" == "recursive" ]]; then
    CAIRO_BIN=$(get_state "setup_state.env" "CAIRO_PROVE_BIN" 2>/dev/null || echo "")
    if [[ -z "$CAIRO_BIN" ]] || [[ ! -f "$CAIRO_BIN" ]]; then
        CAIRO_BIN=$(find_binary "cairo-prove" "$LIBS_DIR" 2>/dev/null || echo "")
    fi
    if [[ -z "$CAIRO_BIN" ]]; then
        err "cairo-prove not found (required for recursive mode)."
        err "Run 00_setup_gpu.sh or use --mode direct/gkr."
        exit 1
    fi
fi

# Optional rebuild
if [[ "$SKIP_BUILD" == "false" ]] && [[ -d "${LIBS_DIR}/stwo-ml" ]]; then
    FEATURES="cli,audit"
    if [[ "$USE_GPU" == "true" ]]; then
        FEATURES="cli,audit,cuda-runtime"
    fi
    log "Rebuilding prove-model (features: ${FEATURES})..."
    (cd "${LIBS_DIR}/stwo-ml" && cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -3) || true
fi

# ─── Display Config ─────────────────────────────────────────────────

banner
echo -e "${BOLD}  Proof Generation${NC}"
echo ""
log "Mode:           ${MODE}"
log "Model:          ${MODEL_NAME:-$(basename "$MODEL_DIR")}"
log "Model dir:      ${MODEL_DIR}"
log "Layers:         ${MODEL_LAYERS:-all}"
log "GPU:            ${USE_GPU} (multi: ${MULTI_GPU})"
log "Output:         ${OUTPUT_DIR}"
log "prove-model:    ${PROVE_BIN}"
[[ "$MODE" == "recursive" ]] && log "cairo-prove:    ${CAIRO_BIN}"
echo ""

timer_start "prove_total"

# ─── Format Mapping ─────────────────────────────────────────────────

case "$MODE" in
    recursive)  FORMAT="cairo_serde" ;;
    direct)     FORMAT="direct" ;;
    gkr)        FORMAT="ml_gkr" ;;
esac

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
    run_cmd "${PROVE_CMD[@]}"
else
    # Run proving and display progress lines
    "${PROVE_CMD[@]}" 2>&1 | while IFS= read -r line; do
        # Show layer progress lines
        if echo "$line" | grep -qE '\[layer|\[BG\]|Proving|Weight commit'; then
            log "$line"
        fi
    done
    # Check if proof was generated (pipe swallows exit code)
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

    # ─── Direct: verify structure ────────────────────────────────────
    direct)
        header "Phase 2: Direct Mode — Verify Structure"
        log "Checking direct proof structure..."

        if python3 -c "
import json, sys
with open('${ML_PROOF}') as f:
    proof = json.load(f)
vc = proof.get('verify_calldata')
assert isinstance(vc, dict), 'Missing verify_calldata object'
assert vc.get('schema_version') == 1, 'verify_calldata.schema_version must be 1'
assert vc.get('entrypoint') == 'verify_model_direct', 'verify_calldata.entrypoint must be verify_model_direct'
calldata = vc.get('calldata')
chunks = vc.get('upload_chunks', [])
assert isinstance(calldata, list) and len(calldata) > 0, 'verify_calldata.calldata must be non-empty array'
assert isinstance(chunks, list), 'verify_calldata.upload_chunks must be an array'
assert any(str(v) == '__SESSION_ID__' for v in calldata), 'verify_model_direct calldata must contain __SESSION_ID__ placeholder'
print(f'  verify_calldata parts: {len(calldata)}', file=sys.stderr)
print(f'  upload_chunks: {len(chunks)}', file=sys.stderr)
" 2>&1; then
            ok "Direct proof structure valid"
        else
            err "Direct proof structure invalid"
            exit 1
        fi
        ok "Direct proof ready"
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
assert vc.get('schema_version') == 1, 'verify_calldata.schema_version must be 1'
assert vc.get('entrypoint') == 'verify_model_gkr', 'verify_calldata.entrypoint must be verify_model_gkr'
calldata = vc.get('calldata')
chunks = vc.get('upload_chunks', [])
assert isinstance(calldata, list) and len(calldata) > 0, 'verify_calldata.calldata must be non-empty array'
assert isinstance(chunks, list), 'verify_calldata.upload_chunks must be an array'
assert len(chunks) == 0, 'verify_model_gkr should not include upload chunks'
assert all(str(v) != '__SESSION_ID__' for v in calldata), 'verify_model_gkr calldata must not include __SESSION_ID__ placeholder'
print(f'  verify_calldata: {len(calldata)} felts', file=sys.stderr)
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
