#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 5: Inference Audit
# ═══════════════════════════════════════════════════════════════════════
#
# Runs the verifiable inference audit: prove + evaluate + report over
# an inference log, with optional encryption and on-chain submission.
#
# Usage:
#   bash scripts/pipeline/05_audit.sh --evaluate
#   bash scripts/pipeline/05_audit.sh --evaluate --submit --privacy private
#   bash scripts/pipeline/05_audit.sh --log-dir /path/to/logs --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/contract_addresses.sh"

# ─── Defaults ────────────────────────────────────────────────────────

LOG_DIR=""
MODEL_DIR_OVERRIDE=""
OUTPUT=""
PRIVACY="public"
ENCRYPTION="poseidon2"
OWNER_PUBKEY=""
DO_EVALUATE=false
DO_PROVE_EVALS=false
DO_SUBMIT=false
DO_DRY_RUN=false
NETWORK="sepolia"
CONTRACT_OVERRIDE=""
ACCOUNT="deployer"
MAX_FEE="0.05"
LAYERS=""
START_WINDOW="all"
END_WINDOW="now"
MAX_INFERENCES=0
EXTRA_FEATURES=""

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)          LOG_DIR="$2"; shift 2 ;;
        --model-dir)        MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
        --output)           OUTPUT="$2"; shift 2 ;;
        --privacy)          PRIVACY="$2"; shift 2 ;;
        --encryption)       ENCRYPTION="$2"; shift 2 ;;
        --owner-pubkey)     OWNER_PUBKEY="$2"; shift 2 ;;
        --evaluate)         DO_EVALUATE=true; shift ;;
        --prove-evals)      DO_PROVE_EVALS=true; DO_EVALUATE=true; shift ;;
        --submit)           DO_SUBMIT=true; shift ;;
        --dry-run)          DO_DRY_RUN=true; shift ;;
        --network)          NETWORK="$2"; shift 2 ;;
        --contract)         CONTRACT_OVERRIDE="$2"; shift 2 ;;
        --account)          ACCOUNT="$2"; shift 2 ;;
        --max-fee)          MAX_FEE="$2"; shift 2 ;;
        --layers)           LAYERS="$2"; shift 2 ;;
        --start)            START_WINDOW="$2"; shift 2 ;;
        --end)              END_WINDOW="$2"; shift 2 ;;
        --max-inferences)   MAX_INFERENCES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run verifiable inference audit over an inference log."
            echo ""
            echo "Options:"
            echo "  --log-dir DIR       Inference log directory (default: from pipeline state)"
            echo "  --model-dir DIR     Model directory override (default: from pipeline state)"
            echo "  --output FILE       Output report path (default: \$PROOF_DIR/audit_report.json)"
            echo "  --privacy TIER      Privacy tier: public, private, selective (default: public)"
            echo "  --encryption MODE   Encryption: poseidon2, aes, none, noop (default: poseidon2)"
            echo "  --owner-pubkey HEX  Owner public key for encryption"
            echo "  --evaluate          Run semantic evaluation"
            echo "  --prove-evals       Prove evaluation forward passes (implies --evaluate)"
            echo "  --submit            Submit audit on-chain via sncast"
            echo "  --dry-run           Skip encryption and on-chain submission"
            echo "  --network NET       Starknet network (default: sepolia)"
            echo "  --contract ADDR     Contract address override"
            echo "  --account NAME      sncast account name (default: deployer)"
            echo "  --max-fee ETH       Max fee in ETH (default: 0.05)"
            echo "  --layers N          Number of layers"
            echo "  --start SPEC        Audit window start (default: all)"
            echo "  --end SPEC          Audit window end (default: now)"
            echo "  --max-inferences N  Max inferences to prove (default: 0 = all)"
            echo ""
            echo "Environment variables:"
            echo "  IRYS_TOKEN          Irys API token for Arweave uploads (required for --privacy private)"
            echo ""
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Resolve Paths from Pipeline State ───────────────────────────────

header "Step 5: Inference Audit"

init_obelysk_dir
timer_start "audit"

# Model directory
if [[ -z "$MODEL_DIR_OVERRIDE" ]]; then
    MODEL_DIR_OVERRIDE=$(get_state "model_state.env" "MODEL_DIR" 2>/dev/null || echo "")
fi

if [[ -z "$MODEL_DIR_OVERRIDE" ]]; then
    err "No model directory found. Specify --model-dir or run 01_setup_model.sh first."
    exit 1
fi
log "Model dir:   ${MODEL_DIR_OVERRIDE}"

# Proof directory (for output location)
PROOF_DIR=$(get_state "prove_state.env" "LAST_PROOF_DIR" 2>/dev/null || echo "")

# Log directory — prefer capture_state.env (from 02b_capture_inference.sh)
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR=$(get_state "capture_state.env" "AUDIT_LOG_DIR" 2>/dev/null || echo "")
fi
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR=$(get_state "audit_state.env" "AUDIT_LOG_DIR" 2>/dev/null || echo "")
fi
if [[ -z "$LOG_DIR" ]]; then
    MODEL_NAME=$(basename "$MODEL_DIR_OVERRIDE")
    LOG_DIR="${OBELYSK_DIR}/logs/${MODEL_NAME}"
fi
log "Log dir:     ${LOG_DIR}"

# Validate log dir has meta.json
if [[ ! -f "${LOG_DIR}/meta.json" ]]; then
    err "Inference log not found: ${LOG_DIR}/meta.json"
    err "Run 02b_capture_inference.sh first to generate an inference log."
    exit 1
fi

# Output path
if [[ -z "$OUTPUT" ]]; then
    if [[ -n "$PROOF_DIR" ]]; then
        OUTPUT="${PROOF_DIR}/audit_report.json"
    else
        OUTPUT="${OBELYSK_DIR}/proofs/audit_report.json"
    fi
fi
log "Output:      ${OUTPUT}"
mkdir -p "$(dirname "$OUTPUT")"

# ─── Determine Features ─────────────────────────────────────────────

FEATURES="cli,audit,model-loading,safetensors"
if command -v nvcc &>/dev/null || [[ -f /usr/local/cuda/bin/nvcc ]]; then
    FEATURES="cli,audit,model-loading,safetensors,cuda-runtime"
fi

if [[ "$ENCRYPTION" == "aes" ]]; then
    FEATURES="${FEATURES},aes-fallback"
fi

if [[ "$PRIVACY" != "public" ]] && [[ "$ENCRYPTION" != "none" ]]; then
    FEATURES="${FEATURES},audit-http"
fi

log "Features:    ${FEATURES}"
log "Privacy:     ${PRIVACY}"
log "Evaluate:    ${DO_EVALUATE}"
log "Submit:      ${DO_SUBMIT}"

# Export marketplace credentials for Rust binary (if registered)
if [[ -n "${MARKETPLACE_API_KEY:-}" ]]; then
    export MARKETPLACE_URL="${MARKETPLACE_URL:-https://marketplace.bitsage.xyz}"
    export MARKETPLACE_API_KEY
    log "Storage:     marketplace (${MARKETPLACE_URL})"
elif [[ -z "${IRYS_TOKEN:-}" ]]; then
    if [[ "$PRIVACY" != "public" ]] && [[ "$ENCRYPTION" != "none" ]]; then
        # No local IRYS_TOKEN — use the Obelysk audit relay (coordinator EC2 holds the token)
        RELAY_URL="${OBELYSK_RELAY_URL:-https://relay.obelysk.xyz}"
        log "No IRYS_TOKEN — audit uploads will route through relay: ${RELAY_URL}"
        export OBELYSK_RELAY_URL="$RELAY_URL"
    fi
else
    log "Storage:     irys-direct"
fi
echo ""

# ─── Resolve Contract Address ────────────────────────────────────────

if [[ -z "$CONTRACT_OVERRIDE" ]]; then
    CONTRACT_OVERRIDE=$(get_audit_contract "$NETWORK")
fi

# ─── Build Command ───────────────────────────────────────────────────

AUDIT_CMD=(
    cargo run --release
    --manifest-path "${SCRIPT_DIR}/../../stwo-ml/Cargo.toml"
    --bin prove-model
    --features "${FEATURES}"
    -- audit
    --log-dir "$LOG_DIR"
    --model-dir "$MODEL_DIR_OVERRIDE"
    --output "$OUTPUT"
    --privacy "$PRIVACY"
    --start "$START_WINDOW"
    --end "$END_WINDOW"
)

[[ -n "$LAYERS" ]] && AUDIT_CMD+=(--layers "$LAYERS")
[[ "$DO_EVALUATE" == "true" ]] && AUDIT_CMD+=(--evaluate)
[[ "$DO_PROVE_EVALS" == "true" ]] && AUDIT_CMD+=(--prove-evals)
[[ "$MAX_INFERENCES" != "0" ]] && AUDIT_CMD+=(--max-inferences "$MAX_INFERENCES")

if [[ "$DO_DRY_RUN" == "true" ]]; then
    AUDIT_CMD+=(--dry-run)
elif [[ "$DO_SUBMIT" == "true" ]]; then
    AUDIT_CMD+=(--submit --contract "$CONTRACT_OVERRIDE" --network "$NETWORK" --account "$ACCOUNT" --max-fee "$MAX_FEE")
fi

if [[ "$ENCRYPTION" != "none" ]]; then
    AUDIT_CMD+=(--encryption "$ENCRYPTION")
    [[ -n "$OWNER_PUBKEY" ]] && AUDIT_CMD+=(--owner-pubkey "$OWNER_PUBKEY")
fi

# ─── Run ─────────────────────────────────────────────────────────────

step "5.1" "Running audit pipeline..."
run_cmd "${AUDIT_CMD[@]}" || { err "Audit pipeline failed"; exit 1; }

# ─── Save State ──────────────────────────────────────────────────────

ELAPSED=$(timer_elapsed "audit")

save_state "audit_state.env" \
    "LAST_AUDIT_REPORT=${OUTPUT}" \
    "LAST_AUDIT_PRIVACY=${PRIVACY}" \
    "LAST_AUDIT_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "AUDIT_LOG_DIR=${LOG_DIR}"

ok "Audit complete in $(format_duration $ELAPSED)"
log "Report: ${OUTPUT}"
