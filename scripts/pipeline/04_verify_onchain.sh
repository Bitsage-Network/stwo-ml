#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 4: On-Chain Verification
# ═══════════════════════════════════════════════════════════════════════
#
# Submits a proof to Starknet and verifies it on-chain.
# Auto-detects proof mode from metadata.json.
#
# Modes:
#   recursive — delegates to submit_recursive_proof.py
#   direct    — upload chunks → verify_model_direct()
#   gkr       — verify_model_gkr()
#
# Usage:
#   bash scripts/pipeline/04_verify_onchain.sh --dry-run
#   bash scripts/pipeline/04_verify_onchain.sh --proof-dir ~/.obelysk/proofs/latest --submit
#   bash scripts/pipeline/04_verify_onchain.sh --proof recursive_proof.json --submit
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/contract_addresses.sh"
source "${SCRIPT_DIR}/lib/starknet_utils.sh"

# ─── Defaults ────────────────────────────────────────────────────────

PROOF_DIR=""
PROOF_FILE=""
NETWORK="sepolia"
CONTRACT_OVERRIDE=""
ACCOUNT="${SNCAST_ACCOUNT:-deployer}"
DO_SUBMIT=false
DO_DRY_RUN=false
MAX_FEE="${MAX_FEE:-0.05}"
MAX_RETRIES=3
ALL_TX_HASHES=()
FORCE_PAYMASTER=false
FORCE_NO_PAYMASTER=false
DEPLOY_ACCOUNT=false

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --proof-dir)   PROOF_DIR="$2"; shift 2 ;;
        --proof)       PROOF_FILE="$2"; shift 2 ;;
        --network)     NETWORK="$2"; shift 2 ;;
        --contract)    CONTRACT_OVERRIDE="$2"; shift 2 ;;
        --account)     ACCOUNT="$2"; shift 2 ;;
        --submit)      DO_SUBMIT=true; shift ;;
        --dry-run)     DO_DRY_RUN=true; shift ;;
        --max-fee)     MAX_FEE="$2"; shift 2 ;;
        --paymaster)   FORCE_PAYMASTER=true; shift ;;
        --no-paymaster) FORCE_NO_PAYMASTER=true; shift ;;
        --deploy-account) DEPLOY_ACCOUNT=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit a proof to Starknet for on-chain verification."
            echo "On Sepolia with no STARKNET_PRIVATE_KEY, uses AVNU paymaster (zero-config)."
            echo ""
            echo "Proof source (pick one):"
            echo "  --proof-dir DIR    Directory with ml_proof.json + metadata.json"
            echo "  --proof FILE       Direct path to proof file"
            echo ""
            echo "Action (required):"
            echo "  --submit           Actually submit transactions on-chain"
            echo "  --dry-run          Print commands without executing"
            echo ""
            echo "Submission mode:"
            echo "  --paymaster        Force AVNU paymaster mode (gasless, sponsored)"
            echo "  --no-paymaster     Force legacy sncast mode (you pay gas in STRK)"
            echo "  --deploy-account   Deploy new agent account via factory before verifying"
            echo "  (default on Sepolia: auto-paymaster when no STARKNET_PRIVATE_KEY)"
            echo ""
            echo "Options:"
            echo "  --network NET      sepolia | mainnet (default: sepolia)"
            echo "  --contract ADDR    Override verifier contract address"
            echo "  --account NAME     sncast account name (default: deployer)"
            echo "  --max-fee ETH      Max fee per TX in ETH (default: 0.05)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Environment variables:"
            echo "  STARKNET_PRIVATE_KEY     Private key (for sncast or paymaster with own account)"
            echo "  STARKNET_ACCOUNT_ADDRESS Account address (when using own key with paymaster)"
            echo "  STARKNET_RPC             Override RPC URL"
            echo "  ALCHEMY_KEY              Alchemy API key for RPC"
            echo "  OBELYSK_DEPLOYER_KEY     Deployer key for factory account creation"
            echo "  OBELYSK_DEPLOYER_ADDRESS Deployer address for factory account creation"
            echo "  AVNU_PAYMASTER_API_KEY   AVNU API key for sponsored mode"
            echo "  SNCAST_ACCOUNT           Default sncast account name"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# Must specify --submit or --dry-run
if [[ "$DO_SUBMIT" == "false" ]] && [[ "$DO_DRY_RUN" == "false" ]]; then
    err "Must specify --submit or --dry-run"
    err "  --dry-run: print commands without executing"
    err "  --submit:  actually submit on-chain"
    exit 1
fi

if [[ "$DO_DRY_RUN" == "true" ]]; then
    DRY_RUN=1
fi

# ─── Resolve Proof ──────────────────────────────────────────────────

init_obelysk_dir

# If neither specified, load from state
if [[ -z "$PROOF_DIR" ]] && [[ -z "$PROOF_FILE" ]]; then
    PROOF_DIR=$(get_state "prove_state.env" "LAST_PROOF_DIR" 2>/dev/null || echo "")
    if [[ -z "$PROOF_DIR" ]]; then
        err "No proof specified. Use --proof-dir or --proof."
        err "Or run 03_prove.sh first."
        exit 1
    fi
    log "Using latest proof: ${PROOF_DIR}"
fi

# Detect mode from metadata or file extension
PROOF_MODE=""
METADATA_FILE=""

if [[ -n "$PROOF_DIR" ]] && [[ -d "$PROOF_DIR" ]]; then
    METADATA_FILE="${PROOF_DIR}/metadata.json"
    if [[ -f "$METADATA_FILE" ]]; then
        PROOF_MODE=$(parse_json_field "$METADATA_FILE" "mode")
        log "Detected mode from metadata: ${PROOF_MODE}"

        # Resolve proof file
        if [[ -z "$PROOF_FILE" ]]; then
            FINAL_PROOF_NAME=$(parse_json_field "$METADATA_FILE" "final_proof")
            PROOF_FILE="${PROOF_DIR}/${FINAL_PROOF_NAME}"
        fi
    else
        warn "metadata.json not found in ${PROOF_DIR}"
    fi
fi

# Verify proof file exists
if [[ -z "$PROOF_FILE" ]] || [[ ! -f "$PROOF_FILE" ]]; then
    # Try common names
    for name in recursive_proof.json ml_proof.json proof.json; do
        if [[ -n "$PROOF_DIR" ]] && [[ -f "${PROOF_DIR}/${name}" ]]; then
            PROOF_FILE="${PROOF_DIR}/${name}"
            break
        fi
    done
fi

check_file "$PROOF_FILE" "Proof file not found: ${PROOF_FILE:-<none>}" || exit 1

# Infer mode if not from metadata
if [[ -z "$PROOF_MODE" ]]; then
    case "$(basename "$PROOF_FILE")" in
        recursive_proof*)  PROOF_MODE="recursive" ;;
        *)                 PROOF_MODE="direct" ;;
    esac
    warn "Inferred mode: ${PROOF_MODE} (from filename)"
fi

# ─── Resolve Contract ───────────────────────────────────────────────

if [[ -n "$CONTRACT_OVERRIDE" ]]; then
    CONTRACT="$CONTRACT_OVERRIDE"
else
    case "$PROOF_MODE" in
        recursive)  CONTRACT=$(get_verifier_address "stark" "$NETWORK") ;;
        direct|gkr) CONTRACT=$(get_verifier_address "elo" "$NETWORK") ;;
    esac
fi

if [[ -z "$CONTRACT" ]]; then
    err "No contract address for mode=${PROOF_MODE} network=${NETWORK}"
    err "Use --contract to specify manually."
    exit 1
fi

RPC_URL=$(get_rpc_url "$NETWORK")

# ─── Determine Submission Mode ──────────────────────────────────────

USE_PAYMASTER=false

if [[ "$FORCE_PAYMASTER" == "true" ]]; then
    USE_PAYMASTER=true
elif [[ "$FORCE_NO_PAYMASTER" == "true" ]]; then
    USE_PAYMASTER=false
elif [[ -z "${STARKNET_PRIVATE_KEY:-}" ]] && [[ "$NETWORK" == "sepolia" ]]; then
    # Zero-config: auto-enable paymaster on Sepolia when no key provided
    USE_PAYMASTER=true
fi

# ─── Check Tools ────────────────────────────────────────────────────

if [[ "$DO_SUBMIT" == "true" ]]; then
    if [[ "$USE_PAYMASTER" == "true" ]]; then
        ensure_node || exit 1
        ensure_starknet_js "${SCRIPT_DIR}/lib" || exit 1
    else
        ensure_sncast || exit 1
        # Auto-create sncast account from STARKNET_PRIVATE_KEY if set
        if [[ -n "${STARKNET_PRIVATE_KEY:-}" ]]; then
            setup_sncast_account "$ACCOUNT" "$NETWORK" || exit 1
        fi
    fi
fi

# CUDA env for any proof parsing — use detected CUDA_PATH if available
_GPU_CONFIG="${OBELYSK_DIR}/gpu_config.env"
if [[ -f "$_GPU_CONFIG" ]]; then
    _DETECTED_CUDA_PATH=$(grep "^CUDA_PATH=" "$_GPU_CONFIG" 2>/dev/null | cut -d'=' -f2-)
fi
_CUDA_LIB="${_DETECTED_CUDA_PATH:-${CUDA_PATH:-/usr/local/cuda}}/lib64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${_CUDA_LIB}:/usr/lib/x86_64-linux-gnu"

# ─── Display Config ─────────────────────────────────────────────────

banner
echo -e "${BOLD}  On-Chain Verification${NC}"
echo ""
log "Mode:       ${PROOF_MODE}"
log "Proof:      ${PROOF_FILE} ($(du -h "$PROOF_FILE" | cut -f1))"
log "Network:    ${NETWORK}"
log "Contract:   ${CONTRACT}"
if [[ "$USE_PAYMASTER" == "true" ]]; then
    log "Submit via: AVNU Paymaster (gasless, sponsored)"
else
    log "Submit via: sncast (account: ${ACCOUNT})"
fi
log "Action:     $([ "$DO_SUBMIT" == "true" ] && echo "SUBMIT" || echo "DRY RUN")"
echo ""

timer_start "verify"

# ─── Invoke Helper with Retry + TX Tracking ─────────────────────────

# Invokes sncast with retry logic. Captures TX hash and waits for confirmation.
# Usage: sncast_invoke_tracked FUNCTION CALLDATA...
sncast_invoke_tracked() {
    local function="$1"; shift
    local calldata=("$@")
    local attempt=0
    local tx_hash=""

    while (( attempt < MAX_RETRIES )); do
        local output
        output=$(sncast --account "$ACCOUNT" --url "$RPC_URL" \
            invoke --contract-address "$CONTRACT" \
            --function "$function" \
            --calldata "${calldata[@]}" \
            --max-fee "$MAX_FEE" 2>&1) || true

        tx_hash=$(echo "$output" | grep -oP '0x[a-fA-F0-9]{50,}' | head -1)

        if [[ -n "$tx_hash" ]]; then
            ALL_TX_HASHES+=("$tx_hash")
            log "TX submitted: ${tx_hash:0:20}..."
            log "  Explorer: $(get_explorer_url "$tx_hash" "$NETWORK")"

            # Wait for confirmation
            if wait_for_tx "$tx_hash" 40 3; then
                return 0
            else
                local status=$?
                if [[ $status -eq 2 ]]; then
                    # REVERTED — do not retry
                    err "TX reverted, not retrying"
                    return 1
                fi
                # Timeout or other — may still succeed, continue
                warn "TX confirmation uncertain, continuing..."
                return 0
            fi
        fi

        (( attempt++ ))
        if (( attempt < MAX_RETRIES )); then
            warn "Invoke failed (attempt ${attempt}/${MAX_RETRIES}), retrying in 5s..."
            sleep 5
        fi
    done

    err "All ${MAX_RETRIES} attempts failed for ${function}"
    return 1
}

# ═══════════════════════════════════════════════════════════════════════
# Mode-specific submission
# ═══════════════════════════════════════════════════════════════════════

MODEL_ID_FROM_META=$(parse_json_field "${METADATA_FILE:-/dev/null}" "model_id" 2>/dev/null || echo "0x1")

if [[ "$USE_PAYMASTER" == "true" ]] && [[ "$PROOF_MODE" != "recursive" ]]; then
    # ─── Paymaster Path (gasless via AVNU) ─────────────────────────────
    header "Paymaster Submission (AVNU Sponsored)"

    PAYMASTER_SCRIPT="${SCRIPT_DIR}/lib/paymaster_submit.mjs"

    # Optional: deploy via factory for ERC-8004 identity (needs deployer key)
    if [[ "$DEPLOY_ACCOUNT" == "true" ]] && [[ -n "${OBELYSK_DEPLOYER_KEY:-}" ]]; then
        log "Deploying pipeline account via factory (ERC-8004 identity)..."
        run_cmd node "$PAYMASTER_SCRIPT" setup \
            --network "$NETWORK" || {
            err "Factory deployment failed — falling back to ephemeral account"
        }
    fi

    # The verify command handles account resolution automatically:
    #   1. STARKNET_PRIVATE_KEY → user's existing account
    #   2. ~/.obelysk/starknet/pipeline_account.json → saved account
    #   3. Neither → auto-generate ephemeral keypair + deploy in same TX
    # No env vars needed for path 3 (true zero-config).

    # Submit via paymaster
    log "Submitting ${PROOF_MODE} proof via AVNU paymaster (gasless)..."
    log "Model ID: ${MODEL_ID_FROM_META}"

    PAYMASTER_OUTPUT=$(run_cmd node "$PAYMASTER_SCRIPT" verify \
        --proof "$PROOF_FILE" \
        --contract "$CONTRACT" \
        --model-id "$MODEL_ID_FROM_META" \
        --network "$NETWORK" 2>&1) || {
        err "Paymaster submission failed"
        echo "$PAYMASTER_OUTPUT" >&2
        exit 1
    }

    # Parse JSON output (last line of stderr is info, stdout is JSON)
    # The script outputs JSON to stdout and logs to stderr
    PAYMASTER_JSON=$(echo "$PAYMASTER_OUTPUT" | grep '^{' | head -1)
    if [[ -z "$PAYMASTER_JSON" ]]; then
        # Try getting JSON from the full output
        PAYMASTER_JSON=$(echo "$PAYMASTER_OUTPUT" | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('{'):
        try:
            json.loads(line)
            print(line)
            break
        except: pass
" 2>/dev/null || echo "")
    fi

    if [[ -n "$PAYMASTER_JSON" ]]; then
        TX_HASH=$(echo "$PAYMASTER_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['txHash'])" 2>/dev/null || echo "")
        IS_VERIFIED_PM=$(echo "$PAYMASTER_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('isVerified','false'))" 2>/dev/null || echo "false")
        EXPLORER_URL=$(echo "$PAYMASTER_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('explorerUrl',''))" 2>/dev/null || echo "")

        if [[ -n "$TX_HASH" ]]; then
            ALL_TX_HASHES+=("$TX_HASH")
            ok "TX submitted via paymaster: ${TX_HASH:0:20}..."
            log "Explorer: ${EXPLORER_URL}"
            log "Gas sponsored: true"
        fi
    else
        warn "Could not parse paymaster output"
        echo "$PAYMASTER_OUTPUT" >&2
    fi

else
    # ─── Legacy sncast Path ────────────────────────────────────────────

    case "$PROOF_MODE" in

        # ─── Recursive: delegate to submit_recursive_proof.py ─────────
        recursive)
            header "Recursive STARK Submission"

            # Find the submit script
            SUBMIT_SCRIPT=""
            for path in \
                "${SCRIPT_DIR}/../submit_recursive_proof.py" \
                "${SCRIPT_DIR}/../../scripts/submit_recursive_proof.py" \
                "scripts/submit_recursive_proof.py" \
                "../scripts/submit_recursive_proof.py"; do
                if [[ -f "$path" ]]; then
                    SUBMIT_SCRIPT="$path"
                    break
                fi
            done

            if [[ -z "$SUBMIT_SCRIPT" ]]; then
                err "submit_recursive_proof.py not found"
                err "Ensure it exists in the scripts/ directory"
                exit 1
            fi

            log "Submit script: ${SUBMIT_SCRIPT}"

            MODE_FLAG="--dry-run"
            if [[ "$DO_SUBMIT" == "true" ]]; then
                MODE_FLAG="--submit"
            fi

            run_cmd python3 "${SUBMIT_SCRIPT}" \
                --proof "${PROOF_FILE}" \
                --account "${ACCOUNT}" \
                ${MODE_FLAG}
            ;;

        # ─── Direct: upload chunks + verify_model_direct ──────────────
        direct)
            header "Direct Proof Submission"

            SESSION_ID="0x$(date +%s | xxd -p 2>/dev/null | head -c 16 || printf '%x' "$(date +%s)")"
            log "Session ID: ${SESSION_ID}"

            # Upload STARK chunks if present
            CHUNK_DIR="${PROOF_DIR:-$(dirname "$PROOF_FILE")}/chunks"
            if [[ -d "$CHUNK_DIR" ]]; then
                CHUNK_COUNT=$(ls "$CHUNK_DIR"/chunk_*.json 2>/dev/null | wc -l | tr -d ' ')
                log "Uploading ${CHUNK_COUNT} STARK chunks..."

                for i in $(seq 0 $((CHUNK_COUNT - 1))); do
                    CHUNK_FILE="${CHUNK_DIR}/chunk_${i}.json"
                    if [[ -f "$CHUNK_FILE" ]]; then
                        CHUNK_SIZE=$(wc -c < "$CHUNK_FILE" | tr -d ' ')
                        log "  Chunk ${i}: ${CHUNK_SIZE} bytes"

                        CHUNK_DATA=$(cat "$CHUNK_FILE")
                        run_cmd sncast_invoke_tracked upload_proof_chunk "$SESSION_ID" "$i" $CHUNK_DATA

                        ok "  Chunk ${i} uploaded"
                    fi
                done
            else
                log "No STARK chunks to upload"
            fi

            # Call verify_model_direct
            log "Calling verify_model_direct..."

            BATCHED_DATA="${PROOF_DIR:-$(dirname "$PROOF_FILE")}/batched_calldata.json"
            CALLDATA=""
            if [[ -f "$BATCHED_DATA" ]]; then
                CALLDATA=$(cat "$BATCHED_DATA")
            fi

            run_cmd sncast_invoke_tracked verify_model_direct "${MODEL_ID_FROM_META}" "$SESSION_ID" $CALLDATA

            ok "verify_model_direct submitted"
            ;;

        # ─── GKR: verify_model_gkr ───────────────────────────────────
        gkr)
            header "GKR Proof Submission"

            log "Submitting GKR proof for model ${MODEL_ID_FROM_META}..."

            # Read the GKR calldata from proof file
            if command -v python3 &>/dev/null && [[ -f "$PROOF_FILE" ]]; then
                GKR_CALLDATA=$(python3 -c "
import json
with open('${PROOF_FILE}') as f:
    proof = json.load(f)
# Extract calldata array if present
if 'calldata' in proof:
    print(' '.join(str(x) for x in proof['calldata']))
elif 'gkr_calldata' in proof:
    print(' '.join(str(x) for x in proof['gkr_calldata']))
else:
    print('')
" 2>/dev/null || echo "")
            fi

            if [[ -n "${GKR_CALLDATA:-}" ]]; then
                run_cmd sncast_invoke_tracked verify_model_gkr "${MODEL_ID_FROM_META}" $GKR_CALLDATA

                ok "verify_model_gkr submitted"
            else
                warn "Could not extract GKR calldata from proof file"
                warn "Manual submission may be required"
            fi
            ;;
    esac
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Post-Verification
# ═══════════════════════════════════════════════════════════════════════

ELAPSED=$(timer_elapsed "verify")

if [[ "$DO_SUBMIT" == "true" ]]; then
    header "Post-Verification"

    # Show all submitted TXs with explorer links
    if (( ${#ALL_TX_HASHES[@]} > 0 )); then
        log "Submitted ${#ALL_TX_HASHES[@]} transaction(s):"
        for tx in "${ALL_TX_HASHES[@]}"; do
            log "  $(get_explorer_url "$tx" "$NETWORK")"
        done
    fi
    echo ""

    # Check on-chain verification status
    MODEL_ID_CHECK=$(parse_json_field "${METADATA_FILE:-/dev/null}" "model_id" 2>/dev/null || echo "0x1")
    if [[ "$USE_PAYMASTER" == "true" ]] && [[ -n "${IS_VERIFIED_PM:-}" ]]; then
        # Paymaster script already checked verification
        IS_VERIFIED="$IS_VERIFIED_PM"
    elif command -v sncast &>/dev/null; then
        IS_VERIFIED=$(check_is_verified "$CONTRACT" "$MODEL_ID_CHECK" "$RPC_URL" 2>/dev/null || echo "false")
    else
        # No sncast and not paymaster — try via node
        IS_VERIFIED=$(node "${SCRIPT_DIR}/lib/paymaster_submit.mjs" status \
            --contract "$CONTRACT" --model-id "$MODEL_ID_CHECK" --network "$NETWORK" 2>/dev/null \
            | python3 -c "import sys,json; print(str(json.load(sys.stdin).get('verification',{}).get('isVerified','false')).lower())" 2>/dev/null || echo "false")
    fi
    if [[ "$IS_VERIFIED" == "true" ]] || [[ "$IS_VERIFIED" == "True" ]]; then
        IS_VERIFIED="true"
        ok "On-chain verification: VERIFIED"
    else
        warn "On-chain verification status: unconfirmed (may need time to propagate)"
    fi
    echo ""

    # Save receipt
    LAST_TX="${ALL_TX_HASHES[${#ALL_TX_HASHES[@]}-1]:-}"
    RECEIPT_FILE="${OUTPUT_DIR:-/tmp}/verify_receipt.json"
    SUBMIT_VIA="sncast"
    [[ "$USE_PAYMASTER" == "true" ]] && SUBMIT_VIA="avnu_paymaster"

    python3 -c "
import json
receipt = {
    'network': '${NETWORK}',
    'contract': '${CONTRACT}',
    'mode': '${PROOF_MODE}',
    'submit_via': '${SUBMIT_VIA}',
    'proof_file': '${PROOF_FILE}',
    'submitted_at': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'elapsed_seconds': ${ELAPSED},
    'tx_hashes': $(python3 -c "import json; print(json.dumps([$(printf "'%s'," "${ALL_TX_HASHES[@]}" | sed 's/,$//')])" 2>/dev/null || echo '[]'),
    'is_verified': '${IS_VERIFIED}',
    'gas_sponsored': ${USE_PAYMASTER},
}
with open('${RECEIPT_FILE}', 'w') as f:
    json.dump(receipt, f, indent=2)
" 2>/dev/null || {
        # Fallback if python fails
        cat > "$RECEIPT_FILE" << RCPTEOF
{
    "network": "${NETWORK}",
    "contract": "${CONTRACT}",
    "mode": "${PROOF_MODE}",
    "submit_via": "${SUBMIT_VIA}",
    "proof_file": "${PROOF_FILE}",
    "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "elapsed_seconds": ${ELAPSED},
    "last_tx": "${LAST_TX}",
    "is_verified": "${IS_VERIFIED}",
    "gas_sponsored": ${USE_PAYMASTER}
}
RCPTEOF
    }
    ok "Receipt saved to ${RECEIPT_FILE}"
fi

# ─── Summary ─────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
if [[ "$DO_SUBMIT" == "true" ]]; then
echo "  ║  ON-CHAIN SUBMISSION COMPLETE                        ║"
else
echo "  ║  DRY RUN COMPLETE                                    ║"
fi
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Mode:         %-36s ║\n" "${PROOF_MODE}"
printf "  ║  Network:      %-36s ║\n" "${NETWORK}"
printf "  ║  Contract:     %-36s ║\n" "${CONTRACT:0:36}"
printf "  ║  Duration:     %-36s ║\n" "$(format_duration $ELAPSED)"
if [[ "$USE_PAYMASTER" == "true" ]]; then
printf "  ║  Submit via:   %-36s ║\n" "AVNU Paymaster (gasless)"
fi
if [[ "$DO_SUBMIT" == "true" ]] && (( ${#ALL_TX_HASHES[@]} > 0 )); then
printf "  ║  TXs:          %-36s ║\n" "${#ALL_TX_HASHES[@]} submitted"
printf "  ║  Last TX:      %-36s ║\n" "${ALL_TX_HASHES[${#ALL_TX_HASHES[@]}-1]:0:36}"
printf "  ║  Verified:     %-36s ║\n" "${IS_VERIFIED:-unknown}"
fi
echo "  ╠══════════════════════════════════════════════════════╣"
if [[ "$DO_SUBMIT" == "false" ]]; then
echo "  ║                                                      ║"
echo "  ║  Re-run with --submit to execute on-chain            ║"
fi
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
