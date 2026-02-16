#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 7: Privacy Transfer (Shielded Send)
# ═══════════════════════════════════════════════════════════════════════
#
# Generates a shielded transfer proof and optionally submits it on-chain.
# Transfers tokens within the privacy pool to a recipient public key.
#
# Usage:
#   bash scripts/pipeline/07_privacy_transfer.sh --amount 500 --to 0xABCD... --dry-run
#   bash scripts/pipeline/07_privacy_transfer.sh --amount 500 --to 0xABCD... --submit
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/contract_addresses.sh"

# ─── Defaults ────────────────────────────────────────────────────────

AMOUNT=""
ASSET_ID="0"
RECIPIENT=""
NETWORK="sepolia"
WALLET_PATH=""
PASSWORD=""
OUTPUT_PATH=""
DO_SUBMIT=false
DO_DRY_RUN=false
ACCOUNT="${SNCAST_ACCOUNT:-deployer}"
POOL_OVERRIDE=""

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --amount)     AMOUNT="$2"; shift 2 ;;
        --asset)      ASSET_ID="$2"; shift 2 ;;
        --to)         RECIPIENT="$2"; shift 2 ;;
        --network)    NETWORK="$2"; shift 2 ;;
        --wallet)     WALLET_PATH="$2"; shift 2 ;;
        --password)   PASSWORD="$2"; shift 2 ;;
        --output)     OUTPUT_PATH="$2"; shift 2 ;;
        --pool-contract) POOL_OVERRIDE="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --submit)     DO_SUBMIT=true; shift ;;
        --dry-run)    DO_DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Send a shielded transfer within the VM31 privacy pool."
            echo ""
            echo "Required:"
            echo "  --amount NUM       Amount to transfer"
            echo "  --to HEX           Recipient public key (hex)"
            echo ""
            echo "Action (required):"
            echo "  --submit           Prove and submit on-chain"
            echo "  --dry-run          Prove only, do not submit"
            echo ""
            echo "Options:"
            echo "  --asset ID         Asset ID (default: 0 = STRK)"
            echo "  --wallet PATH      Path to wallet file (default: ~/.vm31/wallet.json)"
            echo "  --password PW      Wallet password (if encrypted)"
            echo "  --output PATH      Output proof JSON path"
            echo "  --network NET      sepolia | mainnet (default: sepolia)"
            echo "  --pool-contract ADDR  Override pool contract address"
            echo "  --account NAME     sncast account name (default: deployer)"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────

if [[ -z "$AMOUNT" ]]; then
    err "Missing --amount"
    exit 1
fi

if [[ -z "$RECIPIENT" ]]; then
    err "Missing --to (recipient public key)"
    exit 1
fi

if [[ "$DO_SUBMIT" == "false" ]] && [[ "$DO_DRY_RUN" == "false" ]]; then
    err "Must specify --submit or --dry-run"
    exit 1
fi

# ─── Resolve Paths ───────────────────────────────────────────────────

init_obelysk_dir

PROVE_MODEL="${SCRIPT_DIR}/../../stwo-ml/target/release/prove-model"
if [[ ! -x "$PROVE_MODEL" ]]; then
    PROVE_MODEL="prove-model"
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    OUTPUT_PATH="${OBELYSK_DIR}/privacy/transfer_proof.json"
    mkdir -p "$(dirname "$OUTPUT_PATH")"
fi

# ─── Display Config ──────────────────────────────────────────────────

banner
echo -e "${BOLD}  Privacy Transfer${NC}"
echo ""
log "Amount:     ${AMOUNT}"
log "Asset:      ${ASSET_ID}"
log "Recipient:  ${RECIPIENT:0:20}..."
log "Network:    ${NETWORK}"
log "Output:     ${OUTPUT_PATH}"
log "Action:     $([ "$DO_SUBMIT" == "true" ] && echo "PROVE + SUBMIT" || echo "PROVE ONLY")"
echo ""

timer_start "transfer"

# ─── Build CLI args ──────────────────────────────────────────────────

CLI_ARGS=("transfer" "--amount" "$AMOUNT" "--asset" "$ASSET_ID" "--to" "$RECIPIENT" "--output" "$OUTPUT_PATH")

if [[ -n "$WALLET_PATH" ]]; then
    CLI_ARGS+=("--wallet" "$WALLET_PATH")
fi

if [[ -n "$PASSWORD" ]]; then
    CLI_ARGS+=("--password" "$PASSWORD")
fi

if [[ -n "$POOL_OVERRIDE" ]]; then
    CLI_ARGS+=("--pool-contract" "$POOL_OVERRIDE")
elif [[ -n "$(get_pool_address "$NETWORK" 2>/dev/null || echo "")" ]]; then
    CLI_ARGS+=("--pool-contract" "$(get_pool_address "$NETWORK")")
fi

CLI_ARGS+=("--network" "$NETWORK")

# ─── Prove ───────────────────────────────────────────────────────────

header "Generating Transfer Proof"
log "Running: prove-model ${CLI_ARGS[*]}"

run_cmd "$PROVE_MODEL" "${CLI_ARGS[@]}" 2>&1

if [[ ! -f "$OUTPUT_PATH" ]]; then
    err "Proof output not found at ${OUTPUT_PATH}"
    exit 1
fi

ok "Transfer proof saved to ${OUTPUT_PATH}"

# ─── Submit On-Chain ─────────────────────────────────────────────────

if [[ "$DO_SUBMIT" == "true" ]]; then
    header "On-Chain Submission"

    POOL_ADDRESS="${POOL_OVERRIDE:-$(get_pool_address "$NETWORK" 2>/dev/null || echo "")}"
    if [[ -z "$POOL_ADDRESS" ]]; then
        err "No pool contract address for network=${NETWORK}"
        err "Use --pool-contract to specify manually."
        exit 1
    fi

    RPC_URL=$(get_rpc_url "$NETWORK")
    log "Pool contract: ${POOL_ADDRESS}"
    log "RPC: ${RPC_URL}"

    CALLDATA=$(python3 -c "
import json, sys
with open('${OUTPUT_PATH}') as f:
    data = json.load(f)
calldata = data.get('calldata', [])
print(' '.join(str(c) for c in calldata))
" 2>/dev/null || echo "")

    if [[ -z "$CALLDATA" ]]; then
        err "Could not extract calldata from proof"
        exit 1
    fi

    # shellcheck disable=SC2206
    CALLDATA_ARR=($CALLDATA)

    log "Submitting batch proof (${#CALLDATA_ARR[@]} calldata elements)..."
    run_cmd sncast --account "$ACCOUNT" --url "$RPC_URL" \
        invoke --contract-address "$POOL_ADDRESS" \
        --function "submit_batch_proof" \
        --calldata "${CALLDATA_ARR[@]}"

    ok "Transfer submitted on-chain"
fi

# ─── Summary ─────────────────────────────────────────────────────────

ELAPSED=$(timer_elapsed "transfer")
echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  PRIVACY TRANSFER COMPLETE                           ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  Amount:       %-36s ║\n" "${AMOUNT}"
printf "  ║  Asset:        %-36s ║\n" "${ASSET_ID}"
printf "  ║  Recipient:    %-36s ║\n" "${RECIPIENT:0:36}"
printf "  ║  Network:      %-36s ║\n" "${NETWORK}"
printf "  ║  Duration:     %-36s ║\n" "$(format_duration $ELAPSED)"
printf "  ║  Proof:        %-36s ║\n" "$(basename "$OUTPUT_PATH")"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
