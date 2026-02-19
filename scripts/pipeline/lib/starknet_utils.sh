#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Starknet Utilities
# ═══════════════════════════════════════════════════════════════════════
#
# Source this after common.sh:
#   source "${SCRIPT_DIR}/lib/common.sh"
#   source "${SCRIPT_DIR}/lib/starknet_utils.sh"
#
# Provides:
#   - ensure_sncast        → install sncast if missing
#   - setup_sncast_account → create account from STARKNET_PRIVATE_KEY env var
#   - wait_for_tx          → poll TX status until confirmed
#   - call_contract        → read-only contract call
#   - check_is_verified    → check if a model proof is verified on-chain
#   - estimate_gas         → estimate gas for an invocation

[[ -n "${_OBELYSK_STARKNET_UTILS_LOADED:-}" ]] && return 0
_OBELYSK_STARKNET_UTILS_LOADED=1

# ─── Install Node.js ───────────────────────────────────────────────

ensure_node() {
    if command -v node &>/dev/null; then
        local ver
        ver=$(node --version 2>/dev/null)
        debug "Node.js found: $ver"
        return 0
    fi

    log "Node.js not found — installing via nvm..."
    if curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh 2>/dev/null | bash 2>/dev/null; then
        export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
        # shellcheck disable=SC1091
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
        nvm install 22 && nvm use 22
    fi

    if command -v node &>/dev/null; then
        ok "Node.js installed: $(node --version 2>/dev/null)"
        return 0
    fi

    err "Failed to install Node.js."
    err "Install manually: https://nodejs.org or 'brew install node'"
    return 1
}

# ─── Install starknet.js ──────────────────────────────────────────

ensure_starknet_js() {
    local lib_dir="${1:-${SCRIPT_DIR}/lib}"
    if [[ -d "${lib_dir}/node_modules/starknet" ]]; then
        debug "starknet.js already installed"
        return 0
    fi

    log "Installing starknet.js..."
    if (cd "$lib_dir" && npm install --production 2>&1 | tail -3); then
        ok "starknet.js installed"
        return 0
    fi

    err "Failed to install starknet.js."
    err "Manual: cd ${lib_dir} && npm install"
    return 1
}

# ─── Install sncast ─────────────────────────────────────────────────

ensure_sncast() {
    if command -v sncast &>/dev/null; then
        debug "sncast found: $(command -v sncast)"
        return 0
    fi

    log "sncast not found — installing starknet-foundry..."
    if curl -L https://raw.githubusercontent.com/foundry-rs/starknet-foundry/master/scripts/install.sh 2>/dev/null | sh 2>/dev/null; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if command -v sncast &>/dev/null; then
        ok "sncast installed: $(sncast --version 2>/dev/null || echo 'unknown version')"
        return 0
    fi

    err "Failed to install sncast."
    err "Manual: curl -L https://raw.githubusercontent.com/foundry-rs/starknet-foundry/master/scripts/install.sh | sh"
    return 1
}

# ─── Account Setup ──────────────────────────────────────────────────
#
# Creates an sncast account config from STARKNET_PRIVATE_KEY env var.
# If the account file already exists (at ~/.obelysk/starknet/accounts.json),
# it is reused.

setup_sncast_account() {
    local account_name="${1:-deployer}"
    local network="${2:-sepolia}"

    # Require private key
    if [[ -z "${STARKNET_PRIVATE_KEY:-}" ]]; then
        err "STARKNET_PRIVATE_KEY environment variable not set."
        err "Export it before running on-chain commands:"
        err "  export STARKNET_PRIVATE_KEY=0x..."
        return 1
    fi

    local accounts_dir="${OBELYSK_DIR}/starknet"
    local accounts_file="${accounts_dir}/accounts.json"
    mkdir -p "$accounts_dir"

    # Check if account already configured
    if [[ -f "$accounts_file" ]] && python3 -c "
import json, sys
with open('${accounts_file}') as f:
    data = json.load(f)
acc = data.get('alpha-${network}', data.get('${network}', {}))
if '${account_name}' in acc:
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
        debug "Account '${account_name}' already exists in ${accounts_file}"
        SNCAST_ACCOUNTS_FILE="$accounts_file"
        export SNCAST_ACCOUNTS_FILE
        return 0
    fi

    log "Creating sncast account '${account_name}' for ${network}..."

    # Use sncast account create if available
    if command -v sncast &>/dev/null; then
        # sncast needs --accounts-file to know where to save
        run_cmd sncast \
            --accounts-file "$accounts_file" \
            account create \
            --name "$account_name" \
            --network "$network" \
            --private-key "$STARKNET_PRIVATE_KEY" \
            --add-profile 2>/dev/null || true
    fi

    # If sncast create didn't work, build the account file manually
    if [[ ! -f "$accounts_file" ]] || ! python3 -c "
import json
with open('${accounts_file}') as f:
    data = json.load(f)
assert '${account_name}' in data.get('alpha-${network}', data.get('${network}', {}))
" 2>/dev/null; then
        log "Building account config manually..."
        # Derive address from private key using starkli (if available)
        local address=""
        if command -v starkli &>/dev/null; then
            address=$(starkli account address --private-key "$STARKNET_PRIVATE_KEY" 2>/dev/null || echo "")
        fi
        # If no address, use a placeholder — user must provide STARKNET_ACCOUNT_ADDRESS
        if [[ -z "$address" ]]; then
            address="${STARKNET_ACCOUNT_ADDRESS:-0x0}"
            if [[ "$address" == "0x0" ]]; then
                warn "Could not derive account address from private key."
                warn "Set STARKNET_ACCOUNT_ADDRESS env var for correct address."
            fi
        fi

        python3 -c "
import json, os
accounts_file = '${accounts_file}'
try:
    with open(accounts_file) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}

network_key = 'alpha-${network}'
if network_key not in data:
    data[network_key] = {}

data[network_key]['${account_name}'] = {
    'private_key': '${STARKNET_PRIVATE_KEY}',
    'address': '${address}',
    'deployed': True,
    'legacy': False,
}

with open(accounts_file, 'w') as f:
    json.dump(data, f, indent=2)
print(f'Account saved to {accounts_file}')
"
    fi

    SNCAST_ACCOUNTS_FILE="$accounts_file"
    export SNCAST_ACCOUNTS_FILE
    ok "Account '${account_name}' configured"
    return 0
}

# ─── Wait for Transaction ───────────────────────────────────────────

wait_for_tx() {
    local tx_hash="$1"
    local max_attempts="${2:-40}"
    local interval="${3:-3}"

    if [[ -z "$tx_hash" ]]; then
        err "wait_for_tx: no TX hash provided"
        return 1
    fi

    log "Waiting for TX ${tx_hash:0:20}..."

    local attempt=0
    while (( attempt < max_attempts )); do
        local output
        output=$(sncast tx-status "$tx_hash" 2>/dev/null || echo "")

        if echo "$output" | grep -qi "ACCEPTED_ON_L2"; then
            if echo "$output" | grep -qi "REVERTED"; then
                err "TX REVERTED: ${tx_hash}"
                return 2
            fi
            ok "TX accepted on L2: ${tx_hash:0:20}..."
            return 0
        fi

        if echo "$output" | grep -qi "REJECTED"; then
            err "TX REJECTED: ${tx_hash}"
            return 3
        fi

        (( attempt++ ))
        if (( attempt % 5 == 0 )); then
            log "  Still waiting... (attempt ${attempt}/${max_attempts})"
        fi
        sleep "$interval"
    done

    warn "TX confirmation timed out after $((max_attempts * interval))s"
    warn "TX may still be pending: ${tx_hash}"
    return 4
}

# ─── Contract Call (read-only) ──────────────────────────────────────

call_contract() {
    local address="$1"
    local function="$2"
    local calldata="$3"
    local rpc_url="${4:-}"

    local args=("--contract-address" "$address" "--function" "$function")
    if [[ -n "$calldata" ]]; then
        args+=("--calldata" "$calldata")
    fi
    if [[ -n "$rpc_url" ]]; then
        args=("--url" "$rpc_url" "${args[@]}")
    fi

    sncast call "${args[@]}" 2>/dev/null
}

# ─── Check Verification Status ──────────────────────────────────────

get_verification_count() {
    local contract="$1"
    local model_id="$2"
    local rpc_url="${3:-}"

    local args=("--contract-address" "$contract" "--function" "get_verification_count" "--calldata" "$model_id")
    if [[ -n "$rpc_url" ]]; then
        args=("--url" "$rpc_url" "${args[@]}")
    fi

    local result raw
    result=$(sncast call "${args[@]}" 2>/dev/null || echo "")
    raw=$(echo "$result" | grep -oE '0x[0-9a-fA-F]+' | head -1)

    if [[ -z "$raw" ]]; then
        raw=$(echo "$result" | grep -oE '^[0-9]+' | head -1)
    fi

    if [[ -z "$raw" ]]; then
        return 1
    fi

    python3 - "$raw" <<'PY'
import sys
v = sys.argv[1].strip()
print(int(v, 0))
PY
}

check_is_verified() {
    local contract="$1"
    local model_id="$2"
    local rpc_url="${3:-}"

    local count=""
    count=$(get_verification_count "$contract" "$model_id" "$rpc_url" 2>/dev/null || echo "")

    if [[ -n "$count" ]]; then
        if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" != "0" ]]; then
            echo "true"
            return 0
        fi
        echo "false"
        return 1
    fi

    # Fallback for legacy contracts that only expose is_proof_verified
    local args=("--contract-address" "$contract" "--function" "is_proof_verified" "--calldata" "$model_id")
    if [[ -n "$rpc_url" ]]; then
        args=("--url" "$rpc_url" "${args[@]}")
    fi

    local result
    result=$(sncast call "${args[@]}" 2>/dev/null || echo "")

    if echo "$result" | grep -qE "^0x[1-9a-fA-F]"; then
        echo "true"
        return 0
    elif echo "$result" | grep -qE "^[1-9]"; then
        echo "true"
        return 0
    else
        echo "false"
        return 1
    fi
}

# ─── Gas Estimation ─────────────────────────────────────────────────

estimate_gas() {
    local address="$1"
    local function="$2"
    local calldata="$3"
    local rpc_url="${4:-}"

    # Conservative defaults by function name
    case "$function" in
        store_proof_chunk)    echo "0.01" ;;
        init_stark_session)   echo "0.005" ;;
        verify_pow)           echo "0.005" ;;
        verify_fri_step)      echo "0.02" ;;
        verify_merkle_step)   echo "0.02" ;;
        verify_oods)          echo "0.01" ;;
        finalize_session)     echo "0.005" ;;
        verify_model_gkr*)    echo "0.05" ;;
        *)                    echo "0.05" ;;
    esac
}

# ─── Parse TX Hash from sncast Output ──────────────────────────────

parse_tx_hash() {
    local output="$1"
    # sncast outputs "Transaction hash: 0x..." or "transaction_hash: 0x..."
    echo "$output" | grep -oP '0x[a-fA-F0-9]{50,}' | head -1
}
