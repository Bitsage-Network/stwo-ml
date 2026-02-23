#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Starknet Contract Addresses & RPC Config
# ═══════════════════════════════════════════════════════════════════════
#
# Source this after common.sh:
#   source "${SCRIPT_DIR}/lib/common.sh"
#   source "${SCRIPT_DIR}/lib/contract_addresses.sh"
#
# Provides:
#   - get_rpc_url          → RPC endpoint for a given network
#   - get_verifier_address → Contract address for a verifier type + network
#   - get_explorer_url     → Block explorer URL for a tx hash

[[ -n "${_OBELYSK_CONTRACTS_LOADED:-}" ]] && return 0
_OBELYSK_CONTRACTS_LOADED=1

# ─── Contract Addresses ─────────────────────────────────────────────

# StweMlStarkVerifier (recursive STARK verification)
STARK_VERIFIER_SEPOLIA="0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba"

# Obelysk/Elo Cairo Verifier (GKR verification + audit) — v11
ELO_VERIFIER_SEPOLIA="0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005"

# VM31 Privacy Pool
VM31_POOL_SEPOLIA="${VM31_POOL_ADDRESS:-0x07cf94e27a60b94658ec908a00a9bb6dfff03358e952d9d48a8ed0be080ce1f9}"
VM31_POOL_MAINNET=""

# DarkPoolAuction (commit-reveal batch auction with encrypted balances)
DARK_POOL_SEPOLIA="${DARK_POOL_ADDRESS:-0x047765422c66c23d1639a2d93c9c4b91dc41da6273dd4baeab030b4b6ada0d46}"
DARK_POOL_MAINNET=""

# VM31ConfidentialBridge (VM31 withdrawal → ConfidentialTransfer bridge)
VM31_BRIDGE_SEPOLIA="${VM31_BRIDGE_ADDRESS:-0x025a45900864ac136ae56338dc481e2de7bfd9a4ff83ffcceff8439fa1f630a7}"
VM31_BRIDGE_MAINNET=""

# Mainnet (not yet deployed)
STARK_VERIFIER_MAINNET=""
ELO_VERIFIER_MAINNET=""

# ─── Agent Account Factory (Sepolia) ─────────────────────────────────

AGENT_FACTORY_SEPOLIA="0x2f69e566802910359b438ccdb3565dce304a7cc52edbf9fd246d6ad2cd89ce4"
AGENT_CLASS_HASH_SEPOLIA="0x14d44fb938b43e5fbcec27894670cb94898d759e2ef30e7af70058b4da57e7f"
IDENTITY_REGISTRY_SEPOLIA="0x72eb37b0389e570bf8b158ce7f0e1e3489de85ba43ab3876a0594df7231631"

AGENT_FACTORY_MAINNET=""
AGENT_CLASS_HASH_MAINNET=""
IDENTITY_REGISTRY_MAINNET=""

# ─── AVNU Paymaster ──────────────────────────────────────────────────

AVNU_PAYMASTER_SEPOLIA="https://sepolia.paymaster.avnu.fi"
AVNU_PAYMASTER_MAINNET="https://starknet.paymaster.avnu.fi"

# ─── RPC Endpoints ───────────────────────────────────────────────────

# Alchemy v0_8 (required for starkli V3 transactions)
RPC_SEPOLIA="https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}"
RPC_MAINNET="https://starknet-mainnet.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}"

# Public fallbacks (rate-limited, no API key needed)
RPC_SEPOLIA_PUBLIC="https://api.cartridge.gg/x/starknet/sepolia"
RPC_MAINNET_PUBLIC="https://free-rpc.nethermind.io/mainnet-juno/"

# ─── Lookup Functions ────────────────────────────────────────────────

get_rpc_url() {
    local network="${1:-sepolia}"

    # STARKNET_RPC env var overrides everything
    if [[ -n "${STARKNET_RPC:-}" ]]; then
        echo "$STARKNET_RPC"
        return 0
    fi

    case "$network" in
        sepolia)
            if [[ -n "${ALCHEMY_KEY:-}" ]]; then
                echo "$RPC_SEPOLIA"
            else
                echo "$RPC_SEPOLIA_PUBLIC"
            fi
            ;;
        mainnet)
            if [[ -n "${ALCHEMY_KEY:-}" ]]; then
                echo "$RPC_MAINNET"
            else
                echo "$RPC_MAINNET_PUBLIC"
            fi
            ;;
        *)
            err "Unknown network: ${network} (expected: sepolia, mainnet)"
            return 1
            ;;
    esac
}

get_verifier_address() {
    local type="${1:-stark}"
    local network="${2:-sepolia}"
    local addr=""

    case "${type}:${network}" in
        stark:sepolia)   addr="$STARK_VERIFIER_SEPOLIA" ;;
        elo:sepolia)     addr="$ELO_VERIFIER_SEPOLIA" ;;
        stark:mainnet)   addr="$STARK_VERIFIER_MAINNET" ;;
        elo:mainnet)     addr="$ELO_VERIFIER_MAINNET" ;;
        *)
            err "Unknown verifier type/network: ${type}:${network}"
            return 1
            ;;
    esac

    if [[ -z "$addr" ]]; then
        err "No ${type} verifier address configured for ${network}"
        err "  Deploy the contract first, then update lib/contract_addresses.sh"
        return 1
    fi
    echo "$addr"
}

get_explorer_url() {
    local tx_hash="$1"
    local network="${2:-sepolia}"

    case "$network" in
        sepolia)  echo "https://sepolia.starkscan.co/tx/${tx_hash}" ;;
        mainnet)  echo "https://starkscan.co/tx/${tx_hash}" ;;
        *)        echo "https://sepolia.starkscan.co/tx/${tx_hash}" ;;
    esac
}

get_factory_address() {
    local network="${1:-sepolia}"
    local addr=""
    case "$network" in
        sepolia)  addr="$AGENT_FACTORY_SEPOLIA" ;;
        mainnet)  addr="$AGENT_FACTORY_MAINNET" ;;
        *)        err "Unknown network: ${network}"; return 1 ;;
    esac
    if [[ -z "$addr" ]]; then
        err "No agent factory address configured for ${network}"
        return 1
    fi
    echo "$addr"
}

get_paymaster_url() {
    local network="${1:-sepolia}"
    case "$network" in
        sepolia)  echo "$AVNU_PAYMASTER_SEPOLIA" ;;
        mainnet)  echo "$AVNU_PAYMASTER_MAINNET" ;;
        *)        err "Unknown network: ${network}"; return 1 ;;
    esac
}

get_pool_address() {
    local network="${1:-sepolia}"
    local addr=""
    case "$network" in
        sepolia)  addr="$VM31_POOL_SEPOLIA" ;;
        mainnet)  addr="$VM31_POOL_MAINNET" ;;
        *)        err "Unknown network: ${network}"; return 1 ;;
    esac
    if [[ -z "$addr" ]]; then
        err "No VM31 pool address configured for ${network}"
        return 1
    fi
    echo "$addr"
}

get_dark_pool_address() {
    local network="${1:-sepolia}"
    local addr=""
    case "$network" in
        sepolia)  addr="$DARK_POOL_SEPOLIA" ;;
        mainnet)  addr="$DARK_POOL_MAINNET" ;;
        *)        err "Unknown network: ${network}"; return 1 ;;
    esac
    if [[ -z "$addr" ]]; then
        err "No dark pool address configured for ${network}"
        return 1
    fi
    echo "$addr"
}

get_bridge_address() {
    local network="${1:-sepolia}"
    local addr=""
    case "$network" in
        sepolia)  addr="$VM31_BRIDGE_SEPOLIA" ;;
        mainnet)  addr="$VM31_BRIDGE_MAINNET" ;;
        *)        err "Unknown network: ${network}"; return 1 ;;
    esac
    if [[ -z "$addr" ]]; then
        err "No VM31 bridge address configured for ${network}"
        return 1
    fi
    echo "$addr"
}

get_audit_contract() {
    # Audit submissions go to the ELO verifier (same contract, submit_audit entrypoint)
    get_verifier_address "elo" "${1:-sepolia}"
}
