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

# Obelysk/Elo Cairo Verifier (direct + GKR verification)
ELO_VERIFIER_SEPOLIA="0x04f8c5377d94baa15291832dc3821c2fc235a95f0823f86add32f828ea965a15"

# Mainnet (not yet deployed)
STARK_VERIFIER_MAINNET=""
ELO_VERIFIER_MAINNET=""

# ─── RPC Endpoints ───────────────────────────────────────────────────

# Alchemy v0_8 (required for starkli V3 transactions)
RPC_SEPOLIA="https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}"
RPC_MAINNET="https://starknet-mainnet.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}"

# Public fallbacks (rate-limited, no API key needed)
RPC_SEPOLIA_PUBLIC="https://free-rpc.nethermind.io/sepolia-juno/"
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

    case "${type}:${network}" in
        stark:sepolia)   echo "$STARK_VERIFIER_SEPOLIA" ;;
        elo:sepolia)     echo "$ELO_VERIFIER_SEPOLIA" ;;
        stark:mainnet)   echo "$STARK_VERIFIER_MAINNET" ;;
        elo:mainnet)     echo "$ELO_VERIFIER_MAINNET" ;;
        *)
            err "Unknown verifier type/network: ${type}:${network}"
            return 1
            ;;
    esac
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
