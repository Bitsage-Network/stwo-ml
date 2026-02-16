#!/usr/bin/env bash
# Deploy elo-cairo-verifier contracts to Starknet Sepolia/Mainnet.
#
# Supports two contract types:
#   --contract verifier   (default) SumcheckVerifierContract
#   --contract vm31-pool  VM31PoolContract (requires --relayer and --verifier)
#
# Prerequisites:
#   - starkli installed (https://github.com/xJonathanLEI/starkli)
#   - Account and keystore configured
#   - STARKNET_ACCOUNT and STARKNET_KEYSTORE env vars set
#
# Usage:
#   ./scripts/deploy.sh [OWNER_ADDRESS]
#   ./scripts/deploy.sh --contract vm31-pool --relayer 0x... --verifier 0x... [OWNER_ADDRESS]
#   ./scripts/deploy.sh --network mainnet [OWNER_ADDRESS]
#
# If OWNER_ADDRESS is not provided, uses the deployer's address.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$(dirname "$PROJECT_DIR")")/scripts/pipeline"

# Parse args
NETWORK="sepolia"
CONTRACT_TYPE="verifier"
OWNER_ADDRESS=""
RELAYER_ADDRESS=""
VERIFIER_ADDRESS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --network) NETWORK="$2"; shift 2 ;;
        --contract) CONTRACT_TYPE="$2"; shift 2 ;;
        --relayer) RELAYER_ADDRESS="$2"; shift 2 ;;
        --verifier) VERIFIER_ADDRESS="$2"; shift 2 ;;
        *) OWNER_ADDRESS="$1"; shift ;;
    esac
done

# Validate contract type
if [[ "$CONTRACT_TYPE" != "verifier" && "$CONTRACT_TYPE" != "vm31-pool" ]]; then
    echo "Error: --contract must be 'verifier' or 'vm31-pool'"
    exit 1
fi

# VM31Pool requires relayer and verifier
if [[ "$CONTRACT_TYPE" == "vm31-pool" ]]; then
    if [[ -z "$RELAYER_ADDRESS" ]]; then
        echo "Error: --relayer is required for vm31-pool deployment"
        exit 1
    fi
    if [[ -z "$VERIFIER_ADDRESS" ]]; then
        echo "Error: --verifier is required for vm31-pool deployment"
        exit 1
    fi
fi

# Network config
if [[ "$NETWORK" == "mainnet" ]]; then
    RPC_URL="${STARKNET_RPC:-https://starknet-mainnet.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}}"
else
    RPC_URL="${STARKNET_RPC:-https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}}"
fi

# Contract-specific naming
if [[ "$CONTRACT_TYPE" == "vm31-pool" ]]; then
    CONTRACT_NAME="VM31PoolContract"
    DISPLAY_NAME="VM31 Privacy Pool"
else
    CONTRACT_NAME="SumcheckVerifierContract"
    DISPLAY_NAME="ELO Cairo Verifier"
fi

echo "=== Deploying $DISPLAY_NAME ==="
echo "  Contract: $CONTRACT_NAME"
echo "  Network:  $NETWORK"
echo "  RPC:      ${RPC_URL%/*}/.."
echo ""

# Build
echo "Building contract..."
cd "$PROJECT_DIR"
scarb build

# Artifacts
SIERRA_CLASS="$PROJECT_DIR/target/dev/elo_cairo_verifier_${CONTRACT_NAME}.contract_class.json"

if [ ! -f "$SIERRA_CLASS" ]; then
    echo "Error: Sierra class not found at $SIERRA_CLASS"
    echo "Make sure 'scarb build' succeeded."
    exit 1
fi

# Declare
echo "Declaring class..."
DECLARE_OUTPUT=$(starkli declare \
    --rpc "$RPC_URL" \
    "$SIERRA_CLASS" \
    2>&1) || true

CLASS_HASH=$(echo "$DECLARE_OUTPUT" | grep -oE '0x[0-9a-fA-F]+' | head -1)

if [[ -z "$CLASS_HASH" ]]; then
    echo "Error: Failed to declare class. Output:"
    echo "$DECLARE_OUTPUT"
    exit 1
fi

echo "  Class hash: $CLASS_HASH"

# Owner address
if [[ -z "$OWNER_ADDRESS" ]]; then
    OWNER_ADDRESS=$(starkli signer get-public-key 2>/dev/null | head -1 || echo "")
fi
if [[ -z "$OWNER_ADDRESS" ]]; then
    echo "Error: Provide OWNER_ADDRESS as argument or set STARKNET_ACCOUNT."
    exit 1
fi

# Deploy
if [[ "$CONTRACT_TYPE" == "vm31-pool" ]]; then
    echo "Deploying with owner: $OWNER_ADDRESS"
    echo "  Relayer:  $RELAYER_ADDRESS"
    echo "  Verifier: $VERIFIER_ADDRESS"
    DEPLOY_OUTPUT=$(starkli deploy \
        --rpc "$RPC_URL" \
        "$CLASS_HASH" \
        "$OWNER_ADDRESS" \
        "$RELAYER_ADDRESS" \
        "$VERIFIER_ADDRESS" \
        2>&1) || true
else
    echo "Deploying with owner: $OWNER_ADDRESS"
    DEPLOY_OUTPUT=$(starkli deploy \
        --rpc "$RPC_URL" \
        "$CLASS_HASH" \
        "$OWNER_ADDRESS" \
        2>&1) || true
fi

CONTRACT_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE '0x[0-9a-fA-F]+' | tail -1)

if [[ -z "$CONTRACT_ADDRESS" ]]; then
    echo "Error: Failed to deploy. Output:"
    echo "$DEPLOY_OUTPUT"
    exit 1
fi

echo ""
echo "============================================"
echo "  $DISPLAY_NAME deployed!"
echo "  Address:  $CONTRACT_ADDRESS"
echo "  Class:    $CLASS_HASH"
echo "  Owner:    $OWNER_ADDRESS"
echo "  Network:  $NETWORK"
echo "============================================"
echo ""

# Update pipeline contract addresses if available
ADDR_FILE="$PIPELINE_DIR/lib/contract_addresses.sh"
if [[ -f "$ADDR_FILE" ]]; then
    echo "Updating pipeline contract_addresses.sh..."
    if [[ "$CONTRACT_TYPE" == "vm31-pool" ]]; then
        if [[ "$NETWORK" == "sepolia" ]]; then
            sed -i.bak "s|^VM31_POOL_SEPOLIA=.*|VM31_POOL_SEPOLIA=\"$CONTRACT_ADDRESS\"|" "$ADDR_FILE"
        else
            sed -i.bak "s|^VM31_POOL_MAINNET=.*|VM31_POOL_MAINNET=\"$CONTRACT_ADDRESS\"|" "$ADDR_FILE"
        fi
    else
        if [[ "$NETWORK" == "sepolia" ]]; then
            sed -i.bak "s|^ELO_VERIFIER_SEPOLIA=.*|ELO_VERIFIER_SEPOLIA=\"$CONTRACT_ADDRESS\"|" "$ADDR_FILE"
        else
            sed -i.bak "s|^ELO_VERIFIER_MAINNET=.*|ELO_VERIFIER_MAINNET=\"$CONTRACT_ADDRESS\"|" "$ADDR_FILE"
        fi
    fi
    rm -f "${ADDR_FILE}.bak"
    echo "  Updated: $ADDR_FILE"
fi

echo ""
echo "Entrypoints:"
if [[ "$CONTRACT_TYPE" == "vm31-pool" ]]; then
    echo "  Submit batch proof:"
    echo "    starkli invoke $CONTRACT_ADDRESS submit_batch_proof <deposits> <withdrawals> <spends> <proof_hash> <recipients>"
    echo ""
    echo "  Apply batch chunk:"
    echo "    starkli invoke $CONTRACT_ADDRESS apply_batch_chunk <batch_id> <start> <count>"
    echo ""
    echo "  Finalize batch:"
    echo "    starkli invoke $CONTRACT_ADDRESS finalize_batch <batch_id>"
    echo ""
    echo "  Direct deposit:"
    echo "    starkli invoke $CONTRACT_ADDRESS deposit <commitment> <amount> <asset_id> <proof_hash>"
    echo ""
    echo "  Pause/Unpause:"
    echo "    starkli invoke $CONTRACT_ADDRESS pause"
    echo "    starkli invoke $CONTRACT_ADDRESS unpause"
    echo ""
    echo "  Query state:"
    echo "    starkli call $CONTRACT_ADDRESS get_merkle_root"
    echo "    starkli call $CONTRACT_ADDRESS get_tree_size"
    echo "    starkli call $CONTRACT_ADDRESS is_paused"
else
    echo "  Register model:"
    echo "    starkli invoke $CONTRACT_ADDRESS register_model <model_id> <weight_commitment>"
    echo ""
    echo "  Submit audit:"
    echo "    prove-model audit --submit --contract $CONTRACT_ADDRESS"
    echo ""
    echo "  Verify proof:"
    echo "    starkli invoke $CONTRACT_ADDRESS verify_matmul <model_id> <proof_calldata...>"
    echo ""
    echo "  Query audit:"
    echo "    starkli call $CONTRACT_ADDRESS get_latest_audit <model_id>"
fi
