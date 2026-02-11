#!/usr/bin/env bash
# Deploy elo-cairo-verifier to Starknet Sepolia.
#
# Prerequisites:
#   - starkli installed (https://github.com/xJonathanLEI/starkli)
#   - Account and keystore configured
#   - STARKNET_ACCOUNT and STARKNET_KEYSTORE env vars set
#
# Usage:
#   ./scripts/deploy.sh [OWNER_ADDRESS]
#
# If OWNER_ADDRESS is not provided, uses the deployer's address.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Network config
RPC_URL="${STARKNET_RPC:-https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_7/${ALCHEMY_KEY:-}}"

# Build
echo "Building contract..."
cd "$PROJECT_DIR"
scarb build

# Artifacts
SIERRA_CLASS="$PROJECT_DIR/target/dev/elo_cairo_verifier_SumcheckVerifierContract.contract_class.json"
CASM_CLASS="$PROJECT_DIR/target/dev/elo_cairo_verifier_SumcheckVerifierContract.compiled_contract_class.json"

if [ ! -f "$SIERRA_CLASS" ]; then
    echo "Error: Sierra class not found. Run 'scarb build' first."
    exit 1
fi

# Declare
echo "Declaring class..."
CLASS_HASH=$(starkli declare \
    --rpc "$RPC_URL" \
    "$SIERRA_CLASS" \
    --casm-hash "$(starkli class-hash "$CASM_CLASS" 2>/dev/null || echo "auto")" \
    2>&1 | grep -oE '0x[0-9a-fA-F]+' | head -1)

echo "Class hash: $CLASS_HASH"

# Owner address
OWNER_ADDRESS="${1:-$(starkli signer get-public-key 2>/dev/null | head -1 || echo "")}"
if [ -z "$OWNER_ADDRESS" ]; then
    echo "Error: Provide OWNER_ADDRESS as argument or set STARKNET_ACCOUNT."
    exit 1
fi

# Deploy
echo "Deploying with owner: $OWNER_ADDRESS"
CONTRACT_ADDRESS=$(starkli deploy \
    --rpc "$RPC_URL" \
    "$CLASS_HASH" \
    "$OWNER_ADDRESS" \
    2>&1 | grep -oE '0x[0-9a-fA-F]+' | tail -1)

echo ""
echo "============================================"
echo "  SumcheckVerifier deployed!"
echo "  Address: $CONTRACT_ADDRESS"
echo "  Class:   $CLASS_HASH"
echo "  Owner:   $OWNER_ADDRESS"
echo "  Network: Sepolia"
echo "============================================"
echo ""
echo "Register a model:"
echo "  starkli invoke $CONTRACT_ADDRESS register_model <model_id> <weight_commitment>"
echo ""
echo "Verify a proof:"
echo "  starkli invoke $CONTRACT_ADDRESS verify_matmul <model_id> <proof_calldata...>"
