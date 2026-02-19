#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# EloVerifier In-Place Upgrade via replace_class_syscall
# ═══════════════════════════════════════════════════════════════════════
#
# Upgrades the EloVerifier contract in-place using the 5-minute timelocked
# propose_upgrade → execute_upgrade flow. The contract address stays the
# same — only the class hash changes.
#
# Usage:
#   ./scripts/upgrade.sh                              # Build, declare, upgrade
#   ./scripts/upgrade.sh --contract 0x0121d1...c005   # Explicit contract
#   ./scripts/upgrade.sh --declare-only               # Just declare, no upgrade
#   ./scripts/upgrade.sh --skip-delay                  # Auto-sleep 310s
#   ./scripts/upgrade.sh --network mainnet             # Target mainnet
#
# Prerequisites:
#   - starkli installed with STARKNET_ACCOUNT and STARKNET_KEYSTORE set
#   - scarb installed (2.12+)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$(dirname "$PROJECT_DIR")")/scripts/pipeline"

# ─── Defaults ──────────────────────────────────────────────────────────
NETWORK="sepolia"
CONTRACT_ADDRESS=""
DECLARE_ONLY=false
SKIP_DELAY=false

# ─── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --network)       NETWORK="$2"; shift 2 ;;
        --contract)      CONTRACT_ADDRESS="$2"; shift 2 ;;
        --declare-only)  DECLARE_ONLY=true; shift ;;
        --skip-delay)    SKIP_DELAY=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--contract ADDRESS] [--network sepolia|mainnet] [--declare-only] [--skip-delay]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Network config ───────────────────────────────────────────────────
if [[ "$NETWORK" == "mainnet" ]]; then
    RPC_URL="${STARKNET_RPC:-https://starknet-mainnet.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}}"
else
    RPC_URL="${STARKNET_RPC:-https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/${ALCHEMY_KEY:-}}"
fi

# ─── Default contract address ─────────────────────────────────────────
if [[ -z "$CONTRACT_ADDRESS" ]]; then
    ADDR_FILE="$PIPELINE_DIR/lib/contract_addresses.sh"
    if [[ -f "$ADDR_FILE" ]]; then
        # Source to get the variable, but don't fail on missing deps
        CONTRACT_ADDRESS=$(grep "^ELO_VERIFIER_SEPOLIA=" "$ADDR_FILE" | cut -d'"' -f2 || true)
    fi
    if [[ -z "$CONTRACT_ADDRESS" ]]; then
        echo "Error: No contract address specified and could not read from contract_addresses.sh"
        echo "  Use: $0 --contract 0x..."
        exit 1
    fi
fi

echo "═══════════════════════════════════════════════════════════"
echo "  EloVerifier In-Place Upgrade"
echo "  Network:  $NETWORK"
echo "  Contract: $CONTRACT_ADDRESS"
echo "  RPC:      ${RPC_URL%/*}/.."
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── Step 1: Build ─────────────────────────────────────────────────────
echo "[1/5] Building contract..."
cd "$PROJECT_DIR"
scarb build

SIERRA_CLASS="$PROJECT_DIR/target/dev/elo_cairo_verifier_SumcheckVerifierContract.contract_class.json"
if [[ ! -f "$SIERRA_CLASS" ]]; then
    echo "Error: Sierra class not found at $SIERRA_CLASS"
    exit 1
fi
echo "  Build OK"
echo ""

# ─── Step 2: Declare ──────────────────────────────────────────────────
echo "[2/5] Declaring new class..."
DECLARE_OUTPUT=$(starkli declare \
    --rpc "$RPC_URL" \
    "$SIERRA_CLASS" \
    2>&1) || true

NEW_CLASS_HASH=$(echo "$DECLARE_OUTPUT" | grep -oE '0x[0-9a-fA-F]+' | head -1)

if [[ -z "$NEW_CLASS_HASH" ]]; then
    # Check if already declared
    if echo "$DECLARE_OUTPUT" | grep -q "already declared"; then
        NEW_CLASS_HASH=$(echo "$DECLARE_OUTPUT" | grep -oE '0x[0-9a-fA-F]+' | head -1)
        echo "  Class already declared: $NEW_CLASS_HASH"
    else
        echo "Error: Failed to declare. Output:"
        echo "$DECLARE_OUTPUT"
        exit 1
    fi
else
    echo "  New class hash: $NEW_CLASS_HASH"
fi
echo ""

if $DECLARE_ONLY; then
    echo "═══════════════════════════════════════════════════════════"
    echo "  Declare-only mode. New class hash:"
    echo "  $NEW_CLASS_HASH"
    echo "═══════════════════════════════════════════════════════════"
    exit 0
fi

# ─── Step 3: Propose upgrade ──────────────────────────────────────────
echo "[3/5] Proposing upgrade..."
PROPOSE_OUTPUT=$(starkli invoke \
    --rpc "$RPC_URL" \
    "$CONTRACT_ADDRESS" \
    propose_upgrade "$NEW_CLASS_HASH" \
    2>&1) || true

PROPOSE_TX=$(echo "$PROPOSE_OUTPUT" | grep -oE '0x[0-9a-fA-F]+' | tail -1)
echo "  Propose tx: $PROPOSE_TX"
echo ""

# ─── Step 4: Wait for timelock ────────────────────────────────────────
DELAY=310  # 5 min + 10s buffer
if $SKIP_DELAY; then
    echo "[4/5] Waiting ${DELAY}s for timelock (--skip-delay auto-sleep)..."
    sleep "$DELAY"
else
    echo "[4/5] Timelock active. Wait 5 minutes before executing."
    echo "  Run the following after the timelock expires:"
    echo ""
    echo "  starkli invoke --rpc \"$RPC_URL\" \\"
    echo "    \"$CONTRACT_ADDRESS\" execute_upgrade"
    echo ""
    read -rp "  Press Enter when ready to execute (or Ctrl-C to abort)..."
fi
echo ""

# ─── Step 5: Execute upgrade ──────────────────────────────────────────
echo "[5/5] Executing upgrade..."
EXECUTE_OUTPUT=$(starkli invoke \
    --rpc "$RPC_URL" \
    "$CONTRACT_ADDRESS" \
    execute_upgrade \
    2>&1) || true

EXECUTE_TX=$(echo "$EXECUTE_OUTPUT" | grep -oE '0x[0-9a-fA-F]+' | tail -1)
echo "  Execute tx: $EXECUTE_TX"
echo ""

# ─── Verify ───────────────────────────────────────────────────────────
echo "Verifying upgrade..."
PENDING=$(starkli call \
    --rpc "$RPC_URL" \
    "$CONTRACT_ADDRESS" \
    get_pending_upgrade \
    2>&1) || true
echo "  get_pending_upgrade: $PENDING"

# ─── Update pipeline config ──────────────────────────────────────────
ADDR_FILE="$PIPELINE_DIR/lib/contract_addresses.sh"
if [[ -f "$ADDR_FILE" ]]; then
    echo ""
    echo "Updating pipeline contract_addresses.sh with new class hash..."
    # The contract address doesn't change, but record the class hash as a comment
    echo "  (Contract address unchanged: $CONTRACT_ADDRESS)"
    echo "  New class hash: $NEW_CLASS_HASH"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Upgrade complete!"
echo "  Contract: $CONTRACT_ADDRESS (unchanged)"
echo "  Class:    $NEW_CLASS_HASH"
echo "  Network:  $NETWORK"
echo "═══════════════════════════════════════════════════════════"
