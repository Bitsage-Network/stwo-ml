#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Obelysk — Generate on-chain compatible proof
#
# This script sets all required env vars for on-chain streaming
# verification on Starknet Sepolia (v35 contract).
#
# Usage: ./scripts/prove_onchain.sh [prove-model args...]
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROVE_BIN="${SCRIPT_DIR}/../target/release/prove-model"

# On-chain compatibility flags:
# 1. Skip RMS² Part 0 proof (channel ops not replayed in Cairo contract)
export STWO_SKIP_RMS_SQ_PROOF=1
# 2. Allow missing norm proof (relaxed soundness gate for Part 0 skip)
export STWO_ALLOW_MISSING_NORM_PROOF=1
# 3. Disable piecewise activation (channel ops not replayed in Cairo contract)
export STWO_PIECEWISE_ACTIVATION=0
# 4. Allow LogUp-only activation proofs (no piecewise requirement)
export STWO_ALLOW_LOGUP_ACTIVATION=1
# 5. Full aggregated binding (required for streaming v25)
export STWO_AGGREGATED_FULL_BINDING=1
# 6. Skip batch token proving (single-token mode for demo)
export STWO_SKIP_BATCH_TOKENS=1
# 7. Reduce MLE opening queries to fit under 5000-felt calldata limit
export STWO_MLE_N_QUERIES=5

exec "$PROVE_BIN" "$@"
