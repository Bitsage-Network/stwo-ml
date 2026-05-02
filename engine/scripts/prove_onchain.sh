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

# On-chain compatibility flags. The PolicyConfig::standard preset (default)
# now sets all soundness-relevant values — DO NOT add weakening env vars here
# without an entry in engine/docs/SOUNDNESS_GATES_AUDIT.md.
#
# Hardened Apr 30 2026 (two passes):
# - Pass 1 removed STWO_SKIP_RMS_SQ_PROOF=1 and STWO_ALLOW_MISSING_NORM_PROOF=1
#   (audit-verified safe to close).
# - Pass 2 removed STWO_PIECEWISE_ACTIVATION=0 and STWO_ALLOW_LOGUP_ACTIVATION=1
#   (verified safe by reproving SmolLM2 with strict policy locally).
#
# Remaining: STWO_SKIP_BATCH_TOKENS=1 (single-token mode is the common case
# and the audit hasn't cleared multi-token batch closure yet).
export STWO_AGGREGATED_FULL_BINDING=1
export STWO_SKIP_BATCH_TOKENS=1
# Reduce MLE opening queries to fit under 5000-felt calldata limit
export STWO_MLE_N_QUERIES=5

exec "$PROVE_BIN" "$@"
