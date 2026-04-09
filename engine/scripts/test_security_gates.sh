#!/usr/bin/env bash
# test_security_gates.sh — Standalone security gate enforcement tests.
#
# Runs the 10 security gate tests from integration_gate.rs.
# No GPU or model needed — uses synthetic graphs only.
#
# Usage:
#   bash stwo-ml/scripts/test_security_gates.sh
#
# Exit: 0 if all gates pass, 1 otherwise.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Security Gate Enforcement Tests ==="
echo "Running 10 tests (env var gates + bypass checks)..."
echo ""

LOG="/tmp/security_gate_results_$(date +%s).log"

cargo test --features std --test integration_gate -- security_gate \
    --test-threads=1 --nocapture 2>&1 | tee "$LOG"
EXIT=${PIPESTATUS[0]}

echo ""
echo "--- Results ---"
grep -E "^test .* (ok|FAILED)" "$LOG" | sed 's/^/  /' || true

echo ""
if [ "$EXIT" -eq 0 ]; then
    echo "ALL SECURITY GATES PASS"
else
    echo "SECURITY GATE FAILURES — review output above"
fi

rm -f "$LOG"
exit "$EXIT"
