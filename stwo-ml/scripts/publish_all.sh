#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — Publish All SDKs
#
# Publishes all 5 SDK packages. Run: ./scripts/publish_all.sh
# Requires: npm login, twine credentials, cargo login
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

LIME='\033[38;5;118m'; EMERALD='\033[38;5;48m'; VIOLET='\033[38;5;73m'
ORANGE='\033[38;5;208m'; WHITE='\033[38;5;255m'; SLATE='\033[38;5;245m'
GHOST='\033[38;5;240m'; RED='\033[38;5;178m'; BOLD='\033[1m'; X='\033[0m'
CHECK="✓"; CROSS="✗"; ARROW="▸"

MONO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo ""
echo -e "  ${LIME}${BOLD}ObelyZK SDK Publisher${X}"
echo -e "  ${GHOST}──────────────────────────────────────${X}"
echo ""

RESULTS=()

publish_npm() {
    local name=$1 dir=$2
    echo -e "  ${LIME}${ARROW}${X} ${WHITE}${name}${X}"
    cd "$dir"

    if npm publish --access public 2>&1 | tail -3; then
        echo -e "  ${EMERALD}${CHECK} ${name} published${X}"
        RESULTS+=("${EMERALD}${CHECK}${X} ${name}")
    else
        echo -e "  ${RED}${CROSS} ${name} failed${X}"
        RESULTS+=("${RED}${CROSS}${X} ${name}")
    fi
    echo ""
}

# ── 1. @obelyzk/sdk (TypeScript) ────────────────────────────────────
echo -e "  ${GHOST}[1/5] npm${X}"
publish_npm "@obelyzk/sdk@1.1.0" "$MONO/sdk/typescript"

# ── 2. @obelyzk/mcp-server ──────────────────────────────────────────
echo -e "  ${GHOST}[2/5] npm${X}"
publish_npm "@obelyzk/mcp-server@0.1.0" "$MONO/mcp-server"

# ── 3. @obelyzk/prover-sdk ──────────────────────────────────────────
echo -e "  ${GHOST}[3/5] npm${X}"
publish_npm "@obelyzk/prover-sdk@0.1.0" "$MONO/BitSage-Validator/packages/prover-sdk"

# ── 4. bitsage-sdk (Python → PyPI) ─────────────────────────────────
echo -e "  ${GHOST}[4/5] pypi${X}"
echo -e "  ${LIME}${ARROW}${X} ${WHITE}bitsage-sdk (Python)${X}"
cd "$MONO/sdk/python"
if twine upload dist/* 2>&1 | tail -3; then
    echo -e "  ${EMERALD}${CHECK} bitsage-sdk published to PyPI${X}"
    RESULTS+=("${EMERALD}${CHECK}${X} bitsage-sdk (pypi)")
else
    echo -e "  ${RED}${CROSS} bitsage-sdk PyPI failed${X}"
    RESULTS+=("${RED}${CROSS}${X} bitsage-sdk (pypi)")
fi
echo ""

# ── 5. bitsage-sdk (Rust → crates.io) ──────────────────────────────
echo -e "  ${GHOST}[5/5] crates.io${X}"
echo -e "  ${LIME}${ARROW}${X} ${WHITE}bitsage-sdk (Rust)${X}"
cd "$MONO/sdk/rust"
if cargo publish --allow-dirty 2>&1 | tail -3; then
    echo -e "  ${EMERALD}${CHECK} bitsage-sdk published to crates.io${X}"
    RESULTS+=("${EMERALD}${CHECK}${X} bitsage-sdk (crates.io)")
else
    echo -e "  ${RED}${CROSS} bitsage-sdk crates.io failed${X}"
    RESULTS+=("${RED}${CROSS}${X} bitsage-sdk (crates.io)")
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo -e "  ${GHOST}──────────────────────────────────────${X}"
echo -e "  ${WHITE}${BOLD}Summary${X}"
echo ""
for r in "${RESULTS[@]}"; do
    echo -e "  $r"
done
echo ""
echo -e "  ${GHOST}Install commands:${X}"
echo -e "    ${LIME}npm install @obelyzk/sdk${X}"
echo -e "    ${LIME}npm install @obelyzk/mcp-server${X}"
echo -e "    ${LIME}npm install @obelyzk/prover-sdk${X}"
echo -e "    ${LIME}pip install bitsage-sdk${X}"
echo -e "    ${LIME}cargo add bitsage-sdk${X}"
echo ""
