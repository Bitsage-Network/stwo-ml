#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — One-Line Installer
#
# Verifiable ML inference. One command to install, one command to run.
#
# Usage:
#   curl -sSf https://raw.githubusercontent.com/Bitsage-Network/stwo-ml/main/stwo-ml/install.sh | bash
#
# What this does:
#   1. Clones the stwo-ml repository (or updates it)
#   2. Runs scripts/setup.sh to build + download a model
#   3. You get `obelysk` — a TUI to chat with AI and prove every computation
#
# Options (pass after `bash`):
#   curl ... | bash -s -- --model smollm2-135m   # tiny model, fast setup
#   curl ... | bash -s -- --tui-only             # just the TUI binary
#   curl ... | bash -s -- --no-model             # skip model download
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

LIME='\033[38;5;118m'
LIME_DIM='\033[38;5;70m'
EMERALD='\033[38;5;48m'
GHOST='\033[38;5;240m'
WHITE='\033[38;5;255m'
SILVER='\033[38;5;249m'
SLATE='\033[38;5;245m'
ORANGE='\033[38;5;208m'
VIOLET='\033[38;5;73m'
BOLD='\033[1m'
DIM='\033[2m'
X='\033[0m'

H="─"; CHECK="✓"; ARROW="▸"; DOT="·"

clear 2>/dev/null || true
echo ""
echo ""
echo -e "    ${LIME}${BOLD}╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═${X}"
echo -e "    ${LIME}║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗${X}"
echo -e "    ${LIME_DIM}╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩${X}"
echo ""
echo -e "    ${SILVER}Verifiable ML Inference${X}"
echo -e "    ${GHOST}${DIM}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
echo ""

INSTALL_DIR="$HOME/.obelysk/src"
REPO_URL="https://github.com/Bitsage-Network/stwo-ml.git"
BRANCH="main"

# ── Step 1: Get the source ──────────────────────────────────────────

if [[ -d "$INSTALL_DIR" ]] && [[ -f "$INSTALL_DIR/Cargo.toml" ]]; then
    echo -e "  ${EMERALD}${CHECK}${X} Source exists at ${GHOST}${INSTALL_DIR}${X}"
    echo -e "  ${LIME}${ARROW}${X} ${SILVER}Updating...${X}"
    cd "$INSTALL_DIR"
    git fetch origin 2>/dev/null
    git checkout "$BRANCH" 2>/dev/null || true
    git pull origin "$BRANCH" 2>/dev/null || true
    echo -e "  ${EMERALD}${CHECK}${X} Updated to latest"
else
    echo -e "  ${LIME}${ARROW}${X} ${SILVER}Cloning repository...${X}"
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" 2>/dev/null
    echo -e "  ${EMERALD}${CHECK}${X} Cloned to ${GHOST}${INSTALL_DIR}${X}"
fi

echo ""

# ── Step 2: Run setup ───────────────────────────────────────────────

cd "$INSTALL_DIR"

if [[ ! -f "scripts/setup.sh" ]]; then
    echo -e "  ${ORANGE}!${X} scripts/setup.sh not found — repository may be incomplete"
    exit 1
fi

# Forward all arguments to setup.sh
exec bash scripts/setup.sh "$@"
