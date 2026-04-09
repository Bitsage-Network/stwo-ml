#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — SDK Publisher
#
# Build and publish all SDK packages to npm / PyPI / crates.io.
# Dry-run by default. Pass --publish to actually push.
#
# Usage:
#   ./scripts/publish_sdks.sh                  # dry-run (safe preview)
#   ./scripts/publish_sdks.sh --publish        # actually publish
#   ./scripts/publish_sdks.sh --npm-only       # only npm packages
#   ./scripts/publish_sdks.sh --pypi-only      # only Python
#   ./scripts/publish_sdks.sh --crates-only    # only Rust
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Cipher Noir palette ─────────────────────────────────────────────
LIME='\033[38;5;118m'
LIME_DIM='\033[38;5;70m'
EMERALD='\033[38;5;48m'
VIOLET='\033[38;5;73m'
ORANGE='\033[38;5;208m'
LILAC='\033[38;5;141m'
WHITE='\033[38;5;255m'
SILVER='\033[38;5;249m'
SLATE='\033[38;5;245m'
GHOST='\033[38;5;240m'
RED='\033[38;5;178m'
BOLD='\033[1m'
DIM='\033[2m'
X='\033[0m'
BG_LIME='\033[48;5;118m'
FG_BLACK='\033[38;5;0m'

H="─"; V="│"; TL="┌"; TR="┐"; BL="└"; BR="┘"
CHECK="✓"; CROSS="✗"; ARROW="▸"; DOT="·"

step_ok()   { echo -e "  ${EMERALD}${CHECK}${X} $*"; }
step_warn() { echo -e "  ${ORANGE}!${X} $*"; }
step_fail() { echo -e "  ${RED}${CROSS}${X} $*"; }
step_info() { echo -e "  ${LIME}${ARROW}${X} ${SILVER}$*${X}"; }

section() {
    echo ""
    echo -e "  ${BG_LIME}${FG_BLACK}${BOLD} $1 ${X} ${WHITE}${BOLD}$2${X} ${GHOST}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${X}"
    echo ""
}

# ── Parse args ──────────────────────────────────────────────────────
DRY_RUN=true
FILTER="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --publish)     DRY_RUN=false; shift;;
        --npm-only)    FILTER="npm"; shift;;
        --pypi-only)   FILTER="pypi"; shift;;
        --crates-only) FILTER="crates"; shift;;
        -h|--help)
            echo "Usage: publish_sdks.sh [--publish] [--npm-only|--pypi-only|--crates-only]"
            exit 0;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

# ── Monorepo root ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Banner ──────────────────────────────────────────────────────────
echo ""
echo -e "  ${LIME}${BOLD}╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═${X}"
echo -e "  ${LIME}║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗${X}"
echo -e "  ${LIME_DIM}╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩${X}"
echo -e "  ${SILVER}SDK Publisher${X}"
echo ""

if $DRY_RUN; then
    echo -e "  ${ORANGE}${BOLD}DRY RUN${X} ${GHOST}— pass --publish to actually push${X}"
else
    echo -e "  ${EMERALD}${BOLD}LIVE PUBLISH${X} ${GHOST}— pushing to registries${X}"
fi
echo ""

# ── Results tracking ────────────────────────────────────────────────
declare -a PKG_NAMES=()
declare -a PKG_VERSIONS=()
declare -a PKG_REGISTRIES=()
declare -a PKG_STATUSES=()

record() {
    PKG_NAMES+=("$1")
    PKG_VERSIONS+=("$2")
    PKG_REGISTRIES+=("$3")
    PKG_STATUSES+=("$4")
}

# ═══════════════════════════════════════════════════════════════════════
# Already Published
# ═══════════════════════════════════════════════════════════════════════

section "0" "ALREADY PUBLISHED"

step_ok "${WHITE}@obelyzk/sdk${X} ${SLATE}1.0.1${X} ${GHOST}npm${X}"
record "@obelyzk/sdk" "1.0.1" "npm" "published"

step_ok "${WHITE}@bitsagecli/cli${X} ${SLATE}0.2.1${X} ${GHOST}npm${X}"
record "@bitsagecli/cli" "0.2.1" "npm" "published"

# ═══════════════════════════════════════════════════════════════════════
# 1  Python SDK
# ═══════════════════════════════════════════════════════════════════════

if [[ "$FILTER" == "all" ]] || [[ "$FILTER" == "pypi" ]]; then
    section "1" "PYTHON SDK  ${LILAC}bitsage-sdk${X}"

    PY_DIR="$MONO_ROOT/sdk/python"

    if [[ ! -d "$PY_DIR" ]]; then
        step_fail "Not found: $PY_DIR"
        record "bitsage-sdk" "0.1.0" "pypi" "not found"
    else
        cd "$PY_DIR"
        PY_VER=$(grep -o 'version = "[^"]*"' pyproject.toml 2>/dev/null | head -1 | grep -o '"[^"]*"' | tr -d '"' || echo "0.1.0")
        step_info "Version: ${WHITE}${PY_VER}${X}"
        step_info "Registry: ${VIOLET}PyPI${X}"

        # Build
        step_info "Building wheel..."
        if python3 -m build --sdist --wheel 2>/dev/null; then
            step_ok "Built"

            if $DRY_RUN; then
                step_info "${ORANGE}DRY RUN${X} — would run: twine upload dist/*"
                record "bitsage-sdk" "$PY_VER" "pypi" "dry-run"
            else
                if twine upload dist/* 2>/dev/null; then
                    step_ok "${EMERALD}Published to PyPI${X}"
                    record "bitsage-sdk" "$PY_VER" "pypi" "published"
                else
                    step_fail "Publish failed"
                    record "bitsage-sdk" "$PY_VER" "pypi" "failed"
                fi
            fi
        else
            step_warn "Build failed — install: ${WHITE}pip install build${X}"
            record "bitsage-sdk" "$PY_VER" "pypi" "build failed"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# 2  MCP Server
# ═══════════════════════════════════════════════════════════════════════

if [[ "$FILTER" == "all" ]] || [[ "$FILTER" == "npm" ]]; then
    section "2" "MCP SERVER  ${LILAC}@obelyzk/mcp-server${X}"

    MCP_DIR="$MONO_ROOT/mcp-server"

    if [[ ! -d "$MCP_DIR" ]]; then
        step_fail "Not found: $MCP_DIR"
        record "@obelyzk/mcp-server" "0.1.0" "npm" "not found"
    else
        cd "$MCP_DIR"
        MCP_VER=$(node -p "require('./package.json').version" 2>/dev/null || echo "0.1.0")
        step_info "Version: ${WHITE}${MCP_VER}${X}"
        step_info "Registry: ${VIOLET}npm${X}"

        # Install deps + build
        step_info "Installing dependencies..."
        npm install --silent 2>/dev/null

        step_info "Building..."
        if npm run build 2>/dev/null || npx tsc 2>/dev/null; then
            step_ok "Built"

            if $DRY_RUN; then
                step_info "${ORANGE}DRY RUN${X} — would run: npm publish --access public"
                record "@obelyzk/mcp-server" "$MCP_VER" "npm" "dry-run"
            else
                if npm publish --access public 2>/dev/null; then
                    step_ok "${EMERALD}Published to npm${X}"
                    record "@obelyzk/mcp-server" "$MCP_VER" "npm" "published"
                else
                    step_fail "Publish failed"
                    record "@obelyzk/mcp-server" "$MCP_VER" "npm" "failed"
                fi
            fi
        else
            step_warn "Build failed"
            record "@obelyzk/mcp-server" "$MCP_VER" "npm" "build failed"
        fi
    fi

    # ═══════════════════════════════════════════════════════════════════
    # 3  Prover SDK
    # ═══════════════════════════════════════════════════════════════════

    section "3" "PROVER SDK  ${LILAC}@obelyzk/prover-sdk${X}"

    PROVER_DIR="$MONO_ROOT/BitSage-Validator/packages/prover-sdk"

    if [[ ! -d "$PROVER_DIR" ]]; then
        step_fail "Not found: $PROVER_DIR"
        record "@obelyzk/prover-sdk" "0.1.0" "npm" "not found"
    else
        cd "$PROVER_DIR"
        PROVER_VER=$(node -p "require('./package.json').version" 2>/dev/null || echo "0.1.0")
        step_info "Version: ${WHITE}${PROVER_VER}${X}"
        step_info "Registry: ${VIOLET}npm${X}"

        step_info "Installing dependencies..."
        npm install --silent 2>/dev/null

        step_info "Building..."
        if npm run build 2>/dev/null || npx tsc 2>/dev/null; then
            step_ok "Built"

            if $DRY_RUN; then
                step_info "${ORANGE}DRY RUN${X} — would run: npm publish --access public"
                record "@obelyzk/prover-sdk" "$PROVER_VER" "npm" "dry-run"
            else
                if npm publish --access public 2>/dev/null; then
                    step_ok "${EMERALD}Published to npm${X}"
                    record "@obelyzk/prover-sdk" "$PROVER_VER" "npm" "published"
                else
                    step_fail "Publish failed"
                    record "@obelyzk/prover-sdk" "$PROVER_VER" "npm" "failed"
                fi
            fi
        else
            step_warn "Build failed"
            record "@obelyzk/prover-sdk" "$PROVER_VER" "npm" "build failed"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# 4  Rust SDK
# ═══════════════════════════════════════════════════════════════════════

if [[ "$FILTER" == "all" ]] || [[ "$FILTER" == "crates" ]]; then
    section "4" "RUST SDK  ${LILAC}bitsage-sdk${X}"

    RUST_DIR="$MONO_ROOT/sdk/rust"

    if [[ ! -d "$RUST_DIR" ]]; then
        step_fail "Not found: $RUST_DIR"
        record "bitsage-sdk (rust)" "0.1.0" "crates.io" "not found"
    else
        cd "$RUST_DIR"
        RUST_VER=$(grep '^version' Cargo.toml 2>/dev/null | head -1 | grep -o '"[^"]*"' | tr -d '"' || echo "0.1.0")
        step_info "Version: ${WHITE}${RUST_VER}${X}"
        step_info "Registry: ${VIOLET}crates.io${X}"

        step_info "Checking build..."
        if cargo check 2>/dev/null; then
            step_ok "Compiles"

            if $DRY_RUN; then
                step_info "${ORANGE}DRY RUN${X} — would run: cargo publish"
                record "bitsage-sdk (rust)" "$RUST_VER" "crates.io" "dry-run"
            else
                if cargo publish 2>/dev/null; then
                    step_ok "${EMERALD}Published to crates.io${X}"
                    record "bitsage-sdk (rust)" "$RUST_VER" "crates.io" "published"
                else
                    step_fail "Publish failed"
                    record "bitsage-sdk (rust)" "$RUST_VER" "crates.io" "failed"
                fi
            fi
        else
            step_warn "Build failed"
            record "bitsage-sdk (rust)" "$RUST_VER" "crates.io" "build failed"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo ""
echo -e "  ${GHOST}${TL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${TR}${X}"
printf "  ${GHOST}${V}${X} %-25s %-8s %-10s %-12s ${GHOST}${V}${X}\n" "PACKAGE" "VERSION" "REGISTRY" "STATUS"
echo -e "  ${GHOST}${V}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${V}${X}"

for i in "${!PKG_NAMES[@]}"; do
    status="${PKG_STATUSES[$i]}"
    case "$status" in
        published)    color="$EMERALD"; icon="$CHECK";;
        dry-run)      color="$ORANGE"; icon="$DOT";;
        *)            color="$RED"; icon="$CROSS";;
    esac
    printf "  ${GHOST}${V}${X} ${color}${icon}${X} %-23s %-8s %-10s ${color}%-12s${X}${GHOST}${V}${X}\n" \
        "${PKG_NAMES[$i]}" "${PKG_VERSIONS[$i]}" "${PKG_REGISTRIES[$i]}" "$status"
done

echo -e "  ${GHOST}${BL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${BR}${X}"
echo ""

if $DRY_RUN; then
    echo -e "  ${ORANGE}This was a dry run.${X} Run with ${WHITE}--publish${X} to push to registries."
else
    PUBLISHED=$(printf '%s\n' "${PKG_STATUSES[@]}" | grep -c "published" || true)
    echo -e "  ${EMERALD}${PUBLISHED} packages published.${X}"
fi
echo ""
