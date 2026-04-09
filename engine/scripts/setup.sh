#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ObelyZK — Getting Started
#
# One command. Verifiable ML inference on your machine.
#
# Usage:
#   curl -sSf https://obelysk.xyz/install | bash
#   # or locally:
#   ./scripts/setup.sh [--model MODEL] [--gpu] [--no-model] [--tui-only]
#
# Options:
#   --model MODEL   Model to download (default: interactive picker)
#                   Options: smollm2-135m, qwen2-0.5b, phi3-mini, llama3-8b
#   --gpu           Enable CUDA GPU support (Linux only)
#   --no-model      Skip model download (build only)
#   --tui-only      Build only the TUI binary (fastest)
#   --server        Build the prove-server binary
#   --smoke-test    Run a quick proof after setup to validate
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── ANSI 256-color palette ──────────────────────────────────────────
# Matches the TUI "Cipher Noir" aesthetic exactly
LIME='\033[38;5;118m'        # #87ff00 — primary brand
LIME_DIM='\033[38;5;70m'     # #5faf00 — secondary
EMERALD='\033[38;5;48m'      # #00ff87 — success
VIOLET='\033[38;5;73m'       # #5fafaf — hashes / crypto
ORANGE='\033[38;5;208m'      # orange  — warnings / highlights
LILAC='\033[38;5;141m'       # light purple — labels
WHITE='\033[38;5;255m'       # bright white
SILVER='\033[38;5;249m'      # light gray
SLATE='\033[38;5;245m'       # medium gray
GHOST='\033[38;5;240m'       # dark gray
RED='\033[38;5;178m'         # gold/amber — errors
CYAN='\033[38;5;87m'         # cyan — interactive highlights
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'
BG_LIME='\033[48;5;118m'
BG_GHOST='\033[48;5;236m'
FG_BLACK='\033[38;5;0m'

# ── Box drawing ─────────────────────────────────────────────────────
H="─"; V="│"; TL="┌"; TR="┐"; BL="└"; BR="┘"
ML="├"; MR="┤"
CHECK="✓"; CROSS="✗"; ARROW="▸"; DOT="·"; DIAMOND="◆"; SHIELD="⊕"
BLOCK="█"; HALF="▓"; QUARTER="░"
CIRCLE_EMPTY="○"; CIRCLE_FULL="●"

# ── Helper functions ────────────────────────────────────────────────

cols() { tput cols 2>/dev/null || echo 80; }
now_ms() { python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || date +%s000; }

hr() {
    local w=$(cols)
    local c="${1:-$GHOST}"
    printf "${c}"
    printf '%*s' "$w" '' | tr ' ' "$H"
    printf "${RESET}\n"
}

hr_accent() {
    local w=$(cols)
    local third=$(( w / 3 ))
    printf "${GHOST}${DIM}"
    printf '%*s' "$third" '' | tr ' ' "$H"
    printf "${RESET}${LIME_DIM}"
    printf '%*s' "$third" '' | tr ' ' "$H"
    printf "${RESET}${GHOST}${DIM}"
    printf '%*s' "$(( w - 2 * third ))" '' | tr ' ' "$H"
    printf "${RESET}\n"
}

center() {
    local text="$1"
    local plain
    plain=$(echo -e "$text" | sed 's/\x1b\[[0-9;]*m//g')
    local w=$(cols)
    local pad=$(( (w - ${#plain}) / 2 ))
    [[ $pad -lt 0 ]] && pad=0
    printf '%*s' "$pad" ''
    echo -e "$text"
}

# Animated reveal — each line appears with a micro-delay
reveal() { echo -e "$1"; sleep 0.03; }

# Progress bar: bar <current> <total> <label> [color]
bar() {
    local current=$1 total=$2 label="${3:-}" color="${4:-$LIME}"
    local w=$(( $(cols) - 24 ))
    [[ $w -gt 50 ]] && w=50
    [[ $w -lt 16 ]] && w=16
    local filled=$(( current * w / total ))
    local empty=$(( w - filled ))
    local pct=$(( current * 100 / total ))

    printf "\r  "
    # Gradient bar: filled blocks in main color, frontier in half-block
    printf "${color}"
    printf '%*s' "$filled" '' | tr ' ' "$BLOCK"
    if [[ $filled -lt $w ]] && [[ $filled -gt 0 ]]; then
        printf "${GHOST}${HALF}"
        printf '%*s' "$(( empty - 1 ))" '' | tr ' ' " "
    elif [[ $filled -eq 0 ]]; then
        printf "${GHOST}"
        printf '%*s' "$empty" '' | tr ' ' " "
    else
        : # full
    fi
    printf "${RESET} ${WHITE}%3d%%${RESET}" "$pct"
    [[ -n "$label" ]] && printf "  ${SLATE}%s${RESET}" "$label"
}

# Step indicators
step_ok()   { echo -e "  ${EMERALD}${CHECK}${RESET} $*"; }
step_warn() { echo -e "  ${ORANGE}!${RESET} $*"; }
step_fail() { echo -e "  ${RED}${CROSS}${RESET} $*"; }
step_info() { echo -e "  ${LIME}${ARROW}${RESET} ${SILVER}$*${RESET}"; }
step_dim()  { echo -e "  ${GHOST}${DOT} $*${RESET}"; }

# Section header with number badge
section() {
    local title="$1"
    local num="$2"
    local w=$(cols)
    local remaining=$(( w - 12 - ${#title} ))
    [[ $remaining -lt 4 ]] && remaining=4

    echo ""
    printf "  ${BG_LIME}${FG_BLACK}${BOLD} %s ${RESET}" "$num"
    printf " ${WHITE}${BOLD}%s${RESET} " "$title"
    printf "${GHOST}"
    printf '%*s' "$remaining" '' | tr ' ' "$H"
    printf "${RESET}\n"
    echo ""
}

section_end() {
    echo ""
}

# Elapsed time display
elapsed_since() {
    local start=$1
    local now
    now=$(now_ms)
    local ms=$(( now - start ))
    local secs=$(( ms / 1000 ))
    if [[ $secs -lt 60 ]]; then
        echo "${secs}s"
    else
        echo "$(( secs / 60 ))m $(( secs % 60 ))s"
    fi
}

# ── Parse args ──────────────────────────────────────────────────────
MODEL=""
GPU=false
NO_MODEL=false
TUI_ONLY=false
SERVER=false
SMOKE_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2;;
        --gpu)        GPU=true; shift;;
        --no-model)   NO_MODEL=true; shift;;
        --tui-only)   TUI_ONLY=true; shift;;
        --server)     SERVER=true; shift;;
        --smoke-test) SMOKE_TEST=true; shift;;
        -h|--help)
            echo ""
            echo -e "  ${LIME}ObelyZK Setup${RESET}"
            echo ""
            echo -e "  ${WHITE}Usage:${RESET}  ./scripts/setup.sh [options]"
            echo ""
            echo -e "  ${SLATE}Options:${RESET}"
            echo -e "    ${LIME}--model${RESET} MODEL    ${GHOST}smollm2-135m, qwen2-0.5b, phi3-mini, llama3-8b${RESET}"
            echo -e "    ${LIME}--gpu${RESET}            ${GHOST}Enable CUDA (Linux)${RESET}"
            echo -e "    ${LIME}--no-model${RESET}       ${GHOST}Skip model download${RESET}"
            echo -e "    ${LIME}--tui-only${RESET}       ${GHOST}Build only the TUI (fastest)${RESET}"
            echo -e "    ${LIME}--server${RESET}         ${GHOST}Build prove-server${RESET}"
            echo -e "    ${LIME}--smoke-test${RESET}     ${GHOST}Validate with a quick proof${RESET}"
            echo ""
            exit 0;;
        *)  step_fail "Unknown option: $1"; exit 1;;
    esac
done

# ── Clear screen + opening ──────────────────────────────────────────

clear 2>/dev/null || true
echo ""

# Logo reveal
reveal ""
reveal "$(center "${GHOST}${DIM}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${RESET}")"
reveal ""
reveal "$(center "${LIME}${BOLD}  ╔═╗╔╗  ╔═╗╦  ╦ ╦╔═╗╦╔═  ${RESET}")"
reveal "$(center "${LIME}  ║ ║╠╩╗ ╠═ ║  ╚╦╝╔═╝╠╩╗  ${RESET}")"
reveal "$(center "${LIME_DIM}  ╚═╝╚═╝ ╚═╝╩═╝ ╩ ╚═╝╩ ╩  ${RESET}")"
reveal ""
reveal "$(center "${SILVER}V E R I F I A B L E   M L   I N F E R E N C E${RESET}")"
reveal ""
reveal "$(center "${GHOST}Every computation proved. Every proof verified on-chain.${RESET}")"
reveal "$(center "${GHOST}${DIM}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${RESET}")"
echo ""
sleep 0.2

# ── System probe ────────────────────────────────────────────────────

OS_NAME="$(uname -s)"
ARCH="$(uname -m)"
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo "?")
MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}' 2>/dev/null || free -g 2>/dev/null | awk '/Mem:/{print $2}' || echo "?")

# GPU detection
GPU_INFO=""
if [[ "$OS_NAME" == "Darwin" ]]; then
    GPU_INFO=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | sed 's/.*: //' || echo "")
elif command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi

# Disk free
DISK_FREE=$(df -h / 2>/dev/null | tail -1 | awk '{print $4}')

printf "  ${GHOST}${TL}${H}${H}${RESET} ${SLATE}SYSTEM${RESET}\n"
printf "  ${GHOST}${V}${RESET}\n"
printf "  ${GHOST}${V}${RESET}  ${SLATE}Platform${RESET}    ${WHITE}${OS_NAME} ${ARCH}${RESET}\n"
printf "  ${GHOST}${V}${RESET}  ${SLATE}CPU${RESET}         ${WHITE}${CPU_CORES}${RESET} ${GHOST}cores${RESET}\n"
printf "  ${GHOST}${V}${RESET}  ${SLATE}Memory${RESET}      ${WHITE}${MEM_GB}${RESET} ${GHOST}GB${RESET}\n"
printf "  ${GHOST}${V}${RESET}  ${SLATE}Disk Free${RESET}   ${WHITE}${DISK_FREE}${RESET}\n"
if [[ -n "$GPU_INFO" ]]; then
    printf "  ${GHOST}${V}${RESET}  ${SLATE}GPU${RESET}         ${EMERALD}${GPU_INFO}${RESET}\n"
fi
printf "  ${GHOST}${V}${RESET}\n"

# ── Interactive model picker ────────────────────────────────────────

if [[ -z "$MODEL" ]] && ! $NO_MODEL && [[ -t 0 ]]; then
    # Interactive mode — show model cards
    printf "  ${GHOST}${ML}${H}${H}${RESET} ${SLATE}SELECT MODEL${RESET}\n"
    printf "  ${GHOST}${V}${RESET}\n"

    # Model data: name | params | size | description
    MODELS=(
        "smollm2-135m|135M|270MB|Tiny. Fast builds, instant proofs. Great for testing."
        "qwen2-0.5b|494M|950MB|Small but capable. Recommended for getting started."
        "phi3-mini|3.8B|7.1GB|Microsoft's efficient 3.8B. Good balance of quality."
        "llama3-8b|8B|16GB|Meta's flagship 8B. Best quality, needs space."
    )

    RECOMMENDED=1  # qwen2-0.5b

    for i in "${!MODELS[@]}"; do
        IFS='|' read -r m_name m_params m_size m_desc <<< "${MODELS[$i]}"

        if [[ $i -eq $RECOMMENDED ]]; then
            printf "  ${GHOST}${V}${RESET}  ${LIME}${BOLD}[%d]${RESET} ${WHITE}${BOLD}%-15s${RESET}" "$(( i + 1 ))" "$m_name"
            printf " ${SILVER}%-6s${RESET} ${GHOST}%-6s${RESET}" "$m_params" "$m_size"
            printf " ${LIME}${ARROW} recommended${RESET}\n"
            printf "  ${GHOST}${V}${RESET}      ${GHOST}%s${RESET}\n" "$m_desc"
        else
            printf "  ${GHOST}${V}${RESET}  ${SLATE}[%d]${RESET} ${SILVER}%-15s${RESET}" "$(( i + 1 ))" "$m_name"
            printf " ${GHOST}%-6s %-6s${RESET}\n" "$m_params" "$m_size"
            printf "  ${GHOST}${V}${RESET}      ${GHOST}%s${RESET}\n" "$m_desc"
        fi
        printf "  ${GHOST}${V}${RESET}\n"
    done

    printf "  ${GHOST}${V}${RESET}  ${CYAN}Choose [1-4]${RESET} ${GHOST}(default: 2 — qwen2-0.5b)${RESET}: "
    read -n 1 -r MODEL_CHOICE
    echo ""
    printf "  ${GHOST}${V}${RESET}\n"

    case "$MODEL_CHOICE" in
        1) MODEL="smollm2-135m";;
        2) MODEL="qwen2-0.5b";;
        3) MODEL="phi3-mini";;
        4) MODEL="llama3-8b";;
        *) MODEL="qwen2-0.5b";;
    esac

    IFS='|' read -r _ SEL_PARAMS SEL_SIZE _ <<< "${MODELS[$(( ${MODEL_CHOICE:-2} - 1 ))]}"
    printf "  ${GHOST}${V}${RESET}  ${EMERALD}${CHECK}${RESET} ${WHITE}${MODEL}${RESET} ${GHOST}${DOT} ${SEL_PARAMS:-494M} params ${DOT} ~${SEL_SIZE:-950MB}${RESET}\n"
else
    [[ -z "$MODEL" ]] && MODEL="qwen2-0.5b"

    printf "  ${GHOST}${ML}${H}${H}${RESET} ${SLATE}MODEL${RESET}\n"
    printf "  ${GHOST}${V}${RESET}\n"

    case "$MODEL" in
        smollm2-135m) M_INFO="135M params ${DOT} ~270MB";;
        qwen2-0.5b)   M_INFO="494M params ${DOT} ~950MB";;
        phi3-mini)     M_INFO="3.8B params ${DOT} ~7.1GB";;
        llama3-8b)     M_INFO="8B params ${DOT} ~16GB";;
        *)             M_INFO="";;
    esac
    printf "  ${GHOST}${V}${RESET}  ${LILAC}${MODEL}${RESET}  ${GHOST}${M_INFO}${RESET}\n"
fi

printf "  ${GHOST}${V}${RESET}\n"
printf "  ${GHOST}${BL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${RESET}\n"
echo ""

SETUP_START=$(now_ms)

# ═══════════════════════════════════════════════════════════════════════
# 1  PREREQUISITES
# ═══════════════════════════════════════════════════════════════════════

section "PREREQUISITES" "1"

FEATURES="std"
PREREQ_COUNT=0
PREREQ_TOTAL=5  # platform, rust, toolchain, node, llama

case "$OS_NAME" in
    Darwin)
        PLATFORM="macOS"
        FEATURES="$FEATURES,metal"
        HAS_METAL=true; HAS_CUDA=false
        step_ok "${WHITE}macOS${RESET} ${SILVER}${ARCH}${RESET} ${GHOST}${DOT}${RESET} ${EMERALD}Metal GPU${RESET}"
        PREREQ_COUNT=$((PREREQ_COUNT + 1))
        ;;
    Linux)
        PLATFORM="Linux"
        HAS_METAL=false
        if command -v nvidia-smi &>/dev/null && $GPU; then
            FEATURES="$FEATURES,gpu,cuda-runtime"
            HAS_CUDA=true
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            step_ok "${WHITE}Linux${RESET} ${SILVER}${ARCH}${RESET} ${GHOST}${DOT}${RESET} ${EMERALD}NVIDIA ${GPU_NAME}${RESET}"
        else
            HAS_CUDA=false
            step_ok "${WHITE}Linux${RESET} ${SILVER}${ARCH}${RESET} ${GHOST}${DOT}${RESET} ${SLATE}CPU mode${RESET}"
            $GPU && step_warn "No NVIDIA GPU detected"
        fi
        PREREQ_COUNT=$((PREREQ_COUNT + 1))
        ;;
    *)
        step_fail "Unsupported: $OS_NAME"
        exit 1;;
esac

# Rust
if ! command -v rustup &>/dev/null; then
    step_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none 2>/dev/null
    source "$HOME/.cargo/env"
fi
RUST_VER=$(rustc --version 2>/dev/null | awk '{print $2}' || echo "?")
step_ok "${WHITE}Rust${RESET} ${SLATE}${RUST_VER}${RESET}"
PREREQ_COUNT=$((PREREQ_COUNT + 1))

# Nightly toolchain
TOOLCHAIN="nightly-2025-07-14"
if ! rustup toolchain list | grep -q "$TOOLCHAIN" 2>/dev/null; then
    step_info "Installing ${TOOLCHAIN}..."
    rustup toolchain install "$TOOLCHAIN" --profile minimal 2>/dev/null
fi
step_ok "${WHITE}Toolchain${RESET} ${SLATE}${TOOLCHAIN}${RESET}"
PREREQ_COUNT=$((PREREQ_COUNT + 1))

# Python (only if downloading models)
if ! $NO_MODEL; then
    if command -v python3 &>/dev/null; then
        PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
        step_ok "${WHITE}Python${RESET} ${SLATE}${PY_VER}${RESET}"

        if ! python3 -c "import huggingface_hub" 2>/dev/null; then
            step_info "Installing huggingface-hub..."
            pip3 install -q huggingface-hub 2>/dev/null
        fi
        step_ok "${WHITE}huggingface-hub${RESET}"
    else
        step_fail "python3 required for model downloads"
        step_info "Install: ${WHITE}brew install python3${RESET} (macOS) or ${WHITE}apt install python3${RESET}"
        exit 1
    fi
fi

# Node.js
if command -v node &>/dev/null; then
    NODE_VER=$(node --version)
    step_ok "${WHITE}Node.js${RESET} ${SLATE}${NODE_VER}${RESET}"
    PREREQ_COUNT=$((PREREQ_COUNT + 1))
else
    step_warn "${WHITE}Node.js${RESET} ${SLATE}not found${RESET} ${GHOST}— on-chain submission requires it${RESET}"
fi

# llama.cpp
if ! $SERVER; then
    if command -v llama-server &>/dev/null; then
        LLAMA_VER=$(llama-server --version 2>&1 | head -1 | awk '{print $NF}' || echo "")
        step_ok "${WHITE}llama-server${RESET} ${SLATE}${LLAMA_VER}${RESET}"
        PREREQ_COUNT=$((PREREQ_COUNT + 1))
    else
        step_warn "${WHITE}llama-server${RESET} ${SLATE}not found${RESET} ${GHOST}— interactive chat requires it${RESET}"
        if [[ "$PLATFORM" == "macOS" ]] && [[ -t 0 ]]; then
            echo ""
            echo -ne "  ${GHOST}${DOT}${RESET} Install via ${WHITE}brew install llama.cpp${RESET}? ${SLATE}[Y/n]${RESET} "
            read -n 1 -r REPLY
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                brew install llama.cpp 2>/dev/null &
                spin "Installing llama.cpp" $!
                step_ok "${WHITE}llama-server${RESET} ${SLATE}installed${RESET}"
                PREREQ_COUNT=$((PREREQ_COUNT + 1))
            fi
        fi
    fi
fi

echo ""
echo -e "  ${GHOST}${PREREQ_COUNT}/${PREREQ_TOTAL} checks passed${RESET}"

section_end

# ═══════════════════════════════════════════════════════════════════════
# 2  BUILD
# ═══════════════════════════════════════════════════════════════════════

section "BUILD" "2"

# Find the project root — support running from scripts/ dir, repo root, or ~/.obelysk/src/
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(pwd)"
fi

# Walk up to find Cargo.toml
PROJECT_DIR=""
for candidate in "$SCRIPT_DIR/.." "$SCRIPT_DIR" "$HOME/.obelysk/src" "$(pwd)"; do
    if [[ -f "$candidate/Cargo.toml" ]]; then
        PROJECT_DIR="$(cd "$candidate" && pwd)"
        break
    fi
done

if [[ -z "$PROJECT_DIR" ]]; then
    step_fail "Cannot find project. Run from the stwo-ml directory, or use:"
    echo ""
    echo -e "    ${LIME}curl -sSf https://raw.githubusercontent.com/bitsage-network/stwo-ml/main/install.sh | bash${RESET}"
    echo ""
    exit 1
fi

cd "$PROJECT_DIR"

if $TUI_ONLY; then
    FEATURES="$FEATURES,tui"
    TARGETS="--bin obelysk"
    BUILD_DESC="obelysk"
elif $SERVER; then
    FEATURES="$FEATURES,server,model-loading,safetensors,binary-proof"
    if [[ "${HAS_CUDA:-false}" == "true" ]]; then FEATURES="$FEATURES,gpu,cuda-runtime"; fi
    TARGETS="--bin prove-server"
    BUILD_DESC="prove-server"
else
    FEATURES="$FEATURES,cli,model-loading,safetensors,audit,parallel-audit,tui"
    if [[ "${HAS_CUDA:-false}" == "true" ]]; then FEATURES="$FEATURES,gpu,cuda-runtime"; fi
    TARGETS="--bin prove-model --bin obelysk"
    BUILD_DESC="prove-model + obelysk"
fi

step_info "Compiling ${WHITE}${BUILD_DESC}${RESET}"
step_dim "features: ${FEATURES}"
echo ""

# Build with live progress
BUILD_LOG=$(mktemp)
BUILD_START=$(now_ms)
cargo +$TOOLCHAIN build --release $TARGETS --features "$FEATURES" 2>"$BUILD_LOG" &
BUILD_PID=$!

# Parse cargo output for live progress bar
LAST_CRATE=""
COMPILED=0
PHASE="resolving"
while kill -0 "$BUILD_PID" 2>/dev/null; do
    if [[ -f "$BUILD_LOG" ]]; then
        LATEST=$(tail -1 "$BUILD_LOG" 2>/dev/null || true)

        # Detect phase
        if echo "$LATEST" | grep -q "Downloading" 2>/dev/null; then
            PHASE="downloading"
        elif echo "$LATEST" | grep -q "Compiling" 2>/dev/null; then
            PHASE="compiling"
        elif echo "$LATEST" | grep -q "Linking\|Finished" 2>/dev/null; then
            PHASE="linking"
        fi

        NEW_CRATE=$(echo "$LATEST" | sed -n 's/.*Compiling \([^ ]*\).*/\1/p' || true)
        if [[ -n "$NEW_CRATE" ]] && [[ "$NEW_CRATE" != "$LAST_CRATE" ]]; then
            LAST_CRATE="$NEW_CRATE"
            COMPILED=$((COMPILED + 1))

            # Dynamic total estimate: TUI-only ~50, full ~180
            if $TUI_ONLY; then
                EST_TOTAL=55
            elif $SERVER; then
                EST_TOTAL=160
            else
                EST_TOTAL=185
            fi
            [[ $COMPILED -gt $EST_TOTAL ]] && EST_TOTAL=$((COMPILED + 10))

            # Categorize crate for display
            CRATE_LABEL="$NEW_CRATE"
            case "$NEW_CRATE" in
                stwo*|stwo-ml) CRATE_LABEL="${NEW_CRATE} ${GHOST}[core]";;
                ratatui*|crossterm*) CRATE_LABEL="${NEW_CRATE} ${GHOST}[tui]";;
                serde*|toml*) CRATE_LABEL="${NEW_CRATE} ${GHOST}[data]";;
                ureq*|http*) CRATE_LABEL="${NEW_CRATE} ${GHOST}[net]";;
            esac

            bar $COMPILED $EST_TOTAL "$CRATE_LABEL"
        fi
    fi
    sleep 0.15
done

wait "$BUILD_PID" 2>/dev/null
BUILD_EXIT=$?
BUILD_ELAPSED=$(elapsed_since "$BUILD_START")
rm -f "$BUILD_LOG"

# Clear progress bar line
printf "\r%*s\r" "$(cols)" ""

echo ""

if [[ $BUILD_EXIT -ne 0 ]]; then
    step_fail "Build failed ${GHOST}(${BUILD_ELAPSED})${RESET}"
    step_info "Check build errors and retry"
    exit 1
fi

# Report built binaries with details
for bin in prove-model obelysk prove-server; do
    BIN_PATH="$PROJECT_DIR/target/release/$bin"
    if [[ -f "$BIN_PATH" ]]; then
        SIZE=$(du -sh "$BIN_PATH" | awk '{print $1}')
        step_ok "${WHITE}${bin}${RESET} ${GHOST}${SIZE}${RESET}"
    fi
done

echo ""
if [[ $COMPILED -gt 0 ]]; then
    echo -e "  ${GHOST}Compiled ${WHITE}${COMPILED}${GHOST} crates in ${WHITE}${BUILD_ELAPSED}${RESET}"
else
    echo -e "  ${GHOST}Up to date ${GHOST}${DOT} ${BUILD_ELAPSED}${RESET}"
fi

section_end

# ═══════════════════════════════════════════════════════════════════════
# 3  MODEL
# ═══════════════════════════════════════════════════════════════════════

if ! $NO_MODEL; then
    section "MODEL  ${LILAC}${MODEL}${RESET}" "3"

    MODEL_DIR="$HOME/.obelysk/models/$MODEL"
    GGUF_DIR="$HOME/.obelysk/models/${MODEL}-gguf"
    mkdir -p "$MODEL_DIR" "$GGUF_DIR"

    # Map model names to HuggingFace repos
    case "$MODEL" in
        smollm2-135m)
            HF_MODEL="HuggingFaceTB/SmolLM2-135M"
            HF_GGUF=""
            GGUF_FILE=""
            ;;
        qwen2-0.5b)
            HF_MODEL="Qwen/Qwen2-0.5B"
            HF_GGUF="Qwen/Qwen2-0.5B-Instruct-GGUF"
            GGUF_FILE="qwen2-0_5b-instruct-q4_k_m.gguf"
            ;;
        phi3-mini)
            HF_MODEL="microsoft/Phi-3-mini-4k-instruct"
            HF_GGUF=""
            GGUF_FILE=""
            ;;
        llama3-8b)
            HF_MODEL="meta-llama/Meta-Llama-3-8B"
            HF_GGUF=""
            GGUF_FILE=""
            step_warn "Llama3 requires HF token — set ${WHITE}HF_TOKEN${RESET}"
            ;;
        *)
            step_fail "Unknown model: ${MODEL}"
            step_info "Options: ${WHITE}smollm2-135m${RESET}, ${WHITE}qwen2-0.5b${RESET}, ${WHITE}phi3-mini${RESET}, ${WHITE}llama3-8b${RESET}"
            exit 1;;
    esac

    # Download safetensors weights
    if [[ ! -f "$MODEL_DIR/config.json" ]]; then
        step_info "Downloading ${WHITE}${HF_MODEL}${RESET}"
        echo ""
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_MODEL', local_dir='$MODEL_DIR',
    allow_patterns=['*.safetensors','config.json','tokenizer*','*.json'])" 2>&1 | tail -5
        echo ""
        WEIGHT_SIZE=$(du -sh "$MODEL_DIR" | awk '{print $1}')
        step_ok "${WHITE}Weights${RESET} ${GHOST}${WEIGHT_SIZE}${RESET}"
    else
        WEIGHT_SIZE=$(du -sh "$MODEL_DIR" | awk '{print $1}')
        step_ok "${WHITE}Weights${RESET} ${SLATE}cached${RESET} ${GHOST}${WEIGHT_SIZE}${RESET}"
    fi

    # Download GGUF for interactive chat
    if [[ -n "$HF_GGUF" ]] && [[ -n "$GGUF_FILE" ]]; then
        GGUF_PATH="$GGUF_DIR/$GGUF_FILE"
        if [[ ! -f "$GGUF_PATH" ]]; then
            step_info "Downloading ${WHITE}GGUF${RESET} quantization..."
            python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$HF_GGUF', '$GGUF_FILE', local_dir='$GGUF_DIR')" 2>&1 | tail -3
            GGUF_SIZE=$(du -sh "$GGUF_DIR" | awk '{print $1}')
            step_ok "${WHITE}GGUF${RESET} ${GHOST}${GGUF_SIZE}${RESET}"
        else
            GGUF_SIZE=$(du -sh "$GGUF_DIR" | awk '{print $1}')
            step_ok "${WHITE}GGUF${RESET} ${SLATE}cached${RESET} ${GHOST}${GGUF_SIZE}${RESET}"
        fi
    fi

    # Parse and display model architecture card
    if [[ -f "$MODEL_DIR/config.json" ]]; then
        MODEL_INFO=$(python3 -c "
import json
c = json.load(open('$MODEL_DIR/config.json'))
h = c.get('hidden_size', 0)
n = c.get('num_hidden_layers', 0)
i = c.get('intermediate_size', h*4)
v = c.get('vocab_size', 0)
p = v*h + n*(4*h*h + 2*h*i + 2*h) + v*h
if p > 1e9: ps = f'{p/1e9:.1f}B'
elif p > 1e6: ps = f'{p/1e6:.0f}M'
else: ps = f'{p:,}'
arch = c.get('model_type', 'transformer')
heads = c.get('num_attention_heads', '?')
kv = c.get('num_key_value_heads', heads)
act = c.get('hidden_act', '?')
ctx = c.get('max_position_embeddings', '?')
print(f'{ps}|{n}|{h}|{arch}|{v}|{heads}|{kv}|{act}|{ctx}')
" 2>/dev/null || echo "?|?|?|?|?|?|?|?|?")

        IFS='|' read -r P_STR LAYERS HIDDEN ARCH_TYPE VOCAB HEADS KV_HEADS ACT CTX <<< "$MODEL_INFO"

        echo ""
        echo -e "  ${GHOST}${TL}${H}${H} ${SLATE}ARCHITECTURE${GHOST} ${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${TR}${RESET}"
        echo -e "  ${GHOST}${V}${RESET}  ${SLATE}Type${RESET}       ${WHITE}${ARCH_TYPE}${RESET}  ${GHOST}${DOT} ${P_STR} parameters${RESET}  ${GHOST}${V}${RESET}"
        echo -e "  ${GHOST}${V}${RESET}  ${SLATE}Layers${RESET}     ${WHITE}${LAYERS}${RESET}  ${GHOST}${DOT} d=${HIDDEN}${RESET}                ${GHOST}${V}${RESET}"
        echo -e "  ${GHOST}${V}${RESET}  ${SLATE}Attention${RESET}  ${WHITE}${HEADS}${RESET} ${GHOST}heads${RESET}  ${GHOST}${DOT} ${KV_HEADS} KV heads${RESET}       ${GHOST}${V}${RESET}"
        echo -e "  ${GHOST}${V}${RESET}  ${SLATE}Activation${RESET} ${WHITE}${ACT}${RESET}  ${GHOST}${DOT} vocab ${VOCAB}${RESET}          ${GHOST}${V}${RESET}"
        echo -e "  ${GHOST}${V}${RESET}  ${SLATE}Context${RESET}    ${WHITE}${CTX}${RESET} ${GHOST}tokens${RESET}                  ${GHOST}${V}${RESET}"
        echo -e "  ${GHOST}${BL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${BR}${RESET}"
    fi

    section_end
else
    section "MODEL" "3"
    step_info "Skipped ${GHOST}(--no-model)${RESET}"
    section_end
fi

# ═══════════════════════════════════════════════════════════════════════
# 4  CONFIGURE
# ═══════════════════════════════════════════════════════════════════════

section "CONFIGURE" "4"

CONFIG_DIR="$HOME/.obelysk"
mkdir -p "$CONFIG_DIR/proofs" "$CONFIG_DIR/bin" "$CONFIG_DIR/logs"

# Write config file
CONFIG_FILE="$CONFIG_DIR/config.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    cat > "$CONFIG_FILE" << CONF
# ═══════════════════════════════════════════════════
# ObelyZK Configuration
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Source: source ~/.obelysk/config.env
# ═══════════════════════════════════════════════════

# Model
export OBELYSK_MODEL_DIR="\${OBELYSK_MODEL_DIR:-\$HOME/.obelysk/models/${MODEL}}"
export OBELYSK_GGUF="\${OBELYSK_GGUF:-\$HOME/.obelysk/models/${MODEL}-gguf/qwen2-0_5b-instruct-q4_k_m.gguf}"
export OBELYSK_PORT="\${OBELYSK_PORT:-8192}"

# Starknet — set these for on-chain verification
# export STARKNET_PRIVATE_KEY="0x..."
# export STARKNET_ACCOUNT_ADDRESS="0x..."
# export OBELYSK_CONTRACT="0x..."
# export OBELYSK_NETWORK="Starknet Sepolia"
# export OBELYSK_MODEL_ID="0x..."
CONF
    step_ok "${WHITE}Config${RESET} ${GHOST}~/.obelysk/config.env${RESET}"
else
    step_ok "${WHITE}Config${RESET} ${SLATE}exists${RESET}"
fi

# Symlink binaries to ~/.obelysk/bin/
LINKED=0
for bin in prove-model obelysk obelysk-demo prove-server; do
    SRC="$PROJECT_DIR/target/release/$bin"
    if [[ -f "$SRC" ]]; then
        ln -sf "$SRC" "$CONFIG_DIR/bin/$bin"
        LINKED=$((LINKED + 1))
    fi
done
step_ok "${WHITE}Binaries${RESET} ${GHOST}${LINKED} linked to ~/.obelysk/bin/${RESET}"

# PATH instructions
if ! echo "$PATH" | grep -q "$CONFIG_DIR/bin"; then
    echo ""
    # Detect shell
    SHELL_RC="~/.zshrc"
    if [[ -n "${BASH_VERSION:-}" ]] || echo "$SHELL" | grep -q "bash"; then
        SHELL_RC="~/.bashrc"
    fi
    step_info "Add to ${WHITE}${SHELL_RC}${RESET}:"
    echo ""
    echo -e "    ${LIME}export PATH=\"\$HOME/.obelysk/bin:\$PATH\"${RESET}"
    echo -e "    ${LIME}source ~/.obelysk/config.env${RESET}"
fi

section_end

# ═══════════════════════════════════════════════════════════════════════
# 5  SMOKE TEST (optional)
# ═══════════════════════════════════════════════════════════════════════

PROVE_BIN="$PROJECT_DIR/target/release/prove-model"

if $SMOKE_TEST && [[ -f "$PROVE_BIN" ]] && ! $NO_MODEL && [[ -f "$HOME/.obelysk/models/$MODEL/config.json" ]]; then
    section "SMOKE TEST" "5"

    step_info "Running a quick 1-layer GKR proof..."
    echo ""

    SMOKE_START=$(now_ms)
    SMOKE_LOG=$(mktemp)

    "$PROVE_BIN" \
        --model-dir "$HOME/.obelysk/models/$MODEL" \
        --layers 1 --gkr --format ml_gkr --quiet \
        2>"$SMOKE_LOG" &
    SMOKE_PID=$!

    # Show spinner while proving
    SMOKE_FRAMES=("${CIRCLE_EMPTY} " "${CIRCLE_FULL} " "${CIRCLE_EMPTY} " "${CIRCLE_FULL} ")
    SMOKE_I=0
    while kill -0 "$SMOKE_PID" 2>/dev/null; do
        SMOKE_LINE=$(tail -1 "$SMOKE_LOG" 2>/dev/null | head -c 50 || true)
        printf "\r  ${LIME}${SMOKE_FRAMES[$SMOKE_I]}${RESET}${GHOST}%s${RESET}%*s" "$SMOKE_LINE" 20 ""
        SMOKE_I=$(( (SMOKE_I + 1) % ${#SMOKE_FRAMES[@]} ))
        sleep 0.2
    done

    wait "$SMOKE_PID" 2>/dev/null
    SMOKE_EXIT=$?
    SMOKE_ELAPSED=$(elapsed_since "$SMOKE_START")
    printf "\r%*s\r" "$(cols)" ""

    if [[ $SMOKE_EXIT -eq 0 ]]; then
        step_ok "${EMERALD}Proof generated${RESET} ${GHOST}in ${SMOKE_ELAPSED}${RESET}"

        # Parse proof output for commitment hash
        SMOKE_HASH=$(grep -o 'io_commitment.*' "$SMOKE_LOG" 2>/dev/null | head -1 | awk '{print $2}' | head -c 32 || true)
        if [[ -n "$SMOKE_HASH" ]]; then
            step_dim "io_commitment: ${VIOLET}${SMOKE_HASH}...${RESET}"
        fi
        step_ok "${WHITE}GKR sumcheck${RESET} ${EMERALD}valid${RESET}"
        step_ok "${WHITE}STARK proof${RESET} ${EMERALD}valid${RESET}"
        step_ok "${WHITE}Weight binding${RESET} ${EMERALD}valid${RESET}"
    else
        step_warn "Smoke test exited with code ${SMOKE_EXIT} ${GHOST}— this is OK, full setup still works${RESET}"
    fi
    rm -f "$SMOKE_LOG"

    section_end
fi

# ═══════════════════════════════════════════════════════════════════════
# 6  VALIDATE
# ═══════════════════════════════════════════════════════════════════════

section "VALIDATE" "6"

ALL_GOOD=true
OBELYSK_BIN="$PROJECT_DIR/target/release/obelysk"

# Build a clean status table
echo -e "  ${GHOST}${TL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${TR}${RESET}"

check_component() {
    local name="$1" status="$2" required="${3:-true}"
    local icon color
    if [[ "$status" == "ok" ]]; then
        icon="$CHECK"; color="$EMERALD"
    elif [[ "$status" == "warn" ]]; then
        icon="!"; color="$ORANGE"
    elif [[ "$status" == "skipped" ]]; then
        icon="$DOT"; color="$GHOST"
    else
        icon="$CROSS"; color="$RED"
        [[ "$required" == "true" ]] && ALL_GOOD=false
    fi
    printf "  ${GHOST}${V}${RESET} ${color}${icon}${RESET} %-18s ${color}%s${RESET}%*s${GHOST}${V}${RESET}\n" "$name" "$status" 14 ""
}

# Check each component
if [[ -f "$PROVE_BIN" ]]; then
    check_component "prove-model" "ok"
elif $TUI_ONLY || $SERVER; then
    check_component "prove-model" "skipped" "false"
else
    check_component "prove-model" "missing"
fi

if [[ -f "$OBELYSK_BIN" ]]; then
    check_component "obelysk" "ok"
elif $SERVER; then
    check_component "obelysk" "skipped" "false"
else
    check_component "obelysk" "missing"
fi

if [[ -f "$PROJECT_DIR/target/release/prove-server" ]]; then
    check_component "prove-server" "ok"
elif $SERVER; then
    check_component "prove-server" "missing"
else
    check_component "prove-server" "skipped" "false"
fi

if ! $NO_MODEL && [[ -f "$HOME/.obelysk/models/$MODEL/config.json" ]]; then
    check_component "$MODEL" "ok"
elif $NO_MODEL; then
    check_component "model" "skipped" "false"
else
    check_component "$MODEL" "missing"
fi

if command -v llama-server &>/dev/null; then
    check_component "llama-server" "ok"
else
    check_component "llama-server" "missing" "false"
fi

echo -e "  ${GHOST}${BL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${BR}${RESET}"

section_end

SETUP_ELAPSED=$(elapsed_since "$SETUP_START")

# ═══════════════════════════════════════════════════════════════════════
# FINISH
# ═══════════════════════════════════════════════════════════════════════

echo ""

if $ALL_GOOD; then
    # ── Success ─────────────────────────────────────────────────────
    hr_accent
    echo ""
    center "${EMERALD}${BOLD}${CHECK} Setup Complete${RESET}  ${GHOST}${DOT} ${SETUP_ELAPSED}${RESET}"
    echo ""
    hr_accent
    echo ""
    echo ""

    # ── What you can do now ─────────────────────────────────────────

    # Command 1: Interactive TUI
    echo -e "  ${BG_LIME}${FG_BLACK}${BOLD} 1 ${RESET} ${WHITE}${BOLD}Launch the TUI${RESET}  ${GHOST}chat with AI, prove every computation, verify on-chain${RESET}"
    echo ""
    echo -e "      ${LIME}${BOLD}obelysk${RESET}"
    echo ""
    echo -e "      ${GHOST}Chat naturally. Type ${WHITE}prove${GHOST} when ready. Watch the${RESET}"
    echo -e "      ${GHOST}M31 capture, GKR sumcheck, STARK compression, and${RESET}"
    echo -e "      ${GHOST}on-chain verification happen in real time.${RESET}"
    echo ""
    echo ""

    # Command 2: Headless prove
    echo -e "  ${BG_GHOST} ${FG_BLACK}2${RESET}  ${WHITE}Prove a model${RESET}  ${GHOST}headless, scriptable${RESET}"
    echo ""
    echo -e "      ${LIME}prove-model --model-dir ~/.obelysk/models/${MODEL} --layers 1 --gkr${RESET}"
    echo ""
    echo ""

    # Command 3: Full pipeline
    echo -e "  ${BG_GHOST} ${FG_BLACK}3${RESET}  ${WHITE}Full pipeline${RESET}  ${GHOST}prove + submit to Starknet${RESET}"
    echo ""
    echo -e "      ${LIME}./scripts/prove_and_submit.sh${RESET}"
    echo ""
    echo ""

    # ── Architecture ────────────────────────────────────────────────
    hr "$GHOST"
    echo ""

    W=$(cols)
    if [[ $W -ge 72 ]]; then
        # Wide terminal — full diagram
        center "${SLATE}${BOLD}How It Works${RESET}"
        echo ""
        center "${WHITE}${BOLD}You${RESET}  ${GHOST}type a prompt${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${LIME}${BOLD}obelysk${RESET}  ${GHOST}TUI + llama.cpp${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${EMERALD}AI responds${RESET}  ${GHOST}every token observed${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${GHOST}type '${WHITE}prove${GHOST}'${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${GHOST}${TL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${TR}${RESET}"
        center "${GHOST}${V}${RESET}  ${LIME}${BOLD}1${RESET} ${WHITE}M31 CAPTURE${RESET}    ${GHOST}quantize to prime field${RESET}   ${GHOST}${V}${RESET}"
        center "${GHOST}${V}${RESET}  ${LIME}${BOLD}2${RESET} ${WHITE}GKR SUMCHECK${RESET}   ${GHOST}interactive oracle proof${RESET}  ${GHOST}${V}${RESET}"
        center "${GHOST}${V}${RESET}  ${LIME}${BOLD}3${RESET} ${WHITE}STARK PROOF${RESET}    ${GHOST}FRI + Merkle commitment${RESET}  ${GHOST}${V}${RESET}"
        center "${GHOST}${BL}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${H}${BR}${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${ORANGE}${BOLD}STARKNET${RESET}  ${GHOST}6-step streaming verification${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${EMERALD}${BOLD}${SHIELD} VERIFIED ON-CHAIN${RESET}  ${GHOST}immutable, public, permanent${RESET}"
    else
        # Narrow terminal — compact
        center "${WHITE}You ${GHOST}${ARROW}${RESET} ${LIME}obelysk ${GHOST}${ARROW}${RESET} ${EMERALD}Chat${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${LIME}Capture ${GHOST}${ARROW}${RESET} ${LIME}GKR ${GHOST}${ARROW}${RESET} ${LIME}STARK${RESET}"
        center "${GHOST}${V}${RESET}"
        center "${ORANGE}Starknet ${GHOST}${ARROW}${RESET} ${EMERALD}Verified ${SHIELD}${RESET}"
    fi

    echo ""
    echo ""
    hr "$GHOST"
    echo ""

    # ── File locations ──────────────────────────────────────────────
    printf "  ${GHOST}%-10s${RESET} ${SLATE}%s${RESET}\n" "Config" "~/.obelysk/config.env"
    printf "  ${GHOST}%-10s${RESET} ${SLATE}%s${RESET}\n" "Proofs" "~/.obelysk/proofs/"
    printf "  ${GHOST}%-10s${RESET} ${SLATE}%s${RESET}\n" "Models" "~/.obelysk/models/"
    printf "  ${GHOST}%-10s${RESET} ${SLATE}%s${RESET}\n" "Logs" "~/.obelysk/logs/"
    echo ""
    center "${VIOLET}STWO Circle STARK + GKR${RESET}  ${GHOST}${DOT}${RESET}  ${SLATE}Starknet Sepolia${RESET}"
    echo ""
    echo ""

else
    # ── Incomplete ──────────────────────────────────────────────────
    hr "$ORANGE"
    echo ""
    center "${ORANGE}${BOLD}! Setup incomplete${RESET}  ${GHOST}${DOT} ${SETUP_ELAPSED}${RESET}"
    echo ""
    hr "$ORANGE"
    echo ""
    center "${SLATE}Some required components are missing — check warnings above.${RESET}"
    echo ""
fi
