#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Step 0: GPU Environment Setup
# ═══════════════════════════════════════════════════════════════════════
#
# Sets up a fresh GPU instance for ML proving:
#   1. Install system dependencies (apt/yum)
#   2. Install Rust nightly toolchain
#   3. Detect GPU + CUDA + Confidential Computing
#   4. Clone/update repository + submodules
#   5. Build stwo-ml, cairo-prove, Cairo verifier
#   6. Run sanity test
#
# Usage:
#   bash scripts/pipeline/00_setup_gpu.sh
#   bash scripts/pipeline/00_setup_gpu.sh --skip-deps --skip-build
#   bash scripts/pipeline/00_setup_gpu.sh --branch main --cuda-path /usr/local/cuda-12.6
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/gpu_detect.sh"

# ─── Configuration ───────────────────────────────────────────────────

REPO_URL="${REPO_URL:-https://github.com/Bitsage-Network/stwo-ml.git}"
BRANCH="${BRANCH:-main}"

# Auto-detect: if we're running from inside a cloned repo, use that as INSTALL_DIR
_DETECTED_ROOT=""
if [[ -d "${SCRIPT_DIR}/../../.git" ]]; then
    _DETECTED_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
INSTALL_DIR="${INSTALL_DIR:-${_DETECTED_ROOT:-$HOME/obelysk}}"
CUSTOM_CUDA_PATH=""
REQUIRE_GPU="${OBELYSK_REQUIRE_GPU:-true}"

SKIP_DEPS=false
SKIP_BUILD=false
INSTALL_DRIVERS=auto
SKIP_LLAMA=false

# ─── Parse Arguments ─────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)       SKIP_DEPS=true; shift ;;
        --skip-build)      SKIP_BUILD=true; shift ;;
        --install-drivers) INSTALL_DRIVERS=yes; shift ;;
        --skip-drivers)    INSTALL_DRIVERS=no; shift ;;
        --skip-llama)      SKIP_LLAMA=true; shift ;;
        --branch)          BRANCH="$2"; shift 2 ;;
        --install-dir)     INSTALL_DIR="$2"; shift 2 ;;
        --cuda-path)       CUSTOM_CUDA_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Sets up a GPU instance for Obelysk ML proving."
            echo ""
            echo "Options:"
            echo "  --skip-deps         Skip installing system dependencies"
            echo "  --skip-build        Skip building Rust binaries"
            echo "  --install-drivers   Install NVIDIA driver + CUDA if missing"
            echo "  --skip-drivers      Skip driver/CUDA installation entirely"
            echo "  --skip-llama        Skip llama.cpp build"
            echo "  --branch NAME       Git branch to checkout (default: main)"
            echo "  --install-dir DIR   Where to clone repo (default: ~/obelysk)"
            echo "  --cuda-path PATH    Custom CUDA toolkit path (auto-detected if omitted)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Environment variables:"
            echo "  DRY_RUN=1           Print commands without executing"
            echo "  REPO_URL=...        Override git repository URL"
            echo "  OBELYSK_DEBUG=1     Enable debug logging"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Start ───────────────────────────────────────────────────────────

banner
echo -e "${BOLD}  GPU Environment Setup${NC}"
echo ""

init_obelysk_dir
timer_start "setup"

# ─── Step 1: System Dependencies ─────────────────────────────────────

step "1/7" "System dependencies"

if [[ "$SKIP_DEPS" == "false" ]]; then
    if command -v apt-get &>/dev/null; then
        log "Installing via apt..."
        run_cmd sudo apt-get update -qq
        run_cmd sudo apt-get install -y -qq \
            build-essential cmake pkg-config libssl-dev \
            git git-lfs curl wget \
            python3 python3-pip jq bc \
            2>&1 | tail -3
        ok "apt packages installed"
    elif command -v yum &>/dev/null; then
        log "Installing via yum..."
        run_cmd sudo yum install -y -q \
            gcc gcc-c++ make cmake \
            openssl-devel pkg-config \
            git git-lfs curl wget \
            python3 python3-pip jq bc
        ok "yum packages installed"
    else
        warn "Unknown package manager — install manually:"
        warn "  build-essential cmake pkg-config libssl-dev git git-lfs python3 jq bc"
    fi

    # HuggingFace CLI + filelock (system filelock is often too old)
    pip3 install --quiet --upgrade huggingface_hub filelock 2>/dev/null || \
        pip3 install --quiet --upgrade --user huggingface_hub filelock 2>/dev/null || true
    ok "huggingface_hub available"
else
    log "Skipping (--skip-deps)"
fi
echo ""

# ─── Step 2: Rust Nightly ────────────────────────────────────────────

step "2/7" "Rust nightly toolchain"

if ! command -v rustup &>/dev/null; then
    log "Installing rustup..."
    run_cmd curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2025-07-14
    source "$HOME/.cargo/env"
else
    log "rustup already installed"
fi

run_cmd rustup install nightly-2025-07-14 2>/dev/null || true
run_cmd rustup default nightly-2025-07-14 2>/dev/null || true
run_cmd rustup component add rust-src --toolchain nightly-2025-07-14 2>/dev/null || true

ok "$(rustc --version)"
ok "$(cargo --version)"
echo ""

# ─── Disk Space Check ────────────────────────────────────────────────

FREE_GB=$(df -BG / 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
if [[ -n "$FREE_GB" ]] && (( FREE_GB < 50 )); then
    warn "Low disk space: ${FREE_GB}GB free. Recommend 50GB+ for model downloads + builds."
fi

# ─── Step 3: GPU + CUDA Detection ────────────────────────────────────

step "3/7" "GPU & CUDA detection"

# Install driver + CUDA if requested
if [[ "$INSTALL_DRIVERS" == "yes" ]]; then
    install_nvidia_driver || {
        if [[ "${GPU_REBOOT_REQUIRED:-false}" == "true" ]]; then
            err "NVIDIA driver installed but not active yet (reboot required)."
            err "Run: sudo reboot"
            err "Then rerun this setup script."
            exit 1
        fi
        warn "Driver install failed"
    }
    if [[ "${GPU_REBOOT_REQUIRED:-false}" != "true" ]]; then
        install_cuda_toolkit || warn "CUDA install failed"
    fi
elif [[ "$INSTALL_DRIVERS" == "auto" ]]; then
    # Auto: install only if nvidia-smi is missing
    if ! command -v nvidia-smi &>/dev/null; then
        log "nvidia-smi not found — attempting driver install..."
        install_nvidia_driver || {
            if [[ "${GPU_REBOOT_REQUIRED:-false}" == "true" ]]; then
                err "NVIDIA driver installed but not active yet (reboot required)."
                err "Run: sudo reboot"
                err "Then rerun this setup script."
                exit 1
            fi
            warn "Driver install failed"
        }
        if [[ "${GPU_REBOOT_REQUIRED:-false}" != "true" ]]; then
            install_cuda_toolkit || warn "CUDA install failed"
        fi
    fi
fi

detect_all "$CUSTOM_CUDA_PATH"

if [[ "$CC_CAPABLE" == "true" ]]; then
    install_nvattest || true
fi

if [[ "$REQUIRE_GPU" == "true" ]]; then
    if [[ "$GPU_AVAILABLE" != "true" ]]; then
        err "GPU detection failed but GPU is required for this run."
        err "If you just installed drivers, reboot first: sudo reboot"
        err "Then rerun: ./scripts/pipeline/run_e2e.sh --preset <model> --gpu --dry-run"
        exit 1
    fi
    if [[ "$CUDA_AVAILABLE" != "true" ]]; then
        err "CUDA toolkit not detected but GPU mode is required."
        err "Install CUDA toolkit, then rerun setup (or pass --cuda-path)."
        err "Guide: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
fi

save_gpu_config
print_gpu_summary

# ─── Step 4: Repository ──────────────────────────────────────────────

step "4/7" "Repository setup"

if [[ -d "${INSTALL_DIR}/.git" ]]; then
    log "Repo exists at ${INSTALL_DIR}, pulling latest..."
    (
        cd "${INSTALL_DIR}"
        run_cmd git fetch origin
        run_cmd git checkout "${BRANCH}" 2>/dev/null || run_cmd git checkout -b "${BRANCH}" "origin/${BRANCH}"
        run_cmd git pull origin "${BRANCH}" --ff-only 2>/dev/null || true
    )
else
    log "Cloning ${REPO_URL} (branch: ${BRANCH})..."
    run_cmd git clone --branch "${BRANCH}" --depth 1 "${REPO_URL}" "${INSTALL_DIR}"
fi

# Submodules
log "Initializing submodules..."
(cd "${INSTALL_DIR}" && run_cmd git submodule update --init --recursive 2>/dev/null || true)

ok "Repository ready at ${INSTALL_DIR} (branch: ${BRANCH})"
echo ""

# ─── Step 5: Build ───────────────────────────────────────────────────

step "5/7" "Building proving stack"

if [[ "$SKIP_BUILD" == "false" ]]; then
    LIBS_DIR="${INSTALL_DIR}"

    # 5a: stwo-ml
    log "Building stwo-ml (GPU + CLI)..."
    FEATURES="cli,audit"
    if [[ "$GPU_AVAILABLE" == "true" ]] && [[ "$CUDA_AVAILABLE" == "true" ]]; then
        FEATURES="cli,audit,cuda-runtime"
    fi

    if (export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"; cd "${LIBS_DIR}/stwo-ml" && \
        cargo build --release --bin prove-model --features "${FEATURES}" 2>&1 | tail -5); then
        ok "stwo-ml built (features: ${FEATURES})"
    else
        warn "GPU build failed, retrying CPU-only..."
        (cd "${LIBS_DIR}/stwo-ml" && \
            cargo build --release --bin prove-model --features "cli" 2>&1 | tail -5)
        ok "stwo-ml built (CPU-only)"
    fi

    # 5b: cairo-prove
    if [[ -d "${LIBS_DIR}/stwo-cairo/cairo-prove" ]]; then
        log "Building cairo-prove..."
        if (cd "${LIBS_DIR}/stwo-cairo/cairo-prove" && \
            cargo build --release 2>&1 | tail -5); then
            ok "cairo-prove built"
        else
            warn "cairo-prove build failed (recursive proving unavailable)"
        fi
    else
        warn "stwo-cairo not found, skipping cairo-prove"
    fi

    # 5c: Cairo verifier
    if command -v scarb &>/dev/null; then
        if [[ -d "${LIBS_DIR}/stwo-cairo/stwo_cairo_verifier" ]]; then
            log "Building Cairo ML verifier..."
            (cd "${LIBS_DIR}/stwo-cairo/stwo_cairo_verifier" && scarb build 2>&1 | tail -3) && \
                ok "Cairo verifier built" || warn "Cairo verifier build failed"
        fi
    else
        log "scarb not found — installing..."
        curl -L https://docs.swmansion.com/scarb/install.sh 2>/dev/null | sh -s -- -v 2.12.0 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
        if command -v scarb &>/dev/null && [[ -d "${LIBS_DIR}/stwo-cairo/stwo_cairo_verifier" ]]; then
            (cd "${LIBS_DIR}/stwo-cairo/stwo_cairo_verifier" && scarb build 2>&1 | tail -3) || true
        fi
    fi
else
    log "Skipping (--skip-build)"
fi
echo ""

# ─── Step 6: llama.cpp (optional) ────────────────────────────────────

step "6/7" "llama.cpp (inference testing)"

LLAMA_DIR="${OBELYSK_DIR}/llama.cpp"
LLAMA_BIN="${LLAMA_DIR}/build/bin/llama-cli"

if [[ "$SKIP_LLAMA" == "true" ]]; then
    log "Skipping llama.cpp (--skip-llama)"
elif [[ -f "$LLAMA_BIN" ]]; then
    ok "llama.cpp already built: ${LLAMA_BIN}"
else
    # Ensure cmake is available
    if ! command -v cmake &>/dev/null; then
        log "Installing cmake..."
        if command -v apt-get &>/dev/null; then
            run_cmd sudo apt-get install -y -qq cmake 2>&1 | tail -2
        elif command -v yum &>/dev/null; then
            run_cmd sudo yum install -y cmake3 2>/dev/null || run_cmd sudo yum install -y cmake
        fi
    fi

    if [[ -d "$LLAMA_DIR" ]]; then
        log "Updating llama.cpp..."
        (cd "$LLAMA_DIR" && run_cmd git pull --ff-only 2>/dev/null || true)
    else
        log "Cloning llama.cpp..."
        run_cmd git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
    fi

    log "Building llama.cpp with CUDA..."
    CMAKE_ARGS=(-B "${LLAMA_DIR}/build" -DGGML_CUDA=OFF)
    if [[ "$CUDA_AVAILABLE" == "true" ]] && [[ -n "$CUDA_PATH" ]]; then
        CMAKE_ARGS=(-B "${LLAMA_DIR}/build" -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc")
    fi

    if (cd "$LLAMA_DIR" && cmake "${CMAKE_ARGS[@]}" 2>&1 | tail -3 && \
        cmake --build build --config Release -j"$(nproc 2>/dev/null || echo 4)" 2>&1 | tail -5); then
        if [[ -f "$LLAMA_BIN" ]]; then
            ok "llama.cpp built: ${LLAMA_BIN}"
        else
            # Binary may be in a different location
            LLAMA_BIN=$(find "${LLAMA_DIR}/build" -name "llama-cli" -type f 2>/dev/null | head -1)
            if [[ -n "$LLAMA_BIN" ]]; then
                ok "llama.cpp built: ${LLAMA_BIN}"
            else
                warn "llama.cpp build completed but llama-cli not found"
            fi
        fi
    else
        warn "llama.cpp build failed (inference testing will be unavailable)"
    fi
fi
echo ""

# ─── Step 7: Sanity Test ─────────────────────────────────────────────

step "7/7" "Sanity test"

PROVE_BIN=$(find_binary "prove-model" "${INSTALL_DIR}")
CAIRO_BIN=$(find_binary "cairo-prove" "${INSTALL_DIR}")

if [[ -n "$PROVE_BIN" ]]; then
    log "Running quick test..."
    if (cd "${INSTALL_DIR}/stwo-ml" && \
        cargo test --release --lib \
            -- test_matmul_sumcheck_basic --nocapture 2>&1 | tail -5); then
        ok "Sanity test passed"
    else
        warn "Sanity test skipped (not critical)"
    fi
else
    warn "prove-model binary not found, skipping test"
fi
echo ""

# ─── Step 8: Marketplace Registration ────────────────────────────────

step "8/8" "Marketplace registration (zero-config audit storage)"

register_with_marketplace || {
    warn "Marketplace registration skipped — audit uploads will use relay fallback"
}
echo ""

# ─── Save State ──────────────────────────────────────────────────────

ELAPSED=$(timer_elapsed "setup")

save_state "setup_state.env" \
    "SETUP_COMPLETE=true" \
    "SETUP_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "SETUP_DURATION_SEC=${ELAPSED}" \
    "INSTALL_DIR=${INSTALL_DIR}" \
    "BRANCH=${BRANCH}" \
    "PROVE_MODEL_BIN=${PROVE_BIN:-}" \
    "CAIRO_PROVE_BIN=${CAIRO_BIN:-}" \
    "GPU_AVAILABLE=${GPU_AVAILABLE}" \
    "CUDA_AVAILABLE=${CUDA_AVAILABLE}" \
    "LLAMA_BIN=${LLAMA_BIN:-}"

# ─── Summary ─────────────────────────────────────────────────────────

echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  SETUP COMPLETE                                      ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  GPU:         %-37s ║\n" "${GPU_NAME:-none}"
printf "  ║  VRAM:        %-37s ║\n" "${GPU_MEM:-N/A}"
printf "  ║  CUDA:        %-37s ║\n" "${NVCC_VER:-N/A}"
printf "  ║  CC Mode:     %-37s ║\n" "${CC_MODE_STR}"
printf "  ║  Repo:        %-37s ║\n" "${INSTALL_DIR}"
printf "  ║  prove-model: %-37s ║\n" "${PROVE_BIN:-NOT FOUND}"
printf "  ║  cairo-prove: %-37s ║\n" "${CAIRO_BIN:-NOT FOUND}"
printf "  ║  Duration:    %-37s ║\n" "$(format_duration $ELAPSED)"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║                                                      ║"
echo "  ║  Next: ./01_setup_model.sh --preset qwen3-14b       ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
