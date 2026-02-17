#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — GPU & CUDA Detection
# ═══════════════════════════════════════════════════════════════════════
#
# Source this after common.sh:
#   source "${SCRIPT_DIR}/lib/common.sh"
#   source "${SCRIPT_DIR}/lib/gpu_detect.sh"
#
# Provides:
#   - detect_gpu         → GPU_NAME, GPU_MEM, GPU_DRIVER, GPU_COUNT
#   - detect_cuda        → CUDA_PATH, NVCC_VER
#   - detect_compute_cap → COMPUTE_CAP, COMPUTE_MAJOR
#   - detect_cc_mode     → CC_CAPABLE, CC_ACTIVE, CC_MODE_STR
#   - setup_cuda_env     → exports PATH, LD_LIBRARY_PATH, CUDA_HOME
#   - save_gpu_config    → writes ~/.obelysk/gpu_config.env
#   - print_gpu_summary  → pretty-print detected config

[[ -n "${_OBELYSK_GPU_DETECT_LOADED:-}" ]] && return 0
_OBELYSK_GPU_DETECT_LOADED=1
GPU_REBOOT_REQUIRED=false

# ─── Distro Detection ────────────────────────────────────────────────

_detect_distro() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "${ID:-unknown}"
    elif command -v lsb_release &>/dev/null; then
        lsb_release -is 2>/dev/null | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

_ensure_cuda_apt_repo() {
    if ! command -v apt-get &>/dev/null; then
        return 0
    fi

    if apt-cache show cuda-toolkit-12-8 &>/dev/null || apt-cache show cuda-toolkit &>/dev/null; then
        debug "CUDA apt package already available"
        return 0
    fi

    local id version_id repo_base
    id="$(_detect_distro)"
    version_id=""
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        version_id="${VERSION_ID:-}"
    fi

    case "$id" in
        ubuntu)
            repo_base="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${version_id//./}/x86_64"
            ;;
        debian)
            repo_base="https://developer.download.nvidia.com/compute/cuda/repos/debian${version_id%%.*}/x86_64"
            ;;
        *)
            return 0
            ;;
    esac

    if [[ -z "$version_id" ]] || [[ "$repo_base" =~ /ubuntu/x86_64$ ]] || [[ "$repo_base" =~ /debian/x86_64$ ]]; then
        warn "Could not determine distro version for CUDA repo setup"
        return 1
    fi

    if grep -Rqs "developer.download.nvidia.com/compute/cuda/repos" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; then
        debug "CUDA apt repo already configured"
        return 0
    fi

    log "Adding NVIDIA CUDA apt repository..."
    if ! run_cmd wget -qO /tmp/cuda-keyring.deb "${repo_base}/cuda-keyring_1.1-1_all.deb"; then
        warn "Failed to download cuda-keyring from ${repo_base}"
        return 1
    fi
    if ! run_cmd sudo dpkg -i /tmp/cuda-keyring.deb; then
        warn "Failed to install cuda-keyring"
        return 1
    fi
    run_cmd rm -f /tmp/cuda-keyring.deb
    run_cmd sudo apt-get update -qq
    ok "NVIDIA CUDA apt repository configured"
    return 0
}

# ─── NVIDIA Driver Installation ─────────────────────────────────────

install_nvidia_driver() {
    GPU_REBOOT_REQUIRED=false

    # Skip if driver is already installed and working
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        ok "NVIDIA driver already installed ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1))"
        return 0
    fi

    local distro
    distro=$(_detect_distro)
    log "Installing NVIDIA driver for ${distro}..."

    case "$distro" in
        ubuntu|debian)
            run_cmd sudo apt-get update -qq
            # Install the latest recommended driver (550+ supports 4090 through B300)
            if run_cmd sudo apt-get install -y -qq nvidia-driver-550 2>&1 | tail -3; then
                ok "NVIDIA driver 550 installed via apt"
            else
                warn "nvidia-driver-550 not available, trying nvidia-driver..."
                run_cmd sudo apt-get install -y -qq nvidia-driver 2>&1 | tail -3 || true
            fi
            ;;
        rhel|centos|rocky|almalinux|fedora)
            if command -v dnf &>/dev/null; then
                run_cmd sudo dnf install -y nvidia-driver 2>&1 | tail -3 || true
            elif command -v yum &>/dev/null; then
                run_cmd sudo yum install -y nvidia-driver 2>&1 | tail -3 || true
            fi
            ;;
        *)
            warn "Unsupported distro '${distro}' for automatic driver install."
            warn "Install NVIDIA driver manually from: https://www.nvidia.com/drivers"
            return 1
            ;;
    esac

    # Verify installation
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        ok "NVIDIA driver installed successfully"
        return 0
    fi

    # Driver may require reboot to load
    if command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi found but cannot query GPU. A reboot may be required."
        warn "  sudo reboot"
        GPU_REBOOT_REQUIRED=true
    else
        err "NVIDIA driver installation failed."
        err "Try the NVIDIA .run installer: https://www.nvidia.com/drivers"
    fi
    return 1
}

# ─── CUDA Toolkit Installation ──────────────────────────────────────

install_cuda_toolkit() {
    # Skip if nvcc already exists
    if command -v nvcc &>/dev/null; then
        ok "CUDA toolkit already installed ($(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//'))"
        return 0
    fi

    # Also check standard paths
    for dir in /usr/local/cuda/bin /usr/local/cuda-12.8/bin /usr/local/cuda-12.6/bin; do
        if [[ -f "${dir}/nvcc" ]]; then
            ok "CUDA toolkit found at ${dir%/bin}"
            return 0
        fi
    done

    local distro
    distro=$(_detect_distro)
    log "Installing CUDA 12.8 toolkit for ${distro}..."

    case "$distro" in
        ubuntu|debian)
            _ensure_cuda_apt_repo || true

            # Install cuda-toolkit via nvidia repo
            if run_cmd sudo apt-get install -y -qq cuda-toolkit-12-8 2>&1 | tail -3; then
                ok "CUDA 12.8 toolkit installed via apt"
            else
                log "cuda-toolkit-12-8 not in repo, trying generic cuda-toolkit..."
                run_cmd sudo apt-get install -y -qq cuda-toolkit 2>&1 | tail -3 || {
                    warn "CUDA toolkit install failed via apt."
                    warn "Install manually: https://developer.nvidia.com/cuda-downloads"
                    return 1
                }
            fi
            ;;
        rhel|centos|rocky|almalinux|fedora)
            if command -v dnf &>/dev/null; then
                run_cmd sudo dnf install -y cuda-toolkit-12-8 2>&1 | tail -3 || \
                    run_cmd sudo dnf install -y cuda-toolkit 2>&1 | tail -3 || true
            elif command -v yum &>/dev/null; then
                run_cmd sudo yum install -y cuda-toolkit-12-8 2>&1 | tail -3 || \
                    run_cmd sudo yum install -y cuda-toolkit 2>&1 | tail -3 || true
            fi
            ;;
        *)
            warn "Unsupported distro '${distro}' for automatic CUDA install."
            warn "Install from: https://developer.nvidia.com/cuda-downloads"
            return 1
            ;;
    esac

    # Set CUDA_PATH from the installed location
    for dir in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda; do
        if [[ -f "${dir}/bin/nvcc" ]]; then
            CUDA_PATH="$dir"
            ok "CUDA toolkit installed at ${CUDA_PATH}"
            return 0
        fi
    done

    warn "CUDA toolkit install completed but nvcc not found in standard paths."
    return 1
}

# ─── GPU Detection ───────────────────────────────────────────────────

GPU_NAME=""
GPU_MEM=""
GPU_DRIVER=""
GPU_COUNT=0
GPU_AVAILABLE=false

detect_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found — no NVIDIA GPU detected"
        GPU_AVAILABLE=false
        return 1
    fi

    local query_out first_line raw_name raw_mem raw_driver
    if ! query_out="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)"; then
        warn "nvidia-smi failed to query GPUs: $(echo "$query_out" | head -1 | xargs)"
        warn "This is usually a driver/library mismatch. Reboot and rerun setup."
        GPU_AVAILABLE=false
        GPU_NAME=""
        GPU_MEM=""
        GPU_DRIVER=""
        GPU_COUNT=0
        return 1
    fi

    if echo "$query_out" | grep -qiE "failed to initialize nvml|driver/library version mismatch|nvidia-smi has failed"; then
        warn "nvidia-smi returned an NVML error: $(echo "$query_out" | head -1 | xargs)"
        warn "Reboot required after driver update: sudo reboot"
        GPU_AVAILABLE=false
        GPU_NAME=""
        GPU_MEM=""
        GPU_DRIVER=""
        GPU_COUNT=0
        return 1
    fi

    first_line="$(echo "$query_out" | sed '/^[[:space:]]*$/d' | head -1 | xargs)"
    IFS=',' read -r raw_name raw_mem raw_driver <<< "$first_line"
    GPU_NAME="$(echo "${raw_name:-}" | xargs)"
    GPU_MEM="$(echo "${raw_mem:-}" | xargs)"
    GPU_DRIVER="$(echo "${raw_driver:-}" | xargs)"
    GPU_COUNT="$(echo "$query_out" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"

    if [[ -n "$GPU_NAME" ]]; then
        GPU_AVAILABLE=true
        ok "GPU detected: ${GPU_NAME} (${GPU_MEM})"
        if (( GPU_COUNT > 1 )); then
            log "Multi-GPU: ${GPU_COUNT} devices"
        fi
        return 0
    fi

    warn "nvidia-smi present but could not query GPU"
    GPU_AVAILABLE=false
    return 1
}

# ─── CUDA Toolkit Detection ─────────────────────────────────────────

CUDA_PATH=""
NVCC_VER=""
CUDA_AVAILABLE=false

# Default search paths for CUDA toolkit (newest first)
_CUDA_SEARCH_PATHS=(
    /usr/local/cuda-12.8
    /usr/local/cuda-12.6
    /usr/local/cuda-12.4
    /usr/local/cuda-12.2
    /usr/local/cuda
    /opt/cuda
    /usr/local/cuda-12.0
    /opt/cuda/bin/..
)

detect_cuda() {
    local custom_path="${1:-}"

    # If caller specified a path, try that first
    if [[ -n "$custom_path" ]] && [[ -f "${custom_path}/bin/nvcc" ]]; then
        CUDA_PATH="$custom_path"
    else
        # Search standard locations
        CUDA_PATH=""
        for dir in "${_CUDA_SEARCH_PATHS[@]}"; do
            if [[ -d "$dir" ]] && [[ -f "$dir/bin/nvcc" ]]; then
                CUDA_PATH="$dir"
                break
            fi
        done
    fi

    if [[ -z "$CUDA_PATH" ]]; then
        # Try nvcc in PATH as fallback
        if command -v nvcc &>/dev/null; then
            CUDA_PATH="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)" 2>/dev/null || command -v nvcc)")")"
            debug "Found nvcc in PATH: ${CUDA_PATH}"
        else
            # Try dpkg to find nvcc from installed CUDA packages
            local dpkg_nvcc
            dpkg_nvcc=$(dpkg -L cuda-toolkit-* 2>/dev/null | grep 'bin/nvcc$' | head -1 || echo "")
            if [[ -n "$dpkg_nvcc" ]] && [[ -f "$dpkg_nvcc" ]]; then
                CUDA_PATH="$(dirname "$(dirname "$dpkg_nvcc")")"
                debug "Found nvcc via dpkg: ${CUDA_PATH}"
            else
                warn "CUDA toolkit not found"
                warn "Searched: ${_CUDA_SEARCH_PATHS[*]}"
                CUDA_AVAILABLE=false
                return 1
            fi
        fi
    fi

    NVCC_VER=$("${CUDA_PATH}/bin/nvcc" --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_AVAILABLE=true
    ok "CUDA ${NVCC_VER} at ${CUDA_PATH}"
    return 0
}

# ─── Compute Capability ─────────────────────────────────────────────

COMPUTE_CAP=""
COMPUTE_MAJOR=0

detect_compute_cap() {
    if [[ "$GPU_AVAILABLE" != "true" ]]; then
        warn "Cannot detect compute capability: no GPU"
        return 1
    fi

    local compute_out
    compute_out="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | xargs || true)"
    if [[ -z "$compute_out" ]] || ! echo "$compute_out" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
        warn "Could not query compute capability (got: ${compute_out:-empty})"
        COMPUTE_CAP=""
        COMPUTE_MAJOR=0
        return 1
    fi
    COMPUTE_CAP="$compute_out"

    if [[ -n "$COMPUTE_CAP" ]]; then
        COMPUTE_MAJOR="$(echo "$COMPUTE_CAP" | cut -d. -f1)"
        if ! echo "$COMPUTE_MAJOR" | grep -Eq '^[0-9]+$'; then
            COMPUTE_MAJOR=0
        fi
        debug "Compute capability: ${COMPUTE_CAP} (major: ${COMPUTE_MAJOR})"
        return 0
    fi

    warn "Could not query compute capability"
    return 1
}

# ─── GPU Architecture Name ──────────────────────────────────────────

get_gpu_arch() {
    case "${COMPUTE_MAJOR:-0}" in
        7)  echo "Volta/Turing" ;;
        8)  echo "Ampere" ;;
        9)  echo "Hopper" ;;
        10) echo "Blackwell" ;;
        *)  echo "Unknown (SM ${COMPUTE_CAP:-?})" ;;
    esac
}

# ─── GPU Preset Selection ───────────────────────────────────────────
#
# Auto-selects GPU config based on detected GPU. Returns recommended
# chunk budget, layer limits, etc. Configs can also be loaded from
# configs/b200.env, configs/4090.env etc.

GPU_CHUNK_BUDGET_GB=16
GPU_MAX_LAYERS=0

get_gpu_preset() {
    local gpu="${GPU_NAME:-}"
    local mem_str="${GPU_MEM:-0 MiB}"
    local mem_mib
    mem_mib=$(echo "$mem_str" | grep -oP '\d+' | head -1)
    local mem_gb=$(( mem_mib / 1024 ))

    if echo "$gpu" | grep -qi "B300"; then
        GPU_CHUNK_BUDGET_GB=64
        GPU_MAX_LAYERS=0  # no limit
    elif echo "$gpu" | grep -qi "B200"; then
        GPU_CHUNK_BUDGET_GB=48
        GPU_MAX_LAYERS=0
    elif echo "$gpu" | grep -qi "H200"; then
        GPU_CHUNK_BUDGET_GB=32
        GPU_MAX_LAYERS=0
    elif echo "$gpu" | grep -qi "H100"; then
        GPU_CHUNK_BUDGET_GB=24
        GPU_MAX_LAYERS=0
    elif echo "$gpu" | grep -qi "A100"; then
        if (( mem_gb >= 80 )); then
            GPU_CHUNK_BUDGET_GB=24
        else
            GPU_CHUNK_BUDGET_GB=16
        fi
        GPU_MAX_LAYERS=0
    elif echo "$gpu" | grep -qi "4090"; then
        GPU_CHUNK_BUDGET_GB=8
        GPU_MAX_LAYERS=10
    elif echo "$gpu" | grep -qi "3090"; then
        GPU_CHUNK_BUDGET_GB=8
        GPU_MAX_LAYERS=5
    else
        # Default based on VRAM
        if (( mem_gb >= 80 )); then
            GPU_CHUNK_BUDGET_GB=24
        elif (( mem_gb >= 40 )); then
            GPU_CHUNK_BUDGET_GB=16
        elif (( mem_gb >= 16 )); then
            GPU_CHUNK_BUDGET_GB=8
        else
            GPU_CHUNK_BUDGET_GB=4
            GPU_MAX_LAYERS=3
        fi
    fi

    debug "GPU preset: chunk_budget=${GPU_CHUNK_BUDGET_GB}GB, max_layers=${GPU_MAX_LAYERS}"
}

# ─── Confidential Computing (CC) Mode ───────────────────────────────

CC_CAPABLE=false
CC_ACTIVE=false
CC_MODE_STR="N/A"
NVATTEST_AVAILABLE=false

detect_cc_mode() {
    CC_CAPABLE=false
    CC_ACTIVE=false
    CC_MODE_STR="N/A"

    # CC requires Hopper+ (compute >= 9.0)
    local compute_major="${COMPUTE_MAJOR:-0}"
    if ! echo "$compute_major" | grep -Eq '^[0-9]+$'; then
        compute_major=0
    fi
    if (( compute_major < 9 )); then
        debug "CC not supported: compute ${COMPUTE_CAP} < 9.0 (Hopper)"
        CC_MODE_STR="Not supported (compute ${COMPUTE_CAP})"
        return 0
    fi

    CC_CAPABLE=true
    log "CC-capable GPU detected (compute ${COMPUTE_CAP})"

    # Query CC mode
    local cc_output
    cc_output=$(nvidia-smi conf-compute -gcs 2>/dev/null || echo "")

    if echo "$cc_output" | grep -qi "ON\|Enabled"; then
        CC_ACTIVE=true
        CC_MODE_STR="ON"
        ok "CC Mode: ON (Confidential Computing active)"
    elif echo "$cc_output" | grep -qi "DEVTOOLS"; then
        CC_ACTIVE=true
        CC_MODE_STR="DEVTOOLS"
        warn "CC Mode: DEVTOOLS (debugging enabled — use ON for production)"
    else
        CC_MODE_STR="OFF"
        log "CC Mode: OFF"
        log "  To enable: sudo nvidia-smi conf-compute -scc on -i 0 (requires reboot)"
    fi

    # Check nvattest
    if command -v nvattest &>/dev/null; then
        NVATTEST_AVAILABLE=true
        debug "nvattest available"
    fi

    return 0
}

# ─── Install nvattest (optional) ─────────────────────────────────────

install_nvattest() {
    if [[ "$CC_CAPABLE" != "true" ]]; then
        debug "Skipping nvattest install: not CC-capable"
        return 0
    fi

    if command -v nvattest &>/dev/null; then
        NVATTEST_AVAILABLE=true
        ok "nvattest already installed"
        return 0
    fi

    log "Installing NVIDIA Attestation SDK..."
    if pip3 install --quiet nv-attestation-sdk 2>/dev/null || \
       pip3 install --quiet --user nv-attestation-sdk 2>/dev/null; then
        if command -v nvattest &>/dev/null; then
            NVATTEST_AVAILABLE=true
            ok "nvattest installed"
            return 0
        fi
        warn "nv-attestation-sdk installed but nvattest not in PATH"
        warn "Try: export PATH=\$HOME/.local/bin:\$PATH"
    else
        warn "nv-attestation-sdk install failed"
        warn "Manual: pip3 install nv-attestation-sdk"
    fi
    return 1
}

# ─── Setup CUDA Environment ─────────────────────────────────────────

setup_cuda_env() {
    if [[ -z "$CUDA_PATH" ]]; then
        warn "Cannot setup CUDA env: CUDA_PATH not set (run detect_cuda first)"
        return 1
    fi

    export PATH="${CUDA_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="${CUDA_PATH}"

    # Write persistent env file
    local envfile="${OBELYSK_DIR}/cuda_env.sh"
    mkdir -p "${OBELYSK_DIR}"
    cat > "$envfile" << ENVEOF
# Obelysk CUDA environment (auto-generated)
export PATH="${CUDA_PATH}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:\${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_PATH}"
ENVEOF

    debug "CUDA env written to ${envfile}"
    return 0
}

# ─── Save GPU Config ─────────────────────────────────────────────────

save_gpu_config() {
    local config_file="${OBELYSK_DIR}/gpu_config.env"
    mkdir -p "${OBELYSK_DIR}"

    cat > "$config_file" << GPUEOF
# Obelysk GPU Config (auto-generated at $(date -u +%Y-%m-%dT%H:%M:%SZ))
GPU_AVAILABLE=${GPU_AVAILABLE}
GPU_NAME=${GPU_NAME}
GPU_MEM=${GPU_MEM}
GPU_DRIVER=${GPU_DRIVER}
GPU_COUNT=${GPU_COUNT}
GPU_ARCH=$(get_gpu_arch)
CUDA_AVAILABLE=${CUDA_AVAILABLE}
CUDA_PATH=${CUDA_PATH}
NVCC_VER=${NVCC_VER}
COMPUTE_CAP=${COMPUTE_CAP}
COMPUTE_MAJOR=${COMPUTE_MAJOR}
CC_CAPABLE=${CC_CAPABLE}
CC_ACTIVE=${CC_ACTIVE}
CC_MODE_STR=${CC_MODE_STR}
NVATTEST_AVAILABLE=${NVATTEST_AVAILABLE}
GPU_CHUNK_BUDGET_GB=${GPU_CHUNK_BUDGET_GB}
GPU_MAX_LAYERS=${GPU_MAX_LAYERS}
GPUEOF

    ok "GPU config saved to ${config_file}"
}

# ─── Full Detection Pipeline ────────────────────────────────────────

detect_all() {
    local cuda_path="${1:-}"

    detect_gpu || true
    detect_cuda "$cuda_path" || true
    if [[ "$GPU_AVAILABLE" == "true" ]]; then
        if detect_compute_cap; then
            detect_cc_mode || true
        else
            CC_CAPABLE=false
            CC_ACTIVE=false
            CC_MODE_STR="Unknown (compute capability unavailable)"
        fi
        get_gpu_preset
    fi
    if [[ "$CUDA_AVAILABLE" == "true" ]]; then
        setup_cuda_env
    fi
}

# ─── Print Summary ──────────────────────────────────────────────────

print_gpu_summary() {
    echo "" >&2
    echo -e "${BOLD}  GPU Configuration${NC}" >&2
    echo "  ──────────────────────────────────────────" >&2
    echo "  GPU:          ${GPU_NAME:-not detected}" >&2
    echo "  VRAM:         ${GPU_MEM:-N/A}" >&2
    echo "  Driver:       ${GPU_DRIVER:-N/A}" >&2
    echo "  Devices:      ${GPU_COUNT}" >&2
    echo "  CUDA:         ${NVCC_VER:-not found} (${CUDA_PATH:-N/A})" >&2
    echo "  Compute Cap:  ${COMPUTE_CAP:-N/A}" >&2
    echo "  CC Mode:      ${CC_MODE_STR}" >&2
    echo "  nvattest:     ${NVATTEST_AVAILABLE}" >&2
    echo "  ──────────────────────────────────────────" >&2
    echo "" >&2
}
