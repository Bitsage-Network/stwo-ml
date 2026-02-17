#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Shared Utilities
# ═══════════════════════════════════════════════════════════════════════
#
# Source this file in pipeline scripts:
#   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#   source "${SCRIPT_DIR}/lib/common.sh"
#
# Provides:
#   - Colored logging (log, ok, err, warn, header, banner)
#   - State persistence (~/.obelysk/*.env)
#   - Prerequisite checks (check_command, check_file, check_dir)
#   - JSON field extraction (parse_json_field)
#   - Timing utilities (format_duration, timer_start, timer_elapsed)

# Prevent double-sourcing
[[ -n "${_OBELYSK_COMMON_LOADED:-}" ]] && return 0
_OBELYSK_COMMON_LOADED=1

# Ensure cargo/rustup are in PATH (non-login shells may miss this)
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env" 2>/dev/null || true

# ─── Colors ───────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ─── Logging ──────────────────────────────────────────────────────────

log()    { echo -e "${CYAN}[INFO]${NC} $*" >&2; }
ok()     { echo -e "${GREEN}[ OK ]${NC} $*" >&2; }
err()    { echo -e "${RED}[ERR ]${NC} $*" >&2; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
debug()  { [[ "${OBELYSK_DEBUG:-0}" == "1" ]] && echo -e "${DIM}[DBG ]${NC} $*" >&2 || true; }

header() {
    echo "" >&2
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}" >&2
    echo -e "${YELLOW}  $*${NC}" >&2
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}" >&2
}

step() {
    local num="$1"; shift
    echo -e "${YELLOW}[${num}]${NC} $*" >&2
}

banner() {
    echo -e "${CYAN}${BOLD}" >&2
    cat >&2 << 'EOF'
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ║
  ║   ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝██╔════╝██║  ║
  ║   ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝ ███████╗█████╗║
  ║   ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝  ╚════██║██╔═██║
  ║   ╚██████╔╝██████╔╝███████╗███████╗██║   ███████║██║ ██╗║
  ║    ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝ ╚═╝║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}" >&2
}

# ─── Obelysk Directory ───────────────────────────────────────────────

OBELYSK_DIR="${OBELYSK_DIR:-$HOME/.obelysk}"

init_obelysk_dir() {
    mkdir -p "${OBELYSK_DIR}"/{models,proofs,configs,logs}
    debug "Obelysk dir initialized: ${OBELYSK_DIR}"

    # Auto-load encrypted secrets (HF_TOKEN, IRYS_TOKEN, etc.)
    local _secrets_sh="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}/lib/secrets.sh"
    if [[ -f "$_secrets_sh" ]]; then
        source "$_secrets_sh"
        load_secrets
    fi

    # Auto-load marketplace credentials if cached
    if [[ -z "${MARKETPLACE_API_KEY:-}" ]] && [[ -f "${OBELYSK_DIR}/marketplace.env" ]]; then
        source "${OBELYSK_DIR}/marketplace.env"
        debug "Loaded marketplace credentials from cache"
    fi
}

# ─── Marketplace Auto-Registration ───────────────────────────────────
#
# Registers this machine with the BitSage marketplace for zero-config
# audit storage. Auto-provisions an org + API key on first run.
# Credentials are cached in ~/.obelysk/marketplace.env.

MARKETPLACE_URL="${MARKETPLACE_URL:-https://marketplace.bitsage.network}"

register_with_marketplace() {
    # Skip if already registered
    if [[ -n "${MARKETPLACE_API_KEY:-}" ]]; then
        debug "Marketplace already registered (key prefix: ${MARKETPLACE_API_KEY:0:12})"
        return 0
    fi

    # Generate a stable machine fingerprint
    local machine_id=""
    if command -v nvidia-smi &>/dev/null; then
        local gpu_uuid
        gpu_uuid=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        machine_id="$(hostname)-${gpu_uuid}"
    else
        machine_id="$(hostname)-nogpu-$(uname -m)"
    fi

    # Hash the machine ID for privacy
    local machine_hash
    if command -v sha256sum &>/dev/null; then
        machine_hash=$(echo -n "$machine_id" | sha256sum | cut -c1-32)
    elif command -v shasum &>/dev/null; then
        machine_hash=$(echo -n "$machine_id" | shasum -a 256 | cut -c1-32)
    else
        machine_hash=$(echo -n "$machine_id" | md5sum 2>/dev/null | cut -c1-32 || echo "$machine_id" | cut -c1-32)
    fi

    local gpu_model=""
    if command -v nvidia-smi &>/dev/null; then
        gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    fi

    # Prompt for email (optional) — links audits to a dashboard account
    local user_email="${MARKETPLACE_EMAIL:-}"
    if [[ -z "$user_email" ]] && [[ -t 0 ]]; then
        echo "" >&2
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
        echo -e "${BOLD}  BitSage Audit Dashboard${NC}" >&2
        echo -e "  View and manage your proof audit reports at:" >&2
        echo -e "  ${CYAN}https://marketplace.bitsage.network/storage${NC}" >&2
        echo "" >&2
        echo -e "  Enter your email to link this device to your account." >&2
        echo -e "  All future proofs will appear in your dashboard." >&2
        echo -e "  ${DIM}(Press Enter to skip — audits still stored on-chain + Arweave)${NC}" >&2
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
        read -rp "  Email: " user_email
        echo "" >&2
    fi

    log "Registering with marketplace (${MARKETPLACE_URL})..."

    local register_body
    if [[ -n "$user_email" ]]; then
        register_body=$(printf '{"machineId":"%s","gpuModel":"%s","hostname":"%s","email":"%s"}' \
            "$machine_hash" "${gpu_model:-unknown}" "$(hostname)" "$user_email")
    else
        register_body=$(printf '{"machineId":"%s","gpuModel":"%s","hostname":"%s"}' \
            "$machine_hash" "${gpu_model:-unknown}" "$(hostname)")
    fi

    local response
    if command -v curl &>/dev/null; then
        response=$(curl -sS --max-time 30 \
            -X POST "${MARKETPLACE_URL}/api/v1/pipeline/register" \
            -H "Content-Type: application/json" \
            -d "$register_body" 2>&1) || {
            warn "Marketplace registration failed (network error). Audit will use relay fallback."
            return 1
        }
    else
        warn "curl not found. Marketplace registration skipped."
        return 1
    fi

    # Extract fields from response
    local api_key="" status="" message=""
    status=$(echo "$response" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
    message=$(echo "$response" | grep -o '"message":"[^"]*"' | head -1 | cut -d'"' -f4)

    if [[ "$status" == "created" ]]; then
        api_key=$(echo "$response" | grep -o '"apiKey":"[^"]*"' | head -1 | cut -d'"' -f4)
        if [[ -n "$api_key" ]]; then
            # Save credentials
            cat > "${OBELYSK_DIR}/marketplace.env" << MKTEOF
# BitSage Marketplace credentials — auto-generated, do not edit
MARKETPLACE_URL="${MARKETPLACE_URL}"
MARKETPLACE_API_KEY="${api_key}"
MARKETPLACE_MACHINE_ID="${machine_hash}"
MARKETPLACE_EMAIL="${user_email}"
MARKETPLACE_REGISTERED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
MKTEOF
            chmod 600 "${OBELYSK_DIR}/marketplace.env"
            export MARKETPLACE_API_KEY="$api_key"
            export MARKETPLACE_URL
            ok "Marketplace registered."
            [[ -n "$message" ]] && log "$message"
            return 0
        fi
    elif [[ "$status" == "existing" ]]; then
        # Already registered — check if we have cached key
        if [[ -f "${OBELYSK_DIR}/marketplace.env" ]]; then
            source "${OBELYSK_DIR}/marketplace.env"
            ok "Marketplace: device already registered (cached key loaded)"
            return 0
        else
            warn "Marketplace: device registered but API key not cached. Re-register or set MARKETPLACE_API_KEY manually."
            return 1
        fi
    fi

    warn "Marketplace registration response unexpected. Audit will use relay fallback."
    debug "Response: ${response:0:500}"
    return 1
}

# ─── State Persistence ───────────────────────────────────────────────
#
# State files are simple KEY=VALUE files stored in ~/.obelysk/.
# Use save_state / load_state / get_state for inter-script communication.

save_state() {
    local file="${OBELYSK_DIR}/$1"
    shift
    mkdir -p "$(dirname "$file")"
    # Write all KEY=VALUE pairs
    for kv in "$@"; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        # Remove existing key if present, then append
        if [[ -f "$file" ]]; then
            sed -i.bak "/^${key}=/d" "$file" 2>/dev/null || true
            rm -f "${file}.bak"
        fi
        echo "${key}=${val}" >> "$file"
    done
    debug "State saved to ${file}"
}

load_state() {
    local file="${OBELYSK_DIR}/$1"
    if [[ -f "$file" ]]; then
        # Source the file but skip comments and empty lines
        while IFS='=' read -r key val; do
            [[ -z "$key" || "$key" == \#* ]] && continue
            export "${key}=${val}"
        done < "$file"
        debug "State loaded from ${file}"
        return 0
    fi
    return 1
}

get_state() {
    local file="${OBELYSK_DIR}/$1"
    local key="$2"
    if [[ -f "$file" ]]; then
        grep "^${key}=" "$file" 2>/dev/null | head -1 | cut -d'=' -f2-
    fi
}

# ─── Prerequisite Checks ─────────────────────────────────────────────

check_command() {
    local cmd="$1"
    local msg="${2:-$cmd is required but not found}"
    if ! command -v "$cmd" &>/dev/null; then
        err "$msg"
        return 1
    fi
    return 0
}

check_file() {
    local path="$1"
    local msg="${2:-File not found: $path}"
    if [[ ! -f "$path" ]]; then
        err "$msg"
        return 1
    fi
    return 0
}

check_dir() {
    local path="$1"
    local msg="${2:-Directory not found: $path}"
    if [[ ! -d "$path" ]]; then
        err "$msg"
        return 1
    fi
    return 0
}

# ─── JSON Utilities ──────────────────────────────────────────────────

parse_json_field() {
    local file="$1"
    local field="$2"
    python3 -c "
import json, sys
with open('${file}') as f:
    d = json.load(f)
keys = '${field}'.split('.')
v = d
for k in keys:
    if isinstance(v, dict):
        v = v.get(k)
    elif isinstance(v, list) and k.isdigit():
        v = v[int(k)]
    else:
        v = None
        break
if v is not None:
    print(v)
" 2>/dev/null
}

# ─── Timing ──────────────────────────────────────────────────────────

format_duration() {
    local seconds="$1"
    if (( seconds < 60 )); then
        echo "${seconds}s"
    elif (( seconds < 3600 )); then
        local m=$((seconds / 60))
        local s=$((seconds % 60))
        echo "${m}m ${s}s"
    else
        local h=$((seconds / 3600))
        local m=$(( (seconds % 3600) / 60 ))
        local s=$((seconds % 60))
        echo "${h}h ${m}m ${s}s"
    fi
}

# Store start times by label
declare -A _OBELYSK_TIMERS 2>/dev/null || true

timer_start() {
    local label="${1:-default}"
    _OBELYSK_TIMERS[$label]=$(date +%s)
}

timer_elapsed() {
    local label="${1:-default}"
    local start="${_OBELYSK_TIMERS[$label]:-$(date +%s)}"
    local now=$(date +%s)
    echo $(( now - start ))
}

timer_elapsed_fmt() {
    format_duration "$(timer_elapsed "${1:-default}")"
}

# ─── Dry Run Support ─────────────────────────────────────────────────

DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo -e "${DIM}[DRY RUN] $*${NC}" >&2
        return 0
    fi
    "$@"
}

# ─── Misc Helpers ────────────────────────────────────────────────────

# Find a binary in release dirs, checking multiple locations
find_binary() {
    local name="$1"
    local search_dir="${2:-.}"
    find "$search_dir" -name "$name" -path "*/release/*" -type f 2>/dev/null | head -1 || true
}

# Require minimum bash version (for associative arrays)
require_bash4() {
    if (( BASH_VERSINFO[0] < 4 )); then
        err "Bash 4+ required (have ${BASH_VERSION})"
        err "On macOS: brew install bash"
        return 1
    fi
}

# ─── Prove-Server Helpers ───────────────────────────────────────────
#
# curl-based helpers for submitting prove jobs to a remote prove-server.
# Used by 03_prove.sh when --server is specified.

submit_prove_job() {
    local server_url="$1"
    local model_id="$2"
    local gpu="${3:-true}"
    local security="${4:-auto}"

    local body
    body=$(cat <<PROVEJSONEOF
{
    "model_id": "${model_id}",
    "gpu": ${gpu},
    "security": "${security}"
}
PROVEJSONEOF
)

    local response
    response=$(curl -s -w "\n%{http_code}" \
        -X POST "${server_url}/api/v1/prove" \
        -H "Content-Type: application/json" \
        -d "$body")

    local http_code
    http_code=$(echo "$response" | tail -1)
    local body_text
    body_text=$(echo "$response" | sed '$d')

    if [[ "$http_code" != "202" ]] && [[ "$http_code" != "200" ]]; then
        err "Failed to submit prove job (HTTP ${http_code}): ${body_text}"
        return 1
    fi

    # Extract job_id from JSON response
    local job_id
    job_id=$(echo "$body_text" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])" 2>/dev/null)
    echo "$job_id"
}

poll_prove_job() {
    local server_url="$1"
    local job_id="$2"
    local poll_interval="${3:-5}"
    local timeout="${4:-3600}"

    local start_time
    start_time=$(date +%s)

    while true; do
        local now
        now=$(date +%s)
        local elapsed=$(( now - start_time ))

        if (( elapsed > timeout )); then
            err "Prove job timed out after ${timeout}s"
            return 1
        fi

        local response
        response=$(curl -s "${server_url}/api/v1/prove/${job_id}" 2>/dev/null)

        local status
        status=$(echo "$response" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
        local progress
        progress=$(echo "$response" | python3 -c "import json,sys; print(json.load(sys.stdin).get('progress_bps',0))" 2>/dev/null)

        local pct=$(( progress / 100 ))
        log "Job ${job_id}: ${status} (${pct}%, ${elapsed}s elapsed)"

        case "$status" in
            completed) echo "completed"; return 0 ;;
            failed)    err "Prove job failed"; echo "failed"; return 1 ;;
            *) sleep "$poll_interval" ;;
        esac
    done
}

fetch_prove_result() {
    local server_url="$1"
    local job_id="$2"
    local output_file="$3"

    local response
    response=$(curl -s "${server_url}/api/v1/prove/${job_id}/result")

    if [[ -n "$output_file" ]]; then
        echo "$response" > "$output_file"
        ok "Proof result saved to ${output_file}"
    else
        echo "$response"
    fi
}
