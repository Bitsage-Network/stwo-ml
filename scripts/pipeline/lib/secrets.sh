#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Encrypted Secrets Management
# ═══════════════════════════════════════════════════════════════════════
#
# Source this after common.sh:
#   source "${SCRIPT_DIR}/lib/secrets.sh"
#
# Provides:
#   - load_secrets       → auto-decrypt and export HF_TOKEN, IRYS_TOKEN, etc.
#   - encrypt_secrets    → encrypt a .env file for committing to repo
#   - decrypt_secrets    → decrypt to stdout (for debugging)
#
# Encryption: AES-256-CBC via openssl (available on all Linux/macOS)
#
# Usage:
#   1. Team member creates secrets:
#        ./manage_secrets.sh --encrypt
#   2. Commits configs/.secrets.env.enc to repo
#   3. Auditor runs pipeline with OBELYSK_SECRETS_KEY env var:
#        OBELYSK_SECRETS_KEY="the-passphrase" ./run_e2e.sh --preset qwen3-14b --gpu --submit
#   4. Or interactively prompted if key not set

[[ -n "${_OBELYSK_SECRETS_LOADED:-}" ]] && return 0
_OBELYSK_SECRETS_LOADED=1

# Encrypted secrets file (committed to repo)
_SECRETS_ENC="${SCRIPT_DIR:-.}/configs/.secrets.env.enc"

# Decrypted cache (never committed, lives in ~/.obelysk)
_SECRETS_CACHE="${OBELYSK_DIR:-$HOME/.obelysk}/secrets.env"

# ─── Decrypt secrets file ──────────────────────────────────────────

_decrypt_secrets_file() {
    local enc_file="$1"
    local passphrase="$2"

    openssl enc -aes-256-cbc -pbkdf2 -d \
        -in "$enc_file" \
        -pass "pass:${passphrase}" 2>/dev/null
}

# ─── Load secrets into environment ─────────────────────────────────
#
# Priority:
#   1. Already-set env vars (explicit user override always wins)
#   2. Decrypted cache in ~/.obelysk/secrets.env (from prior run)
#   3. Decrypt from configs/.secrets.env.enc using OBELYSK_SECRETS_KEY
#   4. Interactive prompt for passphrase (if tty available)
#   5. Skip silently (tokens only needed for specific operations)

load_secrets() {
    # Track which tokens we still need
    local need_hf=false
    local need_irys=false

    [[ -z "${HF_TOKEN:-}" ]] && need_hf=true
    [[ -z "${IRYS_TOKEN:-}" ]] && need_irys=true

    # Nothing to do if all tokens already set
    if [[ "$need_hf" == "false" ]] && [[ "$need_irys" == "false" ]]; then
        debug "All secrets already set via environment"
        return 0
    fi

    # Try cached decrypted file first
    if [[ -f "$_SECRETS_CACHE" ]]; then
        _source_secrets_file "$_SECRETS_CACHE"
        if [[ "$need_hf" == "false" || -n "${HF_TOKEN:-}" ]] && \
           [[ "$need_irys" == "false" || -n "${IRYS_TOKEN:-}" ]]; then
            debug "Secrets loaded from cache"
            return 0
        fi
    fi

    # No encrypted file → nothing to decrypt
    if [[ ! -f "$_SECRETS_ENC" ]]; then
        debug "No encrypted secrets file found at ${_SECRETS_ENC}"
        return 0
    fi

    # Get passphrase
    local passphrase="${OBELYSK_SECRETS_KEY:-}"

    if [[ -z "$passphrase" ]]; then
        # Try interactive prompt if we have a terminal
        if [[ -t 0 ]]; then
            echo -e "${YELLOW}[KEYS]${NC} Encrypted secrets found. Enter passphrase to unlock:" >&2
            echo -e "${DIM}       (or set OBELYSK_SECRETS_KEY env var to skip prompt)${NC}" >&2
            read -r -s -p "       Passphrase: " passphrase
            echo "" >&2
        fi
    fi

    if [[ -z "$passphrase" ]]; then
        debug "No passphrase available — skipping secrets decryption"
        return 0
    fi

    # Decrypt
    local decrypted
    decrypted=$(_decrypt_secrets_file "$_SECRETS_ENC" "$passphrase") || {
        warn "Failed to decrypt secrets (wrong passphrase?)"
        return 0
    }

    if [[ -z "$decrypted" ]]; then
        warn "Decrypted secrets file is empty"
        return 0
    fi

    # Cache for this session (permissions: owner-only)
    mkdir -p "$(dirname "$_SECRETS_CACHE")"
    echo "$decrypted" > "$_SECRETS_CACHE"
    chmod 600 "$_SECRETS_CACHE"

    _source_secrets_file "$_SECRETS_CACHE"

    # Report what was loaded
    local loaded=()
    [[ "$need_hf" == "true" ]] && [[ -n "${HF_TOKEN:-}" ]] && loaded+=("HF_TOKEN")
    [[ "$need_irys" == "true" ]] && [[ -n "${IRYS_TOKEN:-}" ]] && loaded+=("IRYS_TOKEN")
    [[ -n "${STARKNET_PRIVATE_KEY:-}" ]] && loaded+=("STARKNET_PRIVATE_KEY")
    [[ -n "${ALCHEMY_KEY:-}" ]] && loaded+=("ALCHEMY_KEY")

    if [[ ${#loaded[@]} -gt 0 ]]; then
        ok "Secrets loaded: ${loaded[*]}"
    fi

    return 0
}

# Source a secrets .env file, only filling unset vars
_source_secrets_file() {
    local file="$1"
    while IFS='=' read -r key val; do
        # Skip comments, empty lines
        [[ -z "$key" || "$key" == \#* ]] && continue
        # Trim whitespace
        key="${key// /}"
        # Only set if not already in environment
        if [[ -z "${!key:-}" ]]; then
            export "${key}=${val}"
        fi
    done < "$file"
}

# ─── Encrypt a secrets file ────────────────────────────────────────
#
# Usage: encrypt_secrets <input.env> [passphrase]
# Output: writes to configs/.secrets.env.enc

encrypt_secrets() {
    local input_file="$1"
    local passphrase="${2:-}"
    local output_file="${3:-${_SECRETS_ENC}}"

    if [[ ! -f "$input_file" ]]; then
        err "Input file not found: ${input_file}"
        return 1
    fi

    if [[ -z "$passphrase" ]]; then
        read -r -s -p "Enter passphrase for encryption: " passphrase
        echo ""
        local confirm
        read -r -s -p "Confirm passphrase: " confirm
        echo ""
        if [[ "$passphrase" != "$confirm" ]]; then
            err "Passphrases do not match"
            return 1
        fi
    fi

    mkdir -p "$(dirname "$output_file")"

    openssl enc -aes-256-cbc -pbkdf2 \
        -in "$input_file" \
        -out "$output_file" \
        -pass "pass:${passphrase}" || {
        err "Encryption failed"
        return 1
    }

    ok "Secrets encrypted to ${output_file}"
    log "Add to repo: git add ${output_file}"
}

# ─── Decrypt to stdout (debugging) ─────────────────────────────────

decrypt_secrets() {
    local passphrase="${1:-${OBELYSK_SECRETS_KEY:-}}"

    if [[ ! -f "$_SECRETS_ENC" ]]; then
        err "No encrypted secrets file at ${_SECRETS_ENC}"
        return 1
    fi

    if [[ -z "$passphrase" ]]; then
        read -r -s -p "Passphrase: " passphrase
        echo "" >&2
    fi

    _decrypt_secrets_file "$_SECRETS_ENC" "$passphrase"
}

# ─── Clean up cached secrets ───────────────────────────────────────

clean_secrets() {
    if [[ -f "$_SECRETS_CACHE" ]]; then
        rm -f "$_SECRETS_CACHE"
        ok "Cached secrets removed"
    fi
}
