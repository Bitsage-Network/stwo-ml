#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Obelysk Pipeline — Secrets Management
# ═══════════════════════════════════════════════════════════════════════
#
# Encrypt/decrypt pipeline secrets (HF_TOKEN, IRYS_TOKEN, etc.)
#
# Usage:
#   ./manage_secrets.sh --encrypt              # Interactive: prompts for tokens + passphrase
#   ./manage_secrets.sh --encrypt --from .env  # Encrypt existing .env file
#   ./manage_secrets.sh --decrypt              # Print decrypted secrets (for debugging)
#   ./manage_secrets.sh --clean                # Remove cached decrypted secrets
#   ./manage_secrets.sh --rotate               # Re-encrypt with new passphrase
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/secrets.sh"

ACTION=""
FROM_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --encrypt)  ACTION="encrypt"; shift ;;
        --decrypt)  ACTION="decrypt"; shift ;;
        --clean)    ACTION="clean"; shift ;;
        --rotate)   ACTION="rotate"; shift ;;
        --from)     FROM_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [ACTION] [OPTIONS]"
            echo ""
            echo "Manage encrypted pipeline secrets."
            echo ""
            echo "Actions:"
            echo "  --encrypt          Create encrypted secrets file (interactive or from file)"
            echo "  --decrypt          Print decrypted secrets to stdout"
            echo "  --clean            Remove cached decrypted secrets from ~/.obelysk"
            echo "  --rotate           Re-encrypt with a new passphrase"
            echo ""
            echo "Options:"
            echo "  --from FILE        Encrypt an existing .env file instead of prompting"
            echo ""
            echo "Environment:"
            echo "  OBELYSK_SECRETS_KEY   Passphrase (avoids interactive prompt)"
            echo ""
            echo "The encrypted file is stored at configs/.secrets.env.enc"
            echo "and is safe to commit to the repository."
            echo ""
            echo "Supported secrets:"
            echo "  HF_TOKEN               HuggingFace API token"
            echo "  IRYS_TOKEN             Irys/Arweave upload token"
            echo "  STARKNET_PRIVATE_KEY   Starknet account private key"
            echo "  ALCHEMY_KEY            Alchemy RPC API key"
            echo ""
            echo "Examples:"
            echo "  $0 --encrypt                    # Interactive prompts"
            echo "  $0 --encrypt --from .env        # From existing file"
            echo "  $0 --decrypt                    # View current secrets"
            echo "  $0 --rotate                     # Change passphrase"
            exit 0
            ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    err "No action specified. Use --encrypt, --decrypt, --clean, or --rotate."
    exit 1
fi

case "$ACTION" in
    encrypt)
        if [[ -n "$FROM_FILE" ]]; then
            # Encrypt from existing file
            encrypt_secrets "$FROM_FILE"
        else
            # Interactive: prompt for each token
            header "Create Encrypted Secrets"
            echo ""
            log "Enter values for each secret (leave blank to skip):"
            echo ""

            TMPFILE=$(mktemp)
            trap 'rm -f "$TMPFILE"' EXIT

            echo "# Obelysk pipeline secrets (encrypted at $(date -u +%Y-%m-%dT%H:%M:%SZ))" > "$TMPFILE"

            read -r -p "  HF_TOKEN (HuggingFace): " val
            [[ -n "$val" ]] && echo "HF_TOKEN=${val}" >> "$TMPFILE"

            read -r -p "  IRYS_TOKEN (Arweave): " val
            [[ -n "$val" ]] && echo "IRYS_TOKEN=${val}" >> "$TMPFILE"

            read -r -s -p "  STARKNET_PRIVATE_KEY: " val
            echo ""
            [[ -n "$val" ]] && echo "STARKNET_PRIVATE_KEY=${val}" >> "$TMPFILE"

            read -r -p "  ALCHEMY_KEY (optional): " val
            [[ -n "$val" ]] && echo "ALCHEMY_KEY=${val}" >> "$TMPFILE"

            echo ""

            # Count how many secrets were entered
            local_count=$(grep -c '=' "$TMPFILE" 2>/dev/null || echo "0")
            if [[ "$local_count" -le 1 ]]; then
                warn "No secrets entered"
                exit 0
            fi

            encrypt_secrets "$TMPFILE"
            ok "Encrypted ${local_count} secret(s) to configs/.secrets.env.enc"
            log "Commit: git add configs/.secrets.env.enc"
            log "Share the passphrase with your auditor separately."
        fi
        ;;

    decrypt)
        header "Decrypted Secrets"
        echo ""
        decrypt_secrets || exit 1
        ;;

    clean)
        init_obelysk_dir
        clean_secrets
        ;;

    rotate)
        header "Rotate Secrets Passphrase"
        echo ""

        # Decrypt with old passphrase
        log "Enter CURRENT passphrase:"
        read -r -s -p "  Current: " old_pass
        echo ""

        decrypted=$(_decrypt_secrets_file "$_SECRETS_ENC" "$old_pass") || {
            err "Failed to decrypt with current passphrase"
            exit 1
        }

        # Write to temp, re-encrypt with new passphrase
        TMPFILE=$(mktemp)
        trap 'rm -f "$TMPFILE"' EXIT
        echo "$decrypted" > "$TMPFILE"

        log "Enter NEW passphrase:"
        encrypt_secrets "$TMPFILE"

        # Clear cached version
        init_obelysk_dir
        clean_secrets

        ok "Passphrase rotated. Share new passphrase with your team."
        ;;
esac
