#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# ObelyZK — Connect to H100 Developer Environment
#
# One command: curl -sL https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/development/engine/scripts/connect.sh | bash
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

HOST="62.169.159.231"
USER="obelyzk"
KEY_URL="https://raw.githubusercontent.com/Bitsage-Network/obelyzk.rs/development/engine/scripts/obelyzk_dev_key"
KEY_PATH="$HOME/.ssh/obelyzk_dev_key"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║  ObelyZK — H100 Developer Access         ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# Download key if not present
if [ ! -f "$KEY_PATH" ]; then
    echo "  Downloading SSH key..."
    mkdir -p "$HOME/.ssh"
    curl -sL "$KEY_URL" -o "$KEY_PATH"
    chmod 600 "$KEY_PATH"
    echo "  Key saved to $KEY_PATH"
fi

echo "  Connecting to $USER@$HOST (NVIDIA H100 PCIe)..."
echo ""

ssh -i "$KEY_PATH" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    "$USER@$HOST"
