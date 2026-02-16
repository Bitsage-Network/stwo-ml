#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/pipeline/lib/$(basename "${BASH_SOURCE[0]}")"

if [[ ! -f "$TARGET" ]]; then
  echo "[ERR] Canonical pipeline lib not found: $TARGET" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # Support sourcing: expose canonical functions to current shell.
  source "$TARGET"
else
  exec bash "$TARGET" "$@"
fi
