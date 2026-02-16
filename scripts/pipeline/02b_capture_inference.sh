#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/pipeline/$(basename "$0")"
if [[ ! -f "$TARGET" ]]; then
  echo "[ERR] Canonical pipeline script not found: $TARGET" >&2
  exit 1
fi
exec bash "$TARGET" "$@"
