#!/usr/bin/env bash
set -euo pipefail

# Apply a downloaded Ralph package zip into the current repository root.
#
# This script:
# - extracts the zip to a temp directory
# - copies known Ralph-managed paths into the current repo
# - creates a backup of any overwritten files under .ralph/backup/<timestamp>/
#
# Usage:
#   ./scripts/ralph-apply-package.sh /path/to/github-ralph-loop-v8.zip
#
# Notes:
# - Run from the target repo root.
# - This does NOT run any workflows; it only installs/updates files.

ZIP_PATH="${1:-}"
[[ -n "$ZIP_PATH" ]] || { echo "usage: $0 <zip-path>" 1>&2; exit 2; }
[[ -f "$ZIP_PATH" ]] || { echo "error: zip not found: $ZIP_PATH" 1>&2; exit 2; }

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || { echo "error: not in a git repo" 1>&2; exit 2; }
cd "$ROOT"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR=".ralph/backup/$TS"
mkdir -p "$BACKUP_DIR"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

unzip -q "$ZIP_PATH" -d "$TMP_DIR"

# Paths we manage (copy from package -> repo)
MANAGED_PATHS=(
  ".windsurf"
  ".agent"
  ".claude"
  ".opencode"
  "scripts"
  "docs/adr"
  "docs/projects-v2"
  "docs/claude-code"
  "docs/opencode"
  "AGENTS.md"
  "CLAUDE.md"
)

backup_one() {
  local p="$1"
  if [[ -e "$p" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$p")"
    cp -a "$p" "$BACKUP_DIR/$p"
  fi
}

copy_one() {
  local p="$1"
  local src="$TMP_DIR/$p"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$p")"
    # backup existing
    backup_one "$p"
    rm -rf "$p"
    cp -a "$src" "$p"
  fi
}

for p in "${MANAGED_PATHS[@]}"; do
  copy_one "$p"
done

echo "installed from: $ZIP_PATH"
echo "backup saved to: $BACKUP_DIR"
