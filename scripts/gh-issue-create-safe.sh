#!/usr/bin/env bash
set -euo pipefail

# Create a GitHub issue with a body file to avoid shell escaping problems.
#
# Usage:
#   ./scripts/gh-issue-create-safe.sh "<title>" <body-file> [label ...]

TITLE="${1:-}"
BODY_FILE="${2:-}"
shift 2 || true

[[ -n "$TITLE" && -n "$BODY_FILE" ]] || { echo "usage: $0 <title> <body-file> [label ...]" 1>&2; exit 2; }
[[ -f "$BODY_FILE" ]] || { echo "error: body file not found: $BODY_FILE" 1>&2; exit 2; }

args=(gh issue create --title "$TITLE" --body-file "$BODY_FILE")

for lbl in "$@"; do
  args+=(--label "$lbl")
done

"${args[@]}"
