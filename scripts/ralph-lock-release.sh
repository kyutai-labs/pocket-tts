#!/usr/bin/env bash
set -euo pipefail

ISSUE="${1:-}"
REMOTE="${2:-origin}"
[[ -n "$ISSUE" ]] || { echo "usage: $0 <issue-number> [remote]" 1>&2; exit 2; }

LOCK_REF="refs/ralph-locks/issue-${ISSUE}"

git push "$REMOTE" ":$LOCK_REF" >/dev/null 2>&1 || true
git update-ref -d "$LOCK_REF" >/dev/null 2>&1 || true
