#!/usr/bin/env bash
set -euo pipefail

# Acquire a remote lock ref for a GitHub issue to prevent claim races.
# Lock ref: refs/ralph-locks/issue-<num>
#
# This script creates a synthetic commit object whose commit timestamp reflects the lock acquisition time.
# The commit is created without changing HEAD or the working tree (via `git commit-tree`).
#
# Usage:
#   ./scripts/ralph-lock-acquire.sh <issue-number> [remote]
#
# Exit codes:
#   0: lock acquired; prints lock ref
#   1: lock busy (already exists remotely)
#   2: usage error

ISSUE="${1:-}"
REMOTE="${2:-origin}"
[[ -n "$ISSUE" ]] || { echo "usage: $0 <issue-number> [remote]" 1>&2; exit 2; }

LOCK_REF="refs/ralph-locks/issue-${ISSUE}"

# Build a synthetic commit to encode acquisition time (without touching HEAD).
TREE="$(git rev-parse HEAD^{tree})"
PARENT="$(git rev-parse HEAD)"
MSG="ralph-lock issue=${ISSUE} acquired_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Ensure commit timestamp is 'now' (UTC).
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-ralph-lock}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-ralph-lock@local}"
export GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-ralph-lock}"
export GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-ralph-lock@local}"
export GIT_AUTHOR_DATE="$(date -u +%s)"
export GIT_COMMITTER_DATE="$GIT_AUTHOR_DATE"

COMMIT="$(printf "%s" "$MSG" | git commit-tree "$TREE" -p "$PARENT")"

# Point local lock ref at synthetic commit
git update-ref "$LOCK_REF" "$COMMIT"

# Attempt to push lock ref; if it exists, this will fail.
if git push "$REMOTE" "$LOCK_REF" >/dev/null; then
  echo "$LOCK_REF"
  exit 0
else
  git update-ref -d "$LOCK_REF" >/dev/null 2>&1 || true
  exit 1
fi
