#!/usr/bin/env bash
set -euo pipefail

# Reap stale Ralph lock refs on origin.
#
# Locks are remote refs created by scripts/ralph-lock-acquire.sh:
#   refs/ralph-locks/issue-<num>
#
# The ref points to a synthetic commit whose commit timestamp is the lock acquisition time.
#
# Usage:
#   ./scripts/ralph-lock-reap.sh [--older-than-hours N] [--remote origin] [--dry-run]
#
# Defaults:
#   --older-than-hours 24
#   --remote origin

OLDER_HOURS=24
REMOTE="origin"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --older-than-hours)
      OLDER_HOURS="$2"; shift 2;;
    --remote)
      REMOTE="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift 1;;
    -h|--help)
      echo "usage: $0 [--older-than-hours N] [--remote origin] [--dry-run]"
      exit 0;;
    *)
      echo "unknown arg: $1" 1>&2
      exit 2;;
  esac
done

THRESHOLD_SECS=$((OLDER_HOURS * 3600))
NOW_SECS="$(date -u +%s)"

# Fetch lock refs locally so we can read their commit timestamps.
git fetch "$REMOTE" "refs/ralph-locks/*:refs/ralph-locks/*" >/dev/null 2>&1 || true

mapfile -t LINES < <(git ls-remote --refs "$REMOTE" "refs/ralph-locks/*" || true)
if [[ ${#LINES[@]} -eq 0 ]]; then
  echo "no remote locks found"
  exit 0
fi

reaped=0
kept=0

for line in "${LINES[@]}"; do
  sha="$(awk '{print $1}' <<<"$line")"
  ref="$(awk '{print $2}' <<<"$line")"

  # If we can't read timestamp, skip safely.
  if ! ts="$(git show -s --format=%ct "$sha" 2>/dev/null)"; then
    echo "skip (cannot read timestamp): $ref $sha"
    kept=$((kept+1))
    continue
  fi

  age=$((NOW_SECS - ts))
  if [[ $age -ge $THRESHOLD_SECS ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "DRY-RUN reap: $ref age=${age}s"
    else
      git push "$REMOTE" ":$ref" >/dev/null 2>&1 || true
      git update-ref -d "$ref" >/dev/null 2>&1 || true
      echo "reaped: $ref age=${age}s"
    fi
    reaped=$((reaped+1))
  else
    echo "kept:  $ref age=${age}s"
    kept=$((kept+1))
  fi
done

echo "done: reaped=$reaped kept=$kept older_than_hours=$OLDER_HOURS remote=$REMOTE dry_run=$DRY_RUN"
