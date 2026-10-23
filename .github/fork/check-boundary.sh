#!/usr/bin/env bash
# Fails when the fork changes a file upstream owns.
#
# The fork's changes are `git diff upstream...HEAD`: everything since the last
# upstream commit merged in. Outside the fork-owned paths that must be empty,
# apart from the deliberate edits listed in upstream-edits. An upstream file
# edited here is a conflict at the next merge that touches it; the fix is
# almost always a new file under android/ or pocket-speak/ instead.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

fetch_upstream
mapfile -t excludes < <(owned_excludes)
mapfile -t allowed < <(list_entries "$FORK_DIR/upstream-edits")

changed=$(git diff --name-only "$UPSTREAM_REF...HEAD" -- . "${excludes[@]}")
violations=$(comm -23 <(printf '%s\n' "$changed" | sed '/^$/d' | sort) \
                      <(printf '%s\n' "${allowed[@]}" | sed '/^$/d' | sort))

if [[ -n "$violations" ]]; then
    echo "These upstream-owned files differ from kyutai-labs/pocket-tts:"
    printf '%s\n' "$violations" | sed 's/^/  /'
    echo
    echo "Keep fork changes under the paths in .github/fork/owned-paths, or, if"
    echo "an upstream file truly has to change, list it with a reason in"
    echo ".github/fork/upstream-edits. See FORK.md."
    exit 1
fi
echo "Outside the fork's own paths, the tree matches upstream."
