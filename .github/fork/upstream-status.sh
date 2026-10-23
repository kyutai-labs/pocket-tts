#!/usr/bin/env bash
# Reports what merging upstream's main would do, as Markdown on stdout.
#
# Exit status: 0 nothing to merge, 10 merges cleanly, 11 conflicts. The
# working tree is left as it was: the merge is tried in a throwaway worktree.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

fetch_upstream
behind=$(git rev-list --count "HEAD..$UPSTREAM_REF")
if [[ "$behind" == 0 ]]; then
    echo "The fork has every commit of upstream's main."
    exit 0
fi

tree=$(mktemp -d)
trap 'git worktree remove --force "$tree" >/dev/null 2>&1 || true' EXIT
git worktree add --quiet --detach "$tree" HEAD
conflicts=""
if ! git -C "$tree" -c user.name=fork -c user.email=fork@localhost \
        merge --quiet --no-commit --no-ff "$UPSTREAM_REF" >/dev/null 2>&1; then
    conflicts=$(git -C "$tree" diff --name-only --diff-filter=U)
fi

upstream_sha=$(git rev-parse --short "$UPSTREAM_REF")
echo "Upstream's main ($UPSTREAM_URL/commit/$upstream_sha) has **$behind** commits the fork does not."
echo
if [[ -z "$conflicts" ]]; then
    echo "They merge **cleanly**."
else
    echo "Merging them **conflicts** in:"
    echo
    # The backticks are Markdown, not command substitution.
    # shellcheck disable=SC2016
    printf '%s\n' "$conflicts" | sed 's/.*/- `&`/'
fi
echo
echo "To merge, on a branch, then open a PR and merge it with a merge commit (not squash):"
echo
echo '```'
echo "git fetch $UPSTREAM_URL main"
echo 'git merge --no-ff FETCH_HEAD'
echo '```'
echo
echo "<details><summary>The commits</summary>"
echo
# Limited by git, not by `| head`: head closing the pipe early kills git log
# with SIGPIPE, which pipefail turns into this script failing.
git log --no-merges --max-count=100 --format='- %s (%h)' "HEAD..$UPSTREAM_REF"
shown=$(git rev-list --no-merges --count --max-count=100 "HEAD..$UPSTREAM_REF")
total=$(git rev-list --no-merges --count "HEAD..$UPSTREAM_REF")
if (( total > shown )); then
    echo "- … and $((total - shown)) more"
fi
echo
echo "</details>"
[[ -z "$conflicts" ]] && exit 10 || exit 11
