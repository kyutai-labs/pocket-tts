# shellcheck shell=bash
# Shared by the fork's scripts. Sourced, not run.

UPSTREAM_URL="https://github.com/kyutai-labs/pocket-tts"
UPSTREAM_REF="refs/remotes/upstream/main"
FORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fetches upstream's main into $UPSTREAM_REF, with enough history for a merge
# base with HEAD.
fetch_upstream() {
    git fetch --quiet --no-tags "$UPSTREAM_URL" "+main:$UPSTREAM_REF"
}

# The lines of a list file, without comments or blank lines.
list_entries() {
    sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$1" | grep -v '^$' || true
}

# Exclusion pathspecs for every fork-owned path.
owned_excludes() {
    list_entries "$FORK_DIR/owned-paths" | sed 's/^/:!/'
}
