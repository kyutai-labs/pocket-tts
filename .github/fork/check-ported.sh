#!/usr/bin/env bash
# Fails when upstream changed a file the fork ports, or moved the model's pin.
#
# The engine compiles most of upstream's rust/src in unchanged, so a merge
# updates it. Two files it ports instead (see .github/fork/ported), and one
# dependency it must share with upstream's CLI: ptts, the model, whose tensors
# cross between the two. Neither follows a merge on its own; this is what says
# so, on the PR that brings the change.
#
#   check-ported.sh             against the upstream files in this tree
#   check-ported.sh --upstream  against upstream's main, for the upstream watch
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

ref=HEAD
if [[ "${1:-}" == "--upstream" ]]; then
    fetch_upstream
    ref="$UPSTREAM_REF"
fi

status=0
while read -r path recorded port; do
    current=$(git rev-parse "$ref:$path")
    if [[ "$current" != "$recorded" ]]; then
        echo "$path changed since $port was ported from it. Review with"
        echo "    git diff $recorded $current"
        echo "  port what matters, then record $current in .github/fork/ported."
        status=1
    fi
done < <(list_entries "$FORK_DIR/ported")

pin() { grep -o 'xn-ptts", rev = "[0-9a-f]*"' | grep -o '[0-9a-f]\{40\}'; }
upstream_pin=$(git show "$ref:rust/Cargo.toml" | pin)
fork_pin=$(pin < "$FORK_DIR/../../pocket-speak/crates/engine/Cargo.toml")
if [[ "$upstream_pin" != "$fork_pin" ]]; then
    echo "Upstream's CLI pins ptts at $upstream_pin, the engine at $fork_pin."
    echo "  Set pocket-speak/crates/engine/Cargo.toml to $upstream_pin."
    status=1
fi

[[ $status == 0 ]] && echo "The engine follows upstream's model code and ptts pin."
exit $status
