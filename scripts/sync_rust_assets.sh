#!/usr/bin/env bash
#
# Copies the files the Rust crate embeds from the Python package into rust/assets/.
#
# Why a copy exists at all: `cargo package` only ever includes files under the crate
# directory. It silently ignores an `include` entry pointing at `../`, and a symlink is not
# portable (git's core.symlinks is false on Windows by default). So a published crate cannot
# read ../pocket_tts/, and the files have to live inside rust/.
#
# pocket_tts/ stays canonical. This script is the only way the copy should ever be updated,
# and `cargo test` fails when the two drift apart — see rust/tests/embedded_assets.rs.
#
# Run it from anywhere; it locates the repository itself. It is idempotent, so "is the copy
# current?" is answered by running it and checking whether git reports anything changed.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
src_config="$repo_root/pocket_tts/config"
src_index="$repo_root/pocket_tts/static/index.html"
src_license="$repo_root/LICENSE"
dst="$repo_root/rust/assets"

for path in "$src_config" "$src_index" "$src_license"; do
    if [ ! -e "$path" ]; then
        echo "error: $path is missing — run this from a full checkout of the repository." >&2
        exit 1
    fi
done

mkdir -p "$dst/config"

# Delete first, so a config removed upstream is removed here too rather than lingering.
rm -f "$dst/config"/*.yaml
cp "$src_config"/*.yaml "$dst/config/"
cp "$src_index" "$dst/index.html"
cp "$src_license" "$repo_root/rust/LICENSE"

# Which Python release these copies came from, read back by `PYTHON_PACKAGE_VERSION` in
# rust/src/lib.rs. Recorded as a fact rather than left to a comment, because the comment that
# claimed the two versions were in step was wrong for several releases. No trailing newline:
# the constant is `trim_ascii`d, but keeping the file exact makes the diff obvious.
python_version=$(grep -m 1 '^version' "$repo_root/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
if [ -z "$python_version" ]; then
    echo "error: cannot read the version from pyproject.toml" >&2
    exit 1
fi
printf '%s' "$python_version" > "$dst/python-version"

count=$(find "$dst/config" -name '*.yaml' | wc -l | tr -d ' ')
echo "Synced $count config(s), index.html and LICENSE from pocket_tts $python_version."

if command -v git >/dev/null 2>&1 && git -C "$repo_root" rev-parse --git-dir >/dev/null 2>&1; then
    # Note this reports untracked paths too, so it says "differs from HEAD" rather than
    # "was out of date": before the first commit of rust/assets/ both look the same here.
    changed=$(git -C "$repo_root" status --porcelain -- rust/assets rust/LICENSE)
    if [ -n "$changed" ]; then
        echo
        echo "These now differ from HEAD and need committing:"
        echo "$changed"
    else
        echo "Identical to HEAD; nothing to commit."
    fi
fi
