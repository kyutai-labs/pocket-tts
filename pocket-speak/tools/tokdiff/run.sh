#!/usr/bin/env bash
# Diff sentencepiece-rs against Google's sentencepiece on the pinned model.
# Usage: run.sh WORK_DIR SOURCE [SOURCE ...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$1"; shift
mkdir -p "$work"
base=https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/58a6d00cf13d239b6748cb0769f35c580a8f606c/onnx/english_2026-04
fetch() { # name sha256
  [ -f "$work/$1" ] || curl -sSfL -o "$work/$1" "$base/$1"
  echo "$2  $work/$1" | sha256sum -c --quiet
}
fetch tokenizer.model d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6
fetch bundle.json bab643150f437f37df080a710520ff39ed9ebd9a339f8ebdc739f7eddfc28b3f
py=(uv run --no-project --with sentencepiece==0.2.2 python)
(cd "$here/../.." && cargo build -q --release -p tokdiff)
bin="$here/../../target/release/tokdiff"
python3 "$here/scripts/build_corpus.py" "$work/bundle.json" "$work" "$@"
python3 "$here/scripts/fuzz_corpus.py" "$work/fuzz.jsonl" 5000 20260913
status=0
for set in corpus fuzz; do
  "${py[@]}" "$here/scripts/google_ids.py" "$work/tokenizer.model" "$work/$set.jsonl" "$work/$set.google.ids"
  "$bin" "$work/tokenizer.model" < "$work/$set.jsonl" > "$work/$set.rust.ids"
  echo "== $set"
  "${py[@]}" "$here/scripts/compare.py" "$work/tokenizer.model" "$work/$set.jsonl" "$work/$set.google.ids" "$work/$set.rust.ids" || status=1
done
exit $status
