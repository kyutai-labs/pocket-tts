#!/usr/bin/env bash
# Measures the English models with pocket-bench, as a Markdown table on stdout.
# The other languages' models have the largest English model's shape, and its speed.
#
#   pocket-speak/tools/bench/run.sh MODELS_DIR [THREADS ...]
#
# MODELS_DIR as fetch-models.py fills it. Threads default to 1 and 2: the app's
# default is 2. Speed is seconds of audio per second of generation, summed over
# the four chunks of bench/reply.txt; first audio is the first chunk's.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
models_dir="${1:?usage: run.sh MODELS_DIR [THREADS ...]}"
shift
threads=("${@:-1 2}")
read -r -a threads <<< "${threads[*]}"

(cd "$here/../.." && cargo build --quiet --release -p pocket-tts-android --bin pocket-bench)
bench="$here/../../target/release/pocket-bench"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

echo "| model | weights | threads | load | first audio | speed |"
echo "|---|---|---|---|---|---|"
for model in $(python3 -c "import json,sys; print(' '.join(m['name'] for m in json.load(open(sys.argv[1]))['models'] if m['language'] == 'en'))" "$here/../../models.json"); do
    for t in "${threads[@]}"; do
        "$bench" "$model" "$models_dir/$model" "$models_dir/$model/embeddings/alba.safetensors" \
            "$out/$model-$t" 1 "$t" < "$here/reply.txt" > "$out/$model-$t.jsonl"
        python3 - "$out/$model-$t.jsonl" "$t" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
head, chunks = rows[0], rows[1:]
audio = sum(c["audio_ms"] for c in chunks)
generation = sum(c["generation_ms"] for c in chunks)
weights = "int8" if head["int8"] else "float"
print(f"| {head['model']} | {weights} | {sys.argv[2]} | {head['load_ms']} ms "
      f"| {chunks[0]['first_audio_ms']} ms | {audio / max(generation, 1):.1f}x |")
PY
    done
done
