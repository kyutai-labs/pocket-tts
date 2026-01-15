#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmarks}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/benchmark_${TIMESTAMP}.log}"
RUN_ID="${RUN_ID:-${TIMESTAMP}}"
COMPARE_RUN="${COMPARE_RUN:-}"

TEXTS=(
  "Hello there. This is a quick speed test."
  "Good morning. This is pocket tts."
  "The quick brown fox jumps over the lazy dog. Testing latency and clarity on a short paragraph."
  "Today we measure generation speed on a few sentences to compare float32 and int8 runs."
  "In this benchmark we generate a longer passage to stress the transformer and the audio decoder. We keep the model settings fixed and measure end to end latency on the same text across runs."
  "This is a longer script with multiple sentences intended to keep the model busy for a while. The goal is to measure steady state performance and compute a reliable median and p90 runtime."
)

EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUTPUT_DIR"
echo "benchmark_run_started=${TIMESTAMP}" | tee -a "$OUTPUT_FILE"
echo "output_file=${OUTPUT_FILE}" | tee -a "$OUTPUT_FILE"
echo "run_id=${RUN_ID}" | tee -a "$OUTPUT_FILE"
if [[ -n "${COMPARE_RUN}" ]]; then
  echo "compare_run=${COMPARE_RUN}" | tee -a "$OUTPUT_FILE"
fi
echo "extra_args=${EXTRA_ARGS}" | tee -a "$OUTPUT_FILE"

for text in "${TEXTS[@]}"; do
  echo "----" | tee -a "$OUTPUT_FILE"
  echo "text=${text}" | tee -a "$OUTPUT_FILE"
  pushd "$ROOT_DIR" >/dev/null
  if [[ -n "${COMPARE_RUN}" ]]; then
    uv run python "$SCRIPT_DIR/benchmark.py" --text "$text" --save-audio --run-id "$RUN_ID" --compare-run "$COMPARE_RUN" ${EXTRA_ARGS} 2>&1 | tee -a "$OUTPUT_FILE"
  else
    uv run python "$SCRIPT_DIR/benchmark.py" --text "$text" --save-audio --run-id "$RUN_ID" ${EXTRA_ARGS} 2>&1 | tee -a "$OUTPUT_FILE"
  fi
  popd >/dev/null
done

echo "benchmark_run_finished=$(date +"%Y%m%d_%H%M%S")" | tee -a "$OUTPUT_FILE"
