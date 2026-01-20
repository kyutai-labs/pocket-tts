#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="build"
PROMPT_FILE="${SCRIPT_DIR}/PROMPT_build.md"
MAX_ITERATIONS=0
WORK_SCOPE=""

if [[ "${1-}" == "plan" ]]; then
  MODE="plan"
  PROMPT_FILE="${SCRIPT_DIR}/PROMPT_plan.md"
  if [[ "${2-}" =~ ^[0-9]+$ ]]; then
    MAX_ITERATIONS="$2"
  fi
elif [[ "${1-}" == "plan-work" ]]; then
  if [[ -z "${2-}" ]]; then
    echo "Error: plan-work requires a work description" >&2
    exit 1
  fi
  MODE="plan-work"
  PROMPT_FILE="${SCRIPT_DIR}/PROMPT_plan_work.md"
  WORK_SCOPE="${2}"
  if [[ "${3-}" =~ ^[0-9]+$ ]]; then
    MAX_ITERATIONS="$3"
  fi
elif [[ "${1-}" =~ ^[0-9]+$ ]]; then
  MAX_ITERATIONS="$1"
fi

CLI_BIN="${RALPH_CLI:-claude}"
CLI_ARGS=()

if [[ "${RALPH_HEADLESS:-1}" == "1" ]]; then
  CLI_ARGS+=("-p")
fi

if [[ "${RALPH_DANGEROUS:-0}" == "1" ]]; then
  CLI_ARGS+=("--dangerously-skip-permissions")
fi

if [[ "${RALPH_STREAM_JSON:-0}" == "1" ]]; then
  CLI_ARGS+=("--output-format=stream-json")
fi

if [[ -n "${RALPH_MODEL:-}" ]]; then
  CLI_ARGS+=("--model" "${RALPH_MODEL}")
fi

if [[ "${RALPH_VERBOSE:-0}" == "1" ]]; then
  CLI_ARGS+=("--verbose")
fi

if [[ -n "${RALPH_CLI_ARGS:-}" ]]; then
  CLI_ARGS+=(${RALPH_CLI_ARGS})
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Error: prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

ITERATION=0
CURRENT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo "")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode: ${MODE}"
echo "Prompt: ${PROMPT_FILE}"
if [[ -n "${WORK_SCOPE}" ]]; then
  echo "Scope: ${WORK_SCOPE}"
fi
if [[ -n "${CURRENT_BRANCH}" ]]; then
  echo "Branch: ${CURRENT_BRANCH}"
fi
if [[ "${MAX_ITERATIONS}" -gt 0 ]]; then
  echo "Max: ${MAX_ITERATIONS} iterations"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

while true; do
  if [[ "${MAX_ITERATIONS}" -gt 0 && "${ITERATION}" -ge "${MAX_ITERATIONS}" ]]; then
    echo "Reached max iterations: ${MAX_ITERATIONS}"
    break
  fi

  if [[ -n "${WORK_SCOPE}" ]]; then
    WORK_SCOPE="${WORK_SCOPE}" PROMPT_FILE="${PROMPT_FILE}" python - <<'PY' | "${CLI_BIN}" "${CLI_ARGS[@]}"
import os

prompt_file = os.environ["PROMPT_FILE"]
scope = os.environ["WORK_SCOPE"]

with open(prompt_file, "r", encoding="utf-8") as handle:
    content = handle.read()

print(content.replace("{{WORK_SCOPE}}", scope), end="")
PY
  else
    cat "${PROMPT_FILE}" | "${CLI_BIN}" "${CLI_ARGS[@]}"
  fi

  if [[ "${RALPH_PUSH:-0}" == "1" && -n "${CURRENT_BRANCH}" ]]; then
    git -C "${REPO_ROOT}" push origin "${CURRENT_BRANCH}" || git -C "${REPO_ROOT}" push -u origin "${CURRENT_BRANCH}"
  fi

  ITERATION=$((ITERATION + 1))
  echo -e "\n\n======================== LOOP ${ITERATION} ========================\n"
done
