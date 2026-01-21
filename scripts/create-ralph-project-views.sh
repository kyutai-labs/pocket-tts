#!/usr/bin/env bash
set -euo pipefail

# Create GitHub Projects v2 saved views wired to Ralph labels.
#
# Usage:
#   ./scripts/create-ralph-project-views.sh org <ORG> <PROJECT_NUMBER>
#   ./scripts/create-ralph-project-views.sh user <PROJECT_NUMBER>

API_VERSION="2022-11-28"

die() { echo "error: $*" 1>&2; exit 1; }

if ! command -v gh >/dev/null 2>&1; then
  die "gh CLI is required"
fi

MODE="${1:-}"
case "$MODE" in
  org)
    ORG="${2:-}"
    PROJECT_NUMBER="${3:-}"
    [[ -n "$ORG" && -n "$PROJECT_NUMBER" ]] || die "org mode requires: <ORG> <PROJECT_NUMBER>"
    ENDPOINT="/orgs/${ORG}/projectsV2/${PROJECT_NUMBER}/views"
    ;;
  user)
    PROJECT_NUMBER="${2:-}"
    [[ -n "$PROJECT_NUMBER" ]] || die "user mode requires: <PROJECT_NUMBER>"
    USER_ID="$(gh api user --jq .id)"
    ENDPOINT="/users/${USER_ID}/projectsV2/${PROJECT_NUMBER}/views"
    ;;
  *)
    die "usage: $0 org <ORG> <PROJECT_NUMBER> | user <PROJECT_NUMBER>"
    ;;
esac

# name|layout|filter
VIEWS=(
  "Ralph — Backlog (Queued)|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/status:queued\""
  "Ralph — Claimed|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/status:claimed\""
  "Ralph — In Progress|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/status:in-progress\""
  "Ralph — Blocked|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/status:blocked\""
  "Ralph — Owner: orchestrator|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/owner:orchestrator\""
  "Ralph — Owner: codex|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/owner:codex\""
  "Ralph — Owner: gemini|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/owner:gemini\""
  "Ralph — Owner: zai|table|is:issue is:open label:\"ralph/workset\" label:\"ralph/owner:zai\""
)

echo "Creating views via ${ENDPOINT}"
for spec in "${VIEWS[@]}"; do
  IFS="|" read -r NAME LAYOUT FILTER <<<"$spec"
  echo "- ${NAME}"
  gh api     --method POST     -H "Accept: application/vnd.github+json"     -H "X-GitHub-Api-Version: ${API_VERSION}"     "${ENDPOINT}"     -f "name=${NAME}"     -f "layout=${LAYOUT}"     -f "filter=${FILTER}"     >/dev/null
done

echo "Done."
