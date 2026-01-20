#!/usr/bin/env bash
set -euo pipefail

# Creates a GitHub Projects (v2) board, adds issues, sets a "Stage" single-select field.
#
# Requirements:
#   - gh CLI authenticated
#   - project scope authorized:
#       gh auth refresh -s project
#
# Usage:
#   OWNER="@me" TITLE="NumPy->Rust Port" bash ./create_project_board.sh
#
# Optional:
#   REPO="owner/repo"                         # if not running inside the repo dir
#   ISSUE_NUMBERS="1,2,3,4"                   # explicit issue numbers (preferred)
#   AUTO_ISSUE_LABELS="core,dtype,ufunc,..."  # used if ISSUE_NUMBERS not provided
#   LABEL_TO_STAGE="ci=Ready,testing=Ready,performance=Blocked"

OWNER="${OWNER:-@me}"
TITLE="${TITLE:-NumPy->Rust Port}"
REPO="${REPO:-}"

AUTO_ISSUE_LABELS="${AUTO_ISSUE_LABELS:-core,dtype,ufunc,kernels,api,testing,ci,performance}"
LABEL_TO_STAGE="${LABEL_TO_STAGE:-ci=Ready,testing=Ready,performance=Blocked}"

command -v jq >/dev/null 2>&1 || { echo "jq is required"; exit 1; }

if [[ -z "$REPO" ]]; then
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
fi

echo "Repo:  $REPO"
echo "Owner: $OWNER"
echo "Title: $TITLE"

echo
echo "Creating project..."
PROJECT_NUMBER="$(gh project create --owner "$OWNER" --title "$TITLE" --format json -q '.number')"
echo "Project number: $PROJECT_NUMBER"

echo
echo "Fetching project id..."
PROJECT_ID="$(gh project view "$PROJECT_NUMBER" --owner "$OWNER" --format json -q '.id')"
echo "Project id: $PROJECT_ID"

STAGE_FIELD_NAME="Stage"
STAGE_OPTIONS="Backlog,Ready,In Progress,In Review,Done,Blocked"

echo
echo "Ensuring '$STAGE_FIELD_NAME' field exists (single select)..."
FIELDS_JSON="$(gh project field-list "$PROJECT_NUMBER" --owner "$OWNER" --format json)"
STAGE_FIELD_ID="$(echo "$FIELDS_JSON" | jq -r --arg name "$STAGE_FIELD_NAME" '.fields[] | select(.name == $name) | .id' | head -n1 || true)"

if [[ -z "${STAGE_FIELD_ID}" || "${STAGE_FIELD_ID}" == "null" ]]; then
  gh project field-create "$PROJECT_NUMBER" \
    --owner "$OWNER" \
    --name "$STAGE_FIELD_NAME" \
    --data-type "SINGLE_SELECT" \
    --single-select-options "$STAGE_OPTIONS" >/dev/null
  echo "Created field '$STAGE_FIELD_NAME'"

  FIELDS_JSON="$(gh project field-list "$PROJECT_NUMBER" --owner "$OWNER" --format json)"
  STAGE_FIELD_ID="$(echo "$FIELDS_JSON" | jq -r --arg name "$STAGE_FIELD_NAME" '.fields[] | select(.name == $name) | .id' | head -n1)"
fi

echo "Stage field id: $STAGE_FIELD_ID"

declare -A STAGE_OPT_ID
while IFS=$'\t' read -r opt_name opt_id; do
  STAGE_OPT_ID["$opt_name"]="$opt_id"
done < <(echo "$FIELDS_JSON" | jq -r --arg name "$STAGE_FIELD_NAME" '
  .fields[]
  | select(.name == $name)
  | .options[]
  | [.name, .id] | @tsv
')

choose_stage_from_labels() {
  local labels_csv="$1"
  local stage="Backlog"

  IFS=',' read -ra rules <<< "$LABEL_TO_STAGE"
  for rule in "${rules[@]}"; do
    local key="${rule%%=*}"
    local val="${rule#*=}"
    if [[ ",$labels_csv," == *",$key,"* ]]; then
      stage="$val"
    fi
  done

  echo "$stage"
}

ISSUE_NUMBERS="${ISSUE_NUMBERS:-}"
ISSUE_URLS=()

if [[ -n "$ISSUE_NUMBERS" ]]; then
  IFS=',' read -ra nums <<< "$ISSUE_NUMBERS"
  for n in "${nums[@]}"; do
    n="$(echo "$n" | xargs)"
    ISSUE_URLS+=("https://github.com/${REPO}/issues/${n}")
  done
else
  echo
  echo "Auto-discovering open issues by labels: $AUTO_ISSUE_LABELS"
  tmpfile="$(mktemp)"
  IFS=',' read -ra lbs <<< "$AUTO_ISSUE_LABELS"
  for lb in "${lbs[@]}"; do
    lb="$(echo "$lb" | xargs)"
    gh issue list --repo "$REPO" --state open --label "$lb" --json url --jq '.[].url' >> "$tmpfile"
  done
  mapfile -t ISSUE_URLS < <(sort -u "$tmpfile")
  rm -f "$tmpfile"
fi

if [[ "${#ISSUE_URLS[@]}" -eq 0 ]]; then
  echo "No issues found to add. Set ISSUE_NUMBERS=\"1,2,3\" or ensure labels exist."
  exit 1
fi

echo
echo "Adding ${#ISSUE_URLS[@]} issue(s) to project and setting Stage..."

for url in "${ISSUE_URLS[@]}"; do
  ITEM_ID="$(gh project item-add "$PROJECT_NUMBER" --owner "$OWNER" --url "$url" --format json -q '.id')"
  if [[ -z "$ITEM_ID" || "$ITEM_ID" == "null" ]]; then
    echo "Failed to add: $url"
    continue
  fi

  labels_csv="$(gh issue view "$url" --json labels --jq '[.labels[].name] | join(\",\")' 2>/dev/null || echo "")"
  stage="$(choose_stage_from_labels "$labels_csv")"

  opt_id="${STAGE_OPT_ID[$stage]:-}"
  if [[ -z "$opt_id" ]]; then
    echo "Warning: no option id for stage '$stage' (url: $url). Leaving Stage unset."
    continue
  fi

  gh project item-edit \
    --id "$ITEM_ID" \
    --project-id "$PROJECT_ID" \
    --field-id "$STAGE_FIELD_ID" \
    --single-select-option-id "$opt_id" >/dev/null

  echo "Added: $url  -> Stage=$stage"
done

echo
echo "Done. Open the project with:"
echo "  gh project view $PROJECT_NUMBER --owner \"$OWNER\" --web"
