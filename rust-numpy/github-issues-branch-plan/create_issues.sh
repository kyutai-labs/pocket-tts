#!/usr/bin/env bash
set -euo pipefail

# Creates labels + issues for the NumPy->Rust port plan.
#
# Usage:
#   cd <repo>
#   bash ./create_issues.sh
#
# Optional:
#   REPO=owner/name bash ./create_issues.sh

REPO="${REPO:-}"
if [[ -z "${REPO}" ]]; then
  REPO_ARG=()
else
  REPO_ARG=(--repo "$REPO")
fi

echo "Creating labels (idempotent; ignoring errors if labels already exist)..."

# Area labels
gh label create core        "${REPO_ARG[@]}" --color 1D76DB --description "Core ndarray layout/views/broadcast/iter"       || true
gh label create dtype       "${REPO_ARG[@]}" --color 0E8A16 --description "DType, promotion, casting"                     || true
gh label create ufunc       "${REPO_ARG[@]}" --color 5319E7 --description "UFunc registry/selection/execution/reductions" || true
gh label create kernels     "${REPO_ARG[@]}" --color FBCA04 --description "Baseline kernels"                              || true
gh label create api         "${REPO_ARG[@]}" --color 006B75 --description "Public API facade"                             || true
gh label create testing     "${REPO_ARG[@]}" --color BFDADC --description "Testing harness, parity, golden tests"         || true
gh label create ci          "${REPO_ARG[@]}" --color D4C5F9 --description "CI workflows, tooling"                         || true

# Meta labels
gh label create performance "${REPO_ARG[@]}" --color D93F0B --description "Performance work (gated)"                       || true
gh label create breaking    "${REPO_ARG[@]}" --color B60205 --description "Breaking change"                                || true
gh label create good-first-issue "${REPO_ARG[@]}" --color 7057FF --description "Starter-friendly"                          || true

echo "Creating issues..."

create_issue () {
  local title="$1"
  local labels="$2"
  local body="$3"
  local url
  url="$(gh issue create "${REPO_ARG[@]}" --title "$title" --label "$labels" --body "$body")"
  echo "$url" | sed -n 's#.*/issues/\([0-9]\+\)$#\1#p'
}

ISSUE_1="$(create_issue \
  "[testing][ci] Add PARITY.md and link checklist to tests" \
  "testing,ci" \
"Goal: Introduce a single parity checklist tied to concrete tests.

Acceptance Criteria:
- [ ] PARITY.md exists with sections: Layout, Broadcast, DType, UFunc, Reduce
- [ ] Each section links to a test file path (stubs allowed initially)

Notes / Invariants:
- Treat PARITY.md as the authoritative progress ledger.
")"

ISSUE_2="$(create_issue \
  "[ci] Workspace lint/format enforcement (fmt + clippy -D warnings)" \
  "ci" \
"Goal: Enforce consistent formatting and linting.

Acceptance Criteria:
- [ ] CI runs: cargo fmt --check
- [ ] CI runs: cargo clippy --workspace -- -D warnings

Depends on: #$ISSUE_1 (recommended, not required)
")"

ISSUE_3="$(create_issue \
  "[core] Derive contiguity (C/F) and layout invariants" \
  "core,testing" \
"Goal: Implement derived contiguity checks and core layout invariants.

Acceptance Criteria:
- [ ] Layout::is_c_contiguous()
- [ ] Layout::is_f_contiguous()
- [ ] Tests cover: 0-D, 1-D, and canonical 2-D cases

Notes / Invariants:
- Contiguity must be derived from shape/strides (do not store flags blindly).
")"

ISSUE_4="$(create_issue \
  "[core] Transpose-as-view (stride permutation)" \
  "core,testing" \
"Goal: Implement transpose as a pure view (no copy).

Acceptance Criteria:
- [ ] Layout::transpose(axes: Option<&[usize]>) -> Layout
- [ ] Tests validate shape/strides correctness for default and explicit axes

Depends on: #$ISSUE_3
")"

ISSUE_5="$(create_issue \
  "[core] Slicing-as-view (ranges + step; support negative step)" \
  "core,testing" \
"Goal: Implement slicing as views with steps.

Acceptance Criteria:
- [ ] Slice spec supports:
  - full range ':'
  - start..end
  - step (including negative)
- [ ] Tests cover positive and negative strides

Depends on: #$ISSUE_3
")"

ISSUE_6="$(create_issue \
  "[core] Broadcast layout (stride=0 where dim=1)" \
  "core,testing" \
"Goal: Add broadcast_layout that produces stride=0 broadcasted views.

Acceptance Criteria:
- [ ] broadcast_layout(layout, out_shape) -> Layout
- [ ] Tests validate stride=0 behavior and errors on incompatible shapes

Depends on: #$ISSUE_3
")"

ISSUE_7="$(create_issue \
  "[core] Minimal correct N-D iterator offsets (no coalescing yet)" \
  "core,testing" \
"Goal: Implement a correct baseline N-D iterator/planner.

Acceptance Criteria:
- [ ] Given broadcasted layouts, iterator yields correct per-operand element offsets
- [ ] Tests verify offsets for small shapes against expected sequences

Depends on: #$ISSUE_6
")"

ISSUE_8="$(create_issue \
  "[dtype] Numeric promotion rules (explicit table) for Add/Sub/Mul/TrueDiv/Comparison/Bitwise" \
  "dtype,testing" \
"Goal: Expand dtype promotion into an explicit, auditable rule set.

Acceptance Criteria:
- [ ] promote(left,right,op) supports Bool/Int/UInt/Float/Complex for:
  Add, Sub, Mul, TrueDiv, Comparison, Bitwise
- [ ] Tests include:
  - int + float -> float
  - float + complex -> complex
  - bitwise rejects float/complex with typed error

Notes:
- Use fixed-width dtypes only (no platform int).
")"

ISSUE_9="$(create_issue \
  "[dtype] Casting policy skeleton (Safe/SameKind/Unsafe)" \
  "dtype,testing" \
"Goal: Implement can_cast(from,to,safety) metadata rules.

Acceptance Criteria:
- [ ] can_cast implements Safe/SameKind/Unsafe categories
- [ ] Tests cover representative pairs

Depends on: #$ISSUE_8 (recommended)
")"

ISSUE_10="$(create_issue \
  "[ufunc] UFunc registry + kernel lookup by signature" \
  "ufunc,testing" \
"Goal: Implement registry and kernel selection by dtype signature.

Acceptance Criteria:
- [ ] Registry registers and retrieves UFuncs by name
- [ ] Kernel selection by exact signature works
- [ ] Tests cover registry + selection

Depends on: #$ISSUE_8 (promotion integration next)
")"

ISSUE_11="$(create_issue \
  "[ufunc][kernels] Minimal execution engine (contiguous baseline) for binary ufunc" \
  "ufunc,kernels,testing" \
"Goal: Execute a selected 1-D kernel over planned runs (start contiguous).

Acceptance Criteria:
- [ ] Given kernel + layouts + buffers, exec succeeds for contiguous case
- [ ] Tests: add_f64 contiguous correctness

Depends on: #$ISSUE_7, #$ISSUE_10
")"

ISSUE_12="$(create_issue \
  "[ufunc][core][kernels] Broadcast-aware binary exec path (elementwise add)" \
  "ufunc,core,kernels,testing" \
"Goal: Make binary execution broadcast-correct.

Acceptance Criteria:
- [ ] add works for broadcasted shapes (e.g. (3,1)+(1,4)->(3,4))
- [ ] Tests cover mixed broadcasting patterns

Depends on: #$ISSUE_6, #$ISSUE_7, #$ISSUE_11
")"

ISSUE_13="$(create_issue \
  "[api] Public Array facade + add() wired end-to-end" \
  "api,testing" \
"Goal: Expose minimal public API for Array + add.

Acceptance Criteria:
- [ ] api::Array supports basic construction (start with f64)
- [ ] api::ops::add(&Array,&Array)->Array
- [ ] Tests validate public API behavior

Depends on: #$ISSUE_12
")"

ISSUE_14="$(create_issue \
  "[kernels][ufunc][api] Add mul ufunc (mirror add) + tests" \
  "kernels,ufunc,api,testing" \
"Goal: Implement multiplication ufunc with the same pathway as add.

Acceptance Criteria:
- [ ] mul works for contiguous and broadcasted inputs
- [ ] Tests mirror add coverage for mul

Depends on: #$ISSUE_13
")"

ISSUE_15="$(create_issue \
  "[ufunc][kernels][api] Global sum reduction for f64" \
  "ufunc,kernels,api,testing" \
"Goal: Implement sum reduction over all axes.

Acceptance Criteria:
- [ ] sum(Array)->scalar (or 0-D Array) for f64
- [ ] Tests define and enforce empty-array policy

Depends on: #$ISSUE_13
")"

ISSUE_16="$(create_issue \
  "[ufunc][api] sum(axis=..., keepdims=...): single axis first" \
  "ufunc,api,testing" \
"Goal: Implement sum over a single axis with keepdims.

Acceptance Criteria:
- [ ] sum(axis=i, keepdims=bool) correct for small shapes
- [ ] Tests validate resulting shape + values

Depends on: #$ISSUE_15
")"

ISSUE_17="$(create_issue \
  "[performance][core][ufunc] Dimension coalescing into fewer contiguous runs" \
  "performance,core,ufunc,testing" \
"Goal: Optimize iteration by coalescing dimensions into fewer 1-D kernel calls.

Acceptance Criteria:
- [ ] Outputs identical to baseline across existing test suite
- [ ] Basic perf sanity check (bench optional)

Depends on: #$ISSUE_12, #$ISSUE_16
")"

ISSUE_18="$(create_issue \
  "[performance][kernels] SIMD kernels (feature-gated) + runtime dispatch" \
  "performance,kernels,testing" \
"Goal: Add optional SIMD-specialized kernels with runtime dispatch.

Acceptance Criteria:
- [ ] Baseline path remains default and passes all tests
- [ ] SIMD feature passes identical tests when enabled
- [ ] Dispatch chooses best available implementation safely

Depends on: #$ISSUE_17
")"

ISSUE_19="$(create_issue \
  "[performance] Threading policy for safe kernels (no overlap/alias hazards)" \
  "performance,testing" \
"Goal: Parallelize only where safe and deterministic under defined rules.

Acceptance Criteria:
- [ ] Threading is conditional and respects aliasing constraints
- [ ] Tests confirm correctness; determinism where required

Depends on: #$ISSUE_17
")"

echo
echo "Created issues:"
for n in $ISSUE_1 $ISSUE_2 $ISSUE_3 $ISSUE_4 $ISSUE_5 $ISSUE_6 $ISSUE_7 $ISSUE_8 $ISSUE_9 $ISSUE_10 $ISSUE_11 $ISSUE_12 $ISSUE_13 $ISSUE_14 $ISSUE_15 $ISSUE_16 $ISSUE_17 $ISSUE_18 $ISSUE_19; do
  echo "  #$n"
done

echo
echo "Tip: start branches like:"
echo "  git checkout -b chore/${ISSUE_1}-parity-md"
echo "  git checkout -b feat/${ISSUE_6}-broadcast-layout"
