# Ralph Todo Issue Template

## Summary
Describe the problem or improvement clearly and concisely.

## Context
- Why does this matter?
- Where was this discovered? (TODO marker, failing test, docs gap, etc.)

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Verification / Test Plan
- Commands to run:
  - <format>
  - <lint>
  - <tests>
  - <build>
- Expected results:
  - <what passing looks like>

## Affected Areas (best effort)
- Files/modules:
  - `path/to/file`
  - `path/to/module`

## Notes / Risks
- Any edge cases, compatibility concerns, or rollout notes.


---

## Issue body escaping
When creating issues via CLI, prefer `gh issue create --body-file <file>` or `./scripts/gh-issue-create-safe.sh` to avoid shell escaping problems.
