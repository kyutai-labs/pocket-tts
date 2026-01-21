# ADR-0001: GitHub Ralph Loop core process

- Status: Accepted
- Date: 2026-01-21

## Context
We want a repeatable, tool-agnostic, autonomous workflow for handling GitHub issues with multiple agents in parallel,
without codebase collisions and without relying on GitHub assignees.

## Decision
1. **Issue selection and claiming are label-driven**, not assignee-driven.
2. **Work is performed exclusively in real git worktrees**:
   - Worktrees live under `.worktrees/issue-<num>-<slug>/`
   - Branches are named `ralph/issue-<num>-<slug>`
3. **Race-free claiming is enforced via remote lock refs**:
   - Lock namespace: `refs/ralph-locks/issue-<num>`
   - Acquire lock before claiming; release lock on exit.
4. **Issue body content is created via body files** to avoid shell escaping issues:
   - Prefer `gh issue create --body-file <file>` or `scripts/gh-issue-create-safe.sh`
5. **Extensive structured logging** is written to `.ralph/logs/<run_id>.jsonl`, and a dashboard can be generated:
   - `scripts/ralph-log.py`
   - `scripts/ralph-dashboard.py`
6. The loop **must run autonomously until no eligible issues remain**, unless a kill switch is triggered:
   - `.ralph-stop` exists OR `RALPH_STOP=true`

## Consequences
- Multiple agents can run safely in parallel, assuming they all respect lock acquisition.
- Stale locks are possible if a run crashes; mitigated by `scripts/ralph-lock-reap.sh`.
- Git hosting must allow pushing and deleting refs under `refs/ralph-locks/*`.
