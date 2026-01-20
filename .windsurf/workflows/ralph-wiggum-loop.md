---
description: Ralph Wiggum loop (plan/build) using repo-local prompts
---

1. Add or update specs in `ralph/specs/*.md` (one JTBD topic per file).
2. Make the loop script executable: `chmod +x ralph/loop.sh`.
3. Planning mode (gap analysis -> plan only):
   - `./ralph/loop.sh plan` (optional max iterations: `./ralph/loop.sh plan 2`).
   - Set env vars as needed: `RALPH_CLI=claude`, `RALPH_MODEL=opus`.
4. Scoped planning mode (plan only for a work slice):
   - `./ralph/loop.sh plan-work "short work description"` (optional max iterations: `./ralph/loop.sh plan-work "..." 2`).
5. Building mode (one task per iteration):
   - `./ralph/loop.sh` (optional max iterations: `./ralph/loop.sh 5`).
   - Backpressure: run tests from `ralph/AGENTS.md` and fix failures.
6. Safety: only enable `RALPH_DANGEROUS=1` when running in a sandbox.
7. Tune the loop by updating `ralph/AGENTS.md` and prompt files when you see failure patterns.
8. Use `RALPH_PUSH=1` if you want the loop to `git push` after each iteration.
