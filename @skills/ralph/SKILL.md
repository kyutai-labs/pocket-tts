---
name: ralph-loop
description: Executing the Ralph Wiggum autonomous coding loop (Planning or Building phases).
---

# Ralph Loop Skill

This skill implements the "Ralph Wiggum" autonomous loop logic using the repo-local `ralph/` assets. It has two main modes: **PLANNING** and **BUILDING**.

## Loop Assets

The loop relies on these canonical files:

1. `ralph/specs/*.md`: Jobs To Be Done (JTBD) specifications (Requirements).
2. `ralph/IMPLEMENTATION_PLAN.md`: Prioritized task list (State).
3. `ralph/AGENTS.md`: Operational commands + constraints (Context/backpressure).
4. `ralph/PROMPT_plan.md` + `ralph/PROMPT_build.md`: Loop prompts.
5. `ralph/loop.sh`: Outer loop runner for plan/build iterations.

Each iteration starts from the same prompt + AGENTS + specs context and reads the plan file from disk.

## Mode 1: PLANNING

Use this mode when initializing or when specs change significantly.

**Goal**: Create or Update `IMPLEMENTATION_PLAN.md`.

1. **Orient**: Read all files in `ralph/specs/`.
2. **Context**: Read `ralph/AGENTS.md` for operational constraints.
3. **Analyze**: Read source (`pocket_tts/`, `rust-numpy/`, `training/`, `tests/`, `docs/`).
4. **Gap Analysis**: Compare `ralph/specs/` vs code. Identify missing features, bugs, or discrepancies.
5. **Plan**:
   - Create or overwrite `ralph/IMPLEMENTATION_PLAN.md`.
   - Format: Markdown bullet list, sorted by priority.
   - **CRITICAL**: Do NOT implement code in this phase.
   - **CRITICAL**: "Don't assume not implemented" — confirm via code search first.

## Mode 2: BUILDING

Use this mode to execute the work.

**Goal**: Complete ONE atomic task.

1.  **Orient**: Read `ralph/specs/` and `ralph/AGENTS.md`.
2. **Select**: Read `ralph/IMPLEMENTATION_PLAN.md`. Pick the most important incomplete task.
   - If task is too big, split it and pick the first subtask.
3. **Investigate**: Search the codebase to confirm current state for that task.
4. **Implement**:
   - Write/modify code to satisfy the spec.
   - Follow `ralph/AGENTS.md` patterns.
5. **Verify (Backpressure)**:
   - Run tests defined in `ralph/AGENTS.md` or relevant to the change.
   - **CRITICAL**: If tests fail, fix them. Do not mark done until tests pass.
6. **Record**:
   - Update `ralph/IMPLEMENTATION_PLAN.md` (mark tasks done, add discoveries).
   - Update `ralph/AGENTS.md` if you learned new run/test/build commands.
7. **Commit**: Commit when tests pass. Push if configured by the loop runner.

## Loop Script Usage

- Planning: `./ralph/loop.sh plan` (optional max iterations: `./ralph/loop.sh plan 2`).
- Scoped planning: `./ralph/loop.sh plan-work "short work description"` (optional max iterations: `./ralph/loop.sh plan-work "..." 2`).
- Building: `./ralph/loop.sh` (optional max iterations: `./ralph/loop.sh 5`).

Environment flags:
- `RALPH_DANGEROUS=1` only in a sandbox.
- `RALPH_MODEL=opus` (or `sonnet` for faster build loops).
- `RALPH_PUSH=1` to push after each iteration.

## Safety

Only run with `--dangerously-skip-permissions` in an isolated environment.

## Artifact Formats

### ralph/IMPLEMENTATION_PLAN.md

```markdown
# Implementation Plan

- [ ] #123 [High] Feature A: Description...
- [ ] #124 [Med] Feature B: Description...
- [x] #122 [High] Feature C: Completed...
```

### ralph/AGENTS.md

```markdown
# Ralph Loop Agents (Operational)

(Brief, <60 lines)

## Build & Run
- `uv run pocket-tts --help`
- `uv run pocket-tts generate "Hello" --voice alba --output out.wav`
- `uv run pocket-tts serve --port 8080`

## Validation
- Tests: `uv run pytest -n 3 -v`

## Codebase Patterns
- Absolute imports only (no relative imports).
- Rust must remain safe (no `unsafe`) in `rust-numpy/`.
```
