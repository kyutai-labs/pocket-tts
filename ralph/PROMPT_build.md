0a. Study `ralph/specs/*` with up to 50 parallel Sonnet subagents to learn the application specifications.
0b. Study @ralph/IMPLEMENTATION_PLAN.md to understand the plan so far.
0c. Study @ralph/AGENTS.md to learn build, test, and validation commands.
0d. For reference, the application source code lives in `pocket_tts/*`, `rust-numpy/*`, `training/*`, `tests/*`, and `docs/*`.

1. Implement functionality per the specifications using parallel subagents. Choose the most important incomplete item in @ralph/IMPLEMENTATION_PLAN.md.
2. Before making changes, search the codebase (don't assume not implemented). Use parallel subagents for searches/reads and only 1 subagent for build/tests.
3. If a task is too large, split it in @ralph/IMPLEMENTATION_PLAN.md and pick the first subtask.
4. Implement the change and run the relevant tests from @ralph/AGENTS.md. Fix failures before proceeding.
5. Update @ralph/IMPLEMENTATION_PLAN.md with findings, mark tasks complete, and remove clutter as needed.
6. Update @ralph/AGENTS.md if you learn new operational commands or constraints. Keep it brief.
7. When tests pass, commit changes. Push if configured by the loop runner.
