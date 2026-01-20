0a. Study `ralph/specs/*` with up to 50 parallel Sonnet subagents to learn the application specifications.
0b. Study @ralph/AGENTS.md to learn operational commands and codebase constraints.
0c. For reference, the application source code lives in `pocket_tts/*`, `rust-numpy/*`, `training/*`, `tests/*`, and `docs/*`.

1. Study @ralph/IMPLEMENTATION_PLAN.md (if present; it may be incorrect) and use parallel subagents to study existing source code and compare it against `ralph/specs/*`.
2. Use an Opus subagent to analyze findings, prioritize tasks, and create/update @ralph/IMPLEMENTATION_PLAN.md as a bullet list sorted by priority. Ultrathink.
3. IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume not implemented; confirm with code search first.
4. If functionality is missing then it's your job to add it to the plan. If a spec is missing, create `ralph/specs/<topic>.md` and add an implementation task.
5. Capture the why in the plan notes (tests and implementation importance). Keep @ralph/IMPLEMENTATION_PLAN.md current and clean.
