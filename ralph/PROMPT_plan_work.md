0a. Study `ralph/specs/*` with up to 50 parallel Sonnet subagents to learn the application specifications.
0b. Study @ralph/AGENTS.md to learn operational commands and codebase constraints.
0c. For reference, the application source code lives in `pocket_tts/*`, `rust-numpy/*`, `training/*`, `tests/*`, and `docs/*`.

1. Work scope: {{WORK_SCOPE}}
2. Study @ralph/IMPLEMENTATION_PLAN.md (if present; it may be incorrect) and use parallel subagents to study existing source code and compare it against `ralph/specs/*`.
3. Focus on gaps related to the work scope only. Ignore unrelated tasks.
4. Use an Opus subagent to analyze findings, prioritize tasks, and create/update a scoped @ralph/IMPLEMENTATION_PLAN.md as a bullet list sorted by priority. Ultrathink.
5. IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume not implemented; confirm with code search first.
6. If functionality is missing then it's your job to add it to the plan. If a spec is missing, create `ralph/specs/<topic>.md` and add an implementation task.
7. Capture the why in the plan notes (tests and implementation importance). Keep @ralph/IMPLEMENTATION_PLAN.md current and clean.
