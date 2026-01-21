# Claude Code integration

This folder documents the project-level Claude Code configuration shipped with this repo.

## Files
- `../../CLAUDE.md` — persistent project instructions for Claude Code
- `.claude/skills/github-ralph-loop-setup/SKILL.md` — provides `/github-ralph-loop-setup`
- `.claude/skills/github-ralph-loop/SKILL.md` — provides `/github-ralph-loop`
- `.claude/skills/github-ralph-loop/*` — templates used by the loop

## Usage
From the repo root in Claude Code:
1. Run `/github-ralph-loop-setup`
2. Queue issues by applying:
   - `ralph/workset`
   - `ralph/status:queued`
   - optional lane: `ralph/lane:p0|p1|p2`
3. Run `/github-ralph-loop`

4. (Optional) Run `/github-ralph-plan-todos` to generate and queue new todo issues.
