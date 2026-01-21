# OpenCode integration

This repo ships OpenCode-native skills under `.opencode/skills/`.

OpenCode will discover project skills from either:
- `.opencode/skills/<name>/SKILL.md`
- `.claude/skills/<name>/SKILL.md` (Claude-compatible path)

(Discovery rules are documented by OpenCode.)

## Included skills
- `github-ralph-loop-setup`
- `github-ralph-loop`
- `github-ralph-plan-todos`

## Typical usage (local CLI)
1. Start OpenCode in your repo:
   ```bash
   opencode
   ```
2. Use the skill tool to load a skill, then follow it. For example:
   - load `github-ralph-loop-setup`
   - run setup actions
   - load `github-ralph-loop` and run until finished

## GitHub App / Actions
OpenCode can also run in GitHub via its GitHub integration if you install the OpenCode GitHub app and workflow.
See OpenCode GitHub docs for details.
