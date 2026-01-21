# AGENTS.md — GitHub Ralph Loop Operating Rules

This repository is operated by automated and semi-automated agents (Windsurf Cascade + CLI tools).
These rules are mandatory for agentic changes.

## Non-negotiables

1. **Never commit directly to `main`.**
2. **All changes must map to a GitHub issue.**
3. **One issue per agent at a time** (parallelism allowed across agents).
4. **Prefer small, reviewable diffs.**
5. **Kill switch:** stop immediately if `.ralph-stop` exists or `RALPH_STOP=true`.

## Default workflow model: git worktrees

All issue work must be done in a dedicated worktree.

### Worktree conventions
- Worktrees root: `.worktrees/`
- Worktree directory: `.worktrees/issue-<num>-<slug>/`
- Branch naming: `ralph/issue-<num>-<slug>`
- Slug rules: lowercase, hyphen-separated, max ~40 chars.

### Create a worktree (new branch)
```bash
mkdir -p .worktrees
git fetch origin
git worktree add -b "ralph/issue-<num>-<slug>" ".worktrees/issue-<num>-<slug>"
```

### Cleanup after merge
```bash
git worktree remove ".worktrees/issue-<num>-<slug>"
git branch -D "ralph/issue-<num>-<slug>" 2>/dev/null || true
```

## Label-based claiming and assignment

This repo uses labels (not GitHub assignees) to represent automation “ownership”.

### Workset
- `ralph/workset`

### Status (mutually exclusive; exactly one at a time)
- `ralph/status:queued`
- `ralph/status:claimed`
- `ralph/status:in-progress`
- `ralph/status:blocked`
Optional:
- `ralph/status:needs-review`
- `ralph/status:done`

### Owner (exactly one when claimed/in-progress)
- `ralph/owner:orchestrator`
- `ralph/owner:codex`
- `ralph/owner:gemini`
- `ralph/owner:zai`

### Optional blocked reasons
- `ralph/blocked:needs-info`
- `ralph/blocked:needs-decision`
- `ralph/blocked:upstream`
- `ralph/blocked:ci`
- `ralph/blocked:repro`


## Locking to prevent claim races

When multiple agents run concurrently, they must acquire a remote lock ref before claiming an issue.

- Lock ref: `refs/ralph-locks/issue-<num>`

Acquire:
```bash
ISSUE=<num>
LOCK_REF="refs/ralph-locks/issue-$ISSUE"
git update-ref "$LOCK_REF" HEAD
if git push origin "$LOCK_REF"; then
  echo "lock acquired"
else
  git update-ref -d "$LOCK_REF" || true
  # pick another issue
fi
```

Release (always):
```bash
ISSUE=<num>
LOCK_REF="refs/ralph-locks/issue-$ISSUE"
git push origin ":$LOCK_REF" || true
git update-ref -d "$LOCK_REF" || true
```

Inspect:
```bash
git ls-remote --refs origin "refs/ralph-locks/*"
```
## Testing and verification discipline

Agents must identify and run the repository’s canonical:
- format
- lint
- tests (fast + full if applicable)
- build (if applicable)

If canonical commands are unclear, the agent must:
1. inspect README / CONTRIBUTING / scripts / package scripts / Cargo workspace config
2. propose a command map
3. run at least a minimal test before opening a PR

## Commit / PR rules
- Conventional commit style preferred.
- PR body should include `Resolves #<num>` when correct.
- Prefer squash merge unless repo policy dictates otherwise.

## Blocked policy
When blocked:
1. set `ralph/status:blocked`
2. remove `ralph/status:claimed` / `ralph/status:in-progress`
3. remove `ralph/owner:*`
4. add one `ralph/blocked:*` reason when possible
5. leave a clear comment: what was tried, what was found, what is needed next
