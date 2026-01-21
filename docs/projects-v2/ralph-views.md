# GitHub Projects (v2): Ralph Views (Label-wired)

This repo’s automation uses **labels** as the source of truth:
- `ralph/workset`
- `ralph/status:*`
- `ralph/owner:*`

GitHub Projects v2 can create **saved views** that filter items by labels, and the REST API supports creating views with a `filter` and a `layout` (`table`, `board`, `roadmap`).

## Views created by the script

The provided script creates the following **Table** views:

1. **Ralph — Backlog (Queued)**
   - Filter: `is:issue is:open label:"ralph/workset" label:"ralph/status:queued"`

2. **Ralph — Claimed**
   - Filter: `is:issue is:open label:"ralph/workset" label:"ralph/status:claimed"`

3. **Ralph — In Progress**
   - Filter: `is:issue is:open label:"ralph/workset" label:"ralph/status:in-progress"`

4. **Ralph — Blocked**
   - Filter: `is:issue is:open label:"ralph/workset" label:"ralph/status:blocked"`

5. **Ralph — By Owner**
   - Four views:
     - `... label:"ralph/owner:orchestrator"`
     - `... label:"ralph/owner:codex"`
     - `... label:"ralph/owner:gemini"`
     - `... label:"ralph/owner:zai"`

## Run

### Organization-owned project
```bash
./scripts/create-ralph-project-views.sh org <ORG> <PROJECT_NUMBER>
```

### User-owned project
```bash
./scripts/create-ralph-project-views.sh user <PROJECT_NUMBER>
```

Notes:
- The script uses `gh api` and requires `gh auth login`.
- For user-owned projects, the REST endpoint requires the **numeric user id**, which the script fetches automatically.

## Customize
Edit the `VIEWS` block in `scripts/create-ralph-project-views.sh` if you want to:
- add a board view (layout `board`)
- add a roadmap view (layout `roadmap`)
- include additional filters (e.g., by component labels)
