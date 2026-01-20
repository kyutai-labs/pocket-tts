# Ralph Wiggum Slash Commands

Complete set of slash commands for Ralph Wiggum tools in OpenCode TUI.

## Available Commands

| Command | Description | Usage |
|---------|-----------|-------|
| `/ralph-create` | Create a new Ralph plan interactively | Create plans with multiple tasks and dependencies |
| `/ralph` | Execute a Ralph plan | Execute plan with full GitHub workflow |
| `/ralph-list` | List available Ralph plans | Show all plan files in ralph/plans/ |
| `/ralph-status` | Check Ralph execution status | Display current execution state |

## Quick Start

### Create and Execute a Plan

```bash
# Step 1: Create a plan
/ralph-create

# Step 2: Execute the plan
/ralph plans/your-plan.json

# Step 3: Monitor progress
/ralph-status
```

### List Available Plans

```bash
/ralph-list
```

### Check Execution Status

```bash
/ralph-status
```

## Command Details

### `/ralph-create`

Create a new Ralph Wiggum plan file through interactive prompts.

**What it does**:
- Prompts for plan name and description
- Guides you through creating tasks
- Supports task dependencies
- Validates plan structure
- Saves to `ralph/plans/{plan-name}.json`

**When to use**:
- Planning a new feature
- Organizing complex tasks
- Creating testing strategies
- Planning refactoring work

**Example**:
```bash
/ralph-create
# Follow prompts to create plan
```

### `/ralph`

Execute a Ralph Wiggum plan with full GitHub workflow automation.

**What it does**:
- Parses plan file
- Creates feature branch
- Creates GitHub issues for each task
- Runs OpenCode iteratively
- Auto-commits changes
- Creates and merges pull request

**When to use**:
- Running a pre-created plan
- Automating feature development
- CI/CD integration
- Batch task execution

**Example**:
```bash
/ralph plans/add-unit-tests.json
```

### `/ralph-list`

List all available Ralph plan files.

**What it does**:
- Scans `ralph/plans/` directory
- Displays plan names
- Shows descriptions if available
- Lists task counts

**When to use**:
- Finding existing plans
- Choosing a plan to execute
- Checking available work

**Example**:
```bash
/ralph-list
```

### `/ralph-status`

Check the current Ralph plan execution status.

**What it does**:
- Reads `.opencode/ralph-state.json`
- Displays plan information
- Shows completed tasks
- Shows current task
- Displays issue and PR numbers
- Shows overall status

**When to use**:
- Monitoring execution progress
- Checking if plan completed
- Resuming interrupted executions
- Debugging execution issues

**Example**:
```bash
/ralph-status
```

## Complete Workflow

### Feature Development Workflow

```bash
# 1. Create a plan
/ralph-create

# 2. List plans to confirm
/ralph-list

# 3. Execute the plan
/ralph plans/your-plan.json

# 4. Monitor progress
/ralph-status

# 5. Check status after execution completes
/ralph-status
```

### Bug Fix Workflow

```bash
# 1. Create bug fix plan
/ralph-create "Create a plan for fixing audio clipping bug"

# 2. Execute the plan
/ralph plans/fix-clipping-bug.json

# 3. Monitor progress
/ralph-status
```

### Testing Workflow

```bash
# 1. Create testing plan
/ralph-create "Create a plan for comprehensive unit tests"

# 2. Execute testing plan
/ralph plans/add-comprehensive-tests.json

# 3. Monitor progress
/ralph-status
```

## File Locations

```
.config/opencode/commands/
├── ralph.md              # Main slash command
├── ralph-create.md       # Plan creator command
├── ralph-list.md          # List plans command
└── ralph-status.md        # Status check command
```

## Integration with Agents

Each slash command delegates to the corresponding Ralph agent:

| Slash Command | Agent |
|-------------|--------|
| `/ralph-create` | `ralph-plan-creator` |
| `/ralph` | `ralph-plan-executor` |
| `/ralph-list` | `coder-agent` (for listing) |
| `/ralph-status` | `coder-agent` (for status) |

## Documentation Links

- **[ralph-create.md](ralph-create.md)** - Plan creator command details
- **[ralph.md](ralph.md)** - Plan executor command details
- **[ralph-list.md](ralph-list.md)** - List command details
- **[ralph-status.md](ralph-status.md)** - Status command details
- **[AGENT-README.md](../../.agent/agents/README.md)** - Agent overview
- **[OPENCODE-TUI.md](../../.agent/agents/OPENCODE-TUI.md)** - TUI usage guide
- **[RALPH-README.md](../../ralph/RALPH-README.md)** - Ralph executor documentation
- **[PLAN-CREATOR.md](../../ralph/PLAN-CREATOR.md)** - Plan creator documentation

## Quick Reference

| Task | Command | Alias |
|------|--------|--------|
| Create plan | `/ralph-create` | - |
| Execute plan | `/ralph` | - |
| List plans | `/ralph-list` | - |
| Check status | `/ralph-status` | - |

## Best Practices

### Planning
1. **Be Specific**: Use clear plan names and descriptions
2. **Define Success**: Include completion promises in prompts
3. **Organize Tasks**: Use dependencies to order tasks logically
4. **Validate**: Always check plan structure before executing

### Execution
1. **Start Small**: Test with simple plans first
2. **Monitor**: Use `/ralph-status` to track progress
3. **Review**: Check plan content before executing with `/ralph-list`
4. **Resume**: Interrupted executions can be resumed by re-running `/ralph`

### Integration
1. **Use Commands**: Prefer slash commands over direct CLI
2. **Combine**: Use multiple commands in workflows
3. **Track**: Monitor GitHub issues and PRs
4. **Validate**: Verify plans execute correctly

## Requirements

- OpenCode with slash command support
- Ralph agents registered (`.agent/agents/`)
- Plan files accessible (`ralph/plans/`)

---

**Created**: January 19, 2026
**Commands**: 4 slash commands
**Integration**: Full OpenCode TUI support
