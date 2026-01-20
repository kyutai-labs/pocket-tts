# Slash Command: Ralph Execution Status

```typescript
// Check Ralph execution status
task(
  subagent_type="coder-agent",
  description="Check Ralph plan execution status",
  prompt="Check the Ralph Wiggum execution state from .opencode/ralph-state.json. Display:\n- Current plan name\n- Branch being worked on\n- Completed tasks\n- Current task\n- GitHub issues created\n- Pull request number\n- Execution status (running/paused/completed/failed)"
)
```

## Usage

```bash
# Check execution status
/ralph-status
```

## Output

Displays the current state of Ralph plan execution including:
- Plan being executed
- Number of tasks completed
- Current task (if any)
- GitHub issue numbers
- Pull request information
- Overall status

## See Also

- `/ralph-create` - Create a new Ralph plan
- `/ralph` - Execute a Ralph plan
- `/ralph-list` - List available plans

## Documentation

- [AGENT-README.md](../../.agent/agents/README.md) - Agent overview
- [OPENCODE-TUI.md](../../.agent/agents/OPENCODE-TUI.md) - TUI usage guide
- [RALPH-README.md](../../ralph/RALPH-README.md) - Executor details
