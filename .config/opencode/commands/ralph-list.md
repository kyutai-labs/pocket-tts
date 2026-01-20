# Slash Command: List Ralph Plans

```typescript
// List available Ralph plans
task(
  subagent_type="coder-agent",
  description="List available Ralph plans",
  prompt="List all plan files in ralph/plans/ directory. Display:\n- Plan name\n- Description\n- Number of tasks\n- File path"
)
```

## Usage

```bash
# List all available plans
/ralph-list
```

## Output

Lists all plan files found in `ralph/plans/` directory with their details.

## See Also

- `/ralph-create` - Create a new Ralph plan
- `/ralph` - Execute a Ralph plan
- `/ralph-status` - Check execution status

## Documentation

- [AGENT-README.md](../../.agent/agents/README.md) - Agent overview
- [OPENCODE-TUI.md](../../.agent/agents/OPENCODE-TUI.md) - TUI usage guide
