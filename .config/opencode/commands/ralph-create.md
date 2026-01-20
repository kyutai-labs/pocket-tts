# Slash Command: Ralph Plan Creator

```typescript
// Invoke Plan Creator Agent
task(
  subagent_type="ralph-plan-creator",
  description="Create Ralph plan interactively",
  prompt="Create a Ralph Wiggum plan file for development work. You will be prompted for:\n- Plan name\n- Plan description\n- Tasks (title, prompt, dependencies)\n\nEach task will be automatically assigned an ID (task-1, task-2, etc.). The plan will be validated and saved to ralph/plans/{plan-name}.json."
)
```

## Usage

```bash
# Create a plan for a new feature
/ralph-create

# Create a plan with specific prompt
/ralph-create "Create a plan for implementing a new REST API endpoint"
```

## Examples

```bash
# Plan unit tests
/ralph-create "Create a plan for adding comprehensive unit tests to TTS model"

# Plan feature implementation
/ralph-create "Create a plan for implementing streaming audio output"

# Plan bug fix
/ralph-create "Create a plan for fixing audio clipping bug"
```

## See Also

- `/ralph` - Execute a Ralph plan
- `/ralph-status` - Check execution status
- `/ralph-list` - List available plans

## Documentation

- [AGENT-README.md](.agent/agents/README.md) - Agent overview
- [OPENCODE-TUI.md](.agent/agents/OPENCODE-TUI.md) - TUI usage guide
