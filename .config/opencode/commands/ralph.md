# Slash Command: Ralph Plan Executor

```typescript
// Invoke Plan Executor Agent
task(
  subagent_type="ralph-plan-executor",
  description="Execute Ralph plan",
  prompt="Execute the Ralph Wiggum plan specified. The executor will:\n1. Parse the plan file\n2. Create a feature branch\n3. Create GitHub issues for each task\n4. Run OpenCode iteratively to complete tasks\n5. Auto-commit changes\n6. Create and merge pull request"
)
```

## Usage

```bash
# Execute a specific plan
/ralph plans/my-plan.json

# Execute with reduced iterations
/ralph plans/my-plan.json max-iterations=10

# Resume interrupted execution
/ralph plans/my-plan.json
```

## Examples

```bash
# Execute unit test plan
/ralph plans/add-unit-tests.json

# Execute feature plan
/ralph plans/implement-api.json

# Resume execution
/ralph plans/feature-x.json
```

## Options

- Specify plan file path as argument
- Execution state is automatically persisted
- Can resume interrupted executions

## See Also

- `/ralph-create` - Create a Ralph plan
- `/ralph-list` - List available plans
- `/ralph-status` - Check execution status

## Documentation

- [AGENT-README.md](.agent/agents/README.md) - Agent overview
- [OPENCODE-TUI.md](.agent/agents/OPENCODE-TUI.md) - TUI usage guide
- [RALPH-README.md](../../ralph/RALPH-README.md) - Executor details
