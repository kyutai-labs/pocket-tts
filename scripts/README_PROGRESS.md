# Progress Monitoring for Long-Running Scripts

This directory contains utilities to add progress indicators and heartbeat monitoring to long-running scripts, preventing them from appearing frozen.

## Overview

The progress monitoring system provides several tools:

- **ProgressMonitor**: Detailed progress bars with ETA and status messages
- **TaskProgress**: Context manager for multi-step workflows
- **SimpleHeartbeat**: Basic heartbeat for unknown-duration tasks
- **with_progress**: Decorator/function for processing collections

## Quick Start

### Basic Progress Bar

```python
from progress_monitor import ProgressMonitor

monitor = ProgressMonitor(total=100, width=50)
monitor.start("Processing items...")

for i in range(100):
    # Do work here
    monitor.update(i + 1, f"Processing item {i + 1}/100")

monitor.finish("Complete!")
```

### Task-Based Progress

```python
from progress_monitor import TaskProgress

with TaskProgress("Data Pipeline", total_steps=5, step_descriptions={
    1: "Loading data",
    2: "Validating format",
    3: "Processing records",
    4: "Generating report",
    5: "Saving results"
}) as task:

    # Step 1
    load_data()
    task.step(1)

    # Step 2
    validate_data()
    task.step(2)

    # Continue with remaining steps...
```

### Simple Heartbeat

```python
from progress_monitor import SimpleHeartbeat

heartbeat = SimpleHeartbeat(interval=10.0, message="Background processing")
heartbeat.start()

# Long-running task with unknown duration
process_large_dataset()

heartbeat.stop("Processing complete!")
```

### Processing Collections

```python
from progress_monitor import with_progress

def process_item(item):
    # Process individual item
    return result

items = list_of_items
results = with_progress(process_item, items, "Processing items")
```

## Integration with Existing Scripts

### Golden Data Generator

The golden data generator now includes progress tracking:

```bash
# With progress bars (default)
python scripts/generate_golden_data.py --cases 100

# Without progress bars (uses heartbeat instead)
python scripts/generate_golden_data.py --cases 100 --no-progress
```

### Ralph Dashboard Watch

The dashboard watcher now shows heartbeat indicators:

```bash
# With heartbeat (default)
python scripts/ralph-dashboard-watch.py

# Without heartbeat
python scripts/ralph-dashboard-watch.py --no-heartbeat

# With HTTP server and heartbeat
python scripts/ralph-dashboard-watch.py --serve --port 8735
```

## Features

### ProgressMonitor Features
- Visual progress bar with customizable width
- Percentage completion and item counts
- Elapsed time and ETA calculation
- Status messages for each update
- Automatic heartbeat when no recent updates
- Thread-safe operation

### TaskProgress Features
- Context manager for automatic cleanup
- Step descriptions for better UX
- Nested progress tracking support
- Error handling with graceful exit

### SimpleHeartbeat Features
- Lightweight heartbeat for unknown-duration tasks
- Configurable interval
- Automatic elapsed time tracking
- Clean start/stop interface

## Configuration Options

### ProgressMonitor
- `total`: Total number of items/steps
- `width`: Progress bar width (default: 50)
- `show_heartbeat`: Enable heartbeat indicators (default: True)
- `heartbeat_interval`: Heartbeat frequency in seconds (default: 2.0)
- `show_eta`: Show estimated time remaining (default: True)

### TaskProgress
- `task_name`: Human-readable task name
- `total_steps`: Number of steps in the task
- `show_progress`: Enable progress display (default: True)
- `step_descriptions`: Optional mapping of step numbers to descriptions

### SimpleHeartbeat
- `interval`: Heartbeat interval in seconds (default: 5.0)
- `message`: Descriptive message for the heartbeat

## Best Practices

1. **Choose the right tool**:
   - Use `ProgressMonitor` for known-size collections
   - Use `TaskProgress` for multi-step workflows
   - Use `SimpleHeartbeat` for unknown-duration tasks

2. **Provide meaningful messages**:
   - Update status messages to reflect current operation
   - Use descriptive step names in `TaskProgress`

3. **Handle cleanup properly**:
   - Always call `finish()` or use context managers
   - Handle keyboard interrupts gracefully

4. **Performance considerations**:
   - Don't update progress too frequently (updates take time)
   - Use reasonable heartbeat intervals (5-30 seconds)

5. **Error handling**:
   - Progress monitors handle exceptions gracefully
   - Always stop heartbeats in `finally` blocks

## Examples

See `demo_progress.py` for comprehensive examples of all features:

```bash
python scripts/demo_progress.py
```

This demonstrates:
- Basic progress bars
- Task-based workflows
- Simple heartbeats
- Collection processing
- Nested progress tracking
- Error handling

## Troubleshooting

### Progress bars not showing
- Ensure the script imports `progress_monitor` correctly
- Check that progress is being updated
- Verify the terminal supports ANSI codes

### Heartbeat not working
- Check that the heartbeat thread isn't being blocked
- Ensure the interval is reasonable (not too short/long)
- Make sure `stop()` is called when done

### Performance issues
- Reduce update frequency for very fast operations
- Use larger heartbeat intervals for long-running tasks
- Consider disabling progress bars entirely with `--no-progress`

## Adding Progress to New Scripts

1. Import the required classes:
   ```python
   from progress_monitor import ProgressMonitor, TaskProgress, SimpleHeartbeat
   ```

2. Add fallback imports for compatibility:
   ```python
   try:
       from progress_monitor import ProgressMonitor
   except ImportError:
       # Fallback implementation
       class ProgressMonitor:
           # Minimal fallback...
   ```

3. Choose the appropriate tool based on your use case

4. Add command-line options for disabling progress if needed:
   ```python
   parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
   ```

5. Test with the demo script to verify integration
