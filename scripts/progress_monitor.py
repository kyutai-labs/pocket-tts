#!/usr/bin/env python3
"""
Progress monitoring utility for long-running scripts.

Provides progress bars, status updates, and heartbeat indicators
to prevent scripts from appearing frozen.
"""

import sys
import time
import threading
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ProgressState:
    """Current progress state."""

    current: int
    total: int
    message: str
    start_time: datetime
    last_update: datetime


class ProgressMonitor:
    """Progress monitoring with visual indicators and heartbeat."""

    def __init__(
        self,
        total: int = 100,
        width: int = 50,
        show_heartbeat: bool = True,
        heartbeat_interval: float = 2.0,
        show_eta: bool = True,
    ):
        self.total = total
        self.width = width
        self.show_heartbeat = show_heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.show_eta = show_eta

        self.state = ProgressState(
            current=0,
            total=total,
            message="Starting...",
            start_time=datetime.now(),
            last_update=datetime.now(),
        )

        self.heartbeat_thread = None
        self.heartbeat_running = False
        self.last_heartbeat = ""

    def start(self, message: str = "Starting..."):
        """Start progress monitoring."""
        self.state.message = message
        self.state.start_time = datetime.now()
        self.state.last_update = datetime.now()

        if self.show_heartbeat:
            self.heartbeat_running = True
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_worker, daemon=True
            )
            self.heartbeat_thread.start()

        self._update_display()

    def update(self, current: int, message: Optional[str] = None):
        """Update progress."""
        self.state.current = min(current, self.total)
        if message:
            self.state.message = message
        self.state.last_update = datetime.now()
        self._update_display()

    def increment(self, amount: int = 1, message: Optional[str] = None):
        """Increment progress by amount."""
        self.update(self.state.current + amount, message)

    def finish(self, message: str = "Complete!"):
        """Mark as complete."""
        self.update(self.total, message)
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        print()  # New line after progress bar

    def _update_display(self):
        """Update the progress display."""
        progress = self.state.current / self.state.total if self.state.total > 0 else 0

        # Progress bar
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)

        # Percentage
        pct = progress * 100

        # Time info
        elapsed = datetime.now() - self.state.start_time
        elapsed_str = str(elapsed).split(".")[0]  # Remove microseconds

        eta_str = ""
        if self.show_eta and progress > 0:
            remaining = elapsed / progress - elapsed
            if remaining.total_seconds() > 0:
                eta_str = f" ETA: {str(remaining).split('.')[0]}"

        # Build display line
        line = f"\r[{bar}] {pct:5.1f}% ({self.state.current}/{self.state.total}) {elapsed_str}{eta_str}"

        if self.state.message:
            # Add message on next line if it exists
            full_line = f"{line}\n  {self.state.message}"
        else:
            full_line = line

        # Clear previous lines and print new
        sys.stdout.write("\r" + " " * 200 + "\r")  # Clear line
        sys.stdout.write(full_line)
        sys.stdout.flush()

    def _heartbeat_worker(self):
        """Background thread that shows heartbeat indicators."""
        while self.heartbeat_running:
            time.sleep(self.heartbeat_interval)
            if self.heartbeat_running:
                self._show_heartbeat()

    def _show_heartbeat(self):
        """Show a heartbeat indicator to show the script is still running."""
        now = datetime.now()
        elapsed = now - self.state.start_time
        idle = now - self.state.last_update

        # Only show heartbeat if no recent updates
        if idle.total_seconds() > self.heartbeat_interval:
            heartbeat = f"⚡ Still working... ({str(elapsed).split('.')[0]} elapsed)"
            if heartbeat != self.last_heartbeat:
                sys.stdout.write(f"\n{heartbeat}")
                sys.stdout.flush()
                self.last_heartbeat = heartbeat


class TaskProgress:
    """Context manager for task-based progress tracking."""

    def __init__(
        self,
        task_name: str,
        total_steps: int,
        show_progress: bool = True,
        step_descriptions: Optional[Dict[int, str]] = None,
    ):
        self.task_name = task_name
        self.total_steps = total_steps
        self.show_progress = show_progress
        self.step_descriptions = step_descriptions or {}
        self.monitor = ProgressMonitor(total=total_steps) if show_progress else None
        self.current_step = 0

    def __enter__(self):
        if self.monitor:
            self.monitor.start(f"Starting {self.task_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor:
            message = "Complete!" if exc_type is None else f"Error: {exc_val}"
            self.monitor.finish(message)

    def step(self, step_num: Optional[int] = None, message: Optional[str] = None):
        """Advance to a specific step."""
        if not self.monitor:
            return

        if step_num is None:
            step_num = self.current_step + 1

        self.current_step = step_num

        # Get description for this step
        description = message or self.step_descriptions.get(
            step_num, f"Step {step_num}"
        )
        self.monitor.update(step_num, description)


def with_progress(
    func: Callable,
    items: list,
    description: str = "Processing",
    show_progress: bool = True,
):
    """Decorator/function to add progress to iteration."""
    if not show_progress or len(items) == 0:
        return func(items)

    monitor = ProgressMonitor(total=len(items))
    monitor.start(f"Starting {description}...")

    results = []
    for i, item in enumerate(items):
        result = func(item)
        results.append(result)

        # Update progress
        item_desc = f"{description} item {i + 1}/{len(items)}"
        if hasattr(item, "__name__"):
            item_desc = f"{description}: {item.__name__}"
        elif hasattr(item, "name"):
            item_desc = f"{description}: {item.name}"

        monitor.update(i + 1, item_desc)

    monitor.finish(f"{description} complete!")
    return results


# Simple heartbeat for scripts without detailed progress
class SimpleHeartbeat:
    """Simple heartbeat indicator for long-running operations."""

    def __init__(self, interval: float = 5.0, message: str = "Working..."):
        self.interval = interval
        self.message = message
        self.running = False
        self.thread = None
        self.start_time = None

    def start(self):
        """Start heartbeat."""
        self.running = True
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        print(f"Started: {self.message}")

    def stop(self, message: str = "Done!"):
        """Stop heartbeat."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        elapsed = datetime.now() - self.start_time
        print(f"\n{message} (took {str(elapsed).split('.')[0]})")

    def _heartbeat(self):
        """Heartbeat thread function."""
        while self.running:
            time.sleep(self.interval)
            if self.running:
                elapsed = datetime.now() - self.start_time
                print(f"⚡ {self.message} ({str(elapsed).split('.')[0]} elapsed)")


# Example usage and testing
if __name__ == "__main__":
    # Test progress monitor
    print("Testing ProgressMonitor...")

    with TaskProgress(
        "Test Task",
        5,
        step_descriptions={
            1: "Initializing",
            2: "Loading data",
            3: "Processing",
            4: "Saving results",
            5: "Cleaning up",
        },
    ) as task:
        for i in range(1, 6):
            time.sleep(1)  # Simulate work
            task.step(i)

    # Test simple heartbeat
    print("\nTesting SimpleHeartbeat...")
    heartbeat = SimpleHeartbeat(interval=2.0, message="Background processing")
    heartbeat.start()
    time.sleep(8)
    heartbeat.stop("Background processing complete")
