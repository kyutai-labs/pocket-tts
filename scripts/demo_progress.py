#!/usr/bin/env python3
"""
Demonstration of progress monitoring utilities.

This script shows various ways to add progress indicators to long-running tasks.
"""

import time
import random
from progress_monitor import (
    ProgressMonitor,
    TaskProgress,
    SimpleHeartbeat,
    with_progress,
)


def simulate_heavy_computation(duration: float):
    """Simulate a heavy computation that takes 'duration' seconds."""
    time.sleep(duration)


def demo_basic_progress():
    """Demonstrate basic progress monitoring."""
    print("=== Basic Progress Monitor Demo ===")

    monitor = ProgressMonitor(total=10, width=40)
    monitor.start("Processing items...")

    for i in range(10):
        time.sleep(0.5)  # Simulate work
        monitor.update(i + 1, f"Processing item {i + 1}/10")

    monitor.finish("All items processed!")
    print()


def demo_task_progress():
    """Demonstrate task-based progress tracking."""
    print("=== Task Progress Demo ===")

    with TaskProgress(
        "Data Pipeline",
        total_steps=5,
        step_descriptions={
            1: "Loading data from disk",
            2: "Validating data format",
            3: "Processing records",
            4: "Generating report",
            5: "Saving results",
        },
    ) as task:
        # Step 1: Load data
        time.sleep(1)
        task.step(1)

        # Step 2: Validate
        time.sleep(0.8)
        task.step(2)

        # Step 3: Process
        time.sleep(2)
        task.step(3)

        # Step 4: Generate report
        time.sleep(1.2)
        task.step(4)

        # Step 5: Save
        time.sleep(0.5)
        task.step(5)

    print()


def demo_simple_heartbeat():
    """Demonstrate simple heartbeat for unknown duration tasks."""
    print("=== Simple Heartbeat Demo ===")

    heartbeat = SimpleHeartbeat(interval=2.0, message="Background processing")
    heartbeat.start()

    # Simulate a task with unknown duration
    time.sleep(8)

    heartbeat.stop("Background processing complete!")
    print()


def demo_with_progress():
    """Demonstrate the with_progress decorator."""
    print("=== With Progress Demo ===")

    def process_item(item):
        """Process a single item."""
        time.sleep(random.uniform(0.1, 0.3))  # Variable processing time
        return f"Processed {item}"

    items = [f"item_{i}" for i in range(15)]
    results = with_progress(process_item, items, "Processing items")

    print(f"Results: {len(results)} items processed")
    print()


def demo_nested_progress():
    """Demonstrate nested progress tracking."""
    print("=== Nested Progress Demo ===")

    with TaskProgress("Main Workflow", total_steps=3) as main_task:
        # Phase 1: Data collection
        main_task.step(1, "Collecting data from sources")
        with TaskProgress("Data Sources", total_steps=3) as sub_task:
            for i in range(3):
                time.sleep(0.5)
                sub_task.step(i + 1, f"Source {chr(65 + i)}")

        # Phase 2: Processing
        main_task.step(2, "Processing collected data")
        monitor = ProgressMonitor(total=5, width=30)
        monitor.start("Processing batches")
        for i in range(5):
            time.sleep(0.4)
            monitor.update(i + 1, f"Batch {i + 1}/5")
        monitor.finish()

        # Phase 3: Cleanup
        main_task.step(3, "Cleaning up resources")
        time.sleep(1)

    print()


def demo_error_handling():
    """Demonstrate progress monitoring with error handling."""
    print("=== Error Handling Demo ===")

    try:
        with TaskProgress("Risky Operation", total_steps=4) as task:
            for i in range(4):
                time.sleep(0.5)
                if i == 2:
                    raise ValueError("Something went wrong!")
                task.step(i + 1, f"Step {i + 1}")
    except ValueError as e:
        print(f"Caught error: {e}")

    print()


if __name__ == "__main__":
    print("🚀 Progress Monitoring Demonstrations")
    print("=" * 50)

    demos = [
        demo_basic_progress,
        demo_task_progress,
        demo_simple_heartbeat,
        demo_with_progress,
        demo_nested_progress,
        demo_error_handling,
    ]

    for demo in demos:
        try:
            demo()
            input("Press Enter to continue to next demo...")
            print("\n" + "=" * 50 + "\n")
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            continue

    print("🎉 All demos completed!")
