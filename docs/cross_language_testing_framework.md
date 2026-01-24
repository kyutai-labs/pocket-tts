# Cross-Language Testing Framework

Framework for validating behavioral parity between NumPy and rust-numpy implementations.

## Overview

This framework executes identical test cases in both Python/NumPy and Rust, compares results with configurable tolerances, and generates detailed parity reports.

## Quick Start

```bash
# Run with sample test suite
python scripts/cross_language_test_runner.py

# Run with custom test suite
python scripts/cross_language_test_runner.py --test-file test_suites/my_tests.json
```

## Features

- **Automated Test Execution**: Runs tests in both NumPy and Rust
- **Result Comparison**: Configurable numerical tolerances
- **Performance Tracking**: Execution time for both implementations
- **Detailed Reporting**: JSON output with pass/fail status and error details
- **Extensible**: Easy to add new test functions and suites

## Architecture

The framework consists of:

1. **Test Runner** (`cross_language_test_runner.py`): Orchestrates test execution
2. **Test Suites**: JSON files defining test cases
3. **Comparison Engine**: Compares NumPy and Rust results
4. **Report Generator**: Creates detailed test reports

## Test Suite Format

```json
[
  {
    "name": "array_creation_basic",
    "function": "array",
    "inputs": [[1, 2, 3, 4, 5]]
  },
  {
    "name": "array_sum",
    "function": "sum",
    "inputs": [[1, 2, 3, 4, 5]]
  }
]
```

## Current Status

### ✅ Implemented
- Python test harness with NumPy execution
- Test suite loading from JSON
- Result comparison engine
- Basic reporting (JSON format)
- Sample test suite generation

### ⚠️ Partial Implementation
- Rust test execution (currently placeholder)
- Performance benchmarking (basic timing only)

### ❌ Not Yet Implemented
- Rust FFI/CLI integration
- HTML/PDF report generation
- Parallel test execution
- Regression detection
- Automatic GAP_ANALYSIS.md updates

## Integration with CI/CD

```yaml
- name: Run cross-language tests
  run: |
    python scripts/cross_language_test_runner.py \
      --test-file test_suites/comprehensive.json \
      --output results/ci_results.json
```

## Future Work

1. Complete Rust test execution integration
2. Add comprehensive function coverage
3. Implement detailed performance benchmarking
4. Generate visual HTML/PDF reports
5. Integrate with GAP_ANALYSIS.md updates

## Related Documentation

- [GAP_ANALYSIS.md](../rust-numpy/GAP_ANALYSIS.md)
- [PARITY.md](../rust-numpy/PARITY.md)
- [INVENTORY.md](../rust-numpy/INVENTORY.md)
