# Cross-Language Testing Framework (V1)

This folder contains a minimal, JSON-driven testing framework to compare NumPy
outputs with rust-numpy equivalents. The V1 scope focuses on array creation,
basic reductions, and a small set of binary/comparison ufuncs.

## How it works

1. Define test inputs and operations in `test_cases/basic.json`.
2. Generate NumPy golden data:

```bash
python rust-numpy/testing_framework/generate_golden_data.py
```

3. Run Rust tests (golden data is loaded by `tests/golden_tests.rs`).

## Output

Generated artifacts live in `testing_framework/output/`:

- `golden_data.json` - NumPy reference results for rust-numpy tests
- `report.json` - Summary counts for quick inspection

This V1 format is intentionally small; future expansions can add more suites,
operations, and richer reporting without changing the core schema.
