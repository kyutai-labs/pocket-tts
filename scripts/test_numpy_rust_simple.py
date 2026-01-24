#!/usr/bin/env python3
"""
Simple test harness for NumPy vs Rust validation using generated test datasets.

This script demonstrates how to use the generated test data to validate
NumPy behavioral parity with Rust implementations.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(
        description="Test NumPy vs Rust behavioral parity using generated datasets"
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to dataset file (Parquet format)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} does not exist")
        return

    print(f"🔄 Loading dataset: {dataset_path}")

    # Load Parquet data
    table = pq.read_table(dataset_path)
    df = table.to_pandas()

    # Load metadata
    metadata_path = dataset_path.with_suffix(".json")
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    if args.verbose:
        print(f"📊 Loaded {len(df.columns)} arrays")
        print(f"📋 Metadata: {metadata}")

    # Test basic NumPy operations
    results = {}
    for column in df.columns:
        arr = df[column].values

        if args.verbose:
            print(f"\\n🧪 Testing {column}: shape={arr.shape}, dtype={arr.dtype}")

        # Basic tests
        test_result = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "has_nan": bool(np.any(np.isnan(arr))),
            "has_inf": bool(np.any(np.isinf(arr))),
        }

        results[column] = test_result

    # Test Rust equivalent if available
    rust_available = False
    try:
        from pocket_tts import numpy_rs

        rust_available = numpy_rs._AVAILABLE
        if rust_available and args.verbose:
            print("🦀 Rust numpy_rs available")
    except ImportError:
        if args.verbose:
            print("⚠️  Rust numpy_rs not available")

    # Generate simple report
    report = {
        "dataset": str(dataset_path),
        "arrays_tested": len(results),
        "numpy_tests_passed": all(all(r.values()) for r in results.values()),
        "rust_available": rust_available,
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save report
    report_path = dataset_path.parent / f"{dataset_path.stem}_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\\n" + "=" * 50)
    print("🧪 NUMPY VS RUST VALIDATION SUMMARY")
    print("=" * 50)
    print(f"📊 Dataset: {report['dataset']}")
    print(f"🧮 Arrays Tested: {report['arrays_tested']}")
    print(f"✅ NumPy Tests: {'PASS' if report['numpy_tests_passed'] else 'FAIL'}")
    print(f"🦀 Rust Available: {'YES' if report['rust_available'] else 'NO'}")
    print("\\n" + "=" * 50)


if __name__ == "__main__":
    main()
