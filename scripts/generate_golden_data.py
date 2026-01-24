#!/usr/bin/env python3
"""
Golden Data Generator for rust-numpy conformance testing.

This script generates test data using NumPy and serializes inputs/outputs
so that the Rust implementation can verify exact parity.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# Import progress monitoring
try:
    from progress_monitor import TaskProgress, SimpleHeartbeat
except ImportError:
    # Fallback if progress_monitor is not available
    class TaskProgress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            pass

        def step(self, *args, **kwargs):
            pass

    class SimpleHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self, *args, **kwargs):
            pass


def generate_random_arrays(
    rng: np.random.Generator, shapes: List[Tuple[int, ...]], dtypes: List[str]
) -> List[np.ndarray]:
    """Generate random arrays with various shapes and dtypes."""
    arrays = []

    for shape in shapes:
        for dtype in dtypes:
            if dtype == "bool":
                arr = rng.random(shape) > 0.5
            elif dtype in ["int8", "int16", "int32", "int64"]:
                info = np.iinfo(dtype)
                arr = rng.integers(info.min, info.max + 1, shape, dtype=dtype)
            elif dtype in ["uint8", "uint16", "uint32", "uint64"]:
                info = np.iinfo(dtype)
                arr = rng.integers(0, info.max + 1, shape, dtype=dtype)
            elif dtype in ["float32", "float64"]:
                arr = rng.random(shape).astype(dtype)
            elif dtype == "complex64":
                arr = rng.random(shape) + 1j * rng.random(shape)
                arr = arr.astype("complex64")
            elif dtype == "complex128":
                arr = rng.random(shape) + 1j * rng.random(shape)
            else:
                continue

            arrays.append(arr)

    return arrays


def array_to_dict(arr: np.ndarray) -> Dict[str, Any]:
    """Convert NumPy array to serializable dictionary."""
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tolist(),
    }


def test_basic_operations(arr: np.ndarray) -> Dict[str, Any]:
    """Test basic array operations."""
    results = {}

    # Basic operations that should match exactly
    if arr.size > 0:
        try:
            results["sum"] = array_to_dict(np.sum(arr))
            results["mean"] = array_to_dict(np.mean(arr))
            results["min"] = array_to_dict(np.min(arr))
            results["max"] = array_to_dict(np.max(arr))

            if arr.ndim > 0:
                results["argmin"] = array_to_dict(np.argmin(arr))
                results["argmax"] = array_to_dict(np.argmax(arr))

            # Element-wise operations
            results["negative"] = array_to_dict(-arr)
            results["abs"] = array_to_dict(np.abs(arr))

            if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(
                arr.dtype, np.complexfloating
            ):
                results["sqrt"] = array_to_dict(
                    np.sqrt(np.abs(arr))
                )  # sqrt of abs to avoid complex
                results["square"] = array_to_dict(np.square(arr))

            # Reshape operations
            if arr.size > 1:
                try:
                    flat_shape = (arr.size,)
                    results["flatten"] = array_to_dict(arr.flatten())
                    results["reshape_flat"] = array_to_dict(arr.reshape(flat_shape))
                except Exception:
                    pass

        except Exception as e:
            results["error"] = str(e)

    return results


def test_ufunc_operations(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, Any]:
    """Test universal functions with two arrays."""
    results = {}

    if arr1.shape == arr2.shape and arr1.size > 0:
        try:
            # Arithmetic operations
            results["add"] = array_to_dict(arr1 + arr2)
            results["subtract"] = array_to_dict(arr1 - arr2)
            results["multiply"] = array_to_dict(arr1 * arr2)

            # Avoid division by zero
            if np.any(arr2 != 0):
                safe_divide = np.divide(
                    arr1, arr2, out=np.full_like(arr1, np.nan), where=arr2 != 0
                )
                results["divide"] = array_to_dict(safe_divide)

            # Comparison operations
            results["equal"] = array_to_dict(arr1 == arr2)
            results["less"] = array_to_dict(arr1 < arr2)
            results["greater"] = array_to_dict(arr1 > arr2)

            # Logical operations
            if arr1.dtype in [np.bool_, bool] and arr2.dtype in [np.bool_, bool]:
                results["logical_and"] = array_to_dict(np.logical_and(arr1, arr2))
                results["logical_or"] = array_to_dict(np.logical_or(arr1, arr2))
                results["logical_xor"] = array_to_dict(np.logical_xor(arr1, arr2))

        except Exception as e:
            results["error"] = str(e)

    return results


def test_reduction_operations(arr: np.ndarray) -> Dict[str, Any]:
    """Test reduction operations along axes."""
    results = {}

    if arr.ndim > 1 and arr.size > 0:
        try:
            # Sum along axes
            for axis in range(arr.ndim):
                results[f"sum_axis_{axis}"] = array_to_dict(np.sum(arr, axis=axis))
                results[f"mean_axis_{axis}"] = array_to_dict(np.mean(arr, axis=axis))
                results[f"min_axis_{axis}"] = array_to_dict(np.min(arr, axis=axis))
                results[f"max_axis_{axis}"] = array_to_dict(np.max(arr, axis=axis))

                # Argmin/argmax along axis
                if axis < arr.shape[axis]:
                    results[f"argmin_axis_{axis}"] = array_to_dict(
                        np.argmin(arr, axis=axis)
                    )
                    results[f"argmax_axis_{axis}"] = array_to_dict(
                        np.argmax(arr, axis=axis)
                    )

        except Exception as e:
            results["error"] = str(e)

    return results


def generate_test_cases(
    rng: np.random.Generator, num_cases: int = 50
) -> List[Dict[str, Any]]:
    """Generate comprehensive test cases."""
    test_cases = []

    # Define test parameters
    shapes = [
        (),  # scalar
        (1,),
        (5,),
        (3, 4),
        (2, 3, 4),
        (2, 2, 2, 2),
    ]

    dtypes = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]

    # Generate arrays with progress tracking
    arrays = []
    with TaskProgress("Generating arrays", len(shapes) * len(dtypes)) as task:
        for i, shape in enumerate(shapes):
            for j, dtype in enumerate(dtypes):
                step_num = i * len(dtypes) + j + 1
                task.step(step_num, f"Creating {dtype} array with shape {shape}")

                if dtype == "bool":
                    arr = rng.random(shape) > 0.5
                elif dtype in ["int8", "int16", "int32", "int64"]:
                    info = np.iinfo(dtype)
                    arr = rng.integers(info.min, info.max + 1, shape, dtype=dtype)
                elif dtype in ["uint8", "uint16", "uint32", "uint64"]:
                    info = np.iinfo(dtype)
                    arr = rng.integers(0, info.max + 1, shape, dtype=dtype)
                elif dtype in ["float32", "float64"]:
                    arr = rng.random(shape).astype(dtype)
                elif dtype == "complex64":
                    arr = rng.random(shape) + 1j * rng.random(shape)
                    arr = arr.astype("complex64")
                elif dtype == "complex128":
                    arr = rng.random(shape) + 1j * rng.random(shape)
                else:
                    continue

                arrays.append(arr)

    # Process arrays with progress tracking
    with TaskProgress("Processing test cases", min(len(arrays), num_cases)) as task:
        for i, arr in enumerate(arrays[:num_cases]):
            task.step(i + 1, f"Processing case {i + 1}/{min(len(arrays), num_cases)}")

            case = {"id": i, "input": array_to_dict(arr), "operations": {}}

            # Test basic operations
            case["operations"]["basic"] = test_basic_operations(arr)

            # Test reduction operations
            case["operations"]["reductions"] = test_reduction_operations(arr)

            # Test with another array of same shape
            if i > 0:
                prev_arr = arrays[i - 1]
                if prev_arr.shape == arr.shape:
                    case["operations"]["ufuncs"] = test_ufunc_operations(arr, prev_arr)

            test_cases.append(case)

    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden data for rust-numpy conformance testing"
    )
    parser.add_argument(
        "--output", "-o", default="golden_data.json", help="Output file path"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cases", "-n", type=int, default=50, help="Number of test cases to generate"
    )
    parser.add_argument(
        "--output-dir", "-d", help="Output directory (defaults to same as output file)"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )

    args = parser.parse_args()

    # Set up output path
    output_path = Path(args.output)
    if args.output_dir:
        output_path = Path(args.output_dir) / output_path.name

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create random generator with seed
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.cases} test cases with seed {args.seed}...")
    print(f"Output will be saved to: {output_path}")

    # Use simple heartbeat if progress bars are disabled
    if args.no_progress:
        heartbeat = SimpleHeartbeat(interval=10.0, message="Generating golden data")
        heartbeat.start()

    try:
        # Generate test cases
        test_cases = generate_test_cases(rng, args.cases)

        # Create metadata
        metadata = {
            "version": "1.0.0",
            "numpy_version": np.__version__,
            "seed": args.seed,
            "num_cases": len(test_cases),
            "generated_at": str(np.datetime64("now")),
            "description": "Golden data for rust-numpy conformance testing",
        }

        # Combine and save
        golden_data = {"metadata": metadata, "test_cases": test_cases}

        print("Saving data...")
        with open(output_path, "w") as f:
            json.dump(golden_data, f, indent=2)

        print(f"Generated {len(test_cases)} test cases")
        print(f"Output saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    finally:
        if args.no_progress:
            heartbeat.stop("Golden data generation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
