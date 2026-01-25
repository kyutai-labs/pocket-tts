#!/usr/bin/env python3
"""
Profile NumPy usage in Pocket TTS to identify optimization opportunities.

This script profiles the application to find:
1. Most frequently used numpy_rs functions
2. Performance bottlenecks
3. Opportunities for lazy loading
4. SIMD optimization candidates
"""

import cProfile
import pstats
import time
from io import StringIO
from typing import Dict, List, Tuple
import numpy as np

# Import the modules we want to profile
from pocket_tts.numpy_rs import (
    array,
    arange,
    linspace,
    concatenate,
    clip,
    compute_min,
    compute_std,
    compute_var,
    dot_vec,
    eye,
    hstack,
    vstack,
    ones_vec,
    zeros_vec,
    reshape_vec,
    transpose_2d,
)


def benchmark_function(func, *args, **kwargs) -> Tuple[float, any]:
    """Benchmark a function and return time and result."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return (end_time - start_time), result


def profile_numpy_operations():
    """Profile common NumPy operations with different data sizes."""
    print("=== NumPy Operations Profiling ===\n")

    # Test different data sizes
    sizes = [100, 1000, 10000, 100000]

    results = {}

    for size in sizes:
        print(f"Testing with size {size}:")
        print("-" * 40)

        size_results = {}

        # Test array creation
        try:
            # arange
            time_taken, result = benchmark_function(arange, 0, size, 1)
            size_results["arange"] = time_taken
            print(f"  arange(0, {size}, 1): {time_taken:.6f}s")

            # zeros
            time_taken, result = benchmark_function(zeros_vec, size)
            size_results["zeros_vec"] = time_taken
            print(f"  zeros_vec({size}): {time_taken:.6f}s")

            # ones
            time_taken, result = benchmark_function(ones_vec, size)
            size_results["ones_vec"] = time_taken
            print(f"  ones_vec({size}): {time_taken:.6f}s")

            # linspace
            time_taken, result = benchmark_function(linspace, 0, 1, size)
            size_results["linspace"] = time_taken
            print(f"  linspace(0, 1, {size}): {time_taken:.6f}s")

            # Test array operations (if size is reasonable)
            if size <= 10000:
                # Create test data
                arr1 = array(np.random.randn(size))
                arr2 = array(np.random.randn(size))

                # concatenate
                time_taken, result = benchmark_function(concatenate, [arr1, arr2])
                size_results["concatenate"] = time_taken
                print(f"  concatenate([{size}, {size}]): {time_taken:.6f}s")

                # compute functions
                time_taken, result = benchmark_function(compute_min, arr1)
                size_results["compute_min"] = time_taken
                print(f"  compute_min({size}): {time_taken:.6f}s")

                time_taken, result = benchmark_function(compute_std, arr1)
                size_results["compute_std"] = time_taken
                print(f"  compute_std({size}): {time_taken:.6f}s")

                time_taken, result = benchmark_function(compute_var, arr1)
                size_results["compute_var"] = time_taken
                print(f"  compute_var({size}): {time_taken:.6f}s")

                # dot product
                time_taken, result = benchmark_function(dot_vec, arr1, arr2)
                size_results["dot_vec"] = time_taken
                print(f"  dot_vec({size}, {size}): {time_taken:.6f}s")

        except Exception as e:
            print(f"  Error in operations: {e}")

        results[size] = size_results
        print()

    return results


def profile_rust_vs_numpy():
    """Compare Rust vs NumPy performance."""
    print("=== Rust vs NumPy Performance Comparison ===\n")

    sizes = [1000, 10000, 100000]

    for size in sizes:
        print(f"Size: {size}")
        print("-" * 30)

        # Test data
        data = np.random.randn(size).astype(np.float32)

        # Test arange
        rust_time, rust_result = benchmark_function(arange, 0, size, 1)
        numpy_time, numpy_result = benchmark_function(np.arange, size)
        speedup = numpy_time / rust_time if rust_time > 0 else float("inf")
        print(
            f"arange: Rust {rust_time:.6f}s, NumPy {numpy_time:.6f}s, speedup: {speedup:.2f}x"
        )

        # Test min
        rust_arr = array(data)
        rust_time, rust_result = benchmark_function(compute_min, rust_arr)
        numpy_time, numpy_result = benchmark_function(np.min, data)
        speedup = numpy_time / rust_time if rust_time > 0 else float("inf")
        print(
            f"min: Rust {rust_time:.6f}s, NumPy {numpy_time:.6f}s, speedup: {speedup:.2f}x"
        )

        # Test std
        rust_time, rust_result = benchmark_function(compute_std, rust_arr)
        numpy_time, numpy_result = benchmark_function(np.std, data)
        speedup = numpy_time / rust_time if rust_time > 0 else float("inf")
        print(
            f"std: Rust {rust_time:.6f}s, NumPy {numpy_time:.6f}s, speedup: {speedup:.2f}x"
        )

        print()


def profile_memory_usage():
    """Profile memory usage patterns."""
    print("=== Memory Usage Profiling ===\n")

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Baseline memory: {baseline_memory:.2f} MB")

        # Create large arrays
        arrays = []
        for i in range(10):
            arr = array(np.random.randn(10000))
            arrays.append(arr)

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"After creating 10 arrays (10k each): {current_memory:.2f} MB")
        print(f"Memory increase: {current_memory - baseline_memory:.2f} MB")

        # Clear arrays
        del arrays
        import gc

        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"After cleanup: {final_memory:.2f} MB")

    except ImportError:
        print("psutil not available, skipping memory profiling")


def profile_lazy_loading():
    """Test lazy loading performance."""
    print("=== Lazy Loading Analysis ===\n")

    # Time the import
    start_time = time.perf_counter()
    from pocket_tts import numpy_rs

    import_time = time.perf_counter() - start_time
    print(f"Import time: {import_time:.6f}s")

    # Test if library is available
    print(f"Rust library available: {numpy_rs._AVAILABLE}")

    if not numpy_rs._AVAILABLE:
        print("Rust library not available - all operations will use NumPy fallback")
        return

    # Time first call vs subsequent calls
    test_data = array([1, 2, 3, 4, 5])

    # First call
    start_time = time.perf_counter()
    result1 = numpy_rs.compute_min(test_data)
    first_call_time = time.perf_counter() - start_time

    # Subsequent call
    start_time = time.perf_counter()
    result2 = numpy_rs.compute_min(test_data)
    subsequent_call_time = time.perf_counter() - start_time

    print(f"First call time: {first_call_time:.6f}s")
    print(f"Subsequent call time: {subsequent_call_time:.6f}s")
    print(f"Speedup after warmup: {first_call_time / subsequent_call_time:.2f}x")


def identify_hot_paths():
    """Identify potential hot paths in the application."""
    print("=== Hot Path Analysis ===\n")

    # Simulate common TTS operations
    operations = [
        ("Array creation (small)", lambda: array([1, 2, 3, 4, 5])),
        ("Array creation (medium)", lambda: array(np.random.randn(100))),
        ("Array creation (large)", lambda: array(np.random.randn(10000))),
        ("Arange (small)", lambda: arange(0, 100, 1)),
        ("Arange (medium)", lambda: arange(0, 1000, 1)),
        ("Arange (large)", lambda: arange(0, 10000, 1)),
        (
            "Concatenate (small)",
            lambda: concatenate([array([1, 2, 3]), array([4, 5, 6])]),
        ),
        (
            "Concatenate (medium)",
            lambda: concatenate(
                [array(np.random.randn(100)), array(np.random.randn(100))]
            ),
        ),
        ("Compute min (small)", lambda: compute_min(array(np.random.randn(100)))),
        ("Compute std (small)", lambda: compute_std(array(np.random.randn(100)))),
        (
            "Dot product (small)",
            lambda: dot_vec(array(np.random.randn(100)), array(np.random.randn(100))),
        ),
    ]

    print("Operation performance (100 iterations each):")
    print("-" * 50)

    for name, operation in operations:
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            operation()
            times.append(time.perf_counter() - start_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(
            f"{name:30s}: avg {avg_time:.6f}s, min {min_time:.6f}s, max {max_time:.6f}s"
        )


def main():
    """Run all profiling analyses."""
    print("Pocket TTS NumPy Integration Profiling")
    print("=" * 50)
    print()

    # Run all profiling functions
    profile_numpy_operations()
    profile_rust_vs_numpy()
    profile_memory_usage()
    profile_lazy_loading()
    identify_hot_paths()

    print("\n=== Recommendations ===")
    print("1. Focus optimization on frequently used operations")
    print("2. Implement lazy loading for expensive operations")
    print("3. Consider SIMD optimizations for vector operations")
    print("4. Add caching for repeated computations")
    print("5. Optimize memory allocation patterns")


if __name__ == "__main__":
    main()
