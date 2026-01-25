#!/usr/bin/env python3
"""
Benchmark the optimized numpy_rs implementation against the original.

This script compares performance between:
1. Original numpy_rs implementation
2. Optimized numpy_rs implementation
3. Pure NumPy implementation
"""

import time
import sys
import numpy as np
from typing import Dict, List, Tuple

# Import both implementations
try:
    from pocket_tts.numpy_rs import (
        arange as orig_arange,
        linspace as orig_linspace,
        concatenate as orig_concatenate,
        compute_min as orig_compute_min,
        compute_std as orig_compute_std,
        zeros_vec as orig_zeros_vec,
        ones_vec as orig_ones_vec,
    )

    ORIGINAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import original numpy_rs: {e}")
    ORIGINAL_AVAILABLE = False

# Import optimized implementation
sys.path.insert(0, ".")
try:
    from optimized_numpy_rs import (
        arange as opt_arange,
        linspace as opt_linspace,
        concatenate as opt_concatenate,
        compute_min as opt_compute_min,
        compute_std as opt_compute_std,
        zeros_vec as opt_zeros_vec,
        ones_vec as opt_ones_vec,
        get_cache_stats,
        clear_caches,
    )

    OPTIMIZED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optimized numpy_rs: {e}")
    OPTIMIZED_AVAILABLE = False


def benchmark_function(func, *args, **kwargs) -> Tuple[float, any]:
    """Benchmark a function and return time and result."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return (end_time - start_time), result


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("=== NumPy Integration Optimization Benchmark ===\n")

    if not ORIGINAL_AVAILABLE and not OPTIMIZED_AVAILABLE:
        print("Neither implementation available for comparison")
        return

    # Test configurations
    test_configs = [
        ("Small arrays", 100),
        ("Medium arrays", 1000),
        ("Large arrays", 10000),
        ("Very large arrays", 100000),
    ]

    operations = [
        ("arange", lambda size: (0, size, 1)),
        ("linspace", lambda size: (0, 1, size)),
        ("zeros_vec", lambda size: (size,)),
        ("ones_vec", lambda size: (size,)),
        (
            "concatenate",
            lambda size: ([np.random.randn(size // 2), np.random.randn(size // 2)],),
        ),
        ("compute_min", lambda size: (np.random.randn(size),)),
        ("compute_std", lambda size: (np.random.randn(size),)),
    ]

    results = {}

    for config_name, size in test_configs:
        print(f"{config_name} (size: {size}):")
        print("-" * 50)

        config_results = {}

        for op_name, op_args_func in operations:
            args = op_args_func(size)

            # Test NumPy baseline
            numpy_time, numpy_result = benchmark_numpy_baseline(op_name, args)
            config_results[f"numpy_{op_name}"] = numpy_time

            # Test original implementation
            if ORIGINAL_AVAILABLE:
                orig_time, orig_result = benchmark_original(op_name, args)
                config_results[f"orig_{op_name}"] = orig_time
                speedup_vs_numpy = (
                    numpy_time / orig_time if orig_time > 0 else float("inf")
                )
                print(
                    f"  {op_name:15s}: NumPy {numpy_time:.6f}s, Original {orig_time:.6f}s, speedup: {speedup_vs_numpy:.2f}x"
                )

            # Test optimized implementation
            if OPTIMIZED_AVAILABLE:
                # Clear caches before benchmark
                clear_caches()
                opt_time, opt_result = benchmark_optimized(op_name, args)
                config_results[f"opt_{op_name}"] = opt_time
                speedup_vs_numpy = (
                    numpy_time / opt_time if opt_time > 0 else float("inf")
                )
                print(
                    f"  {op_name:15s}: NumPy {numpy_time:.6f}s, Optimized {opt_time:.6f}s, speedup: {speedup_vs_numpy:.2f}x"
                )

                # Show cache stats
                cache_stats = get_cache_stats()
                if cache_stats["arange_cache_info"]["hits"] > 0:
                    print(f"    Cache hits: {cache_stats['arange_cache_info']['hits']}")

            print()

        results[config_name] = config_results

    # Summary
    print("=== Performance Summary ===")
    print_summary(results)


def benchmark_numpy_baseline(op_name: str, args: tuple) -> Tuple[float, any]:
    """Benchmark pure NumPy implementation."""
    if op_name == "arange":
        return benchmark_function(np.arange, *args)
    elif op_name == "linspace":
        return benchmark_function(np.linspace, *args)
    elif op_name == "zeros_vec":
        return benchmark_function(np.zeros, *args)
    elif op_name == "ones_vec":
        return benchmark_function(np.ones, *args)
    elif op_name == "concatenate":
        return benchmark_function(np.concatenate, *args)
    elif op_name == "compute_min":
        return benchmark_function(np.min, *args)
    elif op_name == "compute_std":
        return benchmark_function(np.std, *args)
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def benchmark_original(op_name: str, args: tuple) -> Tuple[float, any]:
    """Benchmark original numpy_rs implementation."""
    if op_name == "arange":
        return benchmark_function(orig_arange, *args)
    elif op_name == "linspace":
        return benchmark_function(orig_linspace, *args)
    elif op_name == "zeros_vec":
        return benchmark_function(orig_zeros_vec, *args)
    elif op_name == "ones_vec":
        return benchmark_function(orig_ones_vec, *args)
    elif op_name == "concatenate":
        return benchmark_function(orig_concatenate, *args)
    elif op_name == "compute_min":
        return benchmark_function(orig_compute_min, *args)
    elif op_name == "compute_std":
        return benchmark_function(orig_compute_std, *args)
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def benchmark_optimized(op_name: str, args: tuple) -> Tuple[float, any]:
    """Benchmark optimized numpy_rs implementation."""
    if op_name == "arange":
        return benchmark_function(opt_arange, *args)
    elif op_name == "linspace":
        return benchmark_function(opt_linspace, *args)
    elif op_name == "zeros_vec":
        return benchmark_function(opt_zeros_vec, *args)
    elif op_name == "ones_vec":
        return benchmark_function(opt_ones_vec, *args)
    elif op_name == "concatenate":
        return benchmark_function(opt_concatenate, *args)
    elif op_name == "compute_min":
        return benchmark_function(opt_compute_min, *args)
    elif op_name == "compute_std":
        return benchmark_function(opt_compute_std, *args)
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def print_summary(results: Dict):
    """Print performance summary."""
    print("Average speedup across all operations:")
    print("-" * 40)

    for config_name in results.keys():
        config_results = results[config_name]

        numpy_times = [v for k, v in config_results.items() if k.startswith("numpy_")]
        orig_times = (
            [v for k, v in config_results.items() if k.startswith("orig_")]
            if ORIGINAL_AVAILABLE
            else []
        )
        opt_times = (
            [v for k, v in config_results.items() if k.startswith("opt_")]
            if OPTIMIZED_AVAILABLE
            else []
        )

        if numpy_times and orig_times:
            avg_numpy = sum(numpy_times) / len(numpy_times)
            avg_orig = sum(orig_times) / len(orig_times)
            avg_speedup = avg_numpy / avg_orig
            print(f"{config_name:20s}: Original avg speedup: {avg_speedup:.2f}x")

        if numpy_times and opt_times:
            avg_numpy = sum(numpy_times) / len(numpy_times)
            avg_opt = sum(opt_times) / len(opt_times)
            avg_speedup = avg_numpy / avg_opt
            print(f"{config_name:20s}: Optimized avg speedup: {avg_speedup:.2f}x")

    print()


def benchmark_cache_effectiveness():
    """Benchmark cache effectiveness."""
    if not OPTIMIZED_AVAILABLE:
        print("Optimized implementation not available")
        return

    print("=== Cache Effectiveness Benchmark ===\n")

    # Clear caches
    clear_caches()

    # Test repeated operations
    test_args = (0, 1000, 1)  # arange arguments

    print("Testing arange cache effectiveness:")

    # First call (cache miss)
    clear_caches()
    first_time, _ = benchmark_function(opt_arange, *test_args)
    cache_stats = get_cache_stats()
    print(
        f"First call: {first_time:.6f}s, cache hits: {cache_stats['arange_cache_info']['hits']}"
    )

    # Second call (cache hit)
    second_time, _ = benchmark_function(opt_arange, *test_args)
    cache_stats = get_cache_stats()
    print(
        f"Second call: {second_time:.6f}s, cache hits: {cache_stats['arange_cache_info']['hits']}"
    )

    speedup = first_time / second_time if second_time > 0 else float("inf")
    print(f"Cache speedup: {speedup:.2f}x")

    print()


def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("=== Memory Usage Benchmark ===\n")

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Baseline
        baseline = process.memory_info().rss / 1024 / 1024
        print(f"Baseline memory: {baseline:.2f} MB")

        # Create arrays with original implementation
        if ORIGINAL_AVAILABLE:
            arrays_orig = []
            for i in range(100):
                arrays_orig.append(orig_arange(0, 1000, 1))

            memory_after_orig = process.memory_info().rss / 1024 / 1024
            print(
                f"After 100 original arrays: {memory_after_orig:.2f} MB (+{memory_after_orig - baseline:.2f} MB)"
            )

            del arrays_orig

        # Create arrays with optimized implementation
        if OPTIMIZED_AVAILABLE:
            clear_caches()
            arrays_opt = []
            for i in range(100):
                arrays_opt.append(opt_arange(0, 1000, 1))

            memory_after_opt = process.memory_info().rss / 1024 / 1024
            print(
                f"After 100 optimized arrays: {memory_after_opt:.2f} MB (+{memory_after_opt - baseline:.2f} MB)"
            )

            # Show cache memory usage
            cache_stats = get_cache_stats()
            print(f"Cache entries: {cache_stats['arange_cache_info']['currsize']}")

            del arrays_opt

    except ImportError:
        print("psutil not available, skipping memory benchmark")


def main():
    """Run all benchmarks."""
    run_benchmark_suite()
    benchmark_cache_effectiveness()
    benchmark_memory_usage()

    if OPTIMIZED_AVAILABLE:
        print("=== Final Cache Stats ===")
        final_stats = get_cache_stats()
        for key, value in final_stats.items():
            if isinstance(value, dict) and "hits" in value:
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
