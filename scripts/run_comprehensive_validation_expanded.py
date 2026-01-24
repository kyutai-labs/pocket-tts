import json
import sys
from pathlib import Path
import numpy as np

# Ensure worktree root is in path so we can import pocket_tts
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

try:
    from pocket_tts import numpy_rs as rs
except ImportError:
    print(
        "Error: Could not import pocket_tts.numpy_rs. Ensure PYTHONPATH is set correctly."
    )
    sys.exit(1)

# Comprehensive datasets for testing
DATASETS = {
    "DS-ARRAY-1": np.array([0, 1, -2, 3, 4, -5, 6], dtype=np.float32),
    "DS-ARRAY-2": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    "DS-ARRAY-3": np.array(
        [0.0, -1.5, 2.25, np.nan, np.inf, -np.inf], dtype=np.float32
    ),
    "DS-ARRAY-4": np.array([True, False, True, False, True], dtype=bool),
    "DS-ARRAY-5": np.array(["alpha", "Beta", "gamma", ""], dtype=np.str_),
    "DS-ARRAY-6": np.array([1, 1, 2, 3, 5, 8, 13], dtype=np.float32),
    "DS-ARRAY-7": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
    "DS-COMPLEX-1": np.array([1 + 2j, -3 + 0.5j, 0 - 1j], dtype=np.complex64),
}


def validate_function_exists(
    module_name, func_name, rust_func, np_func, dataset_id, extra_args=None
):
    """Check if function exists and can be called"""
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        return None

    print(f"Testing {module_name}.{func_name} with {dataset_id}...")

    try:
        # Check library availability first
        if not rs._AVAILABLE:
            return {
                "module": module_name,
                "function": func_name,
                "dataset": dataset_id,
                "parity": "❌ Library Unavailable",
                "evidence": "Rust library not loaded",
                "notes": "libnumpy.so not found or failed to load",
            }

        # Test function call
        if extra_args:
            if isinstance(dataset, dict):
                np_res = np_func(dataset, **extra_args)
                rs_res = rust_func(dataset, **extra_args)
            else:
                np_res = np_func(dataset, **extra_args)
                rs_res = rust_func(dataset, **extra_args)
        else:
            if isinstance(dataset, dict):
                np_res = np_func(**dataset)
                rs_res = rust_func(**dataset)
            else:
                np_res = np_func(dataset)
                rs_res = rust_func(dataset)

        # Compare results
        if isinstance(np_res, np.ndarray) and isinstance(rs_res, np.ndarray):
            parity = np.allclose(np_res, rs_res, equal_nan=True, rtol=1e-5, atol=1e-8)
            evidence = (
                f"shapes: np{np_res.shape} vs rs{rs_res.shape}, values match: {parity}"
            )
        else:
            # For scalar results
            try:
                if np.isnan(np_res) and np.isnan(rs_res):
                    parity = True
                else:
                    parity = np.isclose(np_res, rs_res, rtol=1e-5, atol=1e-8)
                evidence = f"np: {np_res}, rs: {rs_res}"
            except (TypeError, ValueError):
                parity = str(np_res) == str(rs_res)
                evidence = f"np: {np_res}, rs: {rs_res}"

        return {
            "module": module_name,
            "function": func_name,
            "dataset": dataset_id,
            "parity": "✅ Match" if parity else "❌ Mismatch",
            "evidence": evidence,
            "notes": "",
        }

    except AttributeError as e:
        # Function doesn't exist in Rust bindings
        return {
            "module": module_name,
            "function": func_name,
            "dataset": dataset_id,
            "parity": "❌ Missing FFI",
            "evidence": f"AttributeError: {str(e)}",
            "notes": "Function exists in Rust but not exported for FFI",
        }
    except Exception as e:
        # Check if it's an FFI symbol error
        if "undefined symbol" in str(e):
            return {
                "module": module_name,
                "function": func_name,
                "dataset": dataset_id,
                "parity": "❌ Missing FFI",
                "evidence": str(e),
                "notes": "Function exists in Rust but not exported for FFI",
            }

        return {
            "module": module_name,
            "function": func_name,
            "dataset": dataset_id,
            "parity": "❌ Error",
            "evidence": str(e),
            "notes": f"Execution failed: {e}",
        }


def run_comprehensive_validation():
    results = []

    # 1. Array Creation Functions
    print("=== Array Creation Module ===")
    array_creation_funcs = [
        ("arange", lambda: rs.arange(0, 7, 1), lambda: np.arange(0, 7, 1)),
        ("zeros", lambda: rs.zeros_vec(7), lambda: np.zeros(7, dtype=np.float32)),
        ("ones", lambda: rs.ones_vec(7), lambda: np.ones(7, dtype=np.float32)),
        ("eye", lambda: rs.eye(3), lambda: np.eye(3, dtype=np.float32).flatten()),
        (
            "linspace",
            lambda: rs.linspace(0, 1, 5),
            lambda: np.linspace(0, 1, 5, dtype=np.float32),
        ),
        ("full", None, lambda: np.full([3, 4], 7.0)),  # Not in Rust bindings
        ("empty", None, lambda: np.empty([3, 4])),  # Not in Rust bindings
    ]

    for func_name, rs_call, np_call in array_creation_funcs:
        if rs_call is None:
            # Function not available in Python bindings
            results.append(
                {
                    "module": "array_creation",
                    "function": func_name,
                    "dataset": "DS-ARRAY-1",
                    "parity": "❌ Missing Binding",
                    "evidence": "Function not implemented in numpy_rs.py",
                    "notes": "No Python wrapper for Rust function",
                }
            )
        else:
            results.append(
                validate_function_exists(
                    "array_creation", func_name, rs_call, np_call, "DS-ARRAY-1"
                )
            )

    # 2. Mathematical Ufuncs
    print("\n=== Mathematical Ufuncs ===")
    math_funcs = [
        ("log", rs.log_vec, np.log),
        (
            "power",
            lambda: rs.power_vec(DATASETS["DS-ARRAY-1"], 2),
            lambda: np.power(DATASETS["DS-ARRAY-1"], 2),
        ),
        ("exp", None, lambda: np.exp),  # Not in Rust bindings
        ("sqrt", None, lambda: np.sqrt),  # Not in Rust bindings
        ("sin", None, lambda: np.sin),  # Not in Rust bindings
        ("cos", None, lambda: np.cos),  # Not in Rust bindings
    ]

    for func_name, rs_call, np_call in math_funcs:
        if rs_call is None:
            results.append(
                {
                    "module": "math_ufuncs",
                    "function": func_name,
                    "dataset": "DS-ARRAY-1",
                    "parity": "❌ Missing Binding",
                    "evidence": "Function not implemented in numpy_rs.py",
                    "notes": "No Python wrapper for Rust function",
                }
            )
        else:
            results.append(
                validate_function_exists(
                    "math_ufuncs", func_name, rs_call, np_call, "DS-ARRAY-1"
                )
            )

    # 3. Statistics Functions
    print("\n=== Statistics Module ===")
    stats_funcs = [
        ("min", rs.compute_min, np.min),
        ("max", None, lambda: np.max),  # Not in Rust bindings
        ("mean", None, lambda: np.mean),  # Not in Rust bindings
        ("sum", None, lambda: np.sum),  # Not in Rust bindings
    ]

    for func_name, rs_call, np_call in stats_funcs:
        if rs_call is None:
            results.append(
                {
                    "module": "statistics",
                    "function": func_name,
                    "dataset": "DS-ARRAY-1",
                    "parity": "❌ Missing Binding",
                    "evidence": "Function not implemented in numpy_rs.py",
                    "notes": "No Python wrapper for Rust function",
                }
            )
        else:
            results.append(
                validate_function_exists(
                    "statistics", func_name, rs_call, np_call, "DS-ARRAY-1"
                )
            )

    # 4. Array Manipulation
    print("\n=== Array Manipulation ===")
    manip_funcs = [
        ("transpose_2d", rs.transpose_2d, np.transpose),
        ("reshape", None, lambda: np.reshape),  # In Rust bindings but issues
        ("concatenate", None, lambda: np.concatenate),  # In Rust bindings
        ("vstack", rs.vstack, np.vstack),
        ("hstack", rs.hstack, np.hstack),
    ]

    for func_name, rs_call, np_call in manip_funcs:
        if rs_call is None:
            results.append(
                {
                    "module": "array_manipulation",
                    "function": func_name,
                    "dataset": "DS-ARRAY-2",
                    "parity": "❌ Missing Binding",
                    "evidence": "Function not implemented in numpy_rs.py",
                    "notes": "No Python wrapper for Rust function",
                }
            )
        else:
            results.append(
                validate_function_exists(
                    "array_manipulation", func_name, rs_call, np_call, "DS-ARRAY-2"
                )
            )

    # 5. Linear Algebra
    print("\n=== Linear Algebra ===")
    linalg_funcs = [
        (
            "dot",
            lambda: rs.dot_vec(DATASETS["DS-ARRAY-1"], DATASETS["DS-ARRAY-1"]),
            lambda: np.dot(DATASETS["DS-ARRAY-1"], DATASETS["DS-ARRAY-1"]),
        ),
        (
            "matmul",
            rs.matmul_2d,
            lambda: np.matmul(
                DATASETS["DS-ARRAY-2"],
                np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            ),
        ),
        ("inv", None, lambda: np.linalg.inv),  # Not in Rust bindings
        ("det", None, lambda: np.linalg.det),  # Not in Rust bindings
    ]

    for func_name, rs_call, np_call in linalg_funcs:
        if rs_call is None:
            results.append(
                {
                    "module": "linalg",
                    "function": func_name,
                    "dataset": "DS-ARRAY-2",
                    "parity": "❌ Missing Binding",
                    "evidence": "Function not implemented in numpy_rs.py",
                    "notes": "No Python wrapper for Rust function",
                }
            )
        else:
            results.append(
                validate_function_exists(
                    "linalg", func_name, rs_call, np_call, "DS-ARRAY-2"
                )
            )

    # 6. Advanced Modules (summary only)
    print("\n=== Advanced Modules Status ===")
    advanced_modules = [
        ("fft", "FFT operations"),
        ("random", "Random number generation"),
        ("polynomial", "Polynomial operations"),
        ("datetime", "DateTime/Timedelta operations"),
        ("char", "String/character operations"),
        ("io", "File I/O operations"),
        ("masking", "Masked array operations"),
    ]

    for module_name, description in advanced_modules:
        results.append(
            {
                "module": module_name,
                "function": "module_status",
                "dataset": "N/A",
                "parity": "❌ Not Implemented",
                "evidence": f"Module {module_name} not available in Python bindings",
                "notes": f"No Python wrapper for {description}",
            }
        )

    # Save comprehensive results
    output_path = repo_root / "results" / "comprehensive_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComprehensive results saved to {output_path}")

    # Print summary statistics
    total_funcs = len(results)
    ffi_missing = sum(1 for r in results if "Missing FFI" in r["parity"])
    binding_missing = sum(1 for r in results if "Missing Binding" in r["parity"])
    working = sum(1 for r in results if "✅" in r["parity"])

    print("\n=== SUMMARY ===")
    print(f"Total functions tested: {total_funcs}")
    print(f"✅ Working: {working}")
    print(f"❌ Missing FFI: {ffi_missing}")
    print(f"❌ Missing Python Bindings: {binding_missing}")
    print(f"Completion rate: {100 * working / total_funcs:.1f}%")

    return results


if __name__ == "__main__":
    run_comprehensive_validation()
