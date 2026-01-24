import json
import sys
from pathlib import Path

import numpy as np

# Ensure the worktree root is in path so we can import pocket_tts
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

try:
    from pocket_tts import numpy_rs as rs
except ImportError:
    print(
        "Error: Could not import pocket_tts.numpy_rs. Ensure PYTHONPATH is set correctly."
    )
    sys.exit(1)

# Datasets definition (from GAP_ANALYSIS.md)
DATASETS = {
    "DS-ARRAY-1": np.array([0, 1, -2, 3, 4, -5, 6], dtype=np.float32),
    "DS-ARRAY-2": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    "DS-ARRAY-3": np.array(
        [0.0, -1.5, 2.25, np.nan, np.inf, -np.inf], dtype=np.float32
    ),
    "DS-ARRAY-4": np.array([True, False, True, False, True], dtype=bool),
}


def validate_parity(module_name, func_name, rs_func, np_func, dataset_id):
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        return None

    print(f"Validating {module_name}.{func_name} with {dataset_id}...")

    try:
        # Check if library is available (actual FFI)
        if not rs._AVAILABLE:
            return {
                "module": module_name,
                "function": func_name,
                "dataset": dataset_id,
                "parity": "❌ Missing",
                "evidence": "Rust library not loaded or FFI symbols missing",
                "notes": "Library loader warned about missing .so or symbols",
            }

        if isinstance(dataset, dict):
            np_res = np_func(**dataset)
            rs_res = rs_func(**dataset)
        else:
            np_res = np_func(dataset)
            rs_res = rs_func(dataset)

        parity = np.allclose(np_res, rs_res, equal_nan=True)
        evidence = f"np: {np_res}, rs: {rs_res}"

        return {
            "module": module_name,
            "function": func_name,
            "dataset": dataset_id,
            "parity": "✅ Match" if parity else "❌ Mismatch",
            "evidence": evidence,
            "notes": "",
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


def run_all():
    results = []

    # 1. Array Creation
    results.append(
        validate_parity(
            "array_creation",
            "arange",
            lambda d: rs.arange(0, 10, 1),
            lambda d: np.arange(0, 10, 1),
            "DS-ARRAY-1",
        )
    )
    results.append(
        validate_parity(
            "array_creation",
            "zeros",
            lambda d: rs.zeros_vec(7),
            lambda d: np.zeros(7, dtype=np.float32),
            "DS-ARRAY-1",
        )
    )
    results.append(
        validate_parity(
            "array_creation",
            "ones",
            lambda d: rs.ones_vec(7),
            lambda d: np.ones(7, dtype=np.float32),
            "DS-ARRAY-1",
        )
    )
    results.append(
        validate_parity(
            "array_creation",
            "eye",
            lambda d: rs.eye(3),
            lambda d: np.eye(3, dtype=np.float32).flatten(),
            "DS-ARRAY-1",
        )
    )
    results.append(
        validate_parity(
            "array_creation",
            "linspace",
            lambda d: rs.linspace(0, 1, 5),
            lambda d: np.linspace(0, 1, 5, dtype=np.float32),
            "DS-ARRAY-1",
        )
    )

    # 2. Math Ufuncs
    results.append(
        validate_parity("math_ufuncs", "log", rs.log_vec, np.log, "DS-ARRAY-1")
    )
    results.append(
        validate_parity(
            "math_ufuncs",
            "power",
            lambda d: rs.power_vec(d, 2),
            lambda d: np.power(d, 2),
            "DS-ARRAY-1",
        )
    )

    # 3. Statistics
    results.append(
        validate_parity("statistics", "min", rs.compute_min, np.min, "DS-ARRAY-1")
    )
    results.append(
        validate_parity("statistics", "std", rs.compute_std, np.std, "DS-ARRAY-1")
    )
    results.append(
        validate_parity("statistics", "var", rs.compute_var, np.var, "DS-ARRAY-1")
    )

    # 4. Array Manipulation
    results.append(
        validate_parity(
            "array_manipulation",
            "transpose_2d",
            rs.transpose_2d,
            np.transpose,
            "DS-ARRAY-2",
        )
    )

    # 5. Linalg
    results.append(
        validate_parity(
            "linalg",
            "dot",
            lambda d: rs.dot_vec(d, d),
            lambda d: np.dot(d, d),
            "DS-ARRAY-1",
        )
    )

    # Matmul fix: a is (2,3), so b must be (3, 2 for example)
    matrix_b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    results.append(
        validate_parity(
            "linalg",
            "matmul",
            lambda d: rs.matmul_2d(d, matrix_b),
            lambda d: np.matmul(d, matrix_b),
            "DS-ARRAY-2",
        )
    )

    # Save results
    output_path = repo_root / "results" / "validation_results.json"
    with open(output_path, "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_all()
