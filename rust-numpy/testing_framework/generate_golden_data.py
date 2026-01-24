import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
TEST_CASES_PATH = ROOT / "test_cases" / "basic.json"
OUTPUT_DIR = ROOT / "output"
OUTPUT_GOLDEN = OUTPUT_DIR / "golden_data.json"
OUTPUT_REPORT = OUTPUT_DIR / "report.json"


@dataclass
class OperationSpec:
    name: str
    rhs: Dict[str, Any] | None = None
    axis: List[int] | None = None
    keepdims: bool | None = None


def load_cases() -> Dict[str, Any]:
    with TEST_CASES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_array(array_spec: Dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(array_spec["dtype"])
    data = np.array(array_spec["data"], dtype=dtype)
    shape = array_spec.get("shape", [])
    if shape:
        data = data.reshape(shape)
    return data


def array_to_payload(array: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": array.reshape(-1).tolist(),
    }


def run_basic_ops(array: np.ndarray, ops: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for op in ops:
        if op == "sum":
            results[op] = array_to_payload(np.sum(array))
        elif op == "mean":
            results[op] = array_to_payload(np.mean(array))
        elif op == "min":
            results[op] = array_to_payload(np.min(array))
        elif op == "max":
            results[op] = array_to_payload(np.max(array))
        elif op == "negative":
            results[op] = array_to_payload(np.negative(array))
        elif op == "abs":
            results[op] = array_to_payload(np.abs(array))
        else:
            raise ValueError(f"Unsupported basic operation: {op}")
    return results


def run_reduction_ops(array: np.ndarray, ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for op in ops:
        name = op["name"]
        axis = op.get("axis")
        keepdims = bool(op.get("keepdims", False))
        if name == "sum":
            reduced = np.sum(
                array, axis=tuple(axis) if axis else None, keepdims=keepdims
            )
        else:
            raise ValueError(f"Unsupported reduction operation: {name}")
        axis_suffix = "all" if not axis else "_".join(str(a) for a in axis)
        key = f"{name}_axis_{axis_suffix}"
        results[key] = array_to_payload(reduced)
    return results


def run_ufunc_ops(
    array: np.ndarray, ops: List[Dict[str, Any]]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    results: Dict[str, Any] = {}
    operands: Dict[str, Any] = {"binary": {}, "comparison": {}}
    for op in ops:
        name = op["name"]
        rhs_spec = op.get("rhs")
        if rhs_spec is None:
            raise ValueError(f"Ufunc {name} requires rhs array")
        rhs_array = to_array(rhs_spec)
        if name == "add":
            computed = np.add(array, rhs_array)
            operands["binary"][name] = array_to_payload(rhs_array)
        elif name == "multiply":
            computed = np.multiply(array, rhs_array)
            operands["binary"][name] = array_to_payload(rhs_array)
        elif name == "greater":
            computed = np.greater(array, rhs_array)
            operands["comparison"][name] = array_to_payload(rhs_array)
        elif name == "equal":
            computed = np.equal(array, rhs_array)
            operands["comparison"][name] = array_to_payload(rhs_array)
        else:
            raise ValueError(f"Unsupported ufunc: {name}")
        results[name] = array_to_payload(computed)
    return results, operands


def main() -> None:
    payload = load_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_cases = []
    total_ops = {"basic": 0, "reductions": 0, "ufuncs": 0}

    for case in payload["cases"]:
        input_array = to_array(case["input"])
        operations = case["operations"]

        basic_ops = run_basic_ops(input_array, operations.get("basic", []))
        reductions = run_reduction_ops(input_array, operations.get("reductions", []))
        ufuncs, operands = run_ufunc_ops(input_array, operations.get("ufuncs", []))

        total_ops["basic"] += len(basic_ops)
        total_ops["reductions"] += len(reductions)
        total_ops["ufuncs"] += len(ufuncs)

        test_cases.append(
            {
                "id": case["id"],
                "input": array_to_payload(input_array),
                "operands": operands,
                "operations": {
                    "basic": basic_ops,
                    "reductions": reductions,
                    "ufuncs": ufuncs,
                },
            }
        )

    golden_data = {
        "metadata": {
            "version": "v1",
            "numpy_version": np.__version__,
            "seed": 0,
            "num_cases": len(test_cases),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "description": payload.get("description", ""),
        },
        "test_cases": test_cases,
    }

    OUTPUT_GOLDEN.write_text(json.dumps(golden_data, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(
        json.dumps({"cases": len(test_cases), "operations": total_ops}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote golden data to {OUTPUT_GOLDEN}")
    print(f"Wrote report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
