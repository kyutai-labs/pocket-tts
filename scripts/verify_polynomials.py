#!/usr/bin/env python3
"""
Cross-validation script for rust-numpy polynomial algorithms against NumPy.

This script generates test cases with NumPy and saves the expected results
in a format that can be used by Rust verification tests.

Usage:
    python scripts/verify_polynomials.py
    python scripts/verify_polynomials.py --output rust-numpy/tests/polynomial_golden_data.json
"""

import argparse
import json
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre
from typing import List, Tuple, Dict, Any
import sys


def generate_fitting_test_cases() -> List[Dict[str, Any]]:
    """Generate polynomial fitting test cases."""
    test_cases = []

    # Linear fit
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    p = Polynomial.fit(x, y, 1)
    test_cases.append(
        {
            "test_name": "fit_linear_simple",
            "x": x.tolist(),
            "y": y.tolist(),
            "deg": 1,
            "expected_coeffs": p.convert().coef.tolist(),
        }
    )

    # Quadratic fit
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    p = Polynomial.fit(x, y, 2)
    test_cases.append(
        {
            "test_name": "fit_quadratic",
            "x": x.tolist(),
            "y": y.tolist(),
            "deg": 2,
            "expected_coeffs": p.convert().coef.tolist(),
        }
    )

    # Cubic fit
    x = np.linspace(0, 4, 9)
    y = x**3 - 2 * x**2 + 3 * x - 1
    p = Polynomial.fit(x, y, 3)
    test_cases.append(
        {
            "test_name": "fit_cubic",
            "x": x.tolist(),
            "y": y.tolist(),
            "deg": 3,
            "expected_coeffs": p.convert().coef.tolist(),
        }
    )

    # High degree fit
    x = np.linspace(0, 2, 11)
    y = x**5 - 3 * x**3 + 2 * x + 1
    p = Polynomial.fit(x, y, 5)
    test_cases.append(
        {
            "test_name": "fit_high_degree",
            "x": x.tolist(),
            "y": y.tolist(),
            "deg": 5,
            "expected_coeffs": p.convert().coef.tolist(),
        }
    )

    # Noisy data
    x = np.linspace(0, 9, 10)
    y = np.array([1.1, 2.9, 4.2, 6.1, 7.8, 10.2, 10.9, 13.1, 14.0, 16.2])
    p = Polynomial.fit(x, y, 1)
    test_cases.append(
        {
            "test_name": "fit_noisy_data",
            "x": x.tolist(),
            "y": y.tolist(),
            "deg": 1,
            "expected_coeffs": p.convert().coef.tolist(),
        }
    )

    return test_cases


def generate_roots_test_cases() -> List[Dict[str, Any]]:
    """Generate root-finding test cases."""
    test_cases = []

    # Linear: 2x - 4 = 0 => x = 2
    p = Polynomial([-4, 2])
    roots = p.roots()
    test_cases.append(
        {
            "test_name": "roots_linear",
            "coeffs": p.coef.tolist(),
            "expected_roots": [[r.real, r.imag] for r in roots],
        }
    )

    # Quadratic with real roots: x^2 - 3x + 2 = 0
    p = Polynomial([2, -3, 1])
    roots = p.roots()
    test_cases.append(
        {
            "test_name": "roots_quadratic_real",
            "coeffs": p.coef.tolist(),
            "expected_roots": [[r.real, r.imag] for r in roots],
        }
    )

    # Quadratic with complex roots: x^2 + 1 = 0
    p = Polynomial([1, 0, 1])
    roots = p.roots()
    test_cases.append(
        {
            "test_name": "roots_quadratic_complex",
            "coeffs": p.coef.tolist(),
            "expected_roots": [[r.real, r.imag] for r in roots],
        }
    )

    # Cubic: (x-1)(x-2)(x-3) = 0
    p = Polynomial([-6, 11, -6, 1])
    roots = p.roots()
    test_cases.append(
        {
            "test_name": "roots_cubic_real",
            "coeffs": p.coef.tolist(),
            "expected_roots": [[r.real, r.imag] for r in roots],
        }
    )

    # Quartic: (x-1)(x-2)(x-3)(x-4) = 0
    p = Polynomial([24, -50, 35, -10, 1])
    roots = p.roots()
    test_cases.append(
        {
            "test_name": "roots_quartic",
            "coeffs": p.coef.tolist(),
            "expected_roots": [[r.real, r.imag] for r in roots],
        }
    )

    # Repeated roots: (x-2)^2 = 0
    p = Polynomial([4, -4, 1])
    roots = p.roots()
    test_cases.append(
        {
            "test_name": "roots_repeated",
            "coeffs": p.coef.tolist(),
            "expected_roots": [[r.real, r.imag] for r in roots],
        }
    )

    return test_cases


def generate_arithmetic_test_cases() -> List[Dict[str, Any]]:
    """Generate arithmetic operations test cases."""
    test_cases = []

    # Addition
    p1 = Polynomial([1, 2, 1])  # x^2 + 2x + 1
    p2 = Polynomial([5, 4, 3])  # 3x^2 + 4x + 5
    result = p1 + p2
    test_cases.append(
        {
            "test_name": "add_same_degree",
            "p1_coeffs": p1.coef.tolist(),
            "p2_coeffs": p2.coef.tolist(),
            "operation": "add",
            "expected_coeffs": result.coef.tolist(),
        }
    )

    p1 = Polynomial([1, 1])  # x + 1
    p2 = Polynomial([5, 4, 3])  # 3x^2 + 4x + 5
    result = p1 + p2
    test_cases.append(
        {
            "test_name": "add_different_degree",
            "p1_coeffs": p1.coef.tolist(),
            "p2_coeffs": p2.coef.tolist(),
            "operation": "add",
            "expected_coeffs": result.coef.tolist(),
        }
    )

    # Subtraction
    p1 = Polynomial([7, 6, 5])  # 5x^2 + 6x + 7
    p2 = Polynomial([4, 3, 2])  # 2x^2 + 3x + 4
    result = p1 - p2
    test_cases.append(
        {
            "test_name": "sub_same_degree",
            "p1_coeffs": p1.coef.tolist(),
            "p2_coeffs": p2.coef.tolist(),
            "operation": "sub",
            "expected_coeffs": result.coef.tolist(),
        }
    )

    # Multiplication
    p1 = Polynomial([1, 1])  # x + 1
    p2 = Polynomial([2, 1])  # x + 2
    result = p1 * p2
    test_cases.append(
        {
            "test_name": "mul_simple",
            "p1_coeffs": p1.coef.tolist(),
            "p2_coeffs": p2.coef.tolist(),
            "operation": "mul",
            "expected_coeffs": result.coef.tolist(),
        }
    )

    p1 = Polynomial([1, 1, 1])  # x^2 + x + 1
    p2 = Polynomial([1, 1])  # x + 1
    result = p1 * p2
    test_cases.append(
        {
            "test_name": "mul_different_degree",
            "p1_coeffs": p1.coef.tolist(),
            "p2_coeffs": p2.coef.tolist(),
            "operation": "mul",
            "expected_coeffs": result.coef.tolist(),
        }
    )

    return test_cases


def generate_evaluation_test_cases() -> List[Dict[str, Any]]:
    """Generate polynomial evaluation test cases."""
    test_cases = []

    # Single point
    p = Polynomial([1, 2])  # 2x + 1
    x = np.array([3.0])
    y = p(x)
    test_cases.append(
        {
            "test_name": "eval_single_point",
            "coeffs": p.coef.tolist(),
            "x": x.tolist(),
            "expected_y": y.tolist(),
        }
    )

    # Multiple points
    p = Polynomial([1, 2, 1])  # x^2 + 2x + 1
    x = np.array([0.0, 1.0, 2.0])
    y = p(x)
    test_cases.append(
        {
            "test_name": "eval_multiple_points",
            "coeffs": p.coef.tolist(),
            "x": x.tolist(),
            "expected_y": y.tolist(),
        }
    )

    # High degree
    p = Polynomial([1, 2, 0, -3, 0, 0, 1])  # x^6 - 3x^3 + 2x + 1
    x = np.array([2.0])
    y = p(x)
    test_cases.append(
        {
            "test_name": "eval_high_degree",
            "coeffs": p.coef.tolist(),
            "x": x.tolist(),
            "expected_y": y.tolist(),
        }
    )

    return test_cases


def generate_chebyshev_test_cases() -> List[Dict[str, Any]]:
    """Generate Chebyshev polynomial test cases."""
    test_cases = []

    # Basic Chebyshev fit
    x = np.linspace(-1, 1, 10)
    y = x**2
    c = Chebyshev.fit(x, y, 2)
    test_cases.append(
        {
            "test_name": "chebyshev_fit_quadratic",
            "x": x.tolist(),
            "y": y.tolist(),
            "deg": 2,
            "expected_coeffs": c.coef.tolist(),
        }
    )

    # Chebyshev evaluation
    c = Chebyshev([1, 0.5, 0.25])
    x = np.array([0.0, 0.5, 1.0])
    y = c(x)
    test_cases.append(
        {
            "test_name": "chebyshev_eval",
            "coeffs": c.coef.tolist(),
            "x": x.tolist(),
            "expected_y": y.tolist(),
        }
    )

    return test_cases


def generate_legendre_test_cases() -> List[Dict[str, Any]]:
    """Generate Legendre polynomial test cases."""
    test_cases = []

    # Basic Legendre fit
    x = np.linspace(-1, 1, 10)
    y = x**3
    l = Legendre.fit(x, y, 3)
    test_cases.append(
        {
            "test_name": "legendre_fit_cubic",
            "x": x.tolist(),
            "y": y.tolist(),
            "deg": 3,
            "expected_coeffs": l.coef.tolist(),
        }
    )

    # Legendre evaluation
    l = Legendre([1, 0.5, 0.25, 0.125])
    x = np.array([0.0, 0.5, 1.0])
    y = l(x)
    test_cases.append(
        {
            "test_name": "legendre_eval",
            "coeffs": l.coef.tolist(),
            "x": x.tolist(),
            "expected_y": y.tolist(),
        }
    )

    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden test data for polynomial verification"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="rust-numpy/tests/polynomial_golden_data.json",
        help="Output file path for golden test data",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    # Generate all test cases
    golden_data = {
        "fitting": generate_fitting_test_cases(),
        "roots": generate_roots_test_cases(),
        "arithmetic": generate_arithmetic_test_cases(),
        "evaluation": generate_evaluation_test_cases(),
        "chebyshev": generate_chebyshev_test_cases(),
        "legendre": generate_legendre_test_cases(),
    }

    # Count test cases
    total_cases = sum(len(cases) for cases in golden_data.values())
    print(f"Generated {total_cases} test cases:")
    for category, cases in golden_data.items():
        print(f"  {category}: {len(cases)} cases")

    # Write to file
    with open(args.output, "w") as f:
        if args.pretty:
            json.dump(golden_data, f, indent=2)
        else:
            json.dump(golden_data, f)

    print(f"\nGolden data written to {args.output}")
    print("\nTo use this data in Rust tests, you can:")
    print("1. Include the JSON file in your test")
    print("2. Parse it with serde_json")
    print("3. Compare your results against the expected values")
    print("\nExample tolerance:")
    print("  - Well-conditioned problems: 1e-10")
    print("  - Ill-conditioned problems: 1e-6")
    print("  - Root ordering may differ; sort before comparing")


if __name__ == "__main__":
    main()
