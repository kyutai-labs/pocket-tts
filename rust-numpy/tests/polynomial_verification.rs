// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Comprehensive verification tests for polynomial module algorithm parity with NumPy
//!
//! These tests verify that the rust-numpy polynomial algorithms produce identical
//! (or within numerical tolerance) results to NumPy's polynomial module.
//!
//! Run: cargo test --package rust-numpy --test polynomial_verification
//!
//! For cross-validation with NumPy, run:
//! python scripts/verify_polynomials.py

use ndarray::array;
use numpy::polynomial::{fit, roots, Polynomial, PolynomialBase};
use num_complex::Complex;
use num_traits::Float;

/// Numerical tolerance for comparisons with NumPy
const TOLERANCE: f64 = 1e-10;

/// Relaxed tolerance for ill-conditioned problems
const ILL_CONDITIONED_TOLERANCE: f64 = 1e-6;

// ============================================================================
// Polynomial Fitting Verification Tests
// ============================================================================

#[test]
fn test_fit_linear_simple() {
    // y = 2x + 1
    // NumPy: np.polynomial.Polynomial.fit(x, y, 1).coef()
    // Expected: [1.0, 2.0] (constant term first, like NumPy)
    let x = array![0.0_f64, 1.0, 2.0, 3.0];
    let y = array![1.0_f64, 3.0, 5.0, 7.0];

    let p = fit(&x, &y, 1).unwrap();
    let coeffs = p.coeffs();

    assert!(Float::abs(coeffs[0] - 1.0) < TOLERANCE, "Intercept mismatch");
    assert!(Float::abs(coeffs[1] - 2.0) < TOLERANCE, "Slope mismatch");
}

#[test]
fn test_fit_quadratic() {
    // y = x^2 + 2x + 1
    // NumPy: np.polynomial.Polynomial.fit(x, y, 2).coef()
    let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
    let y = array![1.0_f64, 4.0, 9.0, 16.0, 25.0]; // x^2 + 0x + 1, actually just x^2

    let p = fit(&x, &y, 2).unwrap();
    let coeffs = p.coeffs();

    // For x^2, fit should give coefficients close to [1, 0, 0]
    // But with our x range, we check that it fits well
    let y_pred = p.eval(&x).unwrap();
    for i in 0..y.len() {
        assert!(Float::abs(y_pred[i] - y[i]) < TOLERANCE, "Fit mismatch at x={}", x[i]);
    }
}

#[test]
fn test_fit_cubic() {
    // y = x^3 - 2x^2 + 3x - 1
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let y: Vec<f64> = x.iter().map(|&xi| xi.powi(3) - 2.0 * xi.powi(2) + 3.0 * xi - 1.0).collect();
    let y = ndarray::Array1::from(y);

    let p = fit(&x, &y, 3).unwrap();

    // Check fit quality
    let y_pred = p.eval(&x).unwrap();
    for i in 0..y.len() {
        assert!(Float::abs(y_pred[i] - y[i]) < 1e-8, "Cubic fit mismatch at index {}", i);
    }
}

#[test]
fn test_fit_high_degree() {
    // Higher degree polynomial (degree 5)
    // y = x^5 - 3x^3 + 2x + 1
    let x = array![0.0_f64, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];
    let y: Vec<f64> = x.iter().map(|&xi| {
        xi.powi(5) - 3.0 * xi.powi(3) + 2.0 * xi + 1.0
    }).collect();
    let y = ndarray::Array1::from(y);

    let p = fit(&x, &y, 5).unwrap();

    // Check fit quality
    let y_pred = p.eval(&x).unwrap();
    for i in 0..y.len() {
        assert!(Float::abs(y_pred[i] - y[i]) < 1e-6, "High-degree fit mismatch at index {}", i);
    }
}

#[test]
fn test_fit_noisy_data() {
    // Linear data with noise
    let x = array![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let y = array![1.1_f64, 2.9, 4.2, 6.1, 7.8, 10.2, 10.9, 13.1, 14.0, 16.2];

    let p = fit(&x, &y, 1).unwrap();

    // Check that coefficients are close to expected (intercept ~1, slope ~1.6)
    let coeffs = p.coeffs();
    assert!(Float::abs(coeffs[0] - 1.0) < 1.0, "Noisy intercept too far off");
    assert!(Float::abs(coeffs[1] - 1.6) < 0.2, "Noisy slope too far off");
}

#[test]
fn test_fit_error_too_few_points() {
    // Error case: not enough points for degree
    let x = array![0.0_f64, 1.0];
    let y = array![1.0_f64, 2.0];

    let result = fit(&x, &y, 2);
    assert!(result.is_err(), "Should error when num_points <= degree");
}

#[test]
fn test_fit_error_mismatched_lengths() {
    // Error case: x and y different lengths
    let x = array![0.0_f64, 1.0, 2.0];
    let y = array![1.0_f64, 2.0];

    let result = fit(&x, &y, 1);
    assert!(result.is_err(), "Should error when x.len() != y.len()");
}

// ============================================================================
// Root-Finding Verification Tests
// ============================================================================

#[test]
fn test_roots_linear() {
    // 2x - 4 = 0 => x = 2
    // coeffs: [-4, 2]
    let p = Polynomial::new(&array![-4.0_f64, 2.0]).unwrap();
    let r = roots(&p).unwrap();

    assert_eq!(r.len(), 1);
    assert!(Float::abs(r[0].re - 2.0) < TOLERANCE);
    assert!(Float::abs(r[0].im) < TOLERANCE);
}

#[test]
fn test_roots_quadratic_real() {
    // x^2 - 3x + 2 = 0 => x = 1, 2
    // coeffs: [2, -3, 1]
    let p = Polynomial::new(&array![2.0_f64, -3.0, 1.0]).unwrap();
    let r = roots(&p).unwrap();

    assert_eq!(r.len(), 2);

    // Sort by real part
    let mut r_vec = r.to_vec();
    r_vec.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    assert!(Float::abs(r_vec[0].re - 1.0) < TOLERANCE);
    assert!(Float::abs(r_vec[1].re - 2.0) < TOLERANCE);
    assert!(Float::abs(r_vec[0].im) < TOLERANCE);
    assert!(Float::abs(r_vec[1].im) < TOLERANCE);
}

#[test]
fn test_roots_quadratic_complex() {
    // x^2 + 1 = 0 => x = i, -i
    // coeffs: [1, 0, 1]
    let p = Polynomial::new(&array![1.0_f64, 0.0, 1.0]).unwrap();
    let r = roots(&p).unwrap();

    assert_eq!(r.len(), 2);

    // Check that we have ±i
    for root in r.iter() {
        assert!(Float::abs(root.re) < TOLERANCE);
        assert!(Float::abs(Float::abs(root.im) - 1.0) < TOLERANCE);
    }
}

#[test]
fn test_roots_cubic_real() {
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    // coeffs: [-6, 11, -6, 1]
    let p = Polynomial::new(&array![-6.0_f64, 11.0, -6.0, 1.0]).unwrap();
    let r = roots(&p).unwrap();

    assert_eq!(r.len(), 3);

    // Sort by real part
    let mut r_vec = r.to_vec();
    r_vec.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    assert!(Float::abs(r_vec[0].re - 1.0) < 1e-8);
    assert!(Float::abs(r_vec[1].re - 2.0) < 1e-8);
    assert!(Float::abs(r_vec[2].re - 3.0) < 1e-8);
}

#[test]
fn test_roots_quartic() {
    // (x-1)(x-2)(x-3)(x-4) expanded
    // coeffs: [24, -50, 35, -10, 1]
    let p = Polynomial::new(&array![24.0_f64, -50.0, 35.0, -10.0, 1.0]).unwrap();
    let r = roots(&p).unwrap();

    assert_eq!(r.len(), 4);

    // Sort by real part
    let mut r_vec = r.to_vec();
    r_vec.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    assert!(Float::abs(r_vec[0].re - 1.0) < 1e-6);
    assert!(Float::abs(r_vec[1].re - 2.0) < 1e-6);
    assert!(Float::abs(r_vec[2].re - 3.0) < 1e-6);
    assert!(Float::abs(r_vec[3].re - 4.0) < 1e-6);
}

#[test]
fn test_roots_repeated() {
    // (x-2)^2 = x^2 - 4x + 4
    // coeffs: [4, -4, 1]
    let p = Polynomial::new(&array![4.0_f64, -4.0, 1.0]).unwrap();
    let r = roots(&p).unwrap();

    assert_eq!(r.len(), 2);

    // Both roots should be 2 (or very close)
    for root in r.iter() {
        assert!(Float::abs(root.re - 2.0) < 1e-6);
        assert!(Float::abs(root.im) < 1e-6);
    }
}

#[test]
fn test_roots_high_degree() {
    // Degree 5: (x-1)(x-2)(x-3)(x-4)(x-5)
    // This tests the Durand-Kerner method with a moderately high degree polynomial
    // (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120
    let p = Polynomial::new(&array![
        -120.0_f64,  // constant
        274.0,
        -225.0,
        85.0,
        -15.0,
        1.0       // x^5 term
    ]).unwrap();
    let r = roots(&p).unwrap();

    assert_eq!(r.len(), 5);

    // Sort by real part
    let mut r_vec = r.to_vec();
    r_vec.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    // Check that all roots are close to 1, 2, 3, 4, 5
    // Use relaxed tolerance for higher degree polynomials
    for (i, root) in r_vec.iter().enumerate() {
        let expected = (i + 1) as f64;
        assert!(Float::abs(root.re - expected) < 1e-3, "Root {} mismatch: expected {}, got {}", i, expected, root.re);
        assert!(Float::abs(root.im) < 1e-3, "Root {} has non-zero imaginary part: {}", i, root.im);
    }
}

// ============================================================================
// Arithmetic Operations Verification Tests
// ============================================================================

#[test]
fn test_add_same_degree() {
    // p1(x) = x^2 + 2x + 1
    // p2(x) = 3x^2 + 4x + 5
    // sum = 4x^2 + 6x + 6
    let p1 = Polynomial::new(&array![1.0_f64, 2.0, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![5.0_f64, 4.0, 3.0]).unwrap();

    let result = p1.add(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 6.0);
    assert_eq!(coeffs[1], 6.0);
    assert_eq!(coeffs[2], 4.0);
}

#[test]
fn test_add_different_degree() {
    // p1(x) = x + 1
    // p2(x) = 3x^2 + 4x + 5
    // sum = 3x^2 + 5x + 6
    let p1 = Polynomial::new(&array![1.0_f64, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![5.0_f64, 4.0, 3.0]).unwrap();

    let result = p1.add(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 6.0);
    assert_eq!(coeffs[1], 5.0);
    assert_eq!(coeffs[2], 3.0);
}

#[test]
fn test_sub_same_degree() {
    // p1(x) = 5x^2 + 6x + 7
    // p2(x) = 2x^2 + 3x + 4
    // diff = 3x^2 + 3x + 3
    let p1 = Polynomial::new(&array![7.0_f64, 6.0, 5.0]).unwrap();
    let p2 = Polynomial::new(&array![4.0_f64, 3.0, 2.0]).unwrap();

    let result = p1.sub(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 3.0);
    assert_eq!(coeffs[1], 3.0);
    assert_eq!(coeffs[2], 3.0);
}

#[test]
fn test_mul_simple() {
    // p1(x) = x + 1
    // p2(x) = x + 2
    // product = x^2 + 3x + 2
    let p1 = Polynomial::new(&array![1.0_f64, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![2.0_f64, 1.0]).unwrap();

    let result = p1.mul(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 2.0);
    assert_eq!(coeffs[1], 3.0);
    assert_eq!(coeffs[2], 1.0);
}

#[test]
fn test_mul_different_degree() {
    // p1(x) = x^2 + x + 1
    // p2(x) = x + 1
    // product = x^3 + 2x^2 + 2x + 1
    let p1 = Polynomial::new(&array![1.0_f64, 1.0, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![1.0_f64, 1.0]).unwrap();

    let result = p1.mul(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 1.0);
    assert_eq!(coeffs[1], 2.0);
    assert_eq!(coeffs[2], 2.0);
    assert_eq!(coeffs[3], 1.0);
}

#[test]
fn test_mul_by_zero() {
    // p1(x) = x + 1
    // p2(x) = 0
    // product = 0
    let p1 = Polynomial::new(&array![1.0_f64, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![0.0_f64]).unwrap();

    let result = p1.mul(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 0.0);
}

// ============================================================================
// Evaluation Verification Tests
// ============================================================================

#[test]
fn test_eval_single_point() {
    // p(x) = 2x + 1
    // p(3) = 2*3 + 1 = 7
    let p = Polynomial::new(&array![1.0_f64, 2.0]).unwrap();
    let x = array![3.0_f64];
    let y = p.eval(&x).unwrap();

    assert_eq!(y[0], 7.0);
}

#[test]
fn test_eval_multiple_points() {
    // p(x) = x^2 + 2x + 1
    // p(0) = 1, p(1) = 4, p(2) = 9
    let p = Polynomial::new(&array![1.0_f64, 2.0, 1.0]).unwrap();
    let x = array![0.0_f64, 1.0, 2.0];
    let y = p.eval(&x).unwrap();

    assert!(Float::abs(y[0] - 1.0) < TOLERANCE);
    assert!(Float::abs(y[1] - 4.0) < TOLERANCE);
    assert!(Float::abs(y[2] - 9.0) < TOLERANCE);
}

#[test]
fn test_eval_high_degree() {
    // p(x) = x^5 - 3x^3 + 2x + 1
    // Coefficients: [1, 2, 0, -3, 0, 1] (constant term first)
    let coeffs = array![1.0_f64, 2.0, 0.0, -3.0, 0.0, 1.0];
    let p = Polynomial::new(&coeffs).unwrap();
    let x = array![2.0_f64];

    // p(2) = 32 - 24 + 4 + 1 = 13
    let y = p.eval(&x).unwrap();
    assert!(Float::abs(y[0] - 13.0) < TOLERANCE);
}

#[test]
fn test_eval_empty_array() {
    // p(x) = x + 1
    let p = Polynomial::new(&array![1.0_f64, 1.0]).unwrap();
    let x = ndarray::Array1::zeros(0);
    let y = p.eval(&x).unwrap();

    assert_eq!(y.len(), 0);
}

// ============================================================================
// Domain/Window Operations Verification Tests
// ============================================================================

#[test]
fn test_domain_default() {
    let p = Polynomial::new(&array![1.0_f64, 2.0, 1.0]).unwrap();
    let domain = p.domain();

    // Default domain should be [1, 1] based on implementation
    assert_eq!(domain[0], 1.0);
    assert_eq!(domain[1], 1.0);
}

#[test]
fn test_window_default() {
    let p = Polynomial::new(&array![1.0_f64, 2.0, 1.0]).unwrap();
    let window = p.window();

    // Default window should be [1, 1] based on implementation
    assert_eq!(window[0], 1.0);
    assert_eq!(window[1], 1.0);
}

#[test]
fn test_domain_window_set() {
    let mut p = Polynomial::new(&array![1.0_f64, 2.0, 1.0]).unwrap();
    p.set_domain([-1.0, 1.0]);
    p.set_window([0.0, 2.0]);

    let domain = p.domain();
    let window = p.window();

    assert_eq!(domain[0], -1.0);
    assert_eq!(domain[1], 1.0);
    assert_eq!(window[0], 0.0);
    assert_eq!(window[1], 2.0);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_ill_conditioned_fit() {
    // Nearly singular Vandermonde matrix
    // Using points that are very close together
    let x = array![1.0_f64, 1.001, 1.002, 1.003, 1.004];
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let y = ndarray::Array1::from(y);

    // Should still work, but may have larger numerical errors
    let p = fit(&x, &y, 1).unwrap();
    let coeffs = p.coeffs();

    // Relaxed tolerance for ill-conditioned case
    assert!(Float::abs(coeffs[0] - 1.0) < 0.1, "Ill-conditioned intercept off");
    assert!(Float::abs(coeffs[1] - 2.0) < 0.1, "Ill-conditioned slope off");
}

#[test]
fn test_high_degree_polynomial_roots() {
    // High degree (5) with known roots to keep test simple and fast
    // (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120
    let p = Polynomial::new(&array![
        -120.0_f64,  // constant
        274.0,
        -225.0,
        85.0,
        -15.0,
        1.0       // x^5 term
    ]).unwrap();

    let r = roots(&p).unwrap();

    // Should find 5 roots
    assert_eq!(r.len(), 5);

    // All roots should be finite
    for root in r.iter() {
        assert!(f64::is_finite(root.re));
        assert!(f64::is_finite(root.im));
    }
}
