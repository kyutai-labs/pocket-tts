//! Tests for sign and absolute value functions

use ndarray::array;
use numpy::*;

#[test]
fn test_sign() {
    // Test with various values including negative, positive, zero, NaN
    let x = Array::from_data(vec![-2.5_f64, -0.0, 0.0, 1.5, f64::NAN], vec![5]);
    let result = sign(&x).unwrap();

    assert_eq!(result.get(0).unwrap(), &-1.0);
    assert_eq!(result.get(1).unwrap(), &0.0); // -0.0 should return 0
    assert_eq!(result.get(2).unwrap(), &0.0);
    assert_eq!(result.get(3).unwrap(), &1.0);
    assert!(result.get(4).unwrap().is_nan());
}

#[test]
fn test_signbit() {
    // Test signbit function
    let x = Array::from_data(vec![-2.5_f64, -0.0, 0.0, 1.5, f64::NAN], vec![5]);
    let result = signbit(&x).unwrap();

    assert_eq!(result.get(0).unwrap(), &1.0); // negative
    assert_eq!(result.get(1).unwrap(), &1.0); // -0.0
    assert_eq!(result.get(2).unwrap(), &0.0); // +0.0
    assert_eq!(result.get(3).unwrap(), &0.0); // positive
    assert_eq!(result.get(4).unwrap(), &1.0); // NaN
}

#[test]
fn test_copysign() {
    // Test copysign function
    let x1 = Array::from_data(vec![2.5_f64, -3.7_f64, 0.0], vec![3]);
    let x2 = Array::from_data(vec![-1.0, 1.0, -0.0], vec![3]);
    let result = copysign(&x1, &x2).unwrap();

    assert_eq!(result.get(0).unwrap(), &-2.5); // magnitude of 2.5, sign of -1.0
    assert_eq!(result.get(1).unwrap(), &3.7); // magnitude of -3.7, sign of 1.0
    assert_eq!(result.get(2).unwrap(), &-0.0); // magnitude of 0.0, sign of -0.0
}

#[test]
fn test_absolute() {
    // Test absolute value function
    let x = Array::from_data(vec![-2.5_f64, -0.0, 0.0, 1.5, f64::NAN], vec![5]);
    let result = absolute(&x).unwrap();

    assert_eq!(result.get(0).unwrap(), &2.5);
    assert_eq!(result.get(1).unwrap(), &0.0);
    assert_eq!(result.get(2).unwrap(), &0.0);
    assert_eq!(result.get(3).unwrap(), &1.5);
    assert!(result.get(4).unwrap().is_nan());
}

#[test]
fn test_abs() {
    // Test abs alias
    let x = Array::from_data(vec![-2.5_f64, 0.0, 3.7], vec![3]);
    let result = abs(&x).unwrap();

    assert_eq!(result.get(0).unwrap(), &2.5);
    assert_eq!(result.get(1).unwrap(), &0.0);
    assert_eq!(result.get(2).unwrap(), &3.7);
}

#[test]
fn test_fabs() {
    // Test fabs function (float-only)
    let x = Array::from_data(vec![-2.5_f32, 0.0, 3.7], vec![3]);
    let result = fabs(&x).unwrap();

    assert_eq!(result.get(0).unwrap(), &2.5);
    assert_eq!(result.get(1).unwrap(), &0.0);
    assert_eq!(result.get(2).unwrap(), &3.7);
}

#[test]
fn test_sign_complex() {
    // Test sign with complex numbers
    let x = Array::from_data(
        vec![
            num_complex::Complex64::new(-2.5, 1.0),
            num_complex::Complex64::new(0.0, -1.0),
            num_complex::Complex64::new(1.5, 2.0),
        ],
        vec![3],
    );
    let result = sign(&x).unwrap();

    // For complex numbers, sign should be based on real part
    assert_eq!(result.get(0).unwrap(), &-1.0);
    assert_eq!(result.get(1).unwrap(), &0.0);
    assert_eq!(result.get(2).unwrap(), &1.0);
}

#[test]
fn test_absolute_complex() {
    // Test absolute value with complex numbers
    let x = Array::from_data(
        vec![
            num_complex::Complex64::new(-3.0, 4.0),
            num_complex::Complex64::new(5.0, -2.0),
        ],
        vec![2],
    );
    let result = absolute(&x).unwrap();

    assert_eq!(
        result.get(0).unwrap(),
        &num_complex::Complex64::new(3.0, 4.0)
    );
    assert_eq!(
        result.get(1).unwrap(),
        &num_complex::Complex64::new(5.0, 2.0)
    );
}

#[test]
fn test_sign_2d_array() {
    // Test with 2D array
    let x = Array::from_data(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
    let result = sign(&x).unwrap();

    assert_eq!(result.shape(), &vec![2, 2]);
    assert_eq!(result.get(0).unwrap(), &-1.0);
    assert_eq!(result.get(1).unwrap(), &1.0);
    assert_eq!(result.get(2).unwrap(), &-1.0);
    assert_eq!(result.get(3).unwrap(), &1.0);
}

#[test]
fn test_copysign_broadcasting() {
    // Test copysign with broadcasting
    let x1 = Array::from_data(vec![1.0, 2.0], vec![2]);
    let x2 = Array::from_data(vec![-1.0], vec![1]);
    let result = copysign(&x1, &x2).unwrap();

    assert_eq!(result.get(0).unwrap(), &-1.0);
    assert_eq!(result.get(1).unwrap(), &-2.0);
}
