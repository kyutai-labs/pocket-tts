//! Tests for contiguity (C/F) and layout invariants
//! Per Issue #31: Derive contiguity (C/F) and layout invariants

use numpy::*;

#[test]
fn test_0d_array_contiguity() {
    // 0-D arrays are always contiguous
    let arr: Array<f64> = Array::from_scalar(42.0, vec![]);
    assert!(arr.is_c_contiguous());
    assert!(arr.is_f_contiguous());
    assert!(arr.is_contiguous());
}

#[test]
fn test_1d_array_contiguity() {
    // 1-D arrays are both C and F contiguous
    let arr: Array<f64> = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(arr.is_c_contiguous());
    assert!(arr.is_f_contiguous());
    assert!(arr.is_contiguous());
}

#[test]
fn test_2d_c_contiguous() {
    // 2x3 array in C (row-major) order
    let arr: Array<f64> = Array::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert!(arr.is_c_contiguous());
    // 2D array with default strides is C-contiguous, not F-contiguous
    assert!(!arr.is_f_contiguous());
    assert!(arr.is_contiguous());
}

#[test]
fn test_2d_strides() {
    // Verify strides for 2x3 C-contiguous array
    let arr: Array<f64> = Array::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    // C-order strides for [2, 3]: [3, 1]
    assert_eq!(arr.strides(), &[3, 1]);
}

#[test]
fn test_contiguity_invariants() {
    // Empty arrays
    let empty: Array<f64> = Array::zeros(vec![0]);
    assert!(empty.is_c_contiguous());

    // Single element arrays
    let single: Array<f64> = Array::from_data(vec![1.0], vec![1, 1, 1]);
    assert!(single.is_c_contiguous());

    // 3D array
    let arr3d: Array<f64> = Array::zeros(vec![2, 3, 4]);
    assert!(arr3d.is_c_contiguous());
    // Strides should be [12, 4, 1] for shape [2, 3, 4]
    assert_eq!(arr3d.strides(), &[12, 4, 1]);
}
