// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Tests for diagonal and diag functions

use numpy::array_extra::{diag, diagonal};
use numpy::Array;

#[test]
fn test_diagonal_main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let arr = Array::from_shape_vec(vec![3, 3], data);

    let result = diagonal(&arr, 0, 0, 1).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![1.0, 5.0, 9.0]);
}

#[test]
fn test_diagonal_upper() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let arr = Array::from_shape_vec(vec![3, 3], data);

    let result = diagonal(&arr, 1, 0, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![2.0, 6.0]);
}

#[test]
fn test_diagonal_lower() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let arr = Array::from_shape_vec(vec![3, 3], data);

    let result = diagonal(&arr, -1, 0, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![4.0, 8.0]);
}

#[test]
fn test_diagonal_rectangular() {
    let data = vec
![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let arr = Array::from_shape_vec(vec![2, 4], data);

    let result = diagonal(&arr, 0, 0, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![1.0, 6.0]);

    let result = diagonal(&arr, 1, 0, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![2.0, 7.0]);

    let result = diagonal(&arr, 2, 0, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![3.0, 8.0]);

    let result = diagonal(&arr, -1, 0, 1).unwrap();
    assert_eq!(result.shape(), &[1]);
    assert_eq!(result.to_vec(), vec![5.0]);
}

#[test]
fn test_diagonal_offset_out_of_bounds() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let arr = Array::from_shape_vec(vec![2, 2], data);

    // Upper diagonal beyond array returns empty
    let result = diagonal(&arr, 10, 0, 1).unwrap();
    assert_eq!(result.shape(), &[0]);

    // Lower diagonal beyond array returns empty
    let result = diagonal(&arr, -10, 0, 1).unwrap();
    assert_eq!(result.shape(), &[0]);
}

#[test]
fn test_diagonal_1d_error() {
    let data = vec![1.0, 2.0, 3.0];
    let arr = Array::from_vec(data);

    let result = diagonal(&arr, 0, 0, 1);
    assert!(result.is_err());
}

#[test]
fn test_diag_extract_2d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let arr = Array::from_shape_vec(vec![3, 3], data);

    // Extract main diagonal
    let result = diag(&arr, 0).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![1.0, 5.0, 9.0]);

    // Extract upper diagonal
    let result = diag(&arr, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![2.0, 6.0]);

    // Extract lower diagonal
    let result = diag(&arr, -1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![4.0, 8.0]);
}

#[test]
fn test_diag_construct_1d() {
    let data = vec![1.0, 2.0, 3.0];
    let arr = Array::from_vec(data);

    // Construct with main diagonal (k=0)
    let result = diag(&arr, 0).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    let expected = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    assert_eq!(result.to_vec(), expected);

    // Construct with upper diagonal (k=1)
    let result = diag(&arr, 1).unwrap();
    assert_eq!(result.shape(), &[4, 4]);
    let expected = vec
![
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(result.to_vec(), expected);

    // Construct with lower diagonal (k=-1)
    let result = diag(&arr, -1).unwrap();
    assert_eq!(result.shape(), &[4, 4]);
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
    ];
    assert_eq!(result.to_vec(), expected);
}

#[test]
fn test_diag_1d_integer() {
    let data = vec![1, 2, 3];
    let arr = Array::from_vec(data);

    let result = diag(&arr, 0).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    let expected = vec![1, 0, 0, 0, 2, 0, 0, 0, 3];
    assert_eq!(result.to_vec(), expected);
}

#[test]
fn test_diag_3d_error() {
    let data = vec
![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let arr = Array::from_shape_vec(vec![2, 2, 2], data);

    let result = diag(&arr, 0);
    assert!(result.is_err());
}

#[test]
fn test_diagonal_with_integer_type() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let arr = Array::from_shape_vec(vec![3, 3], data);

    let result = diagonal(&arr, 0, 0, 1).unwrap();
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![1, 5, 9]);
}

#[test]
fn test_diagonal_non_square() {
    // 4x2 array
    let data: Vec<i32> = (1..=8).collect();
    let arr = Array::from_shape_vec(vec![4, 2], data);

    let result = diagonal(&arr, 0, 0, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![1, 4]);

    let result = diagonal(&arr, -1, 0, 1).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.to_vec(), vec![3, 6]);
}
