// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Tests for choose and compress functions

use numpy::array_extra::{choose, compress};
use numpy::Array;

#[test]
fn test_choose_basic() {
    let index = Array::from_vec(vec![0, 1, 2, 0]);
    let c0 = Array::from_vec(vec![1, 2, 3, 4]);
    let c1 = Array::from_vec(vec![10, 20, 30, 40]);
    let c2 = Array::from_vec(vec![100, 200, 300, 400]);
    let choices = vec![&c0, &c1, &c2];

    let result = choose(&index, &choices, "raise").unwrap();
    assert_eq!(result.to_vec(), vec![1, 20, 300, 4]);
}

#[test]
fn test_choose_wrap_mode() {
    let index = Array::from_vec(vec![0, 3, 1, 4]);
    let c0 = Array::from_vec(vec![1, 2, 3, 4, 5]);
    let c1 = Array::from_vec(vec![10, 20, 30, 40, 50]);
    let c2 = Array::from_vec(vec![100, 200, 300, 400, 500]);
    let choices = vec![&c0, &c1, &c2];

    // Wrap mode: wraps both choice index and element index
    let result = choose(&index, &choices, "wrap").unwrap();
    assert_eq!(result.to_vec(), vec![1, 4, 20, 50]);
}

#[test]
fn test_choose_clip_mode() {
    let index = Array::from_vec(vec![0, 3, 1, -1]);
    let c0 = Array::from_vec(vec![1, 2, 3, 4]);
    let c1 = Array::from_vec(vec![10, 20, 30, 40]);
    let choices = vec![&c0, &c1];

    // Clip mode: clips choice index, clamps element index to array bounds
    let result = choose(&index, &choices, "clip").unwrap();
    assert_eq!(result.to_vec(), vec![1, 40, 20, 1]);
}

#[test]
fn test_choose_raise_out_of_bounds() {
    let index = Array::from_vec(vec![0, 5, 1]); // 5 is out of bounds
    let c0 = Array::from_vec(vec![1, 2, 3]);
    let c1 = Array::from_vec(vec![10, 20, 30]);
    let c2 = Array::from_vec(vec![100, 200, 300]);
    let choices = vec![&c0, &c1, &c2];

    let result = choose(&index, &choices, "raise");
    assert!(result.is_err());
}

#[test]
fn test_choose_negative_indices_raise() {
    let index = Array::from_vec(vec![0, -1, 1]); // -1 should raise
    let c0 = Array::from_vec(vec![1, 2, 3]);
    let c1 = Array::from_vec(vec![10, 20, 30]);
    let choices = vec![&c0, &c1];

    let result = choose(&index, &choices, "raise");
    assert!(result.is_err());
}

#[test]
fn test_choose_empty_choices_error() {
    let index = Array::from_vec(vec![0, 1, 2]);
    let choices: Vec<&Array<i32>> = vec![];

    let result = choose(&index, &choices, "raise");
    assert!(result.is_err());
}

#[test]
fn test_choose_invalid_mode_error() {
    let index = Array::from_vec(vec![0, 1]);
    let c0 = Array::from_vec(vec![1, 2]);
    let c1 = Array::from_vec(vec![10, 20]);
    let choices = vec![&c0, &c1];

    let result = choose(&index, &choices, "invalid");
    assert!(result.is_err());
}

#[test]
fn test_compress_1d_basic() {
    let condition = Array::from_vec(vec![true, false, true, false, true]);
    let array = Array::from_vec(vec![1, 2, 3, 4, 5]);

    let result = compress(&condition, &array, None).unwrap();
    assert_eq!(result.to_vec(), vec![1, 3, 5]);
}

#[test]
fn test_compress_1d_all_true() {
    let condition = Array::from_vec(vec![true, true, true]);
    let array = Array::from_vec(vec![10, 20, 30]);

    let result = compress(&condition, &array, None).unwrap();
    assert_eq!(result.to_vec(), vec![10, 20, 30]);
}

#[test]
fn test_compress_1d_all_false() {
    let condition = Array::from_vec(vec![false, false, false]);
    let array = Array::from_vec(vec![1, 2, 3]);

    let result = compress(&condition, &array, None).unwrap();
    assert_eq!(result.to_vec(), Vec::<i32>::new());
}

#[test]
fn test_compress_2d_rows() {
    let condition = Array::from_vec(vec![true, false, true]);
    let array = Array::from_shape_vec(vec![3, 2], vec![1, 2, 3, 4, 5, 6]);

    let result = compress(&condition, &array, Some(0)).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![1, 2, 5, 6]);
}

#[test]
fn test_compress_2d_columns() {
    let condition = Array::from_vec(vec![true, false, true]);
    let array = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);

    let result = compress(&condition, &array, Some(1)).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![1, 3, 4, 6]);
}

#[test]
fn test_compress_2d_axis_negative() {
    let condition = Array::from_vec(vec![true, false, true]);
    let array = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);

    // -1 refers to last axis (columns)
    let result = compress(&condition, &array, Some(-1)).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![1, 3, 4, 6]);
}

#[test]
fn test_compress_shape_mismatch() {
    let condition = Array::from_vec(vec![true, false]); // 2 elements
    let array = Array::from_vec(vec![1, 2, 3]); // 3 elements

    let result = compress(&condition, &array, None);
    assert!(result.is_err());
}

#[test]
fn test_compress_axis_out_of_bounds() {
    let condition = Array::from_vec(vec![true, false]);
    let array = Array::from_vec(vec![1, 2]);

    let result = compress(&condition, &array, Some(5)); // axis 5 doesn't exist
    assert!(result.is_err());
}

#[test]
fn test_compress_2d_axis_shape_mismatch() {
    let condition = Array::from_vec(vec![true, false]); // 2 elements
    let array = Array::from_shape_vec(vec![3, 2], vec![1, 2, 3, 4, 5, 6]); // 3 rows

    let result = compress(&condition, &array, Some(0));
    assert!(result.is_err());
}

#[test]
fn test_compress_float_values() {
    let condition = Array::from_vec(vec![true, false, true]);
    let array = Array::from_vec(vec![1.5, 2.5, 3.5]);

    let result = compress(&condition, &array, None).unwrap();
    assert_eq!(result.to_vec(), vec![1.5, 3.5]);
}

#[test]
fn test_compress_empty_array() {
    let condition = Array::from_vec(vec![]);
    let array: Array<i32> = Array::from_vec(vec![]);

    let result = compress(&condition, &array, None).unwrap();
    assert_eq!(result.to_vec(), Vec::<i32>::new());
}
