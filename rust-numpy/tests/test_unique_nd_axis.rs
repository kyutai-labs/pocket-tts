// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! Tests for N-D axis support in unique() function

use numpy::array::Array;
use numpy::error::Result;
use numpy::set_ops::unique;

#[test]
fn test_unique_2d_axis_0() -> Result<()> {
    // Test 2D array with axis=0 (default behavior)
    let data = vec![1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3];
    let arr = Array::from_data(data, vec![4, 3]);

    let result = unique(&arr, true, true, true, Some(&[0]))?;

    // Should have 2 unique rows: [1,2,3] and [4,5,6]
    assert_eq!(result.values.shape(), &[2, 3]);
    assert_eq!(result.indices.unwrap().to_vec(), vec![0, 2]);
    assert_eq!(result.inverse.unwrap().to_vec(), vec![0, 0, 1, 0]);
    assert_eq!(result.counts.unwrap().to_vec(), vec![3, 1]);

    Ok(())
}

#[test]
fn test_unique_2d_axis_1() -> Result<()> {
    // Test 2D array with axis=1 (columns)
    let data = vec![1, 2, 1, 3, 4, 3, 5, 6, 5];
    let arr = Array::from_data(data, vec![3, 3]);

    let result = unique(&arr, false, false, false, Some(&[1]))?;

    // Should have 2 unique columns: [1,3,5] and [2,4,6]
    assert_eq!(result.values.shape(), &[3, 2]);

    // Check the values
    let values = result.values.to_vec();
    assert_eq!(values[0..2], vec![1, 2]);
    assert_eq!(values[2..4], vec![3, 4]);
    assert_eq!(values[4..6], vec![5, 6]);

    Ok(())
}

#[test]
fn test_unique_3d_axis_0() -> Result<()> {
    // Test 3D array with axis=0
    let data = vec![
        // First 2x2 slice
        1, 2, 3, 4, // Second 2x2 slice (same as first)
        1, 2, 3, 4, // Third 2x2 slice (different)
        5, 6, 7, 8,
    ];
    let arr = Array::from_data(data, vec![3, 2, 2]);

    let result = unique(&arr, true, true, true, Some(&[0]))?;

    // Should have 2 unique slices along axis=0
    assert_eq!(result.values.shape(), &[2, 2, 2]);
    assert_eq!(result.indices.unwrap().to_vec(), vec![0, 2]);
    assert_eq!(result.inverse.unwrap().to_vec(), vec![0, 0, 1]);
    assert_eq!(result.counts.unwrap().to_vec(), vec![2, 1]);

    Ok(())
}

#[test]
fn test_unique_3d_axis_1() -> Result<()> {
    // Test 3D array with axis=1
    let data = vec![
        // First slice
        1, 2, 1, 2, // Second slice
        3, 4, 3, 4,
    ];
    let arr = Array::from_data(data, vec![2, 2, 2]);

    let result = unique(&arr, false, false, false, Some(&[1]))?;

    // Should have 1 unique row along axis=1 (both rows are identical)
    assert_eq!(result.values.shape(), &[2, 1, 2]);

    Ok(())
}

#[test]
fn test_unique_3d_axis_2() -> Result<()> {
    // Test 3D array with axis=2 (last axis)
    let data = vec![
        // First slice
        1, 2, 1, 3, 4, 3, // Second slice
        5, 6, 5, 7, 8, 7,
    ];
    let arr = Array::from_data(data, vec![2, 2, 3]);

    let result = unique(&arr, true, false, false, Some(&[2]))?;

    // Should have 2 unique columns along axis=2
    assert_eq!(result.values.shape(), &[2, 2, 2]);
    assert_eq!(result.indices.unwrap().to_vec(), vec![0, 1]);

    Ok(())
}

#[test]
fn test_unique_4d_axis_2() -> Result<()> {
    // Test 4D array with middle axis
    let shape = vec![2, 3, 2, 2];
    let mut data = Vec::new();

    // Create data with duplicate slices along axis=2
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                for l in 0..2 {
                    // Make slices at k=0 and k=1 identical for some j
                    let val = if k == 0 || (j == 1) {
                        (i * 6 + j * 2 + l + 1) as i32
                    } else {
                        (i * 6 + j * 2 + l + 10) as i32
                    };
                    data.push(val);
                }
            }
        }
    }

    let arr = Array::from_data(data, shape);
    let result = unique(&arr, false, false, true, Some(&[2]))?;

    // Should have unique slices along axis=2
    assert_eq!(result.values.shape(), &[2, 3, 2, 2]);

    Ok(())
}

#[test]
fn test_unique_negative_axis() -> Result<()> {
    // Test negative axis indexing
    let data = vec![1, 2, 3, 4, 5, 6, 1, 2, 3];
    let arr = Array::from_data(data, vec![3, 3]);

    // axis=-1 should be same as axis=1 for 2D array
    let result = unique(&arr, false, false, false, Some(&[-1]))?;
    assert_eq!(result.values.shape(), &[3, 3]);

    // axis=-2 should be same as axis=0 for 2D array
    let result = unique(&arr, false, false, false, Some(&[-2]))?;
    assert_eq!(result.values.shape(), &[2, 3]);

    Ok(())
}

#[test]
fn test_unique_axis_out_of_bounds() {
    let data = vec![1, 2, 3, 4];
    let arr = Array::from_data(data, vec![2, 2]);

    // axis=2 should be out of bounds for 2D array
    let result = unique(&arr, false, false, false, Some(&[2]));
    assert!(result.is_err());

    // axis=-3 should be out of bounds for 2D array
    let result = unique(&arr, false, false, false, Some(&[-3]));
    assert!(result.is_err());
}

#[test]
fn test_unique_empty_array_with_axis() -> Result<()> {
    let arr: Array<i32> = Array::from_data(vec![], vec![0, 3]);
    let result = unique(&arr, false, false, false, Some(&[0]))?;

    assert_eq!(result.values.shape(), &[0, 3]);

    Ok(())
}

#[test]
fn test_unique_single_element_axis() -> Result<()> {
    let data = vec![1, 2, 3];
    let arr = Array::from_data(data, vec![1, 3]);

    let result = unique(&arr, true, true, true, Some(&[0]))?;

    assert_eq!(result.values.shape(), &[1, 3]);
    assert_eq!(result.indices.unwrap().to_vec(), vec![0]);
    assert_eq!(result.inverse.unwrap().to_vec(), vec![0]);
    assert_eq!(result.counts.unwrap().to_vec(), vec![1]);

    Ok(())
}
