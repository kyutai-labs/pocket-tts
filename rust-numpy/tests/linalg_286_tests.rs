use numpy::array_extra::{tril, triu, vander};
use numpy::Array;

#[test]
fn test_triu_basic() {
    let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result = triu(&arr, 0).unwrap();
    assert_eq!(result.to_vec(), vec![1, 2, 3, 0, 5, 6, 0, 0, 9]);
}

#[test]
fn test_triu_offset() {
    let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result = triu(&arr, 1).unwrap();
    assert_eq!(result.to_vec(), vec![0, 2, 3, 0, 0, 6, 0, 0, 0]);
}

#[test]
fn test_tril_basic() {
    let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result = tril(&arr, 0).unwrap();
    assert_eq!(result.to_vec(), vec![1, 0, 0, 4, 5, 0, 7, 8, 9]);
}

#[test]
fn test_tril_offset() {
    let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result = tril(&arr, -1).unwrap();
    assert_eq!(result.to_vec(), vec![0, 0, 0, 4, 0, 0, 7, 8, 0]);
}

#[test]
fn test_vander_basic() {
    let x = Array::from_vec(vec![1, 2, 3]);
    let result = vander(&x, None, false).unwrap();
    assert_eq!(result.shape(), &[3, 3]);
    // [[1^2, 1^1, 1^0], [2^2, 2^1, 2^0], [3^2, 3^1, 3^0]]
    // [[1, 1, 1], [4, 2, 1], [9, 3, 1]]
    assert_eq!(result.to_vec(), vec![1, 1, 1, 4, 2, 1, 9, 3, 1]);
}

#[test]
fn test_vander_increasing() {
    let x = Array::from_vec(vec![1, 2, 3]);
    let result = vander(&x, Some(2), true).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    // [[1^0, 1^1], [2^0, 2^1], [3^0, 3^1]]
    // [[1, 1], [1, 2], [1, 3]]
    assert_eq!(result.to_vec(), vec![1, 1, 1, 2, 1, 3]);
}
