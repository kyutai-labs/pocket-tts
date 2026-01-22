use numpy::array_manipulation::{append, delete, insert};
use numpy::Array;

#[test]
fn test_insert_1d_single() {
    let arr = Array::from_shape_vec(vec![3], vec![1, 2, 3]);
    let indices = Array::from_shape_vec(vec![1], vec![1isize]);
    let values = Array::from_shape_vec(vec![1], vec![10]);
    let result = insert(&arr, &indices, &values, None).unwrap();
    assert_eq!(result.to_vec(), vec![1, 10, 2, 3]);
}

#[test]
fn test_insert_1d_multiple() {
    let arr = Array::from_shape_vec(vec![3], vec![1, 2, 3]);
    let indices = Array::from_shape_vec(vec![2], vec![0isize, 2]);
    let values = Array::from_shape_vec(vec![2], vec![10, 20]);
    let result = insert(&arr, &indices, &values, None).unwrap();
    assert_eq!(result.to_vec(), vec![10, 1, 2, 20, 3]);
}

#[test]
fn test_insert_1d_negative_index() {
    let arr = Array::from_shape_vec(vec![3], vec![1, 2, 3]);
    let indices = Array::from_shape_vec(vec![1], vec![-1isize]); // Insert before last element
    let values = Array::from_shape_vec(vec![1], vec![10]);
    let result = insert(&arr, &indices, &values, None).unwrap();
    assert_eq!(result.to_vec(), vec![1, 2, 10, 3]);
}

#[test]
fn test_insert_2d_axis_0() {
    let arr = Array::from_shape_vec(vec![2, 2], vec![1, 2, 3, 4]);
    let indices = Array::from_shape_vec(vec![1], vec![1isize]);
    let values = Array::from_shape_vec(vec![1, 2], vec![5, 6]);
    let result = insert(&arr, &indices, &values, Some(0)).unwrap();
    // Result should be [[1,2], [5,6], [3,4]]
    assert_eq!(result.shape(), &[3, 2]);
    let result_vec = result.to_vec();
    assert_eq!(result_vec[0..4], vec![1, 2, 5, 6]);
    assert_eq!(result_vec[4..6], vec![3, 4]);
}

#[test]
fn test_delete_1d_single() {
    let arr = Array::from_shape_vec(vec![4], vec![1, 2, 3, 4]);
    let indices = Array::from_shape_vec(vec![1], vec![1isize]);
    let result = delete(&arr, &indices, None).unwrap();
    assert_eq!(result.to_vec(), vec![1, 3, 4]);
}

#[test]
fn test_delete_1d_multiple() {
    let arr = Array::from_shape_vec(vec![5], vec![1, 2, 3, 4, 5]);
    let indices = Array::from_shape_vec(vec![2], vec![0isize, 3]);
    let result = delete(&arr, &indices, None).unwrap();
    assert_eq!(result.to_vec(), vec![2, 3, 5]);
}

#[test]
fn test_delete_1d_negative() {
    let arr = Array::from_shape_vec(vec![4], vec![1, 2, 3, 4]);
    let indices = Array::from_shape_vec(vec![1], vec![-1isize]); // Delete last element
    let result = delete(&arr, &indices, None).unwrap();
    assert_eq!(result.to_vec(), vec![1, 2, 3]);
}

#[test]
fn test_delete_2d_axis_0() {
    let arr = Array::from_shape_vec(vec![3, 2], vec![1, 2, 3, 4, 5, 6]);
    let indices = Array::from_shape_vec(vec![1], vec![1isize]); // Delete middle row
    let result = delete(&arr, &indices, Some(0)).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.to_vec(), vec![1, 2, 5, 6]);
}

#[test]
fn test_append_1d() {
    let arr = Array::from_shape_vec(vec![3], vec![1, 2, 3]);
    let values = Array::from_shape_vec(vec![2], vec![4, 5]);
    let result = append(&arr, &values, None).unwrap();
    assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_append_2d_axis_0() {
    let arr = Array::from_shape_vec(vec![2, 2], vec![1, 2, 3, 4]);
    let values = Array::from_shape_vec(vec![1, 2], vec![5, 6]);
    let result = append(&arr, &values, Some(0)).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_append_2d_axis_1() {
    let arr = Array::from_shape_vec(vec![2, 2], vec![1, 2, 3, 4]);
    let values = Array::from_shape_vec(vec![2, 1], vec![5, 6]);
    let result = append(&arr, &values, Some(1)).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.to_vec(), vec![1, 2, 5, 3, 4, 6]);
}

#[test]
fn test_insert_empty_array() {
    let arr = Array::from_shape_vec(vec![0], vec![0i32]);
    let indices = Array::from_shape_vec(vec![1], vec![0isize]);
    let values = Array::from_shape_vec(vec![1], vec![10]);
    let result = insert(&arr, &indices, &values, None).unwrap();
    assert_eq!(result.to_vec(), vec![10]);
}

#[test]
fn test_delete_all() {
    let arr = Array::from_shape_vec(vec![3], vec![1, 2, 3]);
    let indices = Array::from_shape_vec(vec![3], vec![0isize, 1, 2]);
    let result = delete(&arr, &indices, None).unwrap();
    assert!(result.is_empty());
}
