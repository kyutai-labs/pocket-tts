use numpy::array::Array;
use numpy::array_extra::{dstack, stack};

#[test]
fn test_stack() {
    let a = Array::from_vec(vec![1, 2, 3]);
    let b = Array::from_vec(vec![4, 5, 6]);
    let arrays = vec![&a, &b];

    // stack with axis 0
    // Expected: [[1, 2, 3], [4, 5, 6]] -> [1, 2, 3, 4, 5, 6]
    let result = stack(&arrays, 0).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);

    // stack with axis 1
    // Expected: [[1, 4], [2, 5], [3, 6]] -> [1, 4, 2, 5, 3, 6]
    let result = stack(&arrays, 1).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.to_vec(), vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn test_dstack() {
    let a = Array::from_vec(vec![1, 2, 3]);
    let b = Array::from_vec(vec![2, 3, 4]);
    let arrays = vec![&a, &b];

    let result = dstack(&arrays).unwrap();
    // 1D arrays are treated as (1, N, 1) -> (1, 3, 2)
    // Result: [[[1, 2], [2, 3], [3, 4]]]
    // Flattened: [1, 2, 2, 3, 3, 4]
    assert_eq!(result.shape(), &[1, 3, 2]);
    assert_eq!(result.to_vec(), vec![1, 2, 2, 3, 3, 4]);
}

#[test]
fn test_dstack_2d() {
    let a = Array::from_shape_vec(vec![2, 2], vec![1, 2, 3, 4]);
    let b = Array::from_shape_vec(vec![2, 2], vec![5, 6, 7, 8]);
    let arrays = vec![&a, &b];

    let result = dstack(&arrays).unwrap();
    // 2D arrays (M, N) -> (M, N, 1) + (M, N, 1) -> (M, N, 2)
    // A: [[1, 2], [3, 4]]
    // B: [[5, 6], [7, 8]]
    // Result: [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]
    // Flattened: [1, 5, 2, 6, 3, 7, 4, 8]
    assert_eq!(result.shape(), &[2, 2, 2]);
    assert_eq!(result.to_vec(), vec![1, 5, 2, 6, 3, 7, 4, 8]);
}
