use numpy::array::Array;
use numpy::broadcasting::{are_shapes_compatible, broadcast_arrays, broadcast_to};

// Test advanced broadcasting edge cases to match NumPy behavior exactly

#[test]
fn test_broadcast_scalar_to_2d() {
    let scalar = Array::from_vec(vec![1i32]);
    let matrix = Array::from_vec(vec![1i32, 2, 3, 4])
        .reshape(&[2, 2])
        .unwrap();

    let broadcasted = broadcast_to(&scalar, &[2, 2]).unwrap();
    assert_eq!(broadcasted.shape(), &[2, 2]);
    assert_eq!(broadcasted.strides(), &[0, 0]); // Scalar broadcast
    assert_eq!(broadcasted.to_vec(), vec![1, 1, 1, 1]);
}

#[test]
fn test_broadcast_1d_to_2d_row() {
    let row = Array::from_vec(vec![1i32, 2]);
    let result = broadcast_to(&row, &[3, 2]).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.strides(), &[0, 1]); // 0-stride for broadcasted dim
    assert_eq!(result.to_vec(), vec![1, 2, 1, 2, 1, 2]);
}

#[test]
fn test_broadcast_1d_to_2d_column() {
    let col = Array::from_vec(vec![1i32, 2, 3]).reshape(&[3, 1]).unwrap();
    let result = broadcast_to(&col, &[3, 4]).unwrap();

    assert_eq!(result.shape(), &[3, 4]);
    assert_eq!(result.strides(), &[1, 0]); // 0-stride for broadcasted dim
    assert_eq!(result.to_vec(), vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
}

#[test]
fn test_broadcast_incompatible_shapes() {
    let a = Array::from_vec(vec![1i32, 2, 3]);
    let _b = Array::from_vec(vec![1i32, 2]).reshape(&[2, 1]).unwrap();

    // [3] and [2, 1] should be compatible: (1, 3) and (2, 1) -> (2, 3)
    assert!(are_shapes_compatible(&[1, 3], &[2, 1]));

    let result = broadcast_to(&a, &[2, 3]);
    assert!(result.is_ok()); // [3] -> [2, 3] should work

    let result = broadcast_to(&a, &[2, 2]);
    assert!(result.is_err()); // [3] -> [2, 2] should fail
}

#[test]
fn test_broadcast_arrays_multiple() {
    let a = Array::from_vec(vec![1i32, 2, 3]);
    let b = Array::from_vec(vec![10i32]).reshape(&[1, 1]).unwrap();
    let c = Array::from_vec(vec![100i32]);

    let broadcasted = broadcast_arrays(&[&a, &b, &c]).unwrap();

    // Should all broadcast to [1, 3]
    assert_eq!(broadcasted[0].shape(), &[1, 3]);
    assert_eq!(broadcasted[1].shape(), &[1, 3]);
    assert_eq!(broadcasted[2].shape(), &[1, 3]);

    assert_eq!(broadcasted[0].to_vec(), vec![1, 2, 3]);
    assert_eq!(broadcasted[1].to_vec(), vec![10, 10, 10]);
    assert_eq!(broadcasted[2].to_vec(), vec![100, 100, 100]);
}

#[test]
fn test_broadcast_edge_case_zero_dimensions() {
    let a = Array::from_vec(vec![1i32, 2, 3, 4])
        .reshape(&[2, 2])
        .unwrap();
    let scalar = Array::from_vec(vec![5i32]);

    let broadcasted = broadcast_to(&scalar, &[2, 2]).unwrap();
    assert_eq!(broadcasted.shape(), &[2, 2]);
    assert_eq!(broadcasted.strides(), &[0, 0]);
}

#[test]
fn test_broadcast_complex_shapes() {
    // Test: (3, 1, 4) + (1, 5, 1) -> (3, 5, 4)
    let a = Array::from_vec(vec![1i32; 12]).reshape(&[3, 1, 4]).unwrap();
    let b = Array::from_vec(vec![10i32; 5]).reshape(&[1, 5, 1]).unwrap();

    let broadcasted = broadcast_arrays(&[&a, &b]).unwrap();
    assert_eq!(broadcasted[0].shape(), &[3, 5, 4]);
    assert_eq!(broadcasted[1].shape(), &[3, 5, 4]);
    assert_eq!(broadcasted[0].to_vec(), vec![1; 60]);
    assert_eq!(broadcasted[1].to_vec(), vec![10; 60]);
}

#[test]
fn test_broadcast_comparison_edge_case() {
    // This is the failing test case from comparison_tests.rs
    let a = Array::from_vec(vec![1i32, 2, 3, 4])
        .reshape(&[2, 2])
        .unwrap();
    let b = Array::from_vec(vec![1i32]);

    // Manual broadcasting to debug
    let broadcasted_b = broadcast_to(&b, &[2, 2]).unwrap();
    assert_eq!(broadcasted_b.shape(), &[2, 2]);
    assert_eq!(broadcasted_b.to_vec(), vec![1, 1, 1, 1]);

    // Now test element-wise comparison
    let a_flat = a.to_vec();
    let b_flat = broadcasted_b.to_vec();
    let mut result = Vec::new();

    for (i, &a_val) in a_flat.iter().enumerate() {
        result.push(a_val > b_flat[i]);
    }

    assert_eq!(result, vec![false, true, true, true]);
}

#[test]
fn test_broadcast_memory_view() {
    // Ensure broadcasting creates views, not copies
    let original = Array::from_vec(vec![42i32]);
    let broadcasted = broadcast_to(&original, &[10]).unwrap();

    // Check that strides indicate a view (0-strides)
    assert_eq!(broadcasted.strides(), &[0]);

    // All elements should be the same as original
    for i in 0..broadcasted.size() {
        assert_eq!(broadcasted.get_linear(i), Some(&42));
    }
}

#[test]
fn test_broadcast_to_same_shape() {
    let a = Array::from_vec(vec![1i32, 2, 3, 4])
        .reshape(&[2, 2])
        .unwrap();
    let result = broadcast_to(&a, &[2, 2]).unwrap();

    // Should return the same array (or identical copy)
    assert_eq!(result.shape(), a.shape());
    assert_eq!(result.to_vec(), a.to_vec());
}

#[test]
fn test_broadcast_empty_arrays() {
    let result = broadcast_arrays::<i32>(&[]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}
