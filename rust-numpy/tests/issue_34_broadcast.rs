use numpy::array::Array;
use numpy::broadcasting::broadcast_to;
use numpy::strides::{compute_broadcast_shape, compute_broadcast_strides};

#[test]
fn test_broadcast_stride_zero() {
    // Array of shape (1, 3)
    let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]).unwrap();

    // Check initial strides
    // For [1, 3], strides should be [3, 1] normally.
    // But conceptually for dim 0 (size 1), does stride matter?
    // Yes, linear index = i*3 + j*1.
    // If i=0, j=0, index=0. i=0, j=1, index=1.
    assert_eq!(a.strides(), &[3, 1]);

    // Now pretend we broadcast to (4, 3)
    let target_shape = vec![4, 3];
    let broadcast_strides = compute_broadcast_strides(a.shape(), a.strides(), &target_shape);

    // Strides should be [0, 1] because dim 0 is broadcasted.
    // 0 * i + 1 * j.
    // i=0..3, j=0..2. All map to same row of 'a'.
    assert_eq!(broadcast_strides, &[0, 1]);
}

#[test]
fn test_broadcast_to_view_semantics() {
    // Array of shape (1, 3)
    let a = Array::from_vec(vec![1, 2, 3]).reshape(&[1, 3]).unwrap();

    // Broadcast to (4, 3)
    // Note: We need to use the public API for broadcasting.
    // Assuming numpy::broadcasting::broadcast_to is exposed or will be.

    let b = broadcast_to(&a, &[4, 3]).expect("Broadcast failed");

    // Check shape and strides
    assert_eq!(b.shape(), &[4, 3]);
    assert_eq!(b.strides(), &[0, 1]); // This should be [0, 1] if it's a view

    // Check data sharing (not possible to check pointer equality easily without unsafe,
    // but we can check if it's a view by nature of implementation or by modifying if we had mutable views)
    // For now, checking strides is sufficient proof it's using the 0-stride trick.

    // If it was a copy, it would likely be C-contiguous: [3, 1]
    // because Array::zeros creates C-contiguous arrays.
    assert_ne!(
        b.strides(),
        &[3, 1],
        "Broadcast returned a copy, expected a view"
    );
}

#[test]
fn test_manual_broadcast_creation() {
    let a = Array::from_vec(vec![10]).reshape(&[1]).unwrap();
    // Shape (1), Stride (1)

    let target_shape = vec![5];
    let b_strides = compute_broadcast_strides(a.shape(), a.strides(), &target_shape);

    assert_eq!(b_strides, &[0]);

    // Problem: Can we actually creating an array with these strides?
    // And does get() work?

    // Note: Array::new_with_strides isn't public or doesn't exist yet, we might need to bypass or add it.
    // But we can check if we can conceptually support it.
}
