use numpy::Array;

#[test]
fn test_broadcast_scalar() {
    let arr = Array::from_scalar(42.0, vec![]);
    let broadcasted = arr.broadcast_to(&[2, 3]).unwrap();

    assert_eq!(broadcasted.shape(), &[2, 3]);
    assert_eq!(broadcasted.strides(), &[0, 0]); // Scalar broadcast has 0 strides
    assert_eq!(broadcasted.size(), 6);

    let vec = broadcasted.to_vec();
    assert_eq!(vec, vec![42.0; 6]);
}

#[test]
fn test_broadcast_1d_to_2d() {
    let arr = Array::from_vec(vec![1, 2, 3]);
    let broadcasted = arr.broadcast_to(&[2, 3]).unwrap();

    assert_eq!(broadcasted.shape(), &[2, 3]);
    // Original strides for [3] is [1].
    // Broadcast to [2, 3]: new dim (stride 0), kept dim (stride 1).
    assert_eq!(broadcasted.strides(), &[0, 1]);

    let vec = broadcasted.to_vec();
    assert_eq!(vec, vec![1, 2, 3, 1, 2, 3]);
}

#[test]
fn test_broadcast_view_structure() {
    let arr = Array::from_vec(vec![10]);
    let broadcasted = arr.broadcast_to(&[3]).unwrap();

    // Check structure
    assert_eq!(broadcasted.shape(), &[3]);
    assert_eq!(broadcasted.strides(), &[0]);
    assert_eq!(broadcasted.size(), 3);

    // Check data access
    assert_eq!(broadcasted.get_linear(0), Some(&10));
    assert_eq!(broadcasted.get_linear(1), Some(&10));
    assert_eq!(broadcasted.get_linear(2), Some(&10));

    // NOTE: Shared mutability is not yet supported by Arc<MemoryManager>.
    // Writing to a view currently fails silently or is disallowed by API structure.
    // For Issue #34 we verify the Layout (0-strides) and data sharing (read-only).
}

#[test]
fn test_broadcast_incompatible() {
    let arr = Array::from_vec(vec![1, 2]);
    let res = arr.broadcast_to(&[3]);
    assert!(res.is_err());
}
