use numpy::strides::compute_broadcast_shape;
use numpy::Array;

#[test]
fn test_broadcast_zero_and_one() {
    let shape1 = vec![0];
    let shape2 = vec![1];

    // According to NumPy: np.broadcast_shapes((0,), (1,)) -> (0,)
    let result = compute_broadcast_shape(&shape1, &shape2);
    assert_eq!(
        result,
        vec![0],
        "Broadcasting (0,) and (1,) should result in (0,)"
    );
}

#[test]
fn test_broadcast_zero_and_zero() {
    let shape1 = vec![0];
    let shape2 = vec![0];
    let result = compute_broadcast_shape(&shape1, &shape2);
    assert_eq!(
        result,
        vec![0],
        "Broadcasting (0,) and (0,) should result in (0,)"
    );
}

#[test]
fn test_broadcast_mixed_dims() {
    // (0, 3) and (1, 3) -> (0, 3)
    let shape1 = vec![0, 3];
    let shape2 = vec![1, 3];
    let result = compute_broadcast_shape(&shape1, &shape2);
    assert_eq!(result, vec![0, 3]);
}

#[test]
fn test_broadcast_mixed_dims_2() {
    // (0, 3) and (1, 1) -> (0, 3)
    let shape1 = vec![0, 3];
    let shape2 = vec![1, 1];
    let result = compute_broadcast_shape(&shape1, &shape2);
    assert_eq!(result, vec![0, 3]);
}
