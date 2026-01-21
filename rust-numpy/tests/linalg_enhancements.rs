use numpy::linalg::{det, inv, pinv, solve, trace};
use numpy::Array;

#[test]
fn test_missing_linalg_basic() {
    // 2x2 Identity matrix
    // 2x2 Identity matrix
    // Flat data: [1.0, 0.0, 0.0, 1.0], Shape: [2, 2]
    let a = Array::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

    // Trace
    assert_eq!(trace(&a).unwrap(), 2.0);

    // Determinant
    assert_eq!(det(&a).unwrap(), 1.0);

    // Inverse
    let a_inv = inv(&a).unwrap();
    // Index 0 is [0,0] -> 1.0
    assert_eq!(*a_inv.get(0).unwrap(), 1.0);

    // Solve Ax = b
    // b = [1.0, 2.0]
    let b = Array::from_data(vec![1.0, 2.0], vec![2]);
    let x = solve(&a, &b).unwrap();
    // x should be [1.0, 2.0]
    assert_eq!(*x.get(0).unwrap(), 1.0);
    assert_eq!(*x.get(1).unwrap(), 2.0);

    // Pseudo-Inverse (for square invertible matrix, same as inv)
    let a_pinv = pinv(&a, None).unwrap();
    assert_eq!(*a_pinv.get(0).unwrap(), 1.0);
}
