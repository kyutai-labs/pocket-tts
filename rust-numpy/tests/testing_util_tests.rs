use numpy::*;

#[test]
fn test_assert_array_equal() {
    let a = array![1, 2, 3];
    let b = array![1, 2, 3];
    let c = array![1, 2, 4];
    let d = array![1, 2];

    assert!(assert_array_equal(&a, &b).is_ok());
    assert!(assert_array_equal(&a, &c).is_err());
    assert!(assert_array_equal(&a, &d).is_err());
}

#[test]
fn test_assert_array_almost_equal() {
    let a = array![1.0, 2.0, 3.000001];
    let b = array![1.0, 2.0, 3.000002];

    assert!(assert_array_almost_equal(&a, &b, 5).is_ok());
    assert!(assert_array_almost_equal(&a, &b, 7).is_err());
}

#[test]
fn test_assert_array_shape_equal() {
    let a = array![1, 2, 3];
    let b = array![4, 5, 6];
    let c = array2![[1, 2], [3, 4]];

    assert!(assert_array_shape_equal(&a, &b).is_ok());
    assert!(assert_array_shape_equal(&a, &c).is_err());
}

#[test]
fn test_assert_allclose() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.0000001, 2.0000001, 3.0000001];

    assert!(assert_allclose(&a, &b, 1e-6, 1e-6).is_ok());
    assert!(assert_allclose(&a, &b, 1e-8, 1e-8).is_err());
}

#[test]
fn test_allclose() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.0000001, 2.0000001, 3.0000001];

    assert!(allclose(&a, &b, Some(1e-6), Some(1e-6), None).unwrap());
    assert!(!allclose(&a, &b, Some(1e-8), Some(1e-8), None).unwrap());
}

#[test]
fn test_assert_array_less() {
    let a = array![1, 2, 3];
    let b = array![2, 3, 4];
    let c = array![1, 2, 3];

    assert!(assert_array_less(&a, &b).is_ok());
    assert!(assert_array_less(&a, &c).is_err());
}
