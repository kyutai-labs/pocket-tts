use numpy::*;

#[test]
fn test_svdvals() {
    let a = array2![[1.0f64, 0.0], [0.0, 2.0]];
    let sv = linalg::svdvals(&a).unwrap();
    assert_eq!(sv.size(), 2);
    let mut data = sv.to_vec();
    data.sort_by(|a: &f64, b: &f64| b.partial_cmp(a).unwrap());
    assert!((data[0] - 2.0).abs() < 1e-7);
    assert!((data[1] - 1.0).abs() < 1e-7);
}

#[test]
fn test_matrix_norm() {
    let a = array2![[1.0f64, 1.0], [1.0, 1.0]];
    // Frobenius norm of [[1, 1], [1, 1]] is sqrt(1^2 + 1^2 + 1^2 + 1^2) = 2
    let n = linalg::matrix_norm(&a, Some("fro"), None, false).unwrap();
    let val: f64 = *n.get_linear(0).unwrap();
    assert!((val - 2.0).abs() < 1e-7);
}

#[test]
fn test_vecdot() {
    let a = array![1.0f64, 2.0, 3.0];
    let b = array![4.0f64, 5.0, 6.0];
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let res = linalg::vecdot(&a, &b).unwrap();
    let val: f64 = *res.get_linear(0).unwrap();
    assert!((val - 32.0).abs() < 1e-7);
}

#[test]
fn test_matrix_transpose() {
    let a = array2![[1.0f64, 2.0], [3.0, 4.0]];
    let t = linalg::matrix_transpose(&a).unwrap();
    assert_eq!(t.get_linear(1).cloned().unwrap(), 3.0);
    assert_eq!(t.get_linear(2).cloned().unwrap(), 2.0);
}

#[test]
fn test_diagonal_wrapper() {
    let a = array2![[1.0f64, 2.0], [3.0, 4.0]];
    let d = linalg::diagonal(&a, 0, None, None).unwrap();
    assert_eq!(d.size(), 2);
    assert_eq!(d.get_linear(0).cloned().unwrap(), 1.0);
    assert_eq!(d.get_linear(1).cloned().unwrap(), 4.0);
}

#[test]
fn test_tensordot_default() {
    let a = array2![[1.0f64, 2.0], [3.0, 4.0]];
    let b = array2![[1.0f64, 0.0], [0.0, 1.0]];
    let res = linalg::tensordot(&a, &b, None).unwrap();
    assert_eq!(res.get_linear(0).cloned().unwrap(), 1.0);
    assert_eq!(res.get_linear(1).cloned().unwrap(), 2.0);
}
