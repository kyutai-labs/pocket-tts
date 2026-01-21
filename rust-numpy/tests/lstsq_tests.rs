use approx::assert_abs_diff_eq;
use numpy::linalg::solvers::lstsq;
use numpy::{array2, Array};

#[test]
fn test_lstsq_simple_exact() {
    // Ax = b
    // [[1, 1], [0, 2]] * [1, 1] = [2, 2]
    let a = array2![[1.0, 1.0], [0.0, 2.0]];
    let b = array2![[2.0], [2.0]];

    let (x, residuals, rank, _s) = lstsq(&a, &b, None).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    assert_abs_diff_eq!(x.get_linear(0).unwrap(), &1.0);
    assert_abs_diff_eq!(x.get_linear(1).unwrap(), &1.0);

    // M=N=2. Residuals should be empty [0].
    assert_eq!(residuals.shape(), &[0]);
    assert_eq!(rank, 2);
}

#[test]
fn test_lstsq_overdetermined_approx() {
    // Ax = b
    // [[1], [1], [1], [1]] * [x] = [1, 2, 3, 4]
    // x should be 2.5 (mean)
    let a = array2![[1.0], [1.0], [1.0], [1.0]];
    let b = array2![[1.0], [2.0], [3.0], [4.0]];

    let (x, residuals, rank, _s) = lstsq(&a, &b, None).unwrap();

    assert_eq!(x.shape(), &[1, 1]);
    assert_abs_diff_eq!(x.get_linear(0).unwrap(), &2.5);

    // Residuals: ||b - Ax||^2.
    // [1-2.5, 2-2.5, 3-2.5, 4-2.5] = [-1.5, -0.5, 0.5, 1.5]
    // Squares: [2.25, 0.25, 0.25, 2.25] = 5.0
    // residuals is (1,)
    assert_abs_diff_eq!(residuals.get_linear(0).unwrap(), &5.0);
    assert_eq!(rank, 1);
}

#[test]
fn test_lstsq_1d_b() {
    // Ax = b where b is 1D
    // [[1], [1], [1], [1]] * [x] = [1, 2, 3, 4]
    let a = array2![[1.0], [1.0], [1.0], [1.0]];
    let _b_2d = array2![[1.0, 2.0, 3.0, 4.0]];

    // 1D array construction
    let b_1d = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    let (x, residuals, _rank, _s) = lstsq(&a, &b_1d, None).unwrap();

    // x should be (4,). Shape check.
    // Wait, x is solution. Matrix A is 4x1. M=4, N=1.
    // x should be (N,). So (1,).
    // In sol: x is (1,).
    assert_eq!(x.shape(), &[1]);
    assert_abs_diff_eq!(x.get_linear(0).unwrap(), &2.5);
    assert_abs_diff_eq!(residuals.get_linear(0).unwrap(), &5.0);
}
