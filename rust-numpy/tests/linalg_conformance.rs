// use numpy::linalg;

/*
#[test]
fn test_dot_2d() {
    let a = array2![[1.0, 2.0], [3.0, 4.0]];
    let b = array2![[2.0, 0.0], [1.0, 2.0]];

    // [[1*2+2*1, 1*0+2*2], [3*2+4*1, 3*0+4*2]] = [[4, 4], [10, 8]]
    let c = linalg::dot(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_abs_diff_eq!(c.get_linear(0).unwrap(), &4.0);
    assert_abs_diff_eq!(c.get_linear(1).unwrap(), &4.0);
    assert_abs_diff_eq!(c.get_linear(2).unwrap(), &10.0);
    assert_abs_diff_eq!(c.get_linear(3).unwrap(), &8.0);
}
*/

/*
#[test]
fn test_solve() {
    // Ax = b
    // A = [[3, 1], [1, 2]]
    // b = [9, 8]
    // x = [2, 3] -> 3*2+3=9, 2+2*3=8
    let a = array2![[3.0, 1.0], [1.0, 2.0]];
    let b = array![9.0, 8.0];

    let x = linalg::solve(&a, &b).unwrap();

    assert_abs_diff_eq!(x.get_linear(0).unwrap(), &2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(x.get_linear(1).unwrap(), &3.0, epsilon = 1e-10);
}
*/

/*
#[test]
fn test_cholesky() {
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    let a = array2![
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0]
    ];

    let l = linalg::cholesky(&a).unwrap();

    // Check L is lower triangular
    assert_abs_diff_eq!(l.get(1).unwrap(), &0.0); // row 0, col 1
    assert_abs_diff_eq!(l.get(2).unwrap(), &0.0); // row 0, col 2

    // Check values
    assert_abs_diff_eq!(l.get(0).unwrap(), &2.0);
    assert_abs_diff_eq!(l.get(3).unwrap(), &6.0);
    assert_abs_diff_eq!(l.get(4).unwrap(), &1.0);
}
*/

/*
#[test]
fn test_det() {
    let a = array2![[1.0, 2.0], [3.0, 4.0]];
    // det = 1*4 - 2*3 = -2
    let d = linalg::det(&a).unwrap();
    assert_abs_diff_eq!(d, -2.0, epsilon = 1e-10);
}
*/
