use num_complex::Complex;
use numpy::array::Array;
use numpy::linalg::*;

#[test]
fn test_dot_scalar() {
    let a = Array::from_vec(vec![2.0]).reshape(&[]).unwrap();
    let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let res = dot(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![2.0, 4.0, 6.0]);
    assert_eq!(res.shape(), &[3]);
}

#[test]
fn test_dot_1d() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    let res = dot(&a, &b).unwrap();
    assert_eq!(res.get_linear(0).unwrap(), &32.0); // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
    assert_eq!(res.ndim(), 0);
}

#[test]
fn test_dot_2d() {
    let a = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Array::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let res = dot(&a, &b).unwrap();
    // [1 2] [5 6] = [1*5+2*7 1*6+2*8] = [19 22]
    // [3 4] [7 8] = [3*5+4*7 3*6+4*8] = [43 50]
    assert_eq!(res.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    assert_eq!(res.shape(), &[2, 2]);
}

#[test]
fn test_vdot() {
    let a = Array::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
    let b = Array::from_vec(vec![Complex::new(1.0, -1.0), Complex::new(2.0, -2.0)]);
    let res = vdot(&a, &b).unwrap();
    // (1-1i)*(1-1i) + (2-2i)*(2-2i)
    // = (1 - 1i - 1i - 1) + (4 - 4i - 4i - 4)
    // = -2i - 8i = -10i
    assert_eq!(res, Complex::new(0.0, -10.0));
}

#[test]
fn test_inner() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array::from_vec(vec![0.0, 1.0, 0.0]);
    let res = inner(&a, &b).unwrap();
    assert_eq!(res.get_linear(0).unwrap(), &2.0);
}

#[test]
fn test_outer() {
    let a = Array::from_vec(vec![1.0, 2.0]);
    let b = Array::from_vec(vec![3.0, 4.0, 5.0]);
    let res = outer(&a, &b).unwrap();
    assert_eq!(res.shape(), &[2, 3]);
    assert_eq!(res.to_vec(), vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
}

#[test]
fn test_kron() {
    let a = Array::from_vec(vec![1.0, 2.0]);
    let b = Array::from_vec(vec![3.0, 4.0]);
    let res = kron(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![3.0, 4.0, 6.0, 8.0]);
    assert_eq!(res.shape(), &[4]);
}

#[test]
fn test_matrix_power() {
    let a = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let res = matrix_power(&a, 2).unwrap();
    // [1 2] [1 2] = [1*1+2*3 1*2+2*4] = [7 10]
    // [3 4] [3 4] = [3*1+4*3 3*2+4*4] = [15 22]
    assert_eq!(res.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

    let res0 = matrix_power(&a, 0).unwrap();
    assert_eq!(res0.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_cross() {
    let a = Array::from_vec(vec![1.0, 0.0, 0.0]);
    let b = Array::from_vec(vec![0.0, 1.0, 0.0]);
    let res = cross(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_multi_dot() {
    let a = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Array::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let c = Array::from_vec(vec![1.0, 2.0]); // 1D

    let res = multi_dot(&[a, b, c]).unwrap();
    // (A * B) * c = A * c since B covers identity
    // [1 2] [1] = [1*1 + 2*2] = [5]
    // [3 4] [2] = [3*1 + 4*2] = [11]
    assert_eq!(res.to_vec(), vec![5.0, 11.0]);
    assert_eq!(res.shape(), &[2]);
}
