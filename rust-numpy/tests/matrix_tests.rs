use numpy::array::Array;
use numpy::matrix::Matrix;

#[test]
fn test_matrix_creation() {
    let a = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let m = Matrix::new(a.clone()).unwrap();
    assert_eq!(m.array().shape(), &[2, 2]);
}

#[test]
fn test_matrix_transpose() {
    let m = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let mt = m.T();
    assert_eq!(mt.array().get_linear(1).unwrap(), &3.0);
}

#[test]
fn test_matrix_mul() {
    let m1 = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let m2 = Matrix::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

    // m1 * I = m1
    let res = (m1.clone() * m2.clone()).unwrap();
    assert_eq!(res.array().get_linear(0).unwrap(), &1.0);
    assert_eq!(res.array().get_linear(3).unwrap(), &4.0);
}

#[test]
fn test_matrix_add() {
    let m1 = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let m2 = Matrix::from_data(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();

    let res = (m1 + m2).unwrap();
    assert_eq!(res.array().get_linear(0).unwrap(), &2.0);
    assert_eq!(res.array().get_linear(3).unwrap(), &5.0);
}
