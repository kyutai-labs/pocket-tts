use ndarray::Array1;
use num_traits::Float;
// TODO: Implement polynomial module
// use numpy::polynomial::{fit, roots, Polynomial, PolynomialBase};

// Polynomial tests disabled until module is implemented
/*
#[test]
fn test_polynomial_eval() {
    // p(x) = 1 + 2x + 3x^2
    let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let p = Polynomial::new(&coeffs).unwrap();

    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let y = p.eval(&x).unwrap();

    // p(0) = 1
    // p(1) = 1 + 2 + 3 = 6
    // p(2) = 1 + 4 + 12 = 17
    assert_eq!(y.to_vec(), vec![1.0, 6.0, 17.0]);
}

#[test]
fn test_polynomial_arithmetic() {
    let p1 = Polynomial::new(&Array1::from_vec(vec![1.0, 2.0])).unwrap(); // 1 + 2x
    let p2 = Polynomial::new(&Array1::from_vec(vec![3.0, 4.0])).unwrap(); // 3 + 4x

    let p_add = p1.add(&p2);
    assert_eq!(p_add.coeffs().to_vec(), vec![4.0, 6.0]);

    let p_sub = p1.sub(&p2);
    assert_eq!(p_sub.coeffs().to_vec(), vec![-2.0, -2.0]);

    let p_mul = p1.mul(&p2);
    // (1 + 2x)(3 + 4x) = 3 + 4x + 6x + 8x^2 = 3 + 10x + 8x^2
    assert_eq!(p_mul.coeffs().to_vec(), vec![3.0, 10.0, 8.0]);
}

#[test]
fn test_polynomial_deriv_integ() {
    let p = Polynomial::new(&Array1::from_vec(vec![1.0, 2.0, 3.0])).unwrap(); // 1 + 2x + 3x^2

    let dp = p.deriv(1).unwrap();
    // 2 + 6x
    assert_eq!(dp.coeffs().to_vec(), vec![2.0, 6.0]);

    let ip = p.integ(1, Some(0.0)).unwrap();
    // 0 + 1x + 1x^2 + 1x^3
    assert_eq!(ip.coeffs().to_vec(), vec![0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_polynomial_fit() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let y = Array1::from_vec(vec![1.0, 3.0, 5.0]); // y = 1 + 2x

    let p = fit(&x, &y, 1).unwrap();
    let coeffs = p.coeffs().to_vec();
    assert!((coeffs[0] - 1.0).abs() < 1e-10);
    assert!((coeffs[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_polynomial_roots() {
    // p(x) = x^2 - 1 = (x-1)(x+1)
    let p = Polynomial::new(&Array1::from_vec(vec![-1.0, 0.0, 1.0])).unwrap();
    let r = roots(&p).unwrap();

    let rv = r.to_vec();
    // Roots are 1 and -1
    // find_eigenvalues for n=2 uses quadratic formula
    assert!((rv[0].re - 1.0).abs() < 1e-10 || (rv[1].re - 1.0).abs() < 1e-10);
    assert!((rv[0].re + 1.0).abs() < 1e-10 || (rv[1].re + 1.0).abs() < 1e-10);
}
*/
