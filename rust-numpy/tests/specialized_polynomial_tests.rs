// use ndarray::Array1;
// TODO: Implement polynomial module
// use numpy::polynomial::{
//     Chebyshev, Hermite, HermiteE, Laguerre, Legendre, Polynomial, PolynomialBase,
// };

// Specialized polynomial tests disabled until module is implemented
/*
#[test]
fn test_chebyshev_eval() {
    // T0(x) = 1, T1(x) = x, T2(x) = 2x^2 - 1
    let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]); // 1*T0 + 2*T1 + 3*T2
    let c = Chebyshev::new(&coeffs).unwrap();

    let x = Array1::from_vec(vec![0.0, 1.0]);
    let y = c.eval(&x).unwrap();

    // c(0) = 1*T0(0) + 2*T1(0) + 3*T2(0) = 1*1 + 2*0 + 3*(-1) = 1 - 3 = -2
    // c(1) = 1*1 + 2*1 + 3*(2-1) = 1 + 2 + 3 = 6
    assert!((y[0] - (-2.0f64)).abs() < 1e-10);
    assert!((y[1] - 6.0f64).abs() < 1e-10);
}

#[test]
fn test_legendre_eval() {
    // P0(x) = 1, P1(x) = x, P2(x) = 0.5*(3x^2 - 1)
    let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]); // 1*P0 + 2*P1 + 3*P2
    let l = Legendre::new(&coeffs).unwrap();

    let x = Array1::from_vec(vec![0.0, 1.0]);
    let y = l.eval(&x).unwrap();

    // l(0) = 1*P0(0) + 2*P1(0) + 3*P2(0) = 1*1 + 2*0 + 3*(-0.5) = 1 - 1.5 = -0.5
    // l(1) = 1*1 + 2*1 + 3*1 = 6
    assert!((y[0] - (-0.5f64)).abs() < 1e-10);
    assert!((y[1] - 6.0f64).abs() < 1e-10);
}

#[test]
fn test_laguerre_eval() {
    // L0(x) = 1, L1(x) = 1 - x, L2(x) = 0.5*(x^2 - 4x + 2)
    let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]); // 1*L0 + 2*L1 + 3*L2
    let lag = Laguerre::new(&coeffs).unwrap();

    let x = Array1::from_vec(vec![0.0, 1.0]);
    let y = lag.eval(&x).unwrap();

    // lag(0) = 1*L0(0) + 2*L1(0) + 3*L2(0) = 1*1 + 2*1 + 3*1 = 6
    // lag(1) = 1*1 + 2*0 + 3*0.5*(1-4+2) = 1 + 3*(-0.5) = -0.5
    assert!((y[0] - 6.0f64).abs() < 1e-10);
    assert!((y[1] - (-0.5f64)).abs() < 1e-10);
}

#[test]
fn test_hermite_eval() {
    // H0(x) = 1, H1(x) = 2x, H2(x) = 4x^2 - 2
    let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]); // 1*H0 + 2*H1 + 3*H2
    let h = Hermite::new(&coeffs).unwrap();

    let x = Array1::from_vec(vec![0.0, 1.0]);
    let y = h.eval(&x).unwrap();

    // h(0) = 1*1 + 2*0 + 3*(-2) = -5
    // h(1) = 1*1 + 2*2 + 3*(4-2) = 1 + 4 + 6 = 11
    assert!((y[0] - (-5.0f64)).abs() < 1e-10);
    assert!((y[1] - 11.0f64).abs() < 1e-10);
}

#[test]
fn test_hermite_e_eval() {
    // He0(x) = 1, He1(x) = x, He2(x) = x^2 - 1
    let coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0]); // 1*He0 + 2*He1 + 3*He2
    let he = HermiteE::new(&coeffs).unwrap();

    let x = Array1::from_vec(vec![0.0, 1.0]);
    let y = he.eval(&x).unwrap();

    // he(0) = 1*1 + 2*0 + 3*(-1) = -2
    // he(1) = 1*1 + 2*1 + 3*(1-1) = 3
    assert!((y[0] - (-2.0f64)).abs() < 1e-10);
    assert!((y[1] - 3.0f64).abs() < 1e-10);
}

#[test]
fn test_specialized_to_polynomial() {
    let coeffs = Array1::from_vec(vec![1.0, 2.0]); // 1 + 2x for Chebyshev T0, T1
    let c = Chebyshev::new(&coeffs).unwrap();
    let p = c.to_polynomial().unwrap();

    // T0 = 1, T1 = x. 1*T0 + 2*T1 = 1 + 2x. Coeffs should be [1, 2]
    assert_eq!(p.coeffs().to_vec(), vec![1.0, 2.0]);

    let coeffs2 = Array1::from_vec(vec![1.0, 0.0, 1.0]); // T0 + T2 = 1 + 2x^2 - 1 = 2x^2
    let c2 = Chebyshev::new(&coeffs2).unwrap();
    let p2 = c2.to_polynomial().unwrap();
    assert_eq!(p2.coeffs().to_vec(), vec![0.0, 0.0, 2.0]);
}
*/
