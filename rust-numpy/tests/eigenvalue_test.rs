use num_complex::Complex64;
use numpy::{array2, linalg};

#[test]
fn test_eigenvalue_simple() {
    let matrix = array2![[2.0, 1.0], [1.0, 2.0]];
    let result = linalg::eigvals(&matrix).unwrap();

    let mut vals = result.to_vec();
    vals.sort_by(|a: &Complex64, b: &Complex64| a.norm().partial_cmp(&b.norm()).unwrap());

    // Extract real parts for comparison (matrix is symmetric, so eigenvalues should be real)
    let real_vals: Vec<f64> = vals.iter().map(|c| c.re).collect();
    assert!((real_vals[0] - 3.0).abs() < 1e-10);
    assert!((real_vals[1] - 1.0).abs() < 1e-10);

    println!("Complex eigenvalues: {:?}", vals);
    println!("Real parts: {:?}", real_vals);
}
