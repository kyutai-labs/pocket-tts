use num_complex::Complex64;
use numpy::linalg::{eig, eigh, eigvals, eigvalsh};
use numpy::{array2, Array};

fn assert_approx_eq_c64(a: Complex64, b: Complex64, eps: f64, msg: &str) {
    if (a - b).norm() >= eps {
        panic!(
            "{}: left: {:?}, right: {:?} (diff: {:?})",
            msg,
            a,
            b,
            (a - b).norm()
        );
    }
}

#[test]
fn test_eig_identity() {
    let a = array2!([1.0, 0.0], [0.0, 1.0]);
    let (w, v) = eig(&a).unwrap();

    assert_eq!(w.data().len(), 2);
    // Eigenvalues of Identity are 1, 1
    assert_approx_eq_c64(w.data()[0], Complex64::new(1.0, 0.0), 1e-10, "Eigenvalue 0");
    assert_approx_eq_c64(w.data()[1], Complex64::new(1.0, 0.0), 1e-10, "Eigenvalue 1");

    // Check A*v = lambda*v
    let n = 2;
    for i in 0..n {
        let lambda = w.data()[i];
        for row in 0..n {
            let mut av_row = Complex64::new(0.0, 0.0);
            for col in 0..n {
                let a_val = Complex64::new(a.data()[row * n + col], 0.0);
                let v_val = v.data()[col * n + i];
                av_row += a_val * v_val;
            }
            let lv_row = lambda * v.data()[row * n + i];
            assert_approx_eq_c64(
                av_row,
                lv_row,
                1e-10,
                &format!("A*v = lambda*v for col {}, row {}", i, row),
            );
        }
    }
}

#[test]
fn test_eig_2x2_real() {
    // A = [[0, 1], [-2, -3]]
    // Eigenvalues: -1, -2
    let a = array2!([0.0, 1.0], [-2.0, -3.0]);
    let (w, v) = eig(&a).unwrap();

    let n = 2;
    for i in 0..n {
        let lambda = w.data()[i];
        for row in 0..n {
            let mut av_row = Complex64::new(0.0, 0.0);
            for col in 0..n {
                let a_val = Complex64::new(a.data()[row * n + col], 0.0);
                let v_val = v.data()[col * n + i];
                av_row += a_val * v_val;
            }
            let lv_row = lambda * v.data()[row * n + i];
            assert_approx_eq_c64(
                av_row,
                lv_row,
                1e-10,
                &format!("A*v = lambda*v for col {}, row {}", i, row),
            );
        }
    }
}

#[test]
fn test_eig_2x2_complex() {
    // A = [[0, i], [-i, 0]]
    // Eigenvalues: 1, -1 (Pauli Y matrix)
    let a = array2!(
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        [Complex64::new(0.0, -1.0), Complex64::new(0.0, 0.0)]
    );
    let (w, v) = eig(&a).unwrap();

    let n = 2;
    for i in 0..n {
        let lambda = w.data()[i];
        for row in 0..n {
            let mut av_row = Complex64::new(0.0, 0.0);
            for col in 0..n {
                let a_val = a.data()[row * n + col];
                let v_val = v.data()[col * n + i];
                av_row += a_val * v_val;
            }
            let lv_row = lambda * v.data()[row * n + i];
            assert_approx_eq_c64(
                av_row,
                lv_row,
                1e-10,
                &format!("A*v = lambda*v for col {}, row {}", i, row),
            );
        }
    }
}
#[test]
fn test_eig_3x3_real() {
    // A = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
    // Eigenvalues: 1, 4, 6
    let a = array2!([1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]);
    let (w, v) = eig(&a).unwrap();

    let n = 3;
    for i in 0..n {
        let lambda = w.data()[i];
        for row in 0..n {
            let mut av_row = Complex64::new(0.0, 0.0);
            for col in 0..n {
                let a_val = Complex64::new(a.data()[row * n + col], 0.0);
                let v_val = v.data()[col * n + i];
                av_row += a_val * v_val;
            }
            let lv_row = lambda * v.data()[row * n + i];
            assert_approx_eq_c64(
                av_row,
                lv_row,
                1e-10,
                &format!("A*v = lambda*v for col {}, row {}", i, row),
            );
        }
    }
}

#[test]
fn test_eig_symmetric() {
    // A = [[2, 1], [1, 2]]
    // Eigenvalues: 3, 1
    let a = array2!([2.0, 1.0], [1.0, 2.0]);
    let (w, v) = eig(&a).unwrap();

    let n = 2;
    for i in 0..n {
        let lambda = w.data()[i];
        for row in 0..n {
            let mut av_row = Complex64::new(0.0, 0.0);
            for col in 0..n {
                let a_val = Complex64::new(a.data()[row * n + col], 0.0);
                let v_val = v.data()[col * n + i];
                av_row += a_val * v_val;
            }
            let lv_row = lambda * v.data()[row * n + i];
            assert_approx_eq_c64(av_row, lv_row, 1e-10, "A*v = lambda*v");
        }
    }
}

#[test]
fn test_eig_1x1() {
    let a = array2!([5.0]);
    let (w, v) = eig(&a).unwrap();
    assert_approx_eq_c64(
        w.data()[0],
        Complex64::new(5.0, 0.0),
        1e-10,
        "1x1 eigenvalue",
    );
    assert_approx_eq_c64(
        v.data()[0],
        Complex64::new(1.0, 0.0),
        1e-10,
        "1x1 eigenvector",
    );
}

#[test]
fn test_eig_defective() {
    // A = [[1, 1], [0, 1]] (Jordan block)
    // Eigenvalue: 1 (repeated)
    let a = array2!([1.0, 1.0], [0.0, 1.0]);
    let (w, v) = eig(&a).unwrap();

    let n = 2;
    for i in 0..n {
        let lambda = w.data()[i];
        for row in 0..n {
            let mut av_row = Complex64::new(0.0, 0.0);
            for col in 0..n {
                let a_val = Complex64::new(a.data()[row * n + col], 0.0);
                let v_val = v.data()[col * n + i];
                av_row += a_val * v_val;
            }
            let lv_row = lambda * v.data()[row * n + i];
            assert_approx_eq_c64(av_row, lv_row, 1e-10, "A*v = lambda*v");
        }
    }
}

#[test]
fn test_eig_stacked() {
    // Two identity matrices stacked: shape (2, 2, 2)
    let data = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
    let a = Array::from_data(data, vec![2, 2, 2]);
    let (w, v) = eig(&a).unwrap();

    assert_eq!(w.shape(), &[2, 2]);
    assert_eq!(v.shape(), &[2, 2, 2]);

    // Batch 0: Identity(1) -> eigenvalues 1, 1
    assert_approx_eq_c64(
        w.data()[0],
        Complex64::new(1.0, 0.0),
        1e-10,
        "batch 0 eigenvalue",
    );
    assert_approx_eq_c64(
        w.data()[1],
        Complex64::new(1.0, 0.0),
        1e-10,
        "batch 0 eigenvalue",
    );

    // Batch 1: Identity(2) -> eigenvalues 2, 2
    assert_approx_eq_c64(
        w.data()[2],
        Complex64::new(2.0, 0.0),
        1e-10,
        "batch 1 eigenvalue",
    );
    assert_approx_eq_c64(
        w.data()[3],
        Complex64::new(2.0, 0.0),
        1e-10,
        "batch 1 eigenvalue",
    );

    // Check eigenvectors for batch 1
    let n = 2;
    for b in 0..2 {
        for i in 0..n {
            let lambda = w.data()[b * n + i];
            for row in 0..n {
                let mut av_row = Complex64::new(0.0, 0.0);
                for col in 0..n {
                    let a_val = Complex64::new(a.data()[b * n * n + row * n + col], 0.0);
                    let v_val = v.data()[b * n * n + col * n + i];
                    av_row += a_val * v_val;
                }
                let lv_row = lambda * v.data()[b * n * n + row * n + i];
                assert_approx_eq_c64(av_row, lv_row, 1e-10, "stacked A*v = lambda*v");
            }
        }
    }
}

#[test]
fn test_eigh_identity() {
    // Identity matrix should have eigenvalue 1
    let a = array2!([1.0, 0.0], [0.0, 1.0]);
    let (w, _v) = eigh(&a, None).unwrap();

    assert_eq!(w.data().len(), 2);
    assert!((w.data()[0] - 1.0).abs() < 1e-10);
    assert!((w.data()[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_eigvals() {
    // Test eigvals function
    let a = array2!([2.0, 1.0], [1.0, 2.0]);
    let w = eigvals(&a).unwrap();

    assert_eq!(w.data().len(), 2);
    // Should have 2 eigenvalues (3 and 1)
    let has_three = w.data().iter().any(|&x| (x.re - 3.0).abs() < 1e-10);
    let has_one = w.data().iter().any(|&x| (x.re - 1.0).abs() < 1e-10);
    assert!(has_three && has_one);
}

#[test]
fn test_eigvalsh() {
    // Test eigvalsh function (eigenvalues of Hermitian matrix, real output)
    let a = array2!([2.0, 1.0], [1.0, 2.0]);
    let w = eigvalsh(&a, None).unwrap();

    assert_eq!(w.data().len(), 2);
    // For symmetric real matrix, eigenvalues should be real and positive
    // Eigenvalues of [[2,1],[1,2]] are 3 and 1
    let sorted = {
        let mut vals = w.data().to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals
    };
    assert!((sorted[0] - 1.0).abs() < 1e-10);
    assert!((sorted[1] - 3.0).abs() < 1e-10);
}
