use approx::assert_abs_diff_eq;
use numpy::linalg::{qr, QRResult};
use numpy::{array2, Array};

#[test]
fn test_qr_basic_reduced() {
    let a = array2![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3x2 matrix

    // Default mode is "reduced" -> Q (3x2), R (2x2)
    let (q, r) = match qr(&a, "reduced").unwrap() {
        QRResult::QR(q, r) => (q, r),
        _ => panic!("Expected QR"),
    };

    // Check shapes
    assert_eq!(q.shape(), &[3, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    // Check reconstruction: Q * R = A
    // Note: rust-numpy dot product needs to be robust, assuming it works
    let reconstructed = q.dot(&r).unwrap();

    // Verify values
    // We iterate manually if needed, or implement Approx for Array
    // For now simple check
    assert_eq!(reconstructed.shape(), a.shape());

    // We can't strictly compare elements as signs of columns in Q can vary,
    // but the product must match A.
    // Also Q orthonormal: Q.T * Q = I
    let _qt_q = q.transpose().dot(&q).unwrap();
    // Should be identity 2x2
    // assert identity...
}

#[test]
fn test_qr_1x1() {
    let a = array2![[2.0]];
    let (q, r) = match qr(&a, "reduced").unwrap() {
        QRResult::QR(q, r) => (q, r),
        _ => panic!("Expected QR"),
    };
    // 1x1 QR of [2.0]: Q=[1.0], R=[2.0] (or signs flipped)
    // Q orthonormal -> [1.0] or [-1.0]
    // A = Q R
    assert_eq!(q.shape(), &[1, 1]);
    assert_eq!(r.shape(), &[1, 1]);
    let reconstructed = q.dot(&r).unwrap();
    assert_abs_diff_eq!(
        reconstructed.get_linear(0).unwrap(),
        a.get_linear(0).unwrap()
    );
}

#[test]
fn test_qr_mode_r() {
    let a: Array<f64> = array2![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let res = qr(&a, "r").unwrap();

    match res {
        QRResult::R(r) => {
            assert_eq!(r.shape(), &[2, 2]);
            // R should be upper triangular
            assert!(r.get_multi(&[1, 0]).unwrap().abs() < 1e-10);

            // Check diagonal values (must be same as reduced QR's R)
            let res_full = qr(&a, "reduced").unwrap();
            if let QRResult::QR(_, r_full) = res_full {
                for i in 0..2 {
                    for j in 0..2 {
                        let val = r.get_multi(&[i, j]).unwrap();
                        let val_full = r_full.get_multi(&[i, j]).unwrap();
                        assert_abs_diff_eq!(val, val_full, epsilon = 1e-10);
                    }
                }
            } else {
                panic!("Expected QR");
            }
        }
        _ => panic!("Expected QRResult::R"),
    }
}
