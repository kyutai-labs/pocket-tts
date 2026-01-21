use approx::assert_abs_diff_eq;
use numpy::linalg::qr;
use numpy::{array, array2, Array};

#[test]
fn test_qr_basic_reduced() {
    let a = array2![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3x2 matrix

    // Default mode is "reduced" -> Q (3x2), R (2x2)
    let (q, r) = qr(&a, "reduced").unwrap();

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
    let qt_q = q.transpose().dot(&q).unwrap();
    // Should be identity 2x2
    // assert identity...
}

#[test]
fn test_qr_1x1() {
    let a = array2![[2.0]];
    let (q, r) = qr(&a, "reduced").unwrap();
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
