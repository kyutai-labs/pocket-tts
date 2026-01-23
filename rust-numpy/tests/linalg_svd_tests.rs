use numpy::array::Array;
use numpy::linalg::svd;

#[test]
fn test_svd_2x2() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let a = Array::from_data(data, vec![2, 2]);
    let result = svd(&a, false).expect("SVD should succeed");
    let (u, s, v_h) = result;

    assert_eq!(s.shape(), &[2]);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(v_h.shape(), &[2, 2]);

    // Singular values should be in descending order
    assert!(s.get(0).unwrap() >= s.get(1).unwrap());

    // Check factorization A = U * S * V^H
    // S as 2x2: [s0, 0; 0, s1]
    let s0 = *s.get(0).unwrap();
    let s1 = *s.get(1).unwrap();

    // Recalculate A from U, S, V^H
    let mut a_rec = vec![0.0; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = 0.0;
            // (U * S * V^H)[i, j] = sum_k U[i, k] * S[k] * V^H[k, j]
            for k in 0..2 {
                let sk = if k == 0 { s0 } else { s1 };
                sum += u.get(i * 2 + k).unwrap() * sk * v_h.get(k * 2 + j).unwrap();
            }
            a_rec[i * 2 + j] = sum;
        }
    }

    // A = [1.0, 2.0; 3.0, 4.0]
    let expected = vec![1.0, 2.0, 3.0, 4.0];
    for (r, e) in a_rec.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-10,
            "Reconstructed A should match original. Got {}, expected {}",
            r,
            e
        );
    }
}

#[test]
fn test_svd_3x2() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_data(data, vec![3, 2]);
    let result = svd(&a, false).expect("SVD should succeed");
    let (u, s, v_h) = result;

    assert_eq!(s.shape(), &[2]);
    assert_eq!(u.shape(), &[3, 2]);
    assert_eq!(v_h.shape(), &[2, 2]);

    // Check factorization A = U * S * V^H
    let s0 = *s.get(0).unwrap();
    let s1 = *s.get(1).unwrap();

    let mut a_rec = vec![0.0; 6];
    for i in 0..3 {
        for j in 0..2 {
            let mut sum = 0.0;
            for k in 0..2 {
                let sk = if k == 0 { s0 } else { s1 };
                sum += u.get(i * 2 + k).unwrap() * sk * v_h.get(k * 2 + j).unwrap();
            }
            a_rec[i * 2 + j] = sum;
        }
    }

    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (r, e) in a_rec.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-10,
            "Reconstructed A should match original. Got {}, expected {}",
            r,
            e
        );
    }
}
