use numpy::{array, array2, cdist, pdist, squareform};

#[test]
fn test_pdist_euclidean() {
    // 3 points in 2D space: (0,0), (0,3), (4,0)
    // Distances:
    // (0,0)-(0,3) = 3.0
    // (0,0)-(4,0) = 4.0
    // (0,3)-(4,0) = sqrt(3^2 + 4^2) = 5.0
    let x = array2![[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]];
    let y = pdist(&x, "euclidean").unwrap();

    assert_eq!(y.size(), 3);
    assert!((y.get(0).unwrap() - 3.0).abs() < 1e-9);
    assert!((y.get(1).unwrap() - 4.0).abs() < 1e-9);
    assert!((y.get(2).unwrap() - 5.0).abs() < 1e-9);
}

#[test]
fn test_cdist_euclidean() {
    // XA: (0,0)
    // XB: (0,3), (4,0)
    let xa = array2![[0.0, 0.0]];
    let xb = array2![[0.0, 3.0], [4.0, 0.0]];

    let y = cdist(&xa, &xb, "euclidean").unwrap();

    assert_eq!(y.shape(), &[1, 2]);
    assert!((y.get_multi(&[0, 0]).unwrap() - 3.0).abs() < 1e-9);
    assert!((y.get_multi(&[0, 1]).unwrap() - 4.0).abs() < 1e-9);
}

#[test]
fn test_squareform() {
    // Vector: [3.0, 4.0, 5.0]
    // Matrix:
    // 0 3 4
    // 3 0 5
    // 4 5 0
    let v = array![3.0, 4.0, 5.0];
    let m = squareform(&v).unwrap();

    assert_eq!(m.shape(), &[3, 3]);
    assert_eq!(m.get_multi(&[0, 1]).unwrap(), 3.0);
    assert_eq!(m.get_multi(&[1, 0]).unwrap(), 3.0);
    assert_eq!(m.get_multi(&[0, 2]).unwrap(), 4.0);
    assert_eq!(m.get_multi(&[2, 0]).unwrap(), 4.0);
    assert_eq!(m.get_multi(&[1, 2]).unwrap(), 5.0);
    assert_eq!(m.get_multi(&[2, 1]).unwrap(), 5.0);
    assert_eq!(m.get_multi(&[0, 0]).unwrap(), 0.0);

    // Inverse
    let v2 = squareform(&m).unwrap();
    assert_eq!(v2.size(), 3);
    assert_eq!(v2.get(0).unwrap(), &3.0);
    assert_eq!(v2.get(1).unwrap(), &4.0);
    assert_eq!(v2.get(2).unwrap(), &5.0);
}

#[test]
fn test_metrics() {
    let u = array2![[0.0, 0.0], [1.0, 1.0]];

    // Cityblock: |0-1| + |0-1| = 2
    let y_city = pdist(&u, "cityblock").unwrap();
    assert!((y_city.get(0).unwrap() - 2.0).abs() < 1e-9);

    // Chebyshev: max(|0-1|, |0-1|) = 1
    let y_cheb = pdist(&u, "chebyshev").unwrap();
    assert!((y_cheb.get(0).unwrap() - 1.0).abs() < 1e-9);

    // SqEuclidean: (0-1)^2 + (0-1)^2 = 2
    let y_sq = pdist(&u, "sqeuclidean").unwrap();
    assert!((y_sq.get(0).unwrap() - 2.0).abs() < 1e-9);
}
