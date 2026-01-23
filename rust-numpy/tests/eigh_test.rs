use numpy::{array2, linalg};

#[test]
fn test_eigh_simple() {
    let matrix = array2![[2.0, 1.0], [1.0, 2.0]];

    // Test eigh - should return real eigenvalues for symmetric matrix
    if let Ok(eigvals) = linalg::eigh(&matrix, Some("L")) {
        let mut vals = eigvals.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("eigh eigenvalues: {:?}", vals);

        // Should be approximately [1.0, 3.0] for this symmetric matrix
        assert!(vals.len() == 2);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
    } else {
        panic!("eigh failed");
    }
}

#[test]
fn test_eigvalsh_simple() {
    let matrix = array2![[2.0, 1.0], [1.0, 2.0]];

    // Test eigvalsh - should return real eigenvalues for symmetric matrix
    if let Ok(eigvals) = linalg::eigvalsh(&matrix, Some("L")) {
        let mut vals = eigvals.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("eigvalsh eigenvalues: {:?}", vals);

        // Should be approximately [1.0, 3.0] for this symmetric matrix
        assert!(vals.len() == 2);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
    } else {
        panic!("eigvalsh failed");
    }
}
