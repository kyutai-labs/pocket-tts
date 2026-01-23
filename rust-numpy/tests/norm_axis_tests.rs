use numpy::*;

#[test]
fn test_norm_axis_0_2d() {
    // Test L2 norm along axis 0 of a 2D array
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let arr = Array::from_shape_vec(vec![3, 4], data);

    let result = norm(&arr, Some("2"), Some(&[0isize]), false).unwrap();

    // Expected: sqrt(1^2 + 5^2 + 9^2), sqrt(2^2 + 6^2 + 10^2), etc.
    // sqrt(1 + 25 + 81) = sqrt(107) ≈ 10.344
    // sqrt(4 + 36 + 100) = sqrt(140) ≈ 11.832
    // sqrt(9 + 49 + 121) = sqrt(179) ≈ 13.379
    // sqrt(16 + 64 + 144) = sqrt(224) ≈ 14.967
    assert_eq!(result.shape(), &[4]);
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 10.344).abs() < 0.01);
    assert!((result_vec[1] - 11.832).abs() < 0.01);
}

#[test]
fn test_norm_axis_1_2d() {
    // Test L2 norm along axis 1 of a 2D array
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let arr = Array::from_shape_vec(vec![3, 4], data);

    let result = norm(&arr, Some("2"), Some(&[1isize]), false).unwrap();

    // Expected: sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30) ≈ 5.477
    // sqrt(5^2 + 6^2 + 7^2 + 8^2) = sqrt(174) ≈ 13.191
    // sqrt(9^2 + 10^2 + 11^2 + 12^2) = sqrt(446) ≈ 21.119
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 5.477).abs() < 0.01);
    assert!((result_vec[1] - 13.191).abs() < 0.01);
    assert!((result_vec[2] - 21.119).abs() < 0.01);
}

#[test]
fn test_norm_axis_negative() {
    // Test negative axis indexing
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let arr = Array::from_shape_vec(vec![3, 4], data);

    let result = norm(&arr, Some("2"), Some(&[-1isize]), false).unwrap();

    // axis=-1 should be the same as axis=1
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 5.477).abs() < 0.01);
}

#[test]
fn test_norm_axis_keepdims() {
    // Test keepdims parameter
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let arr = Array::from_shape_vec(vec![3, 4], data);

    let result = norm(&arr, Some("2"), Some(&[0isize]), true).unwrap();

    // With keepdims, shape should be [1, 4]
    assert_eq!(result.shape(), &[1, 4]);
}

#[test]
fn test_norm_axis_multiple() {
    // Test reduction along multiple axes
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];
    let arr = Array::from_shape_vec(vec![2, 3, 4], data);

    let result = norm(&arr, Some("2"), Some(&[0isize, 1isize]), false).unwrap();

    // Should reduce to shape [4]
    assert_eq!(result.shape(), &[4]);

    // Compute expected: sqrt(sum of squares along axes 0 and 1)
    // For position 0: sqrt(1^2 + 5^2 + 9^2 + 13^2 + 17^2 + 21^2)
    // = sqrt(1 + 25 + 81 + 169 + 289 + 441) = sqrt(1006) ≈ 31.717
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 31.717).abs() < 0.1);
}

#[test]
fn test_norm_l1_axis() {
    // Test L1 norm along an axis
    let data: Vec<f64> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let arr = Array::from_shape_vec(vec![2, 3], data);

    let result = norm(&arr, Some("1"), Some(&[0isize]), false).unwrap();

    // L1 norm along axis 0: sum of absolute values
    // |1| + |-4| = 5, |-2| + |5| = 7, |3| + |-6| = 9
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_norm_inf_axis() {
    // Test L-infinity norm along an axis
    let data: Vec<f64> = vec![1.0, -5.0, 3.0, -4.0, 2.0, -6.0];
    let arr = Array::from_shape_vec(vec![2, 3], data);

    let result = norm(&arr, Some("inf"), Some(&[0isize]), false).unwrap();

    // L-inf norm along axis 0: max of absolute values
    // max(|1|, |-4|) = 4, max(|-5|, |2|) = 5, max(|3|, |-6|) = 6
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_norm_neg_inf_axis() {
    // Test L-negative-infinity norm along an axis
    let data: Vec<f64> = vec![1.0, -5.0, 3.0, -4.0, 2.0, -6.0];
    let arr = Array::from_shape_vec(vec![2, 3], data);

    let result = norm(&arr, Some("-inf"), Some(&[0isize]), false).unwrap();

    // L-neg-inf norm along axis 0: min of absolute values
    // min(|1|, |-4|) = 1, min(|-5|, |2|) = 2, min(|3|, |-6|) = 3
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_norm_3d_axis_0() {
    // Test norm on 3D array along axis 0
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let arr = Array::from_shape_vec(vec![2, 3, 2], data);

    let result = norm(&arr, Some("2"), Some(&[0isize]), false).unwrap();

    // Shape should be [3, 2]
    assert_eq!(result.shape(), &[3, 2]);

    // Check first element: sqrt(1^2 + 7^2) = sqrt(50) ≈ 7.071
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 7.071).abs() < 0.01);
}

#[test]
fn test_norm_3d_axis_1() {
    // Test norm on 3D array along axis 1
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let arr = Array::from_shape_vec(vec![2, 3, 2], data);

    let result = norm(&arr, Some("2"), Some(&[1isize]), false).unwrap();

    // Shape should be [2, 2]
    assert_eq!(result.shape(), &[2, 2]);

    // Check first element: sqrt(1^2 + 3^2 + 5^2) = sqrt(35) ≈ 5.916
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 5.916).abs() < 0.01);
}

#[test]
fn test_norm_3d_axis_2() {
    // Test norm on 3D array along axis 2
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let arr = Array::from_shape_vec(vec![2, 3, 2], data);

    let result = norm(&arr, Some("2"), Some(&[2isize]), false).unwrap();

    // Shape should be [2, 3]
    assert_eq!(result.shape(), &[2, 3]);

    // Check first element: sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 2.236).abs() < 0.01);
}

#[test]
fn test_norm_fro_with_axis() {
    // Test that Frobenius norm works with axis
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::from_shape_vec(vec![2, 3], data);

    let result = norm(&arr, Some("fro"), Some(&[0isize]), false).unwrap();

    // Frobenius along axis 0 should be same as L2
    assert_eq!(result.shape(), &[3]);
    let result_vec = result.to_vec();
    assert!((result_vec[0] - 4.123).abs() < 0.01); // sqrt(1^2 + 4^2)
}
