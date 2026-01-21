use num_complex::Complex64;
use numpy::fft::{fft2, fftn, ifft2, ifftn, fftshift, ifftshift, irfftn, rfftn};
use numpy::*;

#[test]
fn test_fftn_2d_basic() {
    let shape = vec![2, 2];
    let data = vec![1.0, 0.0, 0.0, 0.0];
    let arr = Array::from_data(data, shape);

    let res = fftn(&arr, None, None, None).unwrap();
    assert_eq!(res.shape(), &[2, 2]);

    // For [1, 0; 0, 0], FFT should be [1, 1; 1, 1]
    for val in res.iter() {
        assert!((val.re - 1.0).abs() < 1e-10);
        assert!(val.im.abs() < 1e-10);
    }
}

#[test]
fn test_ifftn_2d_basic() {
    let shape = vec![2, 2];
    let data = vec![Complex64::new(1.0, 0.0); 4];
    let arr = Array::from_data(data, shape);

    let res = ifftn(&arr, None, None, None).unwrap();
    assert_eq!(res.shape(), &[2, 2]);

    // IFFT of [1, 1; 1, 1] is [1, 0; 0, 0]
    let expected = vec![1.0, 0.0, 0.0, 0.0];
    for (i, val) in res.iter().enumerate() {
        assert!((val.re - expected[i]).abs() < 1e-10);
    }
}

#[test]
fn test_fftshift_2d() {
    let shape = vec![2, 2];
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let arr = Array::from_data(data, shape);

    // Shift all axes
    let shifted = fftshift(&arr, None);
    assert_eq!(shifted.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);

    let unshifted = ifftshift(&shifted, None);
    assert_eq!(unshifted.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_rfftn_2d() {
    let shape = vec![2, 2];
    let data = vec![1.0, 0.0, 0.0, 0.0];
    let arr = Array::from_data(data, shape);

    let res = rfftn(&arr, None, None, None).unwrap();
    // Shape should be [2, 2/2 + 1] = [2, 2]
    assert_eq!(res.shape(), &[2, 2]);

    let back = irfftn(&res, None, None, None).unwrap();
    assert_eq!(back.shape(), &[2, 2]);
    for (i, &v) in back.iter().enumerate() {
        assert!((v - arr.to_vec()[i]).abs() < 1e-10);
    }
}

#[test]
fn test_fft2_basic() {
    let shape = vec![2, 2];
    let data = vec![1.0, 0.0, 0.0, 0.0];
    let arr = Array::from_data(data, shape);

    let res = fft2(&arr, None, None, None).unwrap();
    assert_eq!(res.shape(), &[2, 2]);

    // For [1, 0; 0, 0], FFT should be [1, 1; 1, 1]
    for val in res.iter() {
        assert!((val.re - 1.0).abs() < 1e-10);
        assert!(val.im.abs() < 1e-10);
    }
}

#[test]
fn test_fft2_with_axes() {
    let shape = vec![3, 2, 2];
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let arr = Array::from_data(data, shape);

    // Test with explicit axes
    let res = fft2(&arr, None, Some(&[1, 2]), None).unwrap();
    assert_eq!(res.shape(), &[3, 2, 2]);
}

#[test]
fn test_ifft2_basic() {
    let shape = vec![2, 2];
    let data = vec![Complex64::new(1.0, 0.0); 4];
    let arr = Array::from_data(data, shape);

    let res = ifft2(&arr, None, None, None).unwrap();
    assert_eq!(res.shape(), &[2, 2]);

    // IFFT2 of [1, 1; 1, 1] is [1, 0; 0, 0]
    let expected = vec![1.0, 0.0, 0.0, 0.0];
    for (i, val) in res.iter().enumerate() {
        assert!((val.re - expected[i]).abs() < 1e-10);
        assert!(val.im.abs() < 1e-10);
    }
}

#[test]
fn test_fft2_ifft2_roundtrip() {
    let shape = vec![4, 4];
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let arr = Array::from_data(data, shape);

    let fwd = fft2(&arr, None, None, None).unwrap();
    let back = ifft2(&fwd, None, None, None).unwrap();

    assert_eq!(back.shape(), &[4, 4]);
    for (i, val) in back.iter().enumerate() {
        assert!((val.re - data[i]).abs() < 1e-8);
        assert!(val.im.abs() < 1e-8);
    }
}
