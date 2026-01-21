use num_complex::Complex64;
use numpy::{
    array,
    fft::{fft, fftfreq, fftshift, hfft, ifft, ifftshift, ihfft, irfft, rfft, rfftfreq},
};

#[test]
fn test_fft_basic() {
    let a = array![1.0, 0.0, -1.0, 0.0];
    let res = fft(&a, None, 0, None).unwrap();
    // Expected: [0, 2, 0, 2]? No.
    // DFT of [1, 0, -1, 0] is:
    // X[0] = 1 + 0 - 1 + 0 = 0
    // X[1] = 1*e^0 + 0 - 1*e^(-j*pi) + 0 = 1 - (-1) = 2
    // X[2] = 1 + 0 - 1*e^(-j*2pi) + 0 = 1 - 1 = 0
    // X[3] = 1 + 0 - 1*e^(-j*3pi) + 0 = 1 - (-1) = 2
    let vals = res.to_vec();
    assert!((vals[0].re).abs() < 1e-10);
    assert!((vals[1].re - 2.0).abs() < 1e-10);
    assert!((vals[2].re).abs() < 1e-10);
    assert!((vals[3].re - 2.0).abs() < 1e-10);
}

#[test]
fn test_ifft_basic() {
    let a = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let arr = numpy::Array::from_vec(a);
    let res = ifft(&arr, None, 0, None).unwrap();
    let vals = res.to_vec();
    // Should be [1, 0, -1, 0]
    assert!((vals[0].re - 1.0).abs() < 1e-10);
    assert!((vals[1].re).abs() < 1e-10);
    assert!((vals[2].re + 1.0).abs() < 1e-10);
    assert!((vals[3].re).abs() < 1e-10);
}

#[test]
fn test_rfft_basic() {
    let a = array![1.0, 0.0, -1.0, 0.0];
    let res = rfft(&a, None, 0, None).unwrap();
    let vals = res.to_vec();
    // n=4, returns n/2 + 1 = 3 elements
    assert_eq!(vals.len(), 3);
    assert!((vals[0].re).abs() < 1e-10);
    assert!((vals[1].re - 2.0).abs() < 1e-10);
    assert!((vals[2].re).abs() < 1e-10);
}

#[test]
fn test_irfft_basic() {
    let a = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let arr = numpy::Array::from_vec(a);
    let res = irfft(&arr, Some(4), 0, None).unwrap();
    let vals = res.to_vec();
    assert_eq!(vals.len(), 4);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1]).abs() < 1e-10);
    assert!((vals[2] + 1.0).abs() < 1e-10);
    assert!((vals[3]).abs() < 1e-10);
}

#[test]
fn test_fftshift() {
    let a = array![0, 1, 2, 3, 4];
    let res = fftshift(&a, None);
    // [3, 4, 0, 1, 2]
    assert_eq!(res.to_vec(), vec![3, 4, 0, 1, 2]);
}

#[test]
fn test_ifftshift() {
    let a = array![3, 4, 0, 1, 2];
    let res = ifftshift(&a, None);
    // [0, 1, 2, 3, 4]
    assert_eq!(res.to_vec(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_fftfreq() {
    let freqs = fftfreq(8, 0.1);
    assert_eq!(freqs.len(), 8);
    assert_eq!(freqs[0], 0.0);
    assert_eq!(freqs[1], 1.25);
}

#[test]
fn test_rfftfreq() {
    let freqs = rfftfreq(8, 0.1);
    assert_eq!(freqs.len(), 5);
    assert_eq!(freqs[0], 0.0);
    assert_eq!(freqs[1], 1.25);
    assert_eq!(freqs[4], 5.0);

    let freqs_odd = rfftfreq(7, 0.1);
    assert_eq!(freqs_odd.len(), 4);
    assert_eq!(freqs_odd[0], 0.0);
    assert!((freqs_odd[3] - 3.0 / (7.0 * 0.1)).abs() < 1e-10);
}

#[test]
fn test_hfft_basic() {
    // hfft takes Hermitian-symmetric input and produces real output
    // For input length m, output length is n = 2*(m-1)
    let a = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let arr = numpy::Array::from_vec(a);
    let res = hfft(&arr, None, 0, None).unwrap();
    let vals = res.to_vec();
    // n = 2*(3-1) = 4
    assert_eq!(vals.len(), 4);
    // Should match rfft round-trip: hfft(rfft(x)) should give scaled x
    // For [1, 0, -1, 0], rfft gives [0, 2, 0]
    // hfft should give back [4, 0, -4, 0] (scaled by n)
    assert!((vals[0] - 4.0).abs() < 1e-10);
    assert!((vals[1]).abs() < 1e-10);
    assert!((vals[2] + 4.0).abs() < 1e-10);
    assert!((vals[3]).abs() < 1e-10);
}

#[test]
fn test_ihfft_basic() {
    // ihfft takes real input and produces Hermitian-symmetric output
    let a = array![1.0, 0.0, -1.0, 0.0];
    let res = ihfft(&a, None, 0, None).unwrap();
    let vals = res.to_vec();
    // Should give m/2 + 1 = 3 elements
    assert_eq!(vals.len(), 3);
    // Should match rfft output (which is also Hermitian-symmetric)
    // rfft([1, 0, -1, 0]) = [0, 2, 0]
    assert!((vals[0].re).abs() < 1e-10);
    assert!((vals[1].re - 2.0).abs() < 1e-10);
    assert!((vals[2].re).abs() < 1e-10);
}

#[test]
fn test_hfft_ihfft_roundtrip() {
    // Test that hfft and ihfft are inverses
    let a = array![1.0, 0.0, -1.0, 0.0];
    let hermitian = ihfft(&a, None, 0, None).unwrap();
    let recovered = hfft(&hermitian, None, 0, None).unwrap();
    let vals = recovered.to_vec();

    // After round-trip, values should be scaled by n
    assert!((vals[0] - 4.0).abs() < 1e-10);
    assert!((vals[1]).abs() < 1e-10);
    assert!((vals[2] + 4.0).abs() < 1e-10);
    assert!((vals[3]).abs() < 1e-10);
}
