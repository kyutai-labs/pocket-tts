use half::f16;
use numpy::{array, Array, Dtype};
use std::mem;

#[test]
fn test_intp_uintp() {
    let arr_isize = Array::from_vec(vec![1isize, 2, 3]);
    let arr_usize = Array::from_vec(vec![1usize, 2, 3]);

    assert!(matches!(arr_isize.dtype(), Dtype::Intp { .. }));
    assert!(matches!(arr_usize.dtype(), Dtype::Uintp { .. }));

    // Verify itemsize matches platform
    assert_eq!(arr_isize.dtype().itemsize(), mem::size_of::<isize>());
    assert_eq!(arr_usize.dtype().itemsize(), mem::size_of::<usize>());
}

#[test]
fn test_float16() {
    let v = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.5)];
    let arr = Array::from_vec(v);

    assert!(matches!(arr.dtype(), Dtype::Float16 { .. }));
    assert_eq!(arr.dtype().itemsize(), 2);
    assert_eq!(arr.dtype().alignment(), 2);

    // Check values
    assert_eq!(arr.get(0).unwrap(), &f16::from_f32(1.0));
    assert_eq!(arr.get(2).unwrap(), &f16::from_f32(3.5));
}

#[test]
fn test_float16_ops() {
    // This tests that we can perform operations if num-traits is working
    let a = f16::from_f32(1.5);
    let b = f16::from_f32(2.5);

    // We don't have Array ops for f16 yet (that requires UFuncs),
    // but we can verify the underlying type works as expected in our environment
    use num_traits::Float;
    assert_eq!(a + b, f16::from_f32(4.0));
    assert_eq!(a * b, f16::from_f32(3.75));
}

#[test]
fn test_complex32_placeholder() {
    // If we were to add Complex32, it would be Complex<f16>.
    // Just checking availability of types.
    use num_complex::Complex;
    let _c = Complex::new(f16::from_f32(1.0), f16::from_f32(1.0));
    // No Dtype::Complex32 yet officially in from_type?
    // Let's check Dtype::from_type::<Complex<f16>>()
    // It likely defaults to Object or isn't specialized yet in dtype.rs
    // Dtype::from_type implementation in dtype.rs needs inspection/update if we want this.
}
