use numpy::dtype::Dtype;
use numpy::type_promotion::{promote_types, promote_types_bitwise, promote_types_division};

#[test]
fn test_promote_types_basic() {
    let i32_type = Dtype::Int32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };

    // i32 + f64 -> f64
    assert_eq!(promote_types(&i32_type, &f64_type).unwrap(), f64_type);

    // i32 + i32 -> i32
    assert_eq!(promote_types(&i32_type, &i32_type).unwrap(), i32_type);
}

#[test]
fn test_promote_mixed_integers() {
    let u8_type = Dtype::UInt8 { byteorder: None };
    let i8_type = Dtype::Int8 { byteorder: None };
    let i16_type = Dtype::Int16 { byteorder: None };

    // u8 + i8 -> i16 (to fit both)
    assert_eq!(promote_types(&u8_type, &i8_type).unwrap(), i16_type);

    let u32_type = Dtype::UInt32 { byteorder: None };
    let i64_type = Dtype::Int64 { byteorder: None };
    // u32 + i32 -> i64
    assert_eq!(
        promote_types(&u32_type, &Dtype::Int32 { byteorder: None }).unwrap(),
        i64_type
    );
}

#[test]
fn test_promote_division() {
    let i32_type = Dtype::Int32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };

    // i32 / i32 -> f64 (NumPy true_divide)
    assert_eq!(
        promote_types_division(&i32_type, &i32_type).unwrap(),
        f64_type
    );
}

#[test]
fn test_promote_bitwise() {
    let i32_type = Dtype::Int32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };

    // i32 & i32 -> i32
    assert!(promote_types_bitwise(&i32_type, &i32_type).is_some());

    // f64 & f64 -> None (invalid for floats)
    assert!(promote_types_bitwise(&f64_type, &f64_type).is_none());
}

#[test]
fn test_promote_complex() {
    let f64_type = Dtype::Float64 { byteorder: None };
    let c128_type = Dtype::Complex128 { byteorder: None };

    // f64 + c128 -> c128
    assert_eq!(promote_types(&f64_type, &c128_type).unwrap(), c128_type);
}
