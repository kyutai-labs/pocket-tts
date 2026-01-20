use numpy::{Casting, Dtype};

#[test]
fn test_casting_modes() {
    let f32 = Dtype::Float32 { byteorder: None };
    let f64 = Dtype::Float64 { byteorder: None };

    // cast f64 -> f32
    // Safe? No (loss of precision)
    assert!(!f64.can_cast(&f32, Casting::Safe));

    // SameKind? Yes (both float)
    assert!(f64.can_cast(&f32, Casting::SameKind));

    // Unsafe? Yes
    assert!(f64.can_cast(&f32, Casting::Unsafe));

    // Equiv? No
    assert!(!f64.can_cast(&f32, Casting::Equiv));

    // No? No
    assert!(!f64.can_cast(&f32, Casting::No));

    // Same type casting
    assert!(f32.can_cast(&f32, Casting::No));
}

#[test]
fn test_safe_conversions() {
    let i32 = Dtype::Int32 { byteorder: None };
    let i64 = Dtype::Int64 { byteorder: None };
    let f32 = Dtype::Float32 { byteorder: None };
    let f64 = Dtype::Float64 { byteorder: None };

    // i32 -> i64 (safe)
    assert!(i32.can_cast(&i64, Casting::Safe));

    // i64 -> i32 (unsafe)
    assert!(!i64.can_cast(&i32, Casting::Safe));

    // i32 -> f64 (safe - approx but generally allowed as safe in NumPy)
    assert!(i32.can_cast(&f64, Casting::Safe));

    // i32 -> f32 (unsafe - 31 bits vs 24 bits)
    assert!(!i32.can_cast(&f32, Casting::Safe));
}

#[test]
fn test_mixed_safe() {
    let u8 = Dtype::UInt8 { byteorder: None };
    let i16 = Dtype::Int16 { byteorder: None };
    let i8 = Dtype::Int8 { byteorder: None };

    // u8 -> i16 (safe)
    assert!(u8.can_cast(&i16, Casting::Safe));

    // u8 -> i8 (unsafe - overflow)
    assert!(!u8.can_cast(&i8, Casting::Safe));
}
