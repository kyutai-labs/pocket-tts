use numpy::{promote_types, Dtype};

#[test]
fn test_same_kind_promotion() {
    let i8 = Dtype::Int8 { byteorder: None };
    let _i16 = Dtype::Int16 { byteorder: None };
    let i32 = Dtype::Int32 { byteorder: None };

    assert_eq!(promote_types(&i8, &i8), Some(i8.clone()));
    assert_eq!(promote_types(&i8, &i16), Some(i16.clone()));
    assert_eq!(promote_types(&i16, &i32), Some(i32.clone()));

    let f32 = Dtype::Float32 { byteorder: None };
    let f64 = Dtype::Float64 { byteorder: None };
    assert_eq!(promote_types(&f32, &f64), Some(f64.clone()));
}

#[test]
fn test_mixed_kind_promotion() {
    let i32 = Dtype::Int32 { byteorder: None };
    let f32 = Dtype::Float32 { byteorder: None };
    let f64 = Dtype::Float64 { byteorder: None };

    // Int + Float -> Float
    assert_eq!(promote_types(&i32, &f32), Some(f64.clone())); // NumPy i32+f32 -> f64

    let i8 = Dtype::Int8 { byteorder: None };
    // i8 (byte) + f32 -> f32 ? (NumPy i8+f32 -> f32? Yes usually)
    // Our implementation does heuristic <= 8 bytes float, >= 4 bytes int -> f64
    // i8 size 1. f32 size 4.
    // logic: if f_size < 8 && i_size >= 4 -> f64.
    // 1 < 4 is false. So returns f32.
    assert_eq!(promote_types(&i8, &f32), Some(f32.clone()));
}

#[test]
fn test_signed_unsigned_promotion() {
    let u8 = Dtype::UInt8 { byteorder: None };
    let i8 = Dtype::Int8 { byteorder: None };
    let i16 = Dtype::Int16 { byteorder: None };

    // u8 + i8 -> i16 (to hold both ranges)
    let res = promote_types(&u8, &i8);
    assert!(matches!(res, Some(Dtype::Int16 { .. })));

    let u32 = Dtype::UInt32 { byteorder: None };
    let i32 = Dtype::Int32 { byteorder: None };
    // u32 + i32 -> i64
    let res2 = promote_types(&u32, &i32);
    assert!(matches!(res2, Some(Dtype::Int64 { .. })));
}

#[test]
fn test_bool_promotion() {
    let b = Dtype::Bool;
    let i8 = Dtype::Int8 { byteorder: None };
    let f32 = Dtype::Float32 { byteorder: None };

    assert_eq!(promote_types(&b, &i8), Some(i8.clone()));
    assert_eq!(promote_types(&b, &f32), Some(f32.clone()));
}
