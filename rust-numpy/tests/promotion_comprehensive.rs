use numpy::dtype::Dtype;
use numpy::type_promotion::promote_types;

#[test]
fn test_promote_string_bytes() {
    let string_type = Dtype::String { length: Some(10) };
    let bytes_type = Dtype::Bytes { length: 10 };

    // NumPy: np.result_type('S', 'U') -> 'U' (String wins)
    // Current impl: Bytes=6, String=5. If Kinds differ, pick higher score.
    // If Bytes > String, Bytes wins. This is WRONG.
    // We expect String to win.

    // So if implementation is wrong, this asserts result is String... and fails if it returns Bytes.
    let res = promote_types(&string_type, &bytes_type).unwrap();
    assert!(
        matches!(res, Dtype::String { .. }),
        "Expected String to win over Bytes, got {:?}",
        res
    );
}

#[test]
fn test_promote_int_string() {
    let int_type = Dtype::Int32 { byteorder: None };
    let string_type = Dtype::String { length: Some(32) };

    let res = promote_types(&int_type, &string_type).unwrap();
    assert!(
        matches!(res, Dtype::String { .. }),
        "Expected String to win over Int"
    );
}

#[test]
fn test_promote_mixed_int_float() {
    let u64_type = Dtype::UInt64 { byteorder: None };
    let f32_type = Dtype::Float32 { byteorder: None };

    // u64 + f32 -> f64
    let res = promote_types(&u64_type, &f32_type).unwrap();
    assert_eq!(res, Dtype::Float64 { byteorder: None });
}
