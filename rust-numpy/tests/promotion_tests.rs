use numpy::dtype::Dtype;
use numpy::type_promotion::promote_types;

#[test]
fn test_string_unicode_promotion() {
    let s = Dtype::String { length: Some(8) }; // 'S'
    let u = Dtype::Unicode { length: Some(8) }; // 'U'

    // In NumPy: S + U -> U
    // Currently in rust-numpy, loop might default to one or the other based on ordering or fail
    let res = promote_types(&s, &u).expect("Promotion failed for String + Unicode");
    match res {
        Dtype::Unicode { .. } => (), // Correct
        Dtype::String { .. } => panic!("Promoted to String instead of Unicode"),
        _ => panic!("Promoted to unexpected type: {:?}", res),
    }

    let res2 = promote_types(&u, &s).expect("Promotion failed for Unicode + String");
    match res2 {
        Dtype::Unicode { .. } => (), // Correct
        Dtype::String { .. } => panic!("Promoted to String instead of Unicode (reversed)"),
        _ => panic!("Promoted to unexpected type: {:?}", res2),
    }
}

#[test]
fn test_bytes_bytes_promotion() {
    // Testing Dtype::Bytes (assumed to be 'V' or raw bytes)
    // If assuming 'S' (String) is bytes, then this test is redundant with String + String.
    // But we have a specific Dtype::Bytes variant.
    let b1 = Dtype::Bytes { length: 10 };
    let b2 = Dtype::Bytes { length: 20 };

    let res = promote_types(&b1, &b2).expect("Promotion failed for Bytes + Bytes");
    match res {
        Dtype::Bytes { length } => assert_eq!(length, 20, "Should take max length"),
        _ => panic!("Promoted to non-Bytes type: {:?}", res),
    }
}
