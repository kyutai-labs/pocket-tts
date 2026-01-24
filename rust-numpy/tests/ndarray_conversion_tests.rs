use num_complex::Complex64;
use numpy::array;
use numpy::dtype::Dtype;
use numpy::Array;
use std::io::Read;

#[test]
fn test_view() {
    let a = array![1, 2, 3, 4];
    let v = a.view();

    assert_eq!(a.shape(), v.shape());
    assert_eq!(a.strides(), v.strides());
    assert_eq!(a.size(), v.size());

    // Ensure data is shared (conceptually, though we can't easily check Arc count here without internal access)
    // But we can check equality
    assert_eq!(a.to_vec(), v.to_vec());
}

#[test]
fn test_astype() {
    let a = array![1, 2, 3, 4];
    // converting i32/i64 to f64
    let b: Array<f64> = a.astype().unwrap();

    assert_eq!(b.dtype(), &Dtype::Float64 { byteorder: None });
    assert_eq!(b.get(0).unwrap(), &1.0);
}

#[test]
fn test_conj() {
    // Real array conj should be no-op (copy)
    let a = array![1.0, 2.0];
    let b = a.conj().unwrap();
    assert_eq!(a.to_vec(), b.to_vec());

    // Complex array
    let c1 = Complex64::new(1.0, 2.0);
    let cArray = Array::from_vec(vec![c1]);
    let conj = cArray.conj().unwrap();

    let res = conj.get(0).unwrap();
    assert_eq!(res.re, 1.0);
    assert_eq!(res.im, -2.0);
}

#[test]
fn test_tobytes() {
    let a = array![1i32, 2i32]; // 4 bytes each = 8 bytes
    let bytes = a.tobytes().unwrap();

    assert_eq!(bytes.len(), 8);
    // Little endian check usually
    // 1 -> 01 00 00 00
    // 2 -> 02 00 00 00
    if cfg!(target_endian = "little") {
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[4], 2);
    }
}

#[test]
fn test_tolist() {
    let a = array![1, 2];
    // For now, tolist might just return Vec<T> for 1D, or we verify string representation if generic return is hard in Rust
    let list = a.tolist();
    // Expectation: some representation we can check.
    // If tolist returns Vec<T>, then:
    assert_eq!(list, vec![1, 2]);
}

#[test]
fn test_tofile() {
    let a = array![1, 2, 3];
    let tmp_dir = tempfile::tempdir().unwrap();
    let file_path = tmp_dir.path().join("test_array.bin");

    a.tofile(&file_path).unwrap();

    let mut f = std::fs::File::open(file_path).unwrap();
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();

    // 3 * size_of(i32) or i64 depending on default
    // assuming i64 default for integer literals in array! macro if inferred?
    // The previous test logic suggests generic T.
    // `array![1, 2, 3]` usually creates i32 or i64.
    // Let's assert based on `a.size() * std::mem::size_of::<i32>()` (or i64)

    // In lib.rs: pub type Int = i64;
    // But macro uses T inference.
    // Let's use explicit type
    let a_i32 = Array::from_vec(vec![1i32, 2, 3]);
    let tmp_dir_2 = tempfile::tempdir().unwrap();
    let file_path_2 = tmp_dir_2.path().join("test_array_i32.bin");
    a_i32.tofile(&file_path_2).unwrap();

    let mut f2 = std::fs::File::open(file_path_2).unwrap();
    let mut buffer2 = Vec::new();
    f2.read_to_end(&mut buffer2).unwrap();

    assert_eq!(buffer2.len(), 12); // 3 * 4 bytes
}
