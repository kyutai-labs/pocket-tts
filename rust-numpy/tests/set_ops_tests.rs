use numpy::array::Array;
use numpy::set_ops::{in1d, isin};

#[test]
fn test_in1d_basic() {
    let ar1 = Array::from_vec(vec![0, 1, 2, 5, 0]);
    let ar2 = Array::from_vec(vec![0, 2]);
    let result = in1d(&ar1, &ar2, false).unwrap();
    assert_eq!(result.to_vec(), vec![true, false, true, false, true]);
}

#[test]
fn test_in1d_empty() {
    let ar1 = Array::from_vec(vec![]);
    let ar2 = Array::from_vec(vec![1, 2]);
    let result = in1d(&ar1, &ar2, false).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_in1d_floats() {
    let ar1 = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let ar2 = Array::from_vec(vec![1.0, 3.0]);
    let result = in1d(&ar1, &ar2, false).unwrap();
    assert_eq!(result.to_vec(), vec![true, false, true]);
}

#[test]
fn test_in1d_strings() {
    let ar1 = Array::from_vec(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    let ar2 = Array::from_vec(vec!["a".to_string(), "c".to_string()]);
    let result = in1d(&ar1, &ar2, false).unwrap();
    assert_eq!(result.to_vec(), vec![true, false, true]);
}

#[test]
fn test_isin_basic() {
    let ar1 = Array::from_vec(vec![0, 1, 2, 5, 0]);
    let ar2 = Array::from_vec(vec![0, 2]);
    let result = isin(&ar1, &ar2, false, false).unwrap();
    assert_eq!(result.to_vec(), vec![true, false, true, false, true]);
}

#[test]
fn test_isin_2d() {
    let ar1 = Array::from_shape_vec(vec![2, 3], vec![0, 1, 2, 3, 4, 5]);
    let ar2 = Array::from_vec(vec![0, 2, 4]);
    let result = isin(&ar1, &ar2, false, false).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.to_vec(), vec![true, false, true, false, true, false]);
}
