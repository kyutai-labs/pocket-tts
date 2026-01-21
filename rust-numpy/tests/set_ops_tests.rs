use numpy::array::Array;
use numpy::set_ops::{in1d, intersect1d, isin, setdiff1d, setxor1d, union1d};

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

#[test]
fn test_intersect1d() {
    let ar1 = Array::from_vec(vec![1, 3, 4, 3]);
    let ar2 = Array::from_vec(vec![3, 1, 2, 1]);
    let result = intersect1d(&ar1, &ar2, false, false).unwrap();
    // Unique common elements, sorted: 1, 3
    assert_eq!(result.values.to_vec(), vec![1, 3]);
}

#[test]
fn test_union1d() {
    let ar1 = Array::from_vec(vec![1, 2, 3]);
    let ar2 = Array::from_vec(vec![2, 3, 4]);
    let result = union1d(&ar1, &ar2).unwrap();
    // Unique elements from both, sorted: 1, 2, 3, 4
    assert_eq!(result.to_vec(), vec![1, 2, 3, 4]);
}

#[test]
fn test_setdiff1d() {
    let ar1 = Array::from_vec(vec![1, 2, 3, 4, 1]);
    let ar2 = Array::from_vec(vec![2, 4]);
    let result = setdiff1d(&ar1, &ar2, false).unwrap();
    // Unique elements in ar1 not in ar2, sorted: 1, 3
    assert_eq!(result.to_vec(), vec![1, 3]);
}

#[test]
fn test_setxor1d() {
    let ar1 = Array::from_vec(vec![1, 2, 3, 2]);
    let ar2 = Array::from_vec(vec![2, 3, 5, 5]);
    let result = setxor1d(&ar1, &ar2, false).unwrap();
    // Unique elements in ar1 or ar2 but not both, sorted: 1, 5
    assert_eq!(result.to_vec(), vec![1, 5]);
}
