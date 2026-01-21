use numpy::array::Array;
use numpy::array_extra::{array_split, dsplit, hsplit, split, vsplit, SplitArg};
use numpy::slicing::Slice; // If needed for checking, but we rely on split returning Arrays

#[test]
fn test_array_split_equal() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    // Split into 3 equal sections
    let result = array_split(&a, SplitArg::Count(3), 0).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].to_vec(), vec![1, 2]);
    assert_eq!(result[1].to_vec(), vec![3, 4]);
    assert_eq!(result[2].to_vec(), vec![5, 6]);
}

#[test]
fn test_array_split_unequal() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
    // Split into 3 sections: [1,2], [3,4], [5]
    let result = array_split(&a, SplitArg::Count(3), 0).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].to_vec(), vec![1, 2]);
    assert_eq!(result[1].to_vec(), vec![3, 4]);
    assert_eq!(result[2].to_vec(), vec![5]);
}

#[test]
fn test_split_indices() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5, 6]);
    // Split at indices [2, 5] -> [0..2], [2..5], [5..]
    let result = split(&a, SplitArg::Indices(vec![2, 5]), 0).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].to_vec(), vec![1, 2]);
    assert_eq!(result[1].to_vec(), vec![3, 4, 5]);
    assert_eq!(result[2].to_vec(), vec![6]);
}

#[test]
fn test_split_error_unequal() {
    let a = Array::from_vec(vec![1, 2, 3, 4, 5]);
    // Split into 2 equal sections -> Error
    let result = split(&a, SplitArg::Count(2), 0);
    assert!(result.is_err());
}

#[test]
fn test_vsplit() {
    let a = Array::from_shape_vec(vec![4, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // vsplit into 2 -> [[1,2], [3,4]] and [[5,6], [7,8]]
    let result = vsplit(&a, SplitArg::Count(2)).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[2, 2]);
    assert_eq!(result[1].shape(), &[2, 2]);
    assert_eq!(result[0].to_vec(), vec![1, 2, 3, 4]);
    assert_eq!(result[1].to_vec(), vec![5, 6, 7, 8]);
}

#[test]
fn test_hsplit() {
    let a = Array::from_shape_vec(vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // hsplit into 2 -> [[1,2], [5,6]] and [[3,4], [7,8]]
    let result = hsplit(&a, SplitArg::Count(2)).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[2, 2]);
    assert_eq!(result[1].shape(), &[2, 2]);
    assert_eq!(result[0].to_vec(), vec![1, 2, 5, 6]);
    assert_eq!(result[1].to_vec(), vec![3, 4, 7, 8]);
}
