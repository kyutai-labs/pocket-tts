use numpy::{array2, slicing::Index, slicing::Index::*, slicing::Slice, Array};

#[test]
fn test_extract_multidim_simple_slices() {
    // 2x3 array
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // Full slice on both dimensions
    let indices = vec![Index::Slice(Slice::Full), Index::Slice(Slice::Full)];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_extract_multidim_integer_indices() {
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // Get element at [1, 2]
    let indices = vec![Index::Integer(1), Index::Integer(2)];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![6]);
}

#[test]
fn test_extract_multidim_range_slice() {
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // Slice rows [0:2], cols [1:3]
    // Should get [[2, 3], [5, 6]]
    let indices = vec![
        Index::Slice(Slice::Range(0, 2)),
        Index::Slice(Slice::Range(1, 3)),
    ];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![2, 3, 5, 6]);
}

#[test]
fn test_extract_multidim_stepped_slice() {
    let arr = array2![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];

    // Every other row, every other column starting from 0
    // Should get [[1, 3], [9, 11]]
    let indices = vec![
        Index::Slice(Slice::RangeStep(0, 3, 2)),
        Index::Slice(Slice::RangeStep(0, 4, 2)),
    ];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![1, 3, 9, 11]);
}

#[test]
fn test_extract_multidim_ellipsis() {
    // 3D array 2x2x3
    let data = vec![
        1, 2, 3, // First 2x3 slice
        4, 5, 6, 7, 8, 9, // Second 2x3 slice
        10, 11, 12,
    ];
    let arr = Array::from_data(data, vec![2, 2, 3]);

    // Use ellipsis to select from first dimension and all from remaining
    // arr[0, ...] should get [1, 2, 3, 4, 5, 6]
    let indices = vec![Index::Integer(0), Index::Ellipsis];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_extract_multidim_negative_index() {
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // Negative index: arr[-1, -1] should get 6
    let indices = vec![Index::Integer(-1), Index::Integer(-1)];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![6]);
}

#[test]
fn test_extract_multidim_from_slice() {
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // arr[1:, :] should get [4, 5, 6]
    let indices = vec![Index::Slice(Slice::From(1)), Index::Slice(Slice::Full)];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![4, 5, 6]);
}

#[test]
fn test_extract_multidim_to_slice() {
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // arr[:, :2] should get [1, 2, 4, 5]
    let indices = vec![Index::Slice(Slice::Full), Index::Slice(Slice::To(2))];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![1, 2, 4, 5]);
}

#[test]
fn test_extract_multidim_mixed_indices() {
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // Mix integer and slice: arr[1, 1:] should get [5, 6]
    let indices = vec![Index::Integer(1), Index::Slice(Slice::From(1))];
    let mut result_data = Vec::new();
    arr.extract_multidim_data(&indices, &mut result_data)
        .unwrap();

    assert_eq!(result_data, vec![5, 6]);
}
