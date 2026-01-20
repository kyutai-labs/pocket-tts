use numpy::{array, s, Array};

#[test]
fn test_iter_1d() {
    let arr = array![10, 20, 30, 40];
    let collected: Vec<_> = arr.iter().cloned().collect();
    assert_eq!(collected, vec![10, 20, 30, 40]);
}

#[test]
fn test_iter_2d() {
    // [[0, 1], [2, 3]]
    let arr = Array::from_shape_vec(vec![2, 2], vec![0, 1, 2, 3]).unwrap();
    let collected: Vec<_> = arr.iter().cloned().collect();
    // Iteration should be logical row-major
    assert_eq!(collected, vec![0, 1, 2, 3]);
}

#[test]
fn test_iter_sliced() {
    let arr = array![0, 1, 2, 3, 4, 5];
    // Slice: [1, 3, 5] (step 2)
    let slice = s!(1..6..2);
    let ms = numpy::slicing::MultiSlice::new(vec![slice]);
    let view = arr.slice(&ms).unwrap();
    let collected: Vec<_> = view.iter().cloned().collect();
    assert_eq!(collected, vec![1, 3, 5]);
}

#[test]
fn test_iter_transposed() {
    // [[0, 1, 2], [3, 4, 5]]
    let arr = Array::from_shape_vec(vec![2, 3], vec![0, 1, 2, 3, 4, 5]).unwrap();
    // Transpose to [[0, 3], [1, 4], [2, 5]]
    let view = arr.transpose_view(None).unwrap(); // None means default reverse axes
    let collected: Vec<_> = view.iter().cloned().collect();
    // Should iterate logical rows of the VIEW: [0, 3, 1, 4, 2, 5]
    assert_eq!(collected, vec![0, 3, 1, 4, 2, 5]);
}

#[test]
fn test_iter_broadcast() {
    // Scalar 9 broadcast to [2, 2]
    // [[9, 9], [9, 9]]
    let arr = Array::from_scalar(9, vec![]);
    let view = arr.broadcast_to(&[2, 2]).unwrap();
    let collected: Vec<_> = view.iter().cloned().collect();
    assert_eq!(collected, vec![9, 9, 9, 9]);
}
