//! Tests for slicing-as-view (Issue #33)
use numpy::*;

#[test]
fn test_basic_slice_view() {
    let arr = array![0, 1, 2, 3, 4, 5];
    let slice = s!(1..4); // [1, 2, 3]
    let ms = numpy::slicing::MultiSlice::new(vec![slice]);
    let view = arr.slice(&ms).unwrap();

    assert_eq!(view.shape(), &[3]);
    assert_eq!(view.to_vec(), vec![1, 2, 3]);

    // Verify it's a view
    // (We can't easily check Arc pointer equality in safe Rust without specific APIs,
    // but we can check if modifying view affects original if we had mutable views.
    // Since Array is currently immutable-ish (Arc<MemoryManager>), we verify structure logic)
    // For now, checks are on valid data extraction.
}

#[test]
fn test_step_slice() {
    let arr = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let slice = s!(::2); // [0, 2, 4, 6, 8]
    let ms = numpy::slicing::MultiSlice::new(vec![slice]);
    let view = arr.slice(&ms).unwrap();

    assert_eq!(view.shape(), &[5]);
    assert_eq!(view.to_vec(), vec![0, 2, 4, 6, 8]);
    assert_eq!(view.strides(), &[2]);
}

#[test]
fn test_negative_step_slice() {
    let arr = array![0, 1, 2, 3, 4, 5];
    let slice = s!(::-1); // [5, 4, 3, 2, 1, 0]
    let ms = numpy::slicing::MultiSlice::new(vec![slice]);
    let view = arr.slice(&ms).unwrap();

    assert_eq!(view.shape(), &[6]);
    assert_eq!(view.to_vec(), vec![5, 4, 3, 2, 1, 0]);
    assert_eq!(view.strides(), &[-1]);
}

#[test]
fn test_multidim_slice() {
    // 3x3 array
    // [[0, 1, 2],
    //  [3, 4, 5],
    //  [6, 7, 8]]
    let data = (0..9).collect::<Vec<_>>();
    let arr = Array::from_shape_vec(vec![3, 3], data).unwrap();

    // Slice: arr[:2, 1:]
    // [[1, 2],
    //  [4, 5]]
    let slice1 = s!(..2);
    let slice2 = s!(1..);
    let ms = numpy::slicing::MultiSlice::new(vec![slice1, slice2]);
    let view = arr.slice(&ms).unwrap();

    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.to_vec(), vec![1, 2, 4, 5]);
}

#[test]
fn test_complex_negative_slice() {
    // 10 elements
    let arr = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    // arr[8:2:-2] -> [8, 6, 4]
    // Start at 8, go down to (but not including) 2, step -2
    let slice = s!(8..2..-2);
    let ms = numpy::slicing::MultiSlice::new(vec![slice]);
    let view = arr.slice(&ms).unwrap();

    assert_eq!(view.to_vec(), vec![8, 6, 4]);
}
