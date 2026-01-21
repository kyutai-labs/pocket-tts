use numpy::array::Array;
use numpy::slicing::{MultiSlice, Slice};

#[test]
fn test_iter_strided() {
    // Array: [0, 1, 2, 3, 4]
    let a = Array::from_vec(vec![0, 1, 2, 3, 4]);

    // Slice: [::2] -> [0, 2, 4]
    // Manual construction of MultiSlice since s! macro might be flaky or I don't know the syntax
    // MultiSlice { slices: vec![Slice::Step(2)] }

    let mut slices = Vec::new();
    slices.push(Slice::Step(2));
    let multi_slice = MultiSlice::new(slices);

    let sliced = a.slice(&multi_slice).unwrap();

    // Check shape and strides to be sure
    // Shape should be [3] (0, 2, 4)
    // Strides should be [2]
    assert_eq!(sliced.shape(), &[3]);
    assert_eq!(sliced.strides(), &[2]);

    // Iterate
    let gathered: Vec<i32> = sliced.iter().cloned().collect();

    assert_eq!(gathered, vec![0, 2, 4]);
}

#[test]
fn test_iter_multi_dim() {
    // 2D Array
    // [[0, 1, 2],
    //  [3, 4, 5]]
    let a = Array::from_vec(vec![0, 1, 2, 3, 4, 5])
        .reshape(&[2, 3])
        .unwrap();

    // Slice: rows 0..2, cols ::2  -> [[0, 2], [3, 5]]
    // Strides for original: [3, 1]
    // Strides for slice: [3, 2]

    let mut slices = Vec::new();
    slices.push(Slice::Full); // dim 0
    slices.push(Slice::Step(2)); // dim 1
    let multi_slice = MultiSlice::new(slices);

    let sliced = a.slice(&multi_slice).unwrap();

    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(sliced.strides(), &[3, 2]);

    let gathered: Vec<i32> = sliced.iter().cloned().collect();

    // Expected order: 0, 2, 3, 5
    assert_eq!(gathered, vec![0, 2, 3, 5]);
}
