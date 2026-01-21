use numpy::array::Array;
use numpy::slicing::{MultiSlice, Slice};

#[test]
fn test_slice_view_semantics() {
    // 1D Array: [0, 1, 2, 3, 4]
    let mut a = Array::from_vec(vec![0, 1, 2, 3, 4]);

    // Slice: [1:4] -> [1, 2, 3]
    let slices = vec![Slice::Range(1, 4)];
    let multi_slice = MultiSlice::new(slices);

    let mut view = a.slice(&multi_slice).unwrap();

    // Check view content
    assert_eq!(view.size(), 3);
    assert_eq!(view.get(0).unwrap(), &1);

    // Modify view
    view.set(0, 99).unwrap();

    // Check if original array is modified (View Semantics)
    // Current implementation copies, so this will fail if it's a copy.
    assert_eq!(
        a.get(1).unwrap(),
        &99,
        "Modifying view should modify original array"
    );
}

#[test]
fn test_slice_step() {
    // 1D Array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let a = Array::from_vec((0..10).collect());

    // Slice: [1:8:2] -> [1, 3, 5, 7]
    let slices = vec![Slice::RangeStep(1, 8, 2)];
    let multi_slice = MultiSlice::new(slices);

    let view = a.slice(&multi_slice).unwrap();

    assert_eq!(view.size(), 4);
    assert_eq!(view.get(0).unwrap(), &1);
    assert_eq!(view.get(1).unwrap(), &3);
    assert_eq!(view.get(2).unwrap(), &5);
    assert_eq!(view.get(3).unwrap(), &7);
}

#[test]
fn test_slice_negative_step() {
    // 1D Array: [0, 1, 2, 3, 4, 5]
    let a = Array::from_vec((0..6).collect());

    // Slice: [::-1] -> [5, 4, 3, 2, 1, 0]
    let slices = vec![Slice::Step(-1)];
    let multi_slice = MultiSlice::new(slices);

    let view = a.slice(&multi_slice).unwrap();

    assert_eq!(view.size(), 6);
    assert_eq!(view.get(0).unwrap(), &5);
    assert_eq!(view.get(5).unwrap(), &0);
}
