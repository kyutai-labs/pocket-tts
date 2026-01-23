use numpy::sorting::{argpartition, partition, ArrayOrInt};
use numpy::*;

#[test]
fn test_partition_array_kth() {
    let mut a = array![3, 4, 2, 1, 5, 0];
    // Partition around index 2 and 4
    // Sorted version: [0, 1, 2, 3, 4, 5]
    // After partitioning at 2 and 4:
    // elements at index 2 should be 2
    // elements at index 4 should be 4
    // elements before index 2 should be < 2
    // elements between 2 and 4 should be > 2 and < 4
    // elements after 4 should be > 4

    let kth = ArrayOrInt::Array(array![2, 4]);
    let res = partition(&mut a, kth, None, "quicksort", "asc").unwrap();

    assert_eq!(*res.get(2).unwrap(), 2);
    assert_eq!(*res.get(4).unwrap(), 4);

    // Check ranges
    for i in 0..2 {
        assert!(*res.get(i).unwrap() < 2);
    }
    assert_eq!(*res.get(3).unwrap(), 3);
    for i in 5..6 {
        assert!(*res.get(i).unwrap() > 4);
    }
}

#[test]
fn test_argpartition_array_kth() {
    let a = array![3, 4, 2, 1, 5, 0];
    let kth = ArrayOrInt::Array(array![1, 3]);
    // Sorted: [0, 1, 2, 3, 4, 5]
    // Indices: [5, 3, 2, 0, 1, 4]

    let res = argpartition(&a, kth, None, "quicksort", "asc").unwrap();

    // At index 1 should be the index of 1 (which is 3)
    // At index 3 should be the index of 3 (which is 0)
    assert_eq!(*res.get(1).unwrap(), 3);
    assert_eq!(*res.get(3).unwrap(), 0);
}

#[test]
fn test_partition_axis_array_kth() {
    let mut a = array2![[3, 4, 2], [1, 5, 0]];
    let kth = ArrayOrInt::Array(array![1]);

    let res = partition(&mut a, kth, Some(1), "quicksort", "asc").unwrap();

    // Row 0: [2, 3, 4] -> index 1 is 3
    // Row 1: [0, 1, 5] -> index 1 is 1
    assert_eq!(res.get_multi(&[0, 1]).unwrap(), 3);
    assert_eq!(res.get_multi(&[1, 1]).unwrap(), 1);
}
