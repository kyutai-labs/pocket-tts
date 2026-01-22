use numpy::*;

#[test]
fn test_boolean_indexing() {
    let arr = array![1, 2, 3, 4, 5];
    let mask = array![true, false, true, false, true];

    let res = arr.get_mask(&mask).unwrap();
    assert_eq!(res.shape(), &[3]);
    assert_eq!(res.to_vec(), vec![1, 3, 5]);
}

#[test]
fn test_take_flat() {
    let arr = array![10, 20, 30, 40, 50];
    let indices = array![0, 4, 1];

    let res = arr.take(&indices, None).unwrap();
    assert_eq!(res.shape(), &[3]);
    assert_eq!(res.to_vec(), vec![10, 50, 20]);
}

#[test]
fn test_take_axis() {
    // 2x3 array
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // Take rows 1 then 0
    let indices = array![1, 0];
    let res = arr.take(&indices, Some(0)).unwrap();

    assert_eq!(res.shape(), &[2, 3]);
    assert_eq!(res.to_vec(), vec![4, 5, 6, 1, 2, 3]);
}

#[test]
fn test_take_axis_1() {
    let arr = array2![[1, 2, 3], [4, 5, 6]];

    // Take columns 2 then 0
    let indices = array![2, 0];
    let res = arr.take(&indices, Some(1)).unwrap();

    assert_eq!(res.shape(), &[2, 2]);
    // [[3, 1],
    //  [6, 4]]
    assert_eq!(res.to_vec(), vec![3, 1, 6, 4]);
}

#[test]
fn test_take_out_of_bounds() {
    let arr = array![1, 2, 3];
    let indices = array![3];

    let res = arr.take(&indices, None);
    assert!(res.is_err());
}
