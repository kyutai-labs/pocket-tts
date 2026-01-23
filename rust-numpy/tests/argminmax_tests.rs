use numpy::*;

#[test]
fn test_argmin_basic() {
    let a = array![3, 1, 4, 1, 5];
    let res = a.argmin(None).unwrap();
    assert_eq!(res.size(), 1);
    assert_eq!(*res.get(0).unwrap(), 1);
}

#[test]
fn test_argmax_basic() {
    let a = array![3, 1, 4, 1, 5];
    let res = a.argmax(None).unwrap();
    assert_eq!(res.size(), 1);
    assert_eq!(*res.get(0).unwrap(), 4);
}

#[test]
fn test_argmin_axis() {
    let a = array2![[10, 20, 30], [5, 40, 2]];

    // argmin along axis 0 (rows)
    let res0 = a.argmin(Some(0)).unwrap();
    assert_eq!(res0.shape(), &[3]);
    assert_eq!(res0.data(), &[1, 0, 1]);

    // argmin along axis 1 (columns)
    let res1 = a.argmin(Some(1)).unwrap();
    assert_eq!(res1.shape(), &[2]);
    assert_eq!(res1.data(), &[0, 2]);
}

#[test]
fn test_argmax_axis() {
    let a = array2![[10, 20, 30], [5, 40, 2]];

    // argmax along axis 0 (rows)
    let res0 = a.argmax(Some(0)).unwrap();
    assert_eq!(res0.shape(), &[3]);
    assert_eq!(res0.data(), &[0, 1, 0]);

    // argmax along axis 1 (columns)
    let res1 = a.argmax(Some(1)).unwrap();
    assert_eq!(res1.shape(), &[2]);
    assert_eq!(res1.data(), &[2, 1]);
}

#[test]
fn test_argminmax_empty() {
    let a: Array<i32> = Array::from_vec(vec![]);
    assert!(a.argmin(None).is_err());
    assert!(a.argmax(None).is_err());
}
