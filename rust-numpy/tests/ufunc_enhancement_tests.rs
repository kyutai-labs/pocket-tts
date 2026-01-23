use numpy::array2;
use numpy::dtype::Casting;

#[test]
fn test_ufunc_where() {
    let a = array2![[1.0, 2.0], [3.0, 4.0]];
    let b = array2![[10.0, 20.0], [30.0, 40.0]];
    let mask = array2![[true, false], [false, true]];

    let res = a.add(&b, Some(&mask), Casting::Safe).unwrap();
    assert_eq!(res.get_multi(&[0, 0]).unwrap(), 11.0);
    assert_eq!(res.get_multi(&[0, 1]).unwrap(), 0.0); // Mask is false, value remains default
    assert_eq!(res.get_multi(&[1, 0]).unwrap(), 0.0); // Mask is false, value remains default
    assert_eq!(res.get_multi(&[1, 1]).unwrap(), 44.0);
}

#[test]
fn test_ufunc_casting_same_type() {
    let a = array2![[1.0, 2.0], [3.0, 4.0]];
    let b = array2![[10.0, 20.0], [30.0, 40.0]];

    let res = a.add(&b, None, Casting::Safe).unwrap();
    assert_eq!(res.get_multi(&[0, 0]).unwrap(), 11.0);
    assert_eq!(res.get_multi(&[1, 1]).unwrap(), 44.0);
}
