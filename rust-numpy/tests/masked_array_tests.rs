use numpy::*;

#[test]
fn test_masked_array_basic() {
    let data = array![1.0, 2.0, 3.0, 4.0];
    let mask = array![false, true, false, true];

    let ma = MaskedArray::new(data.clone(), mask.clone()).unwrap();

    assert_eq!(ma.shape(), &[4]);
    assert_eq!(ma.size(), 4);
    assert_eq!(ma.mask().data(), &[false, true, false, true]);
}

#[test]
fn test_masked_array_filled() {
    let data = array![1.0, 2.0, 3.0, 4.0];
    let mask = array![false, true, false, true];

    let mut ma = MaskedArray::new(data, mask).unwrap();
    ma.set_fill_value(99.0);

    let filled = ma.filled();
    assert_eq!(filled.data(), &[1.0, 99.0, 3.0, 99.0]);
}

#[test]
fn test_masked_array_sum() {
    let data = array![1.0, 2.0, 3.0, 4.0];
    let mask = array![false, true, false, true];

    let ma = MaskedArray::new(data, mask).unwrap();

    // 1.0 + 3.0 = 4.0
    let total = ma.sum().unwrap();
    assert_eq!(total, 4.0);
}

#[test]
fn test_masked_array_binary_op() {
    let data1 = array![1.0, 2.0, 3.0, 4.0];
    let mask1 = array![false, true, false, false];
    let ma1 = MaskedArray::new(data1, mask1).unwrap();

    let data2 = array![10.0, 20.0, 30.0, 40.0];
    let mask2 = array![false, false, true, false];
    let ma2 = MaskedArray::new(data2, mask2).unwrap();

    // ma1 + ma2
    // Result mask should be mask1 | mask2 = [false, true, true, false]
    // Result data (at unmasked) should be [11.0, _, _, 44.0]

    let res = ma1.binary_op(&ma2, |a, b, w, c| a.add(b, w, c)).unwrap();

    assert_eq!(res.mask().data(), &[false, true, true, false]);
    assert_eq!(res.data().get_multi(&[0]).unwrap(), 11.0);
    assert_eq!(res.data().get_multi(&[3]).unwrap(), 44.0);
}
