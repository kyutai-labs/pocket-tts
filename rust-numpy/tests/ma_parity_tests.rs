use numpy::modules::ma::MaskedArray;
use numpy::*;

#[test]
fn test_masked_values() {
    let data = array![1, 2, 3, 2, 5];
    let ma = MaskedArray::masked_values(data, 2).unwrap();

    // Mask should be [false, true, false, true, false]
    let mask = ma.mask();
    let expected_mask = array![false, true, false, true, false];
    assert!(mask
        .data()
        .iter()
        .zip(expected_mask.data().iter())
        .all(|(a, b)| a == b));

    // Count should be 3
    assert_eq!(ma.count(), 3);
}

#[test]
fn test_masked_outside() {
    let data = array![1, 2, 3, 4, 5, 6];
    let ma = MaskedArray::masked_outside(data, 2, 5).unwrap();

    // Mask should be [true, false, false, false, false, true]
    // Values outside [2, 5] are 1 and 6.
    let expected_mask = array![true, false, false, false, false, true];
    let mask = ma.mask();
    assert!(mask
        .data()
        .iter()
        .zip(expected_mask.data().iter())
        .all(|(a, b)| a == b));
}

#[test]
fn test_masked_inside() {
    let data = array![1, 2, 3, 4, 5, 6];
    let ma = MaskedArray::masked_inside(data, 3, 4).unwrap();

    // Mask should be [false, false, true, true, false, false]
    // Values inside [3, 4] are 3 and 4.
    let expected_mask = array![false, false, true, true, false, false];
    let mask = ma.mask();
    assert!(mask
        .data()
        .iter()
        .zip(expected_mask.data().iter())
        .all(|(a, b)| a == b));
}

#[test]
fn test_clump_masked() {
    let data = array![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mask = array![false, true, true, false, false, true, false, true, true];
    let ma = MaskedArray::new(data, mask).unwrap();

    let clumps = ma.clump_masked().unwrap();
    // Clumps of True: 1..3, 5..6, 7..9
    assert_eq!(clumps.len(), 3);
    assert_eq!(clumps[0], 1..3);
    assert_eq!(clumps[1], 5..6);
    assert_eq!(clumps[2], 7..9);
}

#[test]
fn test_clump_unmasked() {
    let data = array![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mask = array![false, true, true, false, false, true, false, true, true]; // [F, T, T, F, F, T, F, T, T]
    let ma = MaskedArray::new(data, mask).unwrap();

    let clumps = ma.clump_unmasked().unwrap();
    // Clumps of False: 0..1, 3..5, 6..7
    assert_eq!(clumps.len(), 3);
    assert_eq!(clumps[0], 0..1);
    assert_eq!(clumps[1], 3..5);
    assert_eq!(clumps[2], 6..7);
}
