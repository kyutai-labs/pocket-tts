use numpy::array::Array;
use numpy::iterator::NDIter;

#[test]
fn test_nditer_broadcasting_2d() {
    // A: (2, 1) -> [[0], [1]]
    let a = Array::from_vec(vec![0, 1]).reshape(&[2, 1]).unwrap();
    // B: (1, 3) -> [[10, 20, 30]]
    let b = Array::from_vec(vec![10, 20, 30]).reshape(&[1, 3]).unwrap();

    // Broadcasted shape: (2, 3)
    // Values:
    // [[10, 20, 30],
    //  [11, 21, 31]]  (if we were adding them)

    let mut iter = NDIter::new(vec![&a, &b]).unwrap();
    assert_eq!(iter.shape(), &[2, 3]);

    let mut offsets = Vec::new();
    while let Some(current_offsets) = iter.next() {
        offsets.push(current_offsets);
    }

    // Expected offsets:
    // (0,0): A[0,0] offset=0, B[0,0] offset=0
    // (0,1): A[0,0] offset=0, B[0,1] offset=1
    // (0,2): A[0,0] offset=0, B[0,2] offset=2
    // (1,0): A[1,0] offset=1, B[0,0] offset=0
    // (1,1): A[1,0] offset=1, B[0,1] offset=1
    // (1,2): A[1,0] offset=1, B[0,2] offset=2

    let expected = vec![
        vec![0, 0],
        vec![0, 1],
        vec![0, 2],
        vec![1, 0],
        vec![1, 1],
        vec![1, 2],
    ];

    assert_eq!(offsets, expected);
}

#[test]
fn test_nditer_3d_complex() {
    // A: (2, 1, 1)
    let a = Array::from_vec(vec![100, 200]).reshape(&[2, 1, 1]).unwrap();
    // B: (1, 2, 1)
    let b = Array::from_vec(vec![10, 20]).reshape(&[1, 2, 1]).unwrap();
    // C: (1, 1, 2)
    let c = Array::from_vec(vec![1, 2]).reshape(&[1, 1, 2]).unwrap();

    let iter = NDIter::new(vec![&a, &b, &c]).unwrap();
    let res: Vec<Vec<usize>> = iter.collect();

    assert_eq!(res.len(), 8);
    // Should be all combinations
    // (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    // A offsets: 0, 0, 0, 0, 1, 1, 1, 1
    // B offsets: 0, 0, 1, 1, 0, 0, 1, 1
    // C offsets: 0, 1, 0, 1, 0, 1, 0, 1

    assert_eq!(res[0], vec![0, 0, 0]);
    assert_eq!(res[1], vec![0, 0, 1]);
    assert_eq!(res[4], vec![1, 0, 0]);
    assert_eq!(res[7], vec![1, 1, 1]);
}
