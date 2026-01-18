use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greater_basic() {
        let a = array![1, 2, 3, 4, 5];
        let b = array![3, 2, 1, 4, 3];
        let result = a.greater(&b).unwrap();
        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.to_vec(), vec![false, false, true, false, true]);
    }

    #[test]
    fn test_less_basic() {
        let a = array![1, 2, 3, 4, 5];
        let b = array![3, 2, 1, 4, 3];
        let result = a.less(&b).unwrap();
        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.to_vec(), vec![true, false, false, false, false]);
    }

    #[test]
    fn test_greater_equal_basic() {
        let a = array![1, 2, 3, 4, 5];
        let b = array![3, 2, 1, 4, 3];
        let result = a.greater_equal(&b).unwrap();
        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.to_vec(), vec![false, true, true, true, true]);
    }

    #[test]
    fn test_less_equal_basic() {
        let a = array![1, 2, 3, 4, 5];
        let b = array![3, 2, 1, 4, 3];
        let result = a.less_equal(&b).unwrap();
        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.to_vec(), vec![true, true, false, true, true]);
    }

    #[test]
    fn test_equal_basic() {
        let a = array![1, 2, 3, 4, 5];
        let b = array![3, 2, 1, 4, 3];
        let result = a.equal(&b).unwrap();
        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.to_vec(), vec![false, true, false, true, false]);
    }

    #[test]
    fn test_not_equal_basic() {
        let a = array![1, 2, 3, 4, 5];
        let b = array![3, 2, 1, 4, 3];
        let result = a.not_equal(&b).unwrap();
        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.to_vec(), vec![true, false, true, false, true]);
    }

    #[test]
    fn test_maximum_basic() {
        let a = array![1, 5, 3];
        let b = array![4, 2, 6];
        let result = a.maximum(&b).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![4, 5, 6]);
    }

    #[test]
    fn test_minimum_basic() {
        let a = array![1, 5, 3];
        let b = array![4, 2, 6];
        let result = a.minimum(&b).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_logical_and_basic() {
        let a: Array<i32> = array![1, 0, 3, 0];
        let b: Array<i32> = array![1, 1, 0, 0];
        let result = a.logical_and(&b).unwrap();
        assert_eq!(result.to_vec(), vec![true, false, false, false]);
    }

    #[test]
    fn test_logical_or_basic() {
        let a: Array<i32> = array![1, 0, 3, 0];
        let b: Array<i32> = array![1, 1, 0, 0];
        let result = a.logical_or(&b).unwrap();
        assert_eq!(result.to_vec(), vec![true, true, true, false]);
    }

    #[test]
    fn test_logical_xor_basic() {
        let a: Array<i32> = array![1, 0, 3, 0];
        let b: Array<i32> = array![1, 1, 0, 0];
        let result = a.logical_xor(&b).unwrap();
        assert_eq!(result.to_vec(), vec![false, false, false, false]);
    }

    #[test]
    fn test_logical_not_basic() {
        let a: Array<i32> = array![0, 1, 2, 0];
        let result = a.logical_not().unwrap();
        assert_eq!(result.to_vec(), vec![true, false, true, true]);
    }

    #[test]
    fn test_greater_broadcasting() {
        let a = array2![[1, 2], [3, 4]];
        let b: Array<i32> = array![1];
        let result = a.greater(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![true, true, false, false]);
    }

    #[test]
    fn test_greater_negative_numbers() {
        let a = array![-5, -3, -1];
        let b = array![-2, -4, 0];
        let result = a.greater(&b).unwrap();
        assert_eq!(result.to_vec(), vec![true, false, false]);
    }

    #[test]
    fn test_equal_dtype() {
        let a: Array<f64> = array![1.0, 2.0, 3.0];
        let b: Array<f64> = array![1.0, 2.0, 3.0];
        let result = a.equal(&b).unwrap();
        assert_eq!(result.to_vec(), vec![true, true, true]);
    }

    #[test]
    fn test_ufunc_registry_contains_comparisons() {
        use numpy::ufunc::{get_ufunc, list_ufuncs};

        assert!(get_ufunc("greater").is_some());
        assert!(get_ufunc("less").is_some());
        assert!(get_ufunc("greater_equal").is_some());
        assert!(get_ufunc("less_equal").is_some());
        assert!(get_ufunc("equal").is_some());
        assert!(get_ufunc("not_equal").is_some());
        assert!(get_ufunc("maximum").is_some());
        assert!(get_ufunc("minimum").is_some());
        assert!(get_ufunc("logical_and").is_some());
        assert!(get_ufunc("logical_or").is_some());
        assert!(get_ufunc("logical_xor").is_some());
        assert!(get_ufunc("logical_not").is_some());

        let ufuncs = list_ufuncs();
        assert!(ufuncs.contains(&"greater"));
        assert!(ufuncs.contains(&"less"));
        assert!(ufuncs.contains(&"greater_equal"));
        assert!(ufuncs.contains(&"less_equal"));
        assert!(ufuncs.contains(&"equal"));
        assert!(ufuncs.contains(&"not_equal"));
        assert!(ufuncs.contains(&"maximum"));
        assert!(ufuncs.contains(&"minimum"));
        assert!(ufuncs.contains(&"logical_and"));
        assert!(ufuncs.contains(&"logical_or"));
        assert!(ufuncs.contains(&"logical_xor"));
        assert!(ufuncs.contains(&"logical_not"));
    }
}
