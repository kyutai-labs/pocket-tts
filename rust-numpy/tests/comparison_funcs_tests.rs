use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allclose_basic() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(allclose(&a, &b, None, None, None).unwrap());
    }

    #[test]
    fn test_allclose_with_tolerance() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0001, 2.0001, 3.0001]);
        assert!(allclose(&a, &b, Some(1e-4), Some(1e-4), None).unwrap());
    }

    #[test]
    fn test_allclose_failure() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.5, 3.0]);
        assert!(!allclose(&a, &b, None, None, None).unwrap());
    }

    #[test]
    fn test_allclose_with_nan_equal() {
        let a = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
        let b = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(allclose(&a, &b, None, None, Some(true)).unwrap());
    }

    #[test]
    fn test_allclose_with_nan_not_equal() {
        let a = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
        let b = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(!allclose(&a, &b, None, None, Some(false)).unwrap());
    }

    #[test]
    fn test_isclose_element_wise() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0001, 2.5, 3.0001]);
        let result = isclose(&a, &b, Some(1e-4), Some(1e-4), None).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.get_linear(0).unwrap(), &true);
        assert_eq!(result.get_linear(1).unwrap(), &false);
        assert_eq!(result.get_linear(2).unwrap(), &true);
    }

    #[test]
    fn test_isclose_broadcasting() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![2.0]);
        let result = isclose(&a, &b, None, Some(1.0), None).unwrap();
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_array_equal_identical() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2, 3]);
        assert!(array_equal(&a, &b));
    }

    #[test]
    fn test_array_equal_different() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2, 4]);
        assert!(!array_equal(&a, &b));
    }

    #[test]
    fn test_array_equal_different_shape() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2]);
        assert!(!array_equal(&a, &b));
    }

    #[test]
    fn test_array_equal_2d() {
        let a = array2![[1, 2], [3, 4]];
        let b = array2![[1, 2], [3, 4]];
        assert!(array_equal(&a, &b));
    }

    #[test]
    fn test_array_equiv_broadcasting() {
        let a = Array::from_vec(vec![1, 1, 1]);
        let b = Array::from_vec(vec![1]);
        assert!(array_equiv(&a, &b));
    }

    #[test]
    fn test_array_equiv_no_broadcasting() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2, 3]);
        assert!(array_equiv(&a, &b));
    }

    #[test]
    fn test_array_equiv_different_values() {
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![1, 2, 4]);
        assert!(!array_equiv(&a, &b));
    }

    #[test]
    fn test_allclose_infinity() {
        let a = Array::from_vec(vec![f64::INFINITY, 1.0]);
        let b = Array::from_vec(vec![f64::INFINITY, 1.0]);
        assert!(allclose(&a, &b, None, None, None).unwrap());
    }

    #[test]
    fn test_allclose_different_infinity() {
        let a = Array::from_vec(vec![f64::INFINITY, 1.0]);
        let b = Array::from_vec(vec![f64::NEG_INFINITY, 1.0]);
        assert!(!allclose(&a, &b, None, None, None).unwrap());
    }

    #[test]
    fn test_array_equal_scalar() {
        let a = Array::from_scalar(42, vec![]);
        let b = Array::from_scalar(42, vec![]);
        assert!(array_equal(&a, &b));
    }

    #[test]
    fn test_array_equiv_scalar_broadcast() {
        let a = Array::from_vec(vec![1, 1, 1]);
        let b = Array::from_scalar(1, vec![]);
        assert!(array_equiv(&a, &b));
    }

    #[test]
    fn test_isclose_default_parameters() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.00001, 2.00001, 3.00001]);
        let result = isclose(&a, &b, None, None, None).unwrap();
        assert_eq!(result.get_linear(0).unwrap(), &true);
    }
}
