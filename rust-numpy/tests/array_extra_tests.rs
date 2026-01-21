//! Tests for array_extra module functions (trim_zeros, ediff1d, etc.)

use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== trim_zeros tests ====================

    #[test]
    fn test_trim_zeros_fb_default() {
        let arr = array![0, 0, 1, 2, 3, 0, 0];
        let trimmed = array_extra::trim_zeros(&arr, "fb").unwrap();
        assert_eq!(trimmed.to_vec(), vec![1i32, 2, 3]);
    }

    #[test]
    fn test_trim_zeros_front() {
        let arr = array![0, 0, 1, 2, 3, 0, 0];
        let trimmed = array_extra::trim_zeros(&arr, "f").unwrap();
        assert_eq!(trimmed.to_vec(), vec![1i32, 2, 3, 0, 0]);
    }

    #[test]
    fn test_trim_zeros_back() {
        let arr = array![0, 0, 1, 2, 3, 0, 0];
        let trimmed = array_extra::trim_zeros(&arr, "b").unwrap();
        assert_eq!(trimmed.to_vec(), vec![0i32, 0, 1, 2, 3]);
    }

    #[test]
    fn test_trim_zeros_all_zeros() {
        let arr = array![0, 0, 0, 0];
        let trimmed = array_extra::trim_zeros(&arr, "fb").unwrap();
        assert_eq!(trimmed.to_vec(), Vec::<i32>::new());
        assert_eq!(trimmed.shape(), &[0]);
    }

    #[test]
    fn test_trim_zeros_empty_array() {
        let arr = Array::<i32>::from_vec(vec![]);
        let trimmed = array_extra::trim_zeros(&arr, "fb").unwrap();
        assert_eq!(trimmed.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_trim_zeros_no_zeros() {
        let arr = array![1, 2, 3, 4, 5];
        let trimmed = array_extra::trim_zeros(&arr, "fb").unwrap();
        assert_eq!(trimmed.to_vec(), vec![1i32, 2, 3, 4, 5]);
    }

    #[test]
    fn test_trim_zeros_only_front_zeros() {
        let arr = array![0, 0, 1, 2, 3];
        let trimmed = array_extra::trim_zeros(&arr, "fb").unwrap();
        assert_eq!(trimmed.to_vec(), vec![1i32, 2, 3]);
    }

    #[test]
    fn test_trim_zeros_only_back_zeros() {
        let arr = array![1, 2, 3, 0, 0];
        let trimmed = array_extra::trim_zeros(&arr, "fb").unwrap();
        assert_eq!(trimmed.to_vec(), vec![1i32, 2, 3]);
    }

    #[test]
    fn test_trim_zeros_invalid_mode() {
        let arr = array![0, 0, 1, 2, 3, 0, 0];
        let result = array_extra::trim_zeros(&arr, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_trim_zeros_2d_array_error() {
        let arr = array2![[0, 0], [1, 2]];
        let result = array_extra::trim_zeros(&arr, "fb");
        assert!(result.is_err());
    }

    #[test]
    fn test_trim_zeros_float_zeros() {
        let arr = Array::<f64>::from_vec(vec![0.0, 0.0, 1.5, 2.5, 0.0]);
        let trimmed = array_extra::trim_zeros(&arr, "fb").unwrap();
        assert_eq!(trimmed.to_vec(), vec![1.5f64, 2.5]);
    }

    // ==================== ediff1d tests ====================

    #[test]
    fn test_ediff1d_basic() {
        let arr = array![1, 2, 4, 7];
        let diff = array_extra::ediff1d(&arr, None, None).unwrap();
        assert_eq!(diff.to_vec(), vec![1i32, 2, 3]);
    }

    #[test]
    fn test_ediff1d_with_to_end() {
        let arr = array![1, 2, 4];
        let diff = array_extra::ediff1d(&arr, Some(&[99]), None).unwrap();
        assert_eq!(diff.to_vec(), vec![1i32, 2, 99]);
    }

    #[test]
    fn test_ediff1d_with_to_begin() {
        let arr = array![1, 2, 4];
        let diff = array_extra::ediff1d(&arr, None, Some(&[88])).unwrap();
        assert_eq!(diff.to_vec(), vec![88i32, 1, 2]);
    }

    #[test]
    fn test_ediff1d_with_both() {
        let arr = array![1, 2, 4];
        let diff = array_extra::ediff1d(&arr, Some(&[99]), Some(&[88])).unwrap();
        assert_eq!(diff.to_vec(), vec![88i32, 1, 2, 99]);
    }

    #[test]
    fn test_ediff1d_empty_array() {
        let arr = Array::<i32>::from_vec(vec![]);
        let diff = array_extra::ediff1d(&arr, Some(&[1, 2]), Some(&[3, 4])).unwrap();
        assert_eq!(diff.to_vec(), vec![3i32, 4, 1, 2]);
    }

    #[test]
    fn test_ediff1d_single_element() {
        let arr = array![5];
        let diff = array_extra::ediff1d(&arr, None, None).unwrap();
        assert_eq!(diff.to_vec(), Vec::<i32>::new());
    }

    #[test]
    fn test_ediff1d_negative_numbers() {
        let arr = array![-5, -3, 0, 3];
        let diff = array_extra::ediff1d(&arr, None, None).unwrap();
        assert_eq!(diff.to_vec(), vec![2i32, 3, 3]);
    }

    #[test]
    fn test_ediff1d_float() {
        let arr = Array::<f64>::from_vec(vec![1.0, 2.5, 4.5]);
        let diff = array_extra::ediff1d(&arr, None, None).unwrap();
        assert_eq!(diff.to_vec(), vec![1.5f64, 2.0]);
    }

    #[test]
    fn test_ediff1d_multiple_to_end() {
        let arr = array![1, 2, 4];
        let diff = array_extra::ediff1d(&arr, Some(&[99, 100]), None).unwrap();
        assert_eq!(diff.to_vec(), vec![1i32, 2, 99, 100]);
    }

    #[test]
    fn test_ediff1d_multiple_to_begin() {
        let arr = array![1, 2, 4];
        let diff = array_extra::ediff1d(&arr, None, Some(&[88, 89])).unwrap();
        assert_eq!(diff.to_vec(), vec![88i32, 89, 1, 2]);
    }

    #[test]
    fn test_ediff1d_empty_array_no_args() {
        let arr = Array::<i32>::from_vec(vec![]);
        let diff = array_extra::ediff1d(&arr, None, None).unwrap();
        assert_eq!(diff.to_vec(), Vec::<i32>::new());
    }
}
