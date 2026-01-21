//! Tests for clip(), round(), and ptp() functions

use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== clip tests ====================

    #[test]
    fn test_clip_both_bounds() {
        let arr = array![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let clipped = array_extra::clip(&arr, Some(3), Some(7)).unwrap();
        assert_eq!(clipped.to_vec(), vec![3i32, 3, 3, 4, 5, 6, 7, 7, 7]);
    }

    #[test]
    fn test_clip_only_min() {
        let arr = array![1, 2, 3, 4, 5];
        let clipped = array_extra::clip(&arr, Some(3), None).unwrap();
        assert_eq!(clipped.to_vec(), vec![3i32, 3, 3, 4, 5]);
    }

    #[test]
    fn test_clip_only_max() {
        let arr = array![1, 2, 3, 4, 5];
        let clipped = array_extra::clip(&arr, None, Some(3)).unwrap();
        assert_eq!(clipped.to_vec(), vec![1i32, 2, 3, 3, 3]);
    }

    #[test]
    fn test_clip_no_bounds() {
        let arr = array![1, 2, 3, 4, 5];
        let clipped = array_extra::clip(&arr, None::<i32>, None).unwrap();
        assert_eq!(clipped.to_vec(), vec![1i32, 2, 3, 4, 5]);
    }

    #[test]
    fn test_clip_all_below_min() {
        let arr = array![1, 2, 3];
        let clipped = array_extra::clip(&arr, Some(5), Some(10)).unwrap();
        assert_eq!(clipped.to_vec(), vec![5i32, 5, 5]);
    }

    #[test]
    fn test_clip_all_above_max() {
        let arr = array![7, 8, 9];
        let clipped = array_extra::clip(&arr, Some(1), Some(5)).unwrap();
        assert_eq!(clipped.to_vec(), vec![5i32, 5, 5]);
    }

    #[test]
    fn test_clip_float() {
        let arr = Array::<f64>::from_vec(vec![1.5, 2.7, 3.2, 4.9]);
        let clipped = array_extra::clip(&arr, Some(2.0), Some(4.0)).unwrap();
        assert_eq!(clipped.to_vec(), vec![2.0f64, 2.7, 3.2, 4.0]);
    }

    #[test]
    fn test_clip_negative_values() {
        let arr = array![-5, -3, 0, 3, 5];
        let clipped = array_extra::clip(&arr, Some(-2), Some(2)).unwrap();
        assert_eq!(clipped.to_vec(), vec![-2i32, -2, 0, 2, 2]);
    }

    #[test]
    fn test_clip_2d_array() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];
        let clipped = array_extra::clip(&arr, Some(2), Some(5)).unwrap();
        assert_eq!(clipped.to_vec(), vec![2i32, 2, 3, 4, 5, 5]);
    }

    // ==================== round tests ====================

    #[test]
    fn test_round_zero_decimals() {
        let arr = Array::<f64>::from_vec(vec![1.2, 2.5, 3.7, 4.9]);
        let rounded = array_extra::round(&arr, 0).unwrap();
        assert_eq!(rounded.to_vec(), vec![1.0f64, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_round_one_decimal() {
        let arr = Array::<f64>::from_vec(vec![1.23, 2.56, 3.74, 4.99]);
        let rounded = array_extra::round(&arr, 1).unwrap();
        assert_eq!(rounded.to_vec(), vec![1.2f64, 2.6, 3.7, 5.0]);
    }

    #[test]
    fn test_round_two_decimals() {
        let arr = Array::<f64>::from_vec(vec![1.234, 2.567, 3.789]);
        let rounded = array_extra::round(&arr, 2).unwrap();
        assert_eq!(rounded.to_vec(), vec![1.23f64, 2.57, 3.79]);
    }

    #[test]
    fn test_round_negative_decimals() {
        let arr = Array::<f64>::from_vec(vec![123.0, 256.0, 789.0]);
        let rounded = array_extra::round(&arr, -1).unwrap();
        // 123.0 -> 120.0, 256.0 -> 260.0, 789.0 -> 790.0
        assert_eq!(rounded.to_vec(), vec![120.0f64, 260.0, 790.0]);
    }

    #[test]
    fn test_round_f32() {
        let arr = Array::<f32>::from_vec(vec![1.234, 2.567, 3.789]);
        let rounded = array_extra::round(&arr, 2).unwrap();
        assert_eq!(rounded.to_vec(), vec![1.23f32, 2.57, 3.79]);
    }

    #[test]
    fn test_round_half_to_even() {
        let arr = Array::<f64>::from_vec(vec![2.5, 3.5]);
        let rounded = array_extra::round(&arr, 0).unwrap();
        // NumPy uses banker's rounding (round half to even)
        assert_eq!(rounded.to_vec(), vec![2.0f64, 4.0]);
    }

    #[test]
    fn test_round_2d_array() {
        let arr = Array::<f64>::from_vec(vec![1.23, 2.56, 3.74, 4.99]);
        let arr_2d = arr.reshape(&[2, 2]).unwrap();
        let rounded = array_extra::round(&arr_2d, 1).unwrap();
        assert_eq!(rounded.to_vec(), vec![1.2f64, 2.6, 3.7, 5.0]);
    }

    // ==================== ptp tests ====================

    #[test]
    fn test_ptp_basic() {
        let arr = array![1, 2, 3, 4, 5];
        let range = statistics::ptp(&arr, None, false).unwrap();
        assert_eq!(range.to_vec(), vec![4i32]); // 5 - 1 = 4
    }

    #[test]
    fn test_ptp_single_element() {
        let arr = array![5];
        let range = statistics::ptp(&arr, None, false).unwrap();
        assert_eq!(range.to_vec(), vec![0i32]); // 5 - 5 = 0
    }

    #[test]
    fn test_ptp_negative_values() {
        let arr = array![-5, -3, 0, 3, 5];
        let range = statistics::ptp(&arr, None, false).unwrap();
        assert_eq!(range.to_vec(), vec![10i32]); // 5 - (-5) = 10
    }

    #[test]
    fn test_ptp_float() {
        let arr = Array::<f64>::from_vec(vec![1.5, 2.7, 3.2, 4.9]);
        let range = statistics::ptp(&arr, None, false).unwrap();
        assert_eq!(range.to_vec(), vec![4.9 - 1.5]); // 4.9 - 1.5 = 3.4
    }

    #[test]
    fn test_ptp_empty_array_error() {
        let arr = Array::<i32>::from_vec(vec![]);
        let result = statistics::ptp(&arr, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_ptp_all_same_values() {
        let arr = array![5, 5, 5, 5];
        let range = statistics::ptp(&arr, None, false).unwrap();
        assert_eq!(range.to_vec(), vec![0i32]); // 5 - 5 = 0
    }

    #[test]
    fn test_ptp_large_range() {
        let arr = array![0, 100, 200, 300, 400, 500];
        let range = statistics::ptp(&arr, None, false).unwrap();
        assert_eq!(range.to_vec(), vec![500i32]); // 500 - 0 = 500
    }

    #[test]
    fn test_ptp_f32() {
        let arr = Array::<f32>::from_vec(vec![1.5, 2.7, 3.2, 4.9]);
        let range = statistics::ptp(&arr, None, false).unwrap();
        assert!((range.to_vec()[0] - (4.9 - 1.5)).abs() < 0.001);
    }
}
