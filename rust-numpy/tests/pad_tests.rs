use numpy::*;
#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array_manipulation::pad;

    fn linear_index<T>(array: &Array<T>, indices: &[usize]) -> usize {
        numpy::strides::compute_linear_index(indices, array.strides()) as usize
    }

    #[test]
    fn test_pad_constant_1d() {
        let arr = array![1, 2, 3];
        let result = pad(&arr, &[(2, 3)], "constant", Some(0)).unwrap();

        assert_eq!(result.shape(), &[8]);
        assert_eq!(result.to_vec(), vec![0, 0, 1, 2, 3, 0, 0, 0]);
    }

    #[test]
    fn test_pad_constant_2d() {
        let arr = array2![[1, 2], [3, 4]];
        let result = pad(&arr, &[(1, 2), (2, 1)], "constant", Some(0)).unwrap();

        assert_eq!(result.shape(), &[5, 5]);
        let expected = vec![
            0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pad_edge_1d() {
        let arr = array![1, 2, 3];
        let result = pad(&arr, &[(2, 1)], "edge", None).unwrap();

        assert_eq!(result.shape(), &[6]);
        assert_eq!(result.to_vec(), vec![1, 1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_pad_edge_2d() {
        let arr = array2![[1, 2], [3, 4]];
        let result = pad(&arr, &[(1, 1), (1, 1)], "edge", None).unwrap();

        assert_eq!(result.shape(), &[4, 4]);
        let expected = vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pad_reflect_1d() {
        let arr = array![1, 2, 3, 4];
        let result = pad(&arr, &[(2, 2)], "reflect", None).unwrap();

        assert_eq!(result.shape(), &[8]);
        assert_eq!(result.to_vec(), vec![3, 2, 1, 2, 3, 4, 3, 2]);
    }

    #[test]
    fn test_pad_reflect_2d() {
        let arr = array2![[1, 2], [3, 4]];
        let result = pad(&arr, &[(1, 1), (1, 1)], "reflect", None).unwrap();

        assert_eq!(result.shape(), &[4, 4]);
        let expected = vec![4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pad_symmetric_1d() {
        let arr = array![1, 2, 3, 4];
        let result = pad(&arr, &[(2, 2)], "symmetric", None).unwrap();

        assert_eq!(result.shape(), &[8]);
        assert_eq!(result.to_vec(), vec![2, 1, 1, 2, 3, 4, 4, 3]);
    }

    #[test]
    fn test_pad_symmetric_2d() {
        let arr = array2![[1, 2], [3, 4]];
        let result = pad(&arr, &[(1, 1), (1, 1)], "symmetric", None).unwrap();

        assert_eq!(result.shape(), &[4, 4]);
        let expected = vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pad_wrap_1d() {
        let arr = array![1, 2, 3];
        let result = pad(&arr, &[(2, 2)], "wrap", None).unwrap();

        assert_eq!(result.shape(), &[7]);
        assert_eq!(result.to_vec(), vec![2, 3, 1, 2, 3, 1, 2]);
    }

    #[test]
    fn test_pad_wrap_2d() {
        let arr = array2![[1, 2], [3, 4]];
        let result = pad(&arr, &[(1, 1), (1, 1)], "wrap", None).unwrap();

        assert_eq!(result.shape(), &[4, 4]);
        let expected = vec![4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pad_linear_ramp_1d() {
        let arr = array![4, 3, 2, 1];
        let result = pad(&arr, &[(2, 2)], "linear_ramp", None).unwrap();

        assert_eq!(result.shape(), &[8]);
        let expected = vec![0, 2, 4, 3, 2, 1, 0, 0];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pad_linear_ramp_2d() {
        let arr = array2![[4, 3], [2, 1]];
        let result = pad(&arr, &[(1, 1), (1, 1)], "linear_ramp", None).unwrap();

        assert_eq!(result.shape(), &[4, 4]);
        let expected = vec![0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_pad_no_padding() {
        let arr = array![1, 2, 3];
        let result = pad(&arr, &[(0, 0)], "constant", Some(0)).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_pad_empty_array() {
        let arr = Array::<f64>::zeros(vec![0]);
        let result = pad(&arr, &[(2, 2)], "constant", Some(5.0)).unwrap();

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.to_vec(), vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_pad_3d_constant() {
        let arr = array3![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        let result = pad(&arr, &[(1, 1), (0, 0), (1, 1)], "constant", Some(0)).unwrap();

        assert_eq!(result.shape(), &[4, 2, 4]);
        assert_eq!(result.get_linear(0), Some(&0));
        assert_eq!(result.get_linear(1), Some(&0));
        assert_eq!(result.get_linear(9), Some(&1));
        assert_eq!(result.get_linear(10), Some(&2));
    }

    #[test]
    fn test_pad_invalid_mode() {
        let arr = array![1, 2, 3];
        let result = pad(&arr, &[(1, 1)], "invalid_mode", Some(0));

        assert!(result.is_err());
        if let Err(NumPyError::InvalidOperation { operation }) = result {
            assert!(operation.contains("Unsupported padding mode"));
        }
    }

    #[test]
    fn test_pad_mismatched_dimensions() {
        let arr = array![1, 2, 3];
        let result = pad(&arr, &[(1, 1), (2, 2)], "constant", Some(0));

        assert!(result.is_err());
        if let Err(NumPyError::InvalidOperation { operation }) = result {
            assert!(operation.contains("pad_width must have 1 entries, got 2"));
        }
    }

    #[test]
    fn test_pad_asymmetric_padding() {
        let arr = array![1, 2, 3];
        let result = pad(&arr, &[(1, 3)], "constant", Some(0)).unwrap();

        assert_eq!(result.shape(), &[7]);
        assert_eq!(result.to_vec(), vec![0, 1, 2, 3, 0, 0, 0]);
    }

    #[test]
    fn test_pad_floating_point() {
        let arr = array![1.5, 2.5, 3.5];
        let result = pad(&arr, &[(1, 1)], "edge", None).unwrap();

        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.to_vec(), vec![1.5, 1.5, 2.5, 3.5, 3.5]);
    }

    #[test]
    fn test_pad_complex_reflect() {
        let arr = array2![[1, 2], [3, 4], [5, 6], [7, 8]];
        let result = pad(&arr, &[(1, 1), (0, 0)], "reflect", None).unwrap();

        assert_eq!(result.shape(), &[6, 2]);

        // Check reflection for the longer dimension

        assert_eq!(result.get_linear(linear_index(&result, &[0, 0])), Some(&3));
        assert_eq!(result.get_linear(linear_index(&result, &[0, 1])), Some(&4));
        assert_eq!(result.get_linear(linear_index(&result, &[5, 0])), Some(&5));
        assert_eq!(result.get_linear(linear_index(&result, &[5, 1])), Some(&6));
    }

    #[test]
    fn test_pad_complex_symmetric() {
        let arr = array2![[1, 2], [3, 4], [5, 6], [7, 8]];
        let result = pad(&arr, &[(1, 1), (0, 0)], "symmetric", None).unwrap();

        assert_eq!(result.shape(), &[6, 2]);

        // Check symmetric reflection for the longer dimension

        assert_eq!(result.get_linear(linear_index(&result, &[0, 0])), Some(&1));
        assert_eq!(result.get_linear(linear_index(&result, &[0, 1])), Some(&2));
        assert_eq!(result.get_linear(linear_index(&result, &[5, 0])), Some(&7));
        assert_eq!(result.get_linear(linear_index(&result, &[5, 1])), Some(&8));
    }

    #[test]
    fn test_pad_integers() {
        let arr = array![1i32, 2i32, 3i32];
        let result = pad(&arr, &[(2, 1)], "constant", Some(9i32)).unwrap();

        assert_eq!(result.shape(), &[6]);
        assert_eq!(result.to_vec(), vec![9, 9, 1, 2, 3, 9]);
    }
}
