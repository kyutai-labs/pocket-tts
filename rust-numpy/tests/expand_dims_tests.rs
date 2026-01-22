use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_dims_1d_to_2d_axis_0() {
        let a = array![1, 2, 3, 4];
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1, 4]);
        assert_eq!(expanded.ndim(), 2);
    }

    #[test]
    fn test_expand_dims_1d_to_2d_axis_1() {
        let a = array![1, 2, 3, 4];
        let expanded = expand_dims(&a, 1).unwrap();
        assert_eq!(expanded.shape(), &[4, 1]);
        assert_eq!(expanded.ndim(), 2);
    }

    #[test]
    fn test_expand_dims_negative_axis() {
        let a = array![1, 2, 3, 4];
        let expanded = expand_dims(&a, -1).unwrap();
        assert_eq!(expanded.shape(), &[4, 1]);
        assert_eq!(expanded.ndim(), 2);
    }

    #[test]
    fn test_expand_dims_2d_to_3d() {
        let a = array2![[1, 2], [3, 4]];
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 2]);
        assert_eq!(expanded.ndim(), 3);
    }

    #[test]
    fn test_expand_dims_2d_middle_axis() {
        let a = array2![[1, 2], [3, 4]];
        let expanded = expand_dims(&a, 1).unwrap();
        assert_eq!(expanded.shape(), &[2, 1, 2]);
        assert_eq!(expanded.ndim(), 3);
    }

    #[test]
    fn test_expand_dims_scalar() {
        let a = Array::from_scalar(42.0, vec![]);
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1]);
        assert_eq!(expanded.ndim(), 1);
    }

    #[test]
    fn test_expand_dims_multiple_calls() {
        let a = array![1, 2, 3, 4];
        let expanded1 = expand_dims(&a, 0).unwrap();
        let expanded2 = expand_dims(&expanded1, 2).unwrap();
        assert_eq!(expanded2.shape(), &[1, 4, 1]);
        assert_eq!(expanded2.ndim(), 3);
    }

    #[test]
    fn test_expand_dims_negative_axis_2d() {
        let a = array2![[1, 2, 3], [4, 5, 6]];
        let expanded = expand_dims(&a, -2).unwrap();
        assert_eq!(expanded.shape(), &[2, 1, 3]);
    }

    #[test]
    fn test_expand_dims_preserves_data() {
        let a = array![1, 2, 3, 4];
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.get_linear(0).unwrap(), &1);
        assert_eq!(expanded.get_linear(1).unwrap(), &2);
        assert_eq!(expanded.get_linear(2).unwrap(), &3);
        assert_eq!(expanded.get_linear(3).unwrap(), &4);
    }

    #[test]
    fn test_expand_dims_axis_out_of_bounds() {
        let a = array![1, 2, 3, 4];
        let result = expand_dims(&a, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_dims_negative_out_of_bounds() {
        let a = array![1, 2, 3, 4];
        let result = expand_dims(&a, -5);
        assert!(result.is_err());
    }

    #[test]
    fn test_newaxis_constant_exists() {
        // Just verify the constant exists and is accessible
        let _ = numpy::NEWAXIS;
    }
}
