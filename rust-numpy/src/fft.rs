#[cfg(test)]
mod fft_tests {
    use super::*;
    use crate::array::Array;
    use num_complex::Complex64;

    #[test]
    fn test_rfft2_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_shape_vec(vec![2, 2], data).unwrap();

        let result = rfft2(&input, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_irfft2_basic() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let input = Array::from_shape_vec(vec![2, 2], data).unwrap();

        let result = irfft2(&input, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rfftn_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Array::from_shape_vec(vec![2, 2, 2], data).unwrap();

        let result = rfftn(&input, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_irfftn_basic() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let input = Array::from_shape_vec(vec![2, 2], data).unwrap();

        let result = irfftn(&input, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hilbert_with_params_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);

        let result = hilbert_with_params(&input, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fft_with_params_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);

        let result = fft_with_params(&input, None, None, None);
        assert!(result.is_ok());
    }
}
