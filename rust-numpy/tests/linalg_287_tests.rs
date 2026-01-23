#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn test_sign_functions() {
        // Test sign function with various values
        let x = Array::from_data(vec![-2.0f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0f64], vec![6]);
        let result = sign(&x).unwrap();
        let expected = vec![-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(result.size(), 6);
        for i in 0..6 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_signbit_function() {
        // Test signbit function with various values
        let x = Array::from_data(vec![-2.0f64, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0f64], vec![6]);
        let result = signbit(&x).unwrap();
        let expected = vec![true, true, true, true, false, false, false];
        assert_eq!(result.size(), 6);
        for i in 0..6 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_copysign_function() {
        // Test copysign function
        let x1 = Array::from_data(vec![1.0, -2.0, 3.0, -4.0f64], vec![4]);
        let x2 = Array::from_data(vec![1.0, -1.0, 1.0, -1.0f64], vec![4]);
        let result = copysign(&x1, &x2).unwrap();
        let expected = vec![1.0, 2.0, 3.0, 4.0f64];
        assert_eq!(result.size(), 4);
        for i in 0..4 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_fabs_function() {
        // Test fabs function (float only)
        let x = Array::from_data(
            vec![-2.5f32, -1.0f32, -0.5f32, 0.0f32, 0.5f32, 1.0f32, 2.0f32],
            vec![6],
        );
        let result = fabs(&x).unwrap();
        let expected = vec![2.5, 1.0, 0.5, 0.0, 0.5, 1.0, 2.0f32];
        assert_eq!(result.size(), 6);
        for i in 0..6 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_absolute_function() {
        // Test absolute function (alias for abs)
        let x = Array::from_data(vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], vec![6]);
        let result = absolute(&x).unwrap();
        let expected = vec![2.0, 1.0, 0.5, 0.0, 0.5, 1.0, 2.0];
        assert_eq!(result.size(), 6);
        for i in 0..6 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_sign_edge_cases() {
        // Test sign with edge cases (should be implemented in the future)
        // NaN, Inf, -Inf should return NaN (not tested here)
        // For now, just test basic behavior
        let x = Array::from_data(vec![-0.0, 0.0f64], vec![2]);
        let result = sign(&x).unwrap();
        assert_eq!(result.size(), 2);
        // Both should be 0 (sign of zero is zero)
        assert_eq!(result.get(0), Some(&0.0f64));
        assert_eq!(result.get(1), Some(&0.0f64));
    }

    #[test]
    fn test_sign_with_integers() {
        // Test sign with integer types
        let x = Array::from_data(vec![-5i64, -1i64, 0i64, 1i64, 5i64], vec![4]);
        let result = sign(&x).unwrap();
        let expected = vec![-1i64, -1i64, 0i64, 1i64, 1i64];
        assert_eq!(result.size(), 4);
        for i in 0..4 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_sign_with_complex_numbers() {
        // Test sign with complex numbers
        let x = Array::from_data(
            vec![
                num_complex::Complex::new(-1.0, -1.0),
                num_complex::Complex::new(1.0, 1.0),
                num_complex::Complex::new(0.0, 2.0),
                num_complex::Complex::new(-1.0, 2.0),
            ],
            vec![2, 2],
        );
        let result = sign(&x).unwrap();
        assert_eq!(result.size(), 4);
        // Complex sign logic: sign = re/|re|
        let expected_re = num_complex::Complex::new(-1.0, 1.0);
        let expected_im = num_complex::Complex::new(1.0, 1.0);
        assert_eq!(result.get(0), Some(&expected_re));
        assert_eq!(result.get(1), Some(&expected_re));
        // Imaginary parts should have sign of real part
        assert_eq!(result.get(2), Some(&expected_im));
        assert_eq!(result.get(3), Some(&expected_im));
    }
}
