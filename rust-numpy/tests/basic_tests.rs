use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let arr = array![1, 2, 3, 4, 5];
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.size(), 5);
        assert_eq!(arr.ndim(), 1);
    }

    #[test]
    fn test_array_2d_creation() {
        let arr = array2![[1, 2, 3], [4, 5, 6]];
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.size(), 6);
        assert_eq!(arr.ndim(), 2);
    }

    #[test]
    fn test_array_zeros() {
        let arr = Array::<f64>::zeros(vec![3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.size(), 12);
    }

    #[test]
    fn test_array_ones() {
        let arr = Array::<i32>::ones(vec![2, 2]);
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.size(), 4);
    }

    #[test]
    fn test_array_full() {
        let arr = Array::<f64>::full(vec![2, 3], 7.0);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.size(), 6);
    }

    #[test]
    fn test_array_reshape() {
        let arr = array![1, 2, 3, 4];
        let reshaped = arr.reshape(&[2, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[2, 2]);
        assert_eq!(reshaped.size(), 4);
    }

    #[test]
    fn test_array_transpose() {
        let arr = array2![[1, 2], [3, 4]];
        let transposed = arr.transpose();
        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(transposed.size(), 4);
    }

    #[test]
    fn test_dtype_creation() {
        let dt = Dtype::from_str("int32").unwrap();
        assert_eq!(dt.kind(), DtypeKind::Integer);
        assert_eq!(dt.itemsize(), 4);
    }

    #[test]
    fn test_dtype_from_type() {
        let dt_f64 = Dtype::from_type::<f64>();
        assert_eq!(dt_f64.kind(), DtypeKind::Float);
        assert_eq!(dt_f64.itemsize(), 8);

        let dt_i32 = Dtype::from_type::<i32>();
        assert_eq!(dt_i32.kind(), DtypeKind::Integer);
        assert_eq!(dt_i32.itemsize(), 4);
    }

    #[test]
    fn test_strides_computation() {
        let shape = vec![2, 3, 4];
        let strides = numpy::strides::compute_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_broadcast_shape() {
        let shape1 = vec![2, 1, 3];
        let shape2 = vec![1, 4, 1];
        let broadcast_shape = numpy::broadcasting::compute_broadcast_shape(&shape1, &shape2);
        assert_eq!(broadcast_shape, vec![2, 4, 3]);
    }

    #[test]
    fn test_constants() {
        assert!((numpy::constants::PI - 3.141592653589793).abs() < 1e-15);
        assert_eq!(numpy::constants::dtype::INT32_MAX, 2147483647);
        assert!(numpy::constants::float::EPSILON > 0.0);
        assert!(numpy::constants::utils::is_nan(numpy::constants::NAN));
        assert!(numpy::constants::utils::is_infinite(numpy::constants::INF));
    }

    #[test]
    fn test_error_creation() {
        let err = NumPyError::shape_mismatch(vec![2, 3], vec![3, 2]);
        match err {
            NumPyError::ShapeMismatch { expected, actual } => {
                assert_eq!(expected, vec![2, 3]);
                assert_eq!(actual, vec![3, 2]);
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_ufunc_registry() {
        use numpy::ufunc::{get_ufunc, UfuncRegistry};

        let _registry = UfuncRegistry::new();
        let add_ufunc = get_ufunc("add");
        assert!(add_ufunc.is_some());

        let nonexistent = get_ufunc("nonexistent_function");
        assert!(nonexistent.is_none());
    }

    #[test]
    fn test_memory_manager() {
        use numpy::memory::MemoryManager;

        let data = vec![1, 2, 3, 4, 5];
        let manager = MemoryManager::from_vec(data);
        assert_eq!(manager.len(), 5);
        assert_eq!(manager.get(2), Some(&3));
    }

    #[test]
    fn test_version_info() {
        assert!(!numpy::VERSION.is_empty());
        assert!(numpy::VERSION.contains('.'));
    }

    #[test]
    fn test_basic_arithmetic() {
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];

        // This would require ufunc implementation
        // let sum = a.add(&b); // Would be [5, 7, 9]

        // For now, just test basic functionality
        assert_eq!(a.size(), 3);
        assert_eq!(b.size(), 3);
    }

    #[test]
    fn test_unique_basic() {
        let arr = array![3, 1, 2, 1, 3];
        let result = numpy::set_ops::unique(&arr, false, false, false, None).unwrap();
        assert_eq!(result.values.to_vec(), vec![1, 2, 3]);
        assert!(result.indices.is_none());
        assert!(result.inverse.is_none());
        assert!(result.counts.is_none());
    }

    #[test]
    fn test_unique_indices_inverse_counts() {
        let arr = array![3, 1, 2, 1, 3];
        let result = numpy::set_ops::unique(&arr, true, true, true, None).unwrap();

        assert_eq!(result.values.to_vec(), vec![1, 2, 3]);
        assert_eq!(result.indices.unwrap().to_vec(), vec![1, 2, 0]);
        assert_eq!(result.inverse.unwrap().to_vec(), vec![2, 0, 1, 0, 2]);
        assert_eq!(result.counts.unwrap().to_vec(), vec![2, 1, 2]);
    }

    #[test]
    fn test_unique_axis_rows() {
        let data = vec![1, 2, 1, 2, 3, 4];
        let arr = Array::from_shape_vec(vec![3, 2], data);

        let result = numpy::set_ops::unique(&arr, true, true, true, Some(&[0])).unwrap();
        assert_eq!(result.values.shape(), &[2, 2]);
        assert_eq!(result.values.to_vec(), vec![1, 2, 3, 4]);
        assert_eq!(result.indices.unwrap().to_vec(), vec![0, 2]);
        assert_eq!(result.inverse.unwrap().to_vec(), vec![0, 0, 1]);
        assert_eq!(result.counts.unwrap().to_vec(), vec![2, 1]);
    }
}
