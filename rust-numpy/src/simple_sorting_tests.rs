// Simple test module for isolated sorting functionality
mod simple_tests {
    use super::*;

    // Create a simple test array that doesn't depend on the full array infrastructure
    fn create_test_array() -> Vec<i32> {
        vec![3, 1, 4, 1, 5, 9, 2, 6]
    }

#[cfg(test)]
mod simple_tests {
    use super::*;

    fn create_test_array() -> Array<i32> {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        Array::from_shape_vec(vec![data.len()], data)
    }

    #[test]
    fn test_sort_basic() {
        let mut array = create_test_array();
        let result = sort(&mut array, None, "quicksort", "asc").unwrap();
        let sorted = result.to_vec();
        assert_eq!(sorted, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_argsort_basic() {
        let array = create_test_array();
        let indices = argsort(&array, None, "quicksort", "asc").unwrap();
        assert_eq!(indices.to_vec(), vec![0, 1, 6, 0, 2, 5, 7, 4]);
    }

    #[test]
    fn test_searchsorted() {
        let sorted_data = vec![1, 2, 2, 3, 4, 5, 6, 9];
        let sorted_array = Array::from_shape_vec(vec![sorted_data.len()], sorted_data);
        let search_data = vec![2, 3, 3, 7];
        let search_array = Array::from_shape_vec(vec![search_data.len()], search_data);
        
        let indices_left = searchsorted(&sorted_array, &search_array, "left", None).unwrap();
        assert_eq!(indices_left.to_vec(), vec![1, 1, 1, 4, 5]);
        
        let indices_right = searchsorted(&sorted_array, &search_array, "right", None).unwrap();
        assert_eq!(indices_right.to_vec(), vec![2, 2, 2, 5, 6]);
    }

    #[test]
    fn test_extract() {
        let condition_data = vec![true, false, true, false];
        let condition = Array::from_shape_vec(vec![condition_data.len()], condition_data);
        let array_data = vec![1, 2, 3, 4];
        let array = Array::from_shape_vec(vec![array_data.len()], array_data);
        
        let result = extract(&condition, &array).unwrap();
        assert_eq!(result.to_vec(), vec![1, 3]);
    }

    #[test]
    fn test_count_nonzero() {
        let data = vec![0, 1, 0, 2, 0, 3];
        let array = Array::from_shape_vec(vec![data.len()], data);
        let count = count_nonzero(&array).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1, 5, 3, 9, 2, 6];
        let array = Array::from_shape_vec(vec![data.len()], data);
        let index = argmax(&array, None, None, false).unwrap();
        assert_eq!(index.to_vec()[0], 3);
    }

    #[test]
    fn test_argmin() {
        let data = vec![1, 5, 3, 9, 2, 6];
        let array = Array::from_shape_vec(vec![data.len()], data);
        let index = argmin(&array, None, None, false).unwrap();
        assert_eq!(index.to_vec()[0], 0);
    }
}