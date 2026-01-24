#[cfg(feature = "rayon")]
use numpy::parallel::ParArrayIter;
use numpy::Array;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "rayon")]
#[test]
fn test_par_iter_sum() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let array = Array::from_vec(data);

    let sum: i32 = array.par_iter().sum();
    assert_eq!(sum, 55);
}

#[cfg(feature = "rayon")]
#[test]
fn test_par_iter_map_reduce() {
    let array = Array::from_vec(vec![1, 2, 3, 4]);
    let squared_sum: i32 = array.par_iter().map(|&x| x * x).sum();
    assert_eq!(squared_sum, 1 + 4 + 9 + 16);
}

#[cfg(feature = "rayon")]
#[test]
#[should_panic(expected = "Parallel iteration currently only supported for C-contiguous arrays")]
fn test_par_iter_strided_panic() {
    // Create a strided array (slice with step > 1)
    let data = vec![1, 2, 3, 4, 5];
    let array = Array::from_vec(data);
    // Use slicing to create a view with strides
    // slice(start, stop, step) - assuming we have slicing implemented or manual stride manipulation

    // Low-level stride manipulation for testing
    let mut strided = array.clone();
    strided.shape = vec![3]; // length 3
    strided.strides = vec![2]; // stride 2 -> elements 0, 2, 4
    strided.offset = 0;

    // Note: The data backing this is still [1,2,3,4,5]
    // Indices: 0->0(1), 1->2(3), 2->4(5)

    // This should panic because strides!=1 (not C-contiguous by default definition for 1D unless stride=1)
    // 1D array is C-contiguous if stride=1.

    let _ = strided.par_iter();
}

#[test]
fn test_simd_module_exists() {
    // Just verify we can import from simd module
    #[allow(unused_imports)]
    use numpy::simd::SimdVector;
    // Nothing to test really as it's a trait, but ensures module is public
}
