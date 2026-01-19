// Parallel operations for array processing using Rayon
//
// This module provides parallel implementations of reduction and binary operations
// for improved performance on multi-core systems.

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::error::NumPyError;

/// Parallel sum of array elements
#[cfg(feature = "rayon")]
pub fn parallel_sum<T>(array: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Add<Output = T> + std::iter::Sum + Send + Sync + 'static,
{
    let size = array.size();
    if size == 0 {
        return Err(NumPyError::invalid_value("Cannot sum empty array"));
    }

    let num_threads = rayon::current_num_threads();
    let _chunk_size = std::cmp::max(1024, size / (num_threads * 4));

    // Process chunks in parallel
    let chunk_sums: Vec<T> = array
        .to_vec()
        .par_chunks(1024)
        .map(|chunk| chunk.iter().cloned().sum())
        .collect();

    let mut result = T::default();
    for chunk_sum in chunk_sums {
        result = result + chunk_sum;
    }

    Ok(crate::Array::from_vec(vec![result]))
}

/// Parallel mean of array elements
#[cfg(feature = "rayon")]
pub fn parallel_mean<T>(array: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + Send
        + Sync
        + 'static
        + num_traits::cast::NumCast,
{
    let sum_result = parallel_sum(array)?;
    let count = array.size() as i64;

    Ok(crate::Array::from_vec(vec![
        sum_result.to_vec()[0].clone() / num_traits::cast::NumCast::from(count).unwrap(),
    ]))
}

/// Parallel binary addition
#[cfg(feature = "rayon")]
pub fn parallel_add<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    let result_data: Vec<T> = a_vec
        .par_iter()
        .zip(b_vec.par_iter())
        .map(|(a_elem, b_elem)| a_elem.clone() + b_elem.clone())
        .collect();

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

/// Parallel binary subtraction
#[cfg(feature = "rayon")]
pub fn parallel_sub<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Sub<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    let result_data: Vec<T> = a_vec
        .par_iter()
        .zip(b_vec.par_iter())
        .map(|(a_elem, b_elem)| a_elem.clone() - b_elem.clone())
        .collect();

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

/// Parallel element-wise multiplication
#[cfg(feature = "rayon")]
pub fn parallel_mul<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Mul<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    let result_data: Vec<T> = a_vec
        .par_iter()
        .zip(b_vec.par_iter())
        .map(|(a_elem, b_elem)| a_elem.clone() * b_elem.clone())
        .collect();

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

/// Parallel element-wise division
#[cfg(feature = "rayon")]
pub fn parallel_div<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Div<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    let result_data: Vec<T> = a_vec
        .par_iter()
        .zip(b_vec.par_iter())
        .map(|(a_elem, b_elem)| a_elem.clone() / b_elem.clone())
        .collect();

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

/// Fallback sequential operations when Rayon is not available
#[cfg(not(feature = "rayon"))]
pub fn parallel_sum<T>(array: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    let size = array.size();
    if size == 0 {
        return Err(NumPyError::invalid_value("Cannot sum empty array"));
    }

    let mut result = T::default();
    for i in 0..size {
        if let Some(val) = array.get(i) {
            result = result + val.clone();
        }
    }

    Ok(crate::Array::from_vec(vec![result]))
}

#[cfg(not(feature = "rayon"))]
pub fn parallel_mean<T>(array: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::NumCast,
{
    let size = array.size();
    if size == 0 {
        return Err(NumPyError::invalid_value(
            "Cannot compute mean of empty array",
        ));
    }

    let mut sum = T::default();
    for i in 0..size {
        if let Some(val) = array.get(i) {
            sum = sum + val.clone();
        }
    }

    let count: T = num_traits::cast::NumCast::from(size as i64)
        .ok_or_else(|| NumPyError::invalid_value("Failed to convert count"))?;

    Ok(crate::Array::from_vec(vec![sum / count]))
}

#[cfg(not(feature = "rayon"))]
pub fn parallel_add<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let mut result_data = Vec::with_capacity(size);
    for i in 0..size {
        if let (Some(a_val), Some(b_val)) = (a.get(i), b.get(i)) {
            result_data.push(a_val.clone() + b_val.clone());
        }
    }

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

#[cfg(not(feature = "rayon"))]
pub fn parallel_sub<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Sub<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let mut result_data = Vec::with_capacity(size);
    for i in 0..size {
        if let (Some(a_val), Some(b_val)) = (a.get(i), b.get(i)) {
            result_data.push(a_val.clone() - b_val.clone());
        }
    }

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

#[cfg(not(feature = "rayon"))]
pub fn parallel_mul<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Mul<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let mut result_data = Vec::with_capacity(size);
    for i in 0..size {
        if let (Some(a_val), Some(b_val)) = (a.get(i), b.get(i)) {
            result_data.push(a_val.clone() * b_val.clone());
        }
    }

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

#[cfg(not(feature = "rayon"))]
pub fn parallel_div<T>(
    a: &crate::Array<T>,
    b: &crate::Array<T>,
) -> Result<crate::Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Div<Output = T> + Send + Sync + 'static,
{
    let size = a.size();
    if size != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let mut result_data = Vec::with_capacity(size);
    for i in 0..size {
        if let (Some(a_val), Some(b_val)) = (a.get(i), b.get(i)) {
            result_data.push(a_val.clone() / b_val.clone());
        }
    }

    Ok(crate::Array::from_data(result_data, a.shape().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_sum() {
        let a = crate::Array::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let result = parallel_sum(&a).unwrap();
        assert_eq!(result.to_vec(), vec![55i64]);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_mean() {
        let a = crate::Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let result = parallel_mean(&a).unwrap();
        assert!((result.to_vec()[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_add() {
        let a = crate::Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = crate::Array::from_vec(vec![4.0, 5.0, 6.0]);
        let result = parallel_add(&a, &b).unwrap();
        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]);
    }
}
