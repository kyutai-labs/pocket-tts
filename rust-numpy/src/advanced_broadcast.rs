// Advanced broadcasting patterns for NumPy compatibility
//
// This module provides array manipulation functions for repeating,
// tiling, and broadcasting arrays to target shapes.

use crate::array::Array;
use crate::broadcasting::broadcast_to;
use crate::error::{NumPyError, Result};
use crate::strides::{compute_linear_index, compute_multi_indices, compute_strides};

/// Repeat array along a given axis
///
/// Arguments:
/// - a: Input array
/// - repeats: Number of times to repeat each element
/// - axis: Axis along which to repeat (0 means flatten and repeat, 1 means repeat along first axis, etc.)
///
/// Returns: Array with repeated elements
///
/// Examples:
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = repeat(&a, 2, Some(0)).unwrap();
/// // result == [1, 1, 2, 2, 3, 3]
/// ```
pub fn repeat<T>(a: &Array<T>, repeats: usize, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if axis.is_none() {
        let size = a.size();
        let mut result_data = Vec::with_capacity(size * repeats);
        for elem in a.to_vec() {
            for _ in 0..repeats {
                result_data.push(elem.clone());
            }
        }
        return Ok(Array::from_data(result_data, vec![size * repeats]));
    }

    let ax = axis.unwrap();
    let ndim = a.ndim();
    let ax = if ax < 0 { ax + ndim as isize } else { ax } as usize;
    if ax >= ndim {
        return Err(NumPyError::index_error(ax, ndim));
    }

    let mut output_shape = a.shape().to_vec();
    output_shape[ax] *= repeats;
    let output_size: usize = output_shape.iter().product();
    let mut result_data = Vec::with_capacity(output_size);
    let source_shape = a.shape();
    let source_strides = compute_strides(source_shape);

    for linear in 0..output_size {
        let mut out_indices = compute_multi_indices(linear, &output_shape);
        if repeats > 0 {
            out_indices[ax] /= repeats;
        }
        let source_linear = compute_linear_index(&out_indices, &source_strides) as usize;
        let value = a
            .get_linear(source_linear)
            .ok_or_else(|| NumPyError::index_error(source_linear, a.size()))?;
        result_data.push(value.clone());
    }

    Ok(Array::from_data(result_data, output_shape))
}

/// Tile array by repeating it
///
/// Arguments:
/// - a: Input array
/// - reps: Number of repetitions per dimension
///
/// Returns: Tiled array
///
/// Examples:
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2]);
/// let result = tile(&a, &[3, 2]).unwrap();
/// // result shape is [3, 2, 2] with 6 elements
/// ```
pub fn tile<T>(a: &Array<T>, reps: &[usize]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if reps.is_empty() {
        return Err(NumPyError::invalid_value("reps must not be empty"));
    }

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let max_len = std::cmp::max(ndim, reps.len());
    let mut shape_full = vec![1; max_len];
    let mut reps_full = vec![1; max_len];

    for (i, &dim) in a_shape.iter().enumerate() {
        shape_full[max_len - ndim + i] = dim;
    }
    for (i, &rep) in reps.iter().enumerate() {
        reps_full[max_len - reps.len() + i] = rep;
    }

    let mut output_shape = Vec::with_capacity(max_len);
    for (dim, rep) in shape_full.iter().zip(reps_full.iter()) {
        output_shape.push(dim * rep);
    }

    let output_size: usize = output_shape.iter().product();
    let mut result_data = Vec::with_capacity(output_size);
    let source_strides = compute_strides(a_shape);

    for linear in 0..output_size {
        let out_indices = compute_multi_indices(linear, &output_shape);
        let mut source_indices_full = Vec::with_capacity(max_len);
        for (idx, dim) in out_indices.iter().zip(shape_full.iter()) {
            source_indices_full.push(if *dim == 0 { 0 } else { idx % dim });
        }

        if ndim == 0 {
            let value = a
                .get_linear(0)
                .ok_or_else(|| NumPyError::index_error(0, a.size()))?;
            result_data.push(value.clone());
            continue;
        }

        let source_indices = &source_indices_full[max_len - ndim..];
        let source_linear = compute_linear_index(source_indices, &source_strides) as usize;
        let value = a
            .get_linear(source_linear)
            .ok_or_else(|| NumPyError::index_error(source_linear, a.size()))?;
        result_data.push(value.clone());
    }

    Ok(Array::from_data(result_data, output_shape))
}

/// Broadcast array to specific shape
///
/// This function extends the broadcast_to functionality to support
/// arbitrary target shapes, similar to numpy.broadcast_to.
///
/// Arguments:
/// - array: Input array
/// - shape: Target shape to broadcast to
///
/// Returns: Broadcasted array
///
/// Examples:
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = broadcast_to(&a, &[3, 4]).unwrap();
/// // result shape is [3, 4] with repeated pattern
/// ```
pub fn broadcast_to_enhanced<T>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    // Delegate to existing broadcast_to function
    // This is a convenience wrapper with NumPy-compatible name
    broadcast_to(array, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_axis_0() {
        let a = Array::from_vec(vec![1i64, 2, 3]);
        let result = repeat(&a, 2, Some(0)).unwrap();
        assert_eq!(result.shape(), vec![6]);
        assert_eq!(result.to_vec(), vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn test_repeat_axis_1() {
        let a = Array::from_vec(vec![1i64, 2, 3]);
        let result = repeat(&a, 2, Some(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_tile_basic() {
        let a = Array::from_vec(vec![1i64, 2]);
        let result = tile(&a, &[3, 2]).unwrap();
        assert_eq!(result.shape(), vec![3, 4]);
        assert_eq!(result.to_vec(), vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    }

    #[test]
    fn test_broadcast_to_enhanced() {
        let a = Array::from_vec(vec![1i64]);
        let result = broadcast_to_enhanced(&a, &[3, 4]).unwrap();
        assert_eq!(result.shape(), vec![3, 4]);
    }
}
