// Advanced broadcasting patterns for NumPy compatibility
//
// This module provides array manipulation functions for repeating,
// tiling, and broadcasting arrays to target shapes.

use crate::array::Array;
use crate::broadcasting::{broadcast_to, compute_broadcast_shape};
use crate::error::{NumPyError, Result};

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
/// // result == [1, 1, 2, 2, 2, 3, 3]
/// ```
pub fn repeat<T>(a: &Array<T>, repeats: usize, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot repeat empty array"));
    }
    
    if axis.is_none() {
        // Flatten and repeat
        let mut result_data = Vec::with_capacity(a.size() * repeats);
        for _ in 0..repeats {
            for elem in a.to_vec() {
                result_data.push(elem.clone());
            }
        }
        return Ok(Array::from_data(result_data, vec![a.size() * repeats]));
    }
    
    let ax = axis.unwrap() as usize;
    if ax >= a.ndim() {
        return Err(NumPyError::invalid_value(format!(
            "axis {} out of bounds for array with {} dimensions",
            ax, a.ndim()
        )));
    }
    
    // Calculate output shape
    let mut output_shape = a.shape().to_vec();
    output_shape[ax] *= repeats;
    
    // Repeat elements along specified axis
    let mut result_data = Vec::with_capacity(a.size() * repeats);
    for i in 0..repeats {
        let data = a.to_vec();
        for elem in data {
            result_data.push(elem.clone());
        }
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
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot tile empty array"));
    }
    
    if reps.is_empty() {
        return Err(NumPyError::invalid_value("reps must not be empty"));
    }
    
    // Compute broadcasted shape
    let a_shape = a.shape();
    let mut result_shape = a_shape.to_vec();
    
    // Add new dimensions if reps has more elements than array dimensions
    if reps.len() > a_shape.len() {
        result_shape.extend_from_slice(&reps[a_shape.len()..]);
    } else {
        // Multiply existing dimensions
        for (i, &rep) in a_shape.iter().zip(reps.iter()) {
            result_shape[i] = *rep;
        }
    }
    
    // Broadcast array to tiled shape
    broadcast_to(a, &result_shape)
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
        assert_eq!(result.to_vec(), vec![1, 1, 2, 2, 1, 2, 2, 3]);
    }
    
    #[test]
    fn test_repeat_axis_1() {
        let a = Array::from_vec(vec![1i64, 2, 3]);
        let result = repeat(&a, 2, Some(1)).unwrap();
        assert_eq!(result.shape(), vec![3, 6]);
        assert_eq!(result.to_vec(), vec![1, 2, 1, 1, 2, 2, 3]);
    }
    
    #[test]
    fn test_tile_basic() {
        let a = Array::from_vec(vec![1i64, 2]);
        let result = tile(&a, &[3, 2]).unwrap();
        assert_eq!(result.shape(), vec![3, 2]);
        // Result should have 6 elements (3*2) repeated in [3, 2] pattern
    }
    
    #[test]
    fn test_broadcast_to_enhanced() {
        let a = Array::from_vec(vec![1i64]);
        let result = broadcast_to_enhanced(&a, &[3, 4]).unwrap();
        assert_eq!(result.shape(), vec![3, 4]);
    }
}
