use crate::array::Array;
// use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
pub use crate::strides::{compute_broadcast_shape, compute_broadcast_strides};

/// Broadcast arrays to common shape
pub fn broadcast_arrays<T>(arrays: &[&Array<T>]) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Ok(vec![]);
    }

    if arrays.len() == 1 {
        return Ok(vec![arrays[0].clone()]);
    }

    // Compute broadcast shape
    let mut broadcast_shape = arrays[0].shape().to_vec();

    for array in arrays.iter().skip(1) {
        broadcast_shape = compute_broadcast_shape(&broadcast_shape, array.shape());
    }

    // Broadcast each array
    let mut result = Vec::with_capacity(arrays.len());

    for array in arrays {
        let broadcasted = broadcast_to(array, &broadcast_shape)?;
        result.push(broadcasted);
    }

    Ok(result)
}

/// Broadcast single array to target shape
pub fn broadcast_to<T>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>> {
    let current_shape = array.shape();

    if current_shape == shape {
        return Ok(array.clone());
    }

    // Check if broadcasting is possible
    if !are_shapes_compatible(current_shape, shape) {
        return Err(NumPyError::broadcast_error(
            current_shape.to_vec(),
            shape.to_vec(),
        ));
    }

    // Compute new strides
    let new_strides = compute_broadcasted_strides(current_shape, array.strides(), shape);

    // Create new array sharing the same data
    Ok(Array {
        data: array.data.clone(),
        shape: shape.to_vec(),
        strides: new_strides,
        dtype: array.dtype.clone(),
        offset: array.offset,
    })
}

/// Check if shapes are compatible for broadcasting
pub fn are_shapes_compatible(shape1: &[usize], shape2: &[usize]) -> bool {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);

    for i in 0..max_len {
        let dim1 = if i >= max_len - len1 {
            shape1[i - (max_len - len1)]
        } else {
            1
        };

        let dim2 = if i >= max_len - len2 {
            shape2[i - (max_len - len2)]
        } else {
            1
        };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}

/// Compute broadcasted strides for an array
pub fn compute_broadcasted_strides(
    original_shape: &[usize],
    original_strides: &[isize],
    target_shape: &[usize],
) -> Vec<isize> {
    let orig_len = original_shape.len();
    let target_len = target_shape.len();
    let mut result = vec![0; target_len];

    for i in 0..target_len {
        if i >= target_len - orig_len {
            let orig_idx = i - (target_len - orig_len);
            let orig_dim = original_shape[orig_idx];

            if orig_dim == 1 {
                result[i] = 0; // Broadcast dimension
            } else {
                result[i] = original_strides[orig_idx];
            }
        } else {
            result[i] = 0; // New dimension being broadcast
        }
    }

    result
}

/// Broadcast shape for reduction operations
pub fn broadcast_shape_for_reduce(shape: &[usize], axis: &[isize], keepdims: bool) -> Vec<usize> {
    if axis.is_empty() {
        return if keepdims {
            vec![1; shape.len()]
        } else {
            vec![]
        };
    }

    let mut result = shape.to_vec();

    // Convert negative axes and standardise
    let mut normalized_axes: Vec<usize> = axis
        .iter()
        .map(|&ax| {
            if ax < 0 {
                (ax + shape.len() as isize) as usize
            } else {
                ax as usize
            }
        })
        .collect();

    // Sort descending to remove correctly
    normalized_axes.sort_by(|a, b| b.cmp(a));
    normalized_axes.dedup();

    for ax in normalized_axes {
        if ax < result.len() || keepdims {
            // For keepdims, we use original index if checking shape len?
            // Wait, result.len() == shape.len() if keepdims=true.
            // If keepdims=false, result shrinks.
            // If we sort descending, we remove highest indices first.

            if keepdims {
                if ax < result.len() {
                    result[ax] = 1;
                }
            } else {
                // If we remove, we must ensure ax is valid in CURRENT result?
                // If we sort descending, ax is valid because lower indices haven't changed.
                // But wait, ax refers to ORIGINAL shape index.
                // If we remove ax=2 (from [0,1,2]). Result [0,1].
                // Then remove ax=1. In Result [0,1], index 1 corresponds to original 1.
                // Yes, because 2 (>1) was removed.
                // So sorting descending works.
                if ax < result.len() {
                    result.remove(ax);
                }
            }
        }
    }

    result
}

/// Broadcast axes for operations
pub fn broadcast_axes(shape: &[usize], target_shape: &[usize]) -> Vec<isize> {
    let mut axes = Vec::new();
    let shape_len = shape.len();
    let target_len = target_shape.len();

    for i in 0..target_len {
        if i >= target_len - shape_len {
            let shape_idx = i - (target_len - shape_len);
            if shape[shape_idx] == 1 {
                axes.push(i as isize); // This dimension is broadcast
            }
        } else {
            axes.push(i as isize); // New dimension
        }
    }

    axes
}
