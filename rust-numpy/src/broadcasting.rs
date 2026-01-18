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
pub fn broadcast_to<T>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
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

    // Create output array with broadcasted shape
    let mut output = Array::zeros(shape.to_vec());

    // Copy data with broadcasting
    broadcast_copy(array, &mut output)?;

    Ok(output)
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

/// Copy data with broadcasting
fn broadcast_copy<T>(src: &Array<T>, dst: &mut Array<T>) -> Result<()>
where
    T: Clone + Copy + Default + 'static,
{
    let src_shape = src.shape();
    let dst_shape = dst.shape();
    
    if src_shape.is_empty() {
        // Scalar broadcasting - use Copy trait to avoid cloning
        if let Some(scalar) = src.get(0) {
            for i in 0..dst.size() {
                dst.set(i, *scalar)?;
            }
        }
        return Ok(());
    }
    
    if src_shape == dst_shape {
        // Simple copy
        return copy_array(src, dst);
    }
    
    // General broadcasting case
    broadcast_general(src, dst)
}

/// Simple array copy
fn copy_array<T>(src: &Array<T>, dst: &mut Array<T>) -> Result<()>
where
    T: Clone + Default + 'static,
{
    if src.size() != dst.size() {
        return Err(NumPyError::shape_mismatch(
            src.shape().to_vec(),
            dst.shape().to_vec(),
        ));
    }

    for i in 0..src.size() {
        if let Some(element) = src.get(i) {
            dst.set(i, element.clone())?;
        }
    }

    Ok(())
}

/// General broadcasting implementation
fn broadcast_general<T>(src: &Array<T>, dst: &mut Array<T>) -> Result<()>
where
    T: Clone + Default + 'static,
{
    let src_shape = src.shape();
    let dst_shape = dst.shape().to_vec();

    // Iterate over destination array
    let mut dst_indices = vec![0; dst_shape.len()];

    for flat_idx in 0..dst.size() {
        // Compute source indices for this destination index
        let src_indices = compute_source_indices(&dst_indices, src_shape, &dst_shape);

        // Copy element from source
        if let Some(element) = get_element_by_indices(src, &src_indices) {
            // Set element in destination
            dst.set(flat_idx, element.clone())?;
        }

        // Increment destination indices
        increment_indices(&mut dst_indices, &dst_shape);
    }

    Ok(())
}

/// Compute source indices for destination index
fn compute_source_indices(
    dst_indices: &[usize],
    src_shape: &[usize],
    dst_shape: &[usize],
) -> Vec<usize> {
    let mut src_indices = Vec::with_capacity(src_shape.len());

    for (i, &src_dim) in src_shape.iter().enumerate() {
        if src_dim == 1 {
            src_indices.push(0); // Broadcast dimension
        } else {
            let dst_dim_idx = dst_shape.len() - src_shape.len() + i;
            src_indices.push(dst_indices[dst_dim_idx]);
        }
    }

    src_indices
}

/// Get element by multi-dimensional indices
fn get_element_by_indices<'a, T>(array: &'a Array<T>, indices: &[usize]) -> Option<&'a T> {
    if indices.len() != array.ndim() {
        return None;
    }

    // Compute linear index from indices
    let mut linear_idx = 0;
    let strides = array.strides();

    for (i, &index) in indices.iter().enumerate() {
        if i < strides.len() {
            linear_idx += index * strides[i] as usize;
        }
    }

    array.get(linear_idx)
}

/// Increment multi-dimensional indices
fn increment_indices(indices: &mut [usize], shape: &[usize]) {
    if indices.is_empty() {
        return;
    }

    let mut i = indices.len() - 1;

    // Increment last dimension
    indices[i] += 1;

    // Handle carry-over
    while i > 0 && indices[i] >= shape[i] {
        indices[i] = 0;
        i -= 1;
        indices[i] += 1;
    }
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

    for &ax in axis {
        let ax = if ax < 0 {
            ax + shape.len() as isize
        } else {
            ax
        } as usize;

        if ax < result.len() {
            if keepdims {
                result[ax] = 1;
            } else {
                result.remove(ax);
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
