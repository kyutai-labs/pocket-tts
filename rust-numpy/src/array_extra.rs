//! Additional array manipulation functions
//!
//! This module provides functions that are missing from the main
//! array_manipulation module but needed for NumPy compatibility.

use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::slicing::{MultiSlice, Slice};

/// Argument for split functions
pub enum SplitArg {
    /// Number of sections
    Count(usize),
    /// Indices to split at
    Indices(Vec<usize>),
}

impl From<usize> for SplitArg {
    fn from(n: usize) -> Self {
        SplitArg::Count(n)
    }
}

impl From<Vec<usize>> for SplitArg {
    fn from(indices: Vec<usize>) -> Self {
        SplitArg::Indices(indices)
    }
}

/// Concatenate arrays along an existing axis (similar to np.concatenate).
///
/// # Arguments
/// - `arrays`: Slice of arrays to concatenate
/// - `axis`: Axis along which to concatenate (default 0)
///
/// # Returns
/// Concatenated array
pub fn concatenate<T>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to concatenate",
        ));
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    if ndim == 0 {
        let mut all_data = Vec::new();
        for arr in arrays {
            all_data.extend_from_slice(&arr.to_vec());
        }
        return Ok(Array::from_vec(all_data));
    }

    let axis = if axis < 0 { ndim as isize + axis } else { axis } as usize;

    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} is out of bounds for {}-dimensional array",
            axis, ndim
        )));
    }

    for arr in arrays {
        if arr.ndim() != ndim {
            return Err(NumPyError::shape_mismatch(vec![ndim], vec![arr.ndim()]));
        }

        let mut arr_shape = arr.shape().to_vec();
        let mut first_shape = first.shape().to_vec();

        arr_shape.remove(axis);
        first_shape.remove(axis);

        if arr_shape != first_shape {
            return Err(NumPyError::shape_mismatch(arr_shape, first_shape));
        }
    }

    // Calculate output shape
    let mut output_shape = first.shape().to_vec();
    let mut total_axis_size = 0;
    for arr in arrays {
        total_axis_size += arr.shape()[axis];
    }
    output_shape[axis] = total_axis_size;

    let output_strides = crate::array::compute_strides(&output_shape);
    let output_size = output_shape.iter().product();
    let mut new_data = vec![T::default(); output_size];

    let mut current_axis_offset = 0;

    for arr in arrays {
        let arr_size = arr.size();
        let arr_shape = arr.shape();

        for i in 0..arr_size {
            if let Some(val) = arr.get_linear(i) {
                let mut coords = crate::strides::compute_multi_indices(i, arr_shape);
                coords[axis] += current_axis_offset;
                let out_linear = crate::strides::compute_linear_index(&coords, &output_strides);
                new_data[out_linear as usize] = val.clone();
            }
        }
        current_axis_offset += arr.shape()[axis];
    }

    Ok(Array::from_shape_vec(output_shape, new_data))
}

/// Stack arrays along a new axis.
pub fn stack<T>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to stack",
        ));
    }
    let first = arrays[0];
    let ndim = first.ndim();
    let shape = first.shape();

    for arr in arrays {
        if arr.shape() != shape {
            return Err(NumPyError::shape_mismatch(
                shape.to_vec(),
                arr.shape().to_vec(),
            ));
        }
    }

    // Normalize axis. The output has ndim + 1 dimensions.
    // For 0D arrays, ndim=0, output is 1D. Axis must be 0.
    let axis_limit = ndim + 1;
    let axis = if axis < 0 {
        axis + axis_limit as isize
    } else {
        axis
    };
    if axis < 0 || axis >= axis_limit as isize {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} out of bounds",
            axis
        )));
    }
    let axis = axis as usize;

    let mut new_shape = shape.to_vec();
    new_shape.insert(axis, 1);

    let mut reshaped_arrays = Vec::with_capacity(arrays.len());
    for arr in arrays {
        reshaped_arrays.push(arr.reshape(&new_shape)?);
    }

    let refs: Vec<&Array<T>> = reshaped_arrays.iter().collect();
    concatenate(&refs, axis as isize)
}

/// Vertically stack arrays (row-wise).
pub fn vstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to vstack",
        ));
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    if ndim == 1 {
        let mut rows = Vec::new();
        for arr in arrays {
            if arr.ndim() != 1 {
                return Err(NumPyError::shape_mismatch(vec![1], vec![arr.ndim()]));
            }
            rows.push((*arr).clone());
        }

        let mut total_length = 0;
        for arr in arrays {
            total_length += arr.shape()[0];
        }

        let mut all_data = Vec::new();
        for arr in arrays {
            all_data.extend_from_slice(&arr.to_vec());
        }

        return Ok(Array::from_shape_vec(
            vec![arrays.len(), total_length / arrays.len()],
            all_data,
        ));
    }

    concatenate(arrays, 0)
}

/// Horizontally stack arrays (column-wise).
pub fn hstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to hstack",
        ));
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    if ndim == 1 {
        let mut all_data = Vec::new();
        for arr in arrays {
            if arr.ndim() != 1 {
                return Err(NumPyError::shape_mismatch(vec![1], vec![arr.ndim()]));
            }
            all_data.extend_from_slice(&arr.to_vec());
        }
        return Ok(Array::from_vec(all_data));
    }

    concatenate(arrays, (ndim - 1) as isize)
}

/// Stack arrays in depth wise (along third axis).
pub fn dstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to dstack",
        ));
    }

    let mut reshaped_arrays = Vec::with_capacity(arrays.len());

    for arr in arrays {
        let ndim = arr.ndim();
        if ndim <= 2 {
            let shape = arr.shape();
            let mut new_shape = vec![1, 1, 1];
            if ndim == 1 {
                // (N,) -> (1, N, 1)
                new_shape[0] = 1;
                new_shape[1] = shape[0];
                new_shape[2] = 1;
            } else {
                // (M, N) -> (M, N, 1)
                new_shape[0] = shape[0];
                new_shape[1] = shape[1];
                new_shape[2] = 1;
            }
            reshaped_arrays.push(arr.reshape(&new_shape)?);
        } else {
            reshaped_arrays.push((*arr).clone());
        }
    }

    let refs: Vec<&Array<T>> = reshaped_arrays.iter().collect();
    concatenate(&refs, 2)
}

/// Split an array into multiple sub-arrays.
///
/// # Arguments
/// - `array`: Array to split
/// - `indices_or_sections`: If integer N, split into N roughly equal sections.
///    If vector of indices, split at those indices.
/// - `axis`: Axis along which to split.
pub fn array_split<T>(
    array: &Array<T>,
    indices_or_sections: SplitArg,
    axis: isize,
) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + 'static,
{
    let ndim = array.ndim();
    let axis = if axis < 0 { ndim as isize + axis } else { axis } as usize;

    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} is out of bounds for {}-dimensional array",
            axis, ndim
        )));
    }

    let dim_len = array.shape()[axis];

    let split_indices = match indices_or_sections {
        SplitArg::Count(n) => {
            if n == 0 {
                return Err(NumPyError::invalid_value(
                    "number sections must be larger than 0",
                ));
            }
            let section_len = dim_len / n;
            let extras = dim_len % n;
            let mut indices = Vec::with_capacity(n + 1);
            let mut curr = 0;
            indices.push(0);
            for i in 0..n {
                let size = section_len + if i < extras { 1 } else { 0 };
                curr += size;
                indices.push(curr);
            }
            indices
        }
        SplitArg::Indices(mut inds) => {
            let mut indices = Vec::with_capacity(inds.len() + 2);
            indices.push(0);
            indices.append(&mut inds);
            indices.push(dim_len);
            indices
        }
    };

    let mut results = Vec::with_capacity(split_indices.len() - 1);
    for i in 0..split_indices.len() - 1 {
        let start = split_indices[i];
        let end = split_indices[i + 1];

        // slice(axis, start, end)
        let mut slices = Vec::new();
        for d in 0..ndim {
            if d == axis {
                slices.push(Slice::Range(start as isize, end as isize));
            } else {
                slices.push(Slice::Full);
            }
        }
        results.push(array.slice(&MultiSlice::new(slices))?);
    }
    Ok(results)
}

/// Split an array into multiple sub-arrays of equal size.
pub fn split<T>(
    array: &Array<T>,
    indices_or_sections: SplitArg,
    axis: isize,
) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + 'static,
{
    let ndim = array.ndim();
    let axis_us = if axis < 0 { ndim as isize + axis } else { axis } as usize;
    if axis_us < ndim {
        if let SplitArg::Count(n) = indices_or_sections {
            let dim_len = array.shape()[axis_us];
            if n > 0 && dim_len % n != 0 {
                return Err(NumPyError::invalid_value(
                    "array split does not result in an equal division",
                ));
            }
        }
    }
    // Pass to array_split (it handles axis validation again)
    // Note: if indices_or_sections was moved/consumed, we need it again?
    // SplitArg is not Copy. But argument is passed by value.
    // We inspected it above. Wait, `if let SplitArg::Count(n) = indices_or_sections` consumes it?
    // No, matching reference `&indices_or_sections`?
    // The arguments assume ownership `SplitArg`.
    // I need to clone it or pass by reference.
    // Or just re-construct.
    // Let's implement logic without cloning.

    // Actually, `split` calls `array_split`, but needs to check condition first.
    // So I will match on reference.
    match &indices_or_sections {
        SplitArg::Count(n) => {
            if axis_us < ndim {
                let dim_len = array.shape()[axis_us];
                if *n > 0 && dim_len % n != 0 {
                    return Err(NumPyError::invalid_value(
                        "array split does not result in an equal division",
                    ));
                }
            }
        }
        _ => {}
    }
    array_split(array, indices_or_sections, axis)
}

/// Split array into multiple sub-arrays horizontally (column-wise).
pub fn hsplit<T>(array: &Array<T>, indices_or_sections: SplitArg) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + 'static,
{
    if array.ndim() == 0 {
        return Err(NumPyError::invalid_value(
            "hsplit only works on arrays of 1 or more dimensions",
        ));
    }
    if array.ndim() > 1 {
        split(array, indices_or_sections, 1)
    } else {
        split(array, indices_or_sections, 0)
    }
}

/// Split array into multiple sub-arrays vertically (row-wise).
pub fn vsplit<T>(array: &Array<T>, indices_or_sections: SplitArg) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + 'static,
{
    if array.ndim() < 2 {
        return Err(NumPyError::invalid_value(
            "vsplit only works on arrays of 2 or more dimensions",
        ));
    }
    split(array, indices_or_sections, 0)
}

/// Split array into multiple sub-arrays along the 3rd axis (depth).
pub fn dsplit<T>(array: &Array<T>, indices_or_sections: SplitArg) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + 'static,
{
    if array.ndim() < 3 {
        return Err(NumPyError::invalid_value(
            "dsplit only works on arrays of 3 or more dimensions",
        ));
    }
    split(array, indices_or_sections, 2)
}

/// Linear interpolation.
pub fn interp<T>(x: &Array<T>, xp: &Array<T>, fp: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    if xp.shape()[0] != fp.shape()[0] {
        return Err(NumPyError::invalid_operation(
            "fp and xp must have the same size",
        ));
    }
    let x_data = x.to_vec();
    let xp_data = xp.to_vec();
    let fp_data = fp.to_vec();
    let mut result = Vec::with_capacity(x_data.len());

    for &x_val in &x_data {
        if x_val <= xp_data[0] {
            result.push(fp_data[0]);
        } else if x_val >= xp_data[xp_data.len() - 1] {
            result.push(fp_data[fp_data.len() - 1]);
        } else {
            let mut i = 0;
            while i < xp_data.len() - 1 && xp_data[i + 1] < x_val {
                i += 1;
            }
            let x0 = xp_data[i];
            let x1 = xp_data[i + 1];
            let y0 = &fp_data[i];
            let y1 = &fp_data[i + 1];
            let t = (x_val - x0) / (x1 - x0);
            let interpolated = *y0 + t * (*y1 - *y0);
            result.push(interpolated);
        }
    }
    Ok(Array::from_vec(result))
}

/// Element-wise power.
pub fn power<T>(array: &Array<T>, exponent: T) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let data: Vec<T> = array.to_vec().iter().map(|&x| x.powf(exponent)).collect();
    Ok(Array::from_vec(data))
}

/// Trim the leading and/or trailing zeros from a 1-D array (similar to np.trim_zeros).
///
/// # Arguments
/// - `array`: Input 1-D array
/// - `trim`: Trim mode - "f" trim from front, "b" trim from back, "fb" trim from both (default)
///
/// # Returns
/// Array with zeros trimmed according to the specified mode
pub fn trim_zeros<T>(array: &Array<T>, trim: &str) -> Result<Array<T>>
where
    T: Clone + Default + PartialEq + num_traits::Zero + 'static,
{
    if array.ndim() != 1 {
        return Err(NumPyError::invalid_operation(
            "trim_zeros() only supports 1-D arrays",
        ));
    }

    if !matches!(trim, "f" | "b" | "fb") {
        return Err(NumPyError::invalid_operation(
            "trim_zeros() trim must be 'f', 'b', or 'fb'",
        ));
    }

    let data = array.to_vec();

    // Handle empty array or all zeros
    if data.is_empty() || data.iter().all(|x| x.is_zero()) {
        return Ok(Array::from_vec(vec![]));
    }

    let mut start = 0;
    let mut end = data.len();

    // Trim from front
    if trim == "f" || trim == "fb" {
        while start < data.len() && data[start].is_zero() {
            start += 1;
        }
    }

    // Trim from back
    if trim == "b" || trim == "fb" {
        while end > start && data[end - 1].is_zero() {
            end -= 1;
        }
    }

    Ok(Array::from_vec(data[start..end].to_vec()))
}

/// The differences between consecutive elements of an array (similar to np.ediff1d).
///
/// # Arguments
/// - `array`: Input array
/// - `to_end`: Optional values to append at the end of the returned differences
/// - `to_begin`: Optional values to prepend at the beginning of the returned differences
///
/// # Returns
/// 1-D array of differences with optional prepended/appended values
pub fn ediff1d<T>(
    array: &Array<T>,
    to_end: Option<&[T]>,
    to_begin: Option<&[T]>,
) -> Result<Array<T>>
where
    T: Clone + Default + std::ops::Sub<Output = T> + 'static,
{
    let data = array.to_vec();

    if data.is_empty() {
        let mut result = Vec::new();
        if let Some(begin) = to_begin {
            result.extend_from_slice(begin);
        }
        if let Some(end) = to_end {
            result.extend_from_slice(end);
        }
        return Ok(Array::from_vec(result));
    }

    // Compute differences
    let mut diffs = Vec::with_capacity(data.len().saturating_sub(1));
    for i in 0..data.len().saturating_sub(1) {
        diffs.push(data[i + 1].clone() - data[i].clone());
    }

    // Build final result
    let mut result = Vec::new();

    // Add to_begin values
    if let Some(begin) = to_begin {
        result.extend_from_slice(begin);
    }

    // Add differences
    result.extend_from_slice(&diffs);

    // Add to_end values
    if let Some(end) = to_end {
        result.extend_from_slice(end);
    }

    Ok(Array::from_vec(result))
}
