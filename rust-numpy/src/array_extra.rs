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

/// Clip (limit) the values in an array (similar to np.clip).
///
/// # Arguments
/// - `array`: Input array
/// - `min`: Minimum value (use None for no minimum)
/// - `max`: Maximum value (use None for no maximum)
///
/// # Returns
/// Array with values clipped to the specified range
pub fn clip<T>(array: &Array<T>, min: Option<T>, max: Option<T>) -> Result<Array<T>>
where
    T: Clone + Default + PartialOrd + 'static,
{
    let data = array.to_vec();

    let clipped: Vec<T> = data
        .iter()
        .map(|x| {
            let mut val = x.clone();
            if let Some(ref min_val) = min {
                if val < *min_val {
                    val = min_val.clone();
                }
            }
            if let Some(ref max_val) = max {
                if val > *max_val {
                    val = max_val.clone();
                }
            }
            val
        })
        .collect();

    Ok(Array::from_shape_vec(array.shape().to_vec(), clipped))
}

/// Round elements of the array to the given number of decimals (similar to np.round).
///
/// Uses NumPy-compatible rounding (banker's rounding: round half to even).
///
/// # Arguments
/// - `array`: Input array
/// - `decimals`: Number of decimal places to round to (default 0)
///
/// # Returns
/// Array with values rounded to the specified number of decimals
pub fn round<T>(array: &Array<T>, decimals: i32) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let factor = T::from(10.0_f64).unwrap().powi(decimals);

    let rounded: Vec<T> = array
        .to_vec()
        .iter()
        .map(|&x| {
            let scaled = x * factor.clone();
            let fract = scaled.fract();
            let whole_scaled = scaled.trunc();

            // Banker's rounding: round half to even
            let result = if fract.abs() == T::from(0.5_f64).unwrap() {
                // Exactly half - round to nearest even integer
                let whole_i64 = whole_scaled.to_i64().unwrap_or(0);
                if whole_i64 % 2 == 0 {
                    whole_scaled
                } else {
                    whole_scaled + T::from(whole_scaled.signum()).unwrap()
                }
            } else {
                scaled.round()
            };

            result / factor.clone()
        })
        .collect();

    Ok(Array::from_shape_vec(array.shape().to_vec(), rounded))
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

/// Extract a diagonal from a 2D array (similar to np.diagonal).
///
/// # Arguments
/// - `array`: Input array (must be at least 2D)
/// - `offset`: Diagonal offset (default 0)
///   - offset > 0: upper diagonals
///   - offset < 0: lower diagonals
///   - offset = 0: main diagonal
/// - `axis1`: First axis of diagonal (default 0)
/// - `axis2`: Second axis of diagonal (default 1)
///
/// # Returns
/// 1D array containing the specified diagonal
pub fn diagonal<T>(
    array: &Array<T>,
    offset: isize,
    axis1: usize,
    axis2: usize,
) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = array.ndim();

    if ndim < 2 {
        return Err(NumPyError::invalid_operation(
            "diagonal() requires array with at least 2 dimensions",
        ));
    }

    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumPyError::index_error(axis1.max(axis2), ndim));
    }

    if axis1 == axis2 {
        return Err(NumPyError::invalid_operation(
            "diagonal() requires axis1 and axis2 to be different",
        ));
    }

    // For 2D arrays, extract the diagonal directly
    if ndim == 2 {
        let rows = array.shape()[0];
        let cols = array.shape()[1];
        let data = array.to_vec();

        let mut diagonal_elements = Vec::new();

        // Calculate diagonal bounds based on offset
        let (row_start, col_start): (isize, isize) = if offset >= 0 {
            (0, offset)
        } else {
            (-offset, 0)
        };

        let row_start = row_start as usize;
        let col_start = col_start as usize;

        // Iterate along the diagonal
        let mut i = row_start;
        let mut j = col_start;

        while i < rows && j < cols {
            let idx = i * cols + j;
            if idx < data.len() {
                diagonal_elements.push(data[idx].clone());
            }
            i += 1;
            j += 1;
        }

        return Ok(Array::from_vec(diagonal_elements));
    }

    // For nD arrays, use axis1 and axis2
    let mut diagonal_shape = array.shape().to_vec();
    diagonal_shape.remove(axis1.max(axis2));
    diagonal_shape.remove(axis1.min(axis2));

    let dim1_size = array.shape()[axis1];
    let dim2_size = array.shape()[axis2];

    let mut diagonal_elements = Vec::new();

    let (start1, start2): (isize, isize) = if offset >= 0 {
        (0, offset)
    } else {
        (-offset, 0)
    };

    let start1 = start1 as usize;
    let start2 = start2 as usize;

    // Collect diagonal elements
    let mut d1 = start1;
    let mut d2 = start2;

    while d1 < dim1_size && d2 < dim2_size {
        let mut indices = vec![0; ndim];
        indices[axis1] = d1;
        indices[axis2] = d2;

        // Collect all elements for this diagonal position across other dimensions
        collect_diagonal_elements(array, &mut indices, axis1.min(axis2), 0, &mut diagonal_elements);

        d1 += 1;
        d2 += 1;
    }

    let final_shape = if diagonal_shape.is_empty() {
        vec![1]
    } else {
        diagonal_shape
    };

    Ok(Array::from_shape_vec(final_shape, diagonal_elements))
}

/// Helper function to collect elements along diagonal for nD arrays
fn collect_diagonal_elements<T>(
    array: &Array<T>,
    indices: &mut [usize],
    skip_axis: usize,
    current_axis: usize,
    result: &mut Vec<T>,
) where
    T: Clone + Default + 'static,
{
    if current_axis == indices.len() {
        let linear_idx = compute_linear_index_from_indices(array, indices);
        if let Some(elem) = array.get_linear(linear_idx) {
            result.push(elem.clone());
        }
        return;
    }

    if current_axis == skip_axis {
        collect_diagonal_elements(array, indices, skip_axis, current_axis + 1, result);
    } else {
        let dim_size = array.shape()[current_axis];
        for i in 0..dim_size {
            indices[current_axis] = i;
            collect_diagonal_elements(array, indices, skip_axis, current_axis + 1, result);
        }
    }
}

/// Compute linear index from multi-dimensional indices
fn compute_linear_index_from_indices<T>(array: &Array<T>, indices: &[usize]) -> usize
where
    T: Clone + Default + 'static,
{
    let strides = array.strides();
    let offset = array.offset;

    let mut linear_idx: isize = offset as isize;
    for (i, &idx) in indices.iter().enumerate() {
        linear_idx += strides[i] * idx as isize;
    }

    linear_idx as usize
}

/// Extract a diagonal or construct a diagonal array (similar to np.diag).
///
/// # Arguments
/// - `v`: Input array
///   - If 2D: extracts the k-th diagonal
///   - If 1D: constructs a 2D array with v on the k-th diagonal
/// - `k`: Diagonal offset (default 0)
///   - k > 0: upper diagonals
///   - k < 0: lower diagonals
///   - k = 0: main diagonal
///
/// # Returns
/// - If input is 2D: 1D array containing the k-th diagonal
/// - If input is 1D: 2D array with input on the k-th diagonal
pub fn diag<T>(v: &Array<T>, k: isize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = v.ndim();

    if ndim == 1 {
        // Construct 2D array with v on the k-th diagonal
        let n = v.shape()[0];
        let data = v.to_vec();

        let size = if k >= 0 {
            n + k as usize
        } else {
            n + (-k) as usize
        };

        let mut matrix = vec![T::default(); size * size];

        for (i, val) in data.iter().enumerate() {
            let (row, col) = if k >= 0 {
                (i, i + k as usize)
            } else {
                (i + (-k) as usize, i)
            };
            if row < size && col < size {
                matrix[row * size + col] = val.clone();
            }
        }

        Ok(Array::from_shape_vec(vec![size, size], matrix))
    } else if ndim == 2 {
        // Extract the k-th diagonal from 2D array
        diagonal(v, k, 0, 1)
    } else {
        Err(NumPyError::invalid_operation(
            "diag() requires 1D or 2D array",
        ))
    }
}

/// Construct an array from an index array and a list of arrays (similar to np.choose).
///
/// # Arguments
/// - `index`: Array of indices (must be same shape as choice arrays)
/// - `choices`: Slice of arrays to choose from
/// - `mode`: Mode for handling out-of-bounds indices
///   - "raise" (default): raise an error
///   - "wrap": wrap around using modulo
///   - "clip": clip to the valid range
///
/// # Returns
/// Array constructed from choices at index positions
pub fn choose<T>(index: &Array<i32>, choices: &[&Array<T>], mode: &str) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if choices.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one choice array",
        ));
    }

    let n_choices = choices.len();

    if !matches!(mode, "raise" | "wrap" | "clip") {
        return Err(NumPyError::invalid_operation(
            "choose() mode must be 'raise', 'wrap', or 'clip'",
        ));
    }

    let index_data = index.to_vec();
    let mut result = Vec::with_capacity(index_data.len());

    for (pos, &idx) in index_data.iter().enumerate() {
        // First, compute the adjusted choice index based on mode
        let choice_idx = if idx < 0 {
            match mode {
                "raise" => {
                    return Err(NumPyError::index_error(
                        (-idx) as usize,
                        n_choices,
                    ))
                }
                "wrap" => {
                    let mut i = idx % n_choices as i32;
                    if i < 0 {
                        i += n_choices as i32;
                    }
                    i as usize
                }
                "clip" => 0,
                _ => unreachable!(),
            }
        } else if (idx as usize) >= n_choices {
            match mode {
                "raise" => {
                    return Err(NumPyError::index_error(
                        idx as usize,
                        n_choices,
                    ))
                }
                "wrap" => idx as usize % n_choices,
                "clip" => n_choices - 1,
                _ => unreachable!(),
            }
        } else {
            idx as usize
        };

        // Now determine the element index
        let choice_array = choices[choice_idx];
        let choice_data = choice_array.to_vec();
        let element_idx = if mode == "wrap" || mode == "clip" {
            // For wrap/clip modes, use the (wrapped) index value for element selection
            let idx_for_element = if mode == "wrap" {
                // Apply same wrap logic to index for element selection
                let mut i = idx % choice_data.len() as i32;
                if i < 0 {
                    i += choice_data.len() as i32;
                }
                i as usize
            } else {
                // Clip mode: clamp to array bounds
                if idx < 0 {
                    0
                } else {
                    (idx as usize).min(choice_data.len().saturating_sub(1))
                }
            };
            idx_for_element
        } else {
            // For raise mode, use the position in the index array
            pos % choice_data.len().max(1)
        };
        result.push(choice_data[element_idx].clone());
    }

    Ok(Array::from_shape_vec(index.shape().to_vec(), result))
}

/// Return selected slices of an array along given axis (similar to np.compress).
///
/// # Arguments
/// - `condition`: Boolean or integer array used to select elements
/// - `array`: Input array
/// - `axis`: Axis along which to select (None for flattened selection)
///
/// # Returns
/// Array with selected elements
pub fn compress<T>(condition: &Array<bool>, array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let cond_data = condition.to_vec();
    let arr_data = array.to_vec();

    if let Some(ax) = axis {
        let ndim = array.ndim();
        if ndim == 0 {
            return Ok(array.clone());
        }

        let ax = if ax < 0 { ndim as isize + ax } else { ax } as usize;

        if ax >= ndim {
            return Err(NumPyError::index_error(ax, ndim));
        }

        // For 2D arrays with axis selection
        if ndim == 2 {
            let axis_len = array.shape()[ax];
            if cond_data.len() != axis_len {
                return Err(NumPyError::shape_mismatch(
                    vec![cond_data.len()],
                    vec![axis_len],
                ));
            }

            let rows = array.shape()[0];
            let cols = array.shape()[1];

            if ax == 0 {
                // Select rows
                let mut result = Vec::new();
                for (i, &keep) in cond_data.iter().enumerate() {
                    if keep && i < rows {
                        for j in 0..cols {
                            result.push(arr_data[i * cols + j].clone());
                        }
                    }
                }
                let kept_count = cond_data.iter().filter(|&&x| x).count();
                return Ok(Array::from_shape_vec(vec![kept_count, cols], result));
            } else {
                // Select columns
                let mut result = Vec::new();
                for i in 0..rows {
                    for (j, &keep) in cond_data.iter().enumerate() {
                        if keep && j < cols {
                            result.push(arr_data[i * cols + j].clone());
                        }
                    }
                }
                let kept_count = cond_data.iter().filter(|&&x| x).count();
                return Ok(Array::from_shape_vec(vec![rows, kept_count], result));
            }
        }

        // For higher dimensions, flatten and use condition
        let mut result = Vec::new();
        for (i, val) in arr_data.iter().enumerate() {
            if cond_data.get(i % cond_data.len()).copied().unwrap_or(false) {
                result.push(val.clone());
            }
        }
        return Ok(Array::from_vec(result));
    } else {
        // No axis specified - flatten and select
        if cond_data.len() != arr_data.len() {
            return Err(NumPyError::shape_mismatch(
                vec![cond_data.len()],
                vec![arr_data.len()],
            ));
        }

        let mut result = Vec::new();
        for (&cond, val) in cond_data.iter().zip(arr_data.iter()) {
            if cond {
                result.push(val.clone());
            }
        }

        Ok(Array::from_vec(result))
    }
}
