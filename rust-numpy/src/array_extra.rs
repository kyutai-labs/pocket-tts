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
pub fn concatenate<T>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "concatenate: must provide at least one array",
        ));
    }

    let ndim = arrays[0].ndim();
    let axis = if axis < 0 { ndim as isize + axis } else { axis } as usize;

    if axis >= ndim {
        return Err(NumPyError::index_error(axis, ndim));
    }

    let mut target_shape = arrays[0].shape().to_vec();
    let mut total_axis_len = target_shape[axis];

    for arr in arrays.iter().skip(1) {
        if arr.ndim() != ndim {
            return Err(NumPyError::shape_mismatch(
                arr.shape().to_vec(),
                target_shape,
            ));
        }
        for (i, (&d1, &d2)) in arr.shape().iter().zip(target_shape.iter()).enumerate() {
            if i != axis && d1 != d2 {
                return Err(NumPyError::shape_mismatch(
                    arr.shape().to_vec(),
                    target_shape,
                ));
            }
        }
        total_axis_len += arr.shape()[axis];
    }

    target_shape[axis] = total_axis_len;
    let total_size: usize = target_shape.iter().product();

    let mut reordered_data = vec![T::default(); total_size];
    let mut current_axis_start = 0;

    for arr in arrays {
        let shape = arr.shape();
        let axis_len = shape[axis];
        let outer_size: usize = shape[0..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();

        for i in 0..outer_size {
            for j in 0..axis_len {
                for k in 0..inner_size {
                    let src_idx = (i * axis_len + j) * inner_size + k;
                    let dst_axis_idx = current_axis_start + j;
                    let dst_idx = (i * total_axis_len + dst_axis_idx) * inner_size + k;

                    if let Some(val) = arr.get_linear(src_idx) {
                        reordered_data[dst_idx] = val.clone();
                    }
                }
            }
        }
        current_axis_start += axis_len;
    }

    Ok(Array::from_shape_vec(target_shape, reordered_data))
}

/// Stack arrays along a new axis.
pub fn stack<T>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "stack: must provide at least one array",
        ));
    }

    let ndim = arrays[0].ndim();
    let axis = if axis < 0 {
        (ndim + 1) as isize + axis
    } else {
        axis
    } as usize;

    let mut expanded = Vec::new();
    for arr in arrays {
        let mut new_shape = arr.shape().to_vec();
        new_shape.insert(axis, 1);
        expanded.push(arr.reshape(&new_shape)?);
    }

    let refs: Vec<&Array<T>> = expanded.iter().collect();
    concatenate(&refs, axis as isize)
}

/// Vertically stack arrays (row-wise).
pub fn vstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let mut promoted = Vec::new();
    for arr in arrays {
        if arr.ndim() < 2 {
            let mut new_shape = vec![1];
            new_shape.extend_from_slice(arr.shape());
            promoted.push(arr.reshape(&new_shape)?);
        } else {
            promoted.push((*arr).clone());
        }
    }
    let refs: Vec<&Array<T>> = promoted.iter().collect();
    concatenate(&refs, 0)
}

/// Horizontally stack arrays (column-wise).
pub fn hstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let mut promoted = Vec::new();
    for arr in arrays {
        promoted.push((*arr).clone());
    }
    let refs: Vec<&Array<T>> = promoted.iter().collect();
    let axis = if !arrays.is_empty() && arrays[0].ndim() == 1 {
        0
    } else {
        1
    };
    concatenate(&refs, axis)
}

/// Stack arrays in depth wise (along third axis).
pub fn dstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let mut promoted = Vec::new();
    for arr in arrays {
        let ndim = arr.ndim();
        if ndim == 0 {
            promoted.push(arr.reshape(&[1, 1, 1])?);
        } else if ndim == 1 {
            promoted.push(arr.reshape(&[1, arr.shape()[0], 1])?);
        } else if ndim == 2 {
            let mut new_shape = arr.shape().to_vec();
            new_shape.push(1);
            promoted.push(arr.reshape(&new_shape)?);
        } else {
            promoted.push((*arr).clone());
        }
    }
    let refs: Vec<&Array<T>> = promoted.iter().collect();
    concatenate(&refs, 2)
}

/// Split an array into multiple sub-arrays.
pub fn array_split<T>(
    array: &Array<T>,
    indices_or_sections: SplitArg,
    axis: isize,
) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let ndim = array.ndim();
    let axis_idx = normalize_axis(axis, ndim)?;
    let axis_len = array.shape()[axis_idx];

    let mut offsets = Vec::new();
    match indices_or_sections {
        SplitArg::Count(n) => {
            let size = axis_len / n;
            let extras = axis_len % n;
            let mut curr = 0;
            for i in 0..n {
                let len = size + if i < extras { 1 } else { 0 };
                offsets.push((curr, curr + len));
                curr += len;
            }
        }
        SplitArg::Indices(indices) => {
            let mut curr = 0;
            for idx in indices {
                offsets.push((curr, idx));
                curr = idx;
            }
            offsets.push((curr, axis_len));
        }
    }

    let mut results = Vec::new();
    for (start, end) in offsets {
        let mut slices = vec![Slice::Full; ndim];
        slices[axis_idx] = Slice::Range(start as isize, end as isize);
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
    T: Clone + Default + Send + Sync + 'static,
{
    if let SplitArg::Count(n) = indices_or_sections {
        let axis_idx = normalize_axis(axis, array.ndim())?;
        if !array.shape()[axis_idx].is_multiple_of(n) {
            return Err(NumPyError::invalid_value(
                "array split does not result in an equal division",
            ));
        }
    }
    array_split(array, indices_or_sections, axis)
}

/// Split array into multiple sub-arrays horizontally (column-wise).
pub fn hsplit<T>(array: &Array<T>, indices_or_sections: SplitArg) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let axis = if array.ndim() < 2 { 0 } else { 1 };
    array_split(array, indices_or_sections, axis)
}

/// Split array into multiple sub-arrays vertically (row-wise).
pub fn vsplit<T>(array: &Array<T>, indices_or_sections: SplitArg) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    array_split(array, indices_or_sections, 0)
}

/// Split array into multiple sub-arrays along the 3rd axis (depth).
pub fn dsplit<T>(array: &Array<T>, indices_or_sections: SplitArg) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    array_split(array, indices_or_sections, 2)
}

/// Linear interpolation.
pub fn interp<T>(x: &Array<T>, xp: &Array<T>, fp: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + PartialOrd + num_traits::Float + Send + Sync + 'static,
{
    let mut result = Vec::with_capacity(x.size());
    for i in 0..x.size() {
        let val = x.get_linear(i).cloned().unwrap_or_default();
        if val <= *xp.get_linear(0).unwrap() {
            result.push(fp.get_linear(0).cloned().unwrap_or_default());
            continue;
        }
        if val >= *xp.get_linear(xp.size() - 1).unwrap() {
            result.push(fp.get_linear(fp.size() - 1).cloned().unwrap_or_default());
            continue;
        }
        let mut found = false;
        for j in 0..xp.size() - 1 {
            let x0 = *xp.get_linear(j).unwrap();
            let x1 = *xp.get_linear(j + 1).unwrap();
            if val >= x0 && val <= x1 {
                let f0 = *fp.get_linear(j).unwrap();
                let f1 = *fp.get_linear(j + 1).unwrap();
                result.push(f0 + (f1 - f0) * (val - x0) / (x1 - x0));
                found = true;
                break;
            }
        }
        if !found {
            result.push(T::nan());
        }
    }
    Ok(Array::from_data(result, x.shape().to_vec()))
}

/// Clip values to be within a specified range.
pub fn clip<T>(array: &Array<T>, min: Option<T>, max: Option<T>) -> Result<Array<T>>
where
    T: Clone + Default + PartialOrd + Send + Sync + 'static,
{
    let mut data = Vec::with_capacity(array.size());
    for i in 0..array.size() {
        let mut val = array.get_linear(i).cloned().unwrap_or_default();
        if let Some(ref m) = min {
            if val < *m {
                val = m.clone();
            }
        }
        if let Some(ref m) = max {
            if val > *m {
                val = m.clone();
            }
        }
        data.push(val);
    }
    Ok(Array::from_data(data, array.shape().to_vec()))
}

/// Trim the leading and/or trailing zeros from a 1-D array.
pub fn trim_zeros<T>(array: &Array<T>, trim: &str) -> Result<Array<T>>
where
    T: Clone + Default + PartialEq + Send + Sync + 'static,
{
    let data = array.data();
    let mut start = 0;
    let mut end = data.len();
    if trim.contains('f') {
        while start < end && data[start] == T::default() {
            start += 1;
        }
    }
    if trim.contains('b') {
        while end > start && data[end - 1] == T::default() {
            end -= 1;
        }
    }
    Ok(Array::from_vec(data[start..end].to_vec()))
}

/// The differences between consecutive elements of an array.
pub fn ediff1d<T>(
    array: &Array<T>,
    to_end: Option<&[T]>,
    to_begin: Option<&[T]>,
) -> Result<Array<T>>
where
    T: Clone + Default + std::ops::Sub<Output = T> + Send + Sync + 'static,
{
    let size = array.size();
    let mut diffs = Vec::new();
    if size >= 2 {
        for i in 0..size - 1 {
            let v1 = array.get_linear(i).unwrap();
            let v2 = array.get_linear(i + 1).unwrap();
            diffs.push(v2.clone() - v1.clone());
        }
    }
    let mut result = Vec::new();
    if let Some(tb) = to_begin {
        result.extend_from_slice(tb);
    }
    result.extend(diffs);
    if let Some(te) = to_end {
        result.extend_from_slice(te);
    }
    Ok(Array::from_vec(result))
}

/// Extract a diagonal from an array.
pub fn diagonal<T>(array: &Array<T>, offset: isize, axis1: usize, axis2: usize) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let ndim = array.ndim();
    if ndim < 2 {
        return Err(NumPyError::invalid_value(
            "diagonal requires at least 2D array",
        ));
    }
    let axis1 = normalize_axis(axis1 as isize, ndim)?;
    let axis2 = normalize_axis(axis2 as isize, ndim)?;
    let dim1 = array.shape()[axis1];
    let dim2 = array.shape()[axis2];
    let mut diag = Vec::new();
    let (mut r, mut c) = if offset >= 0 {
        (0, offset as usize)
    } else {
        ((-offset) as usize, 0)
    };
    while r < dim1 && c < dim2 {
        let mut idx = vec![0; ndim];
        idx[axis1] = r;
        idx[axis2] = c;
        diag.push(array.get_multi(&idx)?);
        r += 1;
        c += 1;
    }
    Ok(Array::from_vec(diag))
}

/// Extract a diagonal or construct a diagonal array.
<<<<<<< HEAD
=======
///
/// If `array` is 2-D, return the diagonal elements (main or offset diagonal).
/// If `array` is 1-D, return a 2-D array with the input as diagonal.
>>>>>>> origin/main
pub fn diag<T>(array: &Array<T>, k: isize) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let ndim = array.ndim();
    if ndim == 1 {
<<<<<<< HEAD
=======
        // Construct a 2D array with array on the k-th diagonal
>>>>>>> origin/main
        let n = array.shape()[0];
        let size = n + k.unsigned_abs();
        let mut data = vec![T::default(); size * size];
        for i in 0..n {
            let (row, col) = if k >= 0 {
                (i, i + k as usize)
            } else {
                (i + (-k) as usize, i)
            };
            data[row * size + col] = array.get_linear(i).cloned().unwrap_or_default();
        }
        Ok(Array::from_shape_vec(vec![size, size], data))
    } else if ndim == 2 {
<<<<<<< HEAD
=======
        // Extract the k-th diagonal
>>>>>>> origin/main
        diagonal(array, k, 0, 1)
    } else {
        Err(NumPyError::invalid_value("diag requires 1D or 2D array"))
    }
}

/// Return the upper triangle of an array.
pub fn triu<T>(array: &Array<T>, k: isize) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let ndim = array.ndim();
    let mut result = Array::from_data(array.to_vec(), array.shape().to_vec());
    let shape = array.shape();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let outer_size: usize = shape[..ndim - 2].iter().product();

    for i in 0..outer_size {
        for r in 0..rows {
            for c in 0..cols {
                if (c as isize) < (r as isize) + k {
                    let idx = (i * rows + r) * cols + c;
                    result.set(idx, T::default())?;
                }
            }
        }
    }
    Ok(result)
}

/// Return the lower triangle of an array.
pub fn tril<T>(array: &Array<T>, k: isize) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let ndim = array.ndim();
    let mut result = Array::from_data(array.to_vec(), array.shape().to_vec());
    let shape = array.shape();
    let rows = shape[ndim - 2];
    let cols = shape[ndim - 1];
    let outer_size: usize = shape[..ndim - 2].iter().product();

    for i in 0..outer_size {
        for r in 0..rows {
            for c in 0..cols {
                if (c as isize) > (r as isize) + k {
                    let idx = (i * rows + r) * cols + c;
                    result.set(idx, T::default())?;
                }
            }
        }
    }
    Ok(result)
}

/// Stack 1-D arrays as columns into a 2-D array.
<<<<<<< HEAD
=======
///
/// Takes a sequence of 1-D arrays and stacks them as columns to make a single 2-D array.
/// 2-D arrays are stacked as-is, just like with hstack.
>>>>>>> origin/main
pub fn column_stack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "column_stack: must provide at least one array",
        ));
    }

    let mut promoted = Vec::new();
    for arr in arrays {
        if arr.ndim() == 1 {
<<<<<<< HEAD
=======
            // Reshape 1-D to (n, 1)
>>>>>>> origin/main
            let n = arr.shape()[0];
            promoted.push(arr.reshape(&[n, 1])?);
        } else {
            promoted.push((*arr).clone());
        }
    }
    let refs: Vec<&Array<T>> = promoted.iter().collect();
    hstack(&refs)
}

/// Stack arrays in sequence vertically (row wise).
<<<<<<< HEAD
=======
///
/// This is equivalent to concatenation along the first axis.
/// row_stack is an alias for vstack.
>>>>>>> origin/main
pub fn row_stack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    vstack(arrays)
}

/// Assemble an nd-array from nested sequences of blocks.
<<<<<<< HEAD
=======
///
/// Blocks in the innermost lists are concatenated along the last axis (-1),
/// then these are concatenated along the second-last axis (-2), etc.
>>>>>>> origin/main
pub fn block<T>(arrays: &[&[&Array<T>]]) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "block: must provide at least one row of arrays",
        ));
    }

<<<<<<< HEAD
=======
    // First, concatenate each row along axis 1 (horizontally)
>>>>>>> origin/main
    let mut rows = Vec::new();
    for row in arrays {
        if row.is_empty() {
            return Err(NumPyError::invalid_value(
                "block: each row must have at least one array",
            ));
        }
        let row_result = hstack(row)?;
        rows.push(row_result);
    }

<<<<<<< HEAD
=======
    // Then concatenate all rows along axis 0 (vertically)
>>>>>>> origin/main
    let refs: Vec<&Array<T>> = rows.iter().collect();
    vstack(&refs)
}

/// Replaces specified elements of an array with given values.
<<<<<<< HEAD
=======
///
/// The indexing works on the flattened target array.
>>>>>>> origin/main
pub fn put<T>(array: &mut Array<T>, indices: &[usize], values: &[T], mode: &str) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let size = array.size();
    for (i, &idx) in indices.iter().enumerate() {
        let actual_idx = match mode {
            "raise" => {
                if idx >= size {
                    return Err(NumPyError::index_error(idx, size));
                }
                idx
            }
            "wrap" => idx % size,
            "clip" => idx.min(size - 1),
            _ => {
                return Err(NumPyError::invalid_value(
                    "mode must be 'raise', 'wrap', or 'clip'",
                ))
            }
        };
        let val = &values[i % values.len()];
        array.set_linear(actual_idx, val.clone());
    }
    Ok(())
}

/// Change elements of an array based on conditional and input values.
<<<<<<< HEAD
=======
///
/// Sets a.flat[n] = values[n] for each n where mask.flat[n]==True.
>>>>>>> origin/main
pub fn putmask<T>(array: &mut Array<T>, mask: &Array<bool>, values: &Array<T>) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if array.shape() != mask.shape() {
        return Err(NumPyError::shape_mismatch(
            array.shape().to_vec(),
            mask.shape().to_vec(),
        ));
    }

    let mut val_idx = 0;
    let values_size = values.size();
    for i in 0..array.size() {
        if let Some(&m) = mask.get_linear(i) {
            if m {
                if let Some(v) = values.get_linear(val_idx % values_size) {
                    array.set_linear(i, v.clone());
                }
                val_idx += 1;
            }
        }
    }
    Ok(())
}

/// Change elements of an array based on conditional and input values.
<<<<<<< HEAD
=======
///
/// Similar to np.putmask, but the indexing is different.
>>>>>>> origin/main
pub fn place<T>(array: &mut Array<T>, mask: &Array<bool>, values: &[T]) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if array.shape() != mask.shape() {
        return Err(NumPyError::shape_mismatch(
            array.shape().to_vec(),
            mask.shape().to_vec(),
        ));
    }

    let mut val_idx = 0;
    for i in 0..array.size() {
        if let Some(&m) = mask.get_linear(i) {
            if m {
                array.set_linear(i, values[val_idx % values.len()].clone());
                val_idx += 1;
            }
        }
    }
    Ok(())
}

/// Put values into the destination array by matching 1d index and data slices.
<<<<<<< HEAD
=======
///
/// This iterates over matching 1d slices oriented along the specified axis in the index
/// and data arrays, and uses the former to place values into the latter.
>>>>>>> origin/main
pub fn put_along_axis<T>(
    array: &mut Array<T>,
    indices: &Array<usize>,
    values: &Array<T>,
    axis: isize,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
{
    let ndim = array.ndim();
    let axis = normalize_axis(axis, ndim)?;
<<<<<<< HEAD
    let shape = array.shape().to_vec();

=======
    let shape = array.shape().to_vec(); // Clone to owned to avoid borrow issues

    // Indices and values must have the same shape
>>>>>>> origin/main
    if indices.shape() != values.shape() {
        return Err(NumPyError::shape_mismatch(
            indices.shape().to_vec(),
            values.shape().to_vec(),
        ));
    }

<<<<<<< HEAD
    for i in 0..indices.size() {
        let mut idx = crate::strides::compute_multi_indices(i, indices.shape());
=======
    // Iterate over all positions in the indices/values array
    for i in 0..indices.size() {
        // Compute multi-index from linear index
        let mut idx = crate::strides::compute_multi_indices(i, indices.shape());

        // Get the index value along the axis
>>>>>>> origin/main
        let axis_idx = *indices
            .get_linear(i)
            .ok_or_else(|| NumPyError::index_error(i, indices.size()))?;

        if axis_idx >= shape[axis] {
            return Err(NumPyError::index_error(axis_idx, shape[axis]));
        }

<<<<<<< HEAD
        idx[axis] = axis_idx;
        let val = values.get_linear(i).cloned().unwrap_or_default();
=======
        // Replace the axis dimension with the actual index value
        idx[axis] = axis_idx;

        // Get the value to place
        let val = values.get_linear(i).cloned().unwrap_or_default();

        // Set the value in the target array
>>>>>>> origin/main
        array.set_multi(&idx, val)?;
    }
    Ok(())
}

/// Generate a Vandermonde matrix.
<<<<<<< HEAD
=======
///
/// The columns of the output matrix are powers of the input vector. The order
/// of the powers is determined by the `increasing` parameter.
>>>>>>> origin/main
pub fn vander<T>(x: &Array<T>, n: Option<usize>, increasing: bool) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Num + num_traits::Pow<u32, Output = T> + Send + Sync + 'static,
{
    if x.ndim() != 1 {
        return Err(NumPyError::invalid_value("vander requires 1D array"));
    }

    let m = x.shape()[0];
    let n_val = n.unwrap_or(m);
    let mut data = vec![T::default(); m * n_val];

    for i in 0..m {
        let val = x.get_linear(i).cloned().unwrap_or_default();
        for j in 0..n_val {
            let power = if increasing {
                j as u32
            } else {
                (n_val - 1 - j) as u32
            };
            data[i * n_val + j] = val.clone().pow(power);
        }
    }

    Ok(Array::from_shape_vec(vec![m, n_val], data))
}

pub mod exports {
    pub use super::{
        array_split, block, column_stack, concatenate, diag, diagonal, dsplit, dstack, hsplit,
        hstack, place, put, put_along_axis, putmask, row_stack, split, stack, tril, triu, vander,
        vsplit, vstack,
    };
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    if axis < 0 {
        let ax = axis + ndim as isize;
        if ax < 0 {
            return Err(NumPyError::invalid_value("axis out of bounds"));
        }
        Ok(ax as usize)
    } else {
        if axis as usize >= ndim {
            return Err(NumPyError::invalid_value("axis out of bounds"));
        }
        Ok(axis as usize)
<<<<<<< HEAD
=======
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;

    #[test]
    fn test_triu_basic() {
        let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let result = triu(&arr, 0).unwrap();
        assert_eq!(result.data(), &[1, 2, 3, 0, 5, 6, 0, 0, 9]);
        // Verify original is untouched
        assert_eq!(arr.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_triu_offset() {
        let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // k = 1: zero out main diagonal and below
        let result = triu(&arr, 1).unwrap();
        assert_eq!(result.data(), &[0, 2, 3, 0, 0, 6, 0, 0, 0]);

        // k = -1: keep one below main diagonal
        let result_neg = triu(&arr, -1).unwrap();
        assert_eq!(result_neg.data(), &[1, 2, 3, 4, 5, 6, 0, 8, 9]);
    }

    #[test]
    fn test_tril_basic() {
        let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let result = tril(&arr, 0).unwrap();
        assert_eq!(result.data(), &[1, 0, 0, 4, 5, 0, 7, 8, 9]);
        // Verify original is untouched
        assert_eq!(arr.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_tril_offset() {
        let arr = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        // k = -1: below main diagonal
        let result = tril(&arr, -1).unwrap();
        assert_eq!(result.data(), &[0, 0, 0, 4, 0, 0, 7, 8, 0]);

        // k = 1: keep one above main diagonal
        let result_pos = tril(&arr, 1).unwrap();
        assert_eq!(result_pos.data(), &[1, 2, 0, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_tri_nd() {
        let arr = Array::from_shape_vec(vec![2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let result = triu(&arr, 0).unwrap();
        // Each 2x2 matrix should be triu
        assert_eq!(result.data(), &[1, 2, 0, 4, 5, 6, 0, 8]);
>>>>>>> origin/main
    }
}
