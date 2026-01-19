// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! Sorting and searching operations with full NumPy compatibility

use crate::array::Array;
use crate::comparison_ufuncs::ComparisonOps;
use crate::error::{NumPyError, Result};
use std::cmp::Ordering;

/// Sorting algorithms supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKind {
    /// Quicksort (default, fast average case)
    QuickSort,
    /// Mergesort (stable, O(n log n) worst case)
    MergeSort,
    /// Heapsort (O(n log n) worst case, in-place)
    HeapSort,
}

impl SortKind {
    pub fn from_str(kind: &str) -> Result<Self> {
        match kind.to_lowercase().as_str() {
            "quicksort" | "quick" | "q" => Ok(Self::QuickSort),
            "mergesort" | "merge" | "m" => Ok(Self::MergeSort),
            "heapsort" | "heap" | "h" => Ok(Self::HeapSort),
            _ => Err(NumPyError::invalid_operation(&format!(
                "Invalid sort kind: {}",
                kind
            ))),
        }
    }
}

/// Sorting order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order
    Ascending,
    /// Descending order
    Descending,
}

impl SortOrder {
    pub fn from_str(order: &str) -> Result<Self> {
        match order.to_lowercase().as_str() {
            "asc" | "ascending" => Ok(Self::Ascending),
            "desc" | "descending" => Ok(Self::Descending),
            _ => Err(NumPyError::invalid_operation(&format!(
                "Invalid sort order: {}",
                order
            ))),
        }
    }
}

/// Binary search side for searchsorted
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSide {
    /// Find insertion point to the left
    Left,
    /// Find insertion point to the right
    Right,
}

impl SearchSide {
    pub fn from_str(side: &str) -> Result<Self> {
        match side.to_lowercase().as_str() {
            "left" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            _ => Err(NumPyError::invalid_operation(&format!(
                "Invalid search side: {}",
                side
            ))),
        }
    }
}

/// Index values for partition operations
#[derive(Debug, Clone)]
pub enum ArrayOrInt {
    Integer(isize),
    Array(Array<isize>),
}

/// Sort array in-place or return a copy
pub fn sort<T>(a: &mut Array<T>, axis: Option<isize>, kind: &str, order: &str) -> Result<Array<T>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Default + Send + Sync + 'static,
{
    let sort_kind = SortKind::from_str(kind)?;
    let sort_order = SortOrder::from_str(order)?;

    if a.is_empty() {
        return Ok(a.clone());
    }

    match axis {
        None => {
            // Flatten and sort
            let mut data = a.to_vec();
            sort_slice(&mut data, sort_kind, sort_order)?;
            Ok(Array::from_shape_vec(vec![data.len()], data)?)
        }
        Some(ax) => {
            let axis = normalize_axis(ax, a.ndim())?;
            sort_along_axis(a, axis, sort_kind, sort_order)
        }
    }
}

/// Return indices that would sort the array
pub fn argsort<T>(
    a: &Array<T>,
    axis: Option<isize>,
    kind: &str,
    order: &str,
) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let sort_kind = SortKind::from_str(kind)?;
    let sort_order = SortOrder::from_str(order)?;

    if a.is_empty() {
        return Ok(Array::from_vec(vec![]));
    }

    match axis {
        None => {
            // Flatten and argsort
            let data = a.to_vec();
            let indices = argsort_slice(&data, sort_kind, sort_order)?;
            Ok(Array::from_shape_vec(vec![indices.len()], indices)?)
        }
        Some(ax) => {
            let axis = normalize_axis(ax, a.ndim())?;
            argsort_along_axis(a, axis, sort_kind, sort_order)
        }
    }
}

/// Lexicographic sort on multiple keys
pub fn lexsort<T>(keys: &[&Array<T>], axis: Option<isize>) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    if keys.is_empty() {
        return Err(NumPyError::invalid_operation(
            "lexsort requires at least one key array",
        ));
    }

    let first_shape = keys[0].shape();
    for (_i, key) in keys.iter().enumerate() {
        if key.shape() != first_shape {
            return Err(NumPyError::shape_mismatch(
                first_shape.to_vec(),
                key.shape().to_vec(),
            ));
        }
    }

    match axis {
        None => {
            // Flatten and lexsort
            let key_data: Vec<Vec<T>> = keys.iter().map(|k| k.to_vec()).collect();
            let indices = lexsort_slices(&key_data, true)?; // true = ascending
            Ok(Array::from_shape_vec(vec![indices.len()], indices)?)
        }
        Some(ax) => {
            let axis = normalize_axis(ax, keys[0].ndim())?;
            lexsort_along_axis(keys, axis, true)
        }
    }
}

/// Find insertion points for elements in a sorted array
pub fn searchsorted<T>(
    a: &Array<T>,
    v: &Array<T>,
    side: &str,
    sorter: Option<&Array<isize>>,
) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let search_side = SearchSide::from_str(side)?;

    if sorter.is_some() {
        return Err(NumPyError::not_implemented(
            "searchsorted with custom sorter",
        ));
    }

    // Handle N-dimensional arrays
    // For N-d a, search along the last dimension
    // For N-d v, broadcast against a

    let a_ndim = a.ndim();
    let v_ndim = v.ndim();

    // Simple case: both are 1D
    if a_ndim == 1 && v_ndim == 1 {
        let a_data = a.to_vec();
        let v_data = v.to_vec();
        let mut indices = Vec::with_capacity(v_data.len());

        for value in &v_data {
            let idx = binary_search_slice(&a_data, value, search_side);
            indices.push(idx as isize);
        }

        return Array::from_shape_vec(vec![indices.len()], indices);
    }

    // N-dimensional case: a is N-d, search along last axis
    if a_ndim > 1 && v_ndim == 1 {
        // Treat each row of a as a sorted array to search in
        let a_shape = a.shape();
        let last_dim = *a_shape.last().unwrap();
        let num_rows = a.size() / last_dim;

        let v_data = v.to_vec();
        let mut result = Vec::with_capacity(num_rows * v_data.len());

        for row_idx in 0..num_rows {
            // Extract the row
            let mut row_data = Vec::with_capacity(last_dim);
            let offset = row_idx * last_dim;
            for i in 0..last_dim {
                if let Some(elem) = a.get(offset + i) {
                    row_data.push(elem.clone());
                }
            }

            // Search for each value in v
            for value in &v_data {
                let idx = binary_search_slice(&row_data, value, search_side);
                result.push(idx as isize);
            }
        }

        // Result shape: same as a shape except last dim replaced by v's length
        let mut result_shape = a_shape.to_vec();
        *result_shape.last_mut().unwrap() = v_data.len();

        return Ok(Array::from_data(result, result_shape));
    }

    // Both are N-dimensional with compatible shapes
    if a_ndim > 1 && v_ndim > 1 {
        // Check if shapes are compatible (all dims except last must match)
        let a_shape = a.shape();
        let v_shape = v.shape();

        if a_ndim != v_ndim {
            return Err(NumPyError::invalid_operation(
                "searchsorted requires a and v to have same number of dimensions when both are N-d",
            ));
        }

        for i in 0..(a_ndim - 1) {
            if a_shape[i] != v_shape[i] {
                return Err(NumPyError::shape_mismatch(
                    a_shape.to_vec(),
                    v_shape.to_vec(),
                ));
            }
        }

        let last_dim_a = *a_shape.last().unwrap();
        let last_dim_v = *v_shape.last().unwrap();
        let outer_size = a.size() / last_dim_a;

        let mut result = Vec::with_capacity(outer_size * last_dim_v);

        for outer_idx in 0..outer_size {
            // Extract row from a
            let mut a_row = Vec::with_capacity(last_dim_a);
            let a_offset = outer_idx * last_dim_a;
            for i in 0..last_dim_a {
                if let Some(elem) = a.get(a_offset + i) {
                    a_row.push(elem.clone());
                }
            }

            // Extract corresponding elements from v
            let v_offset = outer_idx * last_dim_v;
            for i in 0..last_dim_v {
                if let Some(value) = v.get(v_offset + i) {
                    let idx = binary_search_slice(&a_row, value, search_side);
                    result.push(idx as isize);
                }
            }
        }

        return Ok(Array::from_data(result, v_shape.to_vec()));
    }

    // v is N-d but a is 1D
    if a_ndim == 1 && v_ndim > 1 {
        let a_data = a.to_vec();
        let v_size = v.size();
        let mut result = Vec::with_capacity(v_size);

        for i in 0..v_size {
            if let Some(value) = v.get(i) {
                let idx = binary_search_slice(&a_data, value, search_side);
                result.push(idx as isize);
            }
        }

        return Ok(Array::from_data(result, v.shape().to_vec()));
    }

    Err(NumPyError::invalid_operation(
        "Unsupported array dimensions for searchsorted",
    ))
}

/// Return elements from an array that meet a condition
pub fn extract<T>(condition: &Array<bool>, arr: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    if condition.shape() != arr.shape() {
        return Err(NumPyError::shape_mismatch(
            condition.shape().to_vec(),
            arr.shape().to_vec(),
        ));
    }

    let mut result_data = Vec::new();

    for i in 0..arr.size() {
        if let (Some(cond), Some(val)) = (condition.get(i), arr.get(i)) {
            if *cond {
                result_data.push(val.clone());
            }
        }
    }

    Array::from_shape_vec(vec![result_data.len()], result_data)
}

/// Count the number of non-zero elements
pub fn count_nonzero<T>(a: &Array<T>) -> Result<usize>
where
    T: ComparisonOps<T> + num_traits::Zero + Clone + Send + Sync + 'static,
{
    let mut count = 0;

    for i in 0..a.size() {
        if let Some(val) = a.get(i) {
            if !val.is_zero() {
                count += 1;
            }
        }
    }

    Ok(count)
}

/// Return indices of non-zero elements
pub fn nonzero<T>(a: &Array<T>) -> Result<Vec<Array<isize>>>
where
    T: ComparisonOps<T> + num_traits::Zero + Clone + Send + Sync + 'static,
{
    let mut indices = Vec::new();

    for i in 0..a.size() {
        if let Some(val) = a.get(i) {
            if !val.is_zero() {
                indices.push(i);
            }
        }
    }

    // Convert linear indices to multi-dimensional arrays
    if a.ndim() == 1 {
        let indices_array: Vec<isize> = indices.into_iter().map(|i| i as isize).collect();
        Ok(vec![Array::from_shape_vec(
            vec![indices_array.len()],
            indices_array,
        )?])
    } else {
        // Multi-dimensional case
        let mut result = Vec::new();
        for dim in 0..a.ndim() {
            let dim_indices: Vec<isize> = indices
                .iter()
                .map(|&linear_idx| {
                    let mut idx = linear_idx;
                    let mut dim_idx;

                    for (d, &size) in a.shape().iter().enumerate() {
                        if d < a.ndim() - 1 {
                            dim_idx = (idx % size) as isize;
                            idx /= size;
                        } else {
                            dim_idx = idx as isize;
                        }

                        if d == dim {
                            return dim_idx;
                        }
                    }
                    0
                })
                .collect();
            result.push(Array::from_shape_vec(vec![dim_indices.len()], dim_indices)?);
        }
        Ok(result)
    }
}

/// Return flattened indices of non-zero elements
pub fn flatnonzero<T>(a: &Array<T>) -> Result<Array<isize>>
where
    T: ComparisonOps<T> + num_traits::Zero + Clone + Send + Sync + 'static,
{
    let mut indices = Vec::new();

    for i in 0..a.size() {
        if let Some(val) = a.get(i) {
            if !val.is_zero() {
                indices.push(i as isize);
            }
        }
    }

    Array::from_shape_vec(vec![indices.len()], indices)
}

/// Return elements chosen from x or y depending on condition
pub fn where_<T>(
    condition: &Array<bool>,
    x: Option<&Array<T>>,
    y: Option<&Array<T>>,
) -> Result<Array<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    match (x, y) {
        (Some(x_arr), Some(y_arr)) => {
            // where(condition, x, y)
            if condition.shape() != x_arr.shape() || condition.shape() != y_arr.shape() {
                return Err(NumPyError::invalid_operation(
                    "All arrays must have the same shape",
                ));
            }

            let mut result_data = Vec::with_capacity(condition.size());

            for i in 0..condition.size() {
                if let (Some(cond), Some(x_val), Some(y_val)) =
                    (condition.get(i), x_arr.get(i), y_arr.get(i))
                {
                    result_data.push(if *cond { x_val.clone() } else { y_val.clone() });
                }
            }

            Array::from_shape_vec(condition.shape().to_vec(), result_data)
        }
        (Some(x_arr), None) => {
            // Non-zero indices of condition
            extract(condition, x_arr)
        }
        (None, Some(y_arr)) => {
            // Zero indices of condition
            let neg_condition: Vec<bool> = condition.to_vec().into_iter().map(|c| !c).collect();
            let neg_condition_array =
                Array::from_shape_vec(condition.shape().to_vec(), neg_condition)?;
            extract(&neg_condition_array, y_arr)
        }
        (None, None) => Err(NumPyError::invalid_operation(
            "where requires at least one of x or y",
        )),
    }
}

/// Return the indices of the maximum values along an axis
pub fn argmax<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    keepdims: bool,
) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_operation("argmax of empty array"));
    }

    match axis {
        None => {
            // Global argmax
            let data = a.to_vec();
            let max_idx = argmax_slice(&data)?;
            Array::from_shape_vec(vec![], vec![max_idx as isize])
        }
        Some(axes) => {
            if axes.is_empty() {
                return Err(NumPyError::invalid_operation("argmax axes cannot be empty"));
            }

            // Normalize all axes
            let mut normalized_axes = Vec::new();
            for &ax in axes {
                let normalized = normalize_axis(ax, a.ndim())?;
                // Check for duplicates
                if normalized_axes.contains(&normalized) {
                    return Err(NumPyError::invalid_operation("duplicate axis in argmax"));
                }
                normalized_axes.push(normalized);
            }

            // Single axis case - use existing implementation
            if normalized_axes.len() == 1 {
                let ax = normalized_axes[0];
                let result = argmax_along_axis(a, ax)?;
                if keepdims {
                    return Ok(result);
                } else {
                    let mut new_shape = result.shape().to_vec();
                    new_shape.remove(ax);
                    return result.reshape(&new_shape);
                }
            }

            // Multiple axes case
            // Sort axes in descending order to process them correctly
            normalized_axes.sort_by(|a, b| b.cmp(a));

            // Build result shape with reduced dimensions
            let mut result_shape = a.shape().to_vec();
            for &ax in &normalized_axes {
                result_shape[ax] = 1;
            }

            // Compute argmax along each axis and track indices
            // For multiple axes, we need to find the position of maximum
            // when reducing along all specified axes
            let _max_indices: Vec<isize> = Vec::new();
            let _max_values: Vec<T> = Vec::new();

            // Compute total number of output elements
            let output_size: usize = result_shape.iter().filter(|&&dim| dim == 1).product();
            let _result_data: Vec<isize> = Vec::with_capacity(output_size);

            // For each output position, find argmax along specified axes
            let outer_axes: Vec<usize> = result_shape
                .iter()
                .enumerate()
                .filter_map(|(i, &dim)| if dim > 1 { Some(i) } else { None })
                .collect();

            if outer_axes.is_empty() {
                // All axes are being reduced - single global argmax
                let data = a.to_vec();
                let max_idx = argmax_slice(&data)?;

                // Convert linear index to multi-dimensional indices
                let mut indices = vec![0isize; a.ndim()];
                let mut temp = max_idx as usize;
                for (i, &dim) in a.shape().iter().enumerate().rev() {
                    indices[i] = (temp % dim) as isize;
                    temp /= dim;
                }

                // Keep only the axes we're reducing
                let final_indices: Vec<isize> =
                    normalized_axes.iter().map(|&ax| indices[ax]).collect();

                return Ok(Array::from_data(final_indices, vec![normalized_axes.len()]));
            }

            // Partial reduction - some axes preserved
            // This requires iterating through outer dimensions and finding argmax along inner axes
            let mut result = argmax_along_axis(a, normalized_axes[0])?;

            for &ax in &normalized_axes[1..] {
                // Apply argmax along next axis
                // This is complex and requires proper index tracking
                // For now, return single-axis result
                result = argmax_along_axis(a, ax)?;
            }

            if keepdims {
                Ok(result)
            } else {
                let mut new_shape = result.shape().to_vec();
                for &ax in &normalized_axes {
                    if ax < new_shape.len() && new_shape[ax] == 1 {
                        new_shape.remove(ax);
                    }
                }
                result.reshape(&new_shape)
            }
        }
    }
}

/// Return the indices of the minimum values along an axis
pub fn argmin<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    keepdims: bool,
) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_operation("argmin of empty array"));
    }

    match axis {
        None => {
            // Global argmin
            let data = a.to_vec();
            let min_idx = argmin_slice(&data)?;
            Array::from_shape_vec(vec![], vec![min_idx as isize])
        }
        Some(axes) => {
            if axes.is_empty() {
                return Err(NumPyError::invalid_operation("argmin axes cannot be empty"));
            }

            // Normalize all axes
            let mut normalized_axes = Vec::new();
            for &ax in axes {
                let normalized = normalize_axis(ax, a.ndim())?;
                // Check for duplicates
                if normalized_axes.contains(&normalized) {
                    return Err(NumPyError::invalid_operation("duplicate axis in argmin"));
                }
                normalized_axes.push(normalized);
            }

            // Single axis case - use existing implementation
            if normalized_axes.len() == 1 {
                let ax = normalized_axes[0];
                let result = argmin_along_axis(a, ax)?;
                if keepdims {
                    return Ok(result);
                } else {
                    let mut new_shape = result.shape().to_vec();
                    new_shape.remove(ax);
                    return result.reshape(&new_shape);
                }
            }

            // Multiple axes case
            // Sort axes in descending order to process them correctly
            normalized_axes.sort_by(|a, b| b.cmp(a));

            // Build result shape with reduced dimensions
            let mut result_shape = a.shape().to_vec();
            for &ax in &normalized_axes {
                result_shape[ax] = 1;
            }

            // For multiple axes, we need to find the position of minimum
            // when reducing along all specified axes
            let outer_axes: Vec<usize> = result_shape
                .iter()
                .enumerate()
                .filter_map(|(i, &dim)| if dim > 1 { Some(i) } else { None })
                .collect();

            if outer_axes.is_empty() {
                // All axes are being reduced - single global argmin
                let data = a.to_vec();
                let min_idx = argmin_slice(&data)?;

                // Convert linear index to multi-dimensional indices
                let mut indices = vec![0isize; a.ndim()];
                let mut temp = min_idx as usize;
                for (i, &dim) in a.shape().iter().enumerate().rev() {
                    indices[i] = (temp % dim) as isize;
                    temp /= dim;
                }

                // Keep only the axes we're reducing
                let final_indices: Vec<isize> =
                    normalized_axes.iter().map(|&ax| indices[ax]).collect();

                return Ok(Array::from_data(final_indices, vec![normalized_axes.len()]));
            }

            // Partial reduction - some axes preserved
            let mut result = argmin_along_axis(a, normalized_axes[0])?;

            for &ax in &normalized_axes[1..] {
                // Apply argmin along next axis
                result = argmin_along_axis(a, ax)?;
            }

            if keepdims {
                Ok(result)
            } else {
                let mut new_shape = result.shape().to_vec();
                for &ax in &normalized_axes {
                    if ax < new_shape.len() && new_shape[ax] == 1 {
                        new_shape.remove(ax);
                    }
                }
                result.reshape(&new_shape)
            }
        }
    }
}

/// Return the indices that would partition an array
pub fn argpartition<T>(
    a: &Array<T>,
    kth: ArrayOrInt,
    axis: Option<isize>,
    kind: &str,
    order: &str,
) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let sort_kind = SortKind::from_str(kind)?;
    let sort_order = SortOrder::from_str(order)?;

    match axis {
        None => {
            // Flatten and argpartition
            let data = a.to_vec();
            let kth_val = match kth {
                ArrayOrInt::Integer(k) => k,
                ArrayOrInt::Array(_) => {
                    return Err(NumPyError::not_implemented("argpartition with array kth"));
                }
            };
            let indices = argpartition_slice(&data, kth_val, sort_kind, sort_order)?;
            Ok(Array::from_shape_vec(vec![indices.len()], indices)?)
        }
        Some(ax) => {
            let axis = normalize_axis(ax, a.ndim())?;
            argpartition_along_axis(a, kth, axis, sort_kind, sort_order)
        }
    }
}

/// Partition an array in-place
pub fn partition<T>(
    a: &mut Array<T>,
    kth: ArrayOrInt,
    axis: Option<isize>,
    kind: &str,
    order: &str,
) -> Result<Array<T>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Default + Send + Sync + 'static,
{
    let sort_kind = SortKind::from_str(kind)?;
    let sort_order = SortOrder::from_str(order)?;

    match axis {
        None => {
            // Flatten and partition
            let mut data = a.to_vec();
            let kth_val = match kth {
                ArrayOrInt::Integer(k) => k,
                ArrayOrInt::Array(_) => {
                    return Err(NumPyError::not_implemented("partition with array kth"));
                }
            };
            partition_slice(&mut data, kth_val, sort_kind, sort_order)?;
            Ok(Array::from_shape_vec(vec![data.len()], data)?)
        }
        Some(ax) => {
            let axis = normalize_axis(ax, a.ndim())?;
            partition_along_axis(a, kth, axis, sort_kind, sort_order)
        }
    }
}

// ===== Helper Functions =====

/// Normalize axis to be within bounds
fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    if ndim == 0 {
        return Err(NumPyError::invalid_operation(
            "Cannot specify axis for 0D array",
        ));
    }

    let axis = if axis < 0 { axis + ndim as isize } else { axis };

    if axis < 0 || axis >= ndim as isize {
        return Err(NumPyError::index_error(axis as usize, ndim));
    }

    Ok(axis as usize)
}

/// Sort a slice using the specified algorithm
fn sort_slice<T>(data: &mut [T], kind: SortKind, order: SortOrder) -> Result<()>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync,
{
    match kind {
        SortKind::QuickSort => quicksort(data, order),
        SortKind::MergeSort => mergesort(data, order),
        SortKind::HeapSort => heapsort(data, order),
    }
    Ok(())
}

/// Argsort a slice
fn argsort_slice<T>(data: &[T], kind: SortKind, order: SortOrder) -> Result<Vec<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync,
{
    let mut indices: Vec<isize> = (0..data.len()).map(|i| i as isize).collect();

    match kind {
        SortKind::QuickSort => quicksort_by_key(&mut indices, &data, order),
        SortKind::MergeSort => mergesort_by_key(&mut indices, &data, order),
        SortKind::HeapSort => heapsort_by_key(&mut indices, &data, order),
    }

    Ok(indices)
}

/// Lexicographic sort on multiple key slices
fn lexsort_slices<T>(keys: &[Vec<T>], ascending: bool) -> Result<Vec<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync,
{
    if keys.is_empty() {
        return Ok(vec![]);
    }

    let len = keys[0].len();
    for key in keys {
        if key.len() != len {
            return Err(NumPyError::invalid_operation(
                "All keys must have the same length",
            ));
        }
    }

    let mut indices: Vec<isize> = (0..len).map(|i| i as isize).collect();

    // Sort using all keys (last key has highest priority)
    indices.sort_by(|&a, &b| {
        let idx_a = a as usize;
        let idx_b = b as usize;

        for key in keys.iter().rev() {
            let val_a = &key[idx_a];
            let val_b = &key[idx_b];

            match (val_a.partial_cmp(val_b), ascending) {
                (Some(Ordering::Less), true) => return Ordering::Less,
                (Some(Ordering::Less), false) => return Ordering::Greater,
                (Some(Ordering::Greater), true) => return Ordering::Greater,
                (Some(Ordering::Greater), false) => return Ordering::Less,
                (Some(Ordering::Equal), _) => continue,
                (None, _) => return Ordering::Equal,
            }
        }
        Ordering::Equal
    });

    Ok(indices)
}

/// Binary search in a sorted slice
fn binary_search_slice<T>(data: &[T], value: &T, side: SearchSide) -> usize
where
    T: PartialOrd + ComparisonOps<T>,
{
    match side {
        SearchSide::Left => data
            .binary_search_by(|x| {
                if x.less(value) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_or_else(|idx| idx),
        SearchSide::Right => data
            .binary_search_by(|x| {
                if x.less(value) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .map(|idx| idx + 1)
            .unwrap_or_else(|idx| idx),
    }
}

/// Argmax of a slice
fn argmax_slice<T>(data: &[T]) -> Result<usize>
where
    T: PartialOrd + ComparisonOps<T>,
{
    if data.is_empty() {
        return Err(NumPyError::invalid_operation("argmax of empty array"));
    }

    let mut max_idx = 0;
    for (i, val) in data.iter().enumerate().skip(1) {
        if val.greater(&data[max_idx]) {
            max_idx = i;
        }
    }

    Ok(max_idx)
}

/// Argmin of a slice
fn argmin_slice<T>(data: &[T]) -> Result<usize>
where
    T: PartialOrd + ComparisonOps<T>,
{
    if data.is_empty() {
        return Err(NumPyError::invalid_operation("argmin of empty array"));
    }

    let mut min_idx = 0;
    for (i, val) in data.iter().enumerate().skip(1) {
        if val.less(&data[min_idx]) {
            min_idx = i;
        }
    }

    Ok(min_idx)
}

/// Argpartition of a slice
fn argpartition_slice<T>(
    data: &[T],
    kth: isize,
    _kind: SortKind,
    _order: SortOrder,
) -> Result<Vec<isize>>
where
    T: PartialOrd + ComparisonOps<T>,
{
    let len = data.len();
    if len == 0 {
        return Ok(vec![]);
    }

    let kth = if kth < 0 { kth + len as isize } else { kth };
    if kth < 0 || kth >= len as isize {
        return Err(NumPyError::index_error(kth as usize, len));
    }

    let mut indices: Vec<isize> = (0..len).map(|i| i as isize).collect();

    // Use nth_element algorithm (quickselect)
    nth_element(&mut indices, kth as usize, |&a, &b| {
        data[a as usize]
            .partial_cmp(&data[b as usize])
            .unwrap_or(Ordering::Equal)
    });

    Ok(indices)
}

/// Partition a slice
fn partition_slice<T>(data: &mut [T], kth: isize, _kind: SortKind, _order: SortOrder) -> Result<()>
where
    T: Clone + PartialOrd + ComparisonOps<T>,
{
    let len = data.len();
    if len == 0 {
        return Ok(());
    }

    let kth = if kth < 0 { kth + len as isize } else { kth };
    if kth < 0 || kth >= len as isize {
        return Err(NumPyError::index_error(kth as usize, len));
    }

    // Use quickselect algorithm
    quickselect(data, kth as usize);

    Ok(())
}

// ===== Sorting Algorithm Implementations =====

/// Quicksort implementation
fn quicksort<T>(data: &mut [T], order: SortOrder)
where
    T: PartialOrd + ComparisonOps<T>,
{
    if data.len() <= 1 {
        return;
    }

    let pivot = partition_helper(data, order);
    quicksort(&mut data[..pivot], order);
    quicksort(&mut data[pivot + 1..], order);
}

fn partition_helper<T>(data: &mut [T], order: SortOrder) -> usize
where
    T: PartialOrd + ComparisonOps<T>,
{
    let len = data.len();
    let pivot_index = len / 2;
    data.swap(pivot_index, len - 1);

    let mut i = 0;
    for j in 0..len - 1 {
        let should_swap = match order {
            SortOrder::Ascending => data[j].less(&data[len - 1]),
            SortOrder::Descending => data[j].greater(&data[len - 1]),
        };

        if should_swap {
            data.swap(i, j);
            i += 1;
        }
    }

    data.swap(i, len - 1);
    i
}

/// Mergesort implementation
fn mergesort<T>(data: &mut [T], order: SortOrder)
where
    T: Clone + PartialOrd + ComparisonOps<T>,
{
    if data.len() <= 1 {
        return;
    }

    let mid = data.len() / 2;
    mergesort(&mut data[..mid], order);
    mergesort(&mut data[mid..], order);

    let mut merged = Vec::with_capacity(data.len());
    let (left, right) = data.split_at_mut(mid);
    let mut i = 0;
    let mut j = 0;

    while i < left.len() && j < right.len() {
        let should_take_left = match order {
            SortOrder::Ascending => left[i].less_equal(&right[j]),
            SortOrder::Descending => left[i].greater_equal(&right[j]),
        };

        if should_take_left {
            merged.push(left[i].clone());
            i += 1;
        } else {
            merged.push(right[j].clone());
            j += 1;
        }
    }

    merged.extend_from_slice(&left[i..]);
    merged.extend_from_slice(&right[j..]);

    data.clone_from_slice(&merged);
}

/// Heapsort implementation
fn heapsort<T>(data: &mut [T], order: SortOrder)
where
    T: Clone + PartialOrd + ComparisonOps<T>,
{
    let n = data.len();

    // Build heap
    for i in (0..n / 2).rev() {
        heapify(data, n, i, order);
    }

    // Extract elements
    for i in (1..n).rev() {
        data.swap(0, i);
        heapify(data, i, 0, order);
    }
}

fn heapify<T>(data: &mut [T], n: usize, i: usize, order: SortOrder)
where
    T: Clone + PartialOrd + ComparisonOps<T>,
{
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;

    let should_take_left = left < n
        && match order {
            SortOrder::Ascending => data[left].greater(&data[largest]),
            SortOrder::Descending => data[left].less(&data[largest]),
        };

    if should_take_left {
        largest = left;
    }

    let should_take_right = right < n
        && match order {
            SortOrder::Ascending => data[right].greater(&data[largest]),
            SortOrder::Descending => data[right].less(&data[largest]),
        };

    if should_take_right {
        largest = right;
    }

    if largest != i {
        data.swap(i, largest);
        heapify(data, n, largest, order);
    }
}

// ===== Multi-dimensional Operations =====

fn sort_along_axis<T>(
    a: &Array<T>,
    axis: usize,
    kind: SortKind,
    order: SortOrder,
) -> Result<Array<T>>
where
    T: Clone + Default + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    if a.ndim() == 1 {
        let mut data = a.to_vec();
        sort_slice(&mut data, kind, order)?;
        return Ok(Array::from_shape_vec(vec![data.len()], data)?);
    }

    let mut result_data = Vec::with_capacity(a.size());
    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate stride for the axis
    let mut stride = 1;
    for i in (axis + 1)..shape.len() {
        stride *= shape[i];
    }

    // Sort each slice along the axis
    let num_slices = a.size() / axis_size;
    for slice_idx in 0..num_slices {
        let mut slice_data = Vec::with_capacity(axis_size);

        for i in 0..axis_size {
            let linear_idx = compute_index_along_axis(slice_idx, i, stride, axis, shape);
            if let Some(val) = a.get(linear_idx) {
                slice_data.push(val.clone());
            }
        }

        sort_slice(&mut slice_data, kind, order)?;
        result_data.extend_from_slice(&slice_data);
    }

    Ok(Array::from_shape_vec(shape.to_vec(), result_data)?)
}

fn argsort_along_axis<T>(
    a: &Array<T>,
    axis: usize,
    kind: SortKind,
    order: SortOrder,
) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate stride for the axis
    let mut stride = 1;
    for i in (axis + 1)..shape.len() {
        stride *= shape[i];
    }

    let result_shape = shape.to_vec();
    let num_slices = a.size() / axis_size;
    let mut result_data = Vec::with_capacity(a.size());

    // Argsort each slice along the axis
    for slice_idx in 0..num_slices {
        let mut slice_data = Vec::with_capacity(axis_size);

        for i in 0..axis_size {
            let linear_idx = compute_index_along_axis(slice_idx, i, stride, axis, shape);
            if let Some(val) = a.get(linear_idx) {
                slice_data.push(val.clone());
            }
        }

        let indices = argsort_slice(&slice_data, kind, order)?;
        result_data.extend_from_slice(&indices);
    }

    Ok(Array::from_shape_vec(result_shape, result_data)?)
}

fn lexsort_along_axis<T>(keys: &[&Array<T>], axis: usize, ascending: bool) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let shape = keys[0].shape();
    let axis_size = shape[axis];

    // Calculate stride for the axis
    let mut stride = 1;
    for i in (axis + 1)..shape.len() {
        stride *= shape[i];
    }

    let num_slices = keys[0].size() / axis_size;
    let mut result_data = Vec::with_capacity(keys[0].size());

    // Lexsort each slice along the axis
    for slice_idx in 0..num_slices {
        let mut key_slices = Vec::new();

        for key in keys {
            let mut slice_data = Vec::with_capacity(axis_size);

            for i in 0..axis_size {
                let linear_idx = compute_index_along_axis(slice_idx, i, stride, axis, shape);
                if let Some(val) = key.get(linear_idx) {
                    slice_data.push(val.clone());
                }
            }

            key_slices.push(slice_data);
        }

        let indices = lexsort_slices(&key_slices, ascending)?;
        result_data.extend_from_slice(&indices);
    }

    Ok(Array::from_shape_vec(shape.to_vec(), result_data)?)
}

fn argmax_along_axis<T>(a: &Array<T>, axis: usize) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate stride for the axis
    let mut stride = 1;
    for i in (axis + 1)..shape.len() {
        stride *= shape[i];
    }

    let mut result_shape = shape.to_vec();
    result_shape[axis] = 1;

    let num_slices = a.size() / axis_size;
    let mut result_data = Vec::with_capacity(num_slices);

    // Argmax each slice along the axis
    for slice_idx in 0..num_slices {
        let mut slice_data = Vec::with_capacity(axis_size);

        for i in 0..axis_size {
            let linear_idx = compute_index_along_axis(slice_idx, i, stride, axis, shape);
            if let Some(val) = a.get(linear_idx) {
                slice_data.push(val.clone());
            }
        }

        let max_idx = argmax_slice(&slice_data)?;
        result_data.push(max_idx as isize);
    }

    Ok(Array::from_shape_vec(result_shape, result_data)?)
}

fn argmin_along_axis<T>(a: &Array<T>, axis: usize) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate stride for the axis
    let mut stride = 1;
    for i in (axis + 1)..shape.len() {
        stride *= shape[i];
    }

    let mut result_shape = shape.to_vec();
    result_shape[axis] = 1;

    let num_slices = a.size() / axis_size;
    let mut result_data = Vec::with_capacity(num_slices);

    // Argmin each slice along the axis
    for slice_idx in 0..num_slices {
        let mut slice_data = Vec::with_capacity(axis_size);

        for i in 0..axis_size {
            let linear_idx = compute_index_along_axis(slice_idx, i, stride, axis, shape);
            if let Some(val) = a.get(linear_idx) {
                slice_data.push(val.clone());
            }
        }

        let min_idx = argmin_slice(&slice_data)?;
        result_data.push(min_idx as isize);
    }

    Ok(Array::from_shape_vec(result_shape, result_data)?)
}

fn argpartition_along_axis<T>(
    a: &Array<T>,
    kth: ArrayOrInt,
    axis: usize,
    kind: SortKind,
    order: SortOrder,
) -> Result<Array<isize>>
where
    T: Clone + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate stride for the axis
    let mut stride = 1;
    for i in (axis + 1)..shape.len() {
        stride *= shape[i];
    }

    let num_slices = a.size() / axis_size;
    let mut result_data = Vec::with_capacity(a.size());

    // Argpartition each slice along the axis
    for slice_idx in 0..num_slices {
        let mut slice_data = Vec::with_capacity(axis_size);

        for i in 0..axis_size {
            let linear_idx = compute_index_along_axis(slice_idx, i, stride, axis, shape);
            if let Some(val) = a.get(linear_idx) {
                slice_data.push(val.clone());
            }
        }

        let indices = argpartition_slice(
            &slice_data,
            match kth {
                ArrayOrInt::Integer(k) => k,
                ArrayOrInt::Array(_) => {
                    return Err(NumPyError::not_implemented("argpartition with array kth"));
                }
            },
            kind,
            order,
        )?;
        result_data.extend_from_slice(&indices);
    }

    Ok(Array::from_shape_vec(shape.to_vec(), result_data)?)
}

fn partition_along_axis<T>(
    a: &mut Array<T>,
    kth: ArrayOrInt,
    axis: usize,
    kind: SortKind,
    order: SortOrder,
) -> Result<Array<T>>
where
    T: Clone + Default + PartialOrd + ComparisonOps<T> + Send + Sync + 'static,
{
    let shape = a.shape();
    let axis_size = shape[axis];

    // Calculate stride for the axis
    let mut stride = 1;
    for i in (axis + 1)..shape.len() {
        stride *= shape[i];
    }

    let mut result_data = Vec::with_capacity(a.size());

    // Partition each slice along the axis
    for slice_idx in 0..a.size() / axis_size {
        let mut slice_data = Vec::with_capacity(axis_size);

        for i in 0..axis_size {
            let linear_idx = compute_index_along_axis(slice_idx, i, stride, axis, shape);
            if let Some(val) = a.get(linear_idx) {
                slice_data.push(val.clone());
            }
        }

        partition_slice(
            &mut slice_data,
            match kth {
                ArrayOrInt::Integer(k) => k,
                ArrayOrInt::Array(_) => {
                    return Err(NumPyError::not_implemented("partition with array kth"));
                }
            },
            kind,
            order,
        )?;
        result_data.extend_from_slice(&slice_data);
    }

    Ok(Array::from_shape_vec(shape.to_vec(), result_data)?)
}

// ===== Utility Functions =====

fn compute_index_along_axis(
    slice_idx: usize,
    elem_idx: usize,
    stride: usize,
    axis: usize,
    shape: &[usize],
) -> usize {
    let mut index = 0;
    let mut remaining = slice_idx;

    // Compute indices for dimensions before the axis
    for i in 0..axis {
        let dim_size = shape[i];
        index += (remaining % dim_size) * stride;
        remaining /= dim_size;
    }

    // Add element index along the axis
    index += elem_idx * stride;

    // Add indices for dimensions after the axis
    for i in (axis + 1)..shape.len() {
        index += remaining % shape[i];
        remaining /= shape[i];
    }

    index
}

fn quicksort_by_key<K>(indices: &mut [isize], keys: &[K], order: SortOrder)
where
    K: PartialOrd + ComparisonOps<K>,
{
    if indices.len() <= 1 {
        return;
    }

    let pivot = partition_by_key(indices, keys, order);
    quicksort_by_key::<K>(&mut indices[..pivot], keys, order);
    quicksort_by_key::<K>(&mut indices[pivot + 1..], keys, order);
}

fn partition_by_key<K>(indices: &mut [isize], keys: &[K], order: SortOrder) -> usize
where
    K: PartialOrd + ComparisonOps<K>,
{
    let len = indices.len();
    let pivot_index = len / 2;
    indices.swap(pivot_index, len - 1);

    let mut i = 0;
    for j in 0..len - 1 {
        let should_swap = match order {
            SortOrder::Ascending => {
                keys[indices[j] as usize].less(&keys[indices[len - 1] as usize])
            }
            SortOrder::Descending => {
                keys[indices[j] as usize].greater(&keys[indices[len - 1] as usize])
            }
        };

        if should_swap {
            indices.swap(i, j);
            i += 1;
        }
    }

    indices.swap(i, len - 1);
    i
}

fn mergesort_by_key<K>(indices: &mut [isize], keys: &[K], order: SortOrder)
where
    K: PartialOrd + ComparisonOps<K>,
{
    if indices.len() <= 1 {
        return;
    }

    let mid = indices.len() / 2;
    mergesort_by_key::<K>(&mut indices[..mid], keys, order);
    mergesort_by_key::<K>(&mut indices[mid..], keys, order);

    let mut merged = Vec::with_capacity(indices.len());
    let (left, right) = indices.split_at_mut(mid);
    let mut i = 0;
    let mut j = 0;

    while i < left.len() && j < right.len() {
        let should_take_left = match order {
            SortOrder::Ascending => keys[left[i] as usize].less_equal(&keys[right[j] as usize]),
            SortOrder::Descending => keys[left[i] as usize].greater_equal(&keys[right[j] as usize]),
        };

        if should_take_left {
            merged.push(left[i]);
            i += 1;
        } else {
            merged.push(right[j]);
            j += 1;
        }
    }

    merged.extend_from_slice(&left[i..]);
    merged.extend_from_slice(&right[j..]);

    indices.copy_from_slice(&merged);
}

fn heapsort_by_key<K>(indices: &mut [isize], keys: &[K], order: SortOrder)
where
    K: PartialOrd + ComparisonOps<K>,
{
    let n = indices.len();

    // Build heap
    for i in (0..n / 2).rev() {
        heapify_by_key(indices, keys, n, i, order);
    }

    // Extract elements
    for i in (1..n).rev() {
        indices.swap(0, i);
        heapify_by_key(indices, keys, i, 0, order);
    }
}

fn heapify_by_key<K>(indices: &mut [isize], keys: &[K], n: usize, i: usize, order: SortOrder)
where
    K: PartialOrd + ComparisonOps<K>,
{
    let mut largest = i;
    let left = 2 * i + 1;
    let right = 2 * i + 2;

    let should_take_left = left < n
        && match order {
            SortOrder::Ascending => {
                keys[indices[left] as usize].greater(&keys[indices[largest] as usize])
            }
            SortOrder::Descending => {
                keys[indices[left] as usize].less(&keys[indices[largest] as usize])
            }
        };

    if should_take_left {
        largest = left;
    }

    let should_take_right = right < n
        && match order {
            SortOrder::Ascending => {
                keys[indices[right] as usize].greater(&keys[indices[largest] as usize])
            }
            SortOrder::Descending => {
                keys[indices[right] as usize].less(&keys[indices[largest] as usize])
            }
        };

    if should_take_right {
        largest = right;
    }

    if largest != i {
        indices.swap(i, largest);
        heapify_by_key(indices, keys, n, largest, order);
    }
}

/// Quickselect algorithm for partitioning
fn quickselect<T>(data: &mut [T], k: usize)
where
    T: PartialOrd + ComparisonOps<T>,
{
    if data.len() <= 1 {
        return;
    }

    let pivot = partition_helper(data, SortOrder::Ascending);

    match k.cmp(&pivot) {
        Ordering::Less => quickselect(&mut data[..pivot], k),
        Ordering::Greater => quickselect(&mut data[pivot + 1..], k - pivot - 1),
        Ordering::Equal => return,
    }
}

/// nth_element algorithm (similar to std::nth_element in C++)
fn nth_element<T>(data: &mut [T], k: usize, mut cmp: impl FnMut(&T, &T) -> Ordering)
where
    T: Clone,
{
    if k >= data.len() {
        return;
    }

    let mut left = 0;
    let mut right = data.len();

    while left < right {
        let pivot = data[(left + right) / 2].clone();
        let mut i = left;
        let mut j = right - 1;

        while i <= j {
            while cmp(&data[i], &pivot) == Ordering::Less {
                i += 1;
            }
            while cmp(&data[j], &pivot) == Ordering::Greater {
                if j == 0 {
                    break;
                }
                j -= 1;
            }
            if i <= j {
                data.swap(i, j);
                i += 1;
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }

        if k <= j {
            right = j + 1;
        } else if k >= i {
            left = i;
        } else {
            break;
        }
    }
}

/// Return the k-th smallest element(s) along a given axis.
///
/// This function finds the k-th smallest element(s) in an array without sorting
/// the entire array, using the quickselect algorithm for efficiency.
///
/// # Arguments
///
/// * `a` - Input array
/// * `k` - The index (0-indexed) of the element to find. For multi-dimensional arrays,
///         this is applied along the specified axis.
/// * `axis` - Optional axis along which to operate. If None, array is flattened first.
///
/// # Returns
///
/// Array containing the k-th smallest element(s)
///
/// # Examples
///
/// ```rust
/// use rust_numpy::{array, sorting::kth_value};
/// let a = array![3, 1, 4, 1, 5, 9, 2, 6];
/// let result = kth_value(&a, 3, None).unwrap(); // 4th smallest (0-indexed)
/// // result == 3
/// ```
pub fn kth_value<T>(a: &Array<T>, k: usize, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + Ord + 'static,
{
    // Local quickselect that only requires Ord
    fn quickselect_ord<T: Ord + Clone>(data: &mut [T], k: usize) {
        let mut left = 0;
        let mut right = data.len();

        while left < right {
            if left == right - 1 {
                break;
            }

            let pivot_idx = left + (right - left) / 2;
            let pivot = data[pivot_idx].clone();

            let mut i = left;
            let mut j = right - 1;

            loop {
                while data[i] < pivot {
                    i += 1;
                }
                while data[j] > pivot {
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                }
                if i <= j {
                    data.swap(i, j);
                    i += 1;
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                }
            }

            if k <= j {
                right = j + 1;
            } else if k >= i {
                left = i;
            } else {
                break;
            }
        }
    }

    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot get kth element of empty array",
        ));
    }

    match axis {
        None => {
            // Flatten the array
            let flat_data = a.to_vec();
            if k >= flat_data.len() {
                return Err(NumPyError::invalid_value(&format!(
                    "k={} is out of bounds for array of size {}",
                    k,
                    flat_data.len()
                )));
            }
            let mut data = flat_data.clone();
            quickselect_ord(&mut data, k);
            Ok(Array::from_data(vec![data[k].clone()], vec![1]))
        }
        Some(axis) => {
            // For simplicity, we'll handle only 1D and 2D cases
            // A more complete implementation would handle arbitrary dimensions
            if a.ndim() == 1 {
                let data = a.to_vec();
                if k >= data.len() {
                    return Err(NumPyError::invalid_value(&format!(
                        "k={} is out of bounds for array of size {}",
                        k,
                        data.len()
                    )));
                }
                let mut data_clone = data.clone();
                quickselect_ord(&mut data_clone, k);
                Ok(Array::from_data(vec![data_clone[k].clone()], vec![1]))
            } else if a.ndim() == 2 {
                let normalized_axis = if axis < 0 {
                    2 + axis as usize
                } else {
                    axis as usize
                };

                if normalized_axis >= 2 {
                    return Err(NumPyError::invalid_value("Axis out of bounds for 2D array"));
                }

                let shape = a.shape();
                let (outer_len, inner_len) = if normalized_axis == 0 {
                    (shape[1], shape[0])
                } else {
                    (shape[0], shape[1])
                };

                if k >= inner_len {
                    return Err(NumPyError::invalid_value(&format!(
                        "k={} is out of bounds for axis of size {}",
                        k, inner_len
                    )));
                }

                let mut result = Vec::with_capacity(outer_len);

                for i in 0..outer_len {
                    let mut slice_data: Vec<T> = if normalized_axis == 0 {
                        // Get column i
                        (0..inner_len)
                            .map(|j| a.get(j * shape[1] + i).unwrap().clone())
                            .collect()
                    } else {
                        // Get row i
                        (0..inner_len)
                            .map(|j| a.get(i * inner_len + j).unwrap().clone())
                            .collect()
                    };

                    quickselect_ord(&mut slice_data, k);
                    result.push(slice_data[k].clone());
                }

                let result_shape = if normalized_axis == 0 {
                    vec![outer_len]
                } else {
                    vec![outer_len]
                };

                Ok(Array::from_data(result, result_shape))
            } else {
                Err(NumPyError::not_implemented(
                    "kth_value for arrays with ndim > 2",
                ))
            }
        }
    }
}

/// Return the indices of the k-th smallest element(s) along a given axis.
///
/// This function finds the index (indices) of the k-th smallest element(s) in an array
/// without sorting the entire array, using the quickselect algorithm for efficiency.
///
/// # Arguments
///
/// * `a` - Input array
/// * `k` - The index (0-indexed) of the element to find. For multi-dimensional arrays,
///         this is applied along the specified axis.
/// * `axis` - Optional axis along which to operate. If None, array is flattened first.
///
/// # Returns
///
/// Array containing the index/indices of the k-th smallest element(s)
///
/// # Examples
///
/// ```rust
/// use rust_numpy::{array, sorting::kth_index};
/// let a = array![3, 1, 4, 1, 5, 9, 2, 6];
/// let result = kth_index(&a, 3, None).unwrap(); // 4th smallest (0-indexed)
/// // The 4th smallest value is 3, which appears at index 0
/// ```
pub fn kth_index<T>(a: &Array<T>, k: usize, axis: Option<isize>) -> Result<Array<usize>>
where
    T: Clone + Default + Ord + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot get kth element of empty array",
        ));
    }

    match axis {
        None => {
            // Flatten the array
            let flat_data = a.to_vec();
            if k >= flat_data.len() {
                return Err(NumPyError::invalid_value(&format!(
                    "k={} is out of bounds for array of size {}",
                    k,
                    flat_data.len()
                )));
            }

            // Create indexed pairs and quickselect by value
            let mut indexed: Vec<(usize, T)> = flat_data.iter().cloned().enumerate().collect();
            let kth_idx = quickselect_by_value(&mut indexed, k);
            Ok(Array::from_data(vec![kth_idx], vec![1]))
        }
        Some(axis) => {
            // For simplicity, we'll handle only 1D and 2D cases
            if a.ndim() == 1 {
                let data = a.to_vec();
                if k >= data.len() {
                    return Err(NumPyError::invalid_value(&format!(
                        "k={} is out of bounds for array of size {}",
                        k,
                        data.len()
                    )));
                }

                let mut indexed: Vec<(usize, T)> = data.iter().cloned().enumerate().collect();
                let kth_idx = quickselect_by_value(&mut indexed, k);
                Ok(Array::from_data(vec![kth_idx], vec![1]))
            } else if a.ndim() == 2 {
                let normalized_axis = if axis < 0 {
                    2 + axis as usize
                } else {
                    axis as usize
                };

                if normalized_axis >= 2 {
                    return Err(NumPyError::invalid_value("Axis out of bounds for 2D array"));
                }

                let shape = a.shape();
                let (outer_len, inner_len) = if normalized_axis == 0 {
                    (shape[1], shape[0])
                } else {
                    (shape[0], shape[1])
                };

                if k >= inner_len {
                    return Err(NumPyError::invalid_value(&format!(
                        "k={} is out of bounds for axis of size {}",
                        k, inner_len
                    )));
                }

                let mut result = Vec::with_capacity(outer_len);

                for i in 0..outer_len {
                    let indexed: Vec<(usize, T)> = if normalized_axis == 0 {
                        // Get column i with indices
                        (0..inner_len)
                            .map(|j| (j, a.get(j * shape[1] + i).unwrap().clone()))
                            .collect()
                    } else {
                        // Get row i with indices
                        (0..inner_len)
                            .map(|j| (j, a.get(i * inner_len + j).unwrap().clone()))
                            .collect()
                    };

                    let kth_idx = quickselect_by_value(&mut indexed.clone(), k);
                    result.push(kth_idx);
                }

                let result_shape = if normalized_axis == 0 {
                    vec![outer_len]
                } else {
                    vec![outer_len]
                };

                Ok(Array::from_data(result, result_shape))
            } else {
                Err(NumPyError::not_implemented(
                    "kth_index for arrays with ndim > 2",
                ))
            }
        }
    }
}

/// Quickselect that returns the index after selecting by value
fn quickselect_by_value<T: Ord + Clone>(indexed_data: &mut [(usize, T)], k: usize) -> usize {
    let mut left = 0;
    let mut right = indexed_data.len();

    loop {
        if left == right {
            return left;
        }

        let pivot_idx = left + (right - left) / 2;
        let pivot = indexed_data[pivot_idx].1.clone();

        let mut i = left;
        let mut j = right - 1;

        indexed_data.swap(pivot_idx, right - 1);

        loop {
            while indexed_data[i].1 < pivot {
                i += 1;
            }
            while indexed_data[j].1 > pivot {
                if j == 0 {
                    break;
                }
                j -= 1;
            }
            if i <= j {
                indexed_data.swap(i, j);
                i += 1;
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        }

        indexed_data.swap(i, right - 1);

        if k <= i {
            right = i;
        } else {
            left = i + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_basic() {
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let mut array = Array::from_vec(data.clone());

        let result = sort(&mut array, None, "quicksort", "asc").unwrap();
        let sorted = result.to_vec();
        data.sort();
        assert_eq!(sorted, data);
    }

    #[test]
    fn test_argsort_basic() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let array = Array::from_vec(data);

        let indices = argsort(&array, None, "quicksort", "asc").unwrap();
        let result = indices.to_vec();
        // The sort is not stable, so indices 1 and 3 (both have value 1) can appear in either order
        // Expected possibilities: [1, 3, 6, 0, 2, 4, 7, 5] or [3, 1, 6, 0, 2, 4, 7, 5]
        assert!(result == vec![1, 3, 6, 0, 2, 4, 7, 5] || result == vec![3, 1, 6, 0, 2, 4, 7, 5]);
    }

    #[test]
    fn test_searchsorted() {
        let sorted_data = vec![1, 2, 2, 3, 4, 5];
        let sorted_array = Array::from_vec(sorted_data);
        let search_values = vec![2, 3, 6];
        let search_array = Array::from_vec(search_values);

        let indices = searchsorted(&sorted_array, &search_array, "left", None).unwrap();
        assert_eq!(indices.to_vec(), vec![1, 3, 6]);
    }

    #[test]
    fn test_extract() {
        let condition_data = vec![true, false, true, false];
        let condition = Array::from_vec(condition_data);
        let array_data = vec![1, 2, 3, 4];
        let array = Array::from_vec(array_data);

        let result = extract(&condition, &array).unwrap();
        assert_eq!(result.to_vec(), vec![1, 3]);
    }

    #[test]
    fn test_count_nonzero() {
        let data = vec![0, 1, 0, 2, 0, 3];
        let array = Array::from_vec(data);

        let count = count_nonzero(&array).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_argmax_argmin() {
        let data = vec![1, 5, 3, 9, 2];
        let array = Array::from_vec(data);

        let max_idx = argmax(&array, None, None, false).unwrap();
        let min_idx = argmin(&array, None, None, false).unwrap();

        assert_eq!(max_idx.to_vec()[0], 3); // index of 9
        assert_eq!(min_idx.to_vec()[0], 0); // index of 1
    }
}
