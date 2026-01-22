// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! NumPy-compatible array manipulation functions
//!
//! This module provides comprehensive array creation and shape manipulation
//! functions that match the NumPy API exactly, including all parameters,
//! defaults, and edge case handling.

use num_traits::{Float, Num, One, Zero};

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};
use crate::memory::MemoryManager;

/// Compute strides for a given shape (C-order by default)
fn compute_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![0isize; shape.len()];
    strides[shape.len() - 1] = 1;

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }

    strides
}

/// Compute Fortran-order strides for a given shape
fn compute_fortran_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![0isize; shape.len()];
    strides[0] = 1;

    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }

    strides
}

/// Validate newshape for reshape operations
fn validate_reshape(old_shape: &[usize], newshape: &[usize]) -> Result<()> {
    let old_size: usize = old_shape.iter().product();
    let new_size: usize = newshape.iter().product();

    if old_size != new_size {
        return Err(NumPyError::shape_mismatch(
            old_shape.to_vec(),
            newshape.to_vec(),
        ));
    }

    Ok(())
}

/// Compute total elements for a shape
fn compute_size(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    crate::strides::normalize_axis(axis, ndim)
}

fn normalize_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis = normalize_axis(axis, ndim)?;
        if !seen.insert(axis) {
            return Err(NumPyError::invalid_operation("duplicate axis"));
        }
        result.push(axis);
    }
    Ok(result)
}

fn indices_from_fortran(linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = linear;
    for (i, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            indices[i] = 0;
        } else {
            indices[i] = remaining % dim;
            remaining /= dim;
        }
    }
    indices
}

fn linear_from_indices(indices: &[usize], shape: &[usize]) -> usize {
    let strides = compute_strides(shape);
    indices
        .iter()
        .zip(strides.iter())
        .map(|(idx, stride)| *idx as isize * *stride)
        .sum::<isize>() as usize
}

// ==================== ARRAY CREATION FUNCTIONS ====================

/// Create a new uninitialized array with the same shape and type as the prototype
///
/// # Arguments
/// * `prototype` - Array to copy shape and dtype from
///
/// # Returns
/// New uninitialized array with same shape and dtype
pub fn empty_like<T>(prototype: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let size = compute_size(prototype.shape());
    let data = Vec::with_capacity(size);
    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: prototype.shape().to_vec(),
        strides: compute_strides(prototype.shape()),
        dtype: prototype.dtype().clone(),
        offset: 0,
    })
}

/// Create a new array filled with ones, with the same shape and type as the prototype
///
/// # Arguments
/// * `prototype` - Array to copy shape and dtype from
///
/// # Returns
/// New array filled with ones
pub fn ones_like<T>(prototype: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + One + 'static,
{
    let size = compute_size(prototype.shape());
    let data = vec![T::one(); size];
    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: prototype.shape().to_vec(),
        strides: compute_strides(prototype.shape()),
        dtype: prototype.dtype().clone(),
        offset: 0,
    })
}

/// Create a new array filled with zeros, with the same shape and type as the prototype
///
/// # Arguments
/// * `prototype` - Array to copy shape and dtype from
///
/// # Returns
/// New array filled with zeros
pub fn zeros_like<T>(prototype: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + Zero + 'static,
{
    let size = compute_size(prototype.shape());
    let data = vec![T::zero(); size];
    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: prototype.shape().to_vec(),
        strides: compute_strides(prototype.shape()),
        dtype: prototype.dtype().clone(),
        offset: 0,
    })
}

/// Create a new array filled with the given value, with the same shape and type as the prototype
///
/// # Arguments
/// * `prototype` - Array to copy shape and dtype from
/// * `fill_value` - Value to fill the array with
///
/// # Returns
/// New array filled with the specified value
pub fn full_like<T>(prototype: &Array<T>, fill_value: T) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let size = compute_size(prototype.shape());
    let data = vec![fill_value; size];
    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: prototype.shape().to_vec(),
        strides: compute_strides(prototype.shape()),
        dtype: prototype.dtype().clone(),
        offset: 0,
    })
}

/// Create a 2-D array with ones on the diagonal and zeros elsewhere
///
/// # Arguments
/// * `n` - Number of rows in the output
/// * `m` - Optional number of columns in the output (defaults to n)
/// * `k` - Optional diagonal offset (0 for main diagonal, positive for upper, negative for lower)
/// * `dtype` - Optional data type (defaults to appropriate default)
///
/// # Returns
/// 2-D array with specified diagonal of ones
pub fn eye<T>(
    n: usize,
    m: Option<usize>,
    k: Option<isize>,
    dtype: Option<Dtype>,
) -> Result<Array<T>>
where
    T: Clone + Default + Zero + One + 'static,
{
    let m_val = m.unwrap_or(n);
    let k = k.unwrap_or(0);

    if n == 0 || m_val == 0 {
        return Ok(Array {
            data: std::sync::Arc::new(MemoryManager::from_vec(vec![])),
            shape: vec![n, m_val],
            strides: compute_strides(&[n, m_val]),
            dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
            offset: 0,
        });
    }

    let mut data = vec![T::zero(); n * m_val];

    for i in 0..n {
        let j = (i as isize + k) as usize;
        if j < m_val {
            data[i * m_val + j] = T::one();
        }
    }

    let _memory_manager = MemoryManager::from_vec(data.clone());

    Ok(Array::from_data(data, vec![n, m_val]))
}

/// Create the identity array
///
/// # Arguments
/// * `n` - Number of rows and columns in the output
/// * `dtype` - Optional data type (defaults to appropriate default)
///
/// # Returns
/// n x n array with ones on the diagonal and zeros elsewhere
pub fn identity<T>(n: usize, dtype: Option<Dtype>) -> Result<Array<T>>
where
    T: Clone + Default + Zero + One + 'static,
{
    eye(n, Some(n), Some(0), dtype)
}

/// Create evenly spaced values within a given interval
///
/// # Arguments
/// * `start` - Start of interval
/// * `stop` - Optional end of interval (defaults to start + 1)
/// * `step` - Optional spacing between values (defaults to 1)
/// * `dtype` - Optional data type (defaults to appropriate default)
///
/// # Returns
/// 1-D array of evenly spaced values
pub fn arange<T>(
    start: T,
    stop: Option<T>,
    step: Option<T>,
    dtype: Option<Dtype>,
) -> Result<Array<T>>
where
    T: Clone + Default + Num + PartialOrd + 'static,
{
    let stop = stop.unwrap_or_else(|| start.clone() + T::one());
    let step = step.unwrap_or_else(|| T::one());

    if step == T::zero() {
        return Err(NumPyError::invalid_operation(
            "arange() step cannot be zero",
        ));
    }

    let mut count = 0;
    let mut current = start.clone();

    if step > T::zero() {
        while current < stop {
            count += 1;
            current = current + step.clone();
        }
    } else {
        while current > stop {
            count += 1;
            current = current + step.clone();
        }
    }

    let mut data = Vec::with_capacity(count);
    current = start;

    if step > T::zero() {
        while current < stop {
            data.push(current.clone());
            current = current + step.clone();
        }
    } else {
        while current > stop {
            data.push(current.clone());
            current = current + step.clone();
        }
    }

    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: vec![count],
        strides: compute_strides(&[count]),
        dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
        offset: 0,
    })
}

/// Create evenly spaced numbers over a specified interval
///
/// # Arguments
/// * `start` - The starting value of the sequence
/// * `stop` - The end value of the sequence
/// * `num` - Number of samples to generate (must be non-zero)
/// * `endpoint` - If true, stop is the last sample (default: true)
/// * `dtype` - Optional data type (defaults to appropriate default)
///
/// # Returns
/// 1-D array of evenly spaced samples
pub fn linspace<T>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
    dtype: Option<Dtype>,
) -> Result<Array<T>>
where
    T: Clone + Default + Float + Num + 'static,
{
    if num == 0 {
        return Ok(Array {
            data: std::sync::Arc::new(MemoryManager::from_vec(vec![])),
            shape: vec![0],
            strides: compute_strides(&[0]),
            dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
            offset: 0,
        });
    }

    if num == 1 {
        let data = vec![if endpoint { stop } else { start }];
        let memory_manager = MemoryManager::from_vec(data);

        return Ok(Array {
            data: std::sync::Arc::new(memory_manager),
            shape: vec![1],
            strides: compute_strides(&[1]),
            dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
            offset: 0,
        });
    }

    let div = if endpoint { num - 1 } else { num };
    let step = (stop - start) / T::from(div).unwrap();

    let mut data = Vec::with_capacity(num);
    for i in 0..num {
        let value = start + step * T::from(i).unwrap();
        data.push(value);
    }

    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: vec![num],
        strides: compute_strides(&[num]),
        dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
        offset: 0,
    })
}

/// Create numbers spaced evenly on a log scale
///
/// # Arguments
/// * `start` - Base ** start
/// * `stop` - Base ** stop
/// * `num` - Number of samples to generate (must be non-zero)
/// * `endpoint` - If true, stop is the last sample (default: true)
/// * `base` - The base of the log space (default: 10.0)
/// * `dtype` - Optional data type (defaults to appropriate default)
///
/// # Returns
/// 1-D array with equally spaced values on a log scale
pub fn logspace<T>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
    _base: T,
    dtype: Option<Dtype>,
) -> Result<Array<T>>
where
    T: Clone + Default + Float + Num + 'static,
{
    let start_exp = start;
    let stop_exp = stop;

    let lin_array = linspace(
        start_exp,
        stop_exp,
        num,
        endpoint,
        Some(Dtype::from_type::<T>()),
    )?;
    let lin_data = lin_array.to_vec();
    let log_data: Vec<T> = lin_data.into_iter().map(|x| x.exp()).collect();

    let memory_manager = MemoryManager::from_vec(log_data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: vec![num],
        strides: compute_strides(&[num]),
        dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
        offset: 0,
    })
}

/// Create numbers spaced evenly on a log scale (geometric progression)
///
/// # Arguments
/// * `start` - Starting value
/// * `stop` - End value
/// * `num` - Number of samples to generate (must be non-zero)
/// * `endpoint` - If true, stop is the last sample (default: true)
/// * `dtype` - Optional data type (defaults to appropriate default)
///
/// # Returns
/// 1-D array with geometric progression values
pub fn geomspace<T>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
    dtype: Option<Dtype>,
) -> Result<Array<T>>
where
    T: Clone + Default + Float + Num + 'static,
{
    if start == T::zero() || stop == T::zero() {
        return Err(NumPyError::value_error(
            "start and stop must be non-zero for geomspace",
            "geomspace",
        ));
    }

    if num == 0 {
        return Ok(Array {
            data: std::sync::Arc::new(MemoryManager::from_vec(vec![])),
            shape: vec![0],
            strides: compute_strides(&[0]),
            dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
            offset: 0,
        });
    }

    if num == 1 {
        let data = vec![if endpoint { stop } else { start }];
        let memory_manager = MemoryManager::from_vec(data);

        return Ok(Array {
            data: std::sync::Arc::new(memory_manager),
            shape: vec![1],
            strides: compute_strides(&[1]),
            dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
            offset: 0,
        });
    }

    let div = if endpoint { num - 1 } else { num };
    let exponent = T::one() / T::from(div).unwrap();
    let ratio = (stop / start).powf(exponent);

    let mut data = Vec::with_capacity(num);
    for i in 0..num {
        let value = start * ratio.powf(T::from(i).unwrap());
        data.push(value);
    }

    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: vec![num],
        strides: compute_strides(&[num]),
        dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
        offset: 0,
    })
}

/// Return coordinate matrices from coordinate vectors
///
/// # Arguments
/// * `arrays` - One or more 1-D arrays representing coordinates
/// * `indexing` - Indexing mode: 'xy' (Cartesian) or 'ij' (matrix)
///
/// # Returns
/// Vector of coordinate matrices
pub fn meshgrid<T>(arrays: &[&Array<T>], indexing: &str) -> Result<Vec<Array<T>>>
where
    T: Clone + Default + Num + 'static,
{
    if indexing != "xy" && indexing != "ij" {
        return Err(NumPyError::invalid_operation(
            "meshgrid() indexing must be 'xy' or 'ij'",
        ));
    }

    if arrays.is_empty() {
        return Ok(vec![]);
    }

    for (i, arr) in arrays.iter().enumerate() {
        if arr.ndim() != 1 {
            return Err(NumPyError::invalid_operation(format!(
                "meshgrid() array {} must be 1-dimensional",
                i
            )));
        }
    }

    let ndim = arrays.len();
    let mut sizes = Vec::new();

    for arr in arrays.iter() {
        let size = arr.shape()[0];
        sizes.push(size);
    }

    if indexing == "xy" && ndim >= 2 {
        sizes.swap(0, 1);
    }

    let mut result = Vec::new();

    for (i, &arr) in arrays.iter().enumerate() {
        let coord_size = arr.shape()[0];
        let mut shape = vec![1; ndim];

        let idx = if indexing == "xy" && ndim >= 2 {
            if i == 0 {
                1
            } else if i == 1 {
                0
            } else {
                i
            }
        } else {
            i
        };

        shape[idx] = coord_size;

        let mut data = Vec::with_capacity(compute_size(&shape));
        let coord_data = arr.to_vec();

        generate_meshgrid_data(&mut data, &shape, &coord_data, idx);

        let memory_manager = MemoryManager::from_vec(data);

        result.push(Array {
            data: std::sync::Arc::new(memory_manager),
            shape: shape.clone(),
            strides: compute_strides(&shape),
            dtype: arr.dtype().clone(),
            offset: 0,
        });
    }

    Ok(result)
}

/// Helper function to generate meshgrid data for a specific coordinate dimension
fn generate_meshgrid_data<T>(data: &mut Vec<T>, shape: &[usize], coord: &[T], coord_dim: usize)
where
    T: Clone + Default,
{
    if shape.is_empty() {
        return;
    }

    if shape.len() == 1 {
        data.extend_from_slice(coord);
        return;
    }

    let chunk_size = shape[coord_dim + 1..].iter().product();
    let repeat_count = shape[..coord_dim].iter().product();

    for _ in 0..repeat_count {
        for _ in 0..chunk_size {
            data.extend_from_slice(coord);
        }
    }
}

// ==================== SHAPE MANIPULATION FUNCTIONS ====================

/// Give a new shape to an array without changing its data
///
/// # Arguments
/// * `a` - Array to reshape
/// * `newshape` - New shape for the array
/// * `order` - Memory layout order ('C' for row-major, 'F' for column-major, default 'C')
///
/// # Returns
/// Array with new shape (may share data with original)
pub fn reshape<T>(a: &Array<T>, newshape: &[usize], order: &str) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if order != "C" && order != "F" && order != "c" && order != "f" {
        return Err(NumPyError::invalid_operation(
            "reshape() order must be 'C' or 'F'",
        ));
    }

    validate_reshape(a.shape(), newshape)?;

    let strides = if order.to_uppercase() == "F" {
        compute_fortran_strides(newshape)
    } else {
        compute_strides(newshape)
    };

    Ok(Array {
        data: a.data.clone(),
        shape: newshape.to_vec(),
        strides,
        dtype: a.dtype().clone(),
        offset: a.offset,
    })
}

/// Return a contiguous flattened array
///
/// # Arguments
/// * `a` - Array to flatten
/// * `order` - Memory layout order ('C' for row-major, 'F' for column-major, default 'C')
///
/// # Returns
/// 1-D array containing all elements of the input
pub fn ravel<T>(a: &Array<T>, order: &str) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if order != "C" && order != "F" && order != "c" && order != "f" {
        return Err(NumPyError::invalid_operation(
            "ravel() order must be 'C' or 'F'",
        ));
    }

    let size = compute_size(a.shape());
    let mut data = Vec::with_capacity(size);
    let order = order.to_uppercase();
    if order == "C" {
        for idx in 0..size {
            if let Some(value) = a.get_linear(idx) {
                data.push(value.clone());
            }
        }
    } else {
        for idx in 0..size {
            let indices = indices_from_fortran(idx, a.shape());
            let linear = linear_from_indices(&indices, a.shape());
            if let Some(value) = a.get_linear(linear) {
                data.push(value.clone());
            }
        }
    }
    let memory_manager = MemoryManager::from_vec(data);

    Ok(Array {
        data: std::sync::Arc::new(memory_manager),
        shape: vec![size],
        strides: compute_strides(&[size]),
        dtype: a.dtype().clone(),
        offset: 0,
    })
}

/// Return a copy of the array collapsed into one dimension
///
/// # Arguments
/// * `a` - Array to flatten
/// * `order` - Memory layout order ('C' for row-major, 'F' for column-major, default 'C')
///
/// # Returns
/// 1-D copy of the input array
pub fn flatten<T>(a: &Array<T>, order: &str) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    ravel(a, order)
}

/// Repeat elements of an array
///
/// # Arguments
/// * `a` - Array to repeat
/// * `repeats` - Number of repetitions for each element
/// * `axis` - Optional axis along which to repeat
pub fn repeat<T>(a: &Array<T>, repeats: usize, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    crate::advanced_broadcast::repeat(a, repeats, axis)
}

/// Construct an array by repeating `a` the number of times given by `reps`
///
/// # Arguments
/// * `a` - Array to tile
/// * `reps` - Repetitions for each axis
pub fn tile<T>(a: &Array<T>, reps: &[usize]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    crate::advanced_broadcast::tile(a, reps)
}

/// Interchange two axes of an array
pub fn swapaxes<T>(a: &Array<T>, axis1: isize, axis2: isize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = a.ndim();
    let axis1 = normalize_axis(axis1, ndim)?;
    let axis2 = normalize_axis(axis2, ndim)?;
    if axis1 == axis2 {
        return Ok(a.clone());
    }

    let mut shape = a.shape().to_vec();
    let mut strides = a.strides().to_vec();
    shape.swap(axis1, axis2);
    strides.swap(axis1, axis2);

    Ok(Array {
        data: a.data.clone(),
        shape,
        strides,
        dtype: a.dtype().clone(),
        offset: a.offset,
    })
}

/// Roll the specified axis backwards until it lies in a given position
pub fn rollaxis<T>(a: &Array<T>, axis: isize, start: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = a.ndim();
    let axis = normalize_axis(axis, ndim)?;
    let mut start = start.unwrap_or(0);
    if start < 0 {
        start += ndim as isize;
    }
    if start < 0 || start as usize > ndim {
        return Err(NumPyError::index_error(start as usize, ndim));
    }
    moveaxis(a, &[axis as isize], &[start])
}

/// Move axes of an array to new positions
pub fn moveaxis<T>(a: &Array<T>, source: &[isize], destination: &[isize]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if source.len() != destination.len() {
        return Err(NumPyError::invalid_operation(
            "moveaxis requires source and destination to be the same length",
        ));
    }
    let ndim = a.ndim();
    let source = normalize_axes(source, ndim)?;
    let destination: Vec<usize> = destination
        .iter()
        .map(|&axis| {
            if axis < 0 {
                (axis + ndim as isize) as usize
            } else {
                axis as usize
            }
        })
        .collect();

    let mut seen = std::collections::HashSet::new();
    for &axis in &destination {
        if axis > ndim {
            return Err(NumPyError::index_error(axis, ndim));
        }
        if !seen.insert(axis) {
            return Err(NumPyError::invalid_operation("duplicate destination axis"));
        }
    }

    let mut remaining: Vec<usize> = (0..ndim).filter(|ax| !source.contains(ax)).collect();
    let mut order: Vec<usize> = Vec::with_capacity(ndim);
    let mut pairs: Vec<(usize, usize)> = source
        .iter()
        .copied()
        .zip(destination.iter().copied())
        .collect();
    pairs.sort_by_key(|(_, dst)| *dst);

    for (axis, dst) in pairs {
        if dst >= remaining.len() {
            remaining.push(axis);
        } else {
            remaining.insert(dst, axis);
        }
    }
    order.append(&mut remaining);

    let mut shape = Vec::with_capacity(ndim);
    let mut strides = Vec::with_capacity(ndim);
    for axis in order {
        shape.push(a.shape()[axis]);
        strides.push(a.strides()[axis]);
    }

    Ok(Array {
        data: a.data.clone(),
        shape,
        strides,
        dtype: a.dtype().clone(),
        offset: a.offset,
    })
}

/// Remove single-dimensional entries from the shape of an array
///
/// # Arguments
/// * `a` - Array to squeeze
/// * `axis` - Optional axis or axes to remove (None removes all size-1 dimensions)
///
/// # Returns
/// Array with specified dimensions removed
pub fn squeeze<T>(a: &Array<T>, axis: Option<&[isize]>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = a.ndim();

    if ndim == 0 {
        return Ok(a.clone());
    }

    let mut new_shape = a.shape().to_vec();
    let mut new_strides = a.strides().to_vec();

    match axis {
        Some(axes) => {
            let mut axes = normalize_axes(axes, ndim)?;
            axes.sort_unstable_by(|a, b| b.cmp(a));
            for ax in axes {
                if a.shape()[ax] != 1 {
                    return Err(NumPyError::invalid_operation(format!(
                        "squeeze() cannot select axis {} with size {}",
                        ax,
                        a.shape()[ax]
                    )));
                }
                new_shape.remove(ax);
                new_strides.remove(ax);
            }
        }
        None => {
            let mut i = 0;
            while i < new_shape.len() {
                if new_shape[i] == 1 {
                    new_shape.remove(i);
                    new_strides.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    Ok(Array {
        data: a.data.clone(),
        shape: new_shape,
        strides: new_strides,
        dtype: a.dtype().clone(),
        offset: a.offset,
    })
}

/// Expand the shape of an array
///
/// Inserts a new axis at the specified position, increasing the number of dimensions.
///
/// # Arguments
/// * `a` - Input array
/// * `axis` - Position in the expanded axes where the new axis is placed.
///            Can be negative (counts from the end).
///
/// # Returns
/// Array with expanded dimensions
///
/// # Examples
/// ```ignore
/// let a = Array::from_vec(vec![1, 2, 3, 4]);
/// let expanded = expand_dims(&a, 0)?;
/// assert_eq!(expanded.shape(), &[1, 4]);
///
/// let expanded2 = expand_dims(&a, -1)?;
/// assert_eq!(expanded2.shape(), &[4, 1]);
/// ```
pub fn expand_dims<T>(a: &Array<T>, axis: isize) -> Result<Array<T>>
where
    T: Clone + 'static,
{
    let ndim = a.ndim();

    // Normalize axis (allow negative values)
    let axis_norm = if axis < 0 {
        let abs_axis = axis.abs() as usize;
        if abs_axis > ndim {
            return Err(NumPyError::invalid_operation(format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, ndim
            )));
        }
        ndim - abs_axis + 1
    } else {
        let axis_usize = axis as usize;
        if axis_usize > ndim {
            return Err(NumPyError::invalid_operation(format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, ndim
            )));
        }
        axis_usize
    };

    // Insert new dimension of size 1 at the specified position
    let mut new_shape = a.shape().to_vec();
    let mut new_strides = a.strides().to_vec();

    new_shape.insert(axis_norm, 1);
    new_strides.insert(axis_norm, 0); // Zero stride broadcasts the single element

    Ok(Array {
        data: a.data.clone(),
        shape: new_shape,
        strides: new_strides,
        dtype: a.dtype().clone(),
        offset: a.offset,
    })
}

/// Ensure arrays have at least 1 dimension
///
/// # Arguments
/// * `arrays` - Arrays to process
///
/// # Returns
/// Single array with at least 1 dimension (for single input) or first array
pub fn atleast_1d<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_operation(
            "atleast_1d() requires at least one array",
        ));
    }

    let a = arrays[0];

    if a.ndim() == 0 {
        let data = vec![a.get_linear(0).unwrap().clone()];
        let memory_manager = MemoryManager::from_vec(data);

        Ok(Array {
            data: std::sync::Arc::new(memory_manager),
            shape: vec![1],
            strides: compute_strides(&[1]),
            dtype: a.dtype().clone(),
            offset: 0,
        })
    } else {
        Ok(a.clone())
    }
}

/// Ensure arrays have at least 2 dimensions
///
/// # Arguments
/// * `arrays` - Arrays to process
///
/// # Returns
/// Single array with at least 2 dimensions (for single input) or first array
pub fn atleast_2d<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_operation(
            "atleast_2d() requires at least one array",
        ));
    }

    let a = arrays[0];

    match a.ndim() {
        0 => {
            let data = vec![a.get_linear(0).unwrap().clone()];
            let memory_manager = MemoryManager::from_vec(data);

            Ok(Array {
                data: std::sync::Arc::new(memory_manager),
                shape: vec![1, 1],
                strides: compute_strides(&[1, 1]),
                dtype: a.dtype().clone(),
                offset: 0,
            })
        }
        1 => {
            let data = a.to_vec();
            let memory_manager = MemoryManager::from_vec(data);

            Ok(Array {
                data: std::sync::Arc::new(memory_manager),
                shape: vec![1, a.shape()[0]],
                strides: compute_strides(&[1, a.shape()[0]]),
                dtype: a.dtype().clone(),
                offset: 0,
            })
        }
        _ => Ok(a.clone()),
    }
}

/// Ensure arrays have at least 3 dimensions
///
/// # Arguments
/// * `arrays` - Arrays to process
///
/// # Returns
/// Single array with at least 3 dimensions (for single input) or first array
pub fn atleast_3d<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_operation(
            "atleast_3d() requires at least one array",
        ));
    }

    let a = arrays[0];

    match a.ndim() {
        0 => {
            let data = vec![a.get_linear(0).unwrap().clone()];
            let memory_manager = MemoryManager::from_vec(data);

            Ok(Array {
                data: std::sync::Arc::new(memory_manager),
                shape: vec![1, 1, 1],
                strides: compute_strides(&[1, 1, 1]),
                dtype: a.dtype().clone(),
                offset: 0,
            })
        }
        1 => {
            let data = a.to_vec();
            let memory_manager = MemoryManager::from_vec(data);

            Ok(Array {
                data: std::sync::Arc::new(memory_manager),
                shape: vec![1, 1, a.shape()[0]],
                strides: compute_strides(&[1, 1, a.shape()[0]]),
                dtype: a.dtype().clone(),
                offset: 0,
            })
        }
        2 => {
            let data = a.to_vec();
            let memory_manager = MemoryManager::from_vec(data);

            Ok(Array {
                data: std::sync::Arc::new(memory_manager),
                shape: vec![1, a.shape()[0], a.shape()[1]],
                strides: compute_strides(&[1, a.shape()[0], a.shape()[1]]),
                dtype: a.dtype().clone(),
                offset: 0,
            })
        }
        _ => Ok(a.clone()),
    }
}

/// Reverse the order of elements in an array along the given axis
///
/// # Arguments
/// * `a` - Input array
/// * `axis` - Axis or axes along which to flip over. The default, axis=None, will flip over all axes.
///
/// # Returns
/// A view of the array with flipped elements
///
/// # Examples
/// ```ignore
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let flipped = flip(&a, None)?;
/// // Result: [4.0, 3.0, 2.0, 1.0]
/// ```
pub fn flip<T>(a: &Array<T>, axis: Option<&[isize]>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = a.ndim();

    if ndim == 0 {
        return Ok(a.clone());
    }

    // If axis is None, flip all axes
    let axes_to_flip: Vec<usize> = match axis {
        Some(axes) => {
            let normalized = normalize_axes(axes, ndim)?;
            if normalized.is_empty() {
                (0..ndim).collect()
            } else {
                normalized
            }
        }
        None => (0..ndim).collect(),
    };

    // Compute new shape and strides for flipped view
    let new_shape = a.shape().to_vec();
    let mut new_strides = a.strides().to_vec();
    let mut new_offset = a.offset;

    for &ax in &axes_to_flip {
        if new_shape[ax] > 0 {
            new_offset += (new_shape[ax] - 1) as usize * new_strides[ax] as usize;
            new_strides[ax] = -new_strides[ax];
        }
    }

    Ok(Array {
        data: a.data.clone(),
        shape: new_shape,
        strides: new_strides,
        dtype: a.dtype().clone(),
        offset: new_offset,
    })
}

/// Roll array elements along a given axis
///
/// # Arguments
/// * `a` - Input array
/// * `shift` - Number of places by which elements are shifted
/// * `axis` - Axis along which elements are shifted (None for flattened array)
///
/// # Returns
/// New array with rolled elements
///
/// # Examples
/// ```ignore
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let rolled = roll(&a, 2, None)?;
/// // Result: [3.0, 4.0, 1.0, 2.0]
/// ```
pub fn roll<T>(a: &Array<T>, shift: isize, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    // Flatten if axis is None
    let axis_idx = match axis {
        Some(ax) => Some(normalize_axis(ax, a.ndim())?),
        None => None,
    };

    let data = a.to_vec();

    if let Some(ax) = axis_idx {
        // Roll along specific axis
        let dim_size = a.shape()[ax];
        if dim_size == 0 {
            return Ok(a.clone());
        }

        let normalized_shift = if shift < 0 {
            dim_size - ((-shift) as usize % dim_size)
        } else {
            (shift as usize) % dim_size
        };

        if normalized_shift == 0 {
            return Ok(a.clone());
        }

        // Roll along axis by creating new data
        let mut rolled_data = Vec::with_capacity(data.len());
        let stride = a.shape()[ax + 1..].iter().product::<usize>();
        let block_size = stride;
        let num_blocks = data.len() / (dim_size * block_size);

        for block in 0..num_blocks {
            for i in 0..dim_size {
                let src_idx =
                    ((i as isize + normalized_shift as isize) % dim_size as isize) as usize;
                for j in 0..block_size {
                    let src_pos = block * dim_size * block_size + src_idx * block_size + j;
                    rolled_data.push(data[src_pos].clone());
                }
            }
        }

        let memory_manager = MemoryManager::from_vec(rolled_data);
        Ok(Array {
            data: std::sync::Arc::new(memory_manager),
            shape: a.shape().to_vec(),
            strides: compute_strides(a.shape()),
            dtype: a.dtype().clone(),
            offset: 0,
        })
    } else {
        // Roll flattened array
        let len = data.len();
        if len == 0 {
            return Ok(a.clone());
        }

        let normalized_shift = if shift < 0 {
            len - ((-shift) as usize % len)
        } else {
            (shift as usize) % len
        };

        if normalized_shift == 0 {
            return Ok(a.clone());
        }

        let mut rolled_data = Vec::with_capacity(len);
        for i in 0..len {
            rolled_data.push(
                data[((i as isize + normalized_shift as isize) % len as isize) as usize].clone(),
            );
        }

        let memory_manager = MemoryManager::from_vec(rolled_data);
        Ok(Array {
            data: std::sync::Arc::new(memory_manager),
            shape: vec![len],
            strides: vec![1],
            dtype: a.dtype().clone(),
            offset: 0,
        })
    }
}

/// Rotate an array by 90 degrees in the plane specified by axes
///
/// # Arguments
/// * `a` - Input array (must be 2D)
/// * `k` - Number of times to rotate by 90 degrees (default 1)
/// * `axes` - Array of two axes that define the plane of rotation (default [0, 1])
///
/// # Returns
/// Rotated array
///
/// # Examples
/// ```ignore
/// let a = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let rotated = rot90(&a, Some(1), None)?;
/// // Result: [[2.0, 4.0], [1.0, 3.0]]
/// ```
pub fn rot90<T>(a: &Array<T>, k: Option<isize>, axes: Option<[isize; 2]>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if a.ndim() < 2 {
        return Err(NumPyError::invalid_operation(
            "rot90 requires at least 2D array",
        ));
    }

    let ndim = a.ndim();
    let mut axes_arr = axes.unwrap_or([0, 1]);
    let k = k.unwrap_or(1);

    // Normalize axes
    if axes_arr[0] < 0 {
        axes_arr[0] += ndim as isize;
    }
    if axes_arr[1] < 0 {
        axes_arr[1] += ndim as isize;
    }

    if axes_arr[0] < 0
        || axes_arr[0] >= ndim as isize
        || axes_arr[1] < 0
        || axes_arr[1] >= ndim as isize
    {
        return Err(NumPyError::index_error(ndim, ndim));
    }

    if axes_arr[0] == axes_arr[1] {
        return Err(NumPyError::invalid_operation("axes must be different"));
    }

    // Normalize k to [0, 3]
    let k = ((k % 4 + 4) % 4) as usize;

    if k == 0 {
        return Ok(a.clone());
    }

    // Get the data as a 2D slice along the rotation axes
    let data = a.to_vec();

    // For 2D arrays, perform direct rotation
    if ndim == 2 {
        let rows = a.shape()[0];
        let cols = a.shape()[1];

        // For 2D arrays, we need to compute the final rotated array
        // Start with original data
        let mut current_data = data.clone();
        let mut current_rows = rows;
        let mut current_cols = cols;

        for _rot in 0..k {
            let mut rotated = vec![T::default(); current_data.len()];
            for i in 0..current_rows {
                for j in 0..current_cols {
                    rotated[j * current_rows + (current_rows - 1 - i)] =
                        current_data[i * current_cols + j].clone();
                }
            }
            current_data = rotated;
            std::mem::swap(&mut current_rows, &mut current_cols);
        }

        let memory_manager = MemoryManager::from_vec(current_data);
        return Ok(Array {
            data: std::sync::Arc::new(memory_manager),
            shape: vec![current_rows, current_cols],
            strides: compute_strides(&[current_rows, current_cols]),
            dtype: a.dtype().clone(),
            offset: 0,
        });
    }

    // For multi-dimensional arrays, use transpose + flip
    let mut result = a.clone();

    for _ in 0..k {
        // Swap the two axes
        result = swapaxes(&result, axes_arr[0], axes_arr[1])?;

        // Reverse along the first of the swapped axes
        result = flip(&result, Some(&[axes_arr[1]]))?;
    }

    Ok(result)
}

// ==================== ARRAY ELEMENT INSERTION/DELETION FUNCTIONS ====================

/// Insert values along the given axis before the given indices
///
/// # Arguments
/// * `arr` - Input array
/// * `obj` - Index or indices before which to insert values (can be int, slice, or array)
/// * `values` - Values to insert
/// * `axis` - Axis along which to insert (None flattens first)
///
/// # Returns
/// New array with values inserted
pub fn insert<T>(
    arr: &Array<T>,
    obj: &Array<isize>,
    values: &Array<T>,
    axis: Option<isize>,
) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if axis.is_none() {
        // Flatten both arrays and insert
        let flat_arr = flatten(arr, "C")?;
        let flat_values = flatten(values, "C")?;
        return insert_1d(&flat_arr, obj, &flat_values);
    }

    let axis = normalize_axis(axis.unwrap(), arr.ndim())?;
    insert_along_axis(arr, obj, values, axis)
}

/// Delete sub-arrays along the given axis
///
/// # Arguments
/// * `arr` - Input array
/// * `obj` - Indices or slice indicating what to delete
/// * `axis` - Axis along which to delete (None flattens first)
///
/// # Returns
/// New array with specified elements deleted
pub fn delete<T>(arr: &Array<T>, obj: &Array<isize>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if axis.is_none() {
        // Flatten and delete
        let flat_arr = flatten(arr, "C")?;
        return delete_1d(&flat_arr, obj);
    }

    let axis = normalize_axis(axis.unwrap(), arr.ndim())?;
    delete_along_axis(arr, obj, axis)
}

/// Append values to the end of an array
///
/// # Arguments
/// * `arr` - Input array
/// * `values` - Values to append
/// * `axis` - Axis along which to append (None flattens first)
///
/// # Returns
/// New array with values appended
pub fn append<T>(arr: &Array<T>, values: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if axis.is_none() {
        // Flatten both and concatenate
        let flat_arr = flatten(arr, "C")?;
        let flat_values = flatten(values, "C")?;
        return concatenate_1d(&flat_arr, &flat_values);
    }

    let axis = normalize_axis(axis.unwrap(), arr.ndim())?;
    append_along_axis(arr, values, axis)
}

// ==================== HELPER FUNCTIONS FOR INSERT/DELETE/APPEND ====================

fn insert_1d<T>(arr: &Array<T>, indices: &Array<isize>, values: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let arr_data = arr.to_vec();
    let values_data = values.to_vec();
    let indices_vec = indices.to_vec();

    // Normalize indices (handle negative values)
    let mut normalized_indices = Vec::new();
    for &idx in &indices_vec {
        let norm_idx = if idx < 0 {
            (arr_data.len() as isize + idx).max(0) as usize
        } else {
            idx.min(arr_data.len() as isize) as usize
        };
        normalized_indices.push(norm_idx);
    }

    // Sort and deduplicate indices
    normalized_indices.sort();
    normalized_indices.dedup();

    // Calculate new size
    let new_size = arr_data.len() + values_data.len();
    let mut result = Vec::with_capacity(new_size);

    let mut arr_idx = 0;
    let mut values_idx = 0;

    for insert_idx in &normalized_indices {
        // Copy elements from arr up to insert point
        while arr_idx < *insert_idx && arr_idx < arr_data.len() {
            result.push(arr_data[arr_idx].clone());
            arr_idx += 1;
        }

        // Insert value if available
        if values_idx < values_data.len() {
            result.push(values_data[values_idx].clone());
            values_idx += 1;
        }
    }

    // Copy remaining elements from arr
    while arr_idx < arr_data.len() {
        result.push(arr_data[arr_idx].clone());
        arr_idx += 1;
    }

    // Insert any remaining values
    while values_idx < values_data.len() {
        result.push(values_data[values_idx].clone());
        values_idx += 1;
    }

    Ok(Array::from_shape_vec(vec![result.len()], result))
}

fn delete_1d<T>(arr: &Array<T>, indices: &Array<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let arr_data = arr.to_vec();
    let indices_vec = indices.to_vec();

    if arr_data.is_empty() {
        return Ok(Array::from_shape_vec(vec![0], vec![]));
    }

    // Normalize indices and build set of indices to delete
    let mut to_delete = std::collections::HashSet::new();
    for &idx in &indices_vec {
        let norm_idx = if idx < 0 {
            (arr_data.len() as isize + idx) as usize
        } else {
            idx as usize
        };
        if norm_idx < arr_data.len() {
            to_delete.insert(norm_idx);
        }
    }

    // Build result excluding deleted indices
    let result: Vec<T> = arr_data
        .iter()
        .enumerate()
        .filter(|(i, _)| !to_delete.contains(i))
        .map(|(_, v)| v.clone())
        .collect();

    Ok(Array::from_shape_vec(vec![result.len()], result))
}

fn concatenate_1d<T>(arr1: &Array<T>, arr2: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let mut data1 = arr1.to_vec();
    let data2 = arr2.to_vec();
    data1.extend(data2);

    Ok(Array::from_shape_vec(vec![data1.len()], data1))
}

fn insert_along_axis<T>(
    arr: &Array<T>,
    indices: &Array<isize>,
    values: &Array<T>,
    axis: usize,
) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let shape = arr.shape();
    let axis_size = shape[axis];

    // Calculate how many elements to insert along axis
    let indices_vec = indices.to_vec();
    let num_insertions = indices_vec.len();

    // Validate values shape
    if values.ndim() != arr.ndim() {
        return Err(NumPyError::shape_mismatch(
            vec![arr.ndim()],
            vec![values.ndim()],
        ));
    }

    // New shape along axis
    let mut new_shape = shape.to_vec();
    new_shape[axis] = axis_size + num_insertions;

    // Build result array
    let mut result_data = Vec::new();
    let arr_data = arr.to_vec();
    let values_data = values.to_vec();

    // For each position along axis, copy or insert
    let mut insert_positions = std::collections::HashSet::new();
    for &idx in &indices_vec {
        let norm_idx = if idx < 0 {
            (axis_size as isize + idx).max(0) as usize
        } else {
            idx.min(axis_size as isize) as usize
        };
        insert_positions.insert(norm_idx);
    }

    let mut values_idx = 0;
    for i in 0..=axis_size {
        if insert_positions.contains(&i) && values_idx < values_data.len() {
            // Insert values at this position
            result_data.extend_from_slice(
                &values_data[values_idx * arr.size() / axis_size
                    ..(values_idx + 1) * arr.size() / axis_size],
            );
            values_idx += 1;
        }
        if i < axis_size {
            // Copy elements from arr
            result_data.extend_from_slice(
                &arr_data[i * arr.size() / axis_size..(i + 1) * arr.size() / axis_size],
            );
        }
    }

    Ok(Array::from_shape_vec(new_shape, result_data))
}

fn delete_along_axis<T>(arr: &Array<T>, indices: &Array<isize>, axis: usize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let shape = arr.shape();
    let axis_size = shape[axis];

    // Normalize indices
    let indices_vec = indices.to_vec();
    let mut to_delete = std::collections::HashSet::new();
    for &idx in &indices_vec {
        let norm_idx = if idx < 0 {
            (axis_size as isize + idx) as usize
        } else {
            idx as usize
        };
        if norm_idx < axis_size {
            to_delete.insert(norm_idx);
        }
    }

    // New shape along axis
    let new_axis_size = axis_size - to_delete.len();
    let mut new_shape = shape.to_vec();
    new_shape[axis] = new_axis_size;

    // Build result array excluding deleted indices
    let arr_data = arr.to_vec();
    let mut result_data = Vec::new();

    for i in 0..axis_size {
        if !to_delete.contains(&i) {
            result_data.extend_from_slice(
                &arr_data[i * arr.size() / axis_size..(i + 1) * arr.size() / axis_size],
            );
        }
    }

    Ok(Array::from_shape_vec(new_shape, result_data))
}

fn append_along_axis<T>(arr: &Array<T>, values: &Array<T>, axis: usize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let shape = arr.shape();
    let values_shape = values.shape();

    // Validate shapes match except on append axis
    if arr.ndim() != values.ndim() {
        return Err(NumPyError::shape_mismatch(
            vec![arr.ndim()],
            vec![values.ndim()],
        ));
    }

    for (i, (&s1, &s2)) in shape.iter().zip(values_shape.iter()).enumerate() {
        if i != axis && s1 != s2 {
            return Err(NumPyError::shape_mismatch(
                shape.to_vec(),
                values_shape.to_vec(),
            ));
        }
    }

    // Calculate new shape
    let mut new_shape = shape.to_vec();
    new_shape[axis] = shape[axis] + values_shape[axis];

    // Build result by iterating through all dimensions
    let arr_data = arr.to_vec();
    let values_data = values.to_vec();
    let mut result_data = Vec::with_capacity(arr_data.len() + values_data.len());

    // Calculate stride for the target axis
    let mut stride_before_axis = 1;
    for i in 0..axis {
        stride_before_axis *= shape[i];
    }
    let stride_at_axis = shape[axis];
    let mut stride_after_axis = 1;
    for i in (axis + 1)..shape.len() {
        stride_after_axis *= shape[i];
    }

    // For each slice along the axis
    for before_idx in 0..stride_before_axis {
        let base_arr = before_idx * stride_at_axis * stride_after_axis;
        let base_values = before_idx * values_shape[axis] * stride_after_axis;

        // Append arr elements for this slice
        for axis_idx in 0..stride_at_axis {
            let offset = (base_arr + axis_idx * stride_after_axis) as usize;
            result_data.extend_from_slice(
                &arr_data[offset..offset + stride_after_axis]
            );
        }

        // Append values elements for this slice
        for axis_idx in 0..values_shape[axis] {
            let offset = (base_values + axis_idx * stride_after_axis) as usize;
            result_data.extend_from_slice(
                &values_data[offset..offset + stride_after_axis]
            );
        }
    }

    Ok(Array::from_shape_vec(new_shape, result_data))
}

// ==================== PUBLIC EXPORTS ====================

/// Re-export all array manipulation functions for public use
pub mod exports {
    pub use super::{
        append, arange, atleast_1d, atleast_2d, atleast_3d, delete, empty_like, expand_dims, eye,
        flatten, flip, full_like, geomspace, identity, insert, linspace, logspace, meshgrid,
        moveaxis, ones_like, ravel, repeat, reshape, roll, rollaxis, rot90, squeeze, swapaxes,
        tile, zeros_like,
    };
}
// normalize_axis replaced by internal version above
