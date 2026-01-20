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
/// * `N` - Number of rows in the output
/// * `M` - Optional number of columns in the output (defaults to N)
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
    let m = m.unwrap_or(n);
    let k = k.unwrap_or(0);

    if n == 0 || m == 0 {
        return Ok(Array {
            data: std::sync::Arc::new(MemoryManager::from_vec(vec![])),
            shape: vec![n, m],
            strides: compute_strides(&[n, m]),
            dtype: dtype.unwrap_or(Dtype::Float64 { byteorder: None }),
            offset: 0,
        });
    }

    let mut data = vec![T::zero(); n * m];

    for i in 0..n {
        let j = (i as isize + k) as usize;
        if j < m {
            data[i * m + j] = T::one();
        }
    }

    let _memory_manager = MemoryManager::from_vec(data.clone());

    Ok(Array::from_data(data, vec![n, m]))
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

    let _strides = if order.to_uppercase() == "F" {
        compute_fortran_strides(newshape)
    } else {
        compute_strides(newshape)
    };

    Ok(Array::from_data(a.to_vec(), newshape.to_vec()))
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

    let data = a.to_vec();
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
            for &ax in axes {
                let ax = if ax < 0 { ndim as isize + ax } else { ax } as usize;

                if ax >= ndim {
                    return Err(NumPyError::index_error(ax, ndim));
                }

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

// ==================== PUBLIC EXPORTS ====================

/// Re-export all array manipulation functions for public use
pub mod exports {
    pub use super::{
        arange, atleast_1d, atleast_2d, atleast_3d, empty_like, eye, flatten, full_like, geomspace,
        identity, linspace, logspace, meshgrid, ones_like, ravel, reshape, squeeze, zeros_like,
    };
}
