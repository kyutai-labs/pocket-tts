// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! NumPy-compatible array implementation

use std::fmt;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::NumPyError;
use crate::memory::MemoryManager;

/// Main array structure
#[derive(Debug)]
pub struct Array<T> {
    pub data: Arc<MemoryManager<T>>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: Dtype,
    pub offset: usize,
}

// Methods requiring bounds (constructors, etc.)
impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    /// Create array from data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides = crate::strides::compute_strides(&shape);
        let memory_manager = Arc::new(MemoryManager::from_vec(data));

        Self {
            data: memory_manager,
            shape,
            strides,
            dtype: Dtype::from_type::<T>(),
            offset: 0,
        }
    }

    /// Create 2D array from matrix
    pub fn from_array2(array2: ndarray::Array2<T>) -> Self
    where
        T: 'static,
    {
        let shape = array2.shape().to_vec();
        let strides = crate::strides::compute_strides(&shape);
        let memory_manager = Arc::new(MemoryManager::from_vec(array2.into_raw_vec()));
        Self {
            data: memory_manager,
            shape,
            strides,
            dtype: Dtype::from_type::<T>(),
            offset: 0,
        }
    }

    /// Matrix multiplication
    pub fn dot(&self, other: &Array<T>) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + ndarray::LinalgScalar + 'static,
    {
        let a = self.to_ndarray2()?;
        let b = other.to_ndarray2()?;
        let res = a.dot(&b);
        Ok(Array::from_array2(res))
    }

    /// Broadcast array to new shape
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, NumPyError> {
        crate::broadcasting::broadcast_to(self, shape)
    }
    /// Reshape array
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        crate::slicing::reshape(self, new_shape.to_vec())
    }

    /// Create array from shape and vector
    pub fn from_shape_vec(shape: Vec<usize>, data: Vec<T>) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        let size: usize = shape.iter().product();
        if data.len() != size {
            return Err(NumPyError::shape_mismatch(vec![size], vec![data.len()]));
        }
        Ok(Self::from_data(data, shape))
    }
}

// Methods NOT requiring bounds (accessors, iterators)
impl<T> Array<T> {
    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get array strides
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get array dtype
    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    /// Get array size (total elements)
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Get iterator over array elements
    pub fn iter(&self) -> crate::iterator::ArrayIter<'_, T> {
        crate::iterator::ArrayIter::new(self)
    }

    /// Get array data as slice
    pub fn as_slice(&self) -> &[T] {
        self.data.as_ref().as_slice()
    }

    /// Get array data as slice (alias for as_slice, for compatibility)
    pub fn data(&self) -> &[T] {
        self.as_slice()
    }

    /// Set element at linear index
    pub fn set_linear(&mut self, index: usize, value: T) {
        if index >= self.size() {
            return;
        }

        let real_index = if self.is_c_contiguous() {
            self.offset + index
        } else {
            let indices = crate::strides::compute_multi_indices(index, &self.shape);
            let relative_offset = crate::strides::compute_linear_index(&indices, &self.strides);
            self.offset.wrapping_add(relative_offset as usize)
        };

        if let Some(data) = Arc::get_mut(&mut self.data) {
            if let Some(elem) = data.get_mut(real_index) {
                *elem = value;
            }
        }
    }

    /// Set element at linear index with Result
    pub fn set(&mut self, index: usize, value: T) -> Result<(), NumPyError> {
        if index >= self.size() {
            return Err(NumPyError::index_error(index, self.size()));
        }
        self.set_linear(index, value);
        Ok(())
    }

    /// Set element at relative storage offset (internal use)
    pub fn set_storage_at(&mut self, relative_offset: isize, value: T) -> Result<(), NumPyError> {
        let real_index = self.offset.wrapping_add(relative_offset as usize);

        if let Some(data) = Arc::get_mut(&mut self.data) {
            if let Some(elem) = data.get_mut(real_index) {
                *elem = value;
                Ok(())
            } else {
                Err(NumPyError::index_error(real_index, self.data.len()))
            }
        } else {
            // CoW not implemented for generic types yet
            Err(NumPyError::invalid_operation(
                "Cannot mutate shared array data (CoW pending)",
            ))
        }
    }

    /// Convert array to Vec
    ///
    /// Returns a new Vec containing the array elements in logical order.
    /// This respects views and strides.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Get element at linear index
    pub fn get_linear(&self, index: usize) -> Option<&T> {
        if index >= self.size() {
            return None;
        }

        // Fast path for C-contiguous arrays (default layout)
        if self.is_c_contiguous() {
            return self.data.get(self.offset + index);
        }

        // General path for strided arrays / views
        // This is slow (O(N)) per element, but correct for all layouts
        let indices = crate::strides::compute_multi_indices(index, &self.shape);
        let relative_offset = crate::strides::compute_linear_index(&indices, &self.strides);
        let real_index = self.offset.wrapping_add(relative_offset as usize);

        self.data.get(real_index)
    }

    /// Get element at linear index (alias for get_linear)
    pub fn get(&self, index: usize) -> Option<&T> {
        self.get_linear(index)
    }

    /// Get element at relative storage offset (internal use)
    pub fn get_storage_at(&self, relative_offset: isize) -> Option<&T> {
        let real_index = self.offset.wrapping_add(relative_offset as usize);
        self.data.get(real_index)
    }

    /// Convert to ndarray 2D matrix
    pub fn to_ndarray2(&self) -> Result<ndarray::Array2<T>, NumPyError>
    where
        T: Clone,
    {
        if self.ndim() != 2 {
            return Err(NumPyError::invalid_operation(
                "to_ndarray2 requires 2D array",
            ));
        }

        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let data = self.data.as_ref().as_vec();

        // Create ndarray2 with proper shape

        let array2 = ndarray::Array2::from_shape_vec((rows, cols), data.to_vec())
            .map_err(|e| NumPyError::invalid_operation(e.to_string()))?;

        Ok(array2)
    }

    /// Transpose array (copies data - for view-based transpose use transpose_view)
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        if self.ndim() != 2 {
            // For higher dimensions, just return clone (proper transpose requires more work)
            return self.clone();
        }

        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let mut transposed_data = Vec::with_capacity(self.size());

        for i in 0..cols {
            for j in 0..rows {
                transposed_data.push(self.get_linear(j * cols + i).unwrap().clone());
            }
        }

        let new_shape = vec![cols, rows];
        let new_strides = compute_strides(&new_shape);
        let memory_manager = Arc::new(MemoryManager::from_vec(transposed_data));

        Self {
            data: memory_manager,
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype.clone(),
            offset: 0,
        }
    }

    /// Transpose array as a view (no copy) by permuting strides
    ///
    /// # Arguments
    /// * `axes` - Optional permutation of axes. If None, reverses all axes.
    ///
    /// # Returns
    /// A new array that shares the same data but with permuted shape and strides.
    pub fn transpose_view(&self, axes: Option<&[usize]>) -> Result<Self, NumPyError> {
        let ndim = self.ndim();

        // Determine the permutation
        let perm: Vec<usize> = if let Some(axes) = axes {
            if axes.len() != ndim {
                return Err(NumPyError::invalid_operation(format!(
                    "axes must have {} elements, got {}",
                    ndim,
                    axes.len()
                )));
            }
            // Validate axes
            let mut seen = vec![false; ndim];
            for &axis in axes {
                if axis >= ndim {
                    return Err(NumPyError::invalid_operation(format!(
                        "axis {} out of bounds for {}-dimensional array",
                        axis, ndim
                    )));
                }
                if seen[axis] {
                    return Err(NumPyError::invalid_operation("repeated axis in transpose"));
                }
                seen[axis] = true;
            }
            axes.to_vec()
        } else {
            // Default: reverse all axes
            (0..ndim).rev().collect()
        };

        // Permute shape and strides
        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&i| self.strides[i]).collect();

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype.clone(),
            offset: self.offset,
        })
    }

    /// Transpose array (alias for transpose)
    pub fn t(&self) -> Self
    where
        T: Clone,
    {
        self.transpose()
    }

    /// Check if array is C-contiguous
    pub fn is_c_contiguous(&self) -> bool {
        let mut expected_stride = 1isize;
        for (i, &dim) in self.shape.iter().enumerate().rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= dim as isize;
        }
        true
    }

    /// Check if array is Fortran-contiguous (column-major)
    pub fn is_f_contiguous(&self) -> bool {
        let mut expected_stride = 1isize;
        for (i, &dim) in self.shape.iter().enumerate() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= dim as isize;
        }
        true
    }

    /// Check if array is contiguous (either C or F)
    pub fn is_contiguous(&self) -> bool {
        self.is_c_contiguous() || self.is_f_contiguous()
    }
}

/// Display implementation
impl<T> fmt::Display for Array<T>
where
    T: fmt::Display + Clone + Default + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.ndim() == 1 {
            write!(f, "[")?;
            for (i, idx) in (0..self.size()).take(10).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if let Some(val) = self.get_linear(idx) {
                    write!(f, "{}", val)?;
                }
            }
            write!(f, "]")
        } else {
            write!(f, "Array(shape={:?}, dtype={:?})", self.shape, self.dtype)
        }
    }
}

/// Indexing implementation
impl<T> Index<usize> for Array<T>
where
    T: Clone + Default + 'static,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get_linear(index).expect("Index out of bounds")
    }
}

impl<T> IndexMut<usize> for Array<T>
where
    T: Clone + Default + 'static,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        Arc::make_mut(&mut self.data)
            .get_mut(index)
            .expect("Index out of bounds")
    }
}

// Note: Multidimensional indexing removed as [usize] is unsized
// Use get() or get_linear() methods instead

/// Convenience constructor functions
impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    /// Create array filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let data = vec![T::default(); shape.iter().product()];
        Self::from_data(data, shape)
    }

    /// Create array filled with ones
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: num_traits::One,
    {
        let data = vec![T::one(); shape.iter().product()];
        Self::from_data(data, shape)
    }

    /// Create empty array (uninitialized)
    pub fn empty(shape: Vec<usize>) -> Self {
        let data = Vec::with_capacity(shape.iter().product());
        Self::from_data(data, shape)
    }

    /// Create array filled with value
    pub fn full(shape: Vec<usize>, value: T) -> Self {
        let data = vec![value; shape.iter().product()];
        Self::from_data(data, shape)
    }

    /// Create 0-D array from scalar
    pub fn from_scalar(value: T, shape: Vec<usize>) -> Self {
        let data = vec![value];
        Self::from_data(data, shape)
    }

    /// Create array from vector (1D)
    pub fn from_vec(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        let strides = vec![1];
        let memory_manager = Arc::new(MemoryManager::from_vec(data));
        Self {
            data: memory_manager,
            shape,
            strides,
            dtype: Dtype::from_type::<T>(),
            offset: 0,
        }
    }

    /// Create identity matrix
    pub fn eye(size: usize) -> Self
    where
        T: num_traits::Zero + num_traits::One + Clone,
    {
        let mut data = vec![T::zero(); size * size];
        for i in 0..size {
            data[i * size + i] = T::one();
        }
        Self::from_data(data, vec![size, size])
    }
}

/// Compute strides from shape
pub fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;

    // Compute strides in reverse order
    for &dim in shape.iter().rev() {
        strides.push(stride as isize);
        stride *= dim;
    }

    // Reverse to get correct order
    strides.reverse();
    strides
}

impl<T> Clone for Array<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype.clone(),
            offset: self.offset,
        }
    }
}
