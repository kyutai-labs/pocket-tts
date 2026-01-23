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

use num_complex::Complex64;

/// Main array structure
#[derive(Debug)]
pub struct Array<T> {
    pub data: Arc<MemoryManager<T>>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub dtype: Dtype,
    pub offset: usize,
}

impl<T> Array<T> {
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self
    where
        T: Clone + Default + 'static,
    {
        let strides = compute_strides(&shape);
        let memory_manager = Arc::new(MemoryManager::from_vec(data));
        Self {
            data: memory_manager,
            shape,
            strides,
            dtype: Dtype::from_type::<T>(),
            offset: 0,
        }
    }

    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get array strides
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Check if array is C-contiguous
    pub fn is_c_contiguous(&self) -> bool {
        let c_strides = compute_strides(&self.shape);
        self.strides == c_strides
    }

    /// Check if array is Fortran-contiguous
    pub fn is_f_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut stride = 1;
        let f_strides: Vec<isize> = self
            .shape
            .iter()
            .map(|&dim| {
                let s = stride as isize;
                stride *= dim;
                s
            })
            .collect();
        self.strides == f_strides
    }

    /// Check if array is contiguous (either C or Fortran order)
    pub fn is_contiguous(&self) -> bool {
        self.is_c_contiguous() || self.is_f_contiguous()
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
    ///
    /// Note: Returns a Vec by copying the array data.
    /// For non-consuming access, use as_slice() instead to avoid allocation.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        if self.is_c_contiguous() {
            let start = self.offset;
            let end = start + self.size();
            let data = self.data.as_ref().as_vec();
            if end <= data.len() {
                return data[start..end].to_vec();
            }
        }

        let mut result = Vec::with_capacity(self.size());
        for i in 0..self.size() {
            if let Some(val) = self.get_linear(i) {
                result.push(val.clone());
            }
        }
        result
    }

    /// Get array data as slice
    pub fn as_slice(&self) -> &[T] {
        self.data.as_ref().as_slice()
    }

    /// Get array data as slice (alias for as_slice, for compatibility)
    pub fn data(&self) -> &[T] {
        self.as_slice()
    }

    /// Get element at linear index
    pub fn get_linear(&self, index: usize) -> Option<&T> {
        if index >= self.size() {
            return None;
        }
        let indices = crate::strides::compute_multi_indices(index, &self.shape);
        let linear_offset = crate::strides::compute_linear_index(&indices, &self.strides);
        let physical_idx = (self.offset as isize + linear_offset) as usize;
        self.data.get(physical_idx)
    }

    /// Set element at linear index
    pub fn set_linear(&mut self, index: usize, value: T) {
        if index >= self.size() {
            return;
        }
        let indices = crate::strides::compute_multi_indices(index, &self.shape);
        let linear_offset = crate::strides::compute_linear_index(&indices, &self.strides);
        let physical_idx = (self.offset as isize + linear_offset) as usize;

        if let Some(elem) = self.data.get_mut(physical_idx) {
            *elem = value;
        }
    }

    /// Get element at linear index (alias for get_linear)
    pub fn get(&self, index: usize) -> Option<&T> {
        self.get_linear(index)
    }

    /// Get element at physical index (direct access to data buffer)
    pub fn get_linear_physical(&self, physical_index: usize) -> Option<&T> {
        self.data.get(physical_index)
    }

    /// Set element at linear index with Result
    pub fn set(&mut self, index: usize, value: T) -> Result<(), NumPyError> {
        if index >= self.size() {
            return Err(NumPyError::index_error(index, self.size()));
        }
        self.set_linear(index, value);
        Ok(())
    }

    /// Set element at physical index (direct access to data buffer)
    pub fn set_linear_physical(
        &mut self,
        physical_index: usize,
        value: T,
    ) -> Result<(), NumPyError> {
        if physical_index >= self.data.len() {
            return Err(NumPyError::index_error(physical_index, self.data.len()));
        }
        if let Some(elem) = self.data.get_mut(physical_index) {
            *elem = value;
            Ok(())
        } else {
            Err(NumPyError::index_error(physical_index, self.data.len()))
        }
    }

    /// Create 2D array from matrix
    pub fn from_array2(array2: ndarray::Array2<T>) -> Self
    where
        T: 'static,
    {
        let shape = array2.shape().to_vec();
        let strides = compute_strides(&shape);
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

    /// Transpose array
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

        for i in 0..rows {
            for j in 0..cols {
                transposed_data.push(self.get_linear(i * cols + j).unwrap().clone());
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

    /// Transpose array view (non-consuming, shares data)
    /// For 2D arrays, returns a view with swapped axes
    /// For higher dimensions, reverses the axes
    pub fn transpose_view(&self, _axes: Option<&[usize]>) -> Result<Self, NumPyError>
    where
        T: Clone,
    {
        if self.ndim() != 2 {
            // For higher dimensions, reverse axes
            let new_shape: Vec<usize> = self.shape.iter().rev().cloned().collect();
            let new_strides: Vec<isize> = self.strides.iter().rev().cloned().collect();
            return Ok(Self {
                data: Arc::clone(&self.data),
                shape: new_shape,
                strides: new_strides,
                dtype: self.dtype.clone(),
                offset: self.offset,
            });
        }

        // For 2D arrays, swap the two axes
        let new_shape = vec![self.shape[1], self.shape[0]];
        let new_strides = vec![self.strides[1], self.strides[0]];

        Ok(Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype.clone(),
            offset: self.offset,
        })
    }

    /// Broadcast array to new shape
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        crate::broadcasting::broadcast_to(self, shape)
    }

    /// Get element at multi-dimensional indices
    pub fn get_multi(&self, indices: &[usize]) -> Result<T, NumPyError>
    where
        T: Clone,
    {
        if indices.len() != self.ndim() {
            return Err(NumPyError::invalid_operation(format!(
                "Index dimension {} does not match array dimension {}",
                indices.len(),
                self.ndim()
            )));
        }

        let linear_offset = crate::strides::compute_linear_index(indices, &self.strides);
        let physical_idx = (self.offset as isize + linear_offset) as usize;

        self.data
            .get(physical_idx)
            .cloned()
            .ok_or_else(|| NumPyError::index_error(physical_idx, self.size()))
    }

    /// Set element at multi-dimensional indices
    pub fn set_multi(&mut self, indices: &[usize], value: T) -> Result<(), NumPyError>
    where
        T: Clone,
    {
        if indices.len() != self.ndim() {
            return Err(NumPyError::invalid_operation(format!(
                "Index dimension {} does not match array dimension {}",
                indices.len(),
                self.ndim()
            )));
        }

        let linear_offset = crate::strides::compute_linear_index(indices, &self.strides);
        let physical_idx = (self.offset as isize + linear_offset) as usize;

        if let Some(elem) = Arc::make_mut(&mut self.data).get_mut(physical_idx) {
            *elem = value;
            Ok(())
        } else {
            Err(NumPyError::index_error(physical_idx, self.size()))
        }
    }

    /// Clone the array and convert elements to Complex64
    pub fn clone_to_complex(&self) -> Array<Complex64>
    where
        T: Clone + Into<Complex64> + Default + 'static,
    {
        let data: Vec<Complex64> = self.iter().map(|x| x.clone().into()).collect();
        Array::from_data(data, self.shape.to_vec())
    }

    /// Get elements where mask is true (Boolean Indexing)
    pub fn get_mask(&self, mask: &Array<bool>) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        if mask.ndim() > self.ndim() {
            return Err(NumPyError::invalid_operation(format!(
                "Mask ndim {} exceeds array ndim {}",
                mask.ndim(),
                self.ndim()
            )));
        }

        for (i, &dim) in mask.shape().iter().enumerate() {
            if dim != self.shape[i] {
                return Err(NumPyError::invalid_operation(format!(
                    "Mask shape {:?} must match the beginning of array shape {:?}",
                    mask.shape(),
                    self.shape()
                )));
            }
        }

        let mask_size = mask.size();
        let _remaining_ndim = self.ndim() - mask.ndim();
        let remaining_shape = &self.shape[mask.ndim()..];
        let sub_array_size: usize = remaining_shape.iter().product();

        let mut extracted = Vec::new();
        for i in 0..mask_size {
            if let Some(&is_true) = mask.get_linear(i) {
                if is_true {
                    for j in 0..sub_array_size {
                        if let Some(val) = self.get_linear(i * sub_array_size + j) {
                            extracted.push(val.clone());
                        }
                    }
                }
            }
        }

        let mut result_shape = vec![extracted.len() / sub_array_size];
        result_shape.extend_from_slice(remaining_shape);

        Ok(Array::from_data(extracted, result_shape))
    }

    /// Take elements along an axis (Fancy Indexing)
    pub fn take(&self, indices: &Array<usize>, axis: Option<usize>) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        let axis = match axis {
            None => {
                // Flat indexing
                let mut data = Vec::with_capacity(indices.size());
                for &idx in indices.iter() {
                    if let Some(val) = self.get_linear(idx) {
                        data.push(val.clone());
                    } else {
                        return Err(NumPyError::index_error(idx, self.size()));
                    }
                }
                return Ok(Array::from_data(data, indices.shape().to_vec()));
            }
            Some(ax) => normalize_axis(ax as isize, self.ndim())?,
        };

        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape[axis] = indices.size();

        let mut result = Array::zeros(new_shape);
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();
        let axis_len = shape[axis];

        for i in 0..outer_size {
            for (j, &idx) in indices.iter().enumerate() {
                if idx >= axis_len {
                    return Err(NumPyError::index_error(idx, axis_len));
                }
                for k in 0..inner_size {
                    let mut src_idx = vec![0; self.ndim()];
                    let mut temp_i = i;
                    for d in (0..axis).rev() {
                        src_idx[d] = temp_i % shape[d];
                        temp_i /= shape[d];
                    }
                    src_idx[axis] = idx;
                    let mut temp_k = k;
                    for d in (axis + 1..self.ndim()).rev() {
                        src_idx[d] = temp_k % shape[d];
                        temp_k /= shape[d];
                    }

                    let mut dst_idx = src_idx.clone();
                    dst_idx[axis] = j;

                    let val = self.get_multi(&src_idx)?;
                    result.set_multi(&dst_idx, val)?;
                }
            }
        }
        Ok(result)
    }

    /// Fancy indexing with multiple integer arrays (Fancy Indexing)
    pub fn fancy_index(&self, indices: &[&Array<usize>]) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        if indices.is_empty() {
            return Ok(self.clone());
        }

        if indices.len() > self.ndim() {
            return Err(NumPyError::invalid_operation(format!(
                "Too many indices for array: {} > {}",
                indices.len(),
                self.ndim()
            )));
        }

        // 1. Determine broadcasted shape of index arrays
        let mut broadcast_shape = indices[0].shape().to_vec();
        for idx in indices.iter().skip(1) {
            broadcast_shape =
                crate::broadcasting::compute_broadcast_shape(&broadcast_shape, idx.shape());
        }

        // 2. Broadcast all index arrays to the common shape
        let mut broadcasted_indices = Vec::with_capacity(indices.len());
        for idx in indices {
            broadcasted_indices.push(idx.broadcast_to(&broadcast_shape)?);
        }

        // 3. Create result array
        let total_broadcast_elements = broadcast_shape.iter().product::<usize>();
        let remaining_shape = &self.shape[indices.len()..];
        let sub_array_size: usize = remaining_shape.iter().product();

        let mut result_shape = broadcast_shape.clone();
        result_shape.extend_from_slice(remaining_shape);

        let mut result_data = Vec::with_capacity(total_broadcast_elements * sub_array_size);

        // 4. Iterate over indices and extract data
        for i in 0..total_broadcast_elements {
            let mut base_coords = vec![0; indices.len()];

            for (dim, b_idx) in broadcasted_indices.iter().enumerate() {
                let idx_val = *b_idx
                    .get_linear(i)
                    .ok_or_else(|| NumPyError::index_error(i, b_idx.size()))?;

                if idx_val >= self.shape[dim] {
                    return Err(NumPyError::index_error(idx_val, self.shape[dim]));
                }
                base_coords[dim] = idx_val;
            }

            for j in 0..sub_array_size {
                let mut full_coords = base_coords.clone();
                let remaining_coords = crate::strides::compute_multi_indices(j, remaining_shape);
                full_coords.extend(remaining_coords);

                let val = self.get_multi(&full_coords)?;
                result_data.push(val);
            }
        }

        Ok(Array::from_data(result_data, result_shape))
    }

    /// Reshape array
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        crate::slicing::reshape(self, new_shape.to_vec())
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

impl<T> Array<T> {
    pub fn zeros(shape: Vec<usize>) -> Self
    where
        T: Clone + Default + 'static,
    {
        let data = vec![T::default(); shape.iter().product()];
        Self::from_data(data, shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: num_traits::One + Clone + Default + 'static,
    {
        let data = vec![T::one(); shape.iter().product()];
        Self::from_data(data, shape)
    }

    pub fn empty(shape: Vec<usize>) -> Self
    where
        T: Clone + Default + 'static,
    {
        let data = Vec::with_capacity(shape.iter().product());
        Self::from_data(data, shape)
    }

    pub fn full(shape: Vec<usize>, value: T) -> Self
    where
        T: Clone + Default + 'static,
    {
        let data = vec![value; shape.iter().product()];
        Self::from_data(data, shape)
    }

    pub fn from_scalar(value: T, shape: Vec<usize>) -> Self
    where
        T: Clone + Default + 'static,
    {
        let data = vec![value];
        Self::from_data(data, shape)
    }

    pub fn from_vec(data: Vec<T>) -> Self
    where
        T: Clone + Default + 'static,
    {
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

    pub fn from_shape_vec(shape: Vec<usize>, data: Vec<T>) -> Self
    where
        T: Clone + Default + 'static,
    {
        Self::from_data(data, shape)
    }

    pub fn eye(size: usize) -> Self
    where
        T: num_traits::Zero + num_traits::One + Clone + Default + 'static,
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

/// Normalize an axis index to be within bounds [0, ndim)
pub fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, NumPyError> {
    if axis < 0 {
        let ax = axis + ndim as isize;
        if ax < 0 {
            return Err(NumPyError::invalid_operation(format!(
                "Axis {} out of bounds for ndim {}",
                axis, ndim
            )));
        }
        Ok(ax as usize)
    } else {
        if axis as usize >= ndim {
            return Err(NumPyError::invalid_operation(format!(
                "Axis {} out of bounds for ndim {}",
                axis, ndim
            )));
        }
        Ok(axis as usize)
    }
}
