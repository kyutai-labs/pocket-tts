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

    /// Set element at linear index with Result
    pub fn set(&mut self, index: usize, value: T) -> Result<(), NumPyError> {
        if index >= self.size() {
            return Err(NumPyError::index_error(index, self.size()));
        }
        self.set_linear(index, value);
        Ok(())
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
            return self.clone();
        }

        let (rows, cols) = (self.shape()[0], self.shape()[1]);
        let mut transposed_data = Vec::with_capacity(self.size());

        for j in 0..cols {
            for i in 0..rows {
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
    pub fn transpose_view(&self, _axes: Option<&[usize]>) -> Result<Self, NumPyError>
    where
        T: Clone,
    {
        if self.ndim() != 2 {
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
        if self.shape() != mask.shape() {
            return Err(NumPyError::invalid_operation(format!(
                "Mask shape {:?} must match array shape {:?}",
                mask.shape(),
                self.shape()
            )));
        }

        let mut extracted = Vec::new();
        for (i, &is_true) in mask.iter().enumerate() {
            if is_true {
                if let Some(val) = self.get_linear(i) {
                    extracted.push(val.clone());
                }
            }
        }

        Ok(Array::from_vec(extracted))
    }

    /// Take elements along an axis (Fancy Indexing)
    pub fn take(&self, indices: &Array<usize>, axis: Option<usize>) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        match axis {
            None => {
                let mut data = Vec::with_capacity(indices.size());
                for &idx in indices.iter() {
                    if let Some(val) = self.get_linear(idx) {
                        data.push(val.clone());
                    } else {
                        return Err(NumPyError::index_error(idx, self.size()));
                    }
                }
                Ok(Array::from_data(data, indices.shape().to_vec()))
            }
            Some(ax) => {
                let ax = normalize_axis(ax as isize, self.ndim())?;
                let shape = self.shape();
                let mut new_shape = shape.to_vec();
                new_shape[ax] = indices.size();

                let mut result = Array::zeros(new_shape);
                let outer_size: usize = shape[..ax].iter().product();
                let inner_size: usize = shape[ax + 1..].iter().product();
                let axis_len = shape[ax];

                for i in 0..outer_size {
                    for (j, &idx) in indices.iter().enumerate() {
                        if idx >= axis_len {
                            return Err(NumPyError::index_error(idx, axis_len));
                        }
                        for k in 0..inner_size {
                            let mut src_idx = vec![0; self.ndim()];
                            let mut temp_i = i;
                            for d in (0..ax).rev() {
                                src_idx[d] = temp_i % shape[d];
                                temp_i /= shape[d];
                            }
                            src_idx[ax] = idx;
                            let mut temp_k = k;
                            for d in (ax + 1..self.ndim()).rev() {
                                src_idx[d] = temp_k % shape[d];
                                temp_k /= shape[d];
                            }

                            let mut dst_idx = src_idx.clone();
                            dst_idx[ax] = j;

                            let val = self.get_multi(&src_idx)?;
                            result.set_multi(&dst_idx, val)?;
                        }
                    }
                }
                Ok(result)
            }
        }
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

        let mut broadcast_shape = indices[0].shape().to_vec();
        for idx in indices.iter().skip(1) {
            broadcast_shape =
                crate::broadcasting::compute_broadcast_shape(&broadcast_shape, idx.shape());
        }

        let mut broadcasted_indices = Vec::with_capacity(indices.len());
        for idx in indices {
            broadcasted_indices.push(idx.broadcast_to(&broadcast_shape)?);
        }

        let total_elements = broadcast_shape.iter().product();
        let mut result_data = Vec::with_capacity(total_elements);

        for i in 0..total_elements {
            let mut coords = vec![0; self.ndim()];

            for (dim, b_idx) in broadcasted_indices.iter().enumerate() {
                let idx_val = *b_idx
                    .get_linear(i)
                    .ok_or_else(|| NumPyError::index_error(i, b_idx.size()))?;

                if idx_val >= self.shape[dim] {
                    return Err(NumPyError::index_error(idx_val, self.shape[dim]));
                }
                coords[dim] = idx_val;
            }

            if indices.len() < self.ndim() {
                return Err(NumPyError::not_implemented(
                    "Mixed fancy and basic indexing",
                ));
            }

            let val = self.get_multi(&coords)?;
            result_data.push(val);
        }

        Ok(Array::from_data(result_data, broadcast_shape))
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

    // ===== Sorting Methods =====

    /// Sort array in-place or return a sorted copy
    pub fn sort(&mut self, axis: Option<isize>, kind: &str, order: &str) -> Result<Self, NumPyError>
    where
        T: Clone + PartialOrd + crate::sorting::ComparisonOps<T> + Default + Send + Sync + 'static,
    {
        crate::sorting::sort(self, axis, kind, order)
    }

    /// Return indices that would sort the array
    pub fn argsort(&self, axis: Option<isize>, kind: &str, order: &str) -> Result<Array<isize>, NumPyError>
    where
        T: Clone + PartialOrd + crate::sorting::ComparisonOps<T> + Default + Send + Sync + 'static,
    {
        crate::sorting::argsort(self, axis, kind, order)
    }

    /// Find insertion points for elements in a sorted array
    pub fn searchsorted(
        &self,
        v: &Array<T>,
        side: &str,
        sorter: Option<&Array<isize>>,
    ) -> Result<Array<isize>, NumPyError>
    where
        T: Clone + PartialOrd + crate::sorting::ComparisonOps<T> + Default + Send + Sync + 'static,
    {
        crate::sorting::searchsorted(self, v, side, sorter)
    }

    /// Return the indices that would partition an array
    pub fn argpartition(
        &self,
        kth: crate::sorting::ArrayOrInt,
        axis: Option<isize>,
        kind: &str,
        order: &str,
    ) -> Result<Array<isize>, NumPyError>
    where
        T: Clone + PartialOrd + crate::sorting::ComparisonOps<T> + Default + Send + Sync + 'static,
    {
        crate::sorting::argpartition(self, kth, axis, kind, order)
    }

    /// Partition array in-place
    pub fn partition(
        &mut self,
        kth: crate::sorting::ArrayOrInt,
        axis: Option<isize>,
        kind: &str,
        order: &str,
    ) -> Result<Self, NumPyError>
    where
        T: Clone + PartialOrd + crate::sorting::ComparisonOps<T> + Default + Send + Sync + 'static,
    {
        crate::sorting::partition(self, kth, axis, kind, order)
    }
}

/// Compute strides from shape
pub fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;

    for &dim in shape.iter().rev() {
        strides.push(stride as isize);
        stride *= dim;
    }

    strides.reverse();
    strides
}

impl<T> Array<T> {
    /// Create a view of the array (shares data)
    pub fn view(&self) -> Self
    where
        T: Clone,
    {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype.clone(),
            offset: self.offset,
        }
    }

    /// Cast array to new dtype
    pub fn astype<U>(&self) -> Result<Array<U>, NumPyError>
    where
        T: Clone + Default + 'static + num_traits::NumCast + Copy,
        U: Clone + Default + 'static + num_traits::NumCast + Copy,
    {
        let mut new_data = Vec::with_capacity(self.size());
        for item in self.iter() {
            let cast_val = num_traits::cast::<T, U>(*item)
                .ok_or_else(|| NumPyError::invalid_operation("Failed to cast value"))?;
            new_data.push(cast_val);
        }
        Ok(Array::from_shape_vec(self.shape.clone(), new_data))
    }

    /// Return complex conjugate (element-wise).
    pub fn conj(&self) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static + ElementConj,
    {
        let mut new_data = Vec::with_capacity(self.size());
        for item in self.iter() {
            new_data.push(item.element_conj());
        }
        Ok(Array::from_shape_vec(self.shape.clone(), new_data))
    }

    /// Return complex conjugate (alias).
    pub fn conjugate(&self) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static + ElementConj,
    {
        self.conj()
    }

    /// Move array to device (Stub)
    pub fn to_device(&self, device: &str) -> Result<Self, NumPyError>
    where
        T: Clone,
    {
        if device == "cpu" {
            Ok(self.clone())
        } else {
            Err(NumPyError::not_implemented("GPU support not enabled"))
        }
    }

    /// Construct bytes containing raw data
    pub fn tobytes(&self) -> Result<Vec<u8>, NumPyError>
    where
        T: Clone + Copy + 'static,
    {
        // Require contiguous
        if !self.is_contiguous() {
            return Err(NumPyError::invalid_operation(
                "tobytes requires contiguous array",
            ));
        }

        let slice = self.as_slice();
        // unsafe reinterpretation
        let len = slice.len() * std::mem::size_of::<T>();
        let ptr = slice.as_ptr() as *const u8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        Ok(bytes.to_vec())
    }

    /// Write array to file
    pub fn tofile<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), NumPyError>
    where
        T: Clone + Copy + 'static,
    {
        let bytes = self.tobytes()?;
        std::fs::write(path, bytes).map_err(|e| NumPyError::invalid_operation(e.to_string()))
    }

    /// Return array as a list (Vec)
    pub fn tolist(&self) -> Vec<T>
    where
        T: Clone,
    {
        // For 1D, simple vec. For n-D, flat vec for now.
        self.to_vec()
    }

    /// Swap bytes
    pub fn byteswap(&self, _inplace: bool) -> Result<Self, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        Err(NumPyError::not_implemented(
            "byteswap requires trait specialization",
        ))
    }
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

/// Trait for element-wise complex conjugate
pub trait ElementConj {
    fn element_conj(&self) -> Self;
}

impl ElementConj for f32 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for f64 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for i8 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for i16 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for i32 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for i64 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for u8 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for u16 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for u32 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for u64 {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for bool {
    fn element_conj(&self) -> Self {
        *self
    }
}
impl ElementConj for Complex64 {
    fn element_conj(&self) -> Self {
        self.conj()
    }
}
impl ElementConj for num_complex::Complex<f32> {
    fn element_conj(&self) -> Self {
        self.conj()
    }
}
