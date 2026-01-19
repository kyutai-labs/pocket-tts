use crate::array::Array;
// use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};
use crate::strides::compute_linear_index;
// use std::ops::{Index, IndexMut};

/// Indexing trait for arrays
pub trait ArrayIndex<T> {
    type Output;
    fn get(&self, array: &Array<T>) -> Result<Self::Output>;
    fn get_mut(&mut self, array: &mut Array<T>) -> Result<Self::Output>;
}

impl<T> ArrayIndex<T> for usize
where
    T: Clone,
{
    type Output = T;

    fn get(&self, array: &Array<T>) -> Result<Self::Output> {
        array
            .get(*self)
            .cloned()
            .ok_or_else(|| NumPyError::index_error(*self, array.size()))
    }

    fn get_mut(&mut self, _array: &mut Array<T>) -> Result<Self::Output> {
        // This would need mutable access to memory manager
        // For now, return error
        Err(NumPyError::NotImplemented {
            operation: "striding with step > 1".to_string(),
        })
    }
}

impl<T> ArrayIndex<T> for [usize]
where
    T: Clone + Default + 'static,
{
    type Output = Array<T>;

    fn get(&self, array: &Array<T>) -> Result<Self::Output> {
        if self.len() != array.ndim() {
            return Err(NumPyError::index_error(0, array.ndim()));
        }

        for (i, &idx) in self.iter().enumerate() {
            if idx >= array.shape()[i] {
                return Err(NumPyError::index_error(idx, array.shape()[i]));
            }
        }

        let linear_idx = compute_linear_index(self, array.strides());
        if let Some(element) = array.get(linear_idx) {
            let mut result = Array::zeros(vec![1]);
            result.set(0, element.clone())?;
            Ok(result)
        } else {
            Err(NumPyError::index_error(linear_idx, array.size()))
        }
    }

    fn get_mut(&mut self, array: &mut Array<T>) -> Result<Self::Output> {
        ArrayIndex::get(self, array)
    }
}

/// Index types for advanced indexing
#[derive(Debug, Clone, PartialEq)]
pub enum Index {
    /// Integer index
    Integer(isize),
    /// Slice index
    Slice(Slice),
    /// Ellipsis (...)
    Ellipsis,
    /// Boolean mask
    Boolean(bool),
}

/// Slice specification for arrays
#[derive(Debug, Clone, PartialEq)]
pub enum Slice {
    /// Full slice (:)
    Full,
    /// Range slice (start:stop)
    Range(isize, isize),
    /// Range with step (start:stop:step)
    RangeStep(isize, isize, isize),
    /// Single index
    Index(isize),
    /// From start to end (start:)
    From(isize),
    /// To end (:end)
    To(isize),
    /// With step (:step)
    Step(isize),
}

impl Slice {
    /// Convert slice to range with bounds checking
    pub fn to_range(&self, len: isize) -> (isize, isize, isize) {
        match self {
            Slice::Full => (0, len, 1),
            Slice::Range(start, stop) => {
                let start = if *start < 0 { len + *start } else { *start };
                let stop = if *stop < 0 { len + *stop } else { *stop };
                (start, stop, 1)
            }
            Slice::RangeStep(start, stop, step) => {
                let start = if *start < 0 { len + *start } else { *start };
                let stop = if *stop < 0 { len + *stop } else { *stop };
                (start, stop, *step)
            }
            Slice::Index(idx) => {
                let idx = if *idx < 0 { len + *idx } else { *idx };
                (idx, idx + 1, 1)
            }
            Slice::From(start) => {
                let start = if *start < 0 { len + *start } else { *start };
                (start, len, 1)
            }
            Slice::To(stop) => {
                let stop = if *stop < 0 { len + *stop } else { *stop };
                (0, stop, 1)
            }
            Slice::Step(step) => (0, len, *step),
        }
    }

    /// Get length of slice for given dimension
    pub fn len(&self, dim_len: usize) -> usize {
        let len = dim_len as isize;
        let (start, stop, step) = self.to_range(len);

        if step == 0 {
            return 0;
        }

        let actual_start = start.max(0).min(len);
        let actual_stop = stop.max(0).min(len);

        if (step > 0 && actual_start >= actual_stop) || (step < 0 && actual_start <= actual_stop) {
            return 0;
        }

        ((actual_stop - actual_start).abs() as usize + step.abs() as usize - 1)
            / step.abs() as usize
    }
}

/// Slice specification for multiple dimensions
#[derive(Debug, Clone)]
pub struct MultiSlice {
    slices: Vec<Slice>,
}

impl MultiSlice {
    /// Create new multi-slice
    pub fn new(slices: Vec<Slice>) -> Self {
        Self { slices }
    }

    /// Get slice at dimension
    pub fn get(&self, dim: usize) -> &Slice {
        self.slices.get(dim).unwrap_or(&Slice::Full)
    }

    /// Convert to vector of ranges
    pub fn to_ranges(&self, shape: &[usize]) -> Vec<(isize, isize, isize)> {
        self.slices
            .iter()
            .enumerate()
            .map(|(i, slice)| {
                let dim_len = shape.get(i).copied().unwrap_or(0) as isize;
                slice.to_range(dim_len)
            })
            .collect()
    }

    /// Compute resulting shape after slicing
    pub fn result_shape(&self, shape: &[usize]) -> Vec<usize> {
        self.slices
            .iter()
            .enumerate()
            .map(|(i, slice)| {
                let dim_len = shape.get(i).copied().unwrap_or(0);
                slice.len(dim_len)
            })
            .collect()
    }
}

/// Indexing implementation for Array
// Indexing traits for Array are removed because they require returning references
// which is not possible when slicing returns a new Array instance.
// Use .get() or .slice() instead.

impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    /// Get array slice using slice syntax
    pub fn slice(&self, multi_slice: &MultiSlice) -> Result<Array<T>> {
        let result_shape = multi_slice.result_shape(self.shape());
        let _ranges = multi_slice.to_ranges(self.shape());

        // This is a very simplified implementation
        // Real implementation would need complex slicing logic
        let result = Array::zeros(result_shape);

        // Copy sliced data
        for i in 0..result.size() {
            if i < self.size() {
                if let Some(_src_element) = self.get(i) {
                    // This is placeholder - real implementation would calculate source indices
                    // Set element in result - need proper indexing
                    break; // Simplified for now
                }
            }
        }

        Ok(result)
    }

    /// Newaxis support (arr[np.newaxis, :])
    pub fn add_newaxis(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(NumPyError::index_error(axis, self.ndim()));
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);

        // Reshape array with new axis
        self.reshape(&new_shape)
    }

    /// Advanced indexing with mixed types (arr[1, :, :5])
    pub fn advanced_index(&self, indices: &[crate::slicing::Index]) -> Result<Self> {
        // Handle mixed indexing types
        let mut expanded_indices = Vec::new();
        for idx in indices {
            match idx {
                crate::slicing::Index::Integer(i) => {
                    expanded_indices.push(Index::Integer(*i));
                }
                Index::Slice(slice) => {
                    expanded_indices.push(crate::slicing::Index::Slice(slice.clone()));
                }
                Index::Ellipsis => {
                    expanded_indices.push(Index::Ellipsis);
                }
                crate::slicing::Index::Boolean(b) => {
                    expanded_indices.push(Index::Boolean(*b));
                }
            }
        }

        self.ellipsis_index(&expanded_indices)
    }

    /// Multi-dimensional indexing (arr[1:3, 5:, ::-1])
    pub fn multidim_index(&self, indices: &[crate::slicing::Index]) -> Result<Self> {
        if indices.len() != self.ndim() {
            return Err(NumPyError::value_error(
                format!(
                    "Expected {} indices for {}-D array, got {}",
                    self.ndim(),
                    self.ndim(),
                    indices.len()
                ),
                "multidim_index",
            ));
        }

        let mut result_data = Vec::new();
        let mut result_shape = Vec::new();

        // Process each dimension
        for (dim, idx) in indices.iter().enumerate() {
            match idx {
                Index::Slice(slice) => {
                    let length = self.calculate_slice_length(dim, slice);
                    result_shape.push(length);
                }
                Index::Integer(_) => {
                    result_shape.push(1);
                }
                Index::Ellipsis => {
                    result_shape.push(self.shape()[dim]);
                }
                Index::Boolean(_) => {
                    // Boolean indexing reduces dimensionality
                    // Complex implementation needed for full NumPy compatibility
                }
            }
        }

        // Extract data using multi-dimensional indexing
        self.extract_multidim_data(indices, &mut result_data)?;

        Ok(Array::from_data(result_data, result_shape))
    }

    /// Set element at multi-dimensional indices
    pub fn set_by_indices(&mut self, indices: &[usize], value: T) -> Result<()> {
        if indices.len() != self.ndim() {
            return Err(NumPyError::index_error(0, self.ndim()));
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Err(NumPyError::index_error(idx, self.shape()[i]));
            }
        }

        let linear_idx = compute_linear_index(indices, self.strides());
        self.set(linear_idx, value)
    }

    /// Get element at multi-dimensional indices
    pub fn get_by_indices(&self, indices: &[usize]) -> Result<&T> {
        if indices.len() != self.ndim() {
            return Err(NumPyError::index_error(0, self.ndim()));
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Err(NumPyError::index_error(idx, self.shape()[i]));
            }
        }

        let linear_idx = compute_linear_index(indices, self.strides());
        self.get(linear_idx)
            .ok_or_else(|| NumPyError::index_error(linear_idx, self.size()))
    }

    /// Expand ellipsis in indices
    pub fn ellipsis_index(&self, indices: &[Index]) -> Result<Self> {
        // Count non-ellipsis indices and find ellipsis position
        let mut ellipsis_pos = None;
        let mut non_ellipsis_count = 0;

        for (i, idx) in indices.iter().enumerate() {
            match idx {
                Index::Ellipsis => {
                    if ellipsis_pos.is_some() {
                        return Err(NumPyError::invalid_value(
                            "Only one ellipsis allowed in index",
                        ));
                    }
                    ellipsis_pos = Some(i);
                }
                _ => {
                    non_ellipsis_count += 1;
                }
            }
        }

        // If no ellipsis, just pass through to multidim_index
        let ellipsis_pos = match ellipsis_pos {
            Some(pos) => pos,
            None => return self.multidim_index(indices),
        };

        // Calculate how many dimensions the ellipsis expands to
        let ndim = self.ndim();
        if non_ellipsis_count > ndim {
            return Err(NumPyError::value_error(
                format!(
                    "Too many indices for array: array is {}-dimensional, but {} were indexed",
                    ndim, non_ellipsis_count
                ),
                "ellipsis_index",
            ));
        }

        let ellipsis_dims = ndim - non_ellipsis_count;

        // Build expanded indices
        let mut expanded = Vec::with_capacity(ndim);

        // Add indices before ellipsis
        for idx in &indices[..ellipsis_pos] {
            expanded.push(idx.clone());
        }

        // Add Full slices for ellipsis expansion
        for _ in 0..ellipsis_dims {
            expanded.push(Index::Slice(Slice::Full));
        }

        // Add indices after ellipsis
        for idx in &indices[ellipsis_pos + 1..] {
            expanded.push(idx.clone());
        }

        self.multidim_index(&expanded)
    }

    /// Calculate length of a slice for a dimension
    pub fn calculate_slice_length(&self, dim: usize, slice: &Slice) -> usize {
        slice.len(self.shape()[dim])
    }

    /// Extract data using multi-dimensional indices
    pub fn extract_multidim_data(&self, indices: &[Index], result: &mut Vec<T>) -> Result<()> {
        // Convert indices to ranges for each dimension
        let mut ranges: Vec<Vec<usize>> = Vec::with_capacity(self.ndim());

        for (dim, idx) in indices.iter().enumerate() {
            let dim_size = self.shape().get(dim).copied().unwrap_or(0);

            match idx {
                Index::Integer(i) => {
                    // Normalize negative index
                    let normalized = if *i < 0 {
                        (dim_size as isize + *i) as usize
                    } else {
                        *i as usize
                    };

                    if normalized >= dim_size {
                        return Err(NumPyError::index_error(normalized, dim_size));
                    }

                    ranges.push(vec![normalized]);
                }
                Index::Slice(slice) => {
                    let (start, stop, step) = slice.to_range(dim_size as isize);

                    // Clamp to valid bounds
                    let start = start.max(0).min(dim_size as isize) as usize;
                    let stop = stop.max(0).min(dim_size as isize) as usize;

                    let mut dim_indices = Vec::new();

                    if step > 0 {
                        let mut i = start;
                        while i < stop {
                            dim_indices.push(i);
                            i += step as usize;
                        }
                    } else if step < 0 {
                        let mut i = start as isize;
                        while i > stop as isize {
                            dim_indices.push(i as usize);
                            i += step;
                        }
                    }

                    ranges.push(dim_indices);
                }
                Index::Ellipsis => {
                    // Ellipsis should have been expanded by ellipsis_index
                    // If we reach here, treat as full slice
                    ranges.push((0..dim_size).collect());
                }
                Index::Boolean(b) => {
                    // Boolean indexing: if true, include all; if false, include none
                    if *b {
                        ranges.push((0..dim_size).collect());
                    } else {
                        ranges.push(Vec::new());
                    }
                }
            }
        }

        // If any dimension has empty indices, result is empty
        if ranges.iter().any(|r| r.is_empty()) {
            return Ok(());
        }

        // Generate all combinations of indices and extract elements
        let mut current_indices = vec![0usize; ranges.len()];
        let mut counters = vec![0usize; ranges.len()];

        loop {
            // Build current multi-dimensional index
            for (i, counter) in counters.iter().enumerate() {
                current_indices[i] = ranges[i][*counter];
            }

            // Compute linear index and extract element
            let linear_idx = compute_linear_index(&current_indices, self.strides());
            if let Some(val) = self.get(linear_idx) {
                result.push(val.clone());
            }

            // Increment counters (like incrementing a multi-digit number)
            let mut carry = true;
            for i in (0..counters.len()).rev() {
                if carry {
                    counters[i] += 1;
                    if counters[i] >= ranges[i].len() {
                        counters[i] = 0;
                        // carry continues
                    } else {
                        carry = false;
                    }
                }
            }

            // If carry is still true after all dimensions, we're done
            if carry {
                break;
            }
        }

        Ok(())
    }
}

/// Reshape array to new dimensions
pub fn reshape<T: Clone + Default + 'static>(
    array: &Array<T>,
    new_shape: Vec<usize>,
) -> Result<Array<T>> {
    let total_elements: usize = new_shape.iter().product();

    if total_elements != array.size() {
        return Err(NumPyError::value_error(
            format!(
                "Cannot reshape array of size {} into shape {:?}",
                array.size(),
                new_shape
            ),
            "reshape",
        ));
    }

    let mut new_data = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        if let Some(element) = array.get(i) {
            new_data.push(element.clone());
        }
    }

    Ok(Array::from_data(new_data, new_shape))
}

/// Iterator over array elements
pub struct ArrayIter<'a, T> {
    array: &'a Array<T>,
    current: usize,
}

impl<'a, T> Iterator for ArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.array.size() {
            let element = self.array.get(self.current);
            self.current += 1;
            element
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.size() - self.current;
        (remaining, Some(remaining))
    }
}

/// Mutable iterator over array elements
pub struct ArrayIterMut<'a, T> {
    array: &'a mut Array<T>,
    current: usize,
}

impl<'a, T> Iterator for ArrayIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.array.size() {
            // This would need proper mutable access
            self.current += 1;
            None // Placeholder
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.size() - self.current;
        (remaining, Some(remaining))
    }
}

/// Convenience macros for slicing
#[macro_export]
macro_rules! s {
    (:) => {
        $crate::slicing::Slice::Full
    (start:end:step) => {
        $crate::slicing::Slice::RangeStep(start, end, step)
    };
    (..end) => {
        $crate::slicing::Slice::To(end)
    };
    (start..) => {
        $crate::slicing::Slice::From(start)
    };
    (start..end) => {
        $crate::slicing::Slice::Range(start, end)
    };
    (start..end..step) => {
        $crate::slicing::Slice::RangeStep(start, end, step)
    };

}
}
