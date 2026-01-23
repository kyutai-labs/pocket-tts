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
        if let Some(element) = array.get(linear_idx as usize) {
            let mut result = Array::zeros(vec![1]);
            result.set(0, element.clone())?;
            Ok(result)
        } else {
            Err(NumPyError::index_error(linear_idx as usize, array.size()))
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
            Slice::Step(step) => {
                if *step > 0 {
                    (0, len, *step)
                } else {
                    (len - 1, -1, *step)
                }
            }
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

        (actual_stop - actual_start)
            .unsigned_abs()
            .div_ceil(step.unsigned_abs())
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
        if self.ndim() < multi_slice.slices.len() {
            // Handle too many indices? For now assume valid or fewer (broadcasting/newaxis not handled here fully)
        }

        let ranges = multi_slice.to_ranges(self.shape());
        let mut new_shape = Vec::with_capacity(ranges.len());
        let mut new_strides = Vec::with_capacity(ranges.len());
        let mut new_offset = self.offset;

        for (i, (start, stop, step)) in ranges.iter().enumerate() {
            let dim_len = self.shape[i] as isize;
            let current_stride = self.strides[i];

            // Calculate number of elements
            let (count, actual_start) = if *step > 0 {
                let start = (*start).clamp(0, dim_len);
                let stop = (*stop).clamp(0, dim_len);
                let count = if start < stop {
                    (stop - start + step - 1) / step
                } else {
                    0
                };
                (count, start)
            } else {
                // Negative step
                // start is inclusive, stop is exclusive (downwards)
                // e.g. 5, 4, 3, 2, 1, 0. stop at -1.
                // Bounds: start <= len-1, stop >= -1.
                let start = (*start).min(dim_len - 1).max(-1);
                let stop = (*stop).min(dim_len - 1).max(-1); // allows -1

                let count = if start > stop {
                    (start - stop + -step - 1) / -step
                } else {
                    0
                };
                (count, start)
            };

            new_shape.push(count as usize);
            new_strides.push(current_stride * step);
            new_offset = (new_offset as isize + actual_start * current_stride) as usize;
        }

        // Handle any remaining dimensions (if fewer slices provided)
        for i in ranges.len()..self.ndim() {
            new_shape.push(self.shape[i]);
            new_strides.push(self.strides[i]);
        }

        Ok(Array {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype.clone(),
            offset: new_offset,
        })
    }

    /// Newaxis support (arr[np.newaxis, :])
    pub fn add_newaxis(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(NumPyError::invalid_operation(format!(
                "Axis {} out of bounds for {}-D array",
                axis,
                self.ndim()
            )));
        }

        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);

        // Reshape array with new axis
        self.reshape(&new_shape)
    }

    pub fn advanced_index(&self, indices: &[crate::slicing::Index]) -> Result<Self> {
        // For now, if all indices are integers or boolean, we can handle it.
        // Mixed with slices is still complex.

        let mut has_slice = false;

        for idx in indices {
            match idx {
                crate::slicing::Index::Integer(_) => {}
                crate::slicing::Index::Slice(_) => {
                    has_slice = true;
                }
                _ => {
                    has_slice = true;
                }
            }
        }

        if !has_slice && indices.len() == self.ndim() {
            // Simplified: if all are integers, we could use fancy_index by creating 1-elem arrays
            // But usually this just returns a scalar or reduced-dim array.
        }

        self.ellipsis_index(indices)
    }

    /// Multi-dimensional indexing (arr[1:3, 5:, ::-1])
    pub fn multidim_index(&self, indices: &[crate::slicing::Index]) -> Result<Self> {
        if indices.len() != self.ndim() {
            return Err(NumPyError::invalid_operation(format!(
                "Expected {} indices for {}-D array, got {}",
                self.ndim(),
                self.ndim(),
                indices.len()
            )));
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
        self.set(linear_idx as usize, value)
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
        self.get(linear_idx as usize)
            .ok_or_else(|| NumPyError::index_error(linear_idx as usize, self.size()))
    }

    /// Expand ellipsis in indices
    pub fn ellipsis_index(&self, indices: &[Index]) -> Result<Self> {
        let mut expanded_slices = Vec::new();
        let mut ellipsis_found = false;

        // Count non-ellipsis indices
        let non_ellipsis_count = indices
            .iter()
            .filter(|i| !matches!(i, Index::Ellipsis))
            .count();

        if indices.len() - non_ellipsis_count > 1 {
            return Err(NumPyError::invalid_operation(
                "An index can only have one ellipsis (...)",
            ));
        }

        let ellipsis_expansion = self.ndim().saturating_sub(non_ellipsis_count);

        for idx in indices {
            match idx {
                Index::Ellipsis => {
                    if ellipsis_found {
                        // Already handled by check above but for safety
                        continue;
                    }
                    ellipsis_found = true;
                    for _ in 0..ellipsis_expansion {
                        expanded_slices.push(Slice::Full);
                    }
                }
                Index::Slice(s) => expanded_slices.push(s.clone()),
                Index::Integer(i) => expanded_slices.push(Slice::Index(*i)),
                Index::Boolean(b) => {
                    if *b {
                        expanded_slices.push(Slice::Full);
                    } else {
                        expanded_slices.push(Slice::Range(0, 0));
                    }
                }
            }
        }

        // If no ellipsis was found and fewer indices provided, NumPy pads with Full slices at the end
        if !ellipsis_found && expanded_slices.len() < self.ndim() {
            for _ in expanded_slices.len()..self.ndim() {
                expanded_slices.push(Slice::Full);
            }
        }

        self.slice(&MultiSlice::new(expanded_slices))
    }

    /// Calculate length of a slice for a dimension
    pub fn calculate_slice_length(&self, dim: usize, slice: &Slice) -> usize {
        slice.len(self.shape()[dim])
    }

    /// Extract data using multi-dimensional indices
    pub fn extract_multidim_data(&self, _indices: &[Index], _result: &mut Vec<T>) -> Result<()> {
        // Stub
        Err(NumPyError::not_implemented("extract_multidim_data"))
    }
}

/// Reshape array to new dimensions
pub fn reshape<T: Clone + Default + 'static>(
    array: &Array<T>,
    new_shape: Vec<usize>,
) -> Result<Array<T>> {
    let total_elements: usize = new_shape.iter().product();

    if total_elements != array.size() {
        return Err(NumPyError::invalid_value(format!(
            "Cannot reshape array of size {} into shape {:?}",
            array.size(),
            new_shape
        )));
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
    // Full slice: ..
    ( .. ) => {
        $crate::slicing::Slice::Full
    };
    // Python-style step: ::step (using : : to avoid PathSep if needed)
    ( : : $step:expr ) => {
        $crate::slicing::Slice::Step($step as isize)
    };
    // Python-style step: ::step (using :: just in case)
    ( :: $step:expr ) => {
        $crate::slicing::Slice::Step($step as isize)
    };
    // start..stop..step
    ( $start:tt .. $stop:tt .. $step:expr ) => {
        $crate::slicing::Slice::RangeStep($start as isize, $stop as isize, $step as isize)
    };
    // ..stop..step
    ( .. $stop:tt .. $step:expr ) => {
        $crate::slicing::Slice::RangeStep(0, $stop as isize, $step as isize)
    };
    // start.. ..step
    ( $start:tt .. .. $step:expr ) => {
        $crate::slicing::Slice::RangeStep($start as isize, isize::MAX, $step as isize)
    };
    // start..stop
    ( $start:tt .. $stop:tt ) => {
        $crate::slicing::Slice::Range($start as isize, $stop as isize)
    };
    // ..stop
    ( .. $stop:tt ) => {
        $crate::slicing::Slice::To($stop as isize)
    };
    // start..
    ( $start:tt .. ) => {
        $crate::slicing::Slice::From($start as isize)
    };
    // single index
    ( $idx:expr ) => {
        $crate::slicing::Slice::Index($idx as isize)
    };
}
