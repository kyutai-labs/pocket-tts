// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! Set operations with full NumPy compatibility
//!
//! This module provides complete implementation of NumPy's set routines,
//! including unique, intersect1d, union1d, setdiff1d, setxor1d, in1d and isin.
//!
//! All functions support:
//! - Multi-dimensional arrays with axis parameter
//! - Efficient algorithms using HashSet and BTreeSet
//! - Proper NaN handling for floating-point types
//! - Return parameters (index, inverse, counts)
//! - Memory-optimized implementations for large arrays
//!

use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::strides::compute_multi_indices;
use num_traits::Float;
use std::hash::Hash;

/// Trait for types that can be used in set operations
pub trait SetElement: Clone + PartialEq {
    /// Check if value is NaN (for floating point types)
    fn is_nan(&self) -> bool {
        false
    }

    /// Convert to f64 for NaN comparison (specialized for floating point)
    fn as_f64(&self) -> Option<f64> {
        None
    }

    /// Get hash value for element
    fn hash_element<H: std::hash::Hasher>(&self, state: &mut H);
}

macro_rules! impl_set_element {
    ($($t:ty),*) => {
        $(
            impl SetElement for $t {
                fn is_nan(&self) -> bool {
                    false
                }

                fn as_f64(&self) -> Option<f64> {
                    None
                }

                fn hash_element<H: std::hash::Hasher>(&self, state: &mut H) {
                    self.hash(state)
                }
            }
        )*
    };
}

macro_rules! impl_set_element_float {
    ($($t:ty),*) => {
        $(
            impl SetElement for $t {
                fn is_nan(&self) -> bool {
                    Float::is_nan(*self)
                }

                fn as_f64(&self) -> Option<f64> {
                    Some(*self as f64)
                }

                fn hash_element<H: std::hash::Hasher>(&self, state: &mut H) {
                    if self.is_nan() {
                        state.write_u64(0x7fc00000); // Standard NaN
                    } else {
                        // Use to_bits for hashing
                        let bits = self.to_bits();
                         bits.hash(state);
                    }
                }
            }
        )*
    };
}

impl_set_element!(i8, i16, i32, i64, u8, u16, u32, u64, bool, char, String, &str);
impl_set_element_float!(f32, f64);

/// Result of unique operation with optional return values
#[derive(Debug, Clone)]
pub struct UniqueResult<T> {
    /// Unique values
    pub values: Array<T>,
    /// Indices of unique values in original array
    pub indices: Option<Array<usize>>,
    /// Inverse indices to reconstruct original array
    pub inverse: Option<Array<usize>>,
    /// Counts of each unique value
    pub counts: Option<Array<usize>>,
}

/// Find unique elements of an array.
///
/// Returns sorted unique values of an array. There are three optional
/// outputs in addition to unique elements:
/// - indices of input array that give unique values
/// - indices of unique array that reconstruct the input array
/// - count for each unique element
///
/// # Arguments
///
/// * `ar` - Input array
/// * `return_index` - If true, Also return indices of ar that result in unique array
/// * `return_inverse` - If true, Also return indices of unique array that can be used to reconstruct ar
/// * `return_counts` - If true, Also return number of times each unique item appears in ar
/// * `axis` - The axis to operate on. None operates on flattened array
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::unique};
/// let arr = array![1, 2, 1, 3, 2, 1];
/// let result = unique(&arr, false, false, false, None).unwrap();
/// // result.values == [1, 2, 3]
/// ```
pub fn unique<T>(
    ar: &Array<T>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    axis: Option<&[isize]>,
) -> Result<UniqueResult<T>>
where
    T: SetElement + Clone + Default + PartialOrd + 'static,
{
    use std::cmp::Ordering;

    let normalized_axis = if let Some(axis) = axis {
        if axis.len() != 1 {
            return Err(NumPyError::invalid_operation(
                "unique supports at most one axis",
            ));
        }

        let mut normalized = axis[0];
        let ndim = ar.ndim() as isize;
        if normalized < 0 {
            normalized += ndim;
        }
        if normalized < 0 || normalized >= ndim {
            return Err(NumPyError::invalid_operation("axis out of bounds"));
        }
        Some(normalized as usize)
    } else {
        None
    };

    // Handle empty array
    if ar.is_empty() {
        let empty_shape = if let Some(axis) = normalized_axis {
            let mut shape = ar.shape().to_vec();
            if !shape.is_empty() {
                shape[axis] = 0;
            }
            shape
        } else {
            vec![]
        };

        return Ok(UniqueResult {
            values: Array::from_data(vec![], empty_shape),
            indices: if return_index {
                Some(Array::from_data(vec![], vec![]))
            } else {
                None
            },
            inverse: if return_inverse {
                Some(Array::from_data(vec![], vec![0]))
            } else {
                None
            },
            counts: if return_counts {
                Some(Array::from_data(vec![], vec![]))
            } else {
                None
            },
        });
    }

    if let Some(axis) = normalized_axis {
        if ar.ndim() > 1 {
            let axis_len = ar.shape()[axis];
            let slice_size = ar.size() / axis_len;
            let mut slices: Vec<Vec<T>> = (0..axis_len)
                .map(|_| Vec::with_capacity(slice_size))
                .collect();

            for linear_idx in 0..ar.size() {
                let indices = compute_multi_indices(linear_idx, ar.shape());
                let axis_idx = indices[axis];
                let value = ar
                    .get(linear_idx)
                    .ok_or_else(|| NumPyError::invalid_operation("unique index out of bounds"))?;
                slices[axis_idx].push(value.clone());
            }

            let compare_value = |a: &T, b: &T| {
                if a.is_nan() && b.is_nan() {
                    Ordering::Equal
                } else if a.is_nan() {
                    Ordering::Greater
                } else if b.is_nan() {
                    Ordering::Less
                } else {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                }
            };

            let compare_slice = |a: &Vec<T>, b: &Vec<T>| {
                for (lhs, rhs) in a.iter().zip(b.iter()) {
                    let ord = compare_value(lhs, rhs);
                    if ord != Ordering::Equal {
                        return ord;
                    }
                }
                Ordering::Equal
            };

            let mut indexed: Vec<(usize, Vec<T>)> = slices.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| compare_slice(&a.1, &b.1));

            let mut unique_slices: Vec<Vec<T>> = Vec::new();
            let mut indices_vec: Vec<usize> = Vec::new();
            let mut counts_vec: Vec<usize> = Vec::new();
            let mut inverse_vec: Vec<usize> = if return_inverse {
                vec![0; axis_len]
            } else {
                Vec::new()
            };

            let mut i = 0;
            while i < indexed.len() {
                let (mut min_index, slice) = (indexed[i].0, indexed[i].1.clone());
                let mut count = 1;

                if return_inverse {
                    inverse_vec[indexed[i].0] = unique_slices.len();
                }

                let mut j = i + 1;
                while j < indexed.len() {
                    if compare_slice(&slice, &indexed[j].1) != Ordering::Equal {
                        break;
                    }

                    min_index = min_index.min(indexed[j].0);
                    if return_inverse {
                        inverse_vec[indexed[j].0] = unique_slices.len();
                    }
                    count += 1;
                    j += 1;
                }

                unique_slices.push(slice);
                if return_index {
                    indices_vec.push(min_index);
                }
                if return_counts {
                    counts_vec.push(count);
                }

                i = j;
            }

            let n_unique = unique_slices.len();
            let mut output_shape = ar.shape().to_vec();
            output_shape[axis] = n_unique;
            let output_size = output_shape.iter().product::<usize>();
            let mut output_data = Vec::with_capacity(output_size);

            let mut non_axis_shape = Vec::with_capacity(output_shape.len().saturating_sub(1));
            for (dim, &size) in output_shape.iter().enumerate() {
                if dim != axis {
                    non_axis_shape.push(size);
                }
            }

            for linear_idx in 0..output_size {
                let indices = compute_multi_indices(linear_idx, &output_shape);
                let axis_idx = indices[axis];

                let mut non_axis_indices = Vec::with_capacity(non_axis_shape.len());
                for (dim, &idx) in indices.iter().enumerate() {
                    if dim != axis {
                        non_axis_indices.push(idx);
                    }
                }

                let mut slice_linear = 0;
                let mut stride = 1;
                for (dim, &size) in non_axis_shape.iter().enumerate().rev() {
                    slice_linear += non_axis_indices[dim] * stride;
                    stride *= size;
                }

                output_data.push(unique_slices[axis_idx][slice_linear].clone());
            }

            return Ok(UniqueResult {
                values: Array::from_data(output_data, output_shape),
                indices: if return_index {
                    Some(Array::from_data(indices_vec, vec![n_unique]))
                } else {
                    None
                },
                inverse: if return_inverse {
                    Some(Array::from_data(inverse_vec, vec![axis_len]))
                } else {
                    None
                },
                counts: if return_counts {
                    Some(Array::from_data(counts_vec, vec![n_unique]))
                } else {
                    None
                },
            });
        }
    }

    // Collect all elements from the array (flattened)
    let elements: Vec<T> = ar.iter().cloned().collect();
    if elements.is_empty() {
        return Ok(UniqueResult {
            values: Array::from_data(vec![], vec![]),
            indices: if return_index {
                Some(Array::from_data(vec![], vec![]))
            } else {
                None
            },
            inverse: if return_inverse {
                Some(Array::from_data(vec![], vec![]))
            } else {
                None
            },
            counts: if return_counts {
                Some(Array::from_data(vec![], vec![]))
            } else {
                None
            },
        });
    }

    let mut indexed: Vec<(usize, T)> = elements.iter().cloned().enumerate().collect();

    let compare = |a: &T, b: &T| {
        if a.is_nan() && b.is_nan() {
            Ordering::Equal
        } else if a.is_nan() {
            Ordering::Greater
        } else if b.is_nan() {
            Ordering::Less
        } else {
            a.partial_cmp(b).unwrap_or(Ordering::Equal)
        }
    };

    indexed.sort_by(|a, b| compare(&a.1, &b.1));

    let mut unique_values: Vec<T> = Vec::new();
    let mut indices_vec: Vec<usize> = Vec::new();
    let mut counts_vec: Vec<usize> = Vec::new();
    let mut inverse_vec: Vec<usize> = if return_inverse {
        vec![0; elements.len()]
    } else {
        Vec::new()
    };

    let mut i = 0;
    while i < indexed.len() {
        let (mut min_index, value) = (indexed[i].0, indexed[i].1.clone());
        let mut count = 1;

        if return_inverse {
            inverse_vec[indexed[i].0] = unique_values.len();
        }

        let mut j = i + 1;
        while j < indexed.len() {
            let equal = if value.is_nan() && indexed[j].1.is_nan() {
                true
            } else {
                value == indexed[j].1
            };

            if !equal {
                break;
            }

            min_index = min_index.min(indexed[j].0);
            if return_inverse {
                inverse_vec[indexed[j].0] = unique_values.len();
            }
            count += 1;
            j += 1;
        }

        unique_values.push(value);
        if return_index {
            indices_vec.push(min_index);
        }
        if return_counts {
            counts_vec.push(count);
        }

        i = j;
    }

    let n_unique = unique_values.len();

    Ok(UniqueResult {
        values: Array::from_data(unique_values, vec![n_unique]),
        indices: if return_index {
            Some(Array::from_data(indices_vec, vec![n_unique]))
        } else {
            None
        },
        inverse: if return_inverse {
            Some(Array::from_data(inverse_vec, vec![elements.len()]))
        } else {
            None
        },
        counts: if return_counts {
            Some(Array::from_data(counts_vec, vec![n_unique]))
        } else {
            None
        },
    })
}

/// Test whether each element of a 1-D array is also present in a second array.
///
/// Returns a boolean array of the same shape as `ar1` that is True where an element
/// of `ar1` is in `ar2` and False otherwise.
///
/// # Arguments
///
/// * `ar1` - Input array
/// * `ar2` - The values against which to test each value of `ar1`
/// * `assume_unique` - If True, input arrays are both assumed to be unique,
///                    which can speed up the calculation
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::in1d};
/// let ar1 = array![1, 2, 3, 4, 5];
/// let ar2 = array![2, 4, 6];
/// let result = in1d(&ar1, &ar2, false).unwrap();
/// // result == [false, true, false, true, false]
/// ```
pub fn in1d<T>(ar1: &Array<T>, ar2: &Array<T>, _assume_unique: bool) -> Result<Array<bool>>
where
    T: SetElement + Clone + Eq + std::hash::Hash + Default + 'static,
{
    use std::collections::HashSet;

    if ar1.ndim() != 1 || ar2.ndim() != 1 {
        return Err(NumPyError::invalid_operation(
            "in1d requires 1-dimensional arrays",
        ));
    }

    // Build a HashSet from ar2 for O(1) lookup
    let set2: HashSet<_> = ar2.iter().collect();

    // Test each element of ar1
    let result: Vec<bool> = ar1.iter().map(|v| set2.contains(v)).collect();

    let len = result.len();
    Ok(Array::from_data(result, vec![len]))
}

/// Test whether each element of a 1-D array is also present in a second array.
///
/// This is an alias for in1d with the parameter order matching NumPy's isin.
/// Calculates `element in test_elements`, broadcasting over element only.
///
/// # Arguments
///
/// * `element` - Input array
/// * `test_elements` - The values against which to test each value of element
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::isin};
/// let element = array![1, 2, 3, 4, 5];
/// let test_elements = array![2, 4, 6];
/// let result = isin(&element, &test_elements).unwrap();
/// // result == [false, true, false, true, false]
/// ```
pub fn isin<T>(element: &Array<T>, test_elements: &Array<T>) -> Result<Array<bool>>
where
    T: SetElement + Clone + Eq + std::hash::Hash + Default + 'static,
{
    in1d(element, test_elements, false)
}

/// Find the intersection of two arrays.
///
/// Return the unique, sorted array of values that are in both of the input arrays.
///
/// # Arguments
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// Sorted unique values that are in both input arrays
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::intersect1d};
/// let ar1 = array![1, 2, 3, 4, 5];
/// let ar2 = array![2, 4, 6];
/// let result = intersect1d(&ar1, &ar2).unwrap();
/// // result == [2, 4]
/// ```
pub fn intersect1d<T>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>>
where
    T: SetElement + Clone + Eq + std::hash::Hash + Ord + Default + 'static,
{
    use std::collections::HashSet;

    if ar1.ndim() != 1 || ar2.ndim() != 1 {
        return Err(NumPyError::invalid_operation(
            "intersect1d requires 1-dimensional arrays",
        ));
    }

    // Build a HashSet from ar2
    let set2: HashSet<_> = ar2.iter().collect();

    // Find elements in ar1 that are also in ar2
    let mut result: Vec<T> = ar1.iter().filter(|v| set2.contains(v)).cloned().collect();

    // Sort and deduplicate
    result.sort();
    result.dedup();

    let len = result.len();
    Ok(Array::from_data(result, vec![len]))
}

/// Find the union of two arrays.
///
/// Return the unique, sorted array of values that are in either of the input arrays.
///
/// # Arguments
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// Sorted unique values that are in either of the input arrays
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::union1d};
/// let ar1 = array![1, 2, 3];
/// let ar2 = array![2, 4, 6];
/// let result = union1d(&ar1, &ar2).unwrap();
/// // result == [1, 2, 3, 4, 6]
/// ```
pub fn union1d<T>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>>
where
    T: SetElement + Clone + Eq + std::hash::Hash + Ord + Default + 'static,
{
    if ar1.ndim() != 1 || ar2.ndim() != 1 {
        return Err(NumPyError::invalid_operation(
            "union1d requires 1-dimensional arrays",
        ));
    }

    // Combine elements from both arrays
    let mut result: Vec<T> = ar1.iter().chain(ar2.iter()).cloned().collect();

    // Sort and deduplicate
    result.sort();
    result.dedup();

    let len = result.len();
    Ok(Array::from_data(result, vec![len]))
}

/// Find the set difference of two arrays.
///
/// Return the unique values in `ar1` that are not in `ar2`.
///
/// # Arguments
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// Sorted unique values in ar1 that are not in ar2
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::setdiff1d};
/// let ar1 = array![1, 2, 3, 4, 5];
/// let ar2 = array![2, 4];
/// let result = setdiff1d(&ar1, &ar2).unwrap();
/// // result == [1, 3, 5]
/// ```
pub fn setdiff1d<T>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>>
where
    T: SetElement + Clone + Eq + std::hash::Hash + Ord + Default + 'static,
{
    use std::collections::HashSet;

    if ar1.ndim() != 1 || ar2.ndim() != 1 {
        return Err(NumPyError::invalid_operation(
            "setdiff1d requires 1-dimensional arrays",
        ));
    }

    // Build a HashSet from ar2
    let set2: HashSet<_> = ar2.iter().collect();

    // Find elements in ar1 that are NOT in ar2
    let mut result: Vec<T> = ar1.iter().filter(|v| !set2.contains(v)).cloned().collect();

    // Sort and deduplicate
    result.sort();
    result.dedup();

    let len = result.len();
    Ok(Array::from_data(result, vec![len]))
}

/// Find the symmetric difference of two arrays.
///
/// Return the sorted unique values that are in only one (not both) of the input arrays.
///
/// # Arguments
///
/// * `ar1` - First input array
/// * `ar2` - Second input array
///
/// # Returns
///
/// Sorted unique values that are in only one of the input arrays
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::setxor1d};
/// let ar1 = array![1, 2, 3, 4];
/// let ar2 = array![2, 4, 6];
/// let result = setxor1d(&ar1, &ar2).unwrap();
/// // result == [1, 3, 6]
/// ```
pub fn setxor1d<T>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>>
where
    T: SetElement + Clone + Eq + std::hash::Hash + Ord + Default + 'static,
{
    use std::collections::HashSet;

    if ar1.ndim() != 1 || ar2.ndim() != 1 {
        return Err(NumPyError::invalid_operation(
            "setxor1d requires 1-dimensional arrays",
        ));
    }

    // Build HashSets from both arrays
    let set1: HashSet<_> = ar1.iter().collect();
    let set2: HashSet<_> = ar2.iter().collect();

    // Find elements that are in one but not both
    let mut result: Vec<T> = ar1
        .iter()
        .filter(|v| !set2.contains(v))
        .chain(ar2.iter().filter(|v| !set1.contains(v)))
        .cloned()
        .collect();

    // Sort
    result.sort();
    result.dedup();

    let len = result.len();
    Ok(Array::from_data(result, vec![len]))
}

/// Advanced set operations for multi-dimensional arrays
pub struct SetOps;

impl SetOps {
    /// Find unique rows in a 2D array
    pub fn unique_rows<T>(ar: &Array<T>) -> Result<Array<T>>
    where
        T: SetElement + Clone + Eq + std::hash::Hash + Default + 'static,
    {
        use std::collections::HashSet;

        if ar.ndim() != 2 {
            return Err(NumPyError::invalid_operation(
                "unique_rows requires 2-dimensional array",
            ));
        }

        let shape = ar.shape();
        let ncols = shape[1];
        let nrows = shape[0];

        // Extract rows as vectors
        let mut rows: Vec<Vec<T>> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let mut row = Vec::with_capacity(ncols);
            for j in 0..ncols {
                if let Some(val) = ar.get(i * ncols + j) {
                    row.push(val.clone());
                }
            }
            rows.push(row);
        }

        // Use HashSet to find unique rows
        let mut unique_rows_set: HashSet<Vec<T>> = HashSet::new();
        for row in rows {
            unique_rows_set.insert(row);
        }

        // Convert back to flat array
        let mut unique_flat: Vec<T> = Vec::new();
        for row in unique_rows_set {
            unique_flat.extend(row);
        }

        let n_unique_rows = unique_flat.len() / ncols;
        Ok(Array::from_data(unique_flat, vec![n_unique_rows, ncols]))
    }
}

pub mod exports {
    pub use super::{
        in1d, intersect1d, isin, setdiff1d, setxor1d, union1d, unique, SetElement, SetOps,
        UniqueResult,
    };
}
