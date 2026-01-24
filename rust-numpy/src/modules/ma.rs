use crate::array::Array;
use crate::dtype::Casting;
use crate::error::{NumPyError, Result};
use std::fmt::Debug;

/// A MaskedArray is an array that may have missing or invalid entries.
/// It consists of a data array and a boolean mask of the same shape.
/// Elements where the mask is `true` are considered invalid/masked.
#[derive(Debug, Clone)]
pub struct MaskedArray<T> {
    data: Array<T>,
    mask: Array<bool>,
    fill_value: Option<T>,
}

impl<T> MaskedArray<T>
where
    T: Clone + Debug + Default + 'static,
{
    /// Create a new MaskedArray from data and mask.
    pub fn new(data: Array<T>, mask: Array<bool>) -> Result<Self> {
        if data.shape() != mask.shape() {
            return Err(NumPyError::shape_mismatch(
                data.shape().to_vec(),
                mask.shape().to_vec(),
            ));
        }
        Ok(Self {
            data,
            mask,
            fill_value: None,
        })
    }

    /// Create a new MaskedArray from data, with no elements masked.
    pub fn from_data(data: Array<T>) -> Self {
        let mask = Array::from_data(vec![false; data.size()], data.shape().to_vec());
        Self {
            data,
            mask,
            fill_value: None,
        }
    }

    /// Return the data array.
    pub fn data(&self) -> &Array<T> {
        &self.data
    }

    /// Return the mask array.
    pub fn mask(&self) -> &Array<bool> {
        &self.mask
    }

    /// Return the shape of the masked array.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Return the total number of elements.
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Set the fill value.
    pub fn set_fill_value(&mut self, value: T) {
        self.fill_value = Some(value);
    }

    /// Get the fill value, or a default if not set.
    pub fn fill_value(&self) -> T {
        self.fill_value.clone().unwrap_or_else(T::default)
    }

    /// Return an array where masked values are replaced by the fill value.
    pub fn filled(&self) -> Array<T> {
        let mut result = self.data.clone();
        let fill = self.fill_value();

        let mask_data = self.mask.data();

        for (i, &is_masked) in mask_data.iter().enumerate() {
            if is_masked {
                result.set_linear(i, fill.clone());
            }
        }
        result
    }

    /// Sum of array elements over a given axis, respecting the mask.
    pub fn sum(&self) -> Result<T>
    where
        T: std::iter::Sum + Clone,
    {
        let mask_data = self.mask.data();
        let data = self.data.data();

        let filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        Ok(filtered.into_iter().sum())
    }

    /// Mean of array elements, respecting the mask.
    pub fn mean(&self) -> Result<T>
    where
        T: std::iter::Sum + Clone + num_traits::FromPrimitive + std::ops::Div<Output = T> + Default,
    {
        let mask_data = self.mask.data();
        let data = self.data.data();

        let filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        let count = filtered.len();
        if count == 0 {
            return Ok(T::default());
        }

        let sum: T = filtered.into_iter().sum();
        let count_t = T::from_usize(count)
            .ok_or_else(|| NumPyError::value_error("Failed to convert count to type T", "usize"))?;

        Ok(sum / count_t)
    }

    /// Perform a binary operation between two masked arrays.
    pub fn binary_op<F>(&self, other: &MaskedArray<T>, op: F) -> Result<MaskedArray<T>>
    where
        F: Fn(&Array<T>, &Array<T>, Option<&Array<bool>>, Casting) -> Result<Array<T>>,
    {
        // Broadcast masks
        // For simplicity in this initial implementation, we assume same shapes or handle it via ufunc
        // NumPy rule: mask is true if either input mask is true
        let mut new_mask_data = Vec::with_capacity(self.size());
        let m1 = self.mask.data();
        let m2 = other.mask.data();

        for (a, b) in m1.iter().zip(m2.iter()) {
            new_mask_data.push(*a || *b);
        }
        let new_mask = Array::from_data(new_mask_data, self.shape().to_vec());

        // Execute operation on data.
        // We use Safe casting and no mask for the underlying data call to ensure all elements are processed,
        // then we apply the combined mask to the result.
        let new_data = op(&self.data, &other.data, None, Casting::Safe)?;

        MaskedArray::new(new_data, new_mask)
    }

    /// Compress masked array along given axis, returning only unmasked values.
    pub fn compress(&self, condition: &Array<bool>, _axis: Option<usize>) -> Result<MaskedArray<T>>
    where
        T: Clone + Default + 'static,
    {
        let cond_data = condition.data();
        let data = self.data.data();
        let mask_data = self.mask.data();

        let filtered_data: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .zip(cond_data.iter())
            .filter(|((_, &m), &c)| !m && c)
            .map(|((d, _), _)| d.clone())
            .collect();

        let len = filtered_data.len();
        let filtered_mask: Vec<bool> = vec![false; len];

        Ok(MaskedArray::new(
            Array::from_vec(filtered_data),
            Array::from_data(filtered_mask, vec![len]),
        )?)
    }

    /// Median of array elements, respecting the mask.
    pub fn median(&self) -> Result<T>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + PartialOrd
            + num_traits::FromPrimitive
            + num_traits::NumCast
            + 'static,
    {
        let mask_data = self.mask.data();
        let data = self.data.data();

        let mut filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        let count = filtered.len();
        if count == 0 {
            return Ok(T::default());
        }

        // Sort using partial_cmp (for floats)
        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if count % 2 == 0 {
            let mid1 = &filtered[count / 2 - 1];
            let mid2 = &filtered[count / 2];

            // For numeric types, compute average
            if let (Some(a), Some(b)) = (
                num_traits::cast::<T, f64>(mid1.clone()),
                num_traits::cast::<T, f64>(mid2.clone()),
            ) {
                let avg = (a + b) / 2.0;
                if let Some(result) = num_traits::cast::<f64, T>(avg) {
                    return Ok(result);
                }
            }

            // Fallback: just return the first middle value
            Ok(mid1.clone())
        } else {
            Ok(filtered[count / 2].clone())
        }
    }

    /// Variance of array elements, respecting the mask.
    pub fn var(&self, ddof: Option<usize>) -> Result<T>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + num_traits::Float
            + num_traits::FromPrimitive
            + std::iter::Sum
            + 'static,
    {
        let mean_val = self.mean()?;
        let mask_data = self.mask.data();
        let data = self.data.data();

        let filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        let count = filtered.len();
        if count <= ddof.unwrap_or(0) {
            return Ok(T::nan());
        }

        let sum_sq_diff: T = filtered
            .iter()
            .map(|x| {
                let diff = *x - mean_val;
                diff * diff
            })
            .sum();

        let denominator = T::from_usize(count - ddof.unwrap_or(0)).unwrap_or_else(T::one);
        Ok(sum_sq_diff / denominator)
    }

    /// Standard deviation of array elements, respecting the mask.
    pub fn std(&self, ddof: Option<usize>) -> Result<T>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + num_traits::Float
            + num_traits::FromPrimitive
            + std::iter::Sum
            + 'static,
    {
        let variance = self.var(ddof)?;
        Ok(variance.sqrt())
    }

    /// Return unique values, respecting the mask.
    pub fn unique(&self) -> Result<MaskedArray<T>>
    where
        T: Clone + Default + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + 'static,
    {
        use std::collections::HashSet;

        let mask_data = self.mask.data();
        let data = self.data.data();

        // Use HashSet to get unique values
        let unique_set: HashSet<&T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d)
            .collect();

        let unique_data: Vec<T> = unique_set.into_iter().map(|v| v.clone()).collect();
        let len = unique_data.len();
        let unique_mask = vec![false; len];

        Ok(MaskedArray::new(
            Array::from_vec(unique_data),
            Array::from_data(unique_mask, vec![len]),
        )?)
    }
    /// Mask values equal to a given value.
    pub fn masked_values(data: Array<T>, value: T) -> Result<Self>
    where
        T: PartialEq + Clone + Default + Debug + 'static,
    {
        let mask_data: Vec<bool> = data.data().iter().map(|x| *x == value).collect();
        let mask = Array::from_data(mask_data, data.shape().to_vec());
        let mut ma = Self::new(data, mask)?;
        ma.set_fill_value(value);
        Ok(ma)
    }

    /// Mask values outside a given range [min, max].
    pub fn masked_outside(data: Array<T>, min: T, max: T) -> Result<Self>
    where
        T: PartialOrd + Clone + Default + Debug + 'static,
    {
        let mask_data: Vec<bool> = data.data().iter().map(|x| *x < min || *x > max).collect();
        let mask = Array::from_data(mask_data, data.shape().to_vec());
        Self::new(data, mask)
    }

    /// Mask values inside a given range [min, max].
    pub fn masked_inside(data: Array<T>, min: T, max: T) -> Result<Self>
    where
        T: PartialOrd + Clone + Default + Debug + 'static,
    {
        let mask_data: Vec<bool> = data.data().iter().map(|x| *x >= min && *x <= max).collect();
        let mask = Array::from_data(mask_data, data.shape().to_vec());
        Self::new(data, mask)
    }

    /// Count the non-masked elements of the array.
    pub fn count(&self) -> usize {
        self.mask.data().iter().filter(|&&m| !m).count()
    }

    /// Return list of slices corresponding to clustered masked values (1D only).
    pub fn clump_masked(&self) -> Result<Vec<std::ops::Range<usize>>> {
        if self.ndim() != 1 {
            return Err(NumPyError::invalid_value(
                "clump_masked is only supported for 1D arrays",
            ));
        }
        let mask = self.mask.data();
        let mut slices = Vec::new();
        let mut in_clump = false;
        let mut start = 0;

        for (i, &m) in mask.iter().enumerate() {
            if m {
                if !in_clump {
                    in_clump = true;
                    start = i;
                }
            } else if in_clump {
                in_clump = false;
                slices.push(start..i);
            }
        }
        if in_clump {
            slices.push(start..mask.len());
        }
        Ok(slices)
    }

    /// Return list of slices corresponding to clustered unmasked values (1D only).
    pub fn clump_unmasked(&self) -> Result<Vec<std::ops::Range<usize>>> {
        if self.ndim() != 1 {
            return Err(NumPyError::invalid_value(
                "clump_unmasked is only supported for 1D arrays",
            ));
        }
        let mask = self.mask.data();
        let mut slices = Vec::new();
        let mut in_clump = false; // "clump" of UNmasked values
        let mut start = 0;

        for (i, &m) in mask.iter().enumerate() {
            if !m {
                // Unmasked
                if !in_clump {
                    in_clump = true;
                    start = i;
                }
            } else if in_clump {
                in_clump = false;
                slices.push(start..i);
            }
        }
        if in_clump {
            slices.push(start..mask.len());
        }
        Ok(slices)
    }
}

pub mod exports {
    pub use super::MaskedArray;
}
