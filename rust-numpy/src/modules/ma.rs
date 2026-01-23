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
        T: std::iter::Sum + Clone + num_traits::FromPrimitive + std::ops::Div<Output = T>,
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
}

pub mod exports {
    pub use super::MaskedArray;
}
