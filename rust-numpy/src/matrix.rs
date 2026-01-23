// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Matrix class with matrix multiplication as default multiplication

use crate::array::Array;
use crate::comparison_ufuncs::ComparisonOps;
use crate::error::{NumPyError, Result};
use crate::linalg::products::matmul;
use crate::linalg::solvers::inv;
use crate::linalg::LinalgScalar;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

/// A 2D array wrapper that provides matrix multiplication as the default multiplication operator.
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    array: Array<T>,
}

impl<T> Matrix<T>
where
    T: Clone + Default + 'static,
{
    /// Create a new Matrix from an Array. Array must be 2D.
    pub fn new(array: Array<T>) -> Result<Self> {
        if array.ndim() != 2 {
            return Err(NumPyError::invalid_operation(format!(
                "Matrix must be 2D, got {}-D array",
                array.ndim()
            )));
        }
        Ok(Self { array })
    }

    /// Create a new Matrix from data and shape.
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let array = Array::from_shape_vec(shape, data);
        Self::new(array)
    }

    /// Get reference to the underlying Array.
    pub fn array(&self) -> &Array<T> {
        &self.array
    }

    /// Convert to the underlying Array.
    pub fn to_array(self) -> Array<T> {
        self.array
    }

    /// Transpose of the matrix.
    #[allow(non_snake_case)]
    pub fn T(&self) -> Self {
        Self {
            array: self.array.transpose(),
        }
    }
}

impl<T> Matrix<T>
where
    T: LinalgScalar + 'static,
{
    /// Matrix multiplication
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let res = matmul(&self.array, &rhs.array)?;
        Self::new(res)
    }
}

impl<T> Matrix<T>
where
    T: LinalgScalar + ComparisonOps<T> + Debug + 'static,
{
    /// Inverse of the matrix.
    #[allow(non_snake_case)]
    pub fn I(&self) -> Result<Self> {
        let inverse = inv(&self.array)?;
        Self::new(inverse)
    }

    /// Conjugate transpose (Hermitian transpose).
    #[allow(non_snake_case)]
    pub fn H(&self) -> Self {
        // For real types, this is just transpose.
        // For complex types, we would need conjugate.
        // Currently, we'll just return transpose as a placeholder for complex support.
        Self {
            array: self.array.transpose(),
        }
    }
}

// Element-wise addition
impl<T> Add for Matrix<T>
where
    T: LinalgScalar + 'static,
{
    type Output = Result<Matrix<T>>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.array.shape() != rhs.array.shape() {
            return Err(NumPyError::shape_mismatch(
                self.array.shape().to_vec(),
                rhs.array.shape().to_vec(),
            ));
        }

        let mut data = Vec::with_capacity(self.array.size());
        for (a, b) in self.array.iter().zip(rhs.array.iter()) {
            data.push(*a + *b);
        }

        Self::from_data(data, self.array.shape().to_vec())
    }
}

// Element-wise subtraction
impl<T> Sub for Matrix<T>
where
    T: LinalgScalar + 'static,
{
    type Output = Result<Matrix<T>>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.array.shape() != rhs.array.shape() {
            return Err(NumPyError::shape_mismatch(
                self.array.shape().to_vec(),
                rhs.array.shape().to_vec(),
            ));
        }

        let mut data = Vec::with_capacity(self.array.size());
        for (a, b) in self.array.iter().zip(rhs.array.iter()) {
            data.push(*a - *b);
        }

        Self::from_data(data, self.array.shape().to_vec())
    }
}

// Matrix multiplication as default Mul
impl<T> Mul for Matrix<T>
where
    T: LinalgScalar + 'static,
{
    type Output = Result<Matrix<T>>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.matmul(&rhs)
    }
}

pub mod exports {
    pub use super::Matrix;
}
