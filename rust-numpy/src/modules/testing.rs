use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::linalg::LinalgScalar;
use num_traits::Float;
use std::fmt::Debug;

/// Assert that two arrays are equal.
///
/// Raises an error if shapes or elements differ.
pub fn assert_array_equal<T>(actual: &Array<T>, desired: &Array<T>) -> Result<()>
where
    T: Clone + PartialEq + Debug + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        if a != b {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays differ at index {}: actual={:?}, desired={:?}",
                    i, a, b
                ),
                "".to_string(), // desired is already in the message
            ));
        }
    }

    Ok(())
}

/// Assert that two arrays are equal within a certain tolerance.
///
/// Specifically for floating point types.
pub fn assert_array_almost_equal<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    decimal: usize,
) -> Result<()>
where
    T: Clone + Float + Debug + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    let threshold = T::from(10.0).unwrap().powi(-(decimal as i32));

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        if (*a - *b).abs() > threshold {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays differ at index {} by more than {} decimal places: actual={:?}, desired={:?}",
                    i, decimal, a, b
                ),
                "".to_string(),
            ));
        }
    }

    Ok(())
}

/// Assert that two arrays are equal within tolerance
pub fn assert_allclose<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    rtol: T::Real,
    atol: T::Real,
) -> Result<()>
where
    T: LinalgScalar + Debug,
    T::Real: Debug,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        let diff = (*a - *b).abs();
        let b_abs = b.abs();
        if diff > atol + rtol * b_abs {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays not close at index {}: actual={:?}, desired={:?} (diff={:?}, tol={:?})",
                    i,
                    a,
                    b,
                    diff,
                    atol + rtol * b_abs
                ),
                "".to_string(),
            ));
        }
    }
    Ok(())
}

/// Returns True if two arrays are element-wise equal within a tolerance.
pub(crate) fn check_allclose<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    rtol: T::Real,
    atol: T::Real,
) -> bool
where
    T: LinalgScalar,
{
    if actual.shape() != desired.shape() {
        return false;
    }

    for (a, b) in actual.iter().zip(desired.iter()) {
        let diff = (*a - *b).abs();
        let b_abs = b.abs();
        if diff > atol + rtol * b_abs {
            return false;
        }
    }
    true
}

/// Assert that first array is less than second array element-wise.
pub fn assert_array_less<T>(actual: &Array<T>, desired: &Array<T>) -> Result<()>
where
    T: PartialOrd + Debug + Clone + Default + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        if a >= b {
            return Err(NumPyError::value_error(
                format!(
                    "Condition actual < desired failed at index {}: actual={:?}, desired={:?}",
                    i, a, b
                ),
                "".to_string(),
            ));
        }
    }
    Ok(())
}

/// Assert that two arrays have the same shape.
pub fn assert_array_shape_equal<T, U>(a: &Array<T>, b: &Array<U>) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(NumPyError::shape_mismatch(
            b.shape().to_vec(),
            a.shape().to_vec(),
        ));
    }
    Ok(())
}

pub mod exports {
    pub use super::{
        assert_allclose, assert_array_almost_equal, assert_array_equal, assert_array_less,
        assert_array_shape_equal,
    };
}
