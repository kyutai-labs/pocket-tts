//! Array creation and manipulation functions

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};

/// Create an array from a Python-like list of values.
/// This is a simplified version of NumPy's np.array() that creates a 1D array from a slice of f32 values.
pub fn array<T>(data: Vec<T>, _dtype: Option<Dtype>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let arr = Array::from_vec(data);
    Ok(arr)
}

/// Create an array from a range of values (similar to np.arange).
///
/// # Arguments
/// - `start`: Start value (inclusive)
/// - `stop`: Stop value (exclusive)
/// - `step`: Step size (optional, defaults to 1.0)
///
/// # Returns
/// 1D array of f32 values
///
/// # Examples
/// ```ignore
/// let arr = arange(0.0, 5.0, None).unwrap();
/// let arr2 = arange(0.0, 10.0, Some(2.0)).unwrap();
/// ```
pub fn arange(start: f32, stop: f32, step: Option<f32>) -> Result<Array<f32>> {
    let step_val = step.unwrap_or(1.0f32);

    if step_val == 0.0 {
        return Err(NumPyError::invalid_value("step cannot be zero"));
    }

    let num_elements = if step_val > 0.0 {
        if stop <= start {
            return Ok(Array::from_vec(vec![]));
        }
        ((stop - start) / step_val).ceil() as usize
    } else {
        if stop >= start {
            return Ok(Array::from_vec(vec![]));
        }
        ((start - stop) / step_val.abs()).ceil() as usize
    };

    let data: Vec<f32> = (0..num_elements)
        .map(|i| start + (i as f32) * step_val)
        .collect();

    Ok(Array::from_vec(data))
}

/// Clip values to be within a specified range (similar to np.clip).
///
/// # Arguments
/// - `array`: Input array
/// - `a_min`: Minimum value (values below this are set to this, optional)
/// - `a_max`: Maximum value (values above this are set to this, optional)
///
/// # Returns
/// Array with clipped values
///
/// # Examples
/// ```ignore
/// let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// let clipped = clip(&arr, Some(2.0), Some(4.0)).unwrap();
/// // Result: [2.0, 2.0, 3.0, 4.0, 4.0]
/// ```
pub fn clip<T>(array: &Array<T>, a_min: Option<T>, a_max: Option<T>) -> Result<Array<T>>
where
    T: PartialOrd + Clone + num_traits::Bounded + Default + 'static,
{
    let clipped: Vec<T> = array
        .data()
        .iter()
        .map(|x| {
            let mut val = x.clone();
            if let Some(min_v) = &a_min {
                if val < *min_v {
                    val = min_v.clone();
                }
            }
            if let Some(max_v) = &a_max {
                if val > *max_v {
                    val = max_v.clone();
                }
            }
            val
        })
        .collect();

    Ok(Array::from_vec(clipped))
}

/// Find minimum value in an array (similar to np.min).
///
/// # Arguments
/// - `array`: Input array
///
/// # Returns
/// Minimum value or error if array is empty
///
/// # Examples
/// ```ignore
/// let arr = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
/// let min_val = min(&arr).unwrap();  // Returns 1.0
/// ```
pub fn min<T>(array: &Array<T>) -> Result<T>
where
    T: PartialOrd + Copy + num_traits::Bounded + 'static,
{
    if array.size() == 0 {
        return Err(NumPyError::invalid_value("min() arg is an empty sequence"));
    }

    Ok(array
        .data()
        .iter()
        .cloned()
        .fold(T::max_value(), |a, b| if a < b { a } else { b }))
}

/// Compute natural logarithm element-wise (similar to np.log).
///
/// # Arguments
/// - `array`: Input array
///
/// # Returns
/// Array with natural log of each element. Negative values return -inf.
///
/// # Examples
/// ```ignore
/// let arr = Array::from_vec(vec![1.0, 2.0, 10.0]).unwrap();
/// let logged = log(&arr).unwrap();  // Returns [0.0, 0.69..., 2.30...]
/// ```
pub fn log<T>(array: &Array<T>) -> Result<Array<T>>
where
    T: num_traits::Float + Default + 'static,
{
    let logged: Vec<T> = array
        .data()
        .iter()
        .map(|&x| {
            let x_f64 = x.to_f64().unwrap_or(0.0);
            if x_f64 <= 0.0 {
                T::from(f64::NEG_INFINITY).unwrap_or(T::zero())
            } else {
                T::from(x_f64.ln()).unwrap_or(T::zero())
            }
        })
        .collect();

    Ok(Array::from_vec(logged))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arange_basic() {
        let arr = arange(0.0, 10.0, None).unwrap();
        assert_eq!(arr.shape(), &[10]);
        let data = arr.data();
        for i in 0..10 {
            assert_eq!(data[i], i as f32);
        }
    }

    #[test]
    fn test_arange_with_step() {
        let arr = arange(0.0, 10.0, Some(2.0)).unwrap();
        assert_eq!(arr.shape(), &[5]);
        let data = arr.data();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 4.0);
        assert_eq!(data[3], 6.0);
        assert_eq!(data[4], 8.0);
    }

    #[test]
    fn test_arange_negative_step() {
        let arr = arange(10.0, 0.0, Some(-1.0)).unwrap();
        assert_eq!(arr.shape(), &[10]);
        let data = arr.data();
        for i in 0..10 {
            assert_eq!(data[i], (10.0 - i as f32));
        }
    }

    #[test]
    fn test_clip_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(input);
        let clipped = clip(&arr, Some(2.0), Some(4.0)).unwrap();
        let data = clipped.data();
        assert_eq!(data[0], 2.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
        assert_eq!(data[3], 4.0);
        assert_eq!(data[4], 4.0);
    }

    #[test]
    fn test_clip_no_min() {
        let input = vec![1.0, 2.0, 3.0];
        let arr = Array::from_vec(input);
        let clipped = clip(&arr, None, Some(2.0)).unwrap();
        let data = clipped.data();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 2.0);
    }

    #[test]
    fn test_clip_no_max() {
        let input = vec![1.0, 2.0, 3.0];
        let arr = Array::from_vec(input);
        let clipped = clip(&arr, Some(2.0), None).unwrap();
        let data = clipped.data();
        assert_eq!(data[0], 2.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0); // No max, so 3 stays 3
    }

    #[test]
    fn test_min() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let arr = Array::from_vec(input);
        let min_val = min(&arr).unwrap();
        assert_eq!(min_val, 1.0);
    }

    #[test]
    fn test_min_empty() {
        let input: Vec<f32> = vec![];
        let arr = Array::from_vec(input);
        let result = min(&arr);
        assert!(result.is_err());
    }

    #[test]
    fn test_log() {
        let input = vec![1.0, 2.0, 10.0, 0.5, std::f32::consts::E];
        let arr = Array::from_vec(input);
        let logged = log(&arr).unwrap();
        let data = logged.data();
        assert!((data[0] - 0.0).abs() < 1e-6); // log(1.0) = 0.0
        assert!((data[1] - 2.0f32.ln()).abs() < 1e-6);
        assert!((data[2] - 10.0f32.ln()).abs() < 1e-6);
        assert!((data[3] - 0.5f32.ln()).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6); // log(e) = 1.0
    }

    #[test]
    fn test_log_negative() {
        let input = vec![-1.0, 0.0, 1.0];
        let arr = Array::from_vec(input);
        let logged = log(&arr).unwrap();
        let data = logged.data();
        let v0: f32 = data[0];
        let v1: f32 = data[1];
        assert!(v0.is_infinite() && v0 < 0.0);
        assert!(v1.is_infinite() && v1 < 0.0);
        assert!((data[2] - 0.0).abs() < 1e-6); // log(1.0) = 0.0
    }
}
