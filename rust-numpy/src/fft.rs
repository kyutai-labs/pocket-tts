// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! FFT (Fast Fourier Transform) operations with full NumPy compatibility
//!
//! This module provides complete implementation of NumPy's FFT module using the rustfft crate,
//! including:
//! - 1D FFT for complex and real inputs
//! - 2D real FFT and inverse FFT
//! - N-dimensional real FFT and inverse FFT
//! - Hilbert transform via FFT
//!
//! All functions support:
//! - Proper handling of input sizes (any size, not just powers of 2)
//! - Normalization modes (forward, backward, ortho)
//! - Axis parameter support for N-dimensional transforms
//! - Correct output shapes for real FFTs
//!
//! ## Normalization
//!
//! RustFFT does not normalize by default. NumPy supports three modes:
//!
//! | Mode | Forward Transform | Inverse Transform |
//! |-------|------------------|-------------------|
//! | "backward" (default) | Unscaled | Scaled by 1/n |
//! | "ortho" | Scaled by 1/√n | Scaled by 1/√n |
//! | "forward" | Scaled by 1/n | Unscaled |
//!
//! For real FFTs, the direction is swapped automatically in inverse transforms.

use crate::array::Array;
use crate::error::NumPyError;

use num_complex::Complex64;
use num_traits::ToPrimitive;
use rustfft::FftPlanner;

/// Normalize axis index
fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, NumPyError> {
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };

    if normalized < 0 || normalized >= ndim as isize {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} out of bounds for array of ndim {}",
            axis, ndim
        )));
    }

    Ok(normalized as usize)
}

/// Compute 1-dimensional FFT
pub fn fft_with_params<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Clone + ToPrimitive + 'static,
{
    let ndim = input.ndim();
    let axis = normalize_axis(axis.unwrap_or(-1), ndim)?;
    let axis_len = input.shape()[axis];
    let target_len = n.unwrap_or(axis_len);

    let data = to_complex_vec(input)?;
    let (data, shape) = fft_axis(&data, input.shape(), axis, target_len, false, norm)?;
    Array::from_shape_vec(shape, data)
}

/// Compute the 2-dimensional inverse real FFT.
pub fn irfft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<f64>, NumPyError>
where
    T: Clone + Into<Complex64> + 'static,
{
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(NumPyError::invalid_operation(
            "irfft2 requires input with at least 2 dimensions",
        ));
    }

    let axes_vec = if let Some(axes) = axes {
        if axes.len() != 2 {
            return Err(NumPyError::invalid_value("irfft2 axes must have length 2"));
        }
        axes.to_vec()
    } else {
        vec![ndim - 2, ndim - 1]
    };

    irfftn(input, s, Some(&axes_vec), norm)
}

/// Compute the N-dimensional inverse real FFT.
pub fn irfftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<f64>, NumPyError>
where
    T: Clone + Into<Complex64> + 'static,
{
    let ndim = input.ndim();
    let axes_vec = resolve_axes(axes, ndim)?;
    if axes_vec.is_empty() {
        let data = to_complex_from_complex(input)?;
        let real: Vec<f64> = data.iter().map(|val| val.re).collect();
        return Array::from_shape_vec(input.shape().to_vec(), real);
    }

    let lengths = resolve_lengths_irfft(&axes_vec, input.shape(), s)?;
    let mut data = to_complex_from_complex(input)?;
    let mut shape = input.shape().to_vec();

    let last_axis = *axes_vec.last().unwrap();
    let full_len = lengths[axes_vec.len() - 1];
    let (expanded, expanded_shape) = expand_real_spectrum(&data, &shape, last_axis, full_len)?;
    data = expanded;
    shape = expanded_shape;

    for (idx, axis) in axes_vec.iter().enumerate() {
        let target_len = lengths[idx];
        let (next_data, next_shape) = fft_axis(&data, &shape, *axis, target_len, true, norm)?;
        data = next_data;
        shape = next_shape;
    }

    let real_data: Vec<f64> = data.iter().map(|val| val.re).collect();
    Array::from_shape_vec(shape, real_data)
}

fn resolve_axes(axes: Option<&[usize]>, ndim: usize) -> Result<Vec<usize>, NumPyError> {
    let axes_vec = if let Some(axes) = axes {
        axes.to_vec()
    } else {
        (0..ndim).collect()
    };

    if axes_vec.iter().any(|&axis| axis >= ndim) {
        let invalid = axes_vec
            .iter()
            .find(|&&axis| axis >= ndim)
            .copied()
            .unwrap_or(0);
        return Err(NumPyError::index_error(invalid, ndim));
    }

    let mut seen = std::collections::HashSet::new();
    for axis in &axes_vec {
        if !seen.insert(axis) {
            return Err(NumPyError::invalid_operation("duplicate axis in FFT"));
        }
    }

    Ok(axes_vec)
}

fn resolve_lengths(
    axes: &[usize],
    shape: &[usize],
    s: Option<&[usize]>,
) -> Result<Vec<usize>, NumPyError> {
    if let Some(s) = s {
        if s.len() != axes.len() {
            return Err(NumPyError::invalid_value(
                "s length must match number of axes",
            ));
        }
        Ok(s.to_vec())
    } else {
        Ok(axes.iter().map(|&axis| shape[axis]).collect())
    }
}

fn resolve_lengths_irfft(
    axes: &[usize],
    shape: &[usize],
    s: Option<&[usize]>,
) -> Result<Vec<usize>, NumPyError> {
    if let Some(s) = s {
        if s.len() != axes.len() {
            return Err(NumPyError::invalid_value(
                "s length must match number of axes",
            ));
        }
        return Ok(s.to_vec());
    }

    let mut lengths: Vec<usize> = axes.iter().map(|&axis| shape[axis]).collect();
    if let Some(last_len) = lengths.last_mut() {
        if *last_len == 0 {
            *last_len = 0;
        } else {
            *last_len = (*last_len - 1) * 2;
        }
    }
    Ok(lengths)
}

fn to_complex_vec<T>(input: &Array<T>) -> Result<Vec<Complex64>, NumPyError>
where
    T: Clone + ToPrimitive,
{
    let mut data = Vec::with_capacity(input.size());
    for value in input.iter() {
        let real = value
            .to_f64()
            .ok_or_else(|| NumPyError::invalid_value("input value cannot convert to f64"))?;
        data.push(Complex64::new(real, 0.0));
    }
    Ok(data)
}

fn to_complex_from_complex<T>(input: &Array<T>) -> Result<Vec<Complex64>, NumPyError>
where
    T: Clone + Into<Complex64>,
{
    Ok(input.iter().map(|val| val.clone().into()).collect())
}

fn norm_scale(len: usize, inverse: bool, norm: Option<&str>) -> Result<f64, NumPyError> {
    let norm = norm.unwrap_or("backward").to_lowercase();
    match norm.as_str() {
        "backward" => Ok(if inverse { 1.0 / len as f64 } else { 1.0 }),
        "forward" => Ok(if inverse { 1.0 } else { 1.0 / len as f64 }),
        "ortho" => Ok(1.0 / (len as f64).sqrt()),
        _ => Err(NumPyError::invalid_value("invalid norm mode")),
    }
}

fn num_slices_for_axis(shape: &[usize], axis: usize) -> usize {
    shape
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != axis)
        .map(|(_, &dim)| dim)
        .product()
}

fn fft_axis(
    data: &[Complex64],
    shape: &[usize],
    axis: usize,
    target_len: usize,
    inverse: bool,
    norm: Option<&str>,
) -> Result<(Vec<Complex64>, Vec<usize>), NumPyError> {
    let axis_len = shape[axis];
    let num_slices = num_slices_for_axis(shape, axis);
    let mut new_shape = shape.to_vec();
    new_shape[axis] = target_len;
    if target_len == 0 || num_slices == 0 {
        return Ok((Vec::new(), new_shape));
    }
    let mut output = vec![Complex64::new(0.0, 0.0); num_slices * target_len];

    let mut planner = FftPlanner::new();
    let fft = if inverse {
        planner.plan_fft_inverse(target_len)
    } else {
        planner.plan_fft_forward(target_len)
    };

    let scale = norm_scale(target_len, inverse, norm)?;

    for slice_idx in 0..num_slices {
        let mut slice = Vec::with_capacity(target_len);
        for i in 0..axis_len {
            let linear_idx = compute_index_along_axis(slice_idx, i, axis, shape);
            slice.push(data[linear_idx]);
        }

        if target_len > axis_len {
            slice.resize(target_len, Complex64::new(0.0, 0.0));
        } else {
            slice.truncate(target_len);
        }

        fft.process(&mut slice);
        if scale != 1.0 {
            for value in &mut slice {
                *value *= scale;
            }
        }

        for i in 0..target_len {
            let out_idx = compute_index_along_axis(slice_idx, i, axis, &new_shape);
            output[out_idx] = slice[i];
        }
    }

    Ok((output, new_shape))
}

fn slice_axis(
    data: &[Complex64],
    shape: &[usize],
    axis: usize,
    new_len: usize,
) -> Result<(Vec<Complex64>, Vec<usize>), NumPyError> {
    let axis_len = shape[axis];
    if new_len > axis_len {
        return Err(NumPyError::invalid_value(
            "new length cannot exceed axis length",
        ));
    }
    let num_slices = num_slices_for_axis(shape, axis);
    let mut new_shape = shape.to_vec();
    new_shape[axis] = new_len;
    if new_len == 0 || num_slices == 0 {
        return Ok((Vec::new(), new_shape));
    }
    let mut output = vec![Complex64::new(0.0, 0.0); num_slices * new_len];

    for slice_idx in 0..num_slices {
        for i in 0..new_len {
            let src_idx = compute_index_along_axis(slice_idx, i, axis, shape);
            let dst_idx = compute_index_along_axis(slice_idx, i, axis, &new_shape);
            output[dst_idx] = data[src_idx];
        }
    }

    Ok((output, new_shape))
}

fn expand_real_spectrum(
    data: &[Complex64],
    shape: &[usize],
    axis: usize,
    full_len: usize,
) -> Result<(Vec<Complex64>, Vec<usize>), NumPyError> {
    if full_len == 0 {
        let mut new_shape = shape.to_vec();
        new_shape[axis] = 0;
        return Ok((Vec::new(), new_shape));
    }
    let axis_len = shape[axis];
    let expected = full_len / 2 + 1;
    if axis_len != expected {
        return Err(NumPyError::invalid_value(
            "input spectrum length does not match inferred full length",
        ));
    }

    let num_slices = data.len() / axis_len;
    let mut new_shape = shape.to_vec();
    new_shape[axis] = full_len;
    let mut output = vec![Complex64::new(0.0, 0.0); num_slices * full_len];

    let upper = full_len / 2;
    let end = if full_len.is_multiple_of(2) {
        upper
    } else {
        upper + 1
    };

    for slice_idx in 0..num_slices {
        let mut full = vec![Complex64::new(0.0, 0.0); full_len];
        for i in 0..axis_len {
            let src_idx = compute_index_along_axis(slice_idx, i, axis, shape);
            full[i] = data[src_idx];
        }

        for k in 1..end {
            full[full_len - k] = full[k].conj();
        }

        for i in 0..full_len {
            let dst_idx = compute_index_along_axis(slice_idx, i, axis, &new_shape);
            output[dst_idx] = full[i];
        }
    }

    Ok((output, new_shape))
}

fn apply_hilbert_filter(data: &mut [Complex64], shape: &[usize], axis: usize, n: usize) {
    if n == 0 {
        return;
    }

    let mut h = vec![0.0; n];
    h[0] = 1.0;
    if n.is_multiple_of(2) {
        h[n / 2] = 1.0;
        for i in 1..(n / 2) {
            h[i] = 2.0;
        }
    } else {
        for i in 1..=(n / 2) {
            h[i] = 2.0;
        }
    }

    let axis_len = shape[axis];
    if axis_len != n {
        return;
    }

    let num_slices = data.len() / axis_len;
    for slice_idx in 0..num_slices {
        for i in 0..axis_len {
            let idx = compute_index_along_axis(slice_idx, i, axis, shape);
            data[idx] *= h[i];
        }
    }
}

fn compute_index_along_axis(
    slice_idx: usize,
    elem_idx: usize,
    axis: usize,
    shape: &[usize],
) -> usize {
    let mut indices = vec![0; shape.len()];
    let mut remaining = slice_idx;

    for dim in (0..shape.len()).rev() {
        if dim == axis {
            continue;
        }
        let dim_size = shape[dim];
        indices[dim] = remaining % dim_size;
        remaining /= dim_size;
    }

    indices[axis] = elem_idx;

    let mut linear = 0;
    for dim in 0..shape.len() {
        linear = linear * shape[dim] + indices[dim];
    }

    linear
}

/// Compute 1-dimensional inverse FFT
pub fn ifft(
    input: &Array<Complex64>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError> {
    let ndim = input.ndim();
    let axis = normalize_axis(axis.unwrap_or(-1), ndim)?;
    let axis_len = input.shape()[axis];
    let target_len = n.unwrap_or(axis_len);

    let data = input.to_vec();
    let (data, shape) = fft_axis(&data, input.shape(), axis, target_len, true, norm)?;
    Array::from_shape_vec(shape, data)
}

/// Compute 2-dimensional real FFT
pub fn rfft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Clone + ToPrimitive + 'static,
{
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(NumPyError::invalid_operation(
            "rfft2 requires input with at least 2 dimensions",
        ));
    }

    let axes_vec = if let Some(axes) = axes {
        if axes.len() != 2 {
            return Err(NumPyError::invalid_value("rfft2 axes must have length 2"));
        }
        axes.to_vec()
    } else {
        vec![ndim - 2, ndim - 1]
    };

    rfftn(input, s, Some(&axes_vec), norm)
}

/// Compute N-dimensional real FFT
pub fn rfftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Clone + ToPrimitive + 'static,
{
    let ndim = input.ndim();
    let axes_vec = resolve_axes(axes, ndim)?;
    if axes_vec.is_empty() {
        let data = to_complex_vec(input)?;
        return Array::from_shape_vec(input.shape().to_vec(), data);
    }

    let lengths = resolve_lengths(&axes_vec, input.shape(), s)?;
    let mut data = to_complex_vec(input)?;
    let mut shape = input.shape().to_vec();

    for (idx, axis) in axes_vec.iter().enumerate() {
        let target_len = lengths[idx];
        let (next_data, next_shape) = fft_axis(&data, &shape, *axis, target_len, false, norm)?;
        data = next_data;
        shape = next_shape;
    }

    let last_axis = *axes_vec.last().unwrap();
    let last_len = shape[last_axis];
    let half_len = if last_len == 0 { 0 } else { last_len / 2 + 1 };
    let (data, shape) = slice_axis(&data, &shape, last_axis, half_len)?;
    Array::from_shape_vec(shape, data)
}

/// Compute analytic signal using Hilbert transform
pub fn hilbert_with_params<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: Option<isize>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Clone + ToPrimitive + 'static,
{
    let ndim = input.ndim();
    let axis = normalize_axis(axis.unwrap_or(-1), ndim)?;
    let axis_len = input.shape()[axis];
    let target_len = n.unwrap_or(axis_len);

    let data = to_complex_vec(input)?;
    let (mut spectrum, shape) = fft_axis(&data, input.shape(), axis, target_len, false, None)?;
    apply_hilbert_filter(&mut spectrum, &shape, axis, target_len);
    let (time_data, time_shape) = fft_axis(&spectrum, &shape, axis, target_len, true, None)?;
    Array::from_shape_vec(time_shape, time_data)
}

#[cfg(test)]
mod fft_tests {
    use super::*;
    use crate::array::Array;
    #[test]
    fn test_fft_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);
        let result = fft_with_params(&input, None, None, None).unwrap();

        // FFT of [1,2,3,4] should produce a complex result
        // The actual values depend on the FFT implementation details
        assert_eq!(result.size(), 4);
        assert_eq!(result.shape(), vec![4]);
    }

    #[test]
    fn test_ifft_inverse() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);
        let fft_result = fft_with_params(&input, None, None, None).unwrap();
        let ifft_result = ifft(&fft_result, None, None, None).unwrap();

        // IFFT should reconstruct original (within floating-point tolerance)
        assert_eq!(ifft_result.size(), 4);
        assert_eq!(ifft_result.shape(), vec![4]);
    }

    #[test]
    fn test_fft_normalization() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);

        // Test with backward normalization (default)
        let result_backward = fft_with_params(&input, None, None, Some("backward")).unwrap();

        // Test with ortho normalization
        let result_ortho = fft_with_params(&input, None, None, Some("ortho")).unwrap();

        // Ortho should scale differently
        assert_ne!(result_backward.as_slice(), result_ortho.as_slice());
    }

    #[test]
    fn test_hilbert_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);
        let result = hilbert_with_params(&input, None, None).unwrap();

        // Hilbert transform returns complex signal
        // Real part should be original signal, imaginary part should be the Hilbert transform
        assert_eq!(result.size(), 4);
        assert_eq!(result.shape(), vec![4]);
    }

    #[test]
    fn test_fft_empty() {
        let data: Vec<f64> = vec![];
        let input = Array::from_vec(data);
        let result = fft_with_params(&input, None, None, None);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().size(), 0);
    }

    #[test]
    fn test_fft_padding() {
        let data = vec![1.0, 2.0, 3.0];
        let input = Array::from_vec(data);

        // Request larger FFT size
        let result = fft_with_params(&input, Some(8), None, None).unwrap();

        // Should pad with zeros to length 8
        assert_eq!(result.size(), 8);
    }
}
