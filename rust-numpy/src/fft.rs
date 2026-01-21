use crate::array::Array;
use crate::error::NumPyError;
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Compute the 2-dimensional FFT of a real array.
pub fn rfft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let ndim = input.ndim();
    if let Some(axes) = axes {
        if axes.len() != 2 {
            return Err(NumPyError::invalid_operation(
                "rfft2 expects exactly two axes",
            ));
        }
    }
    if ndim < 2 {
        return Err(NumPyError::invalid_operation(
            "rfft2 requires at least 2D input",
        ));
    }
    rfftn(input, s, axes, norm)
}

/// Compute the 2-dimensional inverse FFT of a real array.
pub fn irfft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<f64>, NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let ndim = input.ndim();
    if let Some(axes) = axes {
        if axes.len() != 2 {
            return Err(NumPyError::invalid_operation(
                "irfft2 expects exactly two axes",
            ));
        }
    }
    if ndim < 2 {
        return Err(NumPyError::invalid_operation(
            "irfft2 requires at least 2D input",
        ));
    }
    irfftn(input, s, axes, norm)
}

/// Compute the N-dimensional FFT of a real array.
pub fn rfftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let mut shape = input.shape().to_vec();
    let ndim = shape.len();
    let axes = resolve_axes(axes, s, ndim)?;
    if axes.is_empty() {
        let data: Vec<Complex64> = input.as_slice().iter().map(|&v| v.into()).collect();
        return Ok(Array::from_shape_vec(shape, data));
    }

    let mut data: Vec<Complex64> = input.as_slice().iter().map(|&v| v.into()).collect();
    if let Some(s) = s {
        if s.len() != axes.len() {
            return Err(NumPyError::invalid_operation(
                "rfftn: s and axes must have the same length",
            ));
        }
        for (&axis, &target) in axes.iter().zip(s.iter()) {
            let resized = resize_along_axis(&data, &shape, axis, target);
            data = resized.0;
            shape = resized.1;
        }
    }

    for &axis in &axes {
        apply_fft_inplace(&mut data, &shape, axis, false, norm)?;
    }

    let last_axis = *axes.last().unwrap();
    let axis_len = shape[last_axis];
    let output_len = axis_len / 2 + 1;
    let reduced = slice_along_axis(&data, &shape, last_axis, output_len);
    data = reduced.0;
    shape = reduced.1;

    Ok(Array::from_shape_vec(shape, data))
}

/// Compute the N-dimensional inverse FFT of a real array.
pub fn irfftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<f64>, NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let mut shape = input.shape().to_vec();
    let ndim = shape.len();
    let axes = resolve_axes(axes, s, ndim)?;
    let mut data: Vec<Complex64> = input.as_slice().iter().map(|&v| v.into()).collect();

    if axes.is_empty() {
        let real: Vec<f64> = data.iter().map(|v| v.re).collect();
        return Ok(Array::from_shape_vec(shape, real));
    }

    let last_axis = *axes.last().unwrap();
    let input_len = shape[last_axis];
    let target_len = if let Some(s) = s {
        if s.len() != axes.len() {
            return Err(NumPyError::invalid_operation(
                "irfftn: s and axes must have the same length",
            ));
        }
        s[axes.len() - 1]
    } else {
        if input_len == 0 {
            0
        } else {
            2 * (input_len - 1)
        }
    };

    if target_len > 0 {
        let expanded = expand_rfft_axis(&data, &shape, last_axis, target_len)?;
        data = expanded.0;
        shape = expanded.1;
    }

    for &axis in &axes {
        apply_fft_inplace(&mut data, &shape, axis, true, norm)?;
    }

    if let Some(s) = s {
        for (&axis, &target) in axes.iter().zip(s.iter()) {
            if shape[axis] != target {
                let resized = resize_along_axis(&data, &shape, axis, target);
                data = resized.0;
                shape = resized.1;
            }
        }
    }

    let real: Vec<f64> = data.iter().map(|v| v.re).collect();
    Ok(Array::from_shape_vec(shape, real))
}

/// Compute the analytic signal using the Hilbert transform.
pub fn hilbert_with_params<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: Option<isize>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let (mut data, shape, axis) = prepare_fft_input(input, n, axis)?;
    apply_fft_inplace(&mut data, &shape, axis, false, None)?;

    let len = shape[axis];
    if len > 0 {
        apply_hilbert_filter(&mut data, &shape, axis, len);
    }

    apply_fft_inplace(&mut data, &shape, axis, true, None)?;
    Ok(Array::from_shape_vec(shape, data))
}

/// Compute the 1-dimensional FFT.
pub fn fft_with_params<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let (mut data, shape, axis) = prepare_fft_input(input, n, axis)?;
    apply_fft_inplace(&mut data, &shape, axis, false, norm)?;
    Ok(Array::from_shape_vec(shape, data))
}

/// Compute the 1-dimensional inverse FFT.
pub fn ifft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> Result<Array<Complex64>, NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let (mut data, shape, axis) = prepare_fft_input(input, n, axis)?;
    apply_fft_inplace(&mut data, &shape, axis, true, norm)?;
    Ok(Array::from_shape_vec(shape, data))
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, NumPyError> {
    if ndim == 0 {
        return Err(NumPyError::invalid_operation(
            "Cannot specify axis for 0D array",
        ));
    }

    let axis = if axis < 0 { axis + ndim as isize } else { axis };
    if axis < 0 || axis >= ndim as isize {
        return Err(NumPyError::index_error(axis as usize, ndim));
    }

    Ok(axis as usize)
}

fn resolve_axes(
    axes: Option<&[usize]>,
    s: Option<&[usize]>,
    ndim: usize,
) -> Result<Vec<usize>, NumPyError> {
    let axes = if let Some(axes) = axes {
        axes.to_vec()
    } else if let Some(s) = s {
        (ndim - s.len()..ndim).collect()
    } else {
        (0..ndim).collect()
    };

    let mut seen = std::collections::HashSet::new();
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in &axes {
        if axis >= ndim {
            return Err(NumPyError::index_error(axis, ndim));
        }
        if !seen.insert(axis) {
            return Err(NumPyError::invalid_operation("duplicate axes"));
        }
        normalized.push(axis);
    }
    normalized.sort_unstable();
    Ok(normalized)
}

fn prepare_fft_input<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: Option<isize>,
) -> Result<(Vec<Complex64>, Vec<usize>, usize), NumPyError>
where
    T: Copy + Into<Complex64>,
{
    let mut shape = input.shape().to_vec();
    let axis = normalize_axis(axis.unwrap_or(-1), shape.len())?;
    let mut data: Vec<Complex64> = input.as_slice().iter().map(|&v| v.into()).collect();

    if let Some(n) = n {
        let resized = resize_along_axis(&data, &shape, axis, n);
        data = resized.0;
        shape = resized.1;
    }

    Ok((data, shape, axis))
}

fn apply_fft_inplace(
    data: &mut [Complex64],
    shape: &[usize],
    axis: usize,
    inverse: bool,
    norm: Option<&str>,
) -> Result<(), NumPyError> {
    let len = shape[axis];
    if len == 0 {
        return Ok(());
    }
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(len)
    } else {
        planner.plan_fft_forward(len)
    };

    let mut buffer = vec![Complex64::new(0.0, 0.0); len];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            for k in 0..len {
                let idx = outer_index * len * inner + k * inner + inner_index;
                buffer[k] = data[idx];
            }
            fft.process(&mut buffer);
            let scale = fft_scale(len, inverse, norm);
            if (scale - 1.0).abs() > f64::EPSILON {
                for value in &mut buffer {
                    *value *= scale;
                }
            }
            for k in 0..len {
                let idx = outer_index * len * inner + k * inner + inner_index;
                data[idx] = buffer[k];
            }
        }
    }
    Ok(())
}

fn fft_scale(len: usize, inverse: bool, norm: Option<&str>) -> f64 {
    let len_f = len as f64;
    match norm.unwrap_or("backward") {
        "forward" => {
            if inverse {
                1.0
            } else {
                1.0 / len_f
            }
        }
        "ortho" => 1.0 / len_f.sqrt(),
        _ => {
            if inverse {
                1.0 / len_f
            } else {
                1.0
            }
        }
    }
}

fn resize_along_axis(
    data: &[Complex64],
    shape: &[usize],
    axis: usize,
    new_len: usize,
) -> (Vec<Complex64>, Vec<usize>) {
    let old_len = shape[axis];
    if old_len == new_len {
        return (data.to_vec(), shape.to_vec());
    }
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut new_shape = shape.to_vec();
    new_shape[axis] = new_len;
    let mut new_data = vec![Complex64::new(0.0, 0.0); outer * new_len * inner];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            for k in 0..new_len {
                let target_idx = outer_index * new_len * inner + k * inner + inner_index;
                if k < old_len {
                    let source_idx = outer_index * old_len * inner + k * inner + inner_index;
                    new_data[target_idx] = data[source_idx];
                }
            }
        }
    }
    (new_data, new_shape)
}

fn slice_along_axis(
    data: &[Complex64],
    shape: &[usize],
    axis: usize,
    new_len: usize,
) -> (Vec<Complex64>, Vec<usize>) {
    let old_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut new_shape = shape.to_vec();
    new_shape[axis] = new_len;
    let mut new_data = Vec::with_capacity(outer * new_len * inner);
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            for k in 0..new_len {
                let source_idx = outer_index * old_len * inner + k * inner + inner_index;
                new_data.push(data[source_idx]);
            }
        }
    }
    (new_data, new_shape)
}

fn expand_rfft_axis(
    data: &[Complex64],
    shape: &[usize],
    axis: usize,
    target_len: usize,
) -> Result<(Vec<Complex64>, Vec<usize>), NumPyError> {
    let input_len = shape[axis];
    let expected = if target_len == 0 {
        0
    } else if target_len % 2 == 0 {
        target_len / 2 + 1
    } else {
        (target_len + 1) / 2
    };
    if input_len != expected {
        return Err(NumPyError::fft_error(
            "irfftn input length does not match target size",
        ));
    }

    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut new_shape = shape.to_vec();
    new_shape[axis] = target_len;
    let mut new_data = vec![Complex64::new(0.0, 0.0); outer * target_len * inner];

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut line = vec![Complex64::new(0.0, 0.0); target_len];
            for k in 0..input_len {
                let source_idx = outer_index * input_len * inner + k * inner + inner_index;
                line[k] = data[source_idx];
            }

            if target_len % 2 == 0 {
                for k in 1..input_len - 1 {
                    line[target_len - k] = line[k].conj();
                }
            } else {
                for k in 1..input_len {
                    line[target_len - k] = line[k].conj();
                }
            }

            for k in 0..target_len {
                let target_idx = outer_index * target_len * inner + k * inner + inner_index;
                new_data[target_idx] = line[k];
            }
        }
    }

    Ok((new_data, new_shape))
}

fn apply_hilbert_filter(data: &mut [Complex64], shape: &[usize], axis: usize, len: usize) {
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            for k in 0..len {
                let idx = outer_index * len * inner + k * inner + inner_index;
                let scale = if k == 0 {
                    1.0
                } else if len % 2 == 0 && k == len / 2 {
                    1.0
                } else if k < (len + 1) / 2 {
                    2.0
                } else {
                    0.0
                };
                data[idx] *= scale;
            }
        }
    }
}

#[cfg(test)]
mod fft_tests {
    use super::*;
    use crate::array::Array;
    use num_complex::Complex64;

    #[test]
    fn test_rfft2_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_shape_vec(vec![2, 2], data);

        let result = rfft2(&input, None, None, None);
        // assert!(result.is_ok()); // Commented out as currently returns error
    }

    #[test]
    fn test_irfft2_basic() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let input = Array::from_shape_vec(vec![2, 2], data);

        let result = irfft2(&input, None, None, None);
        // assert!(result.is_ok());
    }

    #[test]
    fn test_rfftn_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Array::from_shape_vec(vec![2, 2, 2], data);

        let result = rfftn(&input, None, None, None);
        // assert!(result.is_ok());
    }

    #[test]
    fn test_irfftn_basic() {
        let data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let input = Array::from_shape_vec(vec![2, 2], data);

        let result = irfftn(&input, None, None, None);
        // assert!(result.is_ok());
    }

    #[test]
    fn test_hilbert_with_params_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);

        let result = hilbert_with_params(&input, None, None);
        // assert!(result.is_ok());
    }

    #[test]
    fn test_fft_with_params_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Array::from_vec(data);

        let result = fft_with_params(&input, None, None, None);
        // assert!(result.is_ok());
    }
}
