use crate::array::Array;
use crate::error::{NumPyError, Result};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Compute the 1-dimensional discrete Fourier Transform.
pub fn fft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let axis = normalize_axis(axis, input.ndim())?;
    let n = n.unwrap_or(input.shape()[axis]);

    let complex_input = input.clone_to_complex();
    fft_axis(&complex_input, n, axis, norm)
}

/// Compute the 1-dimensional inverse discrete Fourier Transform.
pub fn ifft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let axis = normalize_axis(axis, input.ndim())?;
    let n = n.unwrap_or(input.shape()[axis]);

    let complex_input = input.clone_to_complex();
    ifft_axis(&complex_input, n, axis, norm)
}

/// Shift the zero-frequency component to the center of the spectrum.
pub fn fftshift<T: Clone + Default + 'static>(x: &Array<T>, axes: Option<&[usize]>) -> Array<T> {
    let ndim = x.ndim();
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..ndim).collect(),
    };

    let mut result = x.clone();
    let shape = x.shape();

    for &axis in &axes {
        let n = shape[axis];
        let shift = n / 2;
        result = shift_axis(&result, axis, shift);
    }
    result
}

/// Inverse of fftshift.
pub fn ifftshift<T: Clone + Default + 'static>(x: &Array<T>, axes: Option<&[usize]>) -> Array<T> {
    let ndim = x.ndim();
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..ndim).collect(),
    };

    let mut result = x.clone();
    let shape = x.shape();

    for &axis in &axes {
        let n = shape[axis];
        let shift = (n + 1) / 2;
        result = shift_axis(&result, axis, shift);
    }
    result
}

fn shift_axis<T: Clone + Default + 'static>(
    input: &Array<T>,
    axis: usize,
    shift: usize,
) -> Array<T> {
    let shape = input.shape();
    let mut result = Array::zeros(shape.to_vec());
    let n = shape[axis];

    let other_axes: Vec<usize> = (0..input.ndim()).filter(|&ax| ax != axis).collect();
    let outer_size: usize = other_axes.iter().map(|&ax| shape[ax]).product();

    for i in 0..outer_size {
        let mut base_indices = vec![0; input.ndim()];
        let mut temp_idx = i;
        for &ax in other_axes.iter().rev() {
            base_indices[ax] = temp_idx % shape[ax];
            temp_idx /= shape[ax];
        }

        for k in 0..n {
            base_indices[axis] = k;
            let val = input.get_multi(&base_indices).unwrap();
            let new_k = (k + shift) % n;
            base_indices[axis] = new_k;
            result.set_multi(&base_indices, val).unwrap();
        }
    }
    result
}

/// Return the Discrete Fourier Transform sample frequencies.
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    let mut results = vec![0.0; n];
    let n_isize = n as isize;
    let limit = (n_isize - 1) / 2 + 1;

    for i in 0..limit {
        results[i as usize] = i as f64 * val;
    }

    let neg_limit = -(n_isize / 2);
    for i in neg_limit..0 {
        results[(n_isize + i) as usize] = i as f64 * val;
    }

    results
}

/// Return the Real Discrete Fourier Transform sample frequencies.
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    let n_out = n / 2 + 1;
    let mut results = vec![0.0; n_out];

    for i in 0..n_out {
        results[i] = i as f64 * val;
    }

    results
}

fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    if axis < 0 {
        let ax = axis + ndim as isize;
        if ax < 0 {
            return Err(NumPyError::invalid_operation(format!(
                "Axis {} out of bounds for ndim {}",
                axis, ndim
            )));
        }
        Ok(ax as usize)
    } else {
        if axis as usize >= ndim {
            return Err(NumPyError::invalid_operation(format!(
                "Axis {} out of bounds for ndim {}",
                axis, ndim
            )));
        }
        Ok(axis as usize)
    }
}

/// Compute the 1-dimensional discrete Fourier Transform for real input.
pub fn rfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<f64> + Default + 'static,
{
    let axis = normalize_axis(axis, input.ndim())?;
    let n = n.unwrap_or(input.shape()[axis]);

    let complex_input = input.clone_to_complex_real();
    rfft_axis(&complex_input, n, axis, norm)
}

/// Compute the inverse of rfft.
pub fn irfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<&str>,
) -> Result<Array<f64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let axis = normalize_axis(axis, input.ndim())?;
    // For irfft, n defaults to 2*(m-1) where m is length of input along axis
    let m = input.shape()[axis];
    let n = n.unwrap_or(2 * (m - 1));

    let complex_input = input.clone_to_complex();
    irfft_axis(&complex_input, n, axis, norm)
}

fn irfft_axis(
    input: &Array<Complex64>,
    n: usize,
    axis: usize,
    norm: Option<&str>,
) -> Result<Array<f64>> {
    let complex_res = irfft_axis_complex(input, n, axis, norm)?;
    let data: Vec<f64> = complex_res.iter().map(|x| x.re).collect();
    Ok(Array::from_data(data, complex_res.shape().to_vec()))
}

/// Compute the N-dimensional discrete Fourier Transform.
pub fn fftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let ndim = input.ndim();
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..ndim).collect(),
    };
    let s = match s {
        Some(sv) => sv.to_vec(),
        None => axes.iter().map(|&ax| input.shape()[ax]).collect(),
    };

    if axes.len() != s.len() {
        return Err(NumPyError::invalid_value(
            "s and axes must have the same length",
        ));
    }

    let mut current = input.clone_to_complex();

    for (i, &axis) in axes.iter().enumerate() {
        let n = s[i];
        current = fft_axis(&current, n, axis, norm)?;
    }

    Ok(current)
}

/// Compute the N-dimensional inverse discrete Fourier Transform.
pub fn ifftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let ndim = input.ndim();
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..ndim).collect(),
    };
    let s = match s {
        Some(sv) => sv.to_vec(),
        None => axes.iter().map(|&ax| input.shape()[ax]).collect(),
    };

    if axes.len() != s.len() {
        return Err(NumPyError::invalid_value(
            "s and axes must have the same length",
        ));
    }

    let mut current = input.clone_to_complex();

    for (i, &axis) in axes.iter().enumerate() {
        let n = s[i];
        current = ifft_axis(&current, n, axis, norm)?;
    }

    Ok(current)
}

fn fft_axis(
    input: &Array<Complex64>,
    n: usize,
    axis: usize,
    norm: Option<&str>,
) -> Result<Array<Complex64>> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "Axis {} out of bounds for ndim {}",
            axis, ndim
        )));
    }
    let shape = input.shape();

    let mut new_shape = shape.to_vec();
    new_shape[axis] = n;

    let mut result = Array::zeros(new_shape.clone());

    // We need to iterate over all dimensions EXCEPT the transform axis
    let mut other_axes = Vec::new();
    for i in 0..ndim {
        if i != axis {
            other_axes.push(i);
        }
    }

    // This is a naive implementation for now: extracting slices, transforming, and inserting.
    // Optimization with direct strided access would be better.
    let outer_size: usize = other_axes.iter().map(|&i| shape[i]).product();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    for i in 0..outer_size {
        // Map linear index to multi-index for other axes
        let mut indices = vec![0; ndim];
        let mut temp_idx = i;
        for &ax in other_axes.iter().rev() {
            indices[ax] = temp_idx % shape[ax];
            temp_idx /= shape[ax];
        }

        let mut line = Vec::with_capacity(shape[axis]);
        for k in 0..shape[axis] {
            indices[axis] = k;
            line.push(input.get_multi(&indices)?);
        }

        // Pad or truncate
        if line.len() < n {
            line.resize(n, Complex64::new(0.0, 0.0));
        } else if line.len() > n {
            line.truncate(n);
        }

        fft.process(&mut line);

        // Normalization
        if let Some(norm_str) = norm {
            match norm_str {
                "ortho" => {
                    let scale = (n as f64).sqrt();
                    for x in line.iter_mut() {
                        *x /= scale;
                    }
                }
                "forward" => {
                    let scale = n as f64;
                    for x in line.iter_mut() {
                        *x /= scale;
                    }
                }
                _ => {}
            }
        }

        for k in 0..n {
            indices[axis] = k;
            result.set_multi(&indices, line[k])?;
        }
    }

    Ok(result)
}

fn ifft_axis(
    input: &Array<Complex64>,
    n: usize,
    axis: usize,
    norm: Option<&str>,
) -> Result<Array<Complex64>> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "Axis {} out of bounds for ndim {}",
            axis, ndim
        )));
    }
    let shape = input.shape();

    let mut new_shape = shape.to_vec();
    new_shape[axis] = n;

    let mut result = Array::zeros(new_shape.clone());

    let mut other_axes = Vec::new();
    for i in 0..ndim {
        if i != axis {
            other_axes.push(i);
        }
    }

    let outer_size: usize = other_axes.iter().map(|&i| shape[i]).product();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    for i in 0..outer_size {
        let mut indices = vec![0; ndim];
        let mut temp_idx = i;
        for &ax in other_axes.iter().rev() {
            indices[ax] = temp_idx % shape[ax];
            temp_idx /= shape[ax];
        }

        let mut line = Vec::with_capacity(shape[axis]);
        for k in 0..shape[axis] {
            indices[axis] = k;
            line.push(input.get_multi(&indices)?);
        }

        if line.len() < n {
            line.resize(n, Complex64::new(0.0, 0.0));
        } else if line.len() > n {
            line.truncate(n);
        }

        fft.process(&mut line);

        let mut scale = n as f64;
        if let Some(norm_str) = norm {
            match norm_str {
                "ortho" => scale = (n as f64).sqrt(),
                "forward" => scale = 1.0,
                _ => {}
            }
        }

        for k in 0..n {
            indices[axis] = k;
            result.set_multi(&indices, line[k] / scale)?;
        }
    }

    Ok(result)
}

pub fn rfftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<f64> + Default + 'static,
{
    let ndim = input.ndim();
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..ndim).collect(),
    };
    let s = match s {
        Some(sv) => sv.to_vec(),
        None => axes.iter().map(|&ax| input.shape()[ax]).collect(),
    };

    let mut current = input.clone_to_complex_real();

    // Apply standard FFT on all but the last axis in 'axes'
    for i in 0..(axes.len() - 1) {
        current = fft_axis(&current, s[i], axes[i], norm)?;
    }

    // Apply rfft-like transform on the last axis
    let last_ax_idx = axes.len() - 1;
    rfft_axis(&current, s[last_ax_idx], axes[last_ax_idx], norm)
}

pub fn irfftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<f64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let ndim = input.ndim();
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => (0..ndim).collect(),
    };

    // For irfft, last axis 'n' defaults to 2*(m-1)
    let m_last = input.shape()[*axes.last().unwrap()];
    let s = match s {
        Some(sv) => sv.to_vec(),
        None => {
            let mut sv = axes.iter().map(|&ax| input.shape()[ax]).collect::<Vec<_>>();
            sv[axes.len() - 1] = 2 * (m_last - 1);
            sv
        }
    };

    let current = input.clone_to_complex();

    // Apply irfft-like on the last axis
    let last_ax_idx = axes.len() - 1;
    let mut current_complex =
        irfft_axis_complex(&current, s[last_ax_idx], axes[last_ax_idx], norm)?;

    // Apply ifft on remaining axes
    for i in (0..(axes.len() - 1)).rev() {
        current_complex = ifft_axis(&current_complex, s[i], axes[i], norm)?;
    }

    // Convert to real
    let data: Vec<f64> = current_complex.iter().map(|x| x.re).collect();
    Ok(Array::from_data(data, current_complex.shape().to_vec()))
}

fn rfft_axis(
    input: &Array<Complex64>,
    n: usize,
    axis: usize,
    norm: Option<&str>,
) -> Result<Array<Complex64>> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "Axis {} out of bounds for ndim {}",
            axis, ndim
        )));
    }
    let shape = input.shape();

    let n_out = n / 2 + 1;
    let mut new_shape = shape.to_vec();
    new_shape[axis] = n_out;

    let mut result = Array::zeros(new_shape.clone());
    let other_axes = (0..ndim).filter(|&i| i != axis).collect::<Vec<_>>();
    let outer_size: usize = other_axes.iter().map(|&i| shape[i]).product();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    for i in 0..outer_size {
        let mut indices = vec![0; ndim];
        let mut temp_idx = i;
        for &ax in other_axes.iter().rev() {
            indices[ax] = temp_idx % shape[ax];
            temp_idx /= shape[ax];
        }

        let mut line = Vec::with_capacity(shape[axis]);
        for k in 0..shape[axis] {
            indices[axis] = k;
            line.push(input.get_multi(&indices)?);
        }

        if line.len() < n {
            line.resize(n, Complex64::new(0.0, 0.0));
        } else if line.len() > n {
            line.truncate(n);
        }

        fft.process(&mut line);

        line.truncate(n_out);

        if let Some(norm_str) = norm {
            let scale = match norm_str {
                "ortho" => Some((n as f64).sqrt()),
                "forward" => Some(n as f64),
                _ => None,
            };
            if let Some(s) = scale {
                for x in line.iter_mut() {
                    *x /= s;
                }
            }
        }

        for k in 0..n_out {
            indices[axis] = k;
            result.set_multi(&indices, line[k])?;
        }
    }
    Ok(result)
}

fn irfft_axis_complex(
    input: &Array<Complex64>,
    n: usize,
    axis: usize,
    norm: Option<&str>,
) -> Result<Array<Complex64>> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "Axis {} out of bounds for ndim {}",
            axis, ndim
        )));
    }
    let shape = input.shape();
    let m = shape[axis];

    let mut new_shape = shape.to_vec();
    new_shape[axis] = n;

    let mut result = Array::zeros(new_shape.clone());
    let other_axes = (0..ndim).filter(|&i| i != axis).collect::<Vec<_>>();
    let outer_size: usize = other_axes.iter().map(|&i| shape[i]).product();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    for i in 0..outer_size {
        let mut indices = vec![0; ndim];
        let mut temp_idx = i;
        for &ax in other_axes.iter().rev() {
            indices[ax] = temp_idx % shape[ax];
            temp_idx /= shape[ax];
        }

        let mut data = Vec::with_capacity(m);
        for k in 0..m {
            indices[axis] = k;
            data.push(input.get_multi(&indices)?);
        }

        let mut full_data = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..m.min(n / 2 + 1) {
            full_data[k] = data[k];
        }
        for k in 1..((n + 1) / 2) {
            if k < m {
                let target = n - k;
                if target < n {
                    full_data[target] = data[k].conj();
                }
            }
        }

        fft.process(&mut full_data);

        let mut scale = n as f64;
        if let Some(norm_str) = norm {
            match norm_str {
                "ortho" => scale = (n as f64).sqrt(),
                "forward" => scale = 1.0,
                _ => {}
            }
        }

        for k in 0..n {
            indices[axis] = k;
            result.set_multi(&indices, full_data[k] / scale)?;
        }
    }
    Ok(result)
}

impl<T> Array<T>
where
    T: Clone + Into<f64> + Default + 'static,
{
    fn clone_to_complex_real(&self) -> Array<Complex64> {
        let data: Vec<Complex64> = self
            .iter()
            .map(|x| Complex64::new(x.clone().into(), 0.0))
            .collect();
        Array::from_data(data, self.shape.to_vec())
    }
}

/// Compute the 2-dimensional discrete Fourier Transform.
///
/// This computes the n-dimensional discrete Fourier Transform over the specified axes.
/// If no axes are specified, the transform is computed over the last two axes.
pub fn fft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => vec![input.ndim() - 2, input.ndim() - 1],
    };
    fftn(input, s, Some(&axes), norm)
}

/// Compute the 2-dimensional inverse discrete Fourier Transform.
///
/// This computes the n-dimensional inverse discrete Fourier Transform over the specified axes.
/// If no axes are specified, the transform is computed over the last two axes.
pub fn ifft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => vec![input.ndim() - 2, input.ndim() - 1],
    };
    ifftn(input, s, Some(&axes), norm)
}

pub fn rfft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<f64> + Default + 'static,
{
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => vec![input.ndim() - 2, input.ndim() - 1],
    };
    rfftn(input, s, Some(&axes), norm)
}

pub fn irfft2<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[usize]>,
    norm: Option<&str>,
) -> Result<Array<f64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let axes = match axes {
        Some(a) => a.to_vec(),
        None => vec![input.ndim() - 2, input.ndim() - 1],
    };
    irfftn(input, s, Some(&axes), norm)
}

/// Compute the FFT of a signal that has Hermitian symmetry.
///
/// `hfft` represents a 1-D discrete Fourier Transform of a Hermitian symmetric sequence.
/// The input should be Hermitian symmetric (i.e., the negative frequency terms are
/// the complex conjugates of the positive frequency terms). The output is real-valued.
///
/// # Arguments
/// * `input` - Input array with Hermitian symmetry
/// * `n` - Length of the output transform. If None, defaults to 2*(m-1) where m is input length
/// * `axis` - Axis over which to compute the FFT
/// * `norm` - Normalization mode
///
/// # Returns
/// Real-valued FFT of the Hermitian symmetric input
pub fn hfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<&str>,
) -> Result<Array<f64>>
where
    T: Clone + Into<Complex64> + Default + 'static,
{
    let axis = normalize_axis(axis, input.ndim())?;
    // For hfft, n defaults to 2*(m-1) where m is length of input along axis
    let m = input.shape()[axis];
    let n = n.unwrap_or(2 * (m - 1));

    let complex_input = input.clone_to_complex();
    hfft_axis(&complex_input, n, axis, norm)
}

fn hfft_axis(
    input: &Array<Complex64>,
    n: usize,
    axis: usize,
    norm: Option<&str>,
) -> Result<Array<f64>> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "Axis {} out of bounds for ndim {}",
            axis, ndim
        )));
    }
    let shape = input.shape();
    let m = shape[axis];

    let mut new_shape = shape.to_vec();
    new_shape[axis] = n;

    let mut result = Array::zeros(new_shape.clone());
    let other_axes: Vec<usize> = (0..ndim).filter(|&ax| ax != axis).collect();
    let outer_size: usize = other_axes.iter().map(|&ax| shape[ax]).product();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    for i in 0..outer_size {
        let mut indices = vec![0; ndim];
        let mut temp_idx = i;
        for &ax in other_axes.iter().rev() {
            indices[ax] = temp_idx % shape[ax];
            temp_idx /= shape[ax];
        }

        let mut data = Vec::with_capacity(m);
        for k in 0..m {
            indices[axis] = k;
            data.push(input.get_multi(&indices)?);
        }

        let mut full_data = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..m.min(n / 2 + 1) {
            full_data[k] = data[k];
        }
        for k in 1..((n + 1) / 2) {
            if k < m {
                let target = n - k;
                if target < n {
                    full_data[target] = data[k].conj();
                }
            }
        }

        fft.process(&mut full_data);

        let mut scale = 1.0;
        if let Some(norm_str) = norm {
            match norm_str {
                "ortho" => scale = (n as f64).sqrt(),
                "forward" => scale = n as f64,
                _ => {}
            }
        }

        for k in 0..n {
            indices[axis] = k;
            result.set_multi(&indices, full_data[k].re * scale)?;
        }
    }
    Ok(result)
}

/// Compute the inverse FFT of a real signal.
///
/// `ihfft` represents the inverse of `hfft`. It computes the FFT of a real-valued
/// signal and returns the Hermitian symmetric frequency components.
///
/// # Arguments
/// * `input` - Real-valued input array
/// * `n` - Length of the output. If None, defaults to m/2+1 where m is input length
/// * `axis` - Axis over which to compute the FFT
/// * `norm` - Normalization mode
///
/// # Returns
/// Hermitian symmetric FFT coefficients
pub fn ihfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<&str>,
) -> Result<Array<Complex64>>
where
    T: Clone + Into<f64> + Default + 'static,
{
    let axis = normalize_axis(axis, input.ndim())?;
    let m = input.shape()[axis];
    // For ihfft, n defaults to m/2 + 1 (the rfft output length)
    let n_out = n.unwrap_or(m / 2 + 1);

    let complex_input = input.clone_to_complex_real();
    ihfft_axis(&complex_input, n_out, axis, norm)
}

fn ihfft_axis(
    input: &Array<Complex64>,
    n_out: usize,
    axis: usize,
    norm: Option<&str>,
) -> Result<Array<Complex64>> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "Axis {} out of bounds for ndim {}",
            axis, ndim
        )));
    }
    let shape = input.shape();
    let m = shape[axis];

    let mut new_shape = shape.to_vec();
    new_shape[axis] = n_out;

    let mut result = Array::zeros(new_shape.clone());
    let other_axes: Vec<usize> = (0..ndim).filter(|&ax| ax != axis).collect();
    let outer_size: usize = other_axes.iter().map(|&ax| shape[ax]).product();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(m);

    for i in 0..outer_size {
        let mut indices = vec![0; ndim];
        let mut temp_idx = i;
        for &ax in other_axes.iter().rev() {
            indices[ax] = temp_idx % shape[ax];
            temp_idx /= shape[ax];
        }

        let mut data = Vec::with_capacity(m);
        for k in 0..m {
            indices[axis] = k;
            data.push(input.get_multi(&indices)?);
        }

        fft.process(&mut data);

        let mut scale = m as f64;
        if let Some(norm_str) = norm {
            match norm_str {
                "ortho" => scale = (m as f64).sqrt(),
                "forward" => scale = m as f64,
                _ => {}
            }
        }

        for k in 0..n_out {
            indices[axis] = k;
            result.set_multi(&indices, data[k] / scale)?;
        }
    }
    Ok(result)
}
