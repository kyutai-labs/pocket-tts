use crate::array::Array;
use crate::broadcasting::broadcast_shape_for_reduce;
use crate::error::NumPyError;
use crate::strides::compute_multi_indices;
use std::cmp::Ordering;
use std::f64;

pub fn median<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    _overwrite_input: bool,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    let q = Array::from_vec(vec![T::from_f64(50.0)]);
    percentile_internal(a, &q, axis, keepdims, "linear", 100.0, false)
}

pub fn nanmedian<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    _overwrite_input: bool,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    let q = Array::from_vec(vec![T::from_f64(50.0)]);
    percentile_internal(a, &q, axis, keepdims, "linear", 100.0, true)
}

pub fn percentile<T>(
    a: &Array<T>,
    q: &Array<T>,
    axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    _overwrite_input: bool,
    _method: &str,
    keepdims: bool,
    interpolation: &str,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    percentile_internal(a, q, axis, keepdims, interpolation, 100.0, false)
}

pub fn nanpercentile<T>(
    a: &Array<T>,
    q: &Array<T>,
    axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    _overwrite_input: bool,
    _method: &str,
    keepdims: bool,
    interpolation: &str,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    percentile_internal(a, q, axis, keepdims, interpolation, 100.0, true)
}

pub fn quantile<T>(
    a: &Array<T>,
    q: &Array<T>,
    axis: Option<&[isize]>,
    out: Option<&mut Array<T>>,
    overwrite_input: bool,
    method: &str,
    keepdims: bool,
    interpolation: &str,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    let _ = (out, overwrite_input, method);
    percentile_internal(a, q, axis, keepdims, interpolation, 1.0, false)
}

pub fn nanquantile<T>(
    a: &Array<T>,
    q: &Array<T>,
    axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    _overwrite_input: bool,
    _method: &str,
    keepdims: bool,
    interpolation: &str,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    percentile_internal(a, q, axis, keepdims, interpolation, 1.0, true)
}

pub fn average<T>(
    a: &Array<T>,
    _axis: Option<&[isize]>,
    weights: Option<&Array<T>>,
    _returned: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute average of empty array",
        ));
    }

    let data = a.to_vec();

    if let Some(w) = weights {
        if w.size() != a.size() {
            return Err(NumPyError::invalid_value(
                "Weights must have same size as array",
            ));
        }

        let w_data = w.to_vec();
        let sum_weights: f64 = w_data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum();

        if sum_weights == 0.0 {
            return Err(NumPyError::invalid_value("Sum of weights must not be zero"));
        }

        let weighted_sum: f64 = data
            .iter()
            .zip(w_data.iter())
            .map(|(x, w)| x.as_f64().unwrap_or(0.0) * w.as_f64().unwrap_or(0.0))
            .sum();

        Ok(Array::from_vec(vec![T::from_f64(
            weighted_sum / sum_weights,
        )]))
    } else {
        let sum: f64 = data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum();
        Ok(Array::from_vec(vec![T::from_f64(sum / data.len() as f64)]))
    }
}

pub fn std<T>(
    a: &Array<T>,
    _axis: Option<&[isize]>,
    _dtype: Option<crate::dtype::Dtype>,
    _out: Option<&mut Array<T>>,
    ddof: isize,
    _keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute std of empty array",
        ));
    }

    let data = a.to_vec();
    let mean: f64 = data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum::<f64>() / data.len() as f64;
    let variance: f64 = data
        .iter()
        .map(|x| {
            let diff = x.as_f64().unwrap_or(0.0) - mean;
            diff * diff
        })
        .sum::<f64>()
        / ((data.len() as isize - ddof).max(1) as f64);

    Ok(Array::from_vec(vec![T::from_f64(variance.sqrt())]))
}

pub fn var<T>(
    a: &Array<T>,
    _axis: Option<&[isize]>,
    _dtype: Option<crate::dtype::Dtype>,
    _out: Option<&mut Array<T>>,
    ddof: isize,
    _keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute var of empty array",
        ));
    }

    let data = a.to_vec();
    let mean: f64 = data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum::<f64>() / data.len() as f64;
    let variance: f64 = data
        .iter()
        .map(|x| {
            let diff = x.as_f64().unwrap_or(0.0) - mean;
            diff * diff
        })
        .sum::<f64>()
        / (data.len() as isize - ddof).max(1) as f64;

    Ok(Array::from_vec(vec![T::from_f64(variance)]))
}

pub fn corrcoef<T>(
    x: &Array<T>,
    y: Option<&Array<T>>,
    _rowvar: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if x.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute corrcoef of empty array",
        ));
    }

    let x_data = x.to_vec();

    if let Some(y_arr) = y {
        let y_data = y_arr.to_vec();

        if x_data.len() != y_data.len() {
            return Err(NumPyError::invalid_value("x and y must have same length"));
        }

        let n = x_data.len() as f64;
        let x_mean: f64 = x_data
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .sum::<f64>()
            / n;
        let y_mean: f64 = y_data
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .sum::<f64>()
            / n;

        let covariance: f64 = x_data
            .iter()
            .zip(y_data.iter())
            .map(|(xi, yi)| {
                (xi.as_f64().unwrap_or(0.0) - x_mean) * (yi.as_f64().unwrap_or(0.0) - y_mean)
            })
            .sum::<f64>()
            / n;

        let x_var: f64 = x_data
            .iter()
            .map(|xi| {
                let diff = xi.as_f64().unwrap_or(0.0) - x_mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1.0);

        let y_var: f64 = y_data
            .iter()
            .map(|yi| {
                let diff = yi.as_f64().unwrap_or(0.0) - y_mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1.0);

        let std_x = x_var.sqrt();
        let std_y = y_var.sqrt();

        if std_x == 0.0 || std_y == 0.0 {
            return Err(NumPyError::invalid_value("Division by zero in corrcoef"));
        }

        let correlation = covariance / (std_x * std_y);

        Ok(Array::from_vec(vec![T::from_f64(correlation)]))
    } else {
        let n = x_data.len() as f64;
        let x_mean: f64 = x_data
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .sum::<f64>()
            / n;

        let _x_var: f64 = x_data
            .iter()
            .map(|xi| {
                let diff = xi.as_f64().unwrap_or(0.0) - x_mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1.0);

        Ok(Array::from_vec(vec![T::from_f64(1.0)]))
    }
}

pub fn cov<T>(
    m: &Array<T>,
    y: Option<&Array<T>>,
    _rowvar: bool,
    _bias: bool,
    ddof: isize,
    _fweights: Option<&Array<T>>,
    _aweights: Option<&Array<T>>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if m.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute cov of empty array",
        ));
    }

    let m_data = m.to_vec();
    let n = m_data.len() as f64;
    let m_mean: f64 = m_data
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .sum::<f64>()
        / n;

    if let Some(y_arr) = y {
        let y_data = y_arr.to_vec();

        if m_data.len() != y_data.len() {
            return Err(NumPyError::invalid_value("m and y must have same length"));
        }

        let y_mean: f64 = y_data
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .sum::<f64>()
            / n;

        let covariance: f64 = m_data
            .iter()
            .zip(y_data.iter())
            .map(|(mi, yi)| {
                (mi.as_f64().unwrap_or(0.0) - m_mean) * (yi.as_f64().unwrap_or(0.0) - y_mean)
            })
            .sum::<f64>()
            / (n - (ddof as f64));

        Ok(Array::from_vec(vec![T::from_f64(covariance)]))
    } else {
        let variance: f64 = m_data
            .iter()
            .map(|mi| {
                let diff = mi.as_f64().unwrap_or(0.0) - m_mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - (ddof as f64));

        Ok(Array::from_vec(vec![T::from_f64(variance)]))
    }
}

pub fn histogram<T>(
    a: &Array<T>,
    bins: ArrayOrInt<'_, T>,
    range: Option<(T, T)>,
    _density: bool,
    _weights: Option<&Array<T>>,
) -> Result<(Array<T>, Array<T>), NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute histogram of empty array",
        ));
    }

    let data = a.to_vec();
    let num_bins = match bins {
        ArrayOrInt::Array(b) => b.size(),
        ArrayOrInt::Int(n) => n as usize,
    };

    if num_bins < 1 {
        return Err(NumPyError::invalid_value("Number of bins must be >= 1"));
    }

    let (min_val, max_val) = if let Some((ref rmin, ref rmax)) = range {
        (rmin.as_f64().unwrap(), rmax.as_f64().unwrap())
    } else {
        let min_v = data
            .iter()
            .map(|x| x.as_f64().unwrap())
            .fold(f64::INFINITY, |a, b| a.min(b));
        let max_v = data
            .iter()
            .map(|x| x.as_f64().unwrap())
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));
        (min_v, max_v)
    };

    if max_val <= min_val {
        return Err(NumPyError::invalid_value("Range must be positive"));
    }

    let bin_width = (max_val - min_val) / num_bins as f64;
    let mut hist_counts = vec![0usize; num_bins];

    for value in data.iter() {
        let v = value.as_f64().unwrap_or(0.0);
        if v >= min_val && v < max_val {
            let bin_idx = ((v - min_val) / bin_width) as usize;
            let bin_clamped = bin_idx.min(num_bins - 1);
            hist_counts[bin_clamped] += 1;
        }
    }

    let mut bin_edges = vec![0.0; num_bins + 1];
    for i in 0..=num_bins {
        bin_edges[i] = min_val + (i as f64 * bin_width);
    }

    let hist_array: Array<T> =
        Array::from_vec(hist_counts.iter().map(|&c| T::from_f64(c as f64)).collect());
    let edges_array: Array<T> = Array::from_vec(bin_edges.into_iter().map(T::from_f64).collect());

    Ok((hist_array, edges_array))
}

pub fn histogram2d<T>(
    x: &Array<T>,
    y: &Array<T>,
    bins: ArrayOrInt<'_, T>,
    range: &[Option<(T, T)>; 2],
    _density: bool,
    _weights: Option<&Array<T>>,
) -> Result<(Array<T>, Array<T>, Array<T>), NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if x.is_empty() || y.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute histogram2d of empty array",
        ));
    }

    let x_data = x.to_vec();
    let y_data = y.to_vec();

    let num_bins = match bins {
        ArrayOrInt::Array(b) => b.size(),
        ArrayOrInt::Int(n) => n as usize,
    };

    let (x_min, x_max) = match range[0] {
        Some((ref rmin, ref rmax)) => (rmin.as_f64().unwrap(), rmax.as_f64().unwrap()),
        None => {
            let min_v = x_data
                .iter()
                .map(|v| v.as_f64().unwrap())
                .fold(f64::INFINITY, |a, b| a.min(b));
            let max_v = x_data
                .iter()
                .map(|v| v.as_f64().unwrap())
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));
            (min_v, max_v)
        }
    };

    let (y_min, y_max) = match range[1] {
        Some((ref rmin, ref rmax)) => (rmin.as_f64().unwrap(), rmax.as_f64().unwrap()),
        None => {
            let min_v = y_data
                .iter()
                .map(|v| v.as_f64().unwrap())
                .fold(f64::INFINITY, |a, b| a.min(b));
            let max_v = y_data
                .iter()
                .map(|v| v.as_f64().unwrap())
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));
            (min_v, max_v)
        }
    };

    let x_width = (x_max - x_min) / num_bins as f64;
    let y_width = (y_max - y_min) / num_bins as f64;

    let mut hist_counts = vec![0usize; num_bins * num_bins];

    for (xi, yi) in x_data.iter().zip(y_data.iter()) {
        let x_val = xi.as_f64().unwrap_or(0.0);
        let y_val = yi.as_f64().unwrap_or(0.0);

        if x_val >= x_min && x_val < x_max && y_val >= y_min && y_val < y_max {
            let x_bin = ((x_val - x_min) / x_width) as usize;
            let y_bin = ((y_val - y_min) / y_width) as usize;
            let x_clamped = x_bin.min(num_bins - 1);
            let y_clamped = y_bin.min(num_bins - 1);
            hist_counts[y_clamped * num_bins + x_clamped] += 1;
        }
    }

    let hist_array: Array<T> =
        Array::from_vec(hist_counts.iter().map(|&c| T::from_f64(c as f64)).collect());
    let x_edges_array: Array<T> = Array::from_vec(
        (0..=num_bins)
            .map(|i| T::from_f64(x_min + (i as f64 * x_width)))
            .collect(),
    );
    let y_edges_array: Array<T> = Array::from_vec(
        (0..=num_bins)
            .map(|i| T::from_f64(y_min + (i as f64 * y_width)))
            .collect(),
    );

    Ok((hist_array, x_edges_array, y_edges_array))
}

pub fn histogramdd<T>(
    sample: &Array<T>,
    _bins: ArrayOrInt<'_, T>,
    _range: &[Option<(T, T)>],
    _density: bool,
    _weights: Option<&Array<T>>,
) -> Result<(Array<T>, Array<T>, Array<T>), NumPyError>
where
    T: Clone + Default + 'static,
{
    if sample.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute histogramdd of empty array",
        ));
    }

    Ok((
        Array::from_vec(Vec::<T>::new()),
        Array::from_vec(Vec::<T>::new()),
        Array::from_vec(Vec::<T>::new()),
    ))
}

pub fn bincount<T>(
    x: &Array<T>,
    _weights: Option<&Array<T>>,
    minlength: usize,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsI64 + FromF64 + 'static,
{
    if x.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute bincount of empty array",
        ));
    }

    let data = x.to_vec();
    let max_val = data
        .iter()
        .map(|v| v.as_i64().unwrap_or(0))
        .max()
        .unwrap_or(0) as usize
        + 1;
    let length = max_val.max(minlength);

    let mut counts = vec![0usize; length];

    for value in data.iter() {
        if let Some(idx) = value.as_i64() {
            if idx >= 0 && (idx as usize) < length {
                counts[idx as usize] += 1;
            }
        }
    }

    Ok(Array::from_vec(
        counts.iter().map(|&c| T::from_f64(c as f64)).collect(),
    ))
}

pub fn digitize<T>(x: &Array<T>, bins: &Array<T>, right: bool) -> Result<Array<isize>, NumPyError>
where
    T: Clone + AsF64,
{
    if x.is_empty() || bins.is_empty() {
        return Err(NumPyError::invalid_value("Cannot digitize empty arrays"));
    }

    let x_data = x.to_vec();
    let bins_data = bins.to_vec();

    let mut result = vec![0isize; x_data.len()];

    for (i, value) in x_data.iter().enumerate() {
        let v = value.as_f64().unwrap_or(0.0);
        let pos = if right {
            bins_data.iter().position(|b| v < b.as_f64().unwrap_or(0.0))
        } else {
            bins_data
                .iter()
                .position(|b| v <= b.as_f64().unwrap_or(0.0))
        };
        result[i] = pos
            .map(|idx| idx as isize)
            .unwrap_or(bins_data.len() as isize);
    }

    Ok(Array::from_vec(result))
}

pub mod exports {
    pub use super::{
        average, bincount, corrcoef, cov, digitize, histogram, histogram2d, histogramdd, median,
        nanmedian, nanpercentile, nanquantile, percentile, quantile, std, var,
    };
}

fn normalize_axes(axis: Option<&[isize]>, ndim: usize) -> Result<Vec<usize>, NumPyError> {
    match axis {
        None => Ok((0..ndim).collect()),
        Some(axes) => {
            if ndim == 0 {
                return Err(NumPyError::invalid_operation(
                    "Cannot specify axis for 0D array",
                ));
            }

            let mut normalized: Vec<usize> = axes
                .iter()
                .map(|&ax| {
                    let axis = if ax < 0 { ax + ndim as isize } else { ax };
                    if axis < 0 || axis >= ndim as isize {
                        return Err(NumPyError::index_error(axis as usize, ndim));
                    }
                    Ok(axis as usize)
                })
                .collect::<Result<_, _>>()?;

            normalized.sort_unstable();
            normalized.dedup();
            Ok(normalized)
        }
    }
}

fn percentile_internal<T>(
    a: &Array<T>,
    q: &Array<T>,
    axis: Option<&[isize]>,
    keepdims: bool,
    interpolation: &str,
    q_scale: f64,
    skip_nan: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute percentile of empty array",
        ));
    }

    let q_values = q
        .to_vec()
        .into_iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| NumPyError::invalid_value("Invalid percentile value"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let q_shape = q.shape().to_vec();
    let q_size = q.size();
    let q_is_scalar = q_size == 1;

    let reduced_axes = normalize_axes(axis, a.ndim())?;
    let reduction_shape = if axis.is_none() {
        if keepdims {
            vec![1; a.ndim()]
        } else {
            vec![]
        }
    } else {
        broadcast_shape_for_reduce(a.shape(), axis.unwrap_or(&[]), keepdims)
    };

    let mut output_shape = if q_is_scalar {
        reduction_shape.clone()
    } else {
        let mut shape = q_shape.clone();
        shape.extend(reduction_shape.iter().copied());
        shape
    };

    if output_shape.is_empty() {
        output_shape = vec![];
    }

    let mut output = Array::zeros(output_shape.clone());
    let reduction_size = if reduction_shape.is_empty() {
        1
    } else {
        reduction_shape.iter().product()
    };

    for (q_idx, &q_value) in q_values.iter().enumerate() {
        validate_q_range(q_value, q_scale)?;
        let normalized_q = q_value / q_scale;

        for output_idx in 0..reduction_size {
            let output_indices = if reduction_shape.is_empty() {
                vec![]
            } else {
                compute_multi_indices(output_idx, &reduction_shape)
            };

            let mut values =
                collect_reduction_values(a, &output_indices, &reduced_axes, keepdims, skip_nan)?;

            if values.is_empty() {
                let value = if skip_nan { f64::NAN } else { 0.0 };
                set_output_value(
                    &mut output,
                    q_idx,
                    &q_shape,
                    &output_indices,
                    q_is_scalar,
                    T::from_f64(value),
                )?;
                continue;
            }

            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let value = percentile_from_sorted(&values, normalized_q, interpolation)?;

            set_output_value(
                &mut output,
                q_idx,
                &q_shape,
                &output_indices,
                q_is_scalar,
                T::from_f64(value),
            )?;
        }
    }

    Ok(output)
}

fn validate_q_range(q: f64, scale: f64) -> Result<(), NumPyError> {
    let (min, max, label) = if (scale - 100.0).abs() < f64::EPSILON {
        (0.0, 100.0, "Percentiles must be in the range [0, 100]")
    } else {
        (0.0, 1.0, "Quantiles must be in the range [0, 1]")
    };

    if q < min || q > max {
        return Err(NumPyError::invalid_value(label));
    }

    Ok(())
}

fn collect_reduction_values<T>(
    a: &Array<T>,
    output_indices: &[usize],
    reduced_axes: &[usize],
    keepdims: bool,
    skip_nan: bool,
) -> Result<Vec<f64>, NumPyError>
where
    T: Clone + AsF64,
{
    let mut values = Vec::new();

    for linear_idx in 0..a.size() {
        let input_indices = compute_multi_indices(linear_idx, a.shape());
        if should_include_for_reduction(&input_indices, output_indices, reduced_axes, keepdims) {
            if let Some(value) = a.get(linear_idx).and_then(|val| val.as_f64()) {
                if skip_nan && value.is_nan() {
                    continue;
                }
                values.push(value);
            } else {
                return Err(NumPyError::invalid_value("Invalid data value"));
            }
        }
    }

    Ok(values)
}

fn should_include_for_reduction(
    input_indices: &[usize],
    output_indices: &[usize],
    reduced_axes: &[usize],
    keepdims: bool,
) -> bool {
    if keepdims {
        for (dim_idx, &input_dim_val) in input_indices.iter().enumerate() {
            if reduced_axes.contains(&dim_idx) {
                continue;
            }
            if output_indices.get(dim_idx) != Some(&input_dim_val) {
                return false;
            }
        }
        return true;
    }

    let mut output_dim_idx = 0;
    for (dim_idx, &input_dim_val) in input_indices.iter().enumerate() {
        if reduced_axes.contains(&dim_idx) {
            continue;
        }
        if output_indices.get(output_dim_idx) != Some(&input_dim_val) {
            return false;
        }
        output_dim_idx += 1;
    }

    true
}

fn percentile_from_sorted(sorted: &[f64], q: f64, interpolation: &str) -> Result<f64, NumPyError> {
    if sorted.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute percentile of empty slice",
        ));
    }

    if sorted.len() == 1 {
        return Ok(sorted[0]);
    }

    let n = sorted.len() as f64;
    let pos = q * (n - 1.0);
    let lower_idx = pos.floor() as usize;
    let upper_idx = pos.ceil() as usize;
    let frac = pos - lower_idx as f64;

    let lower = sorted[lower_idx];
    let upper = sorted[upper_idx];

    let value = match interpolation {
        "lower" => lower,
        "higher" => upper,
        "nearest" => {
            if frac <= 0.5 {
                lower
            } else {
                upper
            }
        }
        "midpoint" => (lower + upper) / 2.0,
        "linear" => lower * (1.0 - frac) + upper * frac,
        _ => return Err(NumPyError::invalid_value("Invalid interpolation method")),
    };

    Ok(value)
}

fn set_output_value<T>(
    output: &mut Array<T>,
    q_idx: usize,
    q_shape: &[usize],
    reduction_indices: &[usize],
    q_is_scalar: bool,
    value: T,
) -> Result<(), NumPyError>
where
    T: Clone + Default + 'static,
{
    if output.shape().is_empty() {
        return output.set(0, value);
    }

    let indices = if q_is_scalar {
        reduction_indices.to_vec()
    } else {
        let mut idx = if q_shape.is_empty() {
            vec![q_idx]
        } else {
            compute_multi_indices(q_idx, q_shape)
        };
        idx.extend_from_slice(reduction_indices);
        idx
    };

    output.set_by_indices(&indices, value)
}

pub enum ArrayOrInt<'a, T> {
    Array(&'a Array<T>),
    Int(isize),
}

pub trait AsF64 {
    fn as_f64(&self) -> Option<f64>;
}

impl AsF64 for f64 {
    fn as_f64(&self) -> Option<f64> {
        Some(*self)
    }
}

impl AsF64 for f32 {
    fn as_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

impl AsF64 for i64 {
    fn as_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

impl AsF64 for i32 {
    fn as_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

impl AsF64 for usize {
    fn as_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

pub trait AsI64 {
    fn as_i64(&self) -> Option<i64>;
}

impl AsI64 for i64 {
    fn as_i64(&self) -> Option<i64> {
        Some(*self)
    }
}

impl AsI64 for i32 {
    fn as_i64(&self) -> Option<i64> {
        Some(*self as i64)
    }
}

pub trait FromF64 {
    fn from_f64(value: f64) -> Self;
}

impl FromF64 for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

impl FromF64 for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl FromF64 for i64 {
    fn from_f64(value: f64) -> Self {
        value as i64
    }
}

impl FromF64 for i32 {
    fn from_f64(value: f64) -> Self {
        value as i32
    }
}

impl FromF64 for usize {
    fn from_f64(value: f64) -> Self {
        if value <= 0.0 {
            0
        } else {
            value as usize
        }
    }
}
