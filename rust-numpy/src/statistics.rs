use crate::array::Array;
use crate::error::NumPyError;
use std::cmp::Ordering;

pub fn median<T>(
    a: &Array<T>,
    _axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    _overwrite_input: bool,
    _keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute median of empty array",
        ));
    }

    let data = a.to_vec();
    let mut sorted = data.clone();
    sorted.sort_by(|a, b| {
        a.as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&b.as_f64().unwrap_or(0.0))
            .unwrap_or(Ordering::Equal)
    });

    let n = sorted.len();
    let mid = n / 2;

    let median_value = if n.is_multiple_of(2) {
        (sorted[mid - 1].as_f64().unwrap_or(0.0) + sorted[mid].as_f64().unwrap_or(0.0)) / 2.0
    } else {
        sorted[mid].as_f64().unwrap_or(0.0)
    };

    Ok(Array::from_vec(vec![T::from_f64(median_value)]))
}

pub fn percentile<T>(
    a: &Array<T>,
    q: &Array<T>,
    _axis: Option<&[isize]>,
    _out: Option<&mut Array<T>>,
    _overwrite_input: bool,
    _method: &str,
    _keepdims: bool,
    interpolation: &str,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute percentile of empty array",
        ));
    }

    let mut data = a.to_vec();
    data.sort_by(|a, b| {
        a.as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&b.as_f64().unwrap_or(0.0))
            .unwrap_or(Ordering::Equal)
    });

    let n = data.len();
    let mut result = vec![T::default(); q.size()];

    for (i, qv) in q.to_vec().iter().enumerate() {
        if let Some(percentile) = qv.as_f64() {
            let idx = (percentile / 100.0 * (n as f64 - 1.0) / 100.0) as usize;
            let idx_clamped = idx.min(n - 1);
            let value = if interpolation == "nearest" || interpolation == "lower" {
                data[idx_clamped].as_f64().unwrap_or(0.0)
            } else if interpolation == "higher" {
                data[(idx_clamped + 1).min(n - 1)].as_f64().unwrap_or(0.0)
            } else if interpolation == "midpoint" {
                (data[idx_clamped].as_f64().unwrap_or(0.0)
                    + data[(idx_clamped + 1).min(n - 1)].as_f64().unwrap_or(0.0))
                    / 2.0
            } else {
                let lower = data[idx_clamped].as_f64().unwrap_or(0.0);
                let upper = data[(idx_clamped + 1).min(n - 1)].as_f64().unwrap_or(0.0);
                let frac = (percentile / 100.0 * (n as f64 - 1.0) / 100.0) - (idx_clamped as f64);
                lower * (1.0 - frac) + upper * frac
            };
            result[i] = T::from_f64(value);
        }
    }

    Ok(Array::from_vec(result))
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
    percentile(
        a,
        q,
        axis,
        out,
        overwrite_input,
        method,
        keepdims,
        interpolation,
    )
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
    rowvar: bool,
    bias: Option<bool>,
    ddof: Option<isize>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    // corrcoef is essentially cov normalized by standard deviations
    // bias defaults to False for corrcoef (unbiased estimator)
    let bias_val = bias.unwrap_or(false);
    let ddof_val = ddof.unwrap_or(1);

    // Compute covariance matrix
    let c = cov(x, y, rowvar, bias_val, ddof_val, None, None)?;

    // Normalize by standard deviations to get correlation matrix
    let c_data = c.to_vec();
    let c_shape = c.shape();

    // For 2D covariance matrix, normalize: corr[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
    if c_shape.len() == 2 {
        let n = c_shape[0];
        let mut corr_data = vec![0.0; c_data.len()];

        for i in 0..n {
            for j in 0..n {
                let cov_ij = c_data[i * n + j].as_f64().unwrap_or(0.0);
                let cov_ii = c_data[i * n + i].as_f64().unwrap_or(0.0);
                let cov_jj = c_data[j * n + j].as_f64().unwrap_or(0.0);

                let std_i = cov_ii.sqrt();
                let std_j = cov_jj.sqrt();

                if std_i == 0.0 || std_j == 0.0 {
                    return Err(NumPyError::invalid_value(
                        "Division by zero in corrcoef: zero variance variable",
                    ));
                }

                corr_data[i * n + j] = cov_ij / (std_i * std_j);
            }
        }

        Ok(Array::from_shape_vec(c_shape.to_vec(), corr_data.into_iter().map(T::from_f64).collect()))
    } else {
        // For 1D case, single correlation coefficient is 1.0
        Ok(Array::from_vec(vec![T::from_f64(1.0)]))
    }
}

pub fn cov<T>(
    m: &Array<T>,
    y: Option<&Array<T>>,
    rowvar: bool,
    bias: bool,
    ddof: isize,
    fweights: Option<&Array<T>>,
    aweights: Option<&Array<T>>,
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
    let m_shape = m.shape();

    // Handle 1D case
    if m_shape.len() == 1 {
        let n = m_data.len() as f64;

        if let Some(y_arr) = y {
            let y_data = y_arr.to_vec();
            if y_data.len() != m_data.len() {
                return Err(NumPyError::invalid_value(
                    "m and y must have same length",
                ));
            }

            // Compute covariance between two 1D arrays
            let m_mean: f64 = m_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;
            let y_mean: f64 = y_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;

            let covariance: f64 = m_data
                .iter()
                .zip(y_data.iter())
                .map(|(mi, yi)| {
                    (mi.as_f64().unwrap_or(0.0) - m_mean) * (yi.as_f64().unwrap_or(0.0) - y_mean)
                })
                .sum::<f64>();

            let denom = if bias { n } else { (n as isize - ddof).max(1) as f64 };
            let cov_val = covariance / denom;

            // Return 2x2 covariance matrix
            let m_var: f64 = m_data
                .iter()
                .map(|mi| {
                    let diff = mi.as_f64().unwrap_or(0.0) - m_mean;
                    diff * diff
                })
                .sum::<f64>()
                / denom;
            let y_var: f64 = y_data
                .iter()
                .map(|yi| {
                    let diff = yi.as_f64().unwrap_or(0.0) - y_mean;
                    diff * diff
                })
                .sum::<f64>()
                / denom;

            Ok(Array::from_shape_vec(
                vec![2, 2],
                vec![
                    T::from_f64(m_var),
                    T::from_f64(cov_val),
                    T::from_f64(cov_val),
                    T::from_f64(y_var),
                ],
            ))
        } else {
            // Single 1D array - return 1x1 matrix (variance)
            let mean: f64 = m_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;
            let variance: f64 = m_data
                .iter()
                .map(|mi| {
                    let diff = mi.as_f64().unwrap_or(0.0) - mean;
                    diff * diff
                })
                .sum::<f64>();

            let denom = if bias { n } else { (n as isize - ddof).max(1) as f64 };
            Ok(Array::from_shape_vec(vec![1, 1], vec![T::from_f64(variance / denom)]))
        }
    } else if m_shape.len() == 2 {
        // Handle 2D case
        let (rows, cols) = (m_shape[0], m_shape[1]);

        // Determine number of variables and observations
        let (n_vars, n_obs) = if rowvar { (rows, cols) } else { (cols, rows) };

        if n_obs < 2 {
            return Err(NumPyError::invalid_value(
                "Need at least 2 observations to compute covariance",
            ));
        }

        // fweights and aweights validation
        if fweights.is_some() || aweights.is_some() {
            return Err(NumPyError::invalid_value(
                "fweights and aweights are not yet supported for 2D arrays",
            ));
        }

        // Transpose data if rowvar=false so columns become variables (rows)
        let data = if !rowvar {
            // Transpose: rows become columns
            let mut transposed = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = m_data[i * cols + j].as_f64().unwrap_or(0.0);
                }
            }
            transposed
        } else {
            m_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()
        };

        // Compute mean for each variable (row)
        let mut means = vec![0.0; n_vars];
        for i in 0..n_vars {
            let row_sum: f64 = (0..n_obs).map(|j| data[i * n_obs + j]).sum();
            means[i] = row_sum / n_obs as f64;
        }

        // Compute covariance matrix
        let mut cov_matrix = vec![0.0; n_vars * n_vars];
        let denom = if bias {
            n_obs as f64
        } else {
            (n_obs as isize - ddof).max(1) as f64
        };

        for i in 0..n_vars {
            for j in i..n_vars {
                let mut cov_sum = 0.0;
                for k in 0..n_obs {
                    let diff_i = data[i * n_obs + k] - means[i];
                    let diff_j = data[j * n_obs + k] - means[j];
                    cov_sum += diff_i * diff_j;
                }
                let cov_val = cov_sum / denom;
                cov_matrix[i * n_vars + j] = cov_val;
                cov_matrix[j * n_vars + i] = cov_val; // Symmetric
            }
        }

        Ok(Array::from_shape_vec(
            vec![n_vars, n_vars],
            cov_matrix.into_iter().map(T::from_f64).collect(),
        ))
    } else {
        Err(NumPyError::invalid_value(
            "cov requires 1D or 2D arrays",
        ))
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
        percentile, ptp, quantile, std, var,
    };
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

/// Peak-to-peak (maximum - minimum) values along an axis (similar to np.ptp).
///
/// # Arguments
/// - `a`: Input array
/// - `axis`: Optional axis along which to find the peak-to-peak
/// - `keepdims`: If true, the reduced axes are left in the result as dimensions with size one
///
/// # Returns
/// Array containing the peak-to-peak (range) of values
pub fn ptp<T>(
    a: &Array<T>,
    _axis: Option<&[isize]>,
    _keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + AsF64 + FromF64 + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value(
            "Cannot compute ptp of empty array",
        ));
    }

    let data = a.to_vec();

    let min_val = data
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_val = data
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    Ok(Array::from_vec(vec![T::from_f64(max_val - min_val)]))
}
