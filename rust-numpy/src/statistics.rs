use crate::error::NumPyError;
use crate::array::Array;
use std::cmp::Ordering;

pub fn median<T>(a: &Array<T>, axis: Option<&[isize]>, out: Option<&mut Array<T>>, overwrite_input: bool, keepdims: bool) -> Result<Array<T>, NumPyError> {
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute median of empty array"));
    }
    
    let data = a.to_vec();
    let mut sorted = data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    let n = sorted.len();
    let mid = n / 2;
    
    let median_value = if n % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };
    
    Ok(Array::from_vec(vec![median_value]))
}

pub fn percentile<T>(a: &Array<T>, q: &Array<T>, axis: Option<&[isize]>, out: Option<&mut Array<T>>, overwrite_input: bool, method: &str, keepdims: bool, interpolation: &str) -> Result<Array<T>, NumPyError> {
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute percentile of empty array"));
    }
    
    let mut data = a.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    let n = data.len();
    let mut result = vec![T::default(); q.size()];
    
    for (i, &qv) in q.to_vec().iter().enumerate() {
        if let Some(percentile) = qv.as_f64() {
            let idx = (percentile / 100.0 * (n as f64 - 1.0) / 100.0) as usize;
            let idx_clamped = idx.min(n - 1);
            let value = if interpolation == "nearest" || interpolation == "lower" {
                data[idx_clamped]
            } else if interpolation == "higher" {
                data[(idx_clamped + 1).min(n - 1)]
            } else if interpolation == "midpoint" {
                (data[idx_clamped] + data[(idx_clamped + 1).min(n - 1)]) / 2.0
            } else {
                let lower = data[idx_clamped];
                let upper = data[(idx_clamped + 1).min(n - 1)];
                let frac = (percentile / 100.0 * (n as f64 - 1.0) / 100.0) - (idx_clamped as f64);
                lower * (1.0 - frac) + upper * frac
            };
            result[i] = value;
        }
    }
    
    Ok(Array::from_vec(result))
}

pub fn quantile<T>(a: &Array<T>, q: &Array<T>, axis: Option<&[isize]>, out: Option<&mut Array<T>>, overwrite_input: bool, method: &str, keepdims: bool, interpolation: &str) -> Result<Array<T>, NumPyError> {
    percentile(a, q, axis, out, overwrite_input, method, keepdims, interpolation)
}

pub fn average<T>(a: &Array<T>, axis: Option<&[isize]>, weights: Option<&Array<T>>, returned: bool) -> Result<Array<T>, NumPyError> {
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute average of empty array"));
    }
    
    let data = a.to_vec();
    
    if let Some(w) = weights {
        if w.size() != a.size() {
            return Err(NumPyError::invalid_shape("Weights must have same size as array"));
        }
        
        let w_data = w.to_vec();
        let sum_weights: f64 = w_data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum();
        
        if sum_weights == 0.0 {
            return Err(NumPyError::invalid_value("Sum of weights must not be zero"));
        }
        
        let weighted_sum: f64 = data.iter()
            .zip(w_data.iter())
            .map(|(x, w)| x.as_f64().unwrap_or(0.0) * w.as_f64().unwrap_or(0.0))
            .sum();
        
        Ok(Array::from_vec(vec![weighted_sum / sum_weights]))
    } else {
        let sum: f64 = data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum();
        Ok(Array::from_vec(vec![sum / data.len() as f64]))
    }
}

pub fn std<T>(a: &Array<T>, axis: Option<&[isize]>, dtype: Option<crate::dtype::Dtype>, out: Option<&mut Array<T>>, ddof: isize, keepdims: bool) -> Result<Array<T>, NumPyError> {
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute std of empty array"));
    }
    
    let data = a.to_vec();
    let mean: f64 = data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter()
        .map(|x| {
            let diff = x.as_f64().unwrap_or(0.0) - mean;
            diff * diff
        })
        .sum::<f64>() / (data.len() - ddof) as f64;
    
    Ok(Array::from_vec(vec![variance.sqrt()]))
}

pub fn var<T>(a: &Array<T>, axis: Option<&[isize]>, dtype: Option<crate::dtype::Dtype>, out: Option<&mut Array<T>>, ddof: isize, keepdims: bool) -> Result<Array<T>, NumPyError> {
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute var of empty array"));
    }
    
    let data = a.to_vec();
    let mean: f64 = data.iter().map(|x| x.as_f64().unwrap_or(0.0)).sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter()
        .map(|x| {
            let diff = x.as_f64().unwrap_or(0.0) - mean;
            diff * diff
        })
        .sum::<f64>() / (data.len() - ddof) as f64;
    
    Ok(Array::from_vec(vec![variance]))
}

pub fn corrcoef<T>(x: &Array<T>, y: Option<&Array<T>>, rowvar: bool) -> Result<Array<T>, NumPyError> {
    if x.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute corrcoef of empty array"));
    }
    
    let x_data = x.to_vec();
    
    if let Some(y_arr) = y {
        let y_data = y_arr.to_vec();
        
        if x_data.len() != y_data.len() {
            return Err(NumPyError::invalid_shape("x and y must have same length"));
        }
        
        let n = x_data.len() as f64;
        let x_mean: f64 = x_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;
        let y_mean: f64 = y_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;
        
        let covariance: f64 = x_data.iter()
            .zip(y_data.iter())
            .map(|(xi, yi)| {
                (xi.as_f64().unwrap_or(0.0) - x_mean) * (yi.as_f64().unwrap_or(0.0) - y_mean)
            })
            .sum::<f64>() / n;
        
        let x_var: f64 = x_data.iter()
            .map(|xi| {
                let diff = xi.as_f64().unwrap_or(0.0) - x_mean;
                diff * diff
            })
            .sum::<f64>() / (n - 1.0);
        
        let y_var: f64 = y_data.iter()
            .map(|yi| {
                let diff = yi.as_f64().unwrap_or(0.0) - y_mean;
                diff * diff
            })
            .sum::<f64>() / (n - 1.0);
        
        let std_x = x_var.sqrt();
        let std_y = y_var.sqrt();
        
        if std_x == 0.0 || std_y == 0.0 {
            return Err(NumPyError::invalid_value("Division by zero in corrcoef"));
        }
        
        let correlation = covariance / (std_x * std_y);
        
        Ok(Array::from_vec(vec![correlation]))
    } else {
        let n = x_data.len() as f64;
        let x_mean: f64 = x_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;
        
        let x_var: f64 = x_data.iter()
            .map(|xi| {
                let diff = xi.as_f64().unwrap_or(0.0) - x_mean;
                diff * diff
            })
            .sum::<f64>() / (n - 1.0);
        
        Ok(Array::from_vec(vec![1.0]))
    }
}

pub fn cov<T>(m: &Array<T>, y: Option<&Array<T>>, rowvar: bool, bias: bool, ddof: isize, fweights: Option<&Array<T>>, aweights: Option<&Array<T>>) -> Result<Array<T>, NumPyError> {
    if m.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute cov of empty array"));
    }
    
    let m_data = m.to_vec();
    let n = m_data.len() as f64;
    let m_mean: f64 = m_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;
    
    if let Some(y_arr) = y {
        let y_data = y_arr.to_vec();
        
        if m_data.len() != y_data.len() {
            return Err(NumPyError::invalid_shape("m and y must have same length"));
        }
        
        let y_mean: f64 = y_data.iter().map(|v| v.as_f64().unwrap_or(0.0)).sum::<f64>() / n;
        
        let covariance: f64 = m_data.iter()
            .zip(y_data.iter())
            .map(|(mi, yi)| {
                (mi.as_f64().unwrap_or(0.0) - m_mean) * (yi.as_f64().unwrap_or(0.0) - y_mean)
            })
            .sum::<f64>() / (n - (ddof as f64));
        
        Ok(Array::from_vec(vec![covariance]))
    } else {
        let variance: f64 = m_data.iter()
            .map(|mi| {
                let diff = mi.as_f64().unwrap_or(0.0) - m_mean;
                diff * diff
            })
            .sum::<f64>() / (n - (ddof as f64));
        
        Ok(Array::from_vec(vec![variance]))
    }
}

pub fn histogram<T>(a: &Array<T>, bins: ArrayOrInt, range: Option<(T, T)>, density: bool, weights: Option<&Array<T>>) -> Result<(Array<T>, Array<T>), NumPyError> {
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute histogram of empty array"));
    }
    
    let data = a.to_vec();
    let num_bins = match bins {
        ArrayOrInt::Array(b) => b.size(),
        ArrayOrInt::Int(n) => *n as usize,
    };
    
    if num_bins < 1 {
        return Err(NumPyError::invalid_value("Number of bins must be >= 1"));
    }
    
    let (min_val, max_val) = if let Some((rmin, rmax)) = range {
        (rmin.as_f64().unwrap(), rmax.as_f64().unwrap())
    } else {
        let min_v = data.iter().map(|x| x.as_f64().unwrap()).fold(f64::INFINITY, |a, b| a.min(*b));
        let max_v = data.iter().map(|x| x.as_f64().unwrap()).fold(f64::NEG_INFINITY, |a, b| a.max(*b));
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
    
    let hist_array: Array<T> = Array::from_vec(hist_counts.iter().map(|&c| (c as f64).into()).collect());
    let edges_array: Array<T> = Array::from_vec(bin_edges);
    
    Ok((hist_array, edges_array))
}

pub fn histogram2d<T>(x: &Array<T>, y: &Array<T>, bins: ArrayOrInt, range: &[Option<(T, T)>; 2], density: bool, weights: Option<&Array<T>>) -> Result<(Array<T>, Array<T>, Array<T>), NumPyError> {
    if x.is_empty() || y.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute histogram2d of empty array"));
    }
    
    let x_data = x.to_vec();
    let y_data = y.to_vec();
    
    let num_bins = match bins {
        ArrayOrInt::Array(b) => b.size(),
        ArrayOrInt::Int(n) => *n as usize,
    };
    
    let (x_min, x_max) = match range[0] {
        Some((rmin, rmax)) => (rmin.as_f64().unwrap(), rmax.as_f64().unwrap()),
        None => {
            let min_v = x_data.iter().map(|v| v.as_f64().unwrap()).fold(f64::INFINITY, |a, b| a.min(*b));
            let max_v = x_data.iter().map(|v| v.as_f64().unwrap()).fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            (min_v, max_v)
        }
    };
    
    let (y_min, y_max) = match range[1] {
        Some((rmin, rmax)) => (rmin.as_f64().unwrap(), rmax.as_f64().unwrap()),
        None => {
            let min_v = y_data.iter().map(|v| v.as_f64().unwrap()).fold(f64::INFINITY, |a, b| a.min(*b));
            let max_v = y_data.iter().map(|v| v.as_f64().unwrap()).fold(f64::NEG_INFINITY, |a, b| a.max(*b));
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
    
    let hist_array: Array<T> = Array::from_vec(hist_counts.iter().map(|&c| (c as f64).into()).collect());
    let x_edges_array: Array<T> = Array::from_vec((0..=num_bins).map(|i| (x_min + (i as f64 * x_width)).into()).collect());
    let y_edges_array: Array<T> = Array::from_vec((0..=num_bins).map(|i| (y_min + (i as f64 * y_width)).into()).collect());
    
    Ok((hist_array, x_edges_array, y_edges_array))
}

pub fn histogramdd<T>(sample: &Array<T>, bins: ArrayOrInt, range: &[Option<(T, T)>], density: bool, weights: Option<&Array<T>>) -> Result<(Array<T>, Array<T>, Array<T>), NumPyError> {
    if sample.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute histogramdd of empty array"));
    }
    
    Ok((Array::from_vec(vec![]), Array::from_vec(vec![]), Array::from_vec(vec![])))
}

pub fn bincount<T>(x: &Array<T>, weights: Option<&Array<T>>, minlength: usize) -> Result<Array<T>, NumPyError> {
    if x.is_empty() {
        return Err(NumPyError::invalid_value("Cannot compute bincount of empty array"));
    }
    
    let data = x.to_vec();
    let max_val = data.iter().map(|v| v.as_i64().unwrap_or(0)).max().unwrap_or(0) as usize + 1;
    let length = max_val.max(minlength);
    
    let mut counts = vec![0usize; length];
    
    for value in data.iter() {
        if let Some(idx) = value.as_i64() {
            if idx >= 0 && (idx as usize) < length {
                counts[idx as usize] += 1;
            }
        }
    }
    
    Ok(Array::from_vec(counts.iter().map(|&c| (*c as f64).into()).collect()))
}

pub fn digitize<T>(x: &Array<T>, bins: &Array<T>, right: bool) -> Result<Array<isize>, NumPyError> {
    if x.is_empty() || bins.is_empty() {
        return Err(NumPyError::invalid_value("Cannot digitize empty arrays"));
    }
    
    let x_data = x.to_vec();
    let bins_data = bins.to_vec();
    
    let mut result = vec![0isize; x_data.len()];
    
    for (i, value) in x_data.iter().enumerate() {
        let v = value.as_f64().unwrap_or(0.0);
        let mut bin_idx = if right {
            bins_data.iter().position(|b| v < b.as_f64().unwrap_or(0.0)).unwrap_or(bins_data.len() as isize) - 1
        } else {
            bins_data.iter().position(|b| v <= b.as_f64().unwrap_or(0.0)).unwrap_or(bins_data.len() as isize) - 1
        };
        result[i] = bin_idx;
    }
    
    Ok(Array::from_vec(result))
}

pub mod exports {
    pub use super::{
        median, percentile, quantile, average, std, var, corrcoef, cov,
        histogram, histogram2d, histogramdd, bincount, digitize
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