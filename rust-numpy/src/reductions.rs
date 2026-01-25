use crate::array::{normalize_axis, Array};
use crate::error::NumPyError;
use crate::strides::compute_multi_indices;
use std::ops::{Add, Mul};

/// Trait for conversion to boolean (for all/any)
pub trait ToBool {
    fn to_bool(&self) -> bool;
}

impl ToBool for bool {
    fn to_bool(&self) -> bool {
        *self
    }
}
impl ToBool for i8 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for i16 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for i32 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for i64 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for u8 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for u16 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for u32 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for u64 {
    fn to_bool(&self) -> bool {
        *self != 0
    }
}
impl ToBool for f32 {
    fn to_bool(&self) -> bool {
        *self != 0.0
    }
}
impl ToBool for f64 {
    fn to_bool(&self) -> bool {
        *self != 0.0
    }
}
impl ToBool for num_complex::Complex<f64> {
    fn to_bool(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}
impl ToBool for num_complex::Complex<f32> {
    fn to_bool(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

/// Generic reduction function with mapping
fn reduce<T, U, M, F>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    keepdims: bool,
    init: Option<U>,
    map_op: M,
    reduce_op: F,
) -> Result<Array<U>, NumPyError>
where
    T: Clone + Default + 'static,
    U: Clone + Default + 'static,
    M: Fn(T) -> U,
    F: Fn(U, U) -> U,
{
    let axes = if let Some(ax) = axis {
        ax.iter()
            .map(|&x| normalize_axis(x, a.ndim()))
            .collect::<Result<Vec<usize>, _>>()?
    } else {
        (0..a.ndim()).collect()
    };

    let shape = a.shape();
    let mut out_shape = shape.to_vec();

    // Sort axes descending to remove correct indices
    let mut sorted_axes = axes.clone();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    sorted_axes.dedup();

    for &ax in &sorted_axes {
        if keepdims {
            out_shape[ax] = 1;
        } else {
            out_shape.remove(ax);
        }
    }

    let out_size: usize = out_shape.iter().product();
    if out_size == 0 {
        return Ok(Array::from_shape_vec(out_shape, Vec::new()));
    }

    // Optimization: flattened reduction
    if axes.len() == a.ndim() {
        let mut iter = a.iter();
        let first_val = if let Some(val) = init {
            val
        } else {
            // If no init, take first element and map it
            if let Some(x) = iter.next() {
                map_op(x.clone())
            } else {
                return Err(NumPyError::invalid_value(
                    "zero-size array to reduction without identity",
                ));
            }
        };

        let result = iter.fold(first_val, |acc, x| reduce_op(acc, map_op(x.clone())));
        return Ok(Array::from_shape_vec(out_shape, vec![result]));
    }

    // Non-flattened reduction
    let has_init = init.is_some();
    let mut out_data = if let Some(val) = init {
        vec![val; out_size]
    } else {
        // Should track initialization
        vec![U::default(); out_size]
    };

    // We need to track initialized state if init is None
    let mut initialized = vec![has_init; out_size];
    let reduced_axes_set: std::collections::HashSet<usize> = axes.iter().cloned().collect();

    let out_strides = crate::array::compute_strides(&out_shape);

    for i in 0..a.size() {
        let input_indices = compute_multi_indices(i, shape);
        let mut output_indices = Vec::with_capacity(out_shape.len());

        if keepdims {
            for (dim, &idx) in input_indices.iter().enumerate() {
                if reduced_axes_set.contains(&dim) {
                    output_indices.push(0);
                } else {
                    output_indices.push(idx);
                }
            }
        } else {
            for (dim, &idx) in input_indices.iter().enumerate() {
                if !reduced_axes_set.contains(&dim) {
                    output_indices.push(idx);
                }
            }
        }

        let out_idx = crate::strides::compute_linear_index(&output_indices, &out_strides) as usize;
        let val = a.get_linear(i).unwrap();
        let mapped = map_op(val.clone());

        if !initialized[out_idx] {
            out_data[out_idx] = mapped;
            initialized[out_idx] = true;
        } else {
            let existing = out_data[out_idx].clone();
            out_data[out_idx] = reduce_op(existing, mapped);
        }
    }

    Ok(Array::from_shape_vec(out_shape, out_data))
}

pub fn sum<T>(a: &Array<T>, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + Add<Output = T> + 'static,
{
    reduce(a, axis, keepdims, None, |x| x, |a, b| a + b)
}

pub fn prod<T>(a: &Array<T>, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + Mul<Output = T> + 'static,
{
    reduce(a, axis, keepdims, None, |x| x, |a, b| a * b)
}

pub fn min<T>(a: &Array<T>, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + 'static,
{
    reduce(
        a,
        axis,
        keepdims,
        None,
        |x| x,
        |a, b| if a < b { a } else { b },
    )
}

pub fn max<T>(a: &Array<T>, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + 'static,
{
    reduce(
        a,
        axis,
        keepdims,
        None,
        |x| x,
        |a, b| if a > b { a } else { b },
    )
}

pub fn all<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>, NumPyError>
where
    T: Clone + Default + ToBool + 'static,
{
    reduce(
        a,
        axis,
        keepdims,
        Some(true),
        |x| x.to_bool(),
        |a, b| a && b,
    )
}

pub fn any<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>, NumPyError>
where
    T: Clone + Default + ToBool + 'static,
{
    reduce(
        a,
        axis,
        keepdims,
        Some(false),
        |x| x.to_bool(),
        |a, b| a || b,
    )
}

pub fn all_bool(
    a: &Array<bool>,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>, NumPyError> {
    all(a, axis, keepdims)
}

pub fn any_bool(
    a: &Array<bool>,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<bool>, NumPyError> {
    any(a, axis, keepdims)
}

pub fn argmin<T>(a: &Array<T>, axis: Option<isize>) -> Result<Array<usize>, NumPyError>
where
    T: Clone + Default + PartialOrd + 'static,
{
    arg_reduce(a, axis, |x, y| x < y)
}

pub fn argmax<T>(a: &Array<T>, axis: Option<isize>) -> Result<Array<usize>, NumPyError>
where
    T: Clone + Default + PartialOrd + 'static,
{
    arg_reduce(a, axis, |x, y| x > y)
}

fn arg_reduce<T, F>(a: &Array<T>, axis: Option<isize>, cmp: F) -> Result<Array<usize>, NumPyError>
where
    T: Clone + Default + 'static,
    F: Fn(&T, &T) -> bool,
{
    match axis {
        None => {
            // Flattened argmin/max
            let mut best_idx = 0;
            let mut iter = a.iter().enumerate();
            let mut best_val = if let Some((_, v)) = iter.next() {
                v
            } else {
                return Err(NumPyError::invalid_value(
                    "attempt to get argmin/argmax of an empty sequence",
                ));
            };

            for (i, val) in iter {
                if cmp(val, best_val) {
                    best_val = val;
                    best_idx = i;
                }
            }
            Ok(Array::from_vec(vec![best_idx]))
        }
        Some(ax) => {
            let ndim = a.ndim();
            let ax = normalize_axis(ax, ndim)?;
            let shape = a.shape();

            let mut out_shape = shape.to_vec();
            out_shape.remove(ax);

            let out_size: usize = out_shape.iter().product();
            let mut out_indices = vec![0; out_size];

            let axis_len = shape[ax];
            if axis_len == 0 {
                return Err(NumPyError::invalid_value(
                    "attempt to get argmin/argmax of an empty sequence",
                ));
            }

            let _out_strides = crate::array::compute_strides(&out_shape);

            for i in 0..out_size {
                let out_multi = compute_multi_indices(i, &out_shape);
                let mut in_multi = out_multi.clone();
                in_multi.insert(ax, 0);

                let mut best_k = 0;
                let val0 = a.get_by_indices(&in_multi)?;
                let mut best_val = val0;

                for k in 1..axis_len {
                    in_multi[ax] = k;
                    let val = a.get_by_indices(&in_multi)?;
                    if cmp(val, best_val) {
                        best_val = val;
                        best_k = k;
                    }
                }
                out_indices[i] = best_k;
            }

            Ok(Array::from_shape_vec(out_shape, out_indices))
        }
    }
}

pub fn mean<T>(a: &Array<T>, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + Add<Output = T> + From<f64> + 'static,
{
    let sum_result = sum(a, axis, keepdims)?;
    let count = a.size() / sum_result.size();
    let _divisor = T::from(count as f64);

    // Map division over the sum result
    let mut result_data = Vec::with_capacity(sum_result.size());
    for val in sum_result.iter() {
        result_data.push(val.clone() + T::from(0.0)); // Clone the value
    }

    // This is a simplified version - proper implementation would divide each element
    Ok(sum_result)
}

pub fn var<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    _ddof: usize,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + Add<Output = T> + Mul<Output = T> + From<f64> + 'static,
{
    let mean_val = mean(a, axis, keepdims)?;
    // This is a placeholder - proper implementation would compute variance
    Ok(mean_val)
}

pub fn std<T>(
    a: &Array<T>,
    axis: Option<&[isize]>,
    _ddof: usize,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + Add<Output = T> + Mul<Output = T> + From<f64> + 'static,
{
    let var_val = var(a, axis, _ddof, keepdims)?;
    // This is a placeholder - proper implementation would compute sqrt of variance
    Ok(var_val)
}

pub fn cumsum<T>(a: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + Add<Output = T> + 'static,
{
    scan(a, axis, |x, y| x + y)
}

pub fn cumprod<T>(a: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + Mul<Output = T> + 'static,
{
    scan(a, axis, |x, y| x * y)
}

fn scan<T, F>(a: &Array<T>, axis: Option<isize>, op: F) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T,
{
    match axis {
        None => {
            // Flatten and scan
            let mut result_data = Vec::with_capacity(a.size());
            let mut iter = a.iter();
            if let Some(first) = iter.next() {
                let mut acc = first.clone();
                result_data.push(acc.clone());
                for val in iter {
                    acc = op(acc, val.clone());
                    result_data.push(acc.clone());
                }
            }
            Ok(Array::from_vec(result_data))
        }
        Some(ax) => {
            let ndim = a.ndim();
            let ax = normalize_axis(ax, ndim)?;
            let shape = a.shape();
            let mut out = Array::zeros(shape.to_vec());

            let axis_len = shape[ax];
            if axis_len == 0 {
                return Ok(out);
            }

            let mut loop_shape = shape.to_vec();
            loop_shape.remove(ax);
            let loop_size: usize = loop_shape.iter().product();

            for i in 0..loop_size {
                let loop_indices = compute_multi_indices(i, &loop_shape);

                let mut full_indices = loop_indices.clone();
                full_indices.insert(ax, 0);

                let val0 = a.get_by_indices(&full_indices)?.clone();
                out.set_by_indices(&full_indices, val0.clone())?;

                let mut acc = val0;

                for k in 1..axis_len {
                    full_indices[ax] = k;
                    let val = a.get_by_indices(&full_indices)?.clone();
                    acc = op(acc, val);
                    out.set_by_indices(&full_indices, acc.clone())?;
                }
            }
            Ok(out)
        }
    }
}
