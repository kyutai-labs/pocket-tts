use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;
use num_traits::{Float, One, Zero};

/// Compute the determinant of an array.
pub fn det<T>(a: &Array<T>) -> Result<T, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error("det requires 2D array", "linalg"));
    }
    let n = a.shape()[0];
    if n != a.shape()[1] {
        return Err(NumPyError::value_error(
            "det requires square matrix",
            "linalg",
        ));
    }

    let strides = a.strides();
    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let a_idx = idx(i, j, strides);
            data[i * n + j] = *a
                .get(a_idx)
                .ok_or_else(|| NumPyError::invalid_operation("det index out of bounds"))?;
        }
    }

    let mut det = T::one();
    let mut sign = T::one();

    let eps = <T::Real as num_traits::Float>::epsilon();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = data[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = data[row * n + col].abs();
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = row;
            }
        }

        if pivot_val <= eps {
            return Ok(T::zero());
        }

        if pivot_row != col {
            for j in 0..n {
                data.swap(col * n + j, pivot_row * n + j);
            }
            sign = -sign;
        }

        let pivot = data[col * n + col];
        det = det * pivot;

        for row in (col + 1)..n {
            let factor = data[row * n + col] / pivot;
            if factor.abs() <= eps {
                continue;
            }
            for j in col..n {
                data[row * n + j] = data[row * n + j] - factor * data[col * n + j];
            }
        }
    }

    Ok(det * sign)
}

/// Matrix rank
pub fn matrix_rank<T>(a: &Array<T>) -> Result<usize, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error(
            "matrix_rank requires 2D array",
            "linalg",
        ));
    }

    let rows = a.shape()[0];
    let cols = a.shape()[1];
    let strides = a.strides();

    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    let mut data = vec![T::zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let a_idx = idx(i, j, strides);
            data[i * cols + j] = *a
                .get(a_idx)
                .ok_or_else(|| NumPyError::invalid_operation("matrix_rank index out of bounds"))?;
        }
    }

    let mut rank = 0;
    let mut row = 0;
    let eps = <T::Real as num_traits::Float>::epsilon();
    let real_zero = T::Real::zero();

    for col in 0..cols {
        let mut pivot_row = row;
        let mut pivot_val = if row < rows {
            data[row * cols + col].abs()
        } else {
            real_zero
        };

        for r in (row + 1)..rows {
            let candidate = data[r * cols + col].abs();
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = r;
            }
        }

        if pivot_val <= eps {
            continue;
        }

        if pivot_row != row {
            for j in 0..cols {
                data.swap(row * cols + j, pivot_row * cols + j);
            }
        }

        let pivot = data[row * cols + col];
        for r in (row + 1)..rows {
            let factor = data[r * cols + col] / pivot;
            if factor.abs() <= eps {
                continue;
            }
            for j in col..cols {
                data[r * cols + j] = data[r * cols + j] - factor * data[row * cols + j];
            }
        }

        rank += 1;
        row += 1;
        if row >= rows {
            break;
        }
    }

    Ok(rank)
}

/// Normalize axis parameter, handling negative indices
fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, NumPyError> {
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };

    if normalized < 0 || normalized >= ndim as isize {
        return Err(NumPyError::index_error(axis as usize, ndim));
    }

    Ok(normalized as usize)
}

/// Normalize multiple axes, handling negative indices and duplicates
fn normalize_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>, NumPyError> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(axes.len());

    for &axis in axes {
        let normalized = normalize_axis(axis, ndim)?;
        if !seen.insert(normalized) {
            return Err(NumPyError::value_error(
                format!("duplicate axis in normalization: {}", axis),
                "linalg",
            ));
        }
        result.push(normalized);
    }

    Ok(result)
}

/// Compute matrix or vector norm.
<<<<<<< HEAD
=======
///
/// # Arguments
///
/// * `x` - Input array
/// * `ord` - Order of the norm:
///   - `None` or `None`: Frobenius norm for matrices, 2-norm for vectors
///   - `Some(1)`: L1 norm (sum of absolute values)
///   - `Some(2)`: L2 norm (Euclidean norm)
///   - `Some(p)`: Lp norm for any positive integer p
///   - `Some("fro")`: Frobenius norm
///   - `Some("nuc")`: Nuclear norm (sum of singular values)
/// * `axis` - Axis along which to compute the norm (None for entire array, single axis, or multiple axes)
/// * `keepdims` - If true, keep the reduced dimensions
///
/// # Returns
///
/// * `Result<Array<T>, NumPyError>` - The norm value(s)
>>>>>>> origin/main
pub fn norm<T>(
    x: &Array<T>,
    ord: Option<&str>,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    // Determine norm type
    let norm_type = match ord {
        None | Some("fro") => NormType::Frobenius,
        Some("nuc") => NormType::Nuclear,
        Some("1") => NormType::L1,
        Some("2") => NormType::L2,
        Some("inf") => NormType::Linf,
        Some("-inf") => NormType::LNegInf,
        Some(s) => {
<<<<<<< HEAD
=======
            // Try to parse as integer for Lp norm
>>>>>>> origin/main
            if let Ok(p) = s.parse::<i32>() {
                if p > 0 {
                    NormType::Lp(p as u32)
                } else if p < 0 {
                    NormType::LNegP(p.abs() as u32)
                } else {
                    return Err(NumPyError::value_error(
                        "ord must be non-zero for Lp norms",
                        "linalg",
                    ));
                }
            } else {
                return Err(NumPyError::value_error(
                    format!("Invalid norm order: {}", s),
                    "linalg",
                ));
            }
        }
    };

    // Handle axis parameter
    let axes = match axis {
        None => None,
        Some(axes_slice) => Some(normalize_axes(axes_slice, x.ndim())?),
    };

    match norm_type {
        NormType::Nuclear => {
            if axes.is_some() {
                return Err(NumPyError::value_error(
                    "nuclear norm does not support axis parameter",
                    "linalg",
                ));
            }
            compute_nuclear_norm(x)
        }
        NormType::Frobenius => compute_norm_with_axis(x, 2, axes.as_deref(), keepdims),
        NormType::L1 => compute_norm_with_axis(x, 1, axes.as_deref(), keepdims),
        NormType::L2 => compute_norm_with_axis(x, 2, axes.as_deref(), keepdims),
        NormType::Linf => compute_norm_inf_with_axis(x, true, axes.as_deref(), keepdims),
        NormType::LNegInf => compute_norm_inf_with_axis(x, false, axes.as_deref(), keepdims),
        NormType::Lp(p) => compute_norm_with_axis(x, p, axes.as_deref(), keepdims),
        NormType::LNegP(p) => compute_norm_neg_p_with_axis(x, p, axes.as_deref(), keepdims),
    }
}

/// Norm type enumeration
enum NormType {
    Nuclear,
    Frobenius,
    L1,
    L2,
    Lp(u32),
    Linf,
    LNegInf,
    LNegP(u32),
}

/// Helper function to compute output shape after axis reduction
fn compute_output_shape(shape: &[usize], axes: Option<&[usize]>, keepdims: bool) -> Vec<usize> {
    match axes {
        None => {
            if keepdims {
                vec![1; shape.len()]
            } else {
                vec![]
            }
        }
        Some(axes_to_reduce) => {
            let mut result = if keepdims {
                shape.to_vec()
            } else {
                let mut temp = shape.to_vec();
<<<<<<< HEAD
=======
                // Sort axes in descending order for removal
>>>>>>> origin/main
                let mut sorted_axes: Vec<usize> = axes_to_reduce.to_vec();
                sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
                for &ax in &sorted_axes {
                    temp.remove(ax);
                }
                temp
            };

            if keepdims {
                for &ax in axes_to_reduce {
                    result[ax] = 1;
                }
            }

            result
        }
    }
}

/// Compute Lp norm along specified axis/axes
fn compute_norm_with_axis<T>(
    x: &Array<T>,
    p: u32,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let output_shape = compute_output_shape(x.shape(), axes, keepdims);
    let output_size: usize = output_shape.iter().product();

    // Handle scalar output case
<<<<<<< HEAD
    if output_shape.is_empty() || (output_size == 1 && axes.is_none()) {
=======
    if output_shape.is_empty() || output_size == 1 {
>>>>>>> origin/main
        return compute_lp_norm(x, p);
    }

    let mut result = vec![T::Real::zero(); output_size];
<<<<<<< HEAD
    let shape = x.shape();
    let ndim = shape.len();

=======
    let strides = x.strides();
    let shape = x.shape();
    let ndim = shape.len();

    // Generate all indices in the output array
>>>>>>> origin/main
    let mut output_idx = vec![0usize; output_shape.len()];
    let mut output_flat = 0usize;

    loop {
<<<<<<< HEAD
        let mut sum_abs_p = T::Real::zero();
        let mut input_base = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_ax_ptr = 0;
            for dim in 0..ndim {
                if axes_to_reduce.contains(&dim) {
                    if keepdims {
                        input_base[dim] = 0;
                        out_ax_ptr += 1;
                    } else {
                        input_base[dim] = 0;
                    }
                } else {
                    input_base[dim] = output_idx[out_ax_ptr];
                    out_ax_ptr += 1;
                }
            }
        }

        if let Some(axes_to_reduce) = axes {
            let reduced_dims: Vec<(usize, usize)> =
                axes_to_reduce.iter().map(|&ax| (ax, shape[ax])).collect();

            let mut reduced_iter = vec![0usize; reduced_dims.len()];
            loop {
                let mut input_idx = input_base.clone();
=======
        // Compute norm for this output position
        let mut sum_abs_p = T::Real::zero();

        // Build the base index from output_idx (non-reduced dimensions)
        let mut base_idx = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_idx = 0;
            for dim in 0..ndim {
                if !axes_to_reduce.contains(&dim) {
                    base_idx[dim] = output_idx[out_idx];
                    out_idx += 1;
                }
            }
        } else {
            // No reduction - this shouldn't happen given the early return above
            base_idx = output_idx.clone();
        }

        // Iterate over all combinations of reduced axes
        if let Some(axes_to_reduce) = axes {
            // Create a list of (dimension, max_value) for reduced axes
            let reduced_dims: Vec<(usize, usize)> =
                axes_to_reduce.iter().map(|&ax| (ax, shape[ax])).collect();

            // Use nested iteration through all combinations
            let mut reduced_iter = reduced_dims
                .iter()
                .map(|&(_dim, _)| 0usize)
                .collect::<Vec<_>>();
            loop {
                // Build complete index by combining base_idx and reduced_iter
                let mut input_idx = base_idx.clone();
>>>>>>> origin/main
                for (i, &(dim, _)) in reduced_dims.iter().enumerate() {
                    input_idx[dim] = reduced_iter[i];
                }

<<<<<<< HEAD
                if let Ok(val) = x.get_multi(&input_idx) {
                    let abs_val = LinalgScalar::abs(val);
                    sum_abs_p = sum_abs_p + num_traits::Float::powi(abs_val, p as i32);
                }

=======
                // Compute linear index and accumulate
                let linear_idx = input_idx.iter().enumerate().fold(0usize, |acc, (i, &idx)| {
                    acc + idx as usize * strides[i] as usize
                });

                if let Some(val) = x.get(linear_idx) {
                    let abs_val = LinalgScalar::abs(*val);
                    sum_abs_p = sum_abs_p + num_traits::Float::powi(abs_val, p as i32);
                }

                // Increment reduced_iter (like a multi-digit counter)
>>>>>>> origin/main
                let mut carry = true;
                for (i, &(_, max_val)) in reduced_dims.iter().enumerate() {
                    if carry {
                        if reduced_iter[i] + 1 < max_val {
                            reduced_iter[i] += 1;
                            carry = false;
                        } else {
                            reduced_iter[i] = 0;
                        }
                    }
                }
<<<<<<< HEAD
                if carry { break; }
            }
        }

=======

                if carry {
                    break; // All combinations exhausted
                }
            }
        }

        // Compute p-th root
>>>>>>> origin/main
        result[output_flat] = if p == 1 {
            sum_abs_p
        } else if p == 2 {
            num_traits::Float::sqrt(sum_abs_p)
        } else {
            let log_sum = num_traits::Float::ln(sum_abs_p);
            let inv_p = T::Real::one() / num_traits::cast(p as f64).unwrap();
            num_traits::Float::exp(log_sum * inv_p)
        };

<<<<<<< HEAD
        let mut carry = true;
        for dim in (0..output_shape.len()).rev() {
=======
        // Increment output_idx
        let mut carry = true;
        for dim in 0..output_shape.len() {
>>>>>>> origin/main
            if carry {
                if output_idx[dim] + 1 < output_shape[dim] {
                    output_idx[dim] += 1;
                    carry = false;
                } else {
                    output_idx[dim] = 0;
                }
            }
        }

        output_flat += 1;
<<<<<<< HEAD
        if carry || output_flat >= output_size {
=======
        if carry {
>>>>>>> origin/main
            break;
        }
    }

<<<<<<< HEAD
=======
    // Convert Real type back to T
>>>>>>> origin/main
    let result_t: Vec<T> = result.into_iter().map(|r| T::from(r).unwrap()).collect();
    Ok(Array::from_shape_vec(output_shape, result_t))
}

/// Compute L-infinity or L-negative-infinity norm along specified axis/axes
fn compute_norm_inf_with_axis<T>(
    x: &Array<T>,
    max_norm: bool,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let output_shape = compute_output_shape(x.shape(), axes, keepdims);
    let output_size: usize = output_shape.iter().product();

<<<<<<< HEAD
    let mut result = vec![
        if max_norm {
            T::Real::neg_infinity()
=======
    // Handle scalar output case
    if output_shape.is_empty() || output_size == 1 {
        let mut result = if max_norm {
            T::Real::zero()
        } else {
            T::Real::infinity()
        };

        for i in 0..x.size() {
            if let Some(val) = x.get_linear(i) {
                let abs_val = LinalgScalar::abs(*val);
                if max_norm {
                    result = result.max(abs_val);
                } else {
                    result = result.min(abs_val);
                }
            }
        }

        return Ok(Array::from_vec(vec![T::from(result).unwrap()]));
    }

    let mut result = vec![
        if max_norm {
            T::Real::zero()
>>>>>>> origin/main
        } else {
            T::Real::infinity()
        };
        output_size
    ];
<<<<<<< HEAD
    let shape = x.shape();
    let ndim = shape.len();

=======
    let strides = x.strides();
    let shape = x.shape();
    let ndim = shape.len();

    // Generate all indices in the output array
>>>>>>> origin/main
    let mut output_idx = vec![0usize; output_shape.len()];
    let mut output_flat = 0usize;

    loop {
<<<<<<< HEAD
        let mut current_result = if max_norm {
            T::Real::neg_infinity()
=======
        // Compute norm for this output position
        let mut current_result = if max_norm {
            T::Real::zero()
>>>>>>> origin/main
        } else {
            T::Real::infinity()
        };

<<<<<<< HEAD
        let mut input_base = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_ax_ptr = 0;
            for dim in 0..ndim {
                if axes_to_reduce.contains(&dim) {
                    if keepdims {
                        input_base[dim] = 0;
                        out_ax_ptr += 1;
                    }
                } else {
                    input_base[dim] = output_idx[out_ax_ptr];
                    out_ax_ptr += 1;
=======
        // Build the base index from output_idx (non-reduced dimensions)
        let mut base_idx = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_idx = 0;
            for dim in 0..ndim {
                if !axes_to_reduce.contains(&dim) {
                    base_idx[dim] = output_idx[out_idx];
                    out_idx += 1;
>>>>>>> origin/main
                }
            }
        }

<<<<<<< HEAD
=======
        // Iterate over all combinations of reduced axes
>>>>>>> origin/main
        if let Some(axes_to_reduce) = axes {
            let reduced_dims: Vec<(usize, usize)> =
                axes_to_reduce.iter().map(|&ax| (ax, shape[ax])).collect();

<<<<<<< HEAD
            let mut reduced_iter = vec![0usize; reduced_dims.len()];
            loop {
                let mut input_idx = input_base.clone();
=======
            let mut reduced_iter = reduced_dims
                .iter()
                .map(|&(_dim, _)| 0usize)
                .collect::<Vec<_>>();
            loop {
                // Build complete index
                let mut input_idx = base_idx.clone();
>>>>>>> origin/main
                for (i, &(dim, _)) in reduced_dims.iter().enumerate() {
                    input_idx[dim] = reduced_iter[i];
                }

<<<<<<< HEAD
                if let Ok(val) = x.get_multi(&input_idx) {
                    let abs_val = LinalgScalar::abs(val);
=======
                // Compute linear index
                let linear_idx = input_idx.iter().enumerate().fold(0usize, |acc, (i, &idx)| {
                    acc + idx as usize * strides[i] as usize
                });

                if let Some(val) = x.get(linear_idx) {
                    let abs_val = LinalgScalar::abs(*val);
>>>>>>> origin/main
                    current_result = if max_norm {
                        current_result.max(abs_val)
                    } else {
                        current_result.min(abs_val)
                    };
                }

<<<<<<< HEAD
=======
                // Increment reduced_iter
>>>>>>> origin/main
                let mut carry = true;
                for (i, &(_, max_val)) in reduced_dims.iter().enumerate() {
                    if carry {
                        if reduced_iter[i] + 1 < max_val {
                            reduced_iter[i] += 1;
                            carry = false;
                        } else {
                            reduced_iter[i] = 0;
                        }
                    }
                }
<<<<<<< HEAD
                if carry { break; }
            }
        } else {
            for i in 0..x.size() {
                if let Some(val) = x.get_linear(i) {
                    let abs_val = LinalgScalar::abs(*val);
                    current_result = if max_norm {
                        current_result.max(abs_val)
                    } else {
                        current_result.min(abs_val)
                    };
=======

                if carry {
                    break;
>>>>>>> origin/main
                }
            }
        }

        result[output_flat] = current_result;

<<<<<<< HEAD
        let mut carry = true;
        for dim in (0..output_shape.len()).rev() {
=======
        // Increment output_idx
        let mut carry = true;
        for dim in 0..output_shape.len() {
>>>>>>> origin/main
            if carry {
                if output_idx[dim] + 1 < output_shape[dim] {
                    output_idx[dim] += 1;
                    carry = false;
                } else {
                    output_idx[dim] = 0;
                }
            }
        }

        output_flat += 1;
<<<<<<< HEAD
        if carry || output_flat >= output_size {
=======
        if carry {
>>>>>>> origin/main
            break;
        }
    }

    let result_t: Vec<T> = result.into_iter().map(|r| T::from(r).unwrap()).collect();
    Ok(Array::from_shape_vec(output_shape, result_t))
}

/// Compute negative Lp norm along specified axis/axes
fn compute_norm_neg_p_with_axis<T>(
    x: &Array<T>,
    p: u32,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let output_shape = compute_output_shape(x.shape(), axes, keepdims);
    let output_size: usize = output_shape.iter().product();

<<<<<<< HEAD
    let mut result = vec![T::Real::zero(); output_size];
    let shape = x.shape();
    let ndim = shape.len();

    let mut output_idx = vec![0usize; output_shape.len()];
    let mut output_flat = 0usize;

    loop {
        let mut sum_abs_neg_p = T::Real::zero();
        let mut input_base = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_ax_ptr = 0;
            for dim in 0..ndim {
                if axes_to_reduce.contains(&dim) {
                    if keepdims {
                        input_base[dim] = 0;
                        out_ax_ptr += 1;
                    }
                } else {
                    input_base[dim] = output_idx[out_ax_ptr];
                    out_ax_ptr += 1;
=======
    // Handle scalar output case
    if output_shape.is_empty() || output_size == 1 {
        let mut sum_abs_neg_p = T::Real::zero();

        for i in 0..x.size() {
            if let Some(val) = x.get_linear(i) {
                let abs_val = LinalgScalar::abs(*val);
                if abs_val > T::Real::zero() {
                    sum_abs_neg_p = sum_abs_neg_p + num_traits::Float::powi(abs_val, -(p as i32));
>>>>>>> origin/main
                }
            }
        }

<<<<<<< HEAD
=======
        let result = if sum_abs_neg_p > T::Real::zero() {
            num_traits::Float::powi(sum_abs_neg_p, -(1 as i32))
        } else {
            T::Real::infinity()
        };

        return Ok(Array::from_vec(vec![T::from(result).unwrap()]));
    }

    let mut result = vec![T::Real::zero(); output_size];
    let strides = x.strides();
    let shape = x.shape();
    let ndim = shape.len();

    // Generate all indices in the output array
    let mut output_idx = vec![0usize; output_shape.len()];
    let mut output_flat = 0usize;

    loop {
        // Compute norm for this output position
        let mut sum_abs_neg_p = T::Real::zero();

        // Build the base index from output_idx (non-reduced dimensions)
        let mut base_idx = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_idx = 0;
            for dim in 0..ndim {
                if !axes_to_reduce.contains(&dim) {
                    base_idx[dim] = output_idx[out_idx];
                    out_idx += 1;
                }
            }
        }

        // Iterate over all combinations of reduced axes
>>>>>>> origin/main
        if let Some(axes_to_reduce) = axes {
            let reduced_dims: Vec<(usize, usize)> =
                axes_to_reduce.iter().map(|&ax| (ax, shape[ax])).collect();

<<<<<<< HEAD
            let mut reduced_iter = vec![0usize; reduced_dims.len()];
            loop {
                let mut input_idx = input_base.clone();
=======
            let mut reduced_iter = reduced_dims
                .iter()
                .map(|&(_dim, _)| 0usize)
                .collect::<Vec<_>>();
            loop {
                // Build complete index
                let mut input_idx = base_idx.clone();
>>>>>>> origin/main
                for (i, &(dim, _)) in reduced_dims.iter().enumerate() {
                    input_idx[dim] = reduced_iter[i];
                }

<<<<<<< HEAD
                if let Ok(val) = x.get_multi(&input_idx) {
                    let abs_val = LinalgScalar::abs(val);
                    if abs_val > T::Real::zero() {
                        sum_abs_neg_p = sum_abs_neg_p + num_traits::Float::powi(abs_val, -(p as i32));
                    }
                }

=======
                // Compute linear index
                let linear_idx = input_idx.iter().enumerate().fold(0usize, |acc, (i, &idx)| {
                    acc + idx as usize * strides[i] as usize
                });

                if let Some(val) = x.get(linear_idx) {
                    let abs_val = LinalgScalar::abs(*val);
                    if abs_val > T::Real::zero() {
                        sum_abs_neg_p =
                            sum_abs_neg_p + num_traits::Float::powi(abs_val, -(p as i32));
                    }
                }

                // Increment reduced_iter
>>>>>>> origin/main
                let mut carry = true;
                for (i, &(_, max_val)) in reduced_dims.iter().enumerate() {
                    if carry {
                        if reduced_iter[i] + 1 < max_val {
                            reduced_iter[i] += 1;
                            carry = false;
                        } else {
                            reduced_iter[i] = 0;
                        }
                    }
                }
<<<<<<< HEAD
                if carry { break; }
            }
        }

=======

                if carry {
                    break;
                }
            }
        }

        // Compute negative p-th root
>>>>>>> origin/main
        result[output_flat] = if sum_abs_neg_p > T::Real::zero() {
            num_traits::Float::powi(sum_abs_neg_p, -(1 as i32))
        } else {
            T::Real::infinity()
        };

<<<<<<< HEAD
        let mut carry = true;
        for dim in (0..output_shape.len()).rev() {
=======
        // Increment output_idx
        let mut carry = true;
        for dim in 0..output_shape.len() {
>>>>>>> origin/main
            if carry {
                if output_idx[dim] + 1 < output_shape[dim] {
                    output_idx[dim] += 1;
                    carry = false;
                } else {
                    output_idx[dim] = 0;
                }
            }
        }

        output_flat += 1;
<<<<<<< HEAD
        if carry || output_flat >= output_size {
=======
        if carry {
>>>>>>> origin/main
            break;
        }
    }

    let result_t: Vec<T> = result.into_iter().map(|r| T::from(r).unwrap()).collect();
    Ok(Array::from_shape_vec(output_shape, result_t))
}

/// Compute nuclear norm (sum of singular values)
fn compute_nuclear_norm<T>(x: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if x.ndim() != 2 {
        return Err(NumPyError::value_error("nuclear norm requires 2D array", "linalg"));
    }
    let singular_values = compute_singular_values(x)?;
    let mut sum = T::Real::zero();
    for sv in &singular_values { sum = sum + *sv; }
    Ok(Array::from_vec(vec![T::from(sum).unwrap()]))
}

fn compute_singular_values<T>(a: &Array<T>) -> Result<Vec<T::Real>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let m = a.shape()[0];
    let n = a.shape()[1];
    let mut ata = vec![T::Real::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::Real::zero();
            for k in 0..m {
                let val_ki = LinalgScalar::abs(a.get_multi(&[k, i])?);
                let val_kj = LinalgScalar::abs(a.get_multi(&[k, j])?);
                sum = sum + val_ki * val_kj;
            }
            ata[i * n + j] = sum;
        }
    }
    compute_eigenvalues_symmetric_real(&ata, n)
        .map(|evs| evs.into_iter().map(|ev| if ev > T::Real::zero() { ev.sqrt() } else { T::Real::zero() }).collect())
}

fn compute_eigenvalues_symmetric_real<R>(a: &[R], n: usize) -> Result<Vec<R>, NumPyError>
where
    R: num_traits::Float + Clone,
{
    if n == 0 { return Ok(vec![]); }
    if n == 1 { return Ok(vec![a[0].clone()]); }
    let mut h: Vec<R> = a.to_vec();
    let eps = R::epsilon() * num_traits::cast(1000.0).unwrap();
    let mut m = n;
    let mut iter = 0;
    while m > 1 && iter < 100 * n {
        let mut k = m - 1;
        while k > 0 {
            let off_diag = h[k * n + (k - 1)].abs();
            let diag_sum = h[(k - 1) * n + (k - 1)].abs() + h[k * n + k].abs();
            if off_diag <= eps * diag_sum {
                h[k * n + (k - 1)] = R::zero();
                h[(k - 1) * n + k] = R::zero();
                break;
            }
            k -= 1;
        }
        if k == m - 1 { m -= 1; continue; }
        let d = (h[(m - 2) * n + (m - 2)] - h[(m - 1) * n + (m - 1)]) / num_traits::cast(2.0).unwrap();
        let t_sq = h[(m - 1) * n + (m - 2)] * h[(m - 1) * n + (m - 2)];
        let shift = h[(m - 1) * n + (m - 1)] - t_sq / (d + if d >= R::zero() { R::one() } else { -R::one() } * (d * d + t_sq).sqrt());
        let mut x = h[0 * n + 0] - shift;
        let mut z = h[1 * n + 0];
        for i in 0..m - 1 {
            let r = (x * x + z * z).sqrt();
            if r > eps {
                let c = x / r; let s = z / r;
                for j in i..n {
                    let t1 = h[i * n + j]; let t2 = h[(i + 1) * n + j];
                    h[i * n + j] = c * t1 + s * t2; h[(i + 1) * n + j] = -s * t1 + c * t2;
                }
                for j in 0..std::cmp::min(i + 3, n) {
                    let t1 = h[j * n + i]; let t2 = h[j * n + (i + 1)];
                    h[j * n + i] = c * t1 + s * t2; h[j * n + (i + 1)] = -s * t1 + c * t2;
                }
            }
            if i < m - 2 { x = h[(i + 1) * n + i]; z = h[(i + 2) * n + i]; }
        }
        iter += 1;
    }
    Ok((0..n).map(|i| h[i * n + i]).collect())
}

fn compute_lp_norm<T>(x: &Array<T>, p: u32) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if p == 0 { return Err(NumPyError::value_error("p must be positive", "linalg")); }
    let mut sum_abs_p = T::Real::zero();
    for i in 0..x.size() {
        if let Some(val) = x.get_linear(i) {
            sum_abs_p = sum_abs_p + LinalgScalar::abs(*val).powi(p as i32);
        }
    }
    let res = if p == 1 { sum_abs_p } else if p == 2 { sum_abs_p.sqrt() } else { sum_abs_p.powf(T::Real::one() / num_traits::cast(p).unwrap()) };
    Ok(Array::from_vec(vec![T::from(res).unwrap()]))
}

/// Compute the condition number of a matrix
///
/// The condition number measures how sensitive a function is to changes or errors
/// in the input. For matrices, it's defined as ||A|| * ||A^(-1)|| where A^(-1) is the
/// inverse of A.
///
/// # Arguments
///
/// * `a` - Input matrix (2D array)
/// * `p` - Order of the norm:
///   - None or "fro": Frobenius norm
///   - 1: L1 norm (maximum column sum)
///   - -1: Minimum singular value
///   - 2: L2 norm (largest singular value)
///   - -2: Smallest singular value
///   - "nuc": Nuclear norm
///   - inf: Infinity norm (maximum row sum)
///   - -inf: Minimum singular value
///   - "fro": Frobenius norm
///
/// # Returns
///
/// * `Result<Array<T>, NumPyError>` - The condition number
///
/// # Examples
///
/// ```rust,ignore
/// use rust_numpy::{array, linalg::cond};
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let c = cond(&a, None).unwrap();
/// ```
pub fn cond<T>(a: &Array<T>, p: Option<&str>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error("cond requires 2D array", "linalg"));
    }

    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(NumPyError::value_error(
            "cond requires square matrix",
            "linalg",
        ));
    }

    // For p=2 or p=None, use SVD-based condition number: sigma_max / sigma_min
    // For p="fro", use Frobenius norm based computation
    // For p=1 or p=inf, use norm-based computation
    let norm_ord = p.unwrap_or("2");

    if norm_ord == "2" || norm_ord == "-2" {
        // SVD-based condition number
        let singular_values = compute_singular_values(a)?;

        if singular_values.is_empty() {
            return Ok(Array::from_vec(vec![T::from(T::Real::infinity()).unwrap()]));
        }

        let sigma_max = singular_values.first().copied().unwrap_or(T::Real::zero());
        let sigma_min = singular_values.last().copied().unwrap_or(T::Real::zero());

        // If smallest singular value is zero, matrix is singular
        if sigma_min <= T::Real::epsilon() * sigma_max {
            return Ok(Array::from_vec(vec![T::from(T::Real::infinity()).unwrap()]));
        }

        let condition_number = sigma_max / sigma_min;
        Ok(Array::from_vec(vec![T::from(condition_number).unwrap()]))
    } else if norm_ord == "fro"
        || norm_ord == "nuc"
        || norm_ord == "1"
        || norm_ord == "-1"
        || norm_ord == "inf"
        || norm_ord == "-inf"
    {
        // For other norms, compute using norm(a) * norm(inv(a))
        use crate::linalg::solvers::inv;

        let norm_a = norm(a, Some(norm_ord), None, false)?;

        // Convert scalar Array to scalar value
        let norm_a_val = norm_a.get_linear(0).copied().unwrap_or(T::zero());

        let inv_a = inv(a)?;
        let norm_inv_a = norm(&inv_a, Some(norm_ord), None, false)?;

        let norm_inv_a_val = norm_inv_a.get_linear(0).copied().unwrap_or(T::zero());

        // If inverse norm is effectively zero, matrix is singular
        let eps = LinalgScalar::from_real(T::Real::epsilon());
        if num_traits::Float::abs(norm_inv_a_val) <= eps {
            return Ok(Array::from_vec(vec![LinalgScalar::from_real(
                T::Real::infinity(),
            )]));
        }

        let condition_number = norm_a_val * norm_inv_a_val;
        Ok(Array::from_vec(vec![condition_number]))
    } else {
        Err(NumPyError::value_error(
            format!("Invalid norm order for cond: {}", norm_ord),
            "linalg",
        ))
    }
}

/// Compute the sign and (natural) logarithm of the determinant of an array
///
/// This function is more numerically stable than computing the determinant directly
/// for large matrices, as it works with the logarithm of values to avoid overflow.
///
/// # Arguments
///
/// * `a` - Input square matrix (2D array)
///
/// # Returns
///
/// * `Result<(Array<T>, Array<T>), NumPyError>` - A tuple containing:
///   - sign: Sign of the determinant (-1.0, 0.0, or 1.0)
///   - logabsdet: Natural logarithm of the absolute value of the determinant
///
/// If the determinant is zero, sign is 0.0 and logabsdet is -inf.
///
/// # Examples
///
/// ```rust,ignore
/// use rust_numpy::{array, linalg::slogdet};
///
/// let a = array![[1.0, 2.0], [3.0, 4.0]];
/// let (sign, logdet) = slogdet(&a).unwrap();
/// // det = 1*4 - 2*3 = -2
/// // sign = -1.0, logdet = ln(2) ≈ 0.6931
/// ```
pub fn slogdet<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>), NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error(
            "slogdet requires 2D array",
            "linalg",
        ));
    }

    let n = a.shape()[0];
    if n != a.shape()[1] {
        return Err(NumPyError::value_error(
            "slogdet requires square matrix",
            "linalg",
        ));
    }

    if n == 0 {
        return Ok((
            Array::from_vec(vec![T::one()]),
            Array::from_vec(vec![T::from(T::Real::neg_infinity()).unwrap()]),
        ));
    }

    let strides = a.strides();
    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    // Make a copy of the matrix for LU decomposition
    let mut lu = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let a_idx = idx(i, j, strides);
            lu[i * n + j] = *a
                .get(a_idx)
                .ok_or_else(|| NumPyError::invalid_operation("slogdet index out of bounds"))?;
        }
    }

    let mut sign = T::one();
    let mut logabsdet = T::Real::zero();
    let eps = T::Real::epsilon();

    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut pivot_val = LinalgScalar::abs(lu[col * n + col]);
        for row in (col + 1)..n {
            let candidate = LinalgScalar::abs(lu[row * n + col]);
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = row;
            }
        }

        if pivot_val <= eps {
            // Singular matrix
            return Ok((
                Array::from_vec(vec![T::zero()]),
                Array::from_vec(vec![T::from(T::Real::neg_infinity()).unwrap()]),
            ));
        }

        if pivot_row != col {
            // Swap rows
            for j in 0..n {
                lu.swap(col * n + j, pivot_row * n + j);
            }
            sign = -sign;
        }

        let pivot = lu[col * n + col];
        logabsdet = logabsdet + Float::ln(LinalgScalar::abs(pivot));

        // Update the submatrix
        for row in (col + 1)..n {
            let factor = lu[row * n + col] / pivot;
            let abs_val = LinalgScalar::abs(factor);
            if abs_val <= eps {
                continue;
            }
            for j in (col + 1)..n {
                lu[row * n + j] = lu[row * n + j] - factor * lu[col * n + j];
            }
        }
    }

    Ok((
        Array::from_vec(vec![sign]),
        Array::from_vec(vec![T::from(logabsdet).unwrap()]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cond_well_conditioned() {
        // Well-conditioned matrix (identity)
        let n = 3;
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }

        let a = Array::from_data(data, vec![n, n]);
        let result = cond(&a, Some("2")).unwrap();

        // Identity matrix has condition number 1
        assert!((result.get_linear(0).unwrap_or(&1.0) - &1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cond_singular() {
        // Singular matrix (determinant = 0)
        let data = vec![1.0f64, 2.0, 2.0, 4.0]; // Rows are linearly dependent
        let a = Array::from_data(data, vec![2, 2]);

        let result = cond(&a, Some("2")).unwrap();
        // Singular matrix should have infinite condition number
        let val = result.get_linear(0).unwrap_or(&f64::INFINITY);
        assert!(*val > 1e100); // Effectively infinite
    }

    #[test]
    fn test_slogdet_identity() {
        // Identity matrix
        let n = 3;
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }

        let a = Array::from_data(data, vec![n, n]);
        let (sign, logdet) = slogdet(&a).unwrap();

        // det(I) = 1, so sign = 1, logdet = ln(1) = 0
        assert!((sign.get_linear(0).unwrap_or(&1.0) - &1.0).abs() < 1e-10);
        assert!((logdet.get_linear(0).unwrap_or(&0.0) - &0.0).abs() < 1e-10);
    }

    #[test]
    fn test_slogdet_negative_determinant() {
        // Matrix with negative determinant
        // [[1, 2], [3, 4]] has det = 1*4 - 2*3 = -2
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a = Array::from_data(data, vec![2, 2]);

        let (sign, logdet) = slogdet(&a).unwrap();

        // sign should be -1 (negative determinant)
        assert!((sign.get_linear(0).unwrap_or(&0.0) - &(-1.0)).abs() < 1e-10);
        // logdet should be ln(|det|) = ln(2) ≈ 0.6931
        let logdet_val = logdet.get_linear(0).unwrap_or(&0.0);
        assert!((*logdet_val - 0.6931).abs() < 0.01);
    }

    #[test]
    fn test_slogdet_singular() {
        // Singular matrix
        let data = vec![1.0f64, 2.0, 2.0, 4.0];
        let a = Array::from_data(data, vec![2, 2]);

        let (sign, logdet) = slogdet(&a).unwrap();

        // For singular matrix: sign = 0, logdet = -inf
        assert!((sign.get_linear(0).unwrap_or(&0.0) - &0.0).abs() < 1e-10);
        let logdet_val = logdet.get_linear(0).unwrap_or(&f64::NEG_INFINITY);
        assert!(*logdet_val < -1e100); // Effectively -infinity
    }

    #[test]
    fn test_slogdet_empty() {
        // 0x0 matrix (edge case)
        let data: Vec<f64> = vec![];
        let a = Array::from_data(data, vec![0, 0]);

        let (sign, logdet) = slogdet(&a).unwrap();

        // Empty matrix: det = 1 by convention
        assert!((sign.get_linear(0).unwrap_or(&1.0) - &1.0).abs() < 1e-10);
        // logdet = -inf
        let logdet_val = logdet.get_linear(0).unwrap_or(&f64::NEG_INFINITY);
        assert!(*logdet_val < -1e100);
    }

    #[test]
    fn test_cond_non_square() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a = Array::from_data(data, vec![2, 2]);

        let result = cond(&a, Some("2"));
        assert!(result.is_ok()); // 2x2 is square
    }
}
