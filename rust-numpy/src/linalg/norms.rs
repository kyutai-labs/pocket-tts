use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;
use num_traits::{Float, One, ToPrimitive, Zero};

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

/// Compute matrix or vector norm.
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
/// * `axis` - Axis along which to compute the norm (None for entire array)
/// * `keepdims` - If true, keep the reduced dimensions
///
/// # Returns
///
/// * `Result<Array<T>, NumPyError>` - The norm value(s)
pub fn norm<T>(
    x: &Array<T>,
    ord: Option<&str>,
    axis: Option<usize>,
    _keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    // Handle axis parameter (simplified: only None supported for now)
    if axis.is_some() {
        return Err(NumPyError::not_implemented(
            "norm with axis parameter not yet implemented",
        ));
    }

    // Determine norm type
    let norm_type = match ord {
        None | Some("fro") => NormType::Frobenius,
        Some("nuc") => NormType::Nuclear,
        Some("1") => NormType::L1,
        Some("2") => NormType::L2,
        Some(s) => {
            // Try to parse as integer for Lp norm
            if let Ok(p) = s.parse::<u32>() {
                if p > 0 {
                    NormType::Lp(p)
                } else {
                    return Err(NumPyError::value_error(
                        "ord must be positive for Lp norms",
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

    match norm_type {
        NormType::Nuclear => compute_nuclear_norm(x),
        NormType::Frobenius => compute_frobenius_norm(x),
        NormType::L1 => compute_lp_norm(x, 1),
        NormType::L2 => compute_lp_norm(x, 2),
        NormType::Lp(p) => compute_lp_norm(x, p),
    }
}

/// Norm type enumeration
enum NormType {
    Nuclear,
    Frobenius,
    L1,
    L2,
    Lp(u32),
}

/// Compute nuclear norm (sum of singular values)
///
/// Note: This is a simplified implementation. For a full implementation,
/// we need to compute all singular values using SVD and sum them.
fn compute_nuclear_norm<T>(x: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    // For now, return Frobenius norm as an approximation
    // TODO: Implement proper SVD to compute singular values
    compute_frobenius_norm(x)
}

/// Compute Frobenius norm (sqrt of sum of squared absolute values)
fn compute_frobenius_norm<T>(x: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let mut sum_sq = T::Real::zero();

    for i in 0..x.size() {
        if let Some(val) = x.get_linear(i) {
            let abs_val = LinalgScalar::abs(*val);
            sum_sq = sum_sq + abs_val * abs_val;
        }
    }

    let result = Float::sqrt(sum_sq);
    Ok(Array::from_vec(vec![T::from(result).unwrap()]))
}

/// Compute Lp norm: (sum(|x|^p))^(1/p)
fn compute_lp_norm<T>(x: &Array<T>, p: u32) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if p == 0 {
        return Err(NumPyError::value_error(
            "p must be positive for Lp norms",
            "linalg",
        ));
    }

    let mut sum_abs_p = T::Real::zero();

    for i in 0..x.size() {
        if let Some(val) = x.get_linear(i) {
            let abs_val = LinalgScalar::abs(*val);
            sum_abs_p = sum_abs_p + Float::powi(abs_val, p as i32);
        }
    }

    // Compute Lp norm: (sum(|x|^p))^(1/p)
    let result = if p == 1 {
        T::from(sum_abs_p).unwrap()
    } else if p == 2 {
        // For L2 norm, use sqrt for better precision
        T::from(Float::sqrt(sum_abs_p)).unwrap()
    } else {
        // For other p values, use exp(ln(x)/p) to compute pth root
        let log_sum = Float::ln(sum_abs_p);
        // Convert p to the Real type
        let inv_p = T::Real::one() / num_traits::cast(p as f64).unwrap();
        let root = Float::exp(log_sum * inv_p);
        T::from(root).unwrap()
    };

    Ok(Array::from_vec(vec![result]))
}
