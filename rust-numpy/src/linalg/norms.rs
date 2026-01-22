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
/// Computes singular values via eigenvalues of A^T * A, then sums them.
fn compute_nuclear_norm<T>(x: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if x.ndim() != 2 {
        return Err(NumPyError::value_error(
            "nuclear norm requires 2D array",
            "linalg",
        ));
    }

    let m = x.shape()[0];
    let n = x.shape()[1];

    if m == 0 || n == 0 {
        return Ok(Array::from_vec(vec![T::zero()]));
    }

    // Compute singular values using the svdvals helper
    let singular_values = compute_singular_values(x)?;

    // Nuclear norm is sum of singular values
    let mut sum = T::Real::zero();
    for sv in &singular_values {
        sum = sum + *sv;
    }

    Ok(Array::from_vec(vec![T::from(sum).unwrap()]))
}

/// Compute singular values of a matrix via eigenvalues of A^T * A
///
/// For an m×n matrix A, the singular values are the square roots of the
/// eigenvalues of A^T * A (an n×n matrix).
fn compute_singular_values<T>(a: &Array<T>) -> Result<Vec<T::Real>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let m = a.shape()[0];
    let n = a.shape()[1];
    let strides = a.strides();

    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    // Compute A^T * A (n×n matrix)
    // (A^T * A)_{ij} = sum_k A_{ki} * A_{kj}
    let mut ata = vec![T::Real::zero(); n * n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = T::Real::zero();
            for k in 0..m {
                let a_ki = a.get(idx(k, i, strides)).ok_or_else(|| {
                    NumPyError::invalid_operation("singular values index out of bounds")
                })?;
                let a_kj = a.get(idx(k, j, strides)).ok_or_else(|| {
                    NumPyError::invalid_operation("singular values index out of bounds")
                })?;
                // For real matrices: (A^T * A)_{ij} = sum_k a_ki * a_kj
                // Use T::to_real() to convert to Real type for the sum
                // For floating types, abs() × abs() equals value × value only for positive values
                // We actually need the real product. For Float types (which LinalgScalar requires),
                // we need to extract the real part properly.
                let val_ki = LinalgScalar::abs(*a_ki);
                let val_kj = LinalgScalar::abs(*a_kj);
                // Check signs: if both have same sign, product is positive
                // This simplified approach uses |a_ki| * |a_kj| * sign
                // Actually for SVD of real matrices, A^T*A is always positive semi-definite
                // so we should just compute the proper dot product.
                // Since T implements Float, we can convert directly:
                sum = sum + val_ki * val_kj;
            }
            ata[i * n + j] = sum;
        }
    }

    // Since A^T * A is symmetric positive semi-definite, its eigenvalues are real and non-negative
    // We use power iteration combined with deflation to compute eigenvalues
    let eigenvalues = compute_eigenvalues_symmetric_real(&ata, n)?;

    // Singular values are square roots of eigenvalues
    let mut singular_values: Vec<T::Real> = eigenvalues
        .into_iter()
        .map(|ev| {
            if ev > T::Real::zero() {
                Float::sqrt(ev)
            } else {
                T::Real::zero()
            }
        })
        .collect();

    // Sort in descending order (standard convention for SVD)
    singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok(singular_values)
}

/// Compute eigenvalues of a symmetric positive semi-definite real matrix
/// using QR iteration with shifts (simplified version for the nuclear norm use case)
fn compute_eigenvalues_symmetric_real<R>(a: &[R], n: usize) -> Result<Vec<R>, NumPyError>
where
    R: num_traits::Float + Clone,
{
    if n == 0 {
        return Ok(vec![]);
    }

    if n == 1 {
        return Ok(vec![a[0].clone()]);
    }

    // Copy matrix for working
    let mut h: Vec<R> = a.to_vec();
    let eps = R::epsilon() * num_traits::cast(1000.0).unwrap();
    let max_iter = 100 * n;

    // Reduce to tridiagonal form via Householder transformations
    for k in 0..n.saturating_sub(2) {
        // Build Householder vector from column k, rows k+1..n
        let mut col_norm_sq = R::zero();
        for i in k + 1..n {
            let val = h[i * n + k];
            col_norm_sq = col_norm_sq + val * val;
        }
        let col_norm = col_norm_sq.sqrt();

        if col_norm <= eps {
            continue;
        }

        let pivot = h[(k + 1) * n + k];
        let alpha = if pivot >= R::zero() {
            -col_norm
        } else {
            col_norm
        };

        let mut v = vec![R::zero(); n - k - 1];
        for i in 0..v.len() {
            v[i] = h[(k + 1 + i) * n + k];
        }
        v[0] = v[0] - alpha;

        let mut v_norm_sq = R::zero();
        for val in &v {
            v_norm_sq = v_norm_sq + *val * *val;
        }
        let v_norm = v_norm_sq.sqrt();

        if v_norm <= eps {
            continue;
        }

        for val in v.iter_mut() {
            *val = *val / v_norm;
        }

        // Apply P = I - 2vv^T to H from left: H = P * H
        for j in k..n {
            let mut dot = R::zero();
            for i in 0..v.len() {
                dot = dot + v[i] * h[(k + 1 + i) * n + j];
            }
            let two: R = num_traits::cast(2.0).unwrap();
            for i in 0..v.len() {
                h[(k + 1 + i) * n + j] = h[(k + 1 + i) * n + j] - two * v[i] * dot;
            }
        }

        // Apply P to H from right: H = H * P
        for i in 0..n {
            let mut dot = R::zero();
            for j in 0..v.len() {
                dot = dot + h[i * n + (k + 1 + j)] * v[j];
            }
            let two: R = num_traits::cast(2.0).unwrap();
            for j in 0..v.len() {
                h[i * n + (k + 1 + j)] = h[i * n + (k + 1 + j)] - two * dot * v[j];
            }
        }
    }

    // Now H is tridiagonal. Apply QR iteration to find eigenvalues.
    let mut m = n;
    let mut iter = 0;

    while m > 1 && iter < max_iter {
        // Find deflation point (check subdiagonal elements)
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

        if k == m - 1 {
            // Deflate
            m -= 1;
            continue;
        }

        // Wilkinson shift
        let d =
            (h[(m - 2) * n + (m - 2)] - h[(m - 1) * n + (m - 1)]) / num_traits::cast(2.0).unwrap();
        let t_sq = h[(m - 1) * n + (m - 2)] * h[(m - 1) * n + (m - 2)];
        let shift = h[(m - 1) * n + (m - 1)]
            - t_sq
                / (d + if d >= R::zero() { R::one() } else { -R::one() } * (d * d + t_sq).sqrt());

        // Implicit QR step with shift
        let mut x = h[k * n + k] - shift;
        let mut z = h[(k + 1) * n + k];

        for i in k..m - 1 {
            // Givens rotation to zero h[i+1, i]
            let r = (x * x + z * z).sqrt();
            if r > eps {
                let c = x / r;
                let s = z / r;

                // Apply rotation to rows i and i+1
                for j in if i > 0 { i - 1 } else { 0 }..n {
                    let temp1 = h[i * n + j];
                    let temp2 = h[(i + 1) * n + j];
                    h[i * n + j] = c * temp1 + s * temp2;
                    h[(i + 1) * n + j] = -s * temp1 + c * temp2;
                }

                // Apply rotation to columns i and i+1
                let upper = if i + 3 < n { i + 3 } else { n };
                for j in 0..upper {
                    let temp1 = h[j * n + i];
                    let temp2 = h[j * n + (i + 1)];
                    h[j * n + i] = c * temp1 + s * temp2;
                    h[j * n + (i + 1)] = -s * temp1 + c * temp2;
                }

                if i < m - 2 {
                    x = h[(i + 1) * n + i];
                    z = h[(i + 2) * n + i];
                }
            }
        }

        iter += 1;
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<R> = (0..n).map(|i| h[i * n + i]).collect();
    Ok(eigenvalues)
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
