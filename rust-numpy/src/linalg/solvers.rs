use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;

/// Solve a linear matrix equation, or system of linear scalar equations.
/// Computes the "exact" solution, x, of the well-determined, i.e., full rank,
/// linear matrix equation ax = b.
pub fn solve<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error("solve: a must be 2D", "linalg"));
    }
    let n = a.shape()[0];
    if n != a.shape()[1] {
        return Err(NumPyError::value_error("solve: a must be square", "linalg"));
    }

    // b can be 1D or 2D.
    // If 1D, shape (N,). faer expects (N, 1).
    // If 2D, shape (N, K).
    let b_ndim = b.ndim();
    if b_ndim > 2 {
        return Err(NumPyError::value_error(
            "solve: b must be 1D or 2D",
            "linalg",
        ));
    }
    if b.shape()[0] != n {
        return Err(NumPyError::shape_mismatch(vec![n], vec![b.shape()[0]]));
    }

    let k = if b_ndim == 1 { 1 } else { b.shape()[1] };

    let a_strides = a.strides();
    let b_strides = b.strides();

    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    let mut a_data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let a_idx = idx(i, j, a_strides);
            a_data[i * n + j] = *a
                .get(a_idx)
                .ok_or_else(|| NumPyError::invalid_operation("solve index out of bounds"))?;
        }
    }

    let mut b_data = vec![T::zero(); n * k];
    for i in 0..n {
        for j in 0..k {
            let b_idx = if b_ndim == 1 {
                (i as isize * b_strides[0]) as usize
            } else {
                idx(i, j, b_strides)
            };
            b_data[i * k + j] = *b
                .get(b_idx)
                .ok_or_else(|| NumPyError::invalid_operation("solve index out of bounds"))?;
        }
    }

    let eps = <T::Real as num_traits::Float>::epsilon();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = a_data[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = a_data[row * n + col].abs();
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = row;
            }
        }

        if pivot_val <= <T::Real as num_traits::Float>::epsilon() {
            return Err(NumPyError::linalg_error("solve", "Singular matrix"));
        }

        if pivot_row != col {
            for j in 0..n {
                a_data.swap(col * n + j, pivot_row * n + j);
            }
            for j in 0..k {
                b_data.swap(col * k + j, pivot_row * k + j);
            }
        }

        let pivot = a_data[col * n + col];
        for row in (col + 1)..n {
            let factor = a_data[row * n + col] / pivot;
            if factor.abs() <= eps {
                continue;
            }
            for j in col..n {
                a_data[row * n + j] = a_data[row * n + j] - factor * a_data[col * n + j];
            }
            for j in 0..k {
                b_data[row * k + j] = b_data[row * k + j] - factor * b_data[col * k + j];
            }
        }
    }

    let mut x_data = vec![T::zero(); n * k];
    for row in (0..n).rev() {
        let diag = a_data[row * n + row];
        for j in 0..k {
            let mut sum = b_data[row * k + j];
            for col in (row + 1)..n {
                sum = sum - a_data[row * n + col] * x_data[col * k + j];
            }
            x_data[row * k + j] = sum / diag;
        }
    }

    if b_ndim == 1 {
        let mut flat = Vec::with_capacity(n);
        for i in 0..n {
            flat.push(x_data[i * k]);
        }
        Ok(Array::from_data(flat, vec![n]))
    } else {
        Ok(Array::from_data(x_data, vec![n, k]))
    }
}

pub fn inv<T>(a: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    // solve(a, I)
    let n = a.shape()[0];
    let eye = Array::<T>::eye(n);
    solve(a, &eye)
}

/// Return the least-squares solution to a linear matrix equation.
/// Computes the vector x that approximatively solves the equation a @ x = b.
/// The equation may be under-, well-, or over-determined.
///
/// Returns (solution, residuals, rank, singular_values)
pub fn lstsq<T>(
    a: &Array<T>,
    b: &Array<T>,
    _rcond: Option<f64>,
) -> Result<(Array<T>, Array<T>, usize, Array<T>), NumPyError>
where
    T: LinalgScalar,
{
    use crate::linalg::decompositions::qr;
    use num_traits::{NumCast, Zero};

    if a.ndim() != 2 {
        return Err(NumPyError::value_error("lstsq: a must be 2D", "linalg"));
    }
    let m = a.shape()[0];
    let n = a.shape()[1];

    // b can be 1D (M,) or 2D (M, K)
    let b_ndim = b.ndim();
    if b.shape()[0] != m {
        return Err(NumPyError::shape_mismatch(vec![m], vec![b.shape()[0]]));
    }

    let (q, r) = qr(a, "reduced")?;

    // Compute d = Q.T @ b
    let q_t = q.transpose();

    // Handle b reshape for dot if 1D
    let (d, _b_reshaped_guard) = if b_ndim == 1 {
        // b is (M,) -> reshape to (M, 1) to allow dot with (K, M)
        // dot expects (M, 1) to produce (K, 1).
        // Since reshape returns new Array, we keep it alive.
        let b_reshaped = b.reshape(&vec![m, 1])?;
        (q_t.dot(&b_reshaped)?, Some(b_reshaped))
    } else {
        (q_t.dot(b)?, None)
    };

    let x = if m >= n {
        solve(&r, &d)?
    } else {
        return Err(NumPyError::not_implemented(
            "lstsq for M < N not fully implemented",
        ));
    };

    let b_est = a.dot(&x)?;

    // Diff calculation. b - b_est.
    // If b was 1D, x is (N, 1) and b_est is (M, 1).
    // We treat them as matching b's dimension.
    let diff = if b_ndim == 1 {
        let mut diff_data = Vec::with_capacity(m);
        for i in 0..m {
            // b is 1D (M,)
            let val_b = *b.get(i).unwrap();
            // b_est is 2D (M, 1)
            // We access index i (row i, col 0)
            let val_est = *b_est.get_linear(i).unwrap();
            diff_data.push(val_b - val_est);
        }
        Array::from_data(diff_data, vec![m])
    } else {
        let k = b.shape()[1];
        let mut diff_data = Vec::with_capacity(m * k);
        for i in 0..m {
            for j in 0..k {
                let val_b = *b.get_linear(i * k + j).unwrap();
                let val_est = *b_est.get_linear(i * k + j).unwrap();
                diff_data.push(val_b - val_est);
            }
        }
        Array::from_data(diff_data, vec![m, k])
    };

    // Residuals
    let residuals = if m > n {
        if b_ndim == 1 {
            let mut sum: T::Real = Zero::zero();
            for i in 0..m {
                let val = *diff.get(i).unwrap();
                sum = sum + val.abs() * val.abs();
            }
            Array::from_data(vec![T::from_real(sum)], vec![1])
        } else {
            let k = b.shape()[1];
            let mut sums: Vec<T::Real> = vec![Zero::zero(); k];
            for i in 0..m {
                for j in 0..k {
                    let val = *diff.get_linear(i * k + j).unwrap();
                    sums[j] = sums[j] + val.abs() * val.abs();
                }
            }
            let sums_t: Vec<T> = sums.into_iter().map(|s| T::from_real(s)).collect();
            Array::from_data(sums_t, vec![k])
        }
    } else {
        Array::from_data(Vec::new(), vec![0])
    };

    let r_diag_len = r.shape()[0].min(r.shape()[1]);
    let mut rank = 0;
    let machine_eps = <T::Real as num_traits::Float>::epsilon();

    for i in 0..r_diag_len {
        let val = *r.get_linear(i * r.shape()[1] + i).unwrap();
        if let Some(threshold) = <T::Real as NumCast>::from(100.0) {
            if val.abs() > machine_eps * threshold {
                rank += 1;
            }
        }
    }

    let s = Array::from_data(Vec::new(), vec![0]);

    // If input b was 1D, output x should be 1D (N,).
    // Currently x is (N, 1) from solve.
    if b_ndim == 1 {
        // Flatten x: (N, 1) -> (N,)
        let x_size: usize = x.shape.iter().product();
        let x_flat = x.reshape(&vec![x_size])?;
        Ok((x_flat, residuals, rank, s))
    } else {
        Ok((x, residuals, rank, s))
    }
}

/// Tensor solve with axes support
/// Solves linear system ax = b for tensor arrays with specified axes
pub fn tensor_solve<T>(
    a: &Array<T>,
    b: &Array<T>,
    axes: Option<&[usize]>,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + Clone + Default + 'static,
{
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    // For 2D or lower, use existing solve function
    if a_ndim <= 2 && b_ndim <= 2 {
        return solve(a, b);
    }

    // For >2D, use axes to select 2D submatrices
    let (ax1, ax2) = match axes {
        Some(axes) if axes.len() >= 2 => (axes[0], axes[1]),
        _ => {
            // Default: use last two dimensions
            if a_ndim >= 2 {
                (a_ndim - 2, a_ndim - 1)
            } else {
                (0, 1)
            }
        }
    };

    // Validate axes
    if ax1 >= a_ndim || ax2 >= a_ndim {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} or {} out of bounds for {}-dimensional array",
            ax1, ax2, a_ndim
        )));
    }

    // Compute batch dimensions (all dimensions except ax1, ax2)
    let mut batch_shape: Vec<usize> = Vec::new();
    for i in 0..a_ndim {
        if i != ax1 && i != ax2 {
            batch_shape.push(a.shape()[i]);
        }
    }

    // Compute matrix dimensions
    let m = a.shape()[ax1]; // rows
    let n = a.shape()[ax2]; // cols

    // Reshape a to 2D: (batch_size, m * n)
    let batch_size: usize = batch_shape.iter().product();
    let a_2d_shape: Vec<usize> = if batch_size > 0 {
        vec![batch_size, m * n]
    } else {
        vec![m * n]
    };
    let a_2d = a.reshape(&a_2d_shape)?;

    // Reshape b to 2D: (batch_size, m) or (batch_size, m * k)
    let b_m = b.shape()[ax1];
    let b_k = if b_ndim > ax2 { b.shape()[ax2] } else { 1 };
    let b_2d_shape: Vec<usize> = if batch_size > 0 {
        vec![batch_size, b_m * b_k]
    } else {
        vec![b_m * b_k]
    };
    let b_2d = b.reshape(&b_2d_shape)?;

    // Solve each 2D system
    let x_2d = solve(&a_2d, &b_2d)?;

    // Reshape result back: (batch_size, n, k) or (batch_size, n)
    let x_shape: Vec<usize> = if batch_size > 0 {
        let mut shape = batch_shape.clone();
        shape.push(n);
        if b_k > 1 {
            shape.push(b_k);
        }
        shape
    } else {
        vec![n]
    };

    x_2d.reshape(&x_shape)
}

/// Tensor inverse with axes support
/// Computes inverse of tensor matrices along specified axes
pub fn tensor_inv<T>(a: &Array<T>, axes: Option<&[usize]>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + Clone + Default + 'static,
{
    let ndim = a.ndim();

    // For 2D or lower, use existing inv function
    if ndim <= 2 {
        return inv(a);
    }

    // For >2D, use axes to select 2D submatrices
    let (ax1, ax2) = match axes {
        Some(axes) if axes.len() >= 2 => (axes[0], axes[1]),
        _ => {
            // Default: use last two dimensions
            if ndim >= 2 {
                (ndim - 2, ndim - 1)
            } else {
                (0, 1)
            }
        }
    };

    // Validate axes
    if ax1 >= ndim || ax2 >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} or {} out of bounds for {}-dimensional array",
            ax1, ax2, ndim
        )));
    }

    // Compute batch dimensions (all dimensions except ax1, ax2)
    let mut batch_shape: Vec<usize> = Vec::new();
    for i in 0..ndim {
        if i != ax1 && i != ax2 {
            batch_shape.push(a.shape()[i]);
        }
    }

    // Compute matrix dimensions
    let m = a.shape()[ax1]; // rows
    let n = a.shape()[ax2]; // cols

    // Reshape a to 2D: (batch_size, m * n)
    let batch_size: usize = batch_shape.iter().product();
    let a_2d_shape: Vec<usize> = if batch_size > 0 {
        vec![batch_size, m * n]
    } else {
        vec![m * n]
    };
    let a_2d = a.reshape(&a_2d_shape)?;

    // Invert each 2D matrix
    let inv_2d = inv(&a_2d)?;

    // Reshape result back: (batch_size, n, m) or (batch_size, n)
    let inv_shape: Vec<usize> = if batch_size > 0 {
        let mut shape = batch_shape.clone();
        shape.push(n);
        shape.push(m);
        shape
    } else {
        vec![n, m]
    };

    inv_2d.reshape(&inv_shape)
}
