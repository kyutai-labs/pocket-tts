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

pub fn lstsq<T>(_a: &Array<T>, _b: &Array<T>) -> Result<Array<T>, NumPyError> {
    Err(NumPyError::not_implemented("lstsq not implemented"))
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
    let m = a.shape()[ax1];  // rows
    let n = a.shape()[ax2];  // cols

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
pub fn tensor_inv<T>(
    a: &Array<T>,
    axes: Option<&[usize]>,
) -> Result<Array<T>, NumPyError>
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
    let m = a.shape()[ax1];  // rows
    let n = a.shape()[ax2];  // cols

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
