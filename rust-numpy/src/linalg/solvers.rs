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
