use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;
use num_traits::Zero;

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
