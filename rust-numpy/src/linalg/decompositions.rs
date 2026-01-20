use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;

/// Cholesky decomposition.
/// Return the Cholesky decomposition, L * L.H, of the square matrix a,
/// where L is lower-triangular.
pub fn cholesky<T>(a: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error(
            "cholesky requires 2D array",
            "linalg",
        ));
    }
    let n = a.shape()[0];
    if n != a.shape()[1] {
        return Err(NumPyError::value_error(
            "cholesky requires square matrix",
            "linalg",
        ));
    }

    let a_strides = a.strides();
    let mut l_data = vec![T::zero(); n * n];

    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    for i in 0..n {
        for j in 0..=i {
            let mut sum = T::zero();
            for k in 0..j {
                let l_ik = l_data[i * n + k];
                let l_jk = l_data[j * n + k];
                sum = sum + (l_ik * l_jk.conj());
            }

            let a_val = a
                .get(idx(i, j, a_strides))
                .ok_or_else(|| NumPyError::invalid_operation("cholesky index out of bounds"))?;
            let value = *a_val - sum;

            if i == j {
                if !value.is_positive() {
                    return Err(NumPyError::linalg_error(
                        "cholesky",
                        "Matrix is not positive definite",
                    ));
                }
                l_data[i * n + j] = value.sqrt();
            } else {
                let diag = l_data[j * n + j];
                if diag.abs() <= <T::Real as num_traits::Float>::epsilon() {
                    return Err(NumPyError::linalg_error(
                        "cholesky",
                        "Matrix is not positive definite",
                    ));
                }
                l_data[i * n + j] = value / diag;
            }
        }
    }

    Ok(Array::from_data(l_data, vec![n, n]))
}

/// QR decomposition
pub fn qr<T>(_a: &Array<T>) -> Result<(Array<T>, Array<T>), NumPyError>
where
    T: LinalgScalar,
{
    // Stub for now, implementing full QR is complex
    Err(NumPyError::not_implemented("qr not fully implemented"))
}

/// Singular Value Decomposition
pub fn svd<T>(
    _a: &Array<T>,
    _full_matrices: bool,
) -> Result<(Array<T>, Array<f64>, Array<T>), NumPyError>
where
    T: LinalgScalar,
{
    Err(NumPyError::not_implemented("svd not fully implemented"))
}
