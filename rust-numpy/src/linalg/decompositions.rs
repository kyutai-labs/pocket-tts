use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;
use num_complex::Complex;
use num_traits::{Float, One, Zero};

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
pub fn qr<T>(a: &Array<T>, mode: &str) -> Result<(Array<T>, Array<T>), NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error("qr requires 2D array", "linalg"));
    }
    let m = a.shape()[0];
    let n = a.shape()[1];

    // We work with flattened data for Q (m x m) and R (m x n)
    // R initialized with A.
    let mut r_data = vec![T::zero(); m * n];
    let a_strides = a.strides();

    for i in 0..m {
        for j in 0..n {
            // Calculate offset based on strides
            let offset = (i as isize * a_strides[0] + j as isize * a_strides[1]) as usize;
            r_data[i * n + j] = *a.get(offset).unwrap();
        }
    }

    let mut q_data = vec![T::zero(); m * m];
    for i in 0..m {
        q_data[i * m + i] = T::one();
    }

    let iterations = m.min(n);

    for k in 0..iterations {
        // Compute Householder vector v for column k of R (from row k to m)
        // x = R[k:m, k]

        // Compute norm squared of x
        let mut norm_sq = T::Real::zero();
        for i in k..m {
            let val = r_data[i * n + k];
            norm_sq = norm_sq + val.abs() * val.abs();
        }
        let norm = norm_sq.sqrt();

        if norm <= T::Real::zero() {
            continue;
        }

        let pivot = r_data[k * n + k];
        let alpha = if pivot.abs() > T::Real::zero() {
            let sign = pivot * (T::one() / T::from_real(pivot.abs()));
            sign * T::from_real(norm) * (T::zero() - T::one()) // -sign * norm
        } else {
            T::from_real(norm)
        };

        // u = x. u[0] -= alpha.
        // We calculate v = u / ||u|| directly or store u.
        // v length is m - k.
        let mut v = vec![T::zero(); m - k];
        for i in 0..(m - k) {
            v[i] = r_data[(k + i) * n + k];
        }
        v[0] = v[0] - alpha;

        let mut v_norm_sq = T::Real::zero();
        for val in &v {
            v_norm_sq = v_norm_sq + val.abs() * val.abs();
        }
        let v_norm = v_norm_sq.sqrt();

        if v_norm <= T::Real::zero() {
            continue;
        }

        for i in 0..v.len() {
            v[i] = v[i] * (T::one() / T::from_real(v_norm));
        }

        // Apply H to R: R = (I - 2 v v*) R
        // R[k:m, k:n] = R[k:m, k:n] - 2 v (v* R[k:m, k:n])
        // w = v* R[k:m, k:n] (row vector len n-k)
        let n_cols = n - k;
        let mut w = vec![T::zero(); n_cols];

        for j in 0..n_cols {
            let col = k + j;
            let mut sum = T::zero();
            for i in 0..v.len() {
                sum = sum + v[i].conj() * r_data[(k + i) * n + col];
            }
            w[j] = sum;
        }

        for i in 0..v.len() {
            for j in 0..n_cols {
                let col = k + j;
                // Use <T::Real as One>::one() explicitly for generic T::Real
                let one_real = <T::Real as num_traits::One>::one();
                let two = T::from_real(one_real + one_real);
                let update = v[i] * w[j] * two;
                r_data[(k + i) * n + col] = r_data[(k + i) * n + col] - update;
            }
        }

        // Apply H to Q: Q = Q (I - 2 v v*)
        // Q[:, k:m] etc.
        // z = Q[:, k:m] * v
        let mut z = vec![T::zero(); m];
        for i in 0..m {
            let mut sum = T::zero();
            for l in 0..v.len() {
                sum = sum + q_data[i * m + (k + l)] * v[l];
            }
            z[i] = sum;
        }

        for i in 0..m {
            for l in 0..v.len() {
                let one_real = <T::Real as num_traits::One>::one();
                let two = T::from_real(one_real + one_real);
                let update = z[i] * v[l].conj() * two;
                q_data[i * m + (k + l)] = q_data[i * m + (k + l)] - update;
            }
        }
    }

    // Extract result based on mode
    let k_dim = m.min(n);

    let (final_q, final_r) = match mode {
        "reduced" => {
            let mut q_red = Vec::with_capacity(m * k_dim);
            for i in 0..m {
                for j in 0..k_dim {
                    q_red.push(q_data[i * m + j]);
                }
            }
            let mut r_red = Vec::with_capacity(k_dim * n);
            for i in 0..k_dim {
                for j in 0..n {
                    r_red.push(r_data[i * n + j]);
                }
            }
            (
                Array::from_data(q_red, vec![m, k_dim]),
                Array::from_data(r_red, vec![k_dim, n]),
            )
        }
        "complete" => (
            Array::from_data(q_data, vec![m, m]),
            Array::from_data(r_data, vec![m, n]),
        ),
        "r" => {
            return Err(NumPyError::not_implemented("mode 'r' not supported"));
        }
        _ => return Err(NumPyError::value_error("invalid mode", "linalg")),
    };

    Ok((final_q, final_r))
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
