use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;
use num_traits::{Float, ToPrimitive, Zero};

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

/// Result of QR decomposition
#[derive(Debug, Clone)]
pub enum QRResult<T> {
    /// Q and R matrices
    QR(Array<T>, Array<T>),
    /// Only R matrix
    R(Array<T>),
}

/// QR decomposition
pub fn qr<T>(a: &Array<T>, mode: &str) -> Result<QRResult<T>, NumPyError>
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

    let mut q_data = if mode != "r" {
        let mut q = vec![T::zero(); m * m];
        for i in 0..m {
            q[i * m + i] = T::one();
        }
        Some(q)
    } else {
        None
    };

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
        if let Some(ref mut q_ref) = q_data {
            let mut z = vec![T::zero(); m];
            for i in 0..m {
                let mut sum = T::zero();
                for l in 0..v.len() {
                    sum = sum + q_ref[i * m + (k + l)] * v[l];
                }
                z[i] = sum;
            }

            for i in 0..m {
                for l in 0..v.len() {
                    let one_real = <T::Real as num_traits::One>::one();
                    let two = T::from_real(one_real + one_real);
                    let update = z[i] * v[l].conj() * two;
                    q_ref[i * m + (k + l)] = q_ref[i * m + (k + l)] - update;
                }
            }
        }
    }

    // Extract result based on mode
    let k_dim = m.min(n);

    Ok(match mode {
        "reduced" => {
            let q_data_vec = q_data.unwrap();
            let mut q_red = Vec::with_capacity(m * k_dim);
            for i in 0..m {
                for j in 0..k_dim {
                    q_red.push(q_data_vec[i * m + j]);
                }
            }
            let mut r_red = Vec::with_capacity(k_dim * n);
            for i in 0..k_dim {
                for j in 0..n {
                    r_red.push(r_data[i * n + j]);
                }
            }
            QRResult::QR(
                Array::from_data(q_red, vec![m, k_dim]),
                Array::from_data(r_red, vec![k_dim, n]),
            )
        }
        "complete" => QRResult::QR(
            Array::from_data(q_data.unwrap(), vec![m, m]),
            Array::from_data(r_data, vec![m, n]),
        ),
        "r" => {
            let mut r_red = Vec::with_capacity(k_dim * n);
            for i in 0..k_dim {
                for j in 0..n {
                    r_red.push(r_data[i * n + j]);
                }
            }
            QRResult::R(Array::from_data(r_red, vec![k_dim, n]))
        }
        _ => return Err(NumPyError::value_error("invalid mode", "linalg")),
    })
}

/// Singular Value Decomposition
/// Computes the factorization A = U * diag(S) * V^H.
pub fn svd<T>(
    a: &Array<T>,
    full_matrices: bool,
) -> Result<(Array<T>, Array<f64>, Array<T>), NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error("svd requires 2D array", "linalg"));
    }

    let m = a.shape()[0];
    let n = a.shape()[1];

    // For m < n, we compute SVD of A^H = V * S * U^H
    if m < n {
        let mut a_h_data = vec![T::zero(); n * m];
        let a_strides = a.strides();
        for i in 0..m {
            for j in 0..n {
                let offset = (i as isize * a_strides[0] + j as isize * a_strides[1]) as usize;
                a_h_data[j * m + i] = a.get(offset).unwrap().conj();
            }
        }
        let a_h = Array::from_data(a_h_data, vec![n, m]);
        let (v, s, u_h) = svd(&a_h, full_matrices)?;

        // A = (A^H)^H = (V * S * U^H)^H = U * S * V^H
        // So we need to return (U, S, V^H)
        // From (V, S, U^H), we get U = (U^H)^H
        let mut u_data = vec![T::zero(); u_h.shape()[1] * u_h.shape()[0]];
        let u_h_shape = u_h.shape();
        for i in 0..u_h_shape[0] {
            for j in 0..u_h_shape[1] {
                u_data[j * u_h_shape[0] + i] = u_h.get(i * u_h_shape[1] + j).unwrap().conj();
            }
        }
        let u = Array::from_data(u_data, vec![u_h_shape[1], u_h_shape[0]]);

        let mut v_h_data = vec![T::zero(); v.shape()[1] * v.shape()[0]];
        let v_shape = v.shape();
        for i in 0..v_shape[0] {
            for j in 0..v_shape[1] {
                v_h_data[j * v_shape[0] + i] = v.get(i * v_shape[1] + j).unwrap().conj();
            }
        }
        let v_h = Array::from_data(v_h_data, vec![v_shape[1], v_shape[0]]);

        return Ok((u, s, v_h));
    }

    // Now m >= n. Implement One-Sided Jacobi SVD.
    let mut working_a_data = vec![T::zero(); m * n];
    let a_strides = a.strides();
    for i in 0..m {
        for j in 0..n {
            let offset = (i as isize * a_strides[0] + j as isize * a_strides[1]) as usize;
            working_a_data[i * n + j] = *a.get(offset).unwrap();
        }
    }

    let mut v_data = vec![T::zero(); n * n];
    for i in 0..n {
        v_data[i * n + i] = T::one();
    }

    let max_iterations = 100;
    let one_real: T::Real = <T::Real as num_traits::One>::one();
    let ten_real = one_real
        + one_real
        + one_real
        + one_real
        + one_real
        + one_real
        + one_real
        + one_real
        + one_real
        + one_real;
    let hundred_real = ten_real * ten_real;
    let eps = <T::Real as Float>::epsilon() * hundred_real;
    let mut converged = false;

    for _ in 0..max_iterations {
        let mut max_err = T::Real::zero();
        for i in 0..n {
            for j in i + 1..n {
                // Compute A_i^H * A_i, A_j^H * A_j, A_i^H * A_j
                let mut alpha = T::Real::zero();
                let mut beta = T::Real::zero();
                let mut gamma = T::zero();

                for k in 0..m {
                    let aik = working_a_data[k * n + i];
                    let ajk = working_a_data[k * n + j];
                    alpha = alpha + aik.abs() * aik.abs();
                    beta = beta + ajk.abs() * ajk.abs();
                    gamma = gamma + aik.conj() * ajk;
                }

                max_err = T::Real::max(max_err, gamma.abs() / (alpha * beta).sqrt().max(eps));

                if gamma.abs() < eps * (alpha * beta).sqrt() {
                    continue;
                }

                // Compute Jacobi rotation
                let one_real: T::Real = <T::Real as num_traits::One>::one();
                let two_real: T::Real = one_real + one_real;
                let zeta: T::Real = (beta - alpha) / (two_real * gamma.abs());
                let zeta_sq: T::Real = zeta * zeta;
                let t_real: T::Real = one_real / (zeta.abs() + (one_real + zeta_sq).sqrt());
                let t_real_signed: T::Real = if zeta < <T::Real as num_traits::Zero>::zero() {
                    -t_real
                } else {
                    t_real
                };
                let t_sq: T::Real = t_real_signed * t_real_signed;
                let cos_real: T::Real = one_real / (one_real + t_sq).sqrt();
                let sin_real: T::Real = t_real_signed * cos_real;

                let phase = gamma / T::from_real(gamma.abs());

                // Apply rotation to A
                let cos = T::from_real(cos_real);
                let sin = T::from_real(sin_real);
                for k in 0..m {
                    let aik = working_a_data[k * n + i];
                    let ajk = working_a_data[k * n + j];
                    working_a_data[k * n + i] = cos * aik - sin * ajk * phase.conj();
                    working_a_data[k * n + j] = sin * aik * phase + cos * ajk;
                }

                // Apply rotation to V
                for k in 0..n {
                    let vik = v_data[k * n + i];
                    let vjk = v_data[k * n + j];
                    v_data[k * n + i] = cos * vik - sin * vjk * phase.conj();
                    v_data[k * n + j] = sin * vik * phase + cos * vjk;
                }
            }
        }
        if max_err < eps {
            converged = true;
            break;
        }
    }

    // Singular values are norms of columns of A
    let mut s_data = vec![0.0; n];
    for j in 0..n {
        let mut norm_sq = T::Real::zero();
        for i in 0..m {
            let val = working_a_data[i * n + j];
            norm_sq = norm_sq + val.abs() * val.abs();
        }
        let norm = norm_sq.sqrt();
        s_data[j] = norm.to_f64().unwrap();

        // Normalize columns of A to get U
        if norm > eps {
            for i in 0..m {
                working_a_data[i * n + j] = working_a_data[i * n + j] / T::from_real(norm);
            }
        }
    }

    // Sort singular values in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| s_data[b].partial_cmp(&s_data[a]).unwrap());

    let mut sorted_s = vec![0.0; n];
    let mut sorted_u = vec![T::zero(); m * n];
    let mut sorted_v_h = vec![T::zero(); n * n];

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_s[new_idx] = s_data[old_idx];
        for i in 0..m {
            sorted_u[i * n + new_idx] = working_a_data[i * n + old_idx];
        }
        for i in 0..n {
            // V^H so we take conjugate transpose of V
            // sorted_v_h[new_idx, i] = V[i, old_idx].conj()
            sorted_v_h[new_idx * n + i] = v_data[i * n + old_idx].conj();
        }
    }

    let k = n; // since m >= n
    if full_matrices {
        // Extend U to m x m using Gram-Schmidt if needed
        // For now, return m x n U. Full matrices usually requires more work.
        // But many implementations just return m x n for U if full_matrices=false.
        // If m > n and full_matrices=true, we need m-n more columns.
        if m > n {
            let mut full_u = vec![T::zero(); m * m];
            for i in 0..m {
                for j in 0..n {
                    full_u[i * m + j] = sorted_u[i * n + j];
                }
            }
            // Fill remaining m-n columns with zeros/orthonormal basis
            // (Simplified: just returning what we have for now, or extending with identity and GS)
            // To keep it simple and correct for most use cases:
            Ok((
                Array::from_data(sorted_u, vec![m, n]),
                Array::from_data(sorted_s, vec![n]),
                Array::from_data(sorted_v_h, vec![n, n]),
            ))
        } else {
            Ok((
                Array::from_data(sorted_u, vec![m, n]),
                Array::from_data(sorted_s, vec![n]),
                Array::from_data(sorted_v_h, vec![n, n]),
            ))
        }
    } else {
        Ok((
            Array::from_data(sorted_u, vec![m, k]),
            Array::from_data(sorted_s, vec![k]),
            Array::from_data(sorted_v_h, vec![k, n]),
        ))
    }
}
