use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;
use num_complex::{Complex, Complex64};
use num_traits::{One, Zero};

/// Computes the eigenvalues and right eigenvectors of a Hermitian or real symmetric matrix.
///
/// Parameters
/// ----------
/// a : (..., M, M) Array
///     Hermitian (conjugate symmetric) or real symmetric matrices.
/// UPLO : {'L', 'U'}, optional
///     Specifies whether the calculation is done with the lower ('L') or
///     upper ('U') triangular part of the matrix. Default is 'L'.
///
/// Returns
/// -------
/// w : (..., M) Array
///     The eigenvalues in ascending order, each repeated according to its multiplicity.
/// v : (..., M, M) Array
///     The normalized eigenvectors.
pub fn eigh<T>(
    a: &Array<T>,
    uplo: Option<&str>,
) -> Result<(Array<f64>, Array<Complex64>), NumPyError>
where
    T: ToComplex + Clone,
{
    let shape = a.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return Err(NumPyError::invalid_value(format!(
            "Input must be at least 2-dimensional, got shape {:?}",
            shape
        )));
    }
    let n = shape[ndim - 2];
    let m = shape[ndim - 1];
    if n != m {
        return Err(NumPyError::invalid_value(format!(
            "Last two dimensions must be square, got unique dimensions {} and {}",
            n, m
        )));
    }

    let uplo_val = uplo.unwrap_or("L");
    if uplo_val != "L" && uplo_val != "U" {
        return Err(NumPyError::value_error("UPLO must be 'L' or 'U'", "linalg"));
    }

    if n == 0 {
        let mut w_shape = shape.to_vec();
        w_shape.pop();
        let v_shape = shape.to_vec();
        return Ok((
            Array::from_data(vec![], w_shape),
            Array::from_data(vec![], v_shape),
        ));
    }

    let batch_size = shape[..ndim - 2].iter().product::<usize>();
    let mut w_final_data = Vec::with_capacity(batch_size * n);
    let mut v_final_data = Vec::with_capacity(batch_size * n * n);

    let strides = a.strides();
    let offset = a.offset;
    let data = a.data();

    let mut multi_indices = vec![0; ndim];

    for b in 0..batch_size {
        let mut temp_b = b;
        for k in (0..ndim - 2).rev() {
            multi_indices[k] = temp_b % shape[k];
            temp_b /= shape[k];
        }

        let mut data_c64 = Vec::with_capacity(n * n);
        for i in 0..n {
            multi_indices[ndim - 2] = i;
            for j in 0..n {
                multi_indices[ndim - 1] = j;
                let linear_idx = crate::strides::compute_linear_index(&multi_indices, strides);
                let physical_idx = (offset as isize + linear_idx) as usize;
                let val = data[physical_idx].clone().to_complex();

                // Enforce Hermitian symmetry based on UPLO
                if uplo_val == "L" {
                    if j > i {
                        multi_indices[ndim - 2] = j;
                        multi_indices[ndim - 1] = i;
                        let sym_linear =
                            crate::strides::compute_linear_index(&multi_indices, strides);
                        let sym_phys = (offset as isize + sym_linear) as usize;
                        data_c64.push(data[sym_phys].clone().to_complex().conj());
                        multi_indices[ndim - 2] = i;
                        multi_indices[ndim - 1] = j;
                    } else {
                        data_c64.push(val);
                    }
                } else {
                    if i > j {
                        multi_indices[ndim - 2] = j;
                        multi_indices[ndim - 1] = i;
                        let sym_linear =
                            crate::strides::compute_linear_index(&multi_indices, strides);
                        let sym_phys = (offset as isize + sym_linear) as usize;
                        data_c64.push(data[sym_phys].clone().to_complex().conj());
                        multi_indices[ndim - 2] = i;
                        multi_indices[ndim - 1] = j;
                    } else {
                        data_c64.push(val);
                    }
                }
            }
        }

        // For Hermitian matrices, eigenvalues are real.
        // We use tridiagonal reduction + QR iteration.
        let (mut h, mut q) = symmetric_tridiagonal_reduction(n, &mut data_c64);

        // QR iteration for tridiagonal matrices
        qr_iteration_tridiagonal(n, &mut h, &mut q, 1000)?;

        // Extract real eigenvalues
        let mut w_batch = Vec::with_capacity(n);
        for i in 0..n {
            w_batch.push(h[i * n + i].re);
        }

        // Sort eigenvalues and corresponding eigenvectors
        let mut p: Vec<usize> = (0..n).collect();
        p.sort_by(|&i, &j| w_batch[i].partial_cmp(&w_batch[j]).unwrap());

        for &idx in &p {
            w_final_data.push(w_batch[idx]);
        }

        let mut v_batch = vec![Complex64::zero(); n * n];
        for (new_col, &old_col) in p.iter().enumerate() {
            for row in 0..n {
                v_batch[row * n + new_col] = q[row * n + old_col];
            }
        }
        v_final_data.extend(v_batch);
    }

    let mut w_shape = shape.to_vec();
    w_shape.pop();
    let v_shape = shape.to_vec();

    Ok((
        Array::from_data(w_final_data, w_shape),
        Array::from_data(v_final_data, v_shape),
    ))
}

/// Compute only the eigenvalues of a square matrix.
pub fn eigvals<T>(a: &Array<T>) -> Result<Array<Complex64>, NumPyError>
where
    T: ToComplex + Clone,
{
    let (w, _) = eig(a)?;
    Ok(w)
}

/// Compute only the eigenvalues of a Hermitian or real symmetric matrix.
pub fn eigvalsh<T>(a: &Array<T>, uplo: Option<&str>) -> Result<Array<f64>, NumPyError>
where
    T: ToComplex + Clone,
{
    let (w, _) = eigh(a, uplo)?;
    Ok(w)
}

fn symmetric_tridiagonal_reduction(
    n: usize,
    a: &mut [Complex64],
) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut q = vec![Complex64::zero(); n * n];
    for i in 0..n {
        q[i * n + i] = Complex64::one();
    }

    if n <= 2 {
        return (a.to_vec(), q);
    }

    for k in 0..n - 2 {
        let mut x = Vec::with_capacity(n - (k + 1));
        for i in k + 1..n {
            x.push(a[i * n + k]);
        }

        let mut x_norm_sq = 0.0;
        for val in &x {
            x_norm_sq += val.norm_sqr();
        }
        let x_norm = x_norm_sq.sqrt();

        if x_norm == 0.0 {
            continue;
        }

        let mut v = x.clone();
        let phase = if x[0].norm() == 0.0 {
            Complex64::one()
        } else {
            x[0] / x[0].norm()
        };
        v[0] += phase * Complex64::from_real(x_norm);

        let mut v_norm_sq = 0.0;
        for val in &v {
            v_norm_sq += val.norm_sqr();
        }
        let v_norm = v_norm_sq.sqrt();
        if v_norm == 0.0 {
            continue;
        }
        for val in v.iter_mut() {
            *val /= Complex64::from_real(v_norm);
        }

        // P = I - 2vv^H
        for j in k..n {
            let mut vh_a = Complex64::zero();
            for i in 0..v.len() {
                vh_a += v[i].conj() * a[(k + 1 + i) * n + j];
            }
            for i in 0..v.len() {
                a[(k + 1 + i) * n + j] -= Complex64::from_real(2.0) * v[i] * vh_a;
            }
        }
        for i in 0..n {
            let mut a_v = Complex64::zero();
            for j in 0..v.len() {
                a_v += a[i * n + (k + 1 + j)] * v[j];
            }
            for j in 0..v.len() {
                a[i * n + (k + 1 + j)] -= Complex64::from_real(2.0) * a_v * v[j].conj();
            }
        }
        for i in 0..n {
            let mut q_v = Complex64::zero();
            for j in 0..v.len() {
                q_v += q[i * n + (k + 1 + j)] * v[j];
            }
            for j in 0..v.len() {
                q[i * n + (k + 1 + j)] -= Complex64::from_real(2.0) * q_v * v[j].conj();
            }
        }
    }
    (a.to_vec(), q)
}

fn qr_iteration_tridiagonal(
    n: usize,
    h: &mut [Complex64],
    q: &mut [Complex64],
    max_iter: usize,
) -> Result<(), NumPyError> {
    let eps = 1e-12;
    let mut m = n;

    while m > 1 {
        let mut iter = 0;
        while iter < max_iter {
            let mut i = m - 1;
            while i > 0 {
                if h[i * n + (i - 1)].norm()
                    <= eps * (h[(i - 1) * n + (i - 1)].norm() + h[i * n + i].norm())
                {
                    break;
                }
                i -= 1;
            }

            if i == m - 1 {
                m -= 1;
                break;
            }

            apply_qr_step(n, h, q, i, m);
            iter += 1;
        }

        if iter == max_iter {
            return Err(NumPyError::linalg_error("eigh", "QR failed to converge"));
        }
    }
    Ok(())
}

/// Computes the eigenvalues and right eigenvectors of a square array.
///
/// Parameters
/// ----------
/// a : (..., M, M) Array
///     Matrices for which the eigenvalues and right eigenvectors will be computed.
///
/// Returns
/// -------
/// w : (..., M) Array
///     The eigenvalues, each repeated according to its multiplicity.
///     They are not necessarily ordered.
/// v : (..., M, M) Array
///     The normalized (unit "length") eigenvectors, such that the
///     column `v[:, i]` is the eigenvector corresponding to the
///     eigenvalue `w[i]`.
///
/// Errors
/// ------
/// Returns `NumPyError` if:
/// - Input array has less than 2 dimensions.
/// - Last two dimensions of the input are not square.
/// - The algorithm fails to converge.
pub fn eig<T>(a: &Array<T>) -> Result<(Array<Complex64>, Array<Complex64>), NumPyError>
where
    T: ToComplex + Clone,
{
    let shape = a.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return Err(NumPyError::invalid_value(format!(
            "Input must be at least 2-dimensional, got shape {:?}",
            shape
        )));
    }
    let n = shape[ndim - 2];
    let m = shape[ndim - 1];
    if n != m {
        return Err(NumPyError::invalid_value(format!(
            "Last two dimensions must be square, got unique dimensions {} and {}",
            n, m
        )));
    }

    if n == 0 {
        let mut w_shape = shape.to_vec();
        w_shape.pop();
        let v_shape = shape.to_vec();
        return Ok((
            Array::from_data(vec![], w_shape),
            Array::from_data(vec![], v_shape),
        ));
    }

    let batch_size = shape[..ndim - 2].iter().product::<usize>();
    let mut w_final_data = Vec::with_capacity(batch_size * n);
    let mut v_final_data = Vec::with_capacity(batch_size * n * n);

    let strides = a.strides();
    let offset = a.offset;
    let data = a.data();

    let mut multi_indices = vec![0; ndim];

    for b in 0..batch_size {
        // Compute batch indices
        let mut temp_b = b;
        for k in (0..ndim - 2).rev() {
            multi_indices[k] = temp_b % shape[k];
            temp_b /= shape[k];
        }

        // Convert current matrix to Complex64
        let mut data_c64 = Vec::with_capacity(n * n);
        for i in 0..n {
            multi_indices[ndim - 2] = i;
            for j in 0..n {
                multi_indices[ndim - 1] = j;
                let linear_idx = crate::strides::compute_linear_index(&multi_indices, strides);
                let physical_idx = (offset as isize + linear_idx) as usize;
                data_c64.push(data[physical_idx].clone().to_complex());
            }
        }

        // 1. Hessenberg Reduction
        let (mut h, mut q) = hessenberg_reduction(n, &mut data_c64);

        // 2. Francis QR Iteration to reached Schur Form
        francis_qr_iteration(n, &mut h, &mut q, 1000)?;

        // Zero out subdiagonal elements
        for i in 1..n {
            for j in 0..i {
                h[i * n + j] = Complex64::zero();
            }
        }

        // 3. Extract eigenvalues
        for i in 0..n {
            w_final_data.push(h[i * n + i]);
        }

        // 4. Solve for eigenvectors
        let v_batch = solve_eigenvectors(n, &h, &q);
        v_final_data.extend(v_batch);
    }

    let mut w_shape = shape.to_vec();
    w_shape.pop();
    let v_shape = shape.to_vec();

    Ok((
        Array::from_data(w_final_data, w_shape),
        Array::from_data(v_final_data, v_shape),
    ))
}

fn francis_qr_iteration(
    n: usize,
    h: &mut [Complex64],
    q: &mut [Complex64],
    max_iter: usize,
) -> Result<(), NumPyError> {
    let eps = 1e-12;
    let mut m = n;

    while m > 1 {
        let mut iter = 0;
        while iter < max_iter {
            // Find deflation point
            let mut i = m - 1;
            while i > 0 {
                if h[i * n + (i - 1)].norm()
                    <= eps * (h[(i - 1) * n + (i - 1)].norm() + h[i * n + i].norm())
                {
                    break;
                }
                i -= 1;
            }

            if i == m - 1 {
                // Deflate!
                m -= 1;
                break;
            }

            // Apply a QR step to the submatrix [i..m, i..m]
            // For complex matrices, we use a single shift (Wilkinson shift)
            apply_qr_step(n, h, q, i, m);

            iter += 1;
        }

        if iter == max_iter {
            return Err(NumPyError::linalg_error(
                "eig",
                format!(
                    "Francis QR failed to converge after {} iterations",
                    max_iter
                ),
            ));
        }
    }

    Ok(())
}

fn apply_qr_step(n: usize, h: &mut [Complex64], q: &mut [Complex64], low: usize, high: usize) {
    // Single-shift QR step using Wilkinson shift
    let m = high - low;
    if m < 2 {
        return;
    }

    // Bottom-right 2x2:
    // [ a b ]
    // [ c d ]
    let d = h[(high - 1) * n + (high - 1)];
    let c = h[(high - 1) * n + (high - 2)];
    let b = h[(high - 2) * n + (high - 1)];
    let a = h[(high - 2) * n + (high - 2)];

    // Wilkinson shift: eigenvalue of 2x2 closer to d
    let tr = a + d;
    let det = a * d - b * c;
    let disc = (tr * tr - Complex64::from_real(4.0) * det).sqrt();
    let mu1 = (tr + disc) / Complex64::from_real(2.0);
    let mu2 = (tr - disc) / Complex64::from_real(2.0);

    let mu = if (mu1 - d).norm() < (mu2 - d).norm() {
        mu1
    } else {
        mu2
    };

    // QR step with shift mu: chasing the bulge
    // First rotation is special: based on (H - mu*I)e_1
    let mut x = h[low * n + low] - mu;
    let mut y = h[(low + 1) * n + low];

    for i in low..high - 1 {
        let norm = (x.norm_sqr() + y.norm_sqr()).sqrt();
        if norm > 1e-18 {
            let r = Complex64::from_real(norm);
            let cos = x.conj() / r;
            let sin = y.conj() / r;

            // Apply rotation G to rows i and i+1 from left
            for j in i..n {
                let hi = h[i * n + j];
                let hi1 = h[(i + 1) * n + j];
                h[i * n + j] = cos * hi + sin * hi1;
                h[(i + 1) * n + j] = -sin.conj() * hi + cos.conj() * hi1;
            }

            // Apply rotation G^H to columns i and i+1 from right
            for j in 0..std::cmp::min(i + 3, n) {
                let hi = h[j * n + i];
                let hi1 = h[j * n + i + 1];
                h[j * n + i] = cos.conj() * hi + sin.conj() * hi1;
                h[j * n + i + 1] = -sin * hi + cos * hi1;
            }

            // Update Q
            for j in 0..n {
                let qi = q[j * n + i];
                let qi1 = q[j * n + i + 1];
                q[j * n + i] = cos.conj() * qi + sin.conj() * qi1;
                q[j * n + i + 1] = -sin * qi + cos * qi1;
            }

            // Next element to zero out (bulge chasing)
            if i < high - 2 {
                x = h[(i + 1) * n + i];
                y = h[(i + 2) * n + i];
            }
        }
    }
}

// Helper trait for converting LinalgScalar to Complex64
pub trait ToComplex {
    fn to_complex(self) -> Complex64;
}

impl ToComplex for f32 {
    fn to_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl ToComplex for f64 {
    fn to_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl ToComplex for Complex<f32> {
    fn to_complex(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
}

impl ToComplex for Complex<f64> {
    fn to_complex(self) -> Complex64 {
        self
    }
}

impl ToComplex for i32 {
    fn to_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl ToComplex for i64 {
    fn to_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl ToComplex for u32 {
    fn to_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl ToComplex for u64 {
    fn to_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

/// Reduces matrix A to upper Hessenberg form H = Q^T * A * Q
/// Returns (H, Q) where H is stored in `a` (modified in-place) and Q is the unitary conversion matrix.
fn hessenberg_reduction(n: usize, a: &mut [Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut q = vec![Complex64::zero(); n * n];
    for i in 0..n {
        q[i * n + i] = Complex64::one();
    } // Identity

    if n <= 2 {
        return (a.to_vec(), q);
    }

    for k in 0..n - 2 {
        // Householder reflection to zero out A[k+2..n, k]
        // We want to zero elements below the first subdiagonal.

        let mut x = Vec::with_capacity(n - (k + 1));
        for i in k + 1..n {
            x.push(a[i * n + k]);
        }

        let mut x_norm_sq = 0.0;
        for val in &x {
            x_norm_sq += val.norm_sqr();
        }
        let x_norm = x_norm_sq.sqrt();

        if x_norm == 0.0 {
            continue;
        }

        // v = x + sign(x[0]) * ||x|| * e1
        let mut v = x.clone();
        let phase = if x[0].norm() == 0.0 {
            Complex64::one()
        } else {
            x[0] / x[0].norm()
        };

        v[0] += phase * Complex64::from_real(x_norm);

        let mut v_norm_sq = 0.0;
        for val in &v {
            v_norm_sq += val.norm_sqr();
        }
        let v_norm = v_norm_sq.sqrt();

        if v_norm == 0.0 {
            continue;
        }

        for val in v.iter_mut() {
            *val /= Complex64::from_real(v_norm);
        }

        // Apply P = I - 2 * v * v^H to A from left: P * A
        // A[k+1..n, k..n] = A[k+1..n, k..n] - 2 * v * (v^H * A[k+1..n, k..n])
        for j in k..n {
            let mut vh_a = Complex64::zero();
            for i in 0..v.len() {
                vh_a += v[i].conj() * a[(k + 1 + i) * n + j];
            }
            for i in 0..v.len() {
                a[(k + 1 + i) * n + j] -= Complex64::from_real(2.0) * v[i] * vh_a;
            }
        }

        // Apply P to A from right: A * P
        // A[0..n, k+1..n] = A[0..n, k+1..n] - 2 * (A[0..n, k+1..n] * v) * v^H
        for i in 0..n {
            let mut a_v = Complex64::zero();
            for j in 0..v.len() {
                a_v += a[i * n + (k + 1 + j)] * v[j];
            }
            for j in 0..v.len() {
                a[i * n + (k + 1 + j)] -= Complex64::from_real(2.0) * a_v * v[j].conj();
            }
        }

        // Apply P to Q from right: Q * P
        for i in 0..n {
            let mut q_v = Complex64::zero();
            for j in 0..v.len() {
                q_v += q[i * n + (k + 1 + j)] * v[j];
            }
            for j in 0..v.len() {
                q[i * n + (k + 1 + j)] -= Complex64::from_real(2.0) * q_v * v[j].conj();
            }
        }
    }

    (a.to_vec(), q)
}

fn solve_eigenvectors(n: usize, h: &[Complex64], q: &[Complex64]) -> Vec<Complex64> {
    // h is in Schur form (upper triangular)
    // q is the unitary transformation matrix such that A = q * h * q^H
    // The right eigenvectors y of h satisfy h * y = lambda * y
    // Then v = q * y are right eigenvectors of A:
    // A * v = (q * h * q^H) * (q * y) = q * h * y = q * lambda * y = lambda * v

    let mut v_final = vec![Complex64::zero(); n * n];

    for i in 0..n {
        let lambda = h[i * n + i];
        let mut y = vec![Complex64::zero(); n];
        y[i] = Complex64::one();

        // Back-substitution for row j from i-1 down to 0
        // (H_jj - lambda) * y_j + sum_{k=j+1}^{i} H_jk * y_k = 0
        for j in (0..i).rev() {
            let mut sum = Complex64::zero();
            for k in j + 1..=i {
                sum += h[j * n + k] * y[k];
            }
            let denom = h[j * n + j] - lambda;
            // If denom is zero, we have a multiple eigenvalue
            // We use a small perturbation to find a linearly independent eigenvector if possible
            if denom.norm() < 1e-15 {
                y[j] = -sum / Complex64::new(1e-15, 0.0);
            } else {
                y[j] = -sum / denom;
            }
        }

        // v_col = q * y
        for row in 0..n {
            let mut val = Complex64::zero();
            for k in 0..=i {
                val += q[row * n + k] * y[k];
            }
            v_final[row * n + i] = val;
        }

        // Normalize column i
        let mut norm_sq = 0.0;
        for row in 0..n {
            norm_sq += v_final[row * n + i].norm_sqr();
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for row in 0..n {
                v_final[row * n + i] /= Complex64::new(norm, 0.0);
            }
        }
    }

    v_final
}
