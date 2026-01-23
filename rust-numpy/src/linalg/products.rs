use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::solvers::inv;
use crate::linalg::LinalgScalar;

/// Dot product of two arrays.
/// Supports scalar (0D), vector (1D), matrix (2D), and N-D arrays.
pub fn dot<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    match (a.ndim(), b.ndim()) {
        (0, _) | (_, 0) => {
            let a_val = a.get_linear(0).cloned().unwrap_or(T::zero());
            let b_val = b.get_linear(0).cloned().unwrap_or(T::zero());
            if a.ndim() == 0 && b.ndim() == 0 {
                Ok(Array::from_vec(vec![a_val * b_val]).reshape(&[])?)
            } else if a.ndim() == 0 {
                let mut res = b.clone();
                for i in 0..res.size() {
                    let v = *res.get_linear(i).unwrap();
                    res.set_linear(i, v * a_val);
                }
                Ok(res)
            } else {
                let mut res = a.clone();
                for i in 0..res.size() {
                    let v = *res.get_linear(i).unwrap();
                    res.set_linear(i, v * b_val);
                }
                Ok(res)
            }
        }
        (1, 1) => {
            if a.size() != b.size() {
                return Err(NumPyError::shape_mismatch(
                    a.shape().to_vec(),
                    b.shape().to_vec(),
                ));
            }
            let mut sum = T::zero();
            for i in 0..a.size() {
                sum = sum + (*a.get_linear(i).unwrap() * *b.get_linear(i).unwrap());
            }
            Ok(Array::from_vec(vec![sum]).reshape(&[])?)
        }
        (2, 2) => {
            let a_rows = a.shape()[0];
            let a_cols = a.shape()[1];
            let b_rows = b.shape()[0];
            let b_cols = b.shape()[1];

            if a_cols != b_rows {
                return Err(NumPyError::shape_mismatch(
                    vec![a_rows, a_cols],
                    vec![b_rows, b_cols],
                ));
            }

            let a_strides = a.strides();
            let b_strides = b.strides();

            let mut output = Array::<T>::zeros(vec![a_rows, b_cols]);
            let output_strides = output.strides().to_vec();

            for i in 0..a_rows {
                for j in 0..b_cols {
                    let mut sum = T::zero();

                    for k in 0..a_cols {
                        let a_idx =
                            (i as isize * a_strides[0] + k as isize * a_strides[1]) as usize;
                        let b_idx =
                            (k as isize * b_strides[0] + j as isize * b_strides[1]) as usize;

                        let a_val = a.get_linear_physical(a_idx).unwrap();
                        let b_val = b.get_linear_physical(b_idx).unwrap();

                        sum = sum + (*a_val * *b_val);
                    }

                    let out_idx =
                        (i as isize * output_strides[0] + j as isize * output_strides[1]) as usize;
                    output.set_linear_physical(out_idx, sum)?;
                }
            }

            Ok(output)
        }
        _ => dot_nd(a, b),
    }
}

/// Matrix multiplication
pub fn matmul<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    // For now, matmul uses dot. For proper N-D matmul, broadcasting should be implemented.
    dot(a, b)
}

/// N-D dot product implementation.
/// For N-D arrays, it is a sum product over the last axis of a and the second-to-last axis of b.
pub fn dot_nd<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();
    let a_shape = a.shape();
    let b_shape = b.shape();

    let a_last_dim = a_shape[a_ndim - 1];
    let b_contract_dim = if b_ndim == 1 {
        b_shape[0]
    } else {
        b_shape[b_ndim - 2]
    };

    if a_last_dim != b_contract_dim {
        return Err(NumPyError::shape_mismatch(
            a_shape.to_vec(),
            b_shape.to_vec(),
        ));
    }

    // Compute result shape
    let mut res_shape = a_shape[..a_ndim - 1].to_vec();
    if b_ndim > 1 {
        res_shape.extend_from_slice(&b_shape[..b_ndim - 2]);
        res_shape.push(b_shape[b_ndim - 1]);
    }

    let a_outer_size = a.size() / a_last_dim;
    let b_inner_dim = if b_ndim == 1 { 1 } else { b_shape[b_ndim - 1] };
    let b_batch_size = b.size() / (b_contract_dim * b_inner_dim);

    let mut res_data = Vec::with_capacity(a_outer_size * b_batch_size * b_inner_dim);

    for i in 0..a_outer_size {
        for j in 0..b_batch_size {
            for k in 0..b_inner_dim {
                let mut sum = T::zero();
                for l in 0..a_last_dim {
                    // a[i, l]
                    let a_val = *a.get_linear(i * a_last_dim + l).unwrap();
                    // b[j, l, k]
                    let b_val = if b_ndim == 1 {
                        *b.get_linear(l).unwrap()
                    } else {
                        *b.get_linear(j * b_contract_dim * b_inner_dim + l * b_inner_dim + k)
                            .unwrap()
                    };
                    sum = sum + a_val * b_val;
                }
                res_data.push(sum);
            }
        }
    }

    Ok(Array::from_data(res_data, res_shape))
}

/// Extract diagonal with custom axes support
/// Returns the diagonal of the array along specified axes
pub fn diagonal_enhanced<T>(
    a: &Array<T>,
    offset: isize,
    axis1: Option<usize>,
    axis2: Option<usize>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
{
    let ndim = a.ndim();

    // Default axes: last two dimensions
    let (ax1, ax2) = match (axis1, axis2) {
        (Some(a1), Some(a2)) => (a1, a2),
        (Some(_a1), None) => {
            if ndim >= 2 {
                (ndim - 2, ndim - 1)
            } else {
                (0, 1)
            }
        }
        (None, Some(_a2)) => {
            if ndim >= 2 {
                (ndim - 2, ndim - 1)
            } else {
                (0, 1)
            }
        }
        (None, None) => {
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

    // Compute diagonal size
    let diag_size = std::cmp::min(a.shape()[ax1], a.shape()[ax2]);

    // Compute output shape
    let mut output_shape: Vec<usize> = Vec::with_capacity(ndim - 1);
    for i in 0..ndim {
        if i != ax1 && i != ax2 {
            output_shape.push(a.shape()[i]);
        }
    }
    output_shape.push(diag_size);

    // Extract diagonal elements
    let mut output_data = Vec::with_capacity(output_shape.iter().product());
    let output_size = output_shape.iter().product();

    for _ in 0..output_size {
        for i in 0..diag_size {
            let idx1 = if offset >= 0 {
                i + offset as usize
            } else {
                (diag_size - 1 - i) + offset.unsigned_abs()
            };
            let idx2 = i;

            // Compute linear indices for diagonal elements
            let mut a_indices = vec![0usize; ndim];
            a_indices[ax1] = idx1;
            a_indices[ax2] = idx2;

            // Fill other dimensions from output position
            let mut out_idx = 0;
            for (dim, &val) in output_shape.iter().enumerate() {
                out_idx *= val;
                if dim < output_shape.len() - 1 {
                    let orig_dim = if dim < ax1 {
                        dim
                    } else if dim < ax2 {
                        dim + 1
                    } else {
                        dim + 2
                    };
                    let divisor: usize = output_shape[dim + 1..].iter().product();
                    a_indices[orig_dim] = if divisor > 0 { out_idx / divisor } else { 0 };
                }
            }

            // Get diagonal element
            let linear_idx = crate::strides::compute_linear_index(&a_indices, a.strides());
            if let Some(val) = a.get(linear_idx as usize) {
                output_data.push(val.clone());
            }
        }
    }

    Ok(Array::from_data(output_data, output_shape))
}

/// Compute the trace of a matrix (sum of diagonal elements)
pub fn trace<T>(a: &Array<T>) -> Result<T, NumPyError>
where
    T: LinalgScalar + Clone + Default + 'static,
{
    if a.ndim() < 2 {
        return Err(NumPyError::value_error(
            "trace requires at least 2 dimensions",
            "linalg",
        ));
    }

    // Use last two dimensions by default, similar to numpy.trace
    // Numpy allows specifying offset, axis1, axis2. This is a basic implementation.
    let diag = diagonal_enhanced(a, 0, None, None)?;

    let mut sum = T::zero();
    let size: usize = diag.shape.iter().product();
    for i in 0..size {
        if let Some(val) = diag.get(i) {
            sum = sum + *val;
        }
    }

    Ok(sum)
}

/// Compute the n-th power of a square matrix.
pub fn matrix_power<T>(a: &Array<T>, n: i32) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error(
            "matrix_power requires 2D array",
            "linalg",
        ));
    }
    let m = a.shape()[0];
    if m != a.shape()[1] {
        return Err(NumPyError::value_error(
            "matrix_power requires square matrix",
            "linalg",
        ));
    }

    if n == 0 {
        return Ok(Array::eye(m));
    }

    let mut abs_n = n.abs() as u32;
    let base = if n < 0 { inv(a)? } else { a.clone() };

    if abs_n == 1 {
        return Ok(base);
    }

    let mut result = Array::eye(m);
    let mut current_base = base;

    while abs_n > 0 {
        if abs_n % 2 == 1 {
            result = dot(&result, &current_base)?;
        }
        abs_n /= 2;
        if abs_n > 0 {
            current_base = dot(&current_base, &current_base)?;
        }
    }

    Ok(result)
}

/// Return the dot product of two vectors, where the first is complex-conjugated.
pub fn vdot<T>(a: &Array<T>, b: &Array<T>) -> Result<T, NumPyError>
where
    T: LinalgScalar,
{
    if a.size() != b.size() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }
    let a_flat = a.to_vec();
    let b_flat = b.to_vec();
    let mut sum = T::zero();
    for i in 0..a_flat.len() {
        sum = sum + a_flat[i].conj() * b_flat[i];
    }
    Ok(sum)
}

/// Inner product of two arrays.
pub fn inner<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() == 0 || b.ndim() == 0 {
        return dot(a, b);
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();
    if a_shape[a_ndim - 1] != b_shape[b_ndim - 1] {
        return Err(NumPyError::shape_mismatch(
            a_shape.to_vec(),
            b_shape.to_vec(),
        ));
    }
    let last_dim = a_shape[a_ndim - 1];
    let a_outer_size = a.size() / last_dim;
    let b_outer_size = b.size() / last_dim;
    let mut res_data = Vec::with_capacity(a_outer_size * b_outer_size);
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    for i in 0..a_outer_size {
        for j in 0..b_outer_size {
            let mut sum = T::zero();
            for k in 0..last_dim {
                sum = sum + a_vec[i * last_dim + k] * b_vec[j * last_dim + k];
            }
            res_data.push(sum);
        }
    }
    let mut res_shape = a_shape[0..a_ndim - 1].to_vec();
    res_shape.extend_from_slice(&b_shape[0..b_ndim - 1]);
    if res_shape.is_empty() {
        Ok(Array::from_vec(res_data).reshape(&[])?)
    } else {
        Ok(Array::from_data(res_data, res_shape))
    }
}

/// Outer product of two vectors.
pub fn outer<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    let a_flat = a.to_vec();
    let b_flat = b.to_vec();
    let m = a_flat.len();
    let n = b_flat.len();
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(a_flat[i] * b_flat[j]);
        }
    }
    Ok(Array::from_data(data, vec![m, n]))
}

/// Kronecker product of two arrays.
pub fn kron<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();
    let max_ndim = a_ndim.max(b_ndim);

    let mut a_padded_shape = vec![1; max_ndim];
    let mut b_padded_shape = vec![1; max_ndim];

    for i in 0..a_ndim {
        a_padded_shape[max_ndim - 1 - i] = a.shape()[a_ndim - 1 - i];
    }
    for i in 0..b_ndim {
        b_padded_shape[max_ndim - 1 - i] = b.shape()[b_ndim - 1 - i];
    }

    let res_shape: Vec<usize> = a_padded_shape
        .iter()
        .zip(b_padded_shape.iter())
        .map(|(asize, bsize)| asize * bsize)
        .collect();

    let res_size = res_shape.iter().product();
    let mut res_data = Vec::with_capacity(res_size);

    let a_strides = crate::array::compute_strides(&a_padded_shape);
    let b_strides = crate::array::compute_strides(&b_padded_shape);

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    for i in 0..res_size {
        let res_idx = crate::strides::compute_multi_indices(i, &res_shape);
        let mut a_idx = vec![0; max_ndim];
        let mut b_idx = vec![0; max_ndim];
        for d in 0..max_ndim {
            a_idx[d] = res_idx[d] / b_padded_shape[d];
            b_idx[d] = res_idx[d] % b_padded_shape[d];
        }

        let a_lin = crate::strides::compute_linear_index(&a_idx, &a_strides);
        let b_lin = crate::strides::compute_linear_index(&b_idx, &b_strides);

        res_data.push(a_vec[a_lin as usize] * b_vec[b_lin as usize]);
    }

    Ok(Array::from_data(res_data, res_shape))
}

/// Compute the cross product of two (arrays of) 3D vectors.
pub fn cross<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if a.shape().last() != Some(&3) || b.shape().last() != Some(&3) {
        return Err(NumPyError::not_implemented(
            "cross currently only supports 3D vectors as the last dimension",
        ));
    }
    if a.shape() != b.shape() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let n = a_vec.len() / 3;
    let mut res_data = Vec::with_capacity(a_vec.len());
    for i in 0..n {
        let a1 = a_vec[i * 3 + 0];
        let a2 = a_vec[i * 3 + 1];
        let a3 = a_vec[i * 3 + 2];
        let b1 = b_vec[i * 3 + 0];
        let b2 = b_vec[i * 3 + 1];
        let b3 = b_vec[i * 3 + 2];
        res_data.push(a2 * b3 - a3 * b2);
        res_data.push(a3 * b1 - a1 * b3);
        res_data.push(a1 * b2 - a2 * b1);
    }
    Ok(Array::from_data(res_data, a.shape().to_vec()))
}

/// Compute the dot product of two or more matrices in a single function call,
/// while automatically selecting the fastest evaluation order.
pub fn multi_dot<T>(arrays: &[Array<T>]) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    let n = arrays.len();
    if n == 0 {
        return Err(NumPyError::invalid_operation(
            "multi_dot requires at least one array",
        ));
    }
    if n == 1 {
        return Ok(arrays[0].clone());
    }

    // Prepare all arrays as 2D. We own them if they were reshaped.
    let mut prepared = Vec::with_capacity(n);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.ndim() == 1 {
            if idx == 0 {
                prepared.push(arr.reshape(&[1, arr.size()])?);
            } else if idx == n - 1 {
                prepared.push(arr.reshape(&[arr.size(), 1])?);
            } else {
                return Err(NumPyError::invalid_operation(
                    "multi_dot: only first and last arrays can be 1D",
                ));
            }
        } else if arr.ndim() == 2 {
            prepared.push(arr.clone());
        } else {
            return Err(NumPyError::invalid_operation(
                "multi_dot only supports 2D arrays (and 1D at ends)",
            ));
        }
    }

    // Check dimensions
    for i in 0..n - 1 {
        if prepared[i].shape()[1] != prepared[i + 1].shape()[0] {
            return Err(NumPyError::shape_mismatch(
                prepared[i].shape().to_vec(),
                prepared[i + 1].shape().to_vec(),
            ));
        }
    }

    if n == 2 {
        let res = dot(&prepared[0], &prepared[1])?;
        return handle_multi_dot_squeezing(res, arrays[0].ndim() == 1, arrays[n - 1].ndim() == 1);
    }

    let mut dims = Vec::with_capacity(n + 1);
    dims.push(prepared[0].shape()[0]);
    for arr in &prepared {
        dims.push(arr.shape()[1]);
    }

    let s = matrix_chain_order(&dims);
    let res = multi_dot_impl(&prepared, &s, 0, n - 1)?;

    handle_multi_dot_squeezing(res, arrays[0].ndim() == 1, arrays[n - 1].ndim() == 1)
}

fn matrix_chain_order(dims: &[usize]) -> Vec<Vec<usize>> {
    let n = dims.len() - 1;
    let mut m = vec![vec![0; n]; n];
    let mut s = vec![vec![0; n]; n];

    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            m[i][j] = usize::MAX;
            for k in i..j {
                let cost = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1];
                if cost < m[i][j] {
                    m[i][j] = cost;
                    s[i][j] = k;
                }
            }
        }
    }
    s
}

fn multi_dot_impl<T>(
    arrays: &[Array<T>],
    s: &[Vec<usize>],
    i: usize,
    j: usize,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if i == j {
        Ok(arrays[i].clone())
    } else {
        let k = s[i][j];
        let left = multi_dot_impl(arrays, s, i, k)?;
        let right = multi_dot_impl(arrays, s, k + 1, j)?;
        dot(&left, &right)
    }
}

fn handle_multi_dot_squeezing<T>(
    res: Array<T>,
    first_1d: bool,
    last_1d: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if first_1d && last_1d {
        res.reshape(&[])
    } else if first_1d {
        let k = res.shape()[1];
        res.reshape(&[k])
    } else if last_1d {
        let k = res.shape()[0];
        res.reshape(&[k])
    } else {
        Ok(res)
    }
}
