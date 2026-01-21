use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;

/// Dot product of two arrays.
/// Currently supports 2D matrix multiplication.
pub fn dot<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NumPyError::not_implemented(
            "dot only supports 2D arrays currently",
        ));
    }

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
                let a_idx = (i as isize * a_strides[0] + k as isize * a_strides[1]) as usize;
                let b_idx = (k as isize * b_strides[0] + j as isize * b_strides[1]) as usize;

                let a_val = a
                    .get(a_idx)
                    .ok_or_else(|| NumPyError::invalid_operation("dot index out of bounds"))?;
                let b_val = b
                    .get(b_idx)
                    .ok_or_else(|| NumPyError::invalid_operation("dot index out of bounds"))?;

                sum = sum + (*a_val * *b_val);
            }

            let out_idx =
                (i as isize * output_strides[0] + j as isize * output_strides[1]) as usize;
            output.set(out_idx, sum)?;
        }
    }

    Ok(output)
}

/// Matrix multiplication (same as dot for 2D)
pub fn matmul<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar,
{
    dot(a, b)
}

/// Tensor dot product for higher-dimensional arrays
/// Uses reshape + matmul + reshape strategy for >2D arrays
pub fn tensor_dot<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + Clone + Default + 'static,
{
    // For 1D and 2D arrays, use existing dot function
    if a.ndim() <= 2 && b.ndim() <= 2 {
        return dot(a, b);
    }

    // For >2D arrays, use reshape strategy
    // Strategy: reshape to 2D, multiply, reshape back
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    // Reshape a to 2D: (a.shape[0..-2], a.shape[-2] * a.shape[-1])
    let a_shape = a.shape();
    let a_batch_dims = if a_ndim >= 2 {
        &a_shape[0..a_ndim - 2]
    } else {
        &[]
    };
    let a_rows = *a_shape.last().unwrap_or(&1);
    let a_cols = *a_shape.get(a_ndim.saturating_sub(1)).unwrap_or(&1);

    let a_2d_shape: Vec<usize> = a_batch_dims
        .iter()
        .cloned()
        .chain([a_rows * a_cols].iter().cloned())
        .collect();

    let a_2d = a.reshape(&a_2d_shape)?;

    // Reshape b to 2D: (b.shape[0..-2], b.shape[-2] * b.shape[-1])
    let b_shape = b.shape();
    let b_batch_dims = if b_ndim >= 2 {
        &b_shape[0..b_ndim - 2]
    } else {
        &[]
    };
    let b_rows = *b_shape.get(b_ndim.saturating_sub(2)).unwrap_or(&1);
    let b_cols = *b_shape.last().unwrap_or(&1);

    let b_2d_shape: Vec<usize> = b_batch_dims
        .iter()
        .cloned()
        .chain([b_rows * b_cols].iter().cloned())
        .collect();

    let b_2d = b.reshape(&b_2d_shape)?;

    // Perform matrix multiplication on 2D arrays
    let result_2d = dot(&a_2d, &b_2d)?;

    // Compute output shape: (batch_dims, a_rows, b_cols)
    let mut output_shape: Vec<usize> = a_batch_dims.to_vec();
    output_shape.push(a_rows);
    output_shape.push(b_cols);

    result_2d.reshape(&output_shape)
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
