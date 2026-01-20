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
