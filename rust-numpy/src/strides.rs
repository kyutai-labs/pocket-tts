/// Compute strides for C-contiguous array layout
pub fn compute_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![0; shape.len()];
    strides[shape.len() - 1] = 1;

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }

    strides
}

/// Compute strides for Fortran-contiguous array layout
pub fn compute_fortran_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![0; shape.len()];
    strides[0] = 1;

    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }

    strides
}

/// Check if array is C-contiguous
pub fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    if shape.len() != strides.len() {
        return false;
    }

    let expected_strides = compute_strides(shape);
    strides == &expected_strides[..]
}

/// Check if array is Fortran-contiguous
pub fn is_f_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    if shape.len() != strides.len() {
        return false;
    }

    let expected_strides = compute_fortran_strides(shape);
    strides == &expected_strides[..]
}

/// Check if strides are contiguous in either C or Fortran order
pub fn is_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    is_c_contiguous(shape, strides) || is_f_contiguous(shape, strides)
}

/// Compute linear index from multi-dimensional indices
pub fn compute_linear_index(indices: &[usize], strides: &[isize]) -> isize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(i, s)| *i as isize * *s)
        .sum()
}

/// Compute multi-dimensional indices from linear index
pub fn compute_multi_indices(linear_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = linear_index;

    for (i, &dim_size) in shape.iter().enumerate().rev() {
        indices[i] = remaining % dim_size;
        remaining /= dim_size;
    }

    indices
}

/// Check if two shapes are broadcastable
pub fn are_shapes_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);

    for i in 0..max_len {
        let dim1 = if i >= max_len - len1 {
            shape1[i - (max_len - len1)]
        } else {
            1
        };

        let dim2 = if i >= max_len - len2 {
            shape2[i - (max_len - len2)]
        } else {
            1
        };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}

/// Compute broadcasted shape
pub fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);
    let mut result = vec![0; max_len];

    for i in 0..max_len {
        let dim1 = if i >= max_len - len1 {
            shape1[i - (max_len - len1)]
        } else {
            1
        };

        let dim2 = if i >= max_len - len2 {
            shape2[i - (max_len - len2)]
        } else {
            1
        };

        result[i] = std::cmp::max(dim1, dim2);
    }

    result
}

/// Compute broadcast strides for broadcasting
pub fn compute_broadcast_strides(
    original_shape: &[usize],
    original_strides: &[isize],
    broadcast_shape: &[usize],
) -> Vec<isize> {
    let orig_len = original_shape.len();
    let broadcast_len = broadcast_shape.len();
    let mut result = vec![0; broadcast_len];

    for (i, item) in result.iter_mut().enumerate().take(broadcast_len) {
        if i >= broadcast_len - orig_len {
            let orig_idx = i - (broadcast_len - orig_len);
            let orig_dim = original_shape[orig_idx];

            if orig_dim == 1 {
                *item = 0; // Broadcast dimension
            } else {
                *item = original_strides[orig_idx];
            }
        } else {
            *item = 0; // New dimension being broadcast
        }
    }

    result
}

/// Validate strides for a given shape
pub fn validate_strides(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }

    // Check that all dimensions are consistent
    for (dim_size, stride) in shape.iter().zip(strides.iter()) {
        if *dim_size == 0 {
            return *stride == 0;
        } else if *dim_size == 1 {
            // Stride can be anything for size-1 dimensions
            continue;
        } else {
            // For larger dimensions, stride should be reasonable
            if *stride < 0 {
                // Negative strides are allowed but need special handling
                continue;
            }
        }
    }

    true
}

/// Get the order (C or F) of the strides
pub fn stride_order(shape: &[usize], strides: &[isize]) -> StrideOrder {
    if is_c_contiguous(shape, strides) {
        StrideOrder::C
    } else if is_f_contiguous(shape, strides) {
        StrideOrder::F
    } else {
        StrideOrder::Neither
    }
}

/// Order of strides
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrideOrder {
    C,       // C-contiguous (row-major)
    F,       // Fortran-contiguous (column-major)
    Neither, // Neither C nor F contiguous
}
