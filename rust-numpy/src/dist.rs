use crate::array::Array;
use crate::error::NumPyError;
use num_traits::Float;
use std::f64;

/// Compute pairwise distances between observations in n-dimensional space.
///
/// See `scipy.spatial.distance.pdist` for details.
///
/// # Arguments
///
/// * `X` - An m by n array of m original observations in an n-dimensional space.
/// * `metric` - The distance metric to use. Default is "euclidean".
///
/// # Returns
///
/// Returns a condensed distance matrix Y. For each i and j (where i < j < m),
/// where m is the number of original observations. The metric dist(u=X[i], v=X[j])
/// is computed and stored in entry ij.
pub fn pdist<T: Float + Into<f64> + Copy>(
    x: &Array<T>,
    metric: &str,
) -> Result<Array<f64>, NumPyError> {
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(NumPyError::invalid_value("X must be a 2D array"));
    }

    let m = shape[0];
    let n = shape[1];
    let num_pairs = m * (m - 1) / 2;
    let mut y = Vec::with_capacity(num_pairs);

    let data = x.to_vec(); // Simplified for now, should access directly or use strides

    for i in 0..m {
        for j in (i + 1)..m {
            let u_start = i * n;
            let v_start = j * n;
            let u = &data[u_start..u_start + n];
            let v = &data[v_start..v_start + n];

            let dist = compute_distance(u, v, metric)?;
            y.push(dist);
        }
    }

    Ok(Array::from_vec(y))
}

/// Compute distance between each pair of the two collections of inputs.
///
/// See `scipy.spatial.distance.cdist` for details.
pub fn cdist<T: Float + Into<f64> + Copy>(
    xa: &Array<T>,
    xb: &Array<T>,
    metric: &str,
) -> Result<Array<f64>, NumPyError> {
    let shape_a = xa.shape();
    let shape_b = xb.shape();

    if shape_a.len() != 2 || shape_b.len() != 2 {
        return Err(NumPyError::invalid_value("XA and XB must be 2D arrays"));
    }

    if shape_a[1] != shape_b[1] {
        return Err(NumPyError::invalid_value(
            "XA and XB must have the same number of columns",
        ));
    }

    let ma = shape_a[0];
    let mb = shape_b[0];
    let n = shape_a[1];

    let mut y = Vec::with_capacity(ma * mb);

    // Simplification: assuming contiguous implementations or using to_vec
    let data_a = xa.to_vec();
    let data_b = xb.to_vec();

    for i in 0..ma {
        for j in 0..mb {
            let u_start = i * n;
            let v_start = j * n;
            let u = &data_a[u_start..u_start + n];
            let v = &data_b[v_start..v_start + n];

            let dist = compute_distance(u, v, metric)?;
            y.push(dist);
        }
    }

    Ok(Array::from_shape_vec(vec![ma, mb], y))
}

/// Converts a vector-form distance vector to a square-form distance matrix, and vice-versa.
pub fn squareform(x: &Array<f64>) -> Result<Array<f64>, NumPyError> {
    let shape = x.shape();

    if shape.len() == 1 {
        // Vector to Matrix
        let len = shape[0];
        // m * (m - 1) / 2 = len
        // m^2 - m - 2*len = 0
        // m = (1 + sqrt(1 + 8*len)) / 2
        let discriminant = 1.0 + 8.0 * len as f64;
        let m_f64 = (1.0 + discriminant.sqrt()) / 2.0;

        if (m_f64.fract() > 1e-9) && ((m_f64.fract() - 1.0).abs() > 1e-9) {
            return Err(NumPyError::invalid_value(
                "Distance vector has incorrect length",
            ));
        }

        let m = m_f64.round() as usize;
        let mut matrix_data = vec![0.0; m * m];
        let vec_data = x.data();

        // Fill matrix
        let mut k = 0;
        for i in 0..m {
            for j in (i + 1)..m {
                if k < len {
                    let d = vec_data[k];
                    matrix_data[i * m + j] = d;
                    matrix_data[j * m + i] = d;
                    k += 1;
                }
            }
        }

        Ok(Array::from_shape_vec(vec![m, m], matrix_data))
    } else if shape.len() == 2 {
        // Matrix to Vector
        let m = shape[0];
        let n = shape[1];

        if m != n {
            return Err(NumPyError::invalid_value("Distance matrix must be square"));
        }

        let num_pairs = m * (m - 1) / 2;
        let mut vec_data = Vec::with_capacity(num_pairs);
        let matrix_data = x.data();

        for i in 0..m {
            for j in (i + 1)..m {
                vec_data.push(matrix_data[i * n + j]);
            }
        }

        Ok(Array::from_vec(vec_data))
    } else {
        Err(NumPyError::invalid_value("Input must be a 1D or 2D array"))
    }
}

fn compute_distance<T: Float + Into<f64> + Copy>(
    u: &[T],
    v: &[T],
    metric: &str,
) -> Result<f64, NumPyError> {
    match metric {
        "euclidean" => {
            let mut sum_sq = 0.0;
            for k in 0..u.len() {
                let diff = u[k].into() - v[k].into();
                sum_sq += diff * diff;
            }
            Ok(sum_sq.sqrt())
        }
        "sqeuclidean" => {
            let mut sum_sq = 0.0;
            for k in 0..u.len() {
                let diff = u[k].into() - v[k].into();
                sum_sq += diff * diff;
            }
            Ok(sum_sq)
        }
        "cityblock" | "manhattan" => {
            let mut sum_abs = 0.0;
            for k in 0..u.len() {
                let diff = (u[k].into() - v[k].into()).abs();
                sum_abs += diff;
            }
            Ok(sum_abs)
        }
        "chebyshev" => {
            let mut max_diff = 0.0f64;
            for k in 0..u.len() {
                let diff = (u[k].into() - v[k].into()).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            Ok(max_diff)
        }
        "cosine" => {
            let mut dot: f64 = 0.0;
            let mut norm_u: f64 = 0.0;
            let mut norm_v: f64 = 0.0;

            for k in 0..u.len() {
                let uk: f64 = u[k].into();
                let vk: f64 = v[k].into();
                dot += uk * vk;
                norm_u += uk * uk;
                norm_v += vk * vk;
            }

            if norm_u == 0.0 || norm_v == 0.0 {
                return Ok(f64::NAN); // Or handle as error? NumPy returns nan usually or 0 depending on version
                                     // Actually scipy.spatial.distance.cosine(u, v) is 1 - (u.v / (||u|| ||v||))
            }

            Ok(1.0 - (dot / (norm_u.sqrt() * norm_v.sqrt())))
        }
        _ => Err(NumPyError::invalid_value(format!(
            "Unknown metric: {}",
            metric
        ))),
    }
}
