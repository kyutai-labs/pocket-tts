// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Einstein summation (einsum) implementation
//!
//! Provides a powerful notation for array operations including
//! matrix multiplication, trace, transpose, and outer products.

use crate::array::Array;
use crate::error::{NumPyError, Result};
use num_traits::{Float, One, Zero};
use std::collections::HashMap;

/// Einstein summation notation parser and executor
///
/// # Examples
///
/// ```rust,ignore
/// use rust_numpy::{array, einsum};
///
/// let a = array![[1, 2], [3, 4]];
/// let b = array![[5, 6], [7, 8]];
///
/// // Matrix multiplication: 'ij,jk->ik'
/// let result = einsum("ij,jk->ik", &[&a, &b]).unwrap();
/// ```
///
/// # Notation
///
/// - Each operand is represented by subscript labels (e.g., 'ij', 'jk')
/// - Repeated labels across operands indicate summation (contraction)
/// - The output labels are specified after '->'
/// - If '->' is omitted, output is all non-summed labels in alphabetical order
///
/// # Common operations
///
/// - Matrix multiplication: `'ij,jk->ik'`
/// - Diagonal extraction: `'ii->i'`
/// - Trace: `'ii->'`
/// - Transpose: `'ij->ji'`
/// - Outer product: `'i,j->ij'`
pub fn einsum<T>(subscripts: &str, operands: &[&Array<T>]) -> Result<Array<T>>
where
    T: Copy
        + Clone
        + Default
        + Zero
        + One
        + Float
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static,
{
    // Parse the subscripts
    let parsed = parse_subscripts(subscripts)?;

    // Validate number of operands
    if operands.len() != parsed.input_labels.len() {
        return Err(NumPyError::invalid_value(format!(
            "Expected {} operands, got {}",
            parsed.input_labels.len(),
            operands.len()
        )));
    }

    // Validate each operand's dimensions match its labels
    for (idx, (labels, array)) in parsed.input_labels.iter().zip(operands.iter()).enumerate() {
        if array.ndim() != labels.len() {
            return Err(NumPyError::invalid_value(format!(
                "Operand {} has {} dimensions but subscript has {} labels",
                idx,
                array.ndim(),
                labels.len()
            )));
        }
    }

    // Build label dimension mapping
    let mut label_dims: HashMap<char, usize> = HashMap::new();
    for (labels, array) in parsed.input_labels.iter().zip(operands.iter()) {
        for (label, &dim) in labels.chars().zip(array.shape().iter()) {
            label_dims.insert(label, dim);
        }
    }

    // For basic operations, dispatch to specialized implementations
    if let Some(result) = try_specialized(subscripts, operands)? {
        return Ok(result);
    }

    // General einsum implementation
    general_einsum(&parsed, operands, &label_dims)
}

/// Parsed subscripts representation
#[derive(Debug, Clone)]
struct ParsedSubscripts {
    input_labels: Vec<String>,
    output_labels: Option<String>,
}

/// Parse Einstein summation notation
fn parse_subscripts(subscripts: &str) -> Result<ParsedSubscripts> {
    let parts: Vec<&str> = subscripts.split("->").collect();

    let output_labels = if parts.len() > 1 {
        Some(parts[1].trim().to_string())
    } else {
        None
    };

    let input_labels: Vec<String> = parts[0]
        .split(',')
        .map(|s| s.trim().chars().collect())
        .collect();

    // Validate that labels are single characters
    for labels in &input_labels {
        for ch in labels.chars() {
            if !ch.is_alphabetic() || !ch.is_ascii() {
                return Err(NumPyError::invalid_value(format!(
                    "Invalid label '{}' in subscripts",
                    ch
                )));
            }
        }
    }

    // Validate output labels if specified
    if let Some(ref output) = output_labels {
        for ch in output.chars() {
            if !ch.is_alphabetic() || !ch.is_ascii() {
                return Err(NumPyError::invalid_value(format!(
                    "Invalid output label '{}' in subscripts",
                    ch
                )));
            }
        }
    }

    Ok(ParsedSubscripts {
        input_labels,
        output_labels,
    })
}

/// Try to use specialized implementations for common patterns
fn try_specialized<T>(subscripts: &str, operands: &[&Array<T>]) -> Result<Option<Array<T>>>
where
    T: Copy
        + Clone
        + Default
        + Zero
        + One
        + Float
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static,
{
    match subscripts {
        // Matrix multiplication: 'ij,jk->ik'
        "ij,jk->ik" => {
            if operands.len() == 2 && operands[0].ndim() == 2 && operands[1].ndim() == 2 {
                let result = matmul_2d(operands[0], operands[1])?;
                return Ok(Some(result));
            }
        }
        // Trace: 'ii->'
        "ii->" => {
            if operands.len() == 1 && operands[0].ndim() == 2 {
                let result = trace(operands[0])?;
                return Ok(Some(result));
            }
        }
        // Diagonal extraction: 'ii->i'
        "ii->i" => {
            if operands.len() == 1 && operands[0].ndim() == 2 {
                let result = diagonal(operands[0])?;
                return Ok(Some(result));
            }
        }
        // Transpose: 'ij->ji'
        "ij->ji" => {
            if operands.len() == 1 && operands[0].ndim() == 2 {
                let result = transpose(operands[0])?;
                return Ok(Some(result));
            }
        }
        // Outer product: 'i,j->ij'
        "i,j->ij" => {
            if operands.len() == 2 && operands[0].ndim() == 1 && operands[1].ndim() == 1 {
                let result = outer_product(operands[0], operands[1])?;
                return Ok(Some(result));
            }
        }
        _ => {}
    }

    Ok(None)
}

/// General einsum implementation using naive contraction
fn general_einsum<T>(
    parsed: &ParsedSubscripts,
    _operands: &[&Array<T>],
    _label_dims: &HashMap<char, usize>,
) -> Result<Array<T>>
where
    T: Copy
        + Clone
        + Default
        + Zero
        + One
        + Float
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static,
{
    // For now, implement a simplified version that handles basic cases
    // This can be extended to handle more complex patterns

    // Determine which labels are summed (appear in inputs but not in output)
    let mut summed_labels: Vec<char> = vec![];
    for labels in &parsed.input_labels {
        for ch in labels.chars() {
            let in_output = parsed
                .output_labels
                .as_ref()
                .map(|out| out.contains(ch))
                .unwrap_or(false);
            let appears_once = parsed
                .input_labels
                .iter()
                .filter(|l| l.contains(ch))
                .count()
                == 1;

            if !in_output && !appears_once && !summed_labels.contains(&ch) {
                summed_labels.push(ch);
            }
        }
    }

    // For complex contractions, delegate to specialized implementations
    // This is a placeholder - a full implementation would handle arbitrary patterns
    Err(NumPyError::not_implemented(format!(
        "Complex einsum pattern not yet implemented: {:?}",
        parsed
    )))
}

/// 2D matrix multiplication for einsum
fn matmul_2d<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Copy
        + Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + 'static,
{
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[1] != b_shape[0] {
        return Err(NumPyError::invalid_value(format!(
            "Matrix multiplication dimension mismatch: {:?} vs {:?}",
            a_shape, b_shape
        )));
    }

    let m = a_shape[0];
    let n = a_shape[1];
    let p = b_shape[1];

    let mut result_data = vec![T::zero(); m * p];

    for i in 0..m {
        for j in 0..p {
            let mut sum = T::zero();
            for k in 0..n {
                let a_val = a.get_linear(i * n + k).copied().unwrap_or(T::zero());
                let b_val = b.get_linear(k * p + j).copied().unwrap_or(T::zero());
                sum = sum + a_val * b_val;
            }
            result_data[i * p + j] = sum;
        }
    }

    Ok(Array::from_data(result_data, vec![m, p]))
}

/// Compute trace of a matrix
fn trace<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Copy + Clone + Default + Zero + std::ops::Add<Output = T> + 'static,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(NumPyError::invalid_value(format!(
            "Trace requires square matrix, got shape {:?}",
            shape
        )));
    }

    let mut sum = T::zero();
    for i in 0..shape[0] {
        if let Some(val) = a.get_linear(i * shape[1] + i) {
            sum = sum + *val;
        }
    }

    // Return as a 0-dimensional array (scalar)
    Ok(Array::from_data(vec![sum], vec![]))
}

/// Extract diagonal of a matrix
fn diagonal<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Copy + Clone + Default + 'static,
{
    let shape = a.shape();
    let n = shape[0].min(shape[1]);

    let mut diag_data = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(val) = a.get_linear(i * shape[1] + i) {
            diag_data.push(*val);
        }
    }

    Ok(Array::from_data(diag_data, vec![n]))
}

/// Transpose a matrix
fn transpose<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Copy + Clone + Default + 'static,
{
    let shape = a.shape();
    let rows = shape[0];
    let cols = shape[1];

    let mut result_data = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            if let Some(val) = a.get_linear(i * cols + j) {
                result_data.push(*val);
            }
        }
    }

    Ok(Array::from_data(result_data, vec![cols, rows]))
}

/// Compute outer product of two vectors
fn outer_product<T>(a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Copy + Clone + Default + std::ops::Mul<Output = T> + 'static,
{
    let a_len = a.size();
    let b_len = b.size();

    let mut result_data = Vec::with_capacity(a_len * b_len);
    for i in 0..a_len {
        if let Some(a_val) = a.get_linear(i) {
            for j in 0..b_len {
                if let Some(b_val) = b.get_linear(j) {
                    result_data.push(*a_val * *b_val);
                }
            }
        }
    }

    Ok(Array::from_data(result_data, vec![a_len, b_len]))
}

/// Compute the optimal contraction order for einsum
///
/// # Arguments
/// * `subscripts` - Einstein summation notation
/// * `operands` - Input arrays
///
/// # Returns
/// A list of tuples indicating the contraction order
pub fn einsum_path<T>(
    subscripts: &str,
    _operands: &[&Array<T>],
) -> Result<Vec<(String, Vec<usize>)>>
where
    T: Copy + 'static,
{
    // For now, return a simple left-to-right contraction order
    // A full implementation would use dynamic programming for optimization
    let parsed = parse_subscripts(subscripts)?;

    let mut path = Vec::new();
    for i in 0..parsed.input_labels.len() {
        path.push((format!("einsum_subscript_{}", i), vec![i]));
    }

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_subscripts() {
        let parsed = parse_subscripts("ij,jk->ik").unwrap();
        assert_eq!(parsed.input_labels, vec!["ij", "jk"]);
        assert_eq!(parsed.output_labels, Some("ik".to_string()));
    }

    #[test]
    fn test_parse_subscripts_no_output() {
        let parsed = parse_subscripts("ij,jk").unwrap();
        assert_eq!(parsed.input_labels, vec!["ij", "jk"]);
        assert_eq!(parsed.output_labels, None);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f64, 6.0, 7.0, 8.0];

        let a = Array::from_data(a_data, vec![2, 2]);
        let b = Array::from_data(b_data, vec![2, 2]);

        let result = einsum("ij,jk->ik", &[&a, &b]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let data = result.to_vec();
        assert!((data[0] - 19.0).abs() < 1e-10); // 1*5 + 2*7
        assert!((data[1] - 22.0).abs() < 1e-10); // 1*6 + 2*8
        assert!((data[2] - 43.0).abs() < 1e-10); // 3*5 + 4*7
        assert!((data[3] - 50.0).abs() < 1e-10); // 3*6 + 4*8
    }

    #[test]
    fn test_trace() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a = Array::from_data(data, vec![2, 2]);

        let result = einsum("ii->", &[&a]).unwrap();

        assert_eq!(result.shape(), &[]);
        let result_data = result.to_vec();
        assert!((result_data[0] - 5.0).abs() < 1e-10); // 1 + 4
    }

    #[test]
    fn test_diagonal() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a = Array::from_data(data, vec![2, 2]);

        let result = einsum("ii->i", &[&a]).unwrap();

        assert_eq!(result.shape(), &[2]);
        let result_data = result.to_vec();
        assert!((result_data[0] - 1.0).abs() < 1e-10);
        assert!((result_data[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a = Array::from_data(data, vec![2, 2]);

        let result = einsum("ij->ji", &[&a]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let result_data = result.to_vec();
        assert_eq!(result_data[0], 1.0);
        assert_eq!(result_data[1], 3.0);
        assert_eq!(result_data[2], 2.0);
        assert_eq!(result_data[3], 4.0);
    }

    #[test]
    fn test_outer_product() {
        let a_data = vec![1.0f64, 2.0];
        let b_data = vec![3.0f64, 4.0];

        let a = Array::from_data(a_data, vec![2]);
        let b = Array::from_data(b_data, vec![2]);

        let result = einsum("i,j->ij", &[&a, &b]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        let result_data = result.to_vec();
        assert_eq!(result_data[0], 3.0); // 1*3
        assert_eq!(result_data[1], 4.0); // 1*4
        assert_eq!(result_data[2], 6.0); // 2*3
        assert_eq!(result_data[3], 8.0); // 2*4
    }

    #[test]
    fn test_invalid_operand_count() {
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let a = Array::from_data(a_data, vec![2, 2]);

        let result = einsum("ij,jk->ik", &[&a]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_path() {
        let a_data = vec![1.0f64, 2.0];
        let b_data = vec![3.0f64, 4.0];

        let a = Array::from_data(a_data, vec![2]);
        let b = Array::from_data(b_data, vec![2]);

        let path = einsum_path("i,j->ij", &[&a, &b]).unwrap();
        assert!(!path.is_empty());
    }
}
