use crate::error::NumPyError;

/// Character string operations on arrays
///
/// This module provides vectorized string operations similar to numpy.char
pub fn add(
    x1: &crate::Array<String>,
    x2: &crate::Array<String>,
) -> Result<crate::Array<String>, NumPyError> {
    if x1.shape() != x2.shape() {
        return Err(NumPyError::shape_mismatch(
            x1.shape().to_vec(),
            x2.shape().to_vec(),
        ));
    }

    let mut result = vec![String::new(); x1.size()];

    for i in 0..x1.size() {
        if let (Some(s1), Some(s2)) = (get_string(x1, i), get_string(x2, i)) {
            result.push(format!("{}{}", s1, s2));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn multiply(a: &crate::Array<String>, i: isize) -> Result<crate::Array<String>, NumPyError> {
    if i < 0 {
        return Err(NumPyError::invalid_value("i must be >= 0"));
    }

    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.repeat(i as usize));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn capitalize(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let mut chars: Vec<char> = s.chars().collect();
            if !chars.is_empty() {
                chars[0] = chars[0].to_ascii_uppercase();
                for c in chars.iter_mut().skip(1) {
                    *c = c.to_ascii_lowercase();
                }
            }
            result.push(chars.into_iter().collect());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn lower(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.to_lowercase());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn upper(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.to_uppercase());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn strip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    strip_chars(a, " \t\n\r")
}

pub fn lstrip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    lstrip_chars(a, " \t\n\r")
}

pub fn rstrip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    rstrip_chars(a, " \t\n\r")
}

pub fn strip_chars(
    a: &crate::Array<String>,
    chars: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_matches(|c| chars.contains(c)).to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn lstrip_chars(
    a: &crate::Array<String>,
    chars: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_start_matches(|c| chars.contains(c)).to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn rstrip_chars(
    a: &crate::Array<String>,
    chars: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_end_matches(|c| chars.contains(c)).to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn replace(
    a: &crate::Array<String>,
    old: &str,
    new: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.replace(old, new));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn split(a: &crate::Array<String>, sep: &str) -> Result<crate::Array<String>, NumPyError> {
    let mut all_results: Vec<String> = Vec::new();

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let parts: Vec<&str> = s.split(sep).collect();
            all_results.extend(parts.iter().map(|&p| p.to_string()));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(all_results))
}

pub fn join(sep: &str, a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(vec![result.join(sep)]))
}

pub fn startswith(
    a: &crate::Array<String>,
    prefix: &str,
) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = s.starts_with(prefix);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn endswith(a: &crate::Array<String>, suffix: &str) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = s.ends_with(suffix);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

fn get_string(a: &crate::Array<String>, idx: usize) -> Option<String> {
    a.get(idx).cloned()
}

pub mod exports {
    pub use super::{
        add, capitalize, endswith, join, lower, lstrip, lstrip_chars, multiply, replace, rstrip,
        rstrip_chars, split, startswith, strip, strip_chars, upper,
    };
}
