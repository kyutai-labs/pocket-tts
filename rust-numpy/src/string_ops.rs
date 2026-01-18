use crate::error::NumPyError;
use crate::dtype::Dtype;

pub fn add<T>(x1: &crate::Array<T>, x2: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError> {
    if x1.shape() != x2.shape() {
        return Err(NumPyError::invalid_shape("Arrays must have same shape"));
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

pub fn multiply<T>(a: &crate::Array<T>, i: isize) -> Result<crate::Array<T>, NumPyError> {
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

pub fn center<T>(a: &crate::Array<T>, width: isize, fillchar: char) -> Result<crate::Array<T>, NumPyError> {
    let w = if width < 0 { 0 } else { width as usize };
    let fill = if fillchar == '\0' { ' ' } else { fillchar };
    
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let padding = w.saturating_sub(s.len());
            let left = padding / 2;
            let right = padding - left;
            result.push(format!("{}{}{}", fill.repeat(left), s, fill.repeat(right)));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn capitalize<T>(a: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError> {
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

pub fn title<T>(a: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError> {
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let mut words: Vec<String> = s.split_whitespace()
                .map(|w| {
                    let mut chars: Vec<char> = w.chars().collect();
                    if !chars.is_empty() {
                        chars[0] = chars[0].to_ascii_uppercase();
                    }
                    chars.into_iter().collect()
                })
                .collect();
            
            result.push(words.join(" "));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn lower<T>(a: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError> {
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

pub fn upper<T>(a: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError> {
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

pub fn swapcase<T>(a: &crate::Array<T>) -> Result<crate::Array<T>, NumPyError> {
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let swapped: String = s.chars()
                .map(|c| {
                    if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else if c.is_ascii_lowercase() {
                        c.to_ascii_uppercase()
                    } else {
                        *c
                    }
                })
                .collect();
            result.push(swapped);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn join<T>(sep: &str, a: &crate::Array<T>) -> Result<crate::Array<String>, NumPyError> {
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

pub fn split<T>(a: &crate::Array<T>, sep: &str, maxsplit: isize) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let parts: Vec<&str> = if maxsplit > 0 {
                s.split(sep).take(maxsplit as usize).collect()
            } else {
                s.split(sep).collect()
            };
            result.push(parts.join(sep));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn strip<T>(a: &crate::Array<T>, chars: &str) -> Result<crate::Array<T>, NumPyError> {
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_matches(|c| chars.contains(c)));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn lstrip<T>(a: &crate::Array<T>, chars: &str) -> Result<crate::Array<T>, NumPyError> {
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_start_matches(|c| chars.contains(c)));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn rstrip<T>(a: &crate::Array<T>, chars: &str) -> Result<crate::Array<T>, NumPyError> {
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_end_matches(|c| chars.contains(c)));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

fn get_string<T>(a: &crate::Array<T>, idx: usize) -> Option<String> {
    a.get(idx).and_then(|v| {
        unsafe {
            let s = std::mem::transmute::<T, String>(*v);
            Some(s)
        }
    })
}

pub fn count<T>(a: &crate::Array<T>, sub: &str, start: isize, end: isize) -> Result<crate::Array<isize>, NumPyError> {
    let mut result = vec![0isize; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let s_start = if start < 0 { 0 } else { start as usize };
            let s_end = if end < 0 { s.len() } else { end as usize };
            
            if s_start < s_end {
                let substr = &s[s_start..s_end.min(s.len())];
                result[idx] = substr.matches(sub).count() as isize;
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn find<T>(a: &crate::Array<T>, sub: &str, start: isize, end: isize) -> Result<crate::Array<isize>, NumPyError> {
    let mut result = vec![(-1isize); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let s_start = if start < 0 { 0 } else { start as usize };
            let s_end = if end < 0 { s.len() } else { end as usize };
            
            if s_start < s_end && s_end.min(s.len()) <= s.len() {
                let substr = &s[s_start..s_end.min(s.len())];
                if let Some(pos) = s[s_end.min(s.len())..].find(substr) {
                    result[idx] = pos as isize + s_end as isize;
                }
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn findall<T>(a: &crate::Array<T>, sub: &str) -> Result<crate::Array<isize>, NumPyError> {
    let mut all_results: Vec<Vec<isize>> = Vec::new();
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let mut positions: Vec<isize> = Vec::new();
            let mut start = 0;
            while let Some(pos) = s[start..].find(sub) {
                positions.push((start + pos) as isize);
                start = start + pos + sub.len();
            }
            all_results.push(positions);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    let max_len = all_results.iter().map(|v| v.len()).max().unwrap_or(0);
    let mut result = vec![-1isize; a.size() * max_len];
    
    for (i, positions) in all_results.iter().enumerate() {
        for (j, pos) in positions.iter().enumerate() {
            result[i * max_len + j] = *pos;
        }
    }
    
    Ok(crate::Array::from_shape_vec(vec![a.size(), max_len], result).unwrap())
}

pub fn isalpha<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.chars().all(|c| c.is_alphabetic());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn isalnum<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.chars().all(|c| c.is_alphanumeric());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn isdecimal<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.parse::<f64>().is_ok();
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn isdigit<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn islower<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.chars().all(|c| c.is_lowercase());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn isnumeric<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.chars().all(|c| c.is_numeric() || c == '.');
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn isspace<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.chars().all(|c| c.is_whitespace());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn istitle<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let words: Vec<&str> = s.split_whitespace().collect();
            result[idx] = !words.is_empty() && words.iter().all(|w| {
                let mut chars = w.chars();
                chars.next().map_or(false, |c| c.is_ascii_uppercase())
                    && chars.all(|c| c.is_lowercase())
            });
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn isupper<T>(a: &crate::Array<T>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = !s.is_empty() && s.chars().all(|c| c.is_ascii_uppercase());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn ljust<T>(a: &crate::Array<T>, width: isize, fillchar: char) -> Result<crate::Array<T>, NumPyError> {
    let w = if width < 0 { 0 } else { width as usize };
    let fill = if fillchar == '\0' { ' ' } else { fillchar };
    
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let padding = w.saturating_sub(s.len());
            result.push(format!("{}{}", s, fill.repeat(padding)));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn rjust<T>(a: &crate::Array<T>, width: isize, fillchar: char) -> Result<crate::Array<T>, NumPyError> {
    let w = if width < 0 { 0 } else { width as usize };
    let fill = if fillchar == '\0' { ' ' } else { fillchar };
    
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let padding = w.saturating_sub(s.len());
            result.push(format!("{}{}", fill.repeat(padding), s));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub fn zfill<T>(a: &crate::Array<T>, width: isize) -> Result<crate::Array<T>, NumPyError> {
    let w = if width < 0 { 0 } else { width as usize };
    
    let mut result = vec![String::new(); a.size()];
    
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.chars().map(|_| '0').collect());
            result[idx].truncate(w);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    
    Ok(crate::Array::from_vec(result))
}

pub mod exports {
    pub use super::{
        add, multiply, center, capitalize, title, lower, upper, swapcase, join, split,
        strip, lstrip, rstrip, count, find, findall, isalpha, isalnum, isdecimal,
        isdigit, islower, isnumeric, isspace, istitle, isupper, ljust, rjust, zfill
    };
}