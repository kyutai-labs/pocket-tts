use crate::dtype::{Dtype, DtypeKind};

/// Promote two dtypes to a common dtype that can safely hold values of both.
///
/// This implements logic similar to NumPy's `result_type`.
/// - Bool -> Integer -> Float -> Complex
/// - Size increases to max of both (e.g. i8 + i32 -> i32)
/// - Mixed Signed/Unsigned:
///   - Same size: Signed wins (u8 + i8 -> i16 to be safe? NumPy does i16)
///   - Different size: Largest wins, but if unsigned is larger, might need next size up signed.
///     - u8 + i16 -> i16
///     - u32 + i16 -> i64 (to hold u32 range)
///     - u64 + i64 -> float64 (cannot comfortably fit in i64)
pub fn promote_types(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    if t1 == t2 {
        return Some(t1.clone());
    }

    let k1 = t1.kind();
    let k2 = t2.kind();

    use DtypeKind::*;

    // Hierarchy of kinds
    let score = |k: &DtypeKind| match k {
        Bool => 0,
        Integer => 1,
        Unsigned => 1, // Same level, handled specifically
        Float => 2,
        Complex => 3,
        Datetime => 4,
        String => 7, // String > Bytes for promotion (e.g. U + S -> U)
        Bytes => 6,
        Object => 10,
        _ => 20,
    };

    let s1 = score(&k1);
    let s2 = score(&k2);

    // If kinds differ significantly (e.g. Int vs Float), pick higher kind
    if s1 != s2 {
        let (lower, higher_type) = if s1 < s2 { (t1, t2) } else { (t2, t1) };
        let higher_kind = higher_type.kind();

        // If higher is float/complex, we usually just take the higher type's size,
        // unless lower type is actually larger?
        // e.g. i64 + f32 -> f64 (NumPy)
        // Check special case: if integer meets float/complex
        if matches!(lower.kind(), Integer | Unsigned | Bool)
            && matches!(higher_kind, Float | Complex)
        {
            return Some(promote_int_float_complex(lower, higher_type));
        }

        // Default: use the higher kind
        return Some(higher_type.clone());
    }

    // Same kind group
    match (k1, k2) {
        (Bool, Bool) => Some(Dtype::Bool),
        (Integer, Integer) => promote_signed(t1, t2),
        (Unsigned, Unsigned) => promote_unsigned(t1, t2),
        (Integer, Unsigned) => promote_mixed_int(t1, t2),
        (Unsigned, Integer) => promote_mixed_int(t2, t1), // swap
        (Float, Float) => promote_float(t1, t2),
        (Complex, Complex) => promote_complex(t1, t2),
        (Datetime, Datetime) => promote_datetime(t1, t2),
        (String, String) => {
            // Check if either is Unicode
            let is_unicode =
                matches!(t1, Dtype::Unicode { .. }) || matches!(t2, Dtype::Unicode { .. });

            let l1 = match t1 {
                Dtype::String { length } => length.unwrap_or(0),
                Dtype::Unicode { length } => length.unwrap_or(0),
                _ => 0,
            };
            let l2 = match t2 {
                Dtype::String { length } => length.unwrap_or(0),
                Dtype::Unicode { length } => length.unwrap_or(0),
                _ => 0,
            };

            let max_len = l1.max(l2);

            if is_unicode {
                Some(Dtype::Unicode {
                    length: Some(max_len),
                })
            } else {
                Some(Dtype::String {
                    length: Some(max_len),
                })
            }
        }
        (Bytes, Bytes) => {
            let l1 = match t1 {
                Dtype::Bytes { length } => *length,
                _ => 0,
            };
            let l2 = match t2 {
                Dtype::Bytes { length } => *length,
                _ => 0,
            };
            Some(Dtype::Bytes { length: l1.max(l2) })
        }
        _ => {
            if t1 == t2 {
                Some(t1.clone())
            } else {
                Some(Dtype::Object) // Fallback to Object if mixed and no rule
            }
        }
    }
}

/// Promote multiple types to a common result type
pub fn result_type(types: &[&Dtype]) -> Option<Dtype> {
    if types.is_empty() {
        return None;
    }
    let mut res = types[0].clone();
    for i in 1..types.len() {
        res = promote_types(&res, types[i])?;
    }
    Some(res)
}

fn promote_int_float_complex(int_dtype: &Dtype, float_complex_dtype: &Dtype) -> Dtype {
    let f_size = float_complex_dtype.itemsize();
    let i_size = int_dtype.itemsize();

    // NumPy logic:
    // i8/i16/u8/u16 + f32 -> f32
    // i32/u32 + f32 -> f64 (f32 only has 24 bits of precision)
    // i64/u64 + f32/f64 -> f64

    if float_complex_dtype.kind() == DtypeKind::Float {
        if f_size < 8 && i_size >= 4 {
            return Dtype::Float64 { byteorder: None };
        }
        float_complex_dtype.clone()
    } else {
        // Complex
        // c8 (2x f32) + i32 -> c16 (2x f64)
        if f_size < 16 && i_size >= 4 {
            return Dtype::Complex128 { byteorder: None };
        }
        float_complex_dtype.clone()
    }
}

fn promote_signed(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

fn promote_unsigned(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

fn promote_mixed_int(signed: &Dtype, unsigned: &Dtype) -> Option<Dtype> {
    let s_signed = signed.itemsize();
    let s_unsigned = unsigned.itemsize();

    // If signed type is strictly larger than unsigned, it can hold it (e.g. i16 can hold u8)
    if s_signed > s_unsigned {
        return Some(signed.clone());
    }

    // If unsigned is same or larger, we need a larger signed type
    // e.g. i32 + u32 -> i64
    // i64 + u64 -> float64 (NumPy fallback when it cannot fit in signed integer)
    if s_unsigned >= 8 {
        return Some(Dtype::Float64 { byteorder: None });
    }

    size_to_signed(s_unsigned * 2)
}

fn promote_float(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

fn promote_complex(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

/// Promote types specifically for division operations.
///
/// NumPy's `true_divide` (/) always promotes to at least float64 (or float32).
pub fn promote_types_division(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let base = promote_types(t1, t2)?;
    match base.kind() {
        DtypeKind::Integer | DtypeKind::Unsigned | DtypeKind::Bool => {
            Some(Dtype::Float64 { byteorder: None })
        }
        _ => Some(base),
    }
}

/// Promote types for bitwise operations.
///
/// Only valid for integer and boolean types.
pub fn promote_types_bitwise(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    use DtypeKind::*;
    match (t1.kind(), t2.kind()) {
        (Integer | Unsigned | Bool, Integer | Unsigned | Bool) => promote_types(t1, t2),
        _ => None,
    }
}

fn promote_datetime(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    // For datetime/timedelta, NumPy usually takes the finer unit or errors if incompatible
    // For simplicity, we'll return the first one if they match, or error for now
    if t1 == t2 {
        Some(t1.clone())
    } else {
        None
    }
}

fn size_to_signed(size: usize) -> Option<Dtype> {
    match size {
        1 => Some(Dtype::Int8 { byteorder: None }),
        2 => Some(Dtype::Int16 { byteorder: None }),
        4 => Some(Dtype::Int32 { byteorder: None }),
        8 => Some(Dtype::Int64 { byteorder: None }),
        _ => Some(Dtype::Float64 { byteorder: None }), // Fallback
    }
}

fn size_to_unsigned(size: usize) -> Option<Dtype> {
    match size {
        1 => Some(Dtype::UInt8 { byteorder: None }),
        2 => Some(Dtype::UInt16 { byteorder: None }),
        4 => Some(Dtype::UInt32 { byteorder: None }),
        8 => Some(Dtype::UInt64 { byteorder: None }),
        _ => None,
    }
}
