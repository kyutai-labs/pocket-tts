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
        Unsigned => 1, // Treat signed/unsigned roughly same level, handled specifically
        Float => 2,
        Complex => 3,
        _ => 4, // Other types usually don't mix or strictly equal
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
            // For consistency with NumPy:
            // Integers usually promote to at least Float64 if mixed with native python floats,
            // but here we are mixing Dtypes.
            // i16 + f32 -> f32
            // i64 + f32 -> f64
            return Some(promote_int_float(lower, higher_type));
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
        _ => None, // Strings, Objects, etc - strict matching usually required
    }
}

fn promote_int_float(int_dtype: &Dtype, float_dtype: &Dtype) -> Dtype {
    // If int size > float mantissa precision, might need larger float.
    // NumPy rule:
    // i8/i16 + f32 -> f32
    // i32 + f32 -> f64 (because i32 range > f32 exact integer precision? No, i32 fits in f64. f32 only has 24 bits significand)
    // Actually NumPy: float32(1) + int32(1) -> float64.
    // But float32 + int16 -> float32.

    // We'll simplistic heuristic: if float is less than f64 and int is >= 32 bit, go to f64.
    let f_size = float_dtype.itemsize();
    let i_size = int_dtype.itemsize();

    if f_size < 8 && i_size >= 4 {
        return Dtype::Float64 { byteorder: None };
    }

    float_dtype.clone()
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

    // Rule:
    // if signed is larger, it can likely hold unsigned (e.g. i64 > u32) -> Use signed.
    if s_signed > s_unsigned {
        return Some(signed.clone());
    }

    // if sizes equal: i32 + u32 -> i64.
    // if sizes max: i64 + u64 -> f64 (NumPy behavior).
    if s_signed == s_unsigned {
        if s_signed >= 8 {
            return Some(Dtype::Float64 { byteorder: None });
        }
        // Promote to next signed size
        return size_to_signed(s_signed * 2);
    }

    // if unsigned is larger: u64 + i32 -> f64? or try to find signed large enough?
    // u64 needs i128 (not standard supported yet) or f64.
    // u32 + i8 -> i64.
    if s_unsigned < 8 {
        return size_to_signed(s_unsigned * 2);
    }

    Some(Dtype::Float64 { byteorder: None })
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

fn size_to_signed(size: usize) -> Option<Dtype> {
    match size {
        1 => Some(Dtype::Int8 { byteorder: None }),
        2 => Some(Dtype::Int16 { byteorder: None }),
        4 => Some(Dtype::Int32 { byteorder: None }),
        8 => Some(Dtype::Int64 { byteorder: None }),
        _ => None,
    }
}
