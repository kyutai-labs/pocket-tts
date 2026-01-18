# Rust NumPy Dtype - Code Snippets

**Extracted from dtype.rs (358 lines)**

---

## DTYPE ENUM DEFINITION

```rust
#![allow(non_camel_case_types)]
use std::fmt;

/// Comprehensive dtype system matching NumPy's type system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dtype {
    // Signed integer types
    Int8,
    Int16,
    Int32,
    Int64,

    // Unsigned integer types
    UInt8,
    UInt16,
    UInt32,
    UInt64,

    // Floating point types
    Float16,
    Float32,
    Float64,

    // Complex types
    Complex32,
    Complex64,
    Complex128,

    // Boolean type
    Bool,

    // String types
    String,
    Unicode,

    // Datetime types
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),

    // Object type
    Object,

    // Structured type
    Struct(Vec<StructField>),
}
```

---

## DTYPEKIND ENUM

```rust
/// Kind of dtype for type checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtypeKind {
    Integer,
    Unsigned,
    Float,
    Complex,
    Bool,
    String,
    Datetime,
    Object,
    Struct,
}
```

---

## DATETIME UNIT ENUM

```rust
/// Units for datetime64
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatetimeUnit {
    Y,   // Year
    M,   // Month
    W,   // Week
    D,   // Day
    h,   // Hour
    m,   // Minute
    s,   // Second
    ms,  // Millisecond
    us,  // Microsecond
    ns,  // Nanosecond
    ps,  // Picosecond
    fs,  // Femtosecond
    As,  // Attosecond
}
```

---

## TIMEDELTA UNIT ENUM

```rust
/// Units for timedelta64
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimedeltaUnit {
    Y,   // Year
    M,   // Month
    W,   // Week
    D,   // Day
    h,   // Hour
    m,   // Minute
    s,   // Second
    ms,  // Millisecond
    us,  // Microsecond
    ns,  // Nanosecond
    ps,  // Picosecond
    fs,  // Femtosecond
    As,  // Attosecond
}
```

---

## STRUCTFIELD DEFINITION

```rust
/// Field definition for structured dtypes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub dtype: Dtype,
    pub offset: Option<usize>,
}
```

---

## DTYPE::KIND() METHOD

```rust
impl Dtype {
    /// Get the kind of this dtype
    pub fn kind(&self) -> DtypeKind {
        match self {
            Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => DtypeKind::Integer,
            Dtype::UInt8 | Dtype::UInt16 | Dtype::UInt32 | Dtype::UInt64 => DtypeKind::Unsigned,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 => DtypeKind::Float,
            Dtype::Complex32 | Dtype::Complex64 | Dtype::Complex128 => DtypeKind::Complex,
            Dtype::Bool => DtypeKind::Bool,
            Dtype::String | Dtype::Unicode => DtypeKind::String,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => DtypeKind::Datetime,
            Dtype::Object => DtypeKind::Object,
            Dtype::Struct(_) => DtypeKind::Struct,
        }
    }
}
```

---

## DTYPE::ITEMSIZE() METHOD

```rust
impl Dtype {
    /// Get size in bytes
    pub fn itemsize(&self) -> usize {
        match self {
            Dtype::Int8 | Dtype::UInt8 | Dtype::Bool => 1,
            Dtype::Int16 | Dtype::UInt16 | Dtype::Float16 => 2,
            Dtype::Int32 | Dtype::UInt32 | Dtype::Float32 | Dtype::Complex32 => 4,
            Dtype::Int64 | Dtype::UInt64 | Dtype::Float64 | Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
            Dtype::String | Dtype::Unicode => 8, // Pointer size
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => 8,
            Dtype::Object => 8, // Pointer size
            Dtype::Struct(fields) => fields.iter().map(|f| f.dtype.itemsize()).sum(),
        }
    }
}
```

---

## DTYPE::ALIGNMENT() METHOD

```rust
impl Dtype {
    /// Get alignment requirement in bytes
    pub fn alignment(&self) -> usize {
        match self {
            Dtype::Int8 | Dtype::UInt8 | Dtype::Bool => 1,
            Dtype::Int16 | Dtype::UInt16 | Dtype::Float16 => 2,
            Dtype::Int32 | Dtype::UInt32 | Dtype::Float32 | Dtype::Complex32 => 4,
            Dtype::Int64 | Dtype::UInt64 | Dtype::Float64 | Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
            Dtype::String | Dtype::Unicode => 8,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => 8,
            Dtype::Object => 8,
            Dtype::Struct(fields) => fields
                .iter()
                .map(|f| f.dtype.alignment())
                .max()
                .unwrap_or(1),
        }
    }
}
```

---

## DTYPE::FROM_TYPE<T>() METHOD

```rust
impl Dtype {
    /// Create dtype from Rust type
    pub fn from_type<T: 'static>() -> Self {
        use std::any::TypeId;

        let type_id = TypeId::of::<T>();

        if type_id == TypeId::of::<i8>() {
            Dtype::Int8
        } else if type_id == TypeId::of::<i16>() {
            Dtype::Int16
        } else if type_id == TypeId::of::<i32>() {
            Dtype::Int32
        } else if type_id == TypeId::of::<i64>() {
            Dtype::Int64
        } else if type_id == TypeId::of::<u8>() {
            Dtype::UInt8
        } else if type_id == TypeId::of::<u16>() {
            Dtype::UInt16
        } else if type_id == TypeId::of::<u32>() {
            Dtype::UInt32
        } else if type_id == TypeId::of::<u64>() {
            Dtype::UInt64
        } else if type_id == TypeId::of::<f16>() {
            Dtype::Float16
        } else if type_id == TypeId::of::<f32>() {
            Dtype::Float32
        } else if type_id == TypeId::of::<f64>() {
            Dtype::Float64
        } else if type_id == TypeId::of::<bool>() {
            Dtype::Bool
        } else {
            Dtype::Object
        }
    }
}
```

---

## DTYPE::FROM_STR() METHOD (FULL)

```rust
impl Dtype {
    /// Parse dtype from string (NumPy compatible)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "int8" | "i1" => Ok(Dtype::Int8),
            "int16" | "i2" => Ok(Dtype::Int16),
            "int32" | "i4" => Ok(Dtype::Int32),
            "int64" | "i8" => Ok(Dtype::Int64),
            "uint8" | "u1" => Ok(Dtype::UInt8),
            "uint16" | "u2" => Ok(Dtype::UInt16),
            "uint32" | "u4" => Ok(Dtype::UInt32),
            "uint64" | "u8" => Ok(Dtype::UInt64),
            "float16" | "f2" => Ok(Dtype::Float16),
            "float32" | "f4" => Ok(Dtype::Float32),
            "float64" | "f8" => Ok(Dtype::Float64),
            "complex64" | "c8" => Ok(Dtype::Complex64),
            "complex128" | "c16" => Ok(Dtype::Complex128),
            "bool" => Ok(Dtype::Bool),
            "str" => Ok(Dtype::String),
            "unicode" => Ok(Dtype::Unicode),
            "object" => Ok(Dtype::Object),
            _ => {
                // Try to parse datetime types
                if s.starts_with("datetime64") {
                    let unit = s
                        .strip_prefix("datetime64[")
                        .and_then(|s| s.strip_suffix("]"));
                    if let Some(unit_str) = unit {
                        match unit_str {
                            "Y" => Ok(Dtype::Datetime64(DatetimeUnit::Y)),
                            "M" => Ok(Dtype::Datetime64(DatetimeUnit::M)),
                            "W" => Ok(Dtype::Datetime64(DatetimeUnit::W)),
                            "D" => Ok(Dtype::Datetime64(DatetimeUnit::D)),
                            "h" => Ok(Dtype::Datetime64(DatetimeUnit::h)),
                            "m" => Ok(Dtype::Datetime64(DatetimeUnit::m)),
                            "s" => Ok(Dtype::Datetime64(DatetimeUnit::s)),
                            "ms" => Ok(Dtype::Datetime64(DatetimeUnit::ms)),
                            "us" => Ok(Dtype::Datetime64(DatetimeUnit::us)),
                            "ns" => Ok(Dtype::Datetime64(DatetimeUnit::ns)),
                            _ => Err(format!("Unknown datetime unit: {}", unit_str)),
                        }
                    } else {
                        Ok(Dtype::Datetime64(DatetimeUnit::ns)) // Default
                    }
                } else {
                    Err(format!("Unknown dtype: {}", s))
                }
            }
        }
    }
}
```

---

## DTYPE::TO_STRING() METHOD (FULL)

```rust
impl Dtype {
    /// Convert to string (NumPy compatible)
    pub fn to_string(&self) -> String {
        match self {
            Dtype::Int8 => "int8".to_string(),
            Dtype::Int16 => "int16".to_string(),
            Dtype::Int32 => "int32".to_string(),
            Dtype::Int64 => "int64".to_string(),
            Dtype::UInt8 => "uint8".to_string(),
            Dtype::UInt16 => "uint16".to_string(),
            Dtype::UInt32 => "uint32".to_string(),
            Dtype::UInt64 => "uint64".to_string(),
            Dtype::Float16 => "float16".to_string(),
            Dtype::Float32 => "float32".to_string(),
            Dtype::Float64 => "float64".to_string(),
            Dtype::Complex32 => "complex64".to_string(),  // BUG: Should be "complex32"
            Dtype::Complex64 => "complex64".to_string(),
            Dtype::Complex128 => "complex128".to_string(),
            Dtype::Bool => "bool".to_string(),
            Dtype::String => "str".to_string(),
            Dtype::Unicode => "unicode".to_string(),
            Dtype::Datetime64(unit) => format!(
                "datetime64[{}]",
                match unit {
                    DatetimeUnit::Y => "Y",
                    DatetimeUnit::M => "M",
                    DatetimeUnit::W => "W",
                    DatetimeUnit::D => "D",
                    DatetimeUnit::h => "h",
                    DatetimeUnit::m => "m",
                    DatetimeUnit::s => "s",
                    DatetimeUnit::ms => "ms",
                    DatetimeUnit::us => "us",
                    DatetimeUnit::ns => "ns",
                    DatetimeUnit::ps => "ps",
                    DatetimeUnit::fs => "fs",
                    DatetimeUnit::As => "as",
                }
            ),
            Dtype::Timedelta64(unit) => format!(
                "timedelta64[{}]",
                match unit {
                    TimedeltaUnit::Y => "Y",
                    TimedeltaUnit::M => "M",
                    TimedeltaUnit::W => "W",
                    TimedeltaUnit::D => "D",
                    TimedeltaUnit::h => "h",
                    TimedeltaUnit::m => "m",
                    TimedeltaUnit::s => "s",
                    TimedeltaUnit::ms => "ms",
                    TimedeltaUnit::us => "us",
                    TimedeltaUnit::ns => "ns",
                    TimedeltaUnit::ps => "ps",
                    TimedeltaUnit::fs => "fs",
                    TimedeltaUnit::As => "as",
                }
            ),
            Dtype::Object => "object".to_string(),
            Dtype::Struct(_) => "struct".to_string(),
        }
    }
}
```

---

## DTYPE::CAN_CAST_TO() METHOD

```rust
impl Dtype {
    /// Check if dtype can be safely cast to another dtype
    pub fn can_cast_to(&self, other: &Dtype) -> bool {
        use DtypeKind::*;

        let self_kind = self.kind();
        let other_kind = other.kind();

        match (self_kind, other_kind) {
            (Integer, Integer) | (Unsigned, Integer) | (Unsigned, Unsigned) => {
                self.itemsize() <= other.itemsize()
            }
            (Integer, Unsigned) => false,
            (Float, Float) => self.itemsize() <= other.itemsize(),
            (Complex, Complex) => self.itemsize() <= other.itemsize(),
            (Integer | Unsigned | Float, Complex) => true,
            (Complex, Float) => false,
            (Bool, _) => true,
            (_, Bool) => false,
            (String, String) | (Datetime, Datetime) | (Object, _) | (_, Object) => true,
            _ => false,
        }
    }
}
```

---

## F16 STRUCT DEFINITION

```rust
/// Half-precision float type
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct f16(u16);

impl f16 {
    pub fn new(value: f32) -> Self {
        // Simplified conversion - real implementation would be more complex
        Self(((value.to_bits()) >> 16) as u16)
    }

    pub fn to_f32(self) -> f32 {
        // Simplified conversion - real implementation would be more complex
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl From<f32> for f16 {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<f16> for f32 {
    fn from(value: f16) -> Self {
        value.to_f32()
    }
}
```

---

## DISPLAY IMPLEMENTATION

```rust
impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}
```

---

## USAGE IN ARRAY STRUCT

```rust
/// Multi-dimensional array structure mirroring NumPy's ndarray
pub struct Array<T> {
    data: Arc<MemoryManager<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,        // ← Dtype field
    offset: usize,
}

impl<T> Array<T> {
    /// Get dtype
    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }
}
```

---

## DTYPE USAGE IN UFUNC

```rust
pub trait Ufunc: Send + Sync {
    /// Get supported input types
    fn supported_dtypes(&self) -> &[DtypeKind];

    /// Check if ufunc supports given dtypes
    fn supports_dtypes(&self, dtypes: &[&Dtype]) -> bool {
        dtypes
            .iter()
            .all(|dt| self.supported_dtypes().contains(&dt.kind()))
    }
}
```

---

## EXAMPLE: CREATING STRUCTURED DTYPE

```rust
use crate::dtype::{Dtype, StructField};

let struct_dtype = Dtype::Struct(vec![
    StructField {
        name: "id".to_string(),
        dtype: Dtype::Int32,
        offset: Some(0),
    },
    StructField {
        name: "value".to_string(),
        dtype: Dtype::Float64,
        offset: Some(4),
    },
    StructField {
        name: "active".to_string(),
        dtype: Dtype::Bool,
        offset: Some(12),
    },
]);

println!("Struct dtype: {}", struct_dtype); // "struct"
println!("Item size: {} bytes", struct_dtype.itemsize()); // 13 bytes
println!("Alignment: {} bytes", struct_dtype.alignment()); // 8 bytes
```

---

## EXAMPLE: PARSING DTYPE FROM STRING

```rust
use crate::dtype::Dtype;

// Integer types
let int32 = Dtype::from_str("int32").unwrap();
let i4 = Dtype::from_str("i4").unwrap(); // Same as int32

// Float types
let float64 = Dtype::from_str("float64").unwrap();
let f8 = Dtype::from_str("f8").unwrap(); // Same as float64

// Complex types
let complex128 = Dtype::from_str("complex128").unwrap();
let c16 = Dtype::from_str("c16").unwrap(); // Same as complex128

// Datetime types
let datetime_ns = Dtype::from_str("datetime64[ns]").unwrap();
let datetime_s = Dtype::from_str("datetime64[s]").unwrap();
let datetime_default = Dtype::from_str("datetime64").unwrap(); // Uses ns

// Error handling
let invalid = Dtype::from_str("invalid_type");
assert!(invalid.is_err());
```

---

## EXAMPLE: CHECKING DTYPE PROPERTIES

```rust
use crate::dtype::{Dtype, DtypeKind};

let dt = Dtype::Int32;

// Get kind
let kind = dt.kind();
assert_eq!(kind, DtypeKind::Integer);

// Get item size
let size = dt.itemsize();
assert_eq!(size, 4);

// Get alignment
let align = dt.alignment();
assert_eq!(align, 4);

// Check casting safety
assert!(dt.can_cast_to(&Dtype::Int64));  // true
assert!(!dt.can_cast_to(&Dtype::Int8)); // false
assert!(dt.can_cast_to(&Dtype::Float64)); // true
assert!(!dt.can_cast_to(&Dtype::Bool));  // false
```

---

## EXAMPLE: INFERRING DTYPE FROM RUST TYPES

```rust
use crate::dtype::Dtype;

// Infer from Rust types
let int32_dtype = Dtype::from_type::<i32>();
let float64_dtype = Dtype::from_type::<f64>();
let bool_dtype = Dtype::from_type::<bool>();

assert_eq!(int32_dtype, Dtype::Int32);
assert_eq!(float64_dtype, Dtype::Float64);
assert_eq!(bool_dtype, Dtype::Bool);

// Unknown type falls back to Object
let string_dtype = Dtype::from_type::<String>();
assert_eq!(string_dtype, Dtype::Object);
```

---

## EXAMPLE: DTYPE STRING CONVERSION

```rust
use crate::dtype::{Dtype, DatetimeUnit};

let dt = Dtype::Int32;
assert_eq!(dt.to_string(), "int32");

let dt = Dtype::Float64;
assert_eq!(dt.to_string(), "float64");

let dt = Dtype::Datetime64(DatetimeUnit::ns);
assert_eq!(dt.to_string(), "datetime64[ns]");

let dt = Dtype::Datetime64(DatetimeUnit::ms);
assert_eq!(dt.to_string(), "datetime64[ms]");
```

---

## EXAMPLE: CASTING RULES

```rust
use crate::dtype::Dtype;

// Safe casts (same kind, larger size)
assert!(Dtype::Int8.can_cast_to(&Dtype::Int16));
assert!(Dtype::Int16.can_cast_to(&Dtype::Int32));
assert!(Dtype::Int32.can_cast_to(&Dtype::Int64));

// Unsafe casts (same kind, smaller size)
assert!(!Dtype::Int64.can_cast_to(&Dtype::Int32));
assert!(!Dtype::Int32.can_cast_to(&Dtype::Int16));

// Unsigned to signed (safe for same/larger size)
assert!(Dtype::UInt8.can_cast_to(&Dtype::Int16));
assert!(!Dtype::UInt8.can_cast_to(&Dtype::Int8));

// Float to larger float
assert!(Dtype::Float32.can_cast_to(&Dtype::Float64));
assert!(!Dtype::Float64.can_cast_to(&Dtype::Float32));

// Numeric to complex (always safe)
assert!(Dtype::Int32.can_cast_to(&Dtype::Complex64));
assert!(Dtype::Float32.can_cast_to(&Dtype::Complex128));

// Complex to float (never safe)
assert!(!Dtype::Complex64.can_cast_to(&Dtype::Float32));

// Bool to anything (always safe)
assert!(Dtype::Bool.can_cast_to(&Dtype::Int32));
assert!(Dtype::Bool.can_cast_to(&Dtype::Float64));

// Anything to Bool (never safe except Bool)
assert!(!Dtype::Int32.can_cast_to(&Dtype::Bool));
assert!(Dtype::Bool.can_cast_to(&Dtype::Bool));
```

---

## TEST EXAMPLES

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_creation() {
        let dt = Dtype::from_str("int32").unwrap();
        assert_eq!(dt.kind(), DtypeKind::Integer);
        assert_eq!(dt.itemsize(), 4);
    }

    #[test]
    fn test_dtype_from_type() {
        let dt_f64 = Dtype::from_type::<f64>();
        assert_eq!(dt_f64.kind(), DtypeKind::Float);
        assert_eq!(dt_f64.itemsize(), 8);

        let dt_i32 = Dtype::from_type::<i32>();
        assert_eq!(dt_i32.kind(), DtypeKind::Integer);
        assert_eq!(dt_i32.itemsize(), 4);
    }

    #[test]
    fn test_dtype_comprehensive() {
        assert!(Dtype::from_str("int8").is_ok());
        assert!(Dtype::from_str("float64").is_ok());
        assert!(Dtype::from_str("complex128").is_ok());
        assert!(Dtype::from_str("datetime64[ns]").is_ok());
        assert!(Dtype::from_str("bool").is_ok());
        assert!(Dtype::from_str("object").is_ok());

        assert!(Dtype::from_str("invalid_type").is_err());
        assert!(Dtype::from_str("datetime64[invalid]").is_err());
    }
}
```

---

## CONSTANTS FROM constants.rs

```rust
/// Mathematical constants matching NumPy
pub mod math {
    use std::f64;

    pub const E: f64 = f64::consts::E;
    pub const PI: f64 = f64::consts::PI;
    pub const TAU: f64 = f64::consts::TAU;
    pub const INF: f64 = f64::INFINITY;
    pub const NEG_INF: f64 = f64::NEG_INFINITY;
    pub const NAN: f64 = f64::NAN;
}

/// Type-specific constants
pub mod dtype {
    pub const INT8_MAX: i8 = i8::MAX;
    pub const INT8_MIN: i8 = i8::MIN;
    pub const INT16_MAX: i16 = i16::MAX;
    pub const INT16_MIN: i16 = i16::MIN;
    pub const INT32_MAX: i32 = i32::MAX;
    pub const INT32_MIN: i32 = i32::MIN;
    pub const INT64_MAX: i64 = i64::MAX;
    pub const INT64_MIN: i64 = i64::MIN;
    pub const UINT8_MAX: u8 = u8::MAX;
    pub const UINT16_MAX: u16 = u16::MAX;
    pub const UINT32_MAX: u32 = u32::MAX;
    pub const UINT64_MAX: u64 = u64::MAX;
}

/// Floating point special values
pub mod float {
    use std::f64;

    pub const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    pub const EPSILON: f64 = f64::EPSILON;
    pub const MAX: f64 = f64::MAX;
    pub const MIN: f64 = f64::MIN;
    pub const EPSILON_F32: f32 = f32::EPSILON;
}
```

---

**All code extracted from rust-numpy/src/dtype.rs (358 lines)**
**See DTYPE_ARCHITECTURE_ANALYSIS.md for detailed analysis**
**See DTYPE_QUICK_REFERENCE.md for quick lookup**
