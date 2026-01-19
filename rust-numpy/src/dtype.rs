#![allow(non_camel_case_types)]
use std::fmt;

/// Byte order for endianness support
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    Little, // '<' - little-endian
    Big,    // '>' - big-endian
    Native, // '=' - native byte order
    Ignore, // '|' - ignore byte order
}

/// Comprehensive dtype system matching NumPy's type system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dtype {
    // Signed integer types
    Int8 { byteorder: Option<ByteOrder> },
    Int16 { byteorder: Option<ByteOrder> },
    Int32 { byteorder: Option<ByteOrder> },
    Int64 { byteorder: Option<ByteOrder> },
    Intp { byteorder: Option<ByteOrder> }, // Platform-dependent (i32 on 32-bit, i64 on 64-bit)

    // Unsigned integer types
    UInt8 { byteorder: Option<ByteOrder> },
    UInt16 { byteorder: Option<ByteOrder> },
    UInt32 { byteorder: Option<ByteOrder> },
    UInt64 { byteorder: Option<ByteOrder> },
    Uintp { byteorder: Option<ByteOrder> }, // Platform-dependent (u32 on 32-bit, u64 on 64-bit)

    // Floating point types
    Float16 { byteorder: Option<ByteOrder> },
    Float32 { byteorder: Option<ByteOrder> },
    Float64 { byteorder: Option<ByteOrder> },
    Float128 { byteorder: Option<ByteOrder> }, // Extended precision (platform-dependent)

    // Complex types
    Complex32 { byteorder: Option<ByteOrder> },  // 2x f16
    Complex64 { byteorder: Option<ByteOrder> },  // 2x f32
    Complex128 { byteorder: Option<ByteOrder> }, // 2x f64
    Complex256 { byteorder: Option<ByteOrder> }, // 2x f128 (platform-dependent)

    // Boolean type
    Bool,

    // String types with optional length specification
    String { length: Option<usize> },  // Byte strings (S)
    Unicode { length: Option<usize> }, // Unicode strings (U)

    // Binary data type
    Bytes { length: usize }, // Fixed-width bytes (b)

    // Datetime types
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),

    // Object type
    Object,

    // Void type for padding/unstructured data
    Void { size: usize },

    // Structured type with enhanced field support
    Struct(Vec<StructField>),
}

/// Units for datetime64
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatetimeUnit {
    Y,
    M,
    W,
    D,
    h,
    m,
    s,
    ms,
    us,
    ns,
    ps,
    fs,
    As,
}

impl DatetimeUnit {
    pub fn as_str(&self) -> &'static str {
        match self {
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
    }
}

/// Units for timedelta64
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimedeltaUnit {
    Y,
    M,
    W,
    D,
    h,
    m,
    s,
    ms,
    us,
    ns,
    ps,
    fs,
    As,
}

/// Character codes for NumPy dtype string representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DTypeChar {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Bytes,
    Unicode,
    Object,
    Datetime64,
    Timedelta64,
    Void,
}

/// Field definition for structured dtypes with enhanced NumPy compatibility
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub dtype: Dtype,
    pub offset: Option<usize>,
    pub title: Option<String>,     // Field title for display
    pub shape: Option<Vec<usize>>, // Subarray shape, None for scalar
}

/// Kind of dtype for type checking and promotion
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtypeKind {
    Integer,
    Unsigned,
    Float,
    Complex,
    Bool,
    String,
    Bytes,
    Datetime,
    Object,
    Struct,
    Void,
}

impl Dtype {
    /// Get the kind of this dtype
    pub fn kind(&self) -> DtypeKind {
        match self {
            Dtype::Int8 { .. }
            | Dtype::Int16 { .. }
            | Dtype::Int32 { .. }
            | Dtype::Int64 { .. }
            | Dtype::Intp { .. } => DtypeKind::Integer,
            Dtype::UInt8 { .. }
            | Dtype::UInt16 { .. }
            | Dtype::UInt32 { .. }
            | Dtype::UInt64 { .. }
            | Dtype::Uintp { .. } => DtypeKind::Unsigned,
            Dtype::Float16 { .. }
            | Dtype::Float32 { .. }
            | Dtype::Float64 { .. }
            | Dtype::Float128 { .. } => DtypeKind::Float,
            Dtype::Complex32 { .. }
            | Dtype::Complex64 { .. }
            | Dtype::Complex128 { .. }
            | Dtype::Complex256 { .. } => DtypeKind::Complex,
            Dtype::Bool => DtypeKind::Bool,
            Dtype::String { .. } | Dtype::Unicode { .. } => DtypeKind::String,
            Dtype::Bytes { .. } => DtypeKind::Bytes,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => DtypeKind::Datetime,
            Dtype::Object => DtypeKind::Object,
            Dtype::Struct(_) => DtypeKind::Struct,
            Dtype::Void { .. } => DtypeKind::Void,
        }
    }

    /// Get size in bytes
    pub fn itemsize(&self) -> usize {
        match self {
            Dtype::Int8 { .. } | Dtype::UInt8 { .. } | Dtype::Bool => 1,
            Dtype::Int16 { .. } | Dtype::UInt16 { .. } | Dtype::Float16 { .. } => 2,
            Dtype::Int32 { .. }
            | Dtype::UInt32 { .. }
            | Dtype::Float32 { .. }
            | Dtype::Complex32 { .. } => 4,
            Dtype::Int64 { .. }
            | Dtype::UInt64 { .. }
            | Dtype::Float64 { .. }
            | Dtype::Complex64 { .. } => 8,
            Dtype::Complex128 { .. } => 16,
            Dtype::Complex256 { .. } => 32,
            Dtype::Float128 { .. } => 16,
            Dtype::Intp { .. } => std::mem::size_of::<isize>(),
            Dtype::Uintp { .. } => std::mem::size_of::<usize>(),
            Dtype::String { length } => length.unwrap_or(8),
            Dtype::Unicode { length } => length.unwrap_or(8) * 4,
            Dtype::Bytes { length } => *length,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => 8,
            Dtype::Object => 8,
            Dtype::Void { size } => *size,
            Dtype::Struct(fields) => fields.iter().map(|f| f.dtype.itemsize()).sum(),
        }
    }

    /// Get alignment requirement in bytes
    pub fn alignment(&self) -> usize {
        match self {
            Dtype::Int8 { .. } | Dtype::UInt8 { .. } | Dtype::Bool => 1,
            Dtype::Int16 { .. } | Dtype::UInt16 { .. } | Dtype::Float16 { .. } => 2,
            Dtype::Int32 { .. }
            | Dtype::UInt32 { .. }
            | Dtype::Float32 { .. }
            | Dtype::Complex32 { .. } => 4,
            Dtype::Int64 { .. }
            | Dtype::UInt64 { .. }
            | Dtype::Float64 { .. }
            | Dtype::Complex64 { .. } => 8,
            Dtype::Complex128 { .. } => 16,
            Dtype::Complex256 { .. } => 32,
            Dtype::Float128 { .. } => 16,
            Dtype::Intp { .. } => std::mem::align_of::<isize>(),
            Dtype::Uintp { .. } => std::mem::align_of::<usize>(),
            Dtype::String { .. } => 8,
            Dtype::Unicode { .. } => 8,
            Dtype::Bytes { .. } => 1,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => 8,
            Dtype::Object => 8,
            Dtype::Void { .. } => 1,
            Dtype::Struct(fields) => fields
                .iter()
                .map(|f| f.dtype.alignment())
                .max()
                .unwrap_or(1),
        }
    }

    /// Create dtype from Rust type
    pub fn from_type<T: 'static>() -> Self {
        use std::any::TypeId;
        let type_id = TypeId::of::<T>();
        if type_id == TypeId::of::<i8>() {
            Dtype::Int8 { byteorder: None }
        } else if type_id == TypeId::of::<i16>() {
            Dtype::Int16 { byteorder: None }
        } else if type_id == TypeId::of::<i32>() {
            Dtype::Int32 { byteorder: None }
        } else if type_id == TypeId::of::<i64>() {
            Dtype::Int64 { byteorder: None }
        } else if type_id == TypeId::of::<isize>() {
            Dtype::Intp { byteorder: None }
        } else if type_id == TypeId::of::<u8>() {
            Dtype::UInt8 { byteorder: None }
        } else if type_id == TypeId::of::<u16>() {
            Dtype::UInt16 { byteorder: None }
        } else if type_id == TypeId::of::<u32>() {
            Dtype::UInt32 { byteorder: None }
        } else if type_id == TypeId::of::<u64>() {
            Dtype::UInt64 { byteorder: None }
        } else if type_id == TypeId::of::<usize>() {
            Dtype::Uintp { byteorder: None }
        } else if type_id == TypeId::of::<f32>() {
            Dtype::Float32 { byteorder: None }
        } else if type_id == TypeId::of::<f64>() {
            Dtype::Float64 { byteorder: None }
        } else if type_id == TypeId::of::<bool>() {
            Dtype::Bool
        } else if type_id == TypeId::of::<String>() {
            Dtype::Unicode { length: None }
        } else if type_id == TypeId::of::<&str>() {
            Dtype::Unicode { length: None }
        } else {
            Dtype::Object
        }
    }

    /// Parse dtype from string (NumPy compatible)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "int8" | "i1" => Ok(Dtype::Int8 { byteorder: None }),
            "int16" | "i2" => Ok(Dtype::Int16 { byteorder: None }),
            "int32" | "i4" => Ok(Dtype::Int32 { byteorder: None }),
            "int64" | "i8" => Ok(Dtype::Int64 { byteorder: None }),
            "intp" | "ip" => Ok(Dtype::Intp { byteorder: None }),
            "uint8" | "u1" => Ok(Dtype::UInt8 { byteorder: None }),
            "uint16" | "u2" => Ok(Dtype::UInt16 { byteorder: None }),
            "uint32" | "u4" => Ok(Dtype::UInt32 { byteorder: None }),
            "uint64" | "u8" => Ok(Dtype::UInt64 { byteorder: None }),
            "uintp" | "up" => Ok(Dtype::Uintp { byteorder: None }),
            "float16" | "f2" => Ok(Dtype::Float16 { byteorder: None }),
            "float32" | "f4" => Ok(Dtype::Float32 { byteorder: None }),
            "float64" | "f8" => Ok(Dtype::Float64 { byteorder: None }),
            "complex32" | "c4" => Ok(Dtype::Complex32 { byteorder: None }),
            "complex64" | "c8" => Ok(Dtype::Complex64 { byteorder: None }),
            "complex128" | "c16" => Ok(Dtype::Complex128 { byteorder: None }),
            "bool" => Ok(Dtype::Bool),
            "str" => Ok(Dtype::String { length: None }),
            "unicode" => Ok(Dtype::Unicode { length: None }),
            "object" => Ok(Dtype::Object),
            _ => {
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
                        Ok(Dtype::Datetime64(DatetimeUnit::ns))
                    }
                } else {
                    Err(format!("Unknown dtype: {}", s))
                }
            }
        }
    }

    /// Convert to string (NumPy compatible)
    pub fn to_string(&self) -> String {
        match self {
            Dtype::Int8 { .. } => "int8".to_string(),
            Dtype::Int16 { .. } => "int16".to_string(),
            Dtype::Int32 { .. } => "int32".to_string(),
            Dtype::Int64 { .. } => "int64".to_string(),
            Dtype::UInt8 { .. } => "uint8".to_string(),
            Dtype::UInt16 { .. } => "uint16".to_string(),
            Dtype::UInt32 { .. } => "uint32".to_string(),
            Dtype::UInt64 { .. } => "uint64".to_string(),
            Dtype::Float16 { .. } => "float16".to_string(),
            Dtype::Float32 { .. } => "float32".to_string(),
            Dtype::Float64 { .. } => "float64".to_string(),
            Dtype::Complex32 { .. } => "complex32".to_string(),
            Dtype::Complex64 { .. } => "complex64".to_string(),
            Dtype::Complex128 { .. } => "complex128".to_string(),
            Dtype::Bool => "bool".to_string(),
            Dtype::String { .. } => "str".to_string(),
            Dtype::Unicode { .. } => "unicode".to_string(),
            Dtype::Object => "object".to_string(),
            Dtype::Struct(_) => "struct".to_string(),
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
            _ => "unknown".to_string(),
        }
    }

    /// Check if dtype can be safely cast to another dtype
    pub fn can_cast_to(&self, other: &Dtype) -> bool {
        use DtypeKind::*;
        let self_kind = self.kind();
        let other_kind = other.kind();

        match (self_kind, other_kind) {
            (Integer, Integer) | (Unsigned, Integer) => self.itemsize() <= other.itemsize(),
            (Integer, Unsigned) => false,
            (Float, Float) => self.itemsize() <= other.itemsize(),
            (Complex, Complex) => self.itemsize() <= other.itemsize(),
            (Integer | Unsigned | Float, Complex) => true,
            (Complex, Float) => false,
            (Bool, _) => true,
            (_, Bool) => false,
            (String, String) => true,
            (Datetime, Datetime) => true,
            (Object, _) | (_, Object) => true,
            _ => false,
        }
    }

    /// Get byte order of this dtype
    pub fn byteorder(&self) -> Option<ByteOrder> {
        match self {
            Dtype::Int8 { byteorder } => *byteorder,
            Dtype::Int16 { byteorder } => *byteorder,
            Dtype::Int32 { byteorder } => *byteorder,
            Dtype::Int64 { byteorder } => *byteorder,
            Dtype::Intp { byteorder } => *byteorder,
            Dtype::UInt8 { byteorder } => *byteorder,
            Dtype::UInt16 { byteorder } => *byteorder,
            Dtype::UInt32 { byteorder } => *byteorder,
            Dtype::UInt64 { byteorder } => *byteorder,
            Dtype::Uintp { byteorder } => *byteorder,
            Dtype::Float16 { byteorder } => *byteorder,
            Dtype::Float32 { byteorder } => *byteorder,
            Dtype::Float64 { byteorder } => *byteorder,
            Dtype::Float128 { byteorder } => *byteorder,
            Dtype::Complex32 { byteorder } => *byteorder,
            Dtype::Complex64 { byteorder } => *byteorder,
            Dtype::Complex128 { byteorder } => *byteorder,
            Dtype::Complex256 { byteorder } => *byteorder,
            Dtype::Bool => None,
            Dtype::String { .. } => None,
            Dtype::Unicode { .. } => None,
            Dtype::Bytes { .. } => None,
            Dtype::Datetime64(_) => None,
            Dtype::Timedelta64(_) => None,
            Dtype::Object => None,
            Dtype::Void { .. } => None,
            Dtype::Struct(_) => None,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// IEEE 754 half-precision float type (re-exported from half crate)
pub use half::f16;
