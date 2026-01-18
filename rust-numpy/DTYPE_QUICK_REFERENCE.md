# Rust NumPy Dtype System - Quick Reference

## DTYPE CHEAT SHEET

### Complete Dtype Enum (21+ variants)

```rust
pub enum Dtype {
    // Signed integers (4)
    Int8, Int16, Int32, Int64,

    // Unsigned integers (4)
    UInt8, UInt16, UInt32, UInt64,

    // Floating point (3)
    Float16, Float32, Float64,

    // Complex numbers (3)
    Complex32, Complex64, Complex128,

    // Boolean (1)
    Bool,

    // Strings (2)
    String, Unicode,

    // Datetime (12 units each)
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),

    // Object (1)
    Object,

    // Structured (variable)
    Struct(Vec<StructField>),
}
```

### DtypeKind Categories

```rust
pub enum DtypeKind {
    Integer,   // Int8-Int64
    Unsigned,  // UInt8-UInt64
    Float,     // Float16-Float64
    Complex,   // Complex32-Complex128
    Bool,      // Bool
    String,    // String, Unicode
    Datetime,  // Datetime64, Timedelta64
    Object,    // Object
    Struct,    // Struct
}
```

### Datetime/Timedelta Units (12 each)

```rust
pub enum DatetimeUnit {
    Y,   // Years
    M,   // Months
    W,   // Weeks
    D,   // Days
    h,   // Hours
    m,   // Minutes
    s,   // Seconds
    ms,  // Milliseconds
    us,  // Microseconds
    ns,  // Nanoseconds (default)
    ps,  // Picoseconds
    fs,  // Femtoseconds
    As,  // Attoseconds
}
```

### StructField Definition

```rust
pub struct StructField {
    pub name: String,
    pub dtype: Dtype,
    pub offset: Option<usize>,
}
```

---

## DTYPE METHOD SIGNATURES

```rust
impl Dtype {
    // Get dtype category
    pub fn kind(&self) -> DtypeKind;

    // Size in bytes
    pub fn itemsize(&self) -> usize;

    // Alignment in bytes
    pub fn alignment(&self) -> usize;

    // From Rust type
    pub fn from_type<T: 'static>() -> Self;

    // Parse from string
    pub fn from_str(s: &str) -> Result<Self, String>;

    // Convert to string
    pub fn to_string(&self) -> String;

    // Check casting safety
    pub fn can_cast_to(&self, other: &Dtype) -> bool;
}
```

---

## DTYPE STRING PARSING

### Supported Formats

| Long Form | Short Form | Dtype |
|-----------|------------|-------|
| int8 | i1 | Int8 |
| int16 | i2 | Int16 |
| int32 | i4 | Int32 |
| int64 | i8 | Int64 |
| uint8 | u1 | UInt8 |
| uint16 | u2 | UInt16 |
| uint32 | u4 | UInt32 |
| uint64 | u8 | UInt64 |
| float16 | f2 | Float16 |
| float32 | f4 | Float32 |
| float64 | f8 | Float64 |
| complex64 | c8 | Complex64 |
| complex128 | c16 | Complex128 |
| bool | - | Bool |
| str | - | String |
| unicode | - | Unicode |
| object | - | Object |
| datetime64[Y|M|W...|ns] | - | Datetime64 |

### Examples

```rust
let dt1 = Dtype::from_str("int32").unwrap();  // Dtype::Int32
let dt2 = Dtype::from_str("f4").unwrap();     // Dtype::Float32
let dt3 = Dtype::from_str("datetime64[ns]").unwrap(); // Dtype::Datetime64(Nanoseconds)
let dt4 = Dtype::from_str("complex128").unwrap(); // Dtype::Complex128
let dt5 = Dtype::from_str("bool").unwrap();   // Dtype::Bool
```

---

## ITEM SIZES & ALIGNMENT

| Dtype | Item Size | Alignment |
|-------|-----------|-----------|
| Int8, UInt8, Bool | 1 | 1 |
| Int16, UInt16, Float16 | 2 | 2 |
| Int32, UInt32, Float32, Complex32 | 4 | 4 |
| Int64, UInt64, Float64, Complex64 | 8 | 8 |
| Complex128 | 16 | 16 |
| String, Unicode | 8 | 8 |
| Datetime64, Timedelta64 | 8 | 8 |
| Object | 8 | 8 |
| Struct | sum(fields) | max(field alignments) |

---

## CASTING SAFETY

### Safe Casts (can_cast_to() = true)

| From | To | Rule |
|------|-----|------|
| int8 | int16, int32, int64 | Larger same kind |
| uint8 | uint16, uint32, uint64 | Larger same kind |
| float16 | float32, float64 | Larger same kind |
| complex64 | complex128 | Larger same kind |
| int8 | int8 | Same size, same kind |
| uint8 | int16, int32, int64 | Unsigned to signed |
| int8, uint8, float16, float32 | complex32, complex64, complex128 | Numeric to complex |
| bool | anything | Bool promotes |
| String | String | Same string type |
| Datetime64 | Datetime64 | Same datetime type |
| Object | anything | Object is universal |
| anything | Object | Object accepts all |

### Unsafe Casts (can_cast_to() = false)

| From | To | Why |
|------|-----|-----|
| uint8 | int8 | Unsigned to signed (smaller) |
| complex64 | float32 | Complex to float (loses imaginary) |
| int32 | bool | Numeric to bool (not safe) |
| float64 | int32 | Float to int (precision loss) |

---

## CASTING RULES MATRIX

| From \ To | Int | UInt | Float | Complex | Bool | String | Datetime | Object |
|-----------|-----|-------|-------|---------|------|--------|----------|--------|
| **Int** | ✓* | ✗ | ✓* | ✓ | ✗ | ✗ | ✗ | ✓ |
| **UInt** | ✗ | ✓* | ✓* | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Float** | ✗ | ✗ | ✓* | ✓ | ✗ | ✗ | ✗ | ✓ |
| **Complex** | ✗ | ✗ | ✗ | ✓* | ✗ | ✗ | ✗ | ✓ |
| **Bool** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **String** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| **Datetime** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Object** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

\* = Only if itemsize() <= target.itemsize()

---

## DTYPE FROM RUST TYPES

```rust
// Automatic inference
Dtype::from_type::<i8>()    // Dtype::Int8
Dtype::from_type::<i16>()   // Dtype::Int16
Dtype::from_type::<i32>()   // Dtype::Int32
Dtype::from_type::<i64>()   // Dtype::Int64

Dtype::from_type::<u8>()    // Dtype::UInt8
Dtype::from_type::<u16>()   // Dtype::UInt16
Dtype::from_type::<u32>()   // Dtype::UInt32
Dtype::from_type::<u64>()   // Dtype::UInt64

Dtype::from_type::<f16>()   // Dtype::Float16
Dtype::from_type::<f32>()   // Dtype::Float32
Dtype::from_type::<f64>()   // Dtype::Float64

Dtype::from_type::<bool>()  // Dtype::Bool

Dtype::from_type::<String>() // Dtype::Object (fallback)
```

---

## DTYPE TO STRING CONVERSION

```rust
let dt = Dtype::Int32;
dt.to_string();  // "int32"

let dt = Dtype::Float64;
dt.to_string();  // "float64"

let dt = Dtype::Complex128;
dt.to_string();  // "complex128"

let dt = Dtype::Datetime64(DatetimeUnit::ns);
dt.to_string();  // "datetime64[ns]"

let dt = Dtype::Struct(vec![
    StructField { name: "x".to_string(), dtype: Dtype::Int32, offset: Some(0) },
    StructField { name: "y".to_string(), dtype: Dtype::Float64, offset: Some(4) },
]);
dt.to_string();  // "struct"
```

---

## STRUCTURED DTYPES

### Creating Structured Dtypes

```rust
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
        name: "flag".to_string(),
        dtype: Dtype::Bool,
        offset: Some(12),
    },
]);

// itemsize = 4 + 8 + 1 = 13 bytes
// alignment = max(4, 8, 1) = 8 bytes
```

### Auto-calculated Offsets

```rust
let struct_dtype = Dtype::Struct(vec![
    StructField {
        name: "a".to_string(),
        dtype: Dtype::Int32,
        offset: None,  // Auto-calculate
    },
    StructField {
        name: "b".to_string(),
        dtype: Dtype::Float64,
        offset: None,  // Auto-calculate
    },
]);

// NOTE: Auto-calculation not yet implemented
```

---

## F16 HALF-PRECISION FLOAT

### Current Implementation (Simplified)

```rust
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
```

**Status:** Not IEEE 754 compliant. Consider using `half` crate.

---

## CONSTANTS (IN constants.rs)

### Integer Limits

```rust
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
```

### Float Constants

```rust
pub const EPSILON: f64 = f64::EPSILON;
pub const MAX: f64 = f64::MAX;
pub const MIN: f64 = f64::MIN;
pub const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
pub const EPSILON_F32: f32 = f32::EPSILON;
```

---

## USAGE EXAMPLES

### Creating Arrays with Dtypes

```rust
use numpy::*;

// Automatic dtype inference
let arr_int = array![1, 2, 3];        // dtype: Int32 (if i32 elements)
let arr_float = array![1.0, 2.0, 3.0]; // dtype: Float64

// Manual dtype specification (not yet supported)
// let arr = Array::with_dtype(Dtype::Float32, vec![1.0, 2.0, 3.0]);

// Parsing dtype from string
let dt = Dtype::from_str("float32").unwrap();
println!("Dtype: {}, Size: {} bytes", dt, dt.itemsize());
```

### Checking Dtypes

```rust
let arr = array![1, 2, 3];
let dtype = arr.dtype();

println!("Kind: {:?}", dtype.kind());      // DtypeKind::Integer
println!("Item size: {}", dtype.itemsize()); // 4
println!("Alignment: {}", dtype.alignment()); // 4
println!("Can cast to float64? {}",
    dtype.can_cast_to(&Dtype::Float64)); // true
```

### Casting Dtypes

```rust
let int_dtype = Dtype::Int32;
let float_dtype = Dtype::Float64;

if int_dtype.can_cast_to(&float_dtype) {
    println!("Safe cast: int32 -> float64");
}

let complex_dtype = Dtype::Complex128;
if !complex_dtype.can_cast_to(&int_dtype) {
    println!("Unsafe cast: complex128 -> int32");
}
```

---

## MISSING FEATURES

### Not Yet Implemented

1. **Dtype Promotion**
   ```rust
   // NOT YET AVAILABLE
   let result = dtype1.promote(&dtype2);
   ```

2. **Platform-Dependent Types**
   ```rust
   // NOT YET AVAILABLE
   let dt = Dtype::Intp;  // i32 on 32-bit, i64 on 64-bit
   ```

3. **Fixed-Width Strings**
   ```rust
   // NOT YET AVAILABLE
   let dt = Dtype::String { length: 10 };  // "S10"
   ```

4. **IEEE 754 f16**
   - Current implementation is simplified
   - Should use `half` crate for accuracy

### Bugs

1. **Complex32 String**
   ```rust
   Dtype::Complex32.to_string();  // Returns "complex64" (should be "complex32")
   ```

---

## QUICK LOOKUP TABLES

### Dtype Variants by Category

| Category | Dtypes | Count |
|----------|--------|-------|
| Integer | Int8, Int16, Int32, Int64 | 4 |
| Unsigned | UInt8, UInt16, UInt32, UInt64 | 4 |
| Float | Float16, Float32, Float64 | 3 |
| Complex | Complex32, Complex64, Complex128 | 3 |
| Boolean | Bool | 1 |
| String | String, Unicode | 2 |
| Datetime | Datetime64(12 units), Timedelta64(12 units) | 24 |
| Object | Object | 1 |
| Struct | Struct(Vec<StructField>) | 1 |
| **Total** | | **43** |

### Datetime Units Supported

| Unit | Description | Scale |
|------|-------------|-------|
| Y | Years | 365 days |
| M | Months | 30 days |
| W | Weeks | 7 days |
| D | Days | 1 day |
| h | Hours | 1/24 day |
| m | Minutes | 1/1440 day |
| s | Seconds | 1/86400 day |
| ms | Milliseconds | 1/86400000 day |
| us | Microseconds | 1/86400000000 day |
| ns | Nanoseconds | 1/86400000000000 day |
| ps | Picoseconds | 1e-12 seconds |
| fs | Femtoseconds | 1e-15 seconds |
| As | Attoseconds | 1e-18 seconds |

---

## COMMON PATTERNS

### Pattern 1: Type Checking

```rust
fn process_array(arr: &Array<f64>) {
    match arr.dtype().kind() {
        DtypeKind::Float => println!("Processing float array"),
        DtypeKind::Integer => println!("Processing integer array"),
        _ => println!("Unknown dtype"),
    }
}
```

### Pattern 2: Safe Casting

```rust
fn safe_cast_operation(arr: &Array<f32>) {
    let target_dtype = Dtype::Float64;
    if arr.dtype().can_cast_to(&target_dtype) {
        // Perform cast
    } else {
        // Handle error
    }
}
```

### Pattern 3: Dtype from String

```rust
fn create_array_from_dtype_string(dtype_str: &str) {
    let dtype = Dtype::from_str(dtype_str).unwrap();
    match dtype {
        Dtype::Float32 => println!("Float32 array"),
        Dtype::Int32 => println!("Int32 array"),
        _ => println!("Other dtype"),
    }
}
```

---

**Reference Complete**
**See DTYPE_ARCHITECTURE_ANALYSIS.md for detailed analysis**
