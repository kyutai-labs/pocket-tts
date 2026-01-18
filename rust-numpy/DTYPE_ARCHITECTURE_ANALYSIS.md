# Rust NumPy Dtype System - Architecture Analysis

**Date:** 2026-01-17
**Project:** rust-numpy
**Analysis Type:** Research & Documentation

---

## Executive Summary

The rust-numpy dtype system provides a comprehensive foundation for NumPy-style type support with **21+ dtype variants** covering most core NumPy data types. The system is well-structured with enum-based variants, type kind classification, and complete size/alignment metadata. However, **dtype promotion is NOT implemented**, and several advanced NumPy dtypes are missing.

---

## 1. IMPLEMENTED DTYPE VARIANTS

### 1.1 Signed Integers (4 variants)
```rust
Dtype::Int8    // 8-bit signed
Dtype::Int16   // 16-bit signed
Dtype::Int32   // 32-bit signed
Dtype::Int64   // 64-bit signed
```
**Item sizes:** 1, 2, 4, 8 bytes
**Alignment:** 1, 2, 4, 8 bytes

### 1.2 Unsigned Integers (4 variants)
```rust
Dtype::UInt8   // 8-bit unsigned
Dtype::UInt16  // 16-bit unsigned
Dtype::UInt32  // 32-bit unsigned
Dtype::UInt64  // 64-bit unsigned
```
**Item sizes:** 1, 2, 4, 8 bytes
**Alignment:** 1, 2, 4, 8 bytes

### 1.3 Floating Point (3 variants)
```rust
Dtype::Float16 // Half-precision (16-bit)
Dtype::Float32 // Single-precision (32-bit)
Dtype::Float64 // Double-precision (64-bit)
```
**Item sizes:** 2, 4, 8 bytes
**Alignment:** 2, 4, 8 bytes
**Special Implementation:** `f16` struct defined with simplified conversion (not IEEE 754 compliant)

### 1.4 Complex Numbers (3 variants)
```rust
Dtype::Complex32  // 32-bit complex (2x f16)
Dtype::Complex64  // 64-bit complex (2x f32)
Dtype::Complex128 // 128-bit complex (2x f64)
```
**Item sizes:** 4, 8, 16 bytes
**Alignment:** 4, 8, 16 bytes
**External Dependency:** Uses `num-complex = "0.4"` crate for `Complex<f64>` type

### 1.5 Boolean (1 variant)
```rust
Dtype::Bool
```
**Item size:** 1 byte
**Alignment:** 1 byte

### 1.6 String Types (2 variants)
```rust
Dtype::String   // Byte strings
Dtype::Unicode  // Unicode strings
```
**Item size:** 8 bytes (pointer size)
**Alignment:** 8 bytes

### 1.7 Datetime Types (12 variants)
```rust
Dtype::Datetime64(unit)  // Timestamps with units
Dtype::Timedelta64(unit) // Time differences with units
```

**DatetimeUnit/TimedeltaUnit variants:**
```rust
Y   // Years
M   // Months
W   // Weeks
D   // Days
h   // Hours
m   // Minutes
s   // Seconds
ms  // Milliseconds
us  // Microseconds
ns  // Nanoseconds
ps  // Picoseconds
fs  // Femtoseconds
As  // Attoseconds
```
**Item size:** 8 bytes
**Alignment:** 8 bytes
**Default unit:** Nanoseconds (ns) for datetime64

### 1.8 Object Type (1 variant)
```rust
Dtype::Object
```
**Item size:** 8 bytes (pointer size)
**Alignment:** 8 bytes
**Usage:** Generic Python object references

### 1.9 Structured Types (variable)
```rust
Dtype::Struct(Vec<StructField>)
```
```rust
StructField {
    name: String,
    dtype: Dtype,
    offset: Option<usize>,
}
```
**Item size:** Sum of field dtypes
**Alignment:** Maximum field alignment
**Offset:** Optional (auto-calculated if None)

---

## 2. DTYPE SYSTEM ARCHITECTURE

### 2.1 Core Enum Structure
```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dtype {
    // 4 signed integers
    Int8, Int16, Int32, Int64,
    // 4 unsigned integers
    UInt8, UInt16, UInt32, UInt64,
    // 3 floats
    Float16, Float32, Float64,
    // 3 complex
    Complex32, Complex64, Complex128,
    // 1 boolean
    Bool,
    // 2 strings
    String, Unicode,
    // 2 datetime variants with 12 units each
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),
    // 1 object
    Object,
    // 1 structured
    Struct(Vec<StructField>),
}
```

### 2.2 DtypeKind Classification
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
**Purpose:** Type checking, ufunc dtype support verification

### 2.3 Key Methods

#### `kind() -> DtypeKind`
- Classifies dtype into broad category
- Used by ufuncs to check dtype compatibility

#### `itemsize() -> usize`
- Returns size in bytes
- Struct types sum field sizes

#### `alignment() -> usize`
- Returns alignment requirement in bytes
- Struct types use maximum field alignment

#### `from_type<T>() -> Dtype`
- Maps Rust `TypeId` to `Dtype` variant
- Supports: i8-i64, u8-u64, f16-f64, bool
- **Fallback:** Returns `Object` for unknown types

#### `from_str(s: &str) -> Result<Self, String>`
- Parses NumPy-compatible dtype strings
- Supports both long and short forms (e.g., "int32", "i4")
- Parses datetime units: `datetime64[ns]`
- **Missing:** Complex32 parsing (only Complex64/128 supported)

#### `to_string() -> String`
- Converts dtype to NumPy-compatible string
- Output matches NumPy dtype syntax

#### `can_cast_to(&self, other: &Dtype) -> bool`
- Implements NumPy casting rules
- Safe casts:
  - Same kind, larger size (int8->int16)
  - Integer/Unsigned/Float -> Complex
  - Bool -> anything
- Unsafe casts:
  - Unsigned -> Integer
  - Complex -> Float
  - Anything -> Bool (except Bool)
- Always safe:
  - String -> String
  - Datetime -> Datetime
  - Object -> anything
  - anything -> Object

---

## 3. DTYPE PROMOTION STATUS

### 3.1 Current State: **NOT IMPLEMENTED**

**Evidence from code:**
```rust
// ufunc.rs line 187
// Real implementation would need proper broadcasting, dtype promotion, etc.

// ufunc_ops.rs line 322
// This would need proper type promotion in real implementation
```

### 3.2 What NumPy Does (For Reference)

NumPy's dtype promotion follows a type hierarchy:
1. **Scalar promotion** (scalar with array)
2. **Array promotion** (array with array)
3. **Weak/strong dtypes** (e.g., int32 vs uint64)
4. **Result type determination** based on operation

**Promotion table (simplified):**
```
int8   + int8   -> int8
int8   + int16  -> int16
int32  + uint32 -> int64
float32 + int64  -> float64
complex64 + float32 -> complex128
```

### 3.3 Required Implementation

To add dtype promotion, need:
1. **Promotion matrix** - Lookup table for dtype pairs
2. **Weak/strong dtype tracking** - Some dtypes promote more strongly
3. **Safe casting** - Use `can_cast_to()` for validation
4. **Scalar/array distinction** - Scalars promote differently

---

## 4. MISSING NUMPY DTYPES

### 4.1 Completely Missing Types

| NumPy Dtype | Description | Priority |
|-------------|-------------|----------|
| **intp** | Platform-dependent pointer-sized integer | HIGH |
| **uintp** | Platform-dependent unsigned pointer-sized integer | HIGH |
| **half** | IEEE 754 half-precision float | MEDIUM |
| **single** | IEEE 754 single-precision float (alias for float32) | LOW |
| **double** | IEEE 754 double-precision float (alias for float64) | LOW |
| **csingle** | Complex float32 (alias) | LOW |
| **cdouble** | Complex float64 (alias) | LOW |
| **longdouble** | Extended precision float | MEDIUM |
| **clongdouble** | Extended precision complex | MEDIUM |
| **bytes_** | Fixed-size bytes | LOW |
| **str_** | Fixed-size string | LOW |
| **void** | Structured type with specified byte size | LOW |
| **object_** | Python object (same as Object) | NONE |

### 4.2 Partially Missing Features

**Complex32:**
- Enum variant exists but `from_str()` doesn't support it
- String representation returns "complex64" (bug!)

**Complex Types:**
- No actual complex number types defined (only dtype enums)
- Should use `num_complex::Complex<f32>` and `num_complex::Complex<f64>`

**Fixed-width strings:**
- `String`/`Unicode` exist but are always 8 bytes (pointer)
- No support for fixed-length strings like `S10` (10-byte string)

**Fixed-width bytes:**
- No `bytes_` type for binary data

### 4.3 NumPy Reference Dtypes (Not Yet Implemented)

```
bool_         (already as Bool)
int8          (already)
int16         (already)
int32         (already)
int64         (already)
intp          (MISSING - i32/i64 based on platform)
uint8         (already)
uint16        (already)
uint32        (already)
uint64        (already)
uintp         (MISSING - u32/u64 based on platform)
float16       (already - but f16 not IEEE 754)
float32       (already)
float64       (already)
float_        (alias - already)
float64       (already)
complex64     (already)
complex128    (already)
complex256    (MISSING - not in NumPy 1.x anyway)
object        (already)
bytes_        (MISSING)
str_          (MISSING)
void          (MISSING)
datetime64    (already)
timedelta64   (already)
```

---

## 5. DTYPE MODULE STRUCTURE

### 5.1 File Organization

```
rust-numpy/src/
├── dtype.rs              # Main dtype system (358 lines)
│   ├── Dtype enum
│   ├── DtypeKind enum
│   ├── DatetimeUnit enum
│   ├── TimedeltaUnit enum
│   ├── StructField struct
│   ├── f16 struct (simplified)
│   └── impl Dtype methods
│
├── array.rs              # Array<T> with dtype field (360 lines)
│   ├── pub dtype: Dtype
│   └── pub fn dtype(&self) -> &Dtype
│
├── ufunc.rs              # Universal functions (needs promotion)
│   ├── Ufunc trait with supported_dtypes()
│   └── BinaryUfunc, UnaryUfunc
│
├── ufunc_ops.rs          # Ufunc execution (needs promotion)
│   └── UfuncEngine (binary/unary ops)
│
├── constants.rs          # Type-specific constants
│   └── dtype module (INT8_MAX, etc.)
│
└── lib.rs                # Re-exports Dtype, DtypeKind
```

### 5.2 Dependencies Used

| Dependency | Version | Purpose |
|------------|---------|---------|
| num-complex | 0.4 | Complex number types |
| num-traits | 0.2 | Numeric traits |
| num-integer | 0.1 | Integer operations |
| num-bigint | 0.4 | Big integers (for overflow?) |
| num-rational | 0.4 | Rational numbers |
| chrono | 0.4 (optional) | Datetime support |

---

## 6. CURRENT DTYPE USAGE

### 6.1 In Arrays
```rust
pub struct Array<T> {
    data: Arc<MemoryManager<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,  // ← Dtype field
    offset: usize,
}
```

### 6.2 In Views
```rust
pub struct ArrayView<'a, T> {
    data: &'a [T],
    dtype: Dtype,  // ← Dtype field
    // ...
}
```

### 6.3 In Ufuncs
```rust
pub trait Ufunc: Send + Sync {
    fn supported_dtypes(&self) -> &[DtypeKind];
    fn supports_dtypes(&self, dtypes: &[&Dtype]) -> bool;
}
```

### 6.4 Type Inference
```rust
impl<T> Array<T> {
    pub fn from_vec(data: Vec<T>) -> Self {
        let dtype = Dtype::from_type::<T>();  // ← Automatic
        // ...
    }
}
```

---

## 7. TESTING COVERAGE

### 7.1 Test Files

| File | Dtype Tests | Status |
|------|-------------|--------|
| tests/basic_tests.rs | `test_dtype_creation` | ✅ Passes |
| tests/basic_tests.rs | `test_dtype_from_type` | ✅ Passes |
| tests/comprehensive_tests.rs | `test_dtype_comprehensive` | ✅ Passes |

### 7.2 Test Coverage

**Covered:**
- ✅ String parsing (int8, float64, complex128, datetime64[ns], bool, object)
- ✅ Invalid dtype detection
- ✅ Type inference from Rust types
- ✅ Item size calculation
- ✅ Dtype kind classification

**Not Tested:**
- ❌ Complex32 parsing
- ❌ Struct dtype creation
- ❌ Dtype casting (`can_cast_to()`)
- ❌ Dtype promotion (not implemented)
- ❌ Alignment calculations
- ❌ All 12 datetime units

---

## 8. ARCHITECTURAL EXTENSIONS NEEDED

### 8.1 High Priority (Blocker for NumPy Parity)

1. **Dtype Promotion Engine**
   - Add `promote_with(&self, other: &Dtype) -> Dtype`
   - Implement NumPy's promotion table
   - Handle weak/strong dtypes
   - Test scalar vs array promotion

2. **Platform-Dependent Types**
   - Add `Dtype::Intp` (i32 on 32-bit, i64 on 64-bit)
   - Add `Dtype::Uintp` (u32 on 32-bit, u64 on 64-bit)
   - Use `std::mem::size_of::<usize>()` for detection

3. **Correct f16 Implementation**
   - Replace simplified bit shifting
   - Use IEEE 754 half-precision conversion
   - Consider `half` crate (https://crates.io/crates/half)

### 8.2 Medium Priority (Feature Complete)

1. **Complex Number Types**
   - Define type aliases:
     ```rust
     pub type Complex32 = num_complex::Complex<f16>;
     pub type Complex64 = num_complex::Complex<f32>;
     pub type Complex128 = num_complex::Complex<f64>;
     ```
   - Parse Complex32 from "complex64" (current bug!)

2. **Fixed-Width Strings**
   - Add `Dtype::String { length: usize }`
   - Add `Dtype::Unicode { length: usize }`
   - Parse formats like "S10" (10-byte string)

3. **Extended Precision Floats**
   - Add `Dtype::Float128` (if platform supports)
   - Add `Dtype::Complex256`
   - Consider `soft-float` crate for cross-platform

### 8.3 Low Priority (Nice to Have)

1. **Byte Type**
   - Add `Dtype::Bytes { length: usize }`
   - Binary data support

2. **Void Type**
   - Add `Dtype::Void { size: usize }`
   - For structured padding

---

## 9. DTYPE PROMOTION IMPLEMENTATION PLAN

### 9.1 Proposed API

```rust
impl Dtype {
    /// Promote two dtypes to a common dtype
    pub fn promote(&self, other: &Dtype) -> Result<Dtype, String> {
        // Implementation...
    }

    /// Check if this dtype can be safely cast to other
    pub fn can_promote_to(&self, other: &Dtype) -> bool {
        // Use promote() and check if equal to other
    }
}
```

### 9.2 Promotion Rules (NumPy 2.0)

**Same kind:**
- Integer + Integer → Larger size
- Unsigned + Unsigned → Larger size
- Float + Float → Larger size
- Complex + Complex → Larger size

**Mixed kind:**
- Integer + Unsigned → Integer (if fits, else larger)
- Integer/Unsigned + Float → Float
- Integer/Unsigned/Float + Complex → Complex
- Bool + anything → anything

**Datetime/Timedelta:**
- Datetime64 + int → Datetime64
- Timedelta64 + int → Timedelta64
- Datetime64 + Timedelta64 → Timedelta64

**String/Object:**
- String + String → String (longer)
- Object + anything → Object

### 9.3 Weak/Strong Dtypes

**Strong dtypes** (dominate promotion):
- float64 > float32 > float16
- complex128 > complex64 > complex32
- int64 > int32 > int16 > int8
- uint64 > uint32 > uint16 > uint8

**Weak dtypes** (get promoted by strong):
- datetime64, timedelta64
- bool

---

## 10. RECOMMENDATIONS

### 10.1 Immediate Actions

1. **Fix Complex32 bug** - Line 254 in dtype.rs should return "complex64" for Complex32 or add parsing
2. **Implement dtype promotion** - Critical for ufunc operations
3. **Add Intp/Uintp** - Platform-dependent types are fundamental

### 10.2 Short-term (Next Sprint)

1. **Proper f16** - Replace simplified conversion with IEEE 754
2. **Complex type aliases** - Make complex numbers usable
3. **Extend tests** - Add promotion tests, all datetime units

### 10.3 Long-term (Future)

1. **Fixed-width strings** - Parse "S10" format
2. **Extended precision** - Float128 if needed
3. **Performance** - SIMD for promotion operations

---

## 11. REFERENCES

- [NumPy Dtype Documentation](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
- [NumPy dtype promotion rules](https://numpy.org/doc/stable/reference/ufuncs.html#casting-rules)
- [num-complex crate](https://docs.rs/num-complex/latest/num_complex/)
- [half crate](https://docs.rs/half/latest/half/)

---

## 12. SUMMARY STATISTICS

| Metric | Count |
|--------|-------|
| Total dtype variants | 21+ |
| Integer variants | 8 |
| Float variants | 3 |
| Complex variants | 3 |
| String variants | 2 |
| Datetime units | 12 |
| Implemented methods | 7 |
| Missing NumPy dtypes | 8+ |
| Lines of code | ~360 |
| Test files | 2 |

---

**Analysis Complete**
**Next Steps:** Implement dtype promotion system and add missing platform-dependent types
