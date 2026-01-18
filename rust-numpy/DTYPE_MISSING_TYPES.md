# Rust NumPy Dtype System - Missing Types Analysis

**Date:** 2026-01-17
**Project:** rust-numpy
**Purpose:** Identify gaps between current implementation and NumPy dtype system

---

## EXECUTIVE SUMMARY

**Current Implementation:** 21+ dtype variants (43 total with units)
**NumPy 1.x Total:** ~50 dtype variants
**Missing High Priority:** 3 types (intp, uintp, proper f16)
**Missing Medium Priority:** 2 types (float128, complex256)
**Missing Low Priority:** 3 types (bytes_, fixed-width strings, void)
**Status:** Good foundation, needs promotion engine and platform types

---

## COMPLETE MISSING DTYPE LIST

### PRIORITY 1: CRITICAL (Blocking NumPy Parity)

| NumPy Dtype | Current Status | Action Required | Impact |
|-------------|----------------|-----------------|--------|
| **intp** | ❌ NOT IMPLEMENTED | Add `Dtype::Intp` variant | HIGH - Platform-dependent indexing |
| **uintp** | ❌ NOT IMPLEMENTED | Add `Dtype::Uintp` variant | HIGH - Platform-dependent sizing |
| **f16 (IEEE 754)** | ⚠️ PARTIAL | Replace simplified f16 struct | HIGH - Incorrect calculations |

**Why Critical:**
- `intp` and `uintp` are used throughout NumPy for array indices
- `f16` is broken - using simplified bit shifting instead of IEEE 754
- Without these, core NumPy operations fail

---

### PRIORITY 2: HIGH (Feature Complete)

| NumPy Dtype | Current Status | Action Required | Impact |
|-------------|----------------|-----------------|--------|
| **float128** | ❌ NOT IMPLEMENTED | Add `Dtype::Float128` variant | MEDIUM - Extended precision |
| **clongdouble** | ❌ NOT IMPLEMENTED | Add `Dtype::Complex256` variant | MEDIUM - Extended precision |

**Why High:**
- Extended precision needed for scientific computing
- Some NumPy functions use longdouble internally

**Implementation Considerations:**
- Platform-specific support (not available on all systems)
- May need `soft-float` crate for cross-platform support

---

### PRIORITY 3: MEDIUM (Important Features)

| NumPy Dtype | Current Status | Action Required | Impact |
|-------------|----------------|-----------------|--------|
| **complex64 (as alias)** | ⚠️ PARTIAL | Add parsing for "c8" to Complex32 | LOW - Already exists as Complex64 |
| **bytes_** | ❌ NOT IMPLEMENTED | Add `Dtype::Bytes { length: usize }` | LOW - Binary data support |
| **fixed-width strings** | ⚠️ PARTIAL | Add length parameter to String/Unicode | MEDIUM - Fixed-size strings |
| **void** | ❌ NOT IMPLEMENTED | Add `Dtype::Void { size: usize }` | LOW - Padding types |

**Why Medium:**
- Fixed-width strings are common in NumPy (e.g., "S10", "U10")
- Binary data support needed for some operations

---

### PRIORITY 4: LOW (Nice to Have)

| NumPy Dtype | Current Status | Action Required | Impact |
|-------------|----------------|-----------------|--------|
| **single** (alias) | ❌ NOT IMPLEMENTED | Add alias for Float32 | MINIMAL - Convenience |
| **double** (alias) | ❌ NOT IMPLEMENTED | Add alias for Float64 | MINIMAL - Convenience |
| **csingle** (alias) | ❌ NOT IMPLEMENTED | Add alias for Complex64 | MINIMAL - Convenience |
| **cdouble** (alias) | ❌ NOT IMPLEMENTED | Add alias for Complex128 | MINIMAL - Convenience |

**Why Low:**
- Pure convenience aliases, not functionality
- Can be added in parsing layer only

---

## DETAILED MISSING TYPE ANALYSIS

### 1. intp / uintp (Platform-Dependent Integers)

**NumPy Behavior:**
```python
import numpy as np
import sys

# On 64-bit system:
print(np.intp)   # <class 'numpy.int64'>
print(np.uintp)  # <class 'numpy.uint64'>

# On 32-bit system:
print(np.intp)   # <class 'numpy.int32'>
print(np.uintp)  # <class 'numpy.uint32'>
```

**Current Status:**
```rust
// NOT IMPLEMENTED - rust-numpy/src/dtype.rs
```

**Required Implementation:**
```rust
// Add to Dtype enum:
pub enum Dtype {
    // ... existing variants ...

    // Platform-dependent types
    Intp,   // i32 on 32-bit, i64 on 64-bit
    Uintp,  // u32 on 32-bit, u64 on 64-bit
}

// Add to from_type():
pub fn from_type<T: 'static>() -> Self {
    // ... existing code ...

    // Platform-dependent
    if type_id == TypeId::of::<usize>() {
        Dtype::Uintp
    } else if type_id == TypeId::of::<isize>() {
        Dtype::Intp
    }
}

// Add to kind():
Dtype::Intp | Dtype::Uintp => {
    if cfg!(target_pointer_width = "64") {
        DtypeKind::Integer // Or Unsigned
    } else {
        DtypeKind::Integer // Or Unsigned
    }
}

// Add to itemsize():
Dtype::Intp | Dtype::Uintp => std::mem::size_of::<usize>(),
```

**Implementation Complexity:** LOW (2-3 hours)

**Testing Required:**
- [ ] Test on 64-bit system → i64/u64
- [ ] Test on 32-bit system → i32/u32 (if possible)
- [ ] Parse "intp" string
- [ ] Parse "uintp" string
- [ ] from_type::<usize>() → Uintp
- [ ] from_type::<isize>() → Intp

---

### 2. f16 (IEEE 754 Half-Precision)

**Current Implementation (BROKEN):**
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

**Problem:**
- Uses simple bit shifting (not IEEE 754 compliant)
- Incorrect rounding, overflow handling, special values
- Will produce wrong results for many operations

**Required Implementation:**
```rust
// Option 1: Use half crate (RECOMMENDED)
// Add to Cargo.toml:
// half = "2.4"

use half::f16 as HalfF16;

pub type f16 = HalfF16;

// Option 2: Implement IEEE 754 conversion manually
// (Not recommended - complex and error-prone)
```

**Testing Required:**
- [ ] Test rounding behavior
- [ ] Test special values (inf, -inf, nan)
- [ ] Test overflow handling
- [ ] Compare against NumPy's float16 results

**Implementation Complexity:** LOW if using `half` crate (1 hour), HIGH if manual (1-2 weeks)

---

### 3. float128 / complex256 (Extended Precision)

**NumPy Behavior:**
```python
import numpy as np

# float128 is platform-dependent
print(np.float128)  # <class 'numpy.float128'> (on supported systems)

# complex256 uses float128 components
print(np.complex256) # <class 'numpy.complex256'> (on supported systems)
```

**Platform Support:**
- **Linux x86_64:** ✓ Supported (80-bit extended precision)
- **macOS ARM64:** ✗ Not supported (no 80-bit float)
- **Windows:** ⚠️ Variable support
- **WASM:** ✗ Not supported

**Required Implementation:**
```rust
// Add to Dtype enum:
pub enum Dtype {
    // ... existing variants ...

    // Extended precision (platform-dependent)
    Float128,    // 128-bit extended precision
    Complex256,   // 256-bit complex (2x Float128)
}

// Add feature flag:
// [features]
// longdouble = []

// Use soft-float crate for cross-platform support:
// soft-float = "0.1"
```

**Implementation Complexity:** HIGH (requires platform detection, testing on multiple systems)

**Testing Required:**
- [ ] Test on Linux x86_64
- [ ] Test on macOS ARM64 (expect unsupported)
- [ ] Test on Windows
- [ ] Test WASM (expect unsupported)
- [ ] Fallback behavior when not supported

---

### 4. bytes_ (Fixed-Width Binary Data)

**NumPy Behavior:**
```python
import numpy as np

# Fixed-width bytes
arr = np.array([b'a', b'b'], dtype='bytes')
arr = np.array([b'hello'], dtype='S10')  # 10-byte fixed-width
```

**Current Status:**
- `Dtype::String` exists but is always 8 bytes (pointer)
- No support for fixed-width binary data

**Required Implementation:**
```rust
// Add to Dtype enum:
pub enum Dtype {
    // ... existing variants ...

    // Fixed-width binary data
    Bytes { length: usize },
}

// Add parsing:
"b10" => Ok(Dtype::Bytes { length: 10 }),
"bytes" => Ok(Dtype::Bytes { length: 0 }), // Variable length
```

**Implementation Complexity:** MEDIUM (2-3 hours)

**Testing Required:**
- [ ] Parse "bytes" → variable length
- [ ] Parse "b10" → fixed length
- [ ] Calculate itemsize correctly
- [ ] Handle alignment (1 byte)

---

### 5. Fixed-Width Strings (S10, U10)

**NumPy Behavior:**
```python
import numpy as np

# Fixed-width byte strings
arr = np.array(['hello'], dtype='S10')  # 10-byte fixed-width

# Fixed-width unicode strings
arr = np.array(['hello'], dtype='U10')  # 10-char fixed-width unicode
```

**Current Status:**
- `Dtype::String` and `Dtype::Unicode` exist
- Always 8 bytes (pointer size)
- No support for fixed-width

**Required Implementation:**
```rust
// Modify existing enum:
pub enum Dtype {
    // ... existing variants ...

    // String types with optional fixed width
    String { length: Option<usize> },   // None = variable length
    Unicode { length: Option<usize> },  // None = variable length
}

// Parse formats:
"S10" => Ok(Dtype::String { length: Some(10) }),
"U10" => Ok(Dtype::Unicode { length: Some(10) }),
"str" => Ok(Dtype::String { length: None }),
"unicode" => Ok(Dtype::Unicode { length: None }),
```

**Implementation Complexity:** MEDIUM (3-4 hours)

**Testing Required:**
- [ ] Parse "S10" → fixed-width byte string
- [ ] Parse "U10" → fixed-width unicode
- [ ] Parse "str" → variable length
- [ ] Calculate itemsize correctly (char or byte length)
- [ ] Handle alignment

---

### 6. void (Padding Type)

**NumPy Behavior:**
```python
import numpy as np

# Void type for padding
dtype = np.dtype([('a', 'i4'), ('padding', 'V4'), ('b', 'i4')])
```

**Required Implementation:**
```rust
// Add to Dtype enum:
pub enum Dtype {
    // ... existing variants ...

    // Void type (padding)
    Void { size: usize },
}

// Parse format:
"V10" => Ok(Dtype::Void { size: 10 }),
```

**Implementation Complexity:** LOW (1-2 hours)

**Testing Required:**
- [ ] Parse "V10" → 10-byte void
- [ ] Calculate itemsize
- [ ] Handle alignment (1 byte)

---

## BUGS IN CURRENT IMPLEMENTATION

### Bug 1: Complex32 String Representation

**Location:** `dtype.rs` line 254

**Current Code:**
```rust
Dtype::Complex32 => "complex64".to_string(),  // BUG!
```

**Should Be:**
```rust
Dtype::Complex32 => "complex32".to_string(),
```

**Impact:** String conversion is incorrect

**Fix Complexity:** LOW (2 minutes)

---

### Bug 2: Complex32 Not Parsed

**Location:** `dtype.rs` from_str() method

**Current Code:**
```rust
"complex64" | "c8" => Ok(Dtype::Complex64),
"complex128" | "c16" => Ok(Dtype::Complex128),
```

**Missing:**
```rust
"complex32" | "c4" => Ok(Dtype::Complex32),
```

**Impact:** Cannot parse "complex32" or "c4"

**Fix Complexity:** LOW (2 minutes)

---

## MISSING FEATURES (Not Dtypes)

### 1. Dtype Promotion System

**Status:** ❌ NOT IMPLEMENTED

**Evidence:**
```rust
// ufunc.rs line 187
// Real implementation would need proper broadcasting, dtype promotion, etc.

// ufunc_ops.rs line 322
// This would need proper type promotion in real implementation
```

**Required:**
- Promotion matrix (dtype pair → result dtype)
- Weak/strong dtype tracking
- Safe casting validation
- Scalar vs array distinction

**Complexity:** HIGH (1-2 weeks)

**See:** DTYPE_ARCHITECTURE_ANALYSIS.md section 9 for implementation plan

---

### 2. Structured Dtype Offset Calculation

**Status:** ⚠️ PARTIALLY IMPLEMENTED

**Current Code:**
```rust
pub struct StructField {
    pub name: String,
    pub dtype: Dtype,
    pub offset: Option<usize>,  // ← Auto-calculation not implemented
}
```

**Required:**
- Auto-calculate offsets when `offset: None`
- Apply alignment padding between fields
- Calculate total struct size with padding

**Complexity:** MEDIUM (4-6 hours)

---

## IMPLEMENTATION PRIORITY ROADMAP

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix Complex32 string bug (2 minutes)
- [ ] Add Complex32 parsing (2 minutes)
- [ ] Implement `intp` and `uintp` (2-3 hours)
- [ ] Replace f16 with half crate (1 hour)

### Phase 2: Missing Features (Week 2)
- [ ] Implement dtype promotion (1-2 weeks)
- [ ] Implement structured dtype offset calculation (4-6 hours)
- [ ] Add fixed-width strings (3-4 hours)
- [ ] Add bytes_ type (2-3 hours)

### Phase 3: Extended Precision (Week 3-4)
- [ ] Implement float128 (conditional on platform)
- [ ] Implement complex256 (conditional on platform)
- [ ] Cross-platform testing

### Phase 4: Polish (Week 5)
- [ ] Add convenience aliases (single, double, etc.)
- [ ] Add void type
- [ ] Comprehensive testing
- [ ] Documentation

---

## TESTING CHECKLIST

### Priority 1 Tests
- [ ] `intp` / `uintp` parsing on 64-bit
- [ ] `intp` / `uintp` parsing on 32-bit (if possible)
- [ ] `intp` / `uintp` from_type inference
- [ ] `intp` / `uintp` itemsize/alignment
- [ ] `f16` IEEE 754 correctness
- [ ] `f16` special values (inf, -inf, nan)

### Priority 2 Tests
- [ ] Fixed-width string parsing ("S10", "U10")
- [ ] Fixed-width string itemsize
- [ ] Bytes_ parsing ("b10")
- [ ] Bytes_ itemsize/alignment
- [ ] Void parsing ("V10")
- [ ] Void itemsize/alignment

### Priority 3 Tests
- [ ] Float128 platform support
- [ ] Complex256 platform support
- [ ] Float128 fallback when unsupported
- [ ] Dtype promotion for all combinations
- [ ] Structured dtype offset auto-calculation

---

## COMPATIBILITY MATRIX

### Platform Support for Missing Dtypes

| Dtype | Linux x86_64 | macOS ARM64 | Windows x64 | WASM | Rust | Notes |
|-------|---------------|-------------|-------------|-------|-------|-------|
| **intp** | ✓ | ✓ | ✓ | ✓ | ✓ | Always available |
| **uintp** | ✓ | ✓ | ✓ | ✓ | ✓ | Always available |
| **f16 (IEEE)** | ✓ | ✓ | ✓ | ✓ | ✓ | Use half crate |
| **float128** | ✓ | ✗ | ⚠️ | ✗ | ⚠️ | Not on all platforms |
| **complex256** | ✓ | ✗ | ⚠️ | ✗ | ⚠️ | Not on all platforms |
| **bytes_** | ✓ | ✓ | ✓ | ✓ | ✓ | Always available |
| **fixed-string** | ✓ | ✓ | ✓ | ✓ | ✓ | Always available |
| **void** | ✓ | ✓ | ✓ | ✓ | ✓ | Always available |

---

## ESTIMATED IMPLEMENTATION TIME

| Task | Complexity | Time |
|------|------------|------|
| Fix Complex32 bugs | LOW | 5 minutes |
| Implement intp/uintp | LOW | 2-3 hours |
| Replace f16 with half crate | LOW | 1 hour |
| Implement bytes_ | MEDIUM | 2-3 hours |
| Implement fixed-width strings | MEDIUM | 3-4 hours |
| Implement void type | LOW | 1-2 hours |
| Implement dtype promotion | HIGH | 1-2 weeks |
| Implement structured offset calc | MEDIUM | 4-6 hours |
| Implement float128 | HIGH | 1-2 weeks (with testing) |
| Implement complex256 | HIGH | 1-2 weeks (with testing) |
| Comprehensive testing | MEDIUM | 1 week |
| **TOTAL** | | **4-6 weeks** |

---

## RECOMMENDATIONS

### Immediate Actions (This Week)
1. Fix Complex32 string bug (5 minutes)
2. Implement intp/uintp (2-3 hours)
3. Replace f16 with half crate (1 hour)

### Short-Term (Next Sprint)
1. Implement dtype promotion (1-2 weeks)
2. Implement fixed-width strings (3-4 hours)
3. Implement bytes_ (2-3 hours)

### Long-Term (Next Month)
1. Implement float128/complex256 (platform-dependent)
2. Complete structured dtype support
3. Comprehensive testing across platforms

---

## REFERENCES

- [NumPy Dtypes Documentation](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
- [IEEE 754 Half-Precision](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)
- [half crate](https://docs.rs/half/latest/half/)
- [soft-float crate](https://docs.rs/soft-float/latest/soft_float/)

---

**Analysis Complete**
**See DTYPE_ARCHITECTURE_ANALYSIS.md for detailed architecture**
**See DTYPE_QUICK_REFERENCE.md for quick lookup**
**See DTYPE_CODE_SNIPPETS.md for implementation details**
