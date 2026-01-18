# Rust NumPy Dtype System - Complete Documentation

**Documentation Date:** 2026-01-17
**Project:** rust-numpy (pure-Rust NumPy implementation)
**Analysis Coverage:** dtype.rs (358 lines), array.rs, ufunc.rs, constants.rs

---

## DOCUMENTATION INDEX

This directory contains comprehensive documentation about the rust-numpy dtype system architecture, implementation details, and missing features.

### 📚 Documentation Files

| File | Description | For |
|------|-------------|------|
| **[DTYPE_ARCHITECTURE_ANALYSIS.md](./DTYPE_ARCHITECTURE_ANALYSIS.md)** | Complete architecture analysis (12 sections) | Understanding the system |
| **[DTYPE_QUICK_REFERENCE.md](./DTYPE_QUICK_REFERENCE.md)** | Quick lookup tables and patterns | Daily development |
| **[DTYPE_CODE_SNIPPETS.md](./DTYPE_CODE_SNIPPETS.md)** | All code extracted and organized | Implementation reference |
| **[DTYPE_MISSING_TYPES.md](./DTYPE_MISSING_TYPES.md)** | Missing dtypes and bugs | What needs to be done |
| **[README_DTYPE.md](./README_DTYPE.md)** | This file - overview | Getting started |

---

## 🎯 QUICK START

### What This Is

A complete analysis of the rust-numpy dtype system covering:
- **21+ dtype variants** (43 total with datetime units)
- **7 public methods** for dtype manipulation
- **Missing NumPy dtypes** with implementation plans
- **Architecture patterns** for extension

### Current Status

| Category | Status |
|----------|--------|
| **Core Dtypes** | ✅ 80% complete (int8-64, uint8-64, float16-64, complex32-128) |
| **String Types** | ⚠️ 50% complete (missing fixed-width) |
| **Datetime Types** | ✅ 100% complete (12 units each) |
| **Platform Types** | ❌ 0% complete (missing intp, uintp) |
| **Extended Precision** | ❌ 0% complete (missing float128, complex256) |
| **Binary Types** | ❌ 0% complete (missing bytes_) |
| **Dtype Promotion** | ❌ 0% complete (critical blocker) |

---

## 📊 IMPLEMENTED DTYPES

### Numeric Types (15)

| Category | Dtypes | Count |
|----------|--------|-------|
| **Signed Integers** | Int8, Int16, Int32, Int64 | 4 |
| **Unsigned Integers** | UInt8, UInt16, UInt32, UInt64 | 4 |
| **Floating Point** | Float16, Float32, Float64 | 3 |
| **Complex Numbers** | Complex32, Complex64, Complex128 | 3 |

### Other Types (8)

| Category | Dtypes | Count |
|----------|--------|-------|
| **Boolean** | Bool | 1 |
| **Strings** | String, Unicode | 2 |
| **Datetime** | Datetime64(12 units), Timedelta64(12 units) | 24 |
| **Object** | Object | 1 |
| **Structured** | Struct(Vec<StructField>) | 1 |

**Total Variants:** 21+ (43 total with datetime units)

---

## ❌ MISSING DTYPES

### Critical (Blocking NumPy Parity)

| Dtype | Description | Priority |
|-------|-------------|----------|
| **intp** | Platform-dependent int (i32/i64) | HIGH |
| **uintp** | Platform-dependent uint (u32/u64) | HIGH |
| **f16 (IEEE 754)** | Proper half-precision float | HIGH |

### Important (Feature Complete)

| Dtype | Description | Priority |
|-------|-------------|----------|
| **float128** | Extended precision float | MEDIUM |
| **complex256** | Extended precision complex | MEDIUM |
| **bytes_** | Fixed-width binary data | MEDIUM |
| **fixed-string** | S10, U10 format strings | MEDIUM |

### Nice to Have

| Dtype | Description | Priority |
|-------|-------------|----------|
| **void** | Padding type | LOW |
| **single, double** | Convenience aliases | LOW |

**See:** [DTYPE_MISSING_TYPES.md](./DTYPE_MISSING_TYPES.md) for detailed analysis

---

## 🐛 KNOWN BUGS

| Bug | Location | Impact | Fix Time |
|-----|----------|--------|----------|
| Complex32 → "complex64" | dtype.rs:254 | String conversion wrong | 2 min |
| Complex32 not parsed | dtype.rs:204-205 | Can't parse "complex32" | 2 min |

---

## 🏗️ ARCHITECTURE

### Core Files

```
rust-numpy/src/
├── dtype.rs              # Main dtype system (358 lines)
├── array.rs              # Array<T> with dtype field
├── ufunc.rs              # Ufunc trait with dtype support
├── ufunc_ops.rs          # Ufunc execution (needs promotion)
├── constants.rs          # Type-specific constants
└── lib.rs                # Re-exports
```

### Key Components

```rust
// Main enum
pub enum Dtype {
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Complex32, Complex64, Complex128,
    Bool, String, Unicode,
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),
    Object,
    Struct(Vec<StructField>),
}

// Classification
pub enum DtypeKind {
    Integer, Unsigned, Float, Complex,
    Bool, String, Datetime, Object, Struct,
}

// Methods
impl Dtype {
    pub fn kind(&self) -> DtypeKind;
    pub fn itemsize(&self) -> usize;
    pub fn alignment(&self) -> usize;
    pub fn from_type<T: 'static>() -> Self;
    pub fn from_str(s: &str) -> Result<Self, String>;
    pub fn to_string(&self) -> String;
    pub fn can_cast_to(&self, other: &Dtype) -> bool;
}
```

**See:** [DTYPE_ARCHITECTURE_ANALYSIS.md](./DTYPE_ARCHITECTURE_ANALYSIS.md) for complete architecture

---

## 📖 USAGE EXAMPLES

### Creating Dtypes

```rust
use numpy::dtype::*;

// From string
let dt = Dtype::from_str("int32").unwrap();

// From Rust type
let dt = Dtype::from_type::<f64>();

// Complex with datetime
let dt = Dtype::Datetime64(DatetimeUnit::ns);
```

### Inspecting Dtypes

```rust
let dt = Dtype::Int32;

dt.kind();        // DtypeKind::Integer
dt.itemsize();    // 4 bytes
dt.alignment();   // 4 bytes
dt.to_string();   // "int32"
```

### Safe Casting

```rust
let int32 = Dtype::Int32;
let float64 = Dtype::Float64;

if int32.can_cast_to(&float64) {
    println!("Safe to cast");
}
```

**See:** [DTYPE_CODE_SNIPPETS.md](./DTYPE_CODE_SNIPPETS.md) for more examples

---

## 🚀 IMPLEMENTATION ROADMAP

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix Complex32 string bug (2 minutes)
- [ ] Add Complex32 parsing (2 minutes)
- [ ] Implement intp/uintp (2-3 hours)
- [ ] Replace f16 with half crate (1 hour)

### Phase 2: Missing Features (Week 2)
- [ ] Implement dtype promotion (1-2 weeks) ← **BLOCKER**
- [ ] Implement structured offset calculation (4-6 hours)
- [ ] Add fixed-width strings (3-4 hours)
- [ ] Add bytes_ type (2-3 hours)

### Phase 3: Extended Precision (Week 3-4)
- [ ] Implement float128 (platform-dependent)
- [ ] Implement complex256 (platform-dependent)
- [ ] Cross-platform testing

### Phase 4: Polish (Week 5)
- [ ] Add convenience aliases
- [ ] Add void type
- [ ] Comprehensive testing

**Estimated Total:** 4-6 weeks for full NumPy parity

---

## 🔧 DEVELOPMENT NOTES

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| num-complex | 0.4 | Complex number types |
| num-traits | 0.2 | Numeric traits |
| chrono | 0.4 (optional) | Datetime support |

### Recommended Additions

| Crate | Version | For What |
|-------|---------|----------|
| half | 2.4 | IEEE 754 half-precision float |
| soft-float | 0.1 | Cross-platform float128 |

---

## 📚 ADDITIONAL RESOURCES

### NumPy Documentation
- [NumPy Dtypes](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
- [NumPy dtype promotion](https://numpy.org/doc/stable/reference/ufuncs.html#casting-rules)
- [NumPy type promotion](https://numpy.org/doc/stable/reference/arrays.dtypes.html#generalized-ufuncs)

### Rust Crates
- [half crate](https://docs.rs/half/latest/half/)
- [num-complex](https://docs.rs/num-complex/latest/num_complex/)
- [soft-float](https://docs.rs/soft-float/latest/soft_float/)

---

## 📋 QUICK REFERENCE CHEAT SHEET

### All Dtype Variants
```
Int8, Int16, Int32, Int64
UInt8, UInt16, UInt32, UInt64
Float16, Float32, Float64
Complex32, Complex64, Complex128
Bool, String, Unicode
Datetime64(DatetimeUnit)
Timedelta64(TimedeltaUnit)
Object
Struct(Vec<StructField>)
```

### Datetime Units
```
Y, M, W, D, h, m, s, ms, us, ns, ps, fs, As
```

### DtypeKind
```
Integer, Unsigned, Float, Complex, Bool, String, Datetime, Object, Struct
```

### Method Signatures
```rust
dt.kind() -> DtypeKind
dt.itemsize() -> usize
dt.alignment() -> usize
Dtype::from_type<T>() -> Dtype
Dtype::from_str(s: &str) -> Result<Dtype>
dt.to_string() -> String
dt.can_cast_to(&other) -> bool
```

---

## 🤝 CONTRIBUTING

### Adding New Dtypes

1. **Add to Dtype enum** in `dtype.rs`
2. **Add to kind()** method
3. **Add to itemsize()** method
4. **Add to alignment()** method
5. **Add to from_str()** parsing
6. **Add to to_string()** conversion
7. **Add tests** in `tests/basic_tests.rs`
8. **Update documentation**

### Adding New Methods

1. **Add to impl Dtype** block
2. **Add docstring** following NumPy conventions
3. **Add tests** covering all dtype variants
4. **Update quick reference**

**See:** [DTYPE_ARCHITECTURE_ANALYSIS.md](./DTYPE_ARCHITECTURE_ANALYSIS.md) section 9 for dtype promotion implementation

---

## 📞 CONTACT & SUPPORT

- **Project Repository:** https://github.com/grantjr1842/pocket-tts
- **NumPy Issues:** https://github.com/numpy/numpy/issues
- **Rust NumPy Questions:** Use project issues

---

## 📄 DOCUMENTATION METADATA

| Metric | Value |
|--------|-------|
| **Documentation Version** | 1.0 |
| **Analysis Date** | 2026-01-17 |
| **Source Files Analyzed** | 5 (dtype.rs, array.rs, ufunc.rs, constants.rs, lib.rs) |
| **Total Lines of Code** | ~1500 |
| **Dtype Variants** | 21+ (43 with units) |
| **Public Methods** | 7 |
| **Missing Dtypes** | 8+ |
| **Known Bugs** | 2 |
| **Estimated Implementation Time** | 4-6 weeks |

---

## 🎓 LEARNING RESOURCES

### For New Contributors

1. **Start Here:** [DTYPE_QUICK_REFERENCE.md](./DTYPE_QUICK_REFERENCE.md)
2. **Deep Dive:** [DTYPE_ARCHITECTURE_ANALYSIS.md](./DTYPE_ARCHITECTURE_ANALYSIS.md)
3. **Implementation:** [DTYPE_CODE_SNIPPETS.md](./DTYPE_CODE_SNIPPETS.md)
4. **Missing Work:** [DTYPE_MISSING_TYPES.md](./DTYPE_MISSING_TYPES.md)

### For Implementation

1. **Copy Code:** From [DTYPE_CODE_SNIPPETS.md](./DTYPE_CODE_SNIPPETS.md)
2. **Follow Patterns:** From existing dtype variants
3. **Add Tests:** Following existing test structure
4. **Update Docs:** Add to relevant documentation file

---

**Documentation Complete**

Last Updated: 2026-01-17
Maintained by: rust-numpy team

For questions or contributions, please refer to the project repository.
