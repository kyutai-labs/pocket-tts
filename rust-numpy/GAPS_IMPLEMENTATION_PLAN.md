# Plan: Complete rust-numpy Port Gaps

**Date:** 2026-01-18
**Session:** ralph-loop-2025-01-18-rust-numpy
**Status:** Plan Phase

---

## Executive Summary

Based on analysis of `rust-numpy/SESSION_STATE.md` and `DTYPE_MISSING_TYPES.md`, previous session completed 14/14 core tasks (100% completion) including performance optimizations, API enhancements, testing infrastructure, and documentation. However, one critical gap remains:

1. **Missing NumPy dtype parsing support** - intp/uintp strings cannot be parsed
2. **Broken f16 implementation** - Uses incorrect bit-shifting instead of IEEE 754

This plan focuses purely on **Rust dtype fixes** without Python bindings.

---

## Gap Analysis

### Gap 1: Missing Dtype Types (HIGH PRIORITY)

**Reference:** `rust-numpy/DTYPE_MISSING_TYPES.md`

**Current State:**
- Core implementation is complete and production-ready
- 21+ dtype variants implemented (43 total with units)
- NumPy 1.x has ~50 dtype variants
- Missing critical types that prevent NumPy parity

**Missing Types by Priority:**

| Priority | Type | Status | Impact |
|----------|-------|--------|---------|
| **CRITICAL** | `intp` | ❌ Not implemented | Platform-dependent indexing |
| **CRITICAL** | `uintp` | ❌ Not implemented | Platform-dependent indexing |
| **CRITICAL** | `f16` (IEEE 754) | ⚠️ Broken (simplified) | Incorrect calculations |
| **HIGH** | `float128` | ❌ Not implemented | Extended precision |
| **HIGH** | `complex256` | ❌ Not implemented | Extended precision |
| **MEDIUM** | `bytes_` | ❌ Not implemented | Binary data support |
| **MEDIUM** | Fixed-width strings | ⚠️ Partial | NumPy compatibility |
| **LOW** | `void` | ❌ Not implemented | Padding types |

**Issues Found:**
1. **Complex32 string bug** - Displays as "complex64" instead of "complex32"
2. **Complex32 not parsed** - Cannot parse "complex32" or "c4" strings
3. **f16 implementation** - Uses bit shifting instead of IEEE 754 (incorrect results)

---

### Gap 2: Missing Python Bindings (HIGH PRIORITY)

**Reference:** `NUMPY_TO_RUST_COMPLETION_PLAN.md`

**Current State:**
- rust-numpy is a pure Rust library
- No PyO3 bindings exist
- Cannot be used directly from Python
- pocket-tts still relies on standard NumPy

**Requirements:**
1. PyO3 bindings for core array type
2. NumPy function bindings (arange, array, clip, min, max, mean, median, log, sum, sqrt)
3. Python module that compiles to .so/.dylib/.dll
4. Drop-in NumPy replacement with fallback
5. Integration with pocket-tts codebase

---

## Implementation Plan

### Phase 1: Critical Dtype Fixes (Priority 1)

#### Task 1.1: Fix Complex32 Parsing Bug
**File:** `rust-numpy/src/dtype.rs`

**Changes Required:**
- Add "complex32" and "c4" parsing to `from_str()` method (around line 298)

**Note:** The string conversion is already correct (line 347 shows "complex32"), but parsing is missing.

**Estimated Time:** 10 minutes
**Priority:** CRITICAL

---

#### Task 1.2: Add intp and uintp Parsing Support
**File:** `rust-numpy/src/dtype.rs`

**Changes Required:**
- Add "intp" | "ip" parsing to `from_str()` method
- Add "uintp" | "up" parsing to `from_str()` method

**Note:** Intp and Uintp are already IMPLEMENTED in the Dtype enum (lines 21, 28), and from_type() already handles them (lines 57-58, 67-68), but they cannot be parsed from strings!

**Estimated Time:** 10 minutes
**Priority:** CRITICAL

---

#### Task 1.3: Replace f16 with IEEE 754 Compliant Implementation
**File:** `rust-numpy/src/dtype.rs` and `rust-numpy/Cargo.toml`

**Changes Required:**
1. Add `half = "2.4"` to Cargo.toml dependencies
2. Replace simplified f16 struct (lines 440-463) with `use half::f16 as HalfF16`
3. Remove incorrect bit-shifting implementation

**Estimated Time:** 1 hour
**Priority:** CRITICAL

**Tests Required:**
- Test rounding behavior
- Test special values (inf, -inf, nan)
- Test overflow handling
- Compare against NumPy's float16 results

---

### Phase 2: Testing and Validation (Priority 2)

#### Task 4.1: Comprehensive Rust Tests
**File:** `rust-numpy/src/dtype_tests.rs` (NEW FILE)

**Tests Required:**
- Complex32 string representation
- Complex32 parsing
- intp/uintp parsing and type inference
- f16 IEEE 754 correctness
- bytes_ parsing and itemsize
- Fixed-width string parsing
- void parsing

**Estimated Time:** 2-3 hours
**Priority:** HIGH

---

#### Task 4.2: Run Full Test Suite
**Commands:**
```bash
# Run rust-numpy tests
cd rust-numpy
cargo test --all-features

# Run Python integration tests
pytest tests/python_integration_tests.py -v

# Run pocket-tts tests with rust-numpy
cd ..
pytest tests/test_python_api.py -v
pytest tests/test_documentation_examples.py -v
```

**Estimated Time:** 1-2 hours
**Priority:** HIGH

---

## Estimated Effort Summary

| Phase | Tasks | Estimated Time | Lines of Code |
|--------|--------|----------------|----------------|
| **Phase 1: Critical Dtype Fixes** | 3 tasks | 4 hours | ~50 |
| **Phase 2: Medium Priority Dtypes** | 3 tasks | 8 hours | ~150 |
| **Phase 3: Python Bindings** | 7 tasks | 14-17 hours | ~900 |
| **Phase 4: Testing and Validation** | 2 tasks | 3-5 hours | ~200 |
| **TOTAL** | 15 tasks | **29-34 hours** (4-5 days) | **~1,300 lines** |

---

## Success Criteria

✅ **Complete when:**

1. All critical dtype bugs fixed (Complex32, intp, uintp, f16)
2. All missing dtype types implemented (bytes_, fixed-width strings, void)
3. Python bindings compile and import successfully
4. All existing pocket-tts tests pass with rust-numpy
5. Full test coverage for new features
6. Documentation updated with complete usage instructions

---

## Risks and Mitigations

### High Priority Risks

1. **PyO3 Integration Complexity**
   - Risk: Memory ownership and array conversion issues
   - Mitigation: Comprehensive testing with real pocket-tts workloads
   - Fallback: NumPy if rust-numpy fails

2. **Platform-Specific Dtypes**
   - Risk: intp/uintp behavior on 32-bit vs 64-bit
   - Mitigation: Test on both architectures, use cfg! macros

3. **f16 IEEE 754 Compliance**
   - Risk: Incorrect behavior for edge cases
   - Mitigation: Use tested `half` crate, compare with NumPy

### Medium Priority Risks

4. **Build System Complexity**
   - Risk: Cross-platform compilation (Linux/macOS/Windows)
   - Mitigation: Use pyproject.toml, test on all platforms

5. **Performance Regression**
   - Risk: FFI overhead negates rust-numpy benefits
   - Mitigation: Benchmark before/after, keep fallback option

---

## Next Steps After Completion

1. Remove `training/rust_exts/audio_ds/` (deprecated)
2. Archive `rust-numpy/README.md` with completion status
3. Create migration guide for other NumPy users
4. Performance monitoring in production
5. Consider adding more NumPy functions as needed (float128, complex256)

---

## Dependencies

None required - all work uses existing rust-numpy infrastructure and adds standard Rust crates (half) and PyO3.

---

**Estimated completion:** 4-5 days from approval
**Risk level:** Medium-High
**Complexity:** High (requires dtype expertise, PyO3 integration, and cross-platform testing)
