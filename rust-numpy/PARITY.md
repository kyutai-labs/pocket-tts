# Rust-NumPy Parity Analysis & Roadmap

**Last Updated:** 2026-01-24
**Status:** In Progress
**Completion:** ~82% estimated (See [INVENTORY.md](INVENTORY.md) for details)

## Executive Summary

This document provides a comprehensive analysis of rust-numpy's implementation status compared to NumPy 1.26+ API, identifying gaps and providing a prioritized roadmap for achieving 100% parity.

For a detailed list of all implemented functions, see [INVENTORY.md](INVENTORY.md).

---

## Current Implementation Status

### ✅ Completed Modules (Core Functionality)

| Module                 | Status     | Coverage | Notes                                                |
|------------------------|------------|----------|------------------------------------------------------|
| **dtype**              | ✅ Complete | 100%     | All dtypes, byte order, casting rules                |
| **array_creation**     | ✅ Complete | 95%      | zeros, ones, arange, linspace, empty, full           |
| **array_manipulation** | ✅ Complete | 100%     | reshape, transpose, flatten, ravel, squeeze          |
| **broadcasting**       | ✅ Complete | 100%     | NumPy broadcasting semantics                         |
| **math_ufuncs**        | ✅ Complete | 100%     | add, subtract, multiply, divide, power, mod          |
| **comparison_ufuncs**  | ✅ Complete | 100%     | greater, less, equal, logical ops                    |
| **bitwise**            | ✅ Complete | 100%     | and, or, xor, invert, shifts                         |
| **statistics**         | ✅ Complete | 100%     | mean, var, std, min, max, sum, prod, nan-aware stats |
| **sorting**            | ✅ Complete | 100%     | sort, argsort, searchsorted                          |
| **set_ops**            | ✅ Complete | 100%     | unique, intersect1d, setdiff1d, union1d              |
| **string_ops**         | ✅ Complete | 85%      | Basic string operations                              |
| **io**                 | ✅ Complete | 85%      | load, save, text file I/O                            |
| **datetime**           | ✅ Complete | 75%      | datetime64, timedelta64 support                      |
| **random**             | ⚠️ Partial | 60%      | Various distributions, needs trait fixes             |
| **constants**          | ✅ Complete | 100%     | pi, e, inf, nan                                      |
| **strides**            | ✅ Complete | 100%     | Stride manipulation                                  |
| **memory**             | ✅ Complete | 100%     | Memory management                                    |
| **error**              | ✅ Complete | 100%     | Error types                                          |

### 🟡 Partial Implementation (Needs Work)

| Module                 | Status     | Missing                           | Priority                           |
|------------------------|------------|-----------------------------------|------------------------------------|
| **ufunc**              | ✅ Complete | 100%                              | Execution engine, where=, casting= |
| **ufunc_ops**          | ✅ Complete | 95%                               | Reduction ops, axis parameter      |
| **linalg**             | ⚠️ 65%     | See linalg section below          | HIGH                               |
| **fft**                | ⚠️ 50%     | Full FFT suite, multi-dimensional | MEDIUM                             |
| **polynomial**         | ⚠️ 60%     | All polynomial classes            | MEDIUM                             |
| **advanced_broadcast** | ⚠️ 80%     | Edge cases                        | LOW                                |
| **window**             | ⚠️ 70%     | All window functions              | LOW                                |
| **parallel_ops**       | ⚠️ 50%     | Thread safety, performance        | MEDIUM                             |
| **simd_ops**           | ⚠️ 40%     | SIMD kernels, runtime dispatch    | MEDIUM                             |
| **type_promotion**     | ⚠️ 70%     | Edge cases, complex rules         | MEDIUM                             |

### ❌ Missing / Not Started

| Module                   | Status    | Description                | Priority |
|--------------------------|-----------|----------------------------|----------|
| **ma** (masked arrays)   | ✅ 75%    | Mostly complete (see #342) | LOW      |
| **matrix**               | ❌ Missing | @ operator, matrix class   | LOW      |
| **rec** (recarray)       | ❌ Missing | Record arrays              | LOW      |
| **char** (character)     | ✅ 95%    | Nearly complete (see #353) | LOW      |
| **dist**                 | ❌ Missing | Distance matrices          | LOW      |
| **polynomial** (new API) | ✅ 85%    | Mostly complete (see #339)  | LOW      |
| **typing**               | ❌ Missing | Type hints                 | N/A      |
| **testing**              | ✅ 90%     | Testing utilities          | MEDIUM   |

---

## Critical Gaps Analysis

### 1. Linear Algebra (linalg) - 40% Complete

#### ✅ Implemented

- `norm()` - Vector/matrix norms (completed in issue #156)
- `svd()` - Singular Value Decomposition (completed in issue #243)
- Basic bridge functions

#### ❌ Missing (High Priority)

- **Decompositions**:
  - [ ] `qr()` - QR Decomposition (issue #57)
  - [ ] `cholesky()` - Cholesky decomposition
  - [ ] `eig()` - Eigen-decomposition (issue #56)
  - [ ] `lstsq()` - Least squares solver (issue #60)

- **Matrix Operations**:
  - [ ] `inv()` - Matrix inversion
  - [ ] `pinv()` - Pseudo-inverse
  - [ ] `matrix_power()` - Matrix exponentiation
  - [ ] `matrix_rank()` - Matrix rank
  - [ ] `det()` - Determinant
  - [ ] `trace()` - Matrix trace

- **Solvers**:
  - [ ] `solve()` - Linear system solver
  - [ ] `tensorsolve()` - Tensor solver
  - [ ] `tensorinv()` - Tensor inverse

#### Dependencies

- Issue #60 (lstsq)
- Issue #56 (eig)
- Issue #57 (qr)
- Issue #58 (svd)

---

### 2. Universal Functions (ufunc) - 70% Complete

#### ✅ Implemented

- Basic ufunc trait structure
- Registration framework
- Binary operations (add, sub, mul, div)
- Unary operations (sin, cos, exp, log)
- Comparison operations
- Bitwise operations
- `where=` parameter (issue #246)
- `casting=` parameter (issue #246)

#### ❌ Missing (High Priority)

- **Execution Engine**:
  - [ ] Strided execution

- **Kernel System**:
  - [ ] Kernel registry by dtype signature (issue #38)
  - [ ] Kernel selection logic
  - [ ] Type-specific kernels
  - [ ] SIMD kernels (feature-gated) (issue #46)

- **Reductions** (issue #43):
  - [ ] sum with axis parameter
  - [ ] prod with axis parameter
  - [ ] min/max with axis parameter
  - [ ] all/any with axis parameter
  - [ ] keepdims parameter

#### Dependencies (Issues #37-43)

- #37: Casting policy (Safe/SameKind/Unsafe) ✅ COMPLETED
- #38: UFunc registry + kernel lookup
- #39: Minimal execution engine
- #40: Broadcast-aware binary exec
- #41: Public Array facade
- #42: mul ufunc
- #43: Global sum reduction

---

### 3. FFT Module - 50% Complete

#### ✅ Implemented

- Basic FFT interface
- 1D FFT/IFFT
- Complex number support

#### ❌ Missing (Medium Priority)

- [ ] `fft2()` - 2D FFT
- [ ] `fftn()` - N-D FFT
- [ ] `ifft2()` - 2D IFFT
- [ ] `ifftn()` - N-D IFFT
- [ ] `rfft()` - Real FFT
- [ ] `irfft()` - Real IFFT
- [ ] `rfftn()` - N-D real FFT
- [ ] `irfftn()` - N-D real IFFT
- [ ] `hfft()` - Hermitian FFT
- [ ] `ihfft()` - Hermitian IFFT
- [ ] `fftfreq()` - Frequency bins
- [ ] `rfftfreq()` - Real FFT frequency bins
- [ ] `fftshift()` - Shift zero-frequency
- [ ] `ifftshift()` - Inverse shift

---

### 4. Polynomial Module - 60% Complete

#### ✅ Implemented

- Basic polynomial operations
- Polynomial evaluation
- Some polynomial classes (Legendre, etc.)

#### ❌ Missing (Medium Priority - Issue #53)

- [ ] Complete `Polynomial` class
- [ ] `Chebyshev` class (partial exists)
- [ ] `Legendre` class (partial exists)
- [ ] `Hermite` class (partial exists)
- [ ] `HermiteE` class (partial exists)
- [ ] `Laguerre` class (partial exists)
- [ ] Polynomial fitting methods
- [ ] Polynomial arithmetic
- [ ] Roots finding

---

### 5. Array Methods & Operations - 85% Complete

#### ✅ Implemented

- Creation methods (zeros, ones, empty, full, etc.)
- Shape manipulation (reshape, transpose, flatten, etc.)
- Basic indexing/slicing
- Comparison operations
- Mathematical operations
- Statistical operations

#### ❌ Missing / Issues (Medium Priority)

**Advanced Indexing** (Issue #245):

- [ ] Integer array indexing
- [ ] Boolean array indexing (partial support added)
- [ ] Fancy indexing (partial support added)
- [ ] Ellipsis indexing
- [ ] Newaxis

**Missing Methods**:

- [ ] `choose()` - Choose from array
- [ ] `compress()` - Conditional selection
- [ ] `diagonal()` - Array diagonals
- [ ] `diag()` - Diagonal construction/extraction
- [ ] `ptp()` - Peak-to-peak
- [ ] `round()` - Rounding with decimals
- [ ] `clip()` - Limit values
- [ ] `trim_zeros()` - Trim leading/trailing zeros
- [ ] `ediff1d()` - Differences between elements

---

### 6. Statistics Module - 85% Complete

#### ✅ Implemented

- mean, var, std
- min, max, ptp
- sum, prod, cumsum, cumprod
- nanmean, nanvar, nanstd, nanmin, nanmax, nansum, nanprod (issue #244)

#### ❌ Missing (Issue #50 - NaN-Aware Statistics)

- [ ] `nancumsum()` - Cumsum ignoring NaN
- [ ] `nancumprod()` - Cumprod ignoring NaN
- [ ] `argmin()` / `argmax()` - Index of min/max (partially)
- [ ] `nanargmin()` / `nanargmax()` - Index ignoring NaN
- [ ] `median()` - Median value
- [ ] `nanmedian()` - Median ignoring NaN
- [ ] `percentile()` - Percentile
- [ ] `nanpercentile()` - Percentile ignoring NaN
- [ ] `quantile()` - Quantile
- [ ] `nanquantile()` - Quantile ignoring NaN
- [ ] `corrcoef()` - Correlation coefficient
- [ ] `cov()` - Covariance
- [ ] `average()` - Weighted average

---

### 7. Type Promotion - 70% Complete

#### ✅ Implemented

- Basic promotion rules
- Numeric kind detection
- Safe casting checks

#### ❌ Missing (Issue #77 - Numeric Promotion Rules)

- [ ] Complete promotion table for all dtype combinations
- [ ] Division-specific rules (true_div vs floor_div)
- [ ] Complex number promotion
- [ ] Scalar-array promotion
- [ ] Precision preservation rules
- [ ] Operation-specific promotion (add, sub, mul, comparison, bitwise)

---

### 8. Performance & Optimization - 40% Complete

#### ✅ Implemented

- Basic parallel operations
- Some SIMD operations (partial)

#### ❌ Missing (Medium-High Priority)

- [ ] Multi-threading policy (Issue #47)
- [ ] SIMD kernels with runtime dispatch (Issue #46)
- [ ] Dimension coalescing (Issue #45)
- [ ] Memory layout optimization
- [ ] Cache-friendly algorithms
- [ ] Parallel reduction

---

### 9. Testing & Validation - 30% Complete

#### ✅ Implemented

- Basic unit tests in some modules
- Some integration tests
- `assert_array_equal`, `assert_array_almost_equal`, `assert_array_shape_equal` (issue #247)

#### ❌ Missing (High Priority - Issue #54, #65)

- [ ] Comprehensive test coverage (>80%)
- [ ] Property-based tests (proptest)
- [ ] Conformance tests against NumPy
- [ ] Performance benchmarks
- [ ] PARITY.md with checklist (Issue #65)
- [ ] CI test integration

---

## Build & Infrastructure Status

### Current Build Status

- ✅ dtype module: Compiles successfully
- ✅ Core array operations: Compiles
- ✅ Math operations: Compiles
- ⚠️ ufunc system: Has compilation errors (trait implementation issues)
- ⚠️ random module: Has trait bound issues
- ❌ Test suite: 145 compilation errors

### Issues to Fix

1. **Trait implementation mismatches** - RandomGenerator trait stricter than definition
2. **Missing methods** - Some Array methods referenced but not implemented
3. **Type annotation errors** - Need explicit type annotations in some places

---

## Prioritized Roadmap

### Phase 1: Critical Foundation (Weeks 1-4)

**Goal:** Fix compilation, complete core ufunc system

1. **Fix Compilation Errors** (Week 1)
   - Fix RandomGenerator trait implementations
   - Resolve type annotation issues
   - Fix missing Array methods
   - Get test suite compiling

2. **Complete Ufunc System** (Weeks 2-3) - Issues #37-43
   - Finish casting policy ✅
   - Implement ufunc registry
   - Build minimal execution engine
   - Add broadcast-aware execution
   - Implement Public Array API
   - Add multiplication ufunc
   - Implement sum reduction

3. **Test Infrastructure** (Week 4) - Issue #54, #65
   - Set up comprehensive test framework
   - Add property-based testing
   - Create PARITY.md checklist

### Phase 2: Core Numeric Algorithms (Weeks 5-8)

**Goal:** Complete linear algebra and statistics

4. **Linear Algebra** (Weeks 5-7) - Issues #56-60
   - Implement SVD (#58)
   - Implement QR (#57)
   - Implement Eigen-decomposition (#56)
   - Implement Least Squares (#60)
   - Add matrix inversion, det, trace

5. **NaN-Aware Statistics** (Week 8) - Issue #50
   - Implement all nan\* functions
   - Add median/percentile/quantile
   - Add correlation/covariance

### Phase 3: Advanced Features (Weeks 9-12)

**Goal:** FFT, polynomials, advanced operations

6. **Complete FFT Module** (Week 9-10)
   - 2D/N-D FFT
   - Real FFT variants
   - Frequency utilities

7. **Complete Polynomial Module** (Week 11) - Issue #53
   - All polynomial classes
   - Complete arithmetic
   - Fitting methods

8. **Advanced Indexing** (Week 12) - Issue #51
   - Integer array indexing
   - Boolean array indexing
   - Fancy indexing

### Phase 4: Performance & Polish (Weeks 13-16)

**Goal:** Optimization and 100% parity

9. **Performance Optimization** (Weeks 13-14) - Issues #45-47
   - SIMD kernels (#46)
   - Dimension coalescing (#45)
   - Multi-threading policy (#47)
   - Runtime dispatch

10. **Type Promotion** (Week 15) - Issue #77
    - Complete promotion rules
    - All operation-specific rules
    - Test coverage

11. **Final Parity Gap Analysis** (Week 16) - Issue #89
    - Comprehensive audit
    - Test against NumPy conformance
    - Documentation updates

---

## Dependencies

### Issue Dependency Graph

```
# Foundation
Issue #37 (Casting) ✅ -> Issue #38 (Registry) -> Issue #39 (Exec Engine)
                                                   -> Issue #40 (Broadcast)
                                                      -> Issue #41 (Public API)
                                                         -> Issue #42 (mul ufunc)
                                                         -> Issue #43 (sum)

# Testing
Issue #65 (PARITY.md) -> Issue #54 (Test Coverage)

# Feature Sets
Issue #50 (NaN stats) - Independent
Issue #51 (Advanced indexing) - Depends on basic indexing
Issue #53 (Polynomials) - Mostly independent
Issue #56-60 (Linalg) - Can work in parallel
Issue #77 (Type promotion) - Supports all operations
Issue #89 (Gap analysis) - Depends on most other issues
```

---

## Success Metrics

### Completion Criteria

- [ ] All 1000+ NumPy functions have Rust equivalents
- [ ] > 95% test coverage
- [ ] All tests pass
- [ ] Performance within 2x of NumPy for core operations
- [ ] Full documentation
- [ ] PARITY.md shows 100% completion

### Quality Metrics

- [ ] Zero compilation warnings
- [ ] All unsafe blocks audited and documented
- [ ] Memory safety verified (Miri, sanitizers)
- [ ] Benchmarks established
- [ ] Conformance tests passing

---

## Estimated Effort

| Phase                      | Duration     | Engineer-weeks | Complexity  |
|----------------------------|--------------|----------------|-------------|
| Phase 1: Foundation        | 4 weeks      | 4              | HIGH        |
| Phase 2: Core Algorithms   | 4 weeks      | 8              | HIGH        |
| Phase 3: Advanced Features | 4 weeks      | 8              | MEDIUM      |
| Phase 4: Performance       | 4 weeks      | 8              | MEDIUM-HIGH |
| **Total**                  | **16 weeks** | **28**         |             |

**Note:** This is optimistic. With debugging, testing, and unforeseen issues, estimate 24-32 weeks (6-8 months) for 100% parity.

---

## Resource Requirements

### Personnel

- 1-2 Senior Rust engineers (full-time)
- 1 NumPy domain expert (part-time, consultant)
- 1 QA/Test engineer (part-time, Phase 2+)

### Tools & Infrastructure

- CI/CD pipeline (GitHub Actions)
- Benchmarking infrastructure
- Test farm (multiple architectures)
- Documentation tools
- Code coverage tools

---

## Risk Assessment

| Risk                       | Impact | Probability | Mitigation                           |
|----------------------------|--------|-------------|--------------------------------------|
| Complex numeric algorithms | HIGH   | MEDIUM      | Use proven libraries (ndarray, blas) |
| Performance parity         | HIGH   | MEDIUM      | SIMD, parallelization, profiling     |
| Type system complexity     | MEDIUM | HIGH        | Careful design, extensive testing    |
| NumPy version drift        | MEDIUM | LOW         | Target stable NumPy 1.26+ API        |
| Resource constraints       | HIGH   | MEDIUM      | Phased approach, prioritize          |

---

## Next Steps

1. **Immediate (This Week)**:
   - Fix compilation errors
   - Complete Issue #37 ✅ (Casting policy - DONE)
   - Start Issue #38 (UFunc registry)

2. **Short-term (Next 2 Weeks)**:
   - Complete Issues #38-43 (Ufunc system)
   - Fix test suite compilation
   - Create PARITY.md (Issue #65)

3. **Medium-term (Next Month)**:
   - Complete Phase 1 (Foundation)
   - Start Phase 2 (Linear Algebra)
   - Comprehensive testing

---

**Last Updated:** 2026-01-20
**Maintained By:** @grantjr1842
**Status:** Active Planning Document
# Recent Updates (2026-01-24)

## Completed Functionality

### Masked Arrays (ma) - Issue #342 ✅
- **Status**: Now ~75% complete (was listed as "Missing")
- **Added**: compress, median, var, std, unique functions
- **Test Coverage**: 10 comprehensive tests added
- **Still Missing**: choose function (requires complex integer casting)

### Random Module - Issue #354 ✅
- **Status**: Now ~90% complete (was 60%)
- **Added**: 14 probability distributions
  - geometric, negative_binomial, hypergeometric, logseries
  - rayleigh, wald, weibull, triangular, pareto, zipf
  - standard_cauchy, standard_exponential, standard_gamma
  - shuffle utility function
- **Test Coverage**: 17 comprehensive tests

### Char Module - Issue #353 ✅
- **Status**: Now ~95% complete (was listed as "Missing")
- **Added**: 13 string operations
  - ljust, rjust, swapcase, title, rsplit
  - partition, rpartition, splitlines
  - str_len, equal, not_equal, greater, greater_equal, less, less_equal

### NaN-Aware Statistics - Issue #340 ✅
- **Status**: Verified complete (nancumsum, nancumprod)
- **Test Coverage**: 8 tests added

### Polynomial Classes - Issue #339 ✅
- **Status**: Verified complete
- **Features**: Arithmetic (add, sub, mul), operator overloading, fit, roots, eval
- **Test Coverage**: 14 tests

## Updated Module Status

| Module | Previous | Current | Change |
|--------|----------|---------|--------|
| **ma** | ❌ Missing | ✅ 75% | +75% |
| **random** | ⚠️ 60% | ✅ 90% | +30% |
| **char** | ❌ Missing | ✅ 95% | +95% |
| **polynomial** | ⚠️ 60% | ✅ 85% | +25% |
| **statistics** | ✅ 100% | ✅ 100% | NaN-aware verified |

## Overall Parity Estimate

**Previous Estimate**: ~75%
**Current Estimate**: ~82%

### Breakdown by Category:
- **Core Operations**: 95% ✅
- **Math Functions**: 90% ✅
- **Statistics**: 100% ✅
- **Linear Algebra**: 65% ⚠️
- **FFT**: 50% ⚠️
- **String Operations**: 95% ✅
- **Random**: 90% ✅
- **Masked Arrays**: 75% ✅

## Remaining Critical Gaps

### High Priority
1. **Linear Algebra** (35% remaining)
   - QR decomposition
   - Cholesky decomposition
   - Eigen-decomposition
   - Matrix inversion
   - Linear solvers

2. **FFT** (50% remaining)
   - Multi-dimensional FFTs
   - Real FFTs (rfft, irfft)
   - FFT shifts

### Medium Priority
3. **Execution Engine** (ufunc kernels)
4. **Advanced Broadcasting** edge cases
5. **SIMD optimizations**

## Validation Status

### Test Coverage
- **Unit Tests**: ~170 tests passing
- **Integration Tests**: Comprehensive coverage for core modules
- **Golden Data Tests**: Issue #341 harness implemented

### Known Limitations
1. No cross-language validation with NumPy reference
2. Limited performance benchmarking
3. Edge case testing incomplete for some modules

## Recommendations

1. **Short Term** (Next 1-2 weeks):
   - Complete remaining linear algebra functions
   - Implement multi-dimensional FFTs
   - Add cross-language validation tests

2. **Medium Term** (Next month):
   - Implement execution engine with strided kernels
   - Optimize performance with SIMD
   - Comprehensive edge case testing

3. **Long Term**:
   - 100% API parity with NumPy 2.x
   - Performance parity or better
   - Complete validation suite

