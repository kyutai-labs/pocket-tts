# Imported GitHub Issues

## Issue #2: Running the model in the browser

## Feature: Running the Model in the Browser

### Summary
Enable pocket-tts to run directly in web browsers without requiring a local server installation.

### Motivation
**User Problem:** Users want zero-installation TTS capability directly in their web applications.
**Use Cases:**
- Static websites needing TTS without server infrastructure
- Privacy-conscious applications (no audio sent to servers)
- Offline-capable PWAs
- Embedded TTS in client-side applications

### Current State Analysis

Based on codebase investigation:

1. **Existing Research**: `docs/browser_feasibility.md` documents challenges:
   - Model size: ~600MB (prohibitive for browser download)
   - Streaming state: KV caching doesnt export cleanly to ONNX
   - Audio codec: Mimi uses complex convolutions lacking WASM kernels
   - Memory: Browsers limited to 2-4GB

2. **Rust NumPy Progress**: `rust-numpy/` has WASM feature flag in `Cargo.toml` but not yet implemented

3. **Export Foundation**: `pocket_tts/utils/export_model.py` provides TorchScript export

### Proposed Implementation: Hybrid Architecture with Progressive Enhancement

#### Phase 1: WebSocket Streaming Server (Immediate)
- Create lightweight WebSocket server streaming audio chunks to browser
- JavaScript client with Web Audio API for real-time playback
- Add CLI: `uvx pocket-tts websocket --port 8765`

**Files to create:**
- `pocket_tts/websocket_server.py` - WebSocket server
- `pocket_tts/static/client.html` - Browser client UI
- Modify `pocket_tts/main.py` for CLI command

#### Phase 2: ONNX Export and Runtime Web
- Export Mimi decoder to ONNX (stateless version)
- Export Flow LM in chunks with explicit KV cache I/O
- JavaScript wrapper for ONNX Runtime Web

**Files to create:**
- `pocket_tts/utils/export_onnx.py`
- `pocket_tts/static/js/onnx_session.js`

#### Phase 3: Progressive Web App (PWA)
- Installable PWA with offline support
- Service worker for model caching
- Progressive model loading

#### Phase 4: Full Client-Side WASM (Long-term)
- Complete Rust → WASM pipeline using `rust-numpy`
- WebGL-accelerated matrix operations
- WASM SIMD for performance

### Technical Challenges

| Challenge | Mitigation |
|-----------|------------|
| 600MB model size | Quantization, progressive loading, server-assisted hybrid |
| Streaming state | Explicit KV cache management in ONNX |
| Memory limits | Chunked processing, memory-efficient architecture |
| Missing WASM kernels | Start with server-assisted, build WASM support progressively |

### Breaking Changes
- [ ] No - New feature, additive only

### Files to Modify (Phase 1)
- `pocket_tts/main.py` - Add websocket CLI command
- `pocket_tts/static/` - New directory for web assets

### Acceptance Criteria
- [ ] Phase 1: WebSocket server streams audio to browser client
- [ ] Phase 1: Browser client plays audio in real-time
- [ ] Phase 2: Model exports to ONNX successfully
- [ ] Phase 2: ONNX Runtime Web runs inference
- [ ] Tests added for all components
- [ ] Documentation updated

### Estimated Effort
**Phase 1:** 1 week
**Phase 2:** 2-3 weeks
**Phase 3-4:** 4+ weeks

### References
- Original issue: [kyutai-labs/pocket-tts#1](https://github.com/kyutai-labs/pocket-tts/issues/1)
- Feasibility doc: `docs/browser_feasibility.md`
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/

---

## Issue #9: Feature: Full NumPy API parity in pure Rust

## Current State

- rust-numpy crate exists at `rust-numpy/` with ~450KB of source code
- Core foundation implemented: Array type, dtype system, broadcasting, ufuncs
- **Blocking:** Compilation errors prevent build (see #16)

## Sub-Issues (Implementation Plan)

| Issue | Title | Priority | Status |
|-------|-------|----------|--------|
| #16 | Fix critical compilation errors in linalg.rs | P0 Critical | Open |
| #17 | Implement tensor operations with axes support | P1 High | Open |
| #18 | Implement nuclear and L-p norms | P1 High | Open |
| #19 | Implement sorting with array kth | P1 High | Open |
| #20 | Implement set operations | P1 High | Open |
| #21 | Add parallel_ops fallbacks | P2 Medium | Open |

## Desired State

- A complete Rust-native NumPy parity library with full API coverage, conformance tests, and performance benchmarks.
- **No PyO3 or Python dependencies** - 100% pure Rust.

## Success Criteria

### Automated Verification
- [ ] `cargo check` passes without errors (#16)
- [ ] `cargo test --all-features` passes
- [ ] Conformance suite passes against NumPy test vectors
- [ ] Benchmarks meet parity target on core ufuncs

### Manual Verification
- [ ] API parity checklist complete for all modules
- [ ] All `not_implemented()` stubs resolved (#17-#21)
- [ ] Docs/examples parity with NumPy references

## Notes

- See `rust-numpy/IMPLEMENTATION_SUMMARY.md` for current progress
- See `rust-numpy/DTYPE_MISSING_TYPES.md` for dtype gaps analysis

---

## Issue #17: [rust-numpy] Implement tensor operations with axes support

## Feature: Complete tensor operations with axes support

### Summary
Several tensor operations in `linalg.rs` currently return `not_implemented()` errors when called with axes parameters.

### Affected Functions
| Function | Line | Description |
|----------|------|-------------|
| `tensor_solve` | 515-521 | Full tensor solve with axes |
| `tensor_inv` | 589, 604 | Tensor inverse with axes |
| `tensor_dot` | 409 | Higher-dim (>2D) tensor dot |
| `diagonal_enhanced` | 744 | Custom axes for diagonal |

### Motivation
**User Problem:** NumPy users expect tensor operations to work with arbitrary axis specifications.
**Use Cases:** Scientific computing, deep learning tensor manipulations.

### Proposed Implementation
1. Use reshape + matmul + reshape strategy for tensor_dot
2. Permute axes, reshape to 2D, solve/invert, reshape back for tensor_solve/inv
3. Implement axis permutation for diagonal_enhanced

### Files to Modify
- `rust-numpy/src/linalg.rs`

### Acceptance Criteria
- [ ] `tensor_solve` works with axes parameter
- [ ] `tensor_inv` works with axes parameter
- [ ] `tensor_dot` works with >2D arrays
- [ ] `diagonal_enhanced` works with custom axes
- [ ] All tests pass

---

## Issue #18: [rust-numpy] Implement nuclear and L-p norms in linalg::norm

## Feature: Complete norm implementations

### Summary
The `norm()` function in `linalg.rs` is missing nuclear norm and general L-p norm implementations.

### Current Status
- ✅ Frobenius norm (`"fro"`)
- ✅ Infinity norm (`"inf"`)
- ✅ Negative infinity norm (`"-inf"`)
- ✅ L-1 norm (`"1"`)
- ✅ L-2 norm (`"2"`)
- ❌ Nuclear norm (`"nuc"`) - returns `not_implemented()`
- ❌ L-p norms for p > 2 - returns `not_implemented()`

### Proposed Implementation

**Nuclear norm:**
```rust
Some("nuc") => {
    let svd_result = svd(x, false)?;
    let s = svd_result.1; // singular values
    Ok(s.iter().fold(T::zero(), |acc, v| acc + v.abs()))
}
```

**L-p norms:**
```rust
let p_float = T::from(p).unwrap();
let sum_p: T = x.iter()
    .map(|v| v.abs().powf(p_float))
    .fold(T::zero(), |a, b| a + b);
Ok(sum_p.powf(T::one() / p_float))
```

### Files to Modify
- `rust-numpy/src/linalg.rs` - Lines 203, 231

### Acceptance Criteria
- [ ] Nuclear norm returns correct value
- [ ] L-p norms work for any positive integer p
- [ ] Tests added for both

---

## Issue #19: [rust-numpy] Implement sorting functions: argpartition and partition with array kth

## Feature: Complete sorting functions

### Summary
`argpartition` and `partition` in `sorting.rs` return `not_implemented()` when called with an array-based `kth` parameter.

### Current Status
| Function | Scalar kth | Array kth |
|----------|------------|-----------|
| `argpartition` | ✅ Works | ❌ Returns error |
| `partition` | ✅ Works | ❌ Returns error |

### Proposed Implementation
Iterate over all kth values and use quickselect for each.

### Files to Modify
- `rust-numpy/src/sorting.rs` - Lines 489, 523, 1117, 1166

### Acceptance Criteria
- [ ] `argpartition` works with array kth
- [ ] `partition` works with array kth
- [ ] Tests added

---

## Issue #20: [rust-numpy] Implement set operations in set_ops.rs

## Feature: Complete set operations

### Summary
Several set operations in `set_ops.rs` return `not_implemented()` errors.

### Affected Functions
- Set difference
- Symmetric difference
- Additional set utilities

### Files to Modify
- `rust-numpy/src/set_ops.rs` - Lines 162, 199, 220

### Proposed Implementation
Use HashSet-based algorithms with proper handling for empty arrays.

### Acceptance Criteria
- [ ] All set operations implemented
- [ ] Edge cases for empty arrays handled
- [ ] Tests added

---

## Issue #25: [rust-numpy] Verify and implement bitwise operations

Bitwise universal functions and their corresponding tests are currently missing or unverified.

Next steps:
- Implement bitwise_and, bitwise_or, bitwise_xor, invert, left_shift, right_shift.
- Add comprehensive tests in src/bitwise.rs and tests/conformance/.

---

## Issue #26: [rust-numpy] Complete Dtype system (intp, uintp, IEEE 754 f16)

The current Dtype system in rust-numpy is missing critical support for:
- intp / uintp: Platform-dependent indexing types used throughout NumPy.
- IEEE 754 f16: The current implementation uses simple bit-shifts, which leads to incorrect results for many half-precision operations.

Next steps:
- Use the 'half' crate to implement a proper f16.
- Add Intp and Uintp variants to the Dtype enum.
- Update parsing and itemsize logic in src/dtype.rs.

---

## Issue #30: [ci] Workspace lint/format enforcement (fmt + clippy -D warnings)

Goal: Enforce consistent formatting and linting.

Acceptance Criteria:
- [ ] CI runs: cargo fmt --check
- [ ] CI runs: cargo clippy --workspace -- -D warnings

Depends on: #29 (recommended, not required)


---

## Issue #33: [core] Slicing-as-view (ranges + step; support negative step)

Goal: Implement slicing as views with steps.

Acceptance Criteria:
- [ ] Slice spec supports:
  - full range ':'
  - start..end
  - step (including negative)
- [ ] Tests cover positive and negative strides

Depends on: #31


---

## Issue #34: [core] Broadcast layout (stride=0 where dim=1)

Goal: Add broadcast_layout that produces stride=0 broadcasted views.

Acceptance Criteria:
- [ ] broadcast_layout(layout, out_shape) -> Layout
- [ ] Tests validate stride=0 behavior and errors on incompatible shapes

Depends on: #31


---

## Issue #35: [core] Minimal correct N-D iterator offsets (no coalescing yet)

Goal: Implement a correct baseline N-D iterator/planner.

Acceptance Criteria:
- [ ] Given broadcasted layouts, iterator yields correct per-operand element offsets
- [ ] Tests verify offsets for small shapes against expected sequences

Depends on: #34


---

## Issue #36: [dtype] Numeric promotion rules (explicit table) for Add/Sub/Mul/TrueDiv/Comparison/Bitwise

Goal: Expand dtype promotion into an explicit, auditable rule set.

Acceptance Criteria:
- [ ] promote(left,right,op) supports Bool/Int/UInt/Float/Complex for:
  Add, Sub, Mul, TrueDiv, Comparison, Bitwise
- [ ] Tests include:
  - int + float -> float
  - float + complex -> complex
  - bitwise rejects float/complex with typed error

Notes:
- Use fixed-width dtypes only (no platform int).


---

## Issue #37: [dtype] Casting policy skeleton (Safe/SameKind/Unsafe)

Goal: Implement can_cast(from,to,safety) metadata rules.

Acceptance Criteria:
- [ ] can_cast implements Safe/SameKind/Unsafe categories
- [ ] Tests cover representative pairs

Depends on: #36 (recommended)


---

## Issue #38: [ufunc] UFunc registry + kernel lookup by signature

Goal: Implement registry and kernel selection by dtype signature.

Acceptance Criteria:
- [ ] Registry registers and retrieves UFuncs by name
- [ ] Kernel selection by exact signature works
- [ ] Tests cover registry + selection

Depends on: #36 (promotion integration next)


---

## Issue #39: [ufunc][kernels] Minimal execution engine (contiguous baseline) for binary ufunc

Goal: Execute a selected 1-D kernel over planned runs (start contiguous).

Acceptance Criteria:
- [ ] Given kernel + layouts + buffers, exec succeeds for contiguous case
- [ ] Tests: add_f64 contiguous correctness

Depends on: #35, #38


---

## Issue #40: [ufunc][core][kernels] Broadcast-aware binary exec path (elementwise add)

Goal: Make binary execution broadcast-correct.

Acceptance Criteria:
- [ ] add works for broadcasted shapes (e.g. (3,1)+(1,4)->(3,4))
- [ ] Tests cover mixed broadcasting patterns

Depends on: #34, #35, #39


---

## Issue #41: [api] Public Array facade + add() wired end-to-end

Goal: Expose minimal public API for Array + add.

Acceptance Criteria:
- [ ] api::Array supports basic construction (start with f64)
- [ ] api::ops::add(&Array,&Array)->Array
- [ ] Tests validate public API behavior

Depends on: #40


---

## Issue #42: [kernels][ufunc][api] Add mul ufunc (mirror add) + tests

Goal: Implement multiplication ufunc with the same pathway as add.

Acceptance Criteria:
- [ ] mul works for contiguous and broadcasted inputs
- [ ] Tests mirror add coverage for mul

Depends on: #41


---

## Issue #43: [ufunc][kernels][api] Global sum reduction for f64

Goal: Implement sum reduction over all axes.

Acceptance Criteria:
- [ ] sum(Array)->scalar (or 0-D Array) for f64
- [ ] Tests define and enforce empty-array policy

Depends on: #41


---

## Issue #44: [ufunc][api] sum(axis=..., keepdims=...): single axis first

Goal: Implement sum over a single axis with keepdims.

Acceptance Criteria:
- [ ] sum(axis=i, keepdims=bool) correct for small shapes
- [ ] Tests validate resulting shape + values

Depends on: #43


---

## Issue #45: [performance][core][ufunc] Dimension coalescing into fewer contiguous runs

Goal: Optimize iteration by coalescing dimensions into fewer 1-D kernel calls.

Acceptance Criteria:
- [ ] Outputs identical to baseline across existing test suite
- [ ] Basic perf sanity check (bench optional)

Depends on: #40, #44


---

## Issue #46: [performance][kernels] SIMD kernels (feature-gated) + runtime dispatch

Goal: Add optional SIMD-specialized kernels with runtime dispatch.

Acceptance Criteria:
- [ ] Baseline path remains default and passes all tests
- [ ] SIMD feature passes identical tests when enabled
- [ ] Dispatch chooses best available implementation safely

Depends on: #45


---

## Issue #47: [performance] Threading policy for safe kernels (no overlap/alias hazards)

Goal: Parallelize only where safe and deterministic under defined rules.

Acceptance Criteria:
- [ ] Threading is conditional and respects aliasing constraints
- [ ] Tests confirm correctness; determinism where required

Depends on: #45


---

## Issue #48: [rust-numpy] Phase 1: FFT Module Implementation

Complete all 6 stubbed FFT functions using rustfft crate. See `thoughts/plans/fft-module-implementation.md` for details.
Deliverables:
- fft(), ifft()
- rfft2(), irfft2()
- rfftn(), irfftn()
- hilbert()

---

## Issue #49: [rust-numpy] Phase 2: Advanced Array Methods

Implement advanced array manipulation methods. See `thoughts/plans/advanced-array-methods-and-nan-stats.md` for details.
Deliverables:
- flatten(), ravel(), squeeze()
- repeat(), tile()
- swapaxes(), rollaxis(), moveaxis()
- atleast_1d/2d/3d()

---

## Issue #50: [rust-numpy] Phase 3: NaN-Aware Statistics

Implement statistical functions that properly handle NaN values. See `thoughts/plans/advanced-array-methods-and-nan-stats.md` for details.
Deliverables:
- nanmean(), nanstd(), nanvar()
- nanmedian(), nansum(), nanprod()

---

## Issue #51: [rust-numpy] Phase 4: Advanced Indexing & Slicing

Implement fancy indexing, boolean masking, and ellipsis indexing. See `thoughts/plans/advanced-indexing-and-additional-gaps.md` for details.
Deliverables:
- Fancy indexing, Boolean masking
- Ellipsis indexing
- Field access, NewAxis support

---

## Issue #52: [rust-numpy] Phase 5: Missing Utility Ufuncs

Add utility functions for NaN/Inf/Finite checking and angle conversions. See `thoughts/plans/advanced-indexing-and-additional-gaps.md` for details.
Deliverables:
- isnan(), isinf(), isfinite()
- deg2rad(), rad2deg()
- Array versions of above

---

## Issue #53: [rust-numpy] Phase 6: Polynomial Module Comparison

Audit and complete polynomial module. See `thoughts/plans/advanced-indexing-and-additional-gaps.md` for details.
Deliverables:
- Audit existing functions
- Implement missing: roots(), poly(), polyval(), polyfit()
- Comprehensive tests

---

## Issue #54: [rust-numpy] Phase 7: Test Coverage Expansion

Expand test coverage to include NumPy conformance testing. See `thoughts/plans/rust-numpy-port-master-execution-plan.md` for details.
Deliverables:
- Conformance tests against NumPy
- Edge case testing
- Performance benchmarks

---

## Issue #55: [rust-numpy] C-ABI Integration & Python Shim

Integrate Rust library via C-ABI and rewrite Python shim. See `thoughts/plans/rust_numpy_parity_gaps.md` for details.
Deliverables:
- Export all required functions via C-ABI
- Rewrite numpy_rs.py to use ctypes
- Remove PyO3 dependencies

---

## Issue #56: [rust-numpy] Implement Eigen-decomposition

### A. Technical Specification

**What this feature does:**

- Computes eigenvalues and eigenvectors of square matrices
- Supports both real and complex matrices
- Returns eigenvectors as columns of unitary matrix

**Technical requirements:**

- Use Francis double-shift QR iteration algorithm (industry standard)
- Reduce matrix to upper Hessenberg form first
- Handle defective matrices (repeated eigenvalues)
- Sort eigenvalues and eigenvectors consistently
- Return complex eigenvalues even for real matrices

**NumPy API requirements:**

```python
np.linalg.eig(a)  # Returns (eigenvalues, eigenvectors)
```

- Input: Square matrix (n×n)
- Output: Tuple of (eigenvalues array, eigenvectors matrix)
- Eigenvectors as columns: `eigenvectors[:, i]` is eigenvector for `eigenvalues[i]`
- Eigenvalue ordering: Unspecified (typically not sorted)

**Special cases to handle:**

- Non-square matrices (error)
- Singular matrices (ill-conditioned, may produce large errors)
- Defective matrices (incomplete eigenvector basis)
- Empty matrices
- 1×1 matrices

**Algorithms:**

1. Francis double-shift QR iteration with Hessenberg reduction
2. For real matrices: Complex eigenvalue detection
3. Wilkinson shift for convergence

### B. Code Structure

**Files to create/modify:**

- `src/linalg/eigen.rs` - Create eigen-decomposition module (extend existing)
- `src/linalg/mod.rs` - Export eig() function
- `tests/eigen_tests.rs` - Add eigen tests

**Function signatures:**

```rust
pub fn eig<T>(
    a: &Array<T>,
) -> Result<(Array<Complex64>, Array<Complex64>), NumPyError>
where T: LinalgScalar;

pub fn hessenberg_reduction<T>(
    a: &Array<T>,
) -> Array<T>
where T: LinalgScalar;

pub fn francis_double_shift<T>(
    h: &Array<T>,
    max_iterations: usize,
    eps: f64,
) -> (Array<Complex64>, Array<Complex64>)
where T: LinalgScalar;
```

**Data structures to define:**

```rust
// Hessenberg matrix representation
struct HessenbergMatrix<T> {
    data: Array<T>,
    is_upper: bool,
}

// Francis QR iteration state
struct FrancisState {
    iteration: usize,
    converged: bool,
}
```

**Integration points:**

- QR decomposition - Use within Francis iteration
- Complex arithmetic - Use for eigenvalue computation
- Existing `linalg/mod.rs` - Export new eig() function

### C. Implementation Steps

**Phase 1: Foundation (8-10 hours)**

1. [ ] Review existing `src/linalg/eigen.rs` (stub only)
2. [ ] Study Francis double-shift QR algorithm (reference papers)
3. [ ] Implement Wilkinson shift computation
4. [ ] Implement Hessenberg reduction function
5. [ ] Add data structures: `HessenbergMatrix`, `FrancisState`

**Phase 2: Core Algorithm (15-20 hours)** 6. [ ] Implement Francis QR iteration: - Initialize with Hessenberg matrix - Iteratively apply QR steps with Wilkinson shift - Detect convergence (subdiagonal elements below threshold) - Extract 2×2 submatrices for eigenvalues 7. [ ] Handle complex eigenvalues for real matrices: - Detect 2×2 blocks with complex eigenvalues - Convert to complex form 8. [ ] Implement eigenvector back-transformation: - Track accumulated Q matrices from QR steps - Apply to identity to get eigenvectors 9. [ ] Add max iteration limit (e.g., 1000) 10. [ ] Add convergence detection (subdiagonal < eps)

**Phase 3: Integration & Testing (7-10 hours)** 11. [ ] Create `eig()` public function with proper error handling 12. [ ] Validate input is square matrix 13. [ ] Handle edge cases: 1×1, singular, empty 14. [ ] Write unit tests with known examples 15. [ ] Benchmark vs NumPy for correctness and performance 16. [ ] Add documentation with examples 17. [ ] Export from linalg module

**Phase 4: Optimization (Optional, 8-12 hours)** 18. [ ] Optimize Hessenberg reduction (avoid unnecessary computations) 19. [ ] Cache QR results where possible 20. [ ] Add early termination for small matrices (n < 10)

### D. Testing Requirements

**Unit tests in `tests/eigen_tests.rs`:**

```rust
#[test]
fn test_eig_2x2_real()
#[test]
fn test_eig_3x3_real()
#[test]
fn test_eig_complex_matrix()
#[test]
fn test_eig_symmetric()
#[test]
fn test_eig_defective_matrix()
#[test]
fn test_eig_1x1()
#[test]
fn test_non_square_error()
#[test]
fn test_eigenvalue_accuracy()
#[test]
fn test_eigenvector_orthogonality()
#[test]
fn test_convergence_large_matrix()
```

**Integration tests:**

- Test eig() @ eigenvectors ≈ identity (within tolerance)
- Test with real matrices that have complex eigenvalues
- Verify NumPy conformance

**Performance benchmarks:**

- Benchmark eigen-decomposition vs NumPy
- Test scaling with matrix size (10, 50, 100, 500)

**Edge case tests:**

- Singular matrices (near-singular, very ill-conditioned)
- Repeated eigenvalues
- Complex input matrices

### E. Success Criteria

- [ ] All eigen tests pass
- [ ] Eigenvalues match NumPy (within 1e-6 tolerance for well-conditioned)
- [ ] Eigenvectors are orthogonal (for real symmetric matrices)
- [ ] Performance within 3x of NumPy for typical sizes
- [ ] Handles complex eigenvalues correctly
- [ ] No unsafe code
- [ ] Documentation complete with examples
- [ ] Exported in linalg module


---

## Issue #57: [rust-numpy] Implement QR Decomposition

### A. Technical Specification

**What this feature does:**

- Decomposes matrix A into Q (orthonormal/unitary) and R (upper triangular)
- Supports 4 NumPy modes: 'reduced', 'complete', 'r', 'raw'
- Handles rectangular matrices (both m>n and m<n)

**Technical requirements:**

- Use Householder reflections algorithm (standard method)
- Orthogonal/unitary Q matrix via Householder vectors
- Upper triangular R matrix via Gaussian elimination
- Handle complex matrices (unitary Q)
- Return Q and R in correct shapes based on mode

**NumPy API requirements:**

```python
np.linalg.qr(a, mode='reduced')
```

**Modes:**

- `mode='reduced'` (default): Q is m×k, R is k×n where k=min(m,n)
- `mode='complete'`: Q is m×m square, R is m×n (zero-padded)
- `mode='r'`: Returns only R (k×n)
- `mode='raw'`: Returns (h, tau) where h contains Householder vectors, tau contains scaling factors

**Special cases to handle:**

- Non-2D arrays (error)
- Empty arrays
- 1×N matrices (Q is 1×1, R is 1×N)
- Complex matrices

**Algorithm:**

1. Householder reflections for QR factorization
2. Accumulate Q matrix or track Householder vectors
3. Upper triangular R via successive elimination

### B. Code Structure

**Files to create/modify:**

- `src/linalg/decompositions.rs` - Extend existing with qr()
- `src/linalg/mod.rs` - Export qr() function
- `tests/qr_tests.rs` - Add QR tests

**Function signatures:**

```rust
#[derive(Debug, Clone, Copy)]
pub enum QrMode {
    Reduced,
    Complete,
    R,
    Raw,
}

pub fn qr<T>(
    a: &Array<T>,
    mode: QrMode,
) -> Result<QrResult<T>, NumPyError>
where T: LinalgScalar;

pub struct QrResult<T> {
    q: Option<Array<T>>,
    r: Array<T>,
    h: Option<Array<T>>,      // For raw mode
    tau: Option<Array<T>>,     // For raw mode
}

// Householder reflection
fn householder_reflection<T>(
    x: &[T],
) -> (Vec<T>, T)
where T: LinalgScalar;
```

**Data structures to define:**

```rust
pub enum QrResult<T> {
    Reduced(Array<T>, Array<T>),
    Complete(Array<T>, Array<T>),
    R(Array<T>),
    Raw(Array<T>, Array<T>),
}
```

**Integration points:**

- Matrix multiplication - Use in Q accumulation
- Existing decompositions module - Extend with qr()
- Array creation functions - Use for Q, R matrices

### C. Implementation Steps

**Phase 1: Householder Reflections (6-8 hours)**

1. [ ] Review existing `src/linalg/decompositions.rs`
2. [ ] Implement `householder_reflection()`:
   - Compute Householder vector v = x - 2\*(x·u)u where u = x/||x||
   - Return (v, tau) where tau = 2/(v·v)
3. [ ] Implement `apply_householder()` to apply reflection to matrix

**Phase 2: QR Factorization (10-12 hours)** 4. [ ] Implement `qr_reduced()`: - Apply Householder reflections column by column - Stop after k=min(m,n) reflections - Accumulate Q matrix (m×k) - Compute R (k×n upper triangular) 5. [ ] Implement `qr_complete()`: - Continue reflections for all m columns - Q is m×m unitary/square - R is m×n with zeros below diagonal 6. [ ] Implement `qr_r_only()`: - Compute only R matrix (k×n) - Skip Q accumulation (performance) 7. [ ] Implement `qr_raw()`: - Return h (Householder vectors) and tau (scaling factors) - Q can be reconstructed by caller if needed

**Phase 3: Integration & Testing (4-5 hours)** 8. [ ] Create public `qr()` function with mode parameter 9. [ ] Validate input is 2D matrix 10. [ ] Handle m>n and m<n cases 11. [ ] Add unit tests for all modes 12. [ ] Verify NumPy conformance 13. [ ] Document QR modes with examples

### D. Testing Requirements

**Unit tests in `tests/qr_tests.rs`:**

```rust
#[test]
fn test_qr_reduced_square()
#[test]
fn test_qr_reduced_tall()
#[test]
fn test_qr_reduced_wide()
#[test]
fn test_qr_complete()
#[test]
fn test_qr_r_only()
#[test]
fn test_qr_raw()
#[test]
fn test_qr_complex()
#[test]
fn test_qr_orthogonality()
#[test]
fn test_qr_accuracy_a_approx_qr()
```

**Integration tests:**

- Verify Q is orthonormal (Q^T @ Q ≈ I)
- Verify R is upper triangular
- Test A ≈ Q @ R (within tolerance)
- Compare outputs with NumPy.linalg.qr

**Performance benchmarks:**

- Benchmark QR vs NumPy for various sizes (10, 50, 100, 500)

### E. Success Criteria

- [ ] All QR tests pass
- [ ] Q is orthonormal/unitary (within tolerance)
- [ ] R is upper triangular
- [ ] All 4 modes implemented correctly
- [ ] Performance within 2x of NumPy
- [ ] NumPy conformance tests pass
- [ ] No unsafe code
- [ ] Documentation complete with mode examples


---

## Issue #58: [rust-numpy] Implement SVD (Singular Value Decomposition)

### A. Technical Specification

**What this feature does:**

- Decomposes matrix A into U @ Σ @ V^T where U and V are unitary, Σ is diagonal
- Computes singular values (always non-negative, sorted descending)
- Supports full_matrices and compute_uv parameters

**Technical requirements:**

- Use Golub-Kahan bidiagonalization algorithm
- Implicit QR algorithm for bidiagonal SVD
- Singular values ALWAYS returned as f64 (even for complex input)
- Handle rank-deficient matrices
- Support complex matrices (unitary transformations)

**NumPy API requirements:**

```python
np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)
```

- Input: 2D matrix (m×n)
- Output: (u, s, vh) where:
  - u: (m, m) unitary matrix
  - s: (k,) array of singular values where k=min(m,n)
  - vh: (n, n) unitary matrix (V^T)

**Parameters:**

- `full_matrices=True`: u is (m,m), vh is (n,n)
- `full_matrices=False`: u is (m,k), vh is (k,n) where k=min(m,n)
- `compute_uv=False`: Returns only s
- `hermitian=True`: Optimized for Hermitian matrices (not in initial scope)

**Special cases to handle:**

- Non-2D arrays (error)
- Empty arrays
- Rank-deficient matrices (zero singular values)
- Complex matrices

**Algorithm:**

1. Golub-Kahan bidiagonalization (A = U @ B @ V^T where B is bidiagonal)
2. Implicit QR for bidiagonal SVD (Wilkinson shift)
3. Divide and conquer for large matrices (optional optimization)

### B. Code Structure

**Files to create/modify:**

- `src/linalg/decompositions.rs` - Extend existing with svd()
- `src/linalg/mod.rs` - Export svd() function
- `tests/svd_tests.rs` - Add SVD tests

**Function signatures:**

```rust
pub fn svd<T>(
    a: &Array<T>,
    full_matrices: bool,
    compute_uv: bool,
) -> Result<SvdResult, NumPyError>
where T: LinalgScalar;

pub struct SvdResult {
    u: Option<Array<Complex64>>,
    s: Array<f64>,  // Always f64, even for complex input
    vh: Option<Array<Complex64>>,
}

// Bidiagonalization
pub fn golub_kahan_bidiagonal<T>(
    a: &Array<T>,
) -> (Array<T>, Array<T>, Array<T>)  // (U, B, V)
where T: LinalgScalar;

// Bidiagonal SVD (implicit QR)
pub fn bidiagonal_svd<T>(
    b: &Array<T>,
) -> (Array<T>, Array<f64>, Array<T>)  // (U, s, V)
where T: LinalgScalar;
```

**Data structures to define:**

```rust
// Bidiagonal matrix representation
struct BidiagonalMatrix<T> {
    upper: Vec<T>,  // Superdiagonal
    main: Vec<T>,   // Diagonal
    lower: Vec<T>,  // Subdiagonal
}

// Wilkinson shift state
struct WilkinsonShift {
    shift: f64,
    converged: bool,
}
```

**Integration points:**

- QR decomposition - Use in bidiagonalization
- Matrix multiplication - Use for verification
- Existing decompositions module - Extend with svd()

### C. Implementation Steps

**Phase 1: Bidiagonalization (10-12 hours)**

1. [ ] Review existing `src/linalg/decompositions.rs`
2. [ ] Implement Householder reflections for bidiagonalization:
   - Left Householder (zero above diagonal in columns)
   - Right Householder (zero below diagonal in rows)
3. [ ] Implement `golub_kahan_bidiagonal()`:
   - Apply left reflections to zero above diagonal
   - Apply right reflections to zero below diagonal
   - Return U (orthonormal), B (bidiagonal), V (orthonormal)

**Phase 2: Bidiagonal SVD (12-15 hours)** 4. [ ] Implement `bidiagonal_svd()`: - Initialize with bidiagonal matrix - Apply Wilkinson shift for 2×2 submatrix at bottom - Perform Givens rotations to eliminate subdiagonal - Iterate until subdiagonal < eps (converged) 5. [ ] Extract singular values from diagonal 6. [ ] Back-transform singular vectors: - Accumulate left Givens rotations into U - Accumulate right Givens rotations into V 7. [ ] Handle complex singular values

**Phase 3: Integration & Testing (8-10 hours)** 8. [ ] Create public `svd()` function with parameters 9. [ ] Implement full_matrices flag: - True: Return full U (m×m) and full V (n×n) - False: Return reduced U (m×k) and V (k×n) 10. [ ] Implement compute_uv flag: - True: Return (u, s, vh) - False: Return only s 11. [ ] Validate singular values are f64 (convert if needed) 12. [ ] Handle rank-deficient matrices (zero singular values) 13. [ ] Add unit tests with known matrices 14. [ ] Verify NumPy conformance (compare U, s, V) 15. [ ] Benchmark vs NumPy 16. [ ] Document API with examples

### D. Testing Requirements

**Unit tests in `tests/svd_tests.rs`:**

```rust
#[test]
fn test_svd_square()
#[test]
fn test_svd_tall()
#[test]
fn test_svd_wide()
#[test]
fn test_svd_rank_deficient()
#[test]
fn test_svd_singular_values_sorted()
#[test]
fn test_svd_complex()
#[test]
fn test_svd_compute_uv_false()
#[test]
fn test_svd_full_matrices()
#[test]
fn test_svd_accuracy_a_approx_usv()
#[test]
fn test_svd_unitarity()
```

**Integration tests:**

- Verify U is unitary (U @ U^H ≈ I)
- Verify V is unitary (V @ V^H ≈ I)
- Test A ≈ U @ diag(s) @ V^H (within tolerance)
- Compare with NumPy.linalg.svd

**Performance benchmarks:**

- Benchmark SVD vs NumPy for various sizes (10, 50, 100, 500)

### E. Success Criteria

- [ ] All SVD tests pass
- [ ] Singular values are non-negative f64, sorted descending
- [ ] U and V are unitary (within tolerance)
- [ ] A ≈ U @ diag(s) @ V^T (within tolerance)
- [ ] Handles rank-deficient matrices correctly
- [ ] NumPy conformance tests pass
- [ ] Performance within 3x of NumPy
- [ ] No unsafe code
- [ ] Documentation complete with examples


---

## Issue #59: [rust-numpy] Implement Multi-dimensional Dot Products

### A. Technical Specification

**What this feature does:**

- Implements dot(), matmul(), and tensordot() for N-D arrays
- Supports broadcasting between multi-dimensional arrays
- Optimizes for common cases (2D×2D matrix multiplication)

**Technical requirements:**

- Support N-D arrays with broadcasting
- Implement different dimension combinations:
  - 1D × 1D → inner product (scalar)
  - 1D × 2D → matrix-vector multiplication
  - 2D × 1D → vector-matrix multiplication
  - 2D × 2D → matrix multiplication
  - N-D × N-D → tensor contraction
- Proper broadcasting rules (NumPy semantics)
- Clear error messages for shape mismatches

**NumPy API requirements:**

```python
np.dot(a, b)
np.matmul(a, b)  # Python @ operator
np.tensordot(a, b, axes=2)
```

**Shape rules:**

- `dot(a, b)`: Last dim of a must match second-to-last dim of b
- `matmul(a, b)`: (..., n, k) @ (..., k, m) → (..., n, m)

**Special cases to handle:**

- Scalar × Array or Array × Scalar
- Shape mismatches
- Empty arrays
- Large dimensions (>10)

### B. Code Structure

**Files to create/modify:**

- `src/linalg/products.rs` - Extend existing with N-D support
- `src/linalg/mod.rs` - Export enhanced dot(), matmul(), tensordot()
- `tests/tensor_dot_tests.rs` - Add tensor multiplication tests

**Function signatures:**

```rust
pub fn dot<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

pub fn matmul<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

pub fn tensordot<T>(
    a: &Array<T>,
    b: &Array<T>,
    axes: usize,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

// Optimized 2D×2D case
pub fn matrix_multiply_2d<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;
```

**Data structures to define:**

```rust
// Tensor contraction specification
struct TensorContraction {
    a_axes: Vec<usize>,
    b_axes: Vec<usize>,
    output_shape: Vec<usize>,
}
```

**Integration points:**

- Broadcasting module - Use for shape validation and broadcasting
- Existing dot() - Extend for N-D support
- Array operations - Use for efficient computation

### C. Implementation Steps

**Phase 1: Foundation (6-8 hours)**

1. [ ] Review existing `src/linalg/products.rs` (only 2D supported)
2. [ ] Implement shape validation for N-D arrays
3. [ ] Implement broadcasting compatibility check
4. [ ] Create error messages for shape mismatches

**Phase 2: Core Operations (8-10 hours)** 5. [ ] Implement `dot()` for 1D×1D: - Compute inner product - Return scalar (0-D array) 6. [ ] Implement `dot()` for 1D×2D and 2D×1D: - Validate shapes - Compute matrix-vector product 7. [ ] Implement `dot()` for 2D×2D: - Validate (m, k) @ (k, n) → (m, n) - Use optimized loop or call `matrix_multiply_2d()` 8. [ ] Implement `dot()` for N-D×N-D: - Identify contraction dimensions (last dim of a, second-to-last of b) - Contract along specified axes - Compute output shape

**Phase 3: Additional Functions (4-5 hours)** 9. [ ] Implement `matmul()` with same logic as dot() 10. [ ] Implement `tensordot()`: - Generalized tensor contraction - Support arbitrary axes parameter - Contract along specified dimensions

**Phase 4: Testing & Optimization (3-4 hours)** 11. [ ] Add unit tests for all dimension combinations 12. [ ] Add SIMD optimization for 2D×2D (if beneficial) 13. [ ] Benchmark vs NumPy for various sizes 14. [ ] Verify NumPy conformance 15. [ ] Document functions with examples

### D. Testing Requirements

**Unit tests in `tests/tensor_dot_tests.rs`:**

```rust
#[test]
fn test_dot_1d_1d()
#[test]
fn test_dot_1d_2d()
#[test]
fn test_dot_2d_1d()
#[test]
fn test_dot_2d_2d()
#[test]
fn test_dot_nd_nd()
#[test]
fn test_matmul()
#[test]
fn test_tensordot()
#[test]
fn test_shape_error()
#[test]
fn test_scalar_multiplication()
#[test]
fn test_empty_arrays()
#[test]
fn test_large_dimensions()
```

**Integration tests:**

- Test with NumPy.dot() for various shapes
- Verify broadcasting rules match NumPy
- Test numerical accuracy (within tolerance)

**Performance benchmarks:**

- Benchmark dot() vs NumPy for various sizes (100, 1000, 10000)
- Test SIMD optimization impact

### E. Success Criteria

- [ ] All tensor dot tests pass
- [ ] All dimension combinations work correctly
- [ ] Broadcasting matches NumPy exactly
- [ ] Shape mismatches produce clear errors
- [ ] Performance within 2x of NumPy for 2D×2D
- [ ] NumPy conformance tests pass
- [ ] No unsafe code
- [ ] Documentation complete with examples


---

## Issue #60: [rust-numpy] Implement Least Squares Solver

### A. Technical Specification

**What this feature does:**

- Solves linear least squares problem: minimize ||Ax - b||^2
- Returns x, residuals, rank, singular values
- Handles both full-rank and rank-deficient matrices

**Technical requirements:**

- QR method for full-rank matrices (faster)
- SVD method for rank-deficient matrices (Moore-Penrose pseudo-inverse)
- Support multiple right-hand sides (2D b matrix)
- Compute residuals (sum of squared errors)
- Determine matrix rank from singular values

**NumPy API requirements:**

```python
x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
```

- Input: a (m×n matrix), b (m,) or (m, k) array
- Output: Tuple of (x, residuals, rank, s) where:
  - x: (n,) or (n, k) solution
  - residuals: (k,) or () sum of squared errors
  - rank: integer matrix rank
  - s: (min(m,n),) singular values of A

**Parameters:**

- `rcond`: Cutoff for singular values (values < rcond \* s_max are treated as zero)
- Default: max(m,n) \* eps where eps is machine precision

**Special cases to handle:**

- Non-2D arrays (error)
- Empty arrays
- Over-determined (m > n)
- Under-determined (m < n)
- Rank-deficient matrices
- Multiple right-hand sides (2D b)

**Algorithms:**

1. QR-based: Full-rank case, faster, x = R^{-1} @ Q^T @ b
2. SVD-based: Rank-deficient case, x = V @ Σ^+ @ U^T @ b where Σ^+ is pseudo-inverse

### B. Code Structure

**Files to create/modify:**

- `src/linalg/solvers.rs` - Extend existing with lstsq()
- `src/linalg/mod.rs` - Export lstsq() function
- `tests/lstsq_tests.rs` - Add least squares tests

**Function signatures:**

```rust
pub fn lstsq<T>(
    a: &Array<T>,
    b: &Array<T>,
    rcond: Option<f64>,
) -> Result<LstsqResult<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

pub struct LstsqResult<T> {
    x: Array<T>,
    residuals: Array<f64>,
    rank: usize,
    s: Array<f64>,
}

// QR-based least squares
pub fn lstsq_qr<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

// SVD-based least squares (pseudo-inverse)
pub fn lstsq_svd<T>(
    a: &Array<T>,
    b: &Array<T>,
    rcond: f64,
) -> Result<LstsqResult<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;
```

**Data structures to define:**

```rust
// SVD pseudo-inverse helper
struct PseudoInverse {
    v: Array<Complex64>,
    sigma_plus: Array<f64>,
    u_h: Array<Complex64>,
}
```

**Integration points:**

- QR decomposition - Use in QR-based solving
- SVD decomposition - Use in SVD-based solving
- Matrix multiplication - Use in both methods
- Existing solvers module - Extend with lstsq()

### C. Implementation Steps

**Phase 1: QR-based Solver (8-10 hours)**

1. [ ] Review existing `src/linalg/solvers.rs`
2. [ ] Implement `lstsq_qr()`:
   - Compute QR: A = Q @ R
   - Solve R @ x = Q^T @ b (triangular solve)
   - Return x (n,) or (n, k)
3. [ ] Compute residuals:
   - Residuals = ||Ax - b||^2 for each column of b
   - Sum to get total residual
4. [ ] Handle multiple right-hand sides (2D b)

**Phase 2: SVD-based Solver (10-12 hours)** 5. [ ] Compute SVD: A = U @ Σ @ V^T 6. [ ] Implement `lstsq_svd()`: - Compute Σ^+ (pseudo-inverse of Σ) - Set 1/s for s_i > rcond _ s_max, 0 otherwise - Compute x = V @ Σ^+ @ U^T @ b 7. [ ] Compute rank: - Count singular values > rcond _ s_max - Return integer rank 8. [ ] Compute residuals (same as QR method)

**Phase 3: Integration & Testing (4-6 hours)** 9. [ ] Create public `lstsq()` function: - Check condition number of A - Use QR if well-conditioned (fast path) - Use SVD if ill-conditioned or rank-deficient - Default rcond = max(m,n) \* eps 10. [ ] Return (x, residuals, rank, s) tuple 11. [ ] Handle 1D b and 2D b correctly 12. [ ] Add unit tests: - Full-rank cases - Rank-deficient cases - Multiple right-hand sides 13. [ ] Verify NumPy conformance (compare x, residuals, rank, s) 14. [ ] Benchmark vs NumPy 15. [ ] Document API with examples

### D. Testing Requirements

**Unit tests in `tests/lstsq_tests.rs`:**

```rust
#[test]
fn test_lstsq_overdetermined()
#[test]
fn test_lstsq_underdetermined()
#[test]
fn test_lstsq_rank_deficient()
#[test]
fn test_lstsq_multiple_rhs()
#[test]
fn test_lstsq_rcond()
#[test]
fn test_lstsq_residuals()
#[test]
fn test_lstsq_rank()
#[test]
fn test_lstsq_accuracy()
#[test]
fn test_lstsq_qr_vs_svd()
```

**Integration tests:**

- Verify Ax approximates b (within tolerance)
- Test with NumPy.linalg.lstsq for various problems
- Test both QR and SVD paths

**Performance benchmarks:**

- Benchmark lstsq vs NumPy for various sizes (10, 50, 100, 500)

### E. Success Criteria

- [ ] All least squares tests pass
- [ ] QR method works for full-rank matrices
- [ ] SVD method handles rank-deficient correctly
- [ ] Residuals computed correctly
- [ ] Rank matches expected
- [ ] Both 1D and 2D b supported
- [ ] NumPy conformance tests pass
- [ ] Performance within 2x of NumPy
- [ ] No unsafe code
- [ ] Documentation complete with examples


---

## Issue #61: [rust-numpy] Complete Unique Implementation

### A. Technical Specification

**What this feature does:**

- Finds unique elements in array with optional additional outputs
- Returns sorted unique elements (already done)
- Adds return_index, return_inverse, return_counts, axis parameters

**Technical requirements:**

- Return sorted unique elements
- Add return_index parameter (indices of first occurrences)
- Add return_inverse parameter (indices to reconstruct original)
- Add return_counts parameter (counts of each unique element)
- Add axis parameter support (find unique along axis)

**NumPy API requirements:**

```python
np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)
```

### B. Code Structure

**Files to modify:**

- `src/set_ops.rs` - Complete implementation
- `tests/set_ops_tests.rs` - Add tests

### C. Implementation Steps

1. [ ] Review current unique() implementation
2. [ ] Implement return_index parameter
3. [ ] Implement return_inverse parameter
4. [ ] Implement return_counts parameter
5. [ ] Implement axis parameter support
6. [ ] Add comprehensive tests

### D. Testing Requirements

Test all parameter combinations, axis handling, NaN values, edge cases.

### E. Success Criteria

- [ ] All tests pass
- [ ] 100% NumPy parity
- [ ] Performance acceptable


---

## Issue #62: Identify Gaps

**Task ID:** task-1
**Prompt:** Identify all remaining todos required for a 100% port to Rust with 100% parity.

---

