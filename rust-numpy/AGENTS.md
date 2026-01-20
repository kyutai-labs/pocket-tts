# AGENTS.md

This file provides guidance to AI agents when working with the rust-numpy codebase.

## OVERVIEW

Pure Rust NumPy reimplementation achieving 100% API parity through handcrafted safe Rust implementations.

## ARCHITECTURE

**Core Array System**: Multi-dimensional arrays with NumPy-compliant memory model, stride calculations, and broadcasting logic.

**Ufunc System**: Universal functions framework with broadcasting via `UfuncEngine` trait for element-wise operations across arrays.

**Dtype System**: Complete NumPy dtype compatibility (15 dtypes including Complex32-256, Struct, Datetime64/Timedelta64 with all units).

**Memory Management**: Copy-on-write via Arc with SIMD-aligned allocations, zero-copy views where possible.

**Backend-Agnostic**: Pluggable backends (CPU, CUDA, Metal, WASM) - pure Rust library with no PyO3 bindings.

## CODING STANDARDS

**Pure safe Rust only**: Zero unsafe code. Use only essential crates (num-complex, num-traits, rand, chrono, rustfft, faer). All handcrafted implementations must match NumPy functionality 100%.

**NumPy parity**: Function signatures, parameters, defaults, return types, and error messages must match NumPy exactly. Test edge cases against NumPy reference implementations.

**No Python bindings**: This is a pure Rust library with no PyO3/C-API compatibility layer. Designed for Rust-first scientific computing, not Python interop.

**Consistent API patterns**: Follow NumPy conventions exactly (e.g., `arange<T>(start: T, stop: Option<T>, step: Option<T>, dtype: Option<Dtype>) -> Result<Array<T>>`).

**Testing requirements**: All functions require comprehensive unit tests. Conformance tests must verify exact NumPy behavior including edge cases and error conditions.

## ERROR HANDLING

**Result<T> for all operations**: Use `Result<T>` for fallible operations with `NumPyError` enum mapping to NumPy exceptions (ValueError, TypeError, IndexError, etc.).

**Error messages**: Match NumPy exception messages exactly for compatibility. Use descriptive messages that help users identify the issue.

**Validation first**: Validate all inputs before processing (shape compatibility, dtype casting, axis bounds, parameter ranges). Return early errors rather than panicking.

**Edge case handling**: Handle all NumPy edge cases (empty arrays, NaN/Inf propagation, broadcasting failures, dimension mismatches, singular matrices).
