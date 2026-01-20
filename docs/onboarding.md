# Developer Onboarding Guide - Rust-NumPy

Welcome to the `rust-numpy` project! This library aims to be a 100% pure-Rust port of NumPy, providing API parity and high performance.

## Prerequisites

- **Rust**: Latest stable toolchain (`rustup update stable`).
- **Cargo**: Included with Rust.

## Getting Started

### 1. Build the Project

Navigate to the `rust-numpy` directory and build:

```bash
cd rust-numpy
cargo build
```

### 2. Run Tests

We use a combination of unit tests, conformance tests, and integration tests.

```bash
# Run all tests
cargo test

# Run specifically conformance tests (parity checks)
cargo test --test conformance_tests
```

## Key Concepts

### 1. Array Structure

The core data structure is `Array<T>` (`src/array.rs`). It wraps a `Vec<T>` with shape and stride info.

- It is **generic** over type `T`.
- It supports **views** (slicing) without copying data (using `Arc` and offsets).

### 2. Universal Functions (Ufuncs)

Most operations (add, sin, sum) are "ufuncs".

- **Registration**: Ufuncs are registered in a global `UFUNC_REGISTRY` (`src/ufunc.rs`).
- **Execution**: They operate on generic `ArrayView` traits.
- **Dispatch**: The registry looks up the correct implementation based on input dtypes.

### 3. Broadcasting

We support full NumPy-style broadcasting (`src/broadcasting.rs`).

- Scalars are broadcast to array shapes.
- Dimensions of size 1 are stretched to match others.
- **Note**: Always use `broadcast_arrays` before manually iterating over multiple arrays to ensure shapes match.

## Common Development Tasks

### Adding a New Ufunc

1. Define the operation struct (e.g., `MyOp`).
2. Implement `Ufunc` trait for it.
3. Register it in `src/lib.rs` or relevant module loader using `register!` macro.
4. Add generic implementations for valid types (Float, Int).

### Debugging

- Use `println!` or `dbg!` macro.
- If you see "Type mismatch" errors in ufuncs, check that your `get_ufunc_typed::<T>` call matches the array's actual data type.

## Troubleshooting

- **"Unresolved import"**: We often re-export items. Check `src/lib.rs`.
- **"Unsafe code"**: Try to avoid `unsafe`. Use the established safe patterns (like `as_any` downcasting) unless absolutely necessary for FFI or raw memory ops.
- **Tests failing**: Check `tests/conformance_tests.rs` for parity logic. NumPy sometimes has specific behavior (e.g. reduction on empty arrays) that we must mimic.

## Resources

- [Architecture Overview](../RUST_NUMPY_ARCHITECTURE.md)
- [NumPy API Docs](https://numpy.org/doc/) (our reference)
