# Tech Stack and Coding Conventions

## Tech Stack

- Language: Rust
- Framework: None (library)
- Package Manager: Cargo
- Testing: Built-in Rust test framework (`cargo test`)
- Dependencies: ndarray, num-traits, num-complex, rand, lazy_static

## Coding Conventions

- Use `Result<T, NumPyError>` for fallible operations
- Follow NumPy's API naming conventions where applicable
- Prefer generic implementations with trait bounds over concrete types
- Use `#[cfg(feature = "...")]` for optional features like rayon
- Keep functions small and focused - prefer composition over monolithic functions
- Add doc comments to public APIs
- Tests go in `mod tests` blocks at the end of each file
- Use `cargo fmt` and `cargo clippy` for formatting and linting

## Project Structure

- `src/lib.rs` - Main library exports
- `src/array.rs` - Core Array type
- `src/dtype.rs` - Data type system
- `src/ufunc.rs` - Universal function framework
- `src/linalg.rs` - Linear algebra operations
- `src/sorting.rs` - Sort and partition operations
- `src/set_ops.rs` - Set operations (unique, union, intersection)
- `src/bitwise.rs` - Bitwise operations
- `src/math_ufuncs.rs` - Mathematical ufuncs (exp, log, sin, etc.)
- `src/parallel_ops.rs` - Parallel operations with Rayon
- `src/slicing.rs` - Array slicing and indexing
- `src/window.rs` - Window functions

## Testing Commands

```bash
# Run all library tests
cargo test --lib

# Run specific test
cargo test <test_name>

# Check for warnings
cargo build 2>&1 | grep warning

# Check for errors
cargo check
```
