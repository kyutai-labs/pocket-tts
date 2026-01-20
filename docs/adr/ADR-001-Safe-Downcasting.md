# ADR-0001: Safe Downcasting for Universal Functions

## Status

Accepted

## Context

In the initial implementation of the Universal Functions (ufunc) system, generic `ArrayView` traits were used to handle input/output arrays of any type. To perform actual operations, these views needed to be converted back to concrete `Array<T>` types.

The original approach used `unsafe` pointer casts:

```rust
let input = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
```

This was fragile and led to potential undefined behavior if the underlying type did not match `T` exactly, undermining Rust's safety guarantees.

## Decision

We replaced all `unsafe` pointer casts with a safe downcasting mechanism using `std::any::Any`.

1.  Extended `ArrayView` and `ArrayViewMut` traits to include `as_any` and `as_any_mut` methods:
    ```rust
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    ```
2.  Implemented these methods for `Array<T>` to return `self`.
3.  Updated all ufunc executions (`MathBinaryUfunc`, `ComparisonUfunc`, etc.) to use verified downcasting:
    ```rust
    let input = inputs[0].as_any().downcast_ref::<Array<T>>()
        .ok_or_else(|| NumPyError::ufunc_error(...))?;
    ```

## Consequences

### Positive

- **Safety**: Semantic type safety is enforced at runtime. Segfaults and undefined behavior from type mismatches are eliminated.
- **Error Handling**: Type mismatches now produce clear `Result::Err` values instead of undefined crashes.
- **Maintainability**: The codebase is free of complex and dangerous `unsafe` blocks in the core execution path.

### Negative

- **Performance**: Accessing `Any` and downcasting adds a minimal runtime overhead (virtual function call + type check). Given these are interpreted array operations (not inner loops), this overhead is negligible compared to the operation cost on array data.

## Alternatives Considered

- **Enum Dispatch**: wrapping all arrays in an `ArrayEnum` (Int, Float, etc.). Rejected because it would require updating the enum for every new generic type and complicates generic code.
- **Unsafe with TypeName checks**: Checking `std::any::type_name` then casting. Rejected because `downcast_ref` encapsulates this pattern safely.
