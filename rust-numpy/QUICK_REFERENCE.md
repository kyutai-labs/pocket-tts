# Rust-NumPy Array Quick Reference

## Array<T> Structure (360 lines total)

### Core Fields
```rust
pub struct Array<T> {
    data: Arc<MemoryManager<T>>,  // Reference-counted data
    shape: Vec<usize>,           // [dim0, dim1, ...]
    strides: Vec<isize>,         // Byte offsets
    dtype: Dtype,                // NumPy dtype
    offset: usize,               // Start index
}
```

### View Types
```rust
pub struct ArrayView<'a, T> {
    data: &'a [T],              // Borrowed slice
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}

pub struct ArrayViewMut<'a, T> {
    data: &'a mut [T],          // Mutable slice
    // ... same as above
}
```

---

## Memory Layout

### C-Contiguous (Row-Major)
- **Last dimension varies fastest**
- Shape `[2, 3]` → Strides `[3, 1]`
- Memory: `[a00, a01, a02, a10, a11, a12]`

```rust
fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}
```

### F-Contiguous (Column-Major)
- **First dimension varies fastest**
- Shape `[2, 3]` → Strides `[1, 2]`
- Memory: `[a00, a10, a01, a11, a02, a12]`

```rust
fn compute_fortran_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    strides[0] = 1;
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}
```

---

## Stride Calculations

### Linear Index from Multi-Dim
```rust
linear_idx = Σ(indices[i] × strides[i])
```

**Example:** `[1, 2]` with strides `[4, 1]`
```
= (1 × 4) + (2 × 1) = 4 + 2 = 6
```

### Multi-Dim from Linear Index
```rust
// Reverse iteration (C-order)
for (i, &dim_size) in shape.iter().enumerate().rev() {
    indices[i] = remaining % dim_size;
    remaining /= dim_size;
}
indices.reverse();
```

**Example:** Linear `6` in shape `[3, 4]`
```
Step 1: indices[1] = 6 % 4 = 2, remaining = 6 / 4 = 1
Step 2: indices[0] = 1 % 3 = 1, remaining = 1 / 3 = 0
Result: [1, 2]
```

---

## Key Operations (Cost)

| Operation | Cost | Notes |
|-----------|------|-------|
| `clone()` | O(1) | Arc::clone() only |
| `view()` | O(1) | Borrow slice |
| `transpose()` | O(1) | Reverse vectors |
| `reshape()` (contiguous) | O(1) | Update shape |
| `reshape()` (non-contiguous) | O(n) | Copy required |
| `slice()` | O(k) | k = dimensions |

---

## Shape Properties

```rust
ndim()   // len(shape)  - Number of dimensions
size()   // product(shape) - Total elements
shape()  // &shape
strides() // &strides
```

---

## Broadcasting Rules

Two shapes are broadcastable if:
- Equal, OR
- One is 1, OR
- One dimension missing (implicit 1)

**Example:**
- `[3, 4]` + `[4]` → `[3, 4]` ✓
- `[2, 3]` + `[3, 4]` → Shape mismatch ✗
- `[1, 5]` + `[5]` → `[1, 5]` ✓

**Broadcast stride:** 0 means repeat element

---

## Memory Manager

```rust
pub struct MemoryManager<T> {
    data: Vec<T>,
    ref_count: AtomicUsize,
}
```

**Key methods:**
- `as_slice()` / `as_slice_mut()` - Borrow data
- `get()` / `get_mut()` - Element access
- `as_ptr()` / `as_mut_ptr()` - Raw pointers

---

## Copy-on-Write

```rust
// Clone is O(1) - shared data
let b = a.clone();

// First mutation triggers copy
b.set(0, 42);  // Arc::make_mut() copies if shared
```

---

## Contiguity Checks

```rust
is_c_contiguous(shape, strides)  // C-layout
is_f_contiguous(shape, strides)  // F-layout
is_contiguous(shape, strides)    // Either
```

---

## Slice Types

```rust
pub enum Slice {
    Full,                       // (:)
    Range(start, stop),          // (start:stop)
    RangeStep(start, stop, step),// (start:stop:step)
    Index(idx),                  // Single index
    From(start),                 // (start:)
    To(stop),                   // (:end)
    Step(step),                 // (:step)
}
```

---

## Dtype Support

| Category | Types |
|----------|-------|
| Signed | i8, i16, i32, i64 |
| Unsigned | u8, u16, u32, u64 |
| Float | f16, f32, f64 |
| Complex | c32, c64, c128 |
| Other | bool, str, object, datetime64 |

**Properties:**
```rust
dtype.itemsize()   // Bytes per element
dtype.alignment()  // Alignment requirement
dtype.kind()       // Category (Integer, Float, etc.)
dtype.can_cast_to(&other)  // Safe cast check
```

---

## SIMD Alignment

```rust
#[cfg(target_arch = "x86_64")]
pub fn simd_alignment() -> usize { 32 }  // AVX2

#[cfg(target_arch = "aarch64")]
pub fn simd_alignment() -> usize { 16 }  // NEON
```

---

## File Structure

```
src/
├── array.rs      (360 lines) - Array<T>, ArrayView, ArrayViewMut
├── strides.rs    (208 lines) - Stride calculations
├── slicing.rs    (344 lines) - Indexing, slices, iterators
├── memory.rs     (325 lines) - MemoryManager, alignment
├── dtype.rs      (358 lines) - Dtype system
└── lib.rs        (80 lines)  - Module exports
```

---

## Example Usage

```rust
// Create arrays
let a = Array::from_vec(vec![1, 2, 3]);
let b = Array::from_shape_vec(vec![2, 2], vec![1, 2, 3, 4])?;

// Access properties
println!("Shape: {:?}", b.shape());    // [2, 2]
println!("Ndim: {}", b.ndim());         // 2
println!("Size: {}", b.size());         // 4

// Operations
let view = b.view();        // Immutable view
let transposed = b.t();    // O(1) transpose
let c = b.reshape(vec![4])?;  // O(1) reshape

// Indexing
let elem = b.get(2)?;     // Linear indexing
let elem = b.get_by_indices(&[1, 0])?;  // Multi-dim
```

---

## Key Design Principles

1. **Zero-copy:** Views, clone, transpose, reshape (contiguous)
2. **Stride-based:** Flexible memory layouts via strides
3. **Copy-on-write:** Shared data until mutation (Arc)
4. **Type-safe:** Compile-time dtype checking
5. **NumPy-compatible:** Matches NumPy API surface

---

## Extension Points

- **Backends:** CUDA/Metal via trait abstraction
- **SIMD:** Use alignment module for vectorization
- **Lazy eval:** Computation graphs (dask-style)
- **Memory mapping:** File I/O without loading
- **Advanced indexing:** Boolean masks, integer arrays

---

## Comparison: rust-numpy vs NumPy

| Feature | rust-numpy | NumPy |
|---------|------------|-------|
| Zero-copy views | ✓ | ✓ |
| Copy-on-write | ✓ | ✗ |
| Thread safety | Compile-time | GIL |
| Type safety | Static | Dynamic |
| Memory safety | Guaranteed | Manual |

---

**Total Lines Analyzed: 1,675**
**Files: 5** (array.rs, strides.rs, slicing.rs, memory.rs, dtype.rs)
