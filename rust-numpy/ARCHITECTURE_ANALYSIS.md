# Rust-NumPy Array Architecture Analysis

## Executive Summary

This document provides a comprehensive analysis of the rust-numpy array implementation, focusing on the core data structures, memory layout, stride calculations, and dimensional handling.

---

## 1. Core Array Structures

### 1.1 Array<T> - Primary Owned Array

**Location:** `src/array.rs:10-16`

```rust
pub struct Array<T> {
    data: Arc<MemoryManager<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}
```

**Field Descriptions:**

| Field | Type | Purpose |
|-------|------|---------|
| `data` | `Arc<MemoryManager<T>>` | Thread-safe reference-counted data storage |
| `shape` | `Vec<usize>` | Dimension sizes (e.g., `[3, 4]` for 3×4 matrix) |
| `strides` | `Vec<isize>` | Byte offsets between elements in each dimension |
| `dtype` | `Dtype` | NumPy-compatible data type enumeration |
| `offset` | `usize` | Starting index into data buffer |

**Key Characteristics:**
- **Ownership model:** Uses `Arc` for copy-on-write semantics
- **Reference counting:** Enables zero-copy cloning via `MemoryManager`
- **Flexible layout:** Supports arbitrary stride patterns (not just contiguous)

**Memory Management:**
- `MemoryManager<T>` wraps `Vec<T>` with atomic reference counting
- `Arc<MemoryManager<T>>` enables shared ownership without data duplication
- Clone operation is O(1) - only copies the `Arc` pointer
- `Arc::make_mut()` performs copy-on-write when mutability is needed

---

### 1.2 ArrayView<'a, T> - Immutable View

**Location:** `src/array.rs:20-26`

```rust
pub struct ArrayView<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}
```

**Purpose:** Zero-cost immutable view into existing array data

**Key Differences from `Array<T>`:**
- Uses slice reference (`&'a [T]`) instead of owned data
- Lifetime parameter `'a` ties view to source array
- Cannot own data (only borrowed)
- Cheaper to create (no Arc overhead)

**Use Cases:**
- Passing subarrays to functions without copying
- Temporary views during computations
- Slicing operations

---

### 1.3 ArrayViewMut<'a, T> - Mutable View

**Location:** `src/array.rs:30-36`

```rust
pub struct ArrayViewMut<'a, T> {
    data: &'a mut [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}
```

**Purpose:** Zero-cost mutable view into existing array data

**Key Differences:**
- Uses mutable slice reference (`&'a mut [T]`)
- Exclusive ownership during lifetime
- Allows in-place modifications

**Use Cases:**
- In-place operations on array subsets
- Modifying slices without copying
- Avoiding allocation for temporary modifications

---

## 2. Memory Layout Models

### 2.1 C-Contiguous (Row-Major) Layout

**Definition:** Last dimension varies fastest (contiguous in memory)

**Example:** Array with shape `[3, 4]`
```
Memory layout: [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
                  ^^^^ row 0           ^^^^ row 1           ^^^^ row 2
```

**Stride Calculation:**
```rust
fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    strides[shape.len() - 1] = 1;  // Last dimension stride = 1

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }

    strides
}
```

**For shape `[3, 4]`:**
- `strides[1] = 1` (last dimension)
- `strides[0] = 1 * 4 = 4`
- Result: `[4, 1]`

**Verification:**
```rust
fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    strides == &compute_strides(shape)[..]
}
```

---

### 2.2 F-Contiguous (Column-Major) Layout

**Definition:** First dimension varies fastest (Fortran-style)

**Example:** Array with shape `[3, 4]`
```
Memory layout: [a00, a10, a20, a01, a11, a21, a02, a12, a22, a03, a13, a23]
                  ^^^^ col 0           ^^^^ col 1           ^^^^ col 2
```

**Stride Calculation:**
```rust
fn compute_fortran_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    strides[0] = 1;  // First dimension stride = 1

    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }

    strides
}
```

**For shape `[3, 4]`:**
- `strides[0] = 1` (first dimension)
- `strides[1] = 1 * 3 = 3`
- Result: `[1, 3]`

**Verification:**
```rust
fn is_f_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    strides == &compute_fortran_strides(shape)[..]
}
```

---

### 2.3 Contiguity Detection

```rust
pub fn is_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    is_c_contiguous(shape, strides) || is_f_contiguous(shape, strides)
}
```

**Stride Order Enumeration:**
```rust
pub enum StrideOrder {
    C,      // C-contiguous (row-major)
    F,      // Fortran-contiguous (column-major)
    Neither, // Neither C nor F contiguous
}
```

---

## 3. Stride Calculation System

### 3.1 Core Functions

**Location:** `src/strides.rs`

| Function | Purpose | Return Type |
|----------|---------|-------------|
| `compute_strides()` | C-contiguous strides | `Vec<isize>` |
| `compute_fortran_strides()` | F-contiguous strides | `Vec<isize>` |
| `is_c_contiguous()` | Check C layout | `bool` |
| `is_f_contiguous()` | Check F layout | `bool` |
| `is_contiguous()` | Check either layout | `bool` |
| `compute_linear_index()` | Multi-dim to linear | `usize` |
| `compute_multi_indices()` | Linear to multi-dim | `Vec<usize>` |

---

### 3.2 Linear Index Calculation

**Formula:** `linear_idx = Σ(indices[i] × strides[i])`

```rust
pub fn compute_linear_index(indices: &[usize], strides: &[isize]) -> usize {
    indices.iter()
        .zip(strides.iter())
        .map(|(i, s)| i * *s as usize)
        .sum()
}
```

**Example:** For position `[1, 2]` in shape `[3, 4]` with C-strides `[4, 1]`:
```
linear_idx = (1 × 4) + (2 × 1) = 4 + 2 = 6
```

---

### 3.3 Multi-Dimensional Index Calculation

**Formula:** Iterative division modulo shape dimensions

```rust
pub fn compute_multi_indices(linear_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = linear_index;

    // Reverse iteration (last dimension first for C-order)
    for (i, &dim_size) in shape.iter().enumerate().rev() {
        indices[i] = remaining % dim_size;
        remaining /= dim_size;
    }

    indices.reverse();  // Restore original order
    indices
}
```

**Example:** Linear index `6` in shape `[3, 4]`:
```
Iteration 1 (dim=4): indices[1] = 6 % 4 = 2, remaining = 6 / 4 = 1
Iteration 2 (dim=3): indices[0] = 1 % 3 = 1, remaining = 1 / 3 = 0
Result: [1, 2]
```

---

### 3.4 Broadcasting Support

**Shape Broadcasting:**
```rust
pub fn are_shapes_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);

    for i in 0..max_len {
        let dim1 = if i >= max_len - len1 {
            shape1[i - (max_len - len1)]
        } else {
            1  // Implicit broadcast dimension
        };

        let dim2 = if i >= max_len - len2 {
            shape2[i - (max_len - len2)]
        } else {
            1
        };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }

    true
}
```

**Broadcast Shape Computation:**
```rust
pub fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = std::cmp::max(len1, len2);
    let mut result = vec![0; max_len];

    for i in 0..max_len {
        let dim1 = if i >= max_len - len1 {
            shape1[i - (max_len - len1)]
        } else {
            1
        };

        let dim2 = if i >= max_len - len2 {
            shape2[i - (max_len - len2)]
        } else {
            1
        };

        result[i] = std::cmp::max(dim1, dim2);
    }

    result
}
```

**Broadcast Strides:**
```rust
pub fn compute_broadcast_strides(
    original_shape: &[usize],
    original_strides: &[isize],
    broadcast_shape: &[usize]
) -> Vec<isize> {
    let orig_len = original_shape.len();
    let broadcast_len = broadcast_shape.len();
    let mut result = vec![0; broadcast_len];

    for i in 0..broadcast_len {
        if i >= broadcast_len - orig_len {
            let orig_idx = i - (broadcast_len - orig_len);
            let orig_dim = original_shape[orig_idx];

            if orig_dim == 1 {
                result[i] = 0;  // Broadcast: repeat across dimension
            } else {
                result[i] = original_strides[orig_idx];  // Preserve stride
            }
        } else {
            result[i] = 0;  // New dimension being broadcast
        }
    }

    result
}
```

**Key insight:** Stride of 0 indicates broadcasting (element repeats across dimension)

---

## 4. Shape and Dimension Handling

### 4.1 Core Methods

**Location:** `src/array.rs:39-67`

| Method | Return | Description |
|--------|--------|-------------|
| `shape()` | `&[usize]` | Get shape slice |
| `strides()` | `&[isize]` | Get strides slice |
| `offset()` | `usize` | Get data offset |
| `ndim()` | `usize` | Number of dimensions |
| `size()` | `usize` | Total element count (product of shape) |
| `dtype()` | `&Dtype` | Get data type |

**Implementation Details:**

```rust
pub fn ndim(&self) -> usize {
    self.shape.len()  // Direct lookup
}

pub fn size(&self) -> usize {
    self.shape.iter().product()  // Product of all dimensions
}
```

---

### 4.2 Shape Operations

**Reshape:**
```rust
pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
    // Validate: total elements must match
    if self.size() != new_shape.iter().product::<usize>() {
        return Err(NumPyError::shape_mismatch(vec![self.size()], new_shape));
    }

    if self.is_c_contiguous() {
        // Zero-copy: just update shape and compute new strides
        let new_strides = compute_strides(&new_shape);
        Ok(Self {
            data: self.data.clone(),      // Shared data (Arc)
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype.clone(),
            offset: self.offset,         // Preserve offset
        })
    } else {
        // Need to copy and reorganize data
        Err(NumPyError::NotContiguous)
    }
}
```

**Transpose:**
```rust
pub fn t(&self) -> Self {
    if self.ndim() <= 1 {
        return self.clone();  // No-op for 1D arrays
    }

    let mut new_shape = self.shape.clone();
    new_shape.reverse();  // Reverse dimension order

    let mut new_strides = self.strides.clone();
    new_strides.reverse();  // Reverse stride order

    Self {
        data: self.data.clone(),      // Shared data (zero-copy)
        shape: new_shape,
        strides: new_strides,
        dtype: self.dtype.clone(),
        offset: self.offset,
    }
}
```

**Key insight:** Transpose is O(1) - just reverses shape/strides, no data copy

---

### 4.3 Slicing System

**Slice Types:** `src/slicing.rs:69-85`

```rust
pub enum Slice {
    Full,               // (:)
    Range(isize, isize),            // (start:stop)
    RangeStep(isize, isize, isize), // (start:stop:step)
    Index(isize),                  // Single index
    From(isize),                   // (start:)
    To(isize),                     // (:end)
    Step(isize),                   // (:step)
}
```

**Multi-Dimensional Slicing:**
```rust
pub struct MultiSlice {
    slices: Vec<Slice>,
}
```

**Slice Length Calculation:**
```rust
pub fn len(&self, dim_len: usize) -> usize {
    let len = dim_len as isize;
    let (start, stop, step) = self.to_range(len);

    if step == 0 {
        return 0;  // Invalid
    }

    let actual_start = start.max(0).min(len);
    let actual_stop = stop.max(0).min(len);

    if (step > 0 && actual_start >= actual_stop) ||
       (step < 0 && actual_start <= actual_stop) {
        return 0;  // Empty slice
    }

    ((actual_stop - actual_start).abs() as usize + step.abs() as usize - 1)
        / step.abs() as usize
}
```

**Formula:** `length = ⌈|stop - start| / |step|⌉`

---

## 5. Memory Architecture

### 5.1 MemoryManager<T>

**Location:** `src/memory.rs:5-8`

```rust
pub struct MemoryManager<T> {
    data: Vec<T>,
    ref_count: std::sync::atomic::AtomicUsize,
}
```

**Key Features:**
- Wraps `Vec<T>` for managed access
- Atomic reference counting (thread-safe)
- Provides slice access methods
- Raw pointer access for FFI integration

**Key Methods:**
- `as_slice()` / `as_slice_mut()` - Borrow as slice
- `get()` / `get_mut()` - Element access
- `as_ptr()` / `as_mut_ptr()` - Raw pointers
- `inc_ref()` / `dec_ref()` - Reference counting

---

### 5.2 Copy-on-Write Semantics

```rust
impl<T> Array<T> {
    pub fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),  // O(1) - just Arc::clone()
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype.clone(),
            offset: self.offset,
        }
    }
}

impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.size() {
            let linear_idx = self.offset + index;
            // Ensure unique access - performs copy if shared
            Arc::make_mut(&mut self.data).get_mut(linear_idx)
        } else {
            None
        }
    }
}
```

**Behavior:**
- Clone is cheap (just increments Arc refcount)
- First mutation triggers copy-on-write (data becomes unique)
- Enables zero-copy views and sharing until mutation

---

### 5.3 SIMD Alignment Support

**Location:** `src/memory.rs:261-284`

```rust
pub mod alignment {
    pub fn simd_alignment() -> usize {
        #[cfg(target_arch = "x86_64")]
        { 32 }  // AVX2: 32-byte alignment

        #[cfg(target_arch = "aarch64")]
        { 16 }  // NEON: 16-byte alignment

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { 8 }   // Default: 8-byte alignment
    }

    pub fn align_ptr<T>(ptr: *mut T, alignment: usize) -> *mut T {
        let addr = ptr as usize;
        let aligned = (addr + alignment - 1) & !(alignment - 1);
        aligned as *mut T
    }
}
```

**Purpose:** Optimize for vectorized operations (SIMD)

---

## 6. Data Type System

### 6.1 Dtype Enumeration

**Location:** `src/dtype.rs:5-45`

```rust
pub enum Dtype {
    // Signed integers
    Int8, Int16, Int32, Int64,

    // Unsigned integers
    UInt8, UInt16, UInt32, UInt64,

    // Floating point
    Float16, Float32, Float64,

    // Complex
    Complex32, Complex64, Complex128,

    // Other
    Bool, String, Unicode,
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),
    Object,
    Struct(Vec<StructField>),
}
```

---

### 6.2 Type Properties

```rust
impl Dtype {
    pub fn itemsize(&self) -> usize {
        match self {
            Dtype::Int8 | Dtype::UInt8 | Dtype::Bool => 1,
            Dtype::Int16 | Dtype::UInt16 | Dtype::Float16 => 2,
            Dtype::Int32 | Dtype::UInt32 | Dtype::Float32 | Dtype::Complex32 => 4,
            Dtype::Int64 | Dtype::UInt64 | Dtype::Float64 | Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
            Dtype::String | Dtype::Unicode => 8,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => 8,
            Dtype::Object => 8,
            Dtype::Struct(fields) => fields.iter().map(|f| f.dtype.itemsize()).sum(),
        }
    }

    pub fn alignment(&self) -> usize {
        // Similar to itemsize but represents alignment requirements
    }
}
```

---

### 6.3 Type Checking and Casting

```rust
pub fn can_cast_to(&self, other: &Dtype) -> bool {
    use DtypeKind::*;

    let self_kind = self.kind();
    let other_kind = other.kind();

    match (self_kind, other_kind) {
        (Integer, Integer) | (Unsigned, Integer) | (Unsigned, Unsigned) => {
            self.itemsize() <= other.itemsize()
        }
        (Integer, Unsigned) => false,
        (Float, Float) => self.itemsize() <= other.itemsize(),
        (Complex, Complex) => self.itemsize() <= other.itemsize(),
        (Integer | Unsigned | Float, Complex) => true,
        (Complex, Float) => false,
        (Bool, _) => true,
        (_, Bool) => false,
        (String, String) | (Datetime, Datetime) | (Object, _) | (_, Object) => true,
        _ => false,
    }
}
```

---

## 7. Iteration System

### 7.1 Immutable Iterator

**Location:** `src/slicing.rs:264-286`

```rust
pub struct ArrayIter<'a, T> {
    array: &'a Array<T>,
    current: usize,
}

impl<'a, T> Iterator for ArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.array.size() {
            let element = self.array.get(self.current);
            self.current += 1;
            element
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.size() - self.current;
        (remaining, Some(remaining))
    }
}
```

---

### 7.2 Mutable Iterator

**Location:** `src/slicing.rs:289-311`

```rust
pub struct ArrayIterMut<'a, T> {
    array: &'a mut Array<T>,
    current: usize,
}

impl<'a, T> Iterator for ArrayIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.array.size() {
            // Requires proper mutable access implementation
            // Currently returns None (placeholder)
            self.current += 1;
            None
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.size() - self.current;
        (remaining, Some(remaining))
    }
}
```

**Note:** Mutable iterator implementation incomplete (requires copy-on-write handling)

---

## 8. Key Architectural Insights

### 8.1 Zero-Copy Operations

| Operation | Cost | Description |
|-----------|------|-------------|
| `clone()` | O(1) | Just Arc::clone() |
| `view()` | O(1) | Create ArrayView with borrowed data |
| `transpose()` | O(1) | Reverse shape/strides vectors |
| `reshape()` (contiguous) | O(1) | Update shape, compute new strides |
| `slice()` | O(k) | k = number of dimensions |

---

### 8.2 Copy-on-Write Behavior

1. **Initial state:** Data may be shared via Arc
2. **First mutation:** `Arc::make_mut()` triggers copy if refcount > 1
3. **Subsequent mutations:** Direct access to unique data
4. **Benefit:** Efficient for read-heavy workloads

---

### 8.3 Memory Layout Flexibility

- **Arbitrary strides:** Supports non-contiguous arrays
- **Broadcasting:** Zero-stride dimensions repeat elements
- **Views:** Can point to subsets of existing data
- **Offset support:** Arrays can start at non-zero positions

---

### 8.4 Thread Safety

- **Read-only:** Multiple threads can read shared Array<T> (Arc)
- **Mutable:** Requires exclusive ownership (Rust's &mut)
- **MemoryManager:** Uses AtomicUsize for thread-safe refcounting

---

## 9. Comparison with NumPy

| Feature | rust-numpy | NumPy |
|---------|------------|-------|
| Zero-copy views | ✓ | ✓ |
| Copy-on-write | ✓ (via Arc) | ✗ (always copy) |
| Memory layout | Both C/F | Both C/F |
| Stride calculation | ✓ | ✓ |
| Broadcasting | ✓ | ✓ |
| Thread safety | Compile-time (Rust) | GIL (Python) |
| Type safety | Static | Dynamic |

---

## 10. Extension Points

### 10.1 Potential Enhancements

1. **Complete slicing:** Implement full slice-to-array conversion
2. **Advanced indexing:** Boolean masks, integer arrays
3. **SIMD operations:** Leverage alignment module for vectorized ops
4. **GPU backends:** CUDA/Metal integration via trait abstraction
5. **Lazy evaluation:** Computation graphs (dask-style)
6. **Memory mapping:** Direct file I/O without loading

---

### 10.2 Memory Pool Usage

```rust
pub struct MemoryPool<T> {
    available: Vec<Vec<T>>,
    max_size: usize,
}
```

**Purpose:** Reuse allocated buffers to reduce allocation overhead

---

## 11. Summary

### Core Principles

1. **Zero-copy by default:** Views, cloning, transpose, reshape (contiguous)
2. **Copy-on-write:** Efficient sharing until mutation
3. **Stride-based indexing:** Flexible memory layouts (C/F/broadcast)
4. **Type safety:** Compile-time dtype checking via generics
5. **Thread safety:** Arc for sharing, &mut for exclusive mutation

### Strengths

- ✓ Clean separation of concerns (array, strides, memory, dtype)
- ✓ Zero-cost abstractions (views are just pointers)
- ✓ NumPy-compatible API surface
- ✓ Extensible architecture for multiple backends

### Current Limitations

- ✗ Incomplete slicing implementation
- ✗ Mutable iterator placeholder
- ✗ No advanced indexing (masks, fancy indexing)
- ✗ Limited error handling in reshape (non-contiguous case)

---

## 12. Example Usage Patterns

### Creating Arrays

```rust
// From vector
let a = Array::from_vec(vec![1, 2, 3, 4]);

// With shape
let b = Array::from_shape_vec(vec![2, 2], vec![1, 2, 3, 4]).unwrap();

// Zeros/Ones
let c = Array::<f32>::zeros(vec![3, 4]);
let d = Array::<i32>::ones(vec![2, 3]);

// Full
let e = Array::full(vec![2, 2], 42);
```

### Array Operations

```rust
let array = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

// Properties
println!("Shape: {:?}", array.shape());      // [2, 3]
println!("Strides: {:?}", array.strides());  // [3, 1]
println!("Ndim: {}", array.ndim());          // 2
println!("Size: {}", array.size());          // 6

// Contiguity checks
println!("C-contiguous: {}", array.is_c_contiguous());  // true
println!("F-contiguous: {}", array.is_f_contiguous());  // false

// Zero-copy operations
let view = array.view();        // Immutable view
let cloned = array.clone();    // O(1) clone
let transposed = array.t();    // O(1) transpose
```

### Indexing

```rust
// Linear indexing
let elem = array.get(4).unwrap();  // 5

// Multi-dimensional indexing
let multi_indices = vec![1, 2];
let elem = array.get_by_indices(&multi_indices).unwrap();  // 6

// Setting elements
let mut array = array.clone();
array.set(0, 100).unwrap();
array.set_by_indices(&[0, 1], 200).unwrap();
```

---

**End of Analysis**
