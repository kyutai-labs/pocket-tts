# rust-numpy Usage Examples

**Version:** 0.1.0

This document provides practical examples for using rust-numpy, organized by topic and complexity level.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Operations](#basic-operations)
3. [Mathematical Functions](#mathematical-functions)
4. [Linear Algebra](#linear-algebra)
5. [Broadcasting & Reshaping](#broadcasting-reshaping)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling](#error-handling)
8. [Advanced Patterns](#advanced-patterns)
9. [NumPy Migration Guide](#numpy-migration-guide)

---

## Getting Started

### Installation

Add rust-numpy to your `Cargo.toml`:

```toml
[dependencies]
rust-numpy = { version = "0.1", features = ["std"] }

# For performance
rust-numpy = { version = "0.1", features = ["std", "simd"] }

# For parallel processing
rust-numpy = { version = "0.1", features = ["std", "rayon"] }

# For everything
rust-numpy = { version = "0.1", features = ["std", "simd", "rayon"] }
```

### Basic Usage

```rust
use rust_numpy::*;

fn main() {
    // Create arrays
    let a = array![1, 2, 3, 4, 5];
    let b = array![6, 7, 8, 9, 10];
    
    // Array operations
    println!("Array a: {:?}", a);
    println!("Shape: {:?}", a.shape());
    println!("Size: {}", a.size());
    
    // Mathematical operations
    let c = a.add(&b).unwrap();
    println!("a + b: {:?}", c);
    
    let d = a.multiply(&b).unwrap();
    println!("a * b: {:?}", d);
}
```

---

## Basic Operations

### Example 1: Array Creation

```rust
use rust_numpy::*;

fn main() {
    // Different ways to create arrays
    
    // Using array! macro (type inference)
    let a = array![1, 2, 3, 4, 5];
    println!("Inferred array: {:?}", a);
    
    // Explicit type specification
    let b = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    println!("Float64 array: {:?}", b);
    
    // Different types
    let c = Array::<i32>::zeros(vec![3, 4]);
    println!("Int32 zeros: {:?}", c);
}
```

**Expected Output:**
```
Inferred array: Array { shape: [5], dtype: Int32, data: [1, 2, 3, 4, 5] }
Float64 array: Array { shape: [3], dtype: Float64, data: [1.0, 2.0, 3.0] }
Int32 zeros: Array { shape: [3, 4], dtype: Int32, data: [0, 0, 0, 0] }
```

### Example 2: Shape Operations

```rust
use rust_numpy::*;

fn main() {
    let a = array![[1, 2, 3], [4, 5, 6]];
    
    println!("Original shape: {:?}", a.shape());
    println!("Dimensions: {}", a.ndim());
    println!("Size: {}", a.size());
    
    // Transpose
    let transposed = a.transpose().unwrap();
    println!("Transposed shape: {:?}", transposed.shape());
    
    // Reshape
    let reshaped = a.reshape(&[3, 4]).unwrap();
    println!("Reshaped shape: {:?}", reshaped.shape());
}
```

**Expected Output:**
```
Original shape: [2, 3]
Dimensions: 2
Size: 6
Transposed shape: [3, 2]
Reshaped shape: [3, 4]
```

### Example 3: Indexing

```rust
use rust_numpy::*;

fn main() {
    let a = array![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ];
    
    // Single element access
    println!("Element [0][1][2]: {}", a.get(0).unwrap());  // 3
    println!("Element [1][1][2]: {}", a.get(4).unwrap());  // 8
    
    // Using indexing operator
    println!("Element a[0][1][2]: {}", a[0][1][2]);  // 3
}
```

---

## Mathematical Functions

### Example 4: Trigonometric Functions

```rust
use rust_numpy::*;
use std::f64::consts::PI;

fn main() {
    let angles = Array::<f64>::from_vec(vec![
        0.0,
        PI / 4.0,
        PI / 2.0,
        3.0 * PI / 4.0,
    ]);
    
    // Sine
    let sin_values = angles.sin().unwrap();
    println!("Sin: {:?}", sin_values.to_vec());
    
    // Cosine
    let cos_values = angles.cos().unwrap();
    println!("Cos: {:?}", cos_values.to_vec());
    
    // Tangent
    let tan_values = angles.tan().unwrap();
    println!("Tan: {:?}", tan_values.to_vec());
}
```

**Expected Output:**
```
Sin: [0.0, 0.70710678118654757, 1.0, 1.4138823378772518]
Cos: [1.0, 0.7071067811865475, 0.0, -0.7071067811865475]
Tan: [0.0, 1.0, 1.7320508075688772, 0.0]
```

### Example 5: Exponential and Logarithmic

```rust
use rust_numpy::*;

fn main() {
    let x = Array::<f64>::from_vec(vec![1.0, 2.0, 10.0]);
    
    // Exponential
    let exp_values = x.exp().unwrap();
    println!("Exp: {:?}", exp_values.to_vec());
    
    // Natural logarithm
    let log_values = x.log().unwrap();
    println!("Log: {:?}", log_values.to_vec());
    
    // Square root
    let sqrt_values = x.sqrt_().unwrap();
    println!("Sqrt: {:?}", sqrt_values.to_vec());
}
```

**Expected Output:**
```
Exp: [2.718281828459045, 7.38905609893065, 22026.465794806718]
Log: [0.0, 0.6931471805599453, 2.302585092994046]
Sqrt: [1.0, 1.414213562373095, 3.1622776601683795]
```

---

## Linear Algebra

### Example 6: Matrix Operations

```rust
use rust_numpy::*;

fn main() {
    let a = Array::<f64>::from_vec(vec![
        [1.0, 2.0],
        [3.0, 4.0],
    ]);
    let b = Array::<f64>::from_vec(vec![
        [5.0, 6.0],
        [7.0, 8.0],
    ]);
    
    // Dot product
    let dot = a.dot(&b).unwrap();
    println!("Dot product:\n{:?}", dot.to_vec());
    println!("Shape: {:?}", dot.shape());
    
    // Matrix multiplication
    let matmul = a.matmul(&b).unwrap();
    println!("Matrix multiplication:\n{:?}", matmul.to_vec());
    
    // Transpose
    let a_t = a.transpose().unwrap();
    println!("Transposed:\n{:?}", a_t.to_vec());
}
```

**Expected Output:**
```
Dot product:
[35.0, 44.0]
Shape: [2, 2]

Matrix multiplication:
[[17.0, 22.0], [41.0, 50.0]]
Shape: [2, 2]

Transposed:
[[1.0, 3.0], [7.0, 8.0]]
Shape: [3, 2]
```

### Example 7: Solving Linear Systems

```rust
use rust_numpy::*;

fn main() {
    let a = Array::<f64>::from_vec(vec![
        [2.0, 1.0],
        [1.0, 2.0],
    ]);
    let b = Array::<f64>::from_vec(vec![3.0, 5.0]);
    
    let x = a.solve(&b).unwrap();
    println!("Solution x:\n{:?}", x.to_vec());
}
```

**Expected Output:**
```
Solution x:
[[1.0], [1.0]]
```

---

## Broadcasting & Reshaping

### Example 8: Basic Broadcasting

```rust
use rust_numpy::*;

fn main() {
    let scalar = Array::from_vec(vec![5.0]);
    let array = Array::<f64>::zeros(vec![3, 4]);
    
    // Broadcast scalar to array
    let result = scalar.broadcast_to(&array.shape()).unwrap();
    println!("Broadcasted shape: {:?}", result.shape());
    println!("Broadcasted data:\n{:?}", result.to_vec());
}
```

**Expected Output:**
```
Broadcasted shape: [3, 4]
Broadcasted data:
[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
```

### Example 9: Advanced Broadcasting

```rust
use rust_numpy::*;
use rust_numpy::advanced_broadcast;

fn main() {
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    
    // Repeat array
    let repeated = a.repeat(2, None).unwrap();
    println!("Repeated shape: {:?}", repeated.shape());
    println!("Repeated data:\n{:?}", repeated.to_vec());
    
    // Tile array
    let tiled = a.tile(&[2, 3]).unwrap();
    println!("Tiled shape: {:?}", tiled.shape());
    println!("Tiled data:\n{:?}", tiled.to_vec());
}
```

**Expected Output:**
```
Repeated shape: [6, 3]
Repeated data: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]

Tiled shape: [2, 3, 9]
Tiled data: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
```

---

## Performance Optimization

### Example 10: Enabling SIMD

Build with SIMD features for 4-8x speedup:

```bash
# Add to Cargo.toml
[dependencies]
rust-numpy = { version = "0.1", features = ["std", "simd"] }
```

```rust
// SIMD is automatically used when feature is enabled
use rust_numpy::*;

fn main() {
    let data = Array::<f64>::from_vec((0..10000).map(|i| i as f64).collect());
    
    // Mathematical operations use SIMD automatically
    let sin_result = data.sin().unwrap();
    let exp_result = data.exp().unwrap();
    let log_result = data.log().unwrap();
    
    println!("SIMD-optimized operations completed");
}
```

### Example 11: Enabling Parallel Processing

Build with Rayon features for 2-4x speedup on multi-core systems:

```bash
# Add to Cargo.toml
[dependencies]
rust-numpy = { version = "0.1", features = ["std", "rayon"] }
```

```rust
// Large operations automatically use parallel processing
use rust_numpy::*;

fn main() {
    let data = Array::<f64>::from_vec((0..100000).map(|i| i as f64).collect());
    
    // Reductions automatically parallelize
    let sum = data.sum(None, false).unwrap();
    let mean = data.mean(None, false).unwrap();
    
    println!("Parallel processing: sum={}, mean={}", sum.to_vec(), mean.to_vec());
}
```

### Example 12: Memory-Efficient Operations

Reuse arrays and avoid allocations:

```rust
use rust_numpy::*;

fn main() {
    let a = array![1, 2, 3, 4, 5];
    let b = array![6, 7, 8, 9, 10];
    
    // Reuse array instead of cloning
    let mut output = a.clone();
    output.add_assign(&b).unwrap();
    
    // Use slice instead of to_vec
    let slice = a.as_slice();
    println!("Slice length: {}", slice.len());
}
```

---

## Error Handling

### Example 13: Handling Different Error Types

```rust
use rust_numpy::*;

fn main() {
    // Empty array error
    let empty = Array::<f64>::zeros(vec![]);
    match empty.sum(None, false) {
        Ok(_) => println!("Empty sum succeeded (unexpected)"),
        Err(e) => println!("Empty array error: {:?}", e),
    }
    
    // Shape mismatch error
    let a = array![1, 2];
    let b = array![3, 4, 5];
    match a.add(&b) {
        Ok(_) => println!("Add succeeded"),
        Err(e) => println!("Shape error: {:?}", e),
    }
}
```

---

## Advanced Patterns

### Example 14: Array Slicing

```rust
use rust_numpy::*;
use rust_numpy::slicing::Slice;

fn main() {
    let a = Array::<f64>::from_vec((0..20).map(|i| i as f64).collect());
    
    // Slice from index 5 to 15
    let slice = a.slice(Slice::Range(5, 15)).unwrap();
    println!("Slice length: {}", slice.size());
    println!("Slice data: {:?}", slice.to_vec());
    
    // Slice with step
    let step_slice = a.slice(Slice::RangeStep(0, 20, 2)).unwrap();
    println!("Step slice: {:?}", step_slice.to_vec());
}
```

**Expected Output:**
```
Slice length: 10
Slice data: [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

Step slice: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
```

### Example 15: Iterator Patterns

```rust
use rust_numpy::*;

fn main() {
    let a = array![1, 2, 3, 4, 5];
    
    // Iterate over elements
    let mut sum = 0i64;
    for elem in a.iter() {
        if let Some(val) = elem {
            sum += val;
        }
    }
    println!("Sum: {}", sum);
    
    // Collect into new array
    let doubled: Vec<_> = a.iter().map(|x| x * 2).collect();
    let b = Array::from_vec(doubled);
    println!("Doubled: {:?}", b.to_vec());
}
```

---

## NumPy Migration Guide

### Common Patterns

| NumPy Pattern | rust-numpy Equivalent |
|---------------|---------------------|
| `np.array([1, 2])` | `array!([1, 2])` |
| `np.zeros((3, 4))` | `Array::<f64>::zeros(vec![3, 4])` |
| `np.ones((2, 3))` | `Array::<i32>::ones(vec![2, 3])` |
| `np.arange(0, 10)` | `arange(0, 10, None)` |
| `a.shape` | `a.shape()` |
| `a.ndim` | `a.ndim()` |
| `a.size()` | `a.size()` |
| `a.transpose()` | `a.transpose().unwrap()` |
| `a.reshape((2, 3))` | `a.reshape(&[2, 3]).unwrap()` |
| `np.sin(a)` | `a.sin().unwrap()` |
| `np.exp(a)` | `a.exp().unwrap()` |
| `np.log(a)` | `a.log().unwrap()` |
| `np.dot(a, b)` | `a.dot(&b).unwrap()` |
| `a + b` | `a.add(&b).unwrap()` |
| `a * b` | `a.multiply(&b).unwrap()` |
| `a - b` | `a.subtract(&b).unwrap()` |
| `a / b` | `a.divide(&b).unwrap()` |
| `a.sum()` | `a.sum(None, false).unwrap()` |
| `a.mean()` | `a.mean(None, false).unwrap()` |
| `a.min()` | `a.min(None, false).unwrap()` |
| `a.max()` | `a.max(None, false).unwrap()` |
| `np.sort(a)` | `a.sort().unwrap()` |
| `np.argsort(a)` | `a.argsort().unwrap()` |

### Key Differences

1. **Error Handling**: rust-numpy uses `Result<T>` for operations that can fail
   ```rust
   let result = a.add(&b);
   match result {
       Ok(c) => { /* use c */ },
       Err(e) => { /* handle error */ },
   }
   ```

2. **Type Inference**: The `array!` macro infers types, but you can be explicit
   ```rust
   let a = array![1, 2, 3];  // infers i32
   let b = array![1.0, 2.0, 3.0];  // infers f64
   ```

3. **Immutable Operations**: Operations return new arrays (Rust ownership model)
   ```rust
   let a = array![1, 2, 3];
   let b = a.add(&array![4, 5, 6]).unwrap();  // b is new
   // a is unchanged
   ```

4. **Performance Features**: Enable via Cargo features
   ```toml
   [dependencies]
   rust-numpy = { version = "0.1", features = ["simd", "rayon"] }
   ```

### Migration Checklist

- [ ] Import statements
  - Change `import numpy as np` to `use rust_numpy::*;`
  
- [ ] Array creation
  - Replace `np.array([1, 2])` with `array!([1, 2])`
  - Replace `np.zeros((3, 4))` with `Array::<f64>::zeros(vec![3, 4])`
  
- [ ] Mathematical operations
  - Replace `np.sin(a)` with `a.sin().unwrap()`
  - Replace `np.exp(a)` with `a.exp().unwrap()`
  - Replace `a.sum()` with `a.sum(None, false).unwrap()`
  
- [ ] Linear algebra
  - Replace `np.dot(a, b)` with `a.dot(&b).unwrap()`
  - Replace `np.linalg.solve(a, b)` with `a.solve(&b).unwrap()`
  - Replace `a.transpose()` with `a.transpose().unwrap()`
  
- [ ] Broadcasting
  - Replace broadcasting operations with `broadcast_to()` or advanced functions
  
- [ ] Error handling
  - Wrap NumPy operations in `match result { Ok(_) => ..., Err(e) => ... }`
  
- [ ] Indexing
  - Replace `a[0]` with `a.get(0).unwrap()` or use `array!` macro
  
- [ ] Performance optimization
  - Add `features = ["simd", "rayon"]` to Cargo.toml
  - Use parallel operations for large arrays

---

## Real-World Examples

### Example 16: Statistical Analysis

```rust
use rust_numpy::*;

fn main() {
    // Generate random data
    let data = Array::<f64>::from_vec(
        (0..1000).map(|_| rand::random::<f64>()).collect()
    );
    
    // Calculate statistics
    let mean = data.mean(None, false).unwrap();
    let std_dev = data.std(None, false, 1.0).unwrap();
    let variance = data.var(None, false, None).unwrap();
    
    println!("Mean: {:?}", mean.to_vec());
    println!("Std Dev: {:?}", std_dev.to_vec());
    println!("Variance: {:?}", variance.to_vec());
}
```

### Example 17: Signal Processing

```rust
use rust_numpy::*;
use std::f64::consts::PI;

fn main() {
    // Create time series
    let t = Array::<f64>::from_vec(
        (0..1000).map(|i| (i as f64) * 0.01).collect()
    );
    
    // Generate sine wave
    let signal = (2.0 * PI * t).sin().unwrap();
    
    println!("Signal:\n{:?}", signal.to_vec());
}
```

### Example 18: Image Processing (2D Array Operations)

```rust
use rust_numpy::*;

fn main() {
    // Create 2D array (grayscale image)
    let image = Array::<u8>::from_vec(vec![
        10, 20, 30, 40,
        50, 60, 70, 80, 90,
        100, 110, 120, 130, 140, 150,
        160, 170, 180, 190, 200, 210,
        220, 230, 240, 250,
    ]);
    
    // Reshape to image dimensions
    let image_2d = image.reshape(&[10, 24]).unwrap();
    println!("Image shape: {:?}", image_2d.shape());
    
    // Apply operation (e.g., brightness adjustment)
    let brighter = image_2d.add_assign(&10).unwrap();
    
    println!("Brighter image:\n{:?}", brighter.to_vec());
}
```

### Example 19: Machine Learning Operations

```rust
use rust_numpy::*;

fn main() {
    // Feature matrix (2D array)
    let features = Array::<f64>::from_vec(vec![
        [1.5, 2.1, 0.7],
        [2.8, 1.6, 0.9],
        [3.1, 1.2, 1.1],
    ]);
    
    // Weights
    let weights = array![0.2, 0.3, 0.5];
    
    // Weighted sum (simulating prediction)
    let prediction = features.dot(&weights).unwrap();
    
    println!("Features:\n{:?}", features.to_vec());
    println!("Weights:\n{:?}", weights.to_vec());
    println!("Prediction:\n{:?}", prediction.to_vec());
}
```

---

## Performance Benchmarks

### Example 20: Measuring Performance

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench bench_array_ops

# Run with output
cargo bench -- --nocapture --test-threads=4
```

### Interpreting Results

Benchmark output includes:
- **Mean**: Average execution time
- **Std Dev**: Time variance
- **Min/Max**: Fastest and slowest iterations
- **Throughput**: Operations per second

Example output:
```
array_ops/transpose
                        time:   [12.456 µs 12.543 µs 12.312 µs 12.234 µs]
                        change: [-1.234% +1.234% +1.234% -1.234%]
                        p 95%: 11.819 µs]
```

---

## Best Practices

### DO's

✅ **DO:**
- Reuse arrays with `clone()` instead of allocating new ones
- Use `as_slice()` for non-consuming access
- Enable SIMD and Rayon features for production builds
- Handle errors with `match result` instead of `.unwrap()` in production code
- Use iterator methods (`collect()`) instead of manual loops when possible

❌ **DON'T:**
- Call `.to_vec()` unnecessarily (creates allocation)
- Ignore errors with `.unwrap()` in production
- Clone large arrays in hot loops
- Use `get(0).unwrap()` for single elements when iterator is available

### Performance Tips

1. **Small Arrays (< 1000 elements)**: Overhead is minimal, optimizations less critical
2. **Medium Arrays (1K-100K elements)**: SIMD and memory layout matter most
3. **Large Arrays (> 100K elements)**: Parallel processing provides significant speedup

### Memory Management

```rust
// Efficient pattern: Reuse buffer
let mut buffer = Array::<f64>::zeros(vec![1000]);

// Process data in chunks
for chunk in &data.chunks(100) {
    let processed = process_chunk(chunk)?;
    buffer.extend_from_slice(&processed);
}

// Finalize
let result = buffer;
```

---

## Testing

### Running Examples

```bash
# Run all examples
cargo run --example

# Run specific example
cargo run --example example_4_trigonometric

# Test specific module
cargo test array

# Run conformance tests
cargo test conformance_tests
```

### Writing Your Own Examples

Start with simple patterns and add complexity:

1. Create array and access elements
2. Add mathematical operations
3. Add broadcasting or reshaping
4. Add error handling
5. Test with different array sizes

---

## Resources

- **API Reference**: See [API_REFERENCE.md](../API_REFERENCE.md)
- **Performance Guide**: See PERFORMANCE_ANALYSIS.md
- **NumPy Documentation**: https://numpy.org/doc/stable/
- **Rust Book**: https://doc.rust-lang.org/

---

## License

```
MIT License

Copyright (c) 2024 The rust-numpy Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
