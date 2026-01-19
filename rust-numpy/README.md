# Rust NumPy

A 100% pure-Rust implementation of Python's NumPy library with full API parity.

## Status: 🚧 Work in Progress

This is an ambitious project to recreate NumPy's entire functionality in pure Rust. Currently in early development phase with core architecture established.

## Architecture

- **Core Array System**: Multi-dimensional arrays with full NumPy memory model
- **Dtype System**: Complete NumPy dtype compatibility  
- **Ufunc System**: Universal functions with broadcasting
- **Multi-Backend**: Pluggable backends (CPU, CUDA, Metal, WASM)
- **C-API Compatible**: Drop-in replacement for NumPy C extensions

## Current Progress

### ✅ Completed
- [x] Project architecture and build system
- [x] Core array data structures
- [x] Complete dtype system  
- [x] Memory management with SIMD alignment
- [x] Stride calculations and broadcasting logic
- [x] Error handling matching NumPy exceptions
- [x] Universal function framework
- [x] Constants and mathematical values

### 🚧 In Progress
- [ ] Array indexing and slicing operations
- [ ] Ufunc execution engine
- [ ] Broadcasting implementation
- [ ] Linear algebra operations
- [ ] FFT operations
- [ ] Random number generation

### 📋 Planned
- [ ] CUDA backend
- [ ] Metal backend
- [ ] WASM backend
- [ ] Performance benchmarks
- [ ] NumPy conformance tests

## Building

```bash
# Build library only
cargo build

# Build with CUDA support
cargo build --features cuda

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## Usage

```rust
use rust_numpy::*;

// Create arrays
let a = array![1, 2, 3];
let b = array2![[1, 2], [3, 4]];

// Array operations
let c = a.reshape(vec![3]).unwrap();
let d = b.t();

// Mathematical constants
let pi = constants::PI;
let eps = constants::float::EPSILON;
```

## Development Roadmap

### Phase 1: Core Foundation (Current)
- [x] Array structure and memory management
- [x] Dtype system with all NumPy types
- [x] Broadcasting and stride calculations
- [ ] Array indexing, slicing, and views
- [ ] Basic ufunc operations

### Phase 2: Mathematical Operations
- [ ] Complete ufunc system (all mathematical functions)
- [ ] Reduction operations (sum, mean, std, etc.)
- [ ] Comparison and logical operations
- [ ] Linear algebra (matrix operations, decompositions)

### Phase 3: Advanced Modules  
- [ ] FFT operations
- [ ] Random number generation
- [ ] String and datetime operations
- [ ] Masked arrays

### Phase 4: Performance & Integration
- [ ] SIMD optimizations
- [ ] GPU backends (CUDA/Metal)
- [ ] C-API compatibility

### Phase 5: Ecosystem
- [ ] Performance benchmarks
- [ ] NumPy conformance suite
- [ ] Documentation and examples
- [ ] Migration tools

## Requirements

### For Users
- Rust 1.70+

### For Development
- Rust toolchain
- CUDA Toolkit (for CUDA backend)
- Xcode (for Metal backend on macOS)

## Contributing

This is a massive undertaking that requires:
1. **Core contributors**: Array operations, memory management, ufuncs
2. **Mathematical experts**: Linear algebra, FFT, optimization
3. **Performance engineers**: SIMD, GPU programming, benchmarks

## Motivation

The goal is to provide:
- **Performance**: Rust's safety + speed for scientific computing
- **Integration**: Drop-in NumPy replacement for existing code
- **Portability**: Multi-backend support for different platforms
- **Safety**: Memory safety without NumPy's complexity
- **Extensibility**: Modern Rust patterns for scientific computing

## License

BSD-3-Clause License (same as NumPy)

---

**Note**: This is currently an experimental project. For production use, please use the original NumPy library.