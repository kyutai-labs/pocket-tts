# Rust NumPy Ecosystem Research Report

## Executive Summary

This document analyzes key Rust ecosystem crates relevant for implementing NumPy-like functionality. All researched crates use permissive licenses (MIT/Apache-2.0) compatible with commercial/open-source projects.

## Crate Analysis

### 1. ndarray - Core Array Library

**License:** MIT OR Apache-2.0  
**Version:** 0.17.2  
**Downloads:** 13M+ all time, 88 recent versions

**Key Features:**
- N-dimensional arrays with slicing and views
- Efficient memory layout handling (C/Fortran order)
- Optional BLAS support for matrix multiplication
- Parallel iteration via Rayon
- Serde serialization support
- Comprehensive math operations

**Recommendation:** **USE DIRECTLY** - This is the foundational array library for any NumPy implementation. Provides core data structures and operations.

### 2. nalgebra - Linear Algebra

**License:** Apache-2.0  
**Version:** 0.34.1  
**Downloads:** 49M+ all time, 2.1M recent versions

**Key Features:**
- Static and dynamic matrices/vectors
- Geometric transformations (rotation, translation, projection)
- Matrix decompositions (SVD, QR, LU, Cholesky)
- Sparse matrix support
- Extensive type aliases for common dimensions
- Computer graphics focused

**Recommendation:** **USE DIRECTLY** for advanced linear algebra operations. Complements ndarray well.

### 3. rustfft - FFT Implementation

**License:** MIT OR Apache-2.0  
**Version:** 6.4.1  
**Downloads:** 12.5M+ all time

**Key Features:**
- Pure Rust implementation
- Mixed-radix algorithms
- Automatic SIMD optimization (AVX, SSE4.1, NEON)
- WASM SIMD support
- Runtime algorithm selection via FftPlanner

**Recommendation:** **USE DIRECTLY** - High performance, well-maintained FFT implementation.

### 4. rand - Random Number Generation

**License:** MIT OR Apache-2.0  
**Version:** 0.9.2  
**Downloads:** 825M+ all time (extremely popular)

**Key Features:**
- Multiple RNG algorithms (ChaCha, PCG, Xoshiro)
- Distribution sampling
- `no_std` support
- SIMD acceleration
- Serde support

**Recommendation:** **USE DIRECTLY** - Standard for randomness in Rust ecosystem.

### 5. BLAS/LAPACK Bindings

**BLAS License:** Apache-2.0 OR MIT  
**LAPACK License:** Varies (typically BSD-like)

**BLAS Wrapper (blas crate):**
- Low-level Fortran bindings
- Raw BLAS operations
- Multiple backend support (OpenBLAS, Intel MKL)

**LAPACK Bindings (lapack-sys):**
- Low-level Fortran bindings
- System or bundled Netlib
- CBLAS and LAPACKE support

**Recommendation:** **REFERENCE FOR IDEAS** - Consider implementing higher-level interfaces rather than direct bindings.

### 6. CUDA/Metal GPU Bindings

**CUDA (rustacuda):**
**License:** Mixed (check per-project)  
**Status:** Less maintained (last update 2021)

- CUDA Driver API wrapper
- High-level interface
- GPU memory management

**Metal (metal crate):**
**License:** MIT OR Apache-2.0  
**Status:** ⚠️ DEPRECATED - Use objc2-metal instead

- Apple Metal bindings
- Deprecated in favor of objc2 ecosystem

**Modern Alternative:**
**Rust GPU/WGPU approach:**
- Cross-platform GPU compute
- Compiles Rust to SPIR-V/NVVM IR
- Runs on CUDA, Metal, Vulkan, DirectX, WebGPU

**Recommendation:** **USE WGPU/RUST-GPU** for new development. Avoid deprecated metal crate.

### 7. SIMD Crates

**packed_simd:**
**License:** MIT OR Apache-2.0  
**Status:** ⚠️ DEPRECATED - Superseded by std::simd

- Portable SIMD vectors
- Nightly-only
- Performance-focused

**wide:**
**License:** Zlib OR Apache-2.0 OR MIT  
**Status:** Active

- SIMD-compatible types
- Automatic SIMD detection
- Fallback to scalar

**std::simd:**
**License:** Standard library  
**Status:** **RECOMMENDED** - Stable since Rust 1.65

- Official SIMD support
- Cross-platform
- Future-proof

**Recommendation:** **USE STD::SIMD** for new code. Use wide as fallback for older Rust.

### 8. criterion - Performance Benchmarking

**License:** Apache-2.0 OR MIT  
**Version:** 0.8.1  
**Downloads:** 132.5M+ all time, 20.1M recent

**Key Features:**
- Statistical analysis
- Performance regression detection
- Gnuplot chart generation
- Stable Rust compatible
- HTML reports

**Recommendation:** **USE DIRECTLY** - Industry standard for Rust benchmarking.

## License Compatibility Analysis

All major crates use permissive licenses:

✅ **MIT License:** ndarray, rand, criterion, wide, packed_simd  
✅ **Apache-2.0:** nalgebra, rustfft, blas, criterion  
✅ **Dual MIT/Apache:** ndarray, rustfft, rand, blas, criterion  
✅ **Zlib/Apache/MIT:** wide

**No GPL conflicts** - All crates are compatible with commercial use and can be combined in a single project.

## Final Recommendations

### Direct Dependencies (USE)
1. **ndarray** - Core array functionality
2. **nalgebra** - Advanced linear algebra  
3. **rustfft** - FFT operations
4. **rand** - Random number generation
5. **criterion** - Benchmarking
6. **std::simd** - SIMD operations (Rust 1.65+)

### Reference Only (STUDY)
1. **BLAS/LAPACK bindings** - Study for API design, implement Rust-native versions
2. **packed_simd** - Deprecated, study for transition to std::simd
3. **metal crate** - Deprecated, study objc2-metal instead

### GPU Strategy
1. **Preferred:** Use wgpu/rust-gpu for cross-platform compatibility
2. **Fallback:** Platform-specific bindings only if necessary
3. **Avoid:** Deprecated metal crate

## Implementation Priority

**Phase 1 - Foundation:**
- ndarray + nalgebra + rand + criterion

**Phase 2 - Specialized:**
- rustfft + std::simd

**Phase 3 - GPU (if needed):**
- wgpu/rust-gpu ecosystem

This provides a solid foundation for NumPy-like functionality while avoiding deprecated or problematic dependencies.