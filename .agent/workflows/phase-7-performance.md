---
description: Phase 7 - Massive Performance Optimizations for Pocket TTS Candle Port
---

# Phase 7: Massive Performance Optimizations

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the baseline implementation.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

## Overview

This phase focuses on aggressive performance optimizations to make the Rust port significantly faster than the Python reference. The goal is to push CPU performance to its limits and explore every optimization opportunity.

---

## Test-Driven Development Philosophy

> [!NOTE]
> Performance work requires rigorous benchmarking:
> 1. **Establish baselines** before any optimization
> 2. **Measure after each change** to validate improvement
> 3. **Never sacrifice correctness** - all parity tests must still pass

Target: **80% test coverage** with comprehensive benchmarks.

---

## Development Environment

> [!NOTE]
> **Single-PC Development**: All optimizations will be tested on your local machine only. However, the code should be **structured for portability** using conditional compilation (`#[cfg(...)]`) so that platform-specific optimizations can be easily enabled on other systems in the future.

```rust
// Example: Structure code for portability
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn matmul_optimized(a: &[f32], b: &[f32], c: &mut [f32]) {
    matmul_avx2(a, b, c)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn matmul_optimized(a: &[f32], b: &[f32], c: &mut [f32]) {
    matmul_neon(a, b, c)
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
fn matmul_optimized(a: &[f32], b: &[f32], c: &mut [f32]) {
    matmul_naive(a, b, c)  // Fallback
}
```

---

## Performance Targets

| Metric | Baseline (Python) | Target (Rust) | Stretch Goal |
|--------|-------------------|---------------|--------------|
| First chunk latency | ~300ms | <100ms | <50ms |
| Real-time factor | ~0.8x | <0.3x | <0.1x |
| Memory usage | ~500MB | <200MB | <100MB |
| Throughput (samples/sec) | - | 2x Python | 5x Python |

---

## Tasks

### 1. Profiling & Baseline Establishment

**Create benchmark suite:**
```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_full_generation(c: &mut Criterion) {
    let model = TTSModel::load().unwrap();
    let state = model.get_voice_state(&load_wav("ref.wav")).unwrap();
    
    c.bench_function("generate_short", |b| {
        b.iter(|| model.generate("Hello world", &state))
    });
    
    c.bench_function("generate_medium", |b| {
        b.iter(|| model.generate("This is a medium length sentence for benchmarking.", &state))
    });
    
    c.bench_function("generate_long", |b| {
        let text = "word ".repeat(100);
        b.iter(|| model.generate(&text, &state))
    });
}

fn bench_first_chunk_latency(c: &mut Criterion) {
    let model = TTSModel::load().unwrap();
    let state = model.get_voice_state(&load_wav("ref.wav")).unwrap();
    
    c.bench_function("first_chunk", |b| {
        b.iter(|| {
            model.generate_stream("Hello", &state).unwrap().next()
        })
    });
}

criterion_group!(benches, bench_full_generation, bench_first_chunk_latency);
criterion_main!(benches);
```

**Profile hotspots:**
```bash
# Generate flamegraph
cargo flamegraph --bench full_benchmark

# Memory profiling
heaptrack cargo run --release --example generate -- --text "Test"
```

### 2. SIMD Vectorization

**Optimize compute-heavy operations:**

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated softmax
#[target_feature(enable = "avx2")]
unsafe fn softmax_avx2(input: &mut [f32]) {
    // AVX2 implementation
}

/// SIMD-accelerated layer normalization
#[target_feature(enable = "avx2")]
unsafe fn layer_norm_avx2(input: &mut [f32], weight: &[f32], bias: &[f32]) {
    // AVX2 implementation
}
```

**Key operations to vectorize:**
- Matrix multiplication (if not using BLAS)
- Softmax
- Layer normalization
- Activation functions (GELU, SiLU)
- Audio post-processing

**Write tests:**
```rust
#[test]
fn test_simd_softmax_correctness() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let simd_result = simd_softmax(&input);
    let naive_result = naive_softmax(&input);
    assert_tensors_close(&simd_result, &naive_result, 1e-6);
}
```

### 3. Memory Layout Optimization

**Optimize tensor memory layout:**

```rust
/// Use contiguous memory layout for cache efficiency
struct OptimizedTensor {
    data: Vec<f32>,      // Contiguous buffer
    shape: [usize; 4],   // Fixed-size shape (avoids heap allocation)
    strides: [usize; 4], // Pre-computed strides
}

/// Pre-allocated buffer pool to avoid allocations during inference
struct BufferPool {
    buffers: Vec<Vec<f32>>,
    available: Vec<usize>,
}
```

**Strategies:**
- Pre-allocate all intermediate tensors
- Use arena allocation for per-generation buffers
- Minimize tensor copies (use views where possible)
- Ensure cache-friendly access patterns

**Write tests:**
```rust
#[test]
fn test_buffer_pool_reuse() {
    let mut pool = BufferPool::new(10, 1024);
    let buf1 = pool.acquire();
    let id1 = buf1.id();
    pool.release(buf1);
    let buf2 = pool.acquire();
    assert_eq!(buf2.id(), id1); // Should reuse same buffer
}
```

### 4. Threading Optimization

**Optimize thread usage:**

```rust
/// Use rayon for parallel operations
use rayon::prelude::*;

fn parallel_attention_heads(heads: &mut [AttentionHead], input: &Tensor) {
    heads.par_iter_mut().for_each(|head| {
        head.forward(input);
    });
}

/// Configure thread pool for optimal performance
fn configure_threading() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get_physical())
        .build_global()
        .unwrap();
}
```

**Experiments to run:**
- Single-threaded vs multi-threaded matmul
- Optimal thread count for different CPU architectures
- Thread affinity for NUMA systems

### 5. Kernel Fusion

**Fuse operations to reduce memory bandwidth:**

```rust
/// Fused attention + softmax + matmul
fn fused_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    // Single pass instead of 3 separate operations
    // Reduces memory reads/writes
}

/// Fused GELU
fn fused_gelu(x: &Tensor) -> Tensor {
    // x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Single pass instead of multiple operations
}

/// Fused LayerNorm + Linear
fn fused_layernorm_linear(x: &Tensor, ln_weight: &Tensor, ln_bias: &Tensor, 
                          linear_weight: &Tensor, linear_bias: &Tensor) -> Tensor {
    // Combine two operations
}
```

### 6. Weight Preprocessing

**Optimize weight layout for inference:**

```rust
/// Preprocess weights for optimal memory access
fn preprocess_weights(weights: &mut ModelWeights) {
    // Transpose linear weights for better cache locality
    for layer in &mut weights.layers {
        layer.linear.weight = layer.linear.weight.t().contiguous();
    }
    
    // Pack weights for SIMD
    // [out_features, in_features] -> [out_features/8, in_features, 8]
}

/// Save preprocessed weights
fn save_optimized_weights(weights: &ModelWeights, path: &str) {
    // Save in optimized format for fast loading
}
```

### 7. Streaming Pipeline Optimization

**Optimize the streaming path specifically:**

```rust
/// Double-buffered audio output
struct StreamingPipeline {
    model: TTSModel,
    buffer_a: AudioBuffer,
    buffer_b: AudioBuffer,
    active_buffer: AtomicBool,
}

impl StreamingPipeline {
    /// Generate next chunk while previous chunk is being consumed
    fn next_chunk_pipelined(&mut self) -> &AudioBuffer {
        // Start generating into inactive buffer
        // Return active buffer immediately
    }
}
```

**Optimizations:**
- Overlap computation with audio playback
- Minimize state update overhead
- Pre-allocate streaming buffers

### 8. Platform-Specific Optimizations

**x86_64 (Intel/AMD):**
```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn matmul_avx512(a: &[f32], b: &[f32], c: &mut [f32]) {
    // AVX-512 implementation
}
```

**ARM (Apple Silicon, Raspberry Pi):**
```rust
#[cfg(target_arch = "aarch64")]
fn matmul_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
    // NEON implementation
}
```

**Consider using:**
- Intel MKL via `intel-mkl-src`
- OpenBLAS via `openblas-src`
- Apple Accelerate (macOS)

---

## Voice Cloning Test File

Use `ref.wav` in the project root for all performance benchmarks.

---

## Benchmarking Commands

### Internal Rust Benchmarks
```bash
# Run full Criterion benchmark suite
cargo bench

# Profile with flamegraph (requires cargo-flamegraph)
cargo flamegraph --bench full_benchmark -- --bench

# Memory profiling (Linux: heaptrack, Windows: use Visual Studio profiler)
heaptrack cargo run --release --example generate -- --text "Test"
```

### Python vs Rust Comparison

> [!IMPORTANT]
> **This is the key metric.** Use `hyperfine` to compare Python and Rust performance directly on your machine.

```bash
# Install hyperfine (if not installed)
# Windows: winget install sharkdp.hyperfine
# Linux: cargo install hyperfine

# Short text comparison
hyperfine --warmup 2 --runs 5 \
  'uv run pocket-tts generate --text "Hello world" --output NUL' \
  'cargo run --release -p pocket-tts-cli -- generate --text "Hello world" --output NUL' \
  --export-markdown benchmark_short.md

# Medium text comparison
hyperfine --warmup 2 --runs 5 \
  'uv run pocket-tts generate --text "This is a medium length sentence for realistic benchmarking of the TTS system." --output NUL' \
  'cargo run --release -p pocket-tts-cli -- generate --text "This is a medium length sentence for realistic benchmarking of the TTS system." --output NUL' \
  --export-markdown benchmark_medium.md

# Long text comparison
hyperfine --warmup 1 --runs 3 \
  'uv run pocket-tts generate --text "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. We are testing longer form text to speech synthesis." --output NUL' \
  'cargo run --release -p pocket-tts-cli -- generate --text "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. We are testing longer form text to speech synthesis." --output NUL' \
  --export-markdown benchmark_long.md
```

### Expected Output Format

After running `hyperfine`, you'll get output like:

```
Benchmark 1: uv run pocket-tts generate ...
  Time (mean ± σ):      2.341 s ±  0.087 s
  
Benchmark 2: cargo run --release ...
  Time (mean ± σ):      0.892 s ±  0.034 s

Summary
  'cargo run --release ...' ran 2.63 ± 0.13 times faster than 'uv run pocket-tts ...'
```

### First Chunk Latency Test

```bash
# Measure time to first audio byte (streaming)
# Create a small script that exits after first chunk
cargo run --release -p pocket-tts-cli -- stream --text "Test" --measure-first-chunk
```

### Automated Benchmark Script

Create `candle/scripts/benchmark.ps1` (Windows):
```powershell
# Run full Python vs Rust benchmark suite
$texts = @(
    "Hello world",
    "This is a medium length sentence for benchmarking.",
    "The quick brown fox jumps over the lazy dog. " * 3
)

foreach ($i in 0..($texts.Length - 1)) {
    $text = $texts[$i]
    Write-Host "`n=== Benchmark $($i + 1): $($text.Length) chars ===" -ForegroundColor Cyan
    
    hyperfine --warmup 2 --runs 5 `
        "uv run pocket-tts generate --text '$text' --output NUL" `
        "cargo run --release -p pocket-tts-cli -- generate --text '$text' --output NUL"
}
```

---

## Verification Checklist

Before completing this phase:

- [ ] All parity tests still pass (correctness preserved)
- [ ] >2x speedup over Python baseline achieved
- [ ] First chunk latency <100ms
- [ ] Memory usage <200MB
- [ ] All benchmarks documented
- [ ] Optimization decisions documented with data

---

## Optimization Log Template

Document each optimization with:

```markdown
## Optimization: [Name]

**Before:** [Metric]
**After:** [Metric]  
**Improvement:** [X%]

**What changed:**
- [Description]

**Tradeoffs:**
- [Any downsides]

**Code:**
```rust
// Key code changes
```
```

---

## Definition of Done

- [ ] >2x speedup achieved over Python
- [ ] First chunk latency <100ms
- [ ] Memory usage <200MB
- [ ] All parity tests pass
- [ ] All benchmarks pass
- [ ] 80% code coverage maintained
- [ ] Optimization log complete with data
- [ ] Code documented with rustdoc comments
- [ ] Platform-specific optimizations for x86_64 and ARM

---

## Stretch Goals

- [ ] >5x speedup over Python
- [ ] First chunk latency <50ms
- [ ] Memory usage <100MB
- [ ] AVX-512 support for newest Intel CPUs
- [ ] Apple Accelerate support for M1/M2/M3
- [ ] Portable SIMD using `std::simd` (nightly)
