---
description: Phase 6 - Advanced Features (WASM, Quantization, GPU) for Pocket TTS Candle Port
---

# Phase 6: Advanced Features & Optimizations

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the project goals.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

## Overview

This phase implements advanced features: WebAssembly support, int8 quantization, and GPU execution paths.

---

## Test-Driven Development Philosophy

> [!NOTE]
> Advanced features require careful testing:
> 1. **WASM** - Test in browser environment
> 2. **Quantization** - Verify acceptable quality degradation
> 3. **GPU** - Test on multiple hardware configurations

Target: **80% test coverage** with quality benchmarks.

---

## Tasks

### 1. Implement WebAssembly (WASM) Support

**Write tests first:**
```rust
// wasm-specific tests using wasm-bindgen-test
#[wasm_bindgen_test]
fn test_wasm_model_load() {
    let model = TTSModel::load_wasm()?;
    assert!(model.is_ready());
}

#[wasm_bindgen_test]
fn test_wasm_generate() {
    let model = TTSModel::load_wasm()?;
    let audio = model.generate("Hello")?;
    assert!(!audio.is_empty());
}
```

**Implementation notes:**
- Use `candle-core` WASM target: `wasm32-unknown-unknown`
- Create `wasm` feature flag
- Expose simplified API for browser usage
- Consider weight file size (may need quantization)

**Build command:**
```bash
wasm-pack build --target web --features wasm
```

### 2. Implement Silence/Pause Handling

**Write tests first:**
```rust
#[test]
fn test_pause_syntax() {
    let text = "Hello... world";
    let audio = model.generate(text, &state)?;
    // Should contain pause in middle
}

#[test]
fn test_explicit_pause_marker() {
    let text = "Hello [pause:500ms] world";
    let audio = model.generate(text, &state)?;
    // Should have 500ms pause
}
```

**Implementation notes:**
- Detect `...` and `,` for natural pauses
- Support explicit pause markers: `[pause:Xms]`
- Insert silence samples at appropriate points

### 3. Implement int8 Quantization

**Write tests first:**
```rust
#[test]
fn test_quantized_model_load() {
    let model = TTSModel::load_quantized("model_int8.safetensors")?;
    assert!(model.is_quantized());
}

#[test]
fn test_quantized_output_quality() {
    let f32_audio = f32_model.generate("Test", &state)?;
    let int8_audio = int8_model.generate("Test", &state)?;
    
    // Quality should be acceptable (within perceptual threshold)
    let snr = signal_to_noise_ratio(&f32_audio, &int8_audio);
    assert!(snr > 30.0); // 30dB SNR minimum
}

#[test]
fn test_quantized_speedup() {
    // int8 should be faster than f32
}
```

**Implementation notes:**
- Use Candle's quantization support if available
- Create quantization script to convert weights
- Target layers that benefit most from quantization
- Measure quality degradation

### 4. Investigate and Optimize GPU Execution

> [!WARNING]
> The Python implementation notes that GPU does not provide speedup for this small model. Investigate before implementing.

**Write tests first:**
```rust
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_model_load() {
    let device = Device::cuda_if_available(0)?;
    let model = TTSModel::load_on_device(device)?;
    assert!(model.device().is_cuda());
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_generate() {
    let model = TTSModel::load_cuda()?;
    let audio = model.generate("Test", &state)?;
    // Verify correctness
}
```

**Implementation notes:**
- Support CUDA via `candle-cuda` feature
- Profile to determine if GPU is beneficial
- May need batching for GPU efficiency
- Consider Metal support for macOS

---

## Voice Cloning Test File

Use `ref.wav` in the project root for all advanced feature tests.

---

## Verification Checklist

Before completing this phase:

- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` reports no warnings
- [ ] Test coverage ≥80%
- [ ] WASM build succeeds and works in browser
- [ ] Quantized model produces acceptable quality
- [ ] GPU execution verified (if applicable)

---

## Manual Verification Steps

1. **Test WASM in browser:**
```bash
wasm-pack build --target web --features wasm
# Open demo HTML page
```

2. **Test quantization:**
```bash
# Create quantized weights
cargo run --example quantize -- --input model.safetensors --output model_int8.safetensors

# Test quantized model
cargo run --example generate -- --text "Test" --model model_int8.safetensors
```

3. **Test GPU (if available):**
```bash
cargo run --release --features cuda -- generate --text "GPU test"
```

---

## Definition of Done

- [ ] WASM build working in browser
- [ ] Pause/silence handling implemented
- [ ] int8 quantization working with acceptable quality
- [ ] GPU execution investigated and documented
- [ ] All tests pass
- [ ] 80% code coverage achieved
- [ ] Performance documented for each feature
- [ ] Code documented with rustdoc comments

---

## Project Completion Criteria

After all phases are complete, verify:

- [ ] 100% feature parity with Python implementation
- [ ] Numerical parity verified
- [ ] All tests pass
- [ ] 80%+ code coverage overall
- [ ] CLI works: `pocket-tts generate` and `pocket-tts serve`
- [ ] API server works with OpenAI compatibility
- [ ] Documentation complete (README, rustdoc)
- [ ] Voice cloning with `ref.wav` produces correct output
