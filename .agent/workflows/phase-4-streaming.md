---
description: Phase 4 - Streaming Audio Generation & CPU Performance for Pocket TTS Candle Port
---

# Phase 4: Streaming & Performance

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the streaming architecture.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

## Overview

This phase implements real-time streaming audio generation and optimizes CPU performance.

## Key Concepts from Python

**Reference:** `pocket_tts/models/tts_model.py::generate_audio_stream`

- Frame rate: 12.5 Hz (80ms per frame)
- Streaming via `StatefulModule` pattern (KV cache, internal state)
- EOS detection via EOS head prediction
- `frames_after_eos` parameter for clean audio endings

---

## Test-Driven Development Philosophy

> [!NOTE]
> For streaming, focus on:
> 1. **Correctness** - Streaming output matches non-streaming batched output
> 2. **Latency** - First chunk under 200ms
> 3. **Memory** - No unbounded memory growth during long generations

Target: **80% test coverage** with performance benchmarks.

---

## Tasks

### 1. Implement Streaming Audio Generator

**Write tests first:**
```rust
#[test]
fn test_streaming_matches_batch() {
    // Streaming output == non-streaming output
    let batch_audio = model.generate("Hello world", &state)?;
    let stream_audio: Vec<_> = model
        .generate_stream("Hello world", &state)?
        .collect();
    let stream_concat = concat_chunks(&stream_audio);
    assert_tensors_close(&batch_audio, &stream_concat, 1e-6);
}

#[test]
fn test_streaming_yields_chunks() {
    // Verify we get multiple chunks, not one big blob
    let chunks: Vec<_> = model
        .generate_stream("This is a longer sentence.", &state)?
        .collect();
    assert!(chunks.len() > 1);
}

#[test]
fn test_eos_detection() {
    // Verify generation stops after EOS
}

#[test]
fn test_streaming_state_reset() {
    // After generation, state should be clean for next call
}
```

**Implementation notes:**
- Return `impl Iterator<Item = AudioChunk>` or similar
- Each chunk: 80ms of audio (at 12.5 Hz frame rate)
- Maintain streaming state across chunks
- Proper EOS handling with `frames_after_eos`

### 2. CPU Performance Optimization

**Write benchmark tests:**
```rust
#[bench]
fn bench_first_chunk_latency() {
    // Target: <200ms to first audio chunk
}

#[bench]
fn bench_throughput() {
    // Measure samples/second
}

#[bench]
fn bench_memory_usage() {
    // Ensure no memory leaks during streaming
}
```

**Optimization strategies:**
- Set thread count: `std::env::set_var("RAYON_NUM_THREADS", "1")`
- Use `f16` where hardware supports it
- Optimize tensor operations (minimize allocations)
- Profile with `cargo flamegraph`

**Key optimizations from Python:**
```python
# Reference: pocket_tts/models/tts_model.py
torch.set_num_threads(1)  # Optimal for small model
```

### 3. Infinite Text Handling

**Write tests first:**
```rust
#[test]
fn test_long_text_chunking() {
    let long_text = "word ".repeat(1000);
    let chunks: Vec<_> = model.generate_stream(&long_text, &state)?.collect();
    // Should handle without OOM
    assert!(!chunks.is_empty());
}

#[test]
fn test_text_segmentation() {
    // Verify text is split at sentence/phrase boundaries
}
```

**Implementation notes:**
- Split long text at natural breakpoints
- Reset streaming state between segments
- Seamless audio concatenation

---

## Voice Cloning Test File

Use `ref.wav` in the project root for streaming tests.

---

## Performance Targets

| Metric | Target |
|--------|--------|
| First chunk latency | <200ms |
| Real-time factor | <1.0 (faster than real-time) |
| Memory (streaming) | <500MB peak |
| Memory (long text) | Bounded (no growth per chunk) |

---

## Verification Checklist

Before completing this phase:

- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` reports no warnings
- [ ] Test coverage ≥80%
- [ ] Streaming output matches batch output
- [ ] First chunk latency <200ms
- [ ] No memory leaks during long streaming

---

## Manual Verification Steps

1. **Test streaming:**
```bash
cargo run --release --example stream -- --text "Hello, this is streaming." --voice ref.wav
# Audio should start playing almost immediately
```

2. **Test long text:**
```bash
cargo run --release --example stream -- --text "$(cat long_article.txt)" --voice ref.wav
# Should complete without OOM
```

3. **Profile performance:**
```bash
cargo flamegraph --example stream -- --text "Test" --voice ref.wav
```

---

## Definition of Done

- [ ] Streaming generation working
- [ ] Streaming matches batch output
- [ ] All tests pass
- [ ] 80% code coverage achieved
- [ ] First chunk latency <200ms
- [ ] Memory bounded for long text
- [ ] Real-time factor <1.0
- [ ] Code documented with rustdoc comments
