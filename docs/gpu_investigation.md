# GPU Investigation Report

## Summary

**Recommendation: GPU is NOT beneficial for Pocket TTS**

Per the Python implementation's [AGENTS.md](file:///d:/pocket-tts-candle/AGENTS.md):
> "GPU does not provide speedup for this small model."

## Analysis

### Model Size
- Total parameters: ~90MB
- This is a relatively small model compared to LLMs

### Why GPU Doesn't Help

1. **Memory Transfer Overhead**: The time to copy tensors to/from GPU exceeds computational savings
2. **Small Batch Size**: Pocket TTS operates with batch size = 1 (no batching)
3. **CPU SIMD Efficiency**: Modern CPUs with AVX2/AVX-512 handle this workload efficiently
4. **Streaming Architecture**: Frame-by-frame generation (12.5 Hz) creates frequent small operations

### Profiling Notes

The Python implementation explicitly sets:
```python
torch.set_num_threads(1)  # in tts_model.py
```

This suggests that even multi-threaded CPU doesn't provide significant benefit for this workload.

## Feature Flags

The Rust port includes optional `cuda` feature for future investigation:
```toml
[features]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
```

To build with CUDA support:
```bash
cargo build --release --features cuda
```

## Recommendation

- **Default**: CPU-only builds
- **CUDA**: Only enable if users specifically request it and have CUDA hardware
- **Metal**: Consider for macOS users in future if there's demand

## References

- [AGENTS.md](file:///d:/pocket-tts-candle/AGENTS.md) - Line 108: "Device: Defaults to CPU. GPU does not provide speedup for this small model."
- [GitHub Issue #7](https://github.com/kyutai-labs/pocket-tts/issues/7) - Focuses on int8 quantization for CPU, not GPU
