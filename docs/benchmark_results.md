# Benchmark Results

Benchmarks run on system with:
- **GPU 1:** NVIDIA RTX PRO 4500 Blackwell (SM 12.0)
- **GPU 2:** NVIDIA GeForce RTX 3090 (SM 8.6)
- **PyTorch:** 2.11.0+cu128
- **torchao:** 0.17.0

## Results Matrix

| Config | RTF | Latency | TTFB | VRAM |
|--------|-----|---------|------|------|
| CPU | 4.4x | 890ms | 59ms | - |
| CPU + quantize | 3.0x | 1286ms | 184ms | - |
| **GPU** | **13.3x** | **303ms** | **24ms** | 564 MB |
| GPU (no flash) | 12.7x | 318ms | 24ms | 564 MB |
| GPU + compile | 13.2x | 295ms | 23ms | 564 MB |
| GPU + quantize | 3.5x | 1118ms | 63ms | 363 MB |

- **RTF** = Real-time factor (higher is better, >1 means faster than real-time)
- **Latency** = Total time to generate audio for test texts
- **TTFB** = Time to first byte (first audio chunk from streaming)
- **VRAM** = Peak GPU memory usage

## Key Findings

1. **GPU is 3x faster than CPU** (13.3x vs 4.4x RTF)
2. **GPU TTFB is 2.5x faster** (24ms vs 59ms)
3. **Flash Attention provides ~5% improvement** (13.3x vs 12.7x RTF)
4. **torch.compile() provides minimal benefit** (13.2x vs 13.3x RTF) - model may already be memory-bound
5. **INT8 quantization is currently broken** - torchao CUTLASS kernels compiled for SM90 (Ada/Hopper), incompatible with Blackwell (SM 12.0) and Ampere (SM 8.6)
6. **Quantization saves ~35% VRAM** (363 MB vs 564 MB) but kills performance without optimized kernels

## Upstream Comparison

The upstream README claimed:
> "We tried running this TTS model on the GPU but did not observe a speedup compared to CPU execution"

This was likely due to:
1. Model state not being moved to GPU (fixed in benchmark script with `move_state_to_device()`)
2. Default install using CPU-only PyTorch (no Flash Attention)

With proper GPU setup, we see **3x speedup** over CPU.

## Test Texts

Benchmarks used three texts of varying length:
1. "Hello world." (12 chars, 2 words)
2. "The quick brown fox jumps over the lazy dog." (44 chars, 9 words)
3. "Speech synthesis is the artificial production of human speech..." (134 chars, 21 words)

Each text was run 5 times with 2 warmup runs.

## Next Steps

- [x] Fix torch.compile() support (state key `_orig_mod.` prefix issue)
- [ ] Test with torchao built for SM86/SM120
- [ ] Investigate CUDA graphs for lower TTFB
