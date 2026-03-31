# Inference Speed Optimizations for Pocket-TTS

This document outlines potential optimizations to improve inference speed, ordered by ease of implementation and expected impact.

---

## Already Implemented

- **Int8 Dynamic Quantization** (`--quantize` flag) - ~27% speedup, ~48% memory reduction
- **KV Caching** for streaming generation
- **Multithreaded Generation** (FlowLM + Mimi decoder run in parallel)
- **scaled_dot_product_attention** (SDPA) - see note below

> **Note on Flash Attention:** The code uses PyTorch's `scaled_dot_product_attention`, which
> can dispatch to Flash Attention automatically. However, the default install uses **CPU-only
> PyTorch** (via `pytorch-cpu` index in pyproject.toml), so SDPA falls back to the standard
> math implementation. Flash Attention requires CUDA and is only activated when running on GPU.

---

## Optimization Candidates

### 1. GPU Acceleration

**Difficulty:** Easy (but requires reinstalling PyTorch)
**Expected Impact:** High (5-20x on CUDA)
**Status:** [ ] Not tested

The model defaults to CPU, and the default install uses CPU-only PyTorch. Moving to GPU would significantly speed up transformer operations and **enable Flash Attention** via SDPA.

**Setup:**
```bash
# Reinstall PyTorch with CUDA support (not from pytorch-cpu index)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Usage:**
```python
# CLI usage
pocket-tts generate --device cuda "Hello world"

# API usage
model = TTSModel.load_model()
model.to("cuda")
```

**Notes:**
- Requires CUDA-capable GPU and reinstalling PyTorch with CUDA
- Enables Flash Attention automatically via SDPA (PyTorch 2.0+)
- May need to also move Mimi decoder to GPU
- Memory usage will increase (~500MB+ VRAM)

---

### 2. torch.compile()

**Difficulty:** Easy
**Expected Impact:** Medium (20-40% speedup)
**Status:** [x] Working

PyTorch 2.0+ can compile models for faster execution.

```python
model = TTSModel.load_model()
model.flow_lm = torch.compile(model.flow_lm, mode="reduce-overhead")
```

**Notes:**
- First inference will be slow (compilation)
- `mode="reduce-overhead"` is best for small batches
- Requires CUDA for best results
- The benchmark script supports `--compile` flag for easy testing

---

### 3. Half Precision (BF16/FP16)

**Difficulty:** Easy
**Expected Impact:** Medium (1.5-2x speedup on GPU)
**Status:** [ ] Not tested

Reduces memory bandwidth requirements.

```python
model = TTSModel.load_model()
model.to(dtype=torch.bfloat16)
```

**Notes:**
- BF16 preferred over FP16 for stability
- Best combined with GPU acceleration
- May affect audio quality slightly

---

### 4. Quantize Flow Net

**Difficulty:** Easy
**Expected Impact:** Low-Medium
**Status:** [ ] Not tested

The flow_net MLP is not quantized by default. Add it to quantization groups.

```python
# In quantization.py, change RECOMMENDED_CONFIG:
RECOMMENDED_CONFIG = {"attention", "ffn", "flow_net"}
```

**Notes:**
- Already supported in code, just not enabled by default
- May affect generation quality

---

### 5. Optimize Thread Coordination

**Difficulty:** Medium
**Expected Impact:** Low
**Status:** [ ] Not tested

Current implementation in `main.py:97-113` uses blocking queue operations.

Potential improvements:
- Use `asyncio` instead of threads for the web server path
- Batch multiple latents before decoding
- Use a thread pool instead of spawning threads per request

---

### 6. ONNX Runtime Export

**Difficulty:** Medium-Hard
**Expected Impact:** High (2-5x speedup)
**Status:** [ ] Not tested

Export model to ONNX for optimized inference.

```python
# Would require new export script
torch.onnx.export(model.flow_lm, ...)
```

**Notes:**
- Requires handling stateful KV cache
- May lose some PyTorch-specific optimizations
- Good for production deployment

---

### 7. Speculative Decoding

**Difficulty:** Hard
**Expected Impact:** Medium-High (1.5-3x for autoregressive)
**Status:** [ ] Not tested

Use a smaller draft model to predict multiple tokens, then verify with the main model.

**Notes:**
- Requires training/obtaining a draft model
- Complex implementation
- Best for longer generations

---

### 8. INT4 Quantization (AWQ/GPTQ)

**Difficulty:** Medium
**Expected Impact:** Medium (further memory/speed gains over INT8)
**Status:** [ ] Not tested

More aggressive quantization using torchao.

```python
from torchao.quantization import int4_weight_only
quantize_(model.flow_lm, int4_weight_only())
```

**Notes:**
- Requires torchao with INT4 support
- Higher quality degradation risk than INT8
- Best for memory-constrained environments

---

### 9. Paged/Continuous KV Cache

**Difficulty:** Hard
**Expected Impact:** Low-Medium
**Status:** [ ] Not tested

Current KV cache pre-allocates with NaN padding. Paged attention could reduce memory pressure.

**Notes:**
- Complex implementation
- Mainly benefits long sequences
- Libraries like vLLM implement this

---

## Testing Protocol

For each optimization:

1. **Baseline measurement**
   ```bash
   time pocket-tts generate --text "Hello world, this is a test of the text to speech system."
   ```

2. **Apply optimization**

3. **Measure performance**
   - Wall clock time
   - Real-time factor (RTF) from logs
   - Memory usage (`nvidia-smi` for GPU)

4. **Verify quality**
   - Listen to output audio
   - Compare WER if automated testing available

5. **Document results** in the table below

---

## Results

| Optimization | RTF Before | RTF After | Memory | Quality Impact | Notes |
|-------------|-----------|-----------|--------|----------------|-------|
| Baseline (CPU) |        |           |        | N/A            |       |
| INT8 quantize |         |           |        |                | `--quantize` flag |
| GPU (CUDA)  |           |           |        |                | Enables Flash Attention |
| torch.compile |         |           |        |                |       |
| BF16        |           |           |        |                |       |
| Quantize flow_net |     |           |        |                |       |
| ONNX        |           |           |        |                |       |

---

## References

- [PyTorch torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [torchao quantization](https://github.com/pytorch/ao)
- [ONNX Runtime](https://onnxruntime.ai/)
