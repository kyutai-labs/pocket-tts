# Quantisize Plan (Int8 Quantization Options)

This plan is validated against PyTorch documentation and blogs (see Sources at the end).

## Goals
- Reduce memory footprint and (if possible) download size.
- Improve CPU inference speed without hurting audio quality.
- Keep changes scoped and easy to revert if quality regresses.

## Where The Time And Memory Go (Current Structure)
- FlowLM (transformer + MLP) is heavy on `nn.Linear` and should benefit most from int8.
- Mimi uses many Conv1d/ConvTranspose1d layers (SEANet); int8 here is harder and higher risk.
- Streaming attention uses cached KV tensors that must remain float for SDPA ops.

## How We Will Evaluate
- Use `scripts/benchmark.py` and `scripts/benchmark.bash` with a fixed prompt set.
- Compare median latency and RTF across runs.
- Save audio outputs in `audios/` for listening comparisons.
- Run both "default" and "int8" runs with different `--run-label` values.

Example run labels:
- `RUN_LABEL=default bash scripts/benchmark.bash`
- `RUN_LABEL=int8 bash scripts/benchmark.bash`

Audio outputs will be written to `audios/` with run label and hash in the filename.

## Approach A: Dynamic Int8 For Linear Layers (Low Effort)
What it is:
- Apply dynamic (weight-only) quantization to `nn.Linear` layers only (FlowLM + Mimi transformer).
- Convs and LayerNorm/Embedding remain float.

Pros:
- Minimal code changes.
- Supported for Linear/Recurrent layers in PyTorch dynamic quantization; common fit for transformer-style models.
- Memory usage drops for Linear weights.

Cons:
- No download size reduction (weights are quantized after loading). (Inference)
- Speedup depends on CPU backend; may be modest on some CPUs. (Inference)
- Must fix dtype assumptions for KV cache in streaming attention.

Changes Needed:
- Add a quantization toggle in `pocket_tts/models/tts_model.py` to run
  `torch.ao.quantization.quantize_dynamic(...)` on the model after loading.
- Update `pocket_tts/modules/transformer.py` so KV cache dtype stays float
  (avoid `self.in_proj.weight.dtype` once `in_proj` becomes quantized).
- Optional: skip quantizing small or sensitive layers if quality degrades
  (e.g., EOS head or certain MLP layers).

## Approach B: Weight-Only Int8 With Pre-Quantized Weights (Medium Effort)
What it is:
- Offline quantization of Linear weights into int8 + scale/zero-point.
- Store quantized weights in new safetensors files and load them directly.

Pros:
- Smaller downloads.
- Memory reduction on load.
- Can keep compute in int8 or dequantize per layer.

Cons:
- More engineering: custom loading and possible custom Linear modules.
- Must manage scale/zero-point and accuracy per layer.
- Speed gain depends on kernel availability.

Changes Needed:
- Add a conversion script (e.g., `scripts/quantize_weights.py`) to produce
  quantized safetensors and a new config variant.
- Add a quantized Linear module (e.g., `pocket_tts/modules/quant_linear.py`)
  that performs int8 weight-only compute or dequantizes efficiently.
- Update `pocket_tts/utils/weights_loading.py` and
  `pocket_tts/models/tts_model.py` to load quantized weights when configured.

## Approach C: Static Int8 Quantization For Convs (High Effort, High Risk)
What it is:
- Quantize SEANet Conv1d/ConvTranspose1d layers with calibration.

Pros:
- Potentially larger speedups if convs dominate runtime.
- Memory reduction across a larger portion of the model.

Cons:
- Requires calibration data and more complex pipelines.
- Audio quality is at higher risk.
- Requires replacing or refactoring streaming conv layers to quantized variants.

Changes Needed:
- Calibration pipeline and representative audio data.
- Quantized versions of `StreamingConv1d` and `StreamingConvTranspose1d` in
  `pocket_tts/modules/conv.py`.
- Extra tests to ensure streaming behavior matches float.

## Recommended Path
1. Start with Approach A (dynamic int8 for Linear layers).
   - Low risk, fast to implement, easy to evaluate using existing benchmark.
   - Fix KV cache dtype and check for any accuracy regressions by listening.
2. If speedup is too small, move to Approach B to reduce downloads and memory.
3. Only consider Approach C if we still need more speed and are willing to
   invest in calibration and quality validation.

## Accuracy And Regression Checks
- Listen to `audios/` outputs for common prompts.
- Ensure EOS behavior is stable (no early/late cutoffs).
- Compare measured RTF and median latency across runs.
- Optional: add an automated waveform similarity metric if needed.

## Validated Quantization Notes (PyTorch)
- Dynamic quantization pre-quantizes weights, while activations are quantized on-the-fly; it is a one-line API in PyTorch and currently supports Linear and Recurrent layers (LSTM/GRU/RNN). 
- Static post-training quantization pre-calibrates activation ranges using representative data; activations stay quantized between ops and static quantization can be faster than dynamic by avoiding float↔int conversions. 
- On x86, PyTorch 2.0 introduced the “x86” quantization backend which combines FBGEMM and oneDNN; reported INT8 speedups vary by model and CPU. 


## Commands
COMPARE_RUN=20260115_203853 RUN_ID=quantized_$(date +"%Y%m%d_%H%M%S") EXTRA_ARGS=--quantize bash scripts/benchmark.bash

## Sources
- PyTorch blog: Practical Quantization in PyTorch (dynamic vs static, calibration, supported layers)
  ```text
  https://pytorch.org/blog/quantization-in-practice/
  ```
- PyTorch blog: INT8 Quantization for x86 CPU in PyTorch (x86 backend details and speedups)
  ```text
  https://pytorch.org/blog/int8-quantization/
  ```
- Intel technical article: X86 quantization backend overview (backend behavior)
  ```text
  https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-int8-inf-with-new-x86-backend.html
  ```
