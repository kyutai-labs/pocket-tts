# Quantization

Pocket TTS supports dynamic int8 quantization to reduce runtime memory usage and improve inference speed on x86 CPUs.

## Quick Start

### CLI

```bash
pocket-tts generate --quantize --text "Hello world"
pocket-tts serve --quantize
```

### Python API

```python
from pocket_tts import TTSModel

model = TTSModel.load_model(quantize=True)
voice_state = model.get_state_for_audio_prompt("alba")
audio = model.generate_audio(voice_state, "Hello world!")
```

## Installation

Quantization works out of the box on any supported PyTorch version (2.5+) using `torch.ao`.

For optimized performance, install `torchao` (requires torch 2.10+):

```bash
pip install pocket-tts[quantize]
```

The quantization module automatically selects the best available backend:

- **torchao** (torch 2.10+): optimized C++ kernels, faster on both ARM and x86
- **torch.ao** (torch 2.5-2.9): deprecated but functional fallback

## Performance

Benchmarks on the full eval paragraph across 8 voices, 5 isolated runs per config.

### x86 (FBGEMM, ubuntu-latest GitHub Actions runner)

| Config                      | Runtime Memory | RTS       | Speedup vs Baseline |
| --------------------------- | -------------- | --------- | ------------------- |
| baseline                    | 450 MB         | 3.17x     | --                  |
| **attention_ffn (default)** | **234 MB**     | **4.04x** | **1.27x**           |
| all                         | 206 MB         | 4.01x     | 1.26x               |

### ARM (QNNPACK, Apple M4 MacBook Air, torchao backend)

| Config                      | Runtime Memory | RTS       | Speedup vs Baseline |
| --------------------------- | -------------- | --------- | ------------------- |
| baseline                    | 450 MB         | 6.33x     | --                  |
| **attention_ffn (default)** | **234 MB**     | **7.76x** | **1.23x**           |

With the `torch.ao` fallback (torch 2.5-2.9), ARM performance is ~16% slower than baseline rather than faster. Upgrading to torch 2.10+ with torchao is recommended for ARM users.

## Quality

Quantization has no measurable impact on speech quality:

- **WER (Word Error Rate)**: WER delta for the default attention_ffn config is −0.022 ±0.032 - the ± range crosses zero, meaning the delta is indistinguishable from measurement noise.
- **Subjective listening**: no audible difference across all 8 voices

## Advanced Usage

Override which layer groups to quantize:

```python
# Quantize everything (maximum memory savings)
model = TTSModel.load_model(quantize=True, quantize_groups={"attention", "ffn", "flow_net"})

# Quantize only FFN layers
model = TTSModel.load_model(quantize=True, quantize_groups={"ffn"})
```

Available groups:

| Group       | Params | Description                                      |
| ----------- | ------ | ------------------------------------------------ |
| `attention` | ~25M   | Q/K/V/output projections in transformer layers   |
| `ffn`       | ~50M   | Feed-forward linear layers in transformer layers |
| `flow_net`  | ~7M    | MLP sampler / flow matching network              |
