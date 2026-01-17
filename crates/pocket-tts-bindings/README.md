# Pocket TTS Python Bindings

High-performance Rust bindings for Pocket TTS, powered by [PyO3](https://github.com/PyO3/pyo3) and [Candle](https://github.com/huggingface/candle).

## Features

- **Native Rust Performance**: Run the text-to-speech pipeline entirely in backend Rust code.
- **Drop-in Generation**: Generate audio from text using the same model checkpoints as the Python version.
- **No Torch Dependency**: Runs on `candle-core`, removing the need for heavy PyTorch dependencies if only inference is needed.

## Installation

You need [Rust](https://www.rust-lang.org/) and [Maturin](https://github.com/PyO3/maturin) installed.

```bash
# Install maturin
pip install maturin

# Build and install the bindings
maturin develop --release
```

## Usage

```python
import pocket_tts_bindings

# 1. Load the model
# Matches the loading of the main pocket-tts-candle crate
model = pocket_tts_bindings.PyTTSModel.load("b6369a24")

# 2. Generate Audio
# Returns a list of floats (audio samples at 24kHz)
# You can use a .wav file or a .safetensors file (pre-computed embeddings)
samples = model.generate(
    "This is synthesized by Rust.",
    "path/to/reference_voice.wav"
)

# Using Predefined Voices (e.g., 'alba')
# You must download the .safetensors file first
from huggingface_hub import hf_hub_download
alba_path = hf_hub_download(
    repo_id="kyutai/pocket-tts-without-voice-cloning",
    filename="embeddings/alba.safetensors"
)
samples = model.generate(
    "Hello from Alba!",
    alba_path
)

# 3. Save to file (using standard python libs)
import wave, struct

# Scale float samples to 16-bit integers
ints = [max(-32768, min(32767, int(s * 32767))) for s in samples]

with wave.open("output.wav", 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(24000)
    f.writeframes(struct.pack('<' + 'h' * len(ints), *ints))
```

