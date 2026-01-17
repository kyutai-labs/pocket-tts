# Python API Documentation

Kyutai Pocket TTS provides a Python API for integrating text-to-speech capabilities into your applications.

## Installation

```bash
pip install pocket-tts
```

## Quick Start

```python
from pocket_tts import TTSModel
import scipy.io.wavfile

# Load the model
tts_model = TTSModel.load_model()

# Get voice state from an audio file
voice_state = tts_model.get_state_for_audio_prompt(
    "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
)

# Generate audio
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")

# Save to file
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
```

## Core Classes

### TTSModel

The main class for text-to-speech generation.

#### Class Methods

##### `load_model(variant="b6369a24", temp=0.7, lsd_decode_steps=1, noise_clamp=None, eos_threshold=-4.0)`

Load and return a TTSModel instance with pre-trained weights.

**Parameters:**
- `variant` (str): Model variant identifier (default: "b6369a24")
- `temp` (float): Sampling temperature for generation (default: 0.7)
- `lsd_decode_steps` (int): Number of generation steps (default: 1)
- `noise_clamp` (float | None): Maximum value for noise sampling (default: None)
- `eos_threshold` (float): Threshold for end-of-sequence detection (default: -4.0)

**Returns:**
- `TTSModel`: Loaded model instance on CPU

**Example:**
```python
from pocket_tts import TTSModel

# Load with default settings
model = TTSModel.load_model()

# Load with custom parameters
model = TTSModel.load_model(variant="b6369a24", temp=0.5, lsd_decode_steps=5, eos_threshold=-3.0)
```

#### Properties

##### `device` (str)

Returns the device type where the model is running ("cpu" or "cuda").
By default, the model runs on CPU.

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()
print(f"Model running on: {model.device}")
```

##### `sample_rate` (int)

Returns the generated audio sample rate (typically 24000 Hz).

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()
print(f"Sample rate: {model.sample_rate} Hz")
```

#### Methods

##### `get_state_for_audio_prompt(audio_conditioning, truncate=False)`

Extract model state for a given audio file or URL (voice cloning).

**Parameters:**
- `audio_conditioning` (Path | str | torch.Tensor): Audio file path, URL, or tensor
- `truncate` (bool): Whether to truncate the audio (default: False)

**Returns:**
- `dict`: Model state dictionary containing hidden states and positional information

**Example:**
```python
from pocket_tts import TTSModel

model = TTSModel.load_model()
# From HuggingFace URL
voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

# From local file
voice_state = model.get_state_for_audio_prompt("./my_voice.wav")

# From HTTP URL
voice_state = model.get_state_for_audio_prompt(
    "https://huggingface.co/kyutai/tts-voices/resolve"
    "/main/expresso/ex01-ex02_default_001_channel1_168s.wav"
)
```

##### `generate_audio(model_state, text_to_generate, frames_after_eos=None, copy_state=True)`

Generate complete audio tensor from text input.

**Parameters:**
- `model_state` (dict): Model state from `get_state_for_audio_prompt()`
- `text_to_generate` (str): Text to convert to speech
- `frames_after_eos` (int | None): Frames to generate after EOS detection (default: None)
- `copy_state` (bool): Whether to copy the state (default: True)

**Returns:**
- `torch.Tensor`: Audio 1D tensor with shape [samples]

**Example:**
```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

# Generate audio
audio = model.generate_audio(voice_state, "Hello world!", frames_after_eos=2, copy_state=True)

print(f"Generated audio shape: {audio.shape}")
print(f"Audio duration: {audio.shape[-1] / model.sample_rate:.2f} seconds")
```

##### `generate_audio_stream(model_state, text_to_generate, frames_after_eos=None, copy_state=True)`

Generate audio streaming chunks from text input.

**Parameters:** Same as `generate_audio()`

**Yields:**
- `torch.Tensor`: Audio chunks with shape [samples]

**Example:**
```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
# Stream generation
for chunk in model.generate_audio_stream(voice_state, "Long text content..."):
    # Process each chunk as it's generated
    print(f"Generated chunk: {chunk.shape[0]} samples")
    # Could save chunks to file or play in real-time
```

## Advanced Usage

### Voice Management

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()
# Preload multiple voices
voices = {
    "casual": model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav"),
    "funny": model.get_state_for_audio_prompt(
        "https://huggingface.co/kyutai/tts-voices/resolve/main/expresso/ex01-ex02_default_001_channel1_168s.wav"
    ),
}

# Generate with different voices
casual_audio = model.generate_audio(voices["casual"], "Hey there!")
funny_audio = model.generate_audio(voices["funny"], "Good morning.")
```

### Batch Processing

```python
from pocket_tts import TTSModel
import scipy.io.wavfile
import torch

model = TTSModel.load_model()

voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
# Process multiple texts efficiently by re-using the same voice state
texts = [
    "First sentence to generate.",
    "Second sentence to generate.",
    "Third sentence to generate.",
]

audios = []
for text in texts:
    audio = model.generate_audio(voice_state, text)
    audios.append(audio)

# Concatenate all audio
full_audio = torch.cat(audios, dim=0)
scipy.io.wavfile.write("batch_output.wav", model.sample_rate, full_audio.numpy())
```

### Streaming to File
You can refer to our CLI implementation which can stream audio to a wav file.

For more information about the command-line interface, see the [Generate Documentation](generate.md) or [Serve Documentation](serve.md).