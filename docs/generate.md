# Generate Command Documentation

The `generate` command allows you to generate speech from text directly from the command line using Kyutai Pocket TTS.

## Basic Usage

```bash
uvx pocket-tts generate
# or if installed manually:
pocket-tts generate
```

This will generate a WAV file `./tts_output.wav` with the default text and voice.

## Command Options

### Core Options

- `--text TEXT`: Text to generate (default: "Hello world! I am Kyutai Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.")
- `--voice VOICE`: Path to audio conditioning file (voice to clone) (default: "hf://kyutai/tts-voices/alba-mackenna/casual.wav"). Urls and local paths are supported.
- `--output-path OUTPUT_PATH`: Output path for generated audio (default: "./tts_output.wav")

### Generation Parameters

- `--config CONFIG_PATH`: Path to custom config.yaml (for loading local model files) or model signature (default: "b6369a24")
- `--lsd-decode-steps LSD_DECODE_STEPS`: Number of generation steps (default: 1)
- `--temperature TEMPERATURE`: Temperature for generation (default: 0.7)
- `--noise-clamp NOISE_CLAMP`: Noise clamp value (default: None)
- `--eos-threshold EOS_THRESHOLD`: EOS threshold (default: -4.0)
- `--frames-after-eos FRAMES_AFTER_EOS`: Number of frames to generate after EOS (default: None, auto-calculated based on the text length). Each frame is 80ms.

### Performance Options

- `--device DEVICE`: Device to use (default: "cpu", you may not get a speedup by using a gpu since it's a small model)
- `--quiet`, `-q`: Disable logging output

## Examples

### Basic Generation

```bash
# Generate with default settings
pocket-tts generate

# Custom text
pocket-tts generate --text "Hello, this is a custom message."

# Custom output path
pocket-tts generate --output-path "./my_audio.wav"
```

### Voice Selection

```bash
# Use different voice from HuggingFace
pocket-tts generate --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"

# Use local voice file
pocket-tts generate --voice "./my_voice.wav"

# Use a safetensors file (such as one created using `pocket-tts export-voice`)
pocket-tts generate --voice "./my_voice.safetensors"
```


### Quality Tuning

```bash
# Higher quality (more steps)
pocket-tts generate --lsd-decode-steps 5 --temperature 0.5

# More expressive (higher temperature)
pocket-tts generate --temperature 1.0

# Adjust EOS threshold, smaller means finishing earlier.
pocket-tts generate --eos-threshold -3.0
```

### Custom Model Config

If you'd like to override the paths from which the models are loaded, you can provide a custom YAML configuration. 

Copy pocket_tts/config/b6369a24.yaml and change weights_path:, weights_path_without_voice_cloning: and tokenizer_path: to the paths of the models you want to load. 

Then, use the --config option to point to your newly created config.

```bash
# Use a different config
pocket-tts generate --config "C://pocket-tts/my_config.yaml"
```

## Output Format

The generate command always outputs WAV files in the following format:
- **Sample Rate**: 24kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit PCM
- **Format**: Standard WAV file

For more advanced usage, see the [Python API documentation](python-api.md) or consider using the [serve command](serve.md) for web-based generation and quick iteration.
