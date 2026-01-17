# Generate Command Documentation

The `generate` command synthesizes speech from text and saves it to a WAV file.

## Basic Usage

```bash
# Build and run
cargo run --release -p pocket-tts-cli -- generate

# Or if installed
pocket-tts generate
```

This generates `./output.wav` with the default text and voice.

## Command Options

### Core Options

- `--text TEXT`, `-t`: Text to synthesize (default: greeting)
- `--voice VOICE`, `-v`: Voice specification (see below)
- `--output PATH`, `-o`: Output WAV file path (default: `output.wav`)

### Generation Parameters

- `--variant VARIANT`: Model variant identifier (default: `b6369a24`)
- `--temperature FLOAT`: Sampling temperature for variation (default: `0.7`)
- `--lsd-decode-steps INT`: LSD decode steps, more = better quality (default: `1`)
- `--eos-threshold FLOAT`: End-of-speech threshold (default: `-4.0`)
- `--noise-clamp FLOAT`: Optional noise clamp value
- `--frames-after-eos INT`: Frames to generate after EOS (auto-calculated if not set)

### Output Options

- `--stream`: Stream raw PCM audio to stdout (for piping)
- `--quiet`, `-q`: Suppress all output except errors

## Voice Specification

The `--voice` argument supports multiple formats:

### Predefined Voices

```bash
pocket-tts generate --voice alba
pocket-tts generate --voice marius
```

Available: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

### Local WAV File

```bash
pocket-tts generate --voice ./my_voice.wav
```

### Pre-computed Embeddings

```bash
pocket-tts generate --voice ./embeddings.safetensors
```

### HuggingFace URL

```bash
pocket-tts generate --voice "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
```

## Examples

### Basic Generation

```bash
# Default settings
pocket-tts generate

# Custom text
pocket-tts generate --text "Hello, this is a custom message."

# Custom output path
pocket-tts generate --output ./my_audio.wav
```

### Voice Selection

```bash
# Predefined voice
pocket-tts generate --voice marius --text "Good morning!"

# Voice cloning from WAV
pocket-tts generate --voice ./reference.wav --text "Clone my voice"

# HuggingFace voice
pocket-tts generate --voice "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
```

### Quality Tuning

```bash
# Higher quality (more steps, lower temperature)
pocket-tts generate --lsd-decode-steps 5 --temperature 0.5

# More expressive (higher temperature)
pocket-tts generate --temperature 1.0

# Longer audio (more negative EOS threshold)
pocket-tts generate --eos-threshold -5.0
```

### Streaming to Audio Player

```bash
# Stream to ffplay (Linux/macOS)
pocket-tts generate --stream --text "Streaming audio" | \
  ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -

# Stream to SoX play
pocket-tts generate --stream | play -t raw -r 24k -e signed -b 16 -c 1 -
```

## Output Format

Generated audio has the following format:

| Property | Value |
|----------|-------|
| Sample Rate | 24,000 Hz |
| Channels | Mono (1) |
| Bit Depth | 16-bit PCM |
| Format | Standard WAV |

When using `--stream`, raw PCM samples are written to stdout:
- Little-endian 16-bit signed integers
- 24 kHz sample rate
- Mono channel

## Performance Tips

1. **Release build**: Always use `--release` for production
   ```bash
   cargo run --release -p pocket-tts-cli -- generate
   ```

2. **Reuse voice state**: For multiple generations with the same voice, use the HTTP server (`serve` command) to keep the model in memory

3. **LSD steps**: Start with 1 for speed, increase to 3-5 for quality

4. **Temperature**: Use 0.0 for deterministic output, 0.7 for natural variation

## See Also

- [Serve Command](serve.md) - HTTP API server
- [Rust API](rust-api.md) - Library integration
