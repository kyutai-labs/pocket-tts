# Serve

The `serve` command starts a FastAPI web server that provides both a web interface and HTTP API for text-to-speech generation.

## Basic Usage

```bash
uvx pocket-tts serve
# or if installed manually:
pocket-tts serve
```

This starts a server on `http://localhost:8000` with the default voice model.

## Command Options

- `--host HOST`: Host to bind to (default: "localhost")
- `--port PORT`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--language`: Language for the TTS model, one of `'english_2026-01'`, `'english_2026-04'`, `'english'`, `'french_24l'`, `'german_24l'`, `'portuguese_24l'`, `'italian_24l'`, `'spanish_24l'` (default: `english`, which is the same model as `'english_2026-04'`). Incompatible with `--config`. The "24l" variants are bigger models, not distilled yet and here only as preview.
- `--config`: Path to a custom config .yaml — a local path, an `https://` URL, or an `hf://` path. Incompatible with `--language`.
- `--default-voice`: Voice used by the requests that don't ask for one (default: the built-in voice of the language). It accepts anything the `generate --voice` option accepts: a built-in voice name, a local path to an audio file or to a `.safetensors` voice, an `https://` URL, or an `hf://` path. It is loaded at startup, so a voice that cannot be read fails the server immediately instead of the first request.
- `--quantize`: Use int8 quantization for the model (default: False). This can reduce memory usage and increase speed, with minimal impact on audio quality.
## Examples

### Basic Server

```bash
# Start with default settings
pocket-tts serve

# Custom host and port
pocket-tts serve --host "localhost" --port 8080
```

### Custom Language
To select the default language model, pass `--language`:
```bash
pocket-tts serve --language french_24l
```

### Custom Default Voice

The voice served when a request doesn't specify one is the built-in voice of the language.
Use `--default-voice` to serve your own instead:

```bash
# A local audio file
pocket-tts serve --default-voice "./my_voice.wav"

# A voice exported with `export-voice`, which loads much faster than an audio file
pocket-tts serve --default-voice "./my_voice.safetensors"

# A voice hosted on the web or on the Hugging Face Hub
pocket-tts serve --default-voice "https://example.com/my_voice.wav"
pocket-tts serve --default-voice "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

# Another built-in voice
pocket-tts serve --default-voice "marius"
```

Requests are free to ask for another voice: `--default-voice` only changes what is used when the
`voice_url` and `voice_wav` fields of `/tts` are both empty (in the web interface, when the voice
field is left empty).

### Custom Model Config

If you'd like to override the paths from which the models are loaded, you can provide a custom YAML configuration.

Copy one of the files in `pocket_tts/config` (for example `pocket_tts/config/english.yaml`) and change `weights_path`, `weights_path_without_voice_cloning:`, and `tokenizer_path:` to the paths of the models you want to load.

Then, use the --config option to point to your newly created config.

```bash
# Use a different config
pocket-tts serve --config "C://pocket-tts/my_config.yaml"
```

## Web Interface

Once the server is running, navigate to `http://localhost:8000` to access the web interface.

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.
