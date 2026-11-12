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
- `--default-model MODEL` (alias: `--language`): Built-in model loaded when the server starts. Defaults to `english` (the same model as `english_2026-10`); incompatible with `--config`. The `*_24l` models are previews.
- `--config`: Path to a custom config .yaml — a local path, an `https://` URL, or an `hf://` path. Incompatible with `--language`/`--default-model`.
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

### Teacher Forcing Between Chunks

Long texts are generated in chunks. Set the `teacher_force_previous_chunk` form field of `/tts` to
`true` to generate each chunk as the continuation of the previous one instead of starting afresh
from the voice prompt (same as `--teacher-force-previous-chunk` of `generate`; default: `false`,
slightly slower):

```bash
curl -X POST http://localhost:8000/tts \
  -F "text=A long text, made of several sentences..." \
  -F "teacher_force_previous_chunk=true" -o output.wav
```

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

The interface can switch between built-in models and configure whether to keep one model or a
selected set loaded. Keeping multiple models loaded speeds up switching but uses more memory. The
language selector is a preference; when it does not match the selected model's language, the model
may still generate speech in its trained language.

For more advanced usage, see the [Python API documentation](../API%20Reference/python-api.md) for direct integration with the TTS model.
