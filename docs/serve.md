# Serve Command Documentation

The `serve` command starts a FastAPI web server that provides both a web interface and HTTP API for text-to-speech generation.

## Basic Usage

```bash
uvx pocket-tts serve
# or if installed manually:
pocket-tts serve
```

This starts a server on `http://localhost:8000` with the default voice model.

## Command Options

- `--voice VOICE`: Path to voice prompt audio file (voice to clone) (default: "hf://kyutai/tts-voices/alba-mackenna/casual.wav")
- `--host HOST`: Host to bind to (default: "localhost")
- `--port PORT`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--config`: Path to a custom config .yaml

## Examples

### Basic Server

```bash
# Start with default settings
pocket-tts serve

# Custom host and port
pocket-tts serve --host "localhost" --port 8080
```

### Custom Voice

```bash
# Use different voice
pocket-tts serve --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"

# Use local voice file
pocket-tts serve --voice "./my_voice.wav"
```
### Custom Model Config
If you'd like to override the paths from which the models are loaded, you can provide a custom YAML configuration. 

Copy pocket_tts/config/b6369a24.yaml and change weights_path:, weights_path_without_voice_cloning: and tokenizer_path: to the paths of the models you want to load. 

Then, use the --config option to point to your newly created config.

```bash
# Use a different config
pocket-tts serve --config "C://pocket-tts/my_config.yaml"
```

## OpenAI-Compatible Endpoint

The server exposes `POST /v1/audio/speech`, a drop-in replacement for the
[OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).
Existing apps only need to change their `base_url` to point at Pocket TTS.

### curl example

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world","voice":"alloy"}' \
  --output speech.wav
```

### Voice mapping

OpenAI voice names are mapped to Pocket TTS voices automatically.
You can also use the native Pocket TTS names directly.

| OpenAI voice | Pocket TTS voice |
|-------------|------------------|
| alloy       | alba             |
| echo        | marius           |
| fable       | javert           |
| onyx        | jean             |
| nova        | fantine          |
| shimmer     | cosette          |
| coral       | eponine          |
| sage        | azelma           |

### Using with the OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.audio.speech.create(model="tts-1", voice="alloy", input="Hello from Pocket TTS!")
response.stream_to_file("speech.wav")
```

### Supported parameters

| Parameter        | Supported | Notes                                |
|-----------------|-----------|--------------------------------------|
| model           | ignored   | Only one model, any value accepted   |
| input           | yes       | Text to speak                        |
| voice           | yes       | OpenAI or Pocket TTS name            |
| response_format | partial   | `wav` (default) and `pcm` supported  |
| speed           | ignored   | Accepted but has no effect           |

## Web Interface

Once the server is running, navigate to `http://localhost:8000` to access the web interface.

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.