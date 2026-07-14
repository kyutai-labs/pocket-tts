# OpenAI-Compatible API

The `serve` command exposes OpenAI-compatible TTS endpoints under the `/v1` prefix. This allows you to use Pocket TTS with any tool or library that supports the OpenAI speech API.

## Starting the Server

```bash
# Basic usage
uvx pocket-tts serve

# With custom voices from a folder
uvx pocket-tts serve --voices-folder ./my_voices
```

## Endpoints

### Generate Speech

**`POST /v1/audio/speech`**

Generate speech from text and return audio as a WAV file.

**Request Body (JSON):**

| Field             | Type   | Required | Default        | Description                                              |
| ----------------- | ------ | -------- | -------------- | -------------------------------------------------------- |
| `model`           | string | No       | `"pocket-tts"` | Model identifier (ignored)                               |
| `input`           | string | **Yes**  | —              | Text to synthesize                                       |
| `voice`           | string | No       | `"alba"`       | Voice name (built-in or custom)                          |
| `response_format` | string | No       | `"wav"`        | Output format (ignored, only `wav` is actually produced) |
| `speed`           | float  | No       | `1.0`          | Playback speed multiplier (ignored)                      |

**Response:**

- `200 OK` — WAV audio bytes (`Content-Type: audio/wav`)
- `400 Bad Request` — Invalid input (empty text, unknown voice)

**Example:**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test.",
    "voice": "alba"
  }' \
  --output speech.wav
```

**Python (openai package):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="pocket-tts", voice="alba", input="Hello world, this is a test.", response_format="wav"
)
response.write_to_file("speech.wav")
```

## Custom Voices

Use `--voices-folder` to register custom voices at startup:

```bash
# Place voice files in a directory
mkdir -p my_voices
cp my_speaker.wav my_voices/
cp cloned_voice.safetensors my_voices/

# Start the server
uvx pocket-tts serve --voices-folder ./my_voices
```

All `.wav`, `.mp3`, and `.safetensors` files in the folder are registered. The filename without extension becomes the voice name (e.g., `my_speaker.wav` → `"my_speaker"`).

Custom voices take priority over built-in voices if names collide. Use `.safetensors` files (created via [`export-voice`](export_voice.md)) for the fastest loading.

## Differences from OpenAI

- Only `wav` format is actually produced regardless of `response_format`
- `model` and `speed` parameters are also accepted but ignored
- No streaming support (returns complete WAV file)
- Voice names differ from OpenAI's (`alba`, `anna`, etc. instead of `alloy`, `nova`, etc.)
