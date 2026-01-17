# Serve Command Documentation

The `serve` command starts an HTTP API server with a web interface for text-to-speech generation.

## Basic Usage

```bash
cargo run --release -p pocket-tts-cli -- serve
# or if installed:
pocket-tts serve
```

This starts a server on `http://localhost:8000` with the default voice.

## Command Options

- `--host HOST`: Host address to bind (default: `127.0.0.1`)
- `--port PORT`, `-p`: Port number (default: `8000`)
- `--voice VOICE`: Default voice for requests (default: `alba`)
- `--variant VARIANT`: Model variant (default: `b6369a24`)
- `--temperature FLOAT`: Sampling temperature (default: `0.7`)
- `--lsd-decode-steps INT`: LSD decode steps (default: `1`)
- `--eos-threshold FLOAT`: EOS threshold (default: `-4.0`)

## Examples

### Basic Server

```bash
# Start with defaults
pocket-tts serve

# Custom host and port
pocket-tts serve --host 0.0.0.0 --port 8080

# Custom default voice
pocket-tts serve --voice marius
```

## API Endpoints

### Health Check

```
GET /health
```

Returns server status:

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### Generate Audio (JSON)

```
POST /generate
Content-Type: application/json
```

Request body:

```json
{
  "text": "Hello, world!",
  "voice": "alba"
}
```

Response: WAV audio file

**Example:**

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "voice": "alba"}' \
  --output output.wav
```

### Streaming Generation

```
POST /stream
Content-Type: application/json
```

Request body: Same as `/generate`

Response: Chunked audio stream (raw PCM, 16-bit, 24kHz, mono)

**Example:**

```bash
curl -X POST http://localhost:8000/stream \
  -H 'Content-Type: application/json' \
  -d '{"text": "Streaming audio generation"}' | \
  ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

### Python API Compatibility

```
POST /tts
Content-Type: multipart/form-data
```

Form fields:
- `text`: Text to synthesize
- `voice`: Voice name or file

Response: WAV audio file

This endpoint maintains compatibility with the Python server's multipart form API.

### OpenAI Compatibility

```
POST /v1/audio/speech
Content-Type: application/json
```

Request body:

```json
{
  "model": "pocket-tts",
  "input": "Hello, world!",
  "voice": "alba"
}
```

Response: Audio file (WAV format)

This endpoint is compatible with OpenAI's text-to-speech API format.

## Web Interface

Navigate to `http://localhost:8000` to access the built-in web interface:

- Text input field
- Voice selection dropdown
- Generate button
- Audio playback
- Download link

## Voice Options

Voices can be specified as:

1. **Predefined name**: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`
2. **Base64 audio**: Include audio data directly in the request

## Response Formats

| Endpoint | Content-Type | Format |
|----------|--------------|--------|
| `/generate` | `audio/wav` | Complete WAV file |
| `/stream` | `application/octet-stream` | Raw PCM chunks |
| `/tts` | `audio/wav` | Complete WAV file |
| `/v1/audio/speech` | `audio/wav` | Complete WAV file |

## Error Handling

Errors return JSON with status code:

```json
{
  "error": "Error message here"
}
```

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 500 | Server error |

## Performance Notes

- Model is loaded once at startup and kept in memory
- Voice states are resolved per-request (consider caching on client side)
- Streaming endpoint provides lower latency for first audio

## See Also

- [Generate Command](generate.md) - CLI generation
- [Rust API](rust-api.md) - Library integration
