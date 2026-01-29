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

#### Set a different default voice

By default, the server uses the "alba" voice for all requests if no `voice_url` is specified. You can change this via the `--voice` parameter.

```bash
# Use a voice from hugging face
pocket-tts serve --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"

# Use local voice file
pocket-tts serve --voice "./my_voice.wav"

# Use local safetensors file
pocket-tts serve --voice "./my_voice.safetensors"
```

#### Custom Voices Directory

You can place your own custom voices in the `voices/` directory under the project root. They will be made available to the server. A custom voice can be an audio file (prompt audio for cloning) or a .safetensors file (a voice embedding previously exported). Audio files will be automatically converted to safetensors on server start.

For example, place `rob.wav` in `voices/`, then run:

```bash
# start server
pocket-tts serve

# get list of available voices
curl http://localhost:8000/voices

# generate audio and pipe result to ffplay
curl -X POST http://localhost:8000/tts \
   -F "text=Hello everyone. How is it going?" \
   -F "voice_url=rob" | ffplay -autoexit -nodisp -f s16le -ar 24000 -
```

You can override the default location of the custom voices directory via the `serve` command's `--custom-voices-dir` parameter.

### Custom Model Config
If you'd like to override the paths from which the models are loaded, you can provide a custom YAML configuration. 

Copy pocket_tts/config/b6369a24.yaml and change weights_path:, weights_path_without_voice_cloning: and tokenizer_path: to the paths of the models you want to load. 

Then, use the --config option to point to your newly created config.

```bash
# Use a different config
pocket-tts serve --config "C://pocket-tts/my_config.yaml"
```

## Web Interface

Once the server is running, navigate to `http://localhost:8000` to access the web interface.

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.