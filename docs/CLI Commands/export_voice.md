# Export Voice

Kyutai Pocket TTS allows you to generate speech with voice cloning from an audio sample. However, processing an audio file each time is relatively slow and inefficient.

The `export-voice` command allows you to convert an audio file to a voice embedding (it's actually the kvcache) in safetensors format. The safetensors file can then be loaded very quickly whenever you generate speech.

## Basic Usage

```bash
uvx pocket-tts export-voice audio-path export-path
# or if installed manually:
pocket-tts export-voice audio-path export-path
```

Only the first 30 seconds of the audio file will be processed.

## Command Options

### Required Parameters

- `audio-path`: Path of the audio file to convert. `audio-path` can point to an `http:` or `hf:` (hugging face) file. Supports popular audio file formats like wav and mp3.
- `export-path`: Path of the output safetensors file to write.

### Options

- `--quiet`: Do not print any output except errors.
- `--config`: Model config path or signature

## Examples

```bash
# export a single file
pocket-tts export-voice voice_memo127762.mp3 jack.safetensors

# export an online file to current directory
pocket-tts export-voice https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/announcer.wav ./announcer.safetensors

# use the exported safetensors
pocket-tts generate --text "Hello, welcome to today's game between the Bears and Cubs."  --voice announcer.safetensors
```
