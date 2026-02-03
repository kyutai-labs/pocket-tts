# Export Voice Command Documentation

Kyutai Pocket TTS allows you to generate speech from an audio sample. However, processing an audio file each time is relatively slow and inefficient.

The `export-voice` command allows you to convert an audio file to a voice embedding in safetensors format. The safetensors file can then be loaded very quickly whenever you generate speech.

## Basic Usage

```bash
uvx pocket-tts export-voice audio-path export-path
# or if installed manually:
pocket-tts export-voice audio-path export-path
```

## Command Options

### Required Parameters

- `audio-path`: Path of the audio file or directory to convert. `audio-path` can point to an `http:` or `hf:` (hugging face) file. If `audio-path` is a local directory, all audio files found inside it will be batch converted. Supports popular audio file formats like wav and mp3.
- `export-path`: Path of safetensors file or directory to export to. For batch conversion, export-path should be a directory. The directory will be created if it does not exist.

### Options

- `--truncate`: Automatically truncate long audio files down to 30 seconds.

The other parameters such as `--lsd-decode-steps` and `--temperature` are the same as for the `generate` command. See the [generate documentation](https://github.com/kyutai-labs/pocket-tts/tree/main/docs/generate.md) for more details.

## Examples

```bash
# export a single file
pocket-tts export-voice voice_memo127762.mp3 jack.safetensors

# export a single file to a different directory (output is embbeddings/mary.safetensors
pocket-tts export-voice voices/mary.wav embeddings/

# export an entire directory of audio files, truncate long audios
pocket-tts export-voice voices/ embeddings/ --truncate

# export an online file to current directory
pocket-tts export-voice https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/announcer.wav .

# use the exported safetensors
pocket-tts generate --text "Hello, welcome to today's game between the Bears and Cubs."  --voice announcer.safetensors
```

Note: to indicate a directory rather than a file, please be sure to include a trailing / (\ on Windows).
