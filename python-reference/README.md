# Pocket TTS (Python Reference)

This is the original Python implementation of Pocket TTS, preserved as a reference for the Rust/Candle port.

## Usage

You can run this via `uvx` from the repository root:

```bash
uvx --from ./python-reference pocket-tts --help
```

Or by using `uv` inside this directory:

```bash
cd python-reference
uv run pocket-tts generate --text "Hello"
```
