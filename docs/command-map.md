# GitHub Ralph Loop - Command Map

This file defines the canonical commands for the pocket-tts repository used by the Ralph Loop automation.

## Format

```bash
uvx pre-commit run --all-files
```

Or format explicitly:

```bash
uv run ruff format .
```

## Lint

```bash
uv run ruff check .
```

## Type Check

```bash
uv run mypy pocket_tts/
```

## Tests

Fast tests (default):

```bash
uv run pytest -n 3 -v
```

Full tests (with coverage):

```bash
uv run pytest -n 3 -v --cov=pocket_tts
```

## Build

Python wheel build:

```bash
uv build
```

## Rust Extensions (Optional)

For the audio processing Rust extensions:

```bash
cd training/rust_exts/audio_ds
cargo build --release
```

Or use the provided script:

```bash
cd training/rust_exts/audio_ds
./build.sh
```
