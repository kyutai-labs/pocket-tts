# Ralph Loop Agents (Operational)

## Build & Run
- `uv run pocket-tts --help`
- `uv run pocket-tts generate "Hello" --voice alba --output out.wav`
- `uv run pocket-tts serve --port 8080`

## Validation
- Tests: `uv run pytest -n 3 -v`

## Codebase Patterns
- Absolute imports only (no relative imports).
- Rust must remain safe (no `unsafe`) in `rust-numpy/`.
- PyTorch 2.5+ only; avoid 2.4.0.
- Model is not thread-safe; use separate instances for concurrency.

## Notes
- Primary code directories: `pocket_tts/`, `rust-numpy/`, `training/`, `tests/`.
