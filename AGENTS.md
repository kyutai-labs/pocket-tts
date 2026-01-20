# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-19T10:25:08Z
**Commit:** 98a7b5b
**Branch:** main

## OVERVIEW
CPU-based text-to-speech (TTS) system using a flow-based language model (FlowLM) and neural audio codec (Mimi). Built with Python (PyTorch) and high-performance Rust components.

## STRUCTURE
```
.
├── pocket_tts/          # Main TTS package (Python)
├── rust-numpy/          # Pure-Rust NumPy replacement library
├── training/            # Training extensions and Rust audio processing
└── tests/               # API, CLI, and integration tests
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Core TTS Logic | `pocket_tts/models/tts_model.py` | Orchestration, streaming, EOS detection |
| CLI / Web API | `pocket_tts/main.py` | Typer CLI and FastAPI server |
| Audio Processing | `pocket_tts/data/` | Python I/O; Rust optimized in `rust_audio.py` |
| Numerical Ops | `rust-numpy/` | High-performance Rust numerical algorithms |
| Configs | `pocket_tts/utils/config.py` | Pydantic-based hierarchical settings |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `TTSModel` | Class | `pocket_tts/models/tts_model.py` | Primary public API and orchestrator |
| `FlowLMModel` | Class | `pocket_tts/models/flow_lm.py` | Transformer-based flow generator |
| `MimiModel` | Class | `pocket_tts/models/mimi.py` | Neural audio codec |
| `Array` | Struct | `rust-numpy/src/array.rs` | Core Rust numerical container |
| `cli_app` | Variable | `pocket_tts/main.py` | Main CLI entry point |

## CONVENTIONS
- **Absolute Imports**: Relative imports are strictly banned by Ruff.
- **Pure Safe Rust**: 100% safe Rust only (`no unsafe`); no external Rust crates in `rust-numpy`.
- **Streaming-First**: All model components must support stateful streaming (`StatefulModule`).
- **NumPy Parity**: Rust numerical implementations must match NumPy 100%.

## ANTI-PATTERNS (THIS PROJECT)
- **PyTorch 2.4.0**: NEVER use; produces incorrect audio. Requires 2.5+.
- **Unsafe Rust**: Avoid `unsafe` unless absolutely required for FFI.
- **GPU Usage**: Small model optimized for CPU; GPU provides no speedup.
- **Concurrent Requests**: The model is NOT thread-safe; use separate instances for concurrency.

## COMMANDS
```bash
# Setup
uv run pocket-tts --help

# Generate Audio
uv run pocket-tts generate "Hello" --voice alba --output out.wav

# Run Server
uv run pocket-tts serve --port 8080

# Tests
uv run pytest -n 3 -v
```

## NOTES
- **EOS Detection**: Generation continues for `frames_after_eos` after EOS is detected.
- **Voice Caching**: Prompt states are cached via LRU.
- **Dual Package Managers**: `pyproject.toml` (uv) and `pixi.toml` are used.

## DEVELOPMENT WORKFLOW
1. **Streaming Generation**: Model generates audio frame-by-frame (12.5 Hz). Modules maintain state via `StatefulModule`.
2. **Voice Cloning**: Audio encoded via Mimi to create "voice state", cached via `lru_cache`.
3. **Flow Matching**: Uses Lagrangian Self Distillation (LSD) with configurable decode steps.
4. **Beartype**: Runtime type checking enabled via beartype claw in `__init__.py`.
