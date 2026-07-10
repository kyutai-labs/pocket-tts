# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

pocket-tts is a CPU-based text-to-speech (TTS) model. The project uses a flow-based language model architecture with a neural audio codec (Mimi) for efficient speech synthesis.

**Key Architecture Components:**
- **FlowLMModel**: Transformer-based flow language model that generates latent representations from text using Lagrangian Self Distillation (LSD)
- **MimiModel**: Neural audio codec (from the `moshi` package) that compresses/decompresses audio to/from latent representations
- **Conditioners**: Text processing via SentencePiece tokenizer and lookup table embeddings
- **Streaming Architecture**: The entire pipeline supports streaming generation via stateful modules
- **Web API**: FastAPI-based server for HTTP API access with web interface

## Common Commands

### Setup and Development
```bash
# Install pre-commit hooks
uvx pre-commit install

# Run tests (3 parallel workers)
uv run pytest -n 3 -v

# Run a single test
uv run pytest tests/test_python_api.py -v

# Run CLI locally (editable install)
uv run pocket-tts generate
uv run pocket-tts serve
```

### Linting and Formatting
Pre-commit handles this automatically, but you can run manually:
```bash
# Ruff will run automatically on commit via pre-commit
# Includes: ruff-check, ruff-format (with --fix), and import sorting
```

### Building (No Build Step)
This is a pure Python package with Rust extensions in `training/rust_exts/audio_ds/` for training-time audio processing. The main package does not require building.

## Code Structure

### Main Package (`pocket_tts/`)

**Entry Points:**
- `main.py`: CLI implementation with Typer (commands: `generate`, `serve`, and web interface)
- `__init__.py`: Public API exports only `TTSModel`
- `__main__.py`: Python module entry point
- `default_parameters.py`: Default configuration values for generation parameters
- `static/`: Web interface files (HTML for server UI)

**Core Models (`models/`):**
- `tts_model.py`: Main `TTSModel` class - orchestrates the entire TTS pipeline
  - `load_model()`: Downloads weights from HuggingFace and initializes models
  - `get_state_for_audio_prompt()`: Encodes audio prompt (voice) into model state
  - `generate_audio_stream()`: Streaming generation that yields audio chunks
  - Uses LRU cache for voice prompts to avoid reprocessing
- `flow_lm.py`: `FlowLMModel` - transformer that generates latent audio codes from text

**Modules (`modules/`):**
- `transformer.py`: `StreamingTransformer` and `StreamingMultiheadAttention` with RoPE embeddings
- `stateful_module.py`: Base class for streaming support (maintains KV cache and state)
- `rope.py`: Rotary Position Embeddings
- `mlp.py`: `SimpleMLPAdaLN` (AdaLN-conditioned MLP for flow prediction)
- `conv.py`: Convolution utilities
- `seanet.py`: SEANet encoder/decoder (copied from moshi)

**Conditioners (`conditioners/`):**
- `text.py`: `LUTConditioner` - SentencePiece tokenizer + embedding lookup table for text

**Data (`data/`):**
- `audio.py`: Audio I/O utilities (reading, writing WAV, streaming)
- `audio_utils.py`: Audio processing (resampling, conversion)

**Utils (`utils/`):**
- `config.py`: Pydantic config models for FlowLM and Mimi
- `utils.py`: HuggingFace downloads, timing utilities

**Configuration (`config/`):**
- `b6369a24.yaml`: Model configuration (transformer dims, layers, vocab size, etc.)

### Testing (`tests/`)
- `test_python_api.py`: Tests for public Python API
- `test_cli_generate.py`: Tests for CLI generate command
- `test_documentation_examples.py`: Ensures docs examples work

## Development Workflow

### Key Patterns

1. **Streaming Generation**: The model generates audio frame-by-frame (12.5 Hz frame rate, 80ms per frame). All modules inherit from `StatefulModule` to maintain internal state.

2. **Voice Cloning**: Audio prompts are encoded via Mimi encoder to create "voice state" (latent representations + speaker embeddings). This state is cached via `lru_cache` on `_cached_get_state_for_audio_prompt()`.

3. **Flow Matching**: Uses Lagrangian Self Distillation (LSD) with configurable decode steps. Fewer steps = faster but lower quality.

4. **EOS Detection**: Model predicts end-of-speech via an EOS head. Generation continues for `frames_after_eos` frames after EOS is detected.

5. **Config-Driven**: Model architecture is defined in YAML configs. Weights are loaded from HuggingFace Hub via `safetensors`.

### Important Implementation Details

- **Thread Safety**: The code is NOT thread-safe. Server mode does not support concurrent requests.
- **Batching**: Batch size is always 1. No batching support currently.
- **Device**: Defaults to CPU. Whether GPU helps is hardware-dependent: on hardware with strong
  single-thread CPU performance (e.g. Apple Silicon) we did not observe a speedup, but on a cloud
  x86 VM with a Tesla T4, `.to("cuda")` measured ~2.6x speedup over CPU (RTF ~2.3-2.5x CPU vs.
  ~6.28x GPU). There is no `device` argument on `TTSModel.load_model()` (the `generate` CLI has a
  documented `--device`, though its own doc page's "no speedup" note predates this finding; `serve`
  and Docker have none and are CPU-only). `generate_audio()` returns a tensor on the model's device,
  so code moved to GPU must call `.detach().cpu().numpy()` instead of `.numpy()`. `quantize=True` is
  CPU-only and raises `NotImplementedError` on CUDA; the optional `torchao` backend declares
  `torch>=2.11`, which a fresh install satisfies, but installing it on top of an older pinned
  `torch` (e.g. for GPU driver compatibility) can break `quantize=True` even on CPU. See README's
  "Running on GPU" section for details.
- **Torch Threads**: `torch.set_num_threads(1)` in `tts_model.py` for optimal CPU performance
- **dtype**: Models use float32 by default (configurable in YAML)
- **Beartype**: Runtime type checking is enabled via beartype claw in `__init__.py`

### Adding Features

When adding features, be aware of:
- The streaming architecture: any changes to model forward passes need to maintain state correctly
- The config system: new model parameters must be added to config classes in `utils/config.py`
- The public API: only `TTSModel` is exported; keep implementation details internal
- Ruff formatting: line length 100, LF line endings, skip magic trailing comma

### Model Weights

Weights are downloaded from HuggingFace Hub on first use:
- Model weights: `hf://kyutai/pocket-tts/tts_b6369a24.safetensors`
- Tokenizer: `hf://kyutai/pocket-tts/tokenizer.model`
- Voice prompts: `hf://kyutai/tts-voices/<speaker>/<style>.wav`

The `download_if_necessary()` utility handles `hf://` URLs and caches locally.

## Common Gotchas

1. **PyTorch Version**: Requires PyTorch 2.5+. Version 2.4.0 produces incorrect audio.
2. **Python Version**: Supports Python 3.10 through 3.14 (>= 3.10,<3.15).
3. **uv Python Preference**: Set to "only-managed" in pyproject.toml because system Python may lack headers.
4. **CPU-Only PyTorch**: Uses PyTorch CPU index from `download.pytorch.org/whl/cpu` in uv config.
5. **Web Dependencies**: FastAPI and Uvicorn are included for server functionality.
