# AGENTS.md

This file provides guidance to AI agents when working with the pocket_tts Python package.

## OVERVIEW

Core Python package implementing the TTS pipeline with FFI to Rust audio processing.

## STRUCTURE

- `models/`: Core TTS models (TTSModel, FlowLMModel, MimiModel)
- `modules/`: Neural network components (StreamingTransformer, attention, RoPE, SEANet)
- `conditioners/`: Text conditioning (SentencePiece tokenizer, embeddings)
- `data/`: Audio I/O utilities (WAV loading/saving, resampling, streaming)
- `utils/`: Config loading, HuggingFace downloads, logging, export tools
- `config/`: YAML model architecture configurations
- `static/`: Web interface assets (HTML for HTTP server UI)
- `rust_libs/`: Compiled Rust shared libraries for audio acceleration

## WHERE TO LOOK

### Public API
- `__init__.py`: Public exports (TTSModel, load_wav, save_audio, Rust-accelerated functions)
- `models/tts_model.py`: Main TTSModel class with streaming generation
- `data/audio.py`: Audio loading (load_wav), streaming chunk generation
- `data/audio_output.py`: Audio saving utilities

### FFI Wrappers
- `rust_audio.py`: Python bindings for Rust audio processing via ctypes
  - Automatic fallback to pure Python if library not found
  - Functions: normalize_audio, apply_gain, resample_audio, apply_fade, compute_audio_metrics
  - Library search: `training/rust_exts/audio_ds/`, standard system paths

### Monolithic Entry Point
- `main.py`: CLI and server implementations (473 lines)
  - `generate`: CLI command for audio generation
  - `serve`: FastAPI HTTP server with web interface
  - `websocket`: WebSocket server for real-time streaming
  - `benchmark`: Performance testing
  - `export`: TorchScript/ONNX model export
  - Global model instance management for server mode

### Model Architecture
- `models/flow_lm.py`: Transformer-based flow language model
- `models/mimi.py`: Neural audio codec encoder/decoder
- `modules/transformer.py`: Streaming transformer with RoPE
- `modules/stateful_module.py`: Base class for stateful streaming

## CONVENTIONS

### Package Organization
- Only `TTSModel` is exported from `__init__.py` (keep internals private)
- Beartype runtime type checking enabled for entire package
- Thread-unsafe: Server does not support concurrent requests

### Audio Processing
- Default sample rate: 24000 Hz
- Frame rate: 12.5 Hz (80ms per frame)
- Audio tensors: 1D float32 PCM arrays
- Streaming via generators for low-latency output

### Model Loading
- Weights from HuggingFace via `download_if_necessary()`
- Configs from YAML in `config/` directory
- LRU cache for voice prompts (configurable via `POCKET_TTS_PROMPT_CACHE_SIZE`)

### PyTorch Settings
- `torch.set_num_threads(1)` for optimal CPU performance
- Device defaults to CPU (GPU provides no speedup for this small model)
- Batch size always 1 (no batching support)

### Rust Extensions
- Optional build: `cd training/rust_exts/audio_ds && cargo build --release`
- Library detection: searches multiple paths, graceful degradation
- ctypes FFI with automatic Python fallback for all functions

## ANTI-PATTERNS

### ❌ Don't
- Add concurrent request handling (not thread-safe)
- Modify public API exports without updating __init__.py
- Use GPU (no speedup, complicates deployment)
- Increase batch size (not supported by streaming architecture)
- Skip beartype type checking (enabled package-wide)
- Modify streaming state handling without understanding StatefulModule

### ⚠️ Avoid
- Calling `load_model()` or `get_state_for_audio_prompt()` repeatedly (cache them)
- Using PyTorch 2.4.0 (produces incorrect audio; requires 2.5+)
- Python <3.10 or >=3.15 (unsupported versions)
- Modifying torch thread settings (set to 1 for optimal performance)
