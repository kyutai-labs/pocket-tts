# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

pocket-tts-candle is a Rust/Candle port of the pocket-tts CPU-based text-to-speech (TTS) model. It aims for high performance and low latency on CPU.

**Key Architecture Components:**
- **FlowLM**: Transformer-based flow language model that generates latent representations from text using Lagrangian Self Distillation (LSD).
- **Mimi (SEANet)**: Neural audio codec that compresses/decompresses audio to/from latent representations (v0.1.0).
- **Conditioners**: Text processing via SentencePiece tokenizer.
- **Streaming Architecture**: The entire pipeline supports streaming generation via stateful modules.
- **Web API**: Axum-based server for HTTP API access with a built-in web interface.

## Repository Structure

The repository is organized as a Rust workspace at the root level:

- `crates/pocket-tts`: Core library containing model implementations.
- `crates/pocket-tts-cli`: CLI interface and Axum API / Static server.
- `crates/pocket-tts-bindings`: Python bindings using PyO3.
- `assets/`: Centralized reference assets (.wav, .safetensors).
- `python-reference/`: Original Python codebase for reference and parity testing.

## Common Commands

### Setup and Development
```powershell
# Build the project
cargo build --release

# Run all tests (including integration and parity)
$env:HF_TOKEN="your_token_here"; cargo test --release --all-targets

# Run the CLI
cargo run --release -p pocket-tts-cli -- --help

# Serve the Web UI
cargo run --release -p pocket-tts-cli -- serve
```

### Benchmarking
```powershell
# Run benchmarks
cargo bench --release
```

## Code Structure (Rust)

### Library (`crates/pocket-tts/src/`)

- `tts_model.rs`: Orchestrates the TTS pipeline.
- `models/`: Mimi and FlowLM implementations.
- `modules/`: Transformer, MLP, Rope, etc.
- `audio.rs`: Audio I/O and Resampling (robust to any input rate).
- `weights.rs`: HuggingFace weight downloading and management.
- `config.rs`: YAML configuration parsing.

### CLI & Server (`crates/pocket-tts-cli/src/`)

- `main.rs`: CLI entry point using `clap`.
- `server/`: Axum router and handlers.
- `static/`: HTML/JS for the Web UI.

## Numerical Parity

We maintain strict numerical parity with the Python implementation where possible.
- Parity tests are located in `crates/pocket-tts/tests/parity_tests.rs`.
- Reference tensors are in `assets/*.safetensors`.
- The Rust resampler is now robust and theoretically superior to the Python reference, so `ref.wav` (regardless of rate) should be used.

## Development Workflow

1. **Always use --release**: Performance is critical; never benchmark or test audio quality in debug mode.
2. **Streaming first**: All components must support stateful streaming.
3. **CPU Optimization**: Focus on cache-friendly operations and Candle's SIMD capabilities.
4. **Resampling**: The code now handles non-24kHz input automatically via robustness improvements in `audio.rs`.

## Model Weights

Weights are downloaded from HuggingFace Hub:
- Model weights: `hf://kyutai/pocket-tts/tts_b6369a24.safetensors`
- Tokenizer: `hf://kyutai/pocket-tts/tokenizer.model`

## Common Gotchas

1. **HF_TOKEN**: Required for gated weights (`kyutai/pocket-tts`).
2. **Config Discovery**: `find_config_path` looks in `crates/pocket-tts/config` and fallback locations.
3. **MKL/Accelerate**: Ensure appropriate BLAS backends are enabled for maximum performance.
