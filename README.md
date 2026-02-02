# Pocket TTS (Rust/Candle)

A native Rust port of [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) using [Candle](https://github.com/huggingface/candle) for tensor operations.

Text-to-speech that runs entirely on CPU—no Python, no GPU required.

## Features

- **Pure Rust** - No Python runtime, just a single binary
- **CPU-only** - Runs on CPU, no GPU required
- **Metal Acceleration** - Build with `--features metal` for hardware acceleration on macOS
- **int8 Quantization** - Significant speedup and smaller memory footprint
- **Streaming** - Full-pipeline stateful streaming (FlowLM + Mimi) for zero-latency audio
- **Project Structure** - Clean, modular workspace design
- **WebAssembly** - Run the full model in any modern web browser
- **Pause Handling** - Support for natural pauses and explicit `[pause:Xms]` syntax
- **HTTP API** - REST API server with OpenAI-compatible endpoint
- **Web UI** - Built-in web interface (React/Vite) for interactive use
- **Flexible Builds** - Use `--no-default-features` for a "lite" build without web UI assets
- **Python Bindings** - Use the Rust implementation from Python for improved performance

## Quick Start

```bash
# Build with default features (includes Web UI assets)
cargo build --release

# Build "lite" version (no Web UI assets, API only)
cargo build --release --no-default-features

# Build with Metal support (macOS only)
cargo build --release --features metal
```

### Generate audio

```bash
# Using default voice
cargo run --release --package pocket-tts-cli -- generate --text "Hello, world!"

# Using Metal acceleration (if enabled)
cargo run --release --features metal --package pocket-tts-cli -- generate --text "Hello, world!" --use-metal

# Using a custom voice (WAV file)
cargo run --release --package pocket-tts-cli -- generate \
    --text "Hello, world!" \
    --voice ./my_voice.wav \
    --output output.wav

# Using a predefined voice
cargo run --release --package pocket-tts-cli -- generate --voice alba
```

### Start the HTTP server

```bash
cargo run --release -p pocket-tts-cli -- serve
# Navigate to http://localhost:8000

### WebAssembly Demo

The browser demo features a "Zero-Setup" experience with an **embedded tokenizer and config**.

#### 1. Build the WASM package
From the `crates/pocket-tts` directory:
```bash
wasm-pack build --target web --out-dir pkg . -- --features wasm
```

#### 2. Launch the demo server
```bash
cargo run --release -p pocket-tts-cli -- wasm-demo
```
- Navigate to `http://localhost:8080`
- Provides built-in voice cloning and Hugging Face Hub integration.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pocket-tts = { path = "crates/pocket-tts" }
```

## Library Usage

```rust
use pocket_tts::TTSModel;
use anyhow::Result;

fn main() -> Result<()> {
    // Load the model
    let model = TTSModel::load("b6369a24")?;
    
    // Get voice state from audio file
    let voice_state = model.get_voice_state("voice.wav")?;
    
    // Generate audio
    let audio = model.generate("Hello, world!", &voice_state)?;
    
    // Save to file
    pocket_tts::audio::write_wav("output.wav", &audio, model.sample_rate as u32)?;
    
    Ok(())
}
```

### Streaming Generation

```rust
use pocket_tts::TTSModel;

let model = TTSModel::load("b6369a24")?;
let voice_state = model.get_voice_state("voice.wav")?;

// Stream audio chunks as they're generated
for chunk in model.generate_stream("Long text here...", &voice_state) {
    let audio_chunk = chunk?;
    // Process or play each chunk
}
```

### Custom Parameters

```rust
let model = TTSModel::load_with_params(
    "b6369a24",     // variant
    0.7,            // temperature (higher = more variation)
    1,              // lsd_decode_steps (more = better quality, slower)
    -4.0,           // eos_threshold (more negative = longer audio)
)?;
```

### HuggingFace token
If you're using a model that has to be downloaded from huggingface you will need a token in the `HF_TOKEN` environment variable

## CLI Reference

### `generate` command

Generate audio from text and save to a WAV file.

```
pocket-tts generate [OPTIONS]

Options:
  -t, --text <TEXT>              Text to synthesize [default: greeting]
  -v, --voice <VOICE>            Voice: predefined name, .wav file, or .safetensors
  -o, --output <PATH>            Output file [default: output.wav]
      --variant <VARIANT>        Model variant [default: b6369a24]
      --temperature <FLOAT>      Sampling temperature [default: 0.7]
      --lsd-decode-steps <INT>   LSD decode steps [default: 1]
      --eos-threshold <FLOAT>    EOS threshold [default: -4.0]
      --stream                   Stream raw PCM to stdout
  -q, --quiet                    Suppress output
      --use-metal                Use Metal acceleration (macOS)
```

**Predefined voices:** `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

### `serve` command

Start an HTTP API server with web interface.

```
pocket-tts serve [OPTIONS]

Options:
      --host <HOST>              Bind address [default: 127.0.0.1]
  -p, --port <PORT>              Port number [default: 8000]
      --voice <VOICE>            Default voice [default: alba]
      --variant <VARIANT>        Model variant [default: b6369a24]
      --temperature <FLOAT>      Temperature [default: 0.7]
      --lsd-decode-steps <INT>   LSD steps [default: 1]
      --eos-threshold <FLOAT>    EOS threshold [default: -4.0]

## Python Bindings

The Rust implementation can be used as a Python module for improved performance (~1.34x speedup).

### Installation

Requires [maturin](https://github.com/PyO3/maturin).

```bash
cd crates/pocket-tts-bindings
uvx maturin develop --release
```

### Usage

```python
import pocket_tts_bindings

# Load the model
model = pocket_tts_bindings.PyTTSModel.load("b6369a24")

# Generate audio
audio_samples = model.generate(
    "Hello from Rust!",
    "path/to/voice.wav"
)
```

### `wasm-demo` command

Serve the WASM package and browser demo.

```
pocket-tts wasm-demo [OPTIONS]

Options:
  -p, --port <PORT>              Port number [default: 8080]
```
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `GET` | `/health` | Health check |
| `POST` | `/generate` | Generate audio (JSON) |
| `POST` | `/stream` | Streaming generation |
| `POST` | `/tts` | Python-compatible (multipart) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible |

### Example API call

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello world", "voice": "alba"}' \
  --output output.wav
```

## Project Structure

```
candle/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── pocket-tts/         # Core library
│   │   ├── src/
│   │   │   ├── lib.rs          # Public API
│   │   │   ├── tts_model.rs    # Main TTSModel
│   │   │   ├── wasm.rs         # WASM entry points
│   │   │   ├── audio.rs        # WAV I/O, resampling
│   │   │   ├── quantize.rs     # int8 quantization
│   │   │   ├── pause.rs        # Pause/silence handling
│   │   │   ├── config.rs       # YAML config types
│   │   │   ├── models/         # Neural network models
│   │   │   │   ├── flow_lm.rs      # Flow language model
│   │   │   │   ├── mimi.rs         # Audio codec
│   │   │   │   ├── seanet.rs       # Encoder/decoder
│   │   │   │   └── transformer.rs  # Transformer blocks
│   │   │   └── modules/        # Reusable components
│   │   │       ├── attention.rs    # Multi-head attention
│   │   │       ├── conv.rs         # Convolution layers
│   │   │       ├── mlp.rs          # MLP with AdaLN
│   │   │       └── rope.rs         # Rotary embeddings
│   │   ├── tests/
│   │   └── benches/
│   └── pocket-tts-cli/     # CLI binary
│       ├── src/
│       │   ├── main.rs         # Entry point
│       │   ├── commands/       # generate, serve
│       │   ├── server/         # Axum HTTP server
│       │   └── voice.rs        # Voice resolution
│       └── web/                # React/Vite Web UI source
└── docs/                   # Documentation
```

## Architecture

The Rust port mirrors the Python implementation:

1. **Text Conditioning**: SentencePiece tokenizer → embedding lookup table
2. **FlowLM Transformer**: Generates latent representations from text using Lagrangian Self Distillation (LSD)
3. **Mimi Decoder**: Converts latents to audio via SEANet decoder

### Key differences from Python

- Uses [Candle](https://github.com/huggingface/candle) instead of PyTorch
- **Full-pipeline stateful streaming** (KV-caching for Transformer, overlap-add for Mimi)
- Polyphase resampling via [rubato](https://crates.io/crates/rubato) (matches scipy)
- Compiled to native code—no JIT, no Python overhead

## GPU Acceleration

### Metal (macOS)

Build with Metal support for hardware acceleration on Apple Silicon:

```bash
cargo build --release --features metal
```

**Current Status:** Metal support provides ~2x speedup over CPU on Apple Silicon:

| Backend | RTF | Speed | Notes |
|---------|-----|-------|-------|
| CPU | ~0.33 | 3x real-time | Default, cross-platform |
| Metal | ~0.16 | 6x real-time | Requires `--features metal` |

*Benchmarks verified on Apple M4 Max*

For best Apple Silicon performance (~8x real-time), consider the community [MLX implementation](https://github.com/jishnuvenugopal/pocket-tts-mlx).

### CUDA (Linux/Windows)

Build with CUDA support:

```bash
cargo build --release --features cuda
```

**Note:** CUDA support requires a compatible NVIDIA GPU and CUDA toolkit installed.

## Benchmarking

Run benchmarks to measure performance on your hardware:

```bash
cargo bench -p pocket-tts
```

> **Note**: Performance may differ from the Python implementation. Candle is optimized for portability rather than raw speed.

### Performance Results

Benchmarks run on User Hardware (vs Python baseline):

- **Short Text**: ~6.20x speedup
- **Medium Text**: ~3.47x speedup
- **Long Text**: ~3.33x speedup
- **Latency**: ~80ms to first audio chunk (optimized)

Rust is consistently **>3.1x faster** than the optimized Python implementation.

### Cross-Implementation Comparison

| Implementation | RTF | Speed vs Real-Time | Platform |
|----------------|-----|-------------------|----------|
| PyTorch CPU (official) | ~0.25 | 4x faster | Cross-platform |
| **Rust/Candle CPU** | ~0.33 | **3x faster** | Cross-platform |
| **Rust/Candle Metal** | ~0.16 | **6x faster** | macOS (Apple Silicon) |
| [MLX (Apple Silicon)](https://github.com/jishnuvenugopal/pocket-tts-mlx) | ~0.13 | 8x faster | macOS only |

*RTF = Real-Time Factor (lower is better, <1.0 means faster than real-time)*
*All benchmarks verified on Apple M4 Max*

## Numerical Parity

The Rust implementation achieves strong numerical parity with Python:

| Component | Max Difference | Status |
|-----------|----------------|--------|
| Input audio | 0 | ✅ Perfect |
| SEANet Decoder | ~0.000004 | ✅ Excellent |
| Decoder Transformer | ~0.002 | ✅ Good |
| Voice Conditioning | ~0.004 | ✅ Good |
| Full Pipeline | ~0.06 | ✅ Acceptable |

Run parity tests:

```bash
cargo test -p pocket-tts parity --release
```

## Dependencies

Core dependencies (see full list in `Cargo.toml`):

- [`candle-core`](https://crates.io/crates/candle-core) - Tensor operations
- [`candle-nn`](https://crates.io/crates/candle-nn) - Neural network layers
- [`safetensors`](https://crates.io/crates/safetensors) - Weight loading
- [`hf-hub`](https://crates.io/crates/hf-hub) - HuggingFace downloads
- [`tokenizers`](https://crates.io/crates/tokenizers) - Tokenization
- [`rubato`](https://crates.io/crates/rubato) - Audio resampling
- [`hound`](https://crates.io/crates/hound) - WAV I/O
- [`axum`](https://crates.io/crates/axum) - HTTP server
- [`clap`](https://crates.io/crates/clap) - CLI parsing

## License

MIT License - see [LICENSE](../LICENSE)

## Acknowledgements

- **[SmilyOrg](https://github.com/SmilyOrg)** for the [Docker implementation](https://github.com/babybirdprd/pocket-tts/pull/1) that enables completely offline operation.

## Related

- [Pocket TTS (Python)](https://github.com/kyutai-labs/pocket-tts) - Original implementation
- [Candle](https://github.com/huggingface/candle) - Rust ML framework
- [Kyutai](https://kyutai.org) - Research lab
