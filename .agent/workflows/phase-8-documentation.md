---
description: Phase 8 - Documentation & Release Preparation for Pocket TTS Candle Port
---

# Phase 8: Documentation & Release Preparation

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the project context.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

## Overview

This phase focuses on comprehensive documentation, examples, and preparing the crate for publication to crates.io.

---

## Test-Driven Development Philosophy

> [!NOTE]
> Documentation is tested too! Use `cargo test --doc` to verify all doc examples compile and run correctly.

Target: **100% public API documentation** with working examples.

---

## Tasks

### 1. API Documentation (rustdoc)

**Write comprehensive rustdoc comments for all public items:**

```rust
/// A high-performance text-to-speech model using the Candle ML framework.
///
/// # Example
///
/// ```rust
/// use pocket_tts::TTSModel;
///
/// # fn main() -> anyhow::Result<()> {
/// let model = TTSModel::load()?;
/// let audio = model.generate("Hello, world!")?;
/// audio.save("output.wav")?;
/// # Ok(())
/// # }
/// ```
///
/// # Voice Cloning
///
/// ```rust
/// use pocket_tts::{TTSModel, VoiceState};
///
/// # fn main() -> anyhow::Result<()> {
/// let model = TTSModel::load()?;
/// let voice = VoiceState::from_audio_file("speaker.wav")?;
/// let audio = model.generate_with_voice("Hello!", &voice)?;
/// # Ok(())
/// # }
/// ```
pub struct TTSModel { /* ... */ }
```

**Verify doc tests pass:**
```bash
cargo test --doc
```

### 2. README.md

Create `candle/README.md` with:

```markdown
# pocket-tts (Rust/Candle)

A high-performance, CPU-optimized text-to-speech library written in pure Rust.

[![Crates.io](https://img.shields.io/crates/v/pocket-tts.svg)](https://crates.io/crates/pocket-tts)
[![Documentation](https://docs.rs/pocket-tts/badge.svg)](https://docs.rs/pocket-tts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- 🚀 **Fast**: 2-5x faster than the Python implementation
- 🎯 **CPU-Optimized**: Runs efficiently on consumer hardware
- 🎤 **Voice Cloning**: Clone any voice from a short audio sample
- 📡 **Streaming**: Real-time audio generation with <100ms latency
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI TTS API

## Installation

### As a Library
```toml
[dependencies]
pocket-tts = "0.1"
```

### As a CLI
```bash
cargo install pocket-tts-cli
```

## Quick Start

### Library Usage
```rust
use pocket_tts::TTSModel;

fn main() -> anyhow::Result<()> {
    let model = TTSModel::load()?;
    let audio = model.generate("Hello, world!")?;
    audio.save("output.wav")?;
    Ok(())
}
```

### CLI Usage
```bash
# Generate audio
pocket-tts generate --text "Hello, world!" --output hello.wav

# With voice cloning
pocket-tts generate --text "Hello!" --voice speaker.wav --output cloned.wav

# Start API server
pocket-tts serve --port 8000
```

## Voice Cloning

```rust
use pocket_tts::{TTSModel, VoiceState};

let model = TTSModel::load()?;
let voice = VoiceState::from_audio_file("ref.wav")?;
let audio = model.generate_with_voice("Clone this voice!", &voice)?;
```

## Streaming

```rust
use pocket_tts::TTSModel;

let model = TTSModel::load()?;
for chunk in model.generate_stream("Long text here...")? {
    // Play or process each audio chunk
    play_audio(&chunk)?;
}
```

## API Server

The included API server is compatible with the OpenAI TTS API:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "pocket-tts", "input": "Hello!", "voice": "default"}' \
  --output speech.wav
```

## Performance

Benchmarked on [YOUR CPU HERE]:

| Metric | Python | Rust | Speedup |
|--------|--------|------|---------|
| Short text (5 words) | X.XXs | X.XXs | X.Xx |
| Medium text (20 words) | X.XXs | X.XXs | X.Xx |
| Long text (50 words) | X.XXs | X.XXs | X.Xx |
| First chunk latency | XXXms | XXms | X.Xx |

## License

MIT License - see [LICENSE](LICENSE) for details.
```

### 3. Examples Directory

Create working examples in `candle/examples/`:

| File | Description |
|------|-------------|
| `basic.rs` | Minimal TTS generation |
| `voice_clone.rs` | Voice cloning from audio file |
| `streaming.rs` | Real-time streaming generation |
| `server.rs` | Custom API server setup |
| `benchmark.rs` | Performance benchmarking |

**Example: `basic.rs`**
```rust
//! Basic text-to-speech example
//!
//! ```bash
//! cargo run --example basic -- --text "Hello, world!"
//! ```

use anyhow::Result;
use clap::Parser;
use pocket_tts::TTSModel;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    text: String,
    
    #[arg(short, long, default_value = "output.wav")]
    output: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("Loading model...");
    let model = TTSModel::load()?;
    
    println!("Generating audio for: {}", args.text);
    let audio = model.generate(&args.text)?;
    
    println!("Saving to: {}", args.output);
    audio.save(&args.output)?;
    
    println!("Done!");
    Ok(())
}
```

### 4. CHANGELOG.md

Create `candle/CHANGELOG.md`:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - YYYY-MM-DD

### Added
- Initial release of pocket-tts Rust port
- Full feature parity with Python implementation
- Voice cloning support
- Streaming audio generation
- CLI with `generate` and `serve` commands
- OpenAI-compatible API server
- WASM support (feature flag)
- int8 quantization support

### Performance
- 2-5x faster than Python on CPU
- <100ms first chunk latency
- <200MB memory usage
```

### 5. Crates.io Preparation

**Update `candle/crates/pocket-tts/Cargo.toml`:**

```toml
[package]
name = "pocket-tts"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your@email.com>"]
description = "High-performance text-to-speech library using Candle ML framework"
documentation = "https://docs.rs/pocket-tts"
repository = "https://github.com/yourusername/pocket-tts"
license = "MIT"
keywords = ["tts", "text-to-speech", "voice", "audio", "ml"]
categories = ["multimedia::audio", "science::machine-learning"]
readme = "README.md"

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

**Verify package is ready:**
```bash
cargo publish --dry-run
```

### 6. Architecture Documentation

Create `candle/docs/ARCHITECTURE.md`:

```markdown
# Architecture

## Module Overview

```
pocket-tts/
├── src/
│   ├── lib.rs           # Public API
│   ├── tts_model.rs     # Main TTSModel struct
│   ├── models/
│   │   ├── mimi.rs      # Mimi encoder/decoder
│   │   ├── flow_lm.rs   # Flow language model
│   │   └── ...
│   ├── modules/
│   │   ├── transformer.rs
│   │   ├── attention.rs
│   │   └── ...
│   └── audio/
│       ├── wav.rs       # WAV I/O
│       └── stream.rs    # Streaming utilities
```

## Data Flow

```
Text Input
    │
    ▼
┌──────────────┐
│  Tokenizer   │  SentencePiece
└──────────────┘
    │
    ▼
┌──────────────┐
│ LUT Embed    │  Token → Vector
└──────────────┘
    │
    ▼
┌──────────────┐
│  Flow-LM     │  Transformer + Flow Matching
└──────────────┘
    │
    ▼
┌──────────────┐
│ Mimi Decoder │  Latents → Audio
└──────────────┘
    │
    ▼
Audio Output
```

## Key Components

### TTSModel
Main entry point. Orchestrates the full pipeline.

### FlowLM
Transformer-based model using flow matching for latent generation.

### Mimi
Neural audio codec for encoding/decoding audio.

### StreamingState
Manages KV cache and internal state for streaming generation.
```

---

## Voice Cloning Test File

Use `ref.wav` in the project root for documentation examples.

---

## Verification Checklist

Before completing this phase:

- [ ] `cargo doc --no-deps --open` builds without warnings
- [ ] `cargo test --doc` passes all doc tests
- [ ] All public items have documentation
- [ ] README.md is complete and accurate
- [ ] All examples compile and run
- [ ] CHANGELOG.md is up to date
- [ ] `cargo publish --dry-run` succeeds
- [ ] Performance numbers in README are filled in

---

## Manual Verification Steps

1. **Build and review docs:**
```bash
cargo doc --no-deps --open
# Review all public API documentation
```

2. **Run all examples:**
```bash
cargo run --example basic -- --text "Test"
cargo run --example voice_clone -- --voice ref.wav --text "Test"
cargo run --example streaming -- --text "Test"
```

3. **Check doc tests:**
```bash
cargo test --doc
```

---

## Definition of Done

- [ ] 100% public API documented
- [ ] All doc tests pass
- [ ] README.md complete with examples
- [ ] CHANGELOG.md created
- [ ] All examples working
- [ ] Architecture documentation complete
- [ ] Ready for crates.io publication
- [ ] Performance benchmarks documented with real numbers
