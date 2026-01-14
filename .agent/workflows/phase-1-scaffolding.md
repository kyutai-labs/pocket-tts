---
description: Phase 1 - Project Scaffolding & Basic Tensors for Pocket TTS Candle Port
---

# Phase 1: Project Scaffolding & Basic Tensors

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the project architecture and Python reference implementation.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

> [!TIP]
> **Always use `uv` when running Python commands** for reference or comparison:
> ```bash
> uv run pocket-tts generate ...     # CLI commands
> uv run python -c "..."             # Inline Python
> uv run pytest ...                  # Running tests
> ```

---

## Prerequisites

Before starting, ensure the Python reference implementation is ready for comparison:

```bash
# Sync uv environment (downloads Python 3.10+ and installs all dependencies)
uv sync

# Verify Python version (should be 3.10+)
uv run python --version

# Verify pocket-tts CLI works
uv run pocket-tts --help

# Test the reference implementation generates audio correctly
uv run pocket-tts generate --text "Hello world" --output test_reference.wav
```

> [!NOTE]
> The project uses `python-preference = "only-managed"` in `pyproject.toml`, so `uv` will download and manage the correct Python version automatically. You don't need to install Python separately.

---

## Overview

This phase sets up the Rust project structure and implements core utilities needed for the Pocket TTS Candle port.

## Project Structure

```
candle/
├── Cargo.toml          # Workspace root
├── crates/
│   ├── pocket-tts/     # Main library crate
│   └── pocket-tts-cli/ # CLI binary crate
```

Use **Cargo workspace** for best practices. The workspace Cargo.toml should define shared dependencies and features.

---

## Test-Driven Development Philosophy

> [!NOTE]
> The goal is to write **working code on the first attempt**. TDD is not about intentionally writing failing tests—it's about:
> 1. **Defining expectations clearly** before implementation
> 2. **Validating correctness** as you build
> 3. **Maintaining 80% test coverage** across all modules

Write tests that you expect to pass once implementation is complete. Think through the implementation before coding.

---

## Tasks

### 1. Initialize Rust Workspace
// turbo
```bash
cd candle && cargo init --name pocket-tts-workspace
```

Create workspace structure:
- `candle/Cargo.toml` - workspace manifest
- `candle/crates/pocket-tts/` - core library
- `candle/crates/pocket-tts-cli/` - CLI binary

**Key dependencies:**
- `candle-core`, `candle-nn` - ML framework
- `safetensors` - weight loading
- `hf-hub` - HuggingFace downloads
- `anyhow`, `thiserror` - error handling

### 2. Implement Safetensors Weight Loading Utility

**Write tests first:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_tensor_from_safetensors() {
        // Test loading a known tensor shape
    }
    
    #[test]
    fn test_missing_key_error() {
        // Test graceful error on missing weight
    }
}
```

**Then implement** `src/weights.rs`:
- Function to load model weights from `.safetensors` files
- Support for HuggingFace Hub paths (`hf://`)
- Proper error handling with descriptive messages

### 3. Port Basic Utility Functions

**Reference:** `pocket_tts/data/audio.py` and `pocket_tts/data/audio_utils.py`

**Write tests first:**
```rust
#[test]
fn test_audio_normalization() {
    // Peak normalization to [-1, 1]
}

#[test]
fn test_resample_audio() {
    // Verify sample rate conversion
}
```

**Then implement** `src/audio.rs`:
- Audio normalization (peak normalization)
- Sample rate utilities
- WAV file I/O helpers

---

## Voice Cloning Test File

Use `ref.wav` in the project root for any voice cloning tests.

---

## Verification Checklist

Before completing this phase:

- [ ] `cargo build` succeeds with no errors
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` reports no warnings
- [ ] Test coverage ≥80% for new modules
- [ ] Weight loading works with HuggingFace model

**Run coverage:**
```bash
cargo tarpaulin --out Html --output-dir coverage/
```

---

## Definition of Done

- [ ] Workspace structure created correctly
- [ ] All tests pass
- [ ] 80% code coverage achieved
- [ ] No clippy warnings
- [ ] Code documented with rustdoc comments
