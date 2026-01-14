---
description: Phase 3 - TTS Model Integration & Numerical Parity for Pocket TTS Candle Port
---

# Phase 3: TTS Model Integration & Numerical Parity

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the full TTS pipeline.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

## Overview

This phase integrates all model components into the high-level `TTSModel` struct and verifies numerical parity with the Python reference implementation.

## Project Structure

```
candle/crates/pocket-tts/src/
├── lib.rs              # Public API exports
├── tts_model.rs        # Main TTSModel struct
├── config.rs           # Configuration types
├── voice_state.rs      # Voice cloning state
└── ...
```

---

## Test-Driven Development Philosophy

> [!NOTE]
> **Numerical parity is the primary success criterion.** Tests should verify that Rust outputs match Python outputs within floating-point tolerances (typically 1e-5 for f32).

Target: **80% test coverage** with emphasis on end-to-end parity tests.

---

## Tasks

### 1. Implement High-Level TTSModel Struct

**Reference:** `pocket_tts/models/tts_model.py`

**Write tests first:**
```rust
#[test]
fn test_tts_model_initialization() {
    // Load weights and verify model is ready
}

#[test]
fn test_tts_model_config_loading() {
    // Parse YAML config correctly
}

#[test]
fn test_tts_model_generate_audio() {
    // Generate audio from text, verify output shape
}
```

**Implementation notes:**
- `TTSModel::load()` - Download weights from HuggingFace, initialize all components
- `TTSModel::generate()` - Non-streaming generation
- Configuration loading from YAML (`config/b6369a24.yaml`)

**Key methods from Python:**
```python
# Reference: pocket_tts/models/tts_model.py
load_model()                    # -> TTSModel::load()
get_state_for_audio_prompt()    # -> get_voice_state()
generate_audio_stream()         # -> generate_stream() (Phase 4)
```

### 2. Implement Voice Cloning (State Loading)

**Reference:** `pocket_tts/models/tts_model.py::get_state_for_audio_prompt`

**Write tests first using `ref.wav`:**
```rust
#[test]
fn test_voice_cloning_from_audio() {
    let audio = load_wav("ref.wav")?;
    let state = model.get_voice_state(&audio)?;
    assert!(state.latents.dims()[0] > 0);
}

#[test]
fn test_voice_state_caching() {
    // Same audio should return cached state
}

#[test]
fn test_voice_cloning_parity() {
    // Compare voice state with Python output
}
```

**Implementation notes:**
- Encode audio through Mimi encoder
- Extract speaker embeddings
- Cache voice states (like Python's `lru_cache`)

### 3. Numerical Parity Verification

> [!CAUTION]
> This is the most critical task. The Rust port MUST produce audio that matches the Python output within floating-point tolerances.

**Create comprehensive parity tests:**

```rust
#[test]
fn test_full_pipeline_parity() {
    // 1. Load same weights in Python and Rust
    // 2. Use identical text input
    // 3. Use identical voice prompt (ref.wav)
    // 4. Compare generated audio samples
    
    let rust_audio = rust_model.generate("Hello world", &voice_state)?;
    let python_audio = load_reference_output("hello_world_ref.wav");
    
    // Allow small tolerance for floating-point differences
    assert_tensors_close(&rust_audio, &python_audio, 1e-5);
}
```

**Parity test levels:**
1. **Layer-level** - Individual modules (Mimi, Transformer, etc.)
2. **Component-level** - Model subgraphs
3. **End-to-end** - Full text-to-audio pipeline

**Create reference outputs:**
```bash
# In Python, save reference outputs for parity testing (always use uv)
uv run python -c "
from pocket_tts import TTSModel
model = TTSModel.load_model()
audio = model.generate('Hello world')
# Save audio for Rust comparison
"
```

---

## Voice Cloning Test File

Use `ref.wav` in the project root as the standard voice prompt for all tests.

---

## Verification Checklist

Before completing this phase:

- [ ] `cargo build` succeeds with no errors
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` reports no warnings
- [ ] Test coverage ≥80% for integration code
- [ ] End-to-end numerical parity verified
- [ ] Voice cloning produces valid output
- [ ] Audio output is audibly correct

---

## Manual Verification Steps

1. **Generate test audio:**
```bash
cargo run --example generate -- --text "Hello, this is a test." --voice ref.wav --output test.wav
```

2. **Listen to output** and verify it sounds correct

3. **Compare with Python:**
```bash
uv run pocket-tts generate --text "Hello, this is a test." --voice ref.wav --output python_test.wav
# Compare test.wav with python_test.wav
```

---

## Definition of Done

- [ ] TTSModel struct fully implemented
- [ ] Voice cloning working
- [ ] All tests pass
- [ ] 80% code coverage achieved
- [ ] Numerical parity verified (within 1e-5 tolerance)
- [ ] Audio output is audibly correct
- [ ] Code documented with rustdoc comments
