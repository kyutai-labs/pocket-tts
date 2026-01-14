---
description: Phase 2 - Core Model Components (Mimi, Transformer, Seanet, Flow-LM) for Pocket TTS Candle Port
---

# Phase 2: Core Model Components

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the model architecture.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

## Overview

This phase ports the individual neural network modules from Python/PyTorch to Rust/Candle.

## Project Structure

```
candle/crates/pocket-tts/src/
├── lib.rs
├── models/
│   ├── mod.rs
│   ├── mimi.rs        # Mimi encoder/decoder
│   ├── seanet.rs      # SEANet architecture
│   ├── transformer.rs # Streaming transformer
│   └── flow_lm.rs     # Flow language model
├── modules/
│   ├── mod.rs
│   ├── rope.rs        # Rotary position embeddings
│   ├── mlp.rs         # MLP with AdaLN
│   └── attention.rs   # Multi-head attention
└── conditioners/
    └── text.rs        # LUT conditioner
```

---

## Test-Driven Development Philosophy

> [!NOTE]
> The goal is to write **working code on the first attempt**. Design tests around:
> 1. **Shape validation** - Ensure tensor dimensions match Python
> 2. **Numerical parity** - Compare outputs with Python reference
> 3. **Edge cases** - Handle empty inputs, single tokens, etc.

Target: **80% test coverage** across all modules.

---

## Tasks

### 1. Implement Mimi Encoder/Decoder

**Reference:** `pocket_tts/modules/seanet.py` and moshi's Mimi implementation

**Write tests first:**
```rust
#[test]
fn test_mimi_encoder_output_shape() {
    // Input: [batch, channels, samples]
    // Output: [batch, latent_dim, frames]
}

#[test]
fn test_mimi_decoder_output_shape() {
    // Input: [batch, latent_dim, frames]
    // Output: [batch, channels, samples]
}

#[test]
fn test_mimi_roundtrip() {
    // encode -> decode should preserve structure
}
```

**Implementation notes:**
- Port SEANet encoder/decoder architecture
- Implement streaming state management
- Use `candle_nn::Conv1d` for convolution layers

### 2. Implement Transformer & Attention Modules

**Reference:** `pocket_tts/modules/transformer.py`

**Write tests first:**
```rust
#[test]
fn test_streaming_attention_kv_cache() {
    // Verify KV cache accumulates correctly
}

#[test]
fn test_rope_embeddings() {
    // Compare RoPE output with Python reference
}

#[test]
fn test_transformer_block_shape() {
    // Input/output shapes should match
}
```

**Implementation notes:**
- `StreamingMultiheadAttention` with KV cache
- Rotary Position Embeddings (RoPE)
- Layer normalization (RMSNorm preferred in Candle)

### 3. Implement SEANet Architecture

**Reference:** `pocket_tts/modules/seanet.py`

**Write tests first:**
```rust
#[test]
fn test_seanet_encoder_downsample() {
    // Verify correct temporal downsampling
}

#[test]
fn test_seanet_decoder_upsample() {
    // Verify correct temporal upsampling
}
```

**Implementation notes:**
- Residual units with skip connections
- Transposed convolutions for upsampling
- ELU activation functions

### 4. Implement Flow-LM & Conditioners

**Reference:** `pocket_tts/models/flow_lm.py` and `pocket_tts/conditioners/text.py`

**Write tests first:**
```rust
#[test]
fn test_lut_conditioner_embedding() {
    // Token IDs -> embedding vectors
}

#[test]
fn test_flow_lm_forward() {
    // Shape validation for flow prediction
}

#[test]
fn test_flow_matching_step() {
    // Single flow matching decode step
}
```

**Implementation notes:**
- LUT (Lookup Table) conditioner with SentencePiece tokenizer
- Flow matching with configurable decode steps
- AdaLN conditioning in MLP layers

---

## Voice Cloning Test File

Use `ref.wav` in the project root for integration tests that require audio input.

---

## Numerical Parity Testing

For each module, create parity tests that compare Rust output with Python:

```rust
#[test]
fn test_transformer_parity() {
    // Load same weights in both Python and Rust
    // Run identical input through both
    // Assert outputs match within f32 tolerance (1e-5)
}
```

---

## Verification Checklist

Before completing this phase:

- [ ] `cargo build` succeeds with no errors
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` reports no warnings
- [ ] Test coverage ≥80% for all model modules
- [ ] Numerical parity verified for at least one module

---

## Definition of Done

- [ ] All model components implemented
- [ ] All tests pass
- [ ] 80% code coverage achieved
- [ ] Numerical parity with Python verified
- [ ] Streaming state management working
- [ ] Code documented with rustdoc comments
