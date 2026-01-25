# Pocket TTS Architecture

## Overview

Pocket TTS is a lightweight, CPU-optimized text-to-speech system built around a neural architecture that combines flow-based generation with neural audio compression.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pocket TTS                              │
├─────────────────────────────────────────────────────────────────┤
│  Python API Layer                                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ TTSModel    │ │ Audio I/O   │ │ Rust Audio  │              │
│  │             │ │ Functions   │ │ Processing  │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  Core Model Components                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ FlowLM      │ │ Mimi Model  │ │ Conditioner │              │
│  │ (Generation)│ │ (Codec)     │ │ (Text)      │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  Neural Sub-modules                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ SEANet      │ │ Transformer │ │ Quantizer   │              │
│  │ Encoder/Dec │ │ Layers      │ │ (Dummy)     │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ PyTorch     │ │ Rust FFI    │ │ NumPy RS    │              │
│  │ Backend     │ │ Processing  │ │ (Optional)  │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Architecture

### 1. TTSModel (Main Interface)

```python
class TTSModel(nn.Module):
    """Main TTS model interface coordinating all components."""

    Components:
    - FlowLM: Text-to-audio latent generation
    - Mimi: Neural audio codec (encoder/decoder)
    - Conditioner: Text processing and tokenization

    Key Methods:
    - load_model(): Factory method for model initialization
    - generate_audio(): Complete audio generation
    - generate_audio_stream(): Real-time streaming generation
    - get_state_for_audio_prompt(): Voice cloning setup
```

### 2. FlowLM (Flow Language Model)

```
┌─────────────────────────────────────────────────────────┐
│                    FlowLM                                │
├─────────────────────────────────────────────────────────┤
│  Text Processing                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Tokenizer   │ │ Text Embed  │ │ Positional │       │
│  │ (SentenceP) │ │ ding        │ │ Encoding   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
├─────────────────────────────────────────────────────────┤
│  Transformer Architecture                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Multi-Head  │ │ Feed-Forward │ │ LayerNorm   │       │
│  │ Attention   │ │ Network     │ │ & Residual  │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
├─────────────────────────────────────────────────────────┤
│  Flow-Based Generation                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Latent Flow │ │ LSD         │ │ Temperature │       │
│  │ Sampling    │ │ Decoding    │ │ Control     │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**Key Features:**
- Autoregressive latent generation
- Lagrangian Self-Distillation (LSD) for quality improvement
- Teacher forcing for long texts
- End-of-sequence detection

### 3. Mimi Neural Codec

```
┌─────────────────────────────────────────────────────────┐
│                     Mimi Model                           │
├─────────────────────────────────────────────────────────┤
│  Audio Encoder                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ SEANet      │ │ Transformer │ │ Quantizer   │       │
│  │ Encoder     │ │ Encoder     │ │ (Discrete)  │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
├─────────────────────────────────────────────────────────┤
│  Audio Decoder                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Dequantizer │ │ Transformer │ │ SEANet      │       │
│  │             │ │ Decoder     │ │ Decoder     │       │
│  └─────────────┘ └─────────────┘ └─────────────┘       │
├─────────────────────────────────────────────────────────┤
│  Features                                                 │
│  • 24kHz sample rate                                     │
│  • Frame rate: 75 Hz (13.33ms frames)                   │
│  • 12.5x compression ratio                               │
│  • 1024-dimensional latents                              │
│  • 8 codebooks (discrete quantization)                   │
└─────────────────────────────────────────────────────────┘
```

### 4. SEANet (Scale-Encode Audio Network)

```
┌─────────────────────────────────────────────────────────┐
│                   SEANet Blocks                         │
├─────────────────────────────────────────────────────────┤
│  Encoder Path                                            │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Conv │ │ReLU │ │Conv │ │ReLU │ │Conv │ │ReLU │       │
│  │1x1  │ │     │ │1x1  │ │     │ │1x1  │ │     │ ...   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
│     ↓         ↓         ↓         ↓         ↓         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Conv │ │ReLU │ │Conv │ │ReLU │ │Conv │ │ReLU │       │
│  │Str  │ │     │ │Str  │ │     │ │Str  │ │     │ ...   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
├─────────────────────────────────────────────────────────┤
│  Decoder Path (reverse of encoder)                       │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Conv │ │ReLU │ │Conv │ │ReLU │ │Conv │ │ReLU │       │
│  │T    │ │     │ │T    │ │     │ │T    │ │     │ ...   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
└─────────────────────────────────────────────────────────┘
```

**SEANet Characteristics:**
- 1D convolutions with stride and dilation
- Residual connections with skip paths
- ReLU activations
- Multi-scale feature extraction
- Efficient parameter count (~1M parameters)

## Data Flow Architecture

### Text-to-Speech Generation Pipeline

```
Input Text
    ↓
┌─────────────────┐
│ Text            │
│ Preprocessing   │ ← Capitalization, punctuation, spacing
└─────────────────┘
    ↓
┌─────────────────┐
│ Tokenization    │ ← SentencePiece tokenizer
└─────────────────┘
    ↓
┌─────────────────┐
│ Text Embeddings │ ← Learnable embeddings + positional encoding
└─────────────────┘
    ↓
┌─────────────────┐
│ FlowLM          │ ← Autoregressive latent generation
│ Transformer     │   with temperature control
└─────────────────┘
    ↓
┌─────────────────┐
│ Audio Latents   │ ← 1024-dimensional continuous latents
└─────────────────┘
    ↓
┌─────────────────┐
│ Mimi Decoder    │ ← Neural audio codec reconstruction
└─────────────────┘
    ↓
┌─────────────────┐
│ Audio Output    │ ← 24kHz PCM audio waveform
└─────────────────┘
```

### Voice Cloning Pipeline

```
Reference Audio
    ↓
┌─────────────────┐
│ Audio Loading    │ ← WAV/MP3/FLAC support
└─────────────────┘
    ↓
┌─────────────────┐
│ Resampling      │ ← Convert to 24kHz if needed
└─────────────────┘
    ↓
┌─────────────────┐
│ Mimi Encoder    │ ← Encode to discrete latents
└─────────────────┘
    ↓
┌─────────────────┐
│ Projection      │ ← Map to FlowLM latent space
└─────────────────┘
    ↓
┌─────────────────┐
│ Speaker State   │ ← KV cache for conditioning
└─────────────────┘
    ↓
┌─────────────────┐
│ Conditioned     │ ← Used as conditioning for
│ Generation      │   text-to-speech generation
└─────────────────┘
```

## Performance Architecture

### Streaming Generation Architecture

```
Text Input
    ↓
┌─────────────────┐
│ Text Chunking   │ ← Split into optimal chunks
└─────────────────┘
    ↓
┌─────────────────┐
│ Parallel        │ ← Two-thread architecture:
│ Processing      │   Thread 1: Latent generation
│                 │   Thread 2: Audio decoding
└─────────────────┘
    ↓
┌─────────────────┐
│ Queue System    │ ← Thread-safe communication
│ (Producer-      │   between generation and decoding
│ Consumer)       │
└─────────────────┘
    ↓
┌─────────────────┐
│ Streaming       │ ← Real-time audio chunk output
│ Output          │   ~200ms first chunk latency
└─────────────────┘
```

### Memory Management Architecture

```
┌─────────────────┐
│ Model States    │ ← KV caches for transformer layers
│                 │   Optimized with slicing/truncation
└─────────────────┘
    ↓
┌─────────────────┐
│ Cache           │ ← Smart KV cache management:
│ Management      │   • Slice to actual used frames
│                 │   • Expand for generation
│                 │   • Memory-efficient storage
└─────────────────┘
    ↓
┌─────────────────┐
│ Audio Buffers   │ ← Circular buffers for streaming
└─────────────────┘
    ↓
┌─────────────────┐
│ Rust FFI        │ ← Zero-copy operations where possible
│ Integration     │   Efficient audio processing
└─────────────────┘
```

## Model Specifications

### Configuration Summary

| Component       | Parameters | Key Features                     |
|-----------------|------------|----------------------------------|
| **Total Model** | ~100M      | CPU-optimized, streaming capable |
| **FlowLM**      | ~85M       | 12 layers, 8 heads, 512 dim      |
| **Mimi Codec**  | ~15M       | 8 codebooks, 1024 latents        |
| **SEANet**      | ~1M each   | Efficient encoder/decoder        |
| **Sample Rate** | 24kHz      | Broadcast quality audio          |
| **Frame Rate**  | 75Hz       | 13.33ms frame duration           |
| **Compression** | 12.5x      | Efficient audio representation   |

### Performance Characteristics

| Metric               | Value    | Description            |
|----------------------|----------|------------------------|
| **Latency**          | ~200ms   | First audio chunk      |
| **Real-time Factor** | ~6x      | Faster than real-time  |
| **CPU Usage**        | 2 cores  | Minimal resource usage |
| **Memory**           | ~400MB   | Model + buffers        |
| **Quality**          | MOS ~4.0 | High-quality synthesis |

## Extension Points

### Custom Voice Integration
```
Custom Voice Data
    ↓
┌─────────────────┐
│ Audio Analysis  │ ← Feature extraction
└─────────────────┘
    ↓
┌─────────────────┐
│ Voice Embedding  │ ← Speaker representation
└─────────────────┘
    ↓
┌─────────────────┐
│ Model Fine-tune │ ← Optional adaptation
└─────────────────┘
```

### Multi-language Support
```
Language-Specific
Components
    ↓
┌─────────────────┐
│ Tokenizer       │ ← Language-specific tokenization
└─────────────────┘
    ↓
┌─────────────────┐
│ Text Processor  │ ← Language-specific preprocessing
└─────────────────┘
    ↓
┌─────────────────┐
│ Model Adapter   │ ← Language-specific fine-tuning
└─────────────────┘
```

This architecture enables Pocket TTS to achieve high-quality speech synthesis with minimal computational resources while maintaining flexibility for various applications and use cases.
