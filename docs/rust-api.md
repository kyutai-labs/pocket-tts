# Rust API Documentation

The `pocket-tts` library provides a Rust API for integrating text-to-speech into your applications.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pocket-tts = { path = "path/to/candle/crates/pocket-tts" }
```

Or once published to crates.io:

```toml
[dependencies]
pocket-tts = "0.1"
```

## Quick Start

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

## Core Types

### TTSModel

The main struct for text-to-speech generation.

```rust
pub struct TTSModel {
    pub sample_rate: usize,    // Audio sample rate (24000)
    pub temp: f32,             // Sampling temperature
    // ... internal fields
}
```

#### Loading Methods

##### `TTSModel::load(variant: &str) -> Result<Self>`

Load a model with default parameters.

```rust
let model = TTSModel::load("b6369a24")?;
```

##### `TTSModel::load_with_params(...) -> Result<Self>`

Load with custom generation parameters.

```rust
let model = TTSModel::load_with_params(
    "b6369a24",     // variant
    0.7,            // temperature
    1,              // lsd_decode_steps
    -4.0,           // eos_threshold
)?;
```

**Parameters:**
- `variant`: Model variant identifier (e.g., `"b6369a24"`)
- `temp`: Sampling temperature (0.0 = deterministic, 0.7 = natural)
- `lsd_decode_steps`: LSD decode steps (1 = fast, 5 = high quality)
- `eos_threshold`: End-of-speech threshold (more negative = longer audio)

#### Voice State Methods

##### `get_voice_state<P: AsRef<Path>>(&self, audio_path: P) -> Result<ModelState>`

Create voice state from an audio file (voice cloning).

```rust
let voice_state = model.get_voice_state("reference.wav")?;
```

##### `get_voice_state_from_tensor(&self, audio: &Tensor) -> Result<ModelState>`

Create voice state from a tensor (already loaded audio).

```rust
let (audio, _sample_rate) = pocket_tts::audio::read_wav("reference.wav")?;
let audio = audio.unsqueeze(0)?; // Add batch dimension
let voice_state = model.get_voice_state_from_tensor(&audio)?;
```

##### `get_voice_state_from_prompt_file<P: AsRef<Path>>(&self, path: P) -> Result<ModelState>`

Load pre-computed voice embeddings from a `.safetensors` file.

```rust
let voice_state = model.get_voice_state_from_prompt_file("embeddings.safetensors")?;
```

#### Generation Methods

##### `generate(&self, text: &str, voice_state: &ModelState) -> Result<Tensor>`

Generate complete audio from text.

```rust
let audio = model.generate("Hello, world!", &voice_state)?;
// audio shape: [1, samples]
```

##### `generate_stream<'a>(...) -> Box<dyn Iterator<Item = Result<Tensor>> + 'a>`

Generate audio in streaming fashion, yielding chunks.

```rust
for chunk_result in model.generate_stream("Long text...", &voice_state) {
    let chunk = chunk_result?;
    // Process each audio chunk
    // chunk shape: [1, 1, samples_per_frame]
}
```

##### `generate_stream_long(...) -> impl Iterator<Item = Result<Tensor>>`

Generate from long text with automatic segmentation.

```rust
for chunk in model.generate_stream_long("Very long text...", &voice_state) {
    let audio = chunk?;
    // Process chunk
}
```

### ModelState

Type alias for voice conditioning state:

```rust
pub type ModelState = HashMap<String, HashMap<String, Tensor>>;
```

This contains the internal state needed for generation with a specific voice.

## Audio Utilities

The `pocket_tts::audio` module provides audio I/O utilities.

### Reading Audio

```rust
use pocket_tts::audio::read_wav;

let (audio, sample_rate) = read_wav("input.wav")?;
// audio: Tensor [channels, samples]
// sample_rate: u32
```

### Writing Audio

```rust
use pocket_tts::audio::write_wav;

write_wav("output.wav", &audio, 24000)?;
```

### Resampling

```rust
use pocket_tts::audio::resample;

// Resample from 48kHz to 24kHz
let resampled = resample(&audio, 48000, 24000)?;
```

## Example: Batch Processing

```rust
use pocket_tts::TTSModel;
use candle_core::Tensor;

fn batch_generate(texts: &[&str], voice_path: &str) -> anyhow::Result<Vec<Tensor>> {
    let model = TTSModel::load("b6369a24")?;
    let voice_state = model.get_voice_state(voice_path)?;
    
    // Reuse voice state for multiple generations
    let mut outputs = Vec::new();
    for text in texts {
        let audio = model.generate(text, &voice_state)?;
        outputs.push(audio);
    }
    
    Ok(outputs)
}
```

## Example: Streaming Playback

```rust
use pocket_tts::TTSModel;
use std::io::Write;

fn stream_to_stdout(text: &str, voice_path: &str) -> anyhow::Result<()> {
    let model = TTSModel::load("b6369a24")?;
    let voice_state = model.get_voice_state(voice_path)?;
    
    let mut stdout = std::io::stdout();
    
    for chunk in model.generate_stream_long(text, &voice_state) {
        let chunk = chunk?.squeeze(0)?;
        let data = chunk.to_vec2::<f32>()?;
        
        for sample in &data[0] {
            let val = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
            stdout.write_all(&val.to_le_bytes())?;
        }
    }
    
    Ok(())
}
```

## Example: Multiple Voices

```rust
use pocket_tts::TTSModel;
use std::collections::HashMap;

fn multi_voice_generation() -> anyhow::Result<()> {
    let model = TTSModel::load("b6369a24")?;
    
    // Pre-load multiple voice states
    let voices: HashMap<&str, _> = [
        ("alice", model.get_voice_state("alice.wav")?),
        ("bob", model.get_voice_state("bob.wav")?),
    ].into_iter().collect();
    
    // Generate with different voices
    let alice_audio = model.generate("Hello from Alice!", &voices["alice"])?;
    let bob_audio = model.generate("Hello from Bob!", &voices["bob"])?;
    
    pocket_tts::audio::write_wav("alice.wav", &alice_audio, 24000)?;
    pocket_tts::audio::write_wav("bob.wav", &bob_audio, 24000)?;
    
    Ok(())
}
```

## Configuration

Default generation parameters are available in `pocket_tts::config::defaults`:

```rust
use pocket_tts::config::defaults;

println!("Default temperature: {}", defaults::TEMPERATURE);        // 0.7
println!("Default LSD steps: {}", defaults::LSD_DECODE_STEPS);     // 1
println!("Default EOS threshold: {}", defaults::EOS_THRESHOLD);    // -4.0
println!("Default variant: {}", defaults::DEFAULT_VARIANT);        // "b6369a24"
```

## Error Handling

All fallible operations return `anyhow::Result`. Common errors:

- Model weights not found (downloads from HuggingFace on first use)
- Invalid audio file format
- Unsupported sample rate (auto-resampled to 24kHz)
- Empty text input

```rust
use anyhow::Context;

let model = TTSModel::load("b6369a24")
    .context("Failed to load TTS model")?;

let voice_state = model.get_voice_state("voice.wav")
    .context("Failed to process voice file")?;
```

## See Also

- [Generate Command](generate.md) - CLI usage
- [Serve Command](serve.md) - HTTP API server
