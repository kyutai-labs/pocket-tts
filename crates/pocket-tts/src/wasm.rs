//! WebAssembly bindings for Pocket TTS
//!
//! This module provides WASM-compatible entry points for browser usage.
//! Build with: `wasm-pack build --target web --features wasm`

#![cfg(target_arch = "wasm32")]

use crate::tts_model::TTSModel;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

/// Initialize console_error_panic_hook for better error messages in browser
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"Pocket TTS WASM initialized".into());
}

/// WASM-compatible TTS model wrapper
#[wasm_bindgen]
pub struct WasmTTSModel {
    model: Option<TTSModel>,
    voice_state: Option<crate::ModelState>,
    sample_rate: u32,
}

#[wasm_bindgen]
impl WasmTTSModel {
    /// Create a new WASM TTS model
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmTTSModel {
        Self {
            model: None,
            voice_state: None,
            sample_rate: 24000,
        }
    }

    /// Load model from ArrayBuffers
    ///
    /// # Arguments
    /// * `config_yaml` - ArrayBuffer containing config.yaml
    /// * `weights_data` - ArrayBuffer containing safetensors model weights
    /// * `tokenizer_bytes` - Optional ArrayBuffer containing tokenizer.json (falls back to embedded)
    #[wasm_bindgen]
    pub fn load_from_buffer(
        &mut self,
        config_yaml: &[u8],
        weights_data: &[u8],
        tokenizer_bytes: &[u8],
    ) -> Result<(), JsValue> {
        let tok_bytes = if tokenizer_bytes.is_empty() {
            include_bytes!("../assets/tokenizer.json")
        } else {
            tokenizer_bytes
        };

        let model = TTSModel::load_from_bytes(config_yaml, weights_data, tok_bytes)
            .map_err(|e| JsValue::from_str(&format!("Model loading failed: {:?}", e)))?;

        self.sample_rate = model.sample_rate as u32;
        self.model = Some(model);

        web_sys::console::log_1(
            &"Model loaded successfully (using embedded or provided tokenizer)".into(),
        );
        Ok(())
    }

    /// Check if model is ready for generation
    #[wasm_bindgen]
    pub fn is_ready(&self) -> bool {
        self.model.is_some()
    }

    /// Load voice from WAV audio buffer for voice cloning
    #[wasm_bindgen]
    pub fn load_voice_from_buffer(&mut self, wav_bytes: &[u8]) -> Result<(), JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded. Call load_from_buffer first."))?;

        let voice_state = model
            .get_voice_state_from_bytes(wav_bytes)
            .map_err(|e| JsValue::from_str(&format!("Voice cloning failed: {:?}", e)))?;

        self.voice_state = Some(voice_state);
        web_sys::console::log_1(&"Voice loaded successfully (from audio)".into());
        Ok(())
    }

    /// Load voice from safetensors buffer (pre-calculated embedding)
    #[wasm_bindgen]
    pub fn load_voice_from_safetensors(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded. Call load_from_buffer first."))?;

        let voice_state = model
            .get_voice_state_from_prompt_bytes(bytes)
            .map_err(|e| JsValue::from_str(&format!("Voice loading failed: {:?}", e)))?;

        self.voice_state = Some(voice_state);
        web_sys::console::log_1(&"Voice loaded successfully (from embedding)".into());
        Ok(())
    }

    /// Get the sample rate of generated audio
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Generate audio from text
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    ///
    /// # Returns
    /// Float32Array containing audio samples at 24kHz mono
    #[wasm_bindgen]
    pub fn generate(&self, text: &str) -> Result<Float32Array, JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Model not loaded. Call load_from_buffer first."))?;

        let voice_state = self
            .voice_state
            .clone()
            .unwrap_or_else(|| crate::voice_state::init_states(1, 0));

        let audio_tensor = model
            .generate(text, &voice_state)
            .map_err(|e| JsValue::from_str(&format!("Generation failed: {:?}", e)))?;

        // Flatten tensor to Vec<f32>
        let samples = audio_tensor
            .to_vec2::<f32>()
            .map_err(|e| JsValue::from_str(&format!("Failed to extract samples: {:?}", e)))?[0]
            .clone();

        let array = Float32Array::new_with_length(samples.len() as u32);
        array.copy_from(&samples);

        Ok(array)
    }

    /// Generate audio and return as base64-encoded WAV
    #[wasm_bindgen]
    pub fn generate_wav_base64(&self, text: &str) -> Result<String, JsValue> {
        let samples = self.generate(text)?;
        let mut sample_vec = vec![0.0f32; samples.length() as usize];
        samples.copy_to(&mut sample_vec);

        let mut buffer = std::io::Cursor::new(Vec::new());
        {
            // Simple WAV header creation since hound is gated
            write_wav_header(&mut buffer, self.sample_rate, sample_vec.len() as u32)
                .map_err(|e| JsValue::from_str(&format!("WAV header error: {:?}", e)))?;

            for &sample in &sample_vec {
                let s = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                buffer.get_mut().extend_from_slice(&s.to_le_bytes());
            }
        }

        let base64 = base64_encode(buffer.get_ref());
        Ok(format!("data:audio/wav;base64,{}", base64))
    }
}

/// Helper to write a basic 16-bit PCM WAV header
fn write_wav_header(
    w: &mut dyn std::io::Write,
    sample_rate: u32,
    num_samples: u32,
) -> std::io::Result<()> {
    let subchunk2_size = num_samples * 2; // 2 bytes per sample (16-bit)
    let chunk_size = 36 + subchunk2_size;

    w.write_all(b"RIFF")?;
    w.write_all(&chunk_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // Subchunk1Size
    w.write_all(&1u16.to_le_bytes())?; // AudioFormat (PCM)
    w.write_all(&1u16.to_le_bytes())?; // NumChannels (Mono)
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&(sample_rate * 2).to_le_bytes())?; // ByteRate
    w.write_all(&2u16.to_le_bytes())?; // BlockAlign
    w.write_all(&16u16.to_le_bytes())?; // BitsPerSample
    w.write_all(b"data")?;
    w.write_all(&subchunk2_size.to_le_bytes())?;
    Ok(())
}

/// Simple base64 encoder to avoid external dependencies in WASM build
fn base64_encode(input: &[u8]) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity((input.len() + 2) / 3 * 4);

    for chunk in input.chunks(3) {
        let b = match chunk.len() {
            3 => (chunk[0] as u32) << 16 | (chunk[1] as u32) << 8 | (chunk[2] as u32),
            2 => (chunk[0] as u32) << 16 | (chunk[1] as u32) << 8,
            1 => (chunk[0] as u32) << 16,
            _ => unreachable!(),
        };

        output.push(CHARSET[(b >> 18 & 0x3F) as usize] as char);
        output.push(CHARSET[(b >> 12 & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            output.push(CHARSET[(b >> 6 & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }

        if chunk.len() > 2 {
            output.push(CHARSET[(b & 0x3F) as usize] as char);
        } else {
            output.push('=');
        }
    }
    output
}

impl Default for WasmTTSModel {
    fn default() -> Self {
        Self::new()
    }
}
