//! Voice resolution utilities for CLI and server
//!
//! This module provides unified voice resolution logic supporting:
//! - Predefined voice names (alba, marius, etc.)
//! - Local file paths
//! - HuggingFace URLs (hf://...)
//! - Base64-encoded audio data

use anyhow::{Context, Result};
use pocket_tts::TTSModel;
use pocket_tts::weights::download_if_necessary;
use std::path::PathBuf;

/// Predefined stock voices from kyutai/pocket-tts-without-voice-cloning
pub const PREDEFINED_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];

/// HuggingFace repo for stock voice embeddings
const STOCK_VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";

/// Resolve a voice specification to a ModelState
///
/// Supports multiple input formats:
/// - Predefined names: "alba", "marius", etc.
/// - Local paths: "/path/to/audio.wav" or "/path/to/embeddings.safetensors"
/// - HF URLs: "hf://owner/repo/file.wav"
/// - Base64 audio: "data:audio/wav;base64,..." or raw base64 string
pub fn resolve_voice(model: &TTSModel, voice_spec: Option<&str>) -> Result<pocket_tts::ModelState> {
    match voice_spec {
        Some(spec) => resolve_voice_spec(model, spec),
        None => {
            // Default to "alba" stock voice
            resolve_predefined_voice(model, "alba")
        }
    }
}

/// Resolve a specific voice specification
fn resolve_voice_spec(model: &TTSModel, spec: &str) -> Result<pocket_tts::ModelState> {
    let spec = spec.trim();

    // 1. Check if it's a predefined voice name
    if PREDEFINED_VOICES.contains(&spec) {
        return resolve_predefined_voice(model, spec);
    }

    // 2. Check if it's an hf:// URL
    if spec.starts_with("hf://") {
        return resolve_hf_voice(model, spec);
    }

    // 3. Check if it's a file path that exists
    let path = PathBuf::from(spec);
    if path.exists() {
        return resolve_file_voice(model, &path);
    }

    // 4. Check if it's base64 encoded audio
    if is_base64_audio(spec) {
        return resolve_base64_voice(model, spec);
    }

    // Not found as anything - give a helpful error
    anyhow::bail!(
        "Voice '{}' not found. Expected one of:\n\
         - Predefined name: {}\n\
         - File path: /path/to/voice.wav or /path/to/embeddings.safetensors\n\
         - HuggingFace URL: hf://owner/repo/file.wav\n\
         - Base64 audio: data:audio/wav;base64,...",
        spec,
        PREDEFINED_VOICES.join(", ")
    )
}

/// Resolve a predefined voice name to embeddings via HF Hub
fn resolve_predefined_voice(model: &TTSModel, name: &str) -> Result<pocket_tts::ModelState> {
    let hf_path = format!("hf://{}/embeddings/{}.safetensors", STOCK_VOICE_REPO, name);

    let local_path = download_if_necessary(&hf_path)
        .with_context(|| format!("Failed to download stock voice '{}'", name))?;

    model
        .get_voice_state_from_prompt_file(&local_path)
        .with_context(|| format!("Failed to load voice embeddings from {:?}", local_path))
}

/// Resolve an hf:// URL (audio or safetensors)
fn resolve_hf_voice(model: &TTSModel, url: &str) -> Result<pocket_tts::ModelState> {
    let local_path = download_if_necessary(url)
        .with_context(|| format!("Failed to download voice from '{}'", url))?;

    resolve_file_voice(model, &local_path)
}

/// Resolve a local file (WAV audio or safetensors embeddings)
fn resolve_file_voice(model: &TTSModel, path: &PathBuf) -> Result<pocket_tts::ModelState> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "safetensors" => {
            // Pre-computed embeddings
            model
                .get_voice_state_from_prompt_file(path)
                .with_context(|| format!("Failed to load embeddings from {:?}", path))
        }
        "wav" | "wave" => {
            // Raw audio - encode through Mimi
            model
                .get_voice_state(path)
                .with_context(|| format!("Failed to process audio from {:?}", path))
        }
        _ => {
            anyhow::bail!(
                "Unsupported file extension '{}' for voice file. Expected .wav or .safetensors",
                ext
            )
        }
    }
}

/// Check if a string looks like base64 audio
fn is_base64_audio(spec: &str) -> bool {
    // Data URL format
    if spec.starts_with("data:audio/") && spec.contains("base64,") {
        return true;
    }

    // Raw base64 - check if it's reasonably long and looks like base64
    // WAV header is 44 bytes, base64 encoded is ~60 chars minimum
    if spec.len() > 100 {
        // Check if it looks like base64
        let clean = spec.trim();
        return clean
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=');
    }

    false
}

/// Resolve base64-encoded audio data
fn resolve_base64_voice(model: &TTSModel, spec: &str) -> Result<pocket_tts::ModelState> {
    // Strip data URL prefix if present
    let b64_str = if spec.starts_with("data:") {
        spec.split(',').nth(1).unwrap_or(spec)
    } else {
        spec
    };

    use base64::{Engine as _, engine::general_purpose};
    let bytes = general_purpose::STANDARD
        .decode(b64_str)
        .context("Failed to decode base64 audio")?;

    // Decode WAV from bytes
    let (audio, sample_rate) = pocket_tts::audio::read_wav_from_bytes(&bytes)
        .context("Failed to parse WAV from base64 data")?;

    // Resample if needed
    let audio = if sample_rate != model.sample_rate as u32 {
        pocket_tts::audio::resample(&audio, sample_rate, model.sample_rate as u32)?
    } else {
        audio
    };

    // Add batch dimension: [C, T] -> [B, C, T]
    let audio = audio.unsqueeze(0)?;

    model
        .get_voice_state_from_tensor(&audio)
        .context("Failed to encode base64 audio for voice cloning")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predefined_voices_list() {
        assert!(PREDEFINED_VOICES.contains(&"alba"));
        assert!(PREDEFINED_VOICES.contains(&"marius"));
        assert!(!PREDEFINED_VOICES.contains(&"unknown"));
    }

    #[test]
    fn test_is_base64_audio() {
        assert!(is_base64_audio(
            "data:audio/wav;base64,UklGRi4AAABXQVZFZm10IBAAAAABAAIAQB8AAEAfAAABAAgAZGF0YQoAAAAA"
        ));
        assert!(!is_base64_audio("alba"));
        assert!(!is_base64_audio("/path/to/file.wav"));
        assert!(!is_base64_audio("short"));
    }
}
