//! Integration tests for TTSModel with real weights
//!
//! These tests require the HF_TOKEN environment variable to be set
//! for downloading model weights from HuggingFace.

use pocket_tts::audio::{read_wav, write_wav};
use pocket_tts::voice_state::init_states;
use pocket_tts::weights::download_if_necessary;

use std::path::PathBuf;

fn get_ref_wav_path() -> PathBuf {
    // pocket-tts -> crates -> candle -> project_root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("ref.wav")
}

#[test]
fn test_download_non_gated_tokenizer() {
    // Test downloading from non-gated repo (pocket-tts-without-voice-cloning)
    let path = "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3";
    let result = download_if_necessary(path);
    assert!(result.is_ok(), "Failed to download: {:?}", result.err());
    let local_path = result.unwrap();
    assert!(
        local_path.exists(),
        "Downloaded file does not exist: {:?}",
        local_path
    );
    println!("Downloaded to: {:?}", local_path);
}

#[test]
#[ignore = "requires HF_TOKEN and gated model access"]
fn test_download_gated_weights() {
    // Test downloading from gated repo (pocket-tts)
    let path =
        "hf://kyutai/pocket-tts/tts_b6369a24.safetensors@427e3d61b276ed69fdd03de0d185fa8a8d97fc5b";
    let result = download_if_necessary(path);
    assert!(result.is_ok(), "Failed to download: {:?}", result.err());
}

#[test]
#[ignore = "requires HF_TOKEN and model download"]
fn test_tts_model_load() {
    use pocket_tts::TTSModel;
    let model = TTSModel::load("b6369a24").expect("Failed to load model");
    assert_eq!(model.sample_rate, 24000);
    assert_eq!(model.dim, 1024);
    assert_eq!(model.ldim, 32);
}

#[test]
#[ignore = "requires HF_TOKEN and model download"]
fn test_voice_cloning_from_ref_wav() {
    use pocket_tts::TTSModel;
    let model = TTSModel::load("b6369a24").expect("Failed to load model");

    let ref_wav_path = get_ref_wav_path();
    if !ref_wav_path.exists() {
        eprintln!("ref.wav not found at {:?}, skipping test", ref_wav_path);
        return;
    }

    let voice_state = model
        .get_voice_state(&ref_wav_path)
        .expect("Failed to get voice state");

    // Voice state should have entries from running through the transformer
    assert!(!voice_state.is_empty(), "Voice state should not be empty");
}

#[test]
#[ignore = "requires HF_TOKEN and model download"]
fn test_audio_generation_produces_valid_output() {
    use pocket_tts::TTSModel;
    let model = TTSModel::load("b6369a24").expect("Failed to load model");

    let ref_wav_path = get_ref_wav_path();
    if !ref_wav_path.exists() {
        eprintln!("ref.wav not found at {:?}, skipping test", ref_wav_path);
        return;
    }

    let voice_state = model
        .get_voice_state(&ref_wav_path)
        .expect("Failed to get voice state");

    let audio = model
        .generate("Hello world.", &voice_state)
        .expect("Failed to generate audio");

    // Check output shape
    let dims = audio.dims();
    assert_eq!(dims.len(), 2, "Audio should be [channels, samples]");
    assert_eq!(dims[0], 1, "Should have 1 channel");
    assert!(dims[1] > 0, "Should have some samples");

    // Audio should be reasonable length (at least 0.1 seconds for "Hello world")
    let duration_seconds = dims[1] as f32 / model.sample_rate as f32;
    assert!(
        duration_seconds > 0.1,
        "Audio should be at least 0.1 seconds, got {}",
        duration_seconds
    );

    // Optional: save to file for manual inspection
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_output.wav");
    write_wav(&output_path, &audio, model.sample_rate as u32).expect("Failed to write audio");
    println!("Test audio written to {:?}", output_path);
}

#[test]
#[ignore = "requires HF_TOKEN and model download"]
fn test_mimi_encode_decode_roundtrip() {
    use pocket_tts::TTSModel;
    let model = TTSModel::load("b6369a24").expect("Failed to load model");

    let ref_wav_path = get_ref_wav_path();
    if !ref_wav_path.exists() {
        eprintln!("ref.wav not found at {:?}, skipping test", ref_wav_path);
        return;
    }

    let (audio, sample_rate) = read_wav(&ref_wav_path).expect("Failed to read ref.wav");

    // Resample if needed
    let audio = if sample_rate != model.sample_rate as u32 {
        pocket_tts::audio::resample_linear(&audio, sample_rate, model.sample_rate as u32)
            .expect("Failed to resample")
    } else {
        audio
    };

    // Add batch dimension: [C, T] -> [B, C, T]
    let audio = audio.unsqueeze(0).expect("Failed to add batch dim");

    // Pad audio to a multiple of frame size
    let frame_size = model.mimi.frame_size();
    let (b, c, t) = audio.dims3().expect("dims3");
    let pad_len = if t % frame_size != 0 {
        frame_size - (t % frame_size)
    } else {
        0
    };
    let audio = if pad_len > 0 {
        let pad =
            candle_core::Tensor::zeros((b, c, pad_len), audio.dtype(), audio.device()).unwrap();
        candle_core::Tensor::cat(&[&audio, &pad], 2).unwrap()
    } else {
        audio
    };

    // Encode
    let mut encode_state = init_states(1, 1000);
    let latent = model
        .mimi
        .encode_to_latent(&audio, &mut encode_state)
        .expect("Failed to encode");

    println!("Encoded latent shape: {:?}", latent.dims());

    // Decode
    let mut decode_state = init_states(1, 1000);
    let decoded = model
        .mimi
        .decode_from_latent(&latent, &mut decode_state)
        .expect("Failed to decode");

    println!("Decoded audio shape: {:?}", decoded.dims());

    // The decoded audio should have similar length (within a frame)
    let original_len = audio.dims()[2];
    let decoded_len = decoded.dims()[2];

    // Allow for some length difference due to framing
    let max_diff = 1920 * 2; // ~2 frames at 24kHz
    let len_diff = (original_len as i64 - decoded_len as i64).unsigned_abs() as usize;
    assert!(
        len_diff < max_diff,
        "Audio length mismatch: original={}, decoded={}, diff={}",
        original_len,
        decoded_len,
        len_diff
    );
}
