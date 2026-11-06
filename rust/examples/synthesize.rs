//! Text in, WAV out, from outside the crate.
//!
//! An example is compiled as its own crate linking `pocket_tts`, so this file only reaches the
//! public API — which makes it the cheapest check that the library surface is actually sufficient.
//! If something here needs a `pub(crate)` detour, the surface is too narrow and the fix belongs in
//! the library.
//!
//! ```sh
//! cargo run --release --example synthesize -- "Hello from a library." out.wav
//! ```
//!
//! The weights are downloaded from the Hugging Face Hub on first use and cached, so the first run
//! is slow and needs network access.

use pocket_tts::audio::StreamingWavWriter;
use pocket_tts::config::Config;
use pocket_tts::model::{LoadOptions, TtsModel};
use pocket_tts::{F32, defaults};
use std::sync::atomic::AtomicBool;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let text = args
        .next()
        .unwrap_or_else(|| "Hello from a library.".to_string());
    let output = args.next().unwrap_or_else(|| "out.wav".to_string());

    // `F32` is the unquantized CPU backend; `pocket_tts::Int8` is the quantized one, and the model
    // is generic over the two.
    let (config, origin) = Config::load(Some("english"), None)?;
    let model: TtsModel<F32> = TtsModel::load(config, origin, &LoadOptions::default())?;

    // A voice is a name from the catalogue, an audio file, or an exported `.safetensors` state.
    let voice_name = defaults::get_default_voice_for_language(Some("english"), false);
    let voice = model.get_state_for_audio_prompt(&voice_name, false)?;

    // Generation is a callback per decoded chunk, so audio can be played or streamed as it comes
    // rather than buffered whole. Here it goes straight into a WAV file.
    let file = std::fs::File::create(&output)?;
    let mut writer = StreamingWavWriter::new(file, model.sample_rate() as u32, true);
    writer.write_header_initial()?;
    let mut samples = 0usize;
    model.generate_audio_stream(
        &voice,
        &text,
        defaults::MAX_TOKEN_PER_CHUNK,
        None,
        &AtomicBool::new(false),
        &mut |chunk: &[f32]| {
            samples += chunk.len();
            writer.write_pcm(chunk)?;
            Ok(())
        },
    )?;
    writer.finalize()?;

    let seconds = samples as f64 / model.sample_rate() as f64;
    println!("wrote {output}: {seconds:.2}s of audio as {voice_name}");
    Ok(())
}
