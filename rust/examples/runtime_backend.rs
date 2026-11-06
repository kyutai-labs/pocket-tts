//! Choosing the backend at run time, which is what `Tts` is for.
//!
//! `examples/synthesize.rs` names `F32` at compile time, which is the right shape when a program
//! only ever wants one backend. This one takes `--quantize` from the command line instead, and so
//! cannot: generics are resolved before the flag is read.
//!
//! ```sh
//! cargo run --release --example runtime_backend -- "Hello." out.wav
//! cargo run --release --example runtime_backend -- "Hello." out.wav --quantize
//! ```

use pocket_tts::audio::StreamingWavWriter;
use pocket_tts::model::LoadOptions;
use pocket_tts::tts::Tts;
use pocket_tts::{defaults, voices};
use std::sync::atomic::AtomicBool;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quantize = args.iter().any(|a| a == "--quantize");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let text = positional
        .first()
        .map(|s| s.as_str())
        .unwrap_or("Hello from a library.");
    let output = positional.get(1).map(|s| s.as_str()).unwrap_or("out.wav");

    // One call, either backend. No macro, no enum on the caller's side.
    let tts = Tts::load(Some("english"), None, &LoadOptions::default(), quantize)?;

    // The voice catalogue carries its licence, so a program can say what it owes.
    let name = defaults::get_default_voice_for_language(Some("english"), false);
    if let Some(voice) = voices::find(&name) {
        println!("{}", voice.credit_line());
    }
    let voice = tts.voice(&name, false)?;

    let file = std::fs::File::create(output)?;
    let mut writer = StreamingWavWriter::new(file, tts.sample_rate() as u32, true);
    writer.write_header_initial()?;
    let mut samples = 0usize;
    tts.speak(
        &voice,
        text,
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

    println!(
        "wrote {output}: {:.2}s on the {} backend",
        samples as f64 / tts.sample_rate() as f64,
        if tts.is_quantized() { "int8" } else { "f32" }
    );
    Ok(())
}
