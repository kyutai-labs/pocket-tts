//! Runs the engine on a phone from `adb shell`, without the app, so its speed
//! on the phone's CPU and its output can be measured on their own.
//!
//! usage: pocket-bench ORT_LIB MODEL_DIR VOICE.safetensors OUT_DIR SEED THREADS < lines
//!
//! Each stdin line is one chunk, with `\n` standing for a newline inside it.
//! Writes OUT_DIR/NNN.wav per line and one JSON line of stats per chunk.

use std::io::{BufRead, Write};
use std::path::Path;

use pocket_tts_engine::{Cancel, Engine};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!("usage: pocket-bench ORT_LIB MODEL_DIR VOICE OUT_DIR SEED THREADS < lines");
        std::process::exit(2);
    }
    ort::init_from(&args[1]).expect("ONNX Runtime").commit();
    let started = std::time::Instant::now();
    let mut engine = Engine::load(Path::new(&args[2]), args[6].parse().unwrap()).expect("model");
    let voice = engine.load_voice(Path::new(&args[3])).expect("voice");
    println!("{{\"load_ms\":{}}}", started.elapsed().as_millis());
    let out = Path::new(&args[4]);
    std::fs::create_dir_all(out).unwrap();
    let seed: u64 = args[5].parse().unwrap();
    let rate = engine.sample_rate();
    for (i, line) in std::io::stdin().lock().lines().enumerate() {
        let text = line.unwrap().replace("\\n", "\n");
        engine.set_seed(seed);
        let mut samples = Vec::new();
        let stats = engine
            .synthesize(&text, &voice, &Cancel::new(), |pcm| {
                samples.extend_from_slice(pcm);
                true
            })
            .expect("synthesis");
        write_wav(&out.join(format!("{i:03}.wav")), &samples, rate);
        let audio_ms = samples.len() as u128 * 1000 / rate as u128;
        println!(
            "{{\"chunk\":{i},\"tokens\":{},\"frames\":{},\"end\":\"{:?}\",\"audio_ms\":{audio_ms},\"generation_ms\":{},\"first_audio_ms\":{}}}",
            stats.tokens,
            stats.frames,
            stats.end,
            stats.generation_ms,
            stats.first_audio_ms.unwrap_or(0)
        );
        std::io::stdout().flush().unwrap();
    }
}

fn write_wav(path: &Path, samples: &[f32], rate: u32) {
    let data = samples.len() as u32 * 2;
    let mut bytes = Vec::with_capacity(44 + data as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        bytes.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}
