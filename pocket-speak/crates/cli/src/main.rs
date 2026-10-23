//! `pocket-speak`: read text on stdin aloud in a Pocket TTS voice.

mod install;
mod playback;

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

use pocket_tts_engine::text::{self, Chunking};
use pocket_tts_engine::{Cancel, ChunkStats, EndReason, Engine};

const USAGE: &str = "\
usage: pocket-speak [options] < text
       pocket-speak install [--model-dir DIR] [--voice NAME ...] [--all-voices]

Reads text on stdin and plays it. With --out, writes a WAV instead.

  --out PATH          write a 16-bit 24 kHz WAV instead of playing
  --play              play as well, when --out is given
  --model-dir DIR     model directory (default: $POCKET_TTS_MODEL_DIR, else the cache)
  --voice NAME|PATH   voice embedding (default: alba)
  --chunking MODE     sentence | packed | packed-raw (default: packed)
  --plain             do not strip Markdown
  --seed N            reproducible sampling
  --threads N         ONNX Runtime intra-op threads (default: 2)
  --stats             one JSON line per chunk on stderr
";

struct Options {
    out: Option<PathBuf>,
    play: bool,
    model_dir: PathBuf,
    voice: String,
    chunking: Chunking,
    plain: bool,
    seed: Option<u64>,
    threads: usize,
    stats: bool,
}

fn default_model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("POCKET_TTS_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join("pocket-tts-rs")
        .join("english_2026-04")
}

fn parse(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut options = Options {
        out: None,
        play: false,
        model_dir: default_model_dir(),
        voice: "alba".to_string(),
        chunking: Chunking::PACKED_DEFAULT,
        plain: false,
        seed: None,
        threads: 2,
        stats: false,
    };
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--out" => options.out = Some(PathBuf::from(value("--out")?)),
            "--play" => options.play = true,
            "--model-dir" => options.model_dir = PathBuf::from(value("--model-dir")?),
            "--voice" => options.voice = value("--voice")?,
            "--chunking" => {
                options.chunking = match value("--chunking")?.as_str() {
                    "sentence" => Chunking::Sentence,
                    "packed" => Chunking::PACKED_DEFAULT,
                    "packed-raw" => Chunking::Packed {
                        target: 200,
                        max: 400,
                        join_tail: false,
                    },
                    other => return Err(format!("unknown chunking {other:?}")),
                }
            }
            "--plain" => options.plain = true,
            "--seed" => {
                options.seed = Some(
                    value("--seed")?
                        .parse()
                        .map_err(|e| format!("--seed: {e}"))?,
                )
            }
            "--threads" => {
                options.threads = value("--threads")?
                    .parse()
                    .map_err(|e| format!("--threads: {e}"))?
            }
            "--stats" => options.stats = true,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(options)
}

fn install_args(args: Vec<String>) -> Result<(PathBuf, Vec<String>), String> {
    let mut dir = default_model_dir();
    let mut voices = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => {
                dir = PathBuf::from(args.next().ok_or("--model-dir needs a value")?);
            }
            "--voice" => voices.push(args.next().ok_or("--voice needs a value")?),
            "--all-voices" => {
                voices = install::VOICE_FILES
                    .iter()
                    .map(|(name, _)| name.to_string())
                    .collect();
            }
            other => return Err(format!("unknown install argument {other:?}")),
        }
    }
    if voices.is_empty() {
        voices.push("alba".to_string());
    }
    Ok((dir, voices))
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.first().map(String::as_str) == Some("install") {
        args.remove(0);
        match install_args(args) {
            Ok((dir, voices)) => install::install(&dir, &voices),
            Err(message) => return usage_error(&message),
        }
    } else {
        match parse(args.into_iter()) {
            Ok(options) => run(options),
            Err(message) => return usage_error(&message),
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("pocket-speak: {message}");
            ExitCode::FAILURE
        }
    }
}

fn usage_error(message: &str) -> ExitCode {
    if !message.is_empty() {
        eprintln!("pocket-speak: {message}");
    }
    eprint!("{USAGE}");
    ExitCode::from(2)
}

fn run(options: Options) -> Result<(), String> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("reading stdin: {e}"))?;
    let speakable = if options.plain {
        input
    } else {
        text::speakable(&input)
    };
    let chunks = text::chunk(&speakable, options.chunking);
    if chunks.is_empty() {
        return Err("nothing to say".to_string());
    }

    let mut engine =
        Engine::load(&options.model_dir, options.threads).map_err(|e| e.to_string())?;
    if let Some(seed) = options.seed {
        engine.set_seed(seed);
    }
    let voice_path =
        if options.voice.contains(['/', '\\']) || options.voice.ends_with(".safetensors") {
            PathBuf::from(&options.voice)
        } else {
            options
                .model_dir
                .join("embeddings")
                .join(format!("{}.safetensors", options.voice))
        };
    let voice = engine.load_voice(&voice_path).map_err(|e| e.to_string())?;
    let sample_rate = engine.sample_rate();

    let out_name = options
        .out
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let mut writer = match &options.out {
        Some(out) => Some(
            hound::WavWriter::create(
                out,
                hound::WavSpec {
                    channels: 1,
                    sample_rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .map_err(|e| format!("{out_name}: {e}"))?,
        ),
        None => None,
    };
    let mut player = if options.out.is_none() || options.play {
        Some(playback::Player::open(sample_rate)?)
    } else {
        None
    };

    let cancel = Cancel::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let mut failure: Option<String> = None;
        let stats = engine
            .synthesize(&chunk.speech, &voice, &cancel, |pcm| {
                if let Some(writer) = writer.as_mut() {
                    for sample in pcm {
                        let quantized = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
                        if let Err(e) = writer.write_sample(quantized) {
                            failure = Some(format!("{out_name}: {e}"));
                            return false;
                        }
                    }
                }
                if let Some(player) = player.as_mut()
                    && !player.push(pcm)
                {
                    failure = Some("audio output stopped".to_string());
                    return false;
                }
                true
            })
            .map_err(|e| e.to_string())?;
        if let Some(failure) = failure {
            return Err(failure);
        }
        if chunk.pause_after_seconds > 0.0 {
            if let Some(writer) = writer.as_mut() {
                let silence = (chunk.pause_after_seconds * sample_rate as f32) as usize;
                for _ in 0..silence {
                    writer
                        .write_sample(0_i16)
                        .map_err(|e| format!("{out_name}: {e}"))?;
                }
            }
            if let Some(player) = player.as_mut() {
                player.push_silence(chunk.pause_after_seconds, sample_rate);
            }
        }
        if options.stats {
            eprintln!("{}", stats_line(index, &chunk.speech, &stats, sample_rate));
        }
    }
    if let Some(writer) = writer {
        writer.finalize().map_err(|e| format!("{out_name}: {e}"))?;
    }
    if let Some(player) = player {
        player.finish();
    }
    Ok(())
}

fn stats_line(index: usize, speech: &str, stats: &ChunkStats, sample_rate: u32) -> String {
    let audio_seconds = stats.samples as f32 / sample_rate as f32;
    let speed = audio_seconds / (stats.generation_ms.max(1) as f32 / 1000.0);
    let or_null = |value: Option<String>| value.unwrap_or_else(|| "null".to_string());
    let end = match stats.end {
        EndReason::EndOfSpeech => "eos",
        EndReason::Budget => "budget",
        EndReason::Cancelled => "cancelled",
    };
    format!(
        "{{\"chunk\":{index},\"chars\":{},\"tokens\":{},\"frames\":{},\"max_frames\":{},\"eos_frame\":{},\"end\":\"{end}\",\"audio_s\":{audio_seconds:.2},\"gen_ms\":{},\"first_audio_ms\":{},\"x_realtime\":{speed:.2}}}",
        speech.chars().count(),
        stats.tokens,
        stats.frames,
        stats.max_frames,
        or_null(stats.eos_frame.map(|frame| frame.to_string())),
        stats.generation_ms,
        or_null(stats.first_audio_ms.map(|ms| ms.to_string())),
    )
}
