//! `pocket-tts` without Python: the `generate`, `serve` and `export-voice` commands of the
//! Python package's CLI (`pocket_tts/main.py`), with the same options and defaults.

mod audio;
mod config;
mod defaults;
mod download;
mod logging;
mod model;
mod rng;
mod server;
mod text;
mod tokenizer;
mod weights;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use config::Config;
use model::CpuBackend;
use model::{LoadOptions, TtsModel, VoiceState};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

const LANGUAGE_HELP: &str = "Language for the TTS model. 'english_2026-01', 'english_2026-04', 'english_2026-09', 'english_2026-09_2', 'english_2026-10', 'english_drifting_26-09', 'english_18m_2026-10', 'english_18m_refl_2026-10', 'english_7m_2026-10', 'english_7m_refl_2026-10', 'english_2m_2026-10', 'english_2m_refl_2026-10', 'english', 'french', 'french_24l', 'german', 'german_24l', 'portuguese', 'portuguese_24l', 'italian', 'italian_24l', 'spanish', 'spanish_24l', 'dutch', 'dutch_24l'. Incompatible with the config argument. Default is 'english', which is the same model as 'english_2026-10'.";
const CONFIG_HELP: &str = "Path to a model config .yaml file: a local path, an https:// URL, or an hf:// path. Incompatible with the language argument. If not provided, will use the default English model.";

#[derive(Parser)]
#[command(
    name = "pocket-tts",
    version,
    about = "Kyutai Pocket TTS - Text-to-Speech generation tool"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the server, with a web interface to try voices and prompts.
    Serve(ServeArgs),
    /// Generate speech using Kyutai Pocket TTS.
    Generate(GenerateArgs),
    /// Convert and save audio to .safetensors file
    ExportVoice(ExportVoiceArgs),
}

/// `--quantize / --no-quantize`.
#[derive(Args)]
struct QuantizeFlag {
    /// Apply int8 quantization to reduce memory usage
    #[arg(long, overrides_with = "no_quantize")]
    quantize: bool,
    #[arg(long, overrides_with = "quantize", hide = true)]
    no_quantize: bool,
}

#[derive(Args)]
struct ServeArgs {
    /// Host to bind to
    #[arg(long, default_value = "localhost")]
    host: String,
    /// Port to bind to
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Enable auto-reload (a development feature of the Python server, ignored here)
    #[arg(long, overrides_with = "no_reload")]
    reload: bool,
    #[arg(long, overrides_with = "reload", hide = true)]
    no_reload: bool,
    #[arg(long, help = LANGUAGE_HELP)]
    language: Option<String>,
    #[arg(long, help = CONFIG_HELP)]
    config: Option<String>,
    /// Voice used by requests that don't ask for one: a built-in voice name, a local path to an
    /// audio file or to a .safetensors voice, an https:// URL, or an hf:// path. Defaults to the
    /// built-in voice of the language.
    #[arg(long)]
    default_voice: Option<String>,
    #[command(flatten)]
    quantize: QuantizeFlag,
}

#[derive(Args)]
struct GenerateArgs {
    /// Text to generate
    #[arg(long)]
    text: Option<String>,
    /// Path to audio conditioning file (voice to clone). Defaults to a built-in voice chosen from
    /// the language: 'giovanni' for italian, 'lola' for spanish, 'juergen' for german, 'rafael'
    /// for portuguese, 'estelle' for french, 'alba' otherwise. With the config or checkpoint
    /// argument, defaults to alba's audio file, which any model can clone.
    #[arg(long)]
    voice: Option<String>,
    /// Disable logging output
    #[arg(short, long)]
    quiet: bool,
    #[arg(long, help = format!("{LANGUAGE_HELP} The '24l' variants are bigger models, not distilled yet and here only as preview. They're not the final models for those languages."))]
    language: Option<String>,
    #[arg(long, help = CONFIG_HELP)]
    config: Option<String>,
    /// Training checkpoint (.pt) to load instead of the config's weights (not supported by this
    /// binary)
    #[arg(long)]
    checkpoint: Option<String>,
    /// Number of generation steps
    #[arg(long, default_value_t = defaults::DEFAULT_SAMPLER_DECODE_STEPS)]
    sampler_decode_steps: usize,
    /// Deprecated: use --sampler-decode-steps
    #[arg(long, hide = true)]
    lsd_decode_steps: Option<usize>,
    /// Temperature for generation. Defaults to the model's recommended value from its config
    /// (0.3).
    #[arg(long, allow_negative_numbers = true)]
    temperature: Option<f32>,
    /// Noise clamp value
    #[arg(long, allow_negative_numbers = true)]
    noise_clamp: Option<f32>,
    /// EOS threshold. Defaults to the model's value from its config (-4.0).
    #[arg(long, allow_negative_numbers = true)]
    eos_threshold: Option<f32>,
    /// Number of frames to generate after EOS
    #[arg(long)]
    frames_after_eos: Option<usize>,
    /// Output path for generated audio
    #[arg(long, default_value = "./tts_output.wav")]
    output_path: String,
    /// Device to use
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Maximum number of tokens per chunk.
    #[arg(long, default_value_t = defaults::MAX_TOKEN_PER_CHUNK)]
    max_tokens: usize,
    #[command(flatten)]
    quantize: QuantizeFlag,
}

#[derive(Args)]
struct ExportVoiceArgs {
    /// Audio file or directory to convert and export
    audio_path: String,
    /// Output file or directory
    export_path: String,
    /// Disable logging output
    #[arg(short, long)]
    quiet: bool,
    #[arg(long, help = format!("{LANGUAGE_HELP} The '24l' variants are bigger models, not distilled yet and here only as preview."))]
    language: Option<String>,
    #[arg(long, help = CONFIG_HELP)]
    config: Option<String>,
}

/// Threads for the tensor operations unless `RAYON_NUM_THREADS` says otherwise. The model is
/// small enough that more threads cost more to dispatch than they save: four were fastest on
/// the machines measured, ahead of both two and eight.
const DEFAULT_NUM_THREADS: usize = 4;

fn main() -> ExitCode {
    let cli = Cli::parse();
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        xn::set_num_threads(DEFAULT_NUM_THREADS.min(cpus));
    }
    let result = match cli.command {
        Command::Serve(args) => serve(args),
        Command::Generate(args) => generate(args),
        Command::ExportVoice(args) => export_voice(args),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            log::error!("{e:#}");
            ExitCode::FAILURE
        }
    }
}

/// Runs `$body` with `$q` bound to the f32 or the int8 backend.
macro_rules! with_backend {
    ($quantize:expr, $q:ident => $body:expr) => {
        if $quantize {
            type $q = xn::quantized::Q80F32;
            $body
        } else {
            type $q = xn::Unquantized<f32, xn::CpuDevice>;
            $body
        }
    };
}

fn load_model<Q: CpuBackend>(
    language: Option<&str>,
    config: Option<&str>,
    options: &LoadOptions,
) -> Result<TtsModel<Q>> {
    let (config, origin) = Config::load(language, config)?;
    TtsModel::load(config, origin, options)
}

fn generate(args: GenerateArgs) -> Result<ExitCode> {
    logging::init(args.quiet);
    let mut sampler_decode_steps = args.sampler_decode_steps;
    if let Some(steps) = args.lsd_decode_steps {
        log::warn!("--lsd-decode-steps is deprecated, use --sampler-decode-steps");
        sampler_decode_steps = steps;
    }
    let mut text = args.text.clone().unwrap_or_else(|| {
        defaults::get_default_text_for_language(args.language.as_deref()).into()
    });
    if text == "-" {
        text = std::io::read_to_string(std::io::stdin()).context("cannot read stdin")?;
    }
    if text.trim().is_empty() {
        log::error!("No input received from stdin.");
        return Ok(ExitCode::FAILURE);
    }
    if args.checkpoint.is_some() {
        bail!(
            "--checkpoint is not supported by this binary: training checkpoints (.pt) need the Python package"
        )
    }
    if args.device != "cpu" {
        bail!(
            "this binary runs on the cpu only, got --device {}",
            args.device
        )
    }
    let options = LoadOptions {
        temperature: args.temperature,
        sampler_decode_steps,
        noise_clamp: args.noise_clamp,
        eos_threshold: args.eos_threshold,
    };
    with_backend!(args.quantize.quantize, Q => generate_with::<Q>(&args, &text, &options))?;
    if args.output_path != "-" {
        log::info!("Results written in {}", args.output_path);
    }
    log::info!("{}", "-".repeat(20));
    log::info!("If you want to try multiple voices and prompts quickly, try the `serve` command.");
    log::info!(
        "If you like Kyutai projects, comment, like, subscribe at https://x.com/kyutai_labs"
    );
    Ok(ExitCode::SUCCESS)
}

fn generate_with<Q: CpuBackend>(
    args: &GenerateArgs,
    text: &str,
    options: &LoadOptions,
) -> Result<()> {
    let model: TtsModel<Q> = load_model(args.language.as_deref(), args.config.as_deref(), options)?;
    let voice = args.voice.clone().unwrap_or_else(|| {
        defaults::get_default_voice_for_language(
            args.language.as_deref(),
            args.config.is_some() || args.checkpoint.is_some(),
        )
    });
    let voice = model.get_state_for_audio_prompt(&voice, false)?;

    // Stream audio generation directly to the file or stdout.
    let (sink, seekable) = audio::open_output(&args.output_path)?;
    let mut writer = audio::StreamingWavWriter::new(sink, model.sample_rate() as u32, seekable);
    writer.write_header_initial()?;
    {
        let mut on_chunk = |pcm: &[f32]| -> Result<()> { Ok(writer.write_pcm(pcm)?) };
        model.generate_audio_stream(
            &voice,
            text,
            args.max_tokens,
            args.frames_after_eos,
            &AtomicBool::new(false),
            &mut on_chunk,
        )?;
    }
    writer.finalize()?;
    Ok(())
}

fn serve(args: ServeArgs) -> Result<ExitCode> {
    logging::init(false);
    if args.reload {
        log::warn!("--reload is a development feature of the Python server, it is ignored");
    }
    let _ = args.no_reload;
    with_backend!(args.quantize.quantize, Q => serve_with::<Q>(&args))?;
    Ok(ExitCode::SUCCESS)
}

fn serve_with<Q: CpuBackend>(args: &ServeArgs) -> Result<()>
where
    TtsModel<Q>: Send + Sync,
    VoiceState<Q>: Send + Sync,
{
    let model: TtsModel<Q> = load_model(
        args.language.as_deref(),
        args.config.as_deref(),
        &LoadOptions::default(),
    )?;
    let default_voice = args.default_voice.clone().unwrap_or_else(|| {
        defaults::get_default_voice_for_language(args.language.as_deref(), args.config.is_some())
    });
    // Resolved before serving: a voice that cannot be loaded fails at startup instead of on the
    // first request.
    let default_voice = model.get_state_for_audio_prompt(&default_voice, false)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(server::serve(model, default_voice, &args.host, args.port))
}

fn export_voice(args: ExportVoiceArgs) -> Result<ExitCode> {
    logging::init(args.quiet);
    let model: TtsModel<xn::Unquantized<f32, xn::CpuDevice>> = load_model(
        args.language.as_deref(),
        args.config.as_deref(),
        &LoadOptions::default(),
    )?;
    let voice = model.get_state_for_audio_prompt(&args.audio_path, true)?;
    voice.export(std::path::Path::new(&args.export_path))?;
    Ok(ExitCode::SUCCESS)
}
