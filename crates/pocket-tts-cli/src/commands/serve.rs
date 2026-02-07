//! Serve command implementation
//!
//! Provides `pocket-tts serve` for HTTP API server.

use anyhow::Result;
use clap::{ArgAction, Parser};
use owo_colors::OwoColorize;

use crate::voice::PREDEFINED_VOICES;

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Host address to bind (default: 127.0.0.1)
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port number to listen on (default: 8000)
    #[arg(short, long, default_value_t = 8000)]
    pub port: u16,

    /// Default voice for API requests (can be overridden per-request)
    #[arg(long, default_value = "alba")]
    pub voice: String,

    /// Model variant
    #[arg(long, default_value = "b6369a24")]
    pub variant: String,

    /// Sampling temperature
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// LSD decode steps
    #[arg(long, default_value = "1")]
    pub lsd_decode_steps: usize,

    /// EOS threshold
    #[arg(long, default_value = "-4.0")]
    pub eos_threshold: f32,

    /// Use simulated int8 quantization for inference
    #[arg(long)]
    pub quantized: bool,

    /// Maximum number of resolved voice states to keep in server LRU cache.
    #[arg(long, default_value_t = 64)]
    pub voice_cache_capacity: usize,

    /// Comma-separated voices to prewarm at startup (e.g. "alba,marius").
    #[arg(long, default_value = "alba")]
    pub prewarm_voices: String,

    /// Run a tiny startup warmup generation to reduce first-request latency.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub warmup: bool,

    /// Override OMP_NUM_THREADS before model load.
    #[arg(long)]
    pub omp_threads: Option<usize>,

    /// Override MKL_NUM_THREADS before model load.
    #[arg(long)]
    pub mkl_threads: Option<usize>,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    print_banner();

    println!(
        "{} Loading model variant: {}",
        "▶".cyan(),
        args.variant.yellow()
    );

    let server_args = args.clone();
    crate::server::start_server(server_args).await
}

fn print_banner() {
    println!();
    println!(
        "  {}  {} {}",
        "🗣️".bold(),
        "Pocket TTS".bold().cyan(),
        "API Server".bold()
    );
    println!(
        "      {} {}",
        "Rust/Candle port".dimmed(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!();
}

/// Print endpoint information after server starts
pub fn print_endpoints(host: &str, port: u16) {
    let base = format!("http://{}:{}", host, port);

    println!();
    println!(
        "  {} {}",
        "✓".green().bold(),
        "Server is running!".green().bold()
    );
    println!();
    println!("  {}", "Endpoints:".bold());
    println!(
        "    {} {}  {}",
        "GET".cyan(),
        format!("{}/", base).white(),
        "Web interface".dimmed()
    );
    println!(
        "    {} {}  {}",
        "GET".cyan(),
        format!("{}/health", base).white(),
        "Health check".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/generate", base).white(),
        "Generate audio (JSON body)".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/stream", base).white(),
        "Streaming generation".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/tts", base).white(),
        "Python-compatible endpoint (form data)".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/v1/audio/speech", base).white(),
        "OpenAI-compatible".dimmed()
    );
    println!();
    println!(
        "  {} Available voices: {}",
        "💡".dimmed(),
        PREDEFINED_VOICES.join(", ").dimmed()
    );
    println!();
    println!(
        "  {} curl -X POST {}/generate -H 'Content-Type: application/json' -d '{{\"text\": \"Hello world\"}}' --output test.wav",
        "Example:".dimmed(),
        base
    );
    println!();
}
