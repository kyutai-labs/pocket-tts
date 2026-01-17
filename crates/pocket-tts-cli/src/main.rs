//! Pocket TTS CLI - Rust/Candle port
//!
//! A blazingly fast text-to-speech tool.

use anyhow::Result;
use clap::Parser;

use pocket_tts_cli::commands;

/// Pocket TTS - High-quality text-to-speech, blazingly fast on CPU
#[derive(Parser)]
#[command(
    name = "pocket-tts",
    author,
    version,
    about = "Pocket TTS - Blazingly fast text-to-speech",
    long_about = "A Rust/Candle port of Kyutai's Pocket TTS model.\n\n\
                  Generate natural speech from text using neural TTS.\n\
                  Supports voice cloning from audio samples."
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Generate audio from text
    ///
    /// Synthesizes speech from the provided text and saves to a WAV file.
    /// Supports voice cloning using predefined voices or custom audio files.
    Generate(commands::generate::GenerateArgs),

    /// Start the HTTP API server
    ///
    /// Runs a web server providing TTS generation via REST API.
    /// Includes a web interface for interactive use.
    Serve(commands::serve::ServeArgs),

    /// Serve the WASM package and browser demo
    ///
    /// Starts a static file server to test the WebAssembly TTS demo in your browser.
    WasmDemo(commands::wasm_demo::WasmDemoArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Generate(cmd_args) => {
            // Generate is CPU-bound, run synchronously
            commands::generate::run(cmd_args)
        }
        Commands::Serve(cmd_args) => commands::serve::run(cmd_args).await,
        Commands::WasmDemo(cmd_args) => commands::wasm_demo::run(cmd_args).await,
    }
}
