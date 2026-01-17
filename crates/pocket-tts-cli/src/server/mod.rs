//! HTTP API Server
//!
//! Axum-based server providing TTS generation endpoints.

use anyhow::Result;
use pocket_tts::TTSModel;

use crate::commands::serve::{ServeArgs, print_endpoints};
use crate::voice::resolve_voice;

pub mod handlers;
pub mod routes;
pub mod state;

pub async fn start_server(args: ServeArgs) -> Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt::try_init();

    // Load model with configured parameters
    let model = if args.quantized {
        #[cfg(feature = "quantized")]
        {
            TTSModel::load_quantized_with_params(
                &args.variant,
                args.temperature,
                args.lsd_decode_steps,
                args.eos_threshold,
            )?
        }
        #[cfg(not(feature = "quantized"))]
        {
            anyhow::bail!("Quantization feature not enabled. Rebuild with --features quantized");
        }
    } else {
        TTSModel::load_with_params(
            &args.variant,
            args.temperature,
            args.lsd_decode_steps,
            args.eos_threshold,
        )?
    };

    println!(
        "  {} Model loaded (sample rate: {}Hz)",
        "✓".to_string(),
        model.sample_rate
    );

    // Pre-load default voice
    println!("  Loading default voice: {}...", args.voice);
    let default_voice_state = resolve_voice(&model, Some(&args.voice))?;
    println!("  {} Default voice ready", "✓");

    let state = state::AppState::new(model, default_voice_state);
    let app = routes::create_router(state);

    let addr = format!("{}:{}", args.host, args.port);

    print_endpoints(&args.host, args.port);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
