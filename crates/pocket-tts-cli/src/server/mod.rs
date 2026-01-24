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

    println!("  ✓ Model loaded (sample rate: {}Hz)", model.sample_rate);

    // Pre-load default voice
    println!("  Loading default voice: {}...", args.voice);
    let default_voice_state = resolve_voice(&model, Some(&args.voice))?;
    println!("  ✓ Default voice ready");

    let state = state::AppState::new(model, default_voice_state);
    let app = routes::create_router(state);

    let addr = format!("{}:{}", args.host, args.port);

    print_endpoints(&args.host, args.port);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    println!("  👋 Server stopped gracefully");

    Ok(())
}

/// Wait for Ctrl+C or SIGTERM signal
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            println!("\n  ⚠️  Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            println!("\n  ⚠️  Received SIGTERM, shutting down...");
        },
    }
}
