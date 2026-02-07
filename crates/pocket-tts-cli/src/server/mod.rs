//! HTTP API Server
//!
//! Axum-based server providing TTS generation endpoints.

use anyhow::Result;
use pocket_tts::TTSModel;

use crate::commands::serve::{ServeArgs, print_endpoints};
use crate::voice::{resolve_voice, voice_cache_key};

pub mod handlers;
pub mod routes;
pub mod state;

pub async fn start_server(args: ServeArgs) -> Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt::try_init();

    if let Some(omp_threads) = args.omp_threads {
        // SAFETY: environment is configured at startup before serving requests.
        unsafe {
            std::env::set_var("OMP_NUM_THREADS", omp_threads.to_string());
        }
        println!("  Set OMP_NUM_THREADS={omp_threads}");
    }
    if let Some(mkl_threads) = args.mkl_threads {
        // SAFETY: environment is configured at startup before serving requests.
        unsafe {
            std::env::set_var("MKL_NUM_THREADS", mkl_threads.to_string());
        }
        println!("  Set MKL_NUM_THREADS={mkl_threads}");
    }

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

    let state = state::AppState::new(model, default_voice_state, args.voice_cache_capacity);
    {
        let mut cache = state
            .voice_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
        cache.put(
            voice_cache_key(&args.voice),
            state.default_voice_state.clone(),
        );
    }

    for voice in args
        .prewarm_voices
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let key = voice_cache_key(voice);
        let already_cached = {
            let mut cache = state
                .voice_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
            cache.get(&key).is_some()
        };
        if already_cached {
            continue;
        }

        println!("  Prewarming voice: {voice}...");
        match resolve_voice(&state.model, Some(voice)) {
            Ok(vs) => {
                let mut cache = state
                    .voice_cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
                cache.put(key, std::sync::Arc::new(vs));
                println!("  û Voice prewarmed: {voice}");
            }
            Err(e) => {
                println!("  !! Failed to prewarm voice '{voice}': {e}");
            }
        }
    }

    if args.warmup {
        println!("  Running startup warmup...");
        let mut warmup_iter = state
            .model
            .generate_stream_long("warmup", &state.default_voice_state);
        if let Some(frame_res) = warmup_iter.next() {
            frame_res?;
        }
        println!("  û Warmup complete");
    }

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
