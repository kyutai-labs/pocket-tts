use anyhow::Result;
use axum::{Router, response::Redirect};
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use tower_http::services::ServeDir;
use tracing::info;

#[derive(Parser, Debug)]
pub struct WasmDemoArgs {
    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,

    /// Root directory to serve from (default: pocket-tts crate dir)
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Directory containing model assets (config.yaml, etc.) to serve at root
    #[arg(short, long)]
    pub models: Option<PathBuf>,
}

pub async fn run(args: WasmDemoArgs) -> Result<()> {
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    // Resolve project root if not provided
    let serve_root = if let Some(root) = args.root {
        root
    } else {
        // Assume we are running from within the candle workspace
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("pocket-tts")
    };

    if !serve_root.exists() {
        anyhow::bail!(
            "Serve root does not exist: {:?}. Please ensure you are running from the candle workspace.",
            serve_root
        );
    }

    info!("Serving WASM demo from: {:?}", serve_root);

    // The demo expects:
    // /pkg/pocket_tts.js (from ../../pkg/pocket_tts.js relative to index)
    // /examples/wasm/index.html
    // /config.yaml, /model.safetensors, /tokenizer.json (at root)

    let mut app = Router::new()
        .fallback_service(ServeDir::new(&serve_root))
        .route(
            "/",
            axum::routing::get(|| async { Redirect::to("/examples/wasm/index.html") }),
        );

    if let Some(models_dir) = args.models {
        if !models_dir.exists() {
            anyhow::bail!("Models directory does not exist: {:?}", models_dir);
        }
        info!("Serving model assets from: {:?}", models_dir);
        // Serve models directory at the root as well by chaining services
        app = app.fallback_service(ServeDir::new(models_dir).fallback(ServeDir::new(&serve_root)));
    }

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("WASM demo available at http://{}", addr);
    info!(
        "Open your browser to: http://{}/examples/wasm/index.html",
        addr
    );

    println!("\n🚀 WASM Demo started!");
    println!("👉 URL: http://{}\n", addr);
    println!("Note: Ensure you have built the wasm package first:");
    println!("      wasm-pack build --target web --out-dir pkg . -- --features wasm\n");

    axum::serve(listener, app).await?;

    Ok(())
}
