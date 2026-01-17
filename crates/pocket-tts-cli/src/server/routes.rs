//! API routes configuration

use crate::server::handlers;
use crate::server::state::AppState;
use axum::{
    Router,
    routing::{get, post},
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

pub fn create_router(state: AppState) -> Router {
    // CORS layer for web interface
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Static files router
    let static_router = Router::new().fallback(handlers::serve_static);

    Router::new()
        // Web interface
        .route("/", get(handlers::serve_index))
        .nest("/static", static_router)
        // Health check
        .route("/health", get(handlers::health_check))
        // Generation endpoints
        .route("/generate", post(handlers::generate))
        .route("/stream", post(handlers::generate_stream))
        // Python API compatibility (multipart form)
        .route("/tts", post(handlers::tts_form))
        // OpenAI compatibility
        .route("/v1/audio/speech", post(handlers::openai_speech))
        // Middleware
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
