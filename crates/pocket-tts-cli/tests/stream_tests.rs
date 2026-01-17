use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use pocket_tts::TTSModel;
use pocket_tts_cli::server::{routes, state::AppState};
use pocket_tts_cli::voice::resolve_voice;
use serde_json::json;
use tokio_stream::StreamExt;
use tower::ServiceExt;

/// Create test app state
fn create_test_app() -> Option<axum::Router> {
    println!("Loading model for Stream API test...");
    let model = match TTSModel::load("b6369a24") {
        Ok(m) => m,
        Err(e) => {
            println!("Skipping test: could not load model: {}", e);
            return None;
        }
    };

    let default_voice = match resolve_voice(&model, Some("alba")) {
        Ok(v) => v,
        Err(e) => {
            println!("Skipping test: could not load voice: {}", e);
            return None;
        }
    };

    let state = AppState::new(model, default_voice);
    Some(routes::create_router(state))
}

#[tokio::test]
async fn test_api_stream_endpoint() {
    let Some(app) = create_test_app() else { return };

    let body = json!({
        "text": "Streaming test",
        // Use default alba voice
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/stream")
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "application/octet-stream"
    );

    // Collect stream
    let mut stream = response.into_body().into_data_stream();
    let mut total_bytes = 0;
    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.expect("Stream chunk error");
        total_bytes += chunk.len();
    }

    println!("Total streamed bytes: {}", total_bytes);
    assert!(total_bytes > 0);
}
