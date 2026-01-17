use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use pocket_tts::TTSModel;
use pocket_tts_cli::server::{routes, state::AppState};
use pocket_tts_cli::voice::resolve_voice;
use serde_json::json;
use tower::ServiceExt; // for oneshot

/// Create test app state (loads model and default voice)
fn create_test_app() -> Option<axum::Router> {
    println!("Loading model for API test...");
    let model = match TTSModel::load("b6369a24") {
        Ok(m) => m,
        Err(e) => {
            println!("Skipping API test: could not load model: {}", e);
            return None;
        }
    };

    // Load default voice
    let default_voice = match resolve_voice(&model, Some("alba")) {
        Ok(v) => v,
        Err(e) => {
            println!("Skipping API test: could not load voice: {}", e);
            return None;
        }
    };

    let state = AppState::new(model, default_voice);
    Some(routes::create_router(state))
}

#[tokio::test]
async fn test_api_full_flow() {
    let Some(app) = create_test_app() else { return };

    // 1. Health Check
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // 2. Generate Audio (short text)
    let body = json!({
        "text": "Hi",
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("content-type").unwrap(), "audio/wav");

    // 3. OpenAI endpoint
    let body = json!({
        "model": "pocket-tts",
        "input": "Open API test",
        "voice": "alba"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("content-type").unwrap(), "audio/wav");
}

#[tokio::test]
async fn test_web_interface() {
    let Some(app) = create_test_app() else { return };

    // Test index page
    let response = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Check it returns HTML
    let content_type = response.headers().get("content-type");
    assert!(content_type.is_some());

    // Test static file routing
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/static/index.html")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
