use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use base64::{Engine as _, engine::general_purpose};
use pocket_tts::TTSModel;
use pocket_tts_cli::server::{routes, state::AppState};
use pocket_tts_cli::voice::resolve_voice;
use serde_json::json;
use std::path::Path;
use tower::ServiceExt;

/// Create test app state
fn create_test_app() -> Option<axum::Router> {
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
async fn test_api_base64_cloning() {
    println!("Loading model for Base64 API test...");
    let Some(app) = create_test_app() else { return };

    // Read ref.wav to bytes and base64 encode
    let ref_wav = "../../ref.wav";
    if !Path::new(ref_wav).exists() {
        println!("Skipping base64 test: ref.wav not found");
        return;
    }

    let wav_bytes = std::fs::read(ref_wav).unwrap();
    let b64 = general_purpose::STANDARD.encode(&wav_bytes);

    let body = json!({
        "text": "Base64 cloning test",
        "voice": b64
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

    // Check if body is valid wav
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let cursor = std::io::Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor).unwrap();
    assert!(reader.duration() > 0);
}
