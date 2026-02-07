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

    let state = AppState::new(model, default_voice, 64);
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

#[cfg(feature = "web-ui")]
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
                .uri("/index.html")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[cfg(feature = "web-ui")]
#[tokio::test]
async fn test_static_index_content_type_html() {
    use axum::{http::Uri, response::IntoResponse};

    let response = pocket_tts_cli::server::handlers::serve_static(Uri::from_static("/index.html"))
        .await
        .into_response();

    assert_eq!(response.status(), StatusCode::OK);

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("text/html"),
        "unexpected content-type: {content_type}"
    );
}

#[cfg(feature = "web-ui")]
#[tokio::test]
async fn test_static_spa_fallback_for_route() {
    use axum::{http::Uri, response::IntoResponse};

    let response =
        pocket_tts_cli::server::handlers::serve_static(Uri::from_static("/app/settings"))
            .await
            .into_response();

    assert_eq!(response.status(), StatusCode::OK);

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("text/html"),
        "unexpected content-type: {content_type}"
    );
}

#[cfg(feature = "web-ui")]
#[tokio::test]
async fn test_static_missing_file_404() {
    use axum::{http::Uri, response::IntoResponse};

    let response = pocket_tts_cli::server::handlers::serve_static(Uri::from_static(
        "/definitely-missing-file.zzz",
    ))
    .await
    .into_response();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
