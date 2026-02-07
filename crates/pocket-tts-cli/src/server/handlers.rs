//! HTTP request handlers

use crate::server::state::{AppState, VoiceStateCache};
use crate::voice::{resolve_voice, voice_cache_key};
#[cfg(feature = "web-ui")]
use axum::response::Html;
use axum::{
    Json,
    body::Body,
    extract::{Multipart, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
#[cfg(feature = "web-ui")]
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt as _;

type SharedVoiceState = std::sync::Arc<pocket_tts::ModelState>;
type VoiceCache = std::sync::Arc<std::sync::Mutex<VoiceStateCache>>;

// Embed static files at compile time
#[cfg(feature = "web-ui")]
#[derive(Embed)]
#[folder = "web/dist"]
struct StaticAssets;

// ============================================================================
// Static file serving
// ============================================================================

/// Serve static files (CSS, JS, images)
#[cfg(feature = "web-ui")]
pub async fn serve_static(uri: axum::http::Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');

    // If path is empty, serve index.html
    let path = if path.is_empty() { "index.html" } else { path };
    let path = percent_encoding::percent_decode_str(path).decode_utf8_lossy();
    // Force a concrete string type to avoid brittle AsRef inference across dependency/toolchain changes.
    let path: &str = path.as_ref();

    match StaticAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, mime.as_ref().parse().unwrap());
            (headers, content.data.to_vec()).into_response()
        }
        None => {
            // If the request doesn't match a static file, return index.html (for SPA routing)
            // But only if it doesn't look like a file request (to avoid infinite loops on 404s)
            if !path.contains('.')
                && let Some(content) = StaticAssets::get("index.html")
            {
                return Html(content.data.to_vec()).into_response();
            }
            (StatusCode::NOT_FOUND, "File not found").into_response()
        }
    }
}

// ============================================================================
// Health check
// ============================================================================

#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
}

pub async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

// ============================================================================
// Generation (JSON API)
// ============================================================================

#[derive(Deserialize)]
pub struct GenerateRequest {
    text: String,
    voice: Option<String>,
    temperature: Option<f32>,
    lsd_steps: Option<usize>,
    eos_threshold: Option<f32>,
    noise_clamp: Option<f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn resolve_voice_cached(
    model: &pocket_tts::TTSModel,
    default_voice: &SharedVoiceState,
    voice_cache: &VoiceCache,
    voice_spec: Option<&str>,
) -> anyhow::Result<SharedVoiceState> {
    let Some(spec) = voice_spec else {
        return Ok(default_voice.clone());
    };

    let key = voice_cache_key(spec);

    {
        let mut cache = voice_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
        if let Some(cached) = cache.get(&key) {
            return Ok(cached.clone());
        }
    }

    let resolved = std::sync::Arc::new(resolve_voice(model, Some(spec))?);
    let mut cache = voice_cache
        .lock()
        .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
    cache.put(key, resolved.clone());
    Ok(resolved)
}

pub async fn generate(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> Response {
    // Acquire lock for sequential processing
    let _guard = state.lock.lock().await;

    let model = state.model.clone();
    let default_voice = state.default_voice_state.clone();
    let voice_cache = state.voice_cache.clone();
    let text = payload.text.clone();
    let voice_spec = payload.voice.clone();

    // Run generation in blocking thread
    let result = tokio::task::spawn_blocking(move || {
        // Resolve voice (use default if not specified)
        let voice_state =
            resolve_voice_cached(&model, &default_voice, &voice_cache, voice_spec.as_deref())?;

        // Override model params if provided in request
        let mut model_cloned = (*model).clone();
        if let Some(t) = payload.temperature {
            model_cloned.temp = t;
        }
        if let Some(s) = payload.lsd_steps {
            model_cloned.lsd_decode_steps = s;
        }
        if let Some(e) = payload.eos_threshold {
            model_cloned.eos_threshold = e;
        }
        if let Some(nc) = payload.noise_clamp {
            model_cloned.noise_clamp = Some(nc);
        }

        // Generate audio
        tracing::info!("Starting generation for text length: {} chars", text.len());
        let mut audio_chunks = Vec::new();
        for chunk in model_cloned.generate_stream_long(&text, &voice_state) {
            audio_chunks.push(chunk?);
        }
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = candle_core::Tensor::cat(&audio_chunks, 2)?;
        let audio = audio.squeeze(0)?;

        // Encode as WAV
        let mut buffer = std::io::Cursor::new(Vec::new());
        pocket_tts::audio::write_wav_to_writer(&mut buffer, &audio, model.sample_rate as u32)?;

        Ok::<Vec<u8>, anyhow::Error>(buffer.into_inner())
    })
    .await;

    match result {
        Ok(Ok(wav_bytes)) => {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, "audio/wav".parse().unwrap());
            headers.insert(
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"pocket-tts-output.wav\""
                    .parse()
                    .unwrap(),
            );
            (StatusCode::OK, headers, Body::from(wav_bytes)).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Task error: {}", e),
            }),
        )
            .into_response(),
    }
}

// ============================================================================
// Streaming generation
// ============================================================================

pub async fn generate_stream(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> Response {
    let model = state.model.clone();
    let default_voice = state.default_voice_state.clone();
    let voice_cache = state.voice_cache.clone();
    let text = payload.text.clone();
    let voice_spec = payload.voice.clone();
    let lock = state.lock.clone();

    // Channel for streaming chunks
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, anyhow::Error>>(10);

    // Spawn generation task
    tokio::spawn(async move {
        let _guard = lock.lock().await;

        let tx_inner = tx.clone();
        let result = tokio::task::spawn_blocking(move || {
            // Resolve voice
            let voice_state =
                resolve_voice_cached(&model, &default_voice, &voice_cache, voice_spec.as_deref())?;

            // Override model params if provided in request
            let mut model_cloned = (*model).clone();
            if let Some(t) = payload.temperature {
                model_cloned.temp = t;
            }
            if let Some(s) = payload.lsd_steps {
                model_cloned.lsd_decode_steps = s;
            }
            if let Some(e) = payload.eos_threshold {
                model_cloned.eos_threshold = e;
            }
            if let Some(nc) = payload.noise_clamp {
                model_cloned.noise_clamp = Some(nc);
            }

            // Stream audio chunks
            tracing::info!(
                "Starting streaming generation for text length: {} chars",
                text.len()
            );
            for (i, chunk_res) in model_cloned
                .generate_stream_long(&text, &voice_state)
                .enumerate()
            {
                if i > 0 && i % 20 == 0 {
                    tracing::info!("Generated chunk {}", i);
                }
                match chunk_res {
                    Ok(chunk) => {
                        // Convert tensor to 16-bit PCM bytes
                        let chunk = chunk.squeeze(0).map_err(|e| anyhow::anyhow!(e))?;
                        let bytes = pocket_tts::audio::pcm_i16_le_bytes(&chunk)
                            .map_err(|e| anyhow::anyhow!(e))?;

                        if tx_inner.blocking_send(Ok(bytes)).is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        let _ = tx_inner.blocking_send(Err(anyhow::anyhow!(e)));
                        break;
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        if let Err(e) = result.await {
            let _ = tx.send(Err(anyhow::anyhow!("Task error: {}", e))).await;
        }
    });

    // Convert channel to stream
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body_stream = stream.map(
        |res| -> std::result::Result<axum::body::Bytes, std::io::Error> {
            match res {
                Ok(bytes) => Ok(axum::body::Bytes::from(bytes)),
                Err(e) => Err(std::io::Error::other(e.to_string())),
            }
        },
    );

    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(Body::from_stream(body_stream))
        .unwrap()
}

// ============================================================================
// Python API compatibility (/tts with multipart form)
// ============================================================================

pub async fn tts_form(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    let mut text: Option<String> = None;
    let mut voice_url: Option<String> = None;
    let mut voice_wav_bytes: Option<Vec<u8>> = None;

    // Parse multipart form
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "text" => {
                text = field.text().await.ok();
            }
            "voice_url" => {
                voice_url = field.text().await.ok();
            }
            "voice_wav" => {
                voice_wav_bytes = field.bytes().await.ok().map(|b| b.to_vec());
            }
            _ => {}
        }
    }

    let text = match text {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "Text is required".to_string(),
                }),
            )
                .into_response();
        }
    };

    // Determine voice
    let voice = if let Some(bytes) = voice_wav_bytes {
        // Use uploaded WAV - encode as base64 for our resolver
        use base64::{Engine as _, engine::general_purpose};
        Some(format!(
            "data:audio/wav;base64,{}",
            general_purpose::STANDARD.encode(&bytes)
        ))
    } else {
        voice_url
    };

    // Delegate to JSON generate handler
    generate(
        State(state),
        Json(GenerateRequest {
            text,
            voice,
            temperature: None,
            lsd_steps: None,
            eos_threshold: None,
            noise_clamp: None,
        }),
    )
    .await
}

// ============================================================================
// OpenAI compatibility
// ============================================================================

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct OpenAIRequest {
    model: String,
    input: String,
    voice: Option<String>,
    response_format: Option<String>,
}

pub async fn openai_speech(state: State<AppState>, Json(payload): Json<OpenAIRequest>) -> Response {
    // Map OpenAI format to our format
    let req = GenerateRequest {
        text: payload.input,
        voice: payload.voice,
        temperature: None,
        lsd_steps: None,
        eos_threshold: None,
        noise_clamp: None,
    };
    generate(state, Json(req)).await
}
