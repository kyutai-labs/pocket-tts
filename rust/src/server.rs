//! Port of the FastAPI app of `pocket_tts/main.py`: the web page, `/health` and `/tts`, which
//! streams a WAV as it is generated.

// The errors are HTTP responses, built once per failed request.
#![allow(clippy::result_large_err)]

use crate::model::CpuBackend;
use crate::model::{TtsModel, VoiceState};
use anyhow::Result;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Voices requested by URL keep their state, as `lru_cache(maxsize=2)` does in Python.
const VOICE_CACHE_SIZE: usize = 2;

struct App<Q: CpuBackend> {
    model: TtsModel<Q>,
    default_voice: Arc<VoiceState<Q>>,
    voice_cache: Mutex<VecDeque<(String, Arc<VoiceState<Q>>)>>,
    /// Generations run one at a time, as with the Python server.
    generation: Mutex<()>,
}

type Shared<Q> = Arc<App<Q>>;

pub async fn serve<Q: CpuBackend>(
    model: TtsModel<Q>,
    default_voice: VoiceState<Q>,
    host: &str,
    port: u16,
) -> Result<()>
where
    TtsModel<Q>: Send + Sync,
    VoiceState<Q>: Send + Sync,
{
    let app: Shared<Q> = Arc::new(App {
        model,
        default_voice: Arc::new(default_voice),
        voice_cache: Mutex::new(VecDeque::new()),
        generation: Mutex::new(()),
    });
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(
            [
                "http://localhost:3000",
                "https://pod1-10007.internal.kyutai.org",
                "https://kyutai.org",
            ]
            .map(HeaderValue::from_static),
        )
        .allow_credentials(true)
        .allow_methods(tower_http::cors::AllowMethods::mirror_request())
        .allow_headers(tower_http::cors::AllowHeaders::mirror_request());
    let router = Router::new()
        .route("/", get(root::<Q>))
        .route("/health", get(health))
        .route("/tts", post(text_to_speech::<Q>))
        .layer(cors)
        .with_state(app);

    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    log::info!("Started server process [{}]", std::process::id());
    let address = listener.local_addr()?;
    let shown = if address.ip().is_loopback() && host == "localhost" {
        format!("http://localhost:{}", address.port())
    } else {
        format!("http://{address}")
    };
    log::info!("Server running on {shown} (Press CTRL+C to quit)");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            log::info!("Shutting down");
        })
        .await?;
    Ok(())
}

async fn root<Q: CpuBackend>(State(app): State<Shared<Q>>) -> Html<String>
where
    TtsModel<Q>: Send + Sync,
    VoiceState<Q>: Send + Sync,
{
    let origin = app.model.origin.display_path();
    let text = crate::defaults::get_default_text_for_language(Some(&origin));
    Html(crate::config::INDEX_HTML.replace("DEFAULT_TEXT_PROMPT", text))
}

async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({ "status": "healthy" }))
}

fn http_error(status: StatusCode, detail: impl Into<String>) -> Response {
    (
        status,
        axum::Json(serde_json::json!({ "detail": detail.into() })),
    )
        .into_response()
}

/// The form fields of `/tts`, sent as multipart (as the web page does) or urlencoded.
#[derive(Default)]
struct TtsForm {
    text: Option<String>,
    voice_url: Option<String>,
    voice_wav: Option<(Option<String>, Bytes)>,
}

async fn read_form(request: Request) -> Result<TtsForm, Response> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let mut form = TtsForm::default();
    if content_type.starts_with("multipart/form-data") {
        let mut multipart = Multipart::from_request(request, &())
            .await
            .map_err(|e| http_error(StatusCode::BAD_REQUEST, e.body_text()))?;
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| http_error(StatusCode::BAD_REQUEST, e.body_text()))?
        {
            let name = field.name().unwrap_or("").to_string();
            let file_name = field.file_name().map(str::to_string);
            let data = field
                .bytes()
                .await
                .map_err(|e| http_error(StatusCode::BAD_REQUEST, e.body_text()))?;
            match name.as_str() {
                "text" => form.text = Some(String::from_utf8_lossy(&data).into_owned()),
                "voice_url" => form.voice_url = Some(String::from_utf8_lossy(&data).into_owned()),
                "voice_wav" => form.voice_wav = Some((file_name, data)),
                _ => {}
            }
        }
    } else {
        let axum::Form(fields) = axum::Form::<Vec<(String, String)>>::from_request(request, &())
            .await
            .map_err(|e| http_error(StatusCode::UNPROCESSABLE_ENTITY, e.body_text()))?;
        for (name, value) in fields {
            match name.as_str() {
                "text" => form.text = Some(value),
                "voice_url" => form.voice_url = Some(value),
                _ => {}
            }
        }
    }
    Ok(form)
}

async fn text_to_speech<Q: CpuBackend>(State(app): State<Shared<Q>>, request: Request) -> Response
where
    TtsModel<Q>: Send + Sync,
    VoiceState<Q>: Send + Sync,
{
    let form = match read_form(request).await {
        Ok(form) => form,
        Err(response) => return response,
    };
    let Some(text) = form.text else {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            axum::Json(serde_json::json!({ "detail": [{
                "type": "missing", "loc": ["body", "text"], "msg": "Field required", "input": null
            }]})),
        )
            .into_response();
    };
    if text.trim().is_empty() {
        return http_error(StatusCode::BAD_REQUEST, "Text cannot be empty");
    }
    if form.voice_url.is_some() && form.voice_wav.is_some() {
        return http_error(
            StatusCode::BAD_REQUEST,
            "Cannot provide both voice_url and voice_wav",
        );
    }

    let voice_app = app.clone();
    let voice = tokio::task::spawn_blocking(move || -> Result<Arc<VoiceState<Q>>, Response> {
        let app = voice_app;
        if let Some(voice_url) = form.voice_url {
            let allowed = voice_url.starts_with("http://")
                || voice_url.starts_with("https://")
                || voice_url.starts_with("hf://")
                || crate::defaults::is_predefined_voice(&voice_url);
            if !allowed {
                return Err(http_error(
                    StatusCode::BAD_REQUEST,
                    "voice_url must start with http://, https://, or hf://",
                ));
            }
            let voice = cached_voice(&app, &voice_url).map_err(internal_error)?;
            log::warn!("Using voice from URL: {voice_url}");
            Ok(voice)
        } else if let Some((file_name, data)) = form.voice_wav {
            // Keep the extension, the audio decoder goes by it.
            let suffix = file_name
                .as_deref()
                .and_then(|name| std::path::Path::new(name).extension())
                .map(|ext| format!(".{}", ext.to_string_lossy()))
                .unwrap_or_else(|| ".wav".to_string());
            let file = tempfile::Builder::new()
                .suffix(&suffix)
                .tempfile()
                .map_err(internal_error)?;
            std::fs::write(file.path(), &data).map_err(internal_error)?;
            let path = file.path().to_string_lossy().into_owned();
            let voice = app
                .model
                .get_state_for_audio_prompt(&path, true)
                .map_err(internal_error)?;
            Ok(Arc::new(voice))
        } else {
            Ok(app.default_voice.clone())
        }
    })
    .await;
    let voice = match voice {
        Ok(Ok(voice)) => voice,
        Ok(Err(response)) => return response,
        Err(e) => return internal_error(e),
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    std::thread::spawn(move || {
        let _guard = app.generation.lock().unwrap_or_else(|e| e.into_inner());
        let stop = AtomicBool::new(false);
        let sink = crate::audio::Unseekable(ChannelWriter {
            tx: tx.clone(),
            stop: &stop,
        });
        let mut writer =
            crate::audio::StreamingWavWriter::new(sink, app.model.sample_rate() as u32, false);
        let result = (|| -> Result<()> {
            writer.write_header_initial()?;
            let mut on_chunk = |pcm: &[f32]| -> Result<()> { Ok(writer.write_pcm(pcm)?) };
            app.model.generate_audio_stream(
                &voice,
                &text,
                crate::defaults::MAX_TOKEN_PER_CHUNK,
                None,
                &stop,
                &mut on_chunk,
            )?;
            writer.finalize()?;
            Ok(())
        })();
        if let Err(e) = result
            && !stop.load(Ordering::Relaxed)
        {
            log::error!("Error in generation: {e:#}");
            let _ = tx.blocking_send(Err(std::io::Error::other(format!("{e:#}"))));
        }
    });

    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    Response::builder()
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=generated_speech.wav",
        )
        .body(body)
        .unwrap()
}

/// Sends the WAV bytes to the response. When the client has gone, raises `stop`, so that the
/// generation ends instead of finishing for nobody.
struct ChannelWriter<'a> {
    tx: tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    stop: &'a AtomicBool,
}

impl std::io::Write for ChannelWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self
            .tx
            .blocking_send(Ok(Bytes::copy_from_slice(buf)))
            .is_err()
        {
            self.stop.store(true, Ordering::Relaxed);
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "client disconnected",
            ));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn cached_voice<Q: CpuBackend>(app: &App<Q>, voice_url: &str) -> Result<Arc<VoiceState<Q>>> {
    {
        let cache = app.voice_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, voice)) = cache.iter().find(|(url, _)| url == voice_url) {
            return Ok(voice.clone());
        }
    }
    let voice = Arc::new(app.model.get_state_for_audio_prompt(voice_url, false)?);
    let mut cache = app.voice_cache.lock().unwrap_or_else(|e| e.into_inner());
    if cache.len() >= VOICE_CACHE_SIZE {
        cache.pop_front();
    }
    cache.push_back((voice_url.to_string(), voice.clone()));
    Ok(voice)
}

fn internal_error(e: impl std::fmt::Display) -> Response {
    log::error!("{e:#}");
    (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response()
}
