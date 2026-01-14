---
description: Phase 5 - CLI (Clap) & Axum API Server for Pocket TTS Candle Port
---

# Phase 5: CLI & API Server

> [!IMPORTANT]
> **Before starting this workflow, you MUST read `AGENTS.md`** to understand the Python CLI and API.
> 
> **This workflow is for the Rust/Candle port ONLY.** Do not modify any Python code.

## Overview

This phase provides user interfaces: a Clap-based CLI and an Axum-based HTTP API server, mirroring the Python implementation's `generate` and `serve` commands.

## Project Structure

```
candle/crates/pocket-tts-cli/src/
├── main.rs          # Entry point
├── commands/
│   ├── mod.rs
│   ├── generate.rs  # Generate command
│   └── serve.rs     # Serve command
└── server/
    ├── mod.rs
    ├── routes.rs    # API endpoints
    └── handlers.rs  # Request handlers
```

---

## Test-Driven Development Philosophy

> [!NOTE]
> For CLI/API, focus on:
> 1. **Argument parsing** - All options work correctly
> 2. **API contracts** - Endpoints match Python API
> 3. **Error handling** - Graceful failures with helpful messages

Target: **80% test coverage** including integration tests.

---

## Tasks

### 1. Implement CLI with `generate` Command

**Reference:** `pocket_tts/main.py` (Typer CLI)

**Write tests first:**
```rust
#[test]
fn test_cli_generate_basic() {
    let cmd = Command::cargo_bin("pocket-tts")?;
    cmd.arg("generate")
       .arg("--text").arg("Hello")
       .arg("--output").arg("test.wav")
       .assert()
       .success();
    assert!(Path::new("test.wav").exists());
}

#[test]
fn test_cli_generate_with_voice() {
    let cmd = Command::cargo_bin("pocket-tts")?;
    cmd.arg("generate")
       .arg("--text").arg("Hello")
       .arg("--voice").arg("ref.wav")
       .arg("--output").arg("test.wav")
       .assert()
       .success();
}

#[test]
fn test_cli_help() {
    let cmd = Command::cargo_bin("pocket-tts")?;
    cmd.arg("--help")
       .assert()
       .success()
       .stdout(contains("generate"));
}
```

**CLI structure:**
```bash
pocket-tts generate [OPTIONS] --text <TEXT>
    --text <TEXT>       Text to synthesize
    --voice <PATH>      Voice prompt audio file (default: built-in)
    --output <PATH>     Output audio file (default: output.wav)
    --steps <N>         Flow decode steps (default: 16)
    --seed <N>          Random seed

pocket-tts serve [OPTIONS]
    --host <HOST>       Bind address (default: 127.0.0.1)
    --port <PORT>       Port number (default: 8000)
```

### 2. Implement Axum API Server

**Reference:** `pocket_tts/main.py::serve` (FastAPI + Uvicorn)

**Write tests first:**
```rust
#[tokio::test]
async fn test_api_health() {
    let app = create_app();
    let response = app.oneshot(Request::get("/health").body(Body::empty())?).await?;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_api_generate() {
    let app = create_app();
    let body = json!({"text": "Hello", "voice": null});
    let response = app.oneshot(
        Request::post("/generate")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))?
    ).await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("Content-Type").unwrap(), "audio/wav");
}

#[tokio::test]
async fn test_api_stream() {
    let app = create_app();
    let body = json!({"text": "Hello"});
    let response = app.oneshot(
        Request::post("/stream")
            .body(Body::from(body.to_string()))?
    ).await?;
    assert_eq!(response.status(), StatusCode::OK);
    // Verify Server-Sent Events or chunked response
}
```

**API endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/generate` | Generate audio (returns WAV) |
| POST | `/stream` | Streaming generation (SSE or chunked) |
| GET | `/` | Web interface (static HTML) |

### 3. Implement OpenAI Compatibility Layer

**Write tests first:**
```rust
#[tokio::test]
async fn test_openai_tts_endpoint() {
    let app = create_app();
    let body = json!({
        "model": "pocket-tts",
        "input": "Hello world",
        "voice": "default"
    });
    let response = app.oneshot(
        Request::post("/v1/audio/speech")
            .header("Content-Type", "application/json")
            .body(Body::from(body.to_string()))?
    ).await?;
    assert_eq!(response.status(), StatusCode::OK);
}
```

**OpenAI-compatible endpoint:**
- `POST /v1/audio/speech` - Compatible with OpenAI TTS API format
- Accept `model`, `input`, `voice`, `response_format` parameters

---

## Voice Cloning Test File

Use `ref.wav` in the project root for CLI and API tests with voice cloning.

---

## Verification Checklist

Before completing this phase:

- [ ] `cargo build --release` succeeds
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` reports no warnings
- [ ] Test coverage ≥80%
- [ ] `generate` command works correctly
- [ ] `serve` command starts server
- [ ] API endpoints respond correctly
- [ ] OpenAI compatibility verified

---

## Manual Verification Steps

1. **Test CLI generate:**
```bash
cargo run --release -- generate --text "Hello, world!" --output hello.wav
# Verify hello.wav contains correct audio
```

2. **Test CLI with voice cloning:**
```bash
cargo run --release -- generate --text "Clone test" --voice ref.wav --output cloned.wav
```

3. **Test API server:**
```bash
# Terminal 1
cargo run --release -- serve --port 8000

# Terminal 2
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from API"}' \
  --output api_output.wav
```

4. **Test OpenAI compatibility:**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "pocket-tts", "input": "OpenAI test", "voice": "default"}' \
  --output openai_test.wav
```

---

## Definition of Done

- [ ] CLI `generate` command working
- [ ] CLI `serve` command working
- [ ] All API endpoints responding correctly
- [ ] OpenAI compatibility layer working
- [ ] All tests pass
- [ ] 80% code coverage achieved
- [ ] Error messages are user-friendly
- [ ] Code documented with rustdoc comments
