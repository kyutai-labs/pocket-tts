# Docker Deployment Guide

This document describes the Dockerfile for `pocket-tts` (Rust/Candle) and how to use it for containerized deployment.

## Overview

The Dockerfile uses a multi-stage build approach to create a minimal production image with pre-downloaded models, **enabling completely offline operation** without requiring a HuggingFace token or internet access at runtime.

This means that the container *does not* support voice cloning, as the voice cloning model weights are gated and require authentication.

## Quick Start

### Build the Image

```bash
docker build -t pocket-tts .
```

This will take several minutes as it downloads the base image and model weights (~300MB) and builds the server.

### Run the Server

```bash
docker run -p 8000:8000 pocket-tts
```

Access the web UI at `http://localhost:8000`.

### Verify Offline Operation

To confirm the container runs completely offline without internet access:

```bash
# Generate audio completely offline
docker run --network none pocket-tts generate --text "Testing offline mode" --voice alba
```

## Build Stages

### Stage 1: Builder (rust:1.92-bullseye)

The builder stage:
- Installs build dependencies (cmake)
- Copies the entire workspace
- Builds the project in release mode using `cargo build --release`

### Stage 2: Runtime (debian:bullseye-slim)

The runtime stage creates a minimal image by:
- Installing only essential runtime dependencies:
  - `ca-certificates`: Required for HTTPS connections
  - `libssl1.1`: OpenSSL library for secure connections
- Copying the compiled binary (`pocket-tts-cli`) from the builder stage
- Copying the configuration files from `crates/pocket-tts/config`

## Model Caching Strategy

The Dockerfile implements a clever model pre-caching strategy to enable offline operation:

### Config Patching

The build process patches the configuration file to use the public (no-auth) model path:
```bash
sed -i 's|^weights_path: hf://kyutai/pocket-tts/|#weights_path: hf://kyutai/pocket-tts/|' /app/config/b6369a24.yaml
sed -i 's|^weights_path_without_voice_cloning:|weights_path:|' /app/config/b6369a24.yaml
```

This changes the default `weights_path` to use the non-gated model variant, eliminating the need for `HF_TOKEN` at runtime.

### Voice Embedding Pre-Cache

The build process generates audio for each available voice to pre-cache all voice embeddings:
```bash
for voice in alba marius javert jean fantine cosette eponine azelma; do
    pocket-tts generate --text "Initialize cache" --voice "$voice" && rm -f output.wav
done
```

This ensures:
- All model weights are downloaded during build time
- All voice embeddings are cached in the HuggingFace Hub cache
- The container can run completely offline without requiring internet access
- No authentication token is needed at runtime

## Detailed Usage

### Running CLI Commands

Thanks to the `ENTRYPOINT`, you can run CLI commands directly without repeating the binary name:

```bash
# Generate audio
docker run pocket-tts generate --text "Hello world" --voice alba

# Show help
docker run pocket-tts --help

# List available voices
docker run pocket-tts generate --help
```

To extract the generated audio file:
```bash
docker run -v $(pwd):/output:z pocket-tts generate --text "Hello world" --voice alba --output /output/hello.wav
```

## Available Voices

The following voices are pre-cached in the image:
- `alba`
- `marius`
- `javert`
- `jean`
- `fantine`
- `cosette`
- `eponine`
- `azelma`

## Image Size Considerations

The final image size is **339MB**, with the following layer breakdown:

- **Debian base system**: 80.7MB (debian:bullseye-slim)
- **Runtime dependencies**: 3.08MB (ca-certificates, libssl1.1)
- **Compiled binary**: 15.4MB (pocket-tts-cli release build)
- **Configuration files**: ~1.3KB (YAML config)
- **Cached models & voice embeddings**: 240MB (all weights and 8 voice embeddings)

The image is considerably smaller than usual machine learning images, coming in at under 340MB total despite including all model weights and voice embeddings for offline operation. There is of course
a lot of room for improvement (smaller base image, model quantization, etc), but this is a solid starting point for a fully offline TTS Docker image.
