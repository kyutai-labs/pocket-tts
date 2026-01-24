# Multi-stage build for pocket-tts with pre-downloaded models
FROM rust:1.92-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace files
COPY ./ ./

# Build the project in release mode
RUN cargo build --release

# =============================================================================
# Runtime with manual downloads
# =============================================================================
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/pocket-tts-cli /usr/local/bin/pocket-tts
COPY --from=builder /build/crates/pocket-tts/config /app/config

# Patch the config to use the public (no-auth) model path as the default
# This makes it use weights_path_without_voice_cloning instead of weights_path
RUN sed -i 's|^weights_path: hf://kyutai/pocket-tts/|#weights_path: hf://kyutai/pocket-tts/|' /app/config/b6369a24.yaml && \
    sed -i 's|^weights_path_without_voice_cloning:|weights_path:|' /app/config/b6369a24.yaml

WORKDIR /app

# Run the tool once during build to let hf_hub download and create proper cache structure
# This creates all the blob files, refs, and metadata that hf_hub needs for offline operation
# Generate with all available voices to cache all voice embeddings
RUN for voice in alba marius javert jean fantine cosette eponine azelma; do \
        echo "Caching voice: $voice"; \
        pocket-tts generate --text "Initialize cache" --voice "$voice" && rm -f output.wav; \
    done

EXPOSE 8000

ENTRYPOINT ["pocket-tts"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
