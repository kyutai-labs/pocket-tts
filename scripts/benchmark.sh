#!/bin/bash

# Run full Python vs Rust benchmark suite
# Requirements: hyperfine, cargo, uv

# Check for hyperfine
if ! command -v hyperfine &> /dev/null; then
    echo "Error: hyperfine not found in PATH. Please install it: cargo install hyperfine"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found in PATH."
    exit 1
fi

TEXTS=(
    "Hello world"
    "This is a medium length sentence for benchmarking."
    "$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..3})"
)

# Ensure release build is up to date
echo "Building Release..."
cargo build --release -p pocket-tts-cli

# Use the built binary path
RS_BIN="target/release/pocket-tts-cli"

for i in "${!TEXTS[@]}"; do
    text="${TEXTS[$i]}"
    echo -e "\n=== Benchmark $((i + 1)): ${#text} chars ==="
    
    # We use uvx to run the python reference directly
    # Using --from ./python-reference to build local package
    PY_CMD="uvx --from ./python-reference pocket-tts generate --text \"$text\" --output-path bench_py.wav"
    RS_CMD="$RS_BIN generate --text \"$text\" --output bench_rs.wav"
    
    hyperfine --warmup 1 --runs 3 \
        "$PY_CMD" \
        "$RS_CMD"
done
