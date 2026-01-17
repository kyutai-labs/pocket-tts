# Contributing to Pocket TTS Candle

We welcome contributions! This project is now primarily a Rust implementation using the Candle framework.

## Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- (Optional) [uv](https://github.com/astral-sh/uv) for interacting with the Python reference code.

## Development Setup

The project uses a standard Cargo workspace:

```powershell
# Build the project
cargo build --release

# Run all tests
$env:HF_TOKEN="your_token_here"; cargo test --release --all-targets
```

## Repository Structure

- `crates/pocket-tts`: Core library.
- `crates/pocket-tts-cli`: CLI and Web Server.
- `crates/pocket-tts-bindings`: Python bindings.
- `assets/`: Reference assets for testing.
- `python-reference/`: Original Python implementation.

## Style Guidelines

- Run `cargo fmt` before submitting.
- Follow standard Rust naming conventions.
- Keep the streaming architecture in mind for any model changes.
- Performance is a priority; use benchmarking (`cargo bench`) to justify optimizations.

## Numerical Parity

Any core model changes MUST pass the parity tests in `crates/pocket-tts/tests/parity_tests.rs` to ensure they match the Python reference behavioral baseline.

## Coding Agents

If you are using an AI coding agent, please refer to [AGENTS.md](./AGENTS.md) for detailed implementation context and preferred patterns.
