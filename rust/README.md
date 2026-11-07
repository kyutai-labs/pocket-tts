# pocket-tts, the library and the standalone binary

A port of the `pocket-tts` CLI (`generate`, `serve`, `export-voice`) that runs without Python,
built on the [xn-ptts](https://github.com/gradium-ai/xn-ptts) implementation of the model. The
release workflow (`.github/workflows/rust-release.yml`) attaches its builds to every GitHub
release.

The cargo package is named **`kyutai-pocket-tts`**; the library inside it is **`pocket_tts`**,
which is what you write in `use`. The two differ because `pocket-tts` on crates.io belongs to an
unrelated third-party crate, and crates.io normalises `-` and `_` — so `pocket_tts` resolves to
that same crate — while offering no namespaces to disambiguate with.

```bash
cargo build --release
./target/release/pocket-tts generate --text "Hello world."
```

x86 builds target x86-64-v3 (AVX2), see `.cargo/config.toml`.

## Using it as a library

The crate is both a `[[bin]]` and a `[lib]`. Turn the default features off and it pulls neither
the CLI, the web server, nor a C++ toolchain:

```toml
[dependencies]
kyutai-pocket-tts = { version = "0.1", default-features = false }

# Or, to keep `pocket-tts` as the key in your manifest:
pocket-tts = { package = "kyutai-pocket-tts", version = "0.1", default-features = false }

# Or from a git checkout, which works the same way:
kyutai-pocket-tts = { git = "https://github.com/kyutai-labs/pocket-tts", default-features = false }
```

The crate is self-contained — configs and the web page are embedded, nothing is read from the
Python package at build time — so a git or path dependency needs only a Rust toolchain.

[USAGE.md](USAGE.md) is the integration guide — voices, the streaming callback, the Hub cache, the
backends and the caveats. `examples/synthesize.rs` is a complete text-in, WAV-out consumer; run it
with `cargo run --release --example synthesize -- "Hello from a library." out.wav`. The shape is:

```rust
let (config, origin) = Config::load(Some("english"), None)?;
let model: TtsModel<pocket_tts::F32> = TtsModel::load(config, origin, &LoadOptions::default())?;
let voice = model.get_state_for_audio_prompt("marius", false)?;
model.generate_audio_stream(
    &voice, "Hello.", defaults::MAX_TOKEN_PER_CHUNK, None,
    &AtomicBool::new(false),
    &mut |pcm: &[f32]| { /* play or write pcm */ Ok(()) },
)?;
```

### Features

| Feature | What it adds | Default |
| --- | --- | --- |
| `cli` | the `generate`/`serve`/`export-voice` binary, and with it `axum`, `clap`, `tokio`, `tower-http` | on |
| `sentencepiece` | the SentencePiece tokenizer, **which needs CMake and a C++ compiler**. The released models ship a Hugging Face `tokenizer.json` and do not need it; most community models do | off |
| `accelerate` | Apple's Accelerate framework for the f32 matmuls, for macOS | off |

A config that asks for a SentencePiece tokenizer without the feature fails with a message naming
it, rather than with a link error.

### What a consumer should know

- **`-C target-cpu=x86-64-v3` is not inherited.** `.cargo/config.toml` applies only to the
  directory cargo runs in, so a dependent crate gets baseline SIMD and slower matmuls unless it
  sets the flag itself, in its own `.cargo/config.toml` or `RUSTFLAGS`.
- **`[profile.release]` here is ignored.** `codegen-units = 1` and `lto = "thin"` apply only when
  this crate is the root; set them in the consuming crate to get the same code.
- **`xn` is in the public API.** `TtsModel<Q>` is generic over the backend, and `pocket_tts::F32`
  and `pocket_tts::Int8` are aliases for `xn` types, so an `xn` major version bump is a breaking
  change for dependents too. The `xn` version must match the one `ptts` uses — its tensors cross
  the crate boundary.
- **Errors are `anyhow::Error`.** Good for reporting, not for matching on a cause.

### Voice licences

The 27 predefined voices do **not** share one licence, and two of them forbid commercial use. The
crate does not redistribute them — embeddings are downloaded from the Hub on first use — so the
obligation falls on whoever ships the audio. `USAGE.md` has the table; in code:

```rust
for voice in pocket_tts::voices::commercially_usable() { /* … */ }
println!("{}", pocket_tts::voices::attribution_notice(&["alba"])?);
```

`CC0-1.0` for 7 voices, `CC-BY-4.0` for 13, **`CC-BY-NC-4.0` for `cosette` and `jean`**, and
**undetermined** for `giovanni`, `lola`, `juergen`, `rafael` and `daan`, whose sources sit in a
gated repository under no licence statement we could find. `commercial_use()` reports three states
rather than a bool, because "undetermined" is not permission.

The catalogue is data, not code — one JSONL per licence in `assets/voices/` — so a voice can be
added with `voices::load_dir` and no rebuild.

### The small English models need the `sentencepiece` feature

All 27 embedded configs translate and are supported, the four with a low-rank text table included
(`english_2m_2026-10`, `english_2m_refl_2026-10`, `english_7m_2026-10`,
`english_7m_refl_2026-10`). The rank reaches xn-ptts as its `lut_dim` and the projection is renamed
to the tensor it keys the factorisation off — the comments on `Config::to_ptts` in `src/config.rs`
and `remap_key` in `src/weights.rs` explain why that is all it takes.

Two caveats on those four:

- They ask for a **SentencePiece tokenizer**, so they need the `sentencepiece` feature, and with it
  CMake and a C++ compiler. The other configs — `english`, `english_2026-10`, `english_18m`, every
  other language — use pure-Rust `tokenizers`.
- **They have not yet been heard.** The config translation and the weight remapping are unit
  tested, but no low-rank checkpoint has been loaded and generated from on a machine with CMake, so
  treat them as untested rather than known-good.

### Embedded assets

`assets/config/` and `assets/index.html` are copies of `pocket_tts/config/` and
`pocket_tts/static/index.html`, embedded by `build.rs` so the crate is self-contained — a published
crate cannot read `../pocket_tts/`. `assets/python-version` records which release they came from.

The Python copies stay canonical. `cargo test` fails if the two drift, and
`scripts/sync_rust_assets.sh` is the fix; the check skips itself in a published crate, where there
is no `../pocket_tts/` to compare against.

The web page is reachable as `server::INDEX_HTML`, under the `cli` feature, rather than beside the
configs: its only consumer is the `serve` handler, so a library build does not carry it.
`assets/voices/*.jsonl` is this crate's own data — the voice catalogue — and not a copy of
anything.

## Differences with the Python CLI

- Runs on the CPU only (`--device cpu`), and `--quantize` uses xn's int8 weights.

- `serve --reload` is accepted and ignored. The server answers `/`, `/health` and `/tts` as the
  Python one does, without FastAPI's `/docs`.
- The noise is drawn from a different generator, so the audio differs sample by sample from the
  Python package's, as two runs with different seeds would. The seed is fixed, so a command
  always gives the same audio; `POCKET_TTS_SEED` sets another one.
- The tensor operations use 4 threads by default; set `RAYON_NUM_THREADS` to change it.
