# pocket-speak

Reads text aloud in a Pocket TTS voice, from a terminal, on a desktop. The
desk counterpart to the Android app in `android/`: the same engine, no Python,
no PyTorch, one binary.

```
pocket-speak install                        # the default model and its voices
echo "The build is green. One. Two. Three." | pocket-speak
pocket-speak --out reply.wav < reply.md     # write a WAV instead of playing
pocket-speak install --model english_7m_2026-10 && pocket-speak --model english_7m_2026-10 < reply.md
```

It reads Markdown on stdin and drops what should not be spoken (fenced code,
HTML, Markdown syntax), keeping link text and the words in inline code. Bare
URLs are read as their host. A table is announced ("Table, 2 rows.") and read
row by row under its column names ("alpha: before, one; after, two."); one
wider than four columns is announced with its column names and not read. The
app does the same.

| option | |
|---|---|
| `--out PATH` | write a 16-bit 24 kHz WAV instead of playing; `--play` does both |
| `--model NAME` | `english_2026-10` by default; `pocket-speak models` lists the others |
| `--voice NAME\|PATH` | `alba` by default; `install` fetches all 27 unless given `--voice` |
| `--chunking MODE` | `packed` (default), `sentence`, or `packed-raw` for measuring |
| `--plain` | do not strip Markdown |
| `--seed N` | reproducible sampling |
| `--threads N` | threads for the model, 2 by default |
| `--float` | float weights instead of int8 (the 2m model is always float) |
| `--stats` | one JSON line per chunk on stderr: tokens, frames, how it ended, speed |
| `--model-dir DIR` | defaults to `$POCKET_TTS_MODEL_DIR`, else `%LOCALAPPDATA%\pocket-tts-rs\<model>` or `~/.cache/pocket-tts-rs/<model>` |

## Models

| model | download | speed on a 4-core x86 VM, 2 threads | first audio |
|---|---|---|---|
| `english_2026-10` (default) | 220 MB | 4.5x real time | 160 ms |
| `english_18m_2026-10` | 112 MB | 10.6x | 45 ms |
| `english_7m_2026-10` | 67 MB | 9.0x | 28 ms |
| `english_2m_2026-10` | 47 MB | 16.3x | 10 ms |

Kyutai's models, from
[kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning)
(CC-BY-4.0). `models.json` pins every file the engine reads, and
`pocket-speak install` refuses a file whose sha256 does not match. It is
written by `tools/pin-models.py`; the app downloads from the same list.
Pocket TTS is by [Kyutai](https://kyutai.org).

## Layout

- `crates/engine` runs the model on upstream's Rust implementation (xn-ptts),
  compiling in upstream's config, text preparation and weight loading from
  `../rust/src`, and owns the Markdown handling and chunking.
  `../android/docs/pure-rust-engine.md` explains how, and what was measured.
- `crates/cli` is `pocket-speak`: stdin, playback through the default output
  device, WAV output, `install` and `models`.
- `crates/android` is the engine over JNI for the app, and `pocket-bench`,
  which measures the engine without the app, on a phone over `adb` or on any
  Linux machine.
- `tools/pin-models.py` writes `models.json`; `tools/fetch-models.py` fetches
  from it without the CLI; `tools/bench/run.sh` measures every model.
- `tools/tokdiff` checks the pure-Rust tokenizer against Google's.
- `tools/utterances` measures where generations lose words, with Whisper as
  the judge.

## What was measured, 2026-09-13 (the previous, ONNX engine)

**Tokenizer.** `sentencepiece-rs` 0.2.2 against Google `sentencepiece` 0.2.2
on the pinned `tokenizer.model`: 0 mismatches in 14,812 corpus rows and 5,000
random mixed-script strings. See `tools/tokdiff/README.md`. The small models
use the same tokenizer file.

**Early endings.** The model decides for itself when it has finished speaking
and ends early on a generation that closes with very short sentences
(`android/docs/eleven-utterances.md`, 3.1). With faster-whisper base.en as
judge, on WSL x86_64:

| chunking | runs | final word kept | final word doubled |
|---|---|---|---|
| `sentence` | 27 (9 cases, seeds 1-3) | 27 | 0 |
| `packed` | 27 | 27 | 0 |
| `packed-raw` (no tail join) | 45 (seeds 1-5) | 44 | 0 |

The one loss in `packed-raw` dropped a whole countdown ("The build finished
and the log looks clean. One. Two. Three. Four. Five." was read as its first
sentence only), so the failure is the model's and not sherpa-onnx's, and the
tail join in `packed` still earns its place.

**Speed.** 3.2-4.2x real time on WSL x86_64 and 3.6-3.9x on Windows x86_64,
with 2 intra-op threads, for the ONNX engine and `english_2026-04`. First audio
280-450 ms.

**By ear.** The maintainer listened to the Windows smoke tests: at least as
good as speech-kit's implementation, as good as or better than the Android
app, and every trailing short sentence articulated.

Not established for the new engine: Windows and macOS builds (upstream's CLI
builds on both from the same crates), and behaviour on a device whose default
output format is not float, 16- or 32-bit integer.

## Build

```
cd pocket-speak
cargo build --release -p pocket-speak
```

Pure Rust: no CMake, no C++ compiler, nothing downloaded at build time. On
Linux, playback needs ALSA development headers. x86 builds target x86-64-v3
(AVX2, FMA: Intel Haswell and AMD Zen onwards), as upstream's do; see
`.cargo/config.toml`.
