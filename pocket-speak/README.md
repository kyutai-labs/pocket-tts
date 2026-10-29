# pocket-speak

Reads text aloud in a Pocket TTS voice, from a terminal on a desktop. It is the
desk counterpart to the Android app in `android/`: the same engine, with no
Python, no PyTorch, and one binary.

```
pocket-speak install                        # the default model and its voices
echo "The build is green. One. Two. Three." | pocket-speak
pocket-speak --out reply.wav < reply.md     # write a WAV instead of playing
pocket-speak install --model english_7m_2026-10 && pocket-speak --model english_7m_2026-10 < reply.md
pocket-speak install --model french --voice estelle && echo "Le build est vert." | pocket-speak --model french
```

It reads Markdown on stdin and drops what should not be spoken: fenced code,
HTML, and Markdown syntax. It keeps link text and the words in inline code, and
reads a bare URL as its host. A table is announced ("Table, 2 rows.") and read
row by row under its column names ("alpha: before, one; after, two."). A table
wider than four columns is announced with its column names but not read. The
announcement is in the model's language ("Tableau, 2 lignes."), and the app
does the same.

| option | |
|---|---|
| `--out PATH` | write a 16-bit, 24 kHz, mono WAV instead of playing; `--play` does both |
| `--model NAME` | `english_2026-10` by default. The model sets the language; `pocket-speak models` lists them |
| `--voice NAME\|PATH` | the model's default voice: `alba` for English, Kyutai's pick for each other language. `install` fetches all 27 voices of a model unless `--voice` is given (repeat it for several) |
| `--chunking MODE` | `packed` (default), `sentence` (one generation per sentence), or `packed-raw` (no tail join, for measuring) |
| `--plain` | do not strip Markdown |
| `--seed N` | fixed sampling seed, for repeatable audio |
| `--threads N` | threads for the model; 2 by default |
| `--float` | float weights instead of int8. The 2m models are always float |
| `--stats` | on stderr: one JSON line naming the model and its weights, then one line per chunk with token and frame counts, how generation ended, and speed |
| `--model-dir DIR` | defaults to `$POCKET_TTS_MODEL_DIR`. Otherwise `%LOCALAPPDATA%\pocket-tts-rs\<model>` on Windows, or `$XDG_CACHE_HOME/pocket-tts-rs/<model>` (or `~/.cache/pocket-tts-rs/<model>` without `XDG_CACHE_HOME`) elsewhere |

## Models

| model | download | speed, 2 threads (historical) | first audio (historical) |
|---|---|---|---|
| `english_2026-10` (default) | 220 MB | 4.5x real time | 159 ms |
| `english_18m_2026-10` | 112 MB | 10.6x | 45 ms |
| `english_7m_2026-10` | 67 MB | 9.0x | 28 ms |
| `english_2m_2026-10` | 47 MB | 16.3x | 10 ms |
| `french`, `german`, `spanish`, `italian`, `portuguese`, `dutch` | 220 MB each | as `english_2026-10`, whose shape they share | |

The speed and first-audio figures are from `android/docs/pure-rust-engine.md`
(committed 2026-10-23): a 4-core Xeon container at 2.1 GHz, `pocket-bench`
with 2 threads, sharing the CPU with a Gradle build. The engine has changed
since, with commits on 2026-10-24, so re-measure before quoting them.

Each language's default voice is Kyutai's pick for it: Estelle (French), Jürgen
(German), Lola (Spanish), Giovanni (Italian), Rafael (Portuguese), and Daan
(Dutch). Every model has the same 27 voices, so any voice can be used with any
model. The listening check (`tools/listen/check.py`) covers each model's default
voice only. Each model uses the temperature from its upstream config, which is
0.3 for all of them today.

Kyutai's model files come from
[kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning)
(CC-BY-4.0). `models.json` pins every file the engine reads. `pocket-speak install`
downloads each file to a `.part` name, and stops with an error if its sha256
does not match, leaving the existing file alone. A file already present with the
pinned hash is kept. `tools/pin-models.py` writes `models.json`, and the
app downloads from the same list. Pocket TTS is by [Kyutai](https://kyutai.org).

## Layout

- `crates/engine` runs the model on upstream's Rust implementation (xn-ptts).
  It compiles in upstream's config, text preparation, and weight loading from
  `../rust/src`, and owns the Markdown handling and chunking.
  `../android/docs/pure-rust-engine.md` explains why, and what was measured.
- `crates/cli` is `pocket-speak`: stdin, playback through the default output
  device, WAV output, `install`, and `models`.
- `crates/android` is the engine over JNI for the app. Its `pocket-bench` binary
  measures the engine without the app, on a phone over `adb` or on a Linux
  machine.
- `tools/pin-models.py` writes `models.json`. `tools/fetch-models.py` fetches
  from it without the CLI. `tools/bench/run.sh` measures the English models.
- `tools/listen/check.py` has every model read its language's sample sentence
  in its default voice, and Whisper transcribes it. It fails a model whose word
  error rate is over its ceiling in `tools/listen/ceilings.json`. CI runs it on
  arm64 for changes under `android/`, `pocket-speak/`, `rust/src`, or
  `pocket_tts/config`, and writes what Whisper heard to the run's summary. The
  ceilings were measured on x86-64 on 2026-10-24; the file says they have not
  been checked on arm64 yet.
- `tools/tokdiff` checks the pure-Rust tokenizer against Google's.
- `tools/utterances` measures where generations lose words, with Whisper as the
  judge.

## What was measured, 2026-09-13 (the previous, ONNX engine)

**Tokenizer.** `sentencepiece-rs` 0.2.2 gave the same ids as Google's
`sentencepiece` 0.2.2 on the pinned `tokenizer.model`: 0 mismatches in 14,812
corpus rows and 5,000 random mixed-script strings. See
`tools/tokdiff/README.md`. The 18m, 7m, and 2m models read that same file. The
default `english_2026-10` reads a `tokenizer.json` through the Hugging Face
`tokenizers` crate instead, and `tools/tokdiff` does not check that path.

**Early endings.** The model decides for itself when it has finished speaking,
and it ends early on a generation that closes with very short sentences
(`android/docs/eleven-utterances.md`, 3.1). With faster-whisper base.en as the
judge, on WSL x86_64:

| chunking | runs | final word kept | final word doubled |
|---|---|---|---|
| `sentence` | 27 (9 cases, seeds 1-3) | 27 | 0 |
| `packed` | 27 | 27 | 0 |
| `packed-raw` (no tail join) | 45 (seeds 1-5) | 44 | 0 |

The one loss in `packed-raw` dropped a whole countdown ("The build finished and
the log looks clean. One. Two. Three. Four. Five." was read as its first
sentence only). The failure is the model's, not sherpa-onnx's, and the tail join
in `packed` still earns its place.

**Speed.** 3.2-4.2x real time on WSL x86_64 and 3.6-3.9x on Windows x86_64, with
2 intra-op threads, for the ONNX engine and `english_2026-04`. First audio
280-450 ms.

**By ear.** The maintainer listened to the Windows smoke tests. The result was
at least as good as speech-kit's implementation, as good as or better than the
Android app, and every trailing short sentence was articulated.

Not yet established for the new engine: Windows and macOS builds of
`pocket-speak` (upstream's `rust-release.yml` builds upstream's own CLI for both),
and playback on a device whose default sample format is not float, 16- or 32-bit
integer.

## Build

```
cd pocket-speak
cargo build --release -p pocket-speak
```

No CMake and no C++ compiler are needed, and no model files are downloaded at
build time. The HTTP client (`ureq` with rustls) builds `ring`, which compiles
some C and assembly, so a C compiler is still required. On Linux, playback needs
the ALSA development headers (`libasound2-dev` on Debian and Ubuntu, which CI
installs). x86 builds target x86-64-v3 (AVX2, FMA: Intel Haswell and AMD Zen
onwards), as upstream's do; see `.cargo/config.toml`.
