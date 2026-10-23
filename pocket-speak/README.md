# pocket-speak

Reads text aloud in a Pocket TTS voice, from a terminal, on a desktop. The
desk counterpart to the Android app in `android/`: same model family, no
Python, no PyTorch, one binary.

```
pocket-speak install                       # fetch the pinned model and the alba voice
echo "The build is green. One. Two. Three." | pocket-speak
pocket-speak --out reply.wav < reply.md     # write a WAV instead of playing
```

It reads Markdown on stdin and drops what should not be spoken (fenced code,
tables, HTML, Markdown syntax), keeping link text and the words in inline
code. Bare URLs are read as their host.

| option | |
|---|---|
| `--out PATH` | write a 16-bit 24 kHz WAV instead of playing; `--play` does both |
| `--voice NAME\|PATH` | `alba` by default; `install --voice NAME` or `--all-voices` for the others |
| `--chunking MODE` | `packed` (default), `sentence`, or `packed-raw` for measuring |
| `--plain` | do not strip Markdown |
| `--seed N` | reproducible sampling |
| `--stats` | one JSON line per chunk on stderr: tokens, frames, how it ended, speed |
| `--model-dir DIR` | defaults to `$POCKET_TTS_MODEL_DIR`, else `%LOCALAPPDATA%\pocket-tts-rs\english_2026-04` or `~/.cache/pocket-tts-rs/english_2026-04` |

## Layout

- `crates/engine` drives the four ONNX graphs of the `english_2026-04` int8
  export directly on ONNX Runtime, and owns text preparation and chunking.
  Ported from speech-kit's Pocket adapter; see `crates/engine/NOTICE.md`.
- `crates/cli` is `pocket-speak`: stdin, playback through the default output
  device, WAV output, and `install`.
- `tools/tokdiff` checks the pure-Rust tokenizer against Google's.
- `tools/utterances` measures where generations lose words, with Whisper as
  the judge.

## What was measured, 2026-09-13

**Tokenizer.** `sentencepiece-rs` 0.2.2 against Google `sentencepiece` 0.2.2
on the pinned `tokenizer.model`: 0 mismatches in 14,812 corpus rows and 5,000
random mixed-script strings. See `tools/tokdiff/README.md`.

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
with 2 intra-op threads. First audio 280-450 ms. On a 168-word agent reply,
`sentence` and `packed` differed by 4 ms in first audio and 0.03x in speed.

**Windows.** Builds with MSVC in 30 s from a warm registry; the 22 MB
executable runs relocated with nothing beside it, plays through WASAPI, and
its WAV output was transcribed word for word.

**By ear.** The maintainer listened to the Windows smoke tests: at least as
good as speech-kit's implementation, as good as or better than the Android
app, and every trailing short sentence articulated.

Not established: other languages, aarch64, and behaviour on a device whose
default output format is not float, 16- or 32-bit integer.

## Build

```
cd pocket-speak
cargo build --release -p pocket-speak
```

`ort` downloads ONNX Runtime 1.24.2 from `cdn.pyke.io` at build time and
checks it against a pinned sha256. Point `ORT_LIB_LOCATION` at a Microsoft
release to avoid that host. On Linux, playback needs ALSA development headers.

## Models

Not shipped; `pocket-speak install` fetches them from pinned commits and
checks each sha256. The ONNX export is
[KevinAHM/pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx)
and the voices are
[kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning),
both CC-BY-4.0. Pocket TTS is by [Kyutai](https://kyutai.org).
