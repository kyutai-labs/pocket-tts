# The app runs pocket-speak's engine now (2026-10-03)

> Superseded in 2026-10 by upstream's pure-Rust engine: see
> `pure-rust-engine.md`. Kept as the record of the ONNX engine it describes.

sherpa-onnx is gone from the app. `PocketTts` drives the Rust engine the desk
CLI runs (`pocket-speak/crates/engine`), over JNI (`pocket-speak/crates/android`), on
Microsoft's `onnxruntime-android` 1.24.2. `owning-the-pipeline.md` argued for
one Rust core under every front end. This is that core under the Android app.

Each claim is tagged RAN, READ, TOLD or ASSUMED. The scripts and the full
results table are in `android/tools/engine-compare/` (`agg_final.md`).

## Why

The maintainer ran TalkBack with this app as the system engine on a Galaxy
S25. Their words: "many of the rapid-fire lines were sort of rushed thru and
slurred together in places", and "i've seen these same issues in calling 'Read
Aloud' during my own workflows" (both TOLD 2026-10-03). The
desk's pocket-speak was "nearly flawless" for them (TOLD).

The same text was run through each engine and scored with faster-whisper
small.en: five seeds, voice alba, temperature 0.3 (RAN). The TalkBack corpus
is the announcements the Warp fork's Android app sends TalkBack, rebuilt from
its logs. They are short lines with newlines, no closing period and a lowercase
start. The Read Aloud corpus is four paragraphs of this repo's README.

| engine | TalkBack WER | Read Aloud WER | Read Aloud words/s | median pause at a sentence end |
|---|---|---|---|---|
| sherpa-onnx 1.13.6, as the app called it | 53.0% | 3.4% | 4.38 | 0.17 s |
| the same, with the reference's text preparation | 23.8% | 3.4% | 4.38 | 0.17 s |
| reference pocket-tts 3.3.0 (PyTorch) | 10.2% | 2.1% | 3.60 | 0.39 s |
| pocket-speak, desk (x86, Linux) | 3.5% | 1.5% | 3.55 | 0.34 s |
| pocket-speak's engine on the S25 (`pocket-bench`) | 6.7% | 1.2% | 3.59 | 0.35 s |

Most of the TalkBack WER left in the last two rows is Whisper joining "slow
line" or "fast line" into one word ("Slowline 1", "fastline 18"); one
"pears" came back as "pairs". The one real mishearing in the ten pocket-speak
runs was "Flowline 1", from seed 5 on both machines (RAN, `judge.json`).

## What the measurement found about sherpa-onnx (RAN unless tagged)

- **The app handed it raw text.** sherpa-onnx turns a newline into byte token
  `<0x0A>` with no word boundary after it, so lines ran together. The
  reference capitalises the first letter, adds a period and turns newlines into
  spaces (`prepare_text_prompt`). pocket-speak's `Engine::prepare_text` does
  the same, so the swap fixes this inside the engine.
- **The speed setting never did anything.** `offline-tts-pocket-impl.h` at
  v1.13.6 does not read `speed` (READ). Audio at 1.0 and 2.0 was
  byte-identical. Neither the app's slider nor Android's system rate ever took
  effect.
- **PR #5 made no measurable difference.** It generates a whole chunk in one
  pass instead of letting sherpa split it. The TalkBack lines carry no `.!?`,
  so both paths produced identical audio. On prose the two are within noise.
  Its comment says the reference "runs one autoregressive pass over the whole
  text". The reference splits at 50 tokens and again at commas (READ, ef69ab8,
  #143).
- **The TalkBack path skipped the short-tail join.** It passed `chunk.text`
  rather than `chunk.speech` (READ). It passes `chunk.speech` now. Highlight
  offsets still come from the text.
- **sherpa-onnx reads about 22% faster than the reference**, with half the
  pause at sentence ends. Its fp32 model did the same, so int8 is not the
  cause. Voice conditioning is the likely cause (ASSUMED).

## What changed in the app

- **Engine.** `NativeEngine` is the JNI bridge. `PocketTts` keeps its API,
  so `Reader`, the scratchpad and `PocketTtsService` call it as before.
  Synthesis streams: the engine hands over audio every 12 frames, about 0.96 s.
  Each system-engine request makes its own cancel token before it waits for
  the engine, and `onStop` cancels that token. Generation stops at the next
  frame, or at the first if the stop came during the wait, and an in-app read
  sharing the engine is not stopped by it. Samples are clamped to [-1, 1]
  before Sonic and before the caller, because the decoder can overshoot and
  Sonic narrows to 16-bit without a clamp. A panic in the Rust code becomes
  an `IllegalStateException` rather than killing the process.
- **Model.** pocket-speak's bundle: five files plus `bundle.json`, 125 MB,
  pinned to a Hugging Face commit and checked by sha256 before it is kept. These
  are the pins in `pocket-speak/crates/cli/src/install.rs`. A voice is a precomputed
  embedding, about 6 MB, fetched on first use and checked the same way. All 21
  stock voices have one. One download per voice runs at a time. A file
  already on disk is checked against its sha256 once per process before it is
  trusted. A `.part` that already holds the whole file, because the process
  died before the rename, is handed to the hash check rather than failing with
  HTTP 416 on every retry. Installing the model also fetches the selected
  voice's embedding, so the first read does not need the network. The reference wav is still fetched for the picker's
  preview. The old sherpa model directory, about 200 MB, is deleted after the
  new engine first loads, along with any half-finished download of it.
- **Speed works.** Sonic (`app/src/main/java/sonic/Sonic.java`, Bill Cox,
  Apache 2.0, vendored as published) changes tempo after generation and keeps
  pitch. The app's slider runs from 0.5x to 2.0x. Android's system rate is
  clamped to 0.5x–3.0x, because its own slider runs from 10% to 600% (READ,
  AOSP; Samsung's not checked).
- **Temperature** reaches the engine (`Engine::set_temperature`; it used to
  be a constant). The steady-voice switch restarts sampling from the fixed seed
  on every chunk.
- **Hidden, not deleted:** the decode-steps slider, since this engine
  decodes in one step; install-from-archive, since it took sherpa's tar.bz2 and
  now refuses with that reason; and imported voices with the import button.
- **Build.** `:app:preBuild` runs `cargo ndk` for arm64-v8a, armeabi-v7a and
  x86_64 into `app/src/main/jniLibs` (ignored by git). `-PskipNative` packages
  what is already there. Prerequisites: rustup's three Android targets,
  `cargo-ndk`, and the NDK (28.2 here). The debug APK is 85 MB. Most of that is
  three copies of ONNX Runtime, at 18, 26 and 31 MB. The release build,
  minified with R8, is 80 MB.

## On the S25 (RAN, 2026-10-03)

- **`pocket-bench`** is in `pocket-speak/crates/android/src/bin` and is run from
  `adb shell` with two threads:
  - model load 0.43–0.67 s across five runs;
  - first audio 0.21–0.32 s per chunk;
  - generation 3.7–5.0x faster than real time;
  - all 70 chunks (14 per seed, five seeds) ended on the EOS head, none on the
    frame budget.
- **In the app** (`android/tools/engine-compare/in-app-2026-10-03.txt` and
  `timings-release.png`):
  - The debug build, on a broadcast read from a cold process: model load
    0.62 s, 3.46x real time.
  - The release build, after the review fixes: model load 0.63 s, 3.52x.
    That shows R8 kept the JNI callback, which the sherpa path lost twice.
  - Both Timings screens say 0 underruns. Logcat has a framework line in both
    reads restarting the track "due to previous underrun". Why the two
    disagree is not established.
- **Heard by the maintainer:** TalkBack, and a Markdown document from their
  notes app read aloud. "much better than before. Sounded much closer to the
  desktop experience which is exactly what i was hoping for" (TOLD).
- **Installed as a seeded update.** The APK went over the earlier build, and
  the model files were copied in over USB rather than downloaded. They are the
  files the desk installer had already verified. The download path
  (`ModelManager.ensureModel`, `fetchVerified`, `download`) has not run
  anywhere: the unit tests replace the installer with a fake
  (`ModelInstallTest`), and no test reaches the HTTP code.

## Not established, and later steps

- **Voice cloning.** The bundle has no Mimi encoder, so a wav cannot become an
  embedding on the phone. Imported wavs are kept on disk and hidden. Bringing
  them back means adding the encoder graph to the engine, or making the
  embedding on a desk and copying it over.
- **The model download on a real phone**, including resume after a dropped
  connection and the 416 case. Only the seeded install ran.
- **Upgrading switches the system engine off until the model is
  downloaded.** The new model directory is not there after an update, so a
  system-engine request answers `ERROR_NOT_INSTALLED_YET` until the user
  presses Download in the app. Whether TalkBack falls back to another engine
  meanwhile was not checked. The maintainer's phone was seeded, so it never
  saw this.
- **`pocket-bench` aborted on exit** (`FORTIFY: pthread_mutex_lock called on a
  destroyed mutex`) after writing all its output. This is likely ONNX Runtime's
  global environment being torn down after the library state it depends on
  (ASSUMED). The app process is killed rather than exited and never frees
  its engine (`PocketTts.release()` was removed, and `NativeEngine.close()`
  has no caller), so the app should not reach it (READ). It is not
  established.
- **armeabi-v7a and x86_64** were built and packaged, but not run.
- **Whisper is a noisy judge on garbled audio.** Identical TalkBack audio
  (same md5 in four cells) scored 54.0%, 60.3%, 55.6% and 55.6%, the spread
  coming from one item Whisper hallucinated on. On clean audio the spread is
  small.
- **The comparison confounds model export with engine.** sherpa-onnx used its
  own 2026-01-26 int8 export, and pocket-speak uses Kevin AHM's export of
  `english_2026-04`. The two rows compare the two things the app could ship.
  They do not isolate the code.
