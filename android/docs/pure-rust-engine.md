# The pure-Rust engine (2026-10)

The app and `pocket-speak` used to run an ONNX export of `english_2026-04` on
ONNX Runtime (`engine-swap.md`). Since 2026-10 they run upstream's own Rust
implementation of the model, [xn-ptts](https://github.com/gradium-ai/xn-ptts),
through the same fork and commit upstream's Rust CLI uses. This file says why,
what was measured, and how the engine is kept in step with upstream.

## Why

- **The models.** Upstream's 2026-10 release added `english_2026-10` and three
  small English models (18.2M, 6.9M and 1.9M parameters). They are Kyutai
  safetensors with new architecture options (low-rank text tables, one-step
  heads, narrower decoders) that no ONNX export covers. xn-ptts runs them all.
- **Speed.** On the same CPU and the same `english_2026-04` model, xn-ptts with
  int8 weights generated 3.1x faster than ONNX Runtime with the int8 export.
- **Size.** The engine is about 8 MB per ABI, against ONNX Runtime's 18 to 31 MB.
- **Upstream's fixes arrive by merge.** The engine compiles in upstream's own
  config, text preparation and weight loading (below), so a fix upstream makes
  there reaches the app on the next merge of upstream.

## What was measured

On the session's container: 4-core Xeon at 2.1 GHz, x86-64-v3. The same
124-word agent reply (`pocket-speak/tools/bench/reply.txt`), voice `alba`, seed
fixed. Speed is seconds of audio per second of generation; word error rate is
faster-whisper `base.en` against the text.

Before the swap, upstream's Rust CLI against the ONNX engine:

| engine / model | weights | 1 thread | 2 threads | WER |
|---|---|---|---|---|
| ONNX Runtime, `english_2026-04` (the old engine) | int8 | 1.65x | 1.99x | 0.0% |
| xn-ptts, `english_2026-04` | int8 | 3.52x | 6.25x | 0.0% |
| xn-ptts, `english_2026-10` | int8 | 3.54x | 5.95x | 0.0% |
| xn-ptts, `english_18m_2026-10` | int8 | 8.8x | 12.9x | 0.8% |
| xn-ptts, `english_7m_2026-10` | int8 | 7.6x | 10.0x | 0.0% |
| xn-ptts, `english_2m_2026-10` | float | 17.7x | 29.6x | 2.4% |

The old engine's first audio was about 800 ms on that machine. The new engine
(`pocket-bench`, 2 threads, run while a Gradle build shared the CPU):

| model | weights | load | first audio | speed |
|---|---|---|---|---|
| `english_2026-10` | int8 | 668 ms | 159 ms | 4.5x |
| `english_18m_2026-10` | int8 | 125 ms | 45 ms | 10.6x |
| `english_7m_2026-10` | int8 | 43 ms | 28 ms | 9.0x |
| `english_2m_2026-10` | float | 13 ms | 10 ms | 16.3x |

Whisper word error on the new engine's outputs was 0.8% (one word in 124) for
each of the four models. Wall-clock time for the whole reply matched upstream's
CLI to within run-to-run noise.

Two build settings mattered, and both are upstream's: x86 builds target
x86-64-v3 (AVX2), which tripled x86 speed, and release builds use thin LTO. The
v3 setting is for desktop x86 only, not Android's `x86_64` ABI, where a device
without AVX2 would die on the first instruction. Phones are arm64, where NEON is
always on.

On arm64 (GitHub's arm64 runner, `models-arm64` job, 2026-10-23):

| model | weights | threads | load | first audio | speed |
|---|---|---|---|---|---|
| `english_2026-10` | int8 | 1 | 217 ms | 386 ms | 4.8x |
| `english_2026-10` | int8 | 2 | 219 ms | 204 ms | 6.6x |
| `english_2026-10` | int8 | 4 | 219 ms | 113 ms | 6.7x |
| `english_18m_2026-10` | int8 | 2 | 40 ms | 43 ms | 14.5x |
| `english_7m_2026-10` | int8 | 2 | 16 ms | 21 ms | 13.0x |
| `english_2m_2026-10` | float | 2 | 7 ms | 6 ms | 32.4x |

The job reruns on every change to the engine, with 1, 2 and 4 threads for
each model, in its run summary.

**The default model** is the best-quality one that reads at least 3x real
time on that runner with the app's 2 threads: `english_2026-10`, at 6.6x.
A phone's big cores are in the same class as that runner's; its little ones
are several times slower, which is what the smaller models are offered for.
The app's model screen shows the speed the last read was composed at on the
phone itself, and suggests a smaller model below 1.5x.

Not yet established: speed on a phone. `pocket-bench` runs on one over `adb`
as before (`engine-swap.md`).

## How it is built

`pocket-speak/crates/engine` (see its `src/lib.rs`):

| module | what | from |
|---|---|---|
| `config` | model configs, embedded from `pocket_tts/config` by `build.rs` | upstream's `rust/src/config.rs`, compiled in by path |
| `defaults` | | upstream's `rust/src/defaults.rs`, compiled in |
| `prompt` | preparing a chunk's text for the model | upstream's `rust/src/text.rs`, compiled in |
| `weights` | Kyutai safetensors into xn-ptts | upstream's `rust/src/weights.rs`, compiled in |
| `tokenizer` | pure-Rust SentencePiece and `tokenizers` | the fork's, with the interface of upstream's |
| `download` | resolves a config's `hf://` paths to files on disk | the fork's: the engine never touches the network |
| `synth` | the generation loop | ported from upstream's `rust/src/model.rs` and `rng.rs` |
| `models` | `pocket-speak/models.json`: the offered models' files, pinned | the fork's |

Compiled in rather than copied: a merge of upstream updates them, and a change
to their interfaces fails the build rather than drifting. Upstream's own unit
tests of those modules run in the engine's test suite.

The generation loop is ported because what it needs is private upstream and
fixed at load: the noise generator, its temperature and its seed, which the app
sets per read. It keeps upstream's frame budget, EOS rule (ignored for the first
6 frames), frames after EOS, 5 ms fade-in and decoding while generating. It
differs in three ways. The caller's thread decodes and calls the sink, and
generation runs on a scoped thread, because the app's sink calls into the JVM
through a `JNIEnv`, which is only valid on its own thread. Each chunk reports
how it ended. And the text is one chunk the caller has already split.

`.github/fork/ported` records the upstream version each port follows. Fork
checks fail on the PR that merges a change to it, and the weekly upstream watch
says so beforehand. The same check fails if the engine's xn-ptts pin differs
from upstream's.

## Found by testing

Driving the JNI library from a JVM (the app's native signatures, on the host)
and `crates/engine/tests/with_models.rs` found two things worth knowing:

- **A cancel has to reach the decoder too.** The small models generate a chunk
  several times faster than it is decoded. A cancel 150 ms into a chunk of the
  2m model arrived after generation had finished, and the decoder went on
  handing every remaining frame to the sink: a Stop that kept talking. The
  decoding loop checks the token itself now, and the chunk reports `Cancelled`.
- **A seed gives the same speech, not the same bits.** Decoding takes the frames
  queued when it gets to them, so how they are grouped into decoder calls
  depends on timing, and that moves the rounding: one 16-bit step at most, with
  the same length and frames every time. Inaudible, and the test allows it.

`with_models.rs` runs against every model `$POCKET_TTS_TEST_MODELS` holds: each
reads a sentence to its end, clamped, not silent, faded in; a seed repeats; a
cancel before the start yields nothing and a stopped sink yields `Cancelled`;
a voice for another model is refused. CI's `models-arm64` job runs it on arm64.

## Models and voices

`pocket-speak/tools/pin-models.py` writes `pocket-speak/models.json`: for each
offered model, its weights and tokenizer as its upstream config names them, and
the 27 precomputed voice states, each with its URL and sha256. The engine
compiles it in; Gradle packages the same file as an app asset; the desk's
`pocket-speak install` and CI's `tools/fetch-models.py` read it. The engine's
tests fail when upstream moves a model's files without the script being rerun.

The voice states come from the revision upstream's Python package uses
(`a7056f50`). Upstream's Rust CLI pins an older one (`a3055dcb`), which has none
for `english_2026-10`, so it cannot load a stock voice for its own default
model. That is upstream's to fix; this engine is unaffected.

Weights: int8 wherever the model's shapes allow it. The 2m models have a
dimension of 112, not a multiple of int8's 32-wide blocks, so they load as
float; at that size float is fast.

## What did not change

The Kotlin API (`NativeEngine`) kept its methods; `load` takes the model's name
where it took ONNX Runtime's path. Panics are still caught at the JNI boundary,
and the engine still keeps no per-chunk state, which is what that relies on.
Voice cloning is still unavailable: it needs Kyutai's gated weights.

The thread count is read once per process, when the first model loads: xn's
pool is built then, at that width. It is built with `start_pool_with`, not
`set_num_threads`, which sets an environment variable: not thread-safe in a
process other threads are already running in, as the app's always is.
