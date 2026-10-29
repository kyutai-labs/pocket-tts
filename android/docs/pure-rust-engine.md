# The pure-Rust engine (2026-10)

From 2026-10-03 to 2026-10-23, the app and `pocket-speak` ran Kevin AHM's ONNX
export of `english_2026-04` on ONNX Runtime (`engine-swap.md`). Since 2026-10-23
(commit fb538e3) they run upstream's own Rust implementation of the model,
[xn-ptts](https://github.com/gradium-ai/xn-ptts), through the same fork and commit
(`7efa269`) that upstream's Rust CLI uses. This file says why, what was measured,
and how the engine is kept in step with upstream.

## Why

- **The models.** Upstream's 2026-10 release added `english_2026-10` and three
  small English models (18.2M, 6.9M and 1.9M parameters, per upstream's README).
  They are Kyutai safetensors with new architecture options (low-rank text
  tables, one-step heads, narrower decoders) that no ONNX export covers. xn-ptts
  runs them all.
- **Speed.** On the same CPU and the same `english_2026-04` model, xn-ptts with
  int8 weights generated 3.1x faster than ONNX Runtime with the int8 export at
  two threads (6.25x against 1.99x). At one thread the gap is 2.1x (3.52x against
  1.65x).
- **Size.** The engine is 5 to 8 MB per ABI, against ONNX Runtime's 18 to 31 MB.
- **Upstream's fixes arrive by merge.** The engine compiles in upstream's own
  config, text preparation and weight loading (below), so a fix upstream makes
  there reaches the app on the next merge of upstream.

## What was measured

Upstream's Rust CLI, with the ONNX engine and then with xn-ptts, on the session's
container: 4-core Xeon at 2.1 GHz, x86-64-v3. The same 124-word agent reply
(`pocket-speak/tools/bench/reply.txt`), voice `alba`, seed fixed. Speed is seconds
of audio per second of generation. Word error rate is faster-whisper `base.en`
against the text.

| engine / model | weights | 1 thread | 2 threads | WER |
|---|---|---|---|---|
| ONNX Runtime, `english_2026-04` (the old engine) | int8 | 1.65x | 1.99x | 0.0% |
| xn-ptts, `english_2026-04` | int8 | 3.52x | 6.25x | 0.0% |
| xn-ptts, `english_2026-10` | int8 | 3.54x | 5.95x | 0.0% |
| xn-ptts, `english_18m_2026-10` | int8 | 8.8x | 12.9x | 0.8% |
| xn-ptts, `english_7m_2026-10` | int8 | 7.6x | 10.0x | 0.0% |
| xn-ptts, `english_2m_2026-10` | float | 17.7x | 29.6x | 2.4% |

The WER column covers the four 2026-10 models, each at its own row. The 2m model's
2.4% is three words of 124. The old engine's first audio was about 800 ms on that
machine. The new engine (`pocket-bench`, 2 threads, run while a Gradle build shared
the CPU):

| model | weights | load | first audio | speed |
|---|---|---|---|---|
| `english_2026-10` | int8 | 668 ms | 159 ms | 4.5x |
| `english_18m_2026-10` | int8 | 125 ms | 45 ms | 10.6x |
| `english_7m_2026-10` | int8 | 43 ms | 28 ms | 9.0x |
| `english_2m_2026-10` | float | 13 ms | 10 ms | 16.3x |

Two build settings mattered, and both are upstream's: x86 builds target
x86-64-v3 (AVX2), which roughly tripled x86 speed, and release builds use thin
LTO. The v3 setting applies to desktop x86 only, not Android's `x86_64` ABI, where
a device without AVX2 would die on the first instruction. Phones are arm64, where
NEON is always on.

On arm64 (GitHub's arm64 runner, `models-arm64` job). These figures come from the
job's run summary, which is not in the repository, so they are not re-checked here:

| model | weights | threads | load | first audio | speed |
|---|---|---|---|---|---|
| `english_2026-10` | int8 | 1 | 217 ms | 386 ms | 4.8x |
| `english_2026-10` | int8 | 2 | 219 ms | 204 ms | 6.6x |
| `english_2026-10` | int8 | 4 | 219 ms | 113 ms | 6.7x |
| `english_18m_2026-10` | int8 | 2 | 40 ms | 43 ms | 14.5x |
| `english_7m_2026-10` | int8 | 2 | 16 ms | 21 ms | 13.0x |
| `english_2m_2026-10` | float | 2 | 7 ms | 6 ms | 32.4x |

The job runs on pushes to `main` and on pull requests that touch the engine or the
app (the paths in `.github/workflows/android.yml`). It runs `tools/bench/run.sh`
with 1, 2 and 4 threads for each English model, and writes the table to its run
summary.

**The default model** is the best-quality one that reads at least 3x real time on
that runner with the app's 2 threads: `english_2026-10`, at 6.6x. The default
assumes a phone's big cores are close to that runner's. That is not measured. The
little cores on a phone are slower, which is what the smaller models are offered
for. The app's model screen shows the speed the last read was composed at on the
phone itself, and suggests a smaller model below 1.5x.

Not yet established: speed on a phone. `pocket-bench` runs on one over `adb` as
before (`engine-swap.md`).

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

Compiled in rather than copied: a merge of upstream updates them, and a change to
their interfaces fails the build rather than drifting. Upstream's own unit tests of
those modules run in the engine's test suite.

The generation loop is ported because what it needs is private upstream and fixed
at load: the noise generator, its temperature and its seed, which the app sets per
read. It keeps upstream's frame budget, EOS rule (ignored for the first 6 frames),
frames after EOS, 5 ms fade-in and decoding while generating. It differs in three
ways. The caller's thread decodes and calls the sink, and generation runs on a
scoped thread, because the app's sink calls into the JVM through a `JNIEnv`, which
is only valid on its own thread. Each chunk reports how it ended. And the text is
one chunk the caller has already split.

`.github/fork/ported` records the upstream version (git blob) each port follows.
`check-ported.sh` fails when the upstream file changes, and the fork checks run it
on pushes and pull requests. The weekly upstream watch runs the same check against upstream's
`main`, so a change is flagged before a merge. The same check fails if the
engine's xn-ptts pin differs from upstream's.

## Found by testing

Driving the JNI library from a JVM (the app's native signatures, on the host) and
`crates/engine/tests/with_models.rs` found two things worth knowing:

- **A cancel has to reach the decoder too.** The small models generate a chunk
  several times faster than it is decoded. A cancel 150 ms into a chunk of the 2m
  model arrived after generation had finished, and the decoder went on handing
  every remaining frame to the sink: a Stop that kept talking. The decoding loop
  checks the token itself now, and the chunk reports `Cancelled`.
- **A seed gives the same speech, not the same bits.** Decoding takes the frames
  queued when it gets to them, so how they are grouped into decoder calls depends
  on timing. That moves the rounding by one 16-bit step at most, with the same
  length and frames every time. The test's tolerance is 1e-3, which is about 33
  such steps, so the test allows more than the observed difference.

`with_models.rs` runs against every model `$POCKET_TTS_TEST_MODELS` holds: each
reads a sentence to its end, clamped, not silent, faded in; a seed repeats; a
cancel before the start yields nothing and a stopped sink yields `Cancelled`; a
voice for another model is refused. CI's `models-arm64` job runs it on arm64.

## Models and voices

`pocket-speak/tools/pin-models.py` writes `pocket-speak/models.json`: for each
offered model, its weights and tokenizer as its upstream config names them, and
the 27 precomputed voice states, each with its URL and sha256. The engine compiles
it in; Gradle packages the same file as an app asset; the desk's
`pocket-speak install` and CI's `tools/fetch-models.py` read it. The engine's tests
fail when upstream moves a model's files without the script being rerun.

The voice states come from the revision upstream's Python package uses
(`a7056f50`). Upstream's Rust CLI pins an older one (`a3055dcb`), which the author
reports has no voices for `english_2026-10` (not checked here). That would leave
the CLI unable to load a stock voice for its own default model. That is upstream's
to fix; this engine is unaffected.

Weights: int8 wherever the model's shapes allow it. The 2m models have a dimension
of 112, not a multiple of int8's 32-wide blocks, so they load as float. At that
size float is fast.

## What did not change

The Kotlin API (`NativeEngine`) kept its methods; `load` takes the model's name
where it took ONNX Runtime's path. Panics are still caught at the JNI boundary,
and the engine still keeps no per-chunk state, which is what that relies on. Voice
cloning is still unavailable: it needs Kyutai's gated weights.

The thread count is read once per process, when the first model loads: xn's pool
is built then, at that width. It is built with `start_pool_with`, not
`set_num_threads`, which sets an environment variable. That is not thread-safe in
a process other threads are already running in, as the app's always is.
