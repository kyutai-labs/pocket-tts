# Owning the pipeline, not the inference

> Written before the Rust core existed. The core shipped on 2026-10-03
> (`engine-swap.md`) and moved to upstream's pure-Rust engine on 2026-10-23
> (`pure-rust-engine.md`). Several steps in "How to decide" are now done: the
> model name is no longer hardcoded (`ModelCatalog` reads `pocket-speak/models.json`,
> fb538e3), the tokenizer diff exists as `pocket-speak/tools/tokdiff`, and the
> Termux path was run end to end (`eleven-utterances.md`). The rest of the page is
> the reasoning as it stood, left as written.

A position, not a plan of record. Nothing here is built, and the next thing to
build is probably still nothing. It exists so that the question "should this be
one project instead of three" has a written answer to argue with.

`direct-ort.md` asks the Android-shaped version: should this app drive
`OrtSession.run` in Kotlin instead of going through sherpa-onnx. This asks the
wider one, and reaches a different answer about where the code should live.
Everything in that document about the bundle's contents and the frame loop holds
either way, because it is the same graphs and the same arithmetic.

## The situation that prompts it

Three codebases currently run the same model:

| | what it is | how it reaches the model |
|---|---|---|
| this app (as of 2026-09) | Android TTS engine + reader | sherpa-onnx C++ via JNI |
| [speech-kit-obsidian-plugin](https://github.com/mycosavant/speech-kit-obsidian-plugin) | Obsidian plugin | Rust adapter (`native/src/adapters/pocket_tts.rs`) over ONNX Runtime 1.23.2 |
| upstream `pocket_tts` | reference Python | PyTorch |

All three sound different, and until 2026-10-03 nobody could say why. The answer
turned out to be one behaviour in sherpa's C++ that the two implementations do not
share, fixed from the caller's side. That fix is the reason this document is worth
writing rather than acting on. **The case for owning more of the stack got weaker
on the same day it got clearer.**

## What is actually owned, and what only looks owned

The honest inventory:

**Nobody here owns inference.** This app's inference was ONNX Runtime. speech-kit's
inference is ONNX Runtime. sherpa-onnx's inference is ONNX Runtime. That layer is
a commodity, it is Microsoft's, and no amount of rewriting changes which library
multiplies the matrices. A "Rust-core APK" would have had the same C++ under it,
one binding down.

**sherpa-onnx owns packaging, and it is years of it.** A C API, twelve language
bindings, a model zoo of a hundred-odd converted bundles, a build matrix reaching
Android, iOS, HarmonyOS, RISC-V and several NPUs, and the per-model preprocessing
that makes a `.onnx` file into something that speaks. That is the part it would be
foolish to reproduce and cheap to keep depending on.

**speech-kit owns the pipeline, and that is the valuable part.** Chunking, voice
management, streaming, cancellation, and a provider abstraction that lets one front
end speak through several engines. None of it is inference. All of it is the
difference between a model and a product.

So the thing worth owning is the middle: **the pipeline, not the inference.**

## The shape that follows

One Rust crate that is the product:

```
                    ┌─────────────────────────────────┐
                    │  core (Rust)                    │
                    │                                 │
                    │  text → chunks → speech → pcm   │
                    │  voices, cancellation, queueing │
                    │  trait Provider                 │
                    └──────────┬──────────┬───────────┘
                               │          │
              ┌────────────────┘          └────────────────┐
              │                                            │
    ┌─────────▼──────────┐                      ┌──────────▼─────────┐
    │ provider: direct   │                      │ provider: sherpa   │
    │ ONNX Runtime       │                      │ via its C API      │
    │ 1-2 models we care │                      │ the long tail      │
    └────────────────────┘                      └────────────────────┘

    shells:  JNI/AAR (this app) · napi/wasm (Obsidian) · CLI (Termux)
```

Three claims, in order of confidence.

**1. The pipeline belongs in one place, in Rust.** It is the code that is currently
written three times and behaves three ways. It has no platform dependencies: text
in, PCM out, plus a cancellation token. That is the shape that survives being called
from JNI, from Node, and from a terminal.

**2. Direct ONNX Runtime is viable for a small, deliberate set of models.**
speech-kit's Pocket adapter is the existence proof: roughly 300 lines, and it
sounds better than sherpa did because of what it leaves out. One autoregressive
pass, no sentence splitting, no re-encoding the reference per sentence.
`direct-ort.md` reads the bundle graph by graph and finds nothing hidden. Every
graph carries its state as ordinary tensors, so a direct driver needs no new
export, no PyTorch, and no changes to model files already downloaded.

**3. sherpa stays, as one provider among several.** Not as a fallback and not as a
stepping stone. It is the answer for every model we are not going to hand-port,
which is all but one or two of them.

## The cost, stated plainly

**Anything run directly makes us the owner of model conversion and per-model
preprocessing.** That is exactly the labour sherpa performs for a hundred-odd
models, and it does not announce itself when it goes wrong. `direct-ort.md` names
the sharpest instance: the SentencePiece tokenizer is a Viterbi search over a
lattice, a short function, surrounded by a normalizer, a whitespace convention and
a byte-fallback rule. A tokenizer that is subtly wrong does not fail. It
mispronounces, occasionally, in ways nobody notices until someone reads a URL aloud.

So the direct set must stay small and deliberate. One model, possibly two. The
moment it becomes a policy rather than an exception, the project has quietly signed
up to be sherpa, without the contributors.

Three further costs are worth naming:

- **A Rust core does not remove a dependency, it moves one.** ONNX Runtime is still
  there. The gain is that its surface becomes `Session::run` with tensors we own,
  instead of a generation loop we cannot see.
- **Three shells are three build systems.** An AAR with JNI, an N-API or WASM
  module, and a static binary, each with its own cross-compilation story. Android's
  is the worst of them.
- **This app currently works.** Any migration is negative progress until the day it
  lands, measured against a baseline the owner can already hear.

## What today changed, and what it did not

The splitting fix stops sherpa-onnx cutting a chunk back into sentences and
generating each one independently. It sets two sentence-length bounds in the
`extra` map, above `TextChunker`'s cap. The author counted four lines. On the
device: *"much better, first syllable collapse is almost none."*

That matters to this decision in two directions.

**It weakens the performance case.** The largest audible gap between this app and
the reference implementation was not the binding, the language, or the inference
engine. It was one configuration decision in the C++, reachable from the caller. A
Rust rewrite would have inherited the same bug through the same C API and needed
the same change. Performance and quality are not currently arguments for owning
more.

**It strengthens the control case slightly, by showing its shape.** The reason it
took a month to find is that the behaviour was invisible: no setting, no log, no
symptom except a voice that sounded wrong in a way nobody could name. The argument
for owning the pipeline was never that we would write a faster loop. It is that a
defect in code you own is a bug, and a defect in code you call is a mystery.

But that is a weaker argument than speed, and it should be made to stand on its own
rather than borrowing speed's clothes.

## Where the models come from, which is not one place

Worth recording because it was assumed wrong once. The two projects do not share a
model supply chain:

- **sherpa-onnx** publishes pre-converted bundles as GitHub release assets under the
  `tts-models` tag. As of 2026-09-11 it publishes exactly two Pocket bundles, both
  `2026-01-26`, int8 and fp32. Upstream's own C API example on master still points
  at the int8 one. The author probed every `2026-MM-DD` name from February to
  September and found nothing newer (not re-checked here).
- **speech-kit** resolves models from Hugging Face through its own `artifacts.json`
  manifest.

So `english_2026-04` is not a newer version of what this app ran. It is a different
package from a different place, without a Mimi encoder, because its voices are
precomputed embeddings from `kyutai/pocket-tts-without-voice-cloning`. The encoder
is what turns a recorded wav into a voice, and that bundle cannot clone one.

Two consequences:

1. **Model freshness through sherpa is an upstream constraint**, confirmed rather
   than assumed. Nothing to select today.
2. **A provider abstraction has to cover the model source as well as the inference
   call.** Whatever else is true, `ModelManager` should stop hardcoding an
   artifact name: a bundle descriptor with a stable default, a resolvable manifest,
   and an explicit-URL escape hatch. That is small, useful independently, and the
   one piece of this worth doing before any decision is made.

## How to decide

Not by argument, and not soon. In this order:

1. **Exercise what exists.** Done. The Termux path was tested end to end in
   September (`eleven-utterances.md`).
2. **Un-hardcode the model name.** Done in fb538e3. `ModelManager` takes a model
   name, and `ModelCatalog` reads `pocket-speak/models.json`.
3. **Build the tokenizer alone and diff it** against sherpa's tokenisation over a few
   thousand sentences. This was the highest-risk step and the same first step
   `direct-ort.md` names. It was built as `pocket-speak/tools/tokdiff`, which checks
   `sentencepiece-rs` against Google's SentencePiece rather than against sherpa.
4. **Then extract the pipeline, not the inference.** If a Rust core happens, it
   should begin as chunking, voice management, streaming and cancellation behind a
   provider trait with *one* provider, sherpa, and no behaviour change at all. A
   migration that changes how it sounds on the same day it changes where the code
   lives cannot be debugged.

## What would make this not worth doing

- **sherpa keeps answering.** It has now yielded `numSteps`, `temperature`, `seed`
  and both sentence-length bounds to a caller willing to read the C++. The gap has
  narrowed twice in a week. If it narrows again, the control argument is gone.
- **The tokenizer does not agree**, per above.
- **Three shells turn out to be three projects.** The premise is that a Rust core
  makes one pipeline serve three front ends. If in practice each shell needs its own
  scheduling, its own audio session and its own error handling, the shared core is a
  header file with ambitions.
- **The app is good enough.** It is, currently. That is not a small thing, and "we
  own it" is not a feature anybody can hear.
