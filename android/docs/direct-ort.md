# Running the model directly on onnxruntime

> Superseded, and the Kotlin driver described here was never written. On
> 2026-10-03 the app moved to a Rust core (`engine-swap.md`) that ran the model
> through ONNX Runtime. On 2026-10-23 that core moved to upstream's pure-Rust
> engine (`pure-rust-engine.md`). The rest of this page is the reasoning as it
> stood before 2026-10-03, left as written.

A scope, not a plan of record. Nothing here is built. It exists so the decision
to build it, or not to, is made against what is actually in the model bundle
rather than against a guess.

This is the Android-shaped version of the question: Kotlin driving `OrtSession.run`
in this app. `owning-the-pipeline.md` asks the cross-platform version, whether the
thing worth owning is a Rust core shared by this app, the Obsidian plugin and a
Termux CLI. That document reaches a different answer about where the code should
live. Everything below about the bundle's contents and the frame loop holds either
way: it is the same graphs and the same arithmetic.

## Why this comes up

sherpa-onnx has been a good deal. It gave this app a working Pocket TTS in an
afternoon, and its C++ does real work: the generation loop, the streaming state,
the tokenizer, the voice-embedding cache.

It has also cost. Twice the JNI boundary has deleted the audio callback in a way
that compiled, passed every test, and produced silence on a phone: once through
D8's lambda desugaring, once through R8's inlining. Both were found by reading a
`dexdump` of the artefact, and both are now guarded in CI (`check-jni-callback.sh`).
The class of failure is inherent to resolving a Kotlin method from native code by
its exact descriptor. Beyond that, everything the C++ decides is a thing this app
cannot decide: what the tokenizer does with a URL, how text is prepared, how many
flow steps a frame gets. That last one only became reachable because
`GenerationConfig` happens to expose it.

**One of those has since moved.** How text is split turned out to be reachable
too, through the same `extra` map. `max_char_in_sentence` and
`min_char_in_sentence` are read out of it, and setting both above `TextChunker`'s
cap stops the C++ re-splitting a chunk into sentences and generating each one
independently. That was the largest audible defect this app had. It was fixed from
the caller's side in a few lines. That is the strongest evidence yet for the last
entry under *What would make this not worth doing*: the control argument keeps
shrinking as the `extra` map is read more carefully.

## What is actually in the bundle

Read off `sherpa-onnx-pocket-tts-int8-2026-01-26` with `onnx.load`, not inferred.
This is the sherpa bundle, not the `english_2026-04` export the app runs now:

| Graph | Size | Inputs | Outputs |
|---|---|---|---|
| `text_conditioner.onnx` | 16 MB | `token_ids: int64[1, seq]` | `embeddings: float[1, seq, 1024]` |
| `encoder.onnx` | 73 MB | `audio: float[1, 1, n]` | `latents: float[..., ..., latent_len]` |
| `lm_main.int8.onnx` | 76 MB | `sequence: float[1, seq, 32]`, `text_embeddings: float[1, text, 1024]`, 18 state tensors | `conditioning: float[1, 1024]`, `eos_logit: float[1, 1]`, 18 out-states |
| `lm_flow.int8.onnx` | 10 MB | `c: float[b, 1024]`, `s: float[b, 1]`, `t: float[b, 1]`, `x: float[b, 32]` | `flow_dir: float[..., 32]` |
| `decoder.int8.onnx` | 23 MB | `latent: float[1, seq, 32]`, 56 state tensors | `audio_frame`, 56 out-states |
| `vocab.json` | 68 KB | 4000 pieces to ids | |
| `token_scores.json` | 121 KB | 4000 pieces to log-probs | |

The key fact is that every graph carries its state as ordinary tensors, in and
out. There is no hidden session state, nothing that has to be kept alive inside a
C++ object. `lm_main` takes three state tensors per layer for six layers: a
`[2, 1, 1000, 16, 64]` KV cache, an empty `float[0]`, and an `int64[1]` position.
It hands back updated ones. `decoder` does the same for its 28 streaming
convolution positions.

That means a direct port needs no new export, no PyTorch, no C++, and no changes to
the model files people have already downloaded. It is Kotlin driving `OrtSession.run`
with tensors it owns.

## The loop, from the reference implementation

From `pocket_tts/models/flow_lm.py` (`lsd_decode`), `pocket_tts/models/tts_model.py`
(`get_state_for_audio_prompt`) and `pocket_tts/models/text_chunking.py`
(`prepare_text_prompt`) in this repository. Mimi runs at 12.5 Hz over 24 kHz audio,
so one frame is 80 ms. Real time means the whole of the following must finish in
under 80 ms:

1. `lm_main(sequence = previous latent, text_embeddings, states) → conditioning c, eos_logit`
2. `lsd_decode`: start from Gaussian noise `x`, and for `i in 0 until numSteps`,
   with `s = i/numSteps` and `t = (i+1)/numSteps`,
   `x += lm_flow(c, s, t, x) / numSteps`
3. `decoder(x, states) → 1920 samples`
4. stop when `eos_logit` crosses the threshold, plus a few frames of tail

Setting up a voice is `encoder(prompt audio) → latents`, then running those latents
through `lm_main` to warm the KV cache. That is what `get_state_for_audio_prompt`
does, and it is why upstream can save a voice as a state file. A direct port could
do the same: warm a voice once and keep the cache, rather than re-encoding a prompt
for every sentence.

Two numbers to keep in mind. The sherpa bundle's KV cache is fixed at 1000
positions. At 12.5 Hz that is 80 seconds of context, and 49 MB of float, passed in
and back out every frame. And `numSteps` multiplies only step 2, the 10 MB flow net.
It does not touch the 76 MB `lm_main` that runs once per frame. Whatever the
decode-steps slider is worth, it is bounded by the fraction of a frame that the
small model accounts for, and that fraction is unmeasured.

## What would have to be written

In rough order of risk.

**The tokenizer.** `vocab.json` and `token_scores.json` are a SentencePiece unigram
model: 4000 pieces with log-probs, `<unk> <s> </s> <pad>` at 0 to 3, and
`<0x00>`... byte-fallback pieces after them. Encoding is a Viterbi search over the
lattice of matching pieces, which is a short function. The risk is not the Viterbi.
It is everything around it: the normalizer, the `▁` whitespace convention, and how
byte fallback fires. A tokenizer that is subtly wrong does not fail. It mispronounces,
occasionally, in ways nobody notices until someone reads a URL aloud. Build this
part first and test it against sherpa-onnx's own output on a corpus.

**The frame loop and its state.** Mechanical, but the allocation discipline
matters: 49 MB of KV cache cannot be allocated per frame. Use two buffers per state,
swapped, with ORT's `IoBinding` pointing at them. Java's `OnnxTensor.createTensor`
over a `FloatBuffer` allocated once is the shape of it.

**Text preparation.** `prepare_text_prompt` in the reference capitalises the first
letter, adds terminal punctuation when it is missing, replaces semicolons, and pads short inputs
with spaces. These are small rules with audible consequences, and they are
currently invisible to us because they run in C++.

**Voice prompts.** `encoder` then a warm-up pass, cached per voice. Upstream's
`export_model_state` (`pocket_tts/models/model_state.py`) suggests warmed states
could be shipped instead of WAVs, which would remove the encode from first-read
latency entirely.

**Sentence splitting.** Already ours (`TextChunker`), and now actually honoured:
the `extra` map's two sentence-length bounds stop sherpa-onnx re-splitting what
`TextChunker` produced. The reference splits on a token budget of 50
(`MAX_TOKEN_PER_CHUNK`) rather than on characters. That is worth reconciling, and
cheaper to reconcile now that only one splitter is running.

## What it costs

- **Dependency.** This swaps `sherpa-onnx` (a 49 MB AAR carrying four ABIs, which
  the app filtered to two) for `onnxruntime-android`. sherpa bundles onnxruntime
  already, so this may well be a saving. The number is unmeasured, and measuring it
  is one build with the dependency swapped and `assembleRelease` compared.
- **A dependency either way.** This is not a route to zero. It is a route to one
  dependency whose surface is `OrtSession.run`, instead of one whose surface is a
  generation loop we cannot see.
- **Everything sherpa does correctly, which we would then have to keep doing
  correctly**, with no upstream to inherit fixes from.

## How to decide

Not by argument. In this order:

1. **Measure first.** The decode-steps slider and `Timings` are already shipped,
   and each chunk trace line now carries `first=NNNms`. If Pocket TTS through
   sherpa-onnx is comfortably faster than real time on the phone, the performance
   argument for a port is dead and only the control argument remains. *Partly
   answered:* generation runs at about 1.12x real time on the owner's device with a
   stock chunk, so it keeps up, with little margin. That figure is not in the
   repository.
2. **Build the tokenizer alone**, offline, and diff it against sherpa-onnx's
   tokenisation of a few thousand sentences. It is the highest-risk piece and the
   cheapest to abandon. If it does not agree, stop here. (Later done as
   `pocket-speak/tools/tokdiff`, which checks against Google's SentencePiece.)
3. **Then a bench**: one graph at a time, timed on the device, so the per-frame
   budget is known before any of it is load-bearing.
4. **Then, behind the `SpeechEngine` seam.** That interface already exists and is
   already faked in tests, so a second implementation costs the reader nothing and
   can be switched per read.

## What would make this not worth doing

- sherpa-onnx exposing what we want anyway. It exposes `numSteps`, `temperature`,
  `seed`, and, as of the splitting fix, both sentence-length bounds. The gap is
  narrower than it looks, and it has narrowed twice since this was written.
- The tokenizer failing to agree. Everything downstream is arithmetic. That part is
  judgement encoded in a file we do not have.
- Real-time factor on the device coming back comfortable. Control is a weaker
  argument than speed, and it should be made to stand on its own if speed is not in
  question.
