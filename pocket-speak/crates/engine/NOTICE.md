# Notices

## Upstream

`src/synth.rs` is ported from `rust/src/model.rs` and `rust/src/rng.rs` of
[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts), this
repository's upstream, and `src/lib.rs` compiles in four more of its files
unchanged. The model itself is
[xn-ptts](https://github.com/gradium-ai/xn-ptts), through the same fork and
commit upstream uses. Both are under the MIT License (xn-ptts: MIT or
Apache-2.0); see this repository's `LICENSE`.

`src/text.rs`'s chunking follows speech-kit's sentence-per-generation strategy
([speech-kit-obsidian-plugin](https://github.com/brittain9/speech-kit-obsidian-plugin),
MIT), and its trailing-short-sentence join follows this repository's
`android/app/src/main/java/org/pockettts/android/speech/TextChunker.kt`.

## Models

The engine runs models it does not ship. The app and `pocket-speak install`
fetch the files `pocket-speak/models.json` pins, from
[kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning):
the weights, tokenizers and precomputed voice states of each offered model,
CC-BY-4.0.

Pocket TTS is by [Kyutai](https://kyutai.org).
