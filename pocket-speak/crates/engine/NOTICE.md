# Notices

## Upstream

`src/synth.rs` is ported from `rust/src/model.rs` and `rust/src/rng.rs` in
[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts), this
repository's upstream. `src/lib.rs` compiles four more of upstream's files
unchanged: `config.rs`, `defaults.rs`, `text.rs` (as `prompt`), and `weights.rs`.

The model is [xn-ptts](https://github.com/gradium-ai/xn-ptts), used through the
fork at [gabrieldemarmiesse/xn-ptts](https://github.com/gabrieldemarmiesse/xn-ptts)
at the commit upstream pins (`7efa269`). Upstream pocket-tts is MIT licensed, as
this repository's `LICENSE` says. xn-ptts is MIT or Apache-2.0.

`src/text.rs`'s chunking follows speech-kit's sentence-per-generation strategy
([speech-kit-obsidian-plugin](https://github.com/brittain9/speech-kit-obsidian-plugin),
MIT). Its trailing-short-sentence join follows
`android/app/src/main/java/org/pockettts/android/speech/TextChunker.kt` in this
repository.

## Models

The engine runs models it does not ship. The app and `pocket-speak install`
fetch the files that `pocket-speak/models.json` pins, from
[kyutai/pocket-tts-without-voice-cloning](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning).
Those files are the weights, tokenizers, and precomputed voice states of each
offered model, licensed CC-BY-4.0.

Pocket TTS is by [Kyutai](https://kyutai.org).
