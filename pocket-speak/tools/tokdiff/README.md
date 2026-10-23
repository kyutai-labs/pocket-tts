# tokdiff

Checks that `sentencepiece-rs`, the pure-Rust SentencePiece the desk CLI uses,
turns text into the same token ids as Google's `sentencepiece` on the pinned
`english_2026-04` tokenizer. `android/docs/owning-the-pipeline.md` names this
as the first thing to settle, because a tokenizer that is subtly wrong does
not fail: it mispronounces.

```
pocket-speak/tools/tokdiff/run.sh /tmp/tokdiff docs android/docs README.md
```

It downloads `tokenizer.model` and `bundle.json` from a pinned Hugging Face
commit and checks their sha256, builds a corpus of sentences from the given
Markdown trees (raw, and in the engine's prepared form) plus 5,000 seeded
random mixed-script strings, and compares ids line by line. Exits non-zero on
any mismatch.

## Result, 2026-09-13

`sentencepiece-rs` 0.2.2 against Google `sentencepiece` 0.2.2, on a corpus
built from a Warp fork's docs and run records and this repo's docs:

| set | rows | mismatches |
|---|---|---|
| sentences (13,529 raw + 1,283 prepared) | 14,812 | 0 |
| random strings, seed 20260913 | 5,000 | 0 |

The bundle's `tokenizer.model` is byte-identical to the one
`pocket_tts/conditioners/text.py` loads
(`kyutai/pocket-tts-without-voice-cloning@d4fdd22`).

Not covered: decoding, sampling or n-best modes, other languages, texts longer
than 400 characters.
