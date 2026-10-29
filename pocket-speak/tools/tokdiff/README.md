# tokdiff

Checks that `sentencepiece-rs`, the pure-Rust SentencePiece the engine uses for
`tokenizer.model` files, produces the same token ids as Google's
`sentencepiece`, on the pinned `english_2026-04` tokenizer.
`android/docs/owning-the-pipeline.md` ranks this as the highest-risk step,
because a tokenizer that is subtly wrong does not fail. It mispronounces.

Run it from the repository root. It needs `curl`, `uv`, `sha256sum`, and `cargo`:

```
pocket-speak/tools/tokdiff/run.sh /tmp/tokdiff docs android/docs README.md
```

The script downloads `tokenizer.model` and `bundle.json` from a pinned Hugging
Face commit (`KevinAHM/pocket-tts-onnx`) and checks their sha256. It builds a
corpus of sentences from the Markdown trees or files given (each sentence raw
and in the engine's prepared form), plus 5,000 seeded random mixed-script
strings (seed 20260913). It compares the ids line by line and exits non-zero on
any mismatch.

## Result, 2026-09-13

`sentencepiece-rs` 0.2.2 against Google `sentencepiece` 0.2.2, on a corpus built
from a Warp fork's docs and run records, and this repository's docs:

| set | rows | mismatches |
|---|---|---|
| sentences (13,529 raw + 1,283 prepared) | 14,812 | 0 |
| random strings, seed 20260913 | 5,000 | 0 |

The pinned `tokenizer.model` has the same sha256 as the `tokenizer.model` that
the 18m, 7m, and 2m models use in `pocket-speak/models.json`. The `english_2026-10`
default does not use this format. It reads a `tokenizer.json` through the Hugging
Face `tokenizers` crate, and this check does not cover that path.

Not covered: decoding, sampling or n-best modes, other languages, the
`tokenizers` path, and sentences over 400 characters, which the corpus builder
leaves out.
