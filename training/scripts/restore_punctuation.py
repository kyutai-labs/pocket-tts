"""Put punctuation back into a manifest whose transcripts lost it.

ASR- and subtitle-derived corpora (LEMAS, YODAS, GigaSpeech) ship normalized
text: no commas, no full stops. A TTS trained on that has no way to learn
sentence-final intonation or comma pauses, and worse, its tokenizer has no
piece for "," or "." -- so a user typing ordinary text feeds the model <unk>
embeddings it never saw in training.

This restores punctuation (and truecasing) with
1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase, a 47-language ONNX
model that covers Indonesian. It runs on CPU, so it does not compete with a
training job for the GPU.

Only surface forms change: word count, order and timestamps are untouched, and
an utterance whose restored text no longer has one token per aligned word is
written back unchanged rather than dropped.

Run it standalone (no project deps, and `punctuators` stays out of pyproject):
    uv run --with punctuators --no-project python training/scripts/restore_punctuation.py \
        data/id_train.jsonl data/id_train_punct.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, TextIO


def reattach(entry: dict[str, Any], restored: str) -> dict[str, Any]:
    """Copy `restored`'s tokens onto the entry's words, or return it unchanged.

    The punctuation model attaches marks to the token they follow and never
    splits or merges tokens, so a token-count mismatch means something else
    happened (a dropped token, a segmentation quirk) and the entry is left
    alone -- a wrong word/timestamp pairing is worse than missing punctuation.
    """
    tokens = restored.split()
    if len(tokens) != len(entry["words"]):
        return entry
    entry = dict(entry)
    entry["words"] = [{**w, "word": t} for w, t in zip(entry["words"], tokens, strict=True)]
    entry["transcript"] = " ".join(tokens)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_jsonl", type=Path)
    ap.add_argument("output_jsonl", type=Path)
    ap.add_argument("--batch", type=int, default=256, help="utterances per model call")
    ap.add_argument(
        "--model", default="1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    )
    args = ap.parse_args()

    # Deliberately not a project dependency: run this script standalone with
    # `uv run --with punctuators --no-project`.
    from punctuators.models import (  # ty: ignore[unresolved-import]  -- optional extra
        PunctCapSegModelONNX,
    )

    model = PunctCapSegModelONNX.from_pretrained(args.model)

    n = n_changed = 0

    def flush(chunk: list[dict[str, Any]], fout: TextIO) -> None:
        nonlocal n, n_changed
        # infer() returns the sentences it split each input into; the TTS
        # manifest wants one flat transcript, so join them back.
        for entry, sentences in zip(chunk, model.infer([e["transcript"] for e in chunk]), strict=True):
            out = reattach(entry, " ".join(sentences))
            n_changed += out["transcript"] != entry["transcript"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
        if n % (args.batch * 20) == 0:
            print(f"{n} utterances, {n_changed} repunctuated", flush=True)

    with open(args.input_jsonl) as fin, open(args.output_jsonl, "w") as fout:
        chunk: list[dict[str, Any]] = []
        for line in fin:
            chunk.append(json.loads(line))
            if len(chunk) == args.batch:
                flush(chunk, fout)
                chunk = []
        if chunk:
            flush(chunk, fout)
    print(f"done: {n} utterances, {n_changed} repunctuated -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
