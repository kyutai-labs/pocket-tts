"""Train a sentencepiece tokenizer for a new language or domain.

The flow LM looks text up in a fixed-size table, and pocket-tts asserts that
lookup_table.n_bins in the model config equals the tokenizer's vocab size
exactly. The default here is the 4000 the released configs already use, so a
new tokenizer drops in without touching n_bins; the padding id is the table's
extra +1 row, not a reserved bin.

Input is one or more manifest jsonl files (any records with a "transcript"
field, e.g. the training manifests) or plain-text files (one utterance per
line). Point the model config's lookup_table.tokenizer_path at the produced
<prefix>.model to train with it.

The defaults reproduce the spec the released tokenizers were trained with, read
back off one of their .model files: unigram, vocab 4000, character coverage
0.9995, pieces at most 6 characters, digits split, byte fallback, a pad id, and
normalization left off ("identity", no whitespace squashing) so the text reaches
the model as written. sentencepiece's own defaults differ on every one of those,
and the normalization ones change what the tokenizer does to text at inference,
not just how it segments.

Usage:
    python -m training.scripts.train_tokenizer out/tokenizer \
        data/train.jsonl [more files ...]
"""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

import sentencepiece as spm
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)


def iter_texts(paths: list[Path]) -> Iterator[str]:
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    text = json.loads(line).get("transcript", "")
                else:
                    text = line
                if text:
                    yield text


@app.command()
def main(
    output_prefix: Annotated[str, typer.Argument(help="writes <prefix>.model and <prefix>.vocab")],
    inputs: Annotated[list[Path], typer.Argument(help="manifest .jsonl or plain-text files")],
    vocab_size: Annotated[
        int,
        typer.Option(help="the model config's lookup_table.n_bins must be set to this exact value"),
    ] = 4000,
    character_coverage: Annotated[
        float, typer.Option(help="raise to 1.0 for a small alphabet to keep every character")
    ] = 0.9995,
    model_type: Annotated[Literal["unigram", "bpe", "char"], typer.Option()] = "unigram",
):
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        n = 0
        for text in iter_texts(inputs):
            tmp.write(text.replace("\n", " ") + "\n")
            n += 1
        corpus = tmp.name
    print(f"training on {n} utterances")
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        # The rest reproduces the spec the released tokenizers were trained with.
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        max_sentencepiece_length=6,
        allow_whitespace_only_pieces=True,
        split_digits=True,
        byte_fallback=True,
        pad_id=3,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
    )
    sp = spm.SentencePieceProcessor(model_file=output_prefix + ".model")
    print(f"wrote {output_prefix}.model (vocab {sp.get_piece_size()})")
    print(f"convert it with: python -m training.scripts.convert_tokenizer {output_prefix}.model")


if __name__ == "__main__":
    app()
