"""Convert a SentencePiece .model to the tokenizer.json the configs load.

Handles both model types: unigram maps onto tokenizers.Unigram directly, and
bpe onto tokenizers.BPE, whose merges sentencepiece does not store -- they are
reconstructed by taking every (left, right) whose concatenation is itself a
piece, ordered by the merged piece's score.

The .model's normalizer_spec is reproduced as well. sentencepiece applies it
before segmentation, so a converter that skips it silently changes the ids: the
released tokenizers use `identity`, but a tokenizer trained with sentencepiece's
own defaults NFKC-folds (one half becomes 1/2) and collapses repeated spaces.

Conversion is only useful if it is exact, so this checks every sentence it can
against sentencepiece and exits non-zero on any mismatch.

    python -m training.scripts.convert_tokenizer data/tokenizer.model \
        --out data/tokenizer.json --texts data/valid.jsonl
"""

import json
from pathlib import Path
from typing import Annotated, Any

import sentencepiece as spm
import typer
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import ByteFallback, Fuse
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import BPE, Unigram
from tokenizers.normalizers import Precompiled, Prepend, Replace, Strip
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Metaspace

app = typer.Typer(pretty_exceptions_show_locals=False)

# The two .model fields this needs live in submessages of ModelProto, and
# sentencepiece's generated protobuf bindings need protobuf installed, which the
# runtime does not otherwise require. Reading the handful of fields directly off
# the wire keeps the script dependency-free; a misread cannot ship silently
# because every conversion is checked against sentencepiece below.
MODEL_TYPES = {1: "UNIGRAM", 2: "BPE", 3: "WORD", 4: "CHAR"}


def _fields(buf: bytes) -> dict[int, list[Any]]:
    """{field number: values} of one protobuf message, varints and byte strings."""
    out: dict[int, list[Any]] = {}
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            length, i = _varint(buf, i)
            value, i = buf[i : i + length], i + length
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.setdefault(field, []).append(value)
    return out


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


# Whitespace cases first: they exercise add_dummy_prefix and the whitespace rules,
# where the two implementations are easiest to get subtly wrong.
CHECKS = ["", " ", "  hello  ", " Hello, world!", "Hello, world!", "café", "14½-13½", "日本語", "£"]


def _merges(vocab: dict[str, int], scores: dict[str, float]) -> list[tuple[str, str]]:
    pieces = set(vocab)
    merges = [
        (scores[piece], piece[:i], piece[i:])
        for piece in vocab
        for i in range(1, len(piece))
        if piece[:i] in pieces and piece[i:] in pieces
    ]
    merges.sort(key=lambda m: m[0], reverse=True)
    return [(left, right) for _, left, right in merges]


def _normalizer(spec: dict[int, list[Any]]) -> NormalizerSequence:
    """NormalizerSpec: 2 precompiled_charsmap, 3 add_dummy_prefix, 4 remove_extra_whitespaces."""
    charsmap = spec.get(2, [b""])[0]
    add_dummy_prefix = spec.get(3, [1])[0]
    remove_extra_whitespaces = spec.get(4, [1])[0]
    steps: list[Any] = []
    if charsmap:
        steps.append(Precompiled(charsmap))
    if remove_extra_whitespaces:
        steps.append(Replace(Regex(" {2,}"), " "))
        steps.append(Strip())
    if add_dummy_prefix:
        # sentencepiece prepends unconditionally; Metaspace's own prepend_scheme
        # would skip a string that already starts with the marker.
        steps.append(Prepend(prepend="▁"))
    return NormalizerSequence(steps)


def build(model_path: Path) -> Tokenizer:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    pieces = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]

    # ModelProto: 2 trainer_spec (3 model_type), 3 normalizer_spec.
    proto = _fields(model_path.read_bytes())
    trainer_spec = _fields(proto[2][0])
    model_type = MODEL_TYPES.get(trainer_spec.get(3, [1])[0], "UNKNOWN")
    if model_type == "UNIGRAM":
        model = Unigram(pieces, unk_id=sp.unk_id(), byte_fallback=True)
    elif model_type == "BPE":
        vocab = {piece: i for i, (piece, _) in enumerate(pieces)}
        model = BPE(
            vocab,
            _merges(vocab, dict(pieces)),
            unk_token=sp.id_to_piece(sp.unk_id()),
            fuse_unk=True,
            byte_fallback=True,
        )
    else:
        raise typer.BadParameter(f"unsupported sentencepiece model type {model_type}")

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = _normalizer(_fields(proto[3][0]))
    tokenizer.pre_tokenizer = Metaspace(prepend_scheme="always")
    tokenizer.decoder = DecoderSequence(
        [MetaspaceDecoder(prepend_scheme="always"), ByteFallback(), Fuse()]
    )
    for i in range(sp.get_piece_size()):
        if sp.is_control(i) or sp.is_unknown(i):
            tokenizer.add_special_tokens([AddedToken(sp.id_to_piece(i), special=True)])
    return tokenizer


@app.command()
def main(
    model: Annotated[Path, typer.Argument(help="the SentencePiece .model to convert")],
    out: Annotated[
        Path | None, typer.Option(help="default: tokenizer.json beside the model")
    ] = None,
    texts: Annotated[
        Path | None, typer.Option(help="extra sentences to check: a manifest jsonl or one per line")
    ] = None,
) -> None:
    tokenizer = build(model)
    sp = spm.SentencePieceProcessor(model_file=str(model))

    checks = list(CHECKS)
    if texts is not None:
        for line in texts.read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                line = json.loads(line).get("transcript", "")
            if line:
                checks.append(line)
    mismatched = [t for t in checks if tokenizer.encode(t).ids != sp.encode(t, out_type=int)]
    if mismatched:
        for text in mismatched[:5]:
            print(f"MISMATCH {text[:60]!r}")
            print(f"  json {tokenizer.encode(text).ids[:12]}")
            print(f"  spm  {sp.encode(text, out_type=int)[:12]}")
        raise SystemExit(f"{len(mismatched)}/{len(checks)} texts tokenize differently; not written")

    out = out or model.with_name("tokenizer.json")
    tokenizer.save(str(out))
    print(f"wrote {out} ({len(checks)} texts match sentencepiece exactly)")


if __name__ == "__main__":
    app()
