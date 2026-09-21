"""The tokenizer.json backend must reproduce the SentencePiece ids exactly."""

import pytest

from pocket_tts.modules.text_conditioner import (
    DEFAULT_TOKENIZER_N_BINS,
    JsonTokenizer,
    SentencePieceTokenizer,
    build_tokenizer,
)

SP_PATH = (
    "hf://kyutai/pocket-tts-without-voice-cloning/"
    "tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
)
JSON_PATH = (
    "hf://kyutai/pocket-tts-without-voice-cloning/"
    "tokenizer.json@00eac05ed3d16bdc3f6b5d598874019c34a89214"
)
# Leading and repeated whitespace exercise SentencePiece's add_dummy_prefix, the
# one place the two implementations can disagree.
TEXTS = [
    "Hello, world!",
    "café",
    " Hello, world!",
    "  hello  ",
    " ",
    "",
    "14½-13½",
    "Donkey kong uses the front lever to dodge the smash attack from fox.",
]


@pytest.mark.parametrize("text", TEXTS)
def test_json_matches_sentencepiece(text: str) -> None:
    sp = SentencePieceTokenizer(DEFAULT_TOKENIZER_N_BINS, SP_PATH)
    js = JsonTokenizer(DEFAULT_TOKENIZER_N_BINS, JSON_PATH)
    assert js.encode(text) == sp.encode(text)
    assert js.decode(js.encode(text)) == sp.decode(sp.encode(text))


def test_build_tokenizer_picks_the_backend() -> None:
    assert isinstance(build_tokenizer(DEFAULT_TOKENIZER_N_BINS, JSON_PATH), JsonTokenizer)
    # A community model trained before this change keeps working.
    assert isinstance(build_tokenizer(DEFAULT_TOKENIZER_N_BINS, SP_PATH), SentencePieceTokenizer)
    assert isinstance(
        build_tokenizer(DEFAULT_TOKENIZER_N_BINS, SP_PATH, "sentencepiece"), SentencePieceTokenizer
    )
