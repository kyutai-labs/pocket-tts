"""Tests for the user-defined pronunciation lexicon."""

import json
from pathlib import Path

import pytest

from pocket_tts.models.pronunciation import apply_pronunciations, load_pronunciations


def test_no_lexicon_leaves_text_unchanged():
    assert apply_pronunciations("The session is live.", None) == "The session is live."
    assert apply_pronunciations("The session is live.", {}) == "The session is live."


def test_whole_words_only():
    lexicon = {"live": "lyve"}
    assert apply_pronunciations("The session is live.", lexicon) == "The session is lyve."
    assert apply_pronunciations("We delivered it.", lexicon) == "We delivered it."
    assert apply_pronunciations("Lively, alive.", lexicon) == "Lively, alive."


def test_case_insensitive_and_keeps_capitalization():
    lexicon = {"live": "lyve"}
    assert apply_pronunciations("Live now, LIVE!", lexicon) == "Lyve now, Lyve!"


def test_respelling_case_is_kept_otherwise():
    assert apply_pronunciations("Use the update.", {"Update": "Apdäit"}) == "Use the Apdäit."


def test_multi_word_and_punctuated_entries():
    lexicon = {"E-Mail": "Iimeil", "Wingman AI": "Wingmän Ai"}
    text = "Wingman AI schickt eine E-Mail."
    assert apply_pronunciations(text, lexicon) == "Wingmän Ai schickt eine Iimeil."


def test_longest_entry_wins():
    lexicon = {"Arc": "Ark", "ArcCorp": "Ark Korp"}
    assert apply_pronunciations("Fly to ArcCorp, then Arc.", lexicon) == (
        "Fly to Ark Korp, then Ark."
    )


def test_replacements_are_not_applied_twice():
    lexicon = {"read": "red", "red": "rehd"}
    assert apply_pronunciations("I read the red book.", lexicon) == "I red the rehd book."


def test_blank_keys_are_ignored():
    assert apply_pronunciations("Hello world.", {"": "x", " ": "y"}) == "Hello world."


def test_load_pronunciations(tmp_path: Path):
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"Meeting": "Mieting"}), encoding="utf-8")
    assert load_pronunciations(path) == {"Meeting": "Mieting"}


@pytest.mark.parametrize("content", ['["live", "lyve"]', '{"live": 1}'])
def test_load_pronunciations_rejects_invalid_files(tmp_path: Path, content: str):
    path = tmp_path / "lexicon.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_pronunciations(path)
