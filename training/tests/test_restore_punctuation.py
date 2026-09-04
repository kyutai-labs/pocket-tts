from typing import Any

from training.scripts.restore_punctuation import reattach

ENTRY: dict[str, Any] = {
    "path": "/data/a.mp3",
    "duration": 3.0,
    "transcript": "selamat pagi indonesia",
    "words": [
        {"word": "selamat", "start": 0.0, "end": 1.0},
        {"word": "pagi", "start": 1.0, "end": 2.0},
        {"word": "indonesia", "start": 2.0, "end": 3.0},
    ],
}


def test_reattach_keeps_timestamps_and_updates_surfaces():
    out = reattach(ENTRY, "Selamat pagi, Indonesia?")
    assert out["transcript"] == "Selamat pagi, Indonesia?"
    assert out["words"] == [
        {"word": "Selamat", "start": 0.0, "end": 1.0},
        {"word": "pagi,", "start": 1.0, "end": 2.0},
        {"word": "Indonesia?", "start": 2.0, "end": 3.0},
    ]
    assert ENTRY["words"][0]["word"] == "selamat", "input must not be mutated"


def test_reattach_leaves_entry_alone_on_token_count_mismatch():
    assert reattach(ENTRY, "Selamat pagi.") is ENTRY
    assert reattach(ENTRY, "Selamat pagi, Indonesia raya!") is ENTRY
