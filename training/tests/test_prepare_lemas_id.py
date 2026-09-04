from pathlib import Path
from typing import Any

from training.scripts.prepare_lemas import convert, mean_score, speaker_of

RECORD: dict[str, Any] = {
    "key": "id_UJwgKZklwB4-00524-00177227-00177461",
    "audio": "id000/UJwgKZklwB4-00524-00177227-00177461.mp3",
    "dur": 2.33,
    "txt": "menjalankan satu saja",
    "align": {
        "txt": "menjalankan satu saja",
        "words": [
            {"word": "menjalankan", "start": 0.08, "end": 0.703, "score": 1.0},
            {"word": "satu", "start": 0.763, "end": 1.004, "score": 0.5},
            {"word": "saja", "start": None, "end": None, "score": 0.0},
        ],
    },
}


def test_convert_renames_fields_and_drops_untimed_words():
    entry = convert(RECORD, Path("/data/id"), speaker_parts=3)
    assert entry == {
        "path": "/data/id/id000/UJwgKZklwB4-00524-00177227-00177461.mp3",
        "duration": 2.33,
        "transcript": "menjalankan satu saja",
        "words": [
            {"word": "menjalankan", "start": 0.08, "end": 0.703},
            {"word": "satu", "start": 0.763, "end": 1.004},
        ],
        "speaker": "UJwgKZklwB4",
    }


def test_convert_returns_none_without_any_timestamps():
    record = {**RECORD, "align": {"txt": "x", "words": [{"word": "x"}]}}
    assert convert(record, Path("/data/id"), speaker_parts=3) is None


def test_mean_score_averages_over_all_words():
    assert mean_score(RECORD) == 0.5


def test_speaker_of_strips_segment_and_offsets():
    # Segments of one video must hash to one split.
    assert speaker_of(RECORD["key"], 3) == "UJwgKZklwB4"
    assert speaker_of("id_UJwgKZklwB4-00001-00000000-00000100", 3) == "UJwgKZklwB4"


def test_speaker_of_groups_the_non_youtube_shards_too():
    # id000-style keys are `id_train_<recording>-<seg>-<subseg>`: segments of one
    # recording must still land in one split.
    assert speaker_of("id_train_553-18-2", 2) == speaker_of("id_train_553-19-1", 2) == "train_553"
