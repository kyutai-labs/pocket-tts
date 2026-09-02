"""prepare_data.py keeps chapter audio files whole: utterances sharing a
chapter share one manifest `path`, distinguished only by `start`."""

import gzip
import json
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

from training.scripts import prepare_data


def _write_jsonl(path: Path, records: list[dict[str, Any]]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_utterances_in_one_chapter_share_the_manifest_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    chapters_json = tmp_path / "chapters.json"
    manifest_json = tmp_path / "manifest.json"
    _write_jsonl(
        chapters_json,
        [
            {
                "chapter_filepath": "book1/ch1",
                "url": "http://example.invalid/ch1.mp3",
                "duration": 9.0,
                "utterances": [
                    {"audio_filepath": "book1/ch1_utt0", "offset": 0.0, "duration": 5.0},
                    {"audio_filepath": "book1/ch1_utt1", "offset": 5.0, "duration": 4.0},
                ],
            }
        ],
    )
    _write_jsonl(
        manifest_json,
        [
            {
                "audio_filepath": "book1/ch1_utt0",
                "duration": 5.0,
                "normalized_text": "hello world",
                "speaker": "spk1",
                "set": "train",
            },
            {
                "audio_filepath": "book1/ch1_utt1",
                "duration": 4.0,
                "normalized_text": "goodbye",
                "speaker": "spk1",
                "set": "dev",
            },
        ],
    )

    def fake_hf_hub_download(repo: str, filename: str, repo_type: str | None) -> str:
        return str(chapters_json) if "chapters" in filename else str(manifest_json)

    def fake_download(url: str, dest: Path, **kwargs: float):
        # Simulate a successful download: touch the destination.
        open(dest, "w").close()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(prepare_data, "download", fake_download)

    audio_out = tmp_path / "downloads"
    out_dir = tmp_path / "manifests"
    audio_out.mkdir()
    out_dir.mkdir()

    train_m, valid_m = prepare_data.prepare_hifitts2(audio_out, out_dir, None, 1)

    train_recs = [json.loads(line) for line in train_m.open()]
    valid_recs = [json.loads(line) for line in valid_m.open()]
    assert len(train_recs) == 1
    assert len(valid_recs) == 1

    # No per-utterance splitting: both utterances point at the same chapter file.
    assert train_recs[0]["path"] == valid_recs[0]["path"]
    assert Path(train_recs[0]["path"]) == audio_out / "hifitts2_audio" / "ch1.mp3"

    assert train_recs[0]["start"] == 0.0
    assert train_recs[0]["duration"] == 5.0
    assert train_recs[0]["transcript"] == "hello world"

    assert valid_recs[0]["start"] == 5.0
    assert valid_recs[0]["duration"] == 4.0
    assert valid_recs[0]["transcript"] == "goodbye"

    # Exactly one download for the whole chapter, not one per utterance.
    assert list(audio_out.rglob("*.mp3")) == [audio_out / "hifitts2_audio" / "ch1.mp3"]


def test_hf_alignments_join_on_utterance_id_not_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Rows that share a chapter file's `path` still join correctly, because
    the join key is each row's `audio_filepath`, not its `path`."""
    # published alignments: two utterances, utterance-relative timestamps
    snap = tmp_path / "snap"
    (snap / "train").mkdir(parents=True)
    with gzip.open(snap / "train" / "train_aligned-000-of-001.jsonl.gz", "wt") as w:
        w.write(
            json.dumps(
                {
                    "audio_filepath": "book1/utt0.flac",
                    "words": [{"word": "hi", "start": 0.1, "end": 0.4}],
                }
            )
            + "\n"
        )
    with gzip.open(snap / "eval_aligned.jsonl.gz", "wt") as w:
        pass
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))

    # raw manifest: both rows point at the SAME chapter file, distinct ids
    manifest = tmp_path / "raw.jsonl"
    rows = [
        {
            "path": "/audio/book1/chapter.flac",
            "start": 0.0,
            "duration": 2.0,
            "transcript": "hi",
            "audio_filepath": "book1/utt0.flac",
        },
        {
            "path": "/audio/book1/chapter.flac",
            "start": 2.0,
            "duration": 2.0,
            "transcript": "yo",
            "audio_filepath": "book1/utt9.flac",
        },
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = tmp_path / "aligned.jsonl"
    prepare_data.attach_hf_alignments(manifest, out, "hf://x/y", tmp_path)
    got = [json.loads(line) for line in out.read_text().splitlines()]
    assert got[0]["words"][0]["word"] == "hi"  # matched by id
    assert "words" not in got[1]  # unmatched row kept, no words
